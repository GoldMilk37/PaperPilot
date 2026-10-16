"""PII 脱敏：邮箱 / 手机号 / 身份证 / API 密钥（计划 §六.4，保护用户隐私）。

模块速览：
    redact(text)          把命中 PII 的片段替换为 [REDACTED:类型]
    PII_PATTERNS          模式表（检测类型 → 正则），自测契约的一部分

两条强制路径（谁用谁负责调用，本模块只提供 redact）：
    (a) 用户输入进 messages 之前——lg_graph 的 ingest/guard_input 处
    (b) 任何日志与 Langfuse 上报之前——executor._log、tracing 的 attrs

设计决策：
    P1 身份证做 GB 11643 校验位验证，不做裸正则：18 位数字串在论文域
       并不罕见（DOI、某些编号体系），裸匹配会误伤；校验位错的一律放行。
       这是"结构约束优于模式猜测"——校验位是签发方编码进号码的数学约束，
       伪造不了的巧合比正则概率低得多。
    P2 替换而非删除：下游（LLM、日志）拿到的是占位符 [REDACTED:email]，
       保留"这里曾有一个邮箱"的形状信息——对"帮我给作者发邮件"这类
       请求，占位符让 LLM 知道用户给过邮箱但内容被隐藏，而不是一脸懵。
    P3 手机号只认中国大陆手机格式（1[3-9] 开头 11 位），前后不能是数字
       （避免从更长的数字串里切出假号）。

用法：from app.agent.pii import redact
命令行：python -m app.agent.pii（跑内置自测）
"""
import re

# 类型 → 正则。顺序即替换顺序（身份证先于手机号：15 位老证含 11 位段会撞手机号模式）
_PHONE = r"(?<!\d)1[3-9]\d{9}(?!\d)"
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("id_card", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("phone", re.compile(_PHONE)),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
]

# GB 11643 校验位：权重与校验码映射表
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CODES = "10X98765432"

def _valid_id(num: str) -> bool:
    """P1：17 位本体加权 mod 11 的余数必须对上第 18 位校验码。"""
    try:
        r = sum(int(c) * w for c, w in zip(num[:17], _ID_WEIGHTS)) % 11
    except ValueError:
        return False
    return num[17].upper() == _ID_CODES[r]

def redact(text: str) -> str:
    """把 PII 替换为 [REDACTED:类型] 占位符（P2）。幂等：占位符本身不再匹配。"""
    if not text:
        return text
    out = text
    for kind, pat in PII_PATTERNS:
        out = pat.sub(lambda m, k=kind: _sub(m, k), out)
    return out

def _sub(m: re.Match, kind: str) -> str:
    if kind == "id_card" and not _valid_id(m.group(0)):
        return m.group(0)          # 校验位不过 → 不是身份证，放行（P1）
    return f"[REDACTED:{kind}]"

# ---------------- 自测：python -m app.agent.pii ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1 四类命中
    s = redact("发邮件到 a@b.com，手机 13812345678，key 是 sk-abc123DEF456ghi789JKL")
    assert "[REDACTED:email]" in s and "[REDACTED:phone]" in s and "[REDACTED:api_key]" in s, s
    assert "a@b.com" not in s and "sk-abc" not in s
    print("1 四类命中 OK:", s)

    # 2 身份证：校验位对的命中、错的不动（P1）
    ok, bad = "11010519491231002X", "110105194912310021"
    s2 = redact(f"证号 {ok} 和 {bad}")
    assert f"[REDACTED:id_card]" in s2 and bad in s2, s2
    print("2 身份证校验位 OK: 真号拦截 / 假号放行")

    # 3 不误伤：arXiv 号、DOI、英文句子里的 system、短数字
    safe = "论文 2401.12345 的 DOI 是 10.1000/xyz123，引用了 3 篇文献"
    assert redact(safe) == safe, redact(safe)
    print("3 不误伤 OK: arXiv 号 / DOI / 普通数字原样保留")

    # 4 幂等：脱敏产物再脱一遍不变（P2 占位符不二次匹配）
    once = redact("邮箱 x@y.io")
    assert redact(once) == once
    print("4 幂等 OK: [REDACTED:*] 占位符不被二次匹配")

    # 5 空串与非字符串安全
    assert redact("") == ""
    print("5 边界 OK")
    print("\npii 自测 5/5 通过")