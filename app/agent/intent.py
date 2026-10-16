"""意图识别节点：用户一句话 → (意图, 置信度, 槽位, 情绪)。结构照抄 ShopMate，域重画。

模块速览：
    classify(text, history, current_paper_id)   唯一入口，graph 的 classify 节点调
    IntentResult                                结果数据类

六个设计决策：
    D1 意图集合 = 计划 §六.2 的 6 类：search/qa/compare/library/chitchat/unsupported。
       ShopMate 的 human_transfer 在这里换成诚实的 unsupported：本地工具没有
       "真人客服"可转，与其把用户挂起等待一个不存在的出口，不如明说
       "这超出我的能力"。unsupported 不是兜底垃圾桶——chitchat 管寒暄，
       unsupported 管"明确超出论文助手职责的请求"（写作业/写文案）。
    D2 paper_id 定序：正则 → LLM 抽取 → 结构校验，全过才算数。与 ShopMate
       的关键差异：那边校验"在不在商品目录"，这里只校验 arXiv ID 的**结构**
       （\\d{4}.\\d{4,5}）。理由：文献库内容动态变化，classify 不该为验一个 ID
       去读盘或开 MCP 连接；结构性合法但库中不存在的 ID 会被工具优雅地返回
       "找不到"，代价是一次空结果而非错误路由。语义校验（在不在库里）下沉
       到工具层，本层只守结构防线——分类器保持纯函数式的便宜。
    D3 arXiv ID 是确定性字面证据，正则优先于 LLM 复述（ShopMate D5 原则）。
       版本后缀（v2）剥掉：库里的 paper_id 存基础 ID，带版本查库等于查无此文。
    D4 指代消解两条信息源：近 4 条 user 消息（ShopMate 原样）+ current_paper_id
       锚点（session D4 显式状态）。锚点作为**信息**进 prompt（"当前话题论文，
       仅当用户指的就是它时才填"），不作为代码层兜底直接贴——语义归属让
       有上下文的 LLM 判，代码只做 D2 的结构校验。锚点直接贴的风险：用户
       已切话题（"搜一篇新的"）时会把旧论文错贴到新请求上。
    D5 调模型与解析分成两截（ShopMate D7 原样）：网络断了和 JSON 坏了不是
       一件事，混在一个 except 里测了也分不清谁是谁。
    D6 永不向上抛 + confidence 阈值裁决权在状态机（ShopMate D3/D6 原样）：
       任何失败返回 IntentResult()（confidence=0），状态机的低置信分支
       （clarify）就是为这种时刻准备的。

与 guardrails 的接缝：classify 在 guard_input **之后**跑，进来的 text 已中和——
本模块不重复做注入检测，但也没资格假设拿到的必然干净（D6 的兜底人格不变）。

用法：from app.agent.intent import classify
命令行：python -m app.agent.intent（离线部分不联网；在线部分需 .env 的 key）
"""
import json
import re
from dataclasses import dataclass, field

from ..llm import client as llm

# 意图集合（D1）。值与 lg_graph 的路由表一一对应。
INTENTS = (
    "search",       # arXiv 检索：找新论文（没有确定目标）
    "qa",           # 论文问答：问已确定论文的内容（本地 Chroma RAG）
    "compare",      # 论文对比：两篇及以上
    "library",      # 文献库管理：加入/移出/写笔记/列清单
    "chitchat",     # 寒暄
    "unsupported",  # 明确超出职责（D1：诚实地说做不到）
)

# D3：arXiv ID 形如 2401.12345（新版 5 位）或 2006.0999（旧版 4 位）。
_ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")
_ARXIV_FULL = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")

SYSTEM_PROMPT = """你是论文助手的意图分类器。把用户消息分类并抽取信息，输出严格的 json 对象（不要输出其他内容）：
{"intent": "search|qa|compare|library|chitchat|unsupported",
 "confidence": 0到1的小数,
 "dissatisfied": true|false,
 "slots": {"paper_id": "", "query": "", "compare": []}}
当前话题论文：{CURRENT}
判定要点：
- search：到 arXiv 搜**新**论文——用户没有确定目标，要的是"帮我找几篇关于 X 的"。query 填检索关键词
- qa：问**已确定论文**的内容——消息里报了 arXiv ID，或问的就是当前话题论文。问"讲了什么/方法/实验/结论"
- compare：两个及以上论文的对比异同。compare 填要对比的论文（ID、名字或描述）
- library：文献库管理——加入/移出论文、写笔记、列出库里有什么
- chitchat：寒暄问候，与论文无关但可友好回应
- unsupported：明确超出论文助手职责（写作业、写文案、无关任务）——诚实承认做不到
- search 与 qa 的分界：要"找论文"是 search，拿着论文"问内容"是 qa
- paper_id：用户指的具体论文的 arXiv ID（形如 2401.12345）。只在能确定时填：消息里报了
  ID，或结合近期对话能确定问的就是当前话题论文。拿不准留空串——填错会让下游
  检索被过滤到错误的论文上，比不过滤更糟
- dissatisfied：用户表达不满/生气/失望（"怎么又超时了"）
confidence 给你自己的判断把握，拿不准就给低分，不要硬凑高。"""

def _pid_from_text(text: str) -> str:
    """D3：从字面抽出 arXiv 基础 ID（剥版本后缀）。抽不到返回空串。"""
    m = _ARXIV_RE.search(text or "")
    return m.group(1) if m else ""

def _resolve_paper_id(text: str, slots: dict) -> str:
    """D2：正则优先 → LLM 抽的兜底 → 结构不合法一律丢弃。

    只守结构（D2）：一个结构合法但库里没有的 ID 会被工具返回"找不到"，
    优雅降级；而一个结构都不合法的 ID 会让下游 where 过滤变成垃圾进垃圾出。
    """
    pid = _pid_from_text(text) or str(slots.get("paper_id", "") or "").strip()
    return pid if _ARXIV_FULL.match(pid) else ""

def _system_prompt(current_paper_id: str) -> str:
    """D4：锚点作为信息进 prompt，不作为代码兜底。"""
    cur = f"  - {current_paper_id}（用户说\"这篇/它\"时大概率指它，但先确认语境）" \
        if current_paper_id else "  （暂无）"
    return SYSTEM_PROMPT.replace("{CURRENT}", cur)

@dataclass
class IntentResult:
    """classify 的输出。confidence=0 视为"不知道"（D6）。"""
    intent: str = "chitchat"
    confidence: float = 0.0
    dissatisfied: bool = False
    slots: dict = field(default_factory=dict)

def _parse(raw: str, text: str) -> IntentResult:
    """D5：把模型吐的 JSON 变成 IntentResult。每一处降级各留各的痕。"""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[intent] JSON 解析失败，按未知处理: {type(e).__name__}: {e}；"
              f"原文={raw[:120]!r}")
        return IntentResult()
    if not isinstance(d, dict):
        print(f"[intent] 模型返回的不是对象（{type(d).__name__}），按未知处理")
        return IntentResult()

    intent = d.get("intent", "")
    if intent not in INTENTS:
        print(f"[intent] 模型给了意图集外的名字 {intent!r}，按未知处理")
        return IntentResult()

    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        print(f"[intent] confidence 不是数字（{d.get('confidence')!r}），按下限 0 处理")
        conf = 0.0

    slots = d.get("slots")
    if not isinstance(slots, dict):
        if slots is not None:
            print(f"[intent] slots 不是对象（{type(slots).__name__}），按空槽位处理")
        slots = {}
    slots["paper_id"] = _resolve_paper_id(text, slots)
    return IntentResult(intent, max(0.0, min(1.0, conf)),
                        bool(d.get("dissatisfied", False)), slots)

async def classify(text: str, history: list[dict] | None = None, *,
                   current_paper_id: str = "") -> IntentResult:
    """识别意图。LLM/解析任何环节出问题都返回 confidence=0（D6）。

    D5：调模型（safe_call 含重试）与解析分成两截，各报各的错。
    """
    # D4：只带最近 4 条 user 消息作指代消解的上下文
    recent = [m["content"] for m in (history or [])[-8:] if m["role"] == "user"][-4:]
    context = ("\n近期对话（仅供消解指代，如\"它/这篇\"指什么）：\n"
               + "\n".join(f"- {c}" for c in recent)) if recent else ""
    messages = [
        {"role": "system", "content": _system_prompt(current_paper_id)},
        {"role": "user", "content": f"用户消息：{text}{context}"},
    ]

    msg = await llm.safe_call(llm.chat, messages, temperature=0.0, json_mode=True)
    if msg is None:
        return IntentResult()

    try:
        return _parse(msg.content or "", text)
    except Exception as e:
        print(f"[intent] 解析兜底触发，按未知处理: {type(e).__name__}: {e}")
        return IntentResult()

# ---------------- 自测：python -m app.agent.intent（在线部分需 .env 的 key） ----------------
if __name__ == "__main__":
    import asyncio
    import os
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # ---- 离线 1：ID 抽取与结构校验（确定性，不联网） ----
    assert _pid_from_text("2401.12345 讲了什么") == "2401.12345"
    assert _pid_from_text("2006.0999v2 的作者") == "2006.0999", "版本后缀应剥掉"
    assert _pid_from_text("这篇论文怎么样") == ""
    assert _resolve_paper_id("99.999 讲什么", {}) == "", "结构不合法必须丢弃"
    assert _resolve_paper_id("没关系", {"paper_id": "abcd.12345"}) == ""
    assert _resolve_paper_id("没关系", {"paper_id": "2401.12345"}) == "2401.12345"
    print("1 ID 抽取（无需 key）: 正则命中 / 版本剥离 / 结构校验丢弃 全部 OK\n")

    # ---- 离线 2：解析降级五路径（不联网，直接喂坏 JSON） ----
    assert _parse("这不是 json", "随便问问").confidence == 0.0
    assert _parse('["数组，不是对象"]', "随便问问").confidence == 0.0
    assert _parse('{"intent": "human_transfer", "confidence": 0.9}', "x").confidence == 0.0, \
        "ShopMate 的旧意图名必须被拒之门外（意图集已重画）"
    _r = _parse('{"intent": "qa", "confidence": "很高"}', "2401.12345 讲了什么")
    assert _r.intent == "qa" and _r.confidence == 0.0, _r
    _r = _parse('{"intent": "qa", "confidence": 0.9, "slots": "无"}', "2401.12345 讲了什么")
    assert _r.confidence == 0.9 and _r.slots["paper_id"] == "2401.12345", _r
    print("2 解析降级（无需 key）: 坏JSON / 非对象 / 意图集外 / confidence非数字 / "
          "slots非对象 —— 五条路径均按预期降级\n")

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("跳过在线部分：.env 里没有 DEEPSEEK_API_KEY（client.py 已自测过缺 key 路径）")
        sys.exit(0)

    # ---- 在线：意图边界成对测（探针原则同 ShopMate：光测认得出不够，
    #      还得测没把近邻意图误判。search/qa 的探针是"找论文 vs 问内容"） ----
    async def _online() -> None:
        cases = [   # (话术, 期望意图, 期望槽位非空)
            ("帮我搜一下 2024 年关于 RAG 评测的论文", "search", "query"),
            ("注意力机制相关的新论文有哪些", "search", "query"),
            ("2401.12345 的实验设置是什么", "qa", "paper_id"),
            ("对比这两篇论文在长文本任务上的表现", "compare", None),
            ("把这篇加入我的文献库", "library", None),
            ("帮我列一下库里有什么", "library", None),
            ("你好呀", "chitchat", None),
            ("帮我写一份小红书文案", "unsupported", None),
        ]
        ok = 0
        for text, want, slot in cases:
            r = await classify(text)
            hit = r.intent == want and r.confidence >= 0.6
            if hit and slot:
                hit = bool(r.slots.get(slot))
            ok += hit
            print(f"{'✓' if hit else '✗'} [{r.confidence:.2f}] {r.intent:12s} "
                  f"{r.slots}  ← {text}")

        # 指代消解：锚点进 prompt（D4）——"这篇"应落到锚点论文上
        r = await classify("它的实验部分讲了什么", [],
                           current_paper_id="2401.12345")
        print(f"锚点消解: {r.intent} paper_id={r.slots.get('paper_id')!r} "
              "(期望 qa + 2401.12345)")
        # 情绪识别（session.dissatisfaction 计数器的信号源）
        r2 = await classify("这破工具怎么又超时了，火大", [])
        print(f"情绪识别: dissatisfied={r2.dissatisfied} (期望 True)")
        print(f"\n在线用例 {ok}/{len(cases)} 通过")
        await llm.aclose()   # 趁循环活着关连接池，否则退出时 GC 撒噪音

    asyncio.run(_online())