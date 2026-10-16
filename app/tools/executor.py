"""工具层：统一执行入口（能力在 server 侧，策略在 client 侧——计划 §六.1）。

模块速览：
    execute(name, arguments, *, mcp, confirmed)   状态机只调这个（M3 tool_exec）
    ToolResult                                     每次调用的统一结果

    ① 确认门（D1）：写操作未 confirmed → needs_confirmation（不执行）
    ② 查缓存（D5）：只读 + 命中 → 直接返回
    ③ 真执行（D2）：mcp.call + asyncio.wait_for 超时包裹
    ④ 结果三态（D3）：ok / error / timeout，永不裸抛
    ⑤ 写缓存：仅"只读 + ok"才写
    ⑥ 记日志（D4）：每个出口都记，包括被拦截的和缓存命中的

从 ShopMate executor 继承与改造的决策：
    D1 写操作确认门（继承）：WRITE_OPS 命中且未 confirmed → 不执行，返回
       needs_confirmation + 提案文案，状态机拿它去问用户（「提案→确认→
       执行」）。WRITE_OPS 硬编码在 client 侧：这是安全策略，不允许从
       server 的 schema 名字里"猜"——启发式漂了防线就漏了。M3 registry
       内省得到的是"声明"（喂 LLM 的），不是"权限"（这三个字面量）。
    D2 超时（3s → 60s + 线程池 → wait_for）：ShopMate 的 mock 瞬回所以
       3s 够；这里真出网（arXiv 检索 / PDF 下载解析是十几秒的长尾），且
       必须高于 mcp_client 的 read_timeout=30s，否则传输层超时永远轮不
       到自己抛，被执行器先掐掉语义就错乱。用 asyncio.wait_for 而非线程
       池：mcp.call 本来就是协程，包一层线程等于把异步架构倒回同步。
    D3 结果三态（继承）：ok / error / timeout。mcp.call 的 (False, data)
       是业务性失败（is_error，SDK 语义是"作为结果返回给 LLM"）→ error；
       wait_for 超时 → timeout；其余异常（传输层断连等协议层错误）→
       error。结构永远完整，上层不必区分超时和失败。
    D4 日志落 data/logs/tool_calls.jsonl（继承+补强）：追加写、每出口必
       记（含被确认门拦截的；缓存命中也记——命中率是 M7 观测指标）。
    D5 只读 5min 结果缓存（继承+裁剪）：写操作永不缓存（副作用重放是
       事故）。裁掉 ShopMate 缓存 key 里的 user_id 分量：单用户本地工具
       没有串户问题，留一个恒为空的分量只会让人误以为这里有身份概念。
    D6 身份注入（整体裁掉）：ShopMate 的多用户订单域需要"模型说的身份
       不算数"；PaperPilot 无登录态、单用户，没有"别人的数据"可越权。
       将来接多用户时，注入点仍然回到 execute 入口。
    D7 mcp 客户端显式注入（新增）：execute 的 mcp 参数必填——谁持有
       连接（async with McpClient）谁负责传进来，不做模块级单例。MCP
       客户端是有生命周期的异步上下文，藏一个全局实例等于把连接状态
       变成隐式依赖，自测也换不进桩。

用法：from app.tools.executor import execute
命令行：python -m app.tools.executor（跑内置自测，mcp.call 用桩，不真开子进程）
"""
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

# 写操作清单（D1）：与 mcp_server/server.py 的三个写工具一一对应
WRITE_OPS = {"add_to_library", "save_note", "remove_paper"}

# 超时秒数（D2）：必须 > mcp_client 的 read_timeout=30s
TOOL_TIMEOUT_S = 60.0

# 缓存 TTL（D5）
CACHE_TTL_S = 300.0

# 日志目录：项目根/data/logs
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"

TIMEOUT_MSG = "处理超时了（外部检索或 PDF 解析可能较慢），请稍后重试。"
ERROR_PREFIX = "工具调用失败："

@dataclass
class ToolResult:
    """一次工具调用的统一结果（D3 三态 + 确认门特态 + 原始数据）。"""
    status: str                 # ok / error / timeout / needs_confirmation
    data: dict = field(default_factory=dict)     # 成功时的工具返回；is_error 时含 error 文本
    message: str = ""           # 给用户的可直接展示的话术（含降级/提案文案）

    @property
    def ok(self) -> bool:
        return self.status == "ok"

# ---------------- 5min 结果缓存（D5，只缓存只读 ok 结果） ----------------

_cache: dict[str, tuple[float, ToolResult]] = {}

def _cache_key(tool: str, arguments: dict) -> str:
    canon = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return f"tool:cache:{tool}:{hashlib.md5(canon.encode('utf-8')).hexdigest()}"

def _cache_get(key: str) -> ToolResult | None:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    _cache.pop(key, None)          # 过期顺手清掉，避免缓存无限涨
    return None

def _cache_set(key: str, r: ToolResult) -> None:
    _cache[key] = (time.time(), r)

def clear_cache() -> None:
    """测试/运维用：清空结果缓存。"""
    _cache.clear()

# ---------------- 执行入口 ----------------

async def execute(name: str, arguments: dict | None = None, *,
                  mcp, confirmed: bool = False) -> ToolResult:
    """D7：mcp 是持有连接的 McpClient（或同形状的桩），必填。"""
    t0 = time.perf_counter()
    arguments = dict(arguments or {})
    r: ToolResult

    if name in WRITE_OPS and not confirmed:            # D1 确认门：先拦，不执行
        r = ToolResult("needs_confirmation", message=_proposal(name, arguments))
    else:
        key = None
        if name not in WRITE_OPS:                      # D5 写操作永不读缓存
            key = _cache_key(name, arguments)
            hit = _cache_get(key)
            if hit is not None:
                _log(name, arguments, hit, time.perf_counter() - t0)
                return hit

        try:                                           # D2 真执行：wait_for 超时包裹
            ok, data = await asyncio.wait_for(
                mcp.call(name, arguments), timeout=TOOL_TIMEOUT_S)
            r = (ToolResult("ok", data=data) if ok else
                 ToolResult("error", data=data,
                            message=f"{ERROR_PREFIX}{data.get('error', '未知错误')}"))
        except (asyncio.TimeoutError, TimeoutError):   # py3.10 两者不同类，都得接
            r = ToolResult("timeout", message=TIMEOUT_MSG)
        except Exception as e:                         # 协议层/传输层异常也不裸抛（D3）
            r = ToolResult("error", message=f"{ERROR_PREFIX}工具内部错误 ({type(e).__name__})")

        if r.ok and key is not None:                   # D5 仅"只读 + ok"才写缓存
            _cache_set(key, r)

    _log(name, arguments, r, time.perf_counter() - t0)
    return r

# ---------------- 内部辅助 ----------------

def _proposal(name: str, arguments: dict) -> str:
    """写操作确认门给用户看的提案（D1）。计划 §六.4 第 3 层：提案要说清
    「要写什么、为什么」，M3 接 _human_gate 时还要追加标注触发来源
    （论文内容触发 or 用户主动要求）。"""
    if name == "add_to_library":
        return (f"即将把论文 {arguments.get('arxiv_id', '?')} 加入文献库，"
                f"理由：{arguments.get('reason', '未填写')}。确认入库吗？")
    if name == "save_note":
        text = str(arguments.get("text", ""))
        preview = text if len(text) <= 40 else text[:40] + "…"
        return (f"即将给论文 {arguments.get('paper_id', '?')} 写入笔记："
                f"「{preview}」。确认写入吗？")
    if name == "remove_paper":
        return (f"即将把论文 {arguments.get('paper_id', '?')} 移出文献库"
                f"（其笔记、本地 PDF 与解析缓存会一并清理）。确认移除吗？")
    return f"即将执行写操作 {name}，参数：{arguments}。确认吗？"

def _log(name: str, arguments: dict, r: ToolResult, elapsed: float) -> None:
    """D4：追加写一行 JSON 到 data/logs/tool_calls.jsonl。写失败只 print，
    不影响主流程。M4 的 Langfuse MCP span 与这里同源同字段。"""
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": name,
           "arguments": arguments,
           "result": {"status": r.status, "data": r.data, "message": r.message},
           "elapsed_ms": round(elapsed * 1000)}
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with (LOG_DIR / "tool_calls.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[tool_log] 写日志失败: {e}")

# ---------------- 自测：python -m app.tools.executor ----------------
# 按计划 §十：mock 换成 mcp_client.call 的桩，不真开子进程（真子进程归
# mcp_client 的自测管，这里验的是策略层本身）。

class _FakeMcp:
    """mcp_client.call 的桩：形状与 C3 契约一致——(ok, data) 二元组。"""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call(self, name: str, args: dict) -> tuple[bool, dict]:
        self.calls.append((name, dict(args)))
        if name == "list_library":
            return True, {"entries": [], "count": 0}
        if name == "add_to_library":
            if args.get("arxiv_id") == "2401.00001":
                return True, {"arxiv_id": "2401.00001", "created": True}
            return False, {"error": "arXiv 上找不到 9999.99999"}
        if name == "save_note":
            return True, {"paper_id": args["paper_id"], "text": args["text"]}
        if name == "__hang":                           # 超时分支专用的挂起桩
            await asyncio.sleep(10)
        return False, {"error": f"Unknown tool: {name}"}   # 拟真 server 的未知工具响应

async def _selftest_async() -> None:
    global TOOL_TIMEOUT_S

    mcp = _FakeMcp()

    # 1 只读直接执行
    r = await execute("list_library", {}, mcp=mcp)
    assert r.ok and r.data == {"entries": [], "count": 0}, r
    print("1 只读工具:", r.status, r.data)

    # 2 缓存命中：同参二连调，桩只被真调一次
    r2 = await execute("list_library", {}, mcp=mcp)
    assert r2.ok and r2.data == r.data and len(mcp.calls) == 1, (r2, mcp.calls)
    print("2 缓存命中: OK（桩仅被调 1 次）")

    # 3 确认门：写操作未确认 → needs_confirmation，且桩未被调（无副作用）
    args = {"arxiv_id": "2401.00001", "reason": "与方法论章节相关"}
    r3 = await execute("add_to_library", args, mcp=mcp)
    assert r3.status == "needs_confirmation" and len(mcp.calls) == 1, (r3, mcp.calls)
    print("3 确认门:", r3.status, "|", r3.message)

    # 4 确认后真执行
    r4 = await execute("add_to_library", args, mcp=mcp, confirmed=True)
    assert r4.ok and r4.data["created"] is True, r4
    print("4 确认执行:", r4.data)

    # 5 业务失败与未知工具都不裸抛
    r5 = await execute("add_to_library", {"arxiv_id": "9999.99999", "reason": "x"},
                       mcp=mcp, confirmed=True)
    assert r5.status == "error" and r5.message, r5
    r6 = await execute("不存在的工具", {}, mcp=mcp)
    assert r6.status == "error", r6
    print("5 错误三防: error 不裸抛 /", r5.message)

    # 6 超时降级：把常量压到 0.5s，桩挂 10s，验超时分支本身
    TOOL_TIMEOUT_S = 0.5
    t0 = time.perf_counter()
    r7 = await execute("__hang", {}, mcp=mcp)
    assert r7.status == "timeout" and time.perf_counter() - t0 < 5, (r7, time.perf_counter() - t0)
    TOOL_TIMEOUT_S = 60.0
    print("6 超时降级:", r7.status, "|", r7.message)

    # 7 日志落盘：每个出口都记（含被确认门拦截的）
    lines = (LOG_DIR / "tool_calls.jsonl").read_text(encoding="utf-8").strip().splitlines()
    recs = [json.loads(x) for x in lines]
    assert all({"ts", "tool", "arguments", "result", "elapsed_ms"} <= set(x) for x in recs)
    assert any(x["result"]["status"] == "needs_confirmation" for x in recs)
    print(f"7 日志: 共 {len(lines)} 条，含被拦截的调用；末条 tool={recs[-1]['tool']}")

    # 8 写操作永不缓存：同参确认执行两次，桩两次都真调（幂等性由 server 保证）
    clear_cache()
    n0 = len(mcp.calls)
    ra = await execute("save_note", {"paper_id": "2401.00001", "text": "hi"},
                       mcp=mcp, confirmed=True)
    rb = await execute("save_note", {"paper_id": "2401.00001", "text": "hi"},
                       mcp=mcp, confirmed=True)
    assert ra.ok and rb.ok and len(mcp.calls) == n0 + 2, (len(mcp.calls), n0)
    print("8 写操作不缓存: OK（重复执行都真调，副作用不被缓存重放）")

    # 9 传输层异常不裸抛：换一个会断连的桩，验协议层异常也归 error。
    #    注意先清缓存——list_library 在测试 1 已缓存，不清会命中缓存
    #    根本调不到 _Boom，测出来是绿的但 _Boom 有没有生效看不出来（假通过）。
    clear_cache()

    class _Boom:
        async def call(self, name: str, args: dict) -> tuple[bool, dict]:
            raise ConnectionError("stdio pipe closed")

    r9 = await execute("list_library", {}, mcp=_Boom())
    assert r9.status == "error" and "ConnectionError" in r9.message, r9
    print("9 异常兜底: 传输层异常 →", r9.message)

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(_selftest_async())
    print("\nexecutor 自测 9/9 通过（mcp.call 用桩，未真开子进程）")