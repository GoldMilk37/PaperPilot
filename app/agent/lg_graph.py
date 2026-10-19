"""LangGraph 版状态机：PaperPilot 的编排主干（M2 第 5 步）。图骨架先行。

用法：
    python -m app.agent.lg_graph        # 离线自测（全桩：不联网、不开真 MCP 子进程）

与 ShopMate lg_graph.py（母本）的映射关系：

    ShopMate                              PaperPilot
    ──────────────────────────────       ─────────────────────────────────
    8 意图 + human_transfer 兜底          6 意图（intent D1：无人工可转，
                                         超出能力诚实说 unsupported）
    ingest → classify → rag/tool_llm     guard_input → ingest → classify
                                         → recall→qa（桩）/ tool_llm
    Session.current_product_id           state.focus（话题论文锚点，
                                         classify 槽位回写，喂下一轮消解）
    mcp 无（工具是本地函数）             mcp 经 config["configurable"] 注入
                                         （executor D7：谁持有连接谁传入）
    兜底①②③④全汇聚 transfer 节点       无 transfer：连败/轮尽/不满/unsupported
                                         全部收敛为诚实回复

设计决策：
    D1 确认门前置性（继承母本 D1）：确认门话语由 interrupt 暂停承载，用户
       的"确认/取消"作为 resume 值直接回 human_gate，不经过 ingest/classify
       ——不让 LLM 猜意图的前置纪律等价成立。
    D2 interrupt 只在 human_gate 节点顶部且该节点必须 sync（实测教训）:
       本版本 langgraph 的 async 节点不传 config 上下文（context= 参数
       py3.11 才进 asyncio.create_task），interrupt() 里的 get_config()
       会炸。确认门拆成 human_gate（sync，只裁决零副作用）→ gate_exec
       （async，confirmed=True 真执行），副作用仍在裁决之后，D2 等价。
       写操作在 tool_exec 以 confirmed=False 探路。
    D3 听不懂不丢挂起（继承母本 D3）：resume 回来仍是模糊话术时留在门里
       继续 interrupt，与手写版 pending_write 语义一致。
    D4 Session 降级为视图模型（继承母本 D4）：逻辑状态全住 graph state 由
       checkpointer 持久化；adapter 每轮镜像回 SessionStore（history/trace/
       current_paper_id/pending_write/tool_fail_streak/dissatisfaction），
       将来 cli/api 一行不改。
    D5 mcp 注入走 config（新增）：executor D7（mcp 必填、谁持有谁传入）在
       图上的落点——节点不建连接，LgAgent 持有 mcp 并在每次 invoke 的
       configurable 里传入。收益：离线自测把桩从 config 塞进去即可，图代码
       零改动；连接生命周期归调用方，图里不留隐藏的全局异步资源。
    D6 qa/recall 先落桩：检索层（本地 Chroma RAG / 记忆召回）未接入，两个
       节点先返回诚实占位提示，图结构/路由/确认门/断点续跑先全绿，后续
       替换实现不动结构（骨架先行的全部意义）。recall→qa 暂为串行边：
       计划 §六.5 的并行扇出需要 join 语义（messages 合并策略）定稿后再开，
       trace 已预留 memory 字段。
    D7 兜底全收敛为诚实回复：连败（fail_streak≥3）、轮尽（≥3 轮）、连续
       不满（≥2 次）、unsupported 意图，统一给"承认做不到/请稍后再试"——
       本地助手没有人工客服，假装转接比承认做不到更糟。
    D8 guard_input 在 ingest 之前：脱敏与注入中和发生在任何状态落盘之前
       （messages 里进的是已中和文本），trace.injection 记录命中类型。

依赖关系：intent.classify（async）、executor.execute（async + mcp）、
registry.get_schemas（async + mcp）、guardrails.scan_input、pii.redact、
llm.client（async）——业务与安全逻辑零改动，只换编排层。
"""
import json
import time
from pathlib import Path
from typing import TypedDict

import aiosqlite
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..llm import client as llm
from ..tools.executor import clear_cache, execute as tool_execute
from ..tools.registry import get_schemas
from .guardrails import scan_input
from .intent import IntentResult, classify
from .pii import redact
from .session import MAX_ROUNDS, store

# checkpointer 落盘位置（data/ 下运行产物，.gitignore 已忽略）
CHECKPOINT_DB = Path(__file__).resolve().parents[2] / "data" / "graph_checkpoints.sqlite3"

PERSONA = ("你是 PaperPilot，一个本地运行的论文助手：能检索 arXiv、回答论文内容、"
           "对比论文、管理本地文献库。回答用中文，简洁、诚实——不知道就说不知道。")

TOOL_SYSTEM_PROMPT = PERSONA + """
你可以调用工具完成检索与文献库管理。规则：
- search_arxiv 用于找新论文；fetch_paper/read_paper 用于看某篇论文的内容；
  list_library 列出库里有什么
- 写操作（加入文献库/写笔记/移出论文）：收集齐参数后直接调用工具，系统会
  自动向用户出示确认提示。你自己绝不在回复文本里征求确认（那会让系统错过
  确认流程），工具返回的提示原样转达即可
- paper_id 优先用对话里出现过的 arXiv ID（形如 2401.12345）；没有就先检索
- 查询结果原样转述，不要编造论文内容或 ID"""

YES_WORDS = ("确认", "确定", "同意", "可以", "好的", "嗯", "yes", "ok")
NO_WORDS = ("取消", "不要", "算了", "放弃", "不用", "no")

CLARIFY_THRESHOLD = 0.6      # 低于此置信度 → 澄清（intent D6：裁决权在状态机）
MAX_TOOL_ROUNDS = 3          # 工具编排轮次上限
MAX_TOOL_FAILS = 3           # 工具连败上限（跨轮计数）
MAX_DISCONTENT = 2           # 连续不满上限（母本兜底④，收敛为诚实回复）

LLM_FAIL_REPLY = "我这边调用模型出了点问题，请稍后再试。"
NO_INFO_REPLY = "没有查到相关内容。"
MCP_MISSING_REPLY = "工具连接不可用，请稍后再试。"
QA_STUB_REPLY = ("论文内容问答还没接入检索层（后续接入本地向量检索），暂时回答"
                 "不了这个问题。你可以先把论文加入文献库，或让我检索论文。")
UNSUPPORTED_REPLY = ("这超出了我的能力范围。我是论文助手，能做的有：arXiv 检索、"
                     "论文内容问答、论文对比、文献库管理（加入/笔记/移出）。")
DISCONTENT_REPLY = ("抱歉让你反复遇到问题。目前我的能力确实有限，建议换个问法再试，"
                    "或稍后再来。")

# ---------------- State：一轮对话的全部可变状态（住进 checkpointer） ----------------

class LGState(TypedDict, total=False):
    sid: str                 # 会话 id（thread_id 同源）
    user_text: str           # 本轮用户原话（guard_input 中和后）
    guard_hits: list         # guard_input 的注入命中类型（ingest 收进 trace 后清零）
    messages: list           # OpenAI 格式历史（ingest 截 10 轮）
    trace: dict              # 本轮轨迹（每轮 ingest 重建）
    # 意图识别产物
    intent: str
    confidence: float
    slots: dict
    discontent: int          # 连续不满计数（跨轮）
    focus: str               # 话题论文锚点（跨轮，classify 槽位回写）
    # 确认门（HITL）
    pending_write: dict | None    # {name, arguments, message=提案文案}
    gate_hint: str                # "没听清"后再 interrupt 时给用户看的话
    gate_decision: str            # human_gate 的裁决结果（accept 才进 gate_exec）
    # 工具编排
    pending_tool_calls: list      # 本轮 LLM 决定要调的工具 [{id, name, arguments}]
    tool_rounds: int              # 本轮已编排几轮
    fail_streak: int              # 工具连败计数（跨轮）
    last_reply: str               # 本轮最终回复（adapter 返回给用户）

def _gate_decision(text: str) -> str:
    """确认门关键词判定（先否后肯：『不要』必须落在否定，母本原样）。"""
    t = (text or "").strip().lower()
    if any(w in t for w in NO_WORDS):
        return "reject"
    if any(w in t for w in YES_WORDS):
        return "accept"
    return "unclear"

def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}

def _mcp_from(config) -> object | None:
    """D5：mcp 实例从 invoke 的 configurable 里取（LgAgent 持有并传入）。"""
    return ((config or {}).get("configurable") or {}).get("mcp")

def _write_done_reply(name: str, args: dict) -> str:
    """确认门 accept 分支的完成话术（按工具定制，失败话术用 result.message）。"""
    if name == "add_to_library":
        return f"已把论文 {args.get('arxiv_id', '')} 加入文献库。"
    if name == "save_note":
        return f"笔记已保存（论文 {args.get('paper_id', '')}）。"
    if name == "remove_paper":
        return f"已把论文 {args.get('paper_id', '')} 移出文献库。"
    return "操作已完成。"

# ---------------- 节点 ----------------

def _guard_input(state: LGState) -> dict:
    """D8：注入中和 + PII 脱敏，发生在任何状态落盘之前。"""
    g = scan_input(state.get("user_text", ""))
    return {"user_text": redact(g.sanitized), "guard_hits": g.hits}

def _ingest(state: LGState) -> dict:
    """每轮入口：追加用户消息、重置本轮一次性字段、重建轨迹。"""
    text = state.get("user_text", "")
    msgs = list(state.get("messages", []))
    msgs.append({"role": "user", "content": text})
    msgs = msgs[-MAX_ROUNDS * 2:]                     # 与 Session.append_round 同口径
    turn = (len(msgs) - 1) // 2 + 1                   # 含本轮的轮次号
    return {
        "messages": msgs,
        "trace": {"sid": state.get("sid", ""), "turn": turn, "user": text[:200],
                  "injection": list(state.get("guard_hits") or [])},
        # 本轮一次性字段清零（跨轮字段 discontent/fail_streak/focus/pending_write 不动）
        "last_reply": "", "tool_rounds": 0,
        "pending_tool_calls": [], "gate_hint": "", "guard_hits": [],
        "gate_decision": "",
    }

async def _classify(state: LGState) -> dict:
    """意图识别（async）+ 不满计数 + 锚点回写。

    classify 的 confidence=0 失败路径 + safe_call 的网络失败路径都落到
    低置信分支（clarify），不裸抛（intent D6）。
    """
    ir = (await llm.safe_call(classify, state["user_text"],
                              state.get("messages", [])[:-1],
                              current_paper_id=state.get("focus", ""))) or IntentResult()
    discontent = state.get("discontent", 0)
    discontent = discontent + 1 if ir.dissatisfied else 0
    tr = {**state.get("trace", {}), "intent": ir.intent, "confidence": ir.confidence,
          "dissatisfied": ir.dissatisfied, "slots": ir.slots}
    upd: dict = {"intent": ir.intent, "confidence": ir.confidence,
                 "slots": ir.slots, "discontent": discontent, "trace": tr}
    if ir.slots.get("paper_id"):                      # 锚点回写：跨轮消解指代的供给源
        upd["focus"] = ir.slots["paper_id"]
    return upd

def _route_after_classify(state: LGState) -> str:
    """兜底优先于意图路由（母本判定顺序）：不满 > 低置信 > 意图表。"""
    if state.get("discontent", 0) >= MAX_DISCONTENT:
        return "discontent"
    if state.get("confidence", 0.0) < CLARIFY_THRESHOLD:
        return "clarify"
    return {"search": "tool_llm", "qa": "recall", "compare": "tool_llm",
            "library": "tool_llm", "chitchat": "chitchat",
            "unsupported": "unsupported"}.get(state.get("intent", "chitchat"),
                                              "unsupported")

def _clarify(state: LGState) -> dict:
    """兜底①：低置信度追问。"""
    reply = ("我想先确认一下您的需求：是想找论文（arXiv 检索）、问某篇论文的内容，"
             "还是管理文献库？")
    return {"messages": state["messages"] + [_assistant(reply)], "last_reply": reply,
            "trace": {**state["trace"], "stage": "clarify",
                      "fallback": "low_confidence_clarify"}}

def _recall(state: LGState) -> dict:
    """D6 桩：记忆召回（后续接本地笔记/历史检索）。trace 预留 memory 字段。"""
    return {"trace": {**state.get("trace", {}), "stage": "recall",
                      "memory": {"status": "stub", "hits": 0}}}

def _qa(state: LGState) -> dict:
    """D6 桩：论文问答（后续接本地向量检索）。诚实占位，不假装能答。"""
    tr = {**state.get("trace", {}), "stage": "qa", "fallback": "retrieval_not_wired"}
    return {"messages": state["messages"] + [_assistant(QA_STUB_REPLY)],
            "last_reply": QA_STUB_REPLY, "trace": tr}

async def _tool_llm(state: LGState, config: RunnableConfig) -> dict:
    """工具编排的 LLM 半步：决定调什么工具；不调了就出终答。"""
    tr = {**state.get("trace", {}), "stage": "tool"}
    mcp = _mcp_from(config)
    if mcp is None:
        tr["fallback"] = "mcp_missing"
        return {"messages": state["messages"] + [_assistant(MCP_MISSING_REPLY)],
                "last_reply": MCP_MISSING_REPLY, "trace": tr}
    schemas = await llm.safe_call(get_schemas, mcp)   # schema 拉取失败也不裸抛
    if schemas is None:
        tr["fallback"] = "schema_fail"
        return {"messages": state["messages"] + [_assistant(LLM_FAIL_REPLY)],
                "last_reply": LLM_FAIL_REPLY, "trace": tr}
    msgs = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}, *state["messages"]]
    msg = await llm.safe_call(llm.chat, msgs, tools=schemas)
    if msg is None:                                   # 编排断了≠工具失败，不记连败
        tr["fallback"] = "llm_fail"
        return {"messages": state["messages"] + [_assistant(LLM_FAIL_REPLY)],
                "last_reply": LLM_FAIL_REPLY, "trace": tr}
    if not msg.tool_calls:                            # 终答
        reply = msg.content or NO_INFO_REPLY
        return {"messages": state["messages"] + [_assistant(reply)],
                "last_reply": reply, "trace": tr}
    calls = []
    for tc in msg.tool_calls:
        try:
            arguments = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        calls.append({"id": tc.id, "name": tc.function.name, "arguments": arguments})
    msgs.append({"role": "assistant", "content": msg.content or "",
                 "tool_calls": [{"id": c["id"], "type": "function",
                                 "function": {"name": c["name"],
                                              "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                                for c in calls]})
    return {"messages": msgs, "pending_tool_calls": calls, "trace": tr}

def _route_after_tool_llm(state: LGState) -> str:
    return END if state.get("last_reply") else "tool_exec"

async def _tool_exec(state: LGState, config: RunnableConfig) -> dict:
    """工具编排的执行半步：全部过 executor；写操作 → 挂起进 human_gate。"""
    tr = dict(state.get("trace", {}))
    tool_rows = list(tr.get("tool_calls", []))
    msgs = list(state["messages"])
    streak, rounds = state.get("fail_streak", 0), state.get("tool_rounds", 0) + 1
    mcp = _mcp_from(config)
    if mcp is None:
        tr["fallback"] = "mcp_missing"
        return {"messages": msgs + [_assistant(MCP_MISSING_REPLY)],
                "last_reply": MCP_MISSING_REPLY, "trace": tr}

    for tc in state.get("pending_tool_calls", []):
        result = await tool_execute(tc["name"], tc["arguments"], mcp=mcp)
        tool_rows.append({"name": tc["name"], "arguments": tc["arguments"],
                          "status": result.status, "ok": result.ok,
                          "message": result.message})
        msgs.append({"role": "tool", "tool_call_id": tc["id"],
                     "content": json.dumps({"status": result.status, "data": result.data,
                                            "message": result.message}, ensure_ascii=False)})
        if result.status == "needs_confirmation":    # 写操作：不执行，交确认门（D2）
            tr["tool_calls"] = tool_rows
            return {"messages": msgs, "trace": tr, "tool_rounds": rounds,
                    "pending_write": {"name": tc["name"], "arguments": tc["arguments"],
                                      "message": result.message}}
        if result.status != "ok":                    # 连败计数（D7：到顶收敛为诚实回复）
            streak += 1
            if streak >= MAX_TOOL_FAILS:
                tr["tool_calls"] = tool_rows
                tr["fallback"] = "tool_fail_streak"
                reply = "工具连续出错，我先停下来了。请稍后再试，或换个问法。"
                return {"messages": msgs + [_assistant(reply)], "last_reply": reply,
                        "trace": tr, "tool_rounds": rounds, "fail_streak": 0}
        else:
            streak = 0

    if rounds >= MAX_TOOL_ROUNDS:                    # 轮次用尽（D7）
        reply = "这个问题我尝试了几轮都没查成，可能是检索服务不稳，请稍后再试。"
        tr["tool_calls"] = tool_rows
        tr["fallback"] = "tool_rounds_exhausted"
        return {"messages": msgs + [_assistant(reply)], "last_reply": reply,
                "trace": tr, "tool_rounds": rounds, "fail_streak": streak}
    tr["tool_calls"] = tool_rows
    return {"messages": msgs, "trace": tr, "tool_rounds": rounds, "fail_streak": streak}

def _route_after_tool_exec(state: LGState) -> str:
    if state.get("pending_write"):
        return "human_gate"
    return END if state.get("last_reply") else "tool_llm"

def _human_gate(state: LGState, config: RunnableConfig) -> dict:
    """确认门（HITL）裁决半步：interrupt() 挂起等用户，恢复后只做裁决、
    零副作用。必须是 sync 节点：本版本 langgraph 的 async 节点不传 config
    上下文（ASYNCIO_ACCEPTS_CONTEXT 仅 py3.11+ 成立，py3.10 上 interrupt()
    内部的 get_config() 直接炸，自测第 6 组已复现）——确认后的真执行
    拆到 gate_exec，D2 纪律（副作用在裁决之后）等价成立。

    config 上下文注入：langgraph 显式把 config 传给本节点，但 py3.10 上
    它不走 set_config_context（同样只在 py3.11+ 成立），sync 节点被派发
    进线程池时 copy_context() 复制到的上下文里 var_child_runnable_config
    是空的——interrupt() 读不到就炸。这里手动把 langgraph 递进来的 config
    塞进同一个 contextvar（正是 py3.11+ 上 langgraph 自己会做的事），
    py3.11+ 时该注入与框架行为等价、幂等无害，代码可随解释器升级平滑迁移。
    """
    pw = state.get("pending_write")
    tr = dict(state.get("trace", {}))
    if not pw:                                       # 防御：不该到达
        return {"last_reply": "(内部状态异常：没有待确认的写操作)", "pending_write": None}
    show = state.get("gate_hint") or pw.get("message", "")
    tr.update(stage="confirmation")

    _cfg_token = var_child_runnable_config.set(config)
    try:
        ans = interrupt({"reply": show})             # ★ 暂停点：进程重启后仍在此等
    finally:
        var_child_runnable_config.reset(_cfg_token)
    decision = _gate_decision(ans if isinstance(ans, str) else "")
    user_msg = {"role": "user", "content": ans}

    if decision == "reject":
        reply = "好的，已取消本次操作。还有其他可以帮你的吗？"
        tr["gate"] = "reject"
        return {"messages": state["messages"] + [user_msg, _assistant(reply)],
                "last_reply": reply, "pending_write": None, "gate_hint": "", "trace": tr}
    if decision == "unclear":
        reply = "没听清您的意思～ 请回复「确认」执行，或回复「取消」放弃。"
        tr["gate"] = "unclear"
        return {"messages": state["messages"] + [user_msg, _assistant(reply)],
                "last_reply": reply, "gate_hint": reply, "trace": tr}
    # accept：裁决记入状态，真执行拆到 gate_exec
    tr["gate"] = "accept"
    return {"messages": state["messages"] + [user_msg], "gate_decision": "accept",
            "trace": tr}


async def _gate_exec(state: LGState, config: RunnableConfig) -> dict:
    """确认后的真执行半步（async）：带 confirmed=True 重调 executor。"""
    pw = state["pending_write"]
    tr = dict(state.get("trace", {}))
    r = await tool_execute(pw["name"], pw["arguments"],
                           mcp=_mcp_from(config), confirmed=True)
    tr.setdefault("tool_calls", []).append(
        {"name": pw["name"], "arguments": pw["arguments"],
         "status": r.status, "ok": r.ok, "message": r.message})
    reply = _write_done_reply(pw["name"], pw["arguments"]) if r.ok \
        else f"操作没有成功：{r.message}"
    return {"messages": state["messages"] + [_assistant(reply)],
            "last_reply": reply, "pending_write": None, "gate_hint": "", "trace": tr}


def _route_after_gate(state: LGState) -> str:
    """accept → gate_exec；unclear（pending_write 还在）→ 回门口重新
    interrupt；reject（pending_write 已清）→ 结束。"""
    if not state.get("pending_write"):
        return END
    return "gate_exec" if state.get("gate_decision") == "accept" else "human_gate"

async def _chitchat(state: LGState) -> dict:
    tr = {**state.get("trace", {}), "stage": "chitchat"}
    reply = (await llm.safe_call(llm.chat_text,
                                 [{"role": "system", "content": PERSONA},
                                  *state["messages"]], temperature=0.7)) or LLM_FAIL_REPLY
    if reply == LLM_FAIL_REPLY:
        tr["fallback"] = "llm_fail"
    return {"messages": state["messages"] + [_assistant(reply)], "last_reply": reply,
            "trace": tr}

def _unsupported(state: LGState) -> dict:
    """D1/D7：诚实边界——明确超出职责的请求直接说做不到。"""
    return {"messages": state["messages"] + [_assistant(UNSUPPORTED_REPLY)],
            "last_reply": UNSUPPORTED_REPLY,
            "trace": {**state["trace"], "stage": "unsupported"}}

def _discontent(state: LGState) -> dict:
    """D7：连续不满收敛为诚实道歉 + 计数归零（无人工可转，不假装转接）。"""
    return {"messages": state["messages"] + [_assistant(DISCONTENT_REPLY)],
            "last_reply": DISCONTENT_REPLY, "discontent": 0,
            "trace": {**state["trace"], "stage": "discontent",
                      "fallback": "discontent_honest_reply"}}

# ---------------- 图装配 ----------------

def build_graph(checkpointer):
    """装配 StateGraph。checkpointer 必传——interrupt() 依赖它持久化暂停点。"""
    g = StateGraph(LGState)
    for name, fn in (
        ("guard_input", _guard_input), ("ingest", _ingest), ("classify", _classify),
        ("clarify", _clarify), ("recall", _recall), ("qa", _qa),
        ("tool_llm", _tool_llm), ("tool_exec", _tool_exec),
        ("human_gate", _human_gate), ("gate_exec", _gate_exec),
        ("chitchat", _chitchat),
        ("unsupported", _unsupported), ("discontent", _discontent),
    ):
        g.add_node(name, fn)
    g.add_edge(START, "guard_input")
    g.add_edge("guard_input", "ingest")
    g.add_edge("ingest", "classify")
    g.add_conditional_edges("classify", _route_after_classify)
    g.add_edge("clarify", END)
    g.add_edge("recall", "qa")          # D6：暂串行，并行扇出等 join 语义定稿再开
    g.add_edge("qa", END)
    g.add_edge("chitchat", END)
    g.add_edge("unsupported", END)
    g.add_edge("discontent", END)
    g.add_conditional_edges("tool_llm", _route_after_tool_llm)
    g.add_conditional_edges("tool_exec", _route_after_tool_exec)
    g.add_conditional_edges("human_gate", _route_after_gate)
    g.add_edge("gate_exec", END)
    return g.compile(checkpointer=checkpointer)

_default_agent = None

async def get_default_agent(mcp) -> "LgAgent":
    """进程级默认 agent：AsyncSqliteSaver 落盘（断电/重启后确认门仍可续）。

    必须是 async：AsyncSqliteSaver 只支持异步接口（handle 的 aget_state
    走 async 路径，sync 版 SqliteSaver 会直接 NotImplementedError）。
    mcp 必传（D5/D7）：连接生命周期归调用方，本模块不建、不关、不缓存连接。
    """
    global _default_agent
    if _default_agent is None:
        CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(CHECKPOINT_DB)
        _default_agent = LgAgent(build_graph(AsyncSqliteSaver(conn)), mcp)
    return _default_agent

async def aclose_default_agent() -> None:
    """显式关闭默认 agent 的 checkpointer 连接（项目纪律：async 资源
    禁止靠 GC 兑底关闭，进程收尾必须调它）。"""
    global _default_agent
    if _default_agent is not None:
        conn = _default_agent.graph.checkpointer.conn
        _default_agent = None
        await conn.close()

# ---------------- adapter：对 cli/api 暴露 handle()，镜像回 Session（D4） ----------------

class LgAgent:
    """handle(sid, text) → 回复文本。mcp 持有者，经 config 注入每个节点（D5）。"""

    def __init__(self, graph, mcp) -> None:
        self.graph = graph
        self.mcp = mcp

    async def handle(self, session_id: str, text: str) -> str:
        cfg = {"configurable": {"thread_id": session_id, "mcp": self.mcp}}
        paused = bool((await self.graph.aget_state(cfg)).next)   # 暂停中=门在等回答（D1）
        t0 = time.perf_counter()
        if paused:
            result = await self.graph.ainvoke(Command(resume=text), cfg)
        else:
            result = await self.graph.ainvoke({"user_text": text, "sid": session_id}, cfg)

        reply = result.get("last_reply", "")
        intr = result.get("__interrupt__")
        if intr:                                           # 本轮以挂起收尾：回复取 interrupt 载荷
            val = getattr(intr[0], "value", intr[0])
            if isinstance(val, dict):
                reply = str(val.get("reply", "")) or reply

        # D4：镜像给前端/api 的视图层（逻辑真身在 graph state）
        s = store.get(session_id)
        tr = dict(result.get("trace") or {})
        tr["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
        s.trace = tr
        s.history = list(result.get("messages", []))[-MAX_ROUNDS * 2:]
        s.current_paper_id = result.get("focus", "")
        s.pending_write = result.get("pending_write")
        s.tool_fail_streak = result.get("fail_streak", 0)
        s.dissatisfaction = result.get("discontent", 0)
        return reply

# ---------------- 自测：python -m app.agent.lg_graph（离线全桩，不联网） ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    import asyncio
    import tempfile

    from langgraph.checkpoint.memory import MemorySaver

    # ---- 桩件：协议对象 / 假 MCP（call + list_tools 双接口，喂真 executor 与真 registry） ----

    class _Fn:
        pass

    class _F:                                              # 假 tool_calls 协议对象
        def __init__(self, i, name, args):
            self.id, self.content = f"call_{i}", ""
            self.function = _Fn()
            self.function.name, self.function.arguments = name, \
                json.dumps(args, ensure_ascii=False)

    class _Msg:
        def __init__(self, content="", tool_calls=None):
            self.content, self.tool_calls = content, tool_calls

    _STUB_TOOLS = [
        {"name": "search_arxiv", "description": "在 arXiv 检索论文",
         "input_schema": {"type": "object", "title": "SearchArxivArguments",
                          "properties": {"query": {"type": "string", "title": "Query"},
                                         "max_results": {"type": "integer", "title": "Max"}},
                          "required": ["query"]}},
        {"name": "fetch_paper", "description": "拉取并解析论文 PDF",
         "input_schema": {"type": "object", "properties":
                          {"arxiv_id": {"type": "string"}}, "required": ["arxiv_id"]}},
        {"name": "read_paper", "description": "读论文某一部分",
         "input_schema": {"type": "object", "properties":
                          {"paper_id": {"type": "string"},
                           "section": {"type": "string"}},
                          "required": ["paper_id"]}},
        {"name": "list_library", "description": "列出文献库",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "add_to_library", "description": "加入文献库（写）",
         "input_schema": {"type": "object", "properties":
                          {"arxiv_id": {"type": "string"},
                           "reason": {"type": "string"}},
                          "required": ["arxiv_id", "reason"]}},
        {"name": "save_note", "description": "写笔记（写）",
         "input_schema": {"type": "object", "properties":
                          {"paper_id": {"type": "string"},
                           "text": {"type": "string"}},
                          "required": ["paper_id", "text"]}},
        {"name": "remove_paper", "description": "移出文献库（写）",
         "input_schema": {"type": "object", "properties":
                          {"paper_id": {"type": "string"}}, "required": ["paper_id"]}},
    ]

    class _FakeMcp:
        """mcp_client 的桩：list_tools 喂真 registry，call 喂真 executor。"""

        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        async def list_tools(self):
            return _STUB_TOOLS

        async def call(self, name: str, args: dict) -> tuple[bool, dict]:
            self.calls.append((name, dict(args)))
            if name == "search_arxiv":
                return True, {"results": [{"arxiv_id": "2401.12345",
                                           "title": "RAG 评测综述"}], "count": 1}
            if name == "add_to_library":
                if str(args.get("arxiv_id", "")).startswith("24"):
                    return True, {"arxiv_id": args["arxiv_id"], "created": True}
                return False, {"error": "arXiv 上找不到该 ID"}
            return False, {"error": f"Unknown tool: {name}"}

    async def _fake_chat_text(*a, **k):
        return "你好！我是 PaperPilot，本地论文助手。"

    async def _selftest() -> None:
        clear_cache()   # 教训：缓存测试必须先清空，否则历史命中造成假通过
        saved = {"classify": globals()["classify"], "chat": llm.chat,
                 "chat_text": llm.chat_text}

        def _restore():
            globals()["classify"] = saved["classify"]
            llm.chat = saved["chat"]
            llm.chat_text = saved["chat_text"]

        def _stub_classify(intent, confidence, dissatisfied=False, slots=None):
            async def _c(text, history=None, current_paper_id=""):
                return IntentResult(intent, confidence, dissatisfied, slots or {})
            return _c

        # ---- 1. 低置信澄清 ----
        try:
            globals()["classify"] = _stub_classify("chitchat", 0.2)
            a1 = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            r1 = await a1.handle("t1", "那个……")
        finally:
            _restore()
        assert "确认" in r1 and "文献库" in r1, r1
        tr1 = a1.graph.get_state({"configurable": {"thread_id": "t1"}}).values["trace"]
        assert tr1["fallback"] == "low_confidence_clarify"
        print("1 低置信澄清:", r1[:20] + "…")

        # ---- 2. unsupported：诚实边界 ----
        try:
            globals()["classify"] = _stub_classify("unsupported", 0.9)
            a2 = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            r2 = await a2.handle("t2", "帮我写一份小红书文案")
        finally:
            _restore()
        assert "超出了我的能力" in r2, r2
        print("2 unsupported 诚实回复:", r2[:16] + "…")

        # ---- 3. qa 落桩：recall→qa 串行，两个桩都留痕 ----
        try:
            globals()["classify"] = _stub_classify("qa", 0.95,
                                                   slots={"paper_id": "2401.12345"})
            a3 = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            r3 = await a3.handle("t3", "2401.12345 的实验部分讲了什么")
        finally:
            _restore()
        assert "检索层" in r3, r3
        tr3 = a3.graph.get_state({"configurable": {"thread_id": "t3"}}).values["trace"]
        assert tr3["stage"] == "qa" and tr3["fallback"] == "retrieval_not_wired"
        assert tr3["memory"]["status"] == "stub", "recall 桩必须先跑并留痕"
        assert tr3["slots"]["paper_id"] == "2401.12345"
        print("3 qa 落桩: recall(memory=stub) → qa(retrieval_not_wired)，占位回复")

        # ---- 4. chitchat（stub chat_text） ----
        try:
            llm.chat_text = _fake_chat_text
            globals()["classify"] = _stub_classify("chitchat", 0.9)
            a4 = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            r4 = await a4.handle("t4", "你好呀")
        finally:
            _restore()
        assert "PaperPilot" in r4, r4
        print("4 chitchat:", r4[:22] + "…")

        # ---- 5. 工具环：LLM 决定调 search_arxiv → 真 executor + 桩 MCP → 终答 ----
        phase = {"n": 0}

        async def _chat_two_phase(*args, **kwargs):
            phase["n"] += 1
            if phase["n"] == 1:
                return _Msg("", [_F(1, "search_arxiv",
                                    {"query": "RAG evaluation", "max_results": 3})])
            return _Msg("找到 1 篇相关论文：RAG 评测综述（2401.12345）。")

        clear_cache()
        try:
            llm.chat = _chat_two_phase
            globals()["classify"] = _stub_classify("search", 0.95,
                                                   slots={"query": "RAG evaluation"})
            mcp5 = _FakeMcp()
            a5 = LgAgent(build_graph(MemorySaver()), mcp5)
            r5 = await a5.handle("t5", "帮我搜几篇 RAG 评测的论文")
        finally:
            _restore()
        assert "2401.12345" in r5, r5
        tr5 = a5.graph.get_state({"configurable": {"thread_id": "t5"}}).values["trace"]
        assert tr5["stage"] == "tool" and len(tr5["tool_calls"]) == 1
        assert tr5["tool_calls"][0]["status"] == "ok"
        assert ("search_arxiv", {"query": "RAG evaluation", "max_results": 3}) \
            in mcp5.calls, mcp5.calls
        print(f"5 工具环: 2 轮编排，MCP 真被调 {len(mcp5.calls)} 次，终答回传")

        # ---- 6. 确认门全流程：interrupt 挂起 → accept/reject/unclear ----
        async def _chat_write(*a, **k):
            return _Msg("", [_F(1, "add_to_library",
                                {"arxiv_id": "2401.12345", "reason": "与选题相关"})])

        try:
            llm.chat = _chat_write
            globals()["classify"] = _stub_classify("library", 0.95,
                                                   slots={"paper_id": "2401.12345"})
            a6 = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            r6 = await a6.handle("t6", "把 2401.12345 加入文献库")
            assert "确认入库吗" in r6, r6                     # 首轮以 interrupt 挂起
            assert a6.graph.get_state({"configurable": {"thread_id": "t6"}}).next
            r6b = await a6.handle("t6", "确认")                # resume=确认 → 真执行
            assert "已把论文 2401.12345 加入文献库" in r6b, r6b
            tr6 = a6.graph.get_state({"configurable": {"thread_id": "t6"}}).values["trace"]
            assert tr6["gate"] == "accept" and tr6["stage"] == "confirmation"
            assert not a6.graph.get_state({"configurable": {"thread_id": "t6"}}).next
            print("6a 确认门-accept:", r6b[:24] + "…")

            a6r = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            await a6r.handle("t6r", "把 2401.12345 加入文献库")
            r6r = await a6r.handle("t6r", "算了不要了")
            assert "取消" in r6r, r6r
            print("6b 确认门-reject:", r6r[:18] + "…")

            a6u = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            await a6u.handle("t6u", "把 2401.12345 加入文献库")
            r6u = await a6u.handle("t6u", "啊？啥意思")        # unclear：门不丢（D3）
            assert "没听清" in r6u \
                and a6u.graph.get_state({"configurable": {"thread_id": "t6u"}}).next
            r6u2 = await a6u.handle("t6u", "确认")
            assert "加入文献库" in r6u2, r6u2
            print("6c 确认门-unclear→accept: 门保持挂起，后续仍可确认")
        finally:
            _restore()

        # ---- 7. ★ 断点续跑：SqliteSaver 落盘，换图实例（模拟进程重启）仍能续 ----
        # Windows 教训：sqlite 文件被打开时删不掉——连接用完必须 close。
        try:
            llm.chat = _chat_write
            globals()["classify"] = _stub_classify("library", 0.95,
                                                   slots={"paper_id": "2401.12345"})
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
                db = Path(d) / "cp.sqlite3"
                con1 = await aiosqlite.connect(db)
                g1 = build_graph(AsyncSqliteSaver(con1))
                a7 = LgAgent(g1, _FakeMcp())
                await a7.handle("t7", "把 2401.12345 加入文献库")
                assert (await g1.aget_state({"configurable": {"thread_id": "t7"}})).next
                del g1, a7                                     # 模拟进程死掉
                await con1.close()                             # 文件句柄放手，目录才删得掉
                con2 = await aiosqlite.connect(db)
                g2 = build_graph(AsyncSqliteSaver(con2))
                assert (await g2.aget_state({"configurable": {"thread_id": "t7"}})).next, \
                    "换实例后暂停点应仍在（checkpointer 持久化）"
                a7b = LgAgent(g2, _FakeMcp())
                r7 = await a7b.handle("t7", "确认")             # 新进程里 resume
                assert "加入文献库" in r7, r7
                await con2.close()
        finally:
            _restore()
        print("7 断点续跑: 进程重启后 interrupt 暂停点仍在，resume 成功")

        # ---- 8. 连续不满×2 → 诚实道歉（无人工可转） ----
        try:
            llm.chat_text = _fake_chat_text
            globals()["classify"] = _stub_classify("chitchat", 0.9, dissatisfied=True)
            a8 = LgAgent(build_graph(MemorySaver()), _FakeMcp())
            await a8.handle("t8", "怎么又超时了")
            r8 = await a8.handle("t8", "这破工具真难用")
        finally:
            _restore()
        assert "能力确实有限" in r8, r8
        tr8 = a8.graph.get_state({"configurable": {"thread_id": "t8"}}).values["trace"]
        assert tr8["stage"] == "discontent" \
            and tr8["fallback"] == "discontent_honest_reply"
        print("8 连续不满×2:", r8[:20] + "…")

        # ---- 9. 契约：history 键白名单；Session 镜像完整 ----
        for sid, ag in (("t3", a3), ("t6", a6)):
            msgs = ag.graph.get_state({"configurable": {"thread_id": sid}}).values["messages"]
            assert all(set(m) <= {"role", "content", "tool_calls", "tool_call_id"}
                       for m in msgs), msgs
        s = store.get("t6")
        assert s.trace.get("gate") == "accept" and "elapsed_ms" in s.trace
        assert any(m["content"] == "确认" for m in s.history)
        assert s.current_paper_id == "2401.12345", "锚点应镜像回 Session"
        assert s.pending_write is None and s.tool_fail_streak == 0
        assert s.dissatisfaction == 0
        print("9 契约: messages 键白名单 / Session 镜像（trace+history+锚点+门状态）完整")

        await llm.aclose()   # 趁循环活着关连接池（Windows 教训：GC 兜底会撒噪音）

    asyncio.run(_selftest())
    print("\nlg_graph 自测 9/9 通过（离线全桩：无 key、无真实 MCP 子进程）")