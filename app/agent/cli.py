"""聊天入口：命令行 REPL（M2 收尾接线）。

模块速览：
    main()              整个 Agent 的最终用户入口（python -m app.agent.cli）

设计决策（继承 ShopMate cli.py 母本，编号对齐）：
    D1 会话 id 跟进程走：每次启动生成新 session_id（与 graph thread_id 同源）。
       注意实话：CLI 的 sid 不持久，进程重启后 sqlite checkpointer 里挂起的
       确认门虽然还在等（断点续跑能力由 lg_graph 自测第 7 组保证），但 CLI
       拿不到旧 sid、无法 resume——后续接 sid 持久化再兑现这最后一段。
    D2 命令不进状态机：/exit /new /history /trace 在 REPL 层拦截，业务命令
       混进对话历史会污染意图识别的上下文。
    D3 连接生命周期归本进程（lg_graph D5/D7 的对偶「谁持有谁关」）：cli 用
       async with 持有 McpClient 并经 get_default_agent 注入；进程收尾在
       finally 里按「mcp（async with）→ checkpointer（aclose_default_agent）
       → llm 连接池（llm.aclose）」显式关闭，全部趁事件循环活着做完——
       Windows 上靠 GC 兜底关 async 资源必撒 'Event loop is closed'。
       放 finally 而不是循环后：单轮对话异常（LLM/图内部）不该漏关资源。
    D4 async REPL：handle() 是 async，input() 阻塞事件循环无害（等输入时
       本来就没活干），不为它引入线程池（项目纪律：不倒退回同步）。
    D5 trace_view 未移植：/trace 暂用 JSON 原样打印 trace（D5 纪律：它是
       报告不是状态），Streamlit 侧栏接入时再统一换成 render_text。
"""
import asyncio
import json
import random
import sys
import time

from ..llm import client as llm
from ..mcp_client import McpClient
from .lg_graph import aclose_default_agent, get_default_agent
from .session import store

def _new_sid() -> str:
    return f"s-{time.strftime('%H%M%S')}-{random.randint(100, 999)}"

async def _amain() -> None:
    try:
        async with McpClient() as mcp:
            agent = await get_default_agent(mcp)
            sid = _new_sid()
            print("=" * 56)
            print("PaperPilot 论文助手  "
                  "(exit 退出 / new 新会话 / history 看记忆 / trace 看本轮内部状态)")
            print("=" * 56)
            while True:
                try:
                    text = input("\n你: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n再见！")
                    break
                if not text:
                    continue
                if text in ("/exit", "exit", "退出"):
                    print("再见！")
                    break
                if text in ("/new", "new"):                  # 换会话 = 全新上下文
                    sid = _new_sid()
                    print(f"(已开启新会话 {sid})")
                    continue
                if text in ("/history", "history"):
                    h = store.get(sid).history
                    print(f"(共 {len(h) // 2} 轮)")
                    for m in h:
                        print(f"  [{m['role']}] {m['content'][:60]}")
                    continue
                if text in ("/trace", "trace"):
                    print(json.dumps(store.get(sid).trace,
                                     ensure_ascii=False, indent=2))
                    continue
                try:
                    reply = await agent.handle(sid, text)
                except Exception as e:                       # 单轮故障不杀 REPL
                    print(f"\nPilot: (这一轮内部出错了，先回答不了："
                          f"{type(e).__name__}: {e})")
                    continue
                print(f"\nPilot: {reply}")
    finally:
        await aclose_default_agent()
        await llm.aclose()

def main() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:                                # 处理中途 Ctrl+C
        print("\n再见！")

if __name__ == "__main__":
    main()