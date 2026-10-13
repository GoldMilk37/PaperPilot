"""MCP client 适配层：对上只暴露 list_tools / call，对下封装 mcp.Client（M1 第 6 步）。

设计决策：
    C1 适配器模式：底层选一等公民 mcp.Client（transport 差异被吃掉，M5 换
       http 只改构造参数）；对上只暴露最小面。换 SDK 是本文件内的实现细节，
       executor 永远不知道也不关心底层用哪套 API。
    C2 sys.executable 纪律：stdio 子进程命令绝不写 "python"（§三之二.A 两颗雷）。
    C3 三态映射雏形：call -> (ok, data)。is_error=True → ok=False，data 带
       error 文本（SDK 官方语义：is_error 是"作为结果返回给 LLM"，不是协议
       层错误——后者走异常，M3 executor 里对应第三态）；正常 → ok=True，
       data = structured_content（方案 A：即载荷，无解包）。
    C4 数据目录透传：data_dir 参数注入 PAPERPILOT_DATA_DIR，自测与容器共用
       同一开关（server 侧 S3 的对偶）。
"""
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]

class McpClient:
    def __init__(self, transport: str = "stdio", url: str | None = None,
                 data_dir: Path | None = None, read_timeout: float = 30.0):
        if transport == "stdio":
            env = dict(os.environ)
            if data_dir is not None:
                env["PAPERPILOT_DATA_DIR"] = str(data_dir)      # C4
            self._client = Client(
                server=StdioServerParameters(
                    command=sys.executable,                     # C2：绝不写 "python"
                    args=["-m", "mcp_server.server", "--serve"],
                    env=dict(env, PYTHONPATH=str(ROOT)),
                ),
                read_timeout_seconds=read_timeout,
            )
        elif transport == "http":
            if not url:
                raise ValueError("http transport 需要 url")
            self._client = Client(server=url)                   # M5 的口子
        else:
            raise ValueError(f"未知 transport: {transport}")

    async def __aenter__(self) -> "McpClient":
        self._client = await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.__aexit__(*exc)

    async def list_tools(self) -> list[dict]:
        """工具清单：[{name, description, input_schema}]（给 system prompt 用）。"""
        res = await self._client.list_tools()
        return [{"name": t.name, "description": t.description or "",
                 "input_schema": t.input_schema} for t in res.tools]

    async def call(self, name: str, args: dict) -> tuple[bool, dict]:
        """C3：三态映射雏形。返回 (ok, data)，data 直接可填 ToolResult.data。"""
        res = await self._client.call_tool(name, args)
        if res.is_error:
            text = res.content[0].text if res.content else "(server 未返回错误详情)"
            return False, {"error": text}
        data = res.structured_content
        if data is None:                                        # 老式 server 只回 text 的兜底
            text = res.content[0].text if res.content else ""
            import json
            try:
                data = json.loads(text)
            except (ValueError, TypeError):
                data = {"text": text}
        return True, data if isinstance(data, dict) else {"result": data}

# ---------------- 自测：python -m app.mcp_client（真 server 子进程，数据落 tempdir） ----------------
async def _selftest_async(tmp: str) -> None:
    async with McpClient(data_dir=Path(tmp)) as c:
        tools = await c.list_tools()
        assert len(tools) == 7, f"应见 7 个工具，实际 {[t['name'] for t in tools]}"
        assert all(t["input_schema"].get("type") == "object" for t in tools)
        print("1 list_tools OK：7 个工具，schema 齐全")

        ok, data = await c.call("list_library", {})
        assert ok is True and data == {"entries": [], "count": 0}, data
        print("2 call(list_library) OK：structured_content 即载荷（方案 A 直通）")

        ok, data = await c.call("save_note", {"paper_id": "9999.99999", "text": "x"})
        assert ok is False and "error" in data, (ok, data)
        print("3 call(save_note) OK：is_error → (False, {'error': ...}) 三态映射")

        ok, data = await c.call("remove_paper", {"paper_id": "不存在"})
        assert ok is False
        print("4 错误路径复验 OK：诚实失败走 is_error 通道")

if __name__ == "__main__":
    import asyncio
    import tempfile

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_selftest_async(tmp))
    print("\nmcp_client 自测 4/4 通过（真 stdio 子进程，数据落 tempdir）")