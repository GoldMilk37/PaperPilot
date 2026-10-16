"""工具声明层：从 MCP server 内省（list_tools）得到喂 LLM 的工具声明（M2 第 3 步）。

模块速览：
    load_definitions(mcp)   -> {工具名: {name, description, parameters}}
    get_schemas(mcp)        -> OpenAI function-calling 格式（给 LLM tools 参数 / system prompt）

四个设计决策：
    R1 声明只写一处（内省化）：ShopMate 用 data/tools/*.json 双写声明，靠
       加载校验兜底防漂；MCP 下声明与实现合为同一处——server 函数签名是
       唯一产地，list_tools 拿到的 input_schema 是 MCPServer 按注解实时
       生成的，"JSON 定义与实现两边漂"从结构上被消灭。函数全 async 且
       显式收 mcp 实例：与 executor 的 D7 同款纪律，谁持有连接谁传入。
       每轮一次 list_tools 对 stdio 开销可忽略；将来 http transport 变贵
       了，缓存决策归上层 runtime，这里不做。
    R2 schema 清洗（剥 pydantic 工件）：input_schema 带两类生成物——顶层
       title（"saveNoteArguments"，泄漏 Python 命名）和每个属性的 title，
       喂 LLM 是纯 token 噪音。计划 §三之二.D 说"多余键是否被提供商容忍
       要实测不要假设"——清洗让这个问题直接消失：确定性操作、可离线
       断言，不赌任何提供商的宽容度。关键边界：properties 里的**键**是
       参数名，绝不能当工件删——论文域工具随时可能出现叫 title 的参数
       （论文标题），_strip_titles 只删注解层的 title，放过 properties 的键。
    R3 加载即校验（继承 ShopMate D3）：name 重复、parameters 非 object、
       required 不在 properties 里，加载时抛错。内省产物理论上不会坏，
       但校验成本为零——坏结构死在入口，而不是死在 LLM 的一次失败调用里。
    R4 这里没有 is_write_op：写操作清单 WRITE_OPS 归 executor（questions ⑧：
       权限表与能力声明来源必须分开）。ShopMate 的"未知工具按写操作"保守
       判定随之取消——未知工具在 mcp.call 层收到 is_error，职责已移交。

用法：from app.tools.registry import get_schemas
命令行：python -m app.tools.registry（跑内置自测，前 4 组离线，第 5 组真 server 子进程）
"""

def _strip_titles(node):
    """R2：递归剥注解层 title。properties 的键是参数名，整层保留、只递归其值。"""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "title":
                continue
            if k == "properties" and isinstance(v, dict):
                out[k] = {pk: _strip_titles(pv) for pk, pv in v.items()}
            else:
                out[k] = _strip_titles(v)
        return out
    if isinstance(node, list):
        return [_strip_titles(x) for x in node]
    return node

def _validate(name: str, schema) -> dict:
    """R3：坏 schema 死在入口。返回清洗后的 parameters。"""
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError(f"{name}: input_schema 不是 object 型")
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise ValueError(f"{name}: properties 不是 dict")
    for req in schema.get("required", []):
        if req not in props:
            raise ValueError(f"{name}: required 字段 '{req}' 不在 properties 里")
    return _strip_titles(schema)

async def load_definitions(mcp) -> dict[str, dict]:
    """R1：内省全部工具 -> {name: {name, description, parameters}}（已清洗）。"""
    tools = await mcp.list_tools()
    defs: dict[str, dict] = {}
    for t in tools:
        if not t["name"]:
            raise ValueError("工具名为空")
        if t["name"] in defs:
            raise ValueError(f"工具名重复: {t['name']}")
        defs[t["name"]] = {
            "name": t["name"],
            "description": t.get("description") or "",
            "parameters": _validate(t["name"], t["input_schema"]),
        }
    return defs

async def get_schemas(mcp) -> list[dict]:
    """OpenAI tools 参数格式：[{"type": "function", "function": {name, description, parameters}}]。"""
    return [{"type": "function",
             "function": {"name": d["name"], "description": d["description"],
                          "parameters": d["parameters"]}}
            for d in (await load_definitions(mcp)).values()]

# ---------------- 自测：python -m app.tools.registry ----------------

def _stub_tools() -> list[dict]:
    """7 个工具的 pydantic 风格桩：带 title 工件，参数清单照 server.py 真实签名。"""
    spec = {
        "search_arxiv": (["query", "max_results"], ["query"]),
        "fetch_paper": (["arxiv_id"], ["arxiv_id"]),
        "read_paper": (["paper_id", "section"], ["paper_id"]),
        "list_library": ([], []),
        "add_to_library": (["arxiv_id", "reason"], ["arxiv_id", "reason"]),
        "save_note": (["paper_id", "text"], ["paper_id", "text"]),
        "remove_paper": (["paper_id"], ["paper_id"]),
    }
    return [{"name": name, "description": f"{name} 的描述",
             "input_schema": {"type": "object", "title": f"{name}Arguments",
                              "properties": {p: {"type": "string", "title": p.capitalize()}
                                             for p in props},
                              "required": req}}
            for name, (props, req) in spec.items()]

class _StubMcp:
    async def list_tools(self):
        return _stub_tools()

class _BadMcp:
    def __init__(self, tools):
        self._tools = tools

    async def list_tools(self):
        return self._tools

def _has_title(x) -> bool:
    if isinstance(x, dict):
        return any(k == "title" or _has_title(v) for k, v in x.items())
    if isinstance(x, list):
        return any(_has_title(v) for v in x)
    return False

async def _selftest_async() -> None:
    expected = {"search_arxiv", "fetch_paper", "read_paper", "list_library",
                "add_to_library", "save_note", "remove_paper"}

    # 1 内省 + 清洗：7 个工具，工件剥净
    defs = await load_definitions(_StubMcp())
    assert set(defs) == expected, set(defs)
    assert all(not _has_title(d["parameters"]) for d in defs.values())
    print("1 内省清洗 OK：7 个工具，title 工件剥净")

    # 2 get_schemas：OpenAI 格式，required 保留
    schemas = await get_schemas(_StubMcp())
    assert len(schemas) == 7
    for s in schemas:
        assert s["type"] == "function"
        assert set(s["function"]) == {"name", "description", "parameters"}
    add = next(s for s in schemas if s["function"]["name"] == "add_to_library")
    assert add["function"]["parameters"]["required"] == ["arxiv_id", "reason"]
    print("2 get_schemas OK：OpenAI 格式，required 保留")

    # 3 title 逃生舱：参数名叫 title 的不能被误删，注解仍要剥
    probe = _strip_titles({"type": "object", "title": "X",
                           "properties": {"title": {"type": "string", "title": "Title"}}})
    assert "title" in probe["properties"], "参数名叫 title 的被误删了"
    assert "title" not in probe["properties"]["title"], "参数自己的注解没剥"
    assert "title" not in probe, "顶层注解没剥"
    print("3 title 逃生舱 OK：参数名保留 / 注解剥净")

    # 4 坏 schema 死在入口：非 object / required 越界 / 重名
    for bad, why in [
        ([{"name": "x", "description": "", "input_schema": {"type": "array"}}], "非 object"),
        ([{"name": "y", "description": "", "input_schema": {"type": "object",
                                                            "properties": {}, "required": ["a"]}}],
         "required 越界"),
        (_stub_tools()[:1] * 2, "重名"),
    ]:
        try:
            await load_definitions(_BadMcp(bad))
            raise AssertionError(f"{why} 应抛错")
        except ValueError:
            pass
    print("4 校验 OK：非 object / required 越界 / 重名 全部拦截")

    # 5 真内省：真 server 子进程（§三之二.D "实测不要假设"的落点）
    import tempfile
    from pathlib import Path
    from ..mcp_client import McpClient
    with tempfile.TemporaryDirectory() as tmp:
        async with McpClient(data_dir=Path(tmp)) as mcp:
            real = await load_definitions(mcp)
            real_schemas = await get_schemas(mcp)
    assert set(real) == expected, set(real)
    assert all(not _has_title(d["parameters"]) for d in real.values())
    add_fn = next(s["function"] for s in real_schemas
                  if s["function"]["name"] == "add_to_library")
    assert "reason" in add_fn["parameters"].get("required", [])
    print("5 真内省 OK：真 server 子进程 7 工具，工件剥净，reason 必填")

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    import asyncio
    asyncio.run(_selftest_async())
    print("\nregistry 自测 5/5 通过（第 5 组含真 stdio 子进程）")