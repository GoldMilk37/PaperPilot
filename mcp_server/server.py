"""MCP Server 装配：7 工具 + 2 资源 + 3 prompt + transport 开关（W7 核心）。

设计决策：
    S1 能力在此暴露，策略不在此发生：工具只是把 arxiv_api / pdf_parse /
       library 的能力翻译成 schemas 契约的 dict（方案 A）。确认门 / 身份 /
       超时 / 三态全在 client 侧 executor（library.py L3、计划 §六.1）。
    S2 输入校验靠签名注解：MCPServer 按注解生成 input_schema 并在调用前
       校验，server 侧不写手工业 isinstance 防护。reason 无默认值 = schema
       必填 = LLM 不传理由连调用都过不了，天然强制（§六.4 第 3 层的原料）。
    S3 数据目录 = 环境变量 PAPERPILOT_DATA_DIR，缺省 <项目根>/data：本地与
       容器同一开关，compose 命名卷只挂它。
    S4 fetch 产物落盘缓存（parsed/{paper_id}.json）：read_paper 不联网不重
       解析——token 经济学与引用可追溯都建立在"分节结果稳定"上。
    S5 transport 开关走 argv：--serve --transport stdio|streamable-http
       （http 是 M5 的口子，工具逻辑一份、传输两态）。
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server import MCPServer

from . import arxiv_api, pdf_parse
from .library import Library, arxiv_to_paper_id

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PAPERPILOT_DATA_DIR", ROOT / "data"))
_PARSED_DIR = DATA_DIR / "parsed"

server = MCPServer("paperpilot-mcp")
library = Library(DATA_DIR / "library")

def _parsed_path(paper_id: str) -> Path:
    return _PARSED_DIR / f"{arxiv_to_paper_id(paper_id)}.json"

def _locate_meta(arxiv_id: str):
    """给一个 arxiv_id 找 PaperMeta：库里有直接用；没有检索一次定位。

    库外定位必须走 search 而不是拼 URL：PaperMeta 的 title/authors 只能
    来自 API 返回，手拼的字段没有产地。
    """
    entry = library.get_entry(arxiv_id)
    if entry is not None:
        return entry.meta
    base = arxiv_to_paper_id(arxiv_id)
    hits = arxiv_api.search(base, 1)
    if not hits or arxiv_to_paper_id(hits[0].arxiv_id) != base:
        raise RuntimeError(f"arXiv 上找不到 {arxiv_id}")
    return hits[0]

# ================= 工具（7 = 4 读 + 3 写） =================

@server.tool()
def search_arxiv(query: str, max_results: int = 5) -> dict:
    """按关键词检索 arXiv，最新优先。返回列表含摘要（已标记外部不可信）。"""
    query = query.strip()
    if not query:
        raise RuntimeError("检索词为空")
    papers = arxiv_api.search(query, max(1, min(max_results, 20)))
    return {"papers": [p.model_dump() for p in papers], "total": len(papers)}

@server.tool()
def fetch_paper(arxiv_id: str) -> dict:
    """下载并解析一篇论文（PDF→分节），结果缓存。重复 fetch 直接复用本地 PDF。"""
    meta = _locate_meta(arxiv_id.strip())
    pdf_path = arxiv_api.download_pdf(meta.arxiv_id, library.pdfs_dir)
    doc = pdf_parse.parse_pdf(pdf_path, meta.arxiv_id, title=meta.title)
    _PARSED_DIR.mkdir(parents=True, exist_ok=True)
    _parsed_path(meta.arxiv_id).write_text(doc.model_dump_json(), encoding="utf-8")   # S4
    return doc.model_dump()

def _read_paper_impl(paper_id: str, section: int | None = None) -> dict:
    path = _parsed_path(paper_id)
    if not path.exists():
        raise RuntimeError(f"{paper_id} 尚未 fetch，请先调 fetch_paper")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if section is None:
        return doc
    for s in doc["sections"]:
        if s["idx"] == section:
            doc["sections"] = [s]
            return doc
    raise RuntimeError(f"{paper_id} 没有第 {section} 节（共 {len(doc['sections'])} 节）")

@server.tool()
def read_paper(paper_id: str, section: int | None = None) -> dict:
    """读取已 fetch 论文的分节正文；给 section（节号）则只返回该节。"""
    return _read_paper_impl(paper_id, section)

@server.tool()
def list_library() -> dict:
    """列出本地文献库全部条目。"""
    entries = library.list_entries()
    return {"entries": [e.model_dump() for e in entries], "count": len(entries)}

@server.tool()
def add_to_library(arxiv_id: str, reason: str) -> dict:
    """把论文加入文献库（写工具；确认门在 client 侧 executor）。

    reason 必填（S2：schema required），入库后持久化为 added_reason。
    幂等：已在库返回既有条目 + created=False。
    """
    reason = reason.strip()
    if not reason:
        raise RuntimeError("入库理由不能为空")
    entry = library.get_entry(arxiv_id)
    if entry is None:
        entry, _ = library.add_paper(_locate_meta(arxiv_id), reason)
        created = True
    else:
        created = False
    return {**entry.model_dump(), "created": created}

@server.tool()
def save_note(paper_id: str, text: str) -> dict:
    """给库内论文写一条笔记（写工具）。论文必须已入库。"""
    if library.get_entry(paper_id) is None:
        raise RuntimeError(f"{paper_id} 不在库里；笔记只对已入库论文有意义")
    return library.save_note(paper_id, text).model_dump()

@server.tool()
def remove_paper(paper_id: str) -> dict:
    """从文献库移除论文，级联清理笔记、本地 PDF 与解析缓存（写工具）。"""
    existed = library.remove_paper(paper_id)
    if not existed:
        raise RuntimeError(f"库里没有 {paper_id}，无需移除")
    _parsed_path(paper_id).unlink(missing_ok=True)
    return {"removed": arxiv_to_paper_id(paper_id)}

# ================= 资源（2） =================

@server.resource("library://papers")
def library_papers() -> str:
    """文献库全部条目（JSON）。"""
    entries = library.list_entries()
    return json.dumps({"entries": [e.model_dump() for e in entries],
                       "count": len(entries)}, ensure_ascii=False)

@server.resource("paper://{paper_id}/fulltext")
def paper_fulltext(paper_id: str) -> str:
    """一篇已 fetch 论文的分节全文（JSON）。"""
    return json.dumps(_read_paper_impl(paper_id), ensure_ascii=False)

# ================= prompt（3，对齐计划 §四：summarize/compare/critique） =================

@server.prompt()
def summarize(paper_id: str) -> str:
    """总结一篇论文的核心贡献。"""
    return (f"请总结论文 {paper_id} 的核心贡献、方法与结论。"
            f"先逐节 read_paper 阅读再总结，每个要点标注来源节号。")

@server.prompt()
def compare(paper_id_a: str, paper_id_b: str) -> str:
    """对比两篇论文。"""
    return (f"请对比论文 {paper_id_a} 与 {paper_id_b} 的问题设定、方法差异与"
            f"各自局限。先分别 read_paper 再对比，引用节号。")

@server.prompt()
def critique(paper_id: str) -> str:
    """以审稿人视角批判性审视一篇论文。"""
    return (f"请以审稿人视角审视论文 {paper_id}：方法是否有效、实验是否充分、"
            f"结论是否被证据支持。先 read_paper 阅读全文再评估，引用节号。")

# ================= 入口 =================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="启动 MCP 服务（不传则跑自测）")
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not args.serve:
        _selftest()
        return
    if args.transport == "stdio":
        server.run(transport="stdio")                    # S5
    else:                                                # M5 的口子
        server.run(transport="streamable-http", host=args.host, port=args.port)

# ================= 自测：python -m mcp_server.server（全程离线） =================
# 子进程用 python -m 启动（脚本直跑会让相对导入炸），PYTHONPATH 钉死项目根；
# DATA_DIR 指向 tempdir——自测不碰真实 data/。

async def _selftest_async(tmp: str) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable,                          # 教训 §三之二.A：绝不写 "python"
        args=["-m", "mcp_server.server", "--serve"],
        env=dict(os.environ, PAPERPILOT_DATA_DIR=tmp, PYTHONPATH=str(ROOT)),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1 工具齐全 + input_schema 非空（§三之二.D：声明为空不报错，必须断言）
            tools = (await session.list_tools()).tools
            names = {t.name for t in tools}
            assert len(tools) == 7, f"应注册 7 个工具，实际 {names}"
            expected_params = {
                "search_arxiv": {"query", "max_results"},
                "fetch_paper": {"arxiv_id"},
                "read_paper": {"paper_id", "section"},
                "list_library": set(),
                "add_to_library": {"arxiv_id", "reason"},
                "save_note": {"paper_id", "text"},
                "remove_paper": {"paper_id"},
            }
            for t in tools:
                props = set(t.input_schema.get("properties", {}))
                assert props == expected_params[t.name], \
                    f"{t.name}: 参数 {props} != 预期 {expected_params[t.name]}"
            add_schema = next(t for t in tools if t.name == "add_to_library")
            assert "reason" in add_schema.input_schema.get("required", []), "reason 必须必填（S2）"
            print("1 工具注册 OK：7 个齐全，schema 全非空，reason 必填")

            # 2 list_library 空库形状 = 契约
            res = await session.call_tool("list_library", {})
            assert not res.is_error
            assert json.loads(res.content[0].text) == {"entries": [], "count": 0}
            print("2 list_library OK：空库形状符合契约 {entries, count}")

            # 3 save_note 守卫 → is_error（§三之二.B：不抛异常，必须显式查）
            res = await session.call_tool("save_note", {"paper_id": "9999.99999", "text": "x"})
            assert res.is_error, "对不在库的论文写笔记必须报错"
            print("3 save_note 守卫 OK：库外论文被拒，is_error=True")

            # 4 resource 可读
            rres = await session.read_resource("library://papers")
            assert json.loads(rres.contents[0].text)["count"] == 0
            print("4 resource OK：library://papers 可读且为合法 JSON")

            # 5 prompt 可取
            pres = await session.get_prompt("summarize", {"paper_id": "2401.01234"})
            assert pres.messages and pres.messages[0].content.text
            print("5 prompt OK：summarize 模板非空")

def _selftest() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_selftest_async(tmp))
    print("\nserver 自测 5/5 通过（全程离线，数据落在 tempdir）")

if __name__ == "__main__":
    main()