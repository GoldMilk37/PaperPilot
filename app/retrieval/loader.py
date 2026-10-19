"""语料装载：data/library + data/parsed → RawDoc 列表（M3 第 1 步）。

四个设计决策：
    D1 数据单一事实源：只读 mcp_server 侧落盘的两处——data/library
       （papers.jsonl + notes/）与 data/parsed（fetch_paper 的解析缓存）。
       检索是只读消费方，不复制数据也不反向写：写路径仍归 MCP server
       （能力在 server），读路径共享磁盘（事实只有一份）。
    D2 PDF 正文缓存优先：fetch_paper 已把 PaperDoc 落盘（server.py S4），
       loader 读 JSON 即得分节全文，不重跑 pymupdf（几十秒 → 毫秒）。
       缓存缺失但 PDF 在 → 兜底 parse_pdf 现解析；两者都无 → 该论文只有
       abstract 入库（诚实降级，不假装有正文）。
    D3 trust 标记随行入库：论文文本（abstract/section）meta.trust =
       external_untrusted（与 mcp_server.schemas 的 Literal 同名同义），
       笔记/理由 = user_authored。qa 节点拼 prompt 时按它决定包不包
       分隔符——标记来自结构化字段，不是字符串猜（计划 §六.4 第 1 层）。
    D4 锚点原料放 meta：每条 RawDoc 都带 paper_id + title，chunker 直接
       拼 【{paper_id} {title}】。比 ShopMate 的"全库扫一遍建查找表"简单
       ——论文域 title 就在 papers.jsonl 里，不需要跨文档回填。

自测：python -m app.retrieval.loader        # tempdir 造假数据，离线断言
      python -m app.retrieval.loader --real # 真库统计（只读，建库前摸底）
"""
import json
import os
import sys
from pathlib import Path

from .schema import (COLLECTION_MEMORY, COLLECTION_PAPERS, DOC_ABSTRACT,
                     DOC_NOTE, DOC_REASON, DOC_SECTION, RawDoc)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PAPERPILOT_DATA_DIR", ROOT / "data"))
LIBRARY_DIR = DATA_DIR / "library"
PARSED_DIR = DATA_DIR / "parsed"

def _read_entries() -> list[dict]:
    """papers.jsonl → dict 列表。文件不存在 = 空库（首跑合法状态）。"""
    f = LIBRARY_DIR / "papers.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in
            f.read_text(encoding="utf-8").splitlines() if line.strip()]

def _paper_docs(paper_id: str) -> list[dict]:
    """D2：一篇论文的分节（PaperDoc dict）。缓存 → 兜底现解析 → 都无则 []。"""
    cached = PARSED_DIR / f"{paper_id}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))["sections"]
    pdf = LIBRARY_DIR / "pdfs" / f"{paper_id}.pdf"
    if pdf.exists():
        from mcp_server.pdf_parse import parse_pdf
        doc = parse_pdf(pdf, paper_id)              # title 由调用方校正在 add 阶段已有
        return doc["sections"] if isinstance(doc, dict) else doc.sections
    return []

def _paper_rows(e: dict) -> list[RawDoc]:
    """一条库条目 → abstract RawDoc + section RawDoc 列表（D2/D3/D4）。"""
    pid, title = e["paper_id"], e["meta"]["title"]
    ab = e["meta"]["abstract"]
    rows = [RawDoc(
        doc_id=f"abstract:{pid}", doc_type=DOC_ABSTRACT,
        collection=COLLECTION_PAPERS, source=ab["source"], paper_id=pid,
        text=ab["text"],
        meta={"title": title, "trust": "external_untrusted",
              "authors": e["meta"].get("authors", []),
              "published": e["meta"].get("published", "")})]
    for sec in _paper_docs(pid):
        body = sec["body"]
        rows.append(RawDoc(
            doc_id=f"section:{pid}:{sec['idx']}", doc_type=DOC_SECTION,
            collection=COLLECTION_PAPERS, source=body["source"], paper_id=pid,
            text=body["text"],
            meta={"title": title, "trust": "external_untrusted",
                  "heading": sec.get("heading", "")}))
    return rows

def _memory_rows(e: dict) -> list[RawDoc]:
    """一条库条目 → 收藏理由 + 该论文全部笔记（memory collection 的磁盘来源）。"""
    pid, title = e["paper_id"], e["meta"]["title"]
    rows = [RawDoc(
        doc_id=f"reason:{pid}", doc_type=DOC_REASON,
        collection=COLLECTION_MEMORY, source=f"reason:{pid}", paper_id=pid,
        text=f"论文《{title}》（{pid}）的入库理由：{e['added_reason']}",
        meta={"title": title, "trust": "user_authored"})]
    notes_dir = LIBRARY_DIR / "notes" / pid
    for f in sorted(notes_dir.glob("*.md")):
        rows.append(RawDoc(
            doc_id=f"note:{pid}:{f.stem}", doc_type=DOC_NOTE,
            collection=COLLECTION_MEMORY, source=f"note:{pid}/{f.stem}",
            paper_id=pid, text=f.read_text(encoding="utf-8"),
            meta={"title": title, "trust": "user_authored"}))
    return rows

def load_all() -> list[RawDoc]:
    """入口：整库装载。返回的 RawDoc 直接喂 chunker.chunk_all。"""
    rows: list[RawDoc] = []
    for e in _read_entries():
        rows.extend(_paper_rows(e))
        rows.extend(_memory_rows(e))
    return rows

# ---------------- 自测 ----------------
if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    if "--real" in sys.argv:
        rows = load_all()
        by_type: dict[str, int] = {}
        for r in rows:
            by_type[r.doc_type] = by_type.get(r.doc_type, 0) + 1
        print(f"真实库：{len(rows)} 条语料", by_type)
        sys.exit(0)

    import tempfile
    os.environ["PAPERPILOT_DATA_DIR"] = tempfile.mkdtemp()
    import importlib
    import app.retrieval.loader as ld
    importlib.reload(ld)                            # 让模块级 DATA_DIR 吃进新 env

    data = Path(os.environ["PAPERPILOT_DATA_DIR"])
    lib = data / "library"
    (lib / "notes" / "2401.01234").mkdir(parents=True)
    (lib / "pdfs").mkdir(parents=True)
    (data / "parsed").mkdir(parents=True)
    entry = {
        "paper_id": "2401.01234",
        "meta": {"arxiv_id": "2401.01234v2", "title": "A Study of Diffusion",
                 "authors": ["Alice"], "published": "2024-01-05",
                 "pdf_url": "https://arxiv.org/pdf/2401.01234v2.pdf",
                 "abstract": {"text": "We study diffusion models.",
                              "source": "arXiv:2401.01234v2/abstract",
                              "trust": "external_untrusted"}},
        "added_at": "2026-10-19T12:00:00",
        "added_reason": "与选题相关"}
    (lib / "papers.jsonl").write_text(json.dumps(entry, ensure_ascii=False),
                                      encoding="utf-8")
    (data / "parsed" / "2401.01234.json").write_text(json.dumps({
        "arxiv_id": "2401.01234", "title": "A Study of Diffusion", "n_pages": 1,
        "sections": [
            {"idx": 1, "heading": "1 Introduction",
             "body": {"text": "Diffusion models are everywhere.",
                      "source": "paper:2401.01234/section:1",
                      "trust": "external_untrusted"}},
            {"idx": 2, "heading": "",
             "body": {"text": "Methods section body.",
                      "source": "paper:2401.01234/section:2",
                      "trust": "external_untrusted"}}]},
        ensure_ascii=False), encoding="utf-8")
    (lib / "notes" / "2401.01234" / "deadbeef.md").write_text(
        "第 3 节的实验设置值得复现", encoding="utf-8")

    rows = ld.load_all()
    kinds = sorted((r.doc_type, r.doc_id) for r in rows)
    assert [k[0] for k in kinds] == ["abstract", "note", "reason", "section",
                                     "section"], kinds
    sec = next(r for r in rows if r.doc_type == DOC_SECTION)
    assert sec.meta["trust"] == "external_untrusted" and sec.meta["title"]
    assert sec.source == "paper:2401.01234/section:1"
    note = next(r for r in rows if r.doc_type == DOC_NOTE)
    assert note.collection == "memory" and note.meta["trust"] == "user_authored"
    reason = next(r for r in rows if r.doc_type == DOC_REASON)
    assert "入库理由" in reason.text and reason.paper_id == "2401.01234"
    print("1 装载 OK：abstract+2×section+note+reason，trust/锚点原料随行")

    (data / "parsed" / "2401.01234.json").unlink()
    rows2 = ld.load_all()
    assert sum(r.doc_type == DOC_SECTION for r in rows2) == 0, \
        "缓存删了且 PDF 不存在时不得凭空出 section（D2 诚实降级）"
    assert sum(r.doc_type == DOC_ABSTRACT for r in rows2) == 1
    print("2 降级 OK：无缓存无 PDF → 只剩 abstract，不假装有正文")

    (lib / "pdfs" / "2401.01234.pdf").write_bytes(b"%PDF-fake")
    try:
        ld.load_all()
        raise AssertionError("坏 PDF 应让兜底路径炸出来，而不是静默")
    except Exception:
        pass
    print("3 兜底 OK：缓存缺失但 PDF 在 → 走 parse_pdf（坏文件诚实报错）")

    print("\nloader 自测 3/3 通过（离线 tempdir，不碰真实 data/）")
