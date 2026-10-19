"""RawDoc → Chunk（M3-b）：论文域切法。

与 ShopMate chunker.py 的差异（为什么不是复制）：
    - 语料天然自带结构：abstract 一块、section 一节一块、note/reason 本就
      短——不需要按 ## 切、不需要 spec JSON 展平、不需要父标题（ShopMate
      D1 的场景在论文域不存在）。所有 doc_type 共同的问题只有一个：块
      超长 → 全部交给 _split_long 递归对半（机制原样继承）。
    - 锚点简化：ShopMate 要全库扫两遍建"商品ID→名称"查找表（D2），论文域
      title 就在 meta 里（loader D4），直接取。
    - MAX_CHUNK_CHARS 750→1200：中文 1 字≈1.5 token，750 字符≈512 token；
      英文科学文本约 4 字符≈1 token，750 字符才 ~190 token，粒度太碎会把
      一句完整论述拦腰切。1200 字符≈300 token，对 BGE-M3（8192 窗口）
      仍然保守。

自测：python -m app.retrieval.chunker   （纯函数，离线零依赖）
"""
import sys

from .schema import Chunk, RawDoc

MAX_CHUNK_CHARS = 1200


def chunk_all(docs: list[RawDoc]) -> list[Chunk]:
    """批量入口：按 doc_type 组装块正文 → 拼论文锚点 → 产出可入库 Chunk。

    meta 随行继承 loader 的全部键（title/trust/heading/...），再加
    doc_type/paper_id/collection/chunk_id——indexer 按 meta["collection"]
    分组入库，检索端按 doc_type/paper_id 过滤。
    """
    chunks: list[Chunk] = []
    for d in docs:
        for i, text in enumerate(_pieces_for(d), start=1):
            anchor = _anchor(d.paper_id, d.meta.get("title", ""))
            cid = f"{d.doc_id}#{i:02d}"
            chunks.append(Chunk(
                doc_id=d.doc_id, chunk_id=cid,
                text=f"{anchor}{text}" if anchor else text,
                meta={**d.meta, "doc_type": d.doc_type, "paper_id": d.paper_id,
                      "collection": d.collection, "chunk_id": cid}))
    return chunks


def _pieces_for(d: RawDoc) -> list[str]:
    """按 doc_type 组装块正文，统一过 _split_long 限长。"""
    if d.doc_type == "section":
        head = (d.meta.get("heading") or "").strip()
        body = f"{head}\n{d.text}" if head else d.text
    else:                       # abstract/note/reason/qa：整块直给
        body = d.text
    return _split_long(body)


def _anchor(pid: str, title: str) -> str:
    """论文锚点："【2401.01234 Attention Is All You Need】\\n"。

    BM25 靠锚点里的 arXiv ID 定位（用户报型号提问的同款问题，ShopMate
    评测实测过漏召回）；title 让纯语义查询也能蹭到论文身份。
    缺编号或缺标题就不硬造锚点（诚实降级为无锚点）。
    """
    if not pid or not title:
        return ""
    return f"【{pid} {title}】\n"


def _split_long(text: str, max_len: int = MAX_CHUNK_CHARS) -> list[str]:
    """超长块递归对半：在中点 ±150 字符窗口找最近换行下刀，窗口内没有
    换行才硬切。终止性：每层长度严格递减（max_len ≥ 300 即保证）。原样
    继承 ShopMate 的机制，只改默认上限。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    mid = len(text) // 2
    lo, hi = max(0, mid - 150), min(len(text), mid + 150)
    window, target = text[lo:hi], mid - lo
    best, best_dist = -1, 1 << 30
    idx = window.find("\n")
    while idx != -1:
        if abs(idx - target) < best_dist:
            best, best_dist = idx, abs(idx - target)
        idx = window.find("\n", idx + 1)
    cut = (lo + best) if best != -1 else mid

    return _split_long(text[:cut], max_len) + _split_long(text[cut:], max_len)


# ---------------- 自测：python -m app.retrieval.chunker（离线） ----------------
if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    section = RawDoc(
        doc_id="section:2401.01234:3", doc_type="section",
        collection="paper_knowledge", source="paper:2401.01234/section:3",
        paper_id="2401.01234", text="We evaluate on GLUE. " * 80,   # 2200+ 字符
        meta={"title": "RAG Evaluation Survey", "trust": "external_untrusted",
              "heading": "3 Experiments"})
    note = RawDoc(
        doc_id="note:2401.01234:deadbeef", doc_type="note",
        collection="memory", source="note:2401.01234/deadbeef",
        paper_id="2401.01234", text="第 3 节的实验设置值得复现",
        meta={"title": "RAG Evaluation Survey", "trust": "user_authored"})

    chunks = chunk_all([section, note])

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 有重复"
    assert all(c.text.strip() for c in chunks), "有空块"
    assert all(len(c.text) <= MAX_CHUNK_CHARS + 40 for c in chunks), \
        f"有块超过上限：{max(len(c.text) for c in chunks)}"
    sec_chunks = [c for c in chunks if c.meta["doc_type"] == "section"]
    assert len(sec_chunks) >= 2, "2200+ 字符的 section 应切成多块"
    assert sec_chunks[0].text.startswith("【2401.01234 RAG Evaluation Survey】")
    assert sec_chunks[0].text.splitlines()[1] == "3 Experiments", \
        "section 首块应把 heading 放进正文第一行"
    assert all(c.meta["trust"] == "external_untrusted" for c in sec_chunks)
    note_chunk = next(c for c in chunks if c.meta["doc_type"] == "note")
    assert note_chunk.meta["collection"] == "memory"
    assert note_chunk.text.startswith("【") and "复现" in note_chunk.text
    print(f"1 切块 OK：{len(chunks)} 块（section×{len(sec_chunks)} + note×1），"
          "锚点/heading/上限/trust 全部就位")

    empty = chunk_all([RawDoc(doc_id="x", doc_type="note", collection="memory",
                              source="x", paper_id="", text="   ", meta={})])
    assert empty == [], "空白语料应产出零块"
    print("2 空语料 OK：不产出空块")

    print("\nchunker 自测 2/2 通过（离线纯函数）")
