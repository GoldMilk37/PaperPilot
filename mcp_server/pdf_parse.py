"""PDF → 分节文本（pymupdf）。fetch_paper 的解析底层，read_paper 与 M3 RAG 共用。

设计决策：
    P1 分节在 server 侧做：read_paper 按节取（token 经济学）；节号即引用
       出处（paper:{id}/section:{n}）；M3 的 loader 直接复用本分节做 chunk。
    P2 标题判别 = 编号正则 + 字号启发式 双通道：正文行字号按"字符数加权
       众数"确定，显著大于正文（>=1.12 倍）的短行、或匹配 "1 Introduction"
       式编号的行判为标题。启发式会漏（跨行标题、同字号无编号标题），但
       只影响"切分粒度"，不产生错误文本——诚实边界写进 docs/04。
    P3 正文文本在此构造为 ExternalText（trust 标记产地之二），source 格式
       "paper:{arxiv_id}/section:{idx}" 与 schemas 的约定一致。
    P4 无文本层（扫描件）直接报错，不静默返回空——诚实失败优于假成功。
"""
import re
from collections import Counter
from pathlib import Path

import pymupdf

from .schemas import ExternalText, PaperDoc, Section

# 编号式标题："1 Introduction" / "2.3 Related Work"（标题词必须大写开头，
# 排除 "2.3 million samples" 这类以数字开头的普通句子）
_NUMBERED = re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z]")

# 无编号但约定俗成的标题（字号与正文相同时靠这张表兜底）
_UNNUMBERED = {"abstract", "references", "acknowledgments", "acknowledgements",
               "appendix", "discussion", "conclusion", "conclusions"}

_MAX_HEAD_LEN = 90          # 比这更长的"标题"几乎必然是正文句
_BODY_RATIO = 1.12          # 字号达到正文 1.12 倍才算"显著大"

def _clean(text: str) -> str:
    """合并断词换行（"diffu-\nsion"→"diffusion"）+ 压缩空白。"""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return " ".join(text.split())

def _extract_lines(doc: "pymupdf.Document") -> list[tuple[int, float, str, float]]:
    """按阅读顺序抽行：(页码, y坐标, 行文本, 行内最大字号)。"""
    out: list[tuple[int, float, str, float]] = []
    for pno, page in enumerate(doc, start=1):
        data = page.get_text("dict")
        for block in data["blocks"]:
            if block.get("type") != 0:                       # 0=文本块，跳过图片
                continue
            for line in block["lines"]:
                spans = [sp for sp in line["spans"] if sp["text"].strip()]
                if not spans:
                    continue
                text = " ".join(sp["text"] for sp in spans).strip()
                size = max(sp["size"] for sp in spans)
                out.append((pno, line["bbox"][1], text, size))
    out.sort(key=lambda t: (t[0], t[1]))
    return out

def _body_size(lines: list[tuple[int, float, str, float]]) -> float:
    """正文字号 = 按字符数加权的众数（标题行字数少，左右不了统计）。"""
    counter: Counter[float] = Counter()
    for _, _, text, size in lines:
        counter[round(size, 1)] += len(text)
    return counter.most_common(1)[0][0]

def _is_heading(text: str, size: float, body_size: float) -> bool:
    """P2：编号正则 ∨ 显著大字号，再叠加长度/标点/白名单三道闸。"""
    if not text or len(text) > _MAX_HEAD_LEN:
        return False
    if text.strip().lower() in _UNNUMBERED:
        return True
    if text.endswith((".", ",", ";", ":", "?", "!")):
        return False
    numbered = bool(_NUMBERED.match(text))
    big = size >= body_size * _BODY_RATIO
    return numbered or big

def parse_pdf(pdf_path: Path, arxiv_id: str, title: str = "") -> PaperDoc:
    """解析 PDF 为分节文档。title 由调用方传入（检索时已拿到，不重复猜）。"""
    with pymupdf.open(pdf_path) as doc:
        n_pages = doc.page_count
        lines = _extract_lines(doc)
    if sum(len(t) for _, _, t, _ in lines) < 50:
        raise RuntimeError(f"{arxiv_id}: PDF 无文本层（可能是扫描件），无法解析")   # P4

    body_size = _body_size(lines)
    raw: list[tuple[str, list[str]]] = [("", [])]            # (标题, 正文行缓冲)
    for _, _, text, size in lines:
        if _is_heading(text, size, body_size):
            raw.append((text, []))
        else:
            raw[-1][1].append(text)

    sections: list[Section] = []
    for head, buf in raw:
        text = _clean("\n".join(buf))
        if not text:
            continue                                          # 标题行自身不构成空节
        idx = len(sections) + 1
        sections.append(Section(
            idx=idx, heading=head,
            body=ExternalText(text=text, source=f"paper:{arxiv_id}/section:{idx}")))   # P3

    if not sections:                                          # 兜底：整篇一节
        text = _clean("\n".join(t for _, _, t, _ in lines))
        sections = [Section(idx=1, heading="",
                            body=ExternalText(text=text, source=f"paper:{arxiv_id}/section:1"))]
    return PaperDoc(arxiv_id=arxiv_id, title=title, n_pages=n_pages, sections=sections)

# ---------------- 自测：python -m mcp_server.pdf_parse（全程离线） ----------------
def _make_sample_pdf(path: Path) -> None:
    """用 pymupdf 自己造一份两页样例（离线，不依赖真实论文）。"""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Diffusion Models Survey", fontsize=18, fontname="helv")
    p1.insert_text((72, 110), "1 Introduction", fontsize=14, fontname="hebo")
    p1.insert_text((72, 130), "We study diffusion models for image synthesis.", fontsize=10, fontname="helv")
    p1.insert_text((72, 160), "2 Method", fontsize=14, fontname="hebo")
    p1.insert_text((72, 180), "We propose a pilot training method with noise prediction.", fontsize=10, fontname="helv")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "References", fontsize=14, fontname="hebo")
    p2.insert_text((72, 92), "Goodfellow et al. Deep Learning. MIT Press, 2016.", fontsize=10, fontname="helv")
    doc.save(path)
    doc.close()

if __name__ == "__main__":
    import sys
    import tempfile
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. _clean 纯函数（断词合并 / 空白压缩）
    assert _clean("diffu-\nsion models") == "diffusion models"
    assert _clean("A  B\n\nC") == "A B C"
    print("1 _clean 纯函数 OK：断词合并 / 空白压缩")

    with tempfile.TemporaryDirectory() as td:
        # 2. 正常分节：两页样例 → 3 节，trust 标记与 source 格式验收（P3）
        p = Path(td) / "sample.pdf"
        _make_sample_pdf(p)
        doc = parse_pdf(p, "2401.01234v2", title="Diffusion Models Survey")
        assert doc.n_pages == 2 and doc.title == "Diffusion Models Survey"
        heads = [s.heading for s in doc.sections]
        assert heads == ["1 Introduction", "2 Method", "References"], heads
        intro = doc.sections[0]
        assert intro.idx == 1 and "diffusion models" in intro.body.text
        assert intro.body.trust == "external_untrusted", "正文必须天然带标记（P3）"
        assert intro.body.source == "paper:2401.01234v2/section:1"
        assert doc.sections[2].heading == "References", "无编号标题靠白名单命中"
        print("2 分节解析 OK：编号/白名单标题命中，trust 标记与 source 格式验收通过")

        # 3. 兜底：无任何标题 → 整篇单节，heading=""
        p2 = Path(td) / "plain.pdf"
        d2 = pymupdf.open()
        pg = d2.new_page()
        for i in range(6):
            pg.insert_text((72, 60 + 20 * i), f"Plain paragraph number {i} with enough body text.",
                           fontsize=10, fontname="helv")
        d2.save(p2)
        d2.close()
        doc2 = parse_pdf(p2, "1999.99999v1")
        assert len(doc2.sections) == 1 and doc2.sections[0].heading == ""
        assert doc2.sections[0].body.source == "paper:1999.99999v1/section:1"
        print("3 兜底单节 OK：无标题文档不丢内容")

        # 4. 扫描件（无文本层）→ 诚实报错（P4）
        p3 = Path(td) / "blank.pdf"
        d3 = pymupdf.open()
        d3.new_page()
        d3.save(p3)
        d3.close()
        try:
            parse_pdf(p3, "0000.00000v1")
            raise AssertionError("无文本层应该报错")
        except RuntimeError as e:
            assert "无文本层" in str(e)
        print("4 扫描件守卫 OK：无文本层诚实报错，不静默返回空")