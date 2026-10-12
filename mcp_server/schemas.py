"""工具返回结构契约：所有 mcp_server 工具的返回 dict 以此为准。

约定（方案 A）：每个工具声明 -> dict，返回值 = model_dump() 的结果。
trust 标记是 Literal 字段、frozen 实例，外部数据无路径修改它。
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict

class ExternalText(BaseModel):
    """外部不可信文本：arXiv 摘要 / PDF 正文。"""
    model_config = ConfigDict(frozen=True)

    text: str
    source: str                                   # 如 "arXiv:2401.01234v2/abstract" 或 "paper:xxx/section:3"
    trust: Literal["external_untrusted"] = "external_untrusted"


class LibraryText(BaseModel):
    """本地库内文本：已入库论文的正文 / 用户笔记。"""
    model_config = ConfigDict(frozen=True)

    text: str
    paper_id: str
    trust: Literal["local_library"] = "local_library"

class PaperMeta(BaseModel):
    """检索结果里的一篇论文（不含正文）。"""
    arxiv_id: str            # "2401.01234v2"
    title: str
    authors: list[str]
    abstract: ExternalText   # 摘要是外部内容，天然带标记
    published: str           # ISO 日期
    pdf_url: str

class Section(BaseModel):
    """PDF 解析出的一个分节。"""
    idx: int                 # 全文序号，从 1 开始
    heading: str             # "1 Introduction"，无法识别时 = ""
    body: ExternalText

class PaperDoc(BaseModel):
    """fetch_paper 的完整产物：一篇论文的分节全文。"""
    arxiv_id: str
    title: str
    n_pages: int
    sections: list[Section]

class Note(BaseModel):
    """用户笔记。"""
    note_id: str
    paper_id: str
    text: str                # 用户自己的话，无 trust 标记（用户内容 ≠ 攻击面，M4 的 PII 才管它）
    created_at: str

class LibraryEntry(BaseModel):
    """add_to_library 后的库条目。"""
    paper_id: str            # 内部 id = arxiv_id 去版本号，如 "2401.01234"
    meta: PaperMeta
    added_at: str
    added_reason: str        # 为什么入库（用户要求 / Agent 建议+确认）——确认门文案要用  是给 未来的问答和决策 用的

# ---------- 工具返回 dict 的形状（契约，不是代码） ----------
# search_arxiv(query, max_results=5)  -> {"papers": [PaperMeta.model_dump(), ...], "total": int}
# fetch_paper(arxiv_id)               -> PaperDoc.model_dump()          # 下载+解析，落盘到 data/pdfs/
# read_paper(paper_id, section=None)  -> {"paper_id":..., "title":..., "sections":[Section.model_dump()]}
#                                     #   section=None 给全部；给 idx 只给那一节
# list_library()                      -> {"entries": [LibraryEntry.model_dump()], "count": int}
# add_to_library(arxiv_id, reason)    -> LibraryEntry.model_dump()      # 写工具
# save_note(paper_id, text)           -> Note.model_dump()              # 写工具
# remove_paper(paper_id)              -> {"removed": paper_id}          # 写工具