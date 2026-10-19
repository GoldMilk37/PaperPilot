"""检索管线的公共约定（M3）。常量与数据结构集中在这，改这里不用动其他模块。

与 ShopMate schema.py 的差异（论文域重画）：
    - 知识域 6 类 → 论文域 5 类 doc_type；collection 3 个 → 2 个
      （paper_knowledge = 论文本身，qa 节点检索；memory = 用户痕迹，
      recall 节点召回——计划 §六.3 的双通道）
    - RawDoc.product_id → paper_id
    - 阈值沿用 ShopMate 评测结论作初值，但**必须在建库后用论文语料重校**：
      英文科学文本与中文商品评论的 cosine 距离分布不同（ShopMate 实测
      真相关≈0.31/无关≈0.66 是商品域的数，不能直接当论文域的数信）。
"""
from dataclasses import dataclass, field

# ---- 入库 Collection：检索端 get_collection(create=False) 靠这个名字找库 ----
COLLECTION_PAPERS = "paper_knowledge"
COLLECTION_MEMORY = "memory"
COLLECTIONS = (COLLECTION_PAPERS, COLLECTION_MEMORY)

# ---- doc_type：loader 产出 / meta 里随行入库 / 检索端加权与过滤用 ----
DOC_ABSTRACT = "abstract"   # 论文摘要（papers.jsonl）
DOC_SECTION = "section"     # PDF 分节正文（data/parsed 缓存）
DOC_NOTE = "note"           # 用户笔记（notes/{paper_id}/*.md）
DOC_REASON = "reason"       # 收藏理由（added_reason——"为什么收这篇"是高价值记忆）
DOC_QA = "qa"               # 历史问答对（运行期由 memory.py 写入，不经 loader）

# ---- 检索参数（初值 = ShopMate 评测结论，论文域建库后重校） ----
TOP_K = 5
RRF_K = 60
SCORE_THRESHOLD = 0.012
SEMANTIC_MAX_DIST = 0.45

EMBED_DIM = 1024            # BGE-M3 输出维度

@dataclass
class RawDoc:
    """loader 的输出：一份语料原文 + 元数据。"""
    doc_id: str          # 文档唯一 ID，如 "section:2401.01234:3"
    doc_type: str        # abstract/section/note/reason/qa
    collection: str      # 入哪个 ChromaDB collection
    source: str          # 产地，如 "paper:2401.01234/section:3"
    paper_id: str        # 关联论文；note/reason/qa 都带（记忆也能按论文过滤）
    text: str
    meta: dict = field(default_factory=dict)

@dataclass
class Chunk:
    """chunker 的输出：一块可被向量化的文本。text 头部应已拼好论文锚点。"""
    doc_id: str
    chunk_id: str        # 全局唯一，f"{doc_id}#{序号}"
    text: str
    meta: dict = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        return self.text
