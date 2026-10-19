"""Chunk → BGE-M3 向量化 → ChromaDB 入库（M3-b）。

复用 ShopMate indexer.py 的全部机制（D1-D5 编号对齐），差异两处：
    - collection 换成 paper_knowledge / memory 双库（schema.COLLECTIONS）
    - 新增 sanitize_meta：chromadb 的 metadata 只收 str/int/float/bool，
      loader 带来的 authors 是 list[str]、某些键可能是 None——不过清洗直接
      upsert 会在 chroma 侧炸出难以定位的类型错误。在入库边界统一清洗
      （list→顿号串、None 丢弃），loader 不用为存储层改形状，下游 memory.py
      复用同一函数。

模型加载纪律（原样继承，实测教训）：
    D1 懒加载单例：BGE-M3 约 2.3GB，第一次 embed 才加载，之后复用。
    D2 HF 缓存判定：本地已有 models--BAAI--bge-m3 时置 HF_HUB_OFFLINE=1，
       避免 transformers 联网核对仓库模板卡死（实测 WinError 10060）。
       判定必须在 import FlagEmbedding 之前——HF_HUB_OFFLINE 是 import
       时读的常量，运行时改不生效。
    D3 cosine 相似度、D4 元数据随行入库、D5 upsert 幂等：同 ShopMate。

自测：python -m app.retrieval.indexer
    首次运行从本地 HF 缓存加载 BGE-M3（约 10-30 秒，零下载）。建库幂等
    （upsert 同 ID 覆盖），重复执行安全。
"""
import os
import sys
import threading
from pathlib import Path

from .schema import COLLECTION_PAPERS, COLLECTIONS, EMBED_DIM
from .schema import Chunk

PERSIST_ROOT = Path(__file__).resolve().parents[2] / "data" / "chroma"

_embed_batch = 64        # 一次喂给模型的条数：太大爆内存，太小吞吐低

# ---------------- 模型（D1/D2） ----------------

_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:                      # double-check：抢到锁后再看一眼
            hub = Path(os.environ.get("HF_HUB_CACHE",
                      Path.home() / ".cache" / "huggingface" / "hub"))
            if (hub / "models--BAAI--bge-m3").exists():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from FlagEmbedding import BGEM3FlagModel
            _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)   # fp16 内存减半
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化，返回与 texts 等长的 1024 维 list（ChromaDB 不吃 ndarray）。"""
    if not texts:
        return []
    model = get_model()
    out: list[list[float]] = []
    for i in range(0, len(texts), _embed_batch):
        batch = texts[i:i + _embed_batch]
        res = model.encode(batch)["dense_vecs"]
        out.extend(vec.tolist() for vec in res)
    return out


# ---------------- ChromaDB ----------------

def get_client(persist_dir: Path | str | None = None):
    """打开持久化客户端。同一路径重复调用返回同一实例（ChromaDB 自己缓存）。"""
    import chromadb
    return chromadb.PersistentClient(path=str(persist_dir or PERSIST_ROOT))


def get_collection(name: str, client=None, create: bool = True):
    """打开/建一个 collection（cosine）。检索端传 create=False 防误建空库。"""
    client = client or get_client()
    kw = {"metadata": {"hnsw:space": "cosine"}} if create else {}
    return (client.get_or_create_collection(name, **kw) if create
            else client.get_collection(name))


# ---------------- 元数据边界清洗（论文域新增） ----------------

def sanitize_meta(meta: dict) -> dict:
    """chromadb metadata 契约清洗：值只许 str/int/float/bool。

    list/tuple → 顿号连接的字符串（authors 保留可读性）；None 丢弃
    （chroma 拒收）；其余标量原样，未知类型兜底 str()。
    """
    out: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = "、".join(str(x) for x in v)
        else:
            out[k] = str(v)
    return out


# ---------------- 入库 ----------------

def build_index(chunks: list[Chunk], persist_dir: Path | str | None = None) -> dict[str, int]:
    """全量入库（幂等：upsert 同 chunk_id 覆盖）。返回 {collection: 条数}。"""
    client = get_client(persist_dir)
    stats: dict[str, int] = {}
    for coll_name in COLLECTIONS:
        group = [c for c in chunks if c.meta.get("collection") == coll_name]
        if not group:
            continue
        coll = get_collection(coll_name, client)
        for i in range(0, len(group), _embed_batch):
            batch = group[i:i + _embed_batch]
            coll.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embed_texts([c.to_embedding_text() for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[sanitize_meta(c.meta) for c in batch],
            )
        stats[coll_name] = len(group)
    return stats


# ---------------- 自测：python -m app.retrieval.indexer ----------------
# 前置：chromadb/FlagEmbedding 已装；BGE-M3 走本地 HF 缓存（ShopMate 已下载）。
if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    from .chunker import chunk_all
    from .loader import load_all

    chunks = chunk_all(load_all())
    print(f"语料 {len(chunks)} 块，开始建库（首次含模型加载，约 10-30 秒）...")
    stats = build_index(chunks)
    assert stats, "至少一个 collection 有入库"
    assert COLLECTION_PAPERS in stats, \
        f"qa 主通道 {COLLECTION_PAPERS} 不能为空: {stats}"
    for name in COLLECTIONS:
        if name not in stats:
            print(f"  提示：{name} 本次无语料（正常，随使用增长）")
    print(f"入库统计: {stats}")

    client = get_client()
    for coll_name, n in stats.items():
        coll = get_collection(coll_name, client)
        assert coll.count() == n, f"{coll_name}: 库里 {coll.count()} != 应有 {n}"
        one = coll.get(limit=1, include=["embeddings", "metadatas"])
        assert len(one["embeddings"][0]) == EMBED_DIM, \
            f"{coll_name}: 向量维度 {len(one['embeddings'][0])} != {EMBED_DIM}"
        bad = [v for v in one["metadatas"][0].values()
               if not isinstance(v, (str, int, float, bool))]
        assert not bad, f"{coll_name}: metadata 混入 chroma 不支持的类型: {bad}"
        print(f"  {coll_name}: count={coll.count()} dim={EMBED_DIM} meta 纯标量 OK")

    # 幂等复验：重建一次计数不变（upsert 覆盖而非叠加）
    stats2 = build_index(chunks)
    assert stats2 == stats, f"重建后计数应不变: {stats2} vs {stats}"

    # 回路探针：取一条已入库文本自查询，应 top1 命中自己（embed→store→query 通）。
    # 注意 chroma 返回格式差异：query() 的 ids 按查询向量嵌套，get() 的 ids
    # 是扁平列表——下标写错不会报错，只会取到字符串首字符（实测踩过）。
    pap = get_collection(COLLECTION_PAPERS, client)
    got = pap.get(limit=1, include=["documents"])
    got_id = got["ids"][0]
    hit = pap.query(query_embeddings=embed_texts([got["documents"][0]]), n_results=1)
    assert hit["ids"][0][0] == got_id, \
        f"自查询应 top1 命中自己: {hit['ids'][0][0]} vs {got_id}"
    print(f"回路探针 OK：{got_id} 自查询 top1 命中")

    print("\nindexer 自测通过（真实库已建成，幂等可重跑）")
