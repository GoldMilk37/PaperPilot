"""arXiv API 封装：search_arxiv / fetch_paper 的网络底层（mcp_server 侧）。

设计决策：
    A1 代理纪律同 ShopMate D5：显式 HTTPS_PROXY 才走代理，系统/注册表代理
       一律不认。包用 httpx2（mcp 2.x 的依赖），但坑是同一族（urllib 读
       注册表代理），纪律原样适用。
    A2 懒加载单例 client + 模块级节流：arXiv 官方礼貌条款是"请求间隔
       不小于 3 秒"，违反会被限流。节流是全局状态，收在本层别让调用方管。
    A3 三层减速叠加：3s 超时（单请求）+ 3s 间隔（请求之间）+ TTL 缓存
       （同一 query 10 分钟内直接命中）。这层缓存属于"能力"侧，与
       executor（策略侧）的缓存互不替代。
    A4 连接类抖动重试一次（判据照搬 D4 的类名清单），重试仍失败抛
       RuntimeError 给上层——翻译成三态 is_error 是 executor 的职责，
       本模块只负责把异常抛得看得懂。
    A5 摘要在此处被构造为 ExternalText（trust 标记的合法产地之一）；
       正文同理发生在 pdf_parse。外部文本不从这两处来的，类型上就带不上标记。
"""
import os
import re
import time
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx2

from .schemas import ExternalText, PaperMeta

API_URL = "https://export.arxiv.org/api/query"

REQUEST_TIMEOUT_S = 3.0      # A3：单请求超时（计划定死）
MIN_INTERVAL_S = 3.0         # A2：arXiv 礼貌条款
CACHE_TTL_S = 600.0          # A3：同一 query 的缓存存活期
RETRY_DELAY_S = 0.8

_NS = {"a": "http://www.w3.org/2005/Atom"}

_client = None
_lock = threading.Lock()
_last_request_ts = 0.0
_cache: dict[tuple[str, int], tuple[float, list[PaperMeta]]] = {}

def _proxy_from_env() -> str | None:
    """A1：只认显式环境变量（大写优先，小写兜底）。"""
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None

def _get_client() -> httpx2.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                proxy = _proxy_from_env()
                _client = (httpx2.Client(proxy=proxy, trust_env=False, timeout=REQUEST_TIMEOUT_S)
                           if proxy else httpx2.Client(trust_env=False, timeout=REQUEST_TIMEOUT_S))
    return _client

def _throttle() -> None:
    """A2：距上次请求不足 MIN_INTERVAL_S 就等满。"""
    global _last_request_ts
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.monotonic()

_RETRYABLE_HINTS = ("Timeout", "Connection", "Connect", "Read", "RemoteProtocol")

def _retryable(e: Exception) -> bool:
    return any(h in type(e).__name__ for h in _RETRYABLE_HINTS)

def _get(url: str, *, params: dict | None = None) -> str:
    """A4：带一次重试的 GET，失败抛 RuntimeError（带真实异常类名）。"""
    last: Exception | None = None
    for attempt in range(2):
        try:
            resp = _get_client().get(url, params=params)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last = e
            if attempt == 0 and _retryable(e):
                time.sleep(RETRY_DELAY_S)
                continue
            break
    raise RuntimeError(f"arXiv 请求失败: {type(last).__name__}: {last}") from last

def _norm(s: str | None) -> str:
    """arXiv 返回的标题/摘要里常有换行和连续空格，压成单空格。"""
    return " ".join((s or "").split())

def _parse_feed(xml_text: str) -> list[PaperMeta]:
    """Atom XML -> PaperMeta 列表。A5：摘要在此带上 trust 标记。"""
    root = ET.fromstring(xml_text)
    papers: list[PaperMeta] = []
    for entry in root.findall("a:entry", _NS):
        raw_id = entry.findtext("a:id", "", _NS)          # http://arxiv.org/abs/2401.01234v2
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        pdf_url = next((lk.get("href", "") for lk in entry.findall("a:link", _NS)
                        if lk.get("title") == "pdf"), "")
        papers.append(PaperMeta(
            arxiv_id=arxiv_id,
            title=_norm(entry.findtext("a:title", "", _NS)),
            authors=[_norm(a.findtext("a:name", "", _NS)) for a in entry.findall("a:author", _NS)],
            abstract=ExternalText(text=_norm(entry.findtext("a:summary", "", _NS)),
                                  source=f"arXiv:{arxiv_id}/abstract"),
            published=(entry.findtext("a:published", "", _NS) or "")[:10],
            pdf_url=pdf_url,
        ))
    return papers

def search(query: str, max_results: int = 5) -> list[PaperMeta]:
    """按关键词检索 arXiv，返回最新优先的 PaperMeta 列表（A3：TTL 缓存）。"""
    query = query.strip()
    if not query:
        raise RuntimeError("检索词为空")
    key = (query, max_results)
    hit = _cache.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_TTL_S:
        return hit[1]
    _throttle()
    xml_text = _get(API_URL, params={
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    papers = _parse_feed(xml_text)
    _cache[key] = (time.monotonic(), papers)
    return papers

def download_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    """下载 PDF 到 dest_dir，返回本地路径。已存在则直接复用（天然缓存）。

    先写 .part 再原子改名：防止半截文件被下次当成下载成功。
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{arxiv_id}.pdf"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    _throttle()
    # _get 返回 text 会按编码解码破坏二进制，PDF 必须拿原始字节
    last: Exception | None = None
    for attempt in range(2):
        try:
            with _get_client().stream("GET", f"https://arxiv.org/pdf/{arxiv_id}") as resp:
                resp.raise_for_status()
                part = dest.with_suffix(".part")
                with open(part, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
                part.replace(dest)
                return dest
        except Exception as e:
            last = e
            if attempt == 0 and _retryable(e):
                time.sleep(RETRY_DELAY_S)
                continue
            break
    raise RuntimeError(f"PDF 下载失败 {arxiv_id}: {type(last).__name__}: {last}") from last

def clear_cache() -> None:
    """自测用：清空 TTL 缓存与节流时间戳。"""
    global _last_request_ts
    _cache.clear()
    _last_request_ts = 0.0

# ---------------- 自测：python -m mcp_server.arxiv_api ----------------
if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. A1 代理决策（不联网）
    saved_px = {k: os.environ.pop(k, None) for k in ("HTTPS_PROXY", "https_proxy")}
    try:
        assert _proxy_from_env() is None, "环境变量空 → 必须直连（注册表代理无视）"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
        assert _proxy_from_env() == "http://127.0.0.1:7890", "显式设了就走"
        c = _get_client()
        assert c is _get_client(), "单例"
    finally:
        for k, v in saved_px.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("1 A1 代理决策 OK：环境变量空 → 直连；设了 HTTPS_PROXY → 走它")

    # 2. Atom 解析（离线：内嵌最小样例）
    sample = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.01234v2</id>
        <title>  A  Study   of
        Diffusion </title>
        <summary> We study
        diffusion models. </summary>
        <author><name>Alice Chen</name></author>
        <author><name>Bob Li</name></author>
        <published>2024-01-05T18:00:00Z</published>
        <link title="pdf" href="https://arxiv.org/pdf/2401.01234v2.pdf"/>
      </entry>
    </feed>"""
    papers = _parse_feed(sample)
    assert len(papers) == 1
    p = papers[0]
    assert p.arxiv_id == "2401.01234v2"
    assert p.title == "A Study of Diffusion", p.title
    assert p.abstract.trust == "external_untrusted", "摘要必须天然带标记（A5）"
    assert p.abstract.source == "arXiv:2401.01234v2/abstract"
    assert p.authors == ["Alice Chen", "Bob Li"]
    assert p.published == "2024-01-05" and p.pdf_url.endswith(".pdf")
    d = p.abstract.model_dump()
    assert d == {"text": "We study diffusion models.",
                 "source": "arXiv:2401.01234v2/abstract",
                 "trust": "external_untrusted"}, "model_dump 形状即契约"
    print("2 Atom 解析 OK：id 提取 / 空白压缩 / trust 标记 / dump 形状全部断言通过")

    # 3. 联网冒烟（显式开关才跑；走真实 arXiv，免费）
    if "--online" in sys.argv:
        rs = search("diffusion model", 3)
        assert 0 < len(rs) <= 3
        print(f"3 联网冒烟 OK：{len(rs)} 条，首条: {rs[0].arxiv_id} {rs[0].title[:40]}")
    else:
        print("3 联网冒烟: 跳过（加 --online 才发真实请求）")