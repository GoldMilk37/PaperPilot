"""本地文献库读写：papers.jsonl + notes/（mcp_server 侧唯一持久层）。

设计决策：
    L1 存储格式：条目 = JSONL 追加写（一行一个 LibraryEntry，抗中断）；
       笔记 = notes/{paper_id}/{note_id}.md（人可直接打开编辑）。
       不用 SQLite：M1 零依赖跑通；其锁语义在容器里是新坑，M9 换存储
       时再议（与 ShopMate 的"规划中换存储"同款路线）。
    L2 paper_id 规范化：arxiv_id 去版本号（2401.01234v2 → 2401.01234），
       同一篇论文的多个版本是同一个库条目。arxiv_to_paper_id 是唯一换算点。
    L3 写操作（add/remove/save_note）在本模块直接执行、不内嵌确认门：
       确认门是应用策略，属于 client 侧 executor（计划 §六.1）。server 侧
       任何写工具都不得携带策略，否则换一个 MCP client 策略就丢了。
    L4 并发模型：进程内 threading.Lock。stdio 模式单进程单 client 够用；
       http 模式（M5）也按单 worker 部署（与 ShopMate api 同款纪律）。
    L5 幂等 add：重复入库同一 paper_id 返回既有条目 + created=False，
       不报错不重复写——写工具的幂等性让上层重试安全。
"""
import json
import re
import threading
import time
import uuid
from pathlib import Path

from .schemas import LibraryEntry, Note, PaperMeta

_LIBRARY_LOCK = threading.Lock()
_NOTE_ID_RE = re.compile(r"^[a-f0-9]{8}$")

def arxiv_to_paper_id(arxiv_id: str) -> str:
    """L2：arxiv_id 去版本号。'2401.01234v2' → '2401.01234'。"""
    return re.sub(r"v\d+$", "", arxiv_id.strip())

class Library:
    """文献库。每个 MCP server 进程持有一个实例（stdio: 1 个；http: 每 worker 1 个）。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.entries_file = self.root / "papers.jsonl"
        self.notes_dir = self.root / "notes"
        self.pdfs_dir = self.root / "pdfs"          # download_pdf 的落点，remove 时级联清理

    # ---------- 内部 ----------

    def _read_entries(self) -> list[LibraryEntry]:
        """全量读入内存。M1 规模（个人库，几十~几百条）不需要更聪明。"""
        if not self.entries_file.exists():
            return []
        out: list[LibraryEntry] = []
        for line in self.entries_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(LibraryEntry.model_validate(json.loads(line)))
        return out

    def _rewrite_entries(self, entries: list[LibraryEntry]) -> None:
        """全量重写。tmp + 原子 replace，绝不留半截 JSONL。"""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.entries_file.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "\n".join(e.model_dump_json() for e in entries) + ("\n" if entries else ""),
            encoding="utf-8")
        tmp.replace(self.entries_file)

    def _note_path(self, paper_id: str, note_id: str) -> Path:
        return self.notes_dir / paper_id / f"{note_id}.md"

    # ---------- 读操作 ----------

    def get_entry(self, paper_id: str) -> LibraryEntry | None:
        paper_id = arxiv_to_paper_id(paper_id)
        for e in self._read_entries():
            if e.paper_id == paper_id:
                return e
        return None

    def list_entries(self) -> list[LibraryEntry]:
        """按入库时间正序（老的在前）。"""
        return sorted(self._read_entries(), key=lambda e: e.added_at)

    def list_notes(self, paper_id: str) -> list[Note]:
        paper_id = arxiv_to_paper_id(paper_id)
        d = self.notes_dir / paper_id
        if not d.exists():
            return []
        notes: list[Note] = []
        for f in sorted(d.glob("*.md")):
            note_id = f.stem
            if not _NOTE_ID_RE.match(note_id):        # 用户手建的杂文件不进契约
                continue
            notes.append(Note(note_id=note_id, paper_id=paper_id,
                              text=f.read_text(encoding="utf-8"),
                              created_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(f.stat().st_mtime))))
        return notes

    # ---------- 写操作（L3：直接执行，确认门在 client 侧） ----------

    def add_paper(self, meta: PaperMeta, reason: str) -> tuple[LibraryEntry, bool]:
        """入库。返回 (条目, 是否新入库)。L5 幂等：已存在返回既有条目。"""
        paper_id = arxiv_to_paper_id(meta.arxiv_id)
        with _LIBRARY_LOCK:
            existing = self.get_entry(paper_id)
            if existing is not None:
                return existing, False
            entry = LibraryEntry(paper_id=paper_id, meta=meta,
                                 added_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 added_reason=reason)
            with _open_append(self.entries_file) as f:
                f.write(entry.model_dump_json() + "\n")
            return entry, True

    def remove_paper(self, paper_id: str) -> bool:
        """移除条目 + 级联删笔记目录 + 删已下载的 PDF。返回是否真删了东西。"""
        paper_id = arxiv_to_paper_id(paper_id)
        with _LIBRARY_LOCK:
            entries = self._read_entries()
            kept = [e for e in entries if e.paper_id != paper_id]
            existed = len(kept) < len(entries)
            if existed:
                self._rewrite_entries(kept)
            import shutil
            shutil.rmtree(self.notes_dir / paper_id, ignore_errors=True)
            for v in (f"{paper_id}.pdf",):
                p = self.pdfs_dir / v
                p.unlink(missing_ok=True)
            return existed

    def save_note(self, paper_id: str, text: str) -> Note:
        """存一条笔记。text 为空串报 ValueError（诚实拒绝假写）。"""
        paper_id = arxiv_to_paper_id(paper_id)
        text = text.strip()
        if not text:
            raise ValueError("笔记内容为空")
        note_id = uuid.uuid4().hex[:8]
        path = self._note_path(paper_id, note_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return Note(note_id=note_id, paper_id=paper_id, text=text,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

class _open_append:
    """entries_file 的追加写上下文（确保父目录存在）。"""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8")
        return self.f

    def __exit__(self, *exc):
        self.f.close()
        return False

# ---------------- 自测：python -m mcp_server.library（全程离线，tempdir） ----------------
if __name__ == "__main__":
    import sys
    import tempfile
    from .schemas import ExternalText

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 0. L2 规范化纯函数
    assert arxiv_to_paper_id("2401.01234v2") == "2401.01234"
    assert arxiv_to_paper_id("2401.01234") == "2401.01234"
    print("0 arxiv_to_paper_id OK：去版本号 / 无版本号原样")

    meta = PaperMeta(
        arxiv_id="2401.01234v2", title="A Study of Diffusion",
        authors=["Alice Chen"], published="2024-01-05",
        pdf_url="https://arxiv.org/pdf/2401.01234v2.pdf",
        abstract=ExternalText(text="We study diffusion.", source="arXiv:2401.01234v2/abstract"))

    with tempfile.TemporaryDirectory() as td:
        lib = Library(Path(td))

        # 1. add + 幂等（L5）
        entry, created = lib.add_paper(meta, reason="用户要求：把第一篇存进库")
        assert created is True and entry.paper_id == "2401.01234"
        entry2, created2 = lib.add_paper(meta, reason="重复请求")
        assert created2 is False and entry2.paper_id == "2401.01234"
        assert len(lib.list_entries()) == 1, "重复入库不得产生第二条"
        print("1 add_paper OK：新入库 created=True；重复幂等 created=False")

        # 2. get / list / added_reason 持久化
        got = lib.get_entry("2401.01234v2")            # 传带版本号的 id 也能查到（L2）
        assert got is not None and got.added_reason == "用户要求：把第一篇存进库"
        print("2 get_entry OK：带版本 id 归一化命中，added_reason 已持久化")

        # 3. 笔记：save / list / 空内容拒绝
        n1 = lib.save_note("2401.01234", "第 3 节的实验设置值得复现")
        assert n1.note_id and n1.paper_id == "2401.01234"
        notes = lib.list_notes("2401.01234v2")
        assert len(notes) == 1 and notes[0].text == "第 3 节的实验设置值得复现"
        try:
            lib.save_note("2401.01234", "   ")
            raise AssertionError("空笔记应拒绝")
        except ValueError:
            pass
        print("3 save_note OK：落盘可读回；空内容 ValueError 诚实拒绝")

        # 4. remove 级联：条目 + 笔记目录 + PDF 一起消失
        (lib.pdfs_dir).mkdir(parents=True, exist_ok=True)
        (lib.pdfs_dir / "2401.01234.pdf").write_bytes(b"%PDF-fake")
        assert lib.remove_paper("2401.01234") is True
        assert lib.remove_paper("2401.01234") is False, "第二次删应返回 False"
        assert lib.get_entry("2401.01234") is None
        assert lib.list_notes("2401.01234") == []
        assert not (lib.pdfs_dir / "2401.01234.pdf").exists()
        print("4 remove_paper OK：条目/笔记/PDF 级联清理，重复删返回 False")

    print("\nlibrary 自测 5/5 通过（全程离线）")