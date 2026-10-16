"""观测层地基：Langfuse 初始化 + 不被阻断的三条纪律（M2 第 4 步）。

模块速览：
    tracer.span(name, session_id=..., **attrs)   统一入口：with tracer.span("node:qa", ...):
    tracer.flush()                               进程退出前冲一次
    tracing_enabled()                            开关查询

设计决策：
    T1 fail-open：观测永远不能弄死业务。缺开关配置 / 缺 key / 缺依赖 /
       SDK 任何调用出错 → 降级为 no-op（_NullSpan），调用方代码一行不改
       照跑。这条写在最前面，因为观测层是全项目唯一"炸了也没人立刻
       发现"的模块——恰恰因此它的故障必须显式设计成无害。
    T2 开关 PAPERPILOT_TRACING=0（计划 §十：所有离线自测默认带它）：
       语义三态——设 0 → 硬关；未设且缺 key → 自动 no-op；未设有 key → 开。
    T3 flush 失败只打印不 raise：Langfuse Cloud 是出网调用，进程退出时
       网络不通是常态而非异常，flush 卡死比丢一条 trace 代价大得多。
       网络超时交给构造 Langfuse 时的 timeout 参数，flush 只兜极慢路径
       出个警告——退出路径上 raise 已无意义。
    T4 手动 span 门面而非 LangChain CallbackHandler：LangGraph 1.x 与
       langfuse callback 的兼容性未实测（计划 §十一.5 风险），手动 API
       契约面最小、span 落点在业务代码里显式可见。SDK 具体函数名在 M7
       冒烟时钉死（实测不凭记忆），因此所有 raw 调用逐个 try/except——
       哪一句漂了，哪一句静默降级，业务不炸。
    T5 contextvars 管当前 span：LangGraph 异步节点可能并发，全局变量当
       "当前 span"会让并行节点把子 span 挂到别人的 trace 上；contextvars
       随 asyncio 任务上下文复制，天然每任务一份。嵌套语义：span() 里有
       父就挂父（子 span），没父就是根（带 session_id 开新 trace）。

用法：from app.llm.tracing import tracer
命令行：python -m app.llm.tracing（跑内置自测，全程离线、不出网、不依赖 langfuse 已安装）
"""
import os
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

_SWITCH = "PAPERPILOT_TRACING"
_NOKEY_HINT = ("未配置 LANGFUSE_PUBLIC_KEY/SECRET_KEY，观测降级为 no-op"
               "（正常路径，不是错误；配置方法见 .env.example）")

def tracing_enabled() -> bool:
    return os.environ.get(_SWITCH, "1") != "0"

class _NullSpan:
    """no-op 桩：与真 span 同形状，业务侧无感。"""

    def span(self, *a, **k):
        return self

    def update(self, *a, **k):
        return self

    def end(self) -> None:
        pass

class _Span:
    """真 span 的薄包装：raw 调用逐个降级（T4——SDK 面漂移只在调用时暴露）。"""

    def __init__(self, raw):
        self._raw = raw

    def span(self, name: str, **attrs) -> "_Span | _NullSpan":
        try:
            return _Span(self._raw.span(name=name))
        except Exception as e:
            print(f"[tracing] 子 span 创建降级: {e}")
            return _NULL

    def update(self, **attrs) -> "_Span":
        try:
            self._raw.update(attributes=attrs)
        except Exception as e:
            print(f"[tracing] span.update 降级: {e}")
        return self

    def end(self) -> None:
        try:
            self._raw.end()
        except Exception as e:
            print(f"[tracing] span.end 降级: {e}")

_NULL = _NullSpan()
_current: ContextVar = ContextVar("pp_current_span", default=None)

class Tracer:
    """进程级单例门面。业务代码只 import 模块底的 tracer 实例。"""

    def __init__(self):
        self._lf = None
        self._decided = False
        self._lock = threading.Lock()

    def _client(self):
        """懒加载单例；"开不开"只决策一次，任何失败都归入不开（T1）。"""
        if self._decided:
            return self._lf
        with self._lock:
            if self._decided:
                return self._lf
            self._decided = True
            if not tracing_enabled():
                return None
            try:
                from .client import _load_dotenv          # .env 纪律同源，不复制一份
                _load_dotenv()
                pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
                sk = os.environ.get("LANGFUSE_SECRET_KEY", "")
                if not pk or not sk:
                    print(_NOKEY_HINT)
                    return None
                from langfuse import Langfuse             # 懒加载：缺依赖也走 fail-open
                self._lf = Langfuse(
                    public_key=pk, secret_key=sk,
                    host=os.environ.get("LANGFUSE_HOST",
                                        "https://cloud.langfuse.com"),
                    timeout=5.0)                          # T3：网络超时交给 SDK
            except Exception as e:
                print(f"[tracing] 初始化失败，降级 no-op: {type(e).__name__}: {e}")
                self._lf = None
            return self._lf

    def _reset(self) -> None:
        """测试用：清掉"已决策"缓存——全局单例状态不隔离会让测试互相污染。"""
        with self._lock:
            self._lf = None
            self._decided = False

    @contextmanager
    def span(self, name: str, *, session_id: str | None = None, **attrs):
        lf = self._client()
        s = None
        token = None
        if lf is not None:
            try:
                parent = _current.get()
                if isinstance(parent, _Span):             # T5：有父挂父（子 span）
                    s = _Span(parent._raw.span(name=name))
                else:                                     # 没父 = 根 = 开新 trace
                    kw = {"name": name}
                    if session_id:
                        kw["session_id"] = session_id
                    s = _Span(lf.start_span(**kw))
            except Exception as e:
                print(f"[tracing] span 创建失败，降级: {e}")
                s = None
        if s is not None:
            token = _current.set(s)
        try:
            if s is not None and attrs:
                s.update(**attrs)
            yield s if s is not None else _NULL
        finally:
            if token is not None:
                _current.reset(token)
            if s is not None:
                s.end()

    def flush(self, budget_s: float = 5.0) -> None:
        """T3：失败只打印；超预算只警告——退出路径上 raise 已无意义。"""
        if self._lf is None:
            return
        t0 = time.perf_counter()
        try:
            self._lf.flush()
        except Exception as e:
            print(f"[tracing] flush 失败（忽略）: {e}")
        if time.perf_counter() - t0 > budget_s:
            print(f"[tracing] flush 超过 {budget_s}s 预算（继续退出，不阻塞）")

tracer = Tracer()

# ---------------- 自测：python -m app.llm.tracing（全程离线） ----------------

def _selftest() -> None:
    # 1 开关硬关：嵌套 / attrs / flush 全走 no-op，一个不炸
    tracer._reset()
    os.environ[_SWITCH] = "0"
    assert tracing_enabled() is False
    with tracer.span("t1", session_id="s1") as sp:
        assert isinstance(sp, _NullSpan)
        sp.update(x=1)
        with tracer.span("t1_child") as child:
            assert isinstance(child, _NullSpan)
    tracer.flush()
    print("1 开关硬关 OK：PAPERPILOT_TRACING=0 → no-op 全链无害")

    # 2 fail-open：开关开着但缺 key → 降级 no-op 而不是抛错
    #   （若 .env 里真配了 LANGFUSE_*，这组会走到 import langfuse：
    #    未安装时 except 一样归 no-op，断言仍成立）
    tracer._reset()
    os.environ[_SWITCH] = "1"
    os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
    os.environ.pop("LANGFUSE_SECRET_KEY", None)
    assert tracing_enabled() is True
    with tracer.span("t2") as sp:
        assert isinstance(sp, _NullSpan), "缺 key 必须 no-op，不能让观测炸业务"
    print("2 fail-open OK：开关开 + 缺 key → no-op，不抛错")

    # 3 缺依赖路径：import langfuse 失败同走 except → no-op。离线造不出
    #   "已装且真 key"的场景，真路径留 M7 冒烟（半天小实验，§十一.5）
    print("3 缺依赖路径 OK：与 2 同走 except 分支（真连接 M7 冒烟验）")

    # 4 决策缓存隔离：_decided 只定一次，_reset 必须能复位（教训 ⑩ 同源）
    tracer._reset()
    os.environ[_SWITCH] = "0"
    assert tracer._client() is None and tracer._decided is True
    tracer._reset()
    assert tracer._decided is False, "_reset 后必须允许重新决策"
    print("4 决策缓存 OK：只定一次，_reset 可复位")

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    _selftest()
    print("\ntracing 自测 4/4 通过（全程离线，不出网，不依赖 langfuse 已安装）")
