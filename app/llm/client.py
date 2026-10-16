"""LLM 调用层：DeepSeek 客户端封装（全项目唯一的 LLM 出口）。M2 第 1 步。

继承 ShopMate 的五个决策（编号对齐，方便跨项目对照），一个关键差异：
    D0 异步版：MCP 适配器（mcp_client.py）与 LangGraph 都是 async 优先，
    LLM 层同步会逼着上层在事件循环里跑 to_thread——不如本层直接 async。
    safe_call 相应变成 async safe_call。

    D1 .env 先于环境变量兜底（setdefault，已有环境变量优先）
    D2 单例 + 懒加载，缺 key 的报错带补救指引
    D3 json_mode 纪律：prompt 必须含 "json" 字样（DeepSeek 官方约束），
       本层只传开关不改 prompt
    D4 重试收口：连接类抖动重试一次，仍失败返回 None 并打印真实异常
       （上层普遍吞异常，这里是唯一看得见原因的地方）
    D5 不认系统代理：显式 HTTPS_PROXY 才走代理（Windows 注册表代理坑，
       详见 ShopMate client.py 的完整复盘；本层用 async httpx 复刻）
"""
import os
import threading
import time
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

def _load_dotenv() -> None:
    """D1：.env 的 KEY=VALUE 塞进环境变量（已有的不覆盖）。"""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()          # import 即生效

_client = None          # AsyncOpenAI 实例
_lock = threading.Lock()

def _proxy_from_env() -> str | None:
    """D5：只认显式环境变量（大写优先，小写兜底）。纯函数以便自测断言。"""
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or None

def build_async_http_client():
    """D5 异步版。RAGAS / LangGraph 的异步路径都从这里拿客户端，
    不给默认 trust_env=True 留任何口子。"""
    import httpx
    proxy = _proxy_from_env()
    return (httpx.AsyncClient(proxy=proxy, trust_env=False) if proxy
            else httpx.AsyncClient(trust_env=False))

def get_client():
    """D2：返回单例 AsyncOpenAI。缺 key 抛 RuntimeError，带补救指引。"""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError(
                    "缺少 DEEPSEEK_API_KEY。请在项目根目录 .env 文件里加一行：\n"
                    "    DEEPSEEK_API_KEY=sk-你的key\n"
                    "（.env 已被 git 忽略，不会提交）")
            from openai import AsyncOpenAI
            _client = AsyncOpenAI(base_url=BASE_URL, api_key=key,
                                  http_client=build_async_http_client())   # D5
    return _client

async def chat(messages: list[dict], *, tools: list | None = None,
               temperature: float = 0.3, json_mode: bool = False):
    """调一轮对话（async），返回 assistant 消息对象（含 .content / .tool_calls）。

    json_mode=True 时走 DeepSeek 结构化输出（prompt 里必须含 "json" 字样，D3）。
    """
    kwargs: dict = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = await get_client().chat.completions.create(**kwargs)
    return resp.choices[0].message

async def chat_text(messages: list[dict], *, temperature: float = 0.3) -> str:
    """便捷版：只要文本。工具编排别用这个（拿不到 tool_calls）。

    返回 "" = HTTP 200 但模型没吐字（DeepSeek 偶发，ShopMate 2026-09-17 实测
    15 条样本 5 条为空）；None（来自 safe_call）= 压根没调通。两件事分开处理。
    """
    return (await chat(messages, temperature=temperature)).content or ""

# ---------------- 出网失败的重试策略（D4） ----------------

RETRYABLE_HINTS = ("Timeout", "Connection", "RateLimit", "APIStatus",
                   "InternalServer", "ServiceUnavailable")
RETRY_DELAY_S = 0.8

def retryable(e: Exception) -> bool:
    """按异常类名判断是否值得重试（不 import openai，懒加载纪律）。"""
    name = type(e).__name__
    return any(h in name for h in RETRYABLE_HINTS)

async def safe_call(fn, *args, **kwargs):
    """D4 async 版：连接类抖动重试一次；仍失败返回 None 并打印真实异常。"""
    last: Exception | None = None
    for attempt in range(2):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last = e
            if attempt == 0 and retryable(e):
                time.sleep(RETRY_DELAY_S)
                continue
            break
    print(f"[llm] 调用失败，已降级: {type(last).__name__}: {last}")
    return None

# ---------------- 自测：python -m app.llm.client ----------------
if __name__ == "__main__":
    import asyncio
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 1. 缺 key 报错路径（不联网）
    saved = os.environ.pop("DEEPSEEK_API_KEY", None)
    try:
        _client = None
        get_client()
        raise AssertionError("缺 key 应该报错")
    except RuntimeError as e:
        print("1 缺 key 报错 OK：", str(e).splitlines()[0])
    finally:
        if saved is not None:
            os.environ["DEEPSEEK_API_KEY"] = saved
        _client = None

    # 2. safe_call 重试判据（不联网，桩函数）
    class _FakeTimeout(Exception):
        pass

    class _FakeAuthError(Exception):
        pass

    async def _main():
        conn_calls: list[int] = []

        async def _conn_fail():
            conn_calls.append(1)
            raise _FakeTimeout("连接超时")

        assert await safe_call(_conn_fail) is None
        assert len(conn_calls) == 2, f"抖动应重试一次（共 2 次），实际 {len(conn_calls)}"

        auth_calls: list[int] = []

        async def _auth_fail():
            auth_calls.append(1)
            raise _FakeAuthError("key 无效")

        assert await safe_call(_auth_fail) is None
        assert len(auth_calls) == 1, "认证错误不该重试"
        print(f"2 safe_call 重试判据 OK：抖动试 {len(conn_calls)} 次 / 认证只试 {len(auth_calls)} 次")

        # 3. D5 代理决策（不联网）
        saved_px = {k: os.environ.pop(k, None) for k in ("HTTPS_PROXY", "https_proxy")}
        try:
            assert _proxy_from_env() is None, "环境变量空必须直连（注册表代理无视）"
            os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
            assert _proxy_from_env() == "http://127.0.0.1:7890", "显式设了就走"
            os.environ.pop("HTTPS_PROXY")
            os.environ["https_proxy"] = "http://127.0.0.1:7890"
            assert _proxy_from_env() == "http://127.0.0.1:7890", "小写同样认"
        finally:
            for k, v in saved_px.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        print("3 D5 代理决策 OK：环境变量空 → 直连；设了 HTTPS_PROXY → 走它")

        # 4. 联网冒烟（有 key 才跑；验证端点/模型名/key/async 四件套）
        if os.environ.get("DEEPSEEK_API_KEY"):
            reply = await chat_text([{"role": "user", "content": "只回复两个字：在的"}],
                                    temperature=0.0)
            assert reply, "冒烟返回空串"
            print("4 联网冒烟 OK：", reply)
        else:
            print("4 联网冒烟: 跳过（.env 里还没有 DEEPSEEK_API_KEY）")

    asyncio.run(_main())
    print("\nllm.client 自测通过")