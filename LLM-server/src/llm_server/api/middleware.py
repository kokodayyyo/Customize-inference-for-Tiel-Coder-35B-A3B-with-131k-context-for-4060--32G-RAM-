"""网关的鉴权、并发限流与错误封装。"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, MutableMapping

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from .metrics import METRICS

# 用量登记表挂在 ASGI scope 上（同一个可变 dict 对象贯穿请求生命周期），
# 转发层写入、MetricsMiddleware 结算。用对象引用而不是 contextvar，是因为
# 流式响应在独立任务里迭代，contextvar 的写回语义不可靠。
USAGE_KEY = "llm_usage"


def usage_of(request: Request) -> MutableMapping[str, Any] | None:
    """取出当前请求的用量登记表。"""
    scope = getattr(request, "scope", None)
    if isinstance(scope, dict):
        entry = scope.get(USAGE_KEY)
        if isinstance(entry, dict):
            return entry
    return None


def record_usage(
    request: Request,
    *,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """累加本次请求的 token 用量与模型名。"""
    data = usage_of(request)
    if data is None:
        return
    if model:
        data["model"] = model
    data["prompt_tokens"] += max(0, int(prompt_tokens))
    data["completion_tokens"] += max(0, int(completion_tokens))

log = logging.getLogger("llm.gateway")

# 免鉴权的路径（存活探针 / 监控）
# /stats 与 /metrics 同为运维观测端点，若要求鉴权会导致监控系统无法采集，
# 且它们不暴露对话内容，因此一并开放。
PUBLIC_PATHS = {
    "/", "/health", "/healthz", "/stats", "/metrics",
    "/v1/models", "/docs", "/redoc", "/openapi.json",
    # 控制台的**页面外壳**开放（它本身不含任何数据），但 /admin/* 数据接口
    # 仍需鉴权 —— 那些接口能启停进程。页面会提示输入 API Key 并随请求带上。
    "/ui", "/admin",
}


def check_api_key(request: Request, api_key: str) -> None:
    """校验 ``Authorization: Bearer <key>``。``api_key`` 为空表示内网免鉴权。"""
    if not api_key:
        return
    if request.url.path in PUBLIC_PATHS:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = request.headers.get("x-api-key", "").strip()
    if token != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key 无效或缺失，请在 Authorization 头中提供 Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


class ConcurrencyLimiter:
    """限制同时转发到 llama.cpp 的请求数，超出的排队；队列满则返回 503。

    llama.cpp 的 ``--parallel`` 槽位决定了真实并发能力，网关排队可以避免
    请求在 llama.cpp 内部无界堆积导致延迟雪崩。
    """

    def __init__(self, limit: int, max_queue: int) -> None:
        self.limit = max(1, limit)
        self.max_queue = max(0, max_queue)
        self._semaphore = asyncio.Semaphore(self.limit)
        self._waiting = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            if self._waiting >= self.max_queue:
                METRICS.on_rejected()
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"服务繁忙：已有 {self.limit} 个请求在处理、{self._waiting} 个在排队，请稍后重试",
                )
            self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            async with self._lock:
                self._waiting -= 1

    def release(self) -> None:
        self._semaphore.release()

    @property
    def waiting(self) -> int:
        return self._waiting


async def timing_middleware(request: Request, call_next):
    """记录每个请求的耗时并写到访问日志。"""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}"
    return response


class MetricsMiddleware:
    """纯 ASGI 中间件：统计所有请求（含流式响应），不缓冲响应体。"""

    def __init__(self, app, cfg) -> None:  # noqa: ANN001 - ASGI app
        self.app = app
        self.cfg = cfg

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        tracked = path.startswith("/v1/") and path not in ("/v1/models",)
        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message) -> None:  # noqa: ANN001
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        if tracked:
            METRICS.on_start()
        # 每个请求一个独立的用量登记表，并发请求互不影响
        scope[USAGE_KEY] = {"model": "", "prompt_tokens": 0, "completion_tokens": 0}
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            usage = scope.get(USAGE_KEY) or {}
            if tracked:
                METRICS.on_finish(
                    elapsed,
                    model=str(usage.get("model", "")),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    failed=status_code >= 400,
                )
                if self.cfg.log_requests:
                    log.info(
                        "%s %s -> %s  %.2fs  in=%s out=%s",
                        scope.get("method"), path, status_code, elapsed,
                        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                    )


def error_response(message: str, code: str = "internal_error", status_code: int = 500) -> JSONResponse:
    """OpenAI 风格的错误体。"""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "code": code}},
    )
