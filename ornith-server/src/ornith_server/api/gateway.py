"""内网 API 网关：对外提供 OpenAI 兼容接口。

架构：::

    内网客户端 ──HTTP──> FastAPI 网关 (:8000, 0.0.0.0)
                              │  鉴权 / 限流 / 统计 / 流式透传
                              └──HTTP──> llama-server (:8080, 127.0.0.1)
                                              └── CUDA ──> GGUF 模型

之所以在 llama.cpp 前面再套一层，是为了在不牺牲推理速度的前提下拿到
内网服务必需的能力：API Key 鉴权、并发排队保护、Prometheus 指标、
访问日志，以及统一的多模态/重排等扩展入口。推理本身仍由 llama.cpp
的 CUDA 内核完成，网关不做任何逐 token 的加工。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from ..config import ServerConfig, local_ip_addresses
from ..core import LlamaBackendServer, make_backend, query_gpus
from .admin import ModelManager, create_admin_router
from .metrics import METRICS
from .middleware import (
    ConcurrencyLimiter,
    MetricsMiddleware,
    check_api_key,
    error_response,
    record_usage,
)

log = logging.getLogger("ornith.gateway")

# 需要整体缓冲（非流式）的转发路径
BUFFERED_PATHS = {
    "/v1/models": "GET",
    "/v1/embeddings": "POST",
    "/v1/completions": "POST",
    "/v1/chat/completions": "POST",
    "/v1/responses": "POST",
    "/v1/tokenize": "POST",
    "/v1/detokenize": "POST",
    "/props": "GET",
    "/slots": "GET",
}

STREAM_DISABLED_PATHS = {"/v1/embeddings"}


class BackendProxy:
    """把请求转发给 llama-server，并原样（含 SSE 流）返回响应。"""

    def __init__(self, cfg: ServerConfig, backend=None, manager=None) -> None:
        self.cfg = cfg
        self.client: httpx.AsyncClient | None = None
        self.limiter = ConcurrencyLimiter(cfg.parallel_slots, cfg.max_queue)
        # 仅用于在转发失败时判断后端进程是否还活着，给出可诊断的错误信息
        self.backend = backend
        # 换模型期间用它提前拦住请求并返回"正在加载"
        self.manager = manager

    async def open(self) -> None:
        # trust_env=False：本地回环流量绝不能走系统代理。
        # 本机若有 Clash/v2ray 之类的系统代理，httpx 默认会把 127.0.0.1 的
        # 请求也发给代理并拿到 502，整个转发链路都会挂。详见 net.py。
        self.client = httpx.AsyncClient(
            base_url=self.cfg.backend_base_url,
            timeout=httpx.Timeout(self.cfg.request_timeout, connect=10.0),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
            trust_env=False,
        )

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    # ------------------------------------------------------------------
    async def forward(self, request: Request, path: str) -> Response:
        """转发一个请求到后端。"""
        if self.client is None:
            return error_response("网关尚未就绪", "not_ready", 503)

        raw = await request.body()
        payload: dict[str, Any] = {}
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return error_response("请求体不是合法 JSON", "invalid_request_error", 400)
            if not isinstance(payload, dict):
                return error_response("请求体必须是 JSON 对象", "invalid_request_error", 400)

        wants_stream = bool(payload.get("stream")) and path not in STREAM_DISABLED_PATHS
        if wants_stream:
            # 主动要求上游在流末尾附带 usage。llama.cpp 默认不返回，
            # 不设置的话网关的 token 统计会永远是 0。
            options = payload.get("stream_options")
            if not isinstance(options, dict):
                options = {}
            options.setdefault("include_usage", True)
            payload["stream_options"] = options
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        headers = self._forward_headers(request)

        # 换模型期间后端是停着的，这时明确告诉调用方"正在加载"，
        # 比让它去连一个已经关闭的端口、拿到含义模糊的 502 要好得多。
        mgr = getattr(self, "manager", None)
        if mgr is not None and mgr.state == "loading":
            return error_response(
                f"模型正在加载，请稍候重试（{mgr.message}）", "model_loading", 503
            )
        if mgr is not None and mgr.state == "idle":
            return error_response(
                "当前没有模型在运行，请到控制台 /ui 启动一个", "no_model", 503
            )

        await self.limiter.acquire()
        started = time.perf_counter()
        try:
            if wants_stream:
                return await self._stream(request, path, raw, headers, started)
            return await self._buffered(request, path, raw, headers, started)
        except httpx.HTTPError as exc:
            self.limiter.release()
            detail = self._diagnose_upstream(exc)
            log.warning("转发到后端失败: %s", detail)
            return error_response(detail, "upstream_error", 502)
        except BaseException:
            self.limiter.release()
            raise

    # ------------------------------------------------------------------
    def _diagnose_upstream(self, exc: httpx.HTTPError) -> str:
        """把转发失败翻译成能直接照做的提示。

        后端崩溃（例如 CUDA illegal memory access）时 httpx 抛出的异常
        ``str()`` 是空字符串，只回 "后端连接失败: " 对排查毫无帮助。
        """
        text = f"{type(exc).__name__}: {exc}".strip().rstrip(":")
        backend = self.backend
        if backend is not None and not backend.is_running:
            hint = "llama-server 进程已退出（多半是 CUDA 错误或显存不足）。"
            if backend.log_path:
                tail = backend.log_tail(6)
                if tail:
                    hint += "\n后端日志尾部：\n" + tail
                hint += f"\n完整日志：{backend.log_path}"
            return hint
        return f"后端连接失败（{text}）。请检查 llama.cpp 是否仍在运行。"

    # ------------------------------------------------------------------
    async def _buffered(
        self, request: Request, path: str, raw: bytes, headers: dict[str, str], started: float
    ) -> Response:
        assert self.client is not None
        try:
            upstream = await self.client.request(
                request.method,
                path,
                content=raw,
                headers=headers,
                params=dict(request.query_params),
            )
        except BaseException:
            self.limiter.release()
            raise

        self.limiter.release()
        elapsed = time.perf_counter() - started
        self._account(request, upstream, elapsed)

        media_type = upstream.headers.get("content-type", "application/json")
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=media_type.split(";")[0],
            headers={"X-Upstream-Seconds": f"{elapsed:.3f}"},
        )

    async def _stream(
        self, request: Request, path: str, raw: bytes, headers: dict[str, str], started: float
    ) -> Response:
        """SSE 流式透传：边收边发，让客户端尽快看到首 token。"""
        assert self.client is not None
        req = self.client.build_request(
            request.method,
            path,
            content=raw,
            headers=headers,
            params=dict(request.query_params),
        )
        try:
            upstream = await self.client.send(req, stream=True)
        except BaseException:
            self.limiter.release()
            raise

        # 后端报错时不是 SSE，直接缓冲返回，避免客户端解析失败
        if upstream.status_code >= 400:
            content = await upstream.aread()
            await upstream.aclose()
            self.limiter.release()
            return Response(
                content=content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type", "application/json").split(";")[0],
            )

        usage_holder: dict[str, Any] = {}

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    _harvest_usage(chunk, usage_holder)
                    yield chunk
            except httpx.HTTPError as exc:
                log.warning("流式转发中断: %s", exc)
            finally:
                await upstream.aclose()
                self.limiter.release()
                # 结算交给 MetricsMiddleware（本协程与请求共享同一个 usage 字典）
                record_usage(
                    request,
                    model=str(usage_holder.get("model", "")),
                    prompt_tokens=int(usage_holder.get("prompt_tokens", 0)),
                    completion_tokens=int(usage_holder.get("completion_tokens", 0)),
                )

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream").split(";")[0],
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------
    def _forward_headers(self, request: Request) -> dict[str, str]:
        headers: dict[str, str] = {"content-type": "application/json"}
        for name in ("accept", "user-agent"):
            if value := request.headers.get(name):
                headers[name] = value
        return headers

    def _account(self, request: Request, response: httpx.Response, elapsed: float) -> None:
        """从后端响应里提取 token 用量用于统计。"""
        prompt = completion = 0
        model = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                usage = body.get("usage") or {}
                prompt = int(usage.get("prompt_tokens") or 0)
                completion = int(usage.get("completion_tokens") or 0)
                model = str(body.get("model") or "")
        except (ValueError, TypeError):
            pass
        record_usage(
            request, model=model, prompt_tokens=prompt, completion_tokens=completion
        )
        log.debug(
            "-> %s %d %.2fs in=%d out=%d",
            response.request.url.path, response.status_code, elapsed, prompt, completion,
        )


def _harvest_usage(chunk: bytes, holder: dict[str, Any]) -> None:
    """从 SSE 片段里累计 usage / model（OpenAI 流式最后一个 chunk 带 usage）。

    另外统计 ``reasoning_content``：该模型会先输出思考内容，
    调用方若只看 ``content`` 会误以为没有输出。
    """
    text = chunk.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        usage = obj.get("usage")
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                # 流式场景下 usage 是累计值，取最大而不是相加
                if isinstance(value, int) and value > holder.get(key, 0):
                    holder[key] = value
        delta = (obj.get("choices") or [{}])[0].get("delta") or {}
        if delta.get("reasoning_content"):
            holder["reasoning_chunks"] = holder.get("reasoning_chunks", 0) + 1
        if obj.get("model"):
            holder["model"] = str(obj["model"])


# ---------------------------------------------------------------------------
# 应用装配
# ---------------------------------------------------------------------------


def create_app(cfg: ServerConfig, backend_server: LlamaBackendServer | None = None) -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 模型管理器：拥有后端进程，并支持在网页控制台上"点一下换模型"
    manager = ModelManager(cfg, backend_server)

    state: dict[str, Any] = {
        "backend": backend_server,
        "proxy": BackendProxy(cfg, backend_server, manager),
        "manager": manager,
        "ready": False,
        "load_error": "",
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        proxy: BackendProxy = state["proxy"]
        backend: LlamaBackendServer | None = state["backend"]

        if backend is not None and cfg.autostart_backend:
            # 首次启动是同步的：必须等模型就绪，横幅里的信息才准确
            state["ready"] = manager.start_sync()
            if not state["ready"]:
                state["load_error"] = manager.message

        await proxy.open()
        _print_banner(cfg, backend, bool(state["ready"]))
        try:
            yield
        finally:
            await proxy.close()
            if backend is not None and cfg.autostart_backend:
                log.info("正在停止后端进程…")
                backend.stop()

    app = FastAPI(
        title="本地推理内网 API",
        description="基于 llama.cpp CUDA 后端的 OpenAI 兼容内网接口",
        version="1.0.0",
        lifespan=lifespan,
    )

    if cfg.allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.allow_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(MetricsMiddleware, cfg=cfg)

    # 模型管理接口 + 网页控制台（/ui、/admin/*）
    app.include_router(create_admin_router(manager))

    proxy: BackendProxy = state["proxy"]

    # ------------------------------------------------------------------
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):  # noqa: ANN001
        try:
            check_api_key(request, cfg.api_key)
        except HTTPException as exc:
            return error_response(str(exc.detail), "invalid_api_key", exc.status_code)
        return await call_next(request)

    # ------------------------------------------------------------------
    # 运维端点
    # ------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root(request: Request):
        """浏览器访问就给控制台页面，脚本访问（Accept 不含 text/html）给 JSON。"""
        if "text/html" in request.headers.get("accept", ""):
            return RedirectResponse("/ui", status_code=307)
        return {
            "service": "ornith-lan-api",
            "status": "ok" if state["ready"] else "starting",
            "model": cfg.model_alias,
            "endpoints": ["/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/embeddings"],
            "console": "/ui",
            "auth_required": bool(cfg.api_key),
            "docs": "/docs",
        }

    @app.get("/health")
    @app.get("/healthz", include_in_schema=False)
    async def health() -> JSONResponse:
        backend: LlamaBackendServer | None = state["backend"]
        mgr: ModelManager = state["manager"]
        # ready 以管理器为准：换模型期间后端会被停掉，此时必须报 not ready，
        # 否则调用方会在加载窗口期收到一堆难以理解的 502。
        ready = mgr.state == "running" and bool(backend and backend.is_running)
        payload: dict[str, Any] = {
            "gateway": "ok",
            "backend_running": bool(backend and backend.is_running),
            "ready": ready,
            "state": mgr.state,
            "model": cfg.model_alias,
            "pid": backend.pid if backend else None,
            "load_error": mgr.message if mgr.state == "error" else state["load_error"],
        }
        if backend and backend.profile:
            payload["context_size"] = backend.profile.context_size
            payload["context_per_slot"] = cfg.ctx_per_slot
        gpus = query_gpus()
        if gpus:
            payload["gpu"] = [
                {"name": g.name, "total_gib": g.total_gib, "free_gib": g.free_gib} for g in gpus
            ]
        code = 200 if ready else 503
        return JSONResponse(payload, status_code=code)

    @app.get("/stats", include_in_schema=False)
    async def stats() -> dict[str, Any]:
        data = METRICS.snapshot()
        data["queue_waiting"] = proxy.limiter.waiting
        data["parallel_slots"] = cfg.parallel_slots
        return data

    @app.get("/metrics", include_in_schema=False)
    async def prometheus() -> PlainTextResponse:
        return PlainTextResponse(METRICS.render_prometheus(), media_type="text/plain; version=0.0.4")

    # ------------------------------------------------------------------
    # OpenAI 兼容转发
    # ------------------------------------------------------------------
    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "OPTIONS"])
    async def proxy_openai(path: str, request: Request) -> Response:
        target = f"/v1/{path}"
        if not _is_supported(target, request.method):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"不支持的接口: {target}")
        return await proxy.forward(request, target)

    @app.api_route("/props", methods=["GET"], include_in_schema=False)
    async def props(request: Request) -> Response:
        return await proxy.forward(request, "/props")

    @app.api_route("/slots", methods=["GET"], include_in_schema=False)
    async def slots(request: Request) -> Response:
        return await proxy.forward(request, "/slots")

    app.state.cfg = cfg
    app.state.ornith = state
    return app


def _is_supported(path: str, method: str) -> bool:
    if path in ("/v1/models", "/v1/chat/completions", "/v1/completions", "/v1/embeddings",
                "/v1/responses", "/v1/tokenize", "/v1/detokenize"):
        return True
    # 允许 llama.cpp 的其它 v1 扩展（如 /v1/models/<id>）
    return path.startswith("/v1/models/")


def _print_banner(cfg: ServerConfig, backend: LlamaBackendServer | None, ready: bool) -> None:
    """打印启动横幅。

    输出重定向到文件时（如服务化部署）Python 默认沿用系统编码（Windows 是 GBK），
    中文会变成乱码，因此显式按 UTF-8 写字节。
    """
    lines: list[str] = []
    line = "=" * 68
    lines.append(line)
    lines.append("  本地推理内网 API 已启动")
    lines.append(line)
    lines.append(f"  模型      : {cfg.model_alias}")
    lines.append(f"  权重文件  : {cfg.model_file}")
    if backend and backend.profile:
        p = backend.profile
        lines.append(f"  上下文    : {p.context_size} total / {cfg.ctx_per_slot} per slot"
                     f"  (parallel={cfg.parallel_slots})")
        lines.append(f"  KV cache  : {p.kv_type} 位于{p.kv_location}"
                     f"  flash-attn={'on' if cfg.flash_attention else 'off'}"
                     f"  ngl={p.gpu_layers}")
        lines.append(f"  专家权重  : {p.moe_location}")
        lines.append(f"  加载说明  : {p.note}")
    lines.append(f"  后端状态  : {'运行中' if ready else '未就绪'}")
    lines.append(f"  **控制台** : http://127.0.0.1:{cfg.proxy_port}/ui   ← 换模型 / 看状态")
    lines.append(f"  本机访问  : http://127.0.0.1:{cfg.proxy_port}/v1")
    for ip in local_ip_addresses():
        lines.append(f"  内网访问  : http://{ip}:{cfg.proxy_port}/v1")
    lines.append(f"  接口文档  : http://127.0.0.1:{cfg.proxy_port}/docs")
    lines.append(f"  鉴权      : {'已启用 API Key' if cfg.api_key else '未启用（内网开放）'}")
    lines.append(line)

    text = "\n".join(lines) + "\n"
    stream = getattr(sys, "stdout", None)
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        stream.write(text)
        stream.flush()
    except (UnicodeEncodeError, AttributeError):
        # 控制台编码装不下中文时，退化为按 UTF-8 写底层字节
        try:
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except (AttributeError, OSError, ValueError):
            print(text.encode("ascii", errors="replace").decode("ascii"))
    finally:
        log.info(
            "服务已启动：模型=%s 上下文=%s KV=%s/%s 监听=%s:%s 鉴权=%s",
            cfg.model_alias, cfg.context_size,
            backend.profile.kv_type if backend and backend.profile else cfg.kv_cache_type_k,
            backend.profile.kv_location if backend and backend.profile else "?",
            cfg.proxy_host, cfg.proxy_port, bool(cfg.api_key),
        )
