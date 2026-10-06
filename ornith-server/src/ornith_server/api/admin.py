"""模型管理与网页控制台。

职责：
  * **ModelManager** —— 持有当前 ``llama-server`` 进程，支持"点一下换模型"：
    停掉旧进程、把该模型的 profile 应用到 ``ServerConfig``、再启动新进程。
    加载要 10-15 秒，所以走后台线程，前端轮询 ``/admin/status`` 看进度。
  * **管理 API** —— ``/admin/models``（列模型）、``/admin/activate``（启动）、
    ``/admin/stop``、``/admin/status``、``/admin/log``。
  * **网页控制台** —— ``/ui``（浏览器直接访问 ``/`` 也会跳过去）。

安全：只监听内网、且这些接口会启动/停止进程，因此当配置了 ``api_key`` 时
它们同样受鉴权中间件保护（``/admin/*`` 不在 PUBLIC_PATHS 里）。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import ServerConfig
from ..core.backend import query_gpus
from ..core.server import LlamaBackendServer
from ..models_registry import ModelEntry, ModelRegistry

log = logging.getLogger("ornith.admin")

# 网页控制台的静态文件目录：src/ornith_server/web/
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 状态机
STATE_IDLE = "idle"  # 没有后端在跑
STATE_LOADING = "loading"  # 正在加载模型
STATE_RUNNING = "running"  # 就绪
STATE_ERROR = "error"  # 上次加载失败


class ModelManager:
    """拥有后端进程，并允许在运行时换模型。"""

    def __init__(
        self,
        cfg: ServerConfig,
        server: LlamaBackendServer | None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.cfg = cfg
        self.server = server
        self.registry = registry or ModelRegistry()
        self.state = STATE_IDLE
        self.message = ""
        self.active_path = str(cfg.model_path) if cfg.model_path else ""
        self.started_at = 0.0
        self.load_seconds = 0.0
        self.changed_fields: list[str] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self.state == STATE_LOADING

    def status(self) -> dict[str, Any]:
        gpus = []
        query_gpus.cache_clear()  # 让界面看到实时显存，而不是进程启动时的快照
        for g in query_gpus():
            gpus.append({
                "name": g.name, "total_gib": g.total_gib, "free_gib": g.free_gib,
            })
        profile = self.server.profile if self.server else None
        return {
            "state": self.state,
            "message": self.message,
            "busy": self.busy,
            "active_path": self.active_path,
            "alias": self.cfg.model_alias,
            "context_size": profile.context_size if profile else self.cfg.context_size,
            "ubatch": profile.ubatch_size if profile else self.cfg.effective_ubatch,
            "kv_type": profile.kv_type if profile else self.cfg.kv_cache_type_k,
            "kv_location": profile.kv_location if profile else self.cfg.kv_placement,
            "moe_location": profile.moe_location if profile else self.cfg.moe_placement,
            "load_mode": profile.load_mode_label if profile else self.cfg.effective_load_mode,
            "load_note": profile.note if profile else "",
            "load_seconds": self.load_seconds,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0.0,
            "pid": self.server.pid if self.server else None,
            "command": self.server.command_line() if self.server else "",
            "log_path": str(self.server.log_path) if self.server and self.server.log_path else "",
            "gpus": gpus,
            "changed_fields": self.changed_fields,
            "registry_error": self.registry.load_error,
        }

    def log_tail(self, lines: int = 40) -> str:
        return self.server.log_tail(lines) if self.server else ""

    # ------------------------------------------------------------------
    # 扫描
    # ------------------------------------------------------------------
    def list_models(self) -> dict[str, Any]:
        entries = self.registry.scan()
        models = [e.to_dict(self.active_path) for e in entries if not e.is_projector]
        projectors = [e.to_dict(self.active_path) for e in entries if e.is_projector]
        return {
            "search_roots": [str(p) for p in self.registry.search_roots],
            "models": models,
            "projectors": projectors,
            "error": self.registry.load_error,
        }

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------
    def start_sync(self) -> bool:
        """按当前 cfg 同步启动（用于服务刚起来时的 autostart）。"""
        if self.server is None:
            return False
        if self.server.is_running:
            self.state = STATE_RUNNING
            self.started_at = time.time()
            return True
        self.state = STATE_LOADING
        self.message = "正在加载模型…"
        started = time.time()
        try:
            self.server.start()
        except Exception as exc:  # noqa: BLE001 - 要把失败原因暴露到界面
            self.state = STATE_ERROR
            self.message = str(exc)
            log.error("后端启动失败：%s", exc)
            return False
        self.load_seconds = time.time() - started
        self.state = STATE_RUNNING
        self.message = ""
        self.started_at = time.time()
        self.active_path = str(self.cfg.model_path)
        return True

    def activate(self, model_path: str) -> dict[str, Any]:
        """换到指定模型（异步）。返回是否已受理。"""
        if self.busy:
            return {"ok": False, "error": "正在加载中，请等当前模型加载完成"}

        entry = self.registry.find(model_path)
        if entry is None:
            return {"ok": False, "error": f"找不到模型文件: {model_path}"}
        if entry.is_projector:
            return {"ok": False, "error": "这是视觉投影文件（mmproj），不是可加载的模型"}

        with self._lock:
            self.state = STATE_LOADING
            self.message = f"正在切换到 {entry.label}…"
            self._thread = threading.Thread(
                target=self._activate_worker, args=(entry,), daemon=True, name="model-switch"
            )
            self._thread.start()
        return {"ok": True, "model": entry.label, "path": str(entry.path)}

    def _activate_worker(self, entry: ModelEntry) -> None:
        if self.server is None:
            self.state = STATE_ERROR
            self.message = "没有可用的 llama.cpp 后端"
            return
        started = time.time()
        try:
            log.info("切换模型 -> %s", entry.path)
            self.message = "正在停止当前后端…"
            self.server.stop()

            self.changed_fields = self.registry.apply_to_config(entry, self.cfg)
            problems = self.cfg.validate()
            if problems:
                raise RuntimeError("配置有问题：" + "；".join(problems))
            for warn in self.cfg.warnings():
                log.warning("配置告警：%s", warn)

            # server 读的是同一个 cfg 对象，改完 cfg 再 start 就是新模型
            self.message = f"正在加载 {entry.label}（约 10-15 秒）…"
            self.server.start()

            self.load_seconds = time.time() - started
            self.active_path = str(entry.path)
            self.started_at = time.time()
            self.state = STATE_RUNNING
            self.message = ""
            log.info(
                "模型已切换：%s (ctx=%s ubatch=%s) 用时 %.1fs",
                entry.label, self.cfg.context_size, self.cfg.effective_ubatch,
                self.load_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - 失败要显示在界面上
            self.state = STATE_ERROR
            self.message = str(exc)
            self.load_seconds = time.time() - started
            log.error("切换模型失败：%s", exc)

    def stop(self) -> dict[str, Any]:
        if self.busy:
            return {"ok": False, "error": "正在加载中，无法停止"}
        if self.server is not None:
            self.server.stop()
        self.state = STATE_IDLE
        self.message = ""
        self.started_at = 0.0
        return {"ok": True}


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

def create_admin_router(manager: ModelManager) -> APIRouter:
    router = APIRouter()

    @router.get("/admin/models")
    async def list_models() -> dict[str, Any]:
        return manager.list_models()

    @router.get("/admin/status")
    async def status() -> dict[str, Any]:
        return manager.status()

    @router.get("/admin/log", include_in_schema=False)
    async def log_tail(lines: int = 40) -> dict[str, Any]:
        return {"log_path": manager.status().get("log_path", ""),
                "tail": manager.log_tail(max(1, min(lines, 400)))}

    @router.post("/admin/activate")
    async def activate(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"ok": False, "error": "请求体必须是 JSON"}, status_code=400)
        path = str((body or {}).get("path") or "").strip()
        if not path:
            return JSONResponse({"ok": False, "error": "缺少 path"}, status_code=400)
        result = manager.activate(path)
        return JSONResponse(result, status_code=200 if result.get("ok") else 409)

    @router.post("/admin/stop")
    async def stop() -> JSONResponse:
        result = manager.stop()
        return JSONResponse(result, status_code=200 if result.get("ok") else 409)

    @router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    @router.get("/admin", response_class=HTMLResponse, include_in_schema=False)
    async def ui() -> HTMLResponse:
        index = WEB_DIR / "index.html"
        if not index.is_file():
            return HTMLResponse(f"<h1>缺少界面文件</h1><p>{index}</p>", status_code=500)
        return HTMLResponse(index.read_text(encoding="utf-8"))

    return router
