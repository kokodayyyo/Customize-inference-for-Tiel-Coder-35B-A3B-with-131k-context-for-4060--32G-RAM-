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
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..config import ServerConfig
from ..core import sysinfo
from ..core.backend import query_gpus
from ..core.server import LlamaBackendServer
from ..models_registry import ModelEntry, ModelRegistry
from .metrics import METRICS

log = logging.getLogger("llm.admin")

# 网页控制台的静态文件目录：src/llm_server/web/
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 历史曲线保留的采样点数（前端每 2 秒拉一次，120 点 = 4 分钟）
HISTORY_POINTS = 150

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
        # 只有后端**真的起来了**才算"当前模型"。配置里写了 model_path 不代表
        # 它在跑 —— 用 --no-autostart 启动时一个模型都没加载，这里要是预先
        # 填上，界面会把一个没运行的模型标成"使用中"。
        self.active_path = ""
        self.started_at = 0.0
        self.load_seconds = 0.0
        self.changed_fields: list[str] = []
        self._lock = threading.Lock()
        # metrics() 现在跑在线程池里（路由改成了同步 def），可能并发；用它保护
        # _last_metrics 增量基线，避免两次采样交错算出荒谬的实时速度。
        self._metrics_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        # 运行时指标采样：CPU 占用需要两次采样，实时吞吐要用累计值做差
        self._samplers = sysinfo.Samplers.create()
        self._last_metrics: tuple[float, float, float, float] | None = None
        self._history: deque[dict[str, Any]] = deque(maxlen=HISTORY_POINTS)

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
            "mmproj_path": self.cfg.mmproj_path,
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
    # 运行时指标（网页控制台的数据来源）
    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        """线程安全入口：串行化采样，避免并发交错污染实时速度基线。"""
        with self._metrics_lock:
            return self._collect_metrics()

    def _collect_metrics(self) -> dict[str, Any]:
        """聚合展示用的一切：GPU / 内存 / CPU / 后端吞吐 / KV 占用 / 网关统计。

        设计要点：llama.cpp 的 ``/metrics`` 给的是**累计值**，直接用它算平均速度
        会把空闲时间也算进去。这里保存上一次采样，用**增量**算实时速度，
        而且分别用 ``prompt_tokens_seconds`` / ``predicted_tokens_seconds``
        的增量做分母 —— 那是 llama.cpp 自己统计的**各阶段纯耗时**，
        所以得到的是"真正在算的时候有多快"。
        """
        pid = self.server.pid if self.server else None
        snap = sysinfo.snapshot(self._samplers, pid)

        bm: dict[str, float] = {}
        props: dict[str, Any] = {}
        slots: list[dict] = []
        if self.server is not None and self.server.is_running:
            bm = self.server.fetch_backend_metrics()
            props = self.server.fetch_backend_props()
            slots = self.server.fetch_slots()

        now = time.time()
        # llama.cpp 的实际指标名（容易记错，这里以 /metrics 实测为准）：
        #   prompt_tokens_total / prompt_seconds_total          <- 预填充累计 token 与累计耗时
        #   tokens_predicted_total / tokens_predicted_seconds_total  <- 生成同上
        #   prompt_tokens_cached_total / n_decode_total / n_tokens_max
        #   requests_processing / requests_deferred / n_busy_slots_per_decode
        # 注意：本构建**没有** kv_cache_* 指标，KV 占用要从 /slots 推。
        prompt_total = bm.get("prompt_tokens_total", 0.0)
        prompt_seconds = bm.get("prompt_seconds_total", 0.0)
        pred_total = bm.get("tokens_predicted_total", 0.0)
        pred_seconds = bm.get("tokens_predicted_seconds_total", 0.0)

        live: dict[str, float | None] = {"prefill_tps": None, "decode_tps": None}
        if self._last_metrics is not None:
            pt, ps, dt_, ds = self._last_metrics
            d_tok, d_sec = prompt_total - pt, prompt_seconds - ps
            if d_sec > 0.05 and d_tok >= 0:
                live["prefill_tps"] = round(d_tok / d_sec, 1)
            d_tok2, d_sec2 = pred_total - dt_, pred_seconds - ds
            if d_sec2 > 0.05 and d_tok2 >= 0:
                live["decode_tps"] = round(d_tok2 / d_sec2, 2)
        self._last_metrics = (prompt_total, prompt_seconds, pred_total, pred_seconds)

        # 平均速度（累计口径），用于交叉验证实时值
        avg = {
            "prefill_tps": round(prompt_total / prompt_seconds, 1) if prompt_seconds > 0 else None,
            "decode_tps": round(pred_total / pred_seconds, 2) if pred_seconds > 0 else None,
        }

        # ---- KV cache 占用：从 /slots 推（本构建的 /metrics 不提供）----
        slot_list: list[dict[str, Any]] = []
        kv_tokens = 0
        ctx_total = None
        for s in slots:
            n_ctx = s.get("n_ctx")
            n_prompt = s.get("n_prompt_tokens") or 0
            if n_ctx:
                ctx_total = (ctx_total or 0) + n_ctx
            kv_tokens += n_prompt
            # next_token 在不同版本里是 dict 或 list，两种都要兼容
            nxt = s.get("next_token")
            if isinstance(nxt, list):
                nxt = nxt[0] if nxt else {}
            if not isinstance(nxt, dict):
                nxt = {}
            slot_list.append({
                "id": s.get("id"),
                "state": "处理中" if s.get("is_processing") else "空闲",
                "n_ctx": n_ctx,
                "prompt_tokens": n_prompt,
                "processed": s.get("n_prompt_tokens_processed"),
                "cached": s.get("n_prompt_tokens_cache"),
                "n_remain": nxt.get("n_remain"),
                "n_decoded": nxt.get("n_decoded"),
            })
        if ctx_total is None:
            if props:
                ctx_total = (props.get("default_generation_settings") or {}).get("n_ctx")
            if ctx_total is None and self.server is not None and self.server.profile:
                ctx_total = self.server.profile.context_size
        kv_ratio = (kv_tokens / ctx_total) if (ctx_total and kv_tokens is not None) else None

        point = {
            "t": now,
            "prefill_tps": live["prefill_tps"],
            "decode_tps": live["decode_tps"],
            "kv_ratio": kv_ratio,
            "gpu_util": (snap["gpus"][0].get("util_percent") if snap["gpus"] else None),
            "cpu": snap.get("cpu_percent"),
        }
        self._history.append(point)

        gateway = METRICS.snapshot()
        gateway["queue_waiting"] = 0  # 由路由层从限流器补齐
        return {
            "ts": now,
            "system": {
                "memory": snap.get("memory", {}),
                "cpu_percent": snap.get("cpu_percent"),
                "cpu_count": snap.get("cpu_count"),
                "gpus": snap.get("gpus", []),
                "process": snap.get("process", {}),
            },
            "backend": {
                "running": bool(self.server and self.server.is_running),
                "state": self.state,
                "pid": pid,
                "live": live,
                "avg": avg,
                "prompt_tokens_total": prompt_total,
                "predicted_tokens_total": pred_total,
                "prompt_tokens_cached_total": bm.get("prompt_tokens_cached_total"),
                "prompt_seconds_total": round(prompt_seconds, 1),
                "predicted_seconds_total": round(pred_seconds, 1),
                "n_decode_total": bm.get("n_decode_total"),
                "n_tokens_max": bm.get("n_tokens_max"),
                "requests_processing": bm.get("requests_processing"),
                "requests_deferred": bm.get("requests_deferred"),
                "n_busy_slots_per_decode": bm.get("n_busy_slots_per_decode"),
                "kv_cache_usage_ratio": kv_ratio,
                "kv_cache_tokens": kv_tokens,
                "context_total": ctx_total,
                "slots_total": props.get("total_slots"),
                "slots": slot_list,
                "model_path": props.get("model_path"),
                "model_alias": props.get("model_alias"),
                "model_ftype": props.get("model_ftype"),
                "modalities": props.get("modalities"),
                "is_sleeping": props.get("is_sleeping"),
                "build_info": props.get("build_info"),
                "chat_template": bool(props.get("chat_template")),
            },
            "gateway": gateway,
            "history": list(self._history),
        }

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

    def activate(self, model_path: str, vision: bool | None = None) -> dict[str, Any]:
        """换到指定模型（异步）。返回是否已受理。

        ``vision``：前端视觉开关。True=加载 mmproj，False=不加载，None=沿用
        模型 profile 的默认（没写就默认关闭）。
        """
        if self.busy:
            return {"ok": False, "error": "正在加载中，请等当前模型加载完成"}

        entry = self.registry.find(model_path)
        if entry is None:
            return {"ok": False, "error": f"找不到模型文件: {model_path}"}
        if entry.is_projector:
            return {"ok": False, "error": "这是视觉投影文件（mmproj），不是可加载的模型"}

        use_vision = entry.vision_default if vision is None else bool(vision)
        if use_vision and not entry.vision_supported:
            return {"ok": False, "error": "该模型没有检测到视觉组件（mmproj），无法开启视觉"}

        with self._lock:
            self.state = STATE_LOADING
            self.message = f"正在切换到 {entry.label}…"
            self._thread = threading.Thread(
                target=self._activate_worker, args=(entry, vision),
                daemon=True, name="model-switch",
            )
            self._thread.start()
        return {
            "ok": True, "model": entry.label, "path": str(entry.path),
            "vision": use_vision,
            "mmproj": entry.mmproj_path if use_vision else "",
        }

    def _activate_worker(self, entry: ModelEntry, vision: bool | None = None) -> None:
        if self.server is None:
            self.state = STATE_ERROR
            self.message = "没有可用的 llama.cpp 后端"
            return
        use_vision = entry.vision_default if vision is None else bool(vision)
        started = time.time()
        try:
            log.info("切换模型 -> %s (vision=%s)", entry.path, use_vision)
            self.message = "正在停止当前后端…"
            self.server.stop()
            # 旧模型已停：先摘掉"当前模型"标记。否则这次切换若失败，界面会把
            # 已经停掉的旧模型一直标成使用中，按钮灰掉、无法重新启动。
            self.active_path = ""
            self.server.profile = None

            self.changed_fields = self.registry.apply_to_config(entry, self.cfg)
            # 视觉开关：**总是显式设置**，否则从"开了视觉"的模型切到别的模型时
            # cfg.mmproj_path 会残留，导致新模型意外加载旧模型的眼睛。
            if use_vision:
                if not entry.mmproj_path:
                    raise RuntimeError("该模型没有检测到视觉组件（mmproj）")
                self.cfg.mmproj_path = entry.mmproj_path
            else:
                self.cfg.mmproj_path = ""
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
            # 后端已经停了：清掉"当前模型"标记。否则控制台会继续把已停掉的模型
            # 标成使用中（卡片高亮、状态栏还是旧参数），它的「启动此模型」按钮
            # 一直是灰的，用户点不动，看起来就像"启动失败"。
            self.server.profile = None
        self.state = STATE_IDLE
        self.message = ""
        self.active_path = ""
        self.started_at = 0.0
        return {"ok": True}


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

def create_admin_router(manager: ModelManager, proxy: Any = None) -> APIRouter:
    router = APIRouter()

    def _metrics() -> dict[str, Any]:
        data = manager.metrics()
        if proxy is not None:
            # 排队数在网关的限流器里，不在 METRICS 里
            data["gateway"]["queue_waiting"] = proxy.limiter.waiting
        return data

    # 这些处理器内部是同步的（nvidia-smi 子进程 + 阻塞式 httpx），所以声明为普通
    # ``def``：Starlette 会把它们丢到线程池执行，绝不阻塞事件循环。若写成
    # ``async def`` 而没有 await，采样期间整个网关（含 /v1 流式转发）都会卡住。
    @router.get("/admin/models")
    def list_models() -> dict[str, Any]:
        return manager.list_models()

    @router.get("/admin/status")
    def status() -> dict[str, Any]:
        return manager.status()

    @router.get("/admin/metrics")
    def metrics() -> dict[str, Any]:
        """网页控制台的仪表盘数据（GPU / 内存 / CPU / 吞吐 / KV / 网关统计）。"""
        return _metrics()

    @router.get("/admin/log", include_in_schema=False)
    def log_tail(lines: int = 40) -> dict[str, Any]:
        return {"log_path": manager.status().get("log_path", ""),
                "tail": manager.log_tail(max(1, min(lines, 400)))}

    # ---- 扫描目录：前端可增删（持久化到 config/model_roots.json）----
    @router.get("/admin/roots")
    def list_roots() -> dict[str, Any]:
        reg = manager.registry
        return {
            "roots": [str(p) for p in reg.search_roots],
            "defaults": [str(p) for p in reg.default_roots],
            "state_file": str(reg.roots_state_file),
            "error": reg.roots_error or reg.load_error,
        }

    @router.post("/admin/roots")
    async def update_roots(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"ok": False, "error": "请求体必须是 JSON"}, status_code=400)
        body = body or {}
        action = str(body.get("action") or "add")
        reg = manager.registry
        if action == "remove":
            path = str(body.get("path") or "").strip()
            if not path:
                return JSONResponse({"ok": False, "error": "缺少 path"}, status_code=400)
            if not reg.remove_root(path):
                return JSONResponse({"ok": False, "error": "该目录不在列表中"}, status_code=404)
        elif action == "set":
            reg.set_roots([str(p) for p in (body.get("roots") or [])])
        else:  # add
            ok, why = reg.add_root(str(body.get("path") or ""))
            if not ok:
                return JSONResponse({"ok": False, "error": why}, status_code=400)
        return JSONResponse({
            "ok": True, "roots": [str(p) for p in reg.search_roots], "error": reg.roots_error,
        })

    @router.get("/admin/browse")
    def browse(path: str = "") -> dict[str, Any]:
        """列出某目录下的子目录，供前端「选择文件夹」。空路径 → 列出盘符。

        注意：它只是目录枚举（不读文件内容），但会暴露服务器的目录结构，
        所以和 /admin/* 一样受 api_key 保护；内网开放（api_key 为空）时请自行
        评估信任边界。
        """
        raw = (path or "").strip().strip('"')
        if not raw:
            drives = [f"{c}:\\" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      if Path(f"{c}:\\").exists()]
            return {"path": "", "parent": "", "drives": drives, "dirs": [], "error": ""}
        p = Path(raw).expanduser()
        if not p.is_dir():
            return {"path": str(p), "parent": "", "drives": [], "dirs": [],
                    "error": f"目录不存在或不是文件夹: {p}"}
        dirs: list[dict[str, Any]] = []
        try:
            for child in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                if not child.is_dir():
                    continue
                try:
                    has_gguf = any(
                        f.suffix.lower() == ".gguf" for f in child.iterdir() if f.is_file()
                    )
                except OSError:
                    has_gguf = False
                dirs.append({"name": child.name, "path": str(child), "has_gguf": has_gguf})
        except OSError as exc:
            return {"path": str(p), "parent": str(p.parent), "drives": [], "dirs": [],
                    "error": f"无法读取目录: {exc}"}
        parent = str(p.parent) if p.parent != p else ""
        return {"path": str(p), "parent": parent, "drives": [], "dirs": dirs, "error": ""}

    @router.post("/admin/activate")
    async def activate(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"ok": False, "error": "请求体必须是 JSON"}, status_code=400)
        path = str((body or {}).get("path") or "").strip()
        if not path:
            return JSONResponse({"ok": False, "error": "缺少 path"}, status_code=400)
        vision = (body or {}).get("vision")
        if vision is not None:
            vision = bool(vision)
        result = manager.activate(path, vision=vision)
        return JSONResponse(result, status_code=200 if result.get("ok") else 409)

    @router.post("/admin/stop")
    def stop() -> JSONResponse:
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
