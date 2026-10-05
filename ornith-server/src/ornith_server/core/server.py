"""llama.cpp 后端进程管理。

用 Python ``subprocess`` 直接拉起 ``llama-server``，加载 ``--model`` 指向的
GGUF 文件，并以 CUDA 后端把全部层卸载到显存。

关键点：
- **直载文件**：``--model <path.gguf>`` 由 llama.cpp 直接 mmap 加载，不经任何
  中间转换或量化步骤。
- **显存保护**：8GB 笔记本卡上，``context_size`` 过大或 ``ubatch_size`` 过大
  都会 CUDA OOM。这里实现了分级降级：先降上下文，再降 ubatch，最后退部分
  层到 CPU，保证服务一定能起来。
- **就绪判定**：轮询 ``/health``，同时监控进程是否已退出（避免空等）。
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import httpx

from ..config import ServerConfig
from .backend import LlamaBackend
from .jobobject import ChildJob

log = logging.getLogger("ornith.backend")

# 判定 CUDA 显存不足的关键词（llama.cpp / CUDA 运行时输出）
_OOM_MARKERS = (
    "out of memory",
    "cuda_error_out_of_memory",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
    "cudamalloc failed",
    "failed to allocate",
    "unable to allocate",
    "insufficient memory",
    "outofmemory",
)

# 判定"宿主内存（锁页内存）分配失败"的关键词。
#
# 这**不是显存不足**，必须与 _OOM_MARKERS 分开判断，否则会去降 ubatch 之类的
# 显存参数，而病因其实在系统内存。实测遇到：
#   E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 15704850432
#   E unable to allocate CUDA_Host buffer
# 15704850432 B = 14.63 GiB，正是专家权重。
#
# 成因：``--load-mode none`` 会把 --cpu-moe 的专家张量放进 ``CUDA_Host``
# （cudaMallocHost 锁页内存），需要一次性锁定约 14.6 GiB **物理**内存。
# 页缓存占满或内存碎片化时会**瞬时**失败（启动后 1.9 秒就退出）。
# 对策是回落到 mmap —— 那时 CPU 张量来自文件映射，可回收，不需要大块锁定。
_HOST_MEM_MARKERS = (
    "cuda_host",
    "ggml_backend_cpu_buffer_type_alloc_buffer",
    "unable to allocate cuda_host",
    "failed to allocate cuda_host",
)

_EXIT_DLL_NOT_FOUND = 0xC0000135


@dataclass
class LoadProfile:
    """一次加载尝试使用的参数组合。"""

    context_size: int
    ubatch_size: int
    batch_size: int
    gpu_layers: int
    kv_type: str
    note: str
    kv_offload: bool = True
    cpu_moe: bool = True
    n_cpu_moe: int = -1
    # 空字符串 = 用 llama.cpp 默认；"none" 不用 mmap；见 _HOST_MEM_MARKERS 的说明
    load_mode: str = ""

    @property
    def kv_location(self) -> str:
        return "显存" if self.kv_offload else "内存"

    @property
    def moe_location(self) -> str:
        if self.n_cpu_moe >= 0:
            return f"前{self.n_cpu_moe}层专家→内存"
        return "专家→内存" if self.cpu_moe else "专家→显存"

    @property
    def load_mode_label(self) -> str:
        return self.load_mode or "默认"


def build_load_profiles(cfg: ServerConfig) -> list[LoadProfile]:
    """按显存占用从高到低生成降级序列。

    降级顺序的设计依据（RTX 4060 Laptop 8GB + Tile-35B-A3B IQ4_XS 实测）：

    与 9B 稠密模型不同，MoE 下显存压力主要来自 **计算缓冲与 KV**，专家权重
    已经在内存里。实测显存天花板很硬：总占用 6768 MiB 时正常，7894 MiB 时
    decode 从 30.9 崩到 14.6 tok/s（开始颠簸），所以阶梯要**平滑地小步降**：

    1. 配置值；
    2. 把放到显存里的专家退回内存（仅当 n_cpu_moe >= 0，这一步省得最多）；
    3. 逐级降 ubatch（4096 -> 2048 -> 1024 -> 512）——它是 MoE 下影响
       prefill 最大的参数，也是计算缓冲的主要来源；
    4. KV 量化降到 q4_0；
    5. KV cache 整体挪到内存（--no-kv-offload）；
    6. 上下文降到 75%；
    7. 最后才牺牲 GPU 层数（对速度伤害最大）。
    """
    ctx = cfg.context_size

    def prof(ctx_size, ubatch, batch, ngl, kv, note, kv_off=None, cpu_moe=None,
             n_cpu=None, load_mode=...):
        return LoadProfile(
            context_size=ctx_size,
            ubatch_size=ubatch,
            batch_size=batch,
            gpu_layers=ngl,
            kv_type=kv,
            note=note,
            kv_offload=cfg.kv_offload if kv_off is None else kv_off,
            cpu_moe=cfg.cpu_moe if cpu_moe is None else cpu_moe,
            n_cpu_moe=cfg.n_cpu_moe if n_cpu is None else n_cpu,
            load_mode=cfg.effective_load_mode if load_mode is ... else load_mode,
        )

    want = prof(ctx, cfg.effective_ubatch, max(cfg.batch_size, cfg.effective_ubatch),
                cfg.gpu_layers, cfg.kv_cache_type_k, "配置值（首选）")

    fallbacks: list[LoadProfile] = []
    # 2. 显存不够时先把放到显存里的专家收回内存（MoE 独有的第一步，省得最多）
    if cfg.n_cpu_moe >= 0:
        fallbacks.append(
            prof(ctx, cfg.effective_ubatch, max(cfg.batch_size, cfg.effective_ubatch),
                 cfg.gpu_layers,
                 cfg.kv_cache_type_k, "专家全部退回内存（--cpu-moe）",
                 cpu_moe=True, n_cpu=-1)
        )
    # 3. 逐级降 ubatch：计算缓冲随它线性增长，是 MoE 下调显存最有效的一挡
    ub = cfg.effective_ubatch
    for step in (1024, 512):
        if step >= ub:
            continue
        fallbacks.append(
            prof(ctx, step, max(step * 2, 2048), cfg.gpu_layers, cfg.kv_cache_type_k,
                 f"ubatch 降到 {step}（压缩计算缓冲）")
        )
    fallbacks += [
        # 4. 压 KV 精度
        prof(ctx, 1024, 2048, cfg.gpu_layers, "q4_0", "KV 量化降到 q4_0"),
        # 5. KV 挪内存
        prof(ctx, 1024, 2048, cfg.gpu_layers, "q8_0", "KV cache 移到内存", kv_off=False),
        # 6. 才动上下文
        prof(int(ctx * 0.75) // 512 * 512, 1024, 2048, cfg.gpu_layers, "q8_0",
             "上下文降至 75%", kv_off=False),
        prof(min(ctx, 32768), 1024, 2048, cfg.gpu_layers, "q4_0",
             "上下文 32768 + KV 放内存", kv_off=False),
        # 7. 最后牺牲层数
        prof(min(ctx, 16384), 512, 1024, max(1, cfg.gpu_layers - 8), "q4_0",
             "再把 8 层留给 CPU（牺牲速度换稳定）", kv_off=False),
    ]

    profiles: list[LoadProfile] = []
    seen: set[tuple] = set()
    for item in [want, *fallbacks]:
        key = (item.context_size, item.ubatch_size, item.batch_size,
               item.gpu_layers, item.kv_type, item.kv_offload,
               item.cpu_moe, item.n_cpu_moe)
        if key in seen or item.context_size < 512 or item.gpu_layers < 1:
            continue
        seen.add(key)
        profiles.append(item)
    return profiles


class LlamaBackendServer:
    """托管一个 ``llama-server`` 子进程。"""

    def __init__(self, cfg: ServerConfig, backend: LlamaBackend) -> None:
        self.cfg = cfg
        self.backend = backend
        self.process: subprocess.Popen | None = None
        self.profile: LoadProfile | None = None
        self.log_path: Path | None = None
        self._log_handle = None
        # 兜底：父进程被强杀（taskkill /F、任务管理器、崩溃）时，由系统
        # 结束 llama-server，避免孤儿进程一直占着显存。
        self._job: ChildJob | None = None

    # ------------------------------------------------------------------
    # 命令行
    # ------------------------------------------------------------------
    def build_command(self, prof: LoadProfile) -> list[str]:
        cfg = self.cfg
        # ubatch 超过实测安全上限时夹紧（除非显式放开）。4096 会在 qwen35moe +
        # --cpu-moe 下触发 "CUDA error: an illegal memory access"，后端直接死。
        ubatch = prof.ubatch_size
        if not cfg.allow_large_ubatch and ubatch > ServerConfig.SAFE_UBATCH:
            log.warning(
                "ubatch %d 超过实测安全上限 %d（会触发 CUDA 越界），已夹紧。"
                "确要实验请设 allow_large_ubatch: true",
                ubatch, ServerConfig.SAFE_UBATCH,
            )
            ubatch = ServerConfig.SAFE_UBATCH
        batch = max(prof.batch_size, ubatch)
        cmd: list[str] = [
            str(self.backend.exe),
            "--model", str(cfg.model_file),
            "--alias", cfg.model_alias,
            "--host", cfg.backend_host,
            "--port", str(cfg.backend_port),
            "--n-gpu-layers", str(prof.gpu_layers),
            "--ctx-size", str(prof.context_size),
            "--parallel", str(cfg.parallel_slots),
            "--batch-size", str(batch),
            "--ubatch-size", str(ubatch),
            "--threads", str(cfg.threads),
            "--flash-attn", "on" if cfg.flash_attention else "off",
            "--cache-type-k", prof.kv_type,
            "--cache-type-v", prof.kv_type,
            # KV cache 放显存还是内存：长上下文场景下放内存反而更快（避开
            # 显存碎片与计算缓冲争抢），且是 8GB 卡跑 128K 的唯一办法。
            "--kv-offload" if prof.kv_offload else "--no-kv-offload",
        ]

        # ---- MoE 专家权重摆位 ----
        # 只影响 *_exps 张量；attention / SSM / shared expert / embedding 仍按
        # --n-gpu-layers 上显存。n_cpu_moe 优先于 cpu_moe。
        if prof.n_cpu_moe >= 0:
            cmd += ["--n-cpu-moe", str(prof.n_cpu_moe)]
        elif prof.cpu_moe:
            cmd.append("--cpu-moe")
        for rule in cfg.tensor_overrides:
            cmd += ["--override-tensor", rule]

        if cfg.threads_batch > 0:
            cmd += ["--threads-batch", str(cfg.threads_batch)]
        if cfg.mmproj_path:
            cmd += ["--mmproj", str(cfg.mmproj_path)]
        # --load-mode 取代了旧的 --no-mmap / --mlock；profile 的值优先（降级时会换），
        # 其次才是配置。
        load_mode = prof.load_mode
        if load_mode:
            cmd += ["--load-mode", load_mode]
        if cfg.no_warmup:
            cmd.append("--no-warmup")
        if cfg.jinja:
            cmd.append("--jinja")
        if not cfg.cont_batching:
            cmd.append("--no-cont-batching")
        if cfg.cache_reuse > 0:
            cmd += ["--cache-reuse", str(cfg.cache_reuse)]
        if cfg.metrics:
            cmd.append("--metrics")
        if cfg.reasoning_budget >= 0:
            cmd += ["--reasoning-budget", str(cfg.reasoning_budget)]
        return cmd

    def command_line(self) -> str:
        """返回当前（或首选）命令行文本，便于用户复制排查。"""
        prof = self.profile or build_load_profiles(self.cfg)[0]
        return subprocess.list2cmdline(self.build_command(prof))

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------
    def start(
        self,
        profiles: Sequence[LoadProfile] | None = None,
        on_attempt: Callable[[LoadProfile, int, int], None] | None = None,
    ) -> LoadProfile:
        """启动后端，必要时自动降级重试。返回最终成功的参数组合。"""
        if self.is_running:
            assert self.profile is not None
            log.info("后端已在运行 (pid=%s)", self.process.pid if self.process else "?")
            return self.profile

        attempts = list(profiles or build_load_profiles(self.cfg))
        last_error = "未知错误"

        index = 0
        while index < len(attempts):
            prof = attempts[index]
            index += 1
            if on_attempt:
                on_attempt(prof, index, len(attempts))
            log.info(
                "尝试加载 [%d/%d] ctx=%d ubatch=%d batch=%d ngl=%d kv=%s(%s) %s load=%s — %s",
                index, len(attempts), prof.context_size, prof.ubatch_size,
                prof.batch_size, prof.gpu_layers, prof.kv_type, prof.kv_location,
                prof.moe_location, prof.load_mode_label, prof.note,
            )
            outcome, detail = self._attempt(prof)
            if outcome == "ready":
                self.profile = prof
                log.info("后端就绪: pid=%s", self.process.pid if self.process else "?")
                return prof

            last_error = detail
            self._cleanup_process()
            if outcome == "dll_missing":
                raise RuntimeError(
                    "llama-server 启动失败：缺少 CUDA 运行时 DLL (0xC0000135)。\n"
                    f"请确认存在 vendor 目录并把其中的 cudart/cublas DLL 加入 PATH。\n"
                    f"当前 vendor: {self.backend.vendor_dir}\n{detail}"
                )
            # 注意：``fatal`` 也**继续试下一个 profile**，不再直接放弃。
            # 教训：llama.cpp 在内存吃紧时会以硬断言崩溃
            #   GGML_ASSERT(ctx->mem_buffer != NULL) failed
            # 这不是"配置错误"，重试更省的 profile 往往就能起来。只有 DLL 缺失
            # 这种环境问题才值得立刻终止。

            # 宿主内存（锁页内存）分配失败：换 mmap 用**同一套参数**再试一次。
            # mmap 下 CPU 张量来自文件映射、可回收，不需要一次性锁定十几 GiB
            # 物理内存，所以能绕开这个问题；代价是 decode 约慢 17%。
            # 这必须排在降 ubatch 之前——病因不在显存，降 ubatch 治不了。
            #
            # ``fatal``（崩溃/断言，例如内存吃紧时的
            # GGML_ASSERT(ctx->mem_buffer != NULL) failed）原因不明，也先试 mmap；
            # 但 ``oom`` 是明确的显存不足，试 mmap 必然无效，直接走降 ubatch。
            if outcome in ("host_mem", "fatal") and prof.load_mode != "mmap":
                reason = ("宿主内存锁定失败" if outcome == "host_mem"
                          else "进程崩溃/断言失败，原因不明")
                retry = replace(
                    prof,
                    load_mode="mmap",
                    note=f"{prof.note} + {reason}，回落 mmap",
                )
                attempts.insert(index, retry)
                log.warning(
                    "%s（--load-mode none 需一次性锁定约 14.6 GiB 锁页内存），"
                    "改用 mmap 重试同一套参数…",
                    reason,
                )
                time.sleep(1)
                continue

            log.warning("本次加载失败（%s），准备降级重试…", outcome)
            time.sleep(2)

        raise RuntimeError(
            "所有参数组合都无法加载模型，最后错误：\n"
            f"{last_error}\n提示：可减小 context_size 或 parallel_slots 后重试。"
        )

    def _attempt(self, prof: LoadProfile) -> tuple[str, str]:
        """执行一次加载。返回 (结果, 细节)。

        结果取值：``ready`` / ``oom`` / ``host_mem`` / ``dll_missing`` /
        ``fatal`` / ``timeout``

        ``host_mem``（宿主锁页内存分配失败）与 ``oom``（显存不足）必须分开：
        前者要回落 mmap，后者要降 ubatch / KV，两者对策完全不同。
        """
        cmd = self.build_command(prof)
        self.log_path = self._new_log_path(prof)
        self._log_handle = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self._log_handle.write("CMDLINE: " + subprocess.list2cmdline(cmd) + "\n\n")
        self._log_handle.flush()

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                cwd=str(self.backend.home),
                env=self.backend.build_env(),
                creationflags=creationflags,
            )
        except OSError as exc:
            return "fatal", f"无法启动进程: {exc}"

        # 把子进程纳入 Job Object：父进程无论怎么死，llama-server 都会被清理。
        self._job = ChildJob(label=prof.note)
        if self._job.assign(self.process):
            log.debug("已将 llama-server (pid=%s) 加入 Job Object", self.process.pid)
        else:
            log.debug("Job Object 绑定未生效：%s", self._job.error or "不支持")

        deadline = time.time() + self.cfg.startup_timeout
        health_url = f"{self.cfg.backend_base_url}/health"

        while time.time() < deadline:
            code = self.process.poll()
            if code is not None:
                tail = self._log_tail()
                if (code & 0xFFFFFFFF) == _EXIT_DLL_NOT_FOUND:
                    return "dll_missing", f"退出码 0xC0000135\n{tail}"
                # 先判宿主内存：它的日志里也含 "failed to allocate"，
                # 若先判 OOM 就会被误分类。
                if _looks_like_host_mem_failure(tail):
                    return "host_mem", f"退出码 {code}\n{tail}"
                if _looks_like_oom(tail):
                    return "oom", f"退出码 {code}\n{tail}"
                return "fatal", f"退出码 {code}\n{tail}"

            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(health_url)
                if resp.status_code == 200:
                    body = resp.json() if resp.content else {}
                    status = str(body.get("status", "ok"))
                    if status in ("ok", "no slot available"):
                        return "ready", ""
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(1.0)

        tail = self._log_tail()
        return "timeout", f"启动超时（{self.cfg.startup_timeout}s）\n{tail}"

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.is_running and self.process else None

    def wait_ready(self, timeout: int | None = None) -> bool:
        """等待 ``/health`` 返回 ok。"""
        deadline = time.time() + (timeout or self.cfg.startup_timeout)
        while time.time() < deadline:
            if not self.is_running:
                return False
            try:
                with httpx.Client(timeout=2.0) as client:
                    resp = client.get(f"{self.cfg.backend_base_url}/health")
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
        return False

    def stop(self, grace: float = 12.0) -> None:
        """优雅停止：先 terminate，超时后 kill，确保显存释放。"""
        self._cleanup_process(grace=grace)

    def _cleanup_process(self, grace: float = 12.0) -> None:
        proc, self.process = self.process, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                log.warning("后端未在 %.0fs 内退出，强制结束", grace)
                proc.kill()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    pass
            except OSError:
                pass

        # 关闭作业句柄。若上面没能结束进程，KILL_ON_JOB_CLOSE 会补上最后一刀。
        if self._job is not None:
            if proc is not None and proc.poll() is None:
                self._job.terminate()
            self._job.close()
            self._job = None

        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def _new_log_path(self, prof: LoadProfile) -> Path:
        from ..config import LOG_DIR

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return LOG_DIR / f"llama-server-ctx{prof.context_size}-{stamp}.log"

    def _log_tail(self, lines: int = 25) -> str:
        if not self.log_path or not self.log_path.is_file():
            return ""
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def log_tail(self, lines: int = 25) -> str:
        """公开的日志尾部读取，供网关在转发失败时给出可诊断的错误。"""
        return self._log_tail(lines)


def _looks_like_oom(log_text: str) -> bool:
    lowered = log_text.lower()
    return any(marker in lowered for marker in _OOM_MARKERS)


def _looks_like_host_mem_failure(log_text: str) -> bool:
    """是否为宿主（锁页）内存分配失败——不是显存问题。"""
    lowered = log_text.lower()
    return any(marker in lowered for marker in _HOST_MEM_MARKERS)


def make_backend(cfg: ServerConfig, backend: LlamaBackend | None = None) -> LlamaBackendServer:
    """便捷工厂：解析后端并包装成进程管理器。"""
    from .backend import resolve_backend

    resolved = backend or resolve_backend(cfg.llama_dir or None)
    return LlamaBackendServer(cfg, resolved)
