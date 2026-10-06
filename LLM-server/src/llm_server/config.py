"""配置模型与加载逻辑。

配置来源优先级（后者覆盖前者）：
1. 代码内默认值
2. ``config/server.yaml``（若存在），或 ``--config`` 指定的文件
3. 环境变量 ``LLM_*``
4. 命令行参数

默认值是针对 **RTX 4060 Laptop 8GB + Tile-35B-A3B(IQ4_XS)** 实测标定的，
换硬件时主要关注 ``context_size`` / ``parallel_slots`` / ``cpu_moe``。

**MoE 摆位策略**（本项目当前的核心）：
Tile-35B-A3B 是 Qwen3-Next 式混合架构（``qwen35moe``）：41 层里有 4/5 是
线性注意力（SSM/DeltaNet），只有约 10 层是真注意力。因此

* 专家权重（约 33B 参数 / 15 GiB）→ 内存（``cpu_moe``）
* 注意力+SSM+共享专家+embedding（约 2B 参数）→ 显存
* KV cache（只算那 10 层，128K q8_0 约 1.4 GiB）→ 显存（``kv_offload``）

这与 9B 稠密模型的结论相反：那里 128K 的 KV 要 8.5 GiB 只能放内存，这里
KV 放显存既放得下又更快。
"""

from __future__ import annotations

import json
import os
import socket
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# 项目根目录：<root>/src/llm_server/config.py -> <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path(r"D:\models\Tile\Tile-35BA3B\Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf")
# 项目自带的 llama.cpp 运行时（从 LM Studio 复制而来，已自包含，无需再装 LM Studio）
BUNDLED_BACKENDS_DIR = PROJECT_ROOT / "runtime" / "llama.cpp" / "backends"
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "server.yaml"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
LOG_DIR = RUNTIME_DIR / "logs"


@dataclass
class ServerConfig:
    """llama.cpp 后端进程配置。"""

    # ---- 模型 ----
    model_path: str = str(DEFAULT_MODEL)
    model_alias: str = "tile-35b-a3b"
    mmproj_path: str = ""  # 多模态投影文件，留空则不启用视觉（视觉会额外吃显存）

    # ---- 后端 ----
    # 留空 = 优先使用项目自带的 runtime/llama.cpp/backends，找不到才去探测
    # LM Studio 等外部安装。
    llama_dir: str = ""

    # ---- MoE 专家权重摆位（本项目核心）----
    # llama.cpp 的 --cpu-moe / --n-cpu-moe 只影响专家张量（*_exps），
    # attention / SSM / shared expert / embedding 仍按 --n-gpu-layers 上显存。
    cpu_moe: bool = True  # True = 全部专家权重留在内存（--cpu-moe）
    n_cpu_moe: int = -1  # >=0 时只把前 N 层的专家留在内存（--n-cpu-moe N），
    #                      其余层的专家放显存；显存有余量时用它换速度。
    #                      优先级高于 cpu_moe。
    tensor_overrides: list[str] = field(default_factory=list)
    # 逃生舱：直接透传 --override-tensor 规则，例如
    #   ["blk\\.([0-3])\\.ffn_(gate|up|down)_exps\\.weight=CPU"]
    # 用于按张量名做 llama.cpp 覆盖不了的精细摆位。

    # ---- 显存 / 上下文（RTX 4060 Laptop 8GB 实测标定，见 README 的性能矩阵）----
    gpu_layers: int = 99  # 99 = 全部层放 GPU（专家由 cpu_moe 单独控制）
    context_size: int = 131072  # 默认 128K
    parallel_slots: int = 1  # 并发槽位；每槽上下文 = context_size / parallel_slots
    # ubatch 是 MoE 下**最重要**的性能参数：每个 ubatch 都要把 256 个专家全过
    # 一遍，所以 ubatch 越大、同样 prompt 需要的往返越少。实测 prefill：
    #   ubatch  256 ->  322 tok/s     ubatch 2048 -> 1049 tok/s
    #   ubatch  512 ->  443 tok/s     ubatch 4096 -> 1207 tok/s（**会崩，见下**）
    #   ubatch 1024 ->  772 tok/s     ubatch 8192 ->  456 tok/s（显存不够，反而崩）
    #
    # 但 4096 不安全：实测在真实中文长提示下 llama.cpp 会抛
    #   CUDA error: an illegal memory access was encountered
    # （ggml_backend_cuda_synchronize），后端进程直接死掉、请求 502。
    # 4096 时显存只剩 1.78 GiB，推测是 qwen35moe + --cpu-moe 路径下大 ubatch
    # 的缓冲区越界。2048 只比它慢 13% 的 prefill，但显存留 2.76 GiB 且稳定，
    # 因此取 2048。要突破这个上限请设 allow_large_ubatch。
    batch_size: int = 8192  # 逻辑 batch（必须 >= ubatch_size）
    ubatch_size: int = 2048  # 物理 batch；MoE 下直接决定 prefill 速度
    # 允许 ubatch_size 超过实测安全上限（2048）。仅为实验保留，默认关闭：
    # 超限时 build_command 会自动夹到 2048 并打警告。
    allow_large_ubatch: bool = False
    flash_attention: bool = True  # 必须开，配合 KV 量化省显存
    kv_cache_type_k: str = "q8_0"
    kv_cache_type_v: str = "q8_0"
    # KV cache 放显存(True)还是内存(False)。
    # 本模型 41 层里只有 full_attention_interval=4 的那约 10 层需要 KV，其余是
    # 线性注意力（固定大小循环状态，不随上下文增长）。所以 128K 时
    # q8_0 仅约 1.4 GiB、f16 约 2.6 GiB，放显存完全够，且比走 PCIe 快。
    kv_offload: bool = True
    threads: int = 16  # CPU 线程；本机 Ryzen 9 7945HX 有 16 核 32 线程。
    # 实测 8 / 16 / 24 线程的 prefill 是 434 / 426 / 442 tok/s，在噪声内——
    # 瓶颈不在线程数，而在 CPU 侧小批量专家 GEMM 的有效访存（约 13 GB/s）。
    # 留 16 给系统其它进程余量。
    threads_batch: int = 0  # 0 = 与 threads 相同

    # ---- 推理行为 ----
    # 模型加载模式，取代旧的 --no-mmap / --mlock：
    #   auto / mmap / mlock / mmap+mlock / none
    # 实测（128K + cpu-moe）：none 比默认 mmap 的 decode 快约 17%
    # （24.95 -> 29.34 tok/s），prefill 也略快。
    #
    # ⚠ 代价：none 会把 --cpu-moe 的专家张量放进 CUDA_Host（cudaMallocHost
    #   锁页内存），需要**一次性锁定约 14.6 GiB 物理内存**。页缓存占满或内存
    #   碎片化时会在启动后 2 秒内直接失败：
    #     ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer
    #       of size 15704850432
    #     unable to allocate CUDA_Host buffer
    #   这不是显存不足。core/server.py 会识别这种情况并自动回落到 mmap
    #   重试同一套参数（decode 慢 17%，但一定能起来）。
    #   如果启动总是不稳，直接把这里设成 ""（用 llama.cpp 默认的 auto=mmap）。
    load_mode: str = "none"
    no_mmap: bool = False  # 已废弃，保留兼容；True 等价于 load_mode="none"
    mlock: bool = False  # 已废弃，保留兼容；True 等价于 load_mode="mlock"
    no_warmup: bool = True  # 由服务自身做预热，跳过默认 warmup
    jinja: bool = True  # 使用模型的 chat template
    cont_batching: bool = True
    cache_reuse: int = 0  # >0 时启用 prompt cache reuse（适合多轮长对话）
    metrics: bool = True
    reasoning_budget: int = -1  # 思考型模型：>=0 时限制思考 token

    # ---- 网络 ----
    backend_host: str = "127.0.0.1"  # llama.cpp 只监听本机
    backend_port: int = 8080
    proxy_host: str = "0.0.0.0"  # 内网 API 服务对外监听地址
    proxy_port: int = 8000
    api_key: str = ""  # 留空表示内网免鉴权
    allow_origins: list[str] = field(default_factory=lambda: ["*"])

    # ---- 运行时行为 ----
    autostart_backend: bool = True  # 网关启动时自动拉起 llama-server
    startup_timeout: int = 300  # 等待模型加载的最长秒数
    max_queue: int = 8  # 网关允许的排队请求数上限
    request_timeout: int = 600
    log_requests: bool = True

    # ---- 派生属性 ----
    @property
    def effective_backend_port(self) -> int:
        """若配置端口被占用则自动顺延，避免启动失败。"""
        return self.backend_port

    @property
    def ctx_per_slot(self) -> int:
        slots = max(1, self.parallel_slots)
        return max(512, self.context_size // slots)

    @property
    def backend_base_url(self) -> str:
        return f"http://{self.backend_host}:{self.backend_port}"

    @property
    def model_file(self) -> Path:
        return Path(self.model_path)

    @property
    def moe_placement(self) -> str:
        """专家权重的摆位描述，用于日志与横幅。"""
        if self.n_cpu_moe >= 0:
            return f"前 {self.n_cpu_moe} 层专家在内存，其余层专家在显存"
        if self.cpu_moe:
            return "全部专家在内存（--cpu-moe）"
        return "专家权重不动（跟随 --n-gpu-layers，显存优先）"

    @property
    def kv_placement(self) -> str:
        return "显存" if self.kv_offload else "内存"

    @property
    def effective_load_mode(self) -> str:
        """实际生效的加载模式，兼容已废弃的 no_mmap / mlock。"""
        if self.load_mode:
            return self.load_mode
        if self.no_mmap:
            return "none"
        if self.mlock:
            return "mlock"
        return ""

    # 实测安全上限：超过它 llama.cpp 在 qwen35moe + --cpu-moe 下会 CUDA 越界
    SAFE_UBATCH = 2048

    @property
    def effective_ubatch(self) -> int:
        """实际会传给 llama-server 的 ubatch（已按安全上限夹紧）。"""
        ub = max(1, self.ubatch_size)
        if not self.allow_large_ubatch:
            ub = min(ub, self.SAFE_UBATCH)
        return ub

    @property
    def ubatch_clamped(self) -> bool:
        return self.effective_ubatch != self.ubatch_size

    # ---- 校验 ----
    def validate(self) -> list[str]:
        """返回问题列表（空表示配置可用）。"""
        problems: list[str] = []
        if not self.model_file.is_file():
            problems.append(f"模型文件不存在: {self.model_file}")
        if self.mmproj_path and not Path(self.mmproj_path).is_file():
            problems.append(f"mmproj 文件不存在: {self.mmproj_path}")
        if self.context_size < 512:
            problems.append("context_size 至少为 512")
        if self.parallel_slots < 1:
            problems.append("parallel_slots 至少为 1")
        if self.n_cpu_moe < -1:
            problems.append("n_cpu_moe 至少为 -1（-1 表示不用该参数）")
        if self.load_mode and self.load_mode not in (
            "auto", "mmap", "mlock", "mmap+mlock", "none"
        ):
            problems.append(f"load_mode 取值不合法: {self.load_mode}")
        if self.ubatch_size < 1:
            problems.append("ubatch_size 至少为 1")
        if not (1 <= self.proxy_port <= 65535):
            problems.append("proxy_port 不在合法范围")
        for name in ("kv_cache_type_k", "kv_cache_type_v"):
            if getattr(self, name) not in ("f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"):
                problems.append(f"{name} 取值不合法: {getattr(self, name)}")
        for rule in self.tensor_overrides:
            if "=" not in rule:
                problems.append(f"tensor_overrides 规则缺少 '=': {rule}")
        return problems

    def warnings(self) -> list[str]:
        """返回**非致命**告警（可继续启动，但结果可能与预期不同）。"""
        notes: list[str] = []
        if self.ubatch_size > self.SAFE_UBATCH and not self.allow_large_ubatch:
            notes.append(
                f"ubatch_size={self.ubatch_size} 超过实测安全上限 {self.SAFE_UBATCH}"
                f"（4096 会触发 CUDA 越界导致后端崩溃），已自动夹到 {self.SAFE_UBATCH}。"
                f"确要实验请设 allow_large_ubatch: true"
            )
        if self.batch_size < self.effective_ubatch:
            notes.append(
                f"batch_size={self.batch_size} 小于 ubatch={self.effective_ubatch}，"
                f"实际会按 ubatch 生效"
            )
        if self.cpu_moe and self.n_cpu_moe >= 0:
            notes.append(
                f"同时设置了 cpu_moe 和 n_cpu_moe={self.n_cpu_moe}，"
                f"以后者为准（前者被忽略）"
            )
        return notes

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    # ---- 构造 ----
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerConfig":
        known = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"配置中存在未知字段: {', '.join(sorted(unknown))}")
        cfg = cls()
        for key, value in data.items():
            if value is None:
                continue
            current = getattr(cfg, key)
            # YAML 里写数字字符串时做一次宽松转换
            if isinstance(current, bool) and isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(current, int) and not isinstance(current, bool) and isinstance(value, str):
                value = int(value)
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "ServerConfig":
        """从 YAML/JSON 文件加载，再叠加环境变量。"""
        cfg = cls()
        target = Path(path) if path else DEFAULT_CONFIG_FILE
        if target.is_file():
            cfg = cls.from_dict(_read_config_file(target))
        cfg.apply_env()
        return cfg

    def apply_env(self) -> None:
        """``LLM_<字段名大写>`` 覆盖对应字段。"""
        for f in fields(self):
            raw = os.environ.get(f"LLM_{f.name.upper()}")
            if raw is None:
                continue
            current = getattr(self, f.name)
            if isinstance(current, bool):
                setattr(self, f.name, raw.strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(self, f.name, int(raw))
            elif isinstance(current, list):
                setattr(self, f.name, [x.strip() for x in raw.split(",") if x.strip()])
            else:
                setattr(self, f.name, raw)


def _read_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # 没有 PyYAML 时退回 JSON
            stripped = "\n".join(
                line for line in text.splitlines() if not line.strip().startswith("#")
            )
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                raise RuntimeError(
                    "读取 YAML 配置需要 PyYAML：pip install pyyaml"
                ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def port_is_free(host: str, port: int) -> bool:
    """检查端口是否可绑定。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(host: str, start: int, tries: int = 20) -> int:
    """从 ``start`` 起找一个空闲端口。"""
    for offset in range(tries):
        port = start + offset
        if port > 65535:
            break
        if port_is_free(host, port):
            return port
    raise RuntimeError(f"从 {start} 起连续 {tries} 个端口都被占用")


def local_ip_addresses() -> list[str]:
    """列出本机内网 IPv4 地址，便于打印给同事使用。"""
    addrs: set[str] = set()
    try:
        # 连一个外部地址只为拿到默认出口网卡，不会真的发数据
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.5)
            sock.connect(("8.8.8.8", 80))
            addrs.add(sock.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    return sorted(a for a in addrs if not a.startswith("127."))
