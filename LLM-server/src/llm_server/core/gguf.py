"""轻量 GGUF 读取器：元数据 + 张量索引（无需第三方库）。

用途有两个，都是为了让显存/内存规划基于**真实结构**而不是猜测：

1. **元数据**：层数、KV 头数、head_dim、专家数、全注意力层间隔。
   同为 40 层模型，``full_attention_interval`` 为 4（混合线性注意力）与 1
   （纯注意力）在 128K 下的 KV cache 相差 4 倍。

2. **张量索引**：逐个张量的维度与量化类型，据此算出精确字节数。
   MoE 模型里"专家权重"（``*_exps``）占绝大部分且必须放内存，
   "其余权重" 可以放显存——这个划分决定了启动参数，不能靠估。
   实测校验：Tile-35B-A3B 算出的显存占用 3.93 GiB 与服务端实测 3.91 GiB 吻合。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# GGUF 元数据值类型编号
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

_SCALAR = {
    UINT8: ("<B", 1), INT8: ("<b", 1),
    UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4),
    FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

# 各量化类型每个元素占用的字节数（含 scale/zero 开销）
KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,
    "iq4_nl": 0.5625,
}

GIB = 1024**3
MIB = 1024**2

# ggml 类型编号 -> (名称, block_size, type_size)。与 ggml.h 的 ggml_type 一致。
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    36: ("IQ4_NL_4_4", 32, 18),
    39: ("MXFP4", 32, 17),
}


# ---------------------------------------------------------------------------
# 结构（元数据）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelShape:
    """从 GGUF 元数据中提取的模型结构。"""

    arch: str = "unknown"
    name: str = ""
    block_count: int = 0
    embedding_length: int = 0
    head_count: int = 0
    head_count_kv: int = 0
    key_length: int = 0
    context_length: int = 0
    file_size_gib: float = 0.0
    # ---- MoE / 混合注意力 ----
    expert_count: int = 0
    expert_used_count: int = 0
    expert_ff_length: int = 0
    expert_shared_ff_length: int = 0
    full_attention_interval: int = 0
    nextn_predict_layers: int = 0

    @property
    def head_dim(self) -> int:
        if self.key_length:
            return self.key_length
        if self.head_count:
            return self.embedding_length // self.head_count
        return 0

    @property
    def compute_layers(self) -> int:
        """真正参与计算的层数（MTP/nextn 头会被 llama.cpp 忽略）。"""
        n = self.block_count - self.nextn_predict_layers
        return n if n > 0 else self.block_count

    @property
    def attention_layers(self) -> int:
        """需要随上下文增长的 KV cache 的层数。

        混合架构（如 ``full_attention_interval=4``）里只有每 4 层中的 1 层是
        真注意力，其余是线性注意力（固定大小的循环状态）。
        """
        layers = self.compute_layers
        if self.full_attention_interval > 1:
            return max(1, layers // self.full_attention_interval)
        return max(1, layers)

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 1 and self.expert_ff_length > 0

    @property
    def usable(self) -> bool:
        return bool(self.block_count and self.head_count_kv and self.head_dim)

    def kv_gib(self, context: int, quant: str = "q8_0") -> float:
        """计算指定上下文与量化下的 KV cache 占用（GiB）。

        KV 字节数 = 2(K和V) × **全注意力层数** × KV头数 × head_dim × token 数
                    × 每元素字节
        """
        if not self.usable:
            return 0.0
        per_elem = KV_BYTES_PER_ELEMENT.get(quant, 2.0)
        total = (2 * self.attention_layers * self.head_count_kv
                 * self.head_dim * context * per_elem)
        return total / GIB

    def describe(self) -> str:
        bits = [
            f"架构 {self.arch}",
            f"{self.block_count} 层",
            f"{self.head_count} 头 / {self.head_count_kv} KV 头 "
            f"(GQA {self.head_count // max(self.head_count_kv, 1)}:1)",
            f"head_dim {self.head_dim}",
            f"训练上下文 {self.context_length}",
            f"权重 {self.file_size_gib:.2f} GiB",
        ]
        if self.nextn_predict_layers:
            bits.append(f"计算层 {self.compute_layers}（忽略 {self.nextn_predict_layers} 层 MTP）")
        if self.full_attention_interval > 1:
            bits.append(f"全注意力层 {self.attention_layers}"
                        f"（每 {self.full_attention_interval} 层一个，其余为线性注意力）")
        if self.is_moe:
            bits.append(f"MoE {self.expert_count} 专家 / 激活 {self.expert_used_count}"
                        f" / 专家隐藏维 {self.expert_ff_length}")
        return " | ".join(bits)


# ---------------------------------------------------------------------------
# 张量索引
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int = 0

    @property
    def n_elements(self) -> int:
        total = 1
        for d in self.dims:
            total *= d
        return total

    @property
    def type_name(self) -> str:
        return GGML_TYPES.get(self.type_id, (f"type{self.type_id}", 1, 2))[0]

    @property
    def nbytes(self) -> int:
        _, block, size = GGML_TYPES.get(self.type_id, ("?", 1, 2))
        n = self.n_elements
        return (n + block - 1) // block * size

    @property
    def bpw(self) -> float:
        _, block, size = GGML_TYPES.get(self.type_id, ("?", 1, 2))
        return size * 8 / block

    @property
    def layer(self) -> int | None:
        """``blk.<N>.*`` 形式的层号；不是分层张量时返回 None。"""
        parts = self.name.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
            return int(parts[1])
        return None


@dataclass(frozen=True)
class TensorBreakdown:
    """按"放内存 / 放显存"划分的张量字节统计。"""

    total_bytes: int = 0
    expert_bytes: int = 0  # 专家权重（*_exps），--cpu-moe 的作用对象
    resident_bytes: int = 0  # 其余权重（attention/SSM/共享专家/embedding），可上显存
    mtp_bytes: int = 0  # MTP/nextn 头，llama.cpp 当前忽略
    compute_layers: tuple[int, ...] = ()
    per_layer_expert_bytes: tuple[tuple[int, int], ...] = ()  # (层号, 字节)
    by_type: tuple[tuple[str, int, int], ...] = ()  # (类型名, 字节, 张量数)
    tensor_count: int = 0

    @property
    def resident_gib(self) -> float:
        return self.resident_bytes / GIB

    @property
    def expert_gib(self) -> float:
        return self.expert_bytes / GIB

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def avg_layer_expert_gib(self) -> float:
        """计算层里平均每层的专家权重（用于 --n-cpu-moe 预算）。"""
        values = [b for _, b in self.per_layer_expert_bytes]
        return (sum(values) / len(values) / GIB) if values else 0.0

    def experts_gib(self, n_layers: int) -> float:
        """前 ``n_layers`` 层的专家权重大小（GiB）。"""
        values = [b for _, b in self.per_layer_expert_bytes[:n_layers]]
        return sum(values) / GIB

    def describe(self) -> str:
        return (
            f"张量 {self.tensor_count} 个 / {self.total_gib:.3f} GiB："
            f"专家 {self.expert_gib:.3f} GiB（→内存）、"
            f"其余 {self.resident_gib:.3f} GiB（→显存）、"
            f"MTP {self.mtp_bytes / GIB:.3f} GiB（忽略）"
        )


def _read_string(fh) -> str:  # noqa: ANN001
    (length,) = struct.unpack("<Q", fh.read(8))
    return fh.read(length).decode("utf-8", errors="replace")


def _read_value(fh, vtype: int):  # noqa: ANN001
    """读取一个元数据值（顺带把文件指针推进到正确位置）。"""
    if vtype == STRING:
        return _read_string(fh)
    if vtype == ARRAY:
        (etype,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        # 大数组（词表等）只跳过，不构造列表
        if etype in _SCALAR and count > 64:
            _, size = _SCALAR[etype]
            fh.seek(size * count, 1)
            return None
        if etype == STRING and count > 64:
            for _ in range(count):
                _read_string(fh)
            return None
        if count > 100000:
            raise ValueError(f"数组过大: type={etype} count={count}")
        return [_read_value(fh, etype) for _ in range(count)]
    if vtype in _SCALAR:
        fmt, size = _SCALAR[vtype]
        return struct.unpack(fmt, fh.read(size))[0]
    raise ValueError(f"未知 GGUF 值类型: {vtype}")


@lru_cache(maxsize=4)
def _read_metadata_cached(path_str: str, mtime: float) -> dict:
    """读元数据（按路径+mtime 缓存；mtime 变化会自动失效）。"""
    path = Path(path_str)
    meta: dict = {}
    with path.open("rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("不是 GGUF 文件")
        (version,) = struct.unpack("<I", fh.read(4))
        meta["__version__"] = version
        struct.unpack("<Q", fh.read(8))  # tensor_count，不需要
        (kv_count,) = struct.unpack("<Q", fh.read(8))

        for i in range(kv_count):
            try:
                key = _read_string(fh)
                (vtype,) = struct.unpack("<I", fh.read(4))
                meta[key] = _read_value(fh, vtype)
            except (struct.error, ValueError, OSError):
                # 遇到不认识的类型就停下，已读到的字段仍然可用
                meta["__parse_stopped_at__"] = i
                break
    return meta


def read_metadata(path: str | Path) -> dict:
    p = Path(path)
    return _read_metadata_cached(str(p), p.stat().st_mtime if p.is_file() else 0.0)


@lru_cache(maxsize=4)
def _read_tensors_cached(path_str: str, mtime: float) -> tuple[TensorInfo, ...]:
    path = Path(path_str)
    tensors: list[TensorInfo] = []
    with path.open("rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("不是 GGUF 文件")
        struct.unpack("<I", fh.read(4))
        (tensor_count,) = struct.unpack("<Q", fh.read(8))
        (kv_count,) = struct.unpack("<Q", fh.read(8))
        # 跳过元数据区（_read_value 会正确推进指针）
        for _ in range(kv_count):
            _read_string(fh)
            (vtype,) = struct.unpack("<I", fh.read(4))
            _read_value(fh, vtype)
        # 张量信息区
        for _ in range(tensor_count):
            name = _read_string(fh)
            (n_dims,) = struct.unpack("<I", fh.read(4))
            dims = struct.unpack(f"<{n_dims}Q", fh.read(8 * n_dims))
            (type_id,) = struct.unpack("<I", fh.read(4))
            (offset,) = struct.unpack("<Q", fh.read(8))
            tensors.append(
                TensorInfo(name, tuple(int(d) for d in dims), int(type_id), int(offset))
            )
    return tuple(tensors)


def read_tensors(path: str | Path) -> tuple[TensorInfo, ...]:
    """读取全部张量信息（只读文件头，不读张量数据）。"""
    p = Path(path)
    return _read_tensors_cached(str(p), p.stat().st_mtime if p.is_file() else 0.0)


def is_expert_tensor(name: str) -> bool:
    """是否为"按专家分开存放"的权重，即 ``--cpu-moe`` 的作用对象。"""
    return "_exps" in name


def is_mtp_tensor(name: str) -> bool:
    """MTP / nextn 投机头：llama.cpp 当前会忽略这些张量。"""
    return ".nextn." in name or ".mtp." in name


def tensor_breakdown(path: str | Path, shape: ModelShape | None = None) -> TensorBreakdown:
    """把张量索引按"内存/显存"归类并统计字节数。"""
    p = Path(path)
    try:
        tensors = read_tensors(p)
    except (OSError, ValueError, struct.error):
        return TensorBreakdown()

    if shape is None:
        shape = model_shape(p)
    skip_layer = shape.compute_layers if shape.nextn_predict_layers else 0

    total = expert = resident = mtp = 0
    per_layer: dict[int, int] = {}
    by_type: dict[str, list[int]] = {}
    compute_layers: set[int] = set()

    for t in tensors:
        nbytes = t.nbytes
        total += nbytes
        entry = by_type.setdefault(t.type_name, [0, 0])
        entry[0] += nbytes
        entry[1] += 1

        layer = t.layer
        # compute_layers 之后的层（如 blk.40）只承载 MTP 头，llama.cpp 会忽略
        if layer is not None and skip_layer and layer >= skip_layer:
            mtp += nbytes
            continue
        if is_mtp_tensor(t.name):
            mtp += nbytes
            continue
        if layer is not None:
            compute_layers.add(layer)
        if is_expert_tensor(t.name):
            expert += nbytes
            if layer is not None:
                per_layer[layer] = per_layer.get(layer, 0) + nbytes
        else:
            resident += nbytes

    ordered_layers = tuple(sorted(compute_layers))
    return TensorBreakdown(
        total_bytes=total,
        expert_bytes=expert,
        resident_bytes=resident,
        mtp_bytes=mtp,
        compute_layers=ordered_layers,
        per_layer_expert_bytes=tuple(
            (k, per_layer[k]) for k in ordered_layers if k in per_layer
        ),
        by_type=tuple(sorted(
            ((name, v[0], v[1]) for name, v in by_type.items()),
            key=lambda item: -item[1],
        )),
        tensor_count=len(tensors),
    )


def model_shape(path: str | Path) -> ModelShape:
    """读取模型规模信息；解析失败时返回带 file_size 的空结构。"""
    p = Path(path)
    size_gib = p.stat().st_size / GIB if p.is_file() else 0.0
    try:
        meta = read_metadata(p)
    except (OSError, ValueError):
        return ModelShape(file_size_gib=size_gib)

    arch = str(meta.get("general.architecture", "unknown"))

    def get(suffix: str, default: int = 0) -> int:
        value = meta.get(f"{arch}.{suffix}", meta.get(f"general.{suffix}", default))
        return value if isinstance(value, int) else default

    return ModelShape(
        arch=arch,
        name=str(meta.get("general.name", "")),
        block_count=get("block_count"),
        embedding_length=get("embedding_length"),
        head_count=get("attention.head_count"),
        head_count_kv=get("attention.head_count_kv") or get("attention.head_count"),
        key_length=get("attention.key_length"),
        context_length=get("context_length"),
        file_size_gib=size_gib,
        expert_count=get("expert_count"),
        expert_used_count=get("expert_used_count"),
        expert_ff_length=get("expert_feed_forward_length"),
        expert_shared_ff_length=get("expert_shared_feed_forward_length"),
        full_attention_interval=get("full_attention_interval"),
        nextn_predict_layers=get("nextn_predict_layers"),
    )


# ---------------------------------------------------------------------------
# 占用估算
# ---------------------------------------------------------------------------

# ubatch -> 计算缓冲实测值（GiB）。在 RTX 4060 Laptop 8GB + Tile-35B-A3B
# (IQ4_XS, ctx=131072, cpu-moe, kv-offload) 上标定：用总显存净增减去
# 常住权重 2.38 GiB 与 KV 1.33 GiB 得到。
_BUFFER_CURVE: tuple[tuple[int, float], ...] = (
    (256, 0.09),
    (512, 0.20),
    (1024, 0.37),
    (2048, 0.70),
    (4096, 1.68),
)


def compute_buffer_gib(ubatch_size: int) -> float:
    """估算计算缓冲占用（GiB），按实测曲线在 ubatch 之间线性插值。

    MoE 下这个值随 ubatch 增长得比稠密模型快：ubatch 4096 时约 1.7 GiB，
    已经和 KV cache 一个量级。调大 ubatch 提速 prefill 时要把它算进预算。
    """
    points = _BUFFER_CURVE
    if ubatch_size <= points[0][0]:
        return points[0][1]
    if ubatch_size >= points[-1][0]:
        # 超出实测范围时按最后两点的斜率外推
        (x0, y0), (x1, y1) = points[-2], points[-1]
        slope = (y1 - y0) / (x1 - x0)
        return round(y1 + slope * (ubatch_size - x1), 2)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= ubatch_size <= x1:
            ratio = (ubatch_size - x0) / (x1 - x0)
            return round(y0 + ratio * (y1 - y0), 2)
    return 0.55


def estimate_vram(
    shape: ModelShape,
    context: int,
    kv_quant: str = "q8_0",
    parallel: int = 1,
    compute_buffer_gib: float = 0.55,
) -> dict[str, float]:
    """估算显存占用分解（稠密模型口径：全部权重都算在显存里）。

    MoE 模型请用 :func:`estimate_placement`，它会把专家权重从显存里剔除。
    """
    weights = shape.file_size_gib
    kv = shape.kv_gib(context, kv_quant)
    total = weights + kv + compute_buffer_gib
    return {
        "weights_gib": round(weights, 2),
        "kv_cache_gib": round(kv, 2),
        "compute_buffer_gib": round(compute_buffer_gib, 2),
        "total_gib": round(total, 2),
        "context": float(context),
        "parallel": float(parallel),
    }


def estimate_placement(
    shape: ModelShape,
    breakdown: TensorBreakdown,
    *,
    context: int,
    kv_quant: str = "q8_0",
    cpu_moe: bool = True,
    n_cpu_moe: int = -1,
    parallel: int = 1,
    ubatch_size: int = 512,
) -> dict[str, float]:
    """按 MoE 摆位计算显存 / 内存分解占用（GiB）。

    ``n_cpu_moe >= 0`` 时只有前 N 层的专家留在内存，其余层专家进显存；
    否则 ``cpu_moe`` 决定全部专家在内存还是显存。
    """
    buffer = compute_buffer_gib(ubatch_size)
    kv = shape.kv_gib(context, kv_quant) * max(1, parallel)

    total_experts = breakdown.expert_gib
    if n_cpu_moe >= 0:
        experts_ram = breakdown.experts_gib(n_cpu_moe)
    elif cpu_moe:
        experts_ram = total_experts
    else:
        experts_ram = 0.0
    experts_vram = max(0.0, total_experts - experts_ram)

    resident = breakdown.resident_gib
    vram = resident + experts_vram + kv + buffer
    return {
        "resident_gib": round(resident, 2),
        "experts_vram_gib": round(experts_vram, 2),
        "experts_ram_gib": round(experts_ram, 2),
        "kv_cache_gib": round(kv, 2),
        "compute_buffer_gib": round(buffer, 2),
        "vram_total_gib": round(vram, 2),
        "ram_experts_gib": round(experts_ram, 2),
        "kv_attention_layers": float(shape.attention_layers),
        "context": float(context),
    }


def max_experts_on_gpu(
    shape: ModelShape,
    breakdown: TensorBreakdown,
    *,
    context: int,
    kv_quant: str = "q8_0",
    free_vram_gib: float,
    ubatch_size: int = 512,
    margin_gib: float = 0.35,
) -> int:
    """在给定空闲显存下，最多能把多少层的专家放到显存（从最后一层往前数）。

    返回可放到显存的层数；0 表示放不下（应使用 ``--cpu-moe``）。
    """
    per_layer = breakdown.avg_layer_expert_gib
    if per_layer <= 0:
        return 0
    budget = (free_vram_gib - margin_gib
              - breakdown.resident_gib
              - shape.kv_gib(context, kv_quant)
              - compute_buffer_gib(ubatch_size))
    if budget <= 0:
        return 0
    return min(shape.compute_layers, int(budget / per_layer))
