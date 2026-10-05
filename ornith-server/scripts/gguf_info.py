"""读取 GGUF 头部元数据，用于精确计算 KV cache 显存占用。

GGUF 格式（v2/v3）：
  magic "GGUF" | version u32 | tensor_count u64 | metadata_kv_count u64
  随后是 metadata_kv_count 个 (key, value_type, value) 三元组，字符串为
  u64 长度 + UTF-8 字节。

本脚本不需要第三方库，直接按格式解析，用于确定：
  block_count / attention.head_count / attention.head_count_kv /
  attention.key_length / embedding_length / context_length
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

# GGUF 元数据值类型
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

_SCALAR = {
    UINT8: ("<B", 1), INT8: ("<b", 1),
    UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4),
    FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}

ARRAY_LEN = ("<Q", 8)

WANTED = (
    "general.architecture",
    "general.name",
    "general.size_label",
    "context_length",
    "embedding_length",
    "block_count",
    "head_count",
    "head_count_kv",
    "key_length",
    "value_length",
    "rope.freq_base",
    "attention.layer_norm_rms_epsilon",
    "file_type",
)


def _read_string(fh) -> str:
    raw = fh.read(8)
    (length,) = struct.unpack("<Q", raw)
    return fh.read(length).decode("utf-8", errors="replace")


def _read_value(fh, vtype: int, depth: int = 0):
    if vtype == STRING:
        return _read_string(fh)
    if vtype == ARRAY:
        (etype,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        # 大数组（如 tokenizer 词表）跳过，避免读取上百万项
        if etype in _SCALAR and count > 64:
            fmt, size = _SCALAR[etype]
            fh.seek(size * count, 1)
            return f"<array of {count} scalars>"
        if etype == STRING and count > 64:
            for _ in range(count):
                _read_string(fh)
            return f"<array of {count} strings>"
        if count > 100000:
            raise ValueError(f"数组过大无法跳过: type={etype} count={count}")
        return [_read_value(fh, etype, depth + 1) for _ in range(count)]
    if vtype in _SCALAR:
        fmt, size = _SCALAR[vtype]
        return struct.unpack(fmt, fh.read(size))[0]
    raise ValueError(f"未知的 GGUF 值类型: {vtype}")


def read_metadata(path: Path, limit: int = 200000) -> dict:
    meta: dict = {}
    with path.open("rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"不是 GGUF 文件（magic={magic!r}）")
        (version,) = struct.unpack("<I", fh.read(4))
        (tensor_count,) = struct.unpack("<Q", fh.read(8))
        (kv_count,) = struct.unpack("<Q", fh.read(8))
        meta["__version__"] = version
        meta["__tensor_count__"] = tensor_count

        for i in range(min(kv_count, limit)):
            try:
                key = _read_string(fh)
                (vtype,) = struct.unpack("<I", fh.read(4))
                meta[key] = _read_value(fh, vtype)
            except (struct.error, ValueError, OSError) as exc:
                meta["__parse_stopped__"] = f"第 {i} 项后停止: {exc}"
                break
    return meta


def pick(meta: dict, arch: str, suffix: str):
    """按 `<arch>.<suffix>` 取键；不同模型架构前缀不同。"""
    return meta.get(f"{arch}.{suffix}")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                r"D:\models\Tile\Tile-35BA3B\Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf")
    meta = read_metadata(path)
    arch = str(meta.get("general.architecture", "llama"))

    print(f"文件: {path}")
    print(f"大小: {path.stat().st_size / 1024**3:.2f} GiB")
    print(f"GGUF 版本: {meta.get('__version__')}  张量数: {meta.get('__tensor_count__')}")
    print(f"架构: {arch}   名称: {meta.get('general.name')}")
    print()
    print("=== 模型结构 ===")
    block = pick(meta, arch, "block_count")
    embed = pick(meta, arch, "embedding_length")
    heads = pick(meta, arch, "attention.head_count")
    heads_kv = pick(meta, arch, "attention.head_count_kv") or heads
    key_len = pick(meta, arch, "attention.key_length")
    ctx_train = pick(meta, arch, "context_length")
    print(f"  block_count (层数)          : {block}")
    print(f"  embedding_length            : {embed}")
    print(f"  attention.head_count        : {heads}")
    print(f"  attention.head_count_kv     : {heads_kv}   (GQA)")
    print(f"  attention.key_length        : {key_len}")
    print(f"  context_length (训练上下文) : {ctx_train}")

    if not all(isinstance(x, int) for x in (block, heads, heads_kv)):
        print("\n无法确定结构，跳过 KV 估算")
        return 1

    # head_dim：优先用显式 key_length，否则 embed / heads
    head_dim = key_len if isinstance(key_len, int) and key_len else (
        embed // heads if isinstance(embed, int) else None
    )
    print(f"  推导 head_dim               : {head_dim}")
    print()

    # KV cache 每 token 字节数 = 2 (K+V) * 层数 * kv头数 * head_dim * 每元素字节
    per_elem = {
        "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625,
        "q5_1": 0.75, "q5_0": 0.6875, "q4_1": 0.625,
        "q4_0": 0.5625, "iq4_nl": 0.5625,
    }
    print("=== KV cache 显存占用（GiB）===")
    header = f"{'上下文':>10} | " + " | ".join(f"{k:>8}" for k in per_elem)
    print(header)
    print("-" * len(header))
    for ctx in (8192, 16384, 32768, 65536, 98304, 131072, 200000):
        row = []
        for quant, size in per_elem.items():
            total = 2 * block * heads_kv * head_dim * ctx * size
            row.append(f"{total / 1024**3:>8.2f}")
        print(f"{ctx:>10} | " + " | ".join(row))

    print()
    weights = path.stat().st_size / 1024**3
    print(f"权重文件: {weights:.2f} GiB（Q4_K_M，含全部层）")
    print("注：非 KV 张量在 GPU 上还需额外计算缓冲，约 0.5-1.0 GiB（随 ubatch 变化）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
