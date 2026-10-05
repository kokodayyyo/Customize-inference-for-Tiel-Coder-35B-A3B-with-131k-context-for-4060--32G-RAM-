"""转储 GGUF 全部元数据键值，用于确认 MoE 结构（专家数、共享专家等）。

用法:
    python scripts/gguf_raw.py <model.gguf> [过滤子串]

不带过滤子串时只打印结构/注意力/专家相关键，避免刷屏（词表数组很大）。
加 `--all` 打印全部键。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_info import read_metadata  # noqa: E402

# 值得关注的键片段
INTERESTING = (
    "architecture", "name", "block_count", "embedding_length", "feed_forward",
    "expert", "head_count", "key_length", "value_length", "context_length",
    "rope", "attention", "norm", "file_type", "quantization", "ssm", "conv",
    "nextn", "mtp", "dense", "shared", "router", "moe", "layer_types",
    "vocab_size", "tensor_count",
)


def main() -> int:
    argv = [a for a in sys.argv[1:] if a != "--all"]
    show_all = "--all" in sys.argv
    path = Path(argv[0])
    needle = argv[1].lower() if len(argv) > 1 else None

    meta = read_metadata(path)
    print(f"文件: {path}")
    print(f"大小: {path.stat().st_size / 1024**3:.3f} GiB")
    print("=" * 78)

    for key in sorted(meta):
        if key.startswith("__"):
            continue
        low = key.lower()
        if needle is not None:
            if needle not in low:
                continue
        elif not show_all and not any(t in low for t in INTERESTING):
            continue
        value = meta[key]
        text = repr(value)
        if len(text) > 160:
            text = text[:157] + "..."
        print(f"{key:58} = {text}")

    for key in ("__version__", "__tensor_count__", "__parse_stopped__"):
        if key in meta:
            print(f"{key:58} = {meta[key]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
