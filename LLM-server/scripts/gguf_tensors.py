"""打印 GGUF 张量索引的精确统计：专家 vs 其余、量化分布、每层专家体积。

这张表决定了 MoE 摆位参数：

* **专家权重（``*_exps``）** 必须放内存 —— 体积大且每 token 只用到极少数
* **其余权重** 可以放显存
* **MTP/nextn 头** llama.cpp 会整段忽略，属于文件里的死重

用法::

    python scripts/gguf_tensors.py <model.gguf> [--top 20]
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_server.core.gguf import (  # noqa: E402
    GIB,
    MIB,
    GGML_TYPES,
    model_shape,
    read_tensors,
    tensor_breakdown,
)

DEFAULT_MODEL = r"D:\models\Tile\Tile-35BA3B\Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf"


def main() -> int:
    argv = sys.argv[1:]
    top_n = int(argv[argv.index("--top") + 1]) if "--top" in argv else 20
    # 位置参数：跳过 --top 的取值，取第一个看起来像路径的参数
    skip = {argv.index("--top") + 1} if "--top" in argv else set()
    positional = [a for i, a in enumerate(argv) if i not in skip and not a.startswith("--")]
    path = Path(positional[0] if positional else DEFAULT_MODEL)

    if not path.is_file():
        print(f"模型不存在: {path}")
        return 2

    shape = model_shape(path)
    breakdown = tensor_breakdown(path, shape)
    tensors = read_tensors(path)
    file_size = path.stat().st_size

    print(f"文件    : {path}")
    print(f"实际大小: {file_size / GIB:.3f} GiB")
    print(f"张量数  : {breakdown.tensor_count}")
    print(f"张量合计: {breakdown.total_gib:.3f} GiB "
          f"(与文件差 {file_size - breakdown.total_bytes} 字节 = 头部/对齐填充)")
    print()
    print(f"=== 模型结构 ===")
    print(f"  {shape.describe()}")
    print(f"  计算层号    : {breakdown.compute_layers[0]}..{breakdown.compute_layers[-1]}"
          f"（共 {len(breakdown.compute_layers)} 层）")
    print(f"  KV 所需层数 : {shape.attention_layers}"
          f"（full_attention_interval={shape.full_attention_interval or 1}）")
    print()

    print("=== 量化类型分布 ===")
    for name, nbytes, count in breakdown.by_type:
        bpw = next((v[2] * 8 / v[1] for v in GGML_TYPES.values() if v[0] == name), 0.0)
        print(f"  {name:<12} {nbytes / GIB:>7.3f} GiB  {count:>4} 个张量  {bpw:>5.2f} bpw")
    print()

    print("=== 分类（决定谁放内存、谁放显存）===")
    print(f"  专家 *_exps          {breakdown.expert_gib:>7.3f} GiB   → 内存")
    print(f"  其余可上显存          {breakdown.resident_gib:>7.3f} GiB   → 显存")
    print(f"  MTP/nextn（被忽略）   {breakdown.mtp_bytes / GIB:>7.3f} GiB   → 死重")
    print()

    per_layer = breakdown.avg_layer_expert_gib
    if per_layer > 0:
        print("=== 每层专家权重（--n-cpu-moe 的预算单位）===")
        print(f"  平均每层        {per_layer * 1024:>7.1f} MiB")
        print(f"  显存每 1 GiB 可容纳 {1 / per_layer:>5.2f} 层")
        print(f"  KV cache 参考（q8_0）：", end="")
        for ctx in (8192, 32768, 65536, 131072):
            print(f"{ctx // 1024}K={shape.kv_gib(ctx, 'q8_0'):.2f}", end="  ")
        print()
        print()

    print(f"=== 最大的 {top_n} 个张量 ===")
    for t in sorted(tensors, key=lambda x: -x.nbytes)[:top_n]:
        dims = "x".join(str(d) for d in t.dims)
        print(f"  {t.name:<44} [{dims:<20}] {t.type_name:<8} {t.nbytes / MIB:>9.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
