"""MoE 摆位扫描：批量运行 moe_probe，汇总成对比表。

用法::

    python scripts/moe_sweep.py --ctx 131072            # 跑全部预设
    python scripts/moe_sweep.py --only t16,t24          # 只跑指定用例
    python scripts/moe_sweep.py --list                  # 列出预设

每个用例都会完整地启动 → 测量 → 关闭后端，结果写入
``runtime/logs/sweep/<时间戳>/<用例名>.json``，并在最后打印对比表。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import moe_probe  # noqa: E402

# 预设用例：名称 -> moe_probe 的额外参数
CASES: dict[str, list[str]] = {
    # --- 线程数：prefill 是 CPU 算力瓶颈，本机 16 核 32 线程 ---
    "t8": ["--threads", "8", "--cpu-moe"],
    "t16": ["--threads", "16", "--cpu-moe"],
    "t24": ["--threads", "24", "--cpu-moe"],
    "t32": ["--threads", "32", "--cpu-moe"],
    # --- 加载模式：专家放内存时 mmap 有额外开销 ---
    "t16-nommap": ["--threads", "16", "--cpu-moe", "--load-mode", "none"],
    "t16-mlock": ["--threads", "16", "--cpu-moe", "--load-mode", "mmap+mlock"],
    # --- 专家上显存：把 VRAM 余量换成速度 ---
    "t16-ncmoe36": ["--threads", "16", "--n-cpu-moe", "36", "--cpu-moe"],
    "t16-ncmoe34": ["--threads", "16", "--n-cpu-moe", "34"],
    "t16-ncmoe32": ["--threads", "16", "--n-cpu-moe", "32"],
    "t16-ncmoe30": ["--threads", "16", "--n-cpu-moe", "30"],
    # --- KV 精度 / 位置 ---
    "t16-kvf16": ["--threads", "16", "--cpu-moe", "--kv-type", "f16"],
    "t16-kvram": ["--threads", "16", "--cpu-moe", "--no-kv-offload"],
    # --- ubatch：MoE 下每个 ubatch 都要把 256 个专家全过一遍，
    #     所以 ubatch 越大、同样的 prompt 需要的往返越少。第一轮扫描后
    #     确认 --load-mode none 更快，这里统一带上。 ---
    "u128": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
             "--ubatch", "128", "--batch", "2048"],
    "u256": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
             "--ubatch", "256", "--batch", "2048"],
    "u512": ["--threads", "16", "--cpu-moe", "--load-mode", "none"],
    "u1024": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
              "--ubatch", "1024", "--batch", "4096"],
    "u2048": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
              "--ubatch", "2048", "--batch", "8192"],
    "u4096": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
              "--ubatch", "4096", "--batch", "8192"],
    # --- 复核：带 load-mode none 再试一次专家上显存 ---
    "ncmoe34-nm": ["--threads", "16", "--n-cpu-moe", "34", "--load-mode", "none"],
    "ncmoe24-nm": ["--threads", "16", "--n-cpu-moe", "24", "--load-mode", "none"],
    # --- 天花板与边界（`++` 之后的内容会经 --extra 原样透传给 llama-server）---
    "u8192": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
              "--ubatch", "8192", "--batch", "16384"],
    "u2048-f16": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
                  "--ubatch", "2048", "--batch", "8192", "--kv-type", "f16"],
    "u2048-swa": ["--threads", "16", "--cpu-moe", "--load-mode", "none",
                  "--ubatch", "2048", "--batch", "8192", "++", "--swa-full"],
    "u2048-nm36": ["--threads", "16", "--n-cpu-moe", "36", "--load-mode", "none",
                   "--ubatch", "2048", "--batch", "8192"],
    "u2048-nm32": ["--threads", "16", "--n-cpu-moe", "32", "--load-mode", "none",
                   "--ubatch", "2048", "--batch", "8192"],
}

DEFAULT_ORDER = ["t8", "t16", "t24", "t16-nommap", "t16-ncmoe34", "t16-ncmoe32"]
# 第二轮：ubatch 扫描（第一轮确认 load-mode none 更快）
UBATCH_ORDER = ["u256", "u512", "u1024", "u2048", "u4096"]
# 第三轮：天花板与最终候选
FINAL_ORDER = ["u2048", "u4096", "u8192", "u2048-f16", "u2048-swa", "u2048-nm36"]


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE 摆位扫描")
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--only", default="", help="逗号分隔的用例名；默认跑 DEFAULT_ORDER")
    parser.add_argument("--all", action="store_true", help="跑全部预设")
    parser.add_argument("--preset", default="", choices=["", "default", "ubatch", "final"],
                        help="跑命名预设组")
    parser.add_argument("--list", action="store_true", help="列出预设后退出")
    parser.add_argument("--prefill-tokens", type=int, default=8192)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--model", default=str(moe_probe.DEFAULT_MODEL))
    parser.add_argument("--port", type=int, default=8199)
    args = parser.parse_args()

    if args.list:
        for name, extra in CASES.items():
            print(f"  {name:<14} {' '.join(extra)}")
        return 0

    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    elif args.preset == "ubatch":
        names = UBATCH_ORDER
    elif args.preset == "final":
        names = FINAL_ORDER
    elif args.all:
        names = list(CASES)
    else:
        names = DEFAULT_ORDER
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"未知用例: {', '.join(unknown)}", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = PROJECT_ROOT / "runtime" / "logs" / "sweep" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, name in enumerate(names, start=1):
        json_path = out_dir / f"{name}.json"
        log_path = out_dir / f"{name}.out.txt"
        # `++` 之后的内容不再由 moe_probe 解析，而是经 --extra 原样透传给 llama-server
        case_args = list(CASES[name])
        passthrough: list[str] = []
        if "++" in case_args:
            cut = case_args.index("++")
            passthrough = case_args[cut + 1:]
            case_args = case_args[:cut]
        argv = [
            "--model", args.model,
            "--ctx", str(args.ctx),
            "--port", str(args.port),
            *case_args,
            "--prefill-tokens", str(args.prefill_tokens),
            "--decode-tokens", str(args.decode_tokens),
            "--json-out", str(json_path),
        ]
        if passthrough:
            argv += ["--extra", *passthrough]
        print(f"\n{'#' * 78}\n# [{i}/{len(names)}] {name}: "
              f"{' '.join(case_args)} {' '.join(passthrough)}\n{'#' * 78}")

        old_argv, old_stdout = sys.argv, sys.stdout
        sys.argv = ["moe_probe.py", *argv]
        try:
            with log_path.open("w", encoding="utf-8", errors="replace") as fh:
                sys.stdout = fh
                try:
                    code = moe_probe.main()
                finally:
                    sys.stdout = old_stdout
        except Exception as exc:  # noqa: BLE001
            sys.stdout = old_stdout
            print(f"  !! 用例 {name} 异常: {exc}")
            code = -1
        finally:
            sys.argv = old_argv

        record = {"case": name, "exit": code, "extra": CASES[name]}
        if json_path.is_file():
            record.update(json.loads(json_path.read_text(encoding="utf-8")))
        results.append(record)

        print(f"  -> exit={code} "
              f"decode={record.get('decode_tps')} tok/s "
              f"prefill={record.get('prefill_tps')} tok/s "
              f"vram={record.get('vram_used_loaded_mib')} MiB "
              f"ram_avail={record.get('ram_avail_loaded_gib')} GiB")

    # ---- 汇总表 ----
    print(f"\n{'=' * 110}")
    print(f"扫描汇总  ctx={args.ctx}  prefill目标={args.prefill_tokens} token")
    print("=" * 110)
    header = (f"{'用例':<14}{'线程':>5}{'专家':>10}{'decode':>9}{'prefill':>10}"
              f"{'显存MiB':>9}{'净增MiB':>9}{'工作集GiB':>10}{'内存可用':>9}{'加载s':>7}")
    print(header)
    print("-" * 110)
    for r in results:
        extra = r["extra"]
        threads = "-"
        moe = "全部内存" if "--cpu-moe" in extra else "-"
        for j, token in enumerate(extra):
            if token == "--threads":
                threads = extra[j + 1]
            if token == "--n-cpu-moe":
                moe = f"前{extra[j + 1]}层内存"
        def fmt(key, width, digits=2):
            value = r.get(key)
            return f"{value:>{width}.{digits}f}" if isinstance(value, (int, float)) else f"{'-':>{width}}"
        print(f"{r['case']:<14}{threads:>5}{moe:>10}"
              f"{fmt('decode_tps', 9)}{fmt('prefill_tps', 10, 1)}"
              f"{fmt('vram_used_loaded_mib', 9, 0)}{fmt('vram_delta_mib', 9, 0)}"
              f"{fmt('working_set_loaded_gib', 10)}{fmt('ram_avail_loaded_gib', 9)}"
              f"{fmt('load_seconds', 7, 1)}")

    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n原始结果: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
