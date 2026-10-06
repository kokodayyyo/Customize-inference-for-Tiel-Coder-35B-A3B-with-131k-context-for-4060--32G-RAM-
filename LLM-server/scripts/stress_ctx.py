"""上下文压力验证：用中文长提示逐级填充，验证 128K 目标真的可用。

与 `moe_probe.py` 的区别：
  * 复用同一套启动逻辑，但**一个进程内连续测多个填充级别**（更接近真实使用）；
  * 用**中文**长文本（llama.cpp 在大 ubatch 下的越界只在特定提示下暴露，
    英文合成文本测不出来）；
  * 每一步都检查后端进程是否还活着，崩了就立刻停下并打印日志尾部。

用法::

    python scripts/stress_ctx.py --fills 0,8000,32000,65536,100000
    python scripts/stress_ctx.py --fills 0,4000 --ubatch 4096   # 复现崩溃
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import moe_probe as mp  # noqa: E402

# 中文约 1 token ≈ 1.5 个汉字（Qwen 系分词器）；实测后会打印真实 token 数
CHARS_PER_TOKEN = 1.5

FILLER = (
    "长上下文的主要开销来自注意力计算需要读取完整的键值缓存，"
    "因此缓存越大每一步解码的访存压力越高。"
)


def build_chinese_prompt(target_tokens: int) -> str:
    """构造约 target_tokens 个 token 的中文提示。"""
    if target_tokens <= 0:
        return "请用一句话说明键值缓存（KV cache）的作用。"
    need_chars = int(target_tokens * CHARS_PER_TOKEN)
    repeats = max(1, need_chars // len(FILLER) + 1)
    body = FILLER * repeats
    return (
        "下面是一段用于验证长上下文能力的说明，请阅读后回答。\n\n"
        + body
        + "\n\n请用一句话回答：128K 上下文在实际使用中最大的代价是什么？"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="上下文压力验证")
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--fills", default="0,8000,32000,65536,100000",
                        help="要填充到的 token 数，逗号分隔")
    parser.add_argument("--ubatch", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--kv-type", default="q8_0")
    parser.add_argument("--no-kv-offload", action="store_true")
    parser.add_argument("--load-mode", default="none")
    parser.add_argument("--n-cpu-moe", type=int, default=None)
    parser.add_argument("--cpu-moe", action="store_true", default=True)
    parser.add_argument("--no-cpu-moe", dest="cpu_moe", action="store_false")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--load-timeout", type=int, default=900)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    fills = [int(x) for x in args.fills.split(",") if x.strip()]
    engine, vendor = mp.find_backend()

    cmd = [
        str(engine / "llama-server.exe"),
        "--model", str(mp.DEFAULT_MODEL),
        "--host", "127.0.0.1", "--port", str(args.port),
        "--n-gpu-layers", "99",
        "--ctx-size", str(args.ctx),
        "--parallel", "1",
        "--batch-size", str(max(args.batch, args.ubatch)),
        "--ubatch-size", str(args.ubatch),
        "--threads", str(args.threads),
        "--flash-attn", "on",
        "--cache-type-k", args.kv_type, "--cache-type-v", args.kv_type,
        "--no-warmup", "--jinja",
        "--no-kv-offload" if args.no_kv_offload else "--kv-offload",
    ]
    if args.n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(args.n_cpu_moe)]
    elif args.cpu_moe:
        cmd.append("--cpu-moe")
    if args.load_mode:
        cmd += ["--load-mode", args.load_mode]

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(p) for p in (vendor, engine) if p] + [env.get("PATH", "")]
    )
    env.setdefault("GGML_CUDA_NO_PEER_COPY", "1")

    print("=" * 92)
    print(f"上下文压力验证: ctx={args.ctx} ubatch={args.ubatch} "
          f"kv={args.kv_type}({'内存' if args.no_kv_offload else '显存'}) "
          f"load-mode={args.load_mode or '默认'} "
          f"{'n-cpu-moe=' + str(args.n_cpu_moe) if args.n_cpu_moe is not None else ('cpu-moe' if args.cpu_moe else '')}")
    print(f"填充级别: {fills}")
    print("=" * 92)

    log_path = PROJECT_ROOT / "runtime" / "logs" / f"stress-{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    log_handle.write("CMDLINE: " + subprocess.list2cmdline(cmd) + "\n\n")
    log_handle.flush()

    creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT,
        cwd=str(engine), env=env, creationflags=creationflags,
    )

    rows: list[dict] = []
    crashed = False
    try:
        base = f"http://127.0.0.1:{args.port}"
        started = time.time()
        if not mp.wait_health(base, proc, time.time() + args.load_timeout):
            print("加载失败，日志尾部：")
            print("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]))
            return 1
        print(f"加载完成，用时 {time.time() - started:.1f}s")
        gpu = mp.gpu_used_mib()
        mem = mp.system_memory()
        print(f"  显存已用 {gpu['used_mib']} MiB (空闲 {gpu['free_mib']}) | "
              f"内存可用 {mem['avail_gib']} GiB")
        print()

        header = (f"{'填充目标':>9}{'实际prompt':>11}{'prefill':>10}{'prefill耗时':>12}"
                  f"{'首字延迟':>10}{'decode':>9}{'显存MiB':>10}")
        print(header)
        print("-" * len(header))

        for fill in fills:
            if proc.poll() is not None:
                crashed = True
                print(f"{fill:>9}  后端已退出（rc={proc.poll()}），停止。")
                break
            prompt = build_chinese_prompt(fill)
            try:
                resp = mp.run_completion(base, prompt, args.decode_tokens)
            except Exception as exc:  # noqa: BLE001
                crashed = proc.poll() is not None
                print(f"{fill:>9}  请求失败: {type(exc).__name__}: {exc}")
                if crashed:
                    print(f"          -> 后端进程已退出 (rc={proc.poll()})")
                break

            t = resp.get("timings", {})
            pre = t.get("prompt_per_second") or 0
            dec = t.get("predicted_per_second") or 0
            prompt_ms = t.get("prompt_ms") or 0
            pred_n = t.get("predicted_n") or 1
            pred_ms = t.get("predicted_ms") or 0
            # 真正的首字延迟 = 预填充全部耗时 + 生成第一个 token 的时间。
            # 注意不能用 predicted_ms 本身（那只是首字之后的生成耗时）。
            first_s = (prompt_ms + pred_ms / max(pred_n, 1)) / 1000
            gpu = mp.gpu_used_mib()
            row = {
                "fill_target": fill,
                "prompt_n": t.get("prompt_n"),
                "prefill_tps": round(pre, 1),
                "prefill_s": round(prompt_ms / 1000, 1),
                "first_token_s": round(first_s, 2),
                "decode_tps": round(dec, 2),
                "predicted_n": pred_n,
                "vram_used_mib": gpu["used_mib"] if gpu else None,
                "sample": (resp.get("content") or "")[:80].replace("\n", " "),
            }
            rows.append(row)
            print(f"{fill:>9}{t.get('prompt_n', 0):>11}{pre:>9.1f}{prompt_ms / 1000:>11.1f}s"
                  f"{first_s:>9.2f}s{dec:>9.2f}{(gpu['used_mib'] if gpu else 0):>10}")

        print()
        if not crashed and len(rows) == len(fills):
            print("[结论] 全部填充级别通过，后端未崩溃。")
        else:
            print("[结论] **出现崩溃/失败**，见下方日志尾部。")

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log_handle.close()

        if crashed:
            print("\n--- 后端日志尾部 ---")
            text = log_path.read_text(encoding="utf-8", errors="replace")
            print("\n".join(text.splitlines()[-12:]))

        print(f"\n日志: {log_path}")
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps({"rows": rows, "crashed": crashed}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"结果: {args.json_out}")
        time.sleep(1)
        freed = mp.gpu_used_mib()
        if freed:
            print(f"已关闭后端，显存已用 {freed['used_mib']} MiB")
    return 1 if crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
