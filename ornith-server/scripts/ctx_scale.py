"""上下文规模对速度的影响：把 KV cache 填满再测，而不是空载测。

为什么需要这个测试：空上下文时 attention 只需扫过极少量 KV，
decode 速度看起来接近峰值；但当上下文真的用满了（比如 128K），
每一步都要读取完整的 KV cache（q4_0 下约 4.5 GiB），
真实速度会显著下降。用空载数字做容量规划会严重高估。

本脚本会：
  1. 启动指定配置的 llama-server
  2. 用长提示把上下文填充到目标比例
  3. 在"已填充"状态下测 decode 速度
  4. 同时用非流式请求测 prefill 速度（拿到可信的 prompt_tokens）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

import calibrate as cal

PORT = cal.PORT
FILLER = (
    "深度学习推理引擎的性能取决于计算密度、内存带宽以及缓存局部性这三个因素的平衡。"
    "当上下文长度增长时，键值缓存的读取量线性上升，逐渐取代矩阵乘法成为主要瓶颈。"
)


def build_prompt(target_tokens: int) -> str:
    """构造约 target_tokens 个 token 的提示（中文约 1.5 字/token）。"""
    approx_chars = int(target_tokens * 1.5)
    repeats = max(1, approx_chars // len(FILLER) + 1)
    return (FILLER * repeats)[:approx_chars] + "\n\n请用一句话说明上面这段话的主题。"


def non_streaming_probe(client: httpx.Client, prompt: str, max_tokens: int = 8) -> dict:
    """非流式请求，用返回的 usage 得到可信的 token 计数。"""
    started = time.perf_counter()
    resp = client.post(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=900,
    )
    elapsed = time.perf_counter() - started
    resp.raise_for_status()
    usage = resp.json().get("usage") or {}
    return {
        "seconds": elapsed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def streaming_decode(client: httpx.Client, prompt: str, gen_tokens: int) -> dict:
    """流式生成，测首字延迟与 decode 速度。"""
    started = time.perf_counter()
    ttft = 0.0
    pieces = 0
    usage: dict = {}
    with client.stream(
        "POST",
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": gen_tokens,
            "temperature": 0,
            "stream": True,
        },
        timeout=900,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                pieces += 1
                if ttft == 0.0:
                    ttft = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    completion = int(usage.get("completion_tokens") or pieces)
    decode_seconds = max(elapsed - ttft, 1e-3)
    return {
        "seconds": elapsed,
        "ttft": ttft,
        "completion_tokens": completion,
        "decode_tps": (completion - 1) / decode_seconds if completion > 1 else 0.0,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="上下文填充后的真实速度测量")
    parser.add_argument("--ctx", type=int, default=131072)
    parser.add_argument("--kv-type", default="q4_0")
    parser.add_argument("--kv-offload", action="store_true",
                        help="把 KV cache 放显存（默认放内存）")
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--gen-tokens", type=int, default=96)
    parser.add_argument("--fills", default="0,8000,32768,65536,120000",
                        help="要填充到的上下文 token 数，逗号分隔")
    parser.add_argument("--tag", default="ctxscale")
    args = parser.parse_args()

    name = f"{args.tag}-ctx{args.ctx}-{args.kv_type}-{'gpu' if args.kv_offload else 'cpu'}"
    case = cal.Case(
        name=name,
        ctx=args.ctx,
        kv_type=args.kv_type,
        kv_offload=args.kv_offload,
        ubatch=args.ubatch,
        ngl=args.ngl,
    )

    print(f"配置: ctx={case.ctx} kv={case.kv_type} "
          f"kv_offload={'GPU' if case.kv_offload else 'CPU'} ngl={case.ngl}")
    print(f"引擎: {cal.LLAMA_HOME.name}\n")

    baseline_used, baseline_free = cal.vram()
    print(f"启动前显存: used={baseline_used:.2f} free={baseline_free:.2f} GiB")

    results: list[dict] = []

    # 内联加载流程：需要在同一个进程生命周期内做多次填充测量
    import os
    import subprocess

    cal.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = cal.LOG_DIR / f"{name}.log"
    cmd = cal.build_cmd(case)
    logf = log_path.open("w", encoding="utf-8", errors="replace")
    logf.write("CMDLINE: " + subprocess.list2cmdline(cmd) + "\n\n")
    logf.flush()
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(cal.LLAMA_HOME),
        env=cal.env(), creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    try:
        started = time.time()
        ready = False
        while time.time() - started < 600:
            if proc.poll() is not None:
                break
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1.5)

        if not ready:
            text = log_path.read_text(encoding="utf-8", errors="replace").lower()
            reason = "CUDA 显存不足" if "out of memory" in text else f"启动失败 rc={proc.poll()}"
            print(f"加载失败: {reason}")
            return 1

        load_seconds = time.time() - started
        time.sleep(3)
        used, free = cal.vram()
        print(f"加载 {load_seconds:.1f}s | 显存 used={used:.2f} "
              f"(增量 {used - baseline_used:+.2f}) free={free:.2f} GiB\n")

        with httpx.Client(timeout=900) as client:
            # 预热
            client.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "你好"}],
                      "max_tokens": 16, "temperature": 0},
            )

            print(f"{'填充tokens':>12}{'prefill实测':>14}{'prefill t/s':>13}"
                  f"{'TTFT':>9}{'decode t/s':>13}")
            print("-" * 62)

            for fill in [int(x) for x in args.fills.split(",") if x.strip()]:
                if fill >= case.ctx - 512:
                    print(f"{fill:>12}  跳过（超出上下文上限）")
                    continue
                prompt = build_prompt(fill) if fill > 0 else "你好"

                gen = streaming_decode(client, prompt, args.gen_tokens)
                actual_prompt = gen["prompt_tokens"]
                prefill_tps = actual_prompt / max(gen["ttft"], 1e-3)

                row = {
                    "fill_target": fill,
                    "prompt_tokens": actual_prompt,
                    "ttft": round(gen["ttft"], 3),
                    "prefill_tps": round(prefill_tps, 1),
                    "decode_tps": round(gen["decode_tps"], 2),
                    "completion_tokens": gen["completion_tokens"],
                }
                results.append(row)
                print(f"{actual_prompt:>12}{prefill_tps:>13.0f}t/s"
                      f"{'':>13}{gen['ttft']:>8.2f}s{gen['decode_tps']:>12.2f}")

        used, free = cal.vram()
        print(f"\n结束后显存 used={used:.2f} free={free:.2f} GiB")

        out = cal.BASE / "runtime" / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "config": {"ctx": case.ctx, "kv_type": case.kv_type,
                       "kv_offload": case.kv_offload, "ngl": case.ngl,
                       "ubatch": case.ubatch},
            "load_seconds": round(load_seconds, 1),
            "vram_used_gib": round(used, 2),
            "vram_delta_gib": round(used - baseline_used, 2),
            "results": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"结果已写入 {out}")
    finally:
        cal._kill(proc)
        logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
