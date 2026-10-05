"""上下文规模 × KV 位置的性能标定（修正版）。

本版修正了三个测量陷阱：
  1. 推理模型的流式输出走 ``delta.reasoning_content``，必须先计入首字，
     否则 TTFT 会被算成"答案第一个字"的时间，严重偏高；
  2. 吞吐一律取服务端返回的 ``timings``（``prompt_per_second`` /
     ``predicted_per_second``），而不是用客户端计时反推；
  3. 预热文本与测量文本不能共享前缀，否则触发 prompt cache 复用
     （``cache_n``），prefill 会被测成近乎瞬时。

用法::

    python perf_matrix.py                          # 全部场景
    python perf_matrix.py --only 128k-cpu          # 只测一个场景
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

import calibrate as cal

PORT = cal.PORT
OUT_FILE = cal.BASE / "runtime" / "perf_matrix.json"

# 用于填充提示的语料池：每次取不同的起始片段，避免前缀命中缓存
CORPUS = [
    "在显存受限的设备上部署大语言模型时，键值缓存占用往往成为决定上下文长度的瓶颈。",
    "预填充阶段的计算以矩阵乘法为主，属于计算密集型，因此吞吐通常远高于逐字生成。",
    "解码阶段每生成一个词元都要读取完整的键值缓存，因此受显存与内存带宽的约束。",
    "分组查询注意力通过让多个查询头共享一组键值头，显著压缩了缓存的规模。",
    "把键值缓存放在系统内存中，可以腾出显存容纳更长的上下文，代价是注意力需要跨总线取数。",
    "批处理规模决定了显存中计算缓冲的峰值，长上下文场景下应当适当调小物理批。",
    "滑动窗口注意力只保留最近一段上下文，能够在固定显存下处理超长输入。",
    "量化的键值缓存用少量精度损失换取显存占用的大幅下降，是长上下文的常用手段。",
    "上下文切换与缓存复用策略会影响多轮对话的响应速度，尤其在重复前缀较多的场景。",
    "线程数主要影响采样与调度开销，对以图形处理器为主的矩阵运算影响有限。",
]


@dataclass
class Scenario:
    name: str
    ctx: int
    kv_type: str
    kv_offload: bool
    ubatch: int = 512
    ngl: int = 99
    note: str = ""
    fills: list[int] = field(default_factory=lambda: [12000])


@dataclass
class Measurement:
    fill_target: int
    prompt_n: int
    prompt_ms: float
    predicted_n: int
    predicted_ms: float
    ttft: float
    prefill_tps: float
    decode_tps: float


def build_corpus_prompt(target_chars: int, salt: int) -> str:
    """构造指定长度的提示；``salt`` 决定起始语料，避免与预热共享前缀。"""
    rng = random.Random(salt)
    parts: list[str] = []
    total = 0
    while total < target_chars:
        parts.append(rng.choice(CORPUS))
        total += len(parts[-1])
    body = "".join(parts)[:target_chars]
    # 尾部加一个随机标记，确保整段提示不可能与之前请求完全相同
    tag = "".join(rng.choices(string.ascii_letters + string.digits, k=12))
    return f"[会话标记 {tag}]\n{body}\n\n请用一句话概括以上内容。"


def request(
    client: httpx.Client,
    prompt: str,
    max_tokens: int,
    *,
    stream: bool,
    use_cache: bool = True,
    task: str | None = None,
) -> dict:
    """发一次请求，返回 timings、usage、TTFT 等。

    ``use_cache=False`` 会设置 ``cache_prompt: false``，禁止服务端复用前缀
    KV。测预填充速度时必须关掉它，否则同一段提示的第二次请求只会处理
    几个新 token，prefill 会被测成近乎瞬时。

    ``task`` 会替换提示末尾的指令部分，用于让模型持续输出（测量解码速度）。
    """
    user_content = prompt if task is None else f"{prompt}\n\n{task}"
    payload: dict = {
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
        "cache_prompt": use_cache,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    ttft = 0.0
    pieces = 0
    usage: dict = {}
    timings: dict = {}

    def note_piece() -> None:
        nonlocal ttft, pieces
        pieces += 1
        if ttft == 0.0:
            ttft = time.perf_counter() - started

    if stream:
        with client.stream(
            "POST", f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=1800
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
                if isinstance(obj.get("usage"), dict) and obj["usage"]:
                    usage = obj["usage"]
                if isinstance(obj.get("timings"), dict) and obj["timings"]:
                    timings = obj["timings"]
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                # 关键：推理模型的思考内容也走流式，必须一并计入首字时间
                if delta.get("content") or delta.get("reasoning_content"):
                    note_piece()
    else:
        resp = client.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=1800
        )
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage") or {}
        timings = body.get("timings") or {}
        message = ((body.get("choices") or [{}])[0].get("message") or {})
        if message.get("content") or message.get("reasoning_content"):
            note_piece()

    return {
        "wall": time.perf_counter() - started,
        "ttft": ttft,
        "pieces": pieces,
        "usage": usage,
        "timings": timings,
    }


def measure_scenario(scenario: Scenario, warmup_prompt: str) -> dict:
    cal.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = cal.LOG_DIR / f"matrix-{scenario.name}.log"
    case = cal.Case(
        name=scenario.name, ctx=scenario.ctx, kv_type=scenario.kv_type,
        kv_offload=scenario.kv_offload, ubatch=scenario.ubatch, ngl=scenario.ngl,
    )
    cmd = cal.build_cmd(case)

    baseline_used, _ = cal.vram()
    logf = log_path.open("w", encoding="utf-8", errors="replace")
    logf.write("CMDLINE: " + subprocess.list2cmdline(cmd) + "\n\n")
    logf.flush()

    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(cal.LLAMA_HOME),
        env=cal.env(), creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    result: dict = {
        "name": scenario.name, "ctx": scenario.ctx, "kv_type": scenario.kv_type,
        "kv_offload": scenario.kv_offload, "ubatch": scenario.ubatch,
        "ngl": scenario.ngl, "note": scenario.note, "ok": False,
        "measurements": [],
    }

    try:
        started = time.time()
        ready = False
        while time.time() - started < 900:
            if proc.poll() is not None:
                break
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1.5)

        result["load_seconds"] = round(time.time() - started, 1)

        if not ready:
            text = log_path.read_text(encoding="utf-8", errors="replace").lower()
            if "out of memory" in text or "failed to allocate" in text:
                result["error"] = "CUDA 显存不足"
            else:
                result["error"] = f"启动失败 rc={proc.poll()}"
            return result

        time.sleep(3)
        used, free = cal.vram()
        result.update({
            "ok": True,
            "vram_used_gib": round(used, 2),
            "vram_delta_gib": round(used - baseline_used, 2),
            "vram_free_gib": round(free, 2),
        })

        with httpx.Client(timeout=1800) as client:
            # 预热用完全不同的文本，且明确要求它不参与后面的测量
            request(client, warmup_prompt, 32, stream=False)

            for fill in scenario.fills:
                if fill >= scenario.ctx - 1024:
                    continue
                target_chars = int(fill * 1.4)
                prompt = build_corpus_prompt(target_chars, salt=fill + scenario.ctx)

                # 先拿提示的 token 数（用非流式单 token 输出，且禁用缓存复用）
                probe = request(client, prompt, 1, stream=False, use_cache=False)
                prompt_n = int((probe["usage"] or {}).get("prompt_tokens") or 0)

                # 正式测量：禁用缓存复用以测完整预填充；输出用"必须持续生成"
                # 的任务（计数），避免模型直接吐结束标记导致样本过短。
                run = request(client, prompt, 512, stream=True, use_cache=False,
                              task="请从 1 开始连续数数，用空格分隔，至少数到 100 再停。")
                timings = run["timings"] or {}
                predicted_n = int(timings.get("predicted_n") or 0)
                predicted_ms = float(timings.get("predicted_ms") or 0)
                prefill_tps = float(timings.get("prompt_per_second") or 0)
                decode_tps = float(timings.get("predicted_per_second") or 0)

                m = Measurement(
                    fill_target=fill,
                    prompt_n=int(timings.get("prompt_n") or prompt_n),
                    prompt_ms=float(timings.get("prompt_ms") or 0),
                    predicted_n=predicted_n,
                    predicted_ms=predicted_ms,
                    ttft=round(run["ttft"], 3),
                    prefill_tps=round(prefill_tps, 1),
                    decode_tps=round(decode_tps, 2),
                )
                result["measurements"].append(m.__dict__)
                print(f"      fill≈{fill:>6} tok -> 实测 {m.prompt_n:>6} tok | "
                      f"prefill {m.prefill_tps:>8.0f} tok/s | TTFT {m.ttft:>6.2f}s | "
                      f"decode {m.decode_tps:>6.2f} tok/s")
    finally:
        cal._kill(proc)
        logf.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="上下文规模与 KV 位置的性能矩阵")
    parser.add_argument("--only", default=None, help="只跑指定场景名")
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    scenarios = [
        Scenario("8k-q8-gpu", 8192, "q8_0", True, note="8K / KV 在显存",
                 fills=[7000]),
        Scenario("32k-q8-gpu", 32768, "q8_0", True, note="32K / KV 在显存",
                 fills=[8000, 30000]),
        Scenario("64k-q4-gpu", 65536, "q4_0", True, note="64K / KV 在显存",
                 fills=[8000, 60000]),
        Scenario("128k-q8-cpu", 131072, "q8_0", False, note="128K / KV 在内存 q8_0",
                 fills=[8000, 32000, 64000, 120000]),
        Scenario("128k-q4-cpu", 131072, "q4_0", False, note="128K / KV 在内存 q4_0",
                 fills=[8000, 32000, 64000, 120000]),
        Scenario("128k-q4-gpu", 131072, "q4_0", True, note="128K / KV 在显存（对比）",
                 fills=[8000, 64000, 120000]),
    ]
    if args.only:
        scenarios = [s for s in scenarios if s.name == args.only]
        if not scenarios:
            print(f"没有匹配的场景: {args.only}")
            return 2

    print("=" * 96)
    print("  上下文规模 × KV 位置  性能标定")
    print("  吞吐取服务端 timings；提示每次不同以规避 prompt cache 复用")
    print("=" * 96)

    warmup_prompt = "请只回复两个字：收到。"
    results: list[dict] = []

    for scenario in scenarios:
        used, free = cal.vram()
        print(f"\n--- {scenario.name}: {scenario.note} ---")
        print(f"    ctx={scenario.ctx} kv={scenario.kv_type} "
              f"kv位置={'显存' if scenario.kv_offload else '内存'} ub={scenario.ubatch} "
              f"ngl={scenario.ngl}")
        print(f"    启动前显存 used={used:.2f} free={free:.2f} GiB")
        outcome = measure_scenario(scenario, warmup_prompt)
        results.append(outcome)

        if outcome.get("ok"):
            print(f"    加载 {outcome['load_seconds']}s | 显存 used={outcome['vram_used_gib']:.2f} "
                  f"(增量 {outcome['vram_delta_gib']:+.2f}) free={outcome['vram_free_gib']:.2f} GiB")
        else:
            print(f"    失败: {outcome.get('error')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out_path}")

    print("\n" + "=" * 96)
    print(f"{'场景':<16}{'上下文':>8}{'KV位置':>8}{'显存增量':>10}{'实测提示':>10}"
          f"{'prefill':>12}{'decode':>12}")
    print("-" * 96)
    for r in results:
        if not r.get("ok"):
            print(f"{r['name']:<16}{r['ctx']:>8}{'显存' if r['kv_offload'] else '内存':>8}"
                  f"   {r.get('error')}")
            continue
        loc = "显存" if r["kv_offload"] else "内存"
        if not r["measurements"]:
            print(f"{r['name']:<16}{r['ctx']:>8}{loc:>8}{r['vram_delta_gib']:>9.2f}G"
                  f"{'（无测量）':>10}")
            continue
        for m in r["measurements"]:
            print(f"{r['name']:<16}{r['ctx']:>8}{loc:>8}{r['vram_delta_gib']:>9.2f}G"
                  f"{m['prompt_n']:>10}{m['prefill_tps']:>10.0f}t/s"
                  f"{m['decode_tps']:>10.2f}t/s")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
