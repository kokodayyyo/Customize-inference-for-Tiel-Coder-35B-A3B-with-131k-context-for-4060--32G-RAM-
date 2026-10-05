"""诊断：确认服务端的 usage 返回方式与 /metrics 里的真实吞吐。

问题背景：流式响应里 usage 为空，导致按流式片段测出的 prefill 速度全部失真。
本脚本在同一个服务实例上对比三种取数方式：
  1. 流式（不带 include_usage）—— 看它到底给不给 usage
  2. 流式 + stream_options.include_usage
  3. 非流式（直接用返回体的 usage）
  4. /metrics 里的 llamacpp:prompt_tokens_seconds / predicted_tokens_seconds

并打印服务端自己统计的 ``timings``，以此判断哪种口径可信。

**它不自己启动后端**，直接打一个已经在跑的服务（默认网关 8000）：

    start_server.bat
    python scripts/diag_usage.py
"""

from __future__ import annotations

import argparse
import json
import re
import time

import httpx

FILLER = (
    "这段文字用于构造一个足够长的提示，以便准确测量预填充阶段的吞吐表现。"
    "推理引擎在处理长提示时需要先完成全部注意力计算，然后才开始逐 token 生成。"
)


def build_prompt(target_tokens: int) -> str:
    approx_chars = int(target_tokens * 1.5)
    repeats = max(1, approx_chars // len(FILLER) + 1)
    return (FILLER * repeats)[:approx_chars] + "\n\n请概括以上内容。"


def streaming(client: httpx.Client, base: str, prompt: str, max_tokens: int,
              include_usage: bool) -> dict:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    ttft = 0.0
    pieces = 0
    usage: dict = {}
    timings: dict = {}
    with client.stream(
        "POST", f"{base}/v1/chat/completions", json=payload, timeout=900
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
            if isinstance(obj.get("timings"), dict):
                timings = obj["timings"]
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                pieces += 1
                if ttft == 0.0:
                    ttft = time.perf_counter() - started
    return {
        "wall": time.perf_counter() - started,
        "ttft": ttft,
        "chunks": pieces,
        "usage": usage,
        "timings": timings,
    }


def non_streaming(client: httpx.Client, base: str, prompt: str, max_tokens: int) -> dict:
    started = time.perf_counter()
    resp = client.post(
        f"{base}/v1/chat/completions",
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
    body = resp.json()
    return {
        "wall": elapsed,
        "usage": body.get("usage") or {},
        "timings": body.get("timings") or {},
    }


def scrape_metrics(client: httpx.Client, base: str) -> dict:
    text = client.get(f"{base}/metrics", timeout=30).text
    wanted = (
        "prompt_tokens_seconds",
        "predicted_tokens_seconds",
        "prompt_tokens_total",
        "tokens_predicted_total",
        "n_decode_total",
        "n_tokens_max",
    )
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"llamacpp:(\w+)\s+([\d.eE+-]+)", line)
        if m and m.group(1) in wanted:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="对比 usage / timings / metrics 三种取数口径")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="服务地址；看 llama.cpp 原始输出用 http://127.0.0.1:8080")
    parser.add_argument("--api-key", default="", help="服务启用了鉴权时填写")
    parser.add_argument("--prompt-tokens", type=int, default=3000, help="构造多长的提示")
    parser.add_argument("--gen-tokens", type=int, default=64)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    try:
        health = httpx.get(f"{base}/health", timeout=5)
        print(f"健康检查 {health.status_code}: {health.text[:160]}\n")
    except httpx.HTTPError as exc:
        print(f"连不上 {base}：{exc}")
        print("请先启动服务（start_server.bat），或用 --base-url 指定地址。")
        return 2

    prompt = build_prompt(args.prompt_tokens)
    with httpx.Client(timeout=900, headers=headers) as client:
        # 预热：第一次请求包含 CUDA 图捕获等一次性开销，不计入
        client.post(
            f"{base}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "你好"}],
                  "max_tokens": 8, "temperature": 0},
        )

        print("=== 1) 流式（不带 include_usage）===")
        r1 = streaming(client, base, prompt, args.gen_tokens, False)
        print(f"wall={r1['wall']:.2f}s ttft={r1['ttft']:.2f}s chunks={r1['chunks']}")
        print(f"usage={r1['usage']}")
        print(f"timings={json.dumps(r1['timings'], ensure_ascii=False)}\n")

        print("=== 2) 流式（带 stream_options.include_usage）===")
        r2 = streaming(client, base, prompt, args.gen_tokens, True)
        print(f"wall={r2['wall']:.2f}s ttft={r2['ttft']:.2f}s chunks={r2['chunks']}")
        print(f"usage={r2['usage']}")
        print(f"timings={json.dumps(r2['timings'], ensure_ascii=False)}\n")

        print("=== 3) 非流式 ===")
        r3 = non_streaming(client, base, prompt, args.gen_tokens)
        print(f"wall={r3['wall']:.2f}s")
        print(f"usage={r3['usage']}")
        print(f"timings={json.dumps(r3['timings'], ensure_ascii=False)}\n")

        print("=== 4) 非流式 max_tokens=1（近似纯 prefill）===")
        r4 = non_streaming(client, base, prompt, 1)
        print(f"wall={r4['wall']:.2f}s")
        print(f"usage={r4['usage']}\n")

        print("=== 5) /metrics ===")
        print(json.dumps(scrape_metrics(client, base), ensure_ascii=False, indent=2))

        print("\n=== 6) /props（看服务端默认是否返回 usage）===")
        try:
            props = client.get(f"{base}/props", timeout=30).json()
            print(json.dumps({k: v for k, v in props.items()
                              if k in ("default_generation_settings", "total_slots",
                                       "model_path", "chat_template_tool_use")},
                             ensure_ascii=False)[:600])
        except (httpx.HTTPError, ValueError) as exc:
            print(f"（/props 不可用：{exc}）")

    print("\n=== 结论 ===")
    print("以服务端返回的 timings 为准（prompt_per_second / predicted_per_second）。")
    print("客户端计时会把网络与排队算进去；流式片段数也不等于 token 数。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
