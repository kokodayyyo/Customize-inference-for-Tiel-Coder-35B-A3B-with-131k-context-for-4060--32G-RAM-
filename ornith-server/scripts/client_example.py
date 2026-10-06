"""内网 API 调用示例（OpenAI SDK + 原生 HTTP 两种方式）。

用法::

    # 免鉴权
    python scripts/client_example.py

    # 指定服务地址与密钥
    python scripts/client_example.py --base-url http://192.168.1.20:8000/v1 --api-key sk-xxx

    # 只跑原生流式示例
    python scripts/client_example.py --mode raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def chat_openai_sdk(base_url: str, api_key: str, stream: bool) -> None:
    """用官方 openai SDK 调用（最省事的接入方式）。"""
    try:
        from openai import OpenAI
    except ImportError:
        print("[跳过] 未安装 openai 包：pip install openai")
        return

    client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
    print(f"--- OpenAI SDK（stream={stream}）---")
    started = time.perf_counter()

    if stream:
        first = None
        parts: list[str] = []
        reasoning_parts: list[str] = []
        stream_resp = client.chat.completions.create(
            model="tile-35b-a3b",
            messages=[{"role": "user", "content": "用三句话介绍你自己。"}],
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )
        announced_thinking = False
        for chunk in stream_resp:
            delta = chunk.choices[0].delta
            # 该模型会先输出思考内容（reasoning_content），再输出正文（content）
            think = getattr(delta, "reasoning_content", None)
            piece = delta.content or ""
            if think:
                if not announced_thinking:
                    print("[思考] ", end="", flush=True)
                    announced_thinking = True
                reasoning_parts.append(think)
                print(think, end="", flush=True)
            if piece:
                if first is None:
                    first = time.perf_counter() - started
                    if announced_thinking:
                        print("\n[回答] ", end="", flush=True)
                parts.append(piece)
                print(piece, end="", flush=True)
        elapsed = time.perf_counter() - started
        print()
        n = len(parts)
        think_n = len(reasoning_parts)
        print(
            f"[统计] 首字 {first:.2f}s  正文 {n} 段 / 思考 {think_n} 段  "
            f"总耗时 {elapsed:.2f}s"
        )
        if n == 0 and think_n > 0:
            print(
                "[提示] 只产生了思考内容、没有正文：说明 max_tokens 被思考过程耗尽。\n"
                "       请调大 max_tokens，或设置 reasoning_budget 限制思考长度。"
            )
    else:
        resp = client.chat.completions.create(
            model="tile-35b-a3b",
            messages=[{"role": "user", "content": "用三句话介绍你自己。"}],
            max_tokens=1024,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - started
        message = resp.choices[0].message
        think = getattr(message, "reasoning_content", None)
        if think:
            print(f"[思考] {think}\n")
        print(f"[回答] {message.content}")
        usage = resp.usage
        print(
            f"[统计] 总耗时 {elapsed:.2f}s  "
            f"in={getattr(usage, 'prompt_tokens', '?')} out={getattr(usage, 'completion_tokens', '?')}"
        )


def chat_raw(base_url: str, api_key: str) -> None:
    """不依赖任何 SDK，纯 httpx 流式调用。"""
    print("--- 原生 HTTP 流式 ---")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": "tile-35b-a3b",
        "messages": [
            {"role": "system", "content": "你是一个简洁的技术助手。"},
            {"role": "user", "content": "llama.cpp 的 KV cache 量化有什么好处？"},
        ],
        "max_tokens": 300,
        "temperature": 0.6,
        "stream": True,
    }

    started = time.perf_counter()
    ttft = None
    thinking = False
    printed_header = False
    with httpx.Client(timeout=300, trust_env=False) as client:
        with client.stream("POST", f"{base_url}/chat/completions", json=payload, headers=headers) as resp:
            if resp.status_code >= 400:
                print(f"[错误 {resp.status_code}] {resp.read().decode('utf-8', 'replace')}")
                sys.exit(1)
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                obj = json.loads(data)
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                think = delta.get("reasoning_content") or ""
                piece = delta.get("content") or ""
                if think:
                    if not thinking:
                        print("[思考] ", end="", flush=True)
                        thinking = True
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    print(think, end="", flush=True)
                if piece:
                    if not printed_header:
                        if thinking:
                            print("\n[回答] ", end="", flush=True)
                        printed_header = True
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    print(piece, end="", flush=True)
    print(f"\n[统计] 首字 {ttft:.2f}s  总耗时 {time.perf_counter() - started:.2f}s")


def embeddings(base_url: str, api_key: str) -> None:
    """向量接口示例（该模型未专门训练为 embedding 模型，仅演示接口可用性）。"""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    print("--- /v1/embeddings ---")
    with httpx.Client(timeout=120, trust_env=False) as client:
        resp = client.post(
            f"{base_url}/embeddings",
            json={"model": "tile-35b-a3b", "input": ["测试文本"]},
            headers=headers,
        )
    if resp.status_code >= 400:
        print(f"[不支持或错误 {resp.status_code}] {resp.text[:300]}")
        return
    body = resp.json()
    vec = (body.get("data") or [{}])[0].get("embedding") or []
    print(f"维度 {len(vec)}，前 5 个值 {vec[:5]}")


def health(base_url: str, api_key: str) -> None:
    root = base_url.rsplit("/v1", 1)[0]
    print("--- 健康检查 ---")
    with httpx.Client(timeout=30, trust_env=False) as client:
        for path in ("/health", "/stats"):
            try:
                resp = client.get(root + path)
                print(f"{path}: {json.dumps(resp.json(), ensure_ascii=False)[:400]}")
            except httpx.HTTPError as exc:
                print(f"{path}: 失败 {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ornith 内网 API 调用示例")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--mode", choices=["all", "sdk", "sdk-sync", "raw", "embed", "health"], default="all"
    )
    args = parser.parse_args()

    try:
        if args.mode in ("all", "health"):
            health(args.base_url, args.api_key)
        if args.mode in ("all", "sdk"):
            chat_openai_sdk(args.base_url, args.api_key, stream=True)
        if args.mode == "sdk-sync":
            chat_openai_sdk(args.base_url, args.api_key, stream=False)
        if args.mode in ("all", "raw"):
            chat_raw(args.base_url, args.api_key)
        if args.mode in ("all", "embed"):
            embeddings(args.base_url, args.api_key)
    except httpx.ConnectError:
        print(f"[错误] 无法连接 {args.base_url}，服务是否已启动？")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
