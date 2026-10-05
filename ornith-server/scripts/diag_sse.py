"""打印服务端流式响应的原始片段，确认 SSE 实际格式。

这是排查流式问题的第一把工具。当初就是靠它发现这个模型是推理模型、
思考内容走 ``delta.reasoning_content`` 而不是 ``delta.content`` 的。

**它不自己启动后端**，直接打一个已经在跑的服务（默认网关 8000）。
先启动服务再跑本脚本：

    start_server.bat
    python scripts/diag_sse.py

想直接看 llama.cpp 后端的原始输出（绕过网关透传）就加
``--base-url http://127.0.0.1:8080``。
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="打印 SSE 原始片段")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="服务地址；看 llama.cpp 原始输出用 http://127.0.0.1:8080")
    parser.add_argument("--api-key", default="", help="服务启用了鉴权时填写")
    parser.add_argument("--model", default="tile-35b-a3b")
    parser.add_argument("--lines", type=int, default=40, help="打印多少行 SSE")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    # 先确认服务在跑，否则报错信息会很晦涩
    try:
        health = httpx.get(f"{base}/health", timeout=5)
        print(f"健康检查 {health.status_code}: {health.text[:200]}")
    except httpx.HTTPError as exc:
        print(f"连不上 {base}：{exc}", file=sys.stderr)
        print("请先启动服务（start_server.bat），或用 --base-url 指定地址。", file=sys.stderr)
        return 2
    print()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "用一句话说明什么是 KV 缓存。"}],
        "max_tokens": 24,
        "temperature": 0,
        "stream": True,
    }

    with httpx.Client(timeout=300) as client:
        print("=== 逐行看（SSE 事件边界）===")
        with client.stream("POST", f"{base}/v1/chat/completions",
                           json=payload, headers=headers) as resp:
            print("status :", resp.status_code)
            print("headers:", dict(resp.headers))
            if resp.status_code >= 400:
                print(resp.read().decode("utf-8", "replace"))
                return 1
            print()
            seen: set[str] = set()
            count = 0
            for raw in resp.iter_lines():
                print(repr(raw))
                if raw.startswith("data:"):
                    body = raw[5:].strip()
                    if body and body != "[DONE]":
                        try:
                            delta = (json.loads(body).get("choices") or [{}])[0].get("delta") or {}
                            seen.update(delta.keys())
                        except json.JSONDecodeError:
                            pass
                count += 1
                if count >= args.lines:
                    break

        print("\n=== 逐块看（原始字节边界）===")
        payload["messages"] = [{"role": "user", "content": "说三个字。"}]
        payload["max_tokens"] = 8
        with client.stream("POST", f"{base}/v1/chat/completions",
                           json=payload, headers=headers) as resp:
            n = 0
            for chunk in resp.iter_bytes():
                if chunk:
                    print(repr(chunk[:400]))
                    n += 1
                if n >= 8:
                    break

    print()
    if seen:
        print(f"delta 中出现过的字段: {sorted(seen)}")
        if "reasoning_content" in seen:
            print("注意：这是推理模型，思考内容走 reasoning_content，")
            print("      客户端只读 content 会以为「没有输出」。两者都要处理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
