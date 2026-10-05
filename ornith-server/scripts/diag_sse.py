"""打印 llama-server 流式响应的原始片段，确认 SSE 实际格式。"""

from __future__ import annotations

import os
import subprocess
import time

import httpx

import calibrate as cal

PORT = cal.PORT


def main() -> int:
    case = cal.Case(name="diag2", ctx=8192, kv_type="q8_0", kv_offload=True)
    cal.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = cal.LOG_DIR / "diag2.log"
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
        while time.time() - started < 300:
            if proc.poll() is not None:
                print("退出:", log_path.read_text(encoding="utf-8", errors="replace")[-1500:])
                return 1
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1.5)
        print("就绪\n")

        with httpx.Client(timeout=300) as client:
            print("=== 原始字节（前 40 行）===")
            with client.stream(
                "POST",
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "用一句话说明什么是KV缓存。"}],
                    "max_tokens": 24,
                    "temperature": 0,
                    "stream": True,
                },
            ) as resp:
                print("status:", resp.status_code)
                print("headers:", dict(resp.headers))
                count = 0
                for raw in resp.iter_lines():
                    print(repr(raw))
                    count += 1
                    if count >= 40:
                        break

            print("\n=== 用 bytes 逐块看（原始 chunk 边界）===")
            with client.stream(
                "POST",
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "说三个字。"}],
                    "max_tokens": 8,
                    "temperature": 0,
                    "stream": True,
                },
            ) as resp:
                n = 0
                for chunk in resp.iter_bytes():
                    if chunk:
                        print(repr(chunk[:400]))
                        n += 1
                    if n >= 8:
                        break
    finally:
        cal._kill(proc)
        logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
