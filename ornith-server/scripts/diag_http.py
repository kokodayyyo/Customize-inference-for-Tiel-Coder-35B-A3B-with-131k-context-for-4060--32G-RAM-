"""诊断：本机环境下 httpx 与 urllib 访问 127.0.0.1 的行为差异。

背景：网页控制台上线后出现「PowerShell / 浏览器能连上 llama-server 的
/health，但 Python 的 httpx 拿到 502」的现象，导致 ``server.start()`` 轮询
300 秒都看不到 200。本脚本用最小 HTTP 服务器复现并对比各种客户端配置。

用法::

    python scripts/diag_http.py
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading
import time
import urllib.request

PORT = 18999


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler 的接口
        body = json.dumps({"ok": True, "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: ANN002 - 静音
        pass


def main() -> int:
    print("=== 环境变量 ===")
    any_proxy = False
    for key in sorted(os.environ):
        if "PROXY" in key.upper():
            print(f"  {key} = {os.environ[key]}")
            any_proxy = True
    if not any_proxy:
        print("  （没有任何 *PROXY* 变量）")

    try:
        import httpx
    except ImportError:
        print("没有安装 httpx")
        return 1

    print(f"\n=== httpx {httpx.__version__} 认为的代理配置 ===")
    try:
        from httpx._utils import get_environment_proxies

        proxies = get_environment_proxies()
        print(f"  get_environment_proxies() = {proxies or '（空）'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  取不到: {exc}")

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    url = f"http://127.0.0.1:{PORT}/ping"
    print(f"\n=== 目标 {url}（本地最小 HTTP 服务）===")

    def show(label: str, fn) -> None:  # noqa: ANN001
        t0 = time.perf_counter()
        try:
            print(f"  {label:<34}: {fn()}  ({(time.perf_counter() - t0) * 1000:.0f}ms)")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<34}: !! {type(exc).__name__}: {exc}"
                  f"  ({(time.perf_counter() - t0) * 1000:.0f}ms)")

    def with_client(**kw):  # noqa: ANN003, ANN202
        def run():
            with httpx.Client(timeout=5.0, **kw) as c:
                mounts = dict(c._mounts) if c._mounts else "（无）"
                r = c.get(url)
                return f"HTTP {r.status_code} {r.text[:50]}  mounts={mounts}"
        return run

    show("httpx 默认(trust_env=True)", with_client())
    show("httpx trust_env=False", with_client(trust_env=False))
    show("httpx http2=False", with_client(http2=False))

    def sync_get():
        r = httpx.get(url, timeout=5.0)
        return f"HTTP {r.status_code} {r.text[:50]}"

    show("httpx.get 默认", sync_get)

    def url_get():
        with urllib.request.urlopen(url, timeout=5) as r:
            return f"HTTP {r.status} {r.read()[:50].decode()}"

    show("urllib", url_get)

    def asyncio_httpx():
        import asyncio

        async def go():
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(url)
                return f"HTTP {r.status_code} {r.text[:50]}"
        return asyncio.run(go())

    show("httpx.AsyncClient 默认", asyncio_httpx)

    def asyncio_noenv():
        import asyncio

        async def go():
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as c:
                r = await c.get(url)
                return f"HTTP {r.status_code} {r.text[:50]}"
        return asyncio.run(go())

    show("httpx.AsyncClient trust_env=False", asyncio_noenv)

    srv.shutdown()
    print("\n=== 结论 ===")
    print("如果只有默认(trust_env=True)的 httpx 失败、trust_env=False 正常，")
    print("说明环境里有代理配置干扰了本机回环请求。项目里访问 127.0.0.1 的")
    print("httpx 客户端应统一设 trust_env=False —— 本地流量永远不该走代理。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
