"""诊断：llama-server 的 /health 到底能不能被 httpx 打通。

背景：网页控制台上线后出现「后端明明已 listening、但 server.start() 轮询
300 秒都看不到 200」的现象，而 PowerShell 直连同端口返回 ``{"status":"ok"}``。
本脚本把两种客户端放在一起对比，并打印每次尝试的耗时与异常类型。

用法::

    python scripts/diag_health.py                 # 用默认模型起一个后端再测
    python scripts/diag_health.py --url http://127.0.0.1:8080/health --no-launch
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_server.config import ServerConfig  # noqa: E402
from llm_server.core.backend import resolve_backend  # noqa: E402
from llm_server.core.server import build_load_profiles  # noqa: E402


def try_httpx(url: str, timeout: float = 2.0, trust_env: bool = False) -> str:
    """默认 trust_env=False（修复后的行为）；传 True 可复现被代理拦的情况。"""
    import httpx

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout, trust_env=trust_env) as client:
            resp = client.get(url)
        dt = (time.perf_counter() - t0) * 1000
        body = resp.text[:120].replace("\n", " ")
        return f"HTTP {resp.status_code}  {dt:6.0f}ms  {body}"
    except Exception as exc:  # noqa: BLE001 - 诊断脚本要显示任何异常
        dt = (time.perf_counter() - t0) * 1000
        return f"!! {type(exc).__name__}  {dt:6.0f}ms  {exc}"


def try_urllib(url: str, timeout: float = 2.0) -> str:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read()[:120].decode("utf-8", "replace").replace("\n", " ")
        dt = (time.perf_counter() - t0) * 1000
        return f"HTTP {resp.status}  {dt:6.0f}ms  {body}"
    except urllib.error.HTTPError as exc:
        dt = (time.perf_counter() - t0) * 1000
        return f"HTTP {exc.code}  {dt:6.0f}ms  {exc.read()[:120]!r}"
    except Exception as exc:  # noqa: BLE001
        dt = (time.perf_counter() - t0) * 1000
        return f"!! {type(exc).__name__}  {dt:6.0f}ms  {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="对比 httpx / urllib 打 /health")
    parser.add_argument("--url", default="", help="留空则按 config 拼")
    parser.add_argument("--no-launch", action="store_true", help="不自己起后端")
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()

    # 环境里有没有代理变量？httpx 默认 trust_env=True，会走代理
    print("=== 代理相关环境变量 ===")
    found = False
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        val = os.environ.get(key)
        if val:
            found = True
            print(f"  {key} = {val}")
    if not found:
        print("  （都没有设置）")

    cfg = ServerConfig.load()
    url = args.url or f"{cfg.backend_base_url}/health"
    print(f"\n=== 目标 {url} ===")

    proc = None
    log_handle = None
    try:
        if not args.no_launch:
            backend = resolve_backend(cfg.llama_dir or None)
            prof = build_load_profiles(cfg)[0]
            exe = str(backend.exe)
            # 用一条最小命令，缩短诊断时间
            cmd = [
                exe, "--model", str(cfg.model_file),
                "--host", cfg.backend_host, "--port", str(cfg.backend_port),
                "--n-gpu-layers", "0", "--ctx-size", "4096", "--cpu-moe",
                "--no-warmup",
            ]
            log_path = PROJECT_ROOT / "runtime" / "logs" / "diag-health.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("w", encoding="utf-8", errors="replace")
            print(f"启动: {' '.join(cmd[:6])} ... （日志 {log_path}）")
            proc = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                cwd=str(backend.home), env=backend.build_env(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

        for i in range(1, args.rounds + 1):
            print(f"\n--- 第 {i} 轮 ---")
            print(f"  httpx (trust_env=False，修复后) : {try_httpx(url)}")
            print(f"  httpx (trust_env=True，默认)    : {try_httpx(url, trust_env=True)}")
            print(f"  urllib                         : {try_urllib(url)}")
            if proc is not None and proc.poll() is not None:
                print(f"  （后端已退出 rc={proc.poll()}）")
                break
            time.sleep(1.5)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        if log_handle is not None:
            log_handle.close()

    print("\n=== 结论 ===")
    print("httpx 默认 trust_env=True，会通过 urllib.request.getproxies() 读到")
    print("**Windows 注册表里的系统代理**（不只是环境变量）。本机若装了")
    print("Clash / v2ray 之类（例如 127.0.0.1:7897），httpx 会把发往 127.0.0.1 的")
    print("请求也丢给代理，代理返回 502；而 PowerShell / 浏览器 / urllib 都会")
    print("遵守 Windows 的『绕过本地地址』设置，所以症状是『只有 Python 连不上』。")
    print()
    print("修复：本项目内所有访问 llama-server 的 httpx 客户端都用")
    print("      llm_server/net.py 的 local_client()/local_async_client()，")
    print("      或在构造时显式加 trust_env=False。")
    print("      完整对比见 scripts/diag_http.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
