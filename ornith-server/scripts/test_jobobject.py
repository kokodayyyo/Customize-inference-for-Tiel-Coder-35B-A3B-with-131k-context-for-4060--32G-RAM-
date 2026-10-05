"""验证两种关闭路径都不会留下占用显存的孤儿 llama-server。

路径 A（正常）：调用 LlamaBackendServer.stop() —— 应优雅退出并释放显存。
路径 B（强杀）：父进程被 taskkill /F 直接杀死 —— 由 Windows Job Object
              兜底结束 llama-server。这是生产上最容易被忽略的泄漏点。

用法::

    python test_jobobject.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ornith_server.core import ChildJob, describe_support  # noqa: E402

HELPER = ROOT / "scripts" / "_backend_holder.py"
FAILED: list[str] = []
PASSED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED
    if ok:
        PASSED += 1
        print(f"  [通过] {name}" + (f"  — {detail}" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  [失败] {name}" + (f"  — {detail}" if detail else ""))


def vram_used() -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip().splitlines()[0]
    return float(out) / 1024


def llama_server_pids() -> list[int]:
    out = subprocess.run(
        ["wmic", "process", "where", "name='llama-server.exe'", "get", "ProcessId"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        # wmic 在新系统上可能缺失，退回 PowerShell
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process llama-server -ErrorAction SilentlyContinue).Id -join ','"],
            capture_output=True, text=True, timeout=30,
        )
        return [int(x) for x in ps.stdout.strip().split(",") if x.strip().isdigit()]
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def wait_gone(pids: list[int], timeout: float = 60) -> float:
    started = time.time()
    while time.time() - started < timeout:
        alive = [p for p in pids if p in llama_server_pids()]
        if not alive:
            return time.time() - started
        time.sleep(0.5)
    return -1.0


def wait_vram(base: float, timeout: float = 60, tol: float = 0.4) -> float:
    started = time.time()
    while time.time() - started < timeout:
        if vram_used() <= base + tol:
            return time.time() - started
        time.sleep(0.5)
    return -1.0


def run_case(force_kill: bool) -> None:
    title = "路径 B：强杀父进程（Job Object 兜底）" if force_kill else "路径 A：正常 stop()"
    print(f"\n{'=' * 66}\n{title}\n{'=' * 66}")

    base = vram_used()
    print(f"  基线显存 {base:.2f} GiB")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(HELPER)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", env=env,
    )

    # 等 helper 报告 READY
    ready_line = ""
    started = time.time()
    while time.time() - started < 300:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            ready_line = line.strip()
            if ready_line.startswith("READY"):
                break
        if proc.poll() is not None:
            print(f"  helper 提前退出 rc={proc.returncode}")
            print(f"  {ready_line}")
            check("helper 启动后端", False, ready_line)
            return
    check("helper 报告就绪", ready_line.startswith("READY"), ready_line)

    pids = llama_server_pids()
    check("llama-server 正在运行", len(pids) > 0, f"pid={pids}")
    peak = vram_used()
    print(f"  峰值显存 {peak:.2f} GiB (增量 {peak - base:+.2f})")

    if force_kill:
        print("  以 taskkill /F 强杀父进程（不做任何清理）…")
        subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)],
                       capture_output=True, timeout=30)
    else:
        print("  向父进程发送 terminate（走正常清理路径）…")
        proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()

    gone = wait_gone(pids, timeout=90)
    released = wait_vram(base, timeout=90)

    check("llama-server 已结束", gone >= 0,
          f"{gone:.1f}s" if gone >= 0 else f"仍有残留 pid={llama_server_pids()}")
    check("显存已释放", released >= 0,
          f"{released:.1f}s, 当前 {vram_used():.2f} GiB" if released >= 0
          else f"当前 {vram_used():.2f} GiB（基线 {base:.2f}）")


def main() -> int:
    print("Job Object 支持情况:", describe_support())

    # 先确认能力可用
    probe = ChildJob("probe")
    supported = probe.handle is not None
    probe.close()
    if not supported:
        print("当前环境不支持 Job Object，跳过强杀用例")

    run_case(force_kill=False)
    if supported:
        run_case(force_kill=True)
    else:
        print("\n[跳过] 路径 B 需要 Job Object 支持")

    print("\n" + "=" * 66)
    print(f"通过 {PASSED} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败: {name}")
    print("=" * 66)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
