"""MoE 摆位探针：启动 llama-server 实测显存/内存/速度，然后干净退出。

用于验证"专家权重放内存、注意力+KV 放显存"这条路线，并扫描
``--n-cpu-moe`` 找到显存能容纳的最大专家层数。

测量口径（全部取服务端 ``timings`` 字段，不用客户端计时）：
  - prefill: 长 prompt + max_tokens=1 + cache_prompt=false，读 prompt_per_second
  - decode : 短 prompt + 固定 max_tokens，读 predicted_per_second

用法::

    python scripts/moe_probe.py --ctx 131072 --cpu-moe
    python scripts/moe_probe.py --ctx 131072 --n-cpu-moe 34
    python scripts/moe_probe.py --dry-run
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKENDS = PROJECT_ROOT / "runtime" / "llama.cpp" / "backends"
DEFAULT_MODEL = Path(r"D:\models\Tile\Tile-35BA3B\Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf")
GB = 1024**3
MIB = 1024**2


# ---------------------------------------------------------------------------
# 环境探测
# ---------------------------------------------------------------------------

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def system_memory() -> dict[str, float]:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return {
        "total_gib": round(stat.ullTotalPhys / GB, 2),
        "avail_gib": round(stat.ullAvailPhys / GB, 2),
        "load_percent": float(stat.dwMemoryLoad),
    }


def process_memory(pid: int) -> dict[str, float]:
    """子进程的物理内存占用（工作集 / 私有提交）。"""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid
    )
    if not handle:
        return {}
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return {}
        return {
            "working_set_gib": round(counters.WorkingSetSize / GB, 2),
            "peak_working_set_gib": round(counters.PeakWorkingSetSize / GB, 2),
            "pagefile_gib": round(counters.PagefileUsage / GB, 2),
        }
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


_NVIDIA_SMI: str | None = None


def nvidia_smi() -> str | None:
    global _NVIDIA_SMI
    if _NVIDIA_SMI is None:
        for cand in (
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvidia-smi.exe"),
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ):
            if Path(cand).is_file():
                _NVIDIA_SMI = cand
                break
    return _NVIDIA_SMI


def gpu_used_mib() -> float | None:
    exe = nvidia_smi()
    if not exe:
        return None
    out = subprocess.run(
        [exe, "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0:
        return None
    used, free = (int(x.strip()) for x in out.stdout.strip().splitlines()[0].split(","))
    return {"used_mib": used, "free_mib": free}


# ---------------------------------------------------------------------------
# 运行时定位
# ---------------------------------------------------------------------------

def find_backend() -> tuple[Path, Path | None]:
    """在项目自带的 runtime 目录里找 llama-server.exe 及其 vendor DLL。"""
    if not BACKENDS.is_dir():
        raise FileNotFoundError(f"没有找到项目自带运行时: {BACKENDS}")
    engines = sorted(
        (d for d in BACKENDS.iterdir() if d.is_dir() and (d / "llama-server.exe").is_file()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not engines:
        raise FileNotFoundError(f"{BACKENDS} 下没有 llama-server.exe")
    engine = engines[0]
    vendor_root = BACKENDS / "vendor"
    vendor = None
    if vendor_root.is_dir():
        names = sorted(d.name for d in vendor_root.iterdir() if d.is_dir())
        # CUDA 引擎必须配 CUDA vendor
        pick = [n for n in names if "cuda" in n.lower()] or names
        if pick:
            vendor = vendor_root / pick[0]
    return engine, vendor


# ---------------------------------------------------------------------------
# HTTP 小工具
# ---------------------------------------------------------------------------

def http_json(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def wait_health(base: str, proc: subprocess.Popen, deadline: float) -> bool:
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=3) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read() or b"{}")
                    if str(body.get("status", "ok")) in ("ok", "no slot available"):
                        return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# 压测载荷
# ---------------------------------------------------------------------------

def make_prefill_prompt(target_tokens: int) -> str:
    """构造一段确定性的长文本，约 target_tokens 个 token。

    只用纯字母单词（不加数字后缀）：数字会被分词器切成多个 token，使实际
    token 数变成目标的 3 倍以上，导致测试耗时失控。
    """
    words = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
             "kilo lima mike november oscar papa quebec romeo sierra tango "
             "uniform victor whiskey xray yankee zulu").split()
    # 英文纯字母单词约 1.3 token/词（实测校准见 TOKEN_PER_WORD）
    n_words = int(target_tokens / TOKEN_PER_WORD) + 1
    body = " ".join(words[i % len(words)] for i in range(n_words))
    return ("Read the following data and reply with the single word OK.\n\n" + body +
            "\n\nReply with OK only.")


# 实测：纯字母单词的 token/词 比例（含标点与空格）
TOKEN_PER_WORD = 1.35


def run_completion(base: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": False,
        "stream": False,
    }
    return http_json(f"{base}/completion", payload)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MoE 摆位探针")
    p.add_argument("-m", "--model", default=str(DEFAULT_MODEL))
    p.add_argument("--ctx", type=int, default=131072, help="上下文长度")
    p.add_argument("--ngl", type=int, default=99, help="卸载到 GPU 的层数")
    p.add_argument("--cpu-moe", action="store_true", help="全部专家权重放内存")
    p.add_argument("--n-cpu-moe", type=int, default=None,
                   help="前 N 层的专家权重放内存，其余放显存")
    p.add_argument("--kv-type", default="q8_0", help="KV cache 量化类型")
    p.add_argument("--no-kv-offload", action="store_true", help="KV cache 放内存")
    p.add_argument("--ubatch", type=int, default=512)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--threads", type=int, default=16,
                   help="CPU 线程数；本机 16 核 32 线程，prefill 是 CPU 算力瓶颈")
    p.add_argument("--threads-batch", type=int, default=0,
                   help="prompt 处理专用线程数，0 = 与 threads 相同")
    p.add_argument("--load-mode", default="", choices=["", "auto", "mmap", "mlock",
                                                       "mmap+mlock", "none"],
                   help="模型加载模式；专家放内存时 none 往往比 mmap 快")
    p.add_argument("--no-mmap", action="store_true", help="等价于 --load-mode none")
    p.add_argument("--port", type=int, default=8199)
    p.add_argument("--load-timeout", type=int, default=900)
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--prefill-tokens", type=int, default=8192)
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json-out", default="", help="把结果写入 JSON 文件")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="透传给 llama-server 的额外参数")
    return p


def build_command(args, exe: Path) -> list[str]:
    cmd = [
        str(exe),
        "--model", str(args.model),
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--n-gpu-layers", str(args.ngl),
        "--ctx-size", str(args.ctx),
        "--parallel", "1",
        "--batch-size", str(args.batch),
        "--ubatch-size", str(args.ubatch),
        "--threads", str(args.threads),
        "--flash-attn", "on",
        "--cache-type-k", args.kv_type,
        "--cache-type-v", args.kv_type,
        "--no-warmup",
        "--jinja",
    ]
    cmd.append("--no-kv-offload" if args.no_kv_offload else "--kv-offload")
    if args.cpu_moe:
        cmd.append("--cpu-moe")
    if args.n_cpu_moe is not None:
        cmd += ["--n-cpu-moe", str(args.n_cpu_moe)]
    if args.threads_batch > 0:
        cmd += ["--threads-batch", str(args.threads_batch)]
    load_mode = "none" if args.no_mmap else args.load_mode
    if load_mode:
        cmd += ["--load-mode", load_mode]
    cmd += list(args.extra)
    return cmd


def main() -> int:
    args = build_parser().parse_args()
    engine, vendor = find_backend()
    exe = engine / "llama-server.exe"
    cmd = build_command(args, exe)

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(p) for p in (vendor, engine) if p] + [env.get("PATH", "")]
    )
    env.setdefault("GGML_CUDA_NO_PEER_COPY", "1")

    label = f"ctx={args.ctx} ngl={args.ngl} t={args.threads} " + (
        "cpu-moe(全部专家)" if args.cpu_moe
        else (f"n-cpu-moe={args.n_cpu_moe}" if args.n_cpu_moe is not None else "专家默认(显存)")
    ) + f" kv={args.kv_type}{'(内存)' if args.no_kv_offload else '(显存)'}" + (
        f" load={args.no_mmap and 'none' or args.load_mode}" if (args.no_mmap or args.load_mode) else ""
    )

    print("=" * 78)
    print(f"探针: {label}")
    print(f"引擎: {exe}")
    print(f"vendor: {vendor}")
    print("命令行:")
    print("  " + subprocess.list2cmdline(cmd))
    print("=" * 78)

    if args.dry_run:
        return 0

    if not Path(args.model).is_file():
        print(f"模型不存在: {args.model}")
        return 2

    before = {"mem": system_memory(), "gpu": gpu_used_mib()}
    print(f"加载前: 内存可用 {before['mem']['avail_gib']} GiB | "
          f"显存已用 {before['gpu']['used_mib'] if before['gpu'] else '?'} MiB")

    log_path = PROJECT_ROOT / "runtime" / "logs" / f"probe-{int(time.time())}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
    log_handle.write("CMDLINE: " + subprocess.list2cmdline(cmd) + "\n\n")
    log_handle.flush()

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT,
        cwd=str(engine), env=env, creationflags=creationflags,
    )
    result: dict = {"label": label, "cmd": cmd, "ctx": args.ctx,
                    "kv_type": args.kv_type, "kv_offload": not args.no_kv_offload,
                    "cpu_moe": args.cpu_moe, "n_cpu_moe": args.n_cpu_moe,
                    "ok": False}
    started = time.time()
    try:
        base = f"http://127.0.0.1:{args.port}"
        if not wait_health(base, proc, time.time() + args.load_timeout):
            rc = proc.poll()
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            )
            print(f"\n加载失败 (rc={rc})，日志尾部:\n{tail}")
            result["error"] = f"加载失败 rc={rc}"
            return 1

        load_s = time.time() - started
        after = {"mem": system_memory(), "gpu": gpu_used_mib()}
        pmem = process_memory(proc.pid)
        vram_delta = (
            after["gpu"]["used_mib"] - before["gpu"]["used_mib"]
            if before["gpu"] and after["gpu"] else None
        )
        print(f"\n加载成功，用时 {load_s:.1f}s")
        print(f"  显存已用 {after['gpu']['used_mib']} MiB "
              f"(基线 {before['gpu']['used_mib']} → 净增 {vram_delta} MiB)"
              if after["gpu"] else "  显存: 不可用")
        print(f"  内存可用 {after['mem']['avail_gib']} GiB "
              f"(加载前 {before['mem']['avail_gib']} GiB)")
        if pmem:
            print(f"  进程工作集 {pmem['working_set_gib']} GiB "
                  f"(峰值 {pmem['peak_working_set_gib']} GiB, 提交 {pmem['pagefile_gib']} GiB)")
        result.update({
            "ok": True, "load_seconds": round(load_s, 1),
            "vram_used_mib": after["gpu"]["used_mib"] if after["gpu"] else None,
            "vram_delta_mib": vram_delta,
            "ram_avail_gib": after["mem"]["avail_gib"],
            **({"working_set_gib": pmem["working_set_gib"]} if pmem else {}),
        })

        # --- decode ---
        print(f"\n[decode] 生成 {args.decode_tokens} token ...")
        try:
            resp = run_completion(base, "Count from 1 to 200, one number per line.", args.decode_tokens)
            t = resp.get("timings", {})
            dec = t.get("predicted_per_second")
            print(f"  decode {dec:.2f} tok/s  "
                  f"({t.get('predicted_n')} tok / {t.get('predicted_ms', 0)/1000:.2f}s)")
            result["decode_tps"] = round(dec, 2) if dec else None
            result["decode_n"] = t.get("predicted_n")
            result["sample"] = (resp.get("content") or "")[:120].replace("\n", " ")
        except Exception as exc:  # noqa: BLE001
            print(f"  decode 失败: {exc}")
            result["decode_error"] = str(exc)

        # --- prefill ---
        if not args.skip_prefill:
            prompt = make_prefill_prompt(args.prefill_tokens)
            print(f"\n[prefill] 约 {args.prefill_tokens} token 的 prompt ...")
            try:
                resp = run_completion(base, prompt, 1)
                t = resp.get("timings", {})
                pre = t.get("prompt_per_second")
                print(f"  prefill {pre:.1f} tok/s  "
                      f"({t.get('prompt_n')} tok / {t.get('prompt_ms', 0)/1000:.2f}s)")
                if t.get("predicted_per_second"):
                    print(f"  首字延迟 {t.get('predicted_ms', 0)/1000:.2f}s")
                result["prefill_tps"] = round(pre, 1) if pre else None
                result["prefill_n"] = t.get("prompt_n")
                result["prefill_ms"] = t.get("prompt_ms")
            except Exception as exc:  # noqa: BLE001
                print(f"  prefill 失败: {exc}")
                result["prefill_error"] = str(exc)

        # 高水位复核
        peak = {"mem": system_memory(), "gpu": gpu_used_mib()}
        pmem2 = process_memory(proc.pid)
        print(f"\n压测后: 显存已用 {peak['gpu']['used_mib']} MiB | "
              f"内存可用 {peak['mem']['avail_gib']} GiB")
        if pmem2:
            print(f"  进程工作集 {pmem2['working_set_gib']} GiB "
                  f"(峰值 {pmem2['peak_working_set_gib']} GiB)")
            result["working_set_loaded_gib"] = pmem2["working_set_gib"]
            result["peak_working_set_gib"] = pmem2["peak_working_set_gib"]
        result["vram_used_loaded_mib"] = peak["gpu"]["used_mib"] if peak["gpu"] else None
        result["ram_avail_loaded_gib"] = peak["mem"]["avail_gib"]
        return 0

    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log_handle.close()
        time.sleep(2)
        freed = gpu_used_mib()
        print(f"\n已关闭后端。显存已用 {freed['used_mib']} MiB" if freed else "\n已关闭后端。")
        print(f"日志: {log_path}")
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"结果: {args.json_out}")


if __name__ == "__main__":
    raise SystemExit(main())
