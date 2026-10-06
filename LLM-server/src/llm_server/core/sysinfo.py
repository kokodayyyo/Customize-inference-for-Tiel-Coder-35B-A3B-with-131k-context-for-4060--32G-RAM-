"""系统与进程运行时信息（不依赖第三方库、不依赖 WMI）。

网页控制台要展示的"各项数据"都从这里取：

* **GPU** —— 显存占用 / 利用率 / 温度 / 功耗（nvidia-smi）
* **系统内存** —— 专家权重在内存里，内存不够就会退化成读盘
* **系统 CPU** —— 本模型 prefill 是 CPU 侧专家 GEMM 的瓶颈，必须能看见
* **后端进程** —— 工作集、提交、CPU 占用、线程数（`llama-server`）

用 ctypes 直接调 Win32，避免 WMI 在受限环境下的权限问题
（早期版本用 ``Get-CimInstance`` 报 WinError 5，才改成这条路）。

CPU 占用率需要两次采样，所以 ``ProcessSampler`` 保存上一次的
``GetProcessTimes`` 结果，由调用方按固定间隔反复调用。
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass

GB = 1024**3
MIB = 1024**2

if os.name == "nt":
    _kernel32 = ctypes.windll.kernel32
    _psapi = ctypes.windll.psapi
else:  # 仅用于让静态检查通过；本项目面向 Windows
    _kernel32 = None
    _psapi = None


# ---------------------------------------------------------------------------
# 内存
# ---------------------------------------------------------------------------

class _MEMORYSTATUSEX(ctypes.Structure):
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


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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
    """物理内存与提交量（GiB）。"""
    if _kernel32 is None:
        return {}
    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        return {}
    total = stat.ullTotalPhys / GB
    avail = stat.ullAvailPhys / GB
    return {
        "total_gib": round(total, 2),
        "avail_gib": round(avail, 2),
        "used_gib": round(total - avail, 2),
        "percent": round((total - avail) / total * 100, 1) if total else 0.0,
        "commit_total_gib": round(stat.ullTotalPageFile / GB, 2),
        "commit_avail_gib": round(stat.ullAvailPageFile / GB, 2),
    }


def disks(letters: str = "CDEFG") -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    for letter in letters:
        root = f"{letter}:\\"
        if not os.path.exists(root):
            continue
        try:
            usage = shutil.disk_usage(root)
        except OSError:
            continue
        out.append({
            "drive": letter,
            "total_gib": round(usage.total / GB, 1),
            "free_gib": round(usage.free / GB, 1),
            "percent": round(usage.used / usage.total * 100, 1) if usage.total else 0.0,
        })
    return out


# ---------------------------------------------------------------------------
# 系统 CPU（用于判断是否 CPU 瓶颈）
# ---------------------------------------------------------------------------

class _FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", wintypes.DWORD), ("dwHighDateTime", wintypes.DWORD)]


def _ft_to_int(ft: _FILETIME) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


class CpuSampler:
    """系统整体 CPU 占用率；需要两次采样。"""

    def __init__(self) -> None:
        self._last: tuple[float, int, int, int] | None = None  # (t, idle, kernel, user)

    def sample(self) -> float | None:
        if _kernel32 is None:
            return None
        idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
        if not _kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        now = time.perf_counter()
        cur = (now, _ft_to_int(idle), _ft_to_int(kernel), _ft_to_int(user))
        prev, self._last = self._last, cur
        if prev is None:
            return None
        dt = cur[0] - prev[0]
        if dt <= 0:
            return None
        # kernel 时间含 idle，所以"忙"= kernel + user - idle
        busy = (cur[2] - prev[2]) + (cur[3] - prev[3]) - (cur[1] - prev[1])
        total = (cur[2] - prev[2]) + (cur[3] - prev[3])
        if total <= 0:
            return None
        return round(max(0.0, min(100.0, busy / total * 100)), 1)

    @property
    def cpu_count(self) -> int:
        return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# 进程
# ---------------------------------------------------------------------------

class ProcessSampler:
    """某个进程的内存与 CPU 占用（CPU 需要两次采样）。"""

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def __init__(self) -> None:
        self._last: dict[int, tuple[float, int]] = {}

    def sample(self, pid: int | None) -> dict[str, float]:
        if not pid or _kernel32 is None:
            return {}
        handle = _kernel32.OpenProcess(self.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return {}
        try:
            out: dict[str, float] = {}

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            if _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                out["working_set_gib"] = round(counters.WorkingSetSize / GB, 2)
                out["peak_working_set_gib"] = round(counters.PeakWorkingSetSize / GB, 2)
                out["commit_gib"] = round(counters.PagefileUsage / GB, 2)

            created, exited, kernel, user = (
                _FILETIME(), _FILETIME(), _FILETIME(), _FILETIME()
            )
            if _kernel32.GetProcessTimes(
                handle, ctypes.byref(created), ctypes.byref(exited),
                ctypes.byref(kernel), ctypes.byref(user),
            ):
                cpu_100ns = _ft_to_int(kernel) + _ft_to_int(user)
                now = time.perf_counter()
                prev = self._last.get(pid)
                self._last[pid] = (now, cpu_100ns)
                # 换模型会换 pid，旧基线留着没用；只保留最近几次，防止无上限增长。
                if len(self._last) > 8:
                    for stale in list(self._last)[:-8]:
                        self._last.pop(stale, None)
                out["cpu_seconds"] = round(cpu_100ns / 1e7, 1)
                if prev is not None and now > prev[0]:
                    delta = (cpu_100ns - prev[1]) / 1e7
                    cores = os.cpu_count() or 1
                    # 单核百分比：占满 1 个核 = 100%
                    out["cpu_percent"] = round(
                        max(0.0, delta / (now - prev[0]) * 100), 1
                    )
                    out["cpu_percent_of_total"] = round(
                        max(0.0, delta / (now - prev[0]) / cores * 100), 1
                    )
            return out
        finally:
            _kernel32.CloseHandle(handle)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    for cand in (
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvidia-smi.exe"),
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ):
        if os.path.isfile(cand):
            return cand
    return None


# (字段名, nvidia-smi 列名, 是否数值)
_GPU_FIELDS = (
    ("index", "index", False),
    ("name", "name", False),
    ("total_mib", "memory.total", True),
    ("used_mib", "memory.used", True),
    ("free_mib", "memory.free", True),
    ("util_percent", "utilization.gpu", True),
    ("mem_util_percent", "utilization.memory", True),
    ("temp_c", "temperature.gpu", True),
    ("power_w", "power.draw", True),
    ("power_limit_w", "power.limit", True),
)


def gpu_detail() -> list[dict]:
    """显卡实时状态（显存 / 利用率 / 温度 / 功耗）。查不到返回空列表。"""
    exe = _nvidia_smi()
    if not exe:
        return []
    query = ",".join(f[1] for f in _GPU_FIELDS)
    try:
        proc = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    gpus: list[dict] = []
    for line in proc.stdout.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < len(_GPU_FIELDS):
            continue
        item: dict = {}
        for (key, _col, numeric), raw in zip(_GPU_FIELDS, cells):
            if numeric:
                try:
                    item[key] = round(float(raw), 1)
                except ValueError:
                    item[key] = None
            else:
                item[key] = raw
        if item.get("total_mib"):
            item["total_gib"] = round(item["total_mib"] / 1024, 2)
            item["used_gib"] = round((item.get("used_mib") or 0) / 1024, 2)
            item["free_gib"] = round((item.get("free_mib") or 0) / 1024, 2)
            item["used_percent"] = round(
                (item.get("used_mib") or 0) / item["total_mib"] * 100, 1
            )
        gpus.append(item)
    return gpus


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

@dataclass
class Samplers:
    """把需要两次采样的采样器打包，供上层长期持有。"""

    cpu: CpuSampler
    process: ProcessSampler

    @classmethod
    def create(cls) -> "Samplers":
        return cls(cpu=CpuSampler(), process=ProcessSampler())


def snapshot(samplers: Samplers, pid: int | None = None) -> dict:
    """一次性拿到界面需要的全部系统数据。"""
    return {
        "memory": system_memory(),
        "cpu_percent": samplers.cpu.sample(),
        "cpu_count": samplers.cpu.cpu_count,
        "gpus": gpu_detail(),
        "process": samplers.process.sample(pid),
        "ts": time.time(),
    }
