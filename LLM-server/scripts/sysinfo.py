"""读取系统内存、页面文件与磁盘空间（不依赖 WMI，避免权限问题）。

用途：判断 MoE 模型"专家权重放内存"这条路是否可行。决定因素是内存总量与
可用量，而不是显存——35B 的专家权重必须完整驻留或能高效换入。
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
from ctypes import wintypes


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


GB = 1024**3


def memory() -> dict[str, float]:
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx 失败")
    return {
        "total_gib": stat.ullTotalPhys / GB,
        "avail_gib": stat.ullAvailPhys / GB,
        "commit_limit_gib": stat.ullTotalPageFile / GB,
        "commit_avail_gib": stat.ullAvailPageFile / GB,
        "load_percent": float(stat.dwMemoryLoad),
    }


def disks() -> list[tuple[str, float, float]]:
    out = []
    for letter in "CDEFG":
        root = f"{letter}:\\"
        if os.path.exists(root):
            try:
                usage = shutil.disk_usage(root)
                out.append((letter, usage.total / GB, usage.free / GB))
            except OSError:
                continue
    return out


def main() -> int:
    mem = memory()
    print("=== 内存 ===")
    print(f"  物理内存总量    : {mem['total_gib']:>6.1f} GiB")
    print(f"  物理内存可用    : {mem['avail_gib']:>6.1f} GiB")
    print(f"  已用比例        : {mem['load_percent']:>6.1f} %")
    print(f"  提交上限(含页文件): {mem['commit_limit_gib']:>5.1f} GiB")
    print(f"  提交可用        : {mem['commit_avail_gib']:>6.1f} GiB")

    print()
    print("=== 磁盘 ===")
    for letter, total, free in disks():
        print(f"  {letter}: 总 {total:>7.1f} GiB  可用 {free:>7.1f} GiB")

    # 专家权重放内存的可行性判断
    print()
    print("=== MoE 专家权重放内存的可行性 ===")
    avail = mem["avail_gib"]
    print(f"  可用于驻留专家权重的内存约: {avail:.1f} GiB")
    print("  （还需为系统、页缓存、其它程序留 2-3 GiB 余量）")
    print()
    print("  注意：具体需要多少内存取决于模型的真实张量构成，不是按参数量估算的。")
    print("  当前模型（Tile-35B-A3B IQ4_XS）实测：")
    print("    专家 *_exps      14.12 GiB  -> 必须放内存")
    print("    其余权重          2.38 GiB  -> 放显存")
    print("    MTP/nextn 死重    0.36 GiB  -> llama.cpp 忽略")
    print()
    print(f"  按此口径判断: {'可以' if avail >= 14.12 + 2.5 else '偏紧'} "
          f"（需要 14.12 GiB + 约 2.5 GiB 余量）")
    print()
    print("  精确预算请运行: python main.py doctor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
