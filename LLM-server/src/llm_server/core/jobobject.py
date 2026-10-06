"""把子进程绑定到 Job Object，保证父进程被强杀时子进程一并结束。

为什么需要：Windows 上父进程终止**不会**自动结束子进程。如果网关进程被
``taskkill /F``、任务管理器结束或崩溃，``llama-server`` 会变成孤儿进程继续
占着显存（8GB 卡上等于锁死服务）。正常路径由 ``LlamaBackendServer.stop()``
清理，这里只是兜底：给子进程挂一个设置了
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` 的 Job Object，作业句柄随父进程
关闭时，系统会强制结束作业内的所有进程。

非 Windows 平台直接跳过（POSIX 下用进程组 + 信号即可）。
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
from ctypes import wintypes

log = logging.getLogger("llm.jobobject")

IS_WINDOWS = os.name == "nt"

# --- Win32 常量 -------------------------------------------------------------
_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_JOB_OBJECT_TERMINATE = 0x0008


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class ChildJob:
    """一个绑定子进程的 Job Object 包装。

    用法::

        job = ChildJob()
        job.assign(proc)      # 之后父进程被杀，proc 也会被系统结束
        ...
        job.close()           # 正常退出时释放（此时子进程已自行结束）
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.handle: int | None = None
        self.assigned_pid: int | None = None
        self.error: str = ""
        if IS_WINDOWS:
            self._create()

    # ------------------------------------------------------------------
    def _create(self) -> None:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                self.error = f"CreateJobObject 失败 (Win32 {ctypes.get_last_error()})"
                return

            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

            kernel32.SetInformationJobObject.restype = wintypes.BOOL
            kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
            ]
            ok = kernel32.SetInformationJobObject(
                handle,
                _JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                self.error = f"SetInformationJobObject 失败 (Win32 {ctypes.get_last_error()})"
                kernel32.CloseHandle(handle)
                return

            self.handle = int(handle)
        except (OSError, AttributeError) as exc:
            self.error = f"无法创建 Job Object: {exc}"

    # ------------------------------------------------------------------
    def assign(self, proc: subprocess.Popen) -> bool:
        """把进程加入作业。返回是否成功。"""
        if not IS_WINDOWS or self.handle is None:
            return False
        if proc.poll() is not None:
            return False
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            proc_handle = kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, proc.pid
            )
            if not proc_handle:
                self.error = f"OpenProcess 失败 (Win32 {ctypes.get_last_error()})"
                return False
            try:
                kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
                kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
                ok = kernel32.AssignProcessToJobObject(
                    wintypes.HANDLE(self.handle), proc_handle
                )
                if not ok:
                    self.error = (
                        f"AssignProcessToJobObject 失败 (Win32 {ctypes.get_last_error()})"
                    )
                    return False
                self.assigned_pid = proc.pid
                return True
            finally:
                kernel32.CloseHandle(proc_handle)
        except (OSError, AttributeError) as exc:
            self.error = f"加入作业失败: {exc}"
            return False

    # ------------------------------------------------------------------
    def terminate(self) -> None:
        """立即结束作业内所有进程（用于兜底清理）。"""
        if not IS_WINDOWS or self.handle is None:
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateJobObject(wintypes.HANDLE(self.handle), 1)
        except (OSError, AttributeError):
            pass

    # ------------------------------------------------------------------
    def close(self) -> None:
        """关闭作业句柄。

        若句柄关闭时作业内仍有进程，且设置了 KILL_ON_JOB_CLOSE，
        这些进程会被系统结束——这正是我们要的兜底行为。
        """
        if self.handle is None:
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CloseHandle.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle(wintypes.HANDLE(self.handle))
        except (OSError, AttributeError):
            pass
        finally:
            self.handle = None


def describe_support() -> str:
    """返回 Job Object 兜底机制是否可用的说明。"""
    if not IS_WINDOWS:
        return "当前平台非 Windows，跳过 Job Object 兜底"
    job = ChildJob("probe")
    if job.handle is not None:
        job.close()
        return "已启用：父进程被强杀时，llama-server 会被系统一并结束"
    return f"不可用（{job.error or '未知原因'}），请用 stop_server.bat 手动清理"


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print(describe_support())
