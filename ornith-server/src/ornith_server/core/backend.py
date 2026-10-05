"""llama.cpp 后端定位与探测。

本模块负责在本机自动找到可用的 llama.cpp 运行时（llama-server.exe）及其
CUDA vendor DLL 目录，避免重新编译。优先使用 LM Studio 自带的 CUDA 构建。

设计要点：
- LM Studio 把引擎放在 ``~/.lmstudio/extensions/backends/`` 下，CUDA 运行时
  DLL（cublas/cudart）单独放在 ``backends/vendor/win-llama-cuda*-vendor-v2/``，
  必须一并加进 PATH，否则 llama-server.exe 会以 0xC0000135 (DLL not found) 退出。
- 版本目录名形如 ``llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.46.0``，
  末段是引擎版本，按版本号倒序挑选最新的 CUDA 构建。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

LLAMA_SERVER_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"
_CUDA_VENDOR_PREFIX = "win-llama-cuda"
_VENDOR_DLL_NEEDLES = ("cudart64_", "cublas64_", "cublasLt64_")

_VERSION_RE = re.compile(r"-(\d+\.\d+\.\d+)$")


@dataclass(frozen=True)
class LlamaBackend:
    """一套可用的 llama.cpp 运行时。"""

    exe: Path
    home: Path
    vendor_dir: Path | None = None
    version: str = "unknown"
    gpu_framework: str = "cpu"
    gpu_targets: tuple[str, ...] = ()
    source: str = "manual"

    # -- 环境与命令行 ------------------------------------------------------
    def build_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        """返回带好 DLL 搜索路径的环境变量。

        vendor 目录（CUDA 运行时）必须排在引擎目录之前，因为 cudart/cublas 是
        ggml-cuda.dll 的依赖，Windows 加载器按 PATH 顺序解析。
        """
        env = dict(base_env or os.environ)
        parts = [str(p) for p in (self.vendor_dir, self.home) if p]
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
        # 让 CUDA 后端在显存不足时不要把整卡占满（8GB 笔记本卡很敏感）
        env.setdefault("GGML_CUDA_NO_PEER_COPY", "1")
        return env

    def describe(self) -> str:
        gpu = f"{self.gpu_framework}" + (
            f" (sm {','.join(self.gpu_targets)})" if self.gpu_targets else ""
        )
        return (
            f"llama.cpp {self.version} | 后端 {gpu} | 来源 {self.source}\n"
            f"  引擎目录: {self.home}\n"
            f"  运行时库: {self.vendor_dir or '（随引擎目录）'}"
        )


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int
    driver: str = ""

    @property
    def total_gib(self) -> float:
        return round(self.total_mib / 1024, 2)

    @property
    def free_gib(self) -> float:
        return round(self.free_mib / 1024, 2)


# ---------------------------------------------------------------------------
# 候选路径
# ---------------------------------------------------------------------------


def _candidate_roots() -> list[Path]:
    """返回可能存放 llama.cpp 运行时的根目录（不递归太深）。

    顺序即优先级：**项目自带的 runtime 目录排第一**，这样即使卸载了
    LM Studio 服务也能照常启动。环境变量仍然可以强制指定。
    """
    from ..config import BUNDLED_BACKENDS_DIR

    home = Path.home()
    roots: list[Path] = [
        BUNDLED_BACKENDS_DIR,  # <项目>/runtime/llama.cpp/backends（自带，最优先）
        home / ".lmstudio" / "extensions" / "backends",
        Path("D:/LMstudio"),
        Path("C:/Program Files/LM Studio"),
        Path("D:/llama.cpp"),
        Path("D:/tools/llama.cpp"),
        Path("C:/llama.cpp"),
    ]
    # 环境变量显式指定优先
    for var in ("ORNITH_LLAMA_DIR", "LLAMA_CPP_DIR"):
        if raw := os.environ.get(var):
            roots.insert(0, Path(raw))
    return roots


def _read_manifest(engine_dir: Path) -> dict:
    manifest = engine_dir / "backend-manifest.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}


def _find_vendor_dir(engine_dir: Path) -> Path | None:
    """为引擎找到配套的 vendor DLL 目录。

    必须按后端类型配对：CUDA 引擎配 CUDA vendor，Vulkan 引擎配 Vulkan vendor。
    配错会导致加载错误的运行时库。
    """
    backends = engine_dir.parent
    vendor_root = backends / "vendor"
    manifest = _read_manifest(engine_dir)
    wanted = [
        name
        for name in manifest.get("vendor_lib_package_names", [])
        if isinstance(name, str)
    ]
    for name in wanted:
        cand = vendor_root / name
        if cand.is_dir():
            return cand

    if not vendor_root.is_dir():
        return None

    # 从引擎目录名推断需要哪种运行时
    name = engine_dir.name.lower()
    if "nvidia" in name or "cuda" in name:
        families = ("cuda12", "cuda11")
    elif "vulkan" in name:
        families = ("vulkan",)
    elif "rocm" in name or "hip" in name:
        families = ("rocm", "hip")
    else:
        return None  # CPU 引擎不需要额外运行时

    dirs = [d for d in vendor_root.iterdir() if d.is_dir()]
    for family in families:
        for d in sorted(dirs, key=lambda p: p.name, reverse=True):
            if family in d.name.lower():
                return d
    return None


def _iter_engine_dirs() -> Iterable[Path]:
    for root in _candidate_roots():
        if not root.is_dir():
            continue
        # 直接就是引擎目录
        if (root / LLAMA_SERVER_EXE).is_file():
            yield root
            continue
        try:
            children = [d for d in root.iterdir() if d.is_dir()]
        except OSError:
            continue
        for child in children:
            if (child / LLAMA_SERVER_EXE).is_file():
                yield child
            else:
                # LM Studio 布局: backends/<engine>/llama-server.exe
                nested = child / "extensions" / "backends"
                base = nested if nested.is_dir() else child
                try:
                    for grand in base.iterdir():
                        if grand.is_dir() and (grand / LLAMA_SERVER_EXE).is_file():
                            yield grand
                except OSError:
                    continue


def _backend_family(engine_dir: Path) -> str:
    """判定引擎的加速后端类型：cuda / vulkan / metal / rocm / cpu。"""
    manifest = _read_manifest(engine_dir)
    gpu = manifest.get("gpu") or {}
    text = f"{manifest.get('name', '')} {gpu.get('framework', '')} {engine_dir.name}".lower()
    for family, needles in (
        ("cuda", ("cuda", "nvidia")),
        ("vulkan", ("vulkan",)),
        ("metal", ("metal",)),
        ("rocm", ("rocm", "hip")),
    ):
        if any(n in text for n in needles):
            return family
    return "cpu"


def _cuda_targets(engine_dir: Path) -> tuple[str, ...]:
    manifest = _read_manifest(engine_dir)
    gpu = manifest.get("gpu") or {}
    return tuple(str(t) for t in gpu.get("targets", []))


def _preferred_families() -> tuple[str, ...]:
    """根据本机硬件决定后端优先级。

    有 NVIDIA 显卡时 CUDA 明显快于 Vulkan，CPU 则慢一个数量级，
    因此把不匹配的方案排到后面，而不是简单按版本号排序。
    """
    if query_gpus():
        return ("cuda", "vulkan", "cpu")
    return ("cpu", "vulkan")


def _rank_backend(engine_dir: Path) -> tuple:
    """排序键：后端类型匹配优先，其次版本号。"""
    family = _backend_family(engine_dir)
    families = _preferred_families()
    try:
        family_rank = -families.index(family)
    except ValueError:
        family_rank = -len(families)

    m = _VERSION_RE.search(engine_dir.name)
    version = m.group(1) if m else "0.0.0"
    try:
        vkey = tuple(int(x) for x in version.split("."))
    except ValueError:
        vkey = (0, 0, 0)
    return (family_rank, vkey, version)


def discover_backends() -> list[LlamaBackend]:
    """按推荐程度返回所有找到的 llama.cpp 运行时。"""
    found: dict[Path, LlamaBackend] = {}
    for engine_dir in _iter_engine_dirs():
        engine_dir = engine_dir.resolve()
        if engine_dir in found:
            continue
        manifest = _read_manifest(engine_dir)
        gpu = manifest.get("gpu") or {}
        m = _VERSION_RE.search(engine_dir.name)
        found[engine_dir] = LlamaBackend(
            exe=engine_dir / LLAMA_SERVER_EXE,
            home=engine_dir,
            vendor_dir=_find_vendor_dir(engine_dir),
            version=(manifest.get("version") or (m.group(1) if m else "unknown")),
            gpu_framework=_backend_family(engine_dir),
            gpu_targets=_cuda_targets(engine_dir),
            source="LM Studio 扩展" if ".lmstudio" in str(engine_dir) else "本地目录",
        )
    backends = sorted(found.values(), key=lambda b: _rank_backend(b.home), reverse=True)
    # 把 PATH 里现成的 llama-server 也纳入（手动安装 / pip 安装的场景）
    if which := shutil.which(LLAMA_SERVER_EXE.removesuffix(".exe")) or shutil.which(
        LLAMA_SERVER_EXE
    ):
        p = Path(which).resolve()
        if p.parent not in found:
            backends.insert(
                0,
                LlamaBackend(exe=p, home=p.parent, version="unknown", source="PATH"),
            )
    return backends


def resolve_backend(explicit: str | os.PathLike[str] | None = None) -> LlamaBackend:
    """确定要使用的后端；``explicit`` 可以是 exe 路径或引擎目录。"""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            exe = p / LLAMA_SERVER_EXE
            if not exe.is_file():
                raise FileNotFoundError(f"{p} 下没有找到 {LLAMA_SERVER_EXE}")
            return LlamaBackend(
                exe=exe,
                home=p,
                vendor_dir=_find_vendor_dir(p),
                version=_read_manifest(p).get("version", "unknown"),
                gpu_framework=str((_read_manifest(p).get("gpu") or {}).get("framework", "cpu")),
                source="手动指定",
            )
        if p.is_file():
            return LlamaBackend(
                exe=p,
                home=p.parent,
                vendor_dir=_find_vendor_dir(p.parent),
                source="手动指定",
            )
        raise FileNotFoundError(f"找不到指定的 llama.cpp 路径: {p}")

    backends = discover_backends()
    if not backends:
        raise FileNotFoundError(
            "本机没有找到 llama-server。请安装 LM Studio（自带 CUDA 构建），"
            "或用 --llama-dir 指定 llama.cpp 目录。"
        )
    return backends[0]


# ---------------------------------------------------------------------------
# GPU / 运行时探测
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def query_gpus() -> tuple[GpuInfo, ...]:
    """通过 nvidia-smi 查询显卡；没有 NVIDIA 显卡时返回空元组。

    结果缓存在进程内（显卡型号/驱动在运行期不会变），避免每次排序都拉起
    nvidia-smi 进程。需要刷新时调用 ``query_gpus.cache_clear()``。
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        for cand in (
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
        ):
            if Path(cand).is_file():
                exe = cand
                break
    if not exe:
        return ()

    cmd = [
        exe,
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ()
    if proc.returncode != 0:
        return ()

    gpus: list[GpuInfo] = []
    for line in proc.stdout.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 5:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(cells[0]),
                    name=cells[1],
                    total_mib=int(cells[2]),
                    free_mib=int(cells[3]),
                    driver=cells[4],
                )
            )
        except ValueError:
            continue
    return tuple(gpus)


def zero_vram() -> None:
    """尽力释放本机残留的推理进程占用的显存（仅结束 llama-server 自身）。"""
    if os.name != "nt":
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", LLAMA_SERVER_EXE],
            capture_output=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def probe_backend(backend: LlamaBackend, timeout: int = 120) -> str:
    """运行 ``llama-server --list-devices`` 验证后端可用，返回原始输出。"""
    proc = subprocess.run(
        [str(backend.exe), "--list-devices"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(backend.home),
        env=backend.build_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"后端自检失败 (rc=0x{proc.returncode & 0xFFFFFFFF:08X})："
            f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
        )
    return (proc.stdout or "") + (proc.stderr or "")


def dump_environment() -> str:
    """生成一份环境自检报告，便于排查问题。"""
    lines = [f"Python: {sys.version.splitlines()[0]}  ({sys.executable})", ""]

    lines.append("[GPU]")
    gpus = query_gpus()
    if not gpus:
        lines.append("  未检测到 NVIDIA 显卡（将使用 CPU 推理）")
    for g in gpus:
        lines.append(
            f"  #{g.index} {g.name}  {g.total_gib} GiB 总 / {g.free_gib} GiB 空闲 "
            f"(驱动 {g.driver})"
        )

    lines.append("")
    lines.append("[llama.cpp 运行时]")
    backends = discover_backends()
    if not backends:
        lines.append("  未找到 llama-server.exe")
    for i, b in enumerate(backends):
        mark = " <- 默认" if i == 0 else ""
        lines.append(f"  * {b.describe()}{mark}")

    if backends:
        lines.append("")
        lines.append("[设备枚举]")
        try:
            lines.append("  " + probe_backend(backends[0]).strip().replace("\n", "\n  "))
        except Exception as exc:  # noqa: BLE001 - 自检需要展示任何失败
            lines.append(f"  失败: {exc}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(dump_environment())
