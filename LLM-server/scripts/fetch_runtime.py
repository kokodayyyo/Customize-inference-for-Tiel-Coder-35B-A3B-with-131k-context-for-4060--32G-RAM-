"""把 llama.cpp 运行时复制进项目（仓库里不存这些二进制）。

为什么需要这个脚本：``runtime/llama.cpp/`` 约 912 MiB，其中
``cublasLt64_12.dll`` 单个就有 674 MB，远超 GitHub 的 100 MB 单文件上限，
因此被 ``.gitignore`` 排除。克隆仓库后跑一次本脚本即可恢复成可运行状态。

来源优先级：
  1. ``--from`` 指定的目录
  2. ``$LLM_LLAMA_SRC``
  3. LM Studio 的 ``~/.lmstudio/extensions/backends``
  4. ``D:/LMstudio`` 等常见安装位置

会自动挑选**最新的 CUDA12 引擎**，并配对正确的 vendor DLL 目录
（CUDA 引擎必须配 CUDA vendor，配错会加载错误的运行时库）。

用法::

    python scripts/fetch_runtime.py            # 自动寻找并复制
    python scripts/fetch_runtime.py --check    # 只检查现状，不复制
    python scripts/fetch_runtime.py --from "D:/somewhere/backends"
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = PROJECT_ROOT / "runtime" / "llama.cpp" / "backends"
EXE = "llama-server.exe"
MIB = 1024**2

_VERSION_RE = re.compile(r"-(\d+\.\d+\.\d+)$")


def candidate_roots() -> list[Path]:
    home = Path.home()
    roots = [
        home / ".lmstudio" / "extensions" / "backends",
        Path("D:/LMstudio/extensions/backends"),
        Path("C:/Program Files/LM Studio/extensions/backends"),
    ]
    for var in ("LLM_LLAMA_SRC", "LLAMA_CPP_DIR"):
        if raw := os.environ.get(var):
            roots.insert(0, Path(raw))
    return roots


def find_source() -> tuple[Path, Path | None] | None:
    """返回 (引擎目录, vendor 目录)。找不到返回 None。"""
    for root in candidate_roots():
        if not root.is_dir():
            continue
        engines = [
            d for d in root.iterdir()
            if d.is_dir() and (d / EXE).is_file() and "cuda12" in d.name.lower()
        ]
        if not engines:
            engines = [d for d in root.iterdir() if d.is_dir() and (d / EXE).is_file()]
        if not engines:
            continue
        # 版本号倒序取最新
        def vkey(p: Path) -> tuple:
            m = _VERSION_RE.search(p.name)
            return tuple(int(x) for x in m.group(1).split(".")) if m else (0, 0, 0)

        engine = sorted(engines, key=vkey, reverse=True)[0]
        vendor = None
        vendor_root = root / "vendor"
        if vendor_root.is_dir():
            names = sorted(d.name for d in vendor_root.iterdir() if d.is_dir())
            family = "cuda12" if "cuda12" in engine.name.lower() else "cuda"
            pick = [n for n in names if family in n.lower()] or names
            if pick:
                vendor = vendor_root / pick[0]
        return engine, vendor
    return None


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def describe_dest() -> int:
    if not DEST_ROOT.is_dir():
        print(f"[缺失] {DEST_ROOT} 不存在，服务无法独立启动。")
        return 1
    engines = [d for d in DEST_ROOT.iterdir() if d.is_dir() and (d / EXE).is_file()]
    if not engines:
        print(f"[缺失] {DEST_ROOT} 下没有 {EXE}。")
        return 1
    total = dir_size(DEST_ROOT)
    print(f"[就绪] {DEST_ROOT}")
    print(f"       引擎 {len(engines)} 套，合计 {total / MIB:.1f} MiB")
    for e in engines:
        print(f"       - {e.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="复制 llama.cpp 运行时进项目")
    parser.add_argument("--from", dest="src", default="", help="指定 backends 目录")
    parser.add_argument("--check", action="store_true", help="只检查现状")
    parser.add_argument("--force", action="store_true", help="目标已存在时仍覆盖")
    args = parser.parse_args()

    if args.check:
        return describe_dest()

    if DEST_ROOT.is_dir() and not args.force:
        engines = [d for d in DEST_ROOT.iterdir() if d.is_dir() and (d / EXE).is_file()]
        if engines:
            print("运行时已存在，无需复制。")
            return describe_dest()

    if args.src:
        root = Path(args.src)
        if not root.is_dir():
            print(f"指定的来源不存在: {root}", file=sys.stderr)
            return 2
        engines = [d for d in root.iterdir() if d.is_dir() and (d / EXE).is_file()]
        if not engines:
            nested = root / "extensions" / "backends"
            if nested.is_dir():
                root = nested
        # 复用 find_source 的挑选逻辑，这里直接构造
        os.environ["LLM_LLAMA_SRC"] = str(root)
        found = find_source()
    else:
        found = find_source()

    if not found:
        print("没有找到 llama.cpp 运行时。请用 --from 指定目录，或安装 LM Studio。", file=sys.stderr)
        print("需要的结构：", file=sys.stderr)
        print("  <backends>/<引擎目录>/llama-server.exe", file=sys.stderr)
        print("  <backends>/vendor/<cuda vendor 目录>/*.dll", file=sys.stderr)
        return 1

    engine, vendor = found
    print(f"来源引擎: {engine}")
    print(f"来源 vendor: {vendor or '（无，CPU 引擎不需要）'}")

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    dst_engine = DEST_ROOT / engine.name
    if dst_engine.exists() and args.force:
        shutil.rmtree(dst_engine)
    shutil.copytree(engine, dst_engine, dirs_exist_ok=True)
    print(f"已复制引擎 -> {dst_engine}  ({dir_size(dst_engine) / MIB:.1f} MiB)")

    if vendor is not None:
        dst_vendor_root = DEST_ROOT / "vendor"
        dst_vendor = dst_vendor_root / vendor.name
        if dst_vendor.exists() and args.force:
            shutil.rmtree(dst_vendor)
        dst_vendor_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(vendor, dst_vendor, dirs_exist_ok=True)
        print(f"已复制 vendor -> {dst_vendor}  ({dir_size(dst_vendor) / MIB:.1f} MiB)")

    print()
    rc = describe_dest()
    print()
    print("下一步：python main.py doctor  验证引擎能独立启动")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
