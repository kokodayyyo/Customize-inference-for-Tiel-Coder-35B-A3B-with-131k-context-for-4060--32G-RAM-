"""语法检查 + 包导入冒烟测试（不写入字节码缓存）。

为什么需要导入检查：``compile()`` 只验证语法，**查不出相对导入写错层级**
这类错误（例如 ``api/admin.py`` 里把 ``from ..config`` 写成 ``from .config``）。
这种错误在启动服务时才会炸，代价高得多。实测踩过。

用法::

    python scripts/check_syntax.py          # 语法 + 导入
    python scripts/check_syntax.py --syntax # 只做语法检查
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SKIP_DIRS = {".git", "__pycache__", "runtime", ".venv", ".idea", ".piptmp"}

# 导入检查要覆盖的包内模块（新增模块记得加进来）
IMPORT_TARGETS = (
    "ornith_server",
    "ornith_server.config",
    "ornith_server.net",
    "ornith_server.models_registry",
    "ornith_server.bench",
    "ornith_server.core",
    "ornith_server.core.backend",
    "ornith_server.core.server",
    "ornith_server.core.gguf",
    "ornith_server.core.jobobject",
    "ornith_server.api",
    "ornith_server.api.gateway",
    "ornith_server.api.admin",
    "ornith_server.api.middleware",
    "ornith_server.api.metrics",
)


def check_syntax() -> int:
    failed = 0
    checked = 0
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        checked += 1
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            failed += 1
            print(f"语法错误 {path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
        else:
            print(f"OK  {path.relative_to(ROOT)}")
    print(f"\n语法检查 {checked} 个文件，失败 {failed} 个")
    return failed


def check_imports() -> int:
    """把包内模块逐个导入一遍，抓相对导入写错这类问题。"""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    print("\n=== 导入冒烟测试 ===")
    failed = 0
    for name in IMPORT_TARGETS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - 任何导入失败都要报出来
            failed += 1
            print(f"导入失败 {name}: {type(exc).__name__}: {exc}")
            if "--trace" in sys.argv:
                traceback.print_exc()
        else:
            print(f"OK  import {name}")

    # 附带检查网页控制台的静态文件是否在位
    try:
        from ornith_server.api.admin import WEB_DIR

        index = WEB_DIR / "index.html"
        if index.is_file():
            print(f"OK  控制台页面 {index.relative_to(ROOT)} "
                  f"({index.stat().st_size} B)")
        else:
            failed += 1
            print(f"缺少控制台页面: {index}")
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"无法定位控制台页面: {exc}")

    print(f"\n导入检查 {len(IMPORT_TARGETS) + 1} 项，失败 {failed} 项")
    return failed


def main() -> int:
    failed = check_syntax()
    if "--syntax" not in sys.argv:
        failed += check_imports()
    print("\n" + ("全部通过。" if not failed else f"共 {failed} 项失败。"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
