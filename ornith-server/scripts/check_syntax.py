"""对项目内所有 Python 文件做语法检查（不写入字节码缓存）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "__pycache__", "runtime", ".venv"}

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

print(f"\n检查 {checked} 个文件，失败 {failed} 个")
sys.exit(1 if failed else 0)
