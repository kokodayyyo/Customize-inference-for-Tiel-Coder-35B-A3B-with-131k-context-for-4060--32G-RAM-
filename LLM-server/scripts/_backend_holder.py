"""辅助脚本：启动后端并等待被强杀（用于验证 Job Object 兜底）。

由 test_jobobject.py 作为子进程拉起。启动 llama-server 后不再做任何清理，
这样父进程被 /F 强杀时就能验证"子进程是否被系统一并结束"。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_server.config import ServerConfig  # noqa: E402
from llm_server.core import make_backend  # noqa: E402


def main() -> int:
    cfg = ServerConfig.load()
    cfg.context_size = 32768
    cfg.backend_port = 18090

    server = make_backend(cfg)
    profile = server.start()
    # 关键：不注册任何退出清理，模拟"父进程被强杀"
    print(f"READY pid={server.pid} ctx={profile.context_size}", flush=True)

    while True:
        time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
