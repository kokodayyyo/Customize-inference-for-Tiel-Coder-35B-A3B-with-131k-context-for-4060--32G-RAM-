"""ModelManager 停止/切换的状态机测试（不加载模型，秒级完成）。

回归的 bug：在控制台点「停止服务」后 ``active_path`` 没有清空，``server.profile``
也留着旧值。前端据此把已经停掉的模型继续标成"使用中"（卡片高亮、状态栏还是旧
参数），并且把它的「启动此模型」按钮置灰 —— 用户点不动，看起来就像"启动失败"。
切换模型失败时同样会留下这个死状态。

这里用假的 backend / registry 驱动 ``ModelManager``，断言状态机不残留。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_server.api.admin import (  # noqa: E402
    STATE_ERROR,
    STATE_IDLE,
    STATE_RUNNING,
    ModelManager,
)
from llm_server.config import ServerConfig  # noqa: E402

# validate/warnings 会去碰真实模型文件，这里不关心
ServerConfig.validate = lambda self: []  # type: ignore[assignment]
ServerConfig.warnings = lambda self: []  # type: ignore[assignment]

PASSED = 0
FAILED: list[str] = []


def check(name: str, actual, expected) -> None:
    global PASSED
    if actual == expected:
        PASSED += 1
        print(f"  [通过] {name}: {actual!r}")
    else:
        FAILED.append(name)
        print(f"  [失败] {name}: 实际 {actual!r}，期望 {expected!r}")


class FakeProfile:
    context_size = 204800
    ubatch_size = 2048
    kv_type = "q8_0"
    kv_location = "显存"
    moe_location = "内存"
    load_mode_label = "none"
    note = ""


class FakeServer:
    """最小 backend 桩：只记录 start/stop，不碰进程/GPU。"""

    def __init__(self, fail_start: bool = False) -> None:
        self.process = None
        self.profile = None
        self.log_path = None
        self.calls: list[str] = []
        self.fail_start = fail_start

    @property
    def is_running(self):
        return self.process is not None

    @property
    def pid(self):
        return 123 if self.is_running else None

    def start(self):
        self.calls.append("start")
        if self.fail_start:
            raise RuntimeError("模拟加载失败")
        self.process = object()
        self.profile = FakeProfile()
        return self.profile

    def stop(self):
        self.calls.append("stop")
        self.process = None

    def command_line(self):
        return "llama-server ..."

    def log_tail(self, n=40):
        return ""

    def fetch_backend_metrics(self):
        return {}

    def fetch_backend_props(self):
        return {}

    def fetch_slots(self):
        return []


class FakeEntry:
    def __init__(self, path: str) -> None:
        self.path = path
        self.label = "model"
        self.is_projector = False


class FakeRegistry:
    load_error = ""

    def find(self, path):
        return FakeEntry(path)

    def apply_to_config(self, entry, cfg):
        cfg.model_path = str(entry.path)
        return ["model_path"]

    def scan(self):
        return []


def _make(fail_start: bool = False):
    cfg = ServerConfig()
    srv = FakeServer(fail_start=fail_start)
    mgr = ModelManager(cfg, srv, registry=FakeRegistry())
    return mgr, srv


def _running(mgr, srv, path):
    srv.start()
    mgr.state = STATE_RUNNING
    mgr.active_path = path
    mgr.started_at = time.time()


def _join(mgr):
    if mgr._thread is not None:
        mgr._thread.join(timeout=5)


def main() -> int:
    print("[1] 停止服务后不留残留状态")
    mgr, srv = _make()
    _running(mgr, srv, r"D:\models\a.gguf")
    check("停止返回 ok", mgr.stop().get("ok"), True)
    check("active_path 已清空", mgr.active_path, "")
    check("server.profile 已清空", srv.profile, None)
    check("状态回到 idle", mgr.state, STATE_IDLE)

    print()
    print("[2] 停止后仍能重新启动同一个模型（按钮不再被锁死）")
    mgr, srv = _make()
    _running(mgr, srv, r"D:\models\a.gguf")
    mgr.stop()
    srv.calls.clear()
    check("重新启动被受理", mgr.activate(r"D:\models\a.gguf").get("ok"), True)
    _join(mgr)
    check("确实执行了 start", "start" in srv.calls, True)
    check("active_path 指向该模型", mgr.active_path, r"D:\models\a.gguf")
    check("状态为 running", mgr.state, STATE_RUNNING)

    print()
    print("[3] 切换到另一个模型：先停旧、再起新")
    mgr, srv = _make()
    _running(mgr, srv, r"D:\models\a.gguf")
    srv.calls.clear()
    check("切换被受理", mgr.activate(r"D:\models\b.gguf").get("ok"), True)
    _join(mgr)
    check("调用顺序 stop -> start", srv.calls, ["stop", "start"])
    check("active_path 指向新模型", mgr.active_path, r"D:\models\b.gguf")
    check("状态为 running", mgr.state, STATE_RUNNING)

    print()
    print("[4] 切换失败时不遗留「使用中」的旧模型")
    mgr, srv = _make(fail_start=True)
    # 直接摆出"A 在运行"的状态（这个桩的 start() 会失败，不能走 _running）
    srv.process = object()
    srv.profile = FakeProfile()
    mgr.state = STATE_RUNNING
    mgr.active_path = r"D:\models\a.gguf"
    mgr.started_at = time.time()
    mgr.activate(r"D:\models\b.gguf")
    _join(mgr)
    check("失败后状态为 error", mgr.state, STATE_ERROR)
    check("active_path 为空（旧模型已停，不应仍是 active）", mgr.active_path, "")
    check("server.profile 为空", srv.profile, None)

    print()
    print("=" * 60)
    print(f"通过 {PASSED} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败: {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
