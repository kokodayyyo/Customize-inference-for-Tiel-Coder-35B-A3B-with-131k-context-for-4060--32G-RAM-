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
    def __init__(self, path: str, mmproj: str = "", default_vision: bool = False) -> None:
        self.path = path
        self.label = "model"
        self.is_projector = False
        self.mmproj_path = mmproj
        self._settings = {"mmproj_path": mmproj} if default_vision else {}

    @property
    def vision_supported(self):
        return bool(self.mmproj_path)

    @property
    def vision_default(self):
        return bool(str(self._settings.get("mmproj_path") or "").strip())

    @property
    def settings(self):
        return dict(self._settings)


class FakeRegistry:
    load_error = ""

    def __init__(self, entries=None):
        # path -> (mmproj, default_vision)
        self.entries = entries or {}

    def find(self, path):
        mmproj, default = self.entries.get(str(path), ("", False))
        return FakeEntry(path, mmproj=mmproj, default_vision=default)

    def apply_to_config(self, entry, cfg):
        cfg.model_path = str(entry.path)
        return ["model_path"]

    def scan(self):
        return []


def _make(fail_start: bool = False, entries=None):
    cfg = ServerConfig()
    srv = FakeServer(fail_start=fail_start)
    mgr = ModelManager(cfg, srv, registry=FakeRegistry(entries))
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
    print("[5] 视觉开关：只对检测到 mmproj 的模型生效")
    entries = {
        r"D:\models\vision.gguf": (r"D:\models\mmproj-Q8_0.gguf", False),
        r"D:\models\plain.gguf": ("", False),
    }
    mgr, srv = _make(entries=entries)
    check("无视觉组件时开启被拒",
          mgr.activate(r"D:\models\plain.gguf", vision=True).get("ok"), False)
    check("有视觉组件时受理",
          mgr.activate(r"D:\models\vision.gguf", vision=True).get("ok"), True)
    _join(mgr)
    check("配置里写入了 mmproj", mgr.cfg.mmproj_path, r"D:\models\mmproj-Q8_0.gguf")
    check("状态为 running", mgr.state, STATE_RUNNING)

    mgr.activate(r"D:\models\plain.gguf", vision=False)
    _join(mgr)
    check("切到无视觉模型后 mmproj 被清空（不残留旧眼睛）", mgr.cfg.mmproj_path, "")

    mgr.activate(r"D:\models\vision.gguf", vision=False)
    _join(mgr)
    check("有视觉组件但显式关闭则不加载", mgr.cfg.mmproj_path, "")

    print()
    print("[6] 同目录 mmproj 关联（_attach_vision）")
    from pathlib import Path as _P

    from llm_server.models_registry import (  # noqa: PLC0415
        ModelEntry,
        ModelProfile,
        _attach_vision,
    )

    def mk(name: str, proj: bool = False, settings=None):
        e = ModelEntry(path=_P("D:/m") / name, name=name, stem=_P(name).stem,
                       size_gib=1.0, is_projector=proj)
        if settings is not None:
            e.profile = ModelProfile(settings=settings)
        return e

    model, other, projector = mk("model.gguf"), mk("other.gguf"), mk("mmproj-Q8_0.gguf", proj=True)
    _attach_vision([model, other, projector])
    check("同目录投影器被关联", model.mmproj_path, "D:\\m\\mmproj-Q8_0.gguf")
    check("model 标记为支持视觉", model.vision_supported, True)
    check("没有指定时默认不开启视觉", model.vision_default, False)

    pinned = mk("model2.gguf", settings={"mmproj_path": "D:/custom/eye.gguf"})
    _attach_vision([pinned, projector])
    check("profile 显式 mmproj 优先", pinned.mmproj_path, "D:/custom/eye.gguf")
    check("显式指定即默认开启", pinned.vision_default, True)

    lone = mk("lone.gguf")
    _attach_vision([lone])
    check("无投影器的模型不支持视觉", lone.vision_supported, False)

    print()
    print("=" * 60)
    print(f"通过 {PASSED} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败: {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
