"""多模型注册表：扫描模型目录、匹配调优配置。

设计目标：**换模型不用改代码、不用记参数**。
每个模型在 ``config/models.yaml`` 里有一条 profile，记录实测标定好的参数
（上下文、ubatch、摆位、KV 量化……）。网页界面列出扫描到的模型，
点一下就把对应 profile 应用到 ``ServerConfig`` 并重启后端。

不做的事：这里只负责"找到模型 + 给出参数"，启动由
``api/admin.py`` 的 ModelManager 负责。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT, ServerConfig
from .core.gguf import ModelShape, TensorBreakdown, model_shape, tensor_breakdown

log = logging.getLogger("llm.registry")

DEFAULT_REGISTRY_FILE = PROJECT_ROOT / "config" / "models.yaml"

# 这些张量名/元数据特征说明文件是**投影器**（mmproj）而不是可独立加载的模型
_PROJECTOR_KEYS = ("clip.", "mmproj.", "v.", "vision.")


def _norm_path(value: str | os.PathLike[str]) -> str:
    """路径归一化：统一斜杠方向、去掉大小写差异，用于匹配。"""
    return str(value).replace("\\", "/").rstrip("/").lower()


@dataclass
class ModelProfile:
    """一个模型的调优参数。"""

    label: str = ""
    note: str = ""
    # 匹配条件（三选一，按优先级）
    path: str = ""
    match: str = ""
    settings: dict[str, Any] = field(default_factory=dict)
    # **实测**占用与性能。理论估算在某些模型上会偏（实测 apex@200K 比公式
    # 多占 0.76 GiB），所以有实测值时界面优先显示它。
    measured: dict[str, Any] = field(default_factory=dict)

    def matches(self, model_path: Path, stem: str) -> bool:
        if self.path:
            return _norm_path(self.path) == _norm_path(model_path)
        if not self.match:
            return False
        target = self.match.strip()
        if not target:
            return False
        # 文件名完全一致优先于子串
        return target.lower() == stem.lower() or target.lower() in stem.lower()

    def exact_match(self, stem: str) -> bool:
        return bool(self.match) and self.match.strip().lower() == stem.lower()


@dataclass
class ModelEntry:
    """扫描到的一个模型文件。"""

    path: Path
    name: str
    stem: str
    size_gib: float
    is_projector: bool = False
    profile: ModelProfile | None = None
    shape: ModelShape | None = None
    breakdown: TensorBreakdown | None = None
    error: str = ""

    @property
    def label(self) -> str:
        if self.profile and self.profile.label:
            return self.profile.label
        return self.stem

    @property
    def settings(self) -> dict[str, Any]:
        if self.profile:
            return dict(self.profile.settings)
        return {}

    @property
    def alias(self) -> str:
        return str(self.settings.get("model_alias") or self.stem.lower().replace(".", "-"))

    def describe_shape(self) -> str:
        if self.shape is None:
            return "（结构未知）"
        s = self.shape
        bits = [f"{s.block_count} 层"]
        if s.expert_count:
            bits.append(f"{s.expert_count} 专家/激活 {s.expert_used_count}")
        if s.full_attention_interval > 1:
            bits.append(f"{s.attention_layers}/{s.compute_layers} 层全注意力")
        else:
            bits.append(f"{s.attention_layers} 层全注意力")
        return " · ".join(bits)

    def estimate(self, context: int | None = None, kv_quant: str = "q8_0") -> dict[str, float]:
        """按 profile 的上下文估算显存/内存占用（GiB）。

        这是**理论估算**：resident + KV + 计算缓冲。实测在基准版上分毫不差，
        但在 apex@200K 上偏低 0.76 GiB（原因未查明，疑与该 GGUF 的张量摆放或
        llama.cpp 的额外缓冲有关）。所以界面优先显示 profile 里的 `measured`。
        """
        if self.shape is None or self.breakdown is None:
            return {}
        settings = self.settings
        ctx = int(context or settings.get("context_size") or 32768)
        cpm = bool(settings.get("cpu_moe", True))
        ncm = int(settings.get("n_cpu_moe", -1))
        ub = int(settings.get("ubatch_size", 512))
        from .core.gguf import compute_buffer_gib

        kv = self.shape.kv_gib(ctx, str(settings.get("kv_cache_type_k", kv_quant)))
        if ncm >= 0:
            experts_ram = self.breakdown.experts_gib(ncm)
        elif cpm:
            experts_ram = self.breakdown.expert_gib
        else:
            experts_ram = 0.0
        experts_vram = max(0.0, self.breakdown.expert_gib - experts_ram)
        buffer = compute_buffer_gib(ub)
        return {
            "context": ctx,
            "resident_gib": round(self.breakdown.resident_gib, 2),
            "experts_ram_gib": round(experts_ram, 2),
            "experts_vram_gib": round(experts_vram, 2),
            "kv_gib": round(kv, 2),
            "buffer_gib": round(buffer, 2),
            "vram_total_gib": round(
                self.breakdown.resident_gib + experts_vram + kv + buffer, 2
            ),
        }

    def to_dict(self, active_path: str = "", context: int | None = None) -> dict[str, Any]:
        est = self.estimate(context)
        measured = dict(self.profile.measured) if self.profile else {}
        # 有实测值时，用实测的显存/内存覆盖理论估算，并标记来源
        if measured:
            if "vram_gib" in measured:
                est["vram_total_gib"] = measured["vram_gib"]
            if "ram_gib" in measured:
                est["experts_ram_gib"] = measured["ram_gib"]
            est["measured"] = True
        return {
            "path": str(self.path),
            "name": self.name,
            "stem": self.stem,
            "size_gib": round(self.size_gib, 2),
            "is_projector": self.is_projector,
            "label": self.label,
            "note": self.profile.note if self.profile else "",
            "alias": self.alias,
            "matched": bool(self.profile and (self.profile.match or self.profile.path)),
            "profile_label": self.profile.label if self.profile else "",
            "shape": self.describe_shape(),
            "arch": self.shape.arch if self.shape else "",
            "context_max": self.shape.context_length if self.shape else 0,
            "models_paths": str(self.path),
            "estimate": est,
            "measured": measured,
            "settings": self.settings,
            "active": bool(active_path) and _norm_path(active_path) == _norm_path(self.path),
            "error": self.error,
        }


def _looks_like_projector(path: Path, meta: dict | None = None) -> bool:
    """判断是不是 mmproj（视觉投影）而不是可独立加载的语言模型。"""
    name = path.name.lower()
    if name.startswith("mmproj") or "mmproj" in name:
        return True
    if meta is not None:
        arch = str(meta.get("general.architecture", "")).lower()
        if any(arch.startswith(k.rstrip(".")) for k in _PROJECTOR_KEYS):
            return True
        # 语言模型必有 block_count；投影器没有
        if f"{arch}.block_count" not in meta:
            return True
    return False


class ModelRegistry:
    """加载 ``config/models.yaml``，扫描模型目录，解析每个模型的参数。"""

    def __init__(self, registry_file: str | os.PathLike[str] | None = None) -> None:
        self.file = Path(registry_file) if registry_file else DEFAULT_REGISTRY_FILE
        self.search_roots: list[Path] = []
        self.ignore_dirs: list[str] = []
        self.profiles: list[ModelProfile] = []
        self.default_profile = ModelProfile(label="未标定模型", note="使用保守默认值")
        self.load_error = ""
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.file.is_file():
            self.load_error = f"注册表不存在: {self.file}"
            log.warning(self.load_error)
            return
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(self.file.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 - 配置文件坏了也要能给出信息
            self.load_error = f"读取注册表失败: {exc}"
            log.error(self.load_error)
            return

        self.search_roots = [Path(p) for p in data.get("search_roots", []) if p]
        self.ignore_dirs = [str(s) for s in data.get("ignore_dirs", [])]

        default = data.get("default_profile") or {}
        self.default_profile = ModelProfile(
            label=str(default.get("label", "未标定模型")),
            note=str(default.get("note", "")),
            settings=dict(default.get("settings") or {}),
        )

        for raw in data.get("profiles", []) or []:
            if not isinstance(raw, dict):
                continue
            self.profiles.append(
                ModelProfile(
                    label=str(raw.get("label", "")),
                    note=str(raw.get("note", "")),
                    path=str(raw.get("path", "")),
                    match=str(raw.get("match", "")),
                    settings=dict(raw.get("settings") or {}),
                    measured=dict(raw.get("measured") or {}),
                )
            )
        log.info(
            "模型注册表已加载：%d 个 profile，%d 个扫描目录",
            len(self.profiles), len(self.search_roots),
        )

    # ------------------------------------------------------------------
    def resolve_profile(self, model_path: Path) -> ModelProfile:
        """给一个模型文件挑 profile；挑不到就返回 default_profile。"""
        stem = model_path.stem
        # 先找文件名完全一致的，避免 "X" 的子串规则抢走 "X-v2"
        for prof in self.profiles:
            if prof.path or prof.exact_match(stem):
                if prof.matches(model_path, stem):
                    return prof
        for prof in self.profiles:
            if prof.matches(model_path, stem):
                return prof
        return self.default_profile

    def iter_gguf(self) -> Iterable[Path]:
        seen: set[str] = set()
        for root in self.search_roots:
            if not root.is_dir():
                log.warning("扫描目录不存在，已跳过: %s", root)
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                # 原地过滤，避免走进忽略目录
                dirnames[:] = [
                    d for d in dirnames
                    if not any(ig.lower() in d.lower() for ig in self.ignore_dirs)
                ]
                for fn in filenames:
                    if not fn.lower().endswith(".gguf"):
                        continue
                    p = Path(dirpath) / fn
                    key = _norm_path(p)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield p

    def scan(self, *, with_shape: bool = True) -> list[ModelEntry]:
        """扫描所有 root，返回模型条目（含结构解析与参数匹配）。"""
        entries: list[ModelEntry] = []
        for path in self.iter_gguf():
            entry = ModelEntry(
                path=path,
                name=path.name,
                stem=path.stem,
                size_gib=path.stat().st_size / 1024**3 if path.is_file() else 0.0,
            )
            if with_shape:
                try:
                    entry.shape = model_shape(path)
                    if _looks_like_projector(path, None) and not entry.shape.block_count:
                        entry.is_projector = True
                    if entry.is_projector:
                        entries.append(entry)
                        continue
                    entry.breakdown = tensor_breakdown(path, entry.shape)
                except Exception as exc:  # noqa: BLE001 - 单个文件坏了不该中断扫描
                    entry.error = f"解析失败: {exc}"
                    log.warning("解析 %s 失败: %s", path, exc)
            entry.profile = self.resolve_profile(path)
            entries.append(entry)

        # 可按独立加载的模型排前面，再按体积降序
        entries.sort(key=lambda e: (e.is_projector, -e.size_gib))
        return entries

    def find(self, model_path: str | os.PathLike[str]) -> ModelEntry | None:
        """按路径找一个模型条目（用于"点击即加载"）。"""
        target = _norm_path(model_path)
        for entry in self.scan():
            if _norm_path(entry.path) == target:
                return entry
        return None

    def apply_to_config(self, entry: ModelEntry, cfg: ServerConfig) -> list[str]:
        """把模型的 profile 参数应用到 ServerConfig（原地修改）。

        返回被覆盖的字段名列表，便于在界面上展示"这次改了哪些参数"。
        模型路径与别名总是会被设置。
        """
        changed: list[str] = []
        settings = dict(entry.settings)
        settings.pop("model_alias", None)  # 单独处理

        cfg.model_path = str(entry.path)
        changed.append("model_path")
        alias = entry.settings.get("model_alias")
        if alias:
            cfg.model_alias = str(alias)
            changed.append("model_alias")

        valid = set(ServerConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        for key, value in settings.items():
            if key not in valid:
                log.warning("profile 里有未知字段，已忽略: %s", key)
                continue
            if value is None:
                continue
            setattr(cfg, key, value)
            changed.append(key)
        return changed
