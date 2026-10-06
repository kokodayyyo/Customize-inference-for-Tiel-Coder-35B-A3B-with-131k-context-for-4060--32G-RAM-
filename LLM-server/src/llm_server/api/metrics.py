"""内网 API 服务的监控指标（内存内，无外部依赖）。"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Metrics:
    """线程安全的请求/吞吐统计。"""

    started_at: float = field(default_factory=time.time)
    requests_total: int = 0
    requests_failed: int = 0
    requests_rejected: int = 0
    requests_active: int = 0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    _latencies: deque[float] = field(default_factory=lambda: deque(maxlen=500))
    _by_model: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # -- 记录 -----------------------------------------------------------
    def on_start(self) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_active += 1

    def on_finish(
        self,
        elapsed: float,
        *,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        failed: bool = False,
    ) -> None:
        with self._lock:
            self.requests_active = max(0, self.requests_active - 1)
            self._latencies.append(elapsed)
            if failed:
                self.requests_failed += 1
            self.prompt_tokens_total += max(0, prompt_tokens)
            self.completion_tokens_total += max(0, completion_tokens)
            if model:
                self._by_model[model] = self._by_model.get(model, 0) + 1

    def on_rejected(self) -> None:
        with self._lock:
            self.requests_rejected += 1

    # -- 读取 -----------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            lat = sorted(self._latencies)
            uptime = time.time() - self.started_at

            def pct(p: float) -> float:
                if not lat:
                    return 0.0
                idx = min(len(lat) - 1, int(round((len(lat) - 1) * p)))
                return round(lat[idx], 3)

            generated = self.completion_tokens_total
            return {
                "uptime_seconds": round(uptime, 1),
                "requests_total": self.requests_total,
                "requests_active": self.requests_active,
                "requests_failed": self.requests_failed,
                "requests_rejected": self.requests_rejected,
                "prompt_tokens_total": self.prompt_tokens_total,
                "completion_tokens_total": generated,
                "avg_completion_tokens_per_second": (
                    round(generated / uptime, 2) if uptime > 0 and generated else 0.0
                ),
                "latency_p50_seconds": pct(0.5),
                "latency_p95_seconds": pct(0.95),
                "requests_by_model": dict(self._by_model),
            }

    def render_prometheus(self) -> str:
        """输出 Prometheus 文本格式，便于接入现有监控。"""
        snap = self.snapshot()
        lines: list[str] = []

        def metric(name: str, value: Any, help_text: str, mtype: str = "gauge") -> None:
            lines.append(f"# HELP llm_{name} {help_text}")
            lines.append(f"# TYPE llm_{name} {mtype}")
            lines.append(f"llm_{name} {value}")

        metric("uptime_seconds", snap["uptime_seconds"], "网关运行时长（秒）")
        metric("requests_total", snap["requests_total"], "累计请求数", "counter")
        metric("requests_active", snap["requests_active"], "正在处理的请求数")
        metric("requests_failed", snap["requests_failed"], "失败请求数", "counter")
        metric("requests_rejected", snap["requests_rejected"], "因过载被拒请求数", "counter")
        metric("prompt_tokens_total", snap["prompt_tokens_total"], "累计提示 token", "counter")
        metric("completion_tokens_total", snap["completion_tokens_total"], "累计生成 token", "counter")
        metric("latency_p50_seconds", snap["latency_p50_seconds"], "请求延迟 P50")
        metric("latency_p95_seconds", snap["latency_p95_seconds"], "请求延迟 P95")
        return "\n".join(lines) + "\n"


METRICS = Metrics()
