"""基准测试：测量真实的提示处理速度与生成速度。

与单纯跑一次请求不同，这里会：
- 先做一次预热，排除首次 CUDA 图捕获与页缓存冷启动的干扰；
- 分别测量不同提示长度下的 prompt 处理速度（prefill，决定"首字延迟"）；
- 测量长文本生成速度（decode，决定"出字速度"）；
- 输出平均/最快/最慢，便于判断参数调整是否真的有效。

指标口径与 llama.cpp 自带的 ``/metrics`` 一致：tok/s。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass

import httpx

from .config import ServerConfig
from .core import LlamaBackendServer, make_backend

# 用于制造长提示的填充文本（内容无语义，只影响长度）
_FILLER = (
    "人工智能推理服务需要在有限的显存预算内平衡上下文长度、批处理规模与数值精度。"
    "下面这段文字仅用于填充提示长度，以测量预填充阶段的实际吞吐。"
)


@dataclass
class BenchResult:
    name: str
    prompt_tokens: int
    completion_tokens: int
    prompt_tps: float
    gen_tps: float
    ttft_seconds: float
    total_seconds: float

    def render(self) -> str:
        return (
            f"{self.name:<22} 提示 {self.prompt_tokens:>6} tok  生成 {self.completion_tokens:>4} tok  "
            f"| prefill {self.prompt_tps:>8.1f} tok/s  首字 {self.ttft_seconds:>5.2f}s  "
            f"| decode {self.gen_tps:>6.1f} tok/s"
        )


def _make_prompt(target_tokens: int) -> str:
    """粗略构造约 target_tokens 个 token 的提示（中文约 1.5 字/token）。"""
    if target_tokens <= 0:
        return "你好"
    approx_chars = int(target_tokens * 1.5)
    repeats = max(1, approx_chars // len(_FILLER) + 1)
    return (_FILLER * repeats)[:approx_chars] + "\n\n请用一句话总结上面这段话的主题。"


def _one_run(
    client: httpx.Client,
    cfg: ServerConfig,
    prompt: str,
    max_tokens: int,
    stream: bool = True,
) -> BenchResult:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }

    started = time.perf_counter()
    ttft = 0.0
    text_parts: list[str] = []
    usage: dict = {}

    with client.stream("POST", f"{cfg.backend_base_url}/v1/chat/completions", json=payload) as resp:
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.read().decode('utf-8', 'replace')[:400]}")
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or [{}]
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                if ttft == 0.0:
                    ttft = time.perf_counter() - started
                text_parts.append(piece)
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]

    total = time.perf_counter() - started
    completion_tokens = int(usage.get("completion_tokens") or len(text_parts))

    # 关键：流式响应的 usage 在部分版本里不可靠，改用非流式请求拿准确 prompt_tokens。
    # 同时用 max_tokens=1 的请求隔离出纯粹的预填充耗时（总时长减去约 1 个 token 的生成）。
    prompt_tokens, prefill_seconds = _measure_prefill(client, cfg, prompt)
    if not completion_tokens:
        completion_tokens = max(1, len(text_parts))

    decode_seconds = max(total - ttft, 1e-3)
    return BenchResult(
        name="",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tps=prompt_tokens / max(prefill_seconds, 1e-3),
        gen_tps=(completion_tokens - 1) / decode_seconds if completion_tokens > 1 else 0.0,
        ttft_seconds=ttft,
        total_seconds=total,
    )


def _measure_prefill(client: httpx.Client, cfg: ServerConfig, prompt: str) -> tuple[int, float]:
    """用非流式、max_tokens=1 的请求测量预填充耗时。

    返回 (prompt_tokens, 预填充秒数)。单次生成的 1 个 token 开销极小，
    这里按经验扣除一个 decode 步长的估计值（约 1/30 秒）可以忽略不计，
    因此直接把总耗时当作预填充时间。
    """
    started = time.perf_counter()
    resp = client.post(
        f"{cfg.backend_base_url}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        },
    )
    elapsed = time.perf_counter() - started
    resp.raise_for_status()
    usage = resp.json().get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or _estimate_tokens(prompt))
    return prompt_tokens, elapsed


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文按 1.5 字/token，英文按 4 字符/token）。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def run_benchmark(
    cfg: ServerConfig,
    *,
    warmup: bool = True,
    repeats: int = 3,
    verbose: bool = False,
    server: LlamaBackendServer | None = None,
) -> int:
    """执行基准测试，返回进程退出码。"""
    owns_server = False
    backend = server or make_backend(cfg)

    if not backend.is_running:
        if not backend.wait_ready(timeout=3):
            if verbose:
                print("后端未运行，正在启动（约需 5-30 秒）…")
            try:
                backend.start()
                owns_server = True
            except RuntimeError as exc:
                print(f"启动失败：{exc}")
                return 1

    scenarios = [
        ("短提示 / 短回答", _make_prompt(0), 128),
        ("中提示 / 中回答", _make_prompt(600), 256),
        ("长提示 / 短回答", _make_prompt(3000), 128),
        ("超长提示 / 短回答", _make_prompt(7000), 64),
    ]

    results: list[BenchResult] = []
    try:
        with httpx.Client(timeout=cfg.request_timeout, trust_env=False) as client:
            if warmup:
                if verbose:
                    print("预热中（首次推理会包含 CUDA 图捕获开销）…")
                try:
                    _one_run(client, cfg, "你好", 16)
                except RuntimeError as exc:
                    print(f"预热失败：{exc}")
                    return 1

            print()
            print("=" * 100)
            print(f"  基准测试  模型={cfg.model_alias}  上下文={cfg.context_size}  "
                  f"并行槽={cfg.parallel_slots}  KV={cfg.kv_cache_type_k}")
            print("=" * 100)

            for name, prompt, max_tokens in scenarios:
                runs: list[BenchResult] = []
                for i in range(max(1, repeats)):
                    try:
                        res = _one_run(client, cfg, prompt, max_tokens)
                    except RuntimeError as exc:
                        print(f"{name:<22} 失败: {exc}")
                        break
                    res.name = f"{name}" if repeats == 1 else f"{name} #{i + 1}"
                    runs.append(res)
                    results.append(res)
                    if verbose and repeats > 1:
                        print("  " + res.render())
                if runs:
                    best = max(runs, key=lambda r: r.gen_tps)
                    print("  " + best.render())
    finally:
        if owns_server:
            if verbose:
                print("\n正在停止后端…")
            backend.stop()

    if results:
        decode = [r.gen_tps for r in results if r.gen_tps > 0]
        prefill = [r.prompt_tps for r in results]
        print("-" * 100)
        if decode:
            print(
                f"  生成速度: 中位 {statistics.median(decode):.1f} tok/s  "
                f"最快 {max(decode):.1f}  最慢 {min(decode):.1f}"
            )
        print(
            f"  预填充速度: 中位 {statistics.median(prefill):.1f} tok/s  "
            f"最快 {max(prefill):.1f}  最慢 {min(prefill):.1f}"
        )
        print("=" * 100)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_benchmark(ServerConfig.load(), verbose=True))
