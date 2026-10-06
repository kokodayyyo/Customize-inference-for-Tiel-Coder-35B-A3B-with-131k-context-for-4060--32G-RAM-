"""同一批提示词对比两个模型：输出质量、chat template 状态、吞吐。

用途：当"换了个量化后感觉答得更差"时，先分清是**量化本身的损失**还是
**我们套的参数有问题**（例如 GGUF 里没有可用的 chat template、被 llama.cpp
退回到通用模板 —— 那会让质量断崖式下跌，而且完全可以修）。

做法：对每个模型用**各自 profile 里调好的参数**起一次后端，
先抓 /props 看 chat template，再用 temperature=0 跑同一批提示词。

用法::

    python scripts/compare_models.py
    python scripts/compare_models.py --prompts 2 --max-tokens 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_server.config import ServerConfig  # noqa: E402
from llm_server.core.backend import resolve_backend  # noqa: E402
from llm_server.core.gguf import model_shape  # noqa: E402
from llm_server.core.server import LlamaBackendServer, build_load_profiles  # noqa: E402
from llm_server.models_registry import ModelRegistry  # noqa: E402

PORT = 18077

# 固定提示词：覆盖中文问答、指令遵循、推理、代码
PROMPTS: list[tuple[str, str]] = [
    ("中文问答", "用一句话解释什么是 MoE（混合专家）模型，不要超过 50 字。"),
    ("指令遵循", "严格按这个格式回答，不要有任何多余内容：\n答案：[一个数字]\n\n问题：一个班 40 人，60% 是女生，女生有多少人？"),
    ("简单推理", "小明比小红高，小红比小刚高。按从高到低排序这三个人，并说明理由。"),
    ("代码", "写一个 Python 函数 add(a, b) 返回两数之和，只输出代码。"),
]


def http_json(url: str, payload: dict | None = None, timeout: float = 600.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def analyse_template(props: dict) -> dict:
    """判断 chat template 是否可用 —— 这是质量问题最常见的可修原因。"""
    tpl = props.get("chat_template") or ""
    caps = props.get("chat_template_caps") or {}
    default = props.get("default_generation_settings") or {}
    params = default.get("params") or {}
    return {
        "template_len": len(tpl) if isinstance(tpl, str) else 0,
        "template_head": (tpl[:80].replace("\n", "\\n") if isinstance(tpl, str) else ""),
        "caps": caps,
        # llama.cpp 在没读到模型自带模板时会退回这些通用值
        "chat_format": params.get("chat_format"),
        "reasoning_format": params.get("reasoning_format"),
        "n_ctx": default.get("n_ctx"),
        "modalities": props.get("modalities"),
        "ftype": props.get("model_ftype"),
        "build": props.get("build_info"),
    }


def run_model(cfg: ServerConfig, label: str, prompts: list[tuple[str, str]],
              max_tokens: int) -> dict:
    entry = ModelRegistry().find(cfg.model_path)
    print(f"\n{'#' * 78}\n# {label}\n#   {cfg.model_path}\n{'#' * 78}")

    shape = model_shape(cfg.model_path)
    print(f"  结构: {shape.describe()}")

    server = LlamaBackendServer(cfg, resolve_backend(cfg.llama_dir or None))
    prof = build_load_profiles(cfg)[0]
    # 用本次对比专用的端口，避免和其它实例打架
    cfg.backend_port = PORT
    log_path = PROJECT_ROOT / "runtime" / "logs" / f"cmp-{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")

    cmd = server.build_command(prof)
    env = server.backend.build_env()
    log_handle.write("CMDLINE: " + subprocess.list2cmdline(cmd) + "\n\n")
    log_handle.flush()
    proc = subprocess.Popen(
        cmd, stdout=log_handle, stderr=subprocess.STDOUT,
        cwd=str(server.backend.home), env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    result: dict = {"label": label, "path": str(cfg.model_path), "shape": shape.describe()}
    base = f"http://127.0.0.1:{PORT}"
    deadline = time.time() + 400

    try:
        ready = False
        while time.time() < deadline:
            if proc.poll() is not None:
                print("  启动失败:\n" + tail(log_path, 20))
                result["error"] = "启动失败"
                return result
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        if not ready:
            result["error"] = "超时"
            return result

        props = http_json(f"{base}/props")
        info = analyse_template(props)
        result["template"] = info
        print(f"  chat template : {info['template_len']} 字符"
              f"{'  ⚠ 为空！' if info['template_len'] == 0 else ''}")
        print(f"  chat_format   : {info['chat_format']}   reasoning_format: {info['reasoning_format']}")
        print(f"  量化标识      : {info['ftype']}    n_ctx: {info['n_ctx']}")
        print(f"  模板开头      : {info['template_head']}")

        # 模板相关的告警（llama.cpp 会明确提示）
        warnings = [ln for ln in tail(log_path, 400).splitlines()
                    if re.search(r"template|jinja|chat", ln, re.I)
                    and re.search(r"W |E |warn|error|fallback|default", ln, re.I)]
        if warnings:
            print("  日志中的模板相关提示:")
            for w in warnings[:6]:
                print("    " + w.strip()[:150])
        result["warnings"] = warnings[:6]

        # 预热
        http_json(f"{base}/v1/chat/completions", {
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 8, "temperature": 0,
        })

        print()
        outputs = []
        for name, prompt in prompts:
            try:
                resp = http_json(f"{base}/v1/chat/completions", {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "top_k": 1,
                })
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] 请求失败: {exc}")
                continue
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            text = (msg.get("content") or "").strip()
            think = (msg.get("reasoning_content") or "").strip()
            t = resp.get("timings") or {}
            outputs.append({
                "name": name, "prompt": prompt, "content": text,
                "think_len": len(think), "think_head": think[:160],
                "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                "decode_tps": round(t.get("predicted_per_second") or 0, 2),
                "prefill_tps": round(t.get("prompt_per_second") or 0, 1),
            })
            print(f"  [{name}] {len(text)} 字符 / 思考 {len(think)}"
                  f" / {t.get('predicted_per_second', 0):.1f} tok/s")
        result["outputs"] = outputs
        return result
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_handle.close()
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="对比两个模型的实际输出")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "runtime" / "compare.json"))
    args = parser.parse_args()

    registry = ModelRegistry()
    entries = [e for e in registry.scan() if not e.is_projector]
    tile = [e for e in entries if "tiel" in e.name.lower() or "tile" in e.name.lower()]
    if len(tile) < 2:
        print(f"需要至少两个 Tile 模型才能对比，当前找到 {len(tile)} 个")
        return 1

    results = []
    for entry in tile:
        cfg = ServerConfig.load()
        registry.apply_to_config(entry, cfg)
        results.append(run_model(cfg, entry.label, PROMPTS, args.max_tokens))

    Path(args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---- 并排输出 ----
    print(f"\n\n{'=' * 78}\n并排对比\n{'=' * 78}")
    for i, (name, prompt) in enumerate(PROMPTS):
        print(f"\n{'─' * 74}\n【{name}】{prompt}\n{'─' * 74}")
        for r in results:
            outs = r.get("outputs") or []
            if i >= len(outs):
                print(f"\n▸ {r['label']}: （无输出）")
                continue
            o = outs[i]
            print(f"\n▸ {r['label']}  ({o['decode_tps']} tok/s, {o['completion_tokens']} tok)")
            print("  " + (o["content"][:700].replace("\n", "\n  ") or "（正文为空）"))

    print(f"\n\n{'=' * 78}\n模板与量化对照\n{'=' * 78}")
    for r in results:
        t = r.get("template") or {}
        print(f"  {r['label']:<24} 模板 {t.get('template_len', 0):>6} 字符"
              f"  chat_format={t.get('chat_format')}  ftype={t.get('ftype')}")
    print(f"\n完整结果: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
