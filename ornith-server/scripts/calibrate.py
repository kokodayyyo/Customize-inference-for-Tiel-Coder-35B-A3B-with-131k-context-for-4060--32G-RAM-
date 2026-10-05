"""显存与吞吐标定：找出 8GB 卡上能跑多长上下文、代价是多少。

做法是对每个候选配置：
  1. 用 llama-server 加载（记录加载是否成功、失败原因）
  2. 读 nvidia-smi 得到稳定态的显存占用
  3. 跑一次长输出生成，测 decode 吞吐
  4. 跑一次长提示，测 prefill 吞吐
然后回收进程、记录结果。

结果写入 JSON，便于对比不同上下文/量化/卸载策略。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
MODEL = Path(r"D:\models\Tile\Tile-35BA3B\Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf")
# 优先用项目自带的运行时（已从 LM Studio 复制，自包含）
_BUNDLED = PROJECT_ROOT / "runtime" / "llama.cpp" / "backends"
LLAMA_HOME = (_BUNDLED / "llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.51.0"
              if _BUNDLED.is_dir() else
              Path(r"C:\Users\YJC\.lmstudio\extensions\backends"
                   r"\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.51.0"))
VENDOR = (LLAMA_HOME.parent / "vendor" / "win-llama-cuda12-vendor-v2"
          if _BUNDLED.is_dir() else
          Path(r"C:\Users\YJC\.lmstudio\extensions\backends\vendor"
               r"\win-llama-cuda12-vendor-v2"))
LOG_DIR = PROJECT_ROOT / "runtime" / "logs"
RESULT_FILE = PROJECT_ROOT / "runtime" / "calibration.json"
PORT = 18099


@dataclass
class Case:
    """一个待标定的配置。"""

    name: str
    ctx: int
    kv_type: str = "q8_0"
    kv_offload: bool = True
    ubatch: int = 512
    batch: int = 2048
    ngl: int = 99
    note: str = ""
    gen_tokens: int = 128
    prompt_chars: int = 4000
    extra: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    case: str
    ok: bool
    detail: str = ""
    vram_used_gib: float = 0.0
    vram_delta_gib: float = 0.0
    load_seconds: float = 0.0
    gen_tps: float = 0.0
    prefill_tps: float = 0.0
    ttft_seconds: float = 0.0
    ctx: int = 0
    kv_type: str = ""
    kv_offload: bool = True


def vram() -> tuple[float, float]:
    """返回 (used_gib, free_gib)。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    used, free = (float(x) for x in out.split(","))
    return used / 1024, free / 1024


FILLER = (
    "在显存受限的设备上部署大语言模型时，键值缓存的大小往往成为决定上下文长度的瓶颈。"
    "这段文字用于填充提示，以测量预填充阶段的真实吞吐表现。"
)


def build_cmd(case: Case) -> list[str]:
    cmd = [
        str(LLAMA_HOME / "llama-server.exe"),
        "-m", str(MODEL),
        "--host", "127.0.0.1", "--port", str(PORT),
        "-ngl", str(case.ngl),
        "-c", str(case.ctx),
        "-np", "1",
        "-b", str(case.batch),
        "-ub", str(case.ubatch),
        "-fa", "on",
        "-ctk", case.kv_type,
        "-ctv", case.kv_type,
        "--no-warmup",
        "-t", "8",
    ]
    cmd.append("--kv-offload" if case.kv_offload else "--no-kv-offload")
    cmd += case.extra
    return cmd


def env() -> dict[str, str]:
    e = dict(os.environ)
    e["PATH"] = os.pathsep.join([str(VENDOR), str(LLAMA_HOME), e.get("PATH", "")])
    return e


def run_case(case: Case, timeout: int = 420) -> Outcome:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"calib-{case.name}.log"
    baseline_used, baseline_free = vram()

    cmd = build_cmd(case)
    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        logf.write("CMDLINE: " + subprocess.list2cmdline(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(LLAMA_HOME), env=env(),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        started = time.time()
        ready = False
        while time.time() - started < timeout:
            if proc.poll() is not None:
                break
            try:
                r = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1.5)

        load_seconds = time.time() - started

        if not ready:
            code = proc.poll()
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lowered = text.lower()
            reason = "加载失败"
            if "out of memory" in lowered or "cuda_error_out_of_memory" in lowered:
                reason = "CUDA 显存不足"
            elif "failed to allocate" in lowered:
                reason = "分配失败（显存不足）"
            elif code is not None:
                reason = f"进程退出 rc=0x{code & 0xFFFFFFFF:08X}"
            outcome = Outcome(case=case.name, ok=False, detail=reason, ctx=case.ctx,
                              kv_type=case.kv_type, kv_offload=case.kv_offload)
            _kill(proc)
            return outcome

        # 稳定后读显存（等一小会，让 CUDA 上下文分配完成）
        time.sleep(3)
        used, free = vram()

        outcome = Outcome(
            case=case.name, ok=True, ctx=case.ctx, kv_type=case.kv_type,
            kv_offload=case.kv_offload,
            vram_used_gib=round(used, 2),
            vram_delta_gib=round(used - baseline_used, 2),
            load_seconds=round(load_seconds, 1),
        )

        try:
            gen_tps, prefill_tps, ttft = measure(case)
            outcome.gen_tps = round(gen_tps, 1)
            outcome.prefill_tps = round(prefill_tps, 1)
            outcome.ttft_seconds = round(ttft, 2)
        except Exception as exc:  # noqa: BLE001
            outcome.detail = f"推理测量失败: {exc}"

        _kill(proc)
        time.sleep(2)
        return outcome


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
                proc.wait(timeout=8)
            except (subprocess.TimeoutExpired, OSError):
                pass


def measure(case: Case) -> tuple[float, float, float]:
    """返回 (decode_tps, prefill_tps, ttft)。"""
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"

    # --- 预热：排除 CUDA 图捕获开销 ---
    with httpx.Client(timeout=600) as client:
        client.post(url, json={
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 16, "temperature": 0, "stream": False,
        })

        # --- decode：短提示 + 长输出 ---
        started = time.perf_counter()
        ttft = 0.0
        chunks = 0
        with client.stream("POST", url, json={
            "messages": [{"role": "user", "content": "请从 1 数到 200，每个数字之间用空格分隔。"}],
            "max_tokens": case.gen_tokens, "temperature": 0, "stream": True,
        }) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunks += 1
                if ttft == 0.0:
                    ttft = time.perf_counter() - started
        gen_elapsed = time.perf_counter() - started
        decode_seconds = max(gen_elapsed - ttft, 1e-3)
        tokens = max(chunks - 1, 1)
        gen_tps = tokens / decode_seconds

        # --- prefill：长提示 + 极短输出 ---
        repeats = max(1, case.prompt_chars // len(FILLER) + 1)
        long_prompt = (FILLER * repeats)[: case.prompt_chars] + "\n\n用一句话总结上面这段话。"
        started = time.perf_counter()
        ttft2 = 0.0
        with client.stream("POST", url, json={
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 8, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
        }) as resp:
            resp.raise_for_status()
            usage = {}
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
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                if ttft2 == 0.0 and (obj.get("choices") or [{}])[0].get("delta", {}).get("content"):
                    ttft2 = time.perf_counter() - started
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        prefill_tps = prompt_tokens / max(ttft2, 1e-3) if prompt_tokens else 0.0

    return gen_tps, prefill_tps, ttft


def default_cases() -> list[Case]:
    """标定矩阵：从"能全放显存"到"128K 必须放内存"的关键点。"""
    return [
        Case("A-8k-q8-gpu", 8192, "q8_0", True, note="基准：8K 全放显存"),
        Case("B-32k-q4-gpu", 32768, "q4_0", True, note="32K KV q4_0 全放显存"),
        Case("C-64k-q4-gpu", 65536, "q4_0", True, note="64K KV q4_0 全放显存"),
        Case("D-128k-q4-gpu", 131072, "q4_0", True, note="128K 全放显存（预期 OOM）"),
        Case("E-128k-q8-cpu", 131072, "q8_0", False, note="128K KV放内存 q8_0"),
        Case("F-128k-q4-cpu", 131072, "q4_0", False, note="128K KV放内存 q4_0"),
    ]


def main() -> int:
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    cases = [c for c in default_cases() if not only or c.name in only]

    print(f"模型: {MODEL.name}")
    print(f"引擎: {LLAMA_HOME.name}")
    print(f"用例: {len(cases)} 个\n")

    results: list[Outcome] = []
    for case in cases:
        used, free = vram()
        print(f"--- {case.name} ---")
        print(f"    ctx={case.ctx} kv={case.kv_type} "
              f"kv_offload={'GPU' if case.kv_offload else 'CPU'} | {case.note}")
        print(f"    开始前显存 used={used:.2f} free={free:.2f} GiB")
        outcome = run_case(case)
        results.append(outcome)
        if outcome.ok:
            print(f"    加载 {outcome.load_seconds}s | 显存增量 {outcome.vram_delta_gib:.2f} GiB "
                  f"(峰值 {outcome.vram_used_gib:.2f})")
            print(f"    decode {outcome.gen_tps:.1f} tok/s | prefill {outcome.prefill_tps:.1f} tok/s "
                  f"| TTFT {outcome.ttft_seconds:.2f}s")
            if outcome.detail:
                print(f"    备注: {outcome.detail}")
        else:
            print(f"    失败: {outcome.detail}")
        print()

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps([o.__dict__ for o in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"结果已写入 {RESULT_FILE}")

    print("\n=== 汇总 ===")
    print(f"{'用例':<16}{'上下文':>8}{'KV':>8}{'KV位置':>8}{'显存增量':>10}"
          f"{'decode':>10}{'prefill':>10}")
    for o in results:
        if o.ok:
            print(f"{o.case:<16}{o.ctx:>8}{o.kv_type:>8}"
                  f"{'GPU' if o.kv_offload else 'CPU':>8}{o.vram_delta_gib:>9.2f}G"
                  f"{o.gen_tps:>9.1f}t/s{o.prefill_tps:>9.0f}t/s")
        else:
            print(f"{o.case:<16}{o.ctx:>8}{o.kv_type:>8}"
                  f"{'GPU' if o.kv_offload else 'CPU':>8}  {o.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
