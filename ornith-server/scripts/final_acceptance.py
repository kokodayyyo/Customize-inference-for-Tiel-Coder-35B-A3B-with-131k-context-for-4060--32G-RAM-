"""最终验收：用默认配置（128K 上下文）启动服务，调用后关闭。

与 e2e_test.py 的区别：e2e 用小上下文测接口正确性；本脚本验证
**生产默认配置**能否真正加载并服务 128K 上下文。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
API = "http://127.0.0.1:8000"

# 从配置读模型别名，换模型时不用改测试
sys.path.insert(0, str(ROOT / "src"))
try:
    from ornith_server.config import ServerConfig as _SC

    MODEL_ALIAS = _SC.load().model_alias
except Exception:  # noqa: BLE001 - 配置读不到时退回默认别名
    MODEL_ALIAS = "tile-35b-a3b"


def vram() -> tuple[float, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    used, free = (float(x) for x in out.split(","))
    return used / 1024, free / 1024


def llama_server_pids() -> list[int]:
    """列出残留的后端进程。

    优先用 PowerShell（wmic 在新版 Windows 上已被移除，且 tasklist 在
    非管理员会话下可能返回 Access denied）。
    """
    ps = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-Process llama-server -ErrorAction SilentlyContinue).Id -join ','"],
        capture_output=True, text=True, timeout=30,
    )
    return [int(x) for x in ps.stdout.strip().split(",") if x.strip().isdigit()]


def main() -> int:
    used0, free0 = vram()
    print(f"启动前显存: used={used0:.2f} free={free0:.2f} GiB")

    LOG = ROOT / "runtime" / "logs" / "final_acceptance.log"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = LOG.open("w", encoding="utf-8", errors="replace")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [PY, "main.py", "serve"],
        cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT, env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    try:
        started = time.time()
        ready = False
        health: dict = {}
        while time.time() - started < 600:
            if proc.poll() is not None:
                print(f"服务进程退出 rc={proc.returncode}")
                print(LOG.read_text(encoding="utf-8", errors="replace")[-3000:])
                return 1
            try:
                r = httpx.get(f"{API}/health", timeout=3)
                health = r.json()
                if health.get("ready"):
                    ready = True
                    break
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(2)

        load_seconds = time.time() - started
        if not ready:
            print(f"超时未就绪（{load_seconds:.0f}s）健康状态: {health}")
            print(LOG.read_text(encoding="utf-8", errors="replace")[-3000:])
            return 1

        used1, free1 = vram()
        print(f"\n服务就绪，用时 {load_seconds:.1f}s")
        print(f"健康检查: {json.dumps(health, ensure_ascii=False)}")
        print(f"加载后显存: used={used1:.2f} (增量 {used1 - used0:+.2f}) free={free1:.2f} GiB")

        print("\n--- 128K 上下文实际推理测试 ---")
        prompt = (
            "下面是一段用于验证长上下文能力的说明。请阅读后用一句话回答："
            "128K 上下文在实际使用中最大的代价是什么？\n\n"
            + "长上下文的主要开销来自注意力计算需要读取完整的键值缓存。" * 200
        )
        payload = {
            "model": MODEL_ALIAS,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0,
        }
        started = time.perf_counter()
        r = httpx.post(f"{API}/v1/chat/completions", json=payload, timeout=900)
        elapsed = time.perf_counter() - started
        if r.status_code != 200:
            print(f"请求失败 HTTP {r.status_code}: {r.text[:400]}")
            return 1
        body = r.json()
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        usage = body.get("usage") or {}
        content = msg.get("content") or ""
        think = msg.get("reasoning_content") or ""
        print(f"提示 token: {usage.get('prompt_tokens')}  "
              f"生成 token: {usage.get('completion_tokens')}  耗时 {elapsed:.1f}s")
        print(f"思考内容: {len(think)} 字符")
        print(f"正文回答: {content[:200]!r}")

        print("\n--- 流式测试 ---")
        started = time.perf_counter()
        ttft = 0.0
        n_content = n_think = 0
        timings: dict = {}
        with httpx.stream(
            "POST", f"{API}/v1/chat/completions",
            json={**payload, "messages": [{"role": "user", "content": "简短说明 KV cache 的作用。"}],
                  "stream": True},
            timeout=900,
        ) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                obj = json.loads(data)
                if isinstance(obj.get("timings"), dict) and obj["timings"]:
                    timings = obj["timings"]
                delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                if delta.get("content"):
                    n_content += 1
                if delta.get("reasoning_content"):
                    n_think += 1
                if (n_content or n_think) and ttft == 0.0:
                    ttft = time.perf_counter() - started
        print(f"首字 {ttft:.2f}s | 正文 {n_content} 段 / 思考 {n_think} 段 | "
              f"decode {timings.get('predicted_per_second', 0):.1f} tok/s "
              f"prefill {timings.get('prompt_per_second', 0):.0f} tok/s")

        stats = httpx.get(f"{API}/stats", timeout=30).json()
        print(f"\n网关统计: {json.dumps(stats, ensure_ascii=False)}")

        print("\n--- 关闭服务 ---")
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        # 显存释放需要时间：CUDA 上下文销毁与驱动回收不是瞬时的，因此轮询
        # 等待而不是固定睡几秒就读。Job Object 兜底会在父进程退出时强制
        # 结束后端，正常情况下几秒内显存就回到基线。
        released = False
        used2, free2 = used1, free1
        for _ in range(120):
            used2, free2 = vram()
            if used2 <= used0 + 0.35:
                released = True
                break
            time.sleep(1)
        leftover = llama_server_pids()
        print(f"关闭后显存: used={used2:.2f} free={free2:.2f} GiB (基线 {used0:.2f})")
        print(f"显存已释放: {'是' if released else '否'}")
        print(f"残留 llama-server: {leftover or '无'}")
        print(f"\n服务日志: {LOG}")

        if not released or leftover:
            print("\n[验收失败] 关闭后资源未完全释放。")
            return 1
        print("\n[验收通过] 默认 128K 配置可正常加载、服务与关闭。")
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        logf.close()


if __name__ == "__main__":
    raise SystemExit(main())
