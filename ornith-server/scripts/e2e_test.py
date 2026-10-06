"""端到端验证：启动内网 API 服务，跑全部接口，然后关闭。

验证项：
  1. /health、/stats、/metrics 运维接口
  2. /v1/models 模型列表
  3. /v1/chat/completions 非流式
  4. /v1/chat/completions 流式（并确认推理内容 reasoning_content 会被透传）
  5. API Key 鉴权（未带密钥应 401，带正确密钥应 200）
  6. 并发请求（验证排队与并行槽位）
  7. 指标统计是否正确累计 token
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ornith_server.api import create_app  # noqa: E402
from ornith_server.config import ServerConfig  # noqa: E402
from ornith_server.core import make_backend  # noqa: E402

API = "http://127.0.0.1:18080"
KEY = "sk-e2e-test-key"
# 从配置读模型别名，换模型时测试不用跟着改
MODEL_ALIAS = ServerConfig.load().model_alias

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    mark = "通过" if ok else "失败"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    cfg = ServerConfig.load()
    cfg.proxy_host = "127.0.0.1"
    cfg.proxy_port = 18080
    cfg.backend_port = 18081
    cfg.api_key = KEY
    cfg.context_size = 8192
    cfg.parallel_slots = 2

    problems = cfg.validate()
    if problems:
        print("配置问题:", problems)
        return 2

    import uvicorn

    backend = make_backend(cfg)
    app = create_app(cfg, backend)

    server = uvicorn.Server(uvicorn.Config(
        app, host=cfg.proxy_host, port=cfg.proxy_port,
        log_level="warning", access_log=False,
    ))

    import threading

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # 等待服务可用（模型加载需要几秒）
    root = httpx.Client(timeout=30, trust_env=False)
    started = time.time()
    ready = False
    while time.time() - started < 300:
        try:
            if root.get(f"{API}/health").status_code in (200, 503):
                ready = True
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    check("服务启动", ready, f"{time.time() - started:.1f}s")
    if not ready:
        return 1

    # 等后端就绪
    while time.time() - started < 300:
        body = root.get(f"{API}/health").json()
        if body.get("ready"):
            break
        time.sleep(1)
    print(f"  后端就绪: {body.get('ready')} pid={body.get('pid')} "
          f"ctx={body.get('context_size')} 每槽={body.get('context_per_slot')}")
    check("后端就绪", bool(body.get("ready")), str(body.get("load_error") or ""))

    auth = {"Authorization": f"Bearer {KEY}"}
    payload = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": "只回答两个字：收到"}],
        "max_tokens": 256,
        "temperature": 0,
    }

    print("\n[1] 鉴权")
    r = root.post(f"{API}/v1/chat/completions", json=payload)
    check("无密钥被拒绝", r.status_code == 401, f"HTTP {r.status_code}")
    r = root.post(f"{API}/v1/chat/completions", json=payload,
                  headers={"Authorization": "Bearer wrong-key"})
    check("错误密钥被拒绝", r.status_code == 401, f"HTTP {r.status_code}")
    check("健康检查免鉴权", root.get(f"{API}/health").status_code in (200, 503))

    print("\n[2] 模型列表")
    r = root.get(f"{API}/v1/models", headers=auth)
    ids = [str(m.get("id", "")) for m in r.json().get("data", [])] if r.status_code == 200 else []
    ok = r.status_code == 200 and any(MODEL_ALIAS in i for i in ids)
    check("/v1/models", ok, f"期望包含 {MODEL_ALIAS!r}，实际 {ids}")

    print("\n[3] 非流式对话")
    started = time.perf_counter()
    r = root.post(f"{API}/v1/chat/completions", json=payload, headers=auth)
    elapsed = time.perf_counter() - started
    ok = r.status_code == 200
    if ok:
        body = r.json()
        msg = (body.get("choices") or [{}])[0].get("message") or {}
        text = (msg.get("content") or "")
        think = (msg.get("reasoning_content") or "")
        usage = body.get("usage") or {}
        print(f"       正文: {text[:60]!r}")
        print(f"       思考长度: {len(think)} 字符  usage={usage}")
        ok = bool(text or think)
    check("非流式返回内容", ok, f"{elapsed:.2f}s")

    print("\n[4] 流式对话（含推理内容透传）")
    started = time.perf_counter()
    ttft = 0.0
    content_chunks = 0
    reasoning_chunks = 0
    timings: dict = {}
    stream_usage: dict = {}
    with root.stream("POST", f"{API}/v1/chat/completions",
                     json={**payload, "stream": True}, headers=auth) as resp:
        check("流式 HTTP 200", resp.status_code == 200, f"HTTP {resp.status_code}")
        check("流式 Content-Type", "event-stream" in resp.headers.get("content-type", ""),
              resp.headers.get("content-type", ""))
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            obj = json.loads(data)
            if isinstance(obj.get("timings"), dict) and obj["timings"]:
                timings = obj["timings"]
            if isinstance(obj.get("usage"), dict) and obj["usage"]:
                stream_usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                content_chunks += 1
            if delta.get("reasoning_content"):
                reasoning_chunks += 1
            if (content_chunks or reasoning_chunks) and ttft == 0.0:
                ttft = time.perf_counter() - started
    check("流式收到片段", content_chunks + reasoning_chunks > 0,
          f"正文 {content_chunks} 段 / 思考 {reasoning_chunks} 段")
    check("reasoning_content 已透传", reasoning_chunks > 0, "网关未丢弃思考内容")
    check("流式 usage 已透传", bool(stream_usage),
          f"usage={stream_usage or '缺失（网关未能要求上游返回用量）'}")
    if timings:
        print(f"       首字 {ttft:.2f}s  decode {timings.get('predicted_per_second', 0):.1f} tok/s "
              f"prefill {timings.get('prompt_per_second', 0):.0f} tok/s")
    if timings:
        print(f"       首字 {ttft:.2f}s  decode {timings.get('predicted_per_second', 0):.1f} tok/s "
              f"prefill {timings.get('prompt_per_second', 0):.0f} tok/s")

    print("\n[5] 并发（parallel_slots=2）")
    prompts = ["用一个词说你好", "用一个词说再见", "用一个词说谢谢", "用一个词说明天"]
    # 并发测试只关心是否都能成功返回，把输出预算压小以免每请求耗掉上百 token
    conc_payload = {**payload, "max_tokens": 64}

    def ask(text: str) -> tuple[int, float]:
        t0 = time.perf_counter()
        rr = root.post(f"{API}/v1/chat/completions",
                       json={**conc_payload, "messages": [{"role": "user", "content": text}]},
                       headers=auth, timeout=300)
        return rr.status_code, time.perf_counter() - t0

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(ask, prompts))
    wall = time.perf_counter() - started
    codes = [c for c, _ in outcomes]
    check("4 并发全部成功", all(c == 200 for c in codes), f"HTTP {codes} 总耗时 {wall:.2f}s")

    print("\n[6] 运维接口")
    r = root.get(f"{API}/stats")
    stats = r.json() if r.status_code == 200 else {}
    check("/stats 可读", r.status_code == 200)
    if stats:
        print(f"       请求总数={stats.get('requests_total')} 失败={stats.get('requests_failed')} "
              f"活跃={stats.get('requests_active')}")
        print(f"       token: in={stats.get('prompt_tokens_total')} "
              f"out={stats.get('completion_tokens_total')}")
        print(f"       延迟 P50={stats.get('latency_p50_seconds')}s "
              f"P95={stats.get('latency_p95_seconds')}s")
        check("请求计数 > 0", (stats.get("requests_total") or 0) > 0, str(stats.get("requests_total")))
        check("无请求泄漏（活跃数归零）", stats.get("requests_active") == 0,
              f"活跃={stats.get('requests_active')}")
        check("token 统计已累计", (stats.get("completion_tokens_total") or 0) > 0,
              f"out={stats.get('completion_tokens_total')}")

    r = root.get(f"{API}/metrics")
    check("/metrics Prometheus 格式", r.status_code == 200 and "ornith_requests_total" in r.text)

    r = root.get(f"{API}/")
    check("/ 根路径", r.status_code == 200, r.text[:120])

    r = root.get(f"{API}/v1/unknown-endpoint", headers=auth)
    check("未知接口返回 404", r.status_code == 404, f"HTTP {r.status_code}")

    print("\n[7] 关闭")
    server.should_exit = True
    thread.join(timeout=30)
    time.sleep(1)
    check("进程已停止", not thread.is_alive())

    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        for name in FAIL:
            print(f"  失败: {name}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
