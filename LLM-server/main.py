"""本地 GGUF 推理服务入口（MoE 感知）。

子命令：
    serve     启动内网 API 服务（默认，包含 llama.cpp 后端）
    backend   只启动 llama.cpp 后端（不含网关）
    chat      交互式命令行对话（直连本机后端）
    bench     基准测试：提示处理速度与生成速度
    doctor    环境自检：GPU / llama.cpp 运行时 / MoE 摆位预算
    models    列出可用后端与配置摘要

MoE 摆位（本项目核心）：
    专家权重（``*_exps``）放内存、注意力与 KV cache 放显存。用
    ``--cpu-moe`` / ``--n-cpu-moe N`` 控制，用 ``doctor`` 查看精确预算。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 允许直接 `python main.py` 运行（把 src 加进 sys.path）
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows 控制台默认 GBK，中文输出会乱码；统一切到 UTF-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from llm_server.config import (  # noqa: E402
    PROJECT_ROOT,
    ServerConfig,
    find_free_port,
    local_ip_addresses,
    port_is_free,
)


def _load_config(args: argparse.Namespace) -> ServerConfig:
    """装载配置：默认值 -> 配置文件 -> 命令行。

    注意：并非所有选项都存在于每个子命令的命名空间（例如 ``--api-key``
    只属于 ``serve``），因此一律用 ``getattr`` 取值。直接访问属性会在
    ``doctor``、``bench`` 等子命令下抛 AttributeError。
    """
    cfg = ServerConfig.load(args.config)

    # 命令行覆盖
    if getattr(args, "model", None):
        cfg.model_path = args.model
    if getattr(args, "alias", None):
        cfg.model_alias = args.alias
    if getattr(args, "llama_dir", None):
        cfg.llama_dir = args.llama_dir
    if getattr(args, "ctx", None) is not None:
        cfg.context_size = args.ctx
    if getattr(args, "parallel", None) is not None:
        cfg.parallel_slots = args.parallel
    if getattr(args, "ngl", None) is not None:
        cfg.gpu_layers = args.ngl
    if getattr(args, "gpu_layers_cpu", None) is not None:
        cfg.gpu_layers = max(1, cfg.gpu_layers - args.gpu_layers_cpu)
    if getattr(args, "kv_offload", None) is not None:
        cfg.kv_offload = args.kv_offload
    if getattr(args, "cpu_moe", None) is not None:
        cfg.cpu_moe = args.cpu_moe
    if getattr(args, "n_cpu_moe", None) is not None:
        cfg.n_cpu_moe = args.n_cpu_moe
    if getattr(args, "kv_type", None):
        cfg.kv_cache_type_k = args.kv_type
        cfg.kv_cache_type_v = args.kv_type
    if getattr(args, "ubatch", None) is not None:
        cfg.ubatch_size = args.ubatch
    if getattr(args, "batch", None) is not None:
        cfg.batch_size = args.batch
    if getattr(args, "allow_large_ubatch", None) is not None:
        cfg.allow_large_ubatch = args.allow_large_ubatch
    if getattr(args, "threads", None) is not None:
        cfg.threads = args.threads
    if getattr(args, "load_mode", None):
        # --load-mode 取代了旧的 --no-mmap / --mlock
        cfg.load_mode = args.load_mode
    if getattr(args, "autostart", None) is not None:
        # --no-autostart：只起服务与控制台，模型留到网页里手动选
        cfg.autostart_backend = args.autostart
    if getattr(args, "api_key", None) is not None:
        cfg.api_key = args.api_key
    if getattr(args, "port", None) is not None:
        cfg.proxy_port = args.port
    if getattr(args, "backend_port", None) is not None:
        cfg.backend_port = args.backend_port
    if getattr(args, "host", None):
        cfg.proxy_host = args.host

    problems = cfg.validate()
    if problems:
        print("配置有问题：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(2)
    for w in cfg.warnings():
        print(f"[告警] {w}", file=sys.stderr)
    return cfg


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
def _refuse_if_already_running(cfg: ServerConfig) -> int | None:
    """已有实例在跑就拒绝启动，返回退出码；可以启动则返回 None。

    为什么必须拦：``--load-mode none`` 每次加载都要把约 14.6 GiB 的专家权重
    锁进 CUDA_Host 锁页内存。两个实例并发加载时，31 GiB 的机器根本锁不下
    两份，其中一个会以 ``unable to allocate CUDA_Host buffer`` 失败。
    实测踩过这个坑——排查时以为是显存不足，其实是两个实例在抢内存。
    """
    import httpx

    from llm_server.config import port_is_free

    # 只认**我们自己**的服务：必须返回带这些字段的 JSON 才算。
    # 不能"端口上有东西响应就拒绝"——实测踩过误判：端口上有个返回空体的东西
    # （前一次实例正在退出的残留），旧逻辑直接拒绝启动，把正常启动也挡了。
    url = f"http://127.0.0.1:{cfg.proxy_port}/health"
    data = None
    try:
        # trust_env=False：本机若有系统代理（Clash/v2ray 之类），httpx 默认会
        # 把 127.0.0.1 的请求也发给代理并拿到 502，这里会被误判成"连不上"。
        resp = httpx.get(url, timeout=3.0, trust_env=False)
        data = resp.json()
    except (httpx.HTTPError, ValueError):
        data = None

    if isinstance(data, dict) and {"ready", "backend_running"} <= set(data):
        print(f"检测到端口 {cfg.proxy_port} 上已有本服务在运行：")
        print(f"  ready={data.get('ready')} pid={data.get('pid')} "
              f"ctx={data.get('context_size')}")
        if data.get("gpu"):
            print(f"  显存空闲 {data['gpu'][0].get('free_gib')} GiB")
        print()
        print("拒绝启动第二个实例：两个实例会各加载一份 16.9 GiB 的模型，")
        print("而 --load-mode none 每次加载需锁定约 14.6 GiB 锁页内存，")
        print("并发时必然有一个因内存锁定失败而起不来。")
        print()
        print("请选择：")
        print("  1) 直接用现有服务（无需重启）")
        print("  2) 先停掉它：stop_server.bat —— 然后重新启动")
        print(f"  3) 换端口启动：start_server.bat --port {cfg.proxy_port + 1}")
        return 2

    if not port_is_free(cfg.proxy_host, cfg.proxy_port):
        # 端口被别的程序占着。不在这里硬拒——交给 uvicorn 报权威错误，
        # 这里只给一句提示（可能是刚退出的实例还没完全释放）。
        print(
            f"[提示] 端口 {cfg.proxy_port} 当前不可用，可能是别的程序占用，"
            f"或上一个实例正在退出。若启动失败请换端口：--port {cfg.proxy_port + 1}",
            file=sys.stderr,
        )
    return None


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from llm_server.api import create_app
    from llm_server.core import make_backend

    cfg = _load_config(args)

    # 网关端口被占用时自动顺延，避免"启动即失败"
    if args.auto_port:
        try:
            cfg.proxy_port = find_free_port(cfg.proxy_host, cfg.proxy_port)
        except RuntimeError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2
    else:
        # 防呆：端口被占用说明大概率已经有一个实例在跑。
        # 继续启动会很危险——两个实例会各自加载一份模型，而 --load-mode none
        # 每次加载都要锁定约 14.6 GiB 锁页内存，两个并发必然有一个失败
        # （实测就是这样撞出 "unable to allocate CUDA_Host buffer" 的）。
        rc = _refuse_if_already_running(cfg)
        if rc is not None:
            return rc

    backend = make_backend(cfg)
    print(f"[后端] {backend.backend.describe()}")

    app = create_app(cfg, backend)

    uvicorn.run(
        app,
        host=cfg.proxy_host,
        port=cfg.proxy_port,
        log_level=args.log_level,
        access_log=False,  # 由 MetricsMiddleware 统一记录
        timeout_keep_alive=75,
    )
    return 0


# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
def cmd_backend(args: argparse.Namespace) -> int:
    from llm_server.core import make_backend

    cfg = _load_config(args)
    server = make_backend(cfg)
    print(f"[后端] {server.backend.describe()}")
    print(f"[命令] {server.command_line()}")
    print()
    try:
        profile = server.start()
    except RuntimeError as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1

    print(f"已就绪：{cfg.backend_base_url}")
    print(f"  上下文 {profile.context_size} / 每槽 {cfg.ctx_per_slot}")
    print(f"  日志 {server.log_path}")
    print("按 Ctrl+C 停止…")
    try:
        while server.is_running:
            server.process.wait(timeout=1)  # type: ignore[union-attr]
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception:
        pass
    finally:
        server.stop()
    return 0


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------
def cmd_chat(args: argparse.Namespace) -> int:
    """轻量交互式对话，用于不启网关时快速验证模型。"""
    from llm_server.core import make_backend

    cfg = _load_config(args)
    server = make_backend(cfg)

    owns_process = False
    if not server.wait_ready(timeout=3):
        print("后端未运行，正在启动…")
        try:
            server.start()
            owns_process = True
        except RuntimeError as exc:
            print(f"启动失败：{exc}", file=sys.stderr)
            return 1

    import httpx

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"模型 {cfg.model_alias} 已就绪，输入内容开始对话（exit 退出，/clear 清空历史）")
    try:
        with httpx.Client(timeout=cfg.request_timeout, trust_env=False) as client:
            while True:
                try:
                    user = input("\n你 > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not user:
                    continue
                if user.lower() in ("exit", "quit", "/exit"):
                    break
                if user == "/clear":
                    messages = messages[:1] if args.system else []
                    print("（历史已清空）")
                    continue

                messages.append({"role": "user", "content": user})
                payload = {
                    "messages": messages,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "stream": True,
                }
                print("\nAI > ", end="", flush=True)
                answer: list[str] = []
                thinking_shown = False
                try:
                    with client.stream(
                        "POST", f"{cfg.backend_base_url}/v1/chat/completions", json=payload
                    ) as resp:
                        if resp.status_code >= 400:
                            print(f"[错误 {resp.status_code}] {resp.read().decode('utf-8', 'replace')}")
                            messages.pop()
                            continue
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
                            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
                            # 该模型先输出思考内容，再输出正文；两者都要显示，
                            # 否则用户会看到"没反应"。
                            think = delta.get("reasoning_content") or ""
                            piece = delta.get("content") or ""
                            if think:
                                if not thinking_shown:
                                    print("\033[90m[思考] ", end="")
                                    thinking_shown = True
                                print(f"\033[90m{think}\033[0m", end="", flush=True)
                            if piece:
                                if thinking_shown:
                                    print("\n\n[回答] ", end="")
                                    thinking_shown = False
                                answer.append(piece)
                                print(piece, end="", flush=True)
                except httpx.HTTPError as exc:
                    print(f"\n[连接错误] {exc}")
                    messages.pop()
                    continue
                print()
                messages.append({"role": "assistant", "content": "".join(answer)})
                if args.max_turns and len(messages) > args.max_turns * 2 + 1:
                    messages = messages[-args.max_turns * 2 :]
    finally:
        if owns_process:
            server.stop()
    return 0


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
def cmd_bench(args: argparse.Namespace) -> int:
    from llm_server.bench import run_benchmark

    cfg = _load_config(args)
    return run_benchmark(cfg, warmup=not args.no_warmup, repeats=args.repeats, verbose=True)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    from llm_server.core import (
        dump_environment,
        estimate_placement,
        max_experts_on_gpu,
        model_shape,
        query_gpus,
        tensor_breakdown,
    )

    cfg = ServerConfig.load(args.config)
    print(dump_environment())

    print()
    print("[模型结构]")
    shape = model_shape(cfg.model_file) if cfg.model_file.is_file() else None
    if shape is None:
        print(f"  模型文件不存在: {cfg.model_file}")
    else:
        print(f"  文件           : {cfg.model_file}")
        print(f"  大小           : {shape.file_size_gib:.2f} GiB")
        print(f"  {shape.describe()}")

    print()
    print("[配置]")
    print(f"  上下文         : {cfg.context_size} (每槽 {cfg.ctx_per_slot}, parallel={cfg.parallel_slots})")
    print(f"  KV cache 量化  : {cfg.kv_cache_type_k}/{cfg.kv_cache_type_v}")
    print(f"  KV cache 位置  : {cfg.kv_placement}")
    print(f"  专家权重位置   : {cfg.moe_placement}")
    print(f"  flash-attn     : {cfg.flash_attention}")
    print(f"  GPU 层数       : {cfg.gpu_layers}")
    print(f"  ubatch / batch : {cfg.effective_ubatch} / {max(cfg.batch_size, cfg.effective_ubatch)}"
          + (f"  ⚠ 配置的 {cfg.ubatch_size} 超过安全上限 {ServerConfig.SAFE_UBATCH}，已夹紧"
             if cfg.ubatch_clamped else ""))
    print(f"  CPU 线程       : {cfg.threads}")
    print(f"  加载模式       : {cfg.effective_load_mode or 'llama.cpp 默认(auto=mmap)'}"
          + ("   ⚠ 需一次性锁定约 14.6 GiB 锁页内存，失败会自动回落 mmap"
             if cfg.effective_load_mode == "none" else ""))
    print(f"  网关监听       : {cfg.proxy_host}:{cfg.proxy_port}")
    print(f"  后端监听       : {cfg.backend_host}:{cfg.backend_port}")
    print(f"  API Key        : {'已设置' if cfg.api_key else '未设置'}")

    if shape is not None and shape.usable:
        print()
        print("[显存/内存预算]（按 GGUF 张量索引精确计算，非估算假设）")
        breakdown = tensor_breakdown(cfg.model_file, shape)
        budget = estimate_placement(
            shape, breakdown,
            context=cfg.context_size,
            kv_quant=cfg.kv_cache_type_k,
            cpu_moe=cfg.cpu_moe,
            n_cpu_moe=cfg.n_cpu_moe,
            parallel=cfg.parallel_slots,
            ubatch_size=cfg.effective_ubatch,
        )
        print(f"  常住权重(注意力/SSM/embedding) : {budget['resident_gib']:>6.2f} GiB → 显存")
        print(f"  专家权重                        : {budget['experts_ram_gib']:>6.2f} GiB → 内存"
              f"   |   {budget['experts_vram_gib']:.2f} GiB → 显存")
        print(f"  KV cache ({cfg.context_size}, {cfg.kv_cache_type_k}, "
              f"仅 {shape.attention_layers} 个全注意力层) : {budget['kv_cache_gib']:>6.2f} GiB → "
              f"{cfg.kv_placement}")
        print(f"  计算缓冲 (ubatch {cfg.ubatch_size})            : "
              f"{budget['compute_buffer_gib']:>6.2f} GiB → 显存")
        print(f"  ── 显存合计                     : {budget['vram_total_gib']:>6.2f} GiB")
        print(f"  ── 内存需容纳专家               : {budget['ram_experts_gib']:>6.2f} GiB")

        if breakdown.tensor_count:
            print()
            print(f"  {breakdown.describe()}")
            if breakdown.mtp_bytes:
                print(f"  注意：文件里有 {breakdown.mtp_bytes / 1024**3:.2f} GiB 的 MTP/nextn "
                      f"张量（blk.{shape.compute_layers}+），llama.cpp 会忽略，属死重。")

        gpus = query_gpus()
        if gpus:
            free = gpus[0].free_gib
            print()
            print(f"  当前可用显存                    : {free:>6.2f} GiB")
            if budget["vram_total_gib"] < free - 0.3:
                head = free - budget["vram_total_gib"]
                room = max_experts_on_gpu(
                    shape, breakdown,
                    context=cfg.context_size, kv_quant=cfg.kv_cache_type_k,
                    free_vram_gib=free, ubatch_size=cfg.effective_ubatch,
                )
                print(f"  判断                            : 可以加载（余量 {head:.2f} GiB）")
                if room > 0 and cfg.n_cpu_moe < 0:
                    print(f"  可优化                          : 预算上还能放约 {room} 层专家进显存，"
                          f"可试 --n-cpu-moe {shape.compute_layers - room}")
                    print(f"  注意                            : 显存天花板很硬，实测 "
                          f"--n-cpu-moe 36（4 层）正常、32（8 层）会把 decode 从 31 打到 15。"
                          f"改动后请用 scripts/stress_ctx.py 复验。")
            else:
                print(f"  判断                            : 显存偏紧，启动时可能触发自动降级")

        if shape.is_moe:
            print()
            print("  同上下文下不同 KV 量化 / 摆位对比：")
            for ctx in (32768, 65536, 131072):
                row = [f"    {ctx:>7}:"]
                for quant in ("q8_0", "q4_0"):
                    row.append(f"{quant} KV {shape.kv_gib(ctx, quant):.2f} GiB")
                print("  ".join(row))

    problems = cfg.validate()
    print()
    for w in cfg.warnings():
        print(f"[告警] {w}")
    if problems:
        print("[配置问题]")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[结论] 配置可用。")
    return 0


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def cmd_models(args: argparse.Namespace) -> int:
    from llm_server.core import discover_backends

    print("[可用 llama.cpp 运行时]")
    backends = discover_backends()
    if not backends:
        print("  未找到。请安装 LM Studio 或用 --llama-dir 指定目录。")
        return 1
    for i, b in enumerate(backends):
        print(f"  [{i}]{' (默认)' if i == 0 else ''} {b.describe()}")
        print()
    return 0


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def _add_common_options(parser: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    """加入所有子命令都接受的公共选项。

    主解析器与子解析器上都会挂这些选项，从而
    ``main.py --ctx 32768 serve`` 与 ``main.py serve --ctx 32768`` 都可用
    （用户几乎总会写在子命令后面）。

    ``suppress=True`` 用于子解析器：未显式给出该选项时不写入命名空间，
    否则子解析器的 ``None`` 默认值会覆盖主解析器已经解析出的值，
    导致写在子命令前面的参数被静默忽略。
    """
    d = argparse.SUPPRESS if suppress else None

    def opt(*names: str, **kwargs) -> None:
        if suppress:
            kwargs["default"] = argparse.SUPPRESS
        parser.add_argument(*names, **kwargs)

    opt("--config", default=d, help="配置文件路径（YAML 或 JSON）")
    opt("--model", default=d, help="GGUF 模型文件路径")
    opt("--alias", default=d, help="对外暴露的模型名")
    opt("--llama-dir", default=d, help="llama.cpp 目录或 llama-server 路径")
    opt("--ctx", type=int, default=d, help="总上下文长度")
    opt("--parallel", type=int, default=d, help="并发槽位数")
    opt("--ngl", type=int, default=d, help="卸载到 GPU 的层数")
    opt("--gpu-layers-cpu", type=int, default=d,
        help="留多少层给 CPU（在 ngl 基础上相减）")
    opt("--kv-offload", dest="kv_offload", action="store_true", default=d,
        help="把 KV cache 放在显存（MoE 混合架构下 128K 仅约 1.3 GiB，默认显存）")
    opt("--no-kv-offload", dest="kv_offload", action="store_false",
        help="把 KV cache 放在内存")
    opt("--cpu-moe", dest="cpu_moe", action="store_true", default=d,
        help="把全部 MoE 专家权重放在内存（默认）")
    opt("--no-cpu-moe", dest="cpu_moe", action="store_false",
        help="让专家权重跟随 --ngl 上显存")
    opt("--n-cpu-moe", type=int, default=d, metavar="N",
        help="只把前 N 层的专家权重放内存，其余层专家放显存（显存有余量时换速度）")
    opt("--kv-type", default=d, choices=["f16", "bf16", "q8_0", "q4_0", "q4_1",
                                         "iq4_nl", "q5_0", "q5_1"],
        help="KV cache 量化类型，默认 q8_0")
    opt("--ubatch", type=int, default=d, help="物理批大小，调小可降显存峰值")
    opt("--batch", type=int, default=d, help="逻辑批大小")
    opt("--allow-large-ubatch", dest="allow_large_ubatch", action="store_true", default=d,
        help="放开 ubatch 安全上限（2048）。实测 4096 会触发 CUDA 越界崩溃，仅供实验")
    opt("--threads", type=int, default=d,
        help="CPU 线程数（专家权重在内存时，prefill 速度主要取决于它）")
    opt("--load-mode", default=d,
        choices=["auto", "mmap", "mlock", "mmap+mlock", "none"],
        help="模型加载模式。none 更快但需要一次性锁定约 14.6 GiB 锁页内存，"
             "失败时程序会自动回落到 mmap")
    opt("--autostart", dest="autostart", action="store_true", default=d,
        help="启动服务时立即加载配置里的模型（默认行为）")
    opt("--no-autostart", dest="autostart", action="store_false",
        help="**只起服务和控制台，不加载模型**；模型在网页控制台里手动选。"
             "启动只需 1 秒，换模型也不用重启服务")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="LLM-server",
        description="本地 GGUF 高速推理 + 内网 OpenAI 兼容 API（支持 MoE 专家摆位）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    _add_common_options(parser)

    sub = parser.add_subparsers(dest="command")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text)
        _add_common_options(child, suppress=True)
        return child

    p_serve = add("serve", "启动内网 API 服务（默认）")
    p_serve.add_argument("--host", default=None, help="监听地址，默认取配置文件（0.0.0.0）")
    p_serve.add_argument("--port", type=int, default=None, help="监听端口，默认 8000")
    p_serve.add_argument("--backend-port", type=int, default=None, help="llama.cpp 后端端口")
    p_serve.add_argument("--api-key", default=None, help="内网 API Key，留空则免鉴权")
    p_serve.add_argument("--auto-port", action="store_true", help="端口被占用时自动顺延")
    p_serve.add_argument("--log-level", default="info", help="uvicorn 日志级别")
    p_serve.set_defaults(func=cmd_serve)

    add("backend", "只启动 llama.cpp 后端").set_defaults(func=cmd_backend)

    p_chat = add("chat", "命令行对话")
    p_chat.add_argument("--system", default=None, help="系统提示词")
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--max-tokens", type=int, default=1024)
    p_chat.add_argument("--max-turns", type=int, default=0, help="保留最近 N 轮上下文，0 不限制")
    p_chat.set_defaults(func=cmd_chat)

    p_bench = add("bench", "基准测试")
    p_bench.add_argument("--repeats", type=int, default=3)
    p_bench.add_argument("--no-warmup", action="store_true")
    p_bench.set_defaults(func=cmd_bench)

    add("doctor", "环境自检").set_defaults(func=cmd_doctor)
    add("models", "列出可用后端").set_defaults(func=cmd_models)

    return parser


COMMANDS = ("serve", "backend", "chat", "bench", "doctor", "models")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)

    # 没给子命令时默认进入 serve。
    # 必须先判断"有没有子命令关键字"再解析，不能先 parse 一次：
    # `--port 9200` 这种写法在第一次解析时会把 "9200" 当成子命令名而报错。
    if not any(tok in COMMANDS for tok in raw):
        # 把 serve 插到第一个选项之前，这样 `--port 9000` 这类 serve
        # 专属选项才会被交给 serve 子解析器而不是主解析器直接消费。
        insert_at = next((i for i, tok in enumerate(raw) if tok.startswith("-")), len(raw))
        raw = [*raw[:insert_at], "serve", *raw[insert_at:]]

    args = parser.parse_args(raw)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
