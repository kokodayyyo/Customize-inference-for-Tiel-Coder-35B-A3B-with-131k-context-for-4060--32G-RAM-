"""命令行参数解析的单元测试（不启动服务，秒级完成）。

覆盖：参数写在子命令前后、默认子命令注入、布尔开关、config 文件与
命令行的优先级。这些是极易出错且只靠手工试很难发现的地方。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import main as cli  # noqa: E402

FAILED: list[str] = []
PASSED = 0


def check(name: str, actual, expected) -> None:
    global PASSED
    if actual == expected:
        PASSED += 1
        print(f"  [通过] {name}: {actual!r}")
    else:
        FAILED.append(name)
        print(f"  [失败] {name}: 实际 {actual!r}，期望 {expected!r}")


def parse(argv: list[str]):
    """复刻 main() 里的解析逻辑（含默认子命令注入）。"""
    parser = cli.build_parser()
    raw = list(argv)
    if not any(tok in cli.COMMANDS for tok in raw):
        insert_at = next((i for i, tok in enumerate(raw) if tok.startswith("-")), len(raw))
        raw = [*raw[:insert_at], "serve", *raw[insert_at:]]
    return parser.parse_args(raw)


def load_cfg(argv: list[str]):
    """复刻 main() 里的配置装载逻辑。"""
    args = parse(argv)
    return cli._load_config(args), args


def main() -> int:
    print("[1] 参数写在子命令后面")
    for argv, expected_ctx in (
        (["serve", "--ctx", "32768"], 32768),
        (["doctor", "--ctx", "16384"], 16384),
        (["bench", "--ctx", "4096"], 4096),
    ):
        cfg, _ = load_cfg(argv)
        check(f"{' '.join(argv)} -> context_size", cfg.context_size, expected_ctx)

    print("\n[2] 参数写在子命令前面")
    for argv, expected_ctx in (
        (["--ctx", "32768", "serve"], 32768),
        (["--ctx", "16384", "doctor"], 16384),
    ):
        cfg, _ = load_cfg(argv)
        check(f"{' '.join(argv)} -> context_size", cfg.context_size, expected_ctx)

    print("\n[3] 不带子命令（应默认 serve）")
    args = parse(["--ctx", "8192"])
    check("command", args.command, "serve")
    cfg, _ = load_cfg(["--ctx", "8192"])
    check("context_size", cfg.context_size, 8192)

    print("\n[4] 布尔开关")
    for argv, expected in (
        (["serve", "--kv-offload"], True),
        (["serve", "--no-kv-offload"], False),
        (["doctor", "--kv-offload"], True),
        (["--kv-offload", "serve"], True),
    ):
        cfg, _ = load_cfg(argv)
        check(f"{' '.join(argv)} -> kv_offload", cfg.kv_offload, expected)
    # 未指定时沿用配置文件的值（config/server.yaml 里 MoE 混合架构默认 true）
    cfg, _ = load_cfg(["serve"])
    check("serve（未指定）-> kv_offload", cfg.kv_offload, True)

    print("\n[5] MoE 专家摆位")
    cfg, _ = load_cfg(["serve"])
    check("serve（未指定）-> cpu_moe", cfg.cpu_moe, True)
    check("serve（未指定）-> n_cpu_moe", cfg.n_cpu_moe, -1)

    cfg, _ = load_cfg(["serve", "--no-cpu-moe"])
    check("--no-cpu-moe -> cpu_moe", cfg.cpu_moe, False)

    cfg, _ = load_cfg(["serve", "--n-cpu-moe", "32"])
    check("--n-cpu-moe 32 -> n_cpu_moe", cfg.n_cpu_moe, 32)

    cfg, _ = load_cfg(["--cpu-moe", "serve"])
    check("--cpu-moe 写在子命令前", cfg.cpu_moe, True)

    cfg, _ = load_cfg(["serve", "--n-cpu-moe", "30", "--cpu-moe"])
    check("n_cpu_moe 与 cpu_moe 可共存（前者优先）", (cfg.n_cpu_moe, cfg.cpu_moe), (30, True))

    print("\n[6] 其余公共选项")
    cfg, _ = load_cfg(["serve", "--parallel", "4", "--ngl", "32", "--alias", "my-model"])
    check("parallel_slots", cfg.parallel_slots, 4)
    check("gpu_layers", cfg.gpu_layers, 32)
    check("model_alias", cfg.model_alias, "my-model")

    cfg, _ = load_cfg(["serve", "--gpu-layers-cpu", "4"])
    check("--gpu-layers-cpu 在 99 基础上相减", cfg.gpu_layers, 95)

    cfg, _ = load_cfg(["serve", "--kv-type", "f16"])
    check("--kv-type 同时设置 K 和 V", (cfg.kv_cache_type_k, cfg.kv_cache_type_v), ("f16", "f16"))

    cfg, _ = load_cfg(["serve", "--threads", "24", "--ubatch", "1024", "--batch", "4096"])
    check("threads", cfg.threads, 24)
    check("ubatch_size", cfg.ubatch_size, 1024)
    check("batch_size", cfg.batch_size, 4096)

    cfg, _ = load_cfg(["serve", "--load-mode", "none"])
    check("load_mode", cfg.load_mode, "none")

    print("\n[7] ubatch 安全上限（4096 会触发 CUDA 越界崩溃）")
    cfg, _ = load_cfg(["serve"])
    check("默认 ubatch", cfg.ubatch_size, 2048)
    check("默认不再夹紧", cfg.ubatch_clamped, False)

    cfg, _ = load_cfg(["serve", "--ubatch", "4096"])
    check("--ubatch 4096 实际生效值被夹到", cfg.effective_ubatch, 2048)
    check("--ubatch 4096 标记为已夹紧", cfg.ubatch_clamped, True)
    check("--ubatch 4096 会产生告警但不阻断启动",
          any("ubatch" in w for w in cfg.warnings()) and not cfg.validate(), True)

    cfg, _ = load_cfg(["serve", "--ubatch", "4096", "--allow-large-ubatch"])
    check("放开后保留 4096", cfg.effective_ubatch, 4096)
    check("放开后不再告警", any("ubatch" in w for w in cfg.warnings()), False)

    cfg, _ = load_cfg(["serve", "--ubatch", "1024"])
    check("--ubatch 1024 原样保留", cfg.effective_ubatch, 1024)

    print("\n[8] serve 专属选项")
    cfg, args = load_cfg(["serve", "--port", "9100", "--api-key", "sk-x", "--host", "127.0.0.1"])
    check("proxy_port", cfg.proxy_port, 9100)
    check("api_key", cfg.api_key, "sk-x")
    check("proxy_host", cfg.proxy_host, "127.0.0.1")

    print("\n[9] 无子命令时的 serve 专属选项")
    cfg, args = load_cfg(["--port", "9200"])
    check("command", args.command, "serve")
    check("proxy_port", cfg.proxy_port, 9200)

    print("\n[10] MoE 摆位描述（用于日志/横幅）")
    cfg, _ = load_cfg(["serve"])
    check("moe_placement（默认全内存）", "内存" in cfg.moe_placement, True)
    cfg, _ = load_cfg(["serve", "--n-cpu-moe", "32"])
    check("moe_placement（分层）", "32" in cfg.moe_placement, True)
    cfg, _ = load_cfg(["serve", "--no-cpu-moe"])
    check("moe_placement（全显存）", cfg.moe_placement, "专家权重不动（跟随 --n-gpu-layers，显存优先）")

    print("\n[11] 只起控制台（--no-autostart，start_server.bat 用的就是这个）")
    cfg, _ = load_cfg(["serve"])
    check("默认自动加载模型", cfg.autostart_backend, True)
    cfg, _ = load_cfg(["serve", "--no-autostart"])
    check("--no-autostart 关闭自动加载", cfg.autostart_backend, False)
    cfg, _ = load_cfg(["--no-autostart", "serve"])
    check("写在子命令前也生效", cfg.autostart_backend, False)
    cfg, _ = load_cfg(["serve", "--autostart"])
    check("--autostart 显式开启", cfg.autostart_backend, True)

    print("\n" + "=" * 60)
    print(f"通过 {PASSED} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败: {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
