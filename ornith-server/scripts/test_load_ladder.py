"""加载降级阶梯的单元测试（不加载模型，秒级完成）。

覆盖两件容易出错、且出错代价很大的事：

1. **宿主内存失败 vs 显存不足的区分**。``--load-mode none`` 会把 --cpu-moe 的
   专家张量放进 CUDA_Host 锁页内存（约 14.6 GiB）。分配失败时日志里同时含
   "failed to allocate"，若先按显存 OOM 判，就会去降 ubatch —— 病因不在显存，
   治不好。实测踩过这个坑。

2. **降级动作是否正确**。宿主内存失败要回落 mmap 并保留其余参数；显存不足才
   应该降 ubatch。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ornith_server.config import ServerConfig  # noqa: E402
from ornith_server.core import server as srv  # noqa: E402

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


# 实测抓到的真实日志（--load-mode none + 16 GiB 内存紧张时）
HOST_MEM_LOG = """
0.00.037.251 I srv    load_model: loading model 'cyber-tiel.gguf'
0.01.916.978 E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 15704850432
0.01.916.983 E ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA_Host buffer of size 15704850432
0.02.024.393 E llama_model_load: error loading model: unable to allocate CUDA_Host buffer
0.02.024.451 E cmn  common_init_: failed to load model
"""

# 真实的显存不足日志
VRAM_OOM_LOG = """
0.10.941.402 E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2054.28 MiB on device 0: cudaMalloc failed: out of memory
0.10.941.513 E ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 2154070144
0.10.955.328 E llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
"""


def main() -> int:
    print("[1] 失败原因分类")
    check("宿主内存失败被识别", srv._looks_like_host_mem_failure(HOST_MEM_LOG), True)
    check("宿主内存日志不应被当成显存不足",
          srv._looks_like_host_mem_failure(HOST_MEM_LOG)
          and not srv._looks_like_host_mem_failure(VRAM_OOM_LOG), True)
    check("显存不足被识别", srv._looks_like_oom(VRAM_OOM_LOG), True)
    check("显存 OOM 日志不误判为宿主内存",
          not srv._looks_like_host_mem_failure(VRAM_OOM_LOG), True)

    print("\n[2] 降级阶梯的构造")
    cfg = ServerConfig()
    cfg.load_mode = "none"
    cfg.ubatch_size = 2048
    profiles = srv.build_load_profiles(cfg)
    check("首个 profile 用配置的加载模式", profiles[0].load_mode, "none")
    check("首个 profile 用配置的 ubatch", profiles[0].ubatch_size, 2048)
    check("阶梯里有更小的 ubatch 兜底",
          any(p.ubatch_size < 2048 for p in profiles), True)
    check("cfg.effective_load_mode 兼容旧字段", cfg.effective_load_mode, "none")

    legacy = ServerConfig()
    legacy.load_mode = ""
    legacy.no_mmap = True
    check("no_mmap=True 等价于 load_mode=none", legacy.effective_load_mode, "none")

    print("\n[3] 宿主内存失败 → 回落 mmap，且不降 ubatch")
    cfg2 = ServerConfig()
    cfg2.load_mode = "none"
    cfg2.ubatch_size = 2048
    backend = srv.LlamaBackendServer(cfg2, backend=None)  # type: ignore[arg-type]

    seen: list[tuple[int, str]] = []

    def fake_attempt(prof):
        seen.append((prof.ubatch_size, prof.load_mode))
        if len(seen) == 1:
            return "host_mem", "模拟：unable to allocate CUDA_Host buffer"
        return "ready", ""

    backend._attempt = fake_attempt  # type: ignore[method-assign]
    backend._cleanup_process = lambda **kw: None  # type: ignore[method-assign]

    prof = backend.start(profiles=profiles)

    check("第一次尝试的参数", seen[0], (2048, "none"))
    check("第二次尝试回落 mmap", seen[1], (2048, "mmap"))
    check("回落时保持同一 ubatch（不误降）", seen[1][0], seen[0][0])
    check("最终采用的 profile 是 mmap", prof.load_mode, "mmap")
    check("说明里写明了回落原因", "mmap" in prof.note, True)

    print("\n[4] 显存不足 → 正常降 ubatch（不应触发 mmap 回落）")
    cfg3 = ServerConfig()
    cfg3.load_mode = "none"
    cfg3.ubatch_size = 2048
    backend3 = srv.LlamaBackendServer(cfg3, backend=None)  # type: ignore[arg-type]

    seen3: list[tuple[int, str]] = []

    def fake_attempt3(prof):
        seen3.append((prof.ubatch_size, prof.load_mode))
        if len(seen3) == 1:
            return "oom", "模拟：cudaMalloc failed: out of memory"
        return "ready", ""

    backend3._attempt = fake_attempt3  # type: ignore[method-assign]
    backend3._cleanup_process = lambda **kw: None  # type: ignore[method-assign]
    prof3 = backend3.start(profiles=srv.build_load_profiles(cfg3))

    check("第一次尝试", seen3[0], (2048, "none"))
    check("第二次是降 ubatch 而非换 mmap", seen3[1][0] < 2048, True)
    check("真正降级时仍保留 none", seen3[1][1], "none")
    check("最终 profile", prof3.ubatch_size < 2048, True)

    print("\n" + "=" * 60)
    print(f"通过 {PASSED} 项，失败 {len(FAILED)} 项")
    for name in FAILED:
        print(f"  失败: {name}")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
