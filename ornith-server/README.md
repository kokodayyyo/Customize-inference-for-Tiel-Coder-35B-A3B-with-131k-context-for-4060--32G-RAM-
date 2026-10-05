# Tile-35B-A3B 本地高速推理 + 内网 API 服务

在 **RTX 4060 Laptop 8GB + 31 GiB 内存** 上直接加载
`D:\models\Tile\Tile-35BA3B\Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf`（16.88 GiB），
提供内网可访问的 OpenAI 兼容 API。**默认 128K 上下文**。

核心思路是 **MoE 分层摆位**：

| 部分 | 体积 | 放在哪 |
|---|---|---|
| 专家权重 `*_exps`（256 专家 × 40 层）| **14.12 GiB** | **内存** |
| 注意力 / 线性注意力 / embedding / 共享专家 | **2.38 GiB** | **显存** |
| KV cache（128K，q8_0，只算 10 个全注意力层）| **1.33 GiB** | **显存** |
| MTP/nextn 头 `blk.40` | 0.36 GiB | 被 llama.cpp 忽略（死重）|

一句话结论（本机实测）：

| 场景 | prefill | 首字延迟 | **decode** | 显存 |
|---|---|---|---|---|
| 短对话 | — | 0.4 s | **28.3 tok/s** | 5.31 GiB |
| 6.4K 上下文 | 975 tok/s | 6.6 s | **27.9 tok/s** | 5.35 GiB |
| 25.6K 上下文 | 1,068 tok/s | 24.0 s | **24.4 tok/s** | 5.38 GiB |
| 52.3K 上下文 | 1,019 tok/s | 51.4 s | **22.0 tok/s** | 5.51 GiB |
| 79.8K 上下文 | 961 tok/s | 83.0 s | **21.2 tok/s** | 5.78 GiB |
| **95.8K 上下文** | **928 tok/s** | 103.3 s | **19.9 tok/s** | 5.68 GiB |

**显存占用几乎不受填充量影响**（KV 按 128K 全额预分配），decode 从空载填到
9.6 万 token 只下降 **30%**。这与上一个 9B 稠密模型（下降 66%）形成鲜明对比。

---

## 1. 这个模型和普通 MoE 不一样（最重要的一节）

解析 GGUF 得到的关键结构：

| 参数 | 值 |
|---|---|
| 架构 | `qwen35moe` |
| 层数 | 41（`blk.40` 是 MTP 头，**llama.cpp 整段忽略**，实际计算 **40 层**）|
| 专家 | **256 个 / 每 token 激活 8 个**，专家隐藏维 512 |
| 注意力头 | 16 头 / **2 个 KV 头**（GQA 8:1），head_dim 256 |
| **`full_attention_interval`** | **4** |
| `ssm.*` | conv_kernel 4 / state_size 128 / group_count 16 |
| 训练上下文 | 262144 |

### 1.1 混合注意力 → 128K 的 KV 只要 1.33 GiB

`full_attention_interval = 4` 意味着 **每 4 层里只有 1 层是真注意力**，其余 3 层是
**线性注意力（gated DeltaNet）**，其循环状态大小固定、**不随上下文增长**。

所以 KV cache 只按 **10 个全注意力层**计算，而不是 40 层：

| 上下文 | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 32K | 0.63 GiB | 0.33 | 0.18 |
| 64K | 1.26 | 0.66 | 0.35 |
| **128K** | **2.51** | **1.33** | 0.70 |

**这是与上一个 9B 模型结论完全相反的地方**：那个模型 head_dim 256 且 32 层全是
真注意力，128K 的 KV 要 8.5 GiB，只能放内存；这个模型 1.33 GiB，**放显存又放得下
又更快**。所以本项目 `kv_offload: true`。

顺带验证：`--swa-full` 对显存与速度**没有任何影响**（净增 4505 vs 4519 MiB），
反证这些层不是滑窗注意力，**128K 上下文没有被截断**，是真 128K。

### 1.2 张量账本（不靠估算）

`scripts/gguf_tensors.py` 直接解析 GGUF 张量索引算出精确字节：

```
专家 *_exps           14.123 GiB   → 内存
其余可上显存            2.380 GiB   → 显存
MTP/nextn（被忽略）     0.363 GiB   → 死重
```

量化是 Unsloth 动态混合：IQ3_S 8.59 GiB / IQ4_XS 4.91 / Q8_0 1.93 / Q6_K 1.00 GiB。

**这套账本预测显存 5.39 GiB，实测净增 5518 MiB = 5.39 GiB，误差在舍入范围内。**

---

## 2. 实测性能矩阵（本机真实数据）

测试条件：RTX 4060 Laptop 8GB，`--cpu-moe`，KV q8_0 放显存，128K 上下文，
`--load-mode none`，`--threads 16`。吞吐取 llama.cpp 服务端返回的 `timings`，
测 prefill 时设 `cache_prompt: false` 以规避前缀复用。

### 2.1 ubatch 是 MoE 下最重要的参数

原因：**每个 ubatch 都要把 256 个专家全部过一遍**，所以 ubatch 越大，同样长度的
prompt 需要的往返次数越少，读内存的次数也越少。

| ubatch | decode | **prefill** | 显存净增 | 结论 |
|---|---|---|---|---|
| 256 | 28.49 | 322.2 | 3885 MiB | |
| 512 | 28.54 | 442.9 | 3999 MiB | |
| 1024 | 28.76 | 771.8 | 4181 MiB | |
| **2048** | **29.19** | **1049.4** | 4519 MiB | ✅ **默认**（稳定）|
| 4096 | 28.81 | 1207.2 | 5518 MiB | ❌ **会崩，见 2.4** |
| 8192 | 24.93 | 456.5 | 7006 MiB | ❌ 显存挤爆，反而变慢 |

**prefill 从 443 提到 1049（2.4 倍），decode 几乎不变。**
代价只是计算缓冲从 0.20 GiB 涨到 0.70 GiB。

### 2.2 专家摆位与加载模式

| 用例 | decode | prefill | 显存净增 | 说明 |
|---|---|---|---|---|
| `--threads 8` | 24.40 | 433.9 | 4001 MiB | |
| `--threads 16` | 24.95 | 425.6 | 4025 MiB | 线程数在噪声内 |
| `--threads 24` | 24.59 | 442.1 | 4013 MiB | 同上 |
| **`--load-mode none`** | **29.34** | **452.0** | 4072 MiB | ✅ **+17.6% decode** |
| `--n-cpu-moe 34`（6 层专家上显存）| 24.40 | **77.1** | 5921 MiB | ❌ 页面被换出，改读磁盘 |
| `--n-cpu-moe 36` + ubatch 2048 | **30.93** | 1089.3 | 6066 MiB | ✅ 可选 +6% |
| `--n-cpu-moe 32` + ubatch 2048 | **14.56** | **76.3** | 7235 MiB | ❌ 越过显存天花板 |

三条结论：

1. **线程数不是瓶颈。** 有效访存带宽反推：decode 每 token 读 0.441 GiB 专家权重，
   29.34 × 0.441 = **12.9 GB/s**；prefill 也是约 13.4 GB/s。两者卡在同一个数字上，
   说明瓶颈是 **CPU 侧小批量专家 GEMM 的有效访存**，既不是算力也不是内存带宽
   （DDR5 双通道理论 83 GB/s）。
2. **`--load-mode none` 明确更快**（+17.6%）。llama.cpp 自己也会警告
   `tensor overrides to CPU are used with mmap enabled - consider using
   --load-mode none`。代价是加载 3.5s → 11.6s，且需要 17 GiB 常驻内存。
3. **`--n-cpu-moe` 大部分时候是负收益。** 显存天花板非常硬：总占用 6768 MiB 正常，
   7894 MiB 时 decode 从 30.9 崩到 14.6 tok/s。想榨这 6% 必须**同时**把 ubatch
   降到 2048 留出余量。

### 2.3 上下文填充后的真实速度（中文长提示，单进程连续测）

| 实际 prompt token | prefill | prefill 耗时 | 首字延迟 | decode | 显存 |
|---|---|---|---|---|---|
| 12 | — | 0.4 s | 0.43 s | **28.30 tok/s** | 5314 MiB |
| 6,432 | 974.9 tok/s | 6.6 s | 6.63 s | **27.90** | 5348 MiB |
| 25,582 | 1,068.0 | 24.0 s | 23.99 s | **24.43** | 5375 MiB |
| 52,332 | 1,019.1 | 51.4 s | 51.40 s | **21.96** | 5508 MiB |
| 79,832 | 961.4 | 83.0 s | 83.08 s | **21.19** | 5783 MiB |
| 95,782 | 928.1 | 103.2 s | 103.26 s | **19.91** | 5677 MiB |

**全部填充级别通过，后端未崩溃。显存全程平坦**（5314 → 5783 MiB，只涨 470 MiB）
——因为 KV 按 128K 全额预分配，填多少都一样。**decode 只下降 30%**（28.3 → 19.9），
因为每步只有 10 层注意力需要扫 KV。

> 填到 9.6 万 token 时首字约 103 s。这是长上下文的固有代价，但**可以用
> `cache_reuse` 摊掉**：多轮对话里只有新增部分需要预填充。

### 2.4 ⚠ 已知崩溃边界：ubatch 不要超过 2048

`--ubatch-size 4096` 在 `qwen35moe` + `--cpu-moe` 下有两种失败模式：

1. **加载期 OOM**（桌面显存占用稍高时）：
   `ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2054.28 MiB ... cudaMalloc failed`
2. **运行期越界**（能加载成功时）：
   ```
   E CUDA error: an illegal memory access was encountered
   E   in function ggml_backend_cuda_synchronize at ggml-cuda.cu:2553
   ```
   后端进程直接死掉，网关返回 502。

4096 时显存只剩约 1.8 GiB，推测是大 ubatch 的缓冲区越界。**2048 时显存留 2.76 GiB
且每种填充级别都实测通过**，prefill 只比 4096 低 13%，所以取 2048。

代码层面已加防呆：`ubatch_size` 超过 2048 会被自动夹紧并打告警，除非显式设置
`allow_large_ubatch: true`（仅供实验）。网关在后端意外退出时会返回
**后端日志尾部**而不是空白的 502。

### 2.5 ⚠ 另一个坑：`load-mode none` 需要锁定 14.6 GiB 锁页内存

`--load-mode none` 会把 `--cpu-moe` 的专家张量放进 **`CUDA_Host`（cudaMallocHost
锁页内存）**，而不是普通可分页内存。启动日志里是：

```
E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 15704850432
E unable to allocate CUDA_Host buffer
```

`15704850432` 字节 = **14.63 GiB**，正是专家权重。这不是显存不足 —— 它在启动后
**1.9 秒**就失败，根本没到读模型那一步，而且**只在内存被占满/碎片化时触发**。

两个必知后果：

1. **绝对不能同时跑两个实例。** 每个实例都要锁 14.6 GiB，31 GiB 的机器锁不下
   两份，必然有一个起不来。`main.py serve` 现在会先探测 8000 端口，发现已有实例
   就拒绝启动并给出提示（`--auto-port` 除外）。
2. **锁不到时会自动回落 mmap**（decode 慢 17%，但一定能起来），日志里会写明
   `宿主内存不足，回落 mmap`。这是识别出来的**宿主内存**失败，不会被误判成显存
   不足去降 ubatch——那治不了这个病。

如果启动总是不稳，直接把 `config/server.yaml` 的 `load_mode` 设成 `""`
（用 llama.cpp 默认的 auto=mmap）。

---

## 3. 快速开始

```bat
cd /d "D:\personal\AI_output\local LLM\ornith-server"

REM 1) 环境自检（GPU、运行时、精确显存/内存预算）
D:\anaconda\envs\test1\python.exe main.py doctor

REM 2) 启动内网 API 服务（默认 128K）
start_server.bat

REM 带鉴权启动
start_server.bat --api-key sk-my-secret-key
```

启动后终端会打印内网地址：

```
本机访问  : http://127.0.0.1:8000/v1
内网访问  : http://192.168.x.x:8000/v1
接口文档  : http://127.0.0.1:8000/docs
```

验证：

```bat
D:\anaconda\envs\test1\python.exe scripts\client_example.py
```

### 命令行子命令

| 命令 | 作用 |
|---|---|
| `main.py doctor` | 环境自检 + 按张量索引算的精确摆位预算 |
| `main.py serve` | 启动内网 API 服务（默认命令） |
| `main.py backend` | 只启动 llama.cpp 后端，不起网关 |
| `main.py chat` | 终端交互式对话（会显示思考内容） |
| `main.py bench` | 基准测试：prefill / decode 实测吞吐 |
| `main.py models` | 列出本机所有可用 llama.cpp 运行时 |

常用覆盖参数：

```bat
REM 榨取最后 6%：后 4 层专家上显存（必须同时把 ubatch 降到 2048）
main.py serve --n-cpu-moe 36 --ubatch 2048

REM 显存紧张时降到 1024
main.py serve --ubatch 1024

REM 多轮长对话：复用前缀 KV，避免每轮重新预填充整个历史
REM   在 config/server.yaml 里设 cache_reuse: 256

REM 只跑 32K 上下文
main.py serve --ctx 32768
```

---

## 4. 内网 API

### 4.1 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | 对话补全，支持 `stream: true` |
| POST | `/v1/completions` | 传统文本补全 |
| POST | `/v1/embeddings` | 向量接口 |
| GET | `/v1/models` | 模型列表 |
| GET | `/health` | 健康检查（含 GPU 显存与后端进程状态） |
| GET | `/stats` | 网关统计（QPS、延迟、token 数） |
| GET | `/metrics` | Prometheus 格式指标 |
| GET | `/docs` | Swagger 交互文档 |

### 4.2 从其他机器调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.1.20:8000/v1", api_key="sk-my-secret-key")
resp = client.chat.completions.create(
    model="tile-35b-a3b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=512,
)
print(resp.choices[0].message.content)
```

```bat
curl http://192.168.1.20:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer sk-my-secret-key" ^
  -d "{\"model\":\"tile-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

### 4.3 鉴权与访问控制

- 默认 `api_key: ""`，**内网免鉴权**（方便直接接入现有客户端）。
- 设置 `--api-key sk-xxx` 后，除 `/health`、`/stats`、`/metrics`、`/v1/models`、
  `/docs` 外的所有接口都要求 `Authorization: Bearer <key>`。
- 后端 `llama-server` **只监听 127.0.0.1**，不直接暴露到内网。
- 若内网环境不安全，建议同时用 Windows 防火墙限制 8000 端口的来源网段。

### 4.4 推理模型的注意事项（重要）

本模型是**推理模型**：回答前会先输出 `<think>` 思考段。流式响应里思考内容走
`delta.reasoning_content`，正文走 `delta.content`。

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    think = getattr(delta, "reasoning_content", None)   # 思考内容
    text  = delta.content or ""                         # 正文
```

**`max_tokens` 不要设太小**，否则可能被思考过程吃光导致正文为空。建议至少 512。

---

## 5. 性能调优

### 5.1 参数速查（按影响从大到小）

| 参数 | 默认 | 影响 |
|---|---|---|
| **`ubatch_size`** | 2048 | **MoE 下最关键**。每个 ubatch 要过一遍全部 256 个专家，它直接决定 prefill（443 → 1049 tok/s）。**上限 2048**，4096 会崩 |
| **`load_mode`** | `none` | 不用 mmap，decode 快 17%。**代价：需一次性锁定约 14.6 GiB 锁页内存**，锁不到时自动回落 mmap（见 2.5）|
| **`cpu_moe`** | `true` | 专家权重放内存（14.12 GiB）。这是整个方案的前提 |
| `n_cpu_moe` | -1 | 后 N 层专家上显存。**只有**配合 ubatch 2048 才安全，收益 +6% |
| `kv_offload` | `true` | KV 放显存。128K 只要 1.33 GiB，放显存更快 |
| `context_size` | 131072 | 128K。改小不省显存（KV 已很小），但能缩短首字延迟 |
| `kv_cache_type_k/v` | `q8_0` | `f16` 质量更好（多占 1.2 GiB，速度无变化）；`q4_0` 再省一半 |
| `threads` | 16 | **实测 8/16/24 无差异**，瓶颈不在线程数 |
| `batch_size` | 8192 | 逻辑批，需 ≥ `ubatch_size` |
| `cache_reuse` | 0 | 多轮长对话**强烈建议设 256**，避免每轮重填整个历史 |
| `flash_attention` | `true` | 必开，KV 量化需要它 |
| `parallel_slots` | 1 | 并发数。每槽分摊总上下文 |

### 5.2 场景化推荐

```bat
REM A. 默认：128K + 最佳综合速度
main.py serve

REM B. 榨速度：后 4 层专家上显存（+6% decode），显存约 6.8 GiB
main.py serve --n-cpu-moe 36 --ubatch 2048

REM C. 显存很紧（浏览器/IDE 吃显存）
main.py serve --ubatch 1024

REM D. 3-4 人共享，各自上下文不长
main.py serve --ctx 32768 --parallel 4
```

### 5.3 自动降级保护

启动时若 CUDA OOM，会按以下顺序自动重试（每次记录到 `runtime/logs/`）。
设计原则是**优先保住上下文长度**（用户明确要求 128K），先牺牲速度相关的项：

1. 配置值（首选）
2. **宿主内存锁定失败时回落 mmap**（同一套参数，不降 ubatch——病因不在显存）
3. 把放到显存里的专家退回内存（仅当设置了 `n_cpu_moe`，这一步省得最多）
4. 逐级降 `ubatch`：2048 → 1024 → 512
5. KV 量化降到 `q4_0`
6. KV cache 移到内存
7. 上下文降到 75%
8. 上下文 32768 + KV 放内存
9. 最后才把 8 层留给 CPU

### 5.4 显存不会泄漏（Windows Job Object 兜底）

Windows 上父进程终止**不会**自动结束子进程。本项目在启动 `llama-server` 时把它加入
设置了 `KILL_ON_JOB_CLOSE` 的 **Windows Job Object**，无论父进程怎么死，系统都会
一并结束后端。实测正常 `stop()` 与 `taskkill /F` 强杀父进程两条路径都能在 1.2 秒内
释放显存。

```bat
D:\anaconda\envs\test1\python.exe scripts\test_jobobject.py
```

万一仍出现残留，用 `stop_server.bat` 清理。

### 5.5 测量工具

| 脚本 | 用途 |
|---|---|
| `scripts/gguf_tensors.py` | **精确张量账本**：专家 vs 其余、量化分布、每层专家体积 |
| `scripts/fetch_runtime.py` | **克隆后恢复 `runtime/`**（那些二进制不入库）|
| `scripts/moe_probe.py` | 单次摆位实测（加载 / 显存 / 内存 / decode / prefill）|
| `scripts/moe_sweep.py` | 批量扫描与对比表（`--preset default|ubatch|final`）|
| **`scripts/stress_ctx.py`** | **中文长提示逐级填充到 128K，检测崩溃** |
| `scripts/sysinfo.py` | 内存与磁盘（不依赖 WMI）|
| `scripts/perf_matrix.py` | 上下文 × KV 位置性能矩阵 |
| `scripts/diag_sse.py` | 打印流式响应原始片段 |
| `scripts/gguf_info.py` | 模型结构 + KV cache 账 |

### 5.6 自检清单

```bat
cd "D:\personal\AI_output\local LLM\ornith-server"
D:\anaconda\envs\test1\python.exe scripts\check_syntax.py      REM 语法自检（34 文件）
D:\anaconda\envs\test1\python.exe scripts\check_deps.py        REM 依赖自检
D:\anaconda\envs\test1\python.exe scripts\test_cli.py          REM 命令行参数解析（43 项）
D:\anaconda\envs\test1\python.exe scripts\test_load_ladder.py  REM 加载降级阶梯（18 项）
D:\anaconda\envs\test1\python.exe main.py doctor               REM 环境 + 精确摆位预算
D:\anaconda\envs\test1\python.exe scripts\e2e_test.py          REM 接口端到端（21 项）
D:\anaconda\envs\test1\python.exe scripts\test_jobobject.py    REM 显存不泄漏（8 项）
D:\anaconda\envs\test1\python.exe scripts\stress_ctx.py        REM 128K 上下文压力验证
D:\anaconda\envs\test1\python.exe scripts\final_acceptance.py  REM 默认 128K 配置验收
```

---

## 6. 项目结构

```
ornith-server/
├── main.py                       # CLI 入口（serve/backend/chat/bench/doctor/models）
├── start_server.bat              # 一键启动
├── stop_server.bat               # 停止所有 llama-server 进程（不依赖 Python）
├── requirements.txt
├── config/
│   └── server.yaml               # 配置文件（可被环境变量和命令行覆盖）
├── src/ornith_server/
│   ├── config.py                 # 配置模型、校验、告警、ubatch 安全上限
│   ├── bench.py                  # 基准测试
│   ├── core/
│   │   ├── backend.py            # 运行时探测、GPU 查询、DLL 路径配对
│   │   ├── server.py             # 子进程管理、MoE 摆位命令行、分级降级
│   │   ├── gguf.py               # GGUF 元数据 + 张量索引解析、摆位预算
│   │   └── jobobject.py          # Windows Job Object 兜底
│   └── api/
│       ├── gateway.py            # FastAPI 网关（OpenAI 兼容 + 流式透传 + 崩溃诊断）
│       ├── middleware.py         # 鉴权、并发排队、用量统计
│       └── metrics.py            # 指标收集（JSON + Prometheus）
├── scripts/                      # 见 5.5 / 5.6 的表格
└── runtime/
    ├── llama.cpp/backends/       # 自带的 llama.cpp CUDA 运行时（912 MiB，自包含）
    └── logs/                     # llama-server 日志、扫描结果
```

### 架构

```
内网客户端 ──HTTP──> FastAPI 网关 (:8000, 0.0.0.0)
                          │  鉴权 / 限流排队 / 指标 / 日志
                          └──HTTP──> llama-server (:8080, 127.0.0.1)
                                          ├── CUDA ──> 注意力 + KV cache + embedding
                                          └── CPU  ──> MoE 专家权重（14.12 GiB，内存）
```

网关**不做任何逐 token 加工**，只做流式透传，所以不引入额外延迟。

### 运行时是自包含的（但这些二进制不在 Git 仓库里）

`runtime/llama.cpp/backends/` 里的引擎与 CUDA vendor DLL 是从 LM Studio 复制过来的
（共 912 MiB / 24 文件），已验证脱离 LM Studio 可独立运行：

```bat
runtime\llama.cpp\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.51.0\llama-server.exe --list-devices
```

后端探测顺序里项目自带目录**排第一**，LM Studio 路径降级为后备，因此
**LM Studio 可以放心卸载**。

> **`runtime/` 被 `.gitignore` 排除，不在仓库里。**
> 原因：`cublasLt64_12.dll` 单个就有 674 MB，远超 GitHub 的 100 MB 单文件上限，
> push 会被直接拒绝。
>
> **克隆仓库后跑一次下面这条命令即可恢复：**
>
> ```bat
> D:\anaconda\envs\test1\python.exe scripts\fetch_runtime.py
> ```
>
> 它会自动从 LM Studio 的 `extensions/backends/`（或 `--from` 指定的目录）挑选
> **最新的 CUDA12 引擎**并配对正确的 vendor DLL。之后用 `main.py doctor` 验证。
> 加 `--check` 可以只看现状不复制。

---

## 7. 常见问题

**Q: 启动报 `unable to allocate CUDA_Host buffer` / `failed to allocate buffer of size 15704850432`？**
A: 不是显存不足，是**锁页内存**锁不到 14.6 GiB。最常见的原因是**已经有一个实例
在跑**（两个实例各要锁 14.6 GiB）。先 `stop_server.bat`，或在任务管理器里确认没有
残留的 `llama-server.exe`，再启动。程序也会自动回落到 mmap 继续启动。

**Q: 双击 start_server.bat 没反应 / 一闪而过？**
A: 窗口一闪而过说明脚本报错了。先跑 `doctor.bat` 看自检结果。另外**不要双击
`.bat` 后在同一个端口再启一个**——第二个会被拒绝（这是有意的保护）。

**Q: 想确认服务是否已经在跑？**
A: 浏览器打开 `http://127.0.0.1:8000/health`。返回 `"ready": true` 就是在跑。

**Q: 启动报 `0xC0000135` / 找不到 DLL？**
A: CUDA vendor 目录没进 PATH。跑 `main.py doctor` 看"运行时库"一行是否指向
`runtime\llama.cpp\backends\vendor\win-llama-cuda12-vendor-v2`。

**Q: 请求返回 502，错误信息里有后端日志？**
A: 说明 `llama-server` 进程死了。最常见原因是 `ubatch_size` 超过 2048
（CUDA 越界）或显存被挤爆。检查 `config/server.yaml` 的 `ubatch_size`。

**Q: 启动时报 CUDA out of memory？**
A: 显存被别的东西占了。跑 `nvidia-smi` 看占用，或 `main.py serve --ubatch 1024`。
程序也会自动降级重试。

**Q: 生成速度只有个位数 tok/s？**
A: 检查 `doctor` 的"设备枚举"里能否看到 `CUDA0`，以及启动日志里 `ngl` 是否为 99。
另外确认专家权重确实在内存（`--cpu-moe`），否则会退化成读盘。

**Q: 内存不够怎么办？**
A: 专家权重需要 14.12 GiB 常驻 + 约 2.5 GiB 余量。不够时可以：
用 mmap（`load_mode: ""`，慢 17% 但可按需换页），或换更小的量化版本。

**Q: 想再快一点？**
A: 按性价比排序：
1. `cache_reuse: 256` —— 多轮对话必开，省掉重复 prefill（收益最大）；
2. `n_cpu_moe: 36` + `ubatch 2048` —— +6% decode，代价是显存到 6.8 GiB；
3. `kv_cache_type: f16` —— 不提速，但质量更好（多占 1.2 GiB）。

**Q: 为什么不用 4096 的 ubatch？明明 prefill 更快。**
A: 因为它会崩。见 2.4 节，两种失败模式都实测到了。

**Q: 多模态（图片）能用吗？**
A: 该目录下有 `mmproj-BF16.gguf`（0.84 GiB）。在 `config/server.yaml` 里取消
`mmproj_path` 的注释即可启用视觉输入，但会额外占显存。

**Q: MTP（`blk.40`）能用来加速吗？**
A: 现在不行——llama.cpp 会打印 `model has unused tensor ... -- ignoring` 全部跳过，
0.36 GiB 属于死重。等上游支持 `qwen35moe` 的 MTP 后，可以用它做自投机解码。

---

## 8. 参考

- [llama-server 参数文档](https://mintlify.wiki/ggml-org/llama.cpp/api/tools/llama-server)
- [llama.cpp 并行推理参数讨论 #18308](https://github.com/ggml-org/llama.cpp/discussions/18308)
- [llama.cpp vs vLLM 对比](https://theneuralbase.com/llamacpp/qna/llama-cpp-vs-vllm-comparison/)
- 详细排查过程与测量方法论见 [`NOTES.md`](NOTES.md)
