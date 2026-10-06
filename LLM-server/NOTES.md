# 调研与排查笔记

本文件记录选型依据、MoE 摆位的实测过程，以及**测量方法论**。这些坑一旦
踩中，会把性能数据完全测错，因此单独留档。

---

## 1. 本机环境事实（探测得到，非假设）

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 581.80，算力 8.9 |
| 可用显存 | 约 6.6–7.3 GiB（桌面/浏览器占 0.9–1.9 GiB，会波动）|
| CPU | AMD Ryzen 9 7945HX，16 核 32 线程 |
| 内存 | 31.2 GiB 总量，可用约 21 GiB |
| 磁盘 | C: 81 GiB / D: 435 GiB / E: 530 GiB 可用 |
| `test1` 环境 | Python 3.12.7，`torch 2.6.0+cu118`（**本方案不依赖 torch**）|
| `XTTS` 环境 | `torch 2.5.1+cpu`，`cuda_available=False` → 不能用于 GPU 推理 |
| llama.cpp | 本项目自带 `runtime/llama.cpp/backends/`（2.51.0 CUDA12，从 LM Studio 复制）|

## 2. 运行时自包含（已脱离 LM Studio）

原先直接调用 `~/.lmstudio/extensions/backends/` 下的引擎。现已把**引擎目录 +
CUDA vendor 目录**整体复制进项目：

```
runtime/llama.cpp/backends/
├── llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.51.0/   # 引擎 21 个文件 / 160 MiB
└── vendor/win-llama-cuda12-vendor-v2/                # cudart/cublas/cublasLt / 752 MiB
```

共 912 MiB / 24 文件。验证方式：清空 `PATH` 后直接跑 `llama-server.exe
--list-devices`，rc=0 并正确列出 CUDA0 —— 证明不依赖 LM Studio 是否安装。

`core/backend.py` 的候选根目录里，项目自带目录**排第一**，LM Studio 的路径
降级为后备。所以 LM Studio 可以放心卸载。

**关键实现细节**：engine 目录与 vendor 目录必须按**后端家族**配对。CUDA 引擎配
`win-llama-cuda12-vendor-v2`，Vulkan 配 `win-llama-vulkan-vendor-v2`，CPU 引擎
不需要 vendor。配错会加载错误的运行时库。

---

## 3. 模型结构（`qwen35moe`，解析 GGUF 得到）

| 参数 | 值 |
|---|---|
| `general.name` | Huihui Ornith 1.5 35B A3B Abliterated |
| `block_count` | **41**（其中 blk.40 是 MTP 头，**llama.cpp 整段忽略**）|
| 实际计算层数 | **40** |
| `embedding_length` | 2048 |
| `attention.head_count` | 16 |
| `attention.head_count_kv` | **2**（GQA 8:1）|
| `attention.key_length` / `value_length` | 256 |
| `expert_count` | **256** |
| `expert_used_count` | **8** |
| `expert_feed_forward_length` | 512 |
| `expert_shared_feed_forward_length` | 512 |
| **`full_attention_interval`** | **4** |
| `ssm.*` | conv_kernel 4 / group_count 16 / inner_size 4096 / state_size 128 |
| `nextn_predict_layers` | 1 |
| `context_length`（训练） | 262144 |
| 权重文件 | 16.876 GiB（IQ4_XS 混合量化，753 个张量）|

### 3.1 这是混合注意力架构，不是普通 MoE

`full_attention_interval = 4` 表示**每 4 层里只有 1 层是真注意力**，其余 3 层是
**线性注意力（gated DeltaNet）**，其循环状态大小固定、**不随上下文增长**。

于是 KV cache 只算那 10 个全注意力层，而不是 40 层：

| 上下文 | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 32K | 0.63 GiB | 0.33 | 0.18 |
| 64K | 1.26 | 0.66 | 0.35 |
| **128K** | **2.51** | **1.33** | 0.70 |

这是本次改造中最反直觉的一点：**128K 的 KV 只要 1.33 GiB，可以整个放显存**，
与上一个 9B 模型（head_dim 256、32 层全注意力、128K 需 8.5 GiB）结论完全相反。

`--swa-full` 实测对显存与速度**没有任何影响**（净增 4505 vs 4519 MiB），
反证这些层不是滑窗注意力，**128K 上下文没有被截断**。

### 3.2 张量账本（`scripts/gguf_tensors.py` 精确统计）

| 分类 | 体积 | 去向 |
|---|---|---|
| 专家 `*_exps`（层 0–39） | **14.12 GiB** | 内存 |
| 其余（attention/SSM/embedding/shared） | **2.38 GiB** | 显存 |
| MTP/nextn `blk.40` | 0.36 GiB | **被忽略，死重** |

量化分布是 Unsloth 动态混合：IQ3_S 8.59 GiB / IQ4_XS 4.91 / Q8_0 1.93 /
Q6_K 1.00 / Q3_K 0.32 / F32 0.10 GiB。最大的两个张量是
`token_embd.weight`（Q8_0，515 MiB，词表 248320）和 `output.weight`（Q6_K，398 MiB）。

**这套账本预测显存 5.39 GiB，实测净增 5518 MiB = 5.39 GiB，误差在舍入范围内。**

---

## 4. MoE 摆位的实测过程

目标：**专家权重放内存、激活部分与上下文放显存**。

llama.cpp 提供三个旋钮：

- `-cmoe, --cpu-moe` —— 全部专家权重留 CPU
- `-ncmoe, --n-cpu-moe N` —— 只把前 N 层的专家留 CPU，其余层专家进显存
- `-ot, --override-tensor <pattern>=<buffer>` —— 按张量名精确指定

扫描工具：`scripts/moe_probe.py`（单次测量）+ `scripts/moe_sweep.py`（批量对比）。

### 4.1 第一轮：线程数 / 加载模式 / 专家上显存

固定 `ctx=131072`、`--cpu-moe`、KV q8_0 放显存。

| 用例 | decode tok/s | prefill tok/s | 显存净增 MiB |
|---|---|---|---|
| `--threads 8` | 24.40 | 433.9 | 4001 |
| `--threads 16` | 24.95 | 425.6 | 4025 |
| `--threads 24` | 24.59 | 442.1 | 4013 |
| **`--load-mode none`** | **29.34** | **452.0** | 4072 |
| `--n-cpu-moe 34` | 24.40 | **77.1** | 5921 |
| `--n-cpu-moe 32` | 23.65 | 378.0 | 6832 |

**结论一：线程数完全不是瓶颈。** 8 / 16 / 24 线程的差异在噪声内。反推有效访存
带宽：decode 每 token 要读 8/256 × 14.12 GiB = **0.441 GiB** 专家权重，
29.34 tok/s × 0.441 GiB = **12.9 GB/s**；prefill 同样是约 13.4 GB/s。
两者卡在同一个数字上，说明瓶颈是 **CPU 侧小批量专家 GEMM 的有效访存**，
而不是算力也不是内存带宽（DDR5 双通道理论 83 GB/s）。

**结论二：`--load-mode none` 明确更快**（decode +17.6%）。llama.cpp 自己也会警告
`tensor overrides to CPU are used with mmap enabled - consider using --load-mode
none`——mmap 下 CPU 张量走页缓存映射，多一层间接且拿不到大页。代价是加载
3.5s → 11.6s，且需要 17 GiB 常驻内存。

**结论三：`--n-cpu-moe` 单独用是负收益。** 除了吃显存，它还会把专家读取打散。
`ncmoe34` 那次工作集只有 8.45 GiB（说明页面被换出、改从磁盘读），prefill 直接
崩到 77 tok/s。**只有当显存有确定余量时才值得用**（见 4.3）。

### 4.2 第二轮：ubatch 才是决定性的杠杆

发现瓶颈在"每个 ubatch 都要把 256 个专家全过一遍"之后，加大 ubatch 就能减少
同样的 prompt 需要的往返次数：

| ubatch | decode tok/s | prefill tok/s | 显存净增 MiB |
|---|---|---|---|
| 256 | 28.49 | 322.2 | 3885 |
| 512 | 28.54 | 442.9 | 3999 |
| 1024 | 28.76 | 771.8 | 4181 |
| 2048 | 29.19 | 1049.4 | 4519 |
| **4096** | 28.81 | **1207.2** | 5518 |
| 8192 | 24.93 | **456.5** | 7006 |

**prefill 提升 2.7 倍**（443 → 1207），而 decode 几乎不变。ubatch 8192 反而崩，
因为计算缓冲把显存挤爆了（净增 7006 MiB）。**4096 是甜点。**

### 4.3 第三轮：边界与组合

| 用例 | decode | prefill | 显存净增 MiB | 工作集 GiB |
|---|---|---|---|---|
| ubatch 4096 | 28.81 | 1207.2 | 5518 | 15.75 |
| ubatch 2048 + KV f16 | 28.90 | 1047.8 | 5733 | 15.61 |
| ubatch 2048 + `--swa-full` | 28.84 | 1045.3 | 4505 | 15.61 |
| **ubatch 2048 + `--n-cpu-moe 36`** | **30.93** | **1089.3** | 6066 | 14.55 |
| ubatch 2048 + `--n-cpu-moe 32` | **14.56** | **76.3** | 7235 | 13.40 |

**显存天花板非常硬**：总占用 6768 MiB 时一切正常，7894 MiB 时 decode 从 30.9
崩到 14.6 tok/s。所以 `--n-cpu-moe` 只能用"温和"的值。

- **KV f16** 多花 1.2 GiB 显存，速度无变化（28.90 vs 29.19，噪声内）。要质量可以开。
- **`--swa-full`** 无影响 → 确认不是滑窗。
- **`ncmoe 36`**（后 4 层专家进显存）是唯一正向的专家上显存配置：decode +6%、
  prefill +4%，显存仍有 1.4 GiB 余量。`ncmoe 32` 越过了天花板，直接崩。

### 4.4 ⚠ ubatch 4096 不可用（两种崩溃模式）
第三轮的表格里 4096 看起来是最优（prefill 1207），但那是拿**英文合成提示**测的。
换成真实中文长提示后立刻暴露问题：

1. **加载期 OOM**（桌面显存占用稍高时）：
   ```
   E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2054.28 MiB on device 0:
     cudaMalloc failed: out of memory
   E graph_reserve: failed to allocate compute buffers
   ```
2. **运行期越界**（能加载成功时）：
   ```
   E CUDA error: an illegal memory access was encountered
   E   current device: 0, in function ggml_backend_cuda_synchronize
   E   at llm-engine\llama.cpp\ggml\src\ggml-cuda\ggml-cuda.cu:2553
   ```
   后端进程直接死掉，网关返回 502。

4096 时显存只剩约 1.8 GiB，推测是 `qwen35moe` + `--cpu-moe` 路径下大 ubatch 的
缓冲区越界。**2048 时显存留 2.76 GiB，prefill 只低 13%，且所有填充级别实测通过。**

**教训**：用合成英文提示做扫描会漏掉只在特定提示下触发的 bug，
必须再用真实语言的提示、跨多个长度复验一遍。

代码层面已加防呆：`ubatch_size` 超过 `ServerConfig.SAFE_UBATCH`（2048）会被
`build_command` 自动夹紧并打告警，除非显式设 `allow_large_ubatch: true`。
网关在后端意外退出时会返回后端日志尾部，而不是一个空白的 `后端连接失败: `。

### 4.4b ⚠ `--load-mode none` 要锁定 14.6 GiB 锁页内存（真实踩坑记录）

这个坑是在用户实际使用时才暴露的。控制台显示：

```
23:26:59 尝试加载 [1/8] ubatch=2048  — 配置值（首选）
23:27:01 WARNING 本次加载失败（疑似显存不足），准备降级重试…
23:27:03 尝试加载 [2/8] ubatch=1024  — ubatch 降到 1024
23:27:05 后端就绪
```

看起来像显存不足，但翻 llama-server 日志发现完全不是：

```
E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 15704850432
E ggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA_Host buffer of size 15704850432
E llama_model_load: error loading model: unable to allocate CUDA_Host buffer
```

`15704850432` 字节 = **14.63 GiB**，正是专家权重。三个要点：

1. **这是宿主锁页内存，不是显存。** `--load-mode none` 会把 `--cpu-moe` 的专家
   张量放进 `CUDA_Host`（`cudaMallocHost`），需要一次性锁定约 14.6 GiB **物理**
   内存。失败发生在启动后 **1.9 秒**，根本没到读模型那一步。
2. **触发条件是内存争抢。** 事后核对日志文件名的时间戳，发现 **10 秒内有三个
   llama-server 在加载**（23:26:53 / 23:26:59 / 23:27:03）——排查时我自己留了
   一个实例没关，用户又启了一个。每个实例要锁 14.6 GiB，31 GiB 的机器锁不下
   两份，其中一个必然失败。
3. **降级阶梯当时判错了病因。** `_OOM_MARKERS` 里有 "failed to allocate"，所以
   这个宿主内存失败被归类成显存不足，于是去降 ubatch —— 跟病因毫无关系。
   第二次侥幸成功只是因为内存被释放了。

**已做的修复**（见 `core/server.py` 与 `scripts/test_load_ladder.py`）：

- 新增 `_HOST_MEM_MARKERS` 与 `_looks_like_host_mem_failure()`，**先于** OOM 判定；
- 识别为宿主内存失败时，**改用 mmap 重试同一套参数**（mmap 下 CPU 张量来自文件
  映射、可回收，不需要大块锁定），而不是去降 ubatch。代价是 decode 慢 17%；
- `LoadProfile` 增加 `load_mode` 字段，使降级阶梯能换加载模式；
- `main.py serve` 启动前探测 8000 端口，**发现已有实例就拒绝启动**并给出指引
  （防呆，见 `_refuse_if_already_running`）。

**教训**：一个失败症状可能对应完全不同的病因，共享关键词的判据会误导降级策略。
把 "failed to allocate" 同时用于显存和宿主内存就是个典型错误。

### 4.5 上下文填充后的真实速度（中文长提示，单进程连续测）

`scripts/stress_ctx.py`，每一步都检查后端进程是否还活着：

| 实际 prompt | prefill | prefill 耗时 | 首字延迟 | decode | 显存 MiB |
|---|---|---|---|---|---|
| 12 | — | 0.4 s | 0.43 s | 28.30 | 5314 |
| 6,432 | 974.9 tok/s | 6.6 s | 6.63 s | 27.90 | 5348 |
| 25,582 | 1,068.0 | 24.0 s | 23.99 s | 24.43 | 5375 |
| 52,332 | 1,019.1 | 51.4 s | 51.40 s | 21.96 | 5508 |
| 79,832 | 961.4 | 83.0 s | 83.08 s | 21.19 | 5783 |
| 95,782 | 928.1 | 103.2 s | 103.26 s | 19.91 | 5677 |

关键点：

- **显存全程平坦**（5314 → 5783 MiB）。KV 按 128K 全额预分配，填多少都一样。
  这与上一个 9B 模型（KV 放内存、随填充量吃内存）完全不同。
- **decode 只下降 30%**（28.3 → 19.9）。9B 那套要下降 66%（21.6 → 7.4），
  因为这里每步只有 10 层注意力需要扫 KV。
- 首字延迟 ≈ prefill 耗时，填到 9.6 万 token 时约 103 s。这是长上下文的固有
  代价，用 `cache_reuse` 可以摊掉多轮对话里的重复部分。

### 4.6 最终配置

```yaml
context_size: 131072      # 128K
cpu_moe: true             # 专家全部放内存（14.12 GiB）
n_cpu_moe: -1             # 想再榨 6% 可设 36（须同时保持 ubatch 2048）
kv_offload: true          # KV 放显存（仅 1.33 GiB）
kv_cache_type_k/v: q8_0
ubatch_size: 2048         # prefill 的关键；上限就是这里
batch_size: 8192
load_mode: "none"         # +17% decode
threads: 16
```

实测：**decode 约 28 tok/s（空载）～20 tok/s（填 9.6 万 token），
prefill 约 930–1070 tok/s**，显存占用约 5.3–5.8 GiB，内存驻留约 15.8 GiB。

### 4.7 模型文件里的死重

`blk.40.*` 有 21 个张量（约 0.36 GiB）是 MTP / nextn 投机解码头，
当前 llama.cpp 会打印 `model has unused tensor ... -- ignoring` 全部跳过。
**这既是浪费也是机会**：哪天 llama.cpp 支持 `qwen35moe` 的 MTP，用它可以做
自投机解码，decode 还有提升空间。

### 4.8 APEX 版的完整标定（以及两处被证伪的旧结论）

基准版跑了三轮扫描 + 长上下文压力验证，而 APEX 一开始只跑过一次探针 ——
**验证深度不对等**。补测后有三条结论，其中两条推翻了我之前的说法。

**一、`ubatch 4096` 在真实提示下不是更快，而是更慢。**

合成 8K 英文提示的扫描里它看着快 15%，换成真实中文长提示：

| 实际 prompt | ubatch 4096 | ubatch 2048 |
|---|---|---|
| 12,807 | 825.1 | — |
| 38,332 | **505.3** | — |
| 76,632 | **634.1** | **1024.3** |

慢约 40%，且显存冲到 7582~7784 MiB（只剩 376 MiB）。4096 从来不是更快，
只是挑提示 —— **固定用 2048**。

**二、`n_cpu_moe` 在 APEX 上不值得开。**

基准版 @128K 是 +6%（29.19→30.93，显存只到 6768 MiB）。APEX @200K 只有
+3.4%（36.51→37.74），却把显存推到 7531 MiB —— 离 7894 MiB 的崩盘线只剩
363 MiB。参数不能跨模型照搬。

**三、理论显存公式在 APEX 上偏低 0.75 GiB。**

| 模型 | ctx | 公式预测 | 实测净增 | 偏差 |
|---|---|---|---|---|
| 基准版 | 131072 | 4514 MiB | 4519 | +5 ✅ |
| APEX | 131072 | 3654 MiB | 3943 | **+289** |
| APEX | 204800 | 4369 MiB | 5143 | **+774** |

公式 = resident + KV + 计算缓冲。在基准版上分毫不差，在 APEX 上偏差还随
上下文增大。**原因未查明**（怀疑与该 GGUF 的张量摆放或 llama.cpp 的额外
缓冲有关）。既然公式不可靠，就不要拿它当依据：profile 里现在带一个
`measured:` 段记录**实测值**，界面优先显示它并标注「实测 / 估算」。

**四、APEX 与基准版的差别只有量化。**

逐项比对过：chat template（都是 `qwen3.8-froggeric-v22.5.0`，30517 字符）、
层数、注意力结构、rope 参数、专家配置 —— **全部一致**。差别是
APEX 用 Q3_K(6.75G)+IQ3_XXS(5.74G)，基准版用 IQ3_S(8.59G)+IQ4_XS(4.91G)。

同一批提示词、temperature=0 的并排对比（`scripts/compare_models.py`）：
四个题**两个模型全部答对**，输出质量相当 —— 基准版更精炼，APEX 更啰嗦、
爱用 Markdown，且思考长度明显更长（代码题 324 vs 97 字符）。
APEX 换来的是约 20% 的速度和 72K 的额外上下文。

### 4.9 APEX @200K 的长上下文验证（补测）

`scripts/stress_ctx.py`，一路填到 153,232 token，**全程无崩溃**：

| 实际 prompt | prefill | 首字延迟 | decode | 显存 MiB |
|---|---|---|---|---|
| 12 | 31.0 | 0.4 s | 30.50 | 6606 |
| 25,582 | 1137.0 | 22.5 s | 26.91 | 6650 |
| 76,632 | 1024.3 | 74.9 s | 22.28 | 6990 |
| 127,707 | 890.4 | 143.5 s | 18.03 | 6808 |
| **153,232** | **844.8** | 181.4 s | **17.69** | 6758 |

显存全程平坦（6606 → 6990 MiB）。decode 从空载到填 15.3 万 token 只掉 42%。

---

## 5. 测量方法论（重要）

排查过程中先后多次把性能数据测错，记录如下，避免重犯。

### 5.1 坑：推理模型的思考内容走 `reasoning_content`

**现象**：流式响应里数不到任何 `delta.content`，`chunks=0`，prefill 被算成
2,286,000 tok/s 这种荒谬值。

**原因**：这是推理模型，回答前先输出思考内容，字段是
`delta.reasoning_content`；正文才走 `delta.content`。

**正确做法**：
```python
think = getattr(delta, "reasoning_content", None)  # 思考
text  = delta.content or ""                        # 正文
```

### 5.2 坑：预热文本与测试文本共享前缀 → 触发 prompt cache 复用

**现象**：`prompt_n` 只有 4，而提示明明有上万字符；prefill 看起来近乎瞬时。

**原因**：llama.cpp 默认复用与上一请求相同的提示前缀（`cache_n`）。

**正确做法**：测预填充时设置 `cache_prompt: false`。

### 5.3 坑：压测文本里混了数字，token 数变成目标的 3 倍

**现象**：想要 8192 token 的 prompt，实际 `prompt_n = 27607`，一次测试跑了 80 秒。

**原因**：`alpha0`、`alpha1` 这类带数字后缀的词会被分词器切成多个 token。

**正确做法**：用纯字母单词（约 1.35 token/词），并打印实际 `prompt_n` 复核。
`scripts/moe_probe.py` 已修正。

### 5.3b 坑：只用合成英文提示做参数扫描

**现象**：扫描结果显示 `--ubatch-size 4096` 最优（prefill 1207 tok/s），
但换成一个真实的中文长提示后，后端立刻
`CUDA error: an illegal memory access was encountered` 直接死掉。

**正确做法**：参数扫描之后必须用**真实语言的提示、跨多个长度**再复验一遍。
`scripts/stress_ctx.py` 就是为这个目的写的。

### 5.4 坑：用客户端计时反推吞吐

**正确做法**：直接用服务端返回的 `timings` 字段，它给出精确的
`prompt_n` / `prompt_ms` / `prompt_per_second` / `predicted_n` / `predicted_ms` /
`predicted_per_second`。

### 5.5 坑：用空载数字做容量规划

上一个 9B 模型在 128K 配置下空载测得 21.6 tok/s，填充 5.4 万 token 后只剩
7.4 tok/s。**必须按实际填充量评估。**

### 5.6 坑：把"激活参数量"当成显存需求

MoE 的"激活 3B"是**算力成本**，不是**显存需求**。35B 的权重必须完整存放
（显存或内存），每 token 只是不全读而已。可行性瓶颈是**系统内存**。

### 5.7 坑：`--ctx` 写在子命令后面被忽略

公共选项只挂在主解析器上时，写到子命令后面无法识别；挂到子解析器后，
子解析器的 `None` 默认值又会**覆盖**主解析器已解析的值。

**正确做法**：子解析器上的同名选项用 `argparse.SUPPRESS` 作默认值；所有非通用
字段一律用 `getattr` 取值。`scripts/test_cli.py` 覆盖了这些组合。

---

## 6. 其他环境问题

### 6.1 `llama-server.exe` 报 `0xC0000135`

CUDA 运行时不在引擎目录而在 `vendor/` 下，启动前必须把 vendor 加入 `PATH`。
`core/backend.py` 会按后端家族自动配对。

### 6.2 孤儿进程占显存

Windows 上父进程终止不会结束子进程。`core/jobobject.py` 用 Job Object +
`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` 兜底，父进程无论怎么死系统都会结束子进程。
实测两条路径都能在 1.2 秒内释放显存。

### 6.3 工作区权限

会话开始时工作区 `D:\personal\AI_output\local LLM` 缺少 `WRITE_OWNER`，导致任何
命令都失败。已通过 `diagnose-windows-sandbox-acl` 修复。

### 6.4 ⚠ 系统代理会拦掉 httpx 发往 127.0.0.1 的请求（真实踩坑记录）

网页控制台上线后出现一个很诡异的现象：

* PowerShell `Invoke-RestMethod http://127.0.0.1:8080/health` → **200 `{"status":"ok"}`**
* 浏览器打得开控制台
* 但 Python 里 `server.start()` 的健康检查轮询**整整 300 秒都看不到 200**，
  最后超时降级；网关进程 CPU 烧到 75%，端口 8000 一直不监听

用 `scripts/diag_http.py` 在同一个 URL、同一时刻对比，一次就定位了：

```
httpx 默认(trust_env=True)          : HTTP 502   (3570ms)
httpx trust_env=False             : HTTP 200   (1188ms)
urllib                            : HTTP 200   (39ms)
httpx.AsyncClient 默认              : HTTP 502   (3870ms)
```

**原因**：机器上装了 Clash / v2ray 之类的系统代理（WinINET 里设成
`127.0.0.1:7897`）。httpx 的 `trust_env=True` 会通过
`urllib.request.getproxies()` **从 Windows 注册表**读到它 —— 不只是环境变量 ——
然后把发往 `127.0.0.1` 的请求也丢给代理，代理返回 502。

而 PowerShell、浏览器、`urllib` 都会遵守 Windows 的"绕过本地地址"
（ProxyOverride 里的 `<local>`），所以症状表现为"**只有 Python 连不上本机服务**"。

**影响面**：不只是健康检查。网关照样子用 httpx 转发给 llama-server，
等于**整个 API 全部 502**。而且 `server.start()` 会空转 300 秒才降级，
表面上像"模型加载失败"。

**修复**：项目内所有访问 llama-server（127.0.0.1）的 httpx 客户端统一加
`trust_env=False`，收口在 `src/llm_server/net.py`：

```python
from ..net import local_client, local_async_client
```

本地回环流量永远不该走代理。

**教训**：`httpx` 的 `trust_env` 读的是**系统**代理配置，不只环境变量。
凡是访问本机服务的客户端都应显式关闭它 —— 这类问题在装了代理客户端的
开发机上极易出现，且症状会误导成"服务没起来"。

---

## 7. 诊断脚本

| 脚本 | 用途 |
|---|---|
| `scripts/gguf_raw.py` | 转储 GGUF 元数据（看架构/专家数/全注意力层间隔）|
| `scripts/gguf_tensors.py` | **精确张量账本**：专家 vs 其余、量化分布、每层专家体积 |
| `scripts/moe_probe.py` | 单次 MoE 摆位实测（加载/显存/内存/decode/prefill）|
| `scripts/moe_sweep.py` | 批量扫描与对比表，预设组 `default` / `ubatch` / `final` |
| **`scripts/stress_ctx.py`** | **中文长提示逐级填充到 128K，每一步检测后端是否崩溃** |
| `scripts/sysinfo.py` | 内存与磁盘（不依赖 WMI，避免权限问题）|
| `scripts/diag_sse.py` | 打印流式响应原始片段（靠它发现 `reasoning_content`）|
| `scripts/diag_usage.py` | 对比流式/非流式/`/metrics` 三种取数方式 |
| `scripts/diag_http.py` | **对比 httpx / urllib 访问本机服务**（定位系统代理拦截）|
| `scripts/diag_health.py` | 后端 /health 的 httpx vs urllib 实况对比 |
| `scripts/fetch_runtime.py` | 克隆后恢复 `runtime/`（那些二进制不入库）|

> 两个诊断脚本**不自己启动后端**，直接打一个已经在跑的服务（默认
> `http://127.0.0.1:8000`）。早先它们通过 `calibrate.py` 自拉后端，但那个模块
> 构造的命令行不含 MoE 参数，拿这个模型跑必然 OOM，已连同它派生出的
> `ctx_scale.py` / `perf_matrix.py` 一起删除——功能分别由 `moe_probe.py` +
> `moe_sweep.py` + `stress_ctx.py` 覆盖且更准确。

## 8. 参考

- [llama-server 参数文档](https://mintlify.wiki/ggml-org/llama.cpp/api/tools/llama-server)
- [llama.cpp 并行推理参数讨论 #18308](https://github.com/ggml-org/llama.cpp/discussions/18308)
- [llama.cpp vs vLLM 对比](https://theneuralbase.com/llamacpp/qna/llama-cpp-vs-vllm-comparison/)
