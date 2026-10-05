# 在 8GB 显存 + 32GB 内存的笔记本上跑 35B MoE，128K 上下文

把 **Tile-35B-A3B**（Qwen3-Next 式混合架构 MoE）跑在 **RTX 4060 Laptop（8GB 显存）** 上，
提供内网 OpenAI 兼容 API，**默认 128K 上下文**。

核心做法是 **MoE 分层摆位**——专家权重放内存，注意力和 KV cache 放显存：

| 部分 | 体积 | 放哪 |
|---|---|---|
| 专家权重 `*_exps`（256 专家 × 40 层）| **14.12 GiB** | **内存** |
| 注意力 / 线性注意力 / embedding / 共享专家 | **2.38 GiB** | **显存** |
| KV cache（128K，q8_0，只算 10 个全注意力层）| **1.33 GiB** | **显存** |

实测效果：

| 上下文填充量 | prefill | 首字延迟 | **decode** | 显存占用 |
|---|---|---|---|---|
| 空载 | — | 0.4 s | **28.3 tok/s** | 5.31 GiB |
| 6.4K | 975 tok/s | 6.6 s | **27.9 tok/s** | 5.35 GiB |
| 25.6K | 1,068 tok/s | 24.0 s | **24.4 tok/s** | 5.38 GiB |
| 52.3K | 1,019 tok/s | 51.4 s | **22.0 tok/s** | 5.51 GiB |
| **95.8K** | **928 tok/s** | 103.3 s | **19.9 tok/s** | 5.68 GiB |

显存几乎不随填充量变化，decode 从空载填到 9.6 万 token 只掉 **30%**。

---

## 1. 需要什么

### 硬件（在本机实测标定）

| 项 | 最低 | 本机 |
|---|---|---|
| GPU | NVIDIA，**≥8 GiB 显存**，支持 CUDA 12 | RTX 4060 Laptop 8 GiB |
| 内存 | **≥24 GiB**（专家权重需 14.12 GiB 常驻 + 余量）| 31.2 GiB |
| 磁盘 | 17 GiB 放模型 | D: 435 GiB 可用 |
| CPU | 任意；prefill 速度取决于它 | Ryzen 9 7945HX 16C/32T |

> **内存比显存更关键**。35B 的权重必须完整存放，每 token 只是不全读而已；
> "激活 3B" 是算力成本，不是显存需求。

### 软件

- **Windows**（用了 Job Object 防孤儿进程，`start_server.bat` 是批处理）
- **Python 3.10+**（实测 3.12.7）
- **llama.cpp 运行时**：本项目自带获取脚本，从 LM Studio 的 `extensions/backends/` 复制
  （也可以自己编译，或用官方 release，只要含 `llama-server.exe`）
- **GGUF 模型文件**：见下一节

Python 依赖只有网关层需要的几个，**不需要 torch**：

```bat
pip install -r ornith-server\requirements.txt
```

### 模型文件（需自行获取）

```
Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf   16.88 GiB
```

它是 **`qwen35moe` 架构**（Qwen3-Next 式：41 层中 1 层是 MTP 头被忽略、
40 个计算层里每 4 层只有 1 层是真注意力，其余是线性注意力）。

> 本项目**不附带模型**。请从你获取该模型的渠道下载，放到任意位置后改配置指向它。
>
> 换成**其它同架构的 GGUF** 也能跑，但显存/内存预算和性能数字要重新标定——
> 本项目自带全套工具（见第 9 节），流程是：
> `gguf_raw.py` 看结构 → `gguf_tensors.py` 算字节账 → `moe_sweep.py` 扫参数
> → `stress_ctx.py` 验证长上下文。

---

## 2. 快速开始

### 第 1 步：克隆并装依赖

```bat
git clone <本仓库地址> "D:\personal\AI_output\local LLM"
cd /d "D:\personal\AI_output\local LLM"
pip install -r ornith-server\requirements.txt
```

### 第 2 步：恢复 llama.cpp 运行时

`runtime/` 里的二进制**不在仓库里**（`cublasLt64_12.dll` 单个 674 MB，超过 GitHub 的
100 MB 单文件上限）。跑一次脚本从 LM Studio 复制过来：

```bat
D:\anaconda\envs\test1\python.exe ornith-server\scripts\fetch_runtime.py
```

它会自动挑**最新的 CUDA12 引擎**并配对正确的 vendor DLL。没有 LM Studio 的话，
用 `--from` 指定任意含 `llama-server.exe` 的目录：

```bat
python ornith-server\scripts\fetch_runtime.py --from "D:\somewhere\llama.cpp\build\bin"
```

### 第 3 步：指向你的模型

编辑 `ornith-server\config\server.yaml`：

```yaml
model_path: "D:/models/Tile/Tile-35BA3B/Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf"
```

或用命令行覆盖，不改配置：`start_server.bat --model "D:\path\to\model.gguf"`

### 第 4 步：自检并启动

```bat
cd /d "D:\personal\AI_output\local LLM\ornith-server"
doctor.bat          REM 环境自检：GPU、运行时、精确显存/内存预算
start_server.bat    REM 启动（加载约 12 秒）
```

看到这个就是好了：

```
本机访问  : http://127.0.0.1:8000/v1
内网访问  : http://192.168.5.4:8000/v1     ← 你的实际内网 IP
接口文档  : http://127.0.0.1:8000/docs
```

> 横幅里会列出多个 `192.168.x.x`，其中**只有真实网卡那个别人能连**，
> 其余是 VMware / WSL / Hyper-V 的虚拟网卡。

验证一次真实调用：

```bat
D:\anaconda\envs\test1\python.exe scripts\client_example.py
```

---

## 3. 配置

配置文件 `ornith-server\config\server.yaml`，优先级：
**代码默认值 < 配置文件 < 环境变量 `ORNITH_*` < 命令行**。

### 最常改的几项

| 配置 | 默认 | 说明 |
|---|---|---|
| `model_path` | （本机路径）| GGUF 路径 |
| `model_alias` | `tile-35b-a3b` | 对外暴露的模型名，客户端 `model` 字段填这个 |
| `context_size` | `131072` | 128K。改小不省显存，但能缩短首字延迟 |
| `proxy_host` / `proxy_port` | `0.0.0.0` / `8000` | 对外监听 |
| `api_key` | `""` | 空 = 内网免鉴权 |
| `cpu_moe` | `true` | 专家权重放内存（**整个方案的前提**）|
| `load_mode` | `"none"` | 见下方警告 |
| `ubatch_size` | `2048` | MoE 下最重要的性能参数，**上限就是 2048** |

### 命令行速查

```bat
start_server.bat                                    REM 默认：128K + 最佳综合速度
start_server.bat --api-key sk-my-secret-key         REM 开启鉴权
start_server.bat --port 9000                        REM 换端口
start_server.bat --ubatch 1024                      REM 显存紧张时降一档
start_server.bat --n-cpu-moe 36 --ubatch 2048       REM 再榨 6%（显存到 6.8 GiB）
start_server.bat --model "D:\other\model.gguf"      REM 临时换模型
```

完整子命令：

| 命令 | 作用 |
|---|---|
| `doctor.bat` / `main.py doctor` | 环境自检 + 按张量索引算的精确摆位预算 |
| `main.py serve` | 启动内网 API 服务（`start_server.bat` 调的就是它）|
| `main.py backend` | 只启动 llama.cpp 后端，不起网关 |
| `main.py chat` | 终端交互式对话（会显示思考内容）|
| `main.py bench` | 基准测试：prefill / decode 实测吞吐 |
| `main.py models` | 列出本机所有可用 llama.cpp 运行时 |

### 停止

`stop_server.bat`，或在服务窗口按 `Ctrl+C`。

---

## 4. 内网 API

启动后是一个标准 OpenAI 兼容接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | 对话补全，支持 `stream: true` |
| POST | `/v1/completions` | 传统文本补全 |
| POST | `/v1/embeddings` | 向量接口 |
| GET | `/v1/models` | 模型列表 |
| GET | `/health` | 健康检查（含 GPU 显存与后端进程状态）|
| GET | `/stats` | 网关统计（QPS、延迟、token 数）|
| GET | `/metrics` | Prometheus 格式指标 |
| GET | `/docs` | Swagger 交互文档（不想写代码可直接在这里试）|

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://192.168.5.4:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="tile-35b-a3b",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=1024,
)
print(resp.choices[0].message.content)
```

### curl

```bat
curl http://192.168.5.4:8000/v1/chat/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\":\"tile-35b-a3b\",\"messages\":[{\"role\":\"user\",\"content\":\"你好\"}]}"
```

### ⚠ 两个必看的坑

**1. 这是推理模型，思考内容走另一个字段。** 回答前会先输出 `<think>` 段，
流式响应里思考内容在 `delta.reasoning_content`，正文才在 `delta.content`：

```python
for chunk in stream:
    delta = chunk.choices[0].delta
    think = getattr(delta, "reasoning_content", None)   # 思考内容
    text  = delta.content or ""                         # 正文
```

只读 `content` 会以为"没有输出"。本项目的网关、`chat` 命令、`client_example.py`
都已正确处理。

**2. `max_tokens` 不要设太小。** 思考过程会消耗 token 预算，设成 128 可能全部被
思考吃掉导致正文为空。**建议 ≥1024。**

### 鉴权

默认 `api_key: ""` 免鉴权。设置后，除 `/health`、`/stats`、`/metrics`、
`/v1/models`、`/docs` 外的接口都要求 `Authorization: Bearer <key>`。

后端 `llama-server` 只监听 `127.0.0.1`，不直接暴露到内网，避免绕过鉴权。

---

## 5. 实测性能

测试条件：RTX 4060 Laptop 8GB + Ryzen 9 7945HX，128K 上下文，`--cpu-moe`，
KV q8_0 放显存，`--load-mode none`，`--threads 16`，`ubatch 2048`。
吞吐取 llama.cpp 服务端返回的 `timings`（不用客户端计时）。

### 5.1 ubatch 是 MoE 下最重要的参数

原因：**每个 ubatch 都要把 256 个专家全部过一遍**，ubatch 越大，
同样长度的 prompt 需要的往返次数越少。

| ubatch | decode | **prefill** | 显存净增 | |
|---|---|---|---|---|
| 256 | 28.49 | 322.2 | 3885 MiB | |
| 512 | 28.54 | 442.9 | 3999 MiB | |
| 1024 | 28.76 | 771.8 | 4181 MiB | |
| **2048** | **29.19** | **1049.4** | 4519 MiB | ✅ 默认 |
| 4096 | 28.81 | 1207.2 | 5518 MiB | ❌ **会崩**，见 5.3 |
| 8192 | 24.93 | 456.5 | 7006 MiB | ❌ 显存挤爆，反而变慢 |

**prefill 从 443 提到 1049（2.4 倍），decode 几乎不变。**

### 5.2 摆位与加载模式

| 配置 | decode | prefill | 显存净增 | |
|---|---|---|---|---|
| `--threads 8` | 24.40 | 433.9 | 4001 MiB | |
| `--threads 16` | 24.95 | 425.6 | 4025 MiB | 线程数在噪声内 |
| `--threads 24` | 24.59 | 442.1 | 4013 MiB | |
| **`--load-mode none`** | **29.34** | **452.0** | 4072 MiB | ✅ **+17.6%** |
| `--n-cpu-moe 36`（后 4 层专家上显存）| **30.93** | 1089.3 | 6066 MiB | ✅ 可选 +6% |
| `--n-cpu-moe 32`（后 8 层）| **14.56** | **76.3** | 7235 MiB | ❌ 越过显存天花板 |

**线程数不是瓶颈。** 反推有效访存带宽：decode 每 token 读 0.441 GiB 专家权重，
29.34 × 0.441 = **12.9 GB/s**；prefill 也是约 13.4 GB/s。两者卡在同一个数字上，
说明瓶颈是 **CPU 侧小批量专家 GEMM 的访存效率**，既不是算力也不是内存带宽
（DDR5 双通道理论 83 GB/s）。

**显存天花板很硬**：总占用 6768 MiB 正常，7894 MiB 时 decode 从 30.9 崩到 14.6。

### 5.3 ⚠ 两个必须知道的边界

**① `ubatch` 不要超过 2048。** 4096 有两种失败模式：

- 加载期 OOM：`cudaMalloc failed`，要 2054 MiB 计算缓冲
- 运行期越界：`CUDA error: an illegal memory access was encountered`，
  后端进程直接死，请求返回 502

2048 时显存留 2.76 GiB，prefill 只比 4096 低 13%，且所有填充级别实测通过。
代码已加防呆：超过 2048 自动夹紧并告警（`allow_large_ubatch: true` 可放开）。

**② `load_mode: none` 需要锁定 14.6 GiB 锁页内存。** 它把专家张量放进
`CUDA_Host`（`cudaMallocHost`），内存被占满或碎片化时会**在 1.9 秒内**失败：

```
E ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 15704850432
E unable to allocate CUDA_Host buffer
```

`15704850432` 字节 = 14.63 GiB，正是专家权重。**这不是显存不足。**

两个后果：

- **绝对不能同时跑两个实例**（各要锁 14.6 GiB）。`serve` 会先探测端口，
  发现已有实例就拒绝启动。
- **锁不到时自动回落 mmap**（decode 慢 17%，但一定能起来），日志会写明原因。

启动总是不稳的话，把配置里的 `load_mode` 改成 `""`（llama.cpp 默认 mmap）。

---

## 6. 原理：为什么这么摆

### 6.1 这个模型不是普通 MoE

| 参数 | 值 |
|---|---|
| 架构 | `qwen35moe` |
| 层数 | 41（`blk.40` 是 MTP 头，llama.cpp **整段忽略**，实际计算 **40 层**）|
| 专家 | **256 个 / 每 token 激活 8 个** |
| 注意力 | 16 头 / **2 个 KV 头**（GQA 8:1），head_dim 256 |
| **`full_attention_interval`** | **4** |

`full_attention_interval = 4` 意味着**每 4 层里只有 1 层是真注意力**，
其余 3 层是**线性注意力（gated DeltaNet）**，循环状态大小固定、
**不随上下文增长**。

所以 KV cache 只按 **10 个全注意力层**算，而不是 40 层：

| 上下文 | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 32K | 0.63 GiB | 0.33 | 0.18 |
| 64K | 1.26 | 0.66 | 0.35 |
| **128K** | **2.51** | **1.33** | 0.70 |

**128K 的 KV 只要 1.33 GiB，放显存又放得下又更快**，所以 `kv_offload: true`。
（`--swa-full` 实测对显存和速度毫无影响，反证这些层不是滑窗注意力，
128K 上下文没有被截断。）

### 6.2 张量账本（不靠估算）

`scripts/gguf_tensors.py` 直接解析 GGUF 张量索引算出精确字节：

```
专家 *_exps           14.123 GiB   → 内存
其余可上显存            2.380 GiB   → 显存
MTP/nextn（被忽略）     0.363 GiB   → 文件里的死重
```

**这套账本预测显存 4.41 GiB，实测净增 4.40 GiB。**

### 6.3 架构

```
内网客户端 ──HTTP──> FastAPI 网关 (:8000, 0.0.0.0)
                          │  鉴权 / 限流排队 / 指标 / 日志
                          └──HTTP──> llama-server (:8080, 127.0.0.1)
                                          ├── CUDA ──> 注意力 + KV cache + embedding
                                          └── CPU  ──> MoE 专家权重（14.12 GiB，内存）
```

网关**不做任何逐 token 加工**，只做流式透传，所以不引入额外延迟。
套这一层的价值是：API Key 鉴权、并发排队保护、Prometheus 指标、统一访问日志，
以及后端崩溃时返回**后端日志尾部**而不是一个空白的 502。

### 6.4 显存不会泄漏

Windows 上父进程终止**不会**自动结束子进程。本项目在启动 `llama-server` 时把它加入
设置了 `KILL_ON_JOB_CLOSE` 的 **Windows Job Object**，无论父进程怎么死（Ctrl+C、
`taskkill /F`、任务管理器、崩溃），系统都会一并结束后端。
两条路径实测都能在 1 秒级释放显存（`scripts/test_jobobject.py`）。

---

## 7. 调优

### 按性价比排序

1. **`cache_reuse: 256`** —— 多轮长对话**必开**。否则每轮重新预填充整个历史，
   填到 9.6 万 token 时首字要 103 秒，这是最大的浪费。
2. **`n_cpu_moe: 36` + `ubatch 2048`** —— +6% decode，代价是显存到 6.8 GiB。
3. **`kv_cache_type: f16`** —— 不提速，但质量更好（多占 1.2 GiB）。

### 场景推荐

```bat
REM A. 默认：128K + 最佳综合速度
start_server.bat

REM B. 榨速度：后 4 层专家上显存
start_server.bat --n-cpu-moe 36 --ubatch 2048

REM C. 显存很紧（浏览器/IDE 吃显存）
start_server.bat --ubatch 1024

REM D. 3-4 人共享，各自上下文不长
start_server.bat --ctx 32768 --parallel 4
```

### 自动降级

启动时若加载失败，会按以下顺序自动重试（每次记录到 `runtime/logs/`）。
原则是**优先保住上下文长度**，先牺牲速度相关的项：

1. 配置值
2. 宿主内存锁定失败 → **回落 mmap 重试同一套参数**（不降 ubatch，病因不在显存）
3. 进程崩溃/断言 → 同样先试 mmap（原因不明时先排除内存锁定问题）
4. 专家退回内存（仅当设了 `n_cpu_moe`）
5. 逐级降 `ubatch`：2048 → 1024 → 512
6. KV 量化降到 `q4_0` → KV cache 移到内存
7. 上下文降到 75% → 32768
8. 最后才把 8 层留给 CPU

### 参数速查（按影响从大到小）

| 参数 | 默认 | 影响 |
|---|---|---|
| **`ubatch_size`** | 2048 | **MoE 下最关键**，直接决定 prefill（443 → 1049 tok/s）。**上限 2048** |
| **`load_mode`** | `none` | 不用 mmap，decode 快 17%。需锁定 14.6 GiB 锁页内存，失败自动回落 |
| **`cpu_moe`** | `true` | 专家权重放内存，整个方案的前提 |
| `n_cpu_moe` | -1 | 后 N 层专家上显存。**只有**配合 ubatch 2048 才安全，收益 +6% |
| `kv_offload` | `true` | KV 放显存。128K 只要 1.33 GiB |
| `context_size` | 131072 | 改小不省显存（KV 已很小），但能缩短首字延迟 |
| `kv_cache_type_k/v` | `q8_0` | `f16` 质量更好（+1.2 GiB，速度无变化）|
| `threads` | 16 | **实测 8/16/24 无差异**，瓶颈不在线程数 |
| `cache_reuse` | 0 | 多轮长对话**强烈建议设 256** |
| `flash_attention` | `true` | 必开，KV 量化需要它 |

---

## 8. 排障

**Q: 双击 `start_server.bat` 一闪而过？**
A: 说明脚本报错了。跑 `doctor.bat` 看自检结果。

**Q: 启动报 `unable to allocate CUDA_Host buffer`？**
A: 不是显存不足，是**锁页内存**锁不到 14.6 GiB。最常见原因是**已经有一个实例在跑**。
先 `stop_server.bat`，并确认任务管理器里没有残留的 `llama-server.exe`。
程序也会自动回落 mmap 继续启动。

**Q: 启动报 `0xC0000135` / 找不到 DLL？**
A: vendor 目录没进 PATH。跑 `doctor.bat` 看"运行时库"一行是否指向
`runtime\llama.cpp\backends\vendor\win-llama-cuda12-vendor-v2`。
缺失的话重新跑 `scripts/fetch_runtime.py`。

**Q: 请求返回 502，错误信息里有后端日志？**
A: `llama-server` 进程死了。最常见原因是 `ubatch_size` 超过 2048（CUDA 越界）
或显存被挤爆。检查 `config/server.yaml`。

**Q: 启动报 CUDA out of memory？**
A: 显存被别的东西占了。`nvidia-smi` 看占用，或 `start_server.bat --ubatch 1024`。
程序会自动降级重试。

**Q: 生成速度只有个位数 tok/s？**
A: 三个可能：① 层没上 GPU（看 `doctor` 的"设备枚举"里有没有 `CUDA0`）；
② 专家权重掉到磁盘了（内存不足，看是否需要加内存或用更小的量化）；
③ 显存越过天花板开始颠簸（降 `ubatch`）。

**Q: 内网连不上？**
A: 检查 Windows 防火墙是否放行 8000 端口，以及 `proxy_host` 是否为 `0.0.0.0`。
启动横幅会列出所有可用内网地址，注意排除虚拟网卡。

**Q: 想开机自启？**
A: 把 `start_server.bat` 的快捷方式放进 `shell:startup`。

**Q: MTP（`blk.40`）能用来加速吗？**
A: 现在不行——llama.cpp 会打印 `model has unused tensor ... -- ignoring` 全部跳过，
0.36 GiB 属于死重。等上游支持 `qwen35moe` 的 MTP 后可以用它做自投机解码。

更详细的排查过程与**测量方法论**（怎么避免把性能数据测错）见 [`NOTES.md`](NOTES.md)。

---

## 9. 项目结构

```
local LLM/
├── .gitignore
└── ornith-server/
    ├── main.py                 # CLI 入口（serve/backend/chat/bench/doctor/models）
    ├── start_server.bat        # 一键启动
    ├── stop_server.bat         # 停止所有 llama-server（不依赖 Python）
    ├── doctor.bat              # 环境自检
    ├── requirements.txt
    ├── README.md / NOTES.md
    ├── config/
    │   └── server.yaml         # 配置（可被环境变量和命令行覆盖）
    ├── src/ornith_server/
    │   ├── config.py           # 配置模型、校验、告警、ubatch 安全上限
    │   ├── bench.py            # 基准测试
    │   ├── core/
    │   │   ├── backend.py      # 运行时探测、GPU 查询、DLL 路径配对
    │   │   ├── server.py       # 进程管理、MoE 摆位命令行、分级降级
    │   │   ├── gguf.py         # GGUF 元数据 + 张量索引解析、摆位预算
    │   │   └── jobobject.py    # Windows Job Object 兜底
    │   └── api/
    │       ├── gateway.py      # FastAPI 网关（OpenAI 兼容 + 流式透传 + 崩溃诊断）
    │       ├── middleware.py   # 鉴权、并发排队、用量统计
    │       └── metrics.py      # 指标收集（JSON + Prometheus）
    ├── scripts/                # 见下方清单
    └── runtime/                # 运行时与日志（.gitignore 排除，用 fetch_runtime.py 恢复）
        ├── llama.cpp/backends/ #   912 MiB，自包含，不依赖 LM Studio
        └── logs/
```

### 脚本清单

**准备**

| 脚本 | 用途 |
|---|---|
| `fetch_runtime.py` | 克隆后恢复 `runtime/`（那些二进制不入库）|
| `check_deps.py` | 依赖自检 |

**看模型**

| 脚本 | 用途 |
|---|---|
| `gguf_raw.py` | 转储 GGUF 全部元数据（层数、专家数、全注意力层间隔）|
| `gguf_tensors.py` | **精确张量账本**：专家 vs 其余、量化分布、每层专家体积 |
| `sysinfo.py` | 内存与磁盘（不依赖 WMI，避免权限问题）|

**测性能**

| 脚本 | 用途 |
|---|---|
| `moe_probe.py` | 单次摆位实测（加载 / 显存 / 内存 / decode / prefill）|
| `moe_sweep.py` | 批量扫描与对比表（`--preset default\|ubatch\|final`）|
| `stress_ctx.py` | **中文长提示逐级填充到 128K，检测崩溃** |

**验证**

| 脚本 | 用途 |
|---|---|
| `check_syntax.py` | 语法自检 |
| `test_cli.py` | 命令行参数解析单测（43 项）|
| `test_load_ladder.py` | 加载降级阶梯单测（18 项）|
| `test_jobobject.py` | 显存不泄漏验证（8 项）|
| `e2e_test.py` | 接口端到端（21 项）|
| `final_acceptance.py` | 默认 128K 配置验收 |

**诊断 / 示例**

| 脚本 | 用途 |
|---|---|
| `diag_sse.py` | 打印流式响应原始片段（确认 `reasoning_content` 等字段）|
| `diag_usage.py` | 对比 usage / timings / metrics 三种取数口径 |
| `client_example.py` | 调用示例（SDK / 原生 HTTP / 流式）|

> 诊断脚本**不自己启动后端**，直接打一个已经在跑的服务，默认 `http://127.0.0.1:8000`，
> 加 `--base-url http://127.0.0.1:8080` 可看 llama.cpp 的原始输出。

### 自检清单

```bat
cd "D:\personal\AI_output\local LLM\ornith-server"
D:\anaconda\envs\test1\python.exe scripts\check_syntax.py
D:\anaconda\envs\test1\python.exe scripts\test_cli.py
D:\anaconda\envs\test1\python.exe scripts\test_load_ladder.py
D:\anaconda\envs\test1\python.exe main.py doctor
D:\anaconda\envs\test1\python.exe scripts\e2e_test.py
D:\anaconda\envs\test1\python.exe scripts\test_jobobject.py
D:\anaconda\envs\test1\python.exe scripts\stress_ctx.py       REM 128K 上下文压力验证
D:\anaconda\envs\test1\python.exe scripts\final_acceptance.py
```

---

## 10. 参考

- [llama-server 参数文档](https://mintlify.wiki/ggml-org/llama.cpp/api/tools/llama-server)
- [llama.cpp 并行推理参数讨论 #18308](https://github.com/ggml-org/llama.cpp/discussions/18308)
- [llama.cpp vs vLLM 对比](https://theneuralbase.com/llamacpp/qna/llama-cpp-vs-vllm-comparison/)
