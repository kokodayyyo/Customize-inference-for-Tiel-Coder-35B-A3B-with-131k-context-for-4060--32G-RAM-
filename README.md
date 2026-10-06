# Personal参数调优推理项目，试图让Tiel-Coder apex量化模型能在200k上下文保持30tokens的速度

把大 MoE 模型用**分层摆位**跑在普通游戏本上，对外提供 **OpenAI 兼容的内网 API** 和
**网页控制台**。核心是把模型按“谁来算、放哪”拆开，让最贵的部分刚好放下：

| 部分 | 体积（以 Tile-35B 为例）| 放哪 |
|---|---|---|
| MoE 专家权重 `*_exps` | ~12–14 GiB | **内存**（每 token 只读一小撮，放显存装不下）|
| 注意力 / 线性注意力 / embedding / 共享专家 | ~1.5–2.4 GiB | **显存**（每 token 都要算）|
| KV cache（只有少数全注意力层随上下文增长）| 0.7–2.0 GiB | **显存**（放得下又更快）|

> **内存比显存更关键**：权重必须完整存放，每 token 只是不全读；“激活 3B”是算力
> 成本，不是显存需求。

**硬件**（下表“本机”即全部数字的标定环境）：

| 项 | 最低 | 本机 |
|---|---|---|
| GPU | NVIDIA，**≥8 GiB 显存**，CUDA 12 | RTX 4060 Laptop 8 GiB |
| 内存 | **≥24 GiB**（专家权重需 ~13–15 GiB 常驻 + 余量）| 31.2 GiB |
| 磁盘 | ~17 GiB 放模型 | D: 435 GiB 可用 |
| CPU | 任意（prefill 速度取决于它）| Ryzen 9 7945HX 16C/32T |

软件：**Windows** + **Python 3.10+**；Node 18+ 只有跑界面自检才需要。

---

## 1. 这是什么

一个**纯本机的内网推理服务**：`FastAPI 网关` 把请求转发给 `llama-server`（llama.cpp
CUDA 后端），并提供网页控制台来选模型、看实时数据、管理进程。

- 对外是标准 OpenAI 接口，内网任何程序/SDK 都能调；
- 网关**不做逐 token 加工**，只做流式透传，所以不引入额外延迟；
- 模型、参数、扫描目录都能在网页上点，不用改代码。

架构：

```
内网客户端 ──HTTP──> FastAPI 网关 (:8000)  ──HTTP──> llama-server (:8080)
                       鉴权/排队/指标/日志        ├─ CUDA ─> 注意力 + KV + embedding
                                                  └─ CPU  ─> MoE 专家权重（内存）
```

---

## 2. 下载后要补什么

仓库**不含**两类大文件，克隆后按下面补全即可。

**① llama.cpp 运行时**（`runtime/`，约 900 MiB，未入库——`cublasLt64_12.dll` 单个
674 MB 超过 GitHub 单文件上限）：

```bat
D:\anaconda\envs\test1\python.exe LLM-server\scripts\fetch_runtime.py
```

会自动挑**最新的 CUDA12 引擎**并配对正确的 vendor DLL。没有 LM Studio 就用
`--from "任意含 llama-server.exe 的目录"`，或自己编译 / 用官方 release。

**② 模型文件**（GGUF，需自行获取，放到任意位置后改配置指向它）：

| 文件 | 体积 | 上下文 | 说明 |
|---|---|---|---|
| `Cyber-Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf` | 13.74 GiB | **200K** | 更快更省（Q3_K/IQ3_XXS）|
| `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` | 16.88 GiB | 128K | 质量更好（IQ4_XS）|
| `gemma-4-26B-A4B-heretic-APEX-Compact.gguf` | 14.43 GiB | 150K | gemma4 MoE（另一套架构）|

换成**其它同架构的 GGUF** 也能跑，但性能/占用要重新标定（见第 6 节）；非 MoE
（dense）也能列出来，只是没有摆位收益，会用保守默认参数启动。

**③ Python 依赖**（只有网关层需要的几个，**不需要 torch**）：

```bat
pip install -r LLM-server\requirements.txt
```

---

## 3. 怎么用

```bat
cd /d "D:\personal\AI_output\local LLM\LLM-server"
doctor.bat          REM ① 环境自检：GPU / 运行时 / 显存内存预算
start_server.bat    REM ② 起服务 + 控制台（不加载模型，约 1 秒）
```

浏览器打开 **http://127.0.0.1:8000/ui**：

- 自动扫描目录里的 `.gguf`，每个模型一张卡片（体积、结构、显存/内存占用、是否已标定）；
- **目录可自己加/删**：点「＋ 添加目录」用目录选择器（可逐级浏览，含模型的目录会标
  「含模型」，也能直接粘路径）；改动存到 `config/model_roots.json`；
- 点**「启动此模型」**即套用它标定好的参数加载（约 10–15 秒，有进度条）；
- **视觉开关**：模型同目录若有视觉投影（`mmproj*.gguf`），卡片上会出现「视觉」开关，
  **打开才加载**（默认关）；没检测到就不显示；
- 换个模型＝再点另一个，自动“停旧起新”；顶部「停止服务」只停**模型**、保留控制台。

| 脚本 | 用途 |
|---|---|
| `start_server.bat` | 起服务 + 控制台（**不加载模型**）|
| `doctor.bat` | 环境自检 + 精确摆位预算 |
| `stop_server.bat` | 停网关 + 所有 llama-server（不依赖 Python）|
| `main.py serve` | 同上，可加 `--autostart` 开机直接加载 |

**想开机就加载某个模型**（无人值守）：`start_server.bat --autostart --model "D:\path\to.gguf"`。

> 没加载模型时 `/v1/*` 返回明确的 **503**（「请到控制台 /ui 启动一个」），而不是含糊的 502。

---

## 4. 预置了哪些模型配置（按 32G 内存 + 4060 标定）

三套参数都写在 `config/models.yaml`，网页上点一下即用。数字是 `scripts/moe_probe.py`
实测（控制台会标「实测」；以它为准）：

| 模型 | 文件（完整名）| 体积 | 上下文 | decode（空载）| prefill | 显存净增 | 特点 |
|---|---|---|---|---|---|---|---|
| Tile 35B-A3B APEX | `Cyber-Tiel-Coder-35B-A3B.APEX-I-MiniPlus-V2.1.gguf` | 13.74 GiB | **200K** | 30.5 tok/s | 1137 tok/s | 5.02 GiB | 更快更省，量化更低 |
| Tile 35B-A3B 基准版 | `Cyber-Tiel-Coder-35B-A3B-MTP-UD-IQ4_XS.gguf` | 16.88 GiB | 128K | 29.2 tok/s | 1049 tok/s | 4.41 GiB | 质量更好 |
| Gemma 4 26B-A4B Heretic APEX | `gemma-4-26B-A4B-heretic-APEX-Compact.gguf` | 14.43 GiB | 150K | 29.3 tok/s | 1324 tok/s | 5.03 GiB | gemma4 MoE，KV 很小 |

共同的定盘参数：`cpu_moe`（专家放内存）、`ubatch 2048`、KV `q8_0` 放显存、
`load_mode none`（不用 mmap，快 ~17%）。所有模型都在 **8 GiB 显存**里留了安全余量。

> Tile-35B 的上下文能到 200K，是因为它每 4 层只有 1 层真注意力（其余是线性注意力，
> 状态不随上下文增长），KV 只有 ~2 GiB；Gemma-4 是滑窗注意力（window 1024），
> 只有 5 层随上下文增长，KV 更小。**为什么 8G 卡能跑这么长，见 NOTES。**

---

## 5. 必读的坑 + 排障

**① `load_mode: none` 要锁定十几 GiB 锁页内存。**
它把专家权重放进 `CUDA_Host`（一次锁 ~12–15 GiB **物理**内存），换来 decode 快 ~17%：
- **不能同时跑两个实例**（各自都要锁一份）。报 `unable to allocate CUDA_Host buffer`
  就是这个 —— **不是显存不足**。先 `stop_server.bat`，并确认任务管理器里没有残留的
  `llama-server.exe`（换模型/重载前尤其要注意）。
- 锁不到时程序会**自动回落 mmap**（慢 17% 但一定能起）。

**② `ubatch_size` 上限就是 2048。** 4096 在真实长提示下会 CUDA 越界 / OOM，后端直接
死（返回 502）。代码已自动夹紧（`allow_large_ubatch: true` 可放开，不建议）。

**③ 这是推理模型，思考内容在另一个字段。** 流式响应里正文在 `delta.content`，思考在
`delta.reasoning_content`；只读 `content` 会以为“没有输出”。`max_tokens` 别太小
（**建议 ≥1024**），否则会被思考吃光导致正文为空。

**④ “服务起不来 / 只有 Python 连不上本机服务”？** 多半是系统代理（Clash/v2ray）把
`127.0.0.1` 也代理了 → 502。项目已对所有访问本机的 httpx 客户端设 `trust_env=False`；
你自己写的脚本也要加。诊断：`python scripts\diag_http.py`。

**⑤ 内网暴露 / 鉴权。** 默认 `api_key: ""` 表示内网免鉴权，`/admin/*` 还能启停模型、
列目录。**发到内网时建议** `start_server.bat --api-key sk-xxx`。`/admin/*` 的写操作已
拒绝跨站请求、默认不开跨域。

常见错误速查：

| 现象 | 处理 |
|---|---|
| 双击 bat 一闪而过 | 跑 `doctor.bat` 看报错 |
| `unable to allocate CUDA_Host buffer` | 有残留实例占内存 → `stop_server.bat`（不是显存问题）|
| `0xC0000135` / 找不到 DLL | vendor 目录没进 PATH，重跑 `fetch_runtime.py` |
| 请求 502 且带后端日志 | 后端崩了：多为 ubatch>2048 或显存挤爆 |
| 生成只有个位数 tok/s | 层没上 GPU / 专家掉到磁盘 / 显存颠簸（降 ubatch）|
| 内网连不上 | 防火墙放行 8000、`proxy_host` 为 `0.0.0.0` |

---

## 6. 配置与扩展

配置文件 `LLM-server/config/server.yaml`，优先级：**默认值 < 配置文件 < 环境变量
`LLM_*` < 命令行**。最常改的几项：

| 配置 | 默认 | 说明 |
|---|---|---|
| `model_path` | 本机路径 | GGUF 路径（只影响“首次/自动加载”，网页里另选不受影响）|
| `model_alias` | `tile-35b-a3b` | 对外模型名，客户端 `model` 字段填它 |
| `proxy_host` / `proxy_port` | `0.0.0.0` / `8000` | 对外监听 |
| `api_key` | `""` | 空 = 内网免鉴权 |
| `cpu_moe` | `true` | 专家放内存（**整个方案的前提**）|
| `ubatch_size` | `2048` | MoE 下最关键的性能参数，**上限 2048** |
| `load_mode` | `"none"` | 不用 mmap，decode 快 ~17%（见坑 ①）|
| `cache_reuse` | `0` | 多轮长对话**强烈建议设 256**，否则每轮重填整个历史 |

命令行速查：`start_server.bat --api-key sk-xxx`、`--port 9000`、
`--ubatch 1024`（显存紧张）、`--model "D:\other.gguf"`。

**加新模型**：放进扫描目录 → 控制台点「重新扫描」即可用（用保守默认参数）。
想调优就在 `config/models.yaml` 加一条 profile（`match` 用文件名**子串**即可匹配）。
调参流程：

```
scripts/gguf_tensors.py   # 看张量账本（专家多大、谁放内存谁放显存）
scripts/moe_probe.py      # 测一套参数的 显存/内存/decode/prefill
→ 把 settings 和实测值写回 models.yaml 的该条 profile
```

**API**（OpenAI 兼容）：`/v1/chat/completions`（支持 `stream`）、`/v1/completions`、
`/v1/embeddings`、`/v1/models`、`/health`、`/stats`、`/metrics`、`/docs`（Swagger）。

```python
from openai import OpenAI
client = OpenAI(base_url="http://192.168.5.4:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="tile-35b-a3b-apex",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=1024,   # 别设太小，见坑 ③
)
print(resp.choices[0].message.content)
```

**项目结构**（精简）：

```
LLM-server/
├── main.py  start_server.bat  stop_server.bat  doctor.bat
├── config/     server.yaml（全局）  models.yaml（模型注册表）
├── src/llm_server/
│   ├── config.py  net.py  models_registry.py  bench.py
│   ├── core/   server.py（进程/摆位/降级）backend.py  gguf.py  jobobject.py
│   ├── api/    gateway.py（转发）admin.py（/admin + ModelManager）middleware.py  metrics.py
│   └── web/    index.html（控制台，单文件）
├── scripts/    （工具与测试，见 NOTES）
└── runtime/    （llama.cpp 运行时不入库，fetch_runtime.py 恢复）
```

> 原理推导、完整标定过程、测量方法论、脚本清单 → [`LLM-server/NOTES.md`](LLM-server/NOTES.md)。
