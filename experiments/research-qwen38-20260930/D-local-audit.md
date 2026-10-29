# D · 本机审计：Qwen3.8-27B @ DGX Spark 现役部署资产

> 审计日期：**2026-09-30**（宿主本地时区 HKT = UTC+8）· 主机 `spark-XXXX`（DGX Spark GB10，aarch64，Ubuntu 24.04.4，kernel 6.17.0-1031-nvidia）
> 审计方式：**只读**。未重启服务、未改配置、未写入任何模型缓存。
> ⚠️ 重要时间约定：**容器/SGLang 日志时间戳是 UTC**（比宿主 HKT 慢 8 小时）；`journalctl`/watchdog 日志用宿主本地时间。本文凡引用容器日志均标 UTC，并换算 HKT。

---

## 1. 现役确切配置表（2026-09-30 15:39 HKT 实测快照）

### 1.1 运行时身份

| 项 | 值 | 取证来源 |
|---|---|---|
| 容器 | `qwen38-27b-sglang`，ID `1f443a46d0c4…5b6564f` | `docker inspect` |
| 状态 | `Up 11 days`，`Status=running`，**`RestartCount=0`**，**`OOMKilled=false`** | `docker ps` / `docker inspect .State` |
| Created | `2026-09-18T08:15:52Z` = **09-18 16:15:52 HKT** | `docker inspect .Created` |
| StartedAt | `2026-09-18T12:10:43Z` = **09-18 20:10:43 HKT** | `docker inspect .State.StartedAt` |
| FinishedAt（上次退出） | `2026-09-18T12:05:43Z` = 09-18 20:05:43 HKT（**人工 SIGTERM 优雅退出**，非崩溃） | `docker inspect` + 日志 L1661 |
| 宿主 uptime | `up 11 days, 19:28` | `uptime` |
| restart policy | `unless-stopped` | `HostConfig.RestartPolicy` |
| HealthCheck | 无（`Health=<nil>`，靠外部 watchdog） | `docker inspect` |

**镜像**

| 项 | 值 |
|---|---|
| RepoTag | `lmsysorg/sglang:dev-qwen38-27b-dflash2` |
| Image ID | `sha256:60166a99b6610e2b4b3febbeb6411e4cbc9338e996ab6c80d3339f86a7b5b9c3` |
| RepoDigest | `sha256:616a3e97f45191af975896cfa644279096cb31bd408a071c2e99ca7209c3cafe` |
| 镜像构建时间 | `2026-08-22T02:58:08Z`，`Architecture=arm64` |
| 镜像内 SGLang | **`0.0.0.dev1+g5f55db35e`**（实测 `docker exec … sglang.__version__`） |
| 构建指纹 | `SGLANG_BUILD_COMMIT=5f55db35e926d50676f75b812640ea2410b0fe0e`，`SGLANG_BUILD_URL=…/actions/runs/32545892070` |
| torch / transformers | `torch 2.13.0+cu130` / `transformers 5.12.1` |
| CUDA / FlashInfer / NCCL | CUDA `13.0.3`（cudart 13.0.96） / `FLASHINFER_VERSION=0.6.17` / NCCL `2.28.3-1` |
| 同机其他镜像 | `lmsysorg/sglang:qwen38-27b`（38.6GB，旧 DSpark 形态，保留）、`eugr/spark-vllm-b12x:latest`（23GB，ornith 回滚用） |

**模型与变体（容器内路径）**

| 角色 | 路径 | 实际内容 |
|---|---|---|
| target | `/root/.cache/huggingface/variants/qwen38-nvfp4-yarn1024` | `RadixArk/Qwen3.8-27B-NVFP4` snapshot `319f741c…` + **config.json 内联 YaRN factor 4.0**；`text_config.max_position_embeddings=1048576`；`config.json` 是真实文件（73KB），其余 21 项为相对软链 |
| draft | `/root/.cache/huggingface/variants/qwen38-dflash2-yarn1024` | `z-lab/Qwen3.8-27B-DFlash2`（本地目录 `dflash2-draft`）+ 烧入同一套 YaRN；实测 `architectures=["DFlash2DraftModel"]`、`num_hidden_layers=5`、`layer_types` 全 `sliding_attention`、`sliding_window=2048`、32 Q/8 KV、`head_dim=128`、`dflash_config.block_size=8`、`target_layer_ids=[5,19,33,47,61]` |
| `--json-model-override-args` | `'{}'` | **保持为空是硬约束**（见 §3 P1） |

**target 架构（实测 config.json）**：`Qwen3_5ForConditionalGeneration`，64 层 = **48 linear_attention + 16 full_attention**，24 Q heads / **4 KV heads** / `head_dim=256`，`partial_rotary_factor=0.25`，`mrope_section=[11,11,10]`，`mtp_num_hidden_layers=1`（in-checkpoint MTP 头存在但**未启用**），`language_model_only=false`（含 `vision_config`），量化 `modelopt_mixed`（MIXED_PRECISION）。

### 1.2 启动参数（`docker inspect .Args` 原样）

```
python3 -m sglang.launch_server
  --model-path /root/.cache/huggingface/variants/qwen38-nvfp4-yarn1024
  --served-model-name qwen3.8-27b        --trust-remote-code
  --mem-fraction-static 0.70             --attention-backend flashinfer
  --chunked-prefill-size 8192            --disable-prefill-cuda-graph
  --disable-flashinfer-autotune          --kv-cache-dtype fp8_e4m3
  --mamba-ssm-dtype bfloat16             --mamba-full-memory-ratio 4.21
  --mamba-radix-cache-strategy extra_buffer   --max-mamba-cache-size 48
  --max-running-requests 4               --context-length 1010000
  --json-model-override-args {}
  --speculative-algorithm DFLASH         --speculative-dflash-block-size 8
  --speculative-draft-model-quantization unquant
  --speculative-num-draft-tokens 8
  --speculative-draft-model-path /root/.cache/huggingface/variants/qwen38-dflash2-yarn1024
  --enable-torch-compile --torch-compile-max-bs 4
  --cuda-graph-max-bs-decode 4           --num-continuous-decode-steps 2
  --reasoning-parser qwen3               --tool-call-parser qwen3_coder
  --sampling-defaults model              --enable-metrics
  --host 127.0.0.1 --port 8888
```

`/get_server_info` 交叉核对一致：`context_length=1010000`、`mem_fraction_static=0.7`、`max_running_requests=4`、`kv_cache_dtype=fp8_e4m3`、`mamba_radix_cache_strategy=extra_buffer`、`max_mamba_cache_size=48`、`page_size=1`、`retraction_policy=length`、`disable_radix_cache=false`、`disable_flashinfer_autotune=true`、`enable_torch_compile=true`、`cuda_graph_max_bs_decode=4`、`watchdog_timeout=300`、`speculative_algorithm='DFLASH'`、`speculative_dflash_block_size=8`。

### 1.3 上下文 / KV / mamba 实际容量（启动日志 L1794–L1937，09-18 12:13–12:17Z）

| 项 | 值 |
|---|---|
| context-length | **1,010,000**（`/v1/models` 报 `max_model_len: 1010000` ✅；YaRN factor 4 理论上界 1,048,576，留 3.7% 余量） |
| KV 池 | `max_total_num_tokens=1,205,656`（第二次启动）；首启 1,197,439 —— 即约 **1.20M token** fp8_e4m3（≈39GB） |
| 启动后可用显存 | `available_gpu_mem=29.89 GB` |
| CUDA graph | decode `backend=full, bs=[1,2,3,4]`；prefill **disabled**；目标 verify 捕获 179.0s / 2.20GB，草稿 verify 捕获 24.0s / 0.99GB |
| mamba/GDN 状态池 | `max_mamba_cache_size=48`（=并发 4 × 12），`mamba_full_memory_ratio=4.21`，`mamba_ssm_dtype=bfloat16` |
| 权重加载 | target 145.5s / 21.30GB；draft 28.6s / 4.13GB |
| 冷启动 | 重建 16:15:52 → `/health` 200 于 16:23:17 HKT（**7 分 25 秒**），与文档一致 |

### 1.4 Docker 级限制与挂载

| 项 | 值 |
|---|---|
| 网络 / IPC | `NetworkMode=host`，`IpcMode=host`，`Privileged=true` |
| 内存 | `--memory 107374182400`（**100 GiB**），`--memory-swap` 同值（**禁 swap**） |
| CPU | `--cpuset-cpus 5-9,15-19`（GB10 的 10 个 Cortex-X5 大核） |
| shm | `34359738368` = 32 GiB |
| GPU | `DeviceRequests: Capabilities=[[gpu]], Count=-1`（`--gpus all`） |
| env | `HF_HOME=/root/.cache/huggingface`、`TRITON_CACHE_DIR=/root/.triton`、`TORCHINDUCTOR_CACHE_DIR=/root/.torch-cache`、**`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`**、**`HF_HUB_OFFLINE=1`** |
| 挂载（3 个 bind） | `…/qwen38-model-cache/.triton → /root/.triton`；`…/qwen38-model-cache/.torch-cache → /root/.torch-cache`；`…/qwen38-model-cache → /root/.cache/huggingface` |
| LogConfig | `json-file`（**无 max-size/max-file 轮转**；宿主 log 已 239,846 行） |

### 1.5 实时资源（`docker stats --no-stream` + 宿主）

| 指标 | 值 |
|---|---|
| 容器内存 | **8.382 GiB / 100 GiB（8.38%）** |
| 容器 CPU | 216.44%（多线程 prefill/decode 混合） |
| PIDs | 186 |
| Block I/O | 读 6.03GB / 写 1.54GB |
| NET I/O | 0B / 0B（host 网络不计入） |
| 宿主 `free -h` | total 121Gi，**used 101Gi**，free 10Gi，buff/cache 10Gi，**available 19Gi** |
| 宿主 swap | 15Gi total，**618Mi used**（`swapon --show` 16G / 618M） |
| 宿主 load | 1.20 / 1.30 / 1.25 |

### 1.6 对外链路（实测）

| 端口 | 服务 | 实测 |
|---|---|---|
| `127.0.0.1:8888` | 容器 SGLang | `/v1/models` **200**；`/health` **200**（1.0s） |
| `0.0.0.0:4000` | `switchyard.service`（systemd user，active） | `/v1/models` **200**，返回 cloud-test / ornith-local / qwen3.8-27b / switchyard |
| `<LAN-IP-1>:8888` 等 | `vllm-open-proxy.py`（LAN/Tailscale → 127.0.0.1:8888） | **000 = 不可达；进程未运行**（`pgrep -af vllm-open-proxy` 无结果；`ss -ltn` 只有 `0.0.0.0:4000` 与 `127.0.0.1:8888`） |
| `~/.config/systemd/user/switchyard.service` | `switchyard-server --config /home/<user>/routes.toml`，`Restart=always` | active；11 天内 journalctl **无 error/panic**（grep 命中的全是 `error=""` 的正常 INFO 行） |
| `qwen38-watchdog.service` | `dgx-spark-notes/qwen38-sglang/qwen38-watchdog.sh --run`（PID 2349） | active；env：`INTERVAL=30 FAIL_THRESHOLD=4 STARTUP_GRACE=1200 MAX_RESTARTS_PER_HOUR=3 BACKOFF=1800 ALERT_TARGET=discord:#zunni-world` |
| 回滚资产 | `ornith-b12x-serve` 容器 `Exited (0) 2 weeks ago`（保留） | ✅ 仍在 |

---

## 2. 变体 farm 现状（`/home/<user>/projects/qwen38-model-cache/variants/`，11 个目录）

统一实现（`make-yarn-variant.py`）：**只有 `config.json` 是真实文件，权重/tokenizer 全是指向 HF 缓存或 `dflash2-draft` 的相对软链**（零拷贝、不动被校验的 blob）。命名规则 `yarn{round(ctx/1024)}` = **factor × 262144**（所以服务 506,000 用 `yarn512`）。

| 目录 | 架构 | 顶层 max_pos | text max_pos | rope_type / factor / orig_max | 用途与状态 |
|---|---|---|---|---|---|
| `qwen38-nvfp4-yarn512` | Qwen3_5ForConditionalGeneration | — | 524288 | yarn / **2.0** / 262144 | 方案 B（DSpark 506K）的 target |
| **`qwen38-nvfp4-yarn1024`** | Qwen3_5ForConditionalGeneration | — | 1048576 | yarn / **4.0** / 262144 | ★**现役 `--model-path`**（20 项软链 + 真实 config.json） |
| `qwen38-dflash2-yarn512` | DFlash2DraftModel | 524288 | 524288 | yarn / 2.0 / 262144 | DFlash2 草稿烧 YaRN（512K 档） |
| **`qwen38-dflash2-yarn1024`** | DFlash2DraftModel | 1048576 | 1048576 | yarn / 4.0 / 262144 | ★**现役 `--speculative-draft-model-path`** |
| `qwen38-dflash2-textconfig` | DFlash2DraftModel | 262144 | 524288 | default（顶层未改 rope） | 路线 B 验证：给草稿补**完整 text_config** 让 CLI 覆盖走 merge 分支 |
| `qwen38-dspark-yarn512` | DSparkDraftModel | 524288 | 524288 | yarn / **2.0** / 262144 | `MODE=draftyarn` 对照臂（草稿换同套绳） |
| `qwen38-dspark-yarn1024` | DSparkDraftModel | 1048576 | 1048576 | yarn / 4.0 / 262144 | 同上（1M 档） |
| `qwen38-dspark-maxpos-yarn512` | DSparkDraftModel | 524288 | 524288 | yarn / **32.0** / 8192（**训练值不动**） | `MODE=draftmaxpos` 默认臂（方案 B 主推） |
| `qwen38-dspark-maxpos-yarn1024` | DSparkDraftModel | 1048576 | 1048576 | yarn / 32.0 / 8192 | DSpark 1M 待实测（§5 待办 2） |
| `qwen38-dspark-cfg-textconfig` | DSparkDraftModel | 262144 | 524288 | yarn / 32.0 / 8192 | 路线 B 验证（**实测可行**） |
| `qwen38-dspark-cfg-ropeplanB` | DSparkDraftModel | 524288 | 524288 | factor 1.0 / type 缺失 | **已实测证伪**的"骗过校验"方案，留档 |

> 草稿权重目录：`qwen38-model-cache/dflash2-draft`（`z-lab/Qwen3.8-27B-DFlash2`，2.6GB）与 hub 缓存里的 `RadixArk/Qwen3.8-27B-DSpark`、`RadixArk/Qwen3.8-27B-NVFP4`。
> `.concurrency-test/` 只含 14 个原始请求体（`req1..6.json`、`stress1..8.json`），**无结论文档**；唯一结论记录在 `notes/qwen38-dflash2-crash-20260918.md` §5 与 `recreate-concurrency4-20260918.sh` 头部注释。

---

## 3. 已知坑清单（文档 + 日志实证）

**标记说明**：✅ = 在本现役容器的 239,846 行日志里**真的出现过**；⛔ = 出现过但属于**修复前 / 其它容器**，本日志中计数为 0；📄 = 仅有文档记录。

### 3.1 A 类：YaRN × 投机解码草稿

| # | 坑 | 日志实证 |
|---|---|---|
| P1 | 走 `--json-model-override-args` 传 YaRN → 覆盖被**转发给独立草稿模型**，`{text_config:{…}}` 被 setattr 成裸 dict → `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'`，启动即崩。泄漏点 `speculative/dspark_components/dspark_config.py:35,136`、`arg_groups/speculative_hook.py:217`；上游 issue sglang#33505 **open 0 评论**，PR #35583 已关 / #35841 未合并 | ⛔ 本日志 `max_position_embeddings` AttributeError = 0（现役已改为 config 内联；`--json-model-override-args` 实测为 `{}`） |
| P2 | 「给草稿顶层补 `max_position_embeddings` + `rope_parameters{factor:1.0,type:default}` 就能骗过校验」→ **实测证伪**（崩的是覆盖新建的嵌套 stub） | 📄（变体 `qwen38-dspark-cfg-ropeplanB` 留档） |
| P3 | DFLASH **不支持** `--mamba-radix-cache-strategy extra_buffer_lazy` → 启动断言失败 | ⛔ 本日志 0 次断言；现役用 `extra_buffer` ✅（`server_args` 实测 `uses_mamba_radix_cache=True`, `mamba_radix_cache_strategy='extra_buffer'`） |
| P4 | `lmsysorg/sglang:qwen38-27b` 未注册 `DFlash2DraftModel` → `is not a registered model` | ⛔ 0（现役用 dev 镜像） |
| P6 | verify 窗口不做剩余预算裁剪（上游 #33454，PR #33963 未合并）→ 顶格上下文触发 **CUDA illegal address**，引擎直接挂 | ⛔ 本日志 `illegal` = **0**；现役 `--context-length 1010000` 留了 3.7% 余量 |
| P7 | 外挂草稿自身也吃 KV（DFlash2 草稿 2048 滑窗开销小；DSpark 草稿 5 层全注意力 ≈ +10.2 KB/token） | 📄（架构实测已确认：DFlash2 草稿 5 层全 `sliding_attention`/`sliding_window=2048`） |

### 3.2 B 类：DFlash2 采样器并发 bug（本机最严重的历史事故）

| # | 坑 | 日志实证 |
|---|---|---|
| B1 | `dflash_worker_v2.py:206` 按 `max_bs = max(decode cuda_graph.bs)`（= `--cuda-graph-max-bs-decode` = **4**）分配 `greedy_mask`/`temperatures`，而 `:218/:231` 用**本轮 bs**（= 同时在跑的请求数）切片 → **bs ≥ 5 必然 `RuntimeError: The size of tensor a (4) must match the size of tensor b (5/7)`** → scheduler 自杀（SIGQUIT → `kill_process_tree`）→ **容器 exit 0** | ⛔ 现役日志 **`Scheduler hit an exception` = 0**、**`dflash_worker_v2` 关键字 = 0**。历史 5 次崩溃（09-15 11:14 `a(4)vs b(7)`、09-15 12:40、09-17 15:14 `a(4)vs b(5)`、09-17 15:24 `a(4)vs b(5)`、09-18 ~16:01）**都发生在修复前的旧容器**，其 traceback 不在本日志中 |
| B2 | 「exit 0 的崩溃」让 `on-failure` 策略永不触发 → 09-15 服务**静默停摆 66 分钟** | 📄 → 已加 `--restart unless-stopped` |
| B3 | **纯 `docker restart` 治不了**（参数不变）：09-17 第二次崩溃距恢复仅 **2 分 16 秒** | 📄 |
| B4 | 触发链 = 长上下文风暴（单请求 70–90 万 token）→ 排队 → 并发顶到 5~6 → 崩。「卡」和「崩」是同一条因果链 | ✅ 本日志侧证：`#pending-token` 峰值 **1,205,999**、峰值 `full token usage 0.88`（长请求吃掉近 9 成池）；但 `#running-req` 峰值 **恰好 4**（修复生效） |
| B5 | watchdog 冷启动误杀：重建后冷启动 ≈7.5 min，而 watchdog 阈值过紧 → 12:42 把正在加载的实例 `kill -9` | 📄 → 已加 `STARTUP_GRACE=1200` ✅（现役 `/get_server_info` 有 `watchdog_timeout=300` 是 SGLang 自身的，与守护无关） |
| B6 | `/health` 6s 超时 ≠ 挂了：长 prefill 期间 HTTP 事件循环迟滞是常态 | ✅ **本日志真出现**：`health_generate` 探活断连/超时 traceback 3 次（L104347、L150323、L182464，`ValueError: Request is disconnected from the client side` + `TimeoutError`），均在 09-22 / 09-28 / 09-29 的长上下文负载窗口 |
| B7 | 09-18 16:01 那次崩溃**没有 watchdog 告警条目**（alerts.log 从 15:30:25 直接跳到 16:16:04），原因未查清 | 📄 未查（§5 待办 12） |

### 3.3 C 类：GB10 统一内存 / 资源

| # | 坑 | 日志实证 |
|---|---|---|
| P5 | GB10 统一内存陷阱：SGLang 内存记账**看不到 autotuner / CUDA graph 的 25–40GB 瞬时分配**。社区同硬件实测：0.80 三并发崩过、0.85 只剩约 8GB 触发 earlyoom、高于 0.50 或开 autotune 可能把宿主内存打到 0，**只能断电恢复** | ✅ 现役日志侧证：`--disable-flashinfer-autotune` 已设、`mem-fraction 0.70`、docker 硬上限 100g；但宿主 `available 仅 19Gi`、swap 已用 618MiB。⚠️ `dmesg` 在本会话被内核权限拒绝（`读取内核缓冲区失败: 不允许的操作`），**无法核验 earlyoom/OOM killer 历史** |
| P8 | Hub 重新解析会把打过补丁的 config 覆盖回去 | ✅ 已缓解：`HF_HUB_OFFLINE=1` 实测在容器 env 中 |
| 新增 | **stateless 记账盲区仍然存在**：现役 `docker stats` 只报 8.38GiB/100GiB，而宿主 used 已达 **101Gi/121Gi** —— 说明确实有约 60–70GB 内存在容器账本之外（CUDA 预分配 / 宿主其它进程）。**不要因为 `docker stats` 好看就提高 mem-fraction** | ✅ 实测 |
| 新增 | `nvidia-smi` 在 DSH 沙箱内报 `Failed to initialize NVML: Unknown Error` —— **假警报**（crash 复盘 §4 已定案：判断 GPU 要看 `/proc/driver/nvidia/`、`lsmod`、`/dev/nvidia*`） | ✅ 本次复现 |

### 3.4 D 类：脚本 / 运维 / 周边（含本次新发现）

| # | 坑 | 实证 |
|---|---|---|
| P9 | YaRN factor = `round(len/262144)`，**服务长度与变体长度不是一对一** → 变体找不到 | ✅ 现役 `start-dspark-long.sh:72-75` 用 `(ctx+131072)/262144` 同规则解析 |
| P10 | 大海捞针脚本按字符估 token（中文 ≈0.6 token/字），与真实差很多 | ✅ 脚本存在，以服务端 `usage.prompt_tokens` 为准 |
| P11 | 服务端已是 1M，但 **DSH 界面仍按 262144 兜底**（DSH 对自定义 OpenAI 兼容 provider 不探测 `max_model_len`） | ⚠️ **文档声称"已改 `~/.dsh/settings.yaml`"，但该文件不存在**；实测只有 `~/.dsh/settings.yaml.imported`（mtime 09-17 14:58），其中 `qwen3.8-27b.contextWindow: 1010000` 确实写对了 → 声明值存在于 import 产物中，**正式 settings 文件缺失**（见 §6-Ⅳ） |
| 新增 | **`vllm-open-proxy.py` 已不在运行** → 外部直连 `<LAN-IP-1>/<LAN-IP-2>:8888`、`<Tailscale-IP>:8888` **全部失效**（实测 000）；`vllm-proxy.log` mtime 停在 **8月23日 16:49**；进程与 `ss` 监听均无。所有文档仍写"运行中" | ✅ 本次实测 |
| 新增 | `start.sh` 声明的 `LOG_FILE=.sglang.log` **并不存在**（脚本只 `echo` 路径，从未重定向） | ✅ 实测 |
| 新增 | `start.sh 参数区严禁插入行内注释`（`\` 续行 + `#` 会吞掉后续全部参数，曾导致容器以默认参数启动） | 📄 DEPLOYMENT-RECORD §7.4 |
| 新增 | 客户端侧 400：现役日志有 **18 次 `POST /v1/chat/completions 400 Bad Request`**（09-22 ×4、09-28 ×6、09-29 ×2、09-30 ×2 …）与 `create_error_response` 片段 —— 长上下文负载下客户端拿到错误/断流，**与 B6 同源** | ✅ 实测 |
| 新增 | transformers 告警：`Token indices sequence length is longer than the specified maximum sequence length for this model (455553 > 262144)` ×3 —— tokenizer 侧内置 `model_max_length` 仍是 **262144**，模型侧靠 YaRN 兜住 | ✅ 实测 |
| 新增 | 无害噪声：`Ignore import error when loading sglang.srt.multimodal.processors.mimo_v2: No module named 'torchcodec'` ×2 次启动 | ✅ 实测 |

---

## 4. 11 天运行期的稳定性实证结论

**采样窗口**：2026-09-30 15:39 HKT；日志 `2026-09-18 08:16:08Z → 2026-09-30 07:39:30Z`（UTC），共 **239,846 行 / 13 个自然日跨度**。

| 判据 | 结果 |
|---|---|
| 宿主 uptime | **11 天 19 小时 28 分**（`up 11 days, 19:28`） |
| 容器 uptime | **11 天**（`Up 11 days`，`StartedAt=2026-09-18T12:10:43Z`） |
| **本容器启动次数** | 日志里 `The server is fired up and ready to roll!` 共 **2 次**：`09-18 08:22:57Z` 与 **`09-18 12:17:27Z`**。第一次在 `12:05:36Z` 收到 **`SIGTERM received … Draining requests and shutting down`** → `kill_process_tree(parent_pid=1)` → `SystemExit: 0`，是一次**人工/脚本发起的优雅停止**（traceback 是 uvicorn lifespan 的 `CancelledError`，不是崩溃）。 |
| **连续无中断运行** | 自 `2026-09-18 12:17:27Z`（=**09-18 20:17 HKT**）至今 = **11 天 19 小时 22 分，零重启** |
| 崩溃 | **0 次**。`RestartCount=0`、`OOMKilled=false`、`SIGQUIT received` = **0**、`Scheduler hit an exception` = **0**、`dflash_worker_v2` = 0 |
| Traceback 总数 | **11 处**，逐条归类：① 1 处人工 SIGTERM 退出链（L1665–L1697）② 3 处 `health_generate` 探活超时/断连（L104347/104358/104369、L150323/…、L182464/…）③ 其余为 `create_error_response` 上下文里的片段 —— **无一是引擎级崩溃** |
| OOM | **无容器 OOM**（`OOMKilled=false`）。⚠️ **宿主内核 OOM 无法核验**：`dmesg -T | grep -iE "oom|killed process|earlyoom"` 返回空，但同时 `dmesg` 报 `读取内核缓冲区失败: 不允许的操作`（权限不足）→ **本项证据缺失，不可当作"无 OOM"**。旁证：`free -h` available 19Gi、swap 仅 618MiB |
| switchyard | 11 天 journalctl **无 error/panic**；全部命中的 `error` 字样都是日志结构化字段 `error=""`。同期照常处理请求（含 `status=200`、`handling_duration_ms` 1.7s–115s 的长响应） |
| 错误响应 | 18 × 400 Bad Request（客户端/负载侧）、4 × 503 Service Unavailable（均在启动期或长 prefill 挤占期：09-18 08:22、09-18 12:41、09-19 02:17、09-21 03:09） |
| **watchdog 判定（关键）** | 11 天共 **305 次「端口在监听但探活不应答」**、**187 次「服务恢复应答」**、**18 次达 `[WARN]` 疑似假死阈值（连续 4 次失败）**，**自愈动作 0 次**（每条都写明"未做任何重启"）。分布：09-19 44、09-20 19、**09-21 155**、09-22 11、09-23 23、09-24 38、09-25 9、09-28 46、**09-29 123**、09-30 42。`incidents/` 里只有 1 个文件：`20260915-124255-unhealthy.log`（09-15 那次误判抓的现场）。 |
| 结论 | **并发上限收到 4 之后（09-18 20:17 HKT 起）11.8 天：零崩溃、零 OOM、零重启、零 `Scheduler hit an exception`** → crash-0918 §10 待办"观察 3~5 天确认 bs≤4 后不再出现异常"**已超额满足**。但**长上下文"卡"的问题没有消失**：305 次探活不应答 + 18 次 400 + 3 次探活断连 traceback + 峰值 `pending-token` 120 万 / KV 用到 0.88，说明"繁忙到 HTTP 事件循环不响应"仍是常态。 |
| 负向发现 | 守护的保守策略表现良好（0 误杀），但**从未在真实故障上验证过自愈路径**（唯一一次真实演练来自 09-15 的旧容器）；且 `start.sh`/`start-dspark-long.sh` 的重建路径**至今仍会造出"已知必崩"的并发 8 配置**（见 §6-Ⅱ） |

**接受率与吞吐实测基线（全期 55,194 个 decode 样本，来自 `accept len` 行）**

| 指标 | min | p50 | p90 | max |
|---|---|---|---|---|
| accept len | 1.00 | **3.38** | 4.78 | 8.00 |
| accept rate | 0.00 | **0.34** | 0.54 | 1.00 |

最近 8 条 decode 样本（09-30 07:38–07:39Z）：`accept len 2.17–3.61` / `rate 0.17–0.37`，`gen throughput 8.2–44.1 tok/s`（随 batch/上下文长度波动）。
→ 这与文档给的"DFlash2 1M：accept len 2.27–2.80 / rate 0.18–0.26"量级一致；**全期 p50 3.38 显著高于文档早期基线区间**，说明 `LONG-CONTEXT-*` 与 notes 里的"接受率下降"说法需要用配对 A/B 重新量化（文档自己也承认样本非严格配对）。

---

## 5. 文档里明确列出但**还没做**的可改进项

### A. 长上下文 / 投机解码（`notes/qwen38-long-context-1m.md` §7、`LONG-CONTEXT-RESEARCH.md` §10.4 / §10.5 / §12.4、`LONG-CONTEXT-CUTOVER.md` §5）

1. **配对接受率 A/B（最高优先）**：基线与长上下文档用**同一探针**各跑一次，量化 YaRN 对接受率的影响。文档明确说"样本非严格配对"、"严格配对需再切一次基线（约 20 分钟）"。**本次审计用全期 55,194 样本给出 p50 3.38 作为参考，但仍非配对设计。**
2. **DSpark 1M 实测**：变体 `qwen38-nvfp4-yarn1024` + `qwen38-dspark-maxpos-yarn1024` **已就绪但从未启动过**；文档建议"压并发后实测"。
3. **KV dtype 实验**：本镜像 CLI 已支持 `nvfp4` / `fp4_mx_block16` / `fp4_e2m1`，**未实测**。若可用，KV 池可再翻倍（1M 并发从 1 条 → 2 条）。脚本已留 `KV_DTYPE=` 入口。
4. **视觉输入 mrope 验证**：官方 dev 镜像**不含**社区的 mrope 修复（上游 #34446）；图片/视频输入的位置编码可能有偏差。target config 实测确实带 `vision_config` + `mrope_section [11,11,10]`，纯文本无影响，**多模态组合完全未验证**。
5. `--triton-attention-reduce-in-fp32`：社区在同类硬件实测**接受率 +11%**；但它是 triton 后端开关，现役是 flashinfer → **需换后端才有意义，未做**。
6. `--page-size 1` A/B（Keys 报告 page-128 会破坏 triton verify 路径；flashinfer 下未知）。
7. `--sleep-on-idle`（空闲释放池子，省 GB10 统一内存，代价首请求唤醒延迟）、HiCache（`--enable-hierarchical-cache`，提高多轮命中率但不提高单请求上限）、`--enable-unified-memory`（KV 与 GDN 状态池动态共享，**本机未验证**）。
8. **三模式 A/B 顺序从未执行**（`extrap` 零改动基线 → `draftmaxpos` 主推 → `draftyarn` 对照）；实际路线是直接跳到 DFlash2 1M，**extrap 基线从未取过**。

### B. 并发 / 内存 / 调度（`qwen38-dflash2-crash-20260918.md` §7 / §9.5 / §10，`qwen38-dflash2-crash-20260915.md` §10）

9. **备选 2：`--cuda-graph-max-bs-decode 4 → 8`**（保留 8 并发，缓冲区随之 8 行）。未采纳原因是"bs 5–8 路径本机从未验证"；**现已有 11.8 天 bs≤4 的稳定证据**，这是恢复并发的推荐路径（需维护窗口 + 压测）。
10. **长上下文客户端护栏**：`--context-length` 下调 / `--max-queued-requests` 限流 / `--enable-priority-scheduling` —— 文档明确写"**本次未动，留待与用户确认后单独做**"。**本次审计认为这是当前最大的未决风险**（§4 的 305 次探活抖动 = 这条待办的现场代价）。
11. **把 `recreate-concurrency4-20260918.sh` 的两值联动/断言合进 `start-dspark-long.sh`**：从源头消除"并发上限"与"`--cuda-graph-max-bs-decode`"各写各的错配。**未做**（见 §6-Ⅱ）。
12. **向 sgl-project/sglang 报 issue**（`dflash_worker_v2.py stage_sampling_params`：缓冲区按 decode graph max bs 分配但未对 `bs > max_bs` 做保护/扩容），附 traceback + 镜像 commit `5f55db35`。09-15 复盘就列了，**至今仍未报**。
13. **查 09-18 16:01 那次崩溃为何没有 watchdog 告警条目**（alerts.log 15:30:25 → 16:16:04 断档）。**未查**。
14. **watchdog 增强**：检测到 `RestartCount` 增长 / `StartedAt` 变化时**自动 dump 崩溃前日志到 `incidents/`**（本次第 3 次崩溃 traceback 随 `docker rm -f` 丢失）。**未做**。
15. **watchdog"崩溃自动回退到备用形态"开关**（当前只重启同形态，避免无人值守时擅自改规格）。**未做**。

### C. 服务 / 周边

16. `MIGRATION-NOTES.md` §六待办：**灰度期结束后删除 `ornith-local` 旧路由** —— `routes.toml` 里 `[routes.ornith_local]` **仍在**，且**已过期 3 周**（注释自己写"待 1-2 周后删除"）。
17. `MIGRATION-NOTES.md` §8.5 建议 2：**默认关闭思考**（`--default-chat-template-kwargs '{"enable_thinking": false}'` 或客户端传参），理由"每个请求先烧几百 reasoning token"是用户体感慢的主因。注：该建议是切换 DSpark 当天（09-09）提的，**未做**。
18. `DEPLOYMENT-RECORD.md` §6"彻底方案"：重建时把 SGLang 绑 `0.0.0.0` 从而去掉 TCP 代理。**未做** —— 而代理现已挂（§3.4 新增坑），这条从"优化"升级为"待修故障"。
19. P11 的 DSH `contextWindow`：正式 `~/.dsh/settings.yaml` **不存在**（见 §6-Ⅳ）。

---

## 6. 文档互相矛盾 / 已过期之处

### Ⅰ. **文档树本身分裂成两份（最大的结构性问题）**

`/home/<user>/projects/qwen38-sglang/` 与 `/home/<user>/projects/dgx-spark-notes/qwen38-sglang/` 是**同一套文档的两份副本，且 9 个文件全部不同**：

| 文件 | projects 版 | notes 版 | 差异性质 |
|---|---|---|---|
| `DEPLOYMENT-RECORD.md` | 14,816B（09-09） | **15,599B（09-18 16:30）** | notes 版多出 **"⚠️ 2026-09-18 更新（照抄前必读）"** 警告头：明确写"现役已变 DFlash2 + 1,010,000"、"**并发上限已从 8 收到 `--max-running-requests 4`**"、"`MAX_CONCURRENT_REQUESTS=8` 与 `并发×4` 都是首版值，**不要直接照抄**" |
| `LONG-CONTEXT-CUTOVER.md` | 14,063B | **15,554B** | 同上，差异约 1.5KB |
| `LONG-CONTEXT-RESEARCH.md` | 33,223B | 33,232B | 微小 |
| `MIGRATION-NOTES.md` | 19,107B | 19,199B | 微小（脱敏 `dyl`→`<user>` + API key 替换） |
| `CUTOVER.md` / `start.sh` / `start-dspark-long.sh` / `start-long.sh` / `make-yarn-variant.py` / `recreate-concurrency4*.sh` | 各少 9–161B | 各多 | 内容差异小，但**均为脱敏 + 追加** |

> ⚠️ **本次任务书的必读清单点的是 `projects/qwen38-sglang/` 这一份 —— 而它恰恰是缺"09-18 更新头"的旧版**。任何照抄 `projects/qwen38-sglang/DEPLOYMENT-RECORD.md` 的人都拿不到"并发必须改成 4"这条关键更正。**建议：把 `dgx-spark-notes/qwen38-sglang/` 定为唯一权威副本，`projects/qwen38-sglang/` 降级为软链或加醒目指向。**
> 另：`dgx-spark-notes/` 是 git 仓库但 **`main` 分支无任何提交**（`git log` → "尚无任何提交"），所以**没有任何版本历史可用来判断哪份新**——只能靠 mtime。`projects/qwen38-sglang/` 根本不是 git 仓库。

### Ⅱ. **脚本默认值 = 已知会崩的配置（最危险的一条）**

| 文档/脚本 | 写的 | 实际运行 |
|---|---|---|
| `start-dspark-long.sh:47` | `MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"` | **4** |
| `start-dspark-long.sh:129` (DFLASH) | `MAMBA_SLOTS="${MAMBA_SLOTS:-12}"` → `8×12=96` | **48**（=4×12） |
| `start-dspark-long.sh:192` | `--cuda-graph-max-bs-decode 4`（硬编码） | 4 ✅ |

→ 直接执行 `start-dspark-long.sh` **会重建出 8 并发 + bs=4 的错配配置，即 09-15/17/18 崩了 5 次的配置**。crash-0918 §10 已认领这条（"把等价逻辑合进 `start-dspark-long.sh` …并加断言"），**但未执行**。只有 `recreate-concurrency4-20260918.sh` 是安全的（写死 4/48）。
另：`start.sh`（首版脚本，`MAX_CONCURRENT_REQUESTS=8`、mamba=并发×4=32、context 262144、mem 0.85、DSpark + `lmsysorg/sglang:qwen38-27b`）**全文与现役无关**；`projects/README.md` 与 `MIGRATION-NOTES.md` 里多处 `--mamba-radix-cache-strategy extra_buffer_lazy`、`.start.log` 里"并发=8 (mamba池=32)"同样是首版值。

### Ⅲ. **具体数值矛盾清单**

| 项 | 文档说法（出处） | 实测（2026-09-30） |
|---|---|---|
| `--max-mamba-cache-size` | **96**（`LONG-CONTEXT-CUTOVER.md` §1.3 表格、`LONG-CONTEXT-RESEARCH.md` §1、`notes/qwen38-long-context-1m.md:79`、`dgx-spark-model-service.md` §8.1） | **48** |
| `--max-running-requests` | **8**（`DEPLOYMENT-RECORD.md` 全文、`LONG-CONTEXT-*` 表格、脚本默认、`/root/.cache` 之外的多处） | **4** |
| `--mem-fraction-static` | **0.85**（`DEPLOYMENT-RECORD.md` §4.2/§4.3、`MIGRATION-NOTES.md` §8.6、`CUTOVER.md` 隐含） | **0.70** |
| context-length | 262144（`DEPLOYMENT-RECORD.md`、`CUTOVER.md`、`LONG-CONTEXT-RESEARCH.md` §1）／ 506000（同文件 §13） | **1010000** |
| 镜像 | `lmsysorg/sglang:qwen38-27b`（`DEPLOYMENT-RECORD.md`、`CUTOVER.md`、`start.sh`、`LONG-CONTEXT-RESEARCH.md` §11 build 标注） | **`dev-qwen38-27b-dflash2`** |
| SGLang 版本 | `0.0.0.dev0+qwen38.27b.g561c8f3`（`LONG-CONTEXT-RESEARCH.md` §1、§11 抬头） | **`0.0.0.dev1+g5f55db35e`** |
| 模型 | `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`（`DEPLOYMENT-RECORD.md` §4.1 的 `MODEL_ID`，脚本里从未更新） | 变体 `qwen38-nvfp4-yarn1024`（= `Qwen3.8-27B-NVFP4` packed FP4 head） |
| 草稿 | `RadixArk/Qwen3.8-27B-DSpark` block **7**（`DEPLOYMENT-RECORD.md`、`CUTOVER.md`） | **DFlash2** block **8**（z-lab） |
| KV 池 | 1,031,239（`LONG-CONTEXT-CUTOVER.md` §1.4/§1.5、`dgx-spark-model-service.md` §8.1） | **1,205,656**（首启 1,197,439） |
| TCP 代理 | "运行中 / PID 25344 未改动"（`projects/README.md`、`dgx-spark-notes/README.md:36`、`dgx-spark-model-service.md` §3/§4、`DEPLOYMENT-RECORD.md` §8） | **未运行**（LAN 8888 → 000） |
| `~/.dsh/settings.yaml` | "已改"（P11，两处文档） | **文件不存在**；只有 `settings.yaml.imported`（含 `contextWindow: 1010000`） |
| 长上下文与 DSpark | "YaRN 与 DSPARK 不兼容，**必须切 MTP**"（`CUTOVER.md:54`、`DEPLOYMENT-RECORD.md` §12、`MIGRATION-NOTES.md` §六） | **已被 09-10 的 config 内联方案推翻** —— DSpark 506K 与 DFlash2 1M 都跑起来了 |
| "保持 ornith 现状，不建议替代" | `notes/qwen38-27b-alternative-eval.md`（08-25 结论）+ `dgx-spark-model-service.md` §6 | **已被 09-09 实际切换推翻**（`dgx-spark-notes/README.md:16` 自己注明了"后被 9/9 的实际切换推翻"，但 `dgx-spark-model-service.md` §6 与 `projects/README.md` 未加同一句更正） |
| `LONG-CONTEXT-RESEARCH.md` §13 标题 | "**已上线** DSpark 506K (MODE=draftmaxpos)" | 只是**当天的中间态**；数小时后切到 DFlash2 1M。同一文件 §1/§11 用旧容器口径，§13 用中间态口径，**三个时点混排无时间戳标注** |
| `LONG-CONTEXT-RESEARCH.md` §13.2 | "本镜像**未注册** `DFlash2DraftModel`" | 指 `qwen38-27b` 旧镜像；现役 dev 镜像**已注册** |
| `LONG-CONTEXT-RESEARCH.md` §12.1 | "我们现网 **0.85**、free 只剩 18GB" | 已过期（现 0.70 / 19GB available） |
| push 到 HF 的门槛 | `dgx-spark-model-service.md` §0 状态速览 | **同一段落被完整复制了两遍**（L8–19 与 L25–38 逐字重复）—— 疑似合并冲突 |
| `dgx-spark-model-service.md` §1–§7 | 明写"§1-§7 为 ornith 时期（2026-08）原文，保留备查" | 但 §4 仍以肯定语气写"**模型 ID：`ornith-1.5-35b-a3b-nvfp4`**"、§4.5 讲"vLLM 1M 上下文扩展"、§6 讲 ornith 替代评估 —— **一份"本机模型服务总览"里 7/9 的篇幅在描述已下线的服务** |
| `MIGRATION-NOTES.md` §1.5 | "⚠️ 若直接重启 switchyard 而不先恢复全量配置, cloud-test / switchyard 两个模型会消失" | 已过期（`/home/<user>/routes.toml` 现在是全量配置，实测 :4000 返回 4 个模型） |
| `DEPLOYMENT-RECORD.md` §7.3 / §8 | "可用 GPU 内存 9.01GB"、"KV 池 ~135 万 token"、"vllm-open-proxy.py PID 25344" | 全部为 09-09 22:xx 的瞬时快照，现役池 120 万、PID 不存在 |
| 版本/参数四处散落 | `LONG-CONTEXT-RESEARCH.md`（33KB，含 §10–§13 四层追加）、`LONG-CONTEXT-CUTOVER.md`、`notes/qwen38-long-context-1m.md`、`dgx-spark-model-service.md` §8 | **同一套数字在 4 份文件里各写一遍，改一处漏三处** —— §Ⅲ 表格里 8 条数值矛盾全部源于此 |

### Ⅳ. 其它不一致

- `notes/qwen38-long-context-1m.md` 顶部更新头只提到 **09-15 两次崩溃**，**完全没提 09-17/18 又崩 3 次、并发已改 4**（那份独立的 `qwen38-dflash2-crash-20260918.md` 才有）。
- `projects/notes/` 目录（7 个文件）**没有**两份 DFlash2 crash 复盘；它们只存在于 `dgx-spark-notes/notes/`。若按任务书只读 `projects/notes/`，会**完全错过本机最重要的事故知识**。
- `recreate-concurrency4-20260918.sh` 头部注释说"根因（镜像源码 dflash_worker_v2.py，已确认）"并引用了 `container-inspect-20260917-pre-fix.json`（20KB，存在 ✅）。
- `stop.sh` 只说 `docker stop`，**没提** `unless-stopped` 意味着手动 stop 后不会自动拉起（这条只写在 crash-0915 §6 的正文里）。
- `.concurrency-test/` 有 14 个原始请求体但**无任何结论文档**；结论只散落在 crash-0918 §5。
- 宿主时区 vs 容器 UTC 从未在任何文档里注明 —— 本次审计若不换算，会误判"日志停在 07:39，服务似乎 8 小时前就静默了"。

---

## 附：本次审计未能核验的事项

1. **宿主内核 OOM / earlyoom 历史**：`dmesg` 被拒（`读取内核缓冲区失败: 不允许的操作`），本会话为受限子会话，无法提权。**"11 天无 OOM"这一结论在本报告中仅由 `OOMKilled=false` + `free -h` + swap 618MiB 间接支撑，不等同于内核级证明。**
2. **GPU 侧健康**：`nvidia-smi` 报 NVML `Unknown Error`，属已知沙箱假警报；未改用 `/proc/driver/nvidia/` 旁证。
3. **`/metrics` 历史**：只能取当前值（`num_running_reqs=1`、`num_queue_reqs=0`、`token_usage=0.2618`），Prometheus 未做持久化，无 11 天时间序列。
4. **`~/.local/state/qwen38-watchdog/`**：`incidents/` 只有 09-15 的 1 个文件；alerts.log 只有文本，无结构化历史。
5. **watchdog 的 `--diag` 与自愈路径**未演练（审计要求"只读"，且演练会写 `~/.local/state/`，超出本会话工作区写权限）。
