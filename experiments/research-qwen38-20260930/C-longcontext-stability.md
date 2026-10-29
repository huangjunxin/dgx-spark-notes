# C — Qwen3.8-27B / SGLang 长上下文 prefill 加速 与 GB10 统一内存稳定性

调研日期：**2026-09-30** ｜ 目标机型：NVIDIA DGX Spark（GB10, sm_121a, 128GB 统一内存, CUDA 13）
目标模型：Qwen3.8-27B NVFP4（dense 混合 GDN，64 层 = 48 线性注意力 + 16 全注意力），1,010,000 窗口（YaRN factor 4）
现役配置：DFlash2 投机解码、`--mem-fraction-static 0.70`、FP8 KV、`--disable-flashinfer-autotune`、docker `--memory 100g`

**证据分级（全文统一）**

| 标记 | 含义 |
|---|---|
| **【实测·他机】** | 别人在真实 GB10 / 同型号卡上跑出的数字，有 URL + 日期 |
| **【实测·本人推断】** | 用别人的实测点拟合/推算出来的数字，拟合误差已在文中标注 |
| **【文档】** | 官方文档 / 源码 read 出来的事实（SGLang main 分支，2026-09-30 拉取） |
| **【推断】** | 我从架构或文档逻辑推出的，没有直接测量 |
| **【未验证】** | 明确没有证据，必须自己 A/B |

---

## 0. 三条最重要的结论

1. **冷 prefill 慢的根因是二次项，不是你配错了 chunk。** 用社区在**同一台 GB10、同一模型、同一 mem-fraction 0.70、同样 FP8 KV** 的三点实测（100K/200K/300K）拟合 `t(N) = a·N + b·N²`，得 `a = 5.0e-4 s/token`、`b = 3.78e-9 s/token²`。该模型对 466K 预测 1054 s（17.6 min）、对 652K 预测 1933 s（32.2 min），与你实测的 18 min / 32 min **吻合到 2% 以内**。在 652K 处，二次项占总时间的 **83%**。【实测·本人推断】

2. **能真正砍二次项的只有三件事**：把 KV 每 token 字节数减半（FP8→FP4 KV，理论 1.7x）、把前缀复用做满（radix cache 命中时二次项直接归零）、把 prompt 长度降下来（分级上下文/检索而非全量注入）。**chunked-prefill-size 几乎不影响总 prefill 时间**，它只影响混载下的 ITL 公平性——官方 cookbook 之所以在 DGX Spark 上推荐 2048，理由是"8192 的 chunk 会让 decode 每步卡 ~600ms"，不是总量。

3. **统一内存上的稳定性护栏不可能是 docker cgroup**。有实测：GPU 侧分配 10.7 GB 时容器 `memory.current` 只涨了 377 MB；`systemd-run -p MemoryMax=100G` 包 `docker run` 也无效（Docker 自建 cgroup 逃出 scope）。**唯一的真护栏是 `--mem-fraction-static` + 一个真正看 PSI/`MemAvailable` 的看门狗**，而 SGLang 自己提供的 `--watchdog-timeout`（默认 300 s）会在大 chunk 下把 server 自杀掉——这两个都必须在参数表里显式处理。

---

## 1. 定量基线：prefill 到底慢在哪

### 1.1 经验模型（拟合自他机实测）

数据源：hasso5703/dgx-spark-qwen38 `BENCHMARKS.md`，2026-08-29 章节，机器为参考 GB10，配置 `--mem-fraction-static 0.70`、pool 1,008,429 tokens、FP8 KV、27B 车道、cold prefill 且**深度之间清过 cache**。【实测·他机】

| prompt | 实测 prefill | 实测速率 | 拟合 `a·N+b·N²` |
|---|---|---|---|
| 100K | 85.6 s | 1,170 tok/s | 87.8 s |
| 200K | 249 s | 803 tok/s | 251 s |
| 300K | 488 s | 615 tok/s | 490 s |
| 466K（你的） | ~1,080 s | ~431 tok/s | **1,054 s** |
| 652K（你的） | ~1,920 s | ~340 tok/s | **1,933 s** |

**a 项（线性）= 5.0e-4 s/token ≈ 2,000 tok/s 渐近上限**：这是权重读取 + 48 层 GDN 线性注意力 + FFN 的合计成本，本质是**算力受限**（≈108 TFLOPS 等效）。
**b 项（二次）= 3.78e-9 s/token²**：只可能来自 16 层全注意力（48 层 GDN 是 O(N) 的 chunked 扫描，不产生二次项）。

### 1.2 带宽算账（用于判断"该不该量化 KV"）

- GB10 官方内存带宽 **273 GB/s**（LPDDR5x 统一池），Linux `MemTotal` 实测 **121.7 GiB**，CUDA 视图 **119.7 GiB**，做容量规划建议按 ~115 GiB 算。[dev.to/conatusai, 2026-09-02](https://dev.to/conatusai/dgx-spark-gb10-memory-sizing-for-llm-serving-the-numbers-42b7)
- **权重流量不是瓶颈**：NVFP4 权重 ~16.5 GB（[SGLang cookbook Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)），chunk=8192 时 652K prompt 要跑 ~80 个 forward，总权重流量 ≈ 1.43 TB ÷ 273 GB/s ≈ **5.2 s**。相对 1,920 s 可忽略。→ **提高/降低 chunk 大小不会因为"少读权重"而改变总时间。**
- **KV 流量才是二次项的候选解释**：KV = **32.8 KB/token**（16 层 × GQA 4 × head_dim 256 × K+V，FP8；BF16 为 65.5 KB/token —— 这两个数字官方 cookbook 与你实测的 32.8 KB/token 完全一致）。FlashAttention 式分块下，KV 被读的次数 ≈ `N / B_q`（B_q 为 query 分块）。取 B_q=128：
  `KV 读量 = (652137/128) × 652137 × 32800 B ≈ 109 TB ÷ 273 GB/s ≈ 398 s`
  即**理想分块下的 KV 带宽地板约 400 s**，而实测二次项是 1,607 s（约 4x 差距）。
- **另一候选解释是 BF16 注意力算力**：16 层注意力在 652K 处的总 FLOPs ≈ 196,608 × N² ≈ 8.4e16 FLOP，除以实测二次项 1,607 s → **52 TFLOPS**。若 GB10 的 BF16 dense 峰值在 120 TFLOPS 量级，这就是 ~42% 利用率，同样"很合理"。【推断】

> **结论：二次项到底是"KV 带宽受限"还是"BF16 注意力算力受限"，现有公开资料无法判定。这决定了 FP4 KV 值不值得上（前者能拿 1.7x，后者几乎白干）。所以下面第 2.3 节给了一个 30 分钟内能跑完的判别实验，请先做它，再决定参数。**

---

## 2. 长上下文 prefill 加速：可行做法与当前最佳实践

### 2.1 chunked prefill / `--enable-mixed-chunk`：真实作用被普遍误解

- 【文档】`--chunked-prefill-size`：一个 prefill chunk 的最大 token 数，`-1` 关闭 chunked prefill。默认值按显存容量自动选：**GB10 的 CUDA 视图 119.7 GiB 落在 90–160 GB 档 → 默认 8192**（源码 `arg_groups/memory_hook.py::handle_gpu_memory_settings`，2026-09-30）。**同一档还顺带把 decode CUDA graph 的 `max_bs` 默认成 256** —— 见 2.5，这是你内存问题的直接来源之一。
- 【文档】`--enable-mixed-chunk`：默认 `False`，"在 chunked prefill 时允许 prefill 与 decode 混在一个 batch 里"。开它才能让 decode 不被长 prefill 完全饿死。
- 【实测·他机】**chunk=512 反而更慢**：`--chunked-prefill-size 512` 测出 cold prefill 比基准慢 30%（120K：96 s vs 74 s），同时内存峰值地板更低但也更不稳（150K 时地板 6.7 GiB + 13 次驱动分配拒绝）。[hasso5703 BENCHMARKS, 2026-08-29]
- 【实测·他机】**chunk=4096 让 c8 聚合吞吐掉 ~16%**（"boot lottery"控制实验，2026-08-20）。[同上]
- 【文档·官方】cookbook 的 **DGX Spark NVFP4 单元（v0.5.19 上全部实测通过）实际发射的是 `--chunked-prefill-size 2048`**，原文理由："hybrid GDN 模型上 decode 步会被每个 prefill chunk 顶住，8192-token chunk 一次顶 ~600ms；2048 在混合负载下 ITL 更平顺，且单波 TTFT 也更好"。[docs.sglang.io cookbook Qwen3.8-27B, 2026-09-30 拉取]
- 【实测·他机】第三个独立实现（MiaAI-Lab）反过来保留了 **8192**，理由是"chunked prefill 影响的是 prefill/TTFT，不是 decode tok/s"。[MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) `start.sh`

**给你的判断**：你目前是单条长请求为主、几乎不混 decode，那 chunk 大小对你的**总 prefill 时间基本无影响**；但如果你在 652K 冷 prefill 期间还想让短请求能用，必须靠 `2048 + --enable-mixed-chunk`。**不要指望调 chunk 把 32 min 变成 10 min。**

### 2.2 prefix / radix cache 复用：这是最大且最便宜的一刀

- 【实测·他机】官方 cookbook 与两个社区实现都保留 RadixAttention 默认开启。同一 GB10 上：**27K token 前缀复用 2.5 s，冷跑 12.0 s（~5x）**；agent loop 里"每 1000 个新增 prompt token 的 TTFT 增量被压在 ±35 ms（8K 前缀）/ ±85 ms（24K 前缀）"，而**不复用 cache 时该车道冷 prefill 是 ~580 ms / 1k token**。[hasso5703 BENCHMARKS, 2026-08-29 / agent loop 章节]
- 【实测·他机】你已经在享受这一点（652,137 token 召回正确 + "冷 prefill"字样说明第二次是命中的）。
- 【实测·他机】**坑**：带 prefix 共享的 prefill 基准会假快。社区测到 RadixAttention 报 "20,262 tokens in 0.202 s ≈ 100,000 tok/s"，而同一台机器真实冷速率 ~2,100 tok/s —— **50x 高估**。做 prefill 基准时，各深度的 prompt 必须互不为前缀，且每次之间 flush cache。[hasso5703 BENCHMARKS, issue #6]
- 【实测·他机】`--enable-deterministic-inference` 在 GB10 上**必然连带禁用 radix cache**（radix-preserving 的确定性后端在 sm121 上跑不起来：fa4 assert、triton 在 CUDA graph capture 阶段失败），代价约 11% 吞吐。要确定性就得放弃最大的 prefill 加速，二选一。[同上]

### 2.3 KV cache dtype：FP8 vs FP4，以及各 backend 的支持矩阵

- 【文档】`--kv-cache-dtype` 取值：`auto | fp8_e5m2 | fp8_e4m3 | bf16 | bfloat16 | nvfp4 | fp4_mx_block16`。`nvfp4` / `fp4_mx_block16` 需要 CUDA 12.8+ 与 PyTorch 2.8.0+。[SGLang server_arguments, main, 2026-09-30]
- 【文档】FP4 KV 官方标注 **"currently experimental"**。`nvfp4` = NVIDIA FP4（block 16 + E4M3 block scale + per-tensor global scale）；`fp4_mx_block16` = block-16 E2M1 兼容配方（**不是** OCP MXFP4 的 block-32）。[quantized_kv_cache.mdx]
- 【文档】**MHA 后端 × 量化 KV 支持矩阵**（attn.mdx，2026-09-30）关键行：

  | Backend | Page Size>1 | **FP8 KV** | **FP4 KV** | Spec topk=1 | Spec topk>1 |
  |---|---|---|---|---|---|
  | FlashInfer | ✅ | ✅ | **❌** | ✅ | ✅ |
  | FA3 | ✅ | ✅ | ❌ | ✅ | ✅ |
  | FA4 | 128 | ❌ | ✅ | ✅ | ✅ |
  | **Triton** | ❌ | ✅ | **✅** | ✅ | ✅ |
  | TRTLLM MHA | 16/32/64 | ✅ | ✅ | ✅ | ❌ |

  → **在 sm_121 上要 FP4 KV，`--attention-backend triton` 是唯一同时满足 FP4 KV + 投机解码的后端**（FA4 是 SM100/SM103 路线，TRTLLM MHA 在 SM120 只允许当 decode backend 且不支持 topk>1）。
- 【文档】平台约束：**SM120 上 `trtllm_mha` 只能做 `--decode-attention-backend`；prefill/full attention 只能用 `triton` 或 `flashinfer`**。SM121 上 trtllm-gen（`TllmGenFmhaRunner`）会直接 `Unsupported architecture`。[attn.mdx + sglang#40286, 2026-09-19]
- 【文档】`nvfp4` KV 在 **SM100** 才能走 native NVFP4 prefill；`auto` 在 **SM120** 上解析为 **FP8 E4M3 dequant 的 prefill + native NVFP4 decode**（decode 走 SM120 XQA）。native NVFP4 prefill 要求 SM100 + page size 能被 4 整除 + head_dim 能被 64 整除；你的 head_dim 256 满足，但 SM121 不满足"SM100"这一条。[quantized_kv_cache.mdx]
- 【实测·他机】**FP4 KV 有先例，但不是这个模型**：drowzeys 在两台直连 CX7 的 GB10 上跑 thinkingmachines/Inkling-Small-NVFP4（276B MoE / 12B active + DSpark），用 `--kv-cache-dtype fp4_mx_block16 --attention-backend triton --page-size 1 --disable-prefill-cuda-graph --disable-piecewise-cuda-graph`；结论是 **KV 池 354K → 1,104,683 tokens（3.12x）**，且"byte-exact vs bf16 KV on the reference probe；needle 在 21K/64K/113K 深度全部命中"。同时明确警告：**`nvfp4` 那个 recipe 是 flashinfer/trtllm 的打包格式，triton 读不了；`fp8_e4m3` 在这条 lane 上"produces garbage"**。[drowzeys/keys-1M-CTX…, README, 修正日期 2026-08-01](https://github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-SGlang-SM121-optimized-on-Two-DGX-Sparks)
- 【实测·他机】**FP8 KV 的精度争议是真实存在的**：vLLM 侧的 blazux 实测 `KV_DTYPE=fp8_e4m3` "measurable quality cost"，`-10% decode / -30% prefill`，因此**生产保留 bf16 KV、fp8 只做 opt-in**；NVIDIA 论坛 382446（2026-09-05）上，用户实测 MiaAI 的 fp8-KV 1M 配方后回帖说"significantly more degraded"，质量排序 blazux NVFP4 > blazux hybrid > MiaAI。而 SGLang cookbook 与 hasso5703 的 27B 车道都推荐 fp8_e4m3（因为 RadixArk 检查点自带校准过的 KV scale）。[hasso5703 BENCHMARKS "The field agrees on the KV question", 2026-09-09 前后]

**对你（1M + 652K 精确召回）的具体建议：**
- FP4 KV 会把 KV 从 32.8 → 16.4 KB/token，**理论上把 652K 的二次项从 1,607 s 砍到 ~800 s，总时间 1,933 s → ~1,130 s（1.7x）**。但这是**建立在"KV 带宽受限"假设上的上界**。【推断】
- 代价是 `--attention-backend` 必须从 flashinfer 切到 triton（page-size 1、并关闭 prefill/piecewise cuda graph），这会同时改变 16 层全注意力的 kernel 与 GDN 侧的路径；**先跑判别实验**（见 5.2），并在切换后**重跑 652,137 token 的 needle 召回** —— 现有 FP4 KV 的精度证据来自**另一个模型**，对本模型属于【未验证】。

### 2.4 attention backend 选择（sm_121 上到底能用什么）

- 【文档·官方 cookbook】**SM120/SM121 用 `--attention-backend flashinfer`；`trtllm_mha` 是 SM100-only**。MTP + FlashInfer 需要 FlashInfer build 的 prefill `plan` 接受 `uniform_q_len`（新于 0.6.15.post1）；否则用 `--attention-backend triton` 跑 spec。【cookbook 第 2 节】
- 【文档】GDN 混合模型的全注意力层与线性注意力层是两套后端：`--attention-backend`（16 层 full attn）+ Triton 的 linear-attn/Mamba 后端。**SM120 上两种 SSM dtype 都跑 Triton linear-attn prefill** —— FlashInfer 的 GDN prefill 快路 gate 在 SM100 上。【cookbook 第 2 节】
- 【文档】`--enable-unified-memory` 这个名字很有误导性：它**不是**统一内存硬件的开关，而是"给混合 Mamba/GDN 模型用单一 byte buffer 动态切分 KV/SWA/state 子池"的功能，要求 Triton attention + monolithic decode graph，且**与 PD 分离和投机解码尚不兼容**。【server_arguments.mdx】→ 你不会用它，但它会影响 3.2 中那个新特性。
- 【实测·他机】hasso5703 的"boot lottery"实验（2026-08-20）证明：**FlashInfer 的 kernel autotune 每次启动重新选 kernel，同一个配置在不同 boot 之间 c8 吞吐在 92–111 tok/s 之间摆动，verify-heavy 的单流单元 ±15%**；其磁盘 cache 是"advisory"。`--disable-flashinfer-autotune` 修好了这个问题（单流可复现到小数点，c8 落在 ±1.6%，boot 快 ~2 分钟，代价约 2% 平均吞吐）。**你已经开了这个 flag，是对的。**
- 【推断，但值得 A/B】反过来说，autotune 关闭也**冻结了 prefill 的 tile 选择**。如果当前 4x 偏离 roofline 的部分原因就是保守的 prefill tile（小 `B_q`），那 autotune-off 是有代价的。第二次独立实现对 vLLM 测过 "FlashInfer autotune ON + `--max-num-batched-tokens 16384` 无收益"，但那不是 SGLang 也不是你的模型。**建议单独 A/B 一次 prefill（固定 N=300K），这是 30 分钟的实验。**

### 2.5 CUDA graph：这里是你的"隐形 25–40 GB"主要来源

- 【文档·源码】`arg_groups/memory_hook.py`：显存 90–160 GB 档的默认 decode `max_bs = 256`（tp<4）。`cuda_graph_hook.py::generate_decode_cuda_graph_batch_sizes` 在 spec 开启时生成的捕获列表是
  `[1..8] + [10,12,…,32] + [40..64 step4] + [72..256 step8]` → **在 GB10 上默认会捕获 ~37 个 decode CUDA graph**（推测解码下）。这不是小数目。(2026-09-30 main)
- 【实测·他机】**把 `--cuda-graph-max-bs` 从默认降到 8，在 GB10 上 c8 吞吐 +6.5%（且跨 boot 可复现）** —— 因为 batch 5–8 之前跑在 eager 上，捕获后反而更快，同时还省下大量 graph 内存。修正后：稳定 100–104 tok/s c8，而之前是 92–111 的掷骰子。[hasso5703 BENCHMARKS "The boot lottery", 2026-08-20]
- 【实测·他机】**警告：`--cuda-graph-max-bs` 在某些 build 上不生效**。drowzeys 明确写："`--cuda-graph-bs 1 2 3 4 5 6 7 8 10 12 14 16` 要显式列；`--cuda-graph-max-bs` **does not filter** and the default list OOMs the pool"。→ **在 GB10 上请用显式 `--cuda-graph-bs` 列表，不要只给 max。**
- 【文档】prefill 侧：`--disable-prefill-cuda-graph` 等价于 `--cuda-graph-backend-prefill=disabled`；prefill graph 的默认 `max_bs` = `chunked_prefill_size`。【memory_hook.py】
- 【实测·他机】**你这个模型可能根本不捕获 prefill graph**：MiaAI 记录"this build **auto-disables prefill graphs on this model anyway**（GDN layers ≠ standard GQA）"。这条很关键 —— 它直接影响 3.2 节那个新特性是否可用。【MiaAI-Lab README, `PREFILL_CUDA_GRAPH` 说明】
- 【实测·他机】piecewise CUDA graph：drowzeys 在 sm_121 上必须 `--disable-piecewise-cuda-graph`，因为"**the sm_121 piecewise compiler hard-fails**"。【drowzeys README】

### 2.6 `--max-prefill-tokens` 与 `--max-num-batched-tokens`

- 【文档】`--max-prefill-tokens` 默认 **16384**，语义："单个 prefill batch 的最大 token 数。**真正的界是这个值与模型最大上下文长度的较大者**"。cookbook 的 H200 单元会同时显式给 `--chunked-prefill-size 32768` 和 `--max-prefill-tokens 32768`；**DGX Spark 单元不发这个 flag**。[server_arguments.mdx + cookbook cfg.jsx]
- 实务上：**chunked prefill 打开时，`chunked_prefill_size` 才是真正的 batch 上界**（`max_prefill_buffer_tokens` 的定义就是 `chunked_prefill_size`，只有 PP 动态 chunking 才会涨）。所以这个 flag 在你的形态下基本不是杠杆。【文档·源码】

### 2.7 GB10 带宽决定的理论上限（结论）

- 权重流量：**不是问题**（652K / chunk 8192 全程 ~5.2 s）。
- KV 流量地板：**约 400 s @652K**（B_q=128 的理想分块，FP8 KV，32.8 KB/token，273 GB/s）。你现在是 1,933 s，**理论上有 ~4–5x 空间**，但需要 kernel 把 `B_q` 做大或靠 L2 命中。
- 注意力算力地板：**约 8.4e16 FLOP**；按 BF16 120 TFLOPS 算 ≈ 700 s，按 250 TFLOPS 算 ≈ 335 s。这与你实测 1,607 s 之间同样有 2–5x 差距。
- **两条路都指向同一件事：浪费在 kernel 效率上，而不是"内存不够"。** 所以"降 mem-fraction / 加内存"对 prefill 速度没有任何帮助。

---

## 3. 稳定性加固（统一内存机型）

### 3.1 记账盲区：三个有实测支撑的事实

1. **cgroup 看不到 CUDA 统一内存分配。【实测·他机】** GPU 侧分配 10.7 GB，容器 `memory.current` 只涨 **377 MB**。作者结论："cgroups are doubly broken for this use case: Docker escapes the systemd scope; GPU memory bypasses cgroup accounting entirely on unified memory"。[NVIDIA 论坛 t/353752 第 17 楼, 2026-04-10](https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752)
2. **`systemd-run --scope -p MemoryMax=100G -p MemorySwapMax=0 docker run …` 不管用。【实测·他机】** scope 里 `memory.max=107374182400` 是对的，但 **Docker 给容器另建 cgroup（`memory.max = max`）**，进程跑在 Docker 的 cgroup 里，不在 scope 里。[同上]
3. **`nvidia-smi` 在 GB10 上报 `[N/A]` / `Operation not supported`**（`memory.total/used/free` 全是 N/A；`nvmlDeviceGetMemoryInfo.total` 返回的是系统 RAM ~121GB，**不能当作可分配上限**）。可用的替代：求和 `nvidia-smi --query-compute-apps=used_memory`，或直接读 `/proc/meminfo`。[论坛 t/382827 第 1 节, 2026-09-09](https://forums.developer.nvidia.com/t/six-months-of-running-a-dgx-spark-as-a-daily-ai-workstation-real-incidents-root-causes-and-fixes/382827)；[cookbook 第 2 节]

**直接后果**：你现在的 `--mem-fraction-static` 是唯一真正的预算护栏，`docker --memory 100g` 只在"host 堆"层面兜底（也正因为如此，它能保证容器 OOM 时不会把宿主一起拖死，但**不能阻止 CUDA 侧把整机吃干**）。

### 3.2 SGLang 侧可用/不可用的开关（全部核对过 main 分支文档）

| 开关 | 存在？ | 默认 | 语义与建议 |
|---|---|---|---|
| `--crash-dump-folder` | ✅ 存在 | `None` | "存放崩溃诊断：保留 crash-dump buffer 里的已完成请求 + 在飞请求；**在 NVIDIA CUDA 上会在该目录下配置 device coredump**。已存在的 CUDA coredump 环境变量优先。不指定则不启用请求级 crash dump，也不设 coredump 默认。" → **强烈建议设一个持久化目录**（容器外挂载），否则整机 livelock 时什么都留不下 |
| `--watchdog-timeout` | ✅ 存在 | **300 s（float）** | "若一个 forward batch 超过此时间，**server 会 crash 以防止 hang**"。实现是看 `forward_ct` 计数器是否卡住（`utils/watchdog.py` + `scheduler_components/invariant_checker.py`），**粒度是单个 forward batch**。→ 652K 时若你放大 chunk 或关掉 chunked prefill，单个 forward 可能 > 300 s 而被自杀。**放大 chunk 时必须同步提高该值（如 1800）** |
| `--soft-watchdog-timeout` | ✅ 存在 | 平台相关 | 只打日志不杀进程。适合先观察 |
| `--max-running-requests` | ✅ 存在 | `None` | 【实测·他机】**开了投机解码后若不显式 pin，scheduler 会把它重置成 48** —— 在 1M KV 形态下这是灾难。**必须显式 pin**（MiaAI 的做法）|
| `--max-queued-requests` | ✅ 存在 | `None` | 排队上限；**在 disaggregation 模式下被忽略**。配合网关做准入 |
| `--request-timeout-secs` | ❌ **不存在** | — | 我在 server_arguments 里找不到任何 per-request 超时开关（只有 `--watchdog-timeout`、`--soft-watchdog-timeout`、`--dist-timeout`）。**别指望 engine 侧做请求超时，要在网关上做** |
| `--max-total-tokens` | ✅ 存在 | `None` | 手动 pin KV 池 token 数（"通常用于开发和调试"）。**这是把 1M 池固定在确定值、避免"boot lottery"的唯一手段** |
| `--enable-memory-saver` | ✅ 存在 | `False` | 仅启用 `release/resume_memory_occupation` RPC，不自动省内存 |
| `--enable-metrics` / `--enable-cache-report` | ✅ 均存在 | `False` | 开 metrics 与 radix cache 命中报告；cookbook 与两个社区实现都开了 |
| `--schedule-policy` | ✅ 存在 | `fcfs` | 可选 `lpm / random / fcfs / dfs-weight / lof / priority / routing-key`。**`lpm`（最长前缀匹配优先）配合 radix cache 是长上下文场景的正确默认**（SGLang 历史默认） |
| `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`（env） | ✅ 存在，**默认 `False`** | — | **这是最对症的新特性**："在 CUDA graph capture **之后**再给 KV 池定尺寸"。开启后 heuristic 不再预留 graph/activation 的估算量，而是直接量实测空闲内存 → **正面消灭"25–40 GB 不可见分配"**。但有硬条件：非 MLA、`dcp_size==1`、device=cuda、**decode 与 prefill CUDA graph 都必须开启**、未设 `--enable-unified-memory`、未开 `--enable-memory-saver`、chunked_prefill_size>0 且 `max_prefill_buffer_tokens <= max(prefill graph bs)`。【文档·源码 `arg_groups/overrides.py::post_capture_kv_sizing_planned` + `environ.py`, 2026-09-30】**⚠️ 未在 GB10 验证；而且 MiaAI 报告 prefill graph 在这个 GDN 模型上会被自动关掉 —— 若真如此，则此特性对你的配置可能直接不适用。必须实测。** |

### 3.3 earlyoom / docker / 内核层：社区已经踩过的坑

- **DGX OS 自带 earlyoom，默认阈值很激进。**【实测·他机】cookbook 明确：`0.85 × 128GB` 只给 OS 剩 ~8 GB，"**恰好是 DGX OS earlyoom 的 SIGTERM 阈值**"，于是 48 个 cell 里有 **15 个被杀，exit code -15 且无 traceback**，日志在 `journalctl -u earlyoom`。这就是你"0.85 只剩约 8GB 触发 earlyoom"的机制解释。[cookbook 第 2 节]
- **但 earlyoom 也会误杀。**【实测·他机】用户 helge 报告：sparkrun 装的 earlyoom 把此前能跑几十次的 vLLM 启动拦下了，降 `--gpu-memory-utilization` 到 0.75 都没用；**解决办法是把 swap 阈值放宽**：`sudo sed -i '/^EARLYOOM_ARGS=/ s/-s 80/-s 20/' /etc/default/earlyoom`。[论坛 t/378934, 2026-08-02](https://forums.developer.nvidia.com/t/earlyoom-configuration-triggering-too-early/378934)
- **一份可用的 earlyoom 配置（含思路）**：【实测·他机】
  `EARLYOOM_ARGS="-m 3 -s 10 --avoid 'pipewire|wireplumber|systemd|ssh|journald' --prefer 'vllm|python|triton'"`
  外加 systemd override 给 earlyoom `CAP_KILL`（否则杀不掉别人的进程）。配套建议：**装 Dropbear SSH 备用端口 2222**（~500 KB vs OpenSSH 5 MB+），"更可能在内存濒死时活下来，保住你的恢复通道"。[论坛 t/362769, 2026-03-08](https://forums.developer.nvidia.com/t/mitigating-oom-system-freezes-on-uma-based-single-board-computers/362769)
- **但 earlyoom 对 driver 路径冻结无能为力。**【实测·他机】dev.to 作者的两节点事故：`MemAvailable` 在一个 10 s 采样间隔内从 20.3 GB 掉到 0.8 GB，44 s 后归零；worker 卡在 `nvidia_unlocked_ioctl`（持有 rw-semaphore），`earlyoom` 反复 SIGKILL **无效**，worker RSS 只有 202 MiB。他们的配置是 **earlyoom 7 GiB SIGTERM / 5 GiB SIGKILL + 自研 `gb10-oom-guard` 每 3 s 检查、<4 GiB 即杀**，并明确写："These guards can help while the offender remains killable. They cannot stop or recover the driver-path freeze."[dev.to/conatusai, 2026-09-02]
- **看门狗窗口不要设太长。**【实测·他机】同一来源：旧容器看门狗要求"连续 15 次、每 2 s 采样"（即 30 s 持续），结果只记录到 `sample=1/15` —— **完全来不及**。改成"首样本即杀"（阈值由成功启动的配置标定，且只在权重加载完成后启用）。
- **监控要看 PSI，不要只看 MemAvailable。**【实测·他机】多个来源一致：
  - `MemAvailable` 说 28 GB 时 `MemFree` 只有 4.6 GB 且系统明显挣扎；
  - PSI `mem_full` 打到 **4.21%** 时 `MemAvailable` 毫无预警；反之 `MemAvailable` 掉到 6.1 GB 时 PSI 是 **0.00%**（只是"用了"，不是"受罪"）。
  → **`/proc/pressure/memory` 的 `full > 0` 才是预测冻结的那个指标。**[论坛 t/382827 第 2 节, 2026-09-09]
- **swap 要小或关掉。**【实测·他机】统一内存上 swap 让 host page 换出、driver 继续吃共享池，**延长而不是避免 thrash**；事故节点有 16,383 MiB swap，先看到 7.3 GiB 在用。建议"shrink or disable swap for the serving node"。[dev.to, 2026-09-02]
- **别用 `--no-mmap` 类等价物把权重钉成匿名页。**【实测·他机】一份 DGX Spark LLM 栈的 postmortem 结论：统一内存上 `--no-mmap` 把权重推进**匿名页**，匿名页在压力下不可回收（只能 swap 或 OOM），而 mmap 的权重在 page cache 里可被优雅驱逐；"on unified memory it's harmful"。[gardinermichael/spark-llm-stack POSTMORTEM, 拉取 2026-09-30]
- **systemd 单元必须配 `OOMPolicy=stop`。**【实测·他机】同一 postmortem：`Restart=on-failure` 会把 OOM-killer 的 SIGKILL 当作"失败"并重启 → **重启进同一片内存压力 → 砖循环**（他们 12 小时 15 次非计划重启）。正确组合：
  ```ini
  [Service]
  MemoryHigh=70G          # 软限，内核开始回收
  MemoryMax=80G           # 硬限，只 OOM 这个 cgroup
  OOMPolicy=stop          # OOM = 停机，不重启
  OOMScoreAdjust=200
  Restart=on-failure
  RestartSec=15
  StartLimitBurst=3
  StartLimitIntervalSec=600
  Conflicts=<其他重服务>   # 运行时的互斥 ≠ 开机时的互斥，必须写 Conflicts=
  ```
  诊断命令：`systemctl show <unit> -p MemoryMax,OOMPolicy,Restart,Conflicts`、`journalctl --list-boots | tail -10`（一天涨 1 正常，12 小时 15 次就是事故）。
- **kdump 已预装**（`crashkernel=1G-:512M`，`KDUMP_SKIP_VMCORE=1` 跳过 121GB 的 vmcore 是正确默认），**panic 时仍会把内核 dmesg 存到 `/var/crash/<ts>/dmesg.<ts>`**。但真正的掉电/VRM 故障不会留下任何东西。另：**`systemd-coredump` 默认没装、`ulimit -c` 是 0**，一个真实的 hang 会留下零痕迹 —— 要装，并且注意**对已在运行的进程不追溯**（`prlimit --pid <PID> --core=unlimited` 才能给老进程放开）。[论坛 t/382827 第 4/5/9 节]
- **健康检查不能用 `/health`。**【实测·他机】hasso5703："The engine answers `/health` even when it is wedged, so the cockpit runs a **real generation canary** and reports `ready` / `loading` / `wedged` / `stopped` from that." 他们的 keepalive 代理还在 SSE 事件边界每 10 s 注入一次 ping（"a keepalive inside an event corrupts the JSON, measured"），并在**上游真正静默 3600 s**后才报显式错误（"above the worst legitimate prefill"）。[hasso5703 README / docs/context-1m.md]
- **恢复路径要预先准备好**：DGX Spark 没有 BMC。准备**物理接触或"来电自启"的智能插座**；测试新内存配置前先 `docker update --restart=no <id>`，否则容器会自动重启把机器再次拖死。[dev.to, 2026-09-02]
- **CPU 绑定有效**：GB10 是 big.LITTLE（10×3.9 GHz Cortex-X5 = `5-9,15-19`；10×2.8 GHz A725 = `0-4,10-14`）。把容器 pin 到 X5，"否则 scheduler/tokenizer 进程会有一半时间落在小核上"，**实测 +2–7% decode**。[MiaAI-Lab README]

### 3.4 已在他机复现的崩溃签名（照单核对你的日志）

| 签名 | 根因 | 处置 |
|---|---|---|
| 启动后硬重启，内核日志 `task sglang::schedul … blocked …` / `journald … Under memory pressure`，紧跟 `Capture target verify CUDA graph end` | **DFlash2 + 过高的 mem-fraction**：旧的 head 处理在 draft-graph capture 时把整个 NVFP4 lm_head（~2.5–5 GB）**反量化成 dense** | 上游 **#35496** 修好（`lm_head.quant_method.apply` 原地跑，官方镜像已含）。**你跑 DFlash2 + NVFP4，第一件事就是确认镜像含 #35496**；若还见到这个签名，降到 0.90 以下并保留复现记录。[MiaAI-Lab README, 2026-08-19 根因定位] |
| boot 时 `exit code -15` 且**无 traceback** | **DGX OS earlyoom 杀掉了 scheduler** | `journalctl -u earlyoom` 确认；降 mem-fraction（cookbook 因此 pin 0.80）；或按 3.3 调 earlyoom 阈值 |
| `--mamba-radix-cache-strategy extra_buffer_lazy` 启动报 `AssertionError` | 该 build 的 DFLASH 还不支持 lazy 策略 | 上游 **#34763** 加了 lazy 支持；老镜像上退回 `extra_buffer`。[MiaAI-Lab README] |
| DSpark/DFlash2 + YaRN 1M 启动报 `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'` | **YaRN override 泄漏进 draft 的 config**（draft ModelConfig 继承 `text_config` dict） | 官方/MiaAI 的结论是"YaRN 只对 MTP 可用"。**而你已经在跑 DFlash2 + 1M + 内联 YaRN 且工作正常 —— 说明你的做法（把 YaRN 同时写进 target 与 draft 的 config.json）绕过了这个坑。** hasso5703 正是这么做的："patched into **both** cached `config.json` files by `patch-yarn.py`（target model **AND** DFlash2 draft, or the draft crashes at load）"。**升级镜像/换 checkpoint 时必须重新 patch 两个文件**，否则 draft 会在加载时崩。[MiaAI-Lab README + hasso5703 docs/context-1m.md] |

### 3.5 1M KV 与并发的取舍（量化）

- 【实测·他机】你当前：**0.70 → 1.03M token 池 ≈ 33.8 GB**（@32.8 KB/token）。
- 【实测·他机】另一个 27B 实现在 **0.95 上测到 2.48M token ≈ 81 GB 池**，"两条 1M 请求可同时跑，第三条排队直到 KV 释放"。[MiaAI-Lab README]
- 【实测·他机】hasso5703 在 **0.76**（1M 模式）上：graph capture 之后**还剩 21.30 GiB 空闲**（旧的 0.70 pin 是 19.38 GiB），池约 0.89–0.90M token；**0.80 在三并发下实测崩过**（剩 2 GiB，Triton `CUDA operation not permitted`）；**0.85 之后视为 livelock 区间**。十六次 27B boot 的池在 **832,993 – 922,094 token** 之间浮动（"boot lottery"）。
- 【实测·他机】**cookbook 的 0.80 与社区 0.80 不矛盾**：cookbook 的 GB10 cell 是 ISL 8192 / OSL 1024、**并发 1**、只做 boot-and-serve，48 个配置全过；hasso5703 的 0.80 失败是**三并发 + 机器上还跑着别的**。两个数字对应两种工作点。[hasso5703 docs/gb10-memory.md]
- **落地取舍建议**：
  1. **单机就做"一条长 + N 条短"的准入**，别指望 engine 自己扛。用网关（或直接照抄 hasso5703 的 keepalive proxy 思路：给长请求单独计数、给每次请求命名 rid、客户端断开时**在关 socket 之前**发 abort）。
  2. **分级上下文**：日常跑 262144 原生窗口（cookbook/MiaAI 的默认，mem-fraction 可以上到 0.80–0.90 且 10 并发），只在真的需要时开 1M 窗口。这比"常驻 1M + 压并发"在稳定性和日常体验上都好。
  3. 若要提 mem-fraction，**先做一个"boot 后空闲"的下限检查**：hasso5703 的判据是 **graph capture 之后的 GPU 侧空闲量**（0.76 → 21.30 GiB 安全；0.80 → 2 GiB 崩溃）。请把"capture 后空闲 ≥ 15 GiB"作为验收门槛。**注意：`nvidia-smi` 看不到，得用 CUDA 侧的可用内存查询或 `/proc/meminfo` 的 `MemAvailable`**。
  4. **swap/offload 不解决问题**（见 4.1），但它可以在"请求级"上解决问题：把 KV 落盘（HiCache L3）。

---

## 4. 有没有结构性更优解

### 4.1 KV offload 到主机内存 —— **在 GB10 上是 no-op**（重要，别踩）

- 【文档·源码/官方设计文档】HiCache 分层：**L1 = GPU 内存、L2 = host 内存（CPU DRAM）、L3 = 存储后端**。原文两张关键表述：
  - "**L2 is node-local and instance-private.** … HiCache cannot pool the host memory of several machines into one larger L2. Growing `--hicache-ratio` or `--hicache-size` only makes each instance's own private L2 larger."
  - "**Host memory cannot be pooled across instances or across hosts**."
  [hicache_design.mdx / hicache_best_practices.mdx, main, 2026-09-30]
- 【推断，但结论极强】DGX Spark 的 host DRAM 与 GPU 显存是**同一个 128 GB 物理池**（GB10 无独立 framebuffer，见论坛 353752 的分析）。因此 **HiCache 的 L1→L2 只是把 KV 从共享池的一块挪到另一块，净容量收益为零，还多了一次拷贝。** → **不要用 `--hicache-ratio` / `--hicache-size` 来"扩容"KV。**
- **真正有用的是 L3。**【文档】`--hicache-storage-backend` 支持 `file`（默认落 `/tmp/hicache`，可用 `SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR` 改）、`mooncake`、`hf3fs`、`nixl`、`aibrix`；只有 L3 能跨实例共享。开启 HiCache 通常还要 `--page-size 64`、`--hicache-io-backend kernel`、`--hicache-write-policy write_through`、`--hicache-mem-layout page_first`。
- 【推断】对你的场景这是**性价比最高的结构性改动**：把一条 1M 请求的 33 GB KV 落盘，下次同样的长文档 / 重启后可以直接 prefetch 回来，**省掉 32 分钟冷 prefill**。NVMe 顺序读按 5–7 GB/s 估 → 33 GB ≈ 5–7 s（量级上比 1,933 s 好 300 倍以上）。**代价**：33 GB/请求的盘占用 + 写盘带宽与 GPU 抢 IO；`--page-size 64` 与 triton/FP4 KV 所需的 `--page-size 1` **冲突**（若你要 FP4 KV，得在 HiCache L3 和 FP4 KV 之间二选一，或验证 page size 能否取 64 的倍数）。**【未验证】—— 没有任何公开来源在 GB10 上测过 HiCache L3 的吞吐。**

### 4.2 稀疏 / 滑窗 / 长上下文专用 attention

- 【文档】SGLang 有 **HiSparse**（分层稀疏注意力，`docs/advanced_features/hisparse_guide`）和 **DSA / NSA** 后端。
- 【文档+实测·他机】**对你的模型不适用**：Qwen3.8-27B 是 GDN + 全注意力的混合，没有 DSA/indexer 结构。而且 sm_121 上 DSA 路线本身是死的：**sglang#40286（2026-09-19）** 逐条列举 SM121 上 GLM-5.3-Flash 的 DSA 后端全部不可用 —— `trtllm` 报 `Unsupported architecture`（SM100-only）、`tilelang`+fp8 只支持 ROCm、`tilelang`+bf16 超出 GB10 ~100 KB 的 dynamic smem、`triton` 只支持 ROCm、`flashinfer_sparse_mla` 只 gate 在 GlmMoeDsa 架构。**"在 SM121 / DGX Spark 上没有任何可用的 DSA attention 后端"。**
- **所以：不要指望稀疏注意力救你的 1M prefill。** 唯一现实的"减少 N"的手段是**前缀复用**和**分级上下文**。

### 4.3 Prefill/Decode 分离（PD）

- 【文档】SGLang 的 PD 只在**跨节点**有意义（同一台机器上 prefill 与 decode 抢同一个统一内存池），transfer engine 是 `mooncake` 或 `nixl`。
- 【文档】DGX Spark 的官方多节点 Docker flags（cookbook 硬件目录）：`--ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband` —— 注意注释写明 "Multi-node runs over **ConnectX-7 RDMA** (pinned memory + IB passthrough)"。
- 【实测·他机】drowzeys 的两节点配方把这条路的全部坑写清了：NCCL 必须升到 **2.30**（471 MB 的镜像差异"mandatory NCCL upgrade"）、需要 `--device /dev/infiniband --cap-add IPC_LOCK`（否则 NCCL 报裸 `invalid usage`）、`NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME` / `NCCL_IB_GID_INDEX=3` 等一堆环境变量、head 与 worker 启动**顺序**有要求、boot ~8 分钟。
- 【推断】**对你的单条 1M 请求，PD 不解决问题**：KV 传输量是 33 GB/请求，200 Gb/s（25 GB/s）理论 1.3 s，但 PD 的收益是"prefill 与 decode 不互相干扰"，而不是"让单条 prefill 变快"。**在 2 节点上真正能打的是 Prefill CP（见 4.4），不是 PD。**

### 4.4 2 节点 DGX Spark：可行性与代价

- 【实测·他机】**可行性有硬证据**：drowzeys 在 **2×GB10** 上用 SGLang `--tp-size 2 --nnodes 2` 跑 thinkingmachines/Inkling-Small-NVFP4（276B / 12B active）+ DSpark，**full 1,048,576 context**，KV 池 1,082,627 token（`--mem-fraction-static 0.85`），decode ~33 tok/s，accept ~3.5/8，boot ~8 min。关键提醒：**kernel 的 sm100 变体会被加载到 GB10 上**（"`sgl_kernel` loads its **sm100** variant on GB10"，是他们表格里记录的 quirk）。
- 【实测·他机】H200 对比基准：**"dual-Spark SGLang TP2 64 tok/s"**（decode），作为单 Spark 备选的参照。[hasso5703 BENCHMARKS]
- 【实测·他机】**2 节点也会一起挂**：dev.to 作者的两节点 vLLM TP2 复现中，一台节点的 `MemAvailable` 在 10 s 内从 20.3 GB 崩到 0.8 GB，44 s 后归零，worker 卡死在 `nvidia_unlocked_ioctl`，server **从未到达 `/health`**。他们的护栏是 **每节点都装同样的 host-heap guard（earlyoom 7/5 GiB + 自研 guard <4 GiB）**，并有"25 GiB 立即中止"的临时门槛。
- 【文档】Prefill CP（`--enable-prefill-cp` + `--cp-strategy`）在 dense 模型上是 **zigzag + FlashAttention：各 rank 处理自己的 query 块，然后 all-gather KV** 让所有 rank 看到所需的 K/V。**需要 ≥2 个 GPU**，所以单 Spark 上根本不可用。
- 【推断】**2 节点对你的真正价值排序**：
  1. **容量**：把 1M KV 摊到 2×128 GB，从而在**保持 FP8/BF16 KV 精度**的前提下同时支持 2 条 1M 请求（而不是靠降精度换容量）。这是最确定的价值。
  2. **Prefill 时间**：Prefill CP 理论上能把线性项和二次项各除 2（652K 从 1,933 s → 约 970 s + 通信），但 zigzag+FA 每个 chunk 都要 all-gather KV，通信量随 `chunk × 32.8 KB × 16 层` 增长；**收益与代价都没有任何公开实测**。**【未验证，不建议作为第一优先级】**
  3. **运维成本显著上升**：CX7 直连、NCCL 2.30、IB passthrough、两节点同版本镜像、每节点独立的内存护栏、boot 时间翻倍。**在单机问题没解决之前不要上。**

---

## 5. 参数建议表

### 5.1 立即可做（低风险 / 高收益 / 已有他机实测）

| 参数 | 建议值 | 依据 / 来源 | 他机验证 |
|---|---|---|---|
| `--cuda-graph-bs` | **显式列表，如 `1 2 4 8 16`**（不要只给 `--cuda-graph-max-bs`） | 默认在 90–160 GB 档是 `max_bs=256` → 捕获 ~37 个 decode graph（源码推导）；GB10 实测降到 8 后 c8 **+6.5%** 且跨 boot 稳定；drowzeys 报告 `--cuda-graph-max-bs` **不过滤**默认列表 | ✅ **[实测·他机]** hasso5703 2026-08-20；drowzeys |
| `--chunked-prefill-size` | **2048**（若混载 decode）；纯离线长 prefill 可留 8192 | 官方 cookbook DGX Spark 单元发 2048，理由是 8192 chunk 一次顶 decode ~600 ms；8192 在这条 lane 上也实测可用 | ✅ **[实测·他机]** cookbook v0.5.19 / MiaAI |
| `--enable-mixed-chunk` | **开启**（除非你只在离线跑长 prefill） | 文档：允许 prefill/decode 混 batch，否则长 prefill 会把 decode 完全饿死 | ⚠️ **[文档]** 未在 GB10 单独 A/B |
| `--max-running-requests` | **显式 pin**（如 2–4，匹配你的 KV 池） | 开了投机解码后 scheduler 会把它重置为 **48**；MiaAI 因此必须显式 pin | ✅ **[实测·他机]** MiaAI README |
| `--max-mamba-cache-size` | `= 目标并发 × S`（`S=4` 用 `extra_buffer_lazy`+overlap；`S=3` 再开 `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`） | 官方 cookbook 的 `--mamba-full-memory-ratio` 计算式 `ratio=(S+D)×state_bytes/(L×kv_bytes_per_token)`；验证窗口是**独立 buffer**，别把 D 折进 pin（会 2x 超配） | ✅ **[实测·他机]** cookbook + MiaAI |
| `--mamba-radix-cache-strategy` | **`extra_buffer_lazy`**（DFlash 需镜像含上游 #34763；否则退回 `extra_buffer`） | lazy 把 state 从 5 slot/请求降到 4，无精度代价；hasso5703 用它把并发从"被静默截到 ~6"提到 8 | ✅ **[实测·他机]** MiaAI（lazy 报过 AssertionError）+ hasso5703 |
| `--mamba-ssm-dtype` | **`bfloat16`**（省一半 state 池），但**必须做精度验证** | state slot **153.9 MB @fp32 / 78.4 MB @bf16**；官方 cookbook 明确标注"**Treat `bfloat16` as an accuracy gate and validate it for your workload**"；cookbook DGX Spark 单元本身用的是 fp32 | ⚠️ **[文档]**（速度收益他机实测，精度未在你模型验证） |
| `--watchdog-timeout` | 保持默认 300 时**必须保持 chunked prefill**；如果把 chunk 放大到 ≥32K 或关闭 chunked prefill，**同步提到 1800** | 【文档·源码】实现是"某个 forward batch 的 `forward_ct` 卡住超过该值 → server 自杀"；652K 时末尾单个 chunk 的估算耗时约 40 s（chunk 8192），但 chunk 越大越危险 | ⚠️ **[文档]** |
| `--crash-dump-folder` | **设一个容器外的持久目录** | 【文档】会配置 CUDA device coredump 并在崩溃时保存 in-flight 请求；不设则什么都不留。整机 livelock 是这台机器的已知失效模式，事后取证只能靠它 + kdump 的 `/var/crash/*/dmesg.*` | ⚠️ **[文档]** |
| `--enable-metrics` + `--enable-cache-report` | 开启 | 观测 radix cache 命中率 = 你最大的 prefill 杠杆的直接 KPI | ✅ 两个社区实现都开 |
| `--schedule-policy` | **`lpm`**（最长前缀匹配）——若当前不是默认的话显式写 | 长上下文 + radix cache 场景下，前缀复用直接决定 prefill 时间；`fcfs` 会先到先服务而不优先复用 | ⚠️ **[文档/推断]** |
| `--sleep-on-idle` | 可选开启 | MiaAI 用它释放空闲时的显存/内存压力 | ✅ **[实测·他机]** MiaAI |
| 容器 `--cpuset-cpus` | `5-9,15-19` | GB10 big.LITTLE；不 pin 的话 scheduler/tokenizer 有一半时间落小核，**实测 +2–7% decode** | ✅ **[实测·他机]** MiaAI |
| docker 内存 | 保留 `--memory 100g --memory-swap 100g` | 它能兜住 host 堆层的失控（保证容器先死而不是宿主），但**必须知道它看不到 CUDA 分配**（实测 10.7 GB GPU 只计 377 MB） | ✅ **[实测·他机]** hasso5703 / 论坛 t/353752 |
| 主机 earlyoom | `-m 3 -s 20`（放宽 swap 阈值）＋ `--prefer 'python|sglang|triton'`、`--avoid 'systemd|sshd|dockerd|containerd|dbus-daemon|NetworkManager'`；给 earlyoom 加 `CAP_KILL` | sparkrun 默认 `-s 80` 会误杀正常启动；`-s 20` 是实测可用的修正；avoid 列表保护恢复通道 | ✅ **[实测·他机]** 论坛 t/378934 + t/362769 + sparkrun |
| 主机 systemd | `MemoryHigh/MemoryMax + OOMPolicy=stop + Restart=on-failure + StartLimitBurst=3 + Conflicts=` | 缺 `OOMPolicy=stop` 会导致 OOM→重启→更 OOM 的砖循环（12 小时 15 次重启的真实事故） | ✅ **[实测·他机]** spark-llm-stack POSTMORTEM |
| 健康检查 | **不要用 `/health`**；用一次真实的小生成做 canary，并配合 `/proc/pressure/memory` 的 `full` | engine 在 wedged 状态仍然回 `/health` 200；PSI `full > 0` 比 `MemAvailable` 更早预警（实测 `mem_full=4.21%` 时 `MemAvailable` 还在 28 GB） | ✅ **[实测·他机]** hasso5703 + 论坛 t/382827 |
| swap | 关掉或压到很小 | 统一内存上 swap 延长 thrash 而非避免它（事故节点 16 GiB swap，先看到 7.3 GiB 在用） | ✅ **[实测·他机]** dev.to 2026-09-02 |
| 恢复通道 | Dropbear SSH 备用端口 2222；物理接触或来电自启智能插座 | DGX Spark 无 BMC；Dropbear ~500 KB 更可能在内存濒死时活着 | ✅ **[实测·他机]** 论坛 t/362769 + dev.to |

### 5.2 判别实验（30–60 分钟，做完再决定要不要动 KV dtype）

**目的**：判定 652K 的二次项是 **KV 带宽受限** 还是 **注意力算力/kernel 受限**。这决定 FP4 KV（1.7x）值不值得冒精度风险。

固定 prompt 长度 **N ≈ 300K**（互不为前缀、每次 flush radix cache），跑四组：

| 组 | 变量 | 若时间显著变化 → 结论 |
|---|---|---|
| A | 基准（现在的 fp8 KV + flashinfer + chunk 8192 + `--disable-flashinfer-autotune`） | — |
| B | `--kv-cache-dtype bf16`（KV 字节 ×2） | **B 明显慢（≈1.5–2x 二次项）→ KV 带宽受限** → FP4 KV 值得试<br>**B 基本不变 → 算力/kernel 受限** → 别动 KV dtype，去调 backend/tile |
| C | 去掉 `--disable-flashinfer-autotune`（只在 prefill 上 A/B） | C 明显快 → autotune 关闭牺牲了 prefill tile；考虑接受 boot lottery 或固定 autotune 缓存 |
| D | `--attention-backend triton --page-size 1 --disable-prefill-cuda-graph`（先不换 KV dtype） | D 明显快 → 直接走 triton 路线，然后接 FP4 KV<br>D 明显慢 → triton 这条路对你代价太大，只做前缀复用优化 |

补充判据（低成本、必做）：
- 在 B 组跑完 needle，确认 bf16 KV 的召回与 fp8 一致（排除 fp8 已经损伤召回的可能）。
- 记录每组 boot 后 **graph capture 结束时的可用内存**（`/proc/meminfo` 的 `MemAvailable`），作为 3.5 的验收门槛。

### 5.3 中期（需要验证，收益更大）

| 项 | 建议 | 依据 | 状态 |
|---|---|---|---|
| FP4 KV | `--kv-cache-dtype fp4_mx_block16 --attention-backend triton --page-size 1 --disable-prefill-cuda-graph --disable-piecewise-cuda-graph`（**别用 `nvfp4`，triton 读不了那个打包**） | MHA 支持矩阵里 Triton 是 sm_121 上唯一 FP4 KV + spec 都支持的后端；2×GB10 上有 276B MoE @1M 的 byte-exact 先例；**理论收益 652K: 1933 s → ~1130 s** | ❌ **[未验证]** 本模型 + DFlash2 + YaRN 1M 的组合；**必须重跑 652,137 token needle** |
| `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1` | 试开，验证 KV 池是否真的按 capture 后的空闲内存定尺寸 | 【文档·源码】这是唯一正面解决"25–40 GB 不可见分配"的机制 | ❌ **[未验证]** 而且 MiaAI 说这个 GDN 模型上 prefill graph 会被自动关掉，而该特性要求 prefill graph 开启 → **可能直接不适用** |
| HiCache L3 | `--enable-hierarchical-cache --page-size 64 --hicache-storage-backend file --hicache-io-backend kernel --hicache-write-policy write_through --hicache-mem-layout page_first`，存储目录挂到 NVMe | 【文档】只有 L3 能真正加容量；把 33 GB/1M 的 KV 落盘 → 冷 prefill 32 min 变成秒级 prefetch | ❌ **[未验证]** GB10 上无公开吞吐数据；`--page-size 64` 与 FP4 KV 的 `page-size 1` 冲突 |
| `--max-total-tokens` | 固定 KV 池 token 数（如 1030000），消除 boot lottery | 【文档】手动 pin 池大小 | ⚠️ 会牺牲"按 boot 拿到的最大池"，但换来可预测性 |
| DFlash2 镜像含 #35496 | **核对** | 否则 draft-graph capture 会反量化 2.5–5 GB 的 lm_head 并硬重启整机 | ❌ 必须自查 |
| mem-fraction | 目标 **0.76**（1M 单元实测值）；验收门槛"capture 后空闲 ≥ 15 GiB" | 0.70 → 0.76 把池从 ~0.89M 提到 ~0.90M 且空闲更多（21.30 GiB vs 19.38 GiB）；0.80 三并发实测崩 | ✅ **[实测·他机]** hasso5703, v1.14 之后 |

### 5.4 结构性（成本高，先解决单机）

| 选项 | 结论 |
|---|---|
| KV offload 到 host memory（HiCache L2） | **不要做。** 统一内存上是零和搬运，净容量收益为 0 |
| 稀疏/滑窗/DSA attention | **不适用。** Qwen3.8-27B 不是 DSA 结构；且 sm_121 上 DSA 后端全灭（#40286） |
| PD 分离 | 单机无意义；2 节点上解决的是"prefill/decode 互扰"，不是"单条 prefill 变快" |
| 2 节点 | **可行性有实证**（2×GB10 TP2 @1M 已跑通）。最确定的价值是**在保持 KV 精度下把 1M 并发从 1 提到 2**；Prefill CP 的理论收益（二次项减半）**完全未验证**，且通信/运维成本高。**排在单机优化之后** |
| 降 prompt 长度 | 最被低估的一招。分级上下文（默认 262K、按需 1M）+ 前缀复用，是唯一能在**不牺牲精度、不加硬件**的前提下拿数量级收益的路径 |

---

## 6. 需要自己实测的清单（本报告所有 ❌/⚠️ 项）

1. `--cuda-graph-bs 1 2 4 8 16` 在你这台机器上省下多少 capture 内存、decode 掉多少。
2. `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1` 是否生效（看 KV 池 token 数是否变化；若 prefill graph 被自动关掉则直接无效）。
3. FP4 KV（`fp4_mx_block16` + triton）在 **Qwen3.8-27B + DFlash2 + YaRN 1M** 上的 652K needle 召回。
4. triton backend 是否破坏 DFlash2 / YaRN 的加载（YaRN override 与 draft config 的交互）。
5. `--mamba-ssm-dtype bfloat16` 对 652K 召回的影响（精度 gate）。
6. 去掉 `--disable-flashinfer-autotune` 后 prefill 是否变快（5.2 的 C 组）。
7. HiCache L3（`file` backend）在 GB10 NVMe 上的写/预取吞吐。
8. 镜像是否含上游 #35496（DFlash2 lm_head 原地量化）与 #34763（DFLASH lazy radix）。
9. 2 节点 Prefill CP 的实际加速比与通信开销。

---

## 7. 来源清单（URL + 日期）

**官方文档 / 源码（均为 2026-09-30 拉取的 main 分支）**
- SGLang cookbook, Qwen3.8-27B（含 SM121/DGX Spark 指引、202 cell 验证说明）— https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- SGLang server_arguments（`--kv-cache-dtype`、`--watchdog-timeout`、`--crash-dump-folder`、`--max-running-requests`、`--max-queued-requests`、`--max-prefill-tokens`、`--enable-unified-memory` …）— https://docs.sglang.io/docs/advanced_features/server_arguments
- SGLang Attention Backend（MHA 后端 × FP8/FP4 KV 支持矩阵、SM120 约束）— https://docs.sglang.io/docs/advanced_features/attention_backend
- SGLang Quantized KV Cache（`nvfp4` / `fp4_mx_block16` 语义、SM100 native NVFP4 prefill 限制）— https://docs.sglang.io/docs/advanced_features/quantized_kv_cache
- SGLang HiCache 设计（L1/L2/L3 与共享范围）— https://docs.sglang.io/docs/advanced_features/hicache_design
- SGLang HiCache 最佳实践 — https://docs.sglang.io/docs/advanced_features/hicache_best_practices
- SGLang Prefill Context Parallelism — https://docs.sglang.io/docs/advanced_features/prefill_cp
- SGLang PD Disaggregation — https://docs.sglang.io/docs/advanced_features/pd_disaggregation
- 源码：`python/sglang/srt/arg_groups/memory_hook.py`（chunked_prefill_size / decode max_bs / mem_fraction 推导）、`cuda_graph_hook.py`（graph batch size 列表）、`overrides.py::post_capture_kv_sizing_planned`、`environ.py`（`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING = EnvBool(False)`）、`utils/watchdog.py` + `scheduler_components/invariant_checker.py`（watchdog 语义）

**SGLang GitHub issues（日期为创建日）**
- #34872 [Cookbook] Qwen3.8-27B-FP8 validated on one DGX Spark at mem-fraction-static 0.70 — 2026-08-14 — https://github.com/sgl-project/sglang/issues/34872
- #40286 GLM-5.3-Flash has no usable DSA attention backend on SM121 — 2026-09-19 — https://github.com/sgl-project/sglang/issues/40286
- #29317 [GB10/ARM64] sgl_kernel missing SM121 aarch64 wheels — https://github.com/sgl-project/sglang/issues/29317

**社区实测仓库**
- hasso5703/dgx-spark-qwen38：`README.md`、`BENCHMARKS.md`、`docs/gb10-memory.md`、`docs/context-1m.md`（关键日期：2026-08-20 boot lottery / 2026-08-29 27B 长 prefill 与内存地板 / 2026-09-17–18 官方镜像实测）— https://github.com/hasso5703/dgx-spark-qwen38
- MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark：`README.md`、`start.sh`（2026-08-19 DFlash2 崩溃根因、2026-09-05 官方镜像复现）— https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
- drowzeys/keys-1M-CTX-…-Two-DGX-Sparks（2×GB10 TP2 / NVFP4 KV / 1M context；修正日期 2026-08-01）— https://github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-SGlang-SM121-optimized-on-Two-DGX-Sparks
- gardinermichael/spark-llm-stack：`POSTMORTEM.md`（OOM 砖循环与 systemd 加固；拉取 2026-09-30）— https://github.com/gardinermichael/spark-llm-stack/blob/main/POSTMORTEM.md

**NVIDIA 开发者论坛（发布日期）**
- t/353752 DGX Spark becomes unresponsive ("zombie") instead of throwing CUDA OOM — 2025-12-06 起，关键第 16/17 楼为 2026-03-16 / 2026-04-10 — https://forums.developer.nvidia.com/t/dgx-spark-becomes-unresponsive-zombie-instead-of-throwing-cuda-oom/353752
- t/362769 Mitigating OOM System Freezes on UMA-Based Single-Board Computers — 2026-03-08 — https://forums.developer.nvidia.com/t/mitigating-oom-system-freezes-on-uma-based-single-board-computers/362769
- t/364886 Memory Creep on DGX Spark: Where Your 128 GB Actually Goes — 2026-03-27 — https://forums.developer.nvidia.com/t/memory-creep-on-dgx-spark-where-your-128-gb-actually-goes-and-how-to-stop-it/364886
- t/378934 Earlyoom configuration (triggering too early…) — 2026-08-02 — https://forums.developer.nvidia.com/t/earlyoom-configuration-triggering-too-early/378934
- t/382827 Six months of running a DGX Spark as a daily AI workstation — 2026-09-09 — https://forums.developer.nvidia.com/t/six-months-of-running-a-dgx-spark-as-a-daily-ai-workstation-real-incidents-root-causes-and-fixes/382827
- t/367868 Recurring Xid 31 (MMU FAULT_PDE) on DGX Spark / GB10 / SM121 — SGLang + Llama 4 Scout NVFP4 — https://forums.developer.nvidia.com/t/recurring-xid-31-mmu-fault-pde-on-dgx-spark-gb10-sm121-sglang-llama-4-scout-nvfp4/367868

**其他**
- dev.to/conatusai, "DGX Spark (GB10) memory sizing for LLM serving: the numbers" — 2026-09-02 — https://dev.to/conatusai/dgx-spark-gb10-memory-sizing-for-llm-serving-the-numbers-42b7
- DGX Spark 恢复镜像（无 BMC 时的最后手段）— https://www.nvidia.com/en-us/drivers/dgx-spark-recovery-software/

---

## 8. 结合本机现役配置的针对性补充

（现役启动参数取自同目录 `D-local-audit.md` 的 `docker inspect .Args` 快照，2026-09-30 15:39 HKT。以下只列**与上文建议有交叉**的项。）

**8.1 `--cuda-graph-max-bs-decode 4` —— 你已经把它压得很紧了，第 5.1 节的第一条对你不适用。**
本机已经不是"默认 max_bs=256 捕获 37 个 graph"的情况。真正需要你注意的是另一件事：`D-local-audit.md` 记录了 DFlash2 采样器按 `max(decode cuda_graph.bs)` 分配 `greedy_mask`/`temperatures`、却用本轮 bs 切片，**bs ≥ 5 时必然 `RuntimeError`** → 这才是你把并发收到 4 的原因。所以：**"恢复并发"和"抬 `--cuda-graph-max-bs-decode`"必须是同一个动作、同一张变更单**（`4 → 8` + 8 并发压测一起做），不要分开改。hasso5703 在 GB10 上把 decode graph 扩到 8 测得 c8 **+6.5%**，可作为该动作的正面参考。[实测·他机]

**8.2 `--max-mamba-cache-size 48` 疑似按 `并发 × (S+D)` 设的，而引擎只按 `S` 除 —— 可能白占 ~2.5 GB。**
现役是 `并发 4 × 12 = 48`；而 12 恰好 = `S(4, extra_buffer) + D(8, DFLASH block size)`。官方 cookbook 的 ratio 计算式里 D 确实在分子上，但 MiaAI 在**核对过 `kv_cache_configurator.py` 之后**明确写：

> "the engine divides the pool by S **ALONE**; speculative verify states (D, the draft-token window) live in a **separate buffer**… Earlier revisions pinned concurrency x (S+D) and **over-provisioned the pool 2x**."

并且 "`--max-mamba-cache-size` 是等价显式 pin，**会覆盖 ratio**" —— 也就是说现役的 `--mamba-full-memory-ratio 4.21` 实际上没生效。【实测·他机 + 推断】

- 若该结论对本 build 成立：把 `--max-mamba-cache-size` 从 `4×12=48` 改为 `4×S`（`extra_buffer` + overlap → S=5，即 **20**；若再开 `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` → S=4，即 **16**），可释放 `(48-20) × 78.4 MB ≈ 2.2 GB`（bf16 slot）静态预算，按 32.8 KB/token 折合约 **+70K KV token**。
- **收益不大（约 +7% 池），但它的意义在于：这是本机唯一一处"参数之间互相矛盾、且可以安全纠正"的地方。** 顺带把"并发上限"与 `max-mamba-cache-size` 的联动写进启动脚本的断言里（`D-local-audit.md` §6-Ⅱ 已指出脚本默认值与现役不一致的风险）。
- **标为【需实测】：** 改完必须核对启动日志里的 `max_total_num_tokens` 与 `max_running_requests` 两行，确认池变大且并发没被截断。

**8.3 `--mamba-radix-cache-strategy extra_buffer` 是对的，不要改 lazy。**
DFLASH **不支持** `extra_buffer_lazy`（会 `AssertionError`）；上游 #34763 才加上 lazy 支持，且只在较新镜像里。现役日志 0 次断言 → 保持现状。这与第 5.1 节表格里"lazy 优先"的建议冲突，**对本机以 extra_buffer 为准**。[实测·他机：MiaAI + D-local-audit §3.1]

**8.4 `--disable-prefill-cuda-graph` 已开 → 第 5.3 节的 `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` 对你直接无效。**
该特性的硬条件之一是 prefill CUDA graph 必须开启（源码 `post_capture_kv_sizing_planned`）。你已显式 `--disable-prefill-cuda-graph`，加上 MiaAI 报告这个 GDN 模型上 prefill graph 本来就会被自动关掉 → **双重排除**。第 5.3 节的这一项可以从待办里划掉，除非你愿意为了它重开 prefill graph 并承担未知风险。**【未验证，但两个独立理由都指向"不适用"】**

**8.5 `--watchdog-timeout 300` 与 `--chunked-prefill-size 8192` 的组合是安全的。**
652K 时末尾单个 chunk 的估算 forward 时间约 40 s（按 §1.1 的 `b` 求导），远低于 300 s。**但如果你按 §5.2 去试更大的 chunk（≥32K）或 `-1`，必须同步把 watchdog 提到 1800**，否则 server 会在你自己的实验里自杀，而不是真的 hang。**【推断自 §1.1 拟合 + 文档】**

**8.6 你已经开了 `--disable-flashinfer-autotune`、FP8 KV、bf16 SSM、radix cache —— §5.1 里剩下真正还没做的只有：** `--crash-dump-folder`、主机侧 earlyoom/PSI/生成式健康检查、`OOMPolicy=stop`、swap 收敛、Dropbear/智能插座恢复通道。这些是**零风险、纯加固**的，建议与任何参数实验分开、单独上线。

---

## 附：本报告两处最容易被误读的地方

1. **"prefill 慢是因为内存带宽不够"** —— 错。权重流量可忽略（5.2 s / 1,933 s），KV 流量和注意力算力各有 2–5x 的 kernel 效率空间。**加内存、降 mem-fraction、换更大 chunk 都不会让 prefill 变快。**
2. **"KV offload 到主机内存能扩容"** —— 在 GB10 上错。host DRAM 与 GPU 显存是同一个物理池，HiCache L2 只是搬运。要么走 L3 落盘，要么承认容量上限就是那 128 GB。
