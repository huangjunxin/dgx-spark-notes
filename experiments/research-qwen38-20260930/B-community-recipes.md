# B — 社区 / 第三方部署方案调研（Qwen3.8-27B on DGX Spark GB10）

- 调研日期：**2026-09-30**
- 关注窗口：**2026-09-10 以来**的更新（同时收录仍被引用的更早基线）
- 现役基线：SGLang `lmsysorg/sglang:dev-qwen38-27b-dflash2` + DFlash2 草稿（`z-lab/Qwen3.8-27B-DFlash2`）+ YaRN `1,010,000` 上下文，`--mem-fraction-static 0.70`，已稳跑 11 天
- 调研范围：仓库 / HF 页面 / HF discussions / GitHub issues / NVIDIA 开发者论坛 / 第三方博客
- 方法说明：所有性能数字均来自被引页面原文，**未做任何推算或转述放大**；查不到的项在文末"查不到的"一节明确列出

---

## 0. 一句话结论

**不建议换引擎或换部署形态；但现役配置明显落后于社区当前最优档（大约落后 6 个 release），有 3 项低风险、可回滚的升级值得做**：`mem-fraction` 0.70→0.76、草稿换成校准过的 NVFP4 DFlash2 草稿并把深度 8→16、镜像升到官方 release 并关掉 flashinfer autotune。理由与证据见 §1–§7，判断见 §9。

---

## 1. 现役配置在社区坐标里的位置

把你的参数和社区最强方案（hasso5703/dgx-spark-qwen38）逐项对齐后：

| 参数 | 你的现役 | hasso5703 当前（v1.18.7, 2026-09-25） | 差距含义 |
|---|---|---|---|
| 上下文 | 1,010,000 (YaRN) | 1,010,000 (YaRN factor 4.0) | 一致 |
| YaRN 落点 | 生效中（含草稿） | 同时 patch **target 与 draft** 两份 `config.json` | 你已解决同一个坑 |
| 草稿模型 | `z-lab/Qwen3.8-27B-DFlash2`（BF16） | `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@bd7a934` + `--speculative-draft-model-quantization modelopt_fp4` | **落后一代**（v1.9, 2026-09-11 起） |
| 草稿深度 | 8（DFlash2 默认） | **16** | 落后一代 |
| mem-fraction | 0.70 | **0.76**（0.70 是 v1.14 前的旧 pin） | 落后一代（少约 15% KV 池） |
| 镜像 | `nightly`/dev 标签 | 官方 release `lmsysorg/sglang:v0.5.19`（v1.14 起） | 落后（含 69+ 上游提交） |
| 启动确定性 | 未提及 | `--disable-flashinfer-autotune` | 未采纳（社区实测可消除 ±15% 启动抖动） |
| 运维加固 | 稳跑 11 天 | `Restart=always`、`HF_HUB_OFFLINE=1`、keepalive proxy、token-0 腐蚀哨兵、僵尸请求处理 | 未采纳（可选） |

来源：<https://github.com/hasso5703/dgx-spark-qwen38>（README，2026-09-25 状态）、<https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/BENCHMARKS.md>、<https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/docs/context-1m.md>、<https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/docs/gb10-memory.md>

> 该仓库是个人/社区项目（作者 basbunarhasan / hasso5703），不是厂商官方。但它有三个"可验证性"优势：全部 pin 到 digest/revision、给出失败实验记录、并且其 DFlash2 组合自 2026-08-21 起被 SGLang 官方 cookbook 采纳为 DGX Spark 的官方配方。

---

## 2. 自 2026-09-10 以来的仓库 / 页面更新

### 2.1 `hasso5703/dgx-spark-qwen38`（个人，最活跃）— 有大量更新

release 时间线（GitHub Releases API）：**v1.9.0 → v1.18.7**，其中 **v1.18.7 发布于 2026-09-25**（CHANGELOG 条目标注 2026-09-24），v1.17.0 / v1.16.0 / v1.18.0 / v1.18.1 / v1.18.2 / v1.18.3 都集中在 **2026-09-22 ~ 09-23**。

窗口内与"速度/长上下文"直接相关的变更：

1. **v1.9（2026-09-11）：草稿换代 + 深度翻倍**
   - 草稿换为 `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@bd7a934`（5 层草稿量化到 NVFP4，35 个 linear 带校准激活 scale，selector/conv/norm 保 BF16），启动加 `--speculative-draft-model-quantization modelopt_fp4`。草稿显存 **3.53 GB → 1.37 GB**，省下的 2.2 GB 变成 KV 池（该 boot 357,706 tokens）。
   - `--speculative-num-draft-tokens` 由 8 改为 **16**。外部深度扫描：`D4 33.8 / D6 44.1 / D8 47.6 / D12 55.0 / D16 56.6`（pooled tok/s，单台 GB10）；组合配方整请求 **+25.19%**（57.11 → 71.50 tok/s，27 条冻结 payload，质量冻结 24/27 两臂一致、21/27 逐字节相同）。
   - 本机实测：`bench.sh` greedy 中位数 **65.3**（v1.9）vs **50**（v1.2 BF16 草稿 D8）。
   - 代价：NVFP4 草稿理论上只影响接受率，实测 prose 守住、结构化任务上升；**质量不受影响**（草稿的每个 token 仍由 target 在完整词表上验证）。
   - 提交：<https://github.com/hasso5703/dgx-spark-qwen38/commit/0a3458b726567638766a1359a746a2f30d245643>（v1.9.0）

2. **v1.14（2026-09-17 前后）：镜像切到官方 release**
   - 从自建 overlay 切到官方 `v0.5.19` release 镜像，overlay 删除；迁移是实测的：greedy 中位数 **71.4 vs 69.8**，接受长度 **4.29 vs 4.09**，`concat-check` 两臂干净，needle 在 **300,108 prompt tokens 精确命中**（两臂都中）。
   - README 同时明确：**下一步的台阶需要更新的 `lmsysorg/sglang:qwen38-27b` 镜像**——即现役任何 PNG 之外，社区也在等新镜像。

3. **v1.12.1 ~ v1.18.7：1M 成为默认 + 大量稳定性工程**（详见 §4、§6）

### 2.2 `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`（个人）— **窗口内无更新**

CHANGELOG 最后一条是 **2026-09-09**：把 `start-dflash.sh` 的镜像 pin 换成一个修复了僵尸请求的 nightly（`lmsysorg/sglang@sha256:00205b89...`，`nightly-cu134-20260909-708f51e`，含 sglang #35255）。同日的 GB10 观测很值得存档：

> 一次客户端断连留下一个僵尸请求继续解码 **16.75 分钟**（6,466 行 flood、98k tokens）；当天上午累积 10+ 个僵尸 rid、18 次 router 侧 "client hung up"，8–10 个排队请求下瞬时吞吐塌到 **~1 tok/s**。换镜像后同样的 repro 零 flood、15 秒内离开 running batch，RAM 曲线不变（113/121 GiB）。

来源：<https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/blob/main/CHANGELOG.md>

**注意**：该仓库 README 声称 "DSpark 与 DFlash2 都不能用 YaRN（会崩在 rope validator）"，并建议 `YARN=0 / 262144`。这是**它自己的脚本实现**的结论，**不是** SGLang 的能力边界——见 §5。

### 2.3 `RadixArk/Qwen3.8-27B-NVFP4`（权重方）— **窗口内无更新**

HF discussions 列表最新一条是 **2026-08-22**（"The DFlash2 restriction behind the BF16 lm_head change was removed upstream a day earlier"）；窗口内无新讨论、无新 revision 公告。来源：<https://huggingface.co/api/models/RadixArk/Qwen3.8-27B-NVFP4/discussions?p=0>

### 2.4 `incoai/Qwen3.8-27B-DFlash2` / `z-lab` 草稿 — **窗口内无新版本**

HF discussions 共 10 条，最新 #10 创建于 **2026-09-03**；#9 的最后一条回复是 **2026-09-07**。窗口内无新草稿 revision、无 DFlash3。来源：<https://huggingface.co/api/models/incoai/Qwen3.8-27B-DFlash2/discussions?p=0>

### 2.5 NVIDIA 官方权重：`nvidia/Qwen3.8-27B-NVFP4`（2026-09-08 发布）

社区已做过逐层 diff（hasso5703 README，日期 2026-09-10）：

- 量化图与 RadixArk **完全相同**（401 个量化层、MLP 64 组 NVFP4 group-16 + `lm_head`、48 个 linear-attention + 16 个 self-attention 投影 FP8、`mtp*` 排除），三个分片**字节大小完全一致**，仅权重数值不同（两次校准）。
- **唯一实质性差异**：RadixArk 声明 `kv_cache_quant_algo: FP8`，NVIDIA 声明 `null`。因此 NVIDIA 版在 `--kv-cache-dtype auto` 下会退化成 **bf16 KV、池子减半**，除非手动加 `--kv-cache-dtype fp8_e4m3`。
- NVIDIA 卡上自报（GB300 / vLLM）：GPQA Diamond 88.01 vs BF16 88.92、Terminal-Bench 74.02 vs 75.56、IFBench 78.93 vs 80.07、MMMU-Pro 74.86 vs 75.14；AA-LCR 73.38 vs 72.63、SciCode 48.41 vs 47.93 反向。
- 结论（社区）：**不构成换权重的理由**，最多是平局 + 一个 KV 声明陷阱。

---

## 3. 长上下文（≥500K）证据

### 3.1 SGLang + DFlash2 + YaRN 1M 的真实世界战绩（个人，实测）

来自 hasso5703 `docs/context-1m.md`：

| 项目 | 数字 | 备注 |
|---|---|---|
| 窗口 | 1,010,000 tokens | YaRN static scaling，factor 4.0，`original_max_position_embeddings: 262144` |
| KV 池（16 次 27B boot 区间） | **832,993 ~ 922,094 tokens** | 池子是"开机抽签"，没有一次战役的下限能当下限 |
| 官方镜像 @0.76 四次 boot（09-17/18） | 902,398 / 889,131 / 889,722 / 887,797 | |
| 真实 690K-token 请求 | 已服务成功，**冷 prefill ≈ 40 分钟**，之后走缓存 | |
| 单次 opencode 会话 | **535,361 tokens**（两倍原生窗口，**零 compaction**），生成 ~360K tokens，239 步、274 次工具调用、无重试无救场 | 产出单文件 3D FPS：<https://subway-fps.vercel.app> |
| 27B 1M 单元冷 prefill（2026-09-21 实测） | **331 tok/s 平均**；651,583-token prompt 中 614,400 tokens 按 8,192 chunk 灌入，瞬时速率 320 → 184 tok/s | 换算：100K 状态 ≈ 5.0 min；200K ≈ 10.1 min；500K ≈ 25 min |
| 每 token KV 成本 | fp8 ≈ 32.8 KB → 满 1M 序列 ≈ 33 GB；池子 ≈ 75 GB 时约可容纳 **2 个满 1M 请求** | 与 MiaAI 的 81 GB / 2.48M tokens 口径吻合 |

**质量证据（同页）**：作者明确写"原生窗口以外的质量没有正式评测，按实验性 preset 对待"。

**交叉验证**：hasso5703 在 §"1M context mode" 里引用的官方镜像与 nightly 差异，也解释了为什么 0.70 与 0.76 不是"保守 vs 激进"，而是"旧镜像 vs 新镜像的静态预算不同"：

> 官方 release 对同一 fraction 声明更小的 static budget，所以在它上面 0.70 白扔了 15% 池子（**770,118 vs 906,524 tokens**）同时留下 10 GB GPU 内存没认领；0.76 把这块还回来，同时剩下的余量比旧 pin 更宽（graph capture 后 **21.30 GiB vs 19.38 GiB**）。

### 3.2 vLLM 路线的 1M 战绩（个人，实测）

`drowzeys/keys-...-DFlash2-...-1M-Context-YARN-Single-DGXSpark`（GitHub + HF，2026-09-08 实测，engine `vLLM 0.27.1+aeon.sm121a.dspark`）：

| 项目 | 数字 |
|---|---|
| 硬件 | **一台** DGX Spark（明确区别于双 Spark Flash-Next 方案） |
| 窗口 | **1,048,576**（YaRN factor 4.0 × 262,144） |
| 1M needle | **HIT**，prompt **999,714 tokens**，耗时 **2.89 小时 prefill**（单 Spark） |
| KV 池 | 1,113,770 tokens（1M seat, gmu 0.70） |
| 速度 seat（16k / seqs 64 / gmu 0.75） | code **42.3** tok/s、prose ~19–21、tea c=16 **135.7** agg |
| 约束 | gmu **0.70**（声称"Never above 0.85"）；1M seat 上 **seqs 只能 2**（并发被牺牲） |

**关键对照**：同一台机器上，**SGLang 27B 1M 单元的 prefill 是 331 tok/s，而这条 vLLM 路线折算下来 ≈ 96 tok/s（999,714 / 10,404 s）**。即 SGLang 长 prefill 快约 **3×**。两者都远低于 Flash-Next lane 的 ~2,250 tok/s。

### 3.3 512K 档的第三方替代形状：Flash-Next（176B MoE）单机（个人，实测）

MiaAI-Lab 在论坛公布的 KV/长文 profile（聚合贴，2026-09-05，引用其 GitHub 与 sparkDash）：

| 配置 | KV 池 | Prefill | Decode（prose idle） | Needle 5/50/95% |
|---|---|---|---|---|
| 512K YaRN, `KV_TARGET_GIB=20`, BF16 | 704,558 tokens（~1.34× 512K） | 1,537 tok/s @400K，TTFT 260.3 s | 28.3 ±2.6 tok/s | **3/3 PASS** |
| 512K YaRN, `KV_TARGET_GIB=22`, BF16 | 796,196 tokens（~1.52× 512K） | 1,883 tok/s @32K | — | 12/14 |
| 512K YaRN, `KV_TARGET_GIB=22`, FP8 KV | **1,431,164 tokens**（~2.73× 512K） | 1,495 tok/s @400K，TTFT 267.7 s；1,769 @32K | 27.1 ±1.3 tok/s | 15/20 |
| 当前默认（262K, HOST_RESERVE_GIB=26, KV_TARGET 16, FP8） | 992,584 tokens；boot 10 min 51 s；启动后 MemAvailable 15.7 GiB | — | — | — |

也就是说，**≥500K 上有第二条可用形状**：不是 27B，而是单机 176B Flash-Next @512K YaRN。它的 prefill 比 27B lane 快 4–6×，但（a）换模型、（b）窗口是 512K 而非 1M、（c）模型 license 是 Qwen Research **非商用**、（d）FP8 KV 档 needle 掉到 15/20。
来源：<https://forums.developer.nvidia.com/t/382446>

### 3.4 长前缀不会塌掉投机解码（第三方，TP=2 深度阶梯）

Ciprian Ursu 在 spark-arena 上用 hasso5703 的 flag stack + `--tp-size 2` 跨两台 Spark 跑深度阶梯（`tg128 c1`）：

| 前缀深度 | tok/s |
|---|---|
| fresh | 40.09 ± 3.50 |
| 4K / 8K | 36.38 / 39.71 |
| 16K / 32K / 65K | 33.91 / 34.67 / 35.31 |
| **100K** | **28.77**（−28%） |

结论：**深上下文不会让投机解码崩**（32K 时 −13%，100K 时 −28%，16K–65K 基本平）；真实 56K agentic 会话也指向同一结论——**接受率跟内容走，不跟上下文长度走**。来源：<https://spark-arena.com/benchmark/87a93f88-afee-4e93-89de-7cfec34c8345>（经 hasso5703 BENCHMARKS.md 转述，含作者对该 harness 的 caveat）

### 3.5 DFlash2 接受率随上下文塌陷的**反例**（第三方 issue，已关闭）

`mudler/vllm.cpp` issue **#2784**（GB10，driver 580.173.02，2026-09-03 提出，2026-09-03/09-05 关闭）：

| prompt tokens | 无草稿 | + draft k=16 | 接受率 | accepted/pass |
|---:|---:|---:|---:|---:|
| 324 | 16.86 | 86.77 | 0.81 | 12.96 |
| 2,307 | 16.92 | 79.66 | 0.77 | 12.32 |
| **8,159** | **16.34** | **12.57** | **0.06** | **0.96** |

即在该实现上 8K 档投机变成**比不用草稿还慢 23%**（reps 12.60/12.54/12.57，0.5% 波动，是稳定态不是抖动）。作者自己给出的三条 caveat：prompt 是 `vllm-bench --input-len` 合成而非真实文本、只跑了一次 boot、后续 32,741/65,842 档因 worker 崩溃没测到。对照：同架构的 `pangoleen/qwen3.8-27b-dgx-spark-dflash2` 在同一硬件档上 8k 档报告 **7.5 accepted/pass、73.9 tok/s**，接受率从 324 到 65k 只从 6.8 缓降到 8.1。来源：<https://github.com/mudler/vllm.cpp/issues/2784>
**读法**：这是**特定 harness/实现**的反例，不是 DFlash2 的普遍性质；但它提醒"接受率必须在自己内容 + 自己上下文长度上实测"。

---

## 4. 稳定性 / 崩溃 / OOM / 硬重启案例

### 4.1 SGLang 侧的已知风险（社区实测 + 上游 issue）

| 风险 | 证据 | 日期 | 状态 |
|---|---|---|---|
| **`--mem-fraction-static` 过高 → 整机 livelock（要断电）** | SGLang 的内存记账**看不到 25–40 GB** 的瞬时分配（flashinfer fp8 autotuner + CUDA graph capture 在池外）；>0.80 = livelock 区；社区在 3 并发下实测 0.80 崩（剩 2 GiB，Triton `CUDA operation not permitted`）。cgroup **看不到** CUDA unified 分配，所以 `--memory 100g` 不是真护栏，fraction 才是 | 2026-09 | 持续有效 |
| **cookbook 的 0.85 被 earlyoom 杀** | 0.85×128 GB 只剩 ~8 GB，正好是 DGX OS earlyoom 的 SIGTERM 阈值；48 个 cell 里 **15 个被杀**（exit −15，无 traceback，见 `journalctl -u earlyoom`） | — | 官方 cookbook 已回退到 0.80 |
| **0.80 的口径冲突** | cookbook 的 0.80 是 **ISL 8192 / OSL 1024 / 并发 1 / 只 boot-and-serve** 的单流边界；社区 0.80 崩在多客户端场景。两者不矛盾，但**你若是多客户端，按社区的 0.76 起步** | — | |
| **DFlash2 + 高并发 → 跨请求上下文串扰** | sglang **#36548**："last user message is appended to the wrong context"，GPQA 问题会串进 coding workflow；cookbook 设置复现 | 提出 2026-08-26，最后更新 08-27 | **仍 open**（3 条评论） |
| **DFlash2 + thinking → greedy 与 target-only 分歧** | sglang **#38009**：thinking 开启时在 completion token 30 处确定性分歧（关 thinking 则 64 token 全同）；复现环境是官方 dflash2 dev 镜像 digest `616a3e97…` | 提出 2026-09-04，最后更新 **2026-09-10** | **仍 open** |
| **`top_logprobs` 无上界 → scheduler 直接死** | sglang **#40076**：字段 `Optional[int]` 无 validator，超过词表后 `logprobs.topk` 抛 `selected index k out of range` **在 scheduler 内部**，全体客户端一起掉线；社区在 reference box 上误转发一次，调度器死、systemd 花 9 分钟把 lane 拉回来 | 2026-09-18 提出 | **仍 open**（社区用 proxy 在三条路由上拒绝 >1024） |
| **客户端断连 → 僵尸请求** | 上游 **#35255**（2026-09-04 merge）：`abort_request()` 在 rid 已离开 `rid_to_state` 后提前返回，调度器继续解码到 `max_tokens`，占住 running slot 并刷 `state was deleted in TokenizerManager`。MiaAI 在 GB10 实测：一次断连 = 16.75 分钟僵尸解码、6,466 行 flood、10+ 僵尸 rid、吞吐塌到 ~1 tok/s | 2026-09-09 实测 | nightly 已含；**v0.5.19 不含**（注意：这是你现役镜像的重要选择点） |
| **解码状态丢失 → 连续 token id 0（`!`）** | sglang #36537 / #36558 / #36806 / #36845；社区在 proxy 上做"连续 128 个 `!` 就 abort 并报 `corrupted_output`"的哨兵（默认 `CORRUPTION_RUN=128`）。v1.9 配置实测 corruption markers **0** | 2026-08-22 ~ 09 | 部分修 |
| **多轮不可复现（radix cache 开启时）** | 第三方 harness 在 34 个任务里 6 个出现 greedy 多轮分歧（关投机也 3/34；关 radix 则 0/34）；`--enable-deterministic-inference` 可消除，代价 ~11% 吞吐且**在 sm121 上必然连带关掉 radix cache**（fa4 assert / triton 在 CUDA graph capture 期失败） | 2026-08 | 无解，需取舍 |

来源：<https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/docs/gb10-memory.md>、<https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/BENCHMARKS.md>、<https://api.github.com/repos/sgl-project/sglang/issues/36548>、<https://api.github.com/repos/sgl-project/sglang/issues/38009>

### 4.2 硬重启 / 硬冻结案例——**多数是硬件/热，不是引擎**

必须诚实区分，否则会误判：

- **论坛 380286（2026-08-15，个人）**"vLLM 服务 FP8 时静默 exit 255，2–3.5 分钟后整机硬重启"：作者后来自己查明是 **恒温器（thermostat）硬件缺陷**，已 RMA；NVIDIA 员工（aniculescu，Employee）2026-08-26 回复请其更新软件栈，作者确认已在 RMA 流程。**不要把这条记到 vLLM 头上。** <https://forums.developer.nvidia.com/t/380286>
- **论坛 383624（2026-09-18 提出，最后回复 2026-09-24，个人）**：DGX Spark FE 在持续 llama.cpp 推理下**静默硬冻结**（风扇转、系统不可达、只能断电）。10 秒采样遥测显示冻结前最后一次采样毫无异常（81 °C、60 W、无 throttle 位），kernel journal 在冻结瞬间**什么都没写**（无 Xid / NVRM / OOM / hung task / panic），netconsole 建议用于抓"连 printk 都没出"的场景。**`partnerdiag --field` 结果：PowerStress FAIL，MODS-020000600139"温度上限超限或温度传感器损坏/失准"**，已申请 RMA，保修 10-14 到期。同一帖另有用户（ocmkieboom）报告 llama.cpp 在 kernel 7 升级后开始同样的冻结。**这是硬件/热相关类别。** <https://forums.developer.nvidia.com/t/383624>
- **同类事件是一个可复现的"类别"**（论坛 related topics 列表）：`380238`（2× Spark FE 持续推理静默硬锁，48 小时内 RMA）、`381655`、`377365`（partnerdiag PowerStress 可复现硬关机，acpitz 88→97.8 °C / 5 s）、`379195`、`373251`、`368536`。
- **另一条"熵"提醒**：hasso5703 的 reference box 在**统一内存耗尽**时也是硬冻结（不是 swap），且"SSH 常与模型共享同一块内存"。这解释了为什么社区把 `mem-fraction` 当作生死线而非性能旋钮。

### 4.3 fp8 KV 的质量争议（与你的长上下文直接相关）

- `blazux/qwen3.8-Flash-DGX`（个人，最后运行 2026-09-08）对每个选项跑了 **17 场景 agentic tournament**（工具循环、长文抽取、多步推理，temp 0.2，三次重复，满分 51）：
  - `MODE=nvfp4`, MTP=2 → 45/51，~26 tok/s
  - `MODE=hybrid`（NVFP4 experts + blockwise fp8 side layers）→ **45/51**，~31 tok/s（配 reduced draft vocab 可到 37）
  - `KV_DTYPE=fp8_e4m3` → 自测"**measurable quality cost**"，decode −10%、prefill −30%
  - 该仓库的生产配置**保持 bf16 KV**，fp8 KV 只作为 opt-in。
- 论坛 382446（2026-09-05）上，原帖作者当晚回来说用桌面客户端和 coding agent 试过后，**fp8-KV 版本"significantly more degraded"**，质量排序为 `blazux NVFP4 > blazux hybrid > MiaAI Lab`（尽管 MiaAI 吞吐更好）；另有两位用户就安全敏感代码报告同样现象。
- **但对 27B 而言这条争议不直接适用**：SGLang cookbook 明确写"两个 NVFP4 checkpoint 都声明 `kv_cache_quant_algo: FP8`，SGLang 默认 `--kv-cache-dtype auto` 会尊重它，KV 池自动跑 fp8_e4m3 并带 checkpoint 自己的校准 scale"。也就是说**你用 fp8 KV 是官方推荐路径**；上面的质量损失主要是 Flash-Next 那条 lane 上的观察。

### 4.4 vLLM 侧的具体坑（如果你考虑换过去）

- **stock `vllm/vllm-openai` 在 Blackwell sm_121a 上没有 NVFP4 kernel** —— 社区原话："Every stock-vLLM attempt crashed"；必须用 GB10 专用构建（`ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38:mtp3-20260813` / eugr 的 spark-vllm 系列 / AEON 的 `aeon-vllm-ultimate`），env 需要 `FLASHINFER_CUDA_ARCH_LIST=12.1a`、`FLASHINFER_DISABLE_VERSION_CHECK=1`、`VLLM_ALLOW_LONG_MODEL_LEN=1`。来源：<https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/qwen-38-27b.md>（经论坛 380244 post 17 转述，2026-08-18）
- **unsloth NVFP4 checkpoint 的 tokenizer truncation bug**：`unsloth/Qwen3.8-27B-NVFP4` 的 `tokenizer.json` 里编译进了 `"truncation": {"max_length": 2048}`。**文本会被静默截断到 2048 token**（服务器还在宣称 `max_model_len: 262144`）；带图时则报 `Mismatch in image token count`。修法：只把该字段改成 `null`，或用只读挂载覆盖。来源：论坛 380244 post 1（Helge，2026-08-15）——**若你任何时候试 vLLM 路线，这是第一个该查的东西**。
- **vLLM MTP 在长上下文下 decode 明显退化**：同一帖 post 21（jbourny，2026-08-18）实测 filler prompt：d0 = 18.3 tok/s，d16384 = 6.3，d32768 = **3.7**（code/structured 档持平）。<https://forums.developer.nvidia.com/t/380244>

---

## 5. DFlash2 与 YaRN 1M 共存 — 结论已经清楚（且你已在做）

**根因**：Qwen3.8 原生 262,144，1M 靠 YaRN factor 4。SGLang/vLLM 会把 target 的 rope override **复制进草稿**，而 DFlash2/DSpark 草稿是一个扁平 Qwen3 config：

- 要么因为 `text_config` / `max_position_embeddings` 缺失而**崩**（MiaAI 观察到的 `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'`）；<https://raw.githubusercontent.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/refs/heads/main/README.md>
- 要么草稿的 rope 留在 262k 而 target 在 1M（接受率退化）。

**修复（三处独立得到同一结论）**：把 YaRN **直接写到草稿自己的 `config.json`** 上，而不是只写在 target 的 override 里：

1. **SGLang 路线（hasso5703，生产在用）**：`patch-yarn.py` 同时 patch **target 和 DFlash2 draft** 两份缓存 `config.json`（原件备份为 `config.json.pre-yarn`），加 `--context-length 1010000` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`；并在 unit 里设 `HF_HUB_OFFLINE=1`，防止 Hub 元数据检查重解析 checkpoint 把 patch 悄悄还原。
2. **vLLM 路线（HF discussion incoai #8，2026-08-26，编辑于 08-31）**：GB10 用户 GizzmoShifu 把草稿 config 改成 `"max_position_embeddings": 1048576` + `rope_type: yarn, factor 4.0, original_max_position_embeddings: 262144`，**在真实 VS Code coding 会话里跑到 539K tokens**；spec tokens = 6 时接受率约 **47% / ~1.9 accepted per draft**。他明确写"我不据此声称 1M DFlash2 支持，539K 是我在真实负载里验证的最远处"。
3. **vLLM 配置 fork（drowzeys，HF model card）**：把同一修复做成 `config-only fork`，并给出启动参数（target 保留 nested YaRN hf-overrides，草稿用 overlay 绑定 config）：<https://huggingface.co/drowzeys/keys-Qwen3.8-27B-DSpark-vLLM-YaRN-1M-Context>。**该 card 自己声明："Acceptance at 1M is unmeasured until a 1M target + this draft is benched."**

**接受率 / 深度：社区内部有分歧，且与负载相关**

- incoai **#9**（2026-08-28 提出，最后回复 **2026-09-07**）：提问者 Nuno Donato 说"positions 4-7 经常 ≤0.1"；GizzmoShifu 回"我跑 4，coding 场景 60–75%"；**BuiDoan（2026-09-07）**："positions 4–7 通常 ≤0.1，把 spec tokens 降到 3 会让 dflash 表现得像 MTP。"
- hasso5703 自己的深度扫描结论相反（D16 最优，D8 47.6 → D16 56.6 pooled tok/s）。
- **两条并不必然矛盾**：前者是 vLLM + 单卡 GB10 + 各自工作负载，后者是 SGLang + GB10 + 27 条冻结 payload。合理的读法是：**深度是需要在你自己内容上 A/B 的参数**，不是可以照抄的常数。这也和 §3.4"接受率跟内容走"一致。

---

## 6. 其他部署形态对比

| 形态 | 是否有 ≥500K 的公开证据 | 相对现役的判断 | 关键来源（日期） |
|---|---|---|---|
| **SGLang + DFlash2 + YaRN 1M**（现役） | **有**（690K 真实请求、535K 会话、614K prefill 331 tok/s） | 基线；社区最优档仍在这条路上 | hasso5703 docs/context-1m.md（2026-09-25 状态） |
| **vLLM + YaRN fork + DFlash2/DSpark** | **有**（1M needle HIT @999,714，prefill 2.89 h） | 单流更慢（工具评测中位回合 3.6 s vs 7.8–10.3 s）、prefill 慢约 3×、1M 时并发只能 2；但配置层更简单、有正式 1M needle 证据 | drowzeys GH/HF（2026-09-08）；论坛 380957（2026-08-22 / 09-03） |
| **vLLM + MTP（unsloth 权重）** | 仅 262K 级别的 prefill/TTFT 数据 | 明确更慢：code 19.8 vs 32.6、reasoning 25.4 vs 47.3 tok/s（同一评测）；且触发 tokenizer 截断坑 | 论坛 380244（2026-08-15）、380957 |
| **llama.cpp / GGUF** | **查不到** ≥500K 的成功案例 | 实测约为 SGLang lane 的 **1/3 decode**；prefill 也约慢 3×；MTP n=3 调优后 code 25.6、prose 17.7–18.2 tok/s | hasso5703 `extras/gguf/README.md`（回答 issue #12） |
| **TensorRT-LLM (sm_121 / GB10)** | **查不到**任何 Qwen3.8-27B 的 DGX Spark 配方 | 无法评估 | — |
| **双 DGX Spark（TP=2 / CX-7）** | 有深度阶梯（100K 档 28.77 tok/s），但**没有"单请求 500K+ 更快"的证据** | **batch-1 decode 与单台同档（34–40 tok/s）**；第二台只买并发余量（c5 115、c10 97 agg）。社区结论直白："Don't cluster for latency." | spark-arena 阶梯 + hasso5703 BENCHMARKS.md；论坛 380350（dual Sparks, 2026-08-17） |
| **NVIDIA DGX Spark playbook / 官方配方** | 查不到 Qwen3.8-27B 的官方 playbook recipe | 官方存在的只是**权重**：`nvidia/Qwen3.8-27B-NVFP4`（2026-09-08，recipe 与 RadixArk 相同 + 无 KV 声明）、`nvidia/Qwen3.8-Flash-Next-NVFP4` | hasso5703 README §"The seven targets"（2026-09-10） |
| **SGLang 官方 cookbook DGX Spark cell** | 不含 1M/YaRN 的实测数据 | 官方 cell = NVFP4 + **DSpark**（后来 DFlash2），mem **0.80/0.85**、chunk 2048、GDN float32、radix `extra_buffer`；官方口径 DGX Spark 平均 **38.28 tok/s**（temp 0.6 eval 平均）。DFlash2 自 **2026-08-21** 成为官方配方（block 8，draft 路径 `incoai/…`） | <https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B>（含 hasso5703 README 转述） |
| **Flash-Next 176B 单机** | **有**（512K YaRN profile，needle 3/3 PASS @BF16） | 换模型；prefill 快 4–6×（1,495–2,250 tok/s），但窗口 512K、license 非商用、FP8 KV 档 needle 15/20 | 论坛 382446（2026-09-05）；hasso5703 flash lane（2026-09-18 评测） |

**关于 cookbook 的实测口径差异**（值得知道，避免被单个数字误导）：hasso5703 逐项反驳过 cookbook 的 cell：cookbook 用 mem 0.85 / chunk 2048 / GDN **float32** / radix `extra_buffer`，而社区实测 **float32 比 bf16 慢 3%**、FP8 **慢约 30%**，chunk 8192 更好。所以"照抄官方 cell"并不等于最快，但**官方 cell 是保底可用的**。

**关于"哪个数字才是真快"**：论坛 380957 的 Ama5u 与 hasso5703 都反复指出，跨引擎表必须串行（同一台机器只跑一个），并且要注意 `delta.reasoning`（vLLM ≥0.27 把 thinking token 放在 `reasoning`，SGLang 用 `reasoning_content`）——客户端只认一个字段会把 tok/s 虚高到 97（实际 19.9）。这类方法论坑在社区报告里非常常见。

---

## 7. HF discussions / GitHub issues 的新结论汇总

| 主题 | 结论 | 来源 | 日期 |
|---|---|---|---|
| DFlash2 + YaRN 1M 共存 | **可行**，前提是把 YaRN 写进草稿自己的 config.json；vLLM 侧有人在真实 VS Code 会话验证到 539K | <https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/8> | 2026-08-26（编辑 08-31） |
| DFlash2 深度选择 | positions 4–7 接受率常 ≤0.1；有人降到 3 就"退化成 MTP"，有人 4 就拿到 60–75% | <https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/9> | 2026-08-28 → 2026-09-07 |
| DSpark + YaRN 1M | 同样根因、同样修法（rope overlay 落在草稿上）；**1M 接受率尚未实测** | <https://huggingface.co/drowzeys/keys-Qwen3.8-27B-DSpark-vLLM-YaRN-1M-Context> | 卡片无日期（引用 Doopeworld/RadixArk 权重） |
| BF16 `lm_head` vs 打包 FP4 `lm_head` | **BF16 head 更慢且无质量收益**：8/8 probe 更慢、均值 −14%（math −30%）；tool-eval 91 vs 91（4 个场景翻转，属噪声）。作者结论是拒绝 BF16-head | 论坛 380957 post 4（08-29）与 post 17（**2026-09-03**） | 2026-08-29 / 09-03 |
| Qwen 官方 thinking 采样参数用于工具调用 | **greedy 优于 temp 1.0/top_p 0.95/top_k 20**：质量 91 打平、时延 2× 更差，而且 thinking 版在 **TC-58（假 system message 注入）**场景跟着假消息走并泄露假 API key，greedy 全部干净处理 | 论坛 380957 post 1 | 2026-08-22 |
| DFlash2 并发串扰 | **未修** | sglang #36548 | 2026-08-26，最后更新 08-27，open |
| DFlash2 + thinking 的 greedy 分歧 | **未修**；官方 dflash2 镜像可复现；关 thinking 则完全一致 | sglang #38009 | 2026-09-04，最后更新 **2026-09-10**，open |
| 上游对"分歧"的解释 | 社区做了无损性研究：温度 0 下 DSpark 10/10、DFlash2 10/10 都会在推理早期（2–33%）与纯 AR 输出分歧；但 **GSM8K 200 题精确打平 188/200**、IFEval 在噪声内。作者结论："两个草稿的文本都不是模型的真输出，都是同一数值邻域里同样合法的采样" | hasso5703 BENCHMARKS.md §"The losslessness study" | 2026-08-20 |
| DFlash2 接受率在长上下文塌陷 | 特定 vllm.cpp harness 上 8,159 token 时接受率 0.06、比不用草稿慢 23%；同架构另一个 DGX Spark 项目在同一档报告 7.5 accepted/pass、73.9 tok/s | mudler/vllm.cpp #2784 | 2026-09-03，closed |
| RadixArk 权重侧 | **窗口内无任何新讨论/新 revision** | HF discussions API | 最新 2026-08-22 |

---

## 8. 来源链接表（URL | 类型 | 日期 | 关键结论）

| URL | 类型 | 日期 | 关键结论 |
|---|---|---|---|
| <https://github.com/hasso5703/dgx-spark-qwen38> | 个人/社区（最活跃） | 状态 2026-09-25 | 27B lane = SGLang + NVFP4 + DFlash2(NVFP4 校准草稿 D16)；1M YaRN 默认；mem 0.76；71.4 tok/s greedy 中位数 |
| <https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/BENCHMARKS.md> | 个人（附方法论/失败实验） | 2026-08-14 ~ 09-21 | 深度扫描、boot lottery、vLLM/SGLang 同档、losslessness 研究、prefill 331 tok/s |
| <https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/docs/context-1m.md> | 个人 | 2026-09（v1.12.1+） | 1M 预设、池 832,993–922,094、690K 请求 40 min prefill、535,361 token 会话 |
| <https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/docs/gb10-memory.md> | 个人 | 2026-09 | mem-fraction 生死线、0.70→0.76 的 15% 池差、cookbook 0.80 的口径 |
| <https://raw.githubusercontent.com/hasso5703/dgx-spark-qwen38/main/extras/gguf/README.md> | 个人 | 2026-09 | llama.cpp ≈ 1/3 decode、prefill 慢 ~3× |
| <https://api.github.com/repos/hasso5703/dgx-spark-qwen38/releases> | 个人 | v1.9.0(09-11) → v1.18.7(09-25) | 窗口内密集更新；v1.9 草稿换代 |
| <https://github.com/hasso5703/dgx-spark-qwen38/commit/0a3458b...> | 个人 | 2026-09-11 | v1.9.0：NVFP4 校准草稿 + D16 |
| <https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/blob/main/CHANGELOG.md> | 个人 | 最后一条 **2026-09-09** | 僵尸请求修复的 nightly pin；GB10 断连实测；窗口内无更新 |
| <https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/blob/main/README.md> | 个人 | 2026-09（含 09-05） | DFlash2 官方镜像 pin、crash 史（dequant lm_head 硬重启已修）、mem 0.90 |
| <https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B> | **官方（SGLang）** | 2026-08-21 起 DFlash2 为官方配方 | DGX Spark cell = NVFP4 + DSpark/DFlash2、mem 0.80/0.85、chunk 2048；官方 38.28 tok/s |
| <https://api.github.com/repos/sgl-project/sglang/issues/36548> | 上游 issue（未修） | 2026-08-26 | DFlash2 高并发下跨请求上下文串扰 |
| <https://api.github.com/repos/sgl-project/sglang/issues/38009> | 上游 issue（未修） | 2026-09-04 / 更新 09-10 | DFlash2 + thinking 的 greedy 分歧（token 30） |
| <https://github.com/mudler/vllm.cpp/issues/2784> | 第三方 issue（已关） | 2026-09-03 | 8,159 token 档接受率 0.06、比无草稿慢 23%（合成 prompt） |
| <https://huggingface.co/api/models/incoai/Qwen3.8-27B-DFlash2/discussions/8> | HF discussion（个人） | 2026-08-26（编辑 08-31） | vLLM + 草稿 YaRN patch → 真实会话 539K、接受率 ~47% |
| <https://huggingface.co/api/models/incoai/Qwen3.8-27B-DFlash2/discussions/9> | HF discussion（个人） | 2026-08-28 → 09-07 | 深度 4–7 接受率 ≤0.1；降 3 退化成 MTP |
| <https://huggingface.co/drowzeys/keys-Qwen3.8-27B-DSpark-vLLM-YaRN-1M-Context> | 个人（HF model card） | 无日期 | DSpark 草稿 rope overlay；1M 接受率未测 |
| <https://github.com/drowzeys/keys-Qwen3.8-27B-NVFP4-DFlash2-Ablit-Cybersecurity-Unlock-1M-Context-YARN-Single-DGXSpark> | 个人（社区拼装 + 实测） | 实测 **2026-09-08** | 单 Spark 1M needle HIT @999,714、2.89 h prefill、KV 1,113,770、gmu 0.70 |
| <https://huggingface.co/api/models/RadixArk/Qwen3.8-27B-NVFP4/discussions?p=0> | 权重方 HF | 最新 2026-08-22 | 窗口内无新讨论/revision |
| <https://forums.developer.nvidia.com/t/380244> | 论坛（个人） | 2026-08-15 ~ 08-20 | vLLM+MTP 24.7 tok/s、prefill 1,734→853 tok/s、KV 37,169 B/token、unsloth tokenizer 截断坑、长上下文 decode 塌到 3.7 tok/s |
| <https://forums.developer.nvidia.com/t/380957> | 论坛（个人，含后续 A/B） | 2026-08-22 / 08-29 / **09-03** | DFlash2 vs vLLM+MTP 工具评测；greedy > thinking；BF16 head −14~−25% 无质量收益；mem>0.50 危险 |
| <https://forums.developer.nvidia.com/t/382446> | 论坛（个人，含官方仓库链接） | 2026-09-05 | Flash-Next 单机 1M/512K profile、1,431,164 token FP8 池、KV_TARGET 20/22 只剩 6.9–8.8 GiB |
| <https://forums.developer.nvidia.com/t/380286> | 论坛（个人 + NVIDIA 员工） | 2026-08-15 / 08-26 | "硬重启"实为**恒温器硬件缺陷**并已 RMA——勿归因引擎 |
| <https://forums.developer.nvidia.com/t/383624> | 论坛（个人） | 2026-09-18 ~ 09-24 | llama.cpp 持续推理静默硬冻结；PowerStress FAIL MODS-020000600139（热/传感器）；RMA |
| <https://spark-arena.com/benchmark/87a93f88-afee-4e93-89de-7cfec34c8345> | 第三方基准平台 | 经 hasso5703 转述 | 双 Spark TP=2 深度阶梯；100K −28%；TP=2 不提升单流 |
| <https://jishuzhan.net/article/2100796589846220802> | 第三方聚合博客（中文，转载型） | 2026-09-18 | 跨平台综述：DGX Spark+MTP ~20 tok/s 且 prefill 强；llama.cpp 满 262K 但 TTFT 可达 ~30 min。**无版本/参数 pin，仅作趋势参考** |
| <https://github.com/styles01/sparkrun-recipes/blob/main/runbooks/qwen-38-27b.md> | 个人 runbook | 2026-08-18（经论坛转述） | stock vLLM 无 sm_121a NVFP4 kernel；需专用镜像 |

---

## 9. 值不值得换？——判断

### 9.1 结论：**不要换引擎/形态；换"档位"**

**不换的理由（每条都有 ≥500K 或稳定性的直接证据）**

1. **vLLM 在单流上明显更慢**，不是"各有千秋"：同一台机器、同一份 NVFP4、agentic 工具评测里 DFlash2 的 code 32.6 vs 19.8、reasoning 47.3 vs 25.4 tok/s，中位回合 3.6 s vs 7.8–10.3 s（2026-08-22 / 09-03，论坛 380957）。hasso5703 的独立复现（erikvullings）也只在**空载 freshly-rebooted** 的机器上拿到 vLLM +8~30%，并把结论写成"engine choice is not the lever, box load and content are"。
2. **vLLM 的 1M prefill 慢约 3×**：折算 ~96 tok/s（999,714 tokens / 2.89 h，2026-09-08）vs SGLang 27B 1M 单元的 331 tok/s（2026-09-21）。对一个要跑 500K+ 的负载，prefill 才是主成本。
3. **vLLM 路线的 1M 要把并发压到 2**（drowzeys 配方自述），并且依赖第三方 GB10 专用镜像（stock vLLM 在 sm_121a 上根本没有 NVFP4 kernel）、还要绕开 unsloth tokenizer 的 2048 截断坑。
4. **llama.cpp 只有约 1/3 的 decode、约 1/3 的 prefill**，且社区明确说在 GB10 上它是慢路径。
5. **TensorRT-LLM：查不到任何 DGX Spark / sm_121 上的 Qwen3.8-27B 配方**，不能作为候选。
6. **双 Spark 对单请求 500K+ 没有增益**：TP=2 的 batch-1 decode 仍在 34–40 tok/s 同一档（100K 前缀 28.77），只买到并发（c5 115 / c10 97 agg）。社区原话："Don't cluster for latency."
7. **Flash-Next 176B 单机**是唯一在 prefill 上真正跨档的替代（1,495–2,250 tok/s，快 4–6×），512K YaRN profile needle 也能 3/3 PASS；但那是**换模型**（窗口 512K 而非 1M、license 非商用、FP8 KV 档 needle 15/20、需要 230–265 GB 磁盘和 12–15 min 开机写 PLE 表）。**它适合作为"第二条 lane"**，不适合替换 1M 的 27B lane。

### 9.2 值得做的升级（按性价比排序，都在现役 SGLang 路径内，可回滚）

| # | 动作 | 预期收益（来源数字） | 风险 / 前置条件 |
|---|---|---|---|
| 1 | **草稿换成 `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@bd7a934` + `--speculative-draft-model-quantization modelopt_fp4`** | 外部 +25.19% 整请求（57.11→71.50）；作者自测 greedy 中位数 65.3 vs 50；草稿显存 3.53→1.37 GB，转成 KV 池 | 需要 drafter 量化后端支持 `modelopt_fp4`；建议先在 native 262K 上 A/B，再上 YaRN |
| 2 | **`--speculative-num-draft-tokens` 8 → 16** | 作者深度扫描 D8 47.6 → D16 56.6 pooled tok/s | **与 HF discussion 的观察冲突**（多人说 positions 4–7 ≤0.1、降到 3–4 更好）。必须在你自己的工作负载上 A/B；接受率是内容相关的 |
| 3 | **`--mem-fraction-static` 0.70 → 0.76**（前提是镜像口径匹配） | 0.70 在官方 release 上白扔 15% 池（770,118 vs 906,524 tokens），并留 10 GB 未认领；0.76 后 graph capture 余量 **21.30 GiB vs 19.38 GiB** | **这是唯一直接触碰稳定性边界的动作**。>0.80 已实测崩（3 并发、剩 2 GiB）；cookbook 0.85 被 earlyoom 杀 15/48。建议 0.72 → 0.74 → 0.76 小步走，每步盯 `MemAvailable` 与 GPU headroom |
| 4 | **镜像升到官方 release（v0.5.19 或更新），并加 `--disable-flashinfer-autotune`** | 官方镜像 vs 自建：54.58→54.57（modelopt 平）但 compressed-tensors 上 **+14.1% code / +6.3% essay**；v1.14 迁移 71.4 vs 69.8、acceptance 4.29 vs 4.09、needle 300,108 精确。`--disable-flashinfer-autotune` 让单流逐位复现、c8 落在 ±1.6%，开机快 ~2 分钟，代价约 2% 平均吞吐 | **注意你现在的镜像可能已含 #35255（僵尸请求修复）**：MiaAI 的 `nightly-cu134-20260909` 含它，而 `v0.5.19`（2026-09-03 打标）**不含**。升级时务必确认目标镜像同时满足：DFlash2 + 量化 head selector（#35496）+ 僵尸修复（#35255） |
| 5 | **运维加固（可选，不改性能）** | `Restart=always`（Triton 编译崩溃会以 `SystemExit: 0` 结束，`on-failure` 不会重启）、`HF_HUB_OFFLINE=1`（防 YaRN patch 被还原）、keepalive proxy（冷 690K prefill 会让连接静默几十分钟）、token-0 腐蚀哨兵、僵尸守卫 | 都是纯加法，不影响现役成功路径 |

### 9.3 需要持续盯的开放风险

- **sglang#36548（DFlash2 高并发跨请求串扰，open）**：若你有多并发 coding agent，这是"内容被写进错的对话"这类静默错误，而不是崩溃——**最危险的一类**。缓解：限制并发、或在 proxy 层做请求隔离观察。
- **sglang#38009（DFlash2 + thinking 的 greedy 分歧，open）**：你默认 thinking 开启（`reasoning_effort=xhigh`）。社区的无损性研究认为这属浮点邻域分歧而非质量损失（GSM8K 200 题打平），但如果你的工作流需要"确定性可复现输出"，就得知道这个分歧存在。
- **sglang#40076（`top_logprobs` 无上界可干掉 scheduler，open）**：不要向现役 engine 发送无界 `top_logprobs`。
- **`mem-fraction` 与整机 livelock 的耦合**：GB10 上 SGLang 看不到 25–40 GB 瞬时分配，且 `nvidia-smi` 在冻结后不返回、kernel 可能什么都不写。任何调参都应保留 Docker `--memory` 上限 + 逐 boot 记录 GPU headroom。

### 9.4 若你更看重"prefill 快"而非"1M 窗口"

可考虑**并行部署第二条 lane**：Flash-Next 176B 单机（hasso5703 的 `flash` target / MiaAI 的单机 recipe），262K 档 prefill ~2,250 tok/s、code 47.9 tok/s，或 512K YaRN 档（1,495 tok/s @400K、needle 15/20）。这不是替换，是分流；但要注意 disk 需求（230–265 GB）、boot 时间（12–15 min，每次都要重写 47.7 GiB PLE 表）、以及 Qwen Research 非商用 license。

---

## 10. 查不到的（明确声明，未做任何编造）

1. **TensorRT-LLM 在 DGX Spark / sm_121a 上跑 Qwen3.8-27B 的任何配方、版本或实测数字** —— 完全查不到。
2. **NVIDIA 官方 DGX Spark playbook 里的 Qwen3.8-27B 配方** —— 查不到。只找到 NVIDIA 发布的**权重**（`nvidia/Qwen3.8-27B-NVFP4`，2026-09-08）与 `nvidia/Qwen3.8-Flash-Next-NVFP4`。
3. **Fyzen/其他 ≥700K 上下文的 vLLM/SGLang 成功案例的 prefill 时间** —— 除 hasso5703（690K，~40 min，SGLang）与 drowzeys（999,714 tokens，2.89 h，vLLM）之外，没有更多带 prefill 时长的 ≥700K 报告。
4. **DFlash2 在 ≥500K 上下文下的接受率** —— 只有 incoai #8 的 539K / ~47% / ~1.9 accepted（vLLM，单次真实会话）；SGLang 侧没有公开发布的 ≥500K 接受率曲线。hasso5703 自述"受控的单机长前缀 cell（DSpark vs MTP at 0/8K/32K）仍在待办清单上"。
5. **`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal` 在 1M/YaRN 下的表现** —— 该草稿的所有已发布数字都来自 native 262,144 上下文；NVFP4 校准草稿 + YaRN 1M 的组合**未见到任何实测**（这是一项你自己就能补上的空白）。
6. **`hasso5703` 仓库 v1.19 的正式 release 条目**（README 提到 "v1.19 campaign"，CHANGELOG 里我抓到的窗口内最新是 v1.18.7 / 2026-09-25）—— 是否存在 v1.19 release 未能确认。
7. **cookbook 的 DGX Spark cell 的当前原文**（该页正文由 JS 渲染，`.md` 版本只含 Playground 组件代码，未包含 cell 的 flag 明细）—— 本文中 cookbook cell 的参数转述自 hasso5703 的对比段落，非直读官方页面原文。
8. **`MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000`（搜索中出现的另一个仓库）** —— 仅知其存在，未调查内容与日期。

---

*报告完成时间：2026-09-30。所有引用数字均来自上表 URL 的原文；个人/社区来源与官方来源已逐条标注。*
