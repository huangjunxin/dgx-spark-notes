# Qwen3.8 部署方案社区调研：上游版本核查 + GB10 方案横向对比（2026-10-29）

> 状态：**纯调研，零变更**（现役容器/脚本/配置一概未动）
> 结论先行：① 上游三条线（主模型权重 / 架构 / 运行时镜像）均无更新，现役部署仍 pin 在最新版；② 社区不存在整体性更优方案，现役技术路线（SGLang + RadixArk NVFP4 + DFlash2 + YaRN 1M）就是主流最优路线；③ 从 hasso5703 仓库整理出 6 项可 cherry-pick 的工程改进，**本轮决议：暂不更新**，留作下次维护窗口菜单（见 §6）。
> 关联：[DEPLOYMENT-RECORD.md](../qwen38-sglang/DEPLOYMENT-RECORD.md)（9/9 首版部署）· [CHANGELOG-20260930.md](../qwen38-sglang/CHANGELOG-20260930.md)（草稿换代 nvfp4）· [qwen38-dflash2-crash-20260918.md](qwen38-dflash2-crash-20260918.md)（并发 4 的由来）· [qwen38-watchdog-denoise-20261029.md](qwen38-watchdog-denoise-20261029.md)（同日）

## 1. 调研动机与范围

- 动机：现役服务 9/30 换草稿后已稳定运行 29 天，回答"要不要动"之前，先核查上游有没有新版本、社区有没有更优/更稳的 GB10 部署方案。
- 范围：Hugging Face（官方 Qwen org、RadixArk、社区量化）、GitHub（仓库/issue 搜索）、sglang 上游 issue、awesome-dgx-spark 清单、HN。Reddit 反爬（403）未取到，不影响结论。
- 本机基线（核查时的现役状态）：容器 `qwen38-27b-sglang`，镜像 `lmsysorg/sglang:qwen38-27b`（digest `febfb971…`），主模型 `RadixArk/Qwen3.8-27B-NVFP4@319f741c` 的本地 YaRN 1M 变体，草稿 `maurienne-ai/…DFlash2-NVFP4-RTNcal@bd7a934`，`--mem-fraction-static 0.70`，并发 4，bench.py 稳态 27.0 tok/s。

## 2. 上游版本核查：三条线均无更新

### 2.1 模型权重（4 个 pin 的仓库全部与上游最新 commit 一致）

| 仓库 | 本地 pin | 上游最新 sha | 上游 lastModified | 结论 |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B`（官方原版） | — | `1d4bf0f2` | 2026-08-14 | 之后零提交，无新版 |
| `RadixArk/Qwen3.8-27B-NVFP4`（主模型） | `319f741c` | `319f741c` | 2026-08-22 | ✅ 一致 |
| `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`（草稿） | `bd7a934` | `bd7a934` | 2026-08-30 | ✅ 一致 |
| `RadixArk/Qwen3.8-27B-DSpark`（备用草稿） | `b9a5dbdf` | `b9a5dbdf` | 2026-08-29 | ✅ 一致 |

### 2.2 模型架构：无变化

本地 yarn1024 变体 vs 官方 `Qwen/Qwen3.8-27B` main 分支 config 逐项比对全部一致：
`Qwen3_5ForConditionalGeneration` / `qwen3_5`，64 层混合架构 = **48 linear_attention + 16 full_attention（1:4 间隔）**，hidden 5120、GQA 24/4、head_dim 256、intermediate 17408、vocab 248320。本地唯一差异是刻意烧入的 YaRN（`max_position_embeddings` 262144→1048576，factor 4.0 @ 262144），属自家长上下文配置，非上游改动。

### 2.3 运行时镜像：无更新

Docker Hub `lmsysorg/sglang:qwen38-27b` tag 当前 digest 仍为 `sha256:febfb971c735…`，与 9/9 部署记录 pin 的完全一致。
（注意：hasso5703 用的是**另一条线**的镜像 `lmsysorg/sglang:v0.5.22`（digest `e3a4f060`，2026-10-16），比我们的新一个月+，见 §4。）

### 2.4 同系列新发布（都是"新型号"，不是 27B 的更新）

| 发布 | 日期 | 说明 | 对本机 |
|---|---|---|---|
| `Qwen/Qwen3.8-Flash-Next`（+FP8） | 08-24 | **新架构** `Qwen4ExpForConditionalGeneration`/`qwen4_exp`，512 专家 MoE | 单机方案已成熟但需换引擎生态，见 §3.3 |
| `Qwen/Qwen3.8-2.4T-A95B`（+FP8） | 08-08 | 旗舰 MoE | 本机不可能 |
| RadixArk 10 月发布（Minimax-M3.1、Muse-Spark、GLM-5.3 等） | 10 月 | 别的模型线 | 与 27B 无关 |

## 3. 社区 GB10 部署方案盘点

### 3.1 同模型（Qwen3.8-27B）同硬件方案

| 方案 | 热度/活跃 | 核心数据 | 与本机关系 |
|---|---|---|---|
| **hasso5703/dgx-spark-qwen38** | ★242，日更（v1.40.4，10-29） | 同为 NVFP4+DFlash2+SGLang+1M；自称 71.4 tok/s（贪心 code 中位数），8 流 135-148 | ⭐ 深挖见 §4，**唯一值得认真对标的** |
| 0xWhiteMage/…kearuga-sglang-dgx-spark-dflash2 | ★13，10-06 | **蒸馏版 DFlash2 草稿（Kearuga）**：C1 35.3 / C4 108.9 tok/s，比 stock +14.5%/+10.8% | 唯一现成的"草稿再换代"候选，可用 `switch-draft-flavor.sh` 流程 A/B |
| AEON-7/vllm-ultimate-dgx-spark | ★147，10-06 | vLLM 0.29 sm_121a 镜像 + DFlash2，自称单流 3.39×，Triton NVFP4 KV | 换引擎路线，迁移成本高，仅记录 |
| albond/DenseSpark-Qwen3.8-27B | ★16，09-01 | vLLM + INT4 AutoRound + MTP：49.1 C1 / 260 C16 | INT4 低于社区公认质量地板（NVFP4），并发数字再高也不换 |
| liuzl/qwen38-dgx-spark-lab | ★2，10-29 | vLLM 轨 + per-request rank-1 LoRA（去审查向）+ MTP K7 | 与本机需求无关 |

### 3.2 hasso5703 深挖（读完 README/BENCHMARKS/docs/gb10-memory/1M 模板后的定案）

**① 它和我们是同一套方案。** 草稿就是同一个 `maurienne-ai/…DFlash2-NVFP4-RTNcal@bd7a934`；YaRN 也是 patch 进 checkpoint config（我们的 `make-yarn-variant.py` 当初就是参考它的 `patch-yarn.py` 做法）；上下文同为 1,010,000。

**② 它的 71.4 不能直接跟我们的 27.0 比。** 口径警告（它自己在 BENCHMARKS.md 里写得很诚实）：
- 71.4 = `bench.sh` 贪心中位数，**code/math 类内容**；自由散文在同引擎下只有 13-21 tok/s（任何引擎+草稿组合都慢 2-3×）；
- 我们的 27.0 = bench.py 400-token 稳态（13.5-16.6s 墙钟），口径不同；
- 它自己的 DSpark→DFlash2 对照（同 box 同日）：贪心中位 48.7 → 72.0，接受率 3.49 → 4.29——**DFlash2 优于 DSpark 的定案与我们 9/30 的选择互相印证**；
- 它的调参结论（torch-compile +1 tok/s、continuous-decode 2、fp8 草稿更慢、拒绝 compact ragged-verify/block 9/single-batch-overlap）**我们已全部一致或踩过同样的坑**，说明现役参数没有漏掉免费收益。

**③ GB10 统一内存陷阱**（`docs/gb10-memory.md`，对稳定性最有价值的一篇）：
- SGLang 内存记账**看不到 25-40 GB 瞬时分配**（flashinfer fp8 autotuner 与 CUDA graph capture 在池外分配）；裸机运行或 autotuner 开着时 fraction 一高，宿主可用内存可被打到 0 → **整机死锁只能断电**；
- 它的护栏：`--memory 100g --memory-swap 100g`（逃逸杀容器不杀宿主）+ `Restart=always` + `ExecStartPre docker rm -f` + engine-preflight（等上一引擎显存归还再启，防 sglang#40948 冻结）；
- 0.70→0.76 的演变是"数字随镜像走"：官方镜像静态预算更保守，同 0.70 下池子 770,118 vs 本地构建的 906,524，0.76 拿回池子（~902K）且 capture 后余量反而更大（21.3 vs 19.38 GiB）；**0.80 在 3 并发 + 同机跑工具的条件下实测崩**，官方 cookbook 的 0.80 是单流 boot-and-serve 口径，不矛盾；
- 我们的镜像对应哪一档未实测，**换镜像前不要动 0.70**。

**④ 它的 1M 模板里三个我们没开的东西**：`--sleep-on-idle`（治 idle scheduler 忙转——我们的 scheduler 自 9/30 起常驻 ~101% CPU，29 天烧了约 694 核时）、`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`（/health 不真生成 token）、API key 走文件注入不进命令行。

**⑤ chat-template 补丁**（v1.40.3/v1.40.2）：Claude Code 的 ToolSearch（`tool_reference` content block）会让 Qwen 模板 500、opencode 双 system message 会 400。**本机若出现 Claude Code/opencode 直连 8888 的用法，先打它的 `patch-template.py` 再排查别的。**

### 3.3 换模型路线（想要"更优秀"时的另一条路）

Qwen3.8-**Flash-Next**（qwen4_exp MoE）单机 GB10 方案已成熟：blazux ★484、salient-data（pinned nightly+14 patches，71.3 tok/s，GSM8K 99/100）、MiaAI-Lab（512K YaRN，48.7 prose）、hasso flash lane（176B@262K，47.9）。质量显著高于 27B（GSM8K 97+ vs 94.0），MoE 单流速度还更快。代价：本机现役 sglang 镜像不能直接加载 qwen4_exp，要走 vLLM 生态或 hasso 那种自编 sm121 kernel——**是换模型项目，不是换部署参数**。

### 3.4 跨硬件参考（记录备查）

jpezzulli/sglang-rtxpro6000（Penny Royal，SM120 RTX PRO 6000）：HiCache+NIXL 会话持久缓存，490K 会话恢复只 prefill 5 个 token；DFlash2 27B 需额外 24.83 GB 宿主内存放草稿缓存。硬件不同不可直接搬，**"会话缓存跨重启"的思想**留作 1M 长会话场景的参考。

## 4. 现役 vs hasso5703（1M lane 模板逐参数）

| 参数 | 本机现役 | hasso 1M | 评价 |
|---|---|---|---|
| 镜像 | `febfb971`（9 月初谱系） | v0.5.22 `e3a4f060`（10-16） | 他新一个月+，含 #41841 调度停顿、#41737 DFlash2 贪心选择器等修复 |
| mem-fraction | 0.70 | 0.76 | 换镜像后才有意义 |
| **draft depth** | `--speculative-num-draft-tokens 8` | **D=16**（默认） | **最大速度杠杆**（其接受率数据 4.29 支撑） |
| decode graph bs / 并发 / mamba 池 | 4 / 4 / 48 | 8 / 8 / 96 | 我们收到 4 是 9/18 崩溃换来的教训（`qwen38-dflash2-crash-20260918.md`），属选择不是落后 |
| KV / attention / chunked-prefill / autotune off / torch-compile / continuous-decode | ✅ 一致 | ✅ 一致 | 无漏项 |
| `--sleep-on-idle` | 无 | 有 | 省 1 核；上游 #43050 idle-wake bug 未关，我们开多模态（EPD 路径 #45397）需观察 |
| Docker 宿主内存帽 | **无** | `--memory 100g`×2 | **本机最大盲区**（§3.2③ 整机死锁风险） |
| 启动前置 | 脚本断言（参数联动/负向用例） | engine-preflight + decode 腐化 smoke 测（'!' 流） | 可互补：我们缺"显存归还等待"和腐化检测 |
| 健康监控 | **自研 watchdog**（分级自愈/风暴保护/降噪告警） | systemd Restart=always | ✅ 本机明显更强 |
| template 补丁 | 无 | 有（Claude Code/opencode） | 有此类客户端才需要 |

**总评**：不是谁整体更好——**他赢在引擎侧（镜像新、参数激进、进程护栏全），我们赢在运维侧（自愈体系、A/B 纪律、脚本断言）**。整体迁移到它的 installer 不划算（那是整机 appliance：9 条 lane + cockpit + 自己的看护体系，会推平本机与 vllm-open-proxy/switchyard/watchdog 的集成）。

## 5. 可 cherry-pick 清单（按优先级，均已评估、未执行）

| # | 项 | 预期收益 | 风险/前置 |
|---|---|---|---|
| 1 | Docker `--memory 100g --memory-swap 100g` | 补上整机死锁盲区，与 watchdog 叠加 | 零风险，随时可做 |
| 2 | draft depth 8→16 A/B | code 类吞吐可观提升（hasso 接受率 4.29 支撑） | 散文类可能回退；用 `ab-collect.sh` 固化快照，看真实流量构成定夺 |
| 3 | 镜像升 v0.5.22（`CHECK=1` 预检+逐参数 diff，流程现成） | bug 修复 + 为 4/5 铺路 | 单独维护窗口，冷启动 ~7min |
| 4 | mem-fraction 0.70→0.76 | 拿回 ~15% 池子 | **必须先做 #3** |
| 5 | `--sleep-on-idle` | 停掉常驻 101% CPU 忙转 | 先做 #3；开多模态需观察 idle-wake 几天 |
| 6 | chat-template 补丁 | Claude Code/opencode 可用性 | 按客户端情况选配 |

**不采纳项**：整体换 hasso installer（§4 总评）；并发回 8（没有真实需求不为对标而动）；换 vLLM 系（迁移成本）；INT4/AutoRound（质量地板）；本轮换 Flash-Next（换模型 ≠ 换部署，另行立项）。

## 6. 决议（2026-10-29）

**维持现役，不更新。** 理由：三条线均无更新（§2）、现役已在社区最优路线上、稳定期（29 天）价值高于边际收益、且 #2-#5 都依赖镜像升级这个单独窗口，不适合与日常负载混做。

**重开触发条件**（满足其一即重开本轮调研的执行）：
1. 出现 Docker 宿主内存吃紧/(any OOM 迹象 → 立即做 #1，不等窗口)；
2. 决定做镜像升级窗口时，顺路带 #3/#4/#5；
3. 出现 Claude Code/opencode 直连 8888 的 500/400 → §3.2⑤ template 补丁；
4. 对 code 类吞吐有真实需求 → #2（单变量 A/B）；
5. 上游 `Qwen/Qwen3.8-27B` 或草稿仓库出新 commit / 新镜像 digest → 重跑 §2 三条线核查（复现命令见附录）。

## 附录 A：核查方法（可复现）

```bash
# 上游仓库最新 commit / lastModified
curl -s "https://huggingface.co/api/models/RadixArk/Qwen3.8-27B-NVFP4" | python3 -m json.tool | grep -E '"sha"|lastModified'
# 官方 config 架构比对（对照本地 variants/qwen38-nvfp4-yarn1024/config.json）
curl -s "https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json" | python3 -m json.tool
# 运行时镜像 tag 当前 digest（对照 DEPLOYMENT-RECORD pin）
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:lmsysorg/sglang:pull" | python3 -c "import json,sys;print(json.load(sys.stdin)['token'])")
curl -s -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
  -I "https://registry-1.docker.io/v2/lmsysorg/sglang/manifests/qwen38-27b" | grep -i docker-content-digest
# 本机服务架构自报
curl -s http://127.0.0.1:8888/get_model_info | python3 -m json.tool | grep -E "model_type|architectures" -A2
```

## 附录 B：主要来源

- hasso5703/dgx-spark-qwen38：README · BENCHMARKS.md · docs/gb10-memory.md · `qwen38-sglang-1m.service.template` · releases v1.40.1–v1.40.4（github.com/hasso5703/dgx-spark-qwen38）
- MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark（★480，--sleep-on-idle PR #13，09-05 后停更）
- 0xWhiteMage/qwen3.8-27b-kearuga-sglang-dgx-spark-dflash2（★13）· AEON-7/vllm-ultimate-dgx-spark（★147）· albond/DenseSpark-Qwen3.8-27B（★16）· liuzl/qwen38-dgx-spark-lab（★2）
- bidual/awesome-dgx-spark（Flash-Next 单机方案群：blazux ★484、salient-data、MiaAI-Lab、sayyidfareed ~990K token 验证等）
- jpezzulli/sglang-rtxpro6000（Penny Royal，SM120）
- sglang 上游 issue：#43050（idle-wake hang）· #45397（EPD 请求不休眠）· #40948（内存未归还冻结）· #41841 · #41737 · #35371（DFlash2 上游合入）
- HN：blog.kubesimplify.com "Running Qwen3.8-27B on DGX Spark"（08-18，早期内容，已被上述覆盖）
