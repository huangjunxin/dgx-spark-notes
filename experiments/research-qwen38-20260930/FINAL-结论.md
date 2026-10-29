# Qwen3.8-27B on DGX Spark：部署方案调研结论（2026-09-30）

> 问题：有没有**更好的**部署方案？要求**长上下文** + **更稳定**。
> 方法：4 条并行线（上游引擎 / 社区方案 / 长上下文与稳定性工程 / 本机资产审计）+ 父 agent 一手取证（容器实测、镜像源码、HF 制品校验）。
> 证据文件：`A-upstream-engines.md`(624 行) · `B-community-recipes.md`(353 行) · `C-longcontext-stability.md`(408 行) · `D-local-audit.md`(334 行)

---

## 0. 一句话结论

**没有更好的"形态"可换（vLLM / llama.cpp / TensorRT-LLM / 双机都更差或不可用），现役 SGLang + DFlash2 + config 内联 YaRN 1M 就是社区领先路线；但现役配置落后社区最优约 6 个 release，有 4 项"同形态升级"值得做 —— 其中 2 项零风险、1 项是最大单项收益（+25%）、1 项直接回答"稳定一些"。**

真实短板不是"引擎选错了"，而是三件事：
1. `.start` 脚本默认值 = 已知必崩配置（并发 8 与 decode graph bs=4 错配）；
2. 主机侧缺 OOM 护栏，且健康检查用 `/health`（wedged 时仍返回 200）；
3. 长上下文 prefill 慢是 **kernel 效率**问题（不是带宽/内存），652K ≈ 32 分钟，只有三个真杠杆。

---

## 1. 架构层面：不换（有硬数据）

| 候选 | 结论 | 关键证据 |
|---|---|---|
| vLLM v0.30.0 | ❌ 更慢 | 同机同权重：code 32.6 vs **19.8**、reasoning 47.3 vs **25.4** tok/s；1M prefill 折算 ~96 tok/s vs SGLang 331 tok/s（慢 ~3×）；1M 时并发只能 2 |
| llama.cpp | ❌ 更慢 | decode 与 prefill 均约 1/3；另有 open 的 MTP draft 加载 bug #29044 |
| TensorRT-LLM | ❌ 不可用 | 仍 prerelease（rc29, 09-29），无 GB10 + Qwen3.8 支持声明 |
| 双 DGX Spark (TP=2) | ❌ 不解决单流 | batch-1 仍 34–40 tok/s 同档（100K 前缀 28.77，−28%），只买并发 |
| 升官方 `v0.5.20` | ⚠️ 不建议现在 | 唯一新正式 tag（09-18），但 **DFlash2 专用 tag 停在 08-22**；换过去会丢 DFlash2 注册与既有修复；#33505/#33454 仍未修；需处理 17 个 removed flag + `ServerArgs` 改 msgspec.Struct |
| Flash-Next 176B | 🔸 第二条 lane | prefill 快 4–6×，但窗口仅 512K + 非商用 license，不能替代 1M 的 27B |

**上游阻塞点（截至 09-30 仍 open）**：#33505（override 无条件下发 draft —— #35841 已 stale 5 周）、#33454（verify 窗口越界 —— #33963 已 stale）。
⇒ **"两份 config.json 手工打补丁"仍无上游替代方案**，我们的做法就是当前最优解。
> 注：官方 cookbook 的 DGX Spark pin 是 `mem-fraction 0.80 + 并发 1`，验证口径只有 ISL 8192，**从未验证 ≥500K**；官方只用 `fp8_e4m3` KV。我们的 0.70 + 1M 属社区路线（社区 1M 档用 0.76）。

---

## 2. 值得做的升级（按性价比排序）

### P0-A ✅ 零风险：修脚本默认值（今天做）
`qwen38-sglang/start-dspark-long.sh:47,129` 默认 `MAX_CONCURRENT_REQUESTS=8` + `MAMBA_SLOTS=12`(→96)，而 `--cuda-graph-max-bs-decode 4` 硬编码。
⇒ **照默认值重建 = 把 9/15–9/18 崩过 5 次的那套配置装回去**（DFlash2 采样器缓冲区按 graph max bs=4 分配，decode batch ≥5 必崩：`greedy_mask[:bs].copy_` 尺寸不匹配）。
修法：默认改 4、`MAMBA_CACHE_SIZE` 与并发联动、并加断言 `max_running_requests ≤ min(cuda_graph_max_bs_decode, ...)`。`dgx-spark-notes` 版脚本同样有此问题。

### P0-B ✅ 零风险：主机侧稳定性加固（直接回答"稳定一些"）
已查实：**cgroup 对 CUDA 统一内存完全失明**（实测 10.7GB GPU 只计到 377MB），`systemd-run MemoryMax` 会被 Docker 逃逸 → 唯一真护栏是 mem-fraction + PSI 看门狗。
- earlyoom：`-m 3 -s 20 --prefer 'python|sglang|triton' --avoid 'systemd|sshd|dockerd|containerd'` + CAP_KILL
- systemd user 服务：`OOMPolicy=stop`（否则 OOM→重启→砖循环）+ `StartLimitBurst=3`
- **健康检查别用 `/health`**（wedged 时仍 200）→ 换成真实小生成 canary + `/proc/pressure/memory` 的 `full>0`
- `--crash-dump-folder` 指向**容器外**持久目录（9/18 那次崩溃的 traceback 随 `docker rm -f` 丢了）
- 现有 `qwen38-watchdog.service` 已在跑（09-15 上线，分级判定"告警要早、动作要晚"），保留即可

### P1-A ⭐ 最大单项收益：草稿换 NVFP4 量化版（下次维护窗口）
`maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@bd7a934`（1.55 GB）+ `--speculative-draft-model-quantization modelopt_fp4`
- 外部实测整请求 **+25.19%**（57.11→71.50 tok/s）；草稿显存 3.53→1.37 GB，差额转 KV 池
- **父 agent 已核实无需换镜像**：本镜像 CLI 已支持 `modelopt_fp4`；该草稿 config 与现役 z-lab 草稿**逐字段同构**（`max_pos 262144` / `rope_type default` / 5 层 `sliding_attention` / `sliding_window 2048` / `block_size 8` / `selector_top_k 16`）→ 现役 YaRN 覆盖手法原样适用
- 唯一改动点：`make-yarn-variant.py:21` 把草稿源目录硬编码为 `dflash2-draft`、前缀硬编码 `qwen38-dflash2` → 需加 `--dflash2-dir` / `--dflash2-prefix`
- 溯源已闭环：`z-lab/Qwen3.8-27B-DFlash2` 与 `incoai/Qwen3.8-27B-DFlash2` 的 `model.safetensors` **sha256 完全相同**（`67fc76d6…`，3,848,817,896 B），incoai 只是 09-17 的镜像仓 ⇒ 官方 cookbook 指向的草稿 = 本机现役那份，**无需更换**；要换的是它的 NVFP4 衍生版

### P1-B ⚠️ 需小步走：mem-fraction 0.70 → 0.76
- 0.70 在长上下文下白扔约 15% 池（官方 release 上 770,118 vs 906,524 token）；社区 1M 档用 0.76
- 但 **>0.80 已实测崩**（3 并发剩 2 GiB；cookbook 0.85 被 earlyoom 杀 15/48）
- 路径：0.72 → 0.74 → 0.76，每次冷启后核对 `max_total_num_tokens` 与**"capture 后空闲 ≥15 GiB"**，不达标立即回退

---

## 3. 需先做判别实验（不要直接上）

| 项 | 做法 | 判据 |
|---|---|---|
| **FP4 KV** | `--kv-cache-dtype fp4_mx_block16 --attention-backend triton --page-size 1`（**别用 `nvfp4`，triton 读不了**） | 先做 N=300K 的 `bf16` vs `fp8` 判别：bf16 明显更慢→带宽受限→FP4 值得（理论 1.7×）；不变→算力受限→别动。且必须重跑 652,137 needle。注意上游只对 SM100 承诺原生，**SM121 无声明**；换 triton 会离开现役 flashinfer 已验证路径 |
| **chunked-prefill 8192 → 2048** | 官方 DGX Spark cell 用 2048（8192 会让 decode 卡 ~600ms） | 属混载 ITL 优化；改大 chunk 时必须把 `--watchdog-timeout` 提到 1800，否则 server 自杀 |
| **draft depth 8 → 16** | `--speculative-num-draft-tokens 16` | 证据冲突：作者 D16 56.6 vs D8 47.6 pooled tok/s，但 incoai HF #9 多人实测 positions 4–7 接受率 ≤0.1 → 必须本机 A/B |
| **mamba 池 48 → 20/16** | 引擎只按 S 除、verify buffer 独立，48 可能白占 ~2.2 GB | 改后核对日志 `max_total_num_tokens`；可换回 ~70K KV token |
| **恢复 8 并发** | `--cuda-graph-max-bs-decode 4 → 8` **必须与 8 并发压测同一变更单** | 已有 11.8 天 bs≤4 稳定证据；bs 5–8 路径本机从未验证。若不成，`--max-running-requests` 必须显式 pin（投机解码下否则被重置为 48） |

---

## 4. 明确排除（附理由，避免重复踩）

| 选项 | 为什么不做 |
|---|---|
| `--enable-unified-memory` | 父 agent 读本镜像 help 实测：**"not yet compatible with … speculative decoding"**，且需 Triton 后端 → 与 DFlash2 互斥（本地文档曾把它列为候选，应划掉） |
| KV offload 到 host memory | GB10 统一内存下是 **no-op**（同物理池）；只有 HiCache L3 落盘能真加容量 |
| cgroup / systemd 内存限制当护栏 | 对 CUDA 统一内存失明（10.7GB 只计 377MB） |
| `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1` | 已开 `--disable-prefill-cuda-graph` → 该开关直接无效 |
| 向现役 engine 发无界 `top_logprobs` | 上游 #40076 会让 scheduler 直接死。已核实本机暴露面低：DSH 侧仅 SDK 类型定义、switchyard 仅响应解析，均不作为请求参数下发 |
| `--mamba-radix-cache-strategy extra_buffer_lazy` | DFLASH 不支持（引擎断言），现役 `extra_buffer` 正确 |

---

## 5. 长上下文的真实瓶颈（定量，回答"能不能更快"）

- 社区同机同配置三点（100K/200K/300K）拟合：`t(N) = 5.0e-4·N + 3.78e-9·N²`
  → 466K 预测 17.6 min、**652K 预测 32.2 min，与本机实测 18/32 min 吻合到 2%**
  → 652K 时二次项占 83%（注意力项主导）
- 权重流量全程仅 ~5.2 s → **不是内存带宽问题**；理论 KV 带宽地板约 400 s、注意力算力地板 335–700 s ⇒ **2–5× 浪费在 kernel 效率**
- 结论：**加内存 / 降 mem-fraction / 换 chunk 都不会让 prefill 变快**。真杠杆只有三个：
  1. **KV 字节减半**（FP4，理论 1.7×）
  2. **前缀复用做满**（radix 命中即归零 —— 长文档场景固定前缀）
  3. **降 prompt 长度**（客户端侧，最有效）
- 另：单请求 70 万 token ≈ 20–30 分钟独占 GPU，是"卡"的直接来源；客户端护栏（`--max-queued-requests`、`--schedule-policy priority/lof`、`--enable-priority-scheduling`）本镜像全部支持但**均未启用**。

---

## 6. 风险清单（静默型，优先级高）

| 风险 | 状态 | 影响 |
|---|---|---|
| sglang #36548：DFlash2 高并发下**上一个用户消息被追加到错误上下文** | open（08-26） | ⚠️ 多并发 agent 场景最危险 —— 这也是**保持并发 4** 的额外理由 |
| sglang #38009：DFlash2 + thinking 时 greedy 在 token 30 与 target-only 分歧 | open（09-04，09-10 更新） | 输出质量静默偏移 |
| sglang #33454：verify 窗口越界 → CUDA illegal address | open，#33963 stale | 已缓解：现役 1,010,000 对 1,048,576 留了 3.7% 余量 ✅ |
| sglang #35496：DFlash2 反量化整个 NVFP4 lm_head → **硬重启整机** | **父 agent 已在本机镜像源码查实有防护** | ✅ `dflash_worker_v2.py:483-496` 有门控：量化 head 经 `should_apply_lm_head_quant_method` 准入才折进 draft CUDA graph，否则 `_eager("unsupported quantized lm_head")` 优雅降级 |
| 整机硬重启案例 | 已澄清为**硬件** | 恒温器缺陷（已 RMA）+ PowerStress 热-传感器失败，勿归因引擎 |
| 11 天运行期 | ✅ 真稳 | 容器 09-18 20:17 起 **零重启 / 零崩溃 / 零 OOMKilled / `Scheduler hit an exception`=0**；accept len p50 3.38（55,194 样本）；宿主 available 19 Gi、swap 618 Mi |
| watchdog 305 次探活不应答 | ⚠️ 非故障但体感差 | 长 prefill 期间 HTTP 事件循环迟滞（`/health` 6s+），自愈 0 次属预期策略；但说明"卡"是真实体感问题 |

> 未能核验：宿主内核 OOM 历史（`dmesg` 被权限拒绝）→ "无 OOM" 属间接证据。

---

## 7. 本地运维债务（审计发现，建议一并清）

1. **文档树分裂**：`projects/qwen38-sglang/` 是**旧版**，`dgx-spark-notes/qwen38-sglang/`（09-18）才是新版；两份 DFlash2 事故复盘只存在于 `dgx-spark-notes/notes/` ⇒ 只读 projects 会错过本机最重要的事故知识。9 个同名文件内容全不同。
2. **8 条数值矛盾**：mamba 96/48、并发 8/4、mem 0.85/0.70、镜像 tag、SGLang 版本、草稿算法、KV 池 1,031,239/1,205,656 —— 同一套数字散在 4 份文件里各写一遍。
3. **已过期结论仍在文档里**：`CUTOVER.md:54` / `DEPLOYMENT-RECORD.md §12` 仍称"YaRN 与 DSpark 不兼容必须切 MTP"（09-10 已推翻）；`notes/qwen38-27b-alternative-eval.md` 结论"保持 ornith 现状"（09-09 已切换）；`dgx-spark-model-service.md` §0 状态速览逐字重复两遍、§1-§7 仍有 7/9 篇幅描述已下线的 ornith。
4. **`vllm-open-proxy.py` 已停**（`<LAN-IP-1>:8888` → 000，proxy.log 停在 8/23）⇒ 外部 LAN/Tailscale 直连全失效，只剩 switchyard :4000；**五处文档仍写"运行中"**。需你拍板：重启代理，还是让 SGLang 直接绑 `0.0.0.0`。
5. **P11 实际未落地**：文档称"已改 `~/.dsh/settings.yaml`"，实测该文件不存在，只有 `settings.yaml.imported`（09-17，里面 `contextWindow: 1010000` 写对了）⇒ 需要恢复/合并这份配置。
6. `dgx-spark-notes` 是 git 仓库但 main 分支**零提交**（无版本历史）。

---

## 8. 建议执行顺序

**今天（零风险、可回滚）**
1. 修 `start-dspark-long.sh` 并发默认值 + 加参数联动断言（P0-A）
2. 主机加固：earlyoom + `OOMPolicy=stop` + `StartLimitBurst` + crash-dump 持久目录 + 健康检查换 canary/PSI（P0-B）
3. 决策 `vllm-open-proxy`（恢复外网直连 or 改绑 0.0.0.0）
4. 整理文档树（以 `dgx-spark-notes/` 为准）+ 恢复 `~/.dsh/settings.yaml`

**下次维护窗口（约 30–40 分钟）**
5. 草稿换 NVFP4 版（P1-A）→ A/B 验收：accept len（对 p50 3.38）、bench.py 吞吐、652K needle
6. mem-fraction 0.72→0.74→0.76 小步走（P1-B）
7. mamba 池 48→20/16 试回收 ~2.2GB

**实验驱动（先判别、后决定）**
8. FP4 KV（先 300K bf16-vs-fp8 判别）
9. chunked-prefill 2048 / depth 8→16 A/B
10. 客户端护栏（`--max-queued-requests` + 优先级调度）——需你确认策略
11. 恢复 8 并发（`cuda-graph-max-bs-decode 8` + 压测，同一变更单）

---

## 附：证据文件

| 文件 | 内容 |
|---|---|
| `A-upstream-engines.md` | 上游引擎/镜像/issue 状态，624 行，含 9-18 净 diff 与"未见更新"清单 |
| `B-community-recipes.md` | 社区方案与实测数据，353 行，20+ 来源，含"值不值得换"判断 |
| `C-longcontext-stability.md` | 长上下文性能建模与稳定性工程，408 行，31 URL，标注实测/文档/推断 |
| `D-local-audit.md` | 本机资产审计，334 行，含现役配置表、变体 farm 全表、19 条待办 |
| `FINAL-结论.md` | 本文件 |
