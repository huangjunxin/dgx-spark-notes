# DGX Spark 玩机与部署记录

> 主机：`spark-XXXX` · NVIDIA DGX Spark（GB10 / SM121，128 GB 统一内存，aarch64，Ubuntu 24.04，内核 6.17-nvidia，CPU 20 核 = 10×Cortex-X5 + 10×A725）
> 本仓库记录 2026-08 ~ 2026-09 在这台机器上的模型服务部署方案、切换 runbook、评测与踩坑经验。
> 文中命令/路径为记录当时的实际环境（`/home/<user>/projects/...`），直接复用到其他机器需按注释调整。

---

## 时间线速览

| 日期 | 事件 | 文档 |
|---|---|---|
| 2026-08-22 | 按上游仓库部署 **Ornith-1.5-35B-A3B (NVFP4, vLLM, b12x 补丁)** 为主服务 :8888 | [Ornith.../DEPLOYMENT.md](Ornith-1.5-35B-A3B-DGX-Spark/DEPLOYMENT.md) |
| 2026-08-23 | 8888 经 TCP 代理对外放开（零停机）；开始探索本地判官方案（当日结论：显存不够，搁置） | [dgx-spark-model-service.md](dgx-spark-model-service.md) · [notes/LOCAL-JUDGE-LESSONS.md](notes/LOCAL-JUDGE-LESSONS.md) |
| 2026-08-24 | ornith 内存优化 util 0.85→0.65；判官 PK：**Qwen3-8B-AWQ 胜（85% 校准）**，:8889 上线 | [notes/ornith-memory-optimization.md](notes/ornith-memory-optimization.md) · [notes/judge-pk-result.md](notes/judge-pk-result.md) · [notes/minicpm-judge-trial.md](notes/minicpm-judge-trial.md) |
| 2026-08-25 | 评估 Qwen3.8-27B(SGLang) 能否替代 ornith：当时结论"保持现状"（后被 9/9 的实际切换推翻） | [notes/qwen38-27b-alternative-eval.md](notes/qwen38-27b-alternative-eval.md) |
| 2026-09-09 | **主服务切换到 Qwen3.8-27B（SGLang + DSpark 投机解码）**，用户侧仅 model 名变化 | [qwen38-sglang/DEPLOYMENT-RECORD.md](qwen38-sglang/DEPLOYMENT-RECORD.md) · [MIGRATION-NOTES.md](qwen38-sglang/MIGRATION-NOTES.md) · [CUTOVER.md](qwen38-sglang/CUTOVER.md) |
| 2026-09-10 | 判官 :8889 关停（内存压力 + 16 天无调用方） | [notes/LOCAL-JUDGE-LESSONS.md](notes/LOCAL-JUDGE-LESSONS.md) §7 |
| 2026-09-10 | **上下文 262K → 1M**：解决 YaRN 与投机解码草稿的参数污染冲突，上线 **DFlash2 + 1,010,000**（另备 DSpark + 506,000 路线）；含 11 条踩坑 | [notes/qwen38-long-context-1m.md](notes/qwen38-long-context-1m.md) · [qwen38-sglang/LONG-CONTEXT-CUTOVER.md](qwen38-sglang/LONG-CONTEXT-CUTOVER.md) · [RESEARCH](qwen38-sglang/LONG-CONTEXT-RESEARCH.md) |
| 2026-09-15 | **PaddleOCR-VL-1.6 本机可行性**（+ HunyuanOCR-1.5 对比）：官方 Blackwell Docker 为 x86_64 不可用；aarch64 走手动路线（PaddlePaddle CPU 版面 + vLLM 识别）；HunyuanOCR-1.5 单模型 + DFlash 与主服务同栈（许可地域受限） | [notes/paddleocr-vl-1.6-dgx-spark-feasibility.md](notes/paddleocr-vl-1.6-dgx-spark-feasibility.md) |
| 2026-09-15 | **⚠️ 事故：主服务静默停摆 66 分钟**（DFlash2 batch 尺寸 bug → scheduler 崩溃 → 容器 exit 0，当时 RestartPolicy=no 无自愈）；恢复后加装 **restart 策略 + watchdog 探活自愈 + 告警** 三重护栏；同日 12:40 **再次复发**（真实负载 25 分钟）→ 策略自动拉起，watchdog 冷启动误判已修 | [notes/qwen38-dflash2-crash-20260915.md](notes/qwen38-dflash2-crash-20260915.md) |
| 2026-09-15 | **★ GLM-OCR 本机部署方案（本机 OCR 首选）**：本机 vLLM 原生 glm_ocr+MTP；0.9B / 权重 MIT / SDK Apache-2.0 / OmniDocBench V1.5 #1；**版面经 SDK 的 torch 路径可上 GPU（不依赖 PaddlePaddle）**→ 比 PaddleOCR-VL-1.6 更适配 aarch64 | [notes/glm-ocr-dgx-spark-deployment.md](notes/glm-ocr-dgx-spark-deployment.md) |
| 2026-09-15 | **★ GLM-OCR 客户端接入（实测）**：SDK 服务 `0.0.0.0:5002` + 上传网关 `:5010`；客户端 HTTP 直连传 `data:image/*;base64`（单图 2.6–4.9 s）；**发现官方 SDK 客户端传 PDF 会静默返回空**（内置服务只认 `data:image`/`file://`）→ 网关补上 `/glmocr/parse`（官方 SDK 只改 `api_url` 即可传 PDF）；含调用链拆解（网络上只有一来一回）、一键启停脚本、8 条补充踩坑 | [notes/glm-ocr-client-integration.md](notes/glm-ocr-client-integration.md) · [部署文档 §12](notes/glm-ocr-dgx-spark-deployment.md) |
| 2026-09-15 | **MinerU2.5-Pro 本机可行性：不可用**。客户端装得干净、模型也能被本机 vLLM 原生加载（qwen2_vl，2.41 GiB），**但推理输出退化成连续 !**，解析产物为空；根因是 MinerU 要求 `transformers<5` + `vllm<0.22`，本机是 5.15.1/0.27.1 | [notes/mineru-2.5-pro-dgx-spark-feasibility.md](notes/mineru-2.5-pro-dgx-spark-feasibility.md) |
| 2026-09-17~18 | **⚠️ 事故（续）：同一 DFlash2 bug 再崩 3 次 → 源码级定案 + 修复**。读源码确认草稿采样器缓冲区尺寸 = decode CUDA graph 的 max bs（`--cuda-graph-max-bs-decode` = **4**），而并发上限是 **8** → **decode batch ≥5 必崩**（09-15 的 `a(4) vs b(7)` 与本轮 `a(4) vs b(5)` 是同一个等式；09-15 那份“缓冲区按上次 batch 复用”的解释已更正）。触发链是“长上下文风暴（单请求 70 万 token）→ 排队 → 并发顶到 5~6 → 崩”，**纯重启无效**（第 2 次崩溃距恢复仅 2 分 16 秒）。改为 **`--max-running-requests 4`**（+ mamba 池 96→48）后 8 并发压测通过：running 峰值恰好 4、8/8 全 200、异常 0 | [notes/qwen38-dflash2-crash-20260918.md](notes/qwen38-dflash2-crash-20260918.md) |

| 2026-09-15 | **★ 终端 agent 直接调用 GLM-OCR：`glm-ocr-remote` skill**（对标 [PaddleOCR-Skill](https://github.com/huangjunxin/PaddleOCR-Skill)）：装到 `~/.agents/skills`，图片/PDF 一条命令；纯标准库零依赖；内置"空结果守卫"专治 PDF 静默失败。含 `install.sh` + 实测记录，及 SKILL.md frontmatter 被 YAML 静默判死的坑 | [skills/glm-ocr-remote/](skills/glm-ocr-remote/) · [笔记 §12](notes/glm-ocr-client-integration.md) |
| 2026-10-29 | **告警降噪：Hermes 通道"沉默即健康"**。watchdog 头 45 天 1117 条告警中 **987 条是未达告警线的"恢复"虚惊播报**（长 prefill `/health` 偶发超时自愈），日均 ~10 条刷屏 Discord/hermes 上下文。修复：恢复播报加"真发过告警"门槛 + `STALL_ALERT=0` 降级 2min 级 WARN；`dgx-health-monitor` 同步修 hung task 重复告警与断网每小时催办。验证、踩坑（bash 增量读脚本/`exec 2>/dev/null` 吞 trace/ACL mask 被压/`systemctl --user` 每用户隔离/测试桩契约）见笔记 | [notes/qwen38-watchdog-denoise-20261029.md](notes/qwen38-watchdog-denoise-20261029.md) |
| 2026-10-29 | **Qwen3.8 社区调研（纯调研零变更）**：上游三条线核查（主模型/草稿权重、架构、运行时镜像）**均无更新**，现役仍 pin 在最新；社区横评结论——现役路线（SGLang+NVFP4+DFlash2+YaRN 1M）就是 GB10 主流最优，hasso5703 ★242 与本机同草稿同上下文（其 71.4 tok/s 为 code 贪心口径，勿与 bench.py 27.0 直接比）。整理 6 项可 cherry-pick 改进（Docker 内存帽 / draft D=16 / 镜像升 v0.5.22 / 0.76 / sleep-on-idle / template 补丁），**决议：暂不更新**，留 5 条重开触发条件 | [notes/qwen38-community-survey-20261029.md](notes/qwen38-community-survey-20261029.md) |

## 当前服务速览（记录时点 2026-09-15）

| 端口 | 服务 | 状态 |
|---|---|---|
| 4000 | switchyard（`0.0.0.0`，用户唯一入口，systemd user 服务） | active |
| 8888 | 容器 `qwen38-27b-sglang`（SGLang + Qwen3.8-27B-NVFP4 + DFlash2） | Up（09-15 11:14 崩溃 → 12:20 重启恢复；`--restart unless-stopped`） |
| - | 守护 [qwen38-watchdog.sh](qwen38-sglang/qwen38-watchdog.sh)（systemd user 服务 `qwen38-watchdog`）— 探活 / 抓现场 / 自愈 / 告警 | active |
| 8888 | [vllm-open-proxy.py](vllm-open-proxy.py) — 外部 IP（LAN/Tailscale）→ `127.0.0.1:8888` 纯 TCP 转发 | 运行中 |
| 5002 / 5010 | **GLM-OCR**：SDK HTTP 服务 + 上传网关（按需启停，`glm-ocr-test/{start,stop}-service.sh`） | 已停（用则起，全栈约 10 GB 统一内存） |
| 8889 | Qwen3-8B-AWQ 评测判官（venv-vllm） | 已停（可重启，脚本在 [scripts/](scripts/)） |
| - | 旧容器 `ornith-b12x-serve`（vLLM / Ornith） | Exited，保留待回滚 |

用户 API：主入口 `http://<host>:4000/v1`（OpenAI 兼容；model：`qwen3.8-27b` / `ornith-local`（灰度）/ `cloud-test` / `switchyard` 智能路由）。

## 目录结构

```
dgx-spark-notes/
├── README.md                          # 本文件：总览 + 时间线
├── dgx-spark-model-service.md         # 模型服务总览（服务/端口/对外暴露/访问方式）
├── vllm-open-proxy.py                 # 外部 IP → 127.0.0.1:8888 TCP 转发代理（纯标准库）
├── notes/                             # 📄 玩机经验 / 评测 / worklog
│   ├── LOCAL-JUDGE-LESSONS.md         # 本地判官方案完整探索（vLLM 生态/显存/沙箱教训）
│   ├── minicpm-judge-trial.md         # MiniCPM5-1B 判官试跑（格式全过、判断力 65%）
│   ├── judge-pk-result.md             # 判官 PK：MiniCPM5-1B vs Qwen3-8B-AWQ（85% 胜）
│   ├── ornith-memory-optimization.md  # ornith util 0.85→0.65 内存优化
│   ├── qwen38-27b-alternative-eval.md # Qwen3.8-27B(SGLang) 替代评估（2026-08-25）
│   ├── qwen38-long-context-1m.md      # ★ 长上下文 262K->1M 完整记录 + 11 条踩坑（2026-09-10）
│   ├── paddleocr-vl-1.6-dgx-spark-feasibility.md # PaddleOCR-VL-1.6 本机可行性 + HunyuanOCR-1.5 对比（2026-09-15）
│   ├── glm-ocr-dgx-spark-deployment.md # ★ 本机 OCR 首选：GLM-OCR 部署方案（vLLM+MTP / SDK pipeline / 服务化）
│   ├── glm-ocr-client-integration.md  # ★ GLM-OCR 客户端接入：HTTP 契约 / SDK MaaS 模式 / 上传网关（2026-09-15 实测）
│   ├── mineru-2.5-pro-dgx-spark-feasibility.md # MinerU2.5-Pro 可行性：当前栈不可用（transformers<5 / vllm<0.22 闸门）
│   ├── qwen38-dflash2-crash-20260915.md # ★ 事故复盘：主服务静默停摆 66 分钟 + 三重护栏（2026-09-15）
│   ├── qwen38-dflash2-crash-20260918.md # ★ 事故（续）：同 bug 复发 3 次 + 源码级定案 + 并发上限 4 修复（2026-09-17~18）
│   └── ollama-uninstall-worklog.md    # Ollama 卸载调查（46GB 模型清理）
├── Ornith-1.5-35B-A3B-DGX-Spark/      # 旧主服务部署记录（vLLM + b12x 内核补丁）
│   ├── DEPLOYMENT.md                  # 部署全流程：前置条件/步骤/踩坑/运维
│   ├── start.sh / stop.sh / patches/  # 启动脚本与 GB10 b12x 修复补丁
│   └── README.md / LICENSE            # 上游 MiaAI-Lab 仓库 README（MIT）
├── qwen38-sglang/                     # 现主服务部署记录（SGLang + DSpark）
│   ├── DEPLOYMENT-RECORD.md           # ★ 首读：部署全流程（参数/执行/验证/运维/回滚）
│   ├── MIGRATION-NOTES.md             # 切换前调研 + 现状核查 + 修正备忘
│   ├── CUTOVER.md                     # 切换/回滚 runbook
│   ├── LONG-CONTEXT-CUTOVER.md        # ★ 长上下文切换记录（262144 -> 1010000，含三臂实测对比）
│   ├── LONG-CONTEXT-RESEARCH.md       # 长上下文调研（官方配方/社区方案/逐条裁决，12 节）
│   ├── start-long.sh                  # 长上下文启动脚本 A：YaRN 覆盖 + 原生 MTP
│   ├── start-dspark-long.sh           # 长上下文启动脚本 B：三模式（draftmaxpos/draftyarn/yarn）+ ALGO=DFLASH 支线
│   ├── make-yarn-variant.py           # 生成 target/draft 软链变体（内联 YaRN / 草稿 rope overlay，零拷贝）
│   ├── bench-long-ctx.py              # 大海捞针验证（真实 prompt_tokens / TTFT / 命中）
│   ├── verify-*.py                    # 零停机预检（复现参数污染崩溃 + 各变体能否加载）
│   ├── qwen38-watchdog.sh             # ★ 探活/抓现场/自愈/告警守护（systemd user 服务 qwen38-watchdog）
│   ├── qwen38-watchdog.service        # 守护的 systemd user 单元（安装说明见文件头）
│   ├── qwen38-watchdog.env.example    # 守护配置样例（探活间隔 / 自愈限流 / 告警通道）
│   ├── recreate-concurrency4-20260918.sh # ★ 并发上限修复：从 docker inspect 导出重建，只改 2 个值（含根因与回滚）
│   └── start.sh / stop.sh / bench.py / routes.toml.next
├── skills/                            # 可安装的 agent skill（对标 PaddleOCR-Skill）
│   ├── install.sh                     # 安装到 ~/.agents/skills（重复安装保留 config.env）+ Claude Code 软链
│   └── glm-ocr-remote/                # ★ GLM-OCR skill：图片/PDF → 文字与 Markdown
│       ├── SKILL.md                   # 触发词 / 用法 / 故障排查
│       ├── config.env.example         # 服务地址模板（--server > GLMOCR_URL > config.env > 默认）
│       └── scripts/{ocr.py,check_server.py,_common.py}   # 纯标准库，零依赖
├── experiments/                       # 🧪 本机可行性实验记录（脚本 + 产物 + 日志，可复跑）
│   ├── glm-ocr-test/                  # GLM-OCR 客户端接入实测（HTTP/SDK/网关，2026-09-15）
│   ├── mineru-test/                   # MinerU2.5-Pro 可行性实测（结论：当前栈不可用）
│   └── research-qwen38-20260930/      # Qwen3.8 引擎/长上下文稳定性调研（A~D + FINAL 结论）
├── archive/                           # 🗄️ 历史快照（不入 git，见 archive/README.md）
│   └── 2026-09-snapshots/             # projects 根旧副本 / dist 打包产物 / tarball 快照（2026-10-29 收拢）
└── scripts/                           # 判官相关脚本（start-judge-* / swap-ornith-util / test-judge-schema）
```

## 阅读建议

- **想快速上手本机服务**：[dgx-spark-model-service.md](dgx-spark-model-service.md)（服务与访问）→ [qwen38-sglang/DEPLOYMENT-RECORD.md](qwen38-sglang/DEPLOYMENT-RECORD.md)（现主服务全部参数）。
- **想在自己的 Spark 上复刻 Ornith 部署**：[Ornith.../DEPLOYMENT.md](Ornith-1.5-35B-A3B-DGX-Spark/DEPLOYMENT.md)（含前置条件清单与 3 行 start.sh 补丁）。
- **想做难度分流（本地判官）**：[LOCAL-JUDGE-LESSONS.md](notes/LOCAL-JUDGE-LESSONS.md) → [judge-pk-result.md](notes/judge-pk-result.md)（选型结论 + 阈值 + switchyard 接入预案）。
- **回滚/切换演练**：[CUTOVER.md](qwen38-sglang/CUTOVER.md)（任一步失败 <1 min 回滚）。
- **运维/自愈**：[notes/qwen38-dflash2-crash-20260915.md](notes/qwen38-dflash2-crash-20260915.md)（事故复盘 + watchdog/告警的安装与演练命令）→ [notes/qwen38-dflash2-crash-20260918.md](notes/qwen38-dflash2-crash-20260918.md)（**同一 bug 的源码级定案 + 并发上限 4 修复与压测**，先看这份的 TL;DR）。
- **想让别的机器调用本机 OCR**：[notes/glm-ocr-client-integration.md](notes/glm-ocr-client-integration.md)（调用链拆解 / 三种接法 / PDF 的坑 / 一键启停）。
- **想在终端里让 agent 直接 OCR**：`bash skills/install.sh` → [notes/glm-ocr-client-integration.md](notes/glm-ocr-client-integration.md) §12（skill 设计与实测）。

## 说明

- 敏感信息已脱敏：`qwen38-sglang/MIGRATION-NOTES.md` §1.5 中的 API key 值已替换为 `<已脱敏>`（原文在机器 ~/.config/systemd/user/switchyard.service 的环境变量里）。
- 已做公开化模糊处理：主机名 → `spark-XXXX`，局域网 IP → `<LAN-IP-1>`/`<LAN-IP-2>`（原为有线/无线各一个），Tailscale IP → `<Tailscale-IP>`，用户名 → `<user>`；`vllm-open-proxy.py` 中的 `LISTEN_IPS` 列表同步为占位符，实际部署时请改回本机 IP。
- `Ornith-1.5-35B-A3B-DGX-Spark/` 子目录内容源自上游 MIT 仓库（README/LICENSE 已保留）；其余文档为个人记录。
