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

## 当前服务速览（记录时点 2026-09-10）

| 端口 | 服务 | 状态 |
|---|---|---|
| 4000 | switchyard（`0.0.0.0`，用户唯一入口，systemd user 服务） | active |
| 8888 | 容器 `qwen38-27b-sglang`（SGLang + Qwen3.8-27B-NVFP4 + DSpark） | Up |
| 8888 | [vllm-open-proxy.py](vllm-open-proxy.py) — 外部 IP（LAN/Tailscale）→ `127.0.0.1:8888` 纯 TCP 转发 | 运行中 |
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
│   └── ollama-uninstall-worklog.md    # Ollama 卸载调查（46GB 模型清理）
├── Ornith-1.5-35B-A3B-DGX-Spark/      # 旧主服务部署记录（vLLM + b12x 内核补丁）
│   ├── DEPLOYMENT.md                  # 部署全流程：前置条件/步骤/踩坑/运维
│   ├── start.sh / stop.sh / patches/  # 启动脚本与 GB10 b12x 修复补丁
│   └── README.md / LICENSE            # 上游 MiaAI-Lab 仓库 README（MIT）
├── qwen38-sglang/                     # 现主服务部署记录（SGLang + DSpark）
│   ├── DEPLOYMENT-RECORD.md           # ★ 首读：部署全流程（参数/执行/验证/运维/回滚）
│   ├── MIGRATION-NOTES.md             # 切换前调研 + 现状核查 + 修正备忘
│   ├── CUTOVER.md                     # 切换/回滚 runbook
│   └── start.sh / stop.sh / bench.py / routes.toml.next
└── scripts/                           # 判官相关脚本（start-judge-* / swap-ornith-util / test-judge-schema）
```

## 阅读建议

- **想快速上手本机服务**：[dgx-spark-model-service.md](dgx-spark-model-service.md)（服务与访问）→ [qwen38-sglang/DEPLOYMENT-RECORD.md](qwen38-sglang/DEPLOYMENT-RECORD.md)（现主服务全部参数）。
- **想在自己的 Spark 上复刻 Ornith 部署**：[Ornith.../DEPLOYMENT.md](Ornith-1.5-35B-A3B-DGX-Spark/DEPLOYMENT.md)（含前置条件清单与 3 行 start.sh 补丁）。
- **想做难度分流（本地判官）**：[LOCAL-JUDGE-LESSONS.md](notes/LOCAL-JUDGE-LESSONS.md) → [judge-pk-result.md](notes/judge-pk-result.md)（选型结论 + 阈值 + switchyard 接入预案）。
- **回滚/切换演练**：[CUTOVER.md](qwen38-sglang/CUTOVER.md)（任一步失败 <1 min 回滚）。

## 说明

- 敏感信息已脱敏：`qwen38-sglang/MIGRATION-NOTES.md` §1.5 中的 API key 值已替换为 `<已脱敏>`（原文在机器 ~/.config/systemd/user/switchyard.service 的环境变量里）。
- 已做公开化模糊处理：主机名 → `spark-XXXX`，局域网 IP → `<LAN-IP-1>`/`<LAN-IP-2>`（原为有线/无线各一个），Tailscale IP → `<Tailscale-IP>`，用户名 → `<user>`；`vllm-open-proxy.py` 中的 `LISTEN_IPS` 列表同步为占位符，实际部署时请改回本机 IP。
- `Ornith-1.5-35B-A3B-DGX-Spark/` 子目录内容源自上游 MIT 仓库（README/LICENSE 已保留）；其余文档为个人记录。
