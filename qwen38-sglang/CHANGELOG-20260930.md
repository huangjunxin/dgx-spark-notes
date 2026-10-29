# 变更记录：部署脚本加固 + 草稿换代准备（2026-09-30）

> 依据：`../research-qwen38-20260930/`（上游/社区/长上下文与本机审计四条线）
> 原则：**本轮只改了"下次重建时才会生效"的脚本与新增文件，没有触碰正在运行的容器**。

## 0. 最重要的一句

**本轮已执行草稿换代：现役容器现在是 `DRAFT_FLAVOR=nvfp4`**（2026-09-30 22:09 切换，冷启动 6m55s，全部门槛通过）。
稳态吞吐 bench.py **17.6 → 27.0 tok/s（+53%）**、宿主可用内存 +10 GiB、`Scheduler hit an exception`=0。
完整对比与回滚见 `ab-records/AB-CONCLUSION-nvfp4-draft.md`。

脚本侧同时完成加固：本轮的改动与换档前配置**逐参数等价**（见 §3 验证），唯一新增是 `--crash-dump-folder`。
也就是说：**重建不会再退回 8 并发的必崩配置**，而是自动带上修复与护栏。

## 1. `start-dspark-long.sh` 的改动

| # | 改动 | 为什么 |
|---|---|---|
| 1 | `MAX_CONCURRENT_REQUESTS` 默认 **8 → 4** | 旧默认值与 `--cuda-graph-max-bs-decode 4` 错配 = 09-15/17/18 崩 5 次的那套配置 |
| 2 | 新增**参数联动断言**：`ALGO=DFLASH` 时并发必须 ≤ decode graph bs，否则拒绝启动并打印两种正确做法 | 从源头消除"两个参数各写各的"（09-18 复盘 §10 最后一条待办） |
| 3 | `CUDA_GRAPH_MAX_BS_DECODE` / `TORCH_COMPILE_MAX_BS` 提为变量（原为硬编码 4） | 恢复高并发时必须两个值一起改，现在可显式指定且有断言 |
| 4 | 新增 `ATTENTION_BACKEND`（默认 flashinfer）+ **fp4 KV 必须配 triton 的断言** | 调研结论：`fp4_mx_block16` 只有 triton 能读、`nvfp4` 连 triton 都读不了；避免照着旧注释试出崩溃 |
| 5 | 新增 `DRAFT_FLAVOR=zlab\|nvfp4` | 草稿换代的 A/B 开关（见 §2），一条环境变量切换 |
| 6 | 新增 `--crash-dump-folder`（落到已挂载的 `qwen38-model-cache/crash-dumps/`） | 09-18 那次崩溃的 traceback 随 `docker rm -f` 一起丢了 |
| 7 | 补上 `--restart unless-stopped` | 之前只有 notes 版有；scheduler 异常时容器以 exit 0 结束，`on-failure` 不会触发（曾静默停摆 66 分钟） |
| 8 | 新增 `CHECK=1` 预检模式 + `CHECK_ARGS_OUT` | 重建前先看完整命令行，且能与 `docker inspect .Args` 逐条 diff；重建不再"盲开" |
| 9 | mamba 池断言：`并发×槽位 ≥ 并发` | 防止有人只改并发不改池子 |

**未改**（刻意）：mem-fraction 0.70、KV dtype fp8_e4m3、chunked-prefill 8192、`--max-mamba-cache-size 48`、
`extra_buffer` 策略、`--disable-flashinfer-autotune`。这些都在 11 天稳定期内验证过，要动应单独做 A/B。

## 2. 草稿换代（P1，已执行完成）

- 已下载 `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@bd7a934` → `qwen38-model-cache/dflash2-nvfp4-draft`（1.5GB，原版 3.6GB）
- 已扩展 `make-yarn-variant.py`：新增 `--dflash2-dir` / `--dflash2-prefix`（原先把草稿目录硬编码）
- 已生成两个变体（与 zlab 变体并存，便于回滚）：
  - `variants/qwen38-dflash2nvfp4-yarn1024`（max_pos 1048576 + YaRN f4.0，NVFP4 量化配置保留）
  - `variants/qwen38-dflash2nvfp4-yarn512`（f2.0）
- 新增 `ab-collect.sh`：单卡无法并存两套配置，所以每个档位跑完立刻固化快照

## 3. 验证证据（都是本轮实际执行的）

| 验证 | 结果 |
|---|---|
| `bash -n` 语法 | ✅ 两棵树均通过 |
| 预检 vs 现役容器**逐参数 diff** | ✅ **只差 `--crash-dump-folder` 一项**（有意新增），其余 60 项完全一致 |
| 负向断言：`MAX_CONCURRENT_REQUESTS=8` | ✅ 拒绝启动，打印崩溃机理与两种正确做法 |
| 负向断言：`KV_DTYPE=fp4_mx_block16` + flashinfer | ✅ 拒绝启动，提示需 `ATTENTION_BACKEND=triton` |
| 负向断言：非法 `DRAFT_FLAVOR` | ✅ 拒绝启动 |
| 新草稿变体生成 | ✅ YaRN 已烧入、`quant_algo=NVFP4` 保留、软链为相对路径（零拷贝） |
| 默认路径回归（`--dflash2` 不带新参数） | ✅ 仍指向 `dflash2-draft`，行为不变 |
| `ab-collect.sh baseline` | ✅ 复现审计数字：accept len p50 **3.38** / rate p50 **0.34** / KV 池 1,205,656 / 0 崩溃 |
| `canary-check.sh` | ✅ 真实生成 488ms 返回健康（退出码 0） |
| `restart-storm-guard.sh` | ✅ 正常态 0；模拟 3 次重启后正确报 STORM（退出码 1） |
| 两棵文档树同步 | ✅ 差异仅脱敏路径（`/home/<user>` → `/home/<user>`） |

## 4. 已执行的维护窗口操作（2026-09-30 22:09–22:23）

**草稿换代 zlab → nvfp4：✅ 成功，已保留**（一条命令完成：停+删 → 冷启动 6m55s → 自动采集）

```bash
cd ~/projects/qwen38-sglang
./switch-draft-flavor.sh nvfp4        # 已执行; BENCH=1 可顺带跑吞吐
```

| 结果 | 数值 |
|---|---|
| bench.py 稳态吞吐 | 17.6 → **27.0 tok/s（+53%）** |
| 400-token 墙钟 | 21.0–23.6s → **13.5–16.6s（−33%）** |
| KV 池 / 宿主可用内存 | 1,205,656 → 1,236,510 token ／ 19 → **29 GiB** |
| accept len 中位数 | 3.40 → **2.08**（口径不可比，待长期观察） |
| 硬门槛 | `Scheduler hit an exception`=0、RestartCount=0、4 并发压测全 200、300K 长文召回命中 |

详见 `ab-records/AB-CONCLUSION-nvfp4-draft.md`。**回滚**：`./switch-draft-flavor.sh zlab`（权重与变体都在原位，零下载）。

**主机加固（仍需你手动，沙箱内 sudo 被禁）**
```bash
cd ~/projects/qwen38-sglang/host-hardening
./install.sh                 # dry-run
./install.sh --apply --install-earlyoom
```

## 5. 本轮**没有**做的事（及原因）

| 项 | 原因 |
|---|---|
| 改 mem-fraction 0.70→0.76 | 触碰稳定性边界，需逐档冷启观察"capture 后空闲 ≥15GiB"，属维护窗口动作 |
| 改 KV dtype / chunked-prefill / draft depth / mamba 池 | 都需先做判别实验或 A/B，见调研结论 §3 |
| 改 18KB 的现场 watchdog 脚本 | 有行为风险，应单独做并演练（`host-hardening/README.md` 给了接法） |
| 恢复 8 并发 | 需要 `CUDA_GRAPH_MAX_BS_DECODE=8` + 并发压测同一变更单；脚本现在会拒绝错配配置 |
| 清理 4 份过期文档的数值矛盾 | 属编辑工作，已列在审计报告里；本轮只加了本变更记录，未静默重写历史文档 |
| 恢复 `vllm-open-proxy.py` | 取决于你要"重启代理"还是"SGLang 直接绑 0.0.0.0"，需你拍板 |
