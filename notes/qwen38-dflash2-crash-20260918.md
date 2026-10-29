# 事故复盘（续）：DFlash2 batch 尺寸 bug 定案 —— 并发 8 与 decode CUDA graph 的 bs=4 不匹配

> 日期: 2026-09-17 ~ 2026-09-18 · 主机: spark-XXXX（DGX Spark GB10）· 容器: `qwen38-27b-sglang`
> 镜像: `lmsysorg/sglang:dev-qwen38-27b-dflash2`（commit `5f55db35`）
> 影响: **三次崩溃**（09-17 15:14:11 / 09-17 15:24:18 / 09-18 约 16:01），每次 ≈7 分钟中断；09-17 14:43–15:15 期间服务严重卡顿（非崩溃）
> 状态: ✅ **已定案并修复** —— `--max-running-requests 8 → 4`（与 `--cuda-graph-max-bs-decode 4` 对齐），8 并发压测通过
> 前次复盘: [qwen38-dflash2-crash-20260915.md](qwen38-dflash2-crash-20260915.md)（2026-09-15 两次崩溃，同一 bug）

---

## 0. TL;DR

- 09-15 那两次崩溃的**真正机理这次读源码定案了**：DFlash2 草稿采样器的缓冲区尺寸取自 **decode CUDA graph 的最大 batch**
  （`max_bs = max(get_exec().graph.cuda_graph_config.decode.bs)` = `--cuda-graph-max-bs-decode` = **4**），
  而调度器允许 `--max-running-requests`（当时 **8**）个请求同时在跑。**decode batch ≥5 时必然越界崩溃**，与"batch 尺寸抖动""buffer 复用"无关。
- 于是 09-15 的 `a(4) vs b(7)`、本次的 `a(4) vs b(5)` 是**同一个等式**：`greedy_mask[:bs].copy_(src)` 里 **a = 缓冲区行数 4、b = bs**。
- 触发条件：**长上下文 + 多并发混合负载**。09-17 是一波 70 万~90 万 token 的超长请求（"风暴"）把并发顶到 6，09-18 是并发到 5。
- 关键教训：**纯重启治不了**——09-17 两次崩溃后各自 7 分钟内就把状态清零重来，第二次崩溃只隔了 2 分 16 秒。
- 修复（方案 1）：把并发收到 4，与图形捕获的 bs=4 对齐，让 bs≥5 不可能出现；`--max-mamba-cache-size` 按项目配方 96 → 48（并发 × 12）。
- 验证：**8 个并发请求 → running 峰值恰好 4、queue 峰值 4、8/8 全部 200**，`Scheduler hit an exception` = 0，容器全程未重启。

## 1. 时间线（HKT）

| 时刻 | 事件 |
|---|---|
| 09-17 ~14:43 | **"卡"的开始**：客户端开始连续发 25 万~90 万 token 的超长请求（`#pending-token` 峰值 735,933）；prefill 速度被压到 460–1000 tok/s，队列 2–4，普通请求 TTFT 涨到几十分钟 |
| 09-17 14:43–15:15 | 探活抖动：`/health` 反复 6s 超时（长 prefill 期间 HTTP 事件循环迟滞，**非故障**，watchdog 判为"繁忙"未动手，只留下多条 `[OK] 恢复应答` 记录） |
| 09-17 15:14:10 | `#running-req: 6`；`dflash_worker_v2.py:226` 报 `shape [4] ... required [5]` |
| **09-17 15:14:11** | **第 1 次崩溃**：`Scheduler hit an exception` → `greedy_mask[:bs].copy_` → `RuntimeError: The size of tensor a (4) must match the size of tensor b (5)` → SIGQUIT → `kill_process_tree` → 容器 **exit 0** |
| 09-17 15:15:33 | `unless-stopped` 策略**自动拉起**（`RestartCount=1`） |
| 09-17 15:22:02 | 第 1 次冷启动完成（6 分 29 秒），恢复服务 |
| 09-17 15:24:18 | **第 2 次崩溃**（同一 traceback；崩溃前 `#running-req: 4`，新请求加入后 decode batch = 5）→ **恢复后仅 2 分 16 秒** |
| 09-17 15:24:23 | 策略自动拉起（`RestartCount=2`），冷启动进行中 |
| 09-17 15:30:15 | 人工 `docker restart`（用户要求"重启一下"，此操作把 `RestartCount` 归零）——**注意：纯重启不改变任何参数，属无效动作** |
| 09-17 15:36:52 | 冷启动完成，`/health` 200 |
| 09-17 15:44:20 | 一次探活 6s 超时（繁忙），随即恢复；`Scheduler hit an exception` 计数仍为 0（= 不是崩溃） |
| 09-18 12:04–13:09 | 数次探活失败后自行恢复（watchdog 记录，未自愈） |
| **09-18 ~16:01:4x** | **第 3 次崩溃** → 16:01:49 策略自动拉起（`RestartCount=1`，`StartedAt 08:01:49Z`）。⚠️ watchdog 告警日志**没有**这次变化的条目，见 §10 待办 |
| 09-18 16:15:52 | **应用修复**：`docker rm -f` + 按修复参数重建容器（方案 1） |
| 09-18 16:23:17 | 冷启动完成（7 分 25 秒），`/health` 200 |
| 09-18 16:23–16:24 | 6 并发 + 8 并发压测通过（§5）；watchdog 判定 `healthy` |

## 2. 定案的根因（源码级）

镜像内 `/sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py`：

```python
# :503 / :532  构造采样器 —— 缓冲区尺寸取自 decode CUDA graph 的最大 batch
return _DflashDraftSampler(
    draft_model=self.draft_model,
    block_size=self.block_size,
    max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),   # ← 就是 --cuda-graph-max-bs-decode（=4）
    device=self.device,
)

# :206  按 max_bs 分配，只有 max_bs 行
self.temperatures = torch.ones((max_bs,), dtype=torch.float32, device=device)
self.greedy_mask  = torch.ones((max_bs,), dtype=torch.bool,    device=device)

# :218 / :231  stage_sampling_params(bs=...) —— bs 来自本轮 decode batch（= 同时在跑的请求数）
torch.clamp(..., out=self.temperatures[:bs])          # 先出现 "shape [4] ... required [5]" 警告
self.greedy_mask[:bs].copy_(resolve_greedy_mask(bs=bs, ...))   # bs > max_bs → RuntimeError
```

崩溃现场（09-17 15:14:11，与 09-15 完全同型）：

```
dflash_worker_v2.py:226: UserWarning: An output with one or more elements was resized since it had shape [4],
                         which does not match the required output shape [5]. ...
[2026-09-17 07:14:11] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/speculative/dflash_worker_v2.py", line 1898, in forward_batch_generation
    self._draft_sampler.stage_sampling_params(
  File ".../sglang/srt/speculative/dflash_worker_v2.py", line 231, in stage_sampling_params
    self.greedy_mask[:bs].copy_(
RuntimeError: The size of tensor a (4) must match the size of tensor b (5) at non-singleton dimension 0
```

**统一解释三次（乃至 09-15 两次）崩溃**：`copy_` 报错里 **a 是目标（缓冲区行数 = max_bs = 4，启动即固定），b 是源（= 本轮 bs）**。

| 事故 | 报错 | 含义 | 崩溃瞬间并发 |
|---|---|---|---|
| 09-15 11:14 | `a(4) vs b(7)` | bs=7 > 4 | `#running-req: 5`（同一批次后续到 7） |
| 09-17 15:14 | `a(4) vs b(5)` | bs=5 > 4 | `#running-req: 6` |
| 09-17 15:24 | `a(4) vs b(5)` | bs=5 > 4 | `#running-req: 4` + 新请求加入 |
| 09-18 ~16:01 | （日志随容器删除丢失） | 推断同为 bs>4 | — |

> 🔧 **对 09-15 复盘的一处更正**：那份笔记把机理写成"缓冲区按上一次 batch 的尺寸（7）resize/复用"。
> 读到源码后确认：**缓冲区尺寸在启动时就固定为 decode CUDA graph 的 max bs（4），与历史 batch 无关**。
> 这解释了为什么"同一 bug 反复发作"且"重启后几分钟就复发"—— 只要并发再上 5，它必崩。

## 3. 触发条件：为什么是"先卡后崩"

- 09-17 14:43 起有客户端连续发 **25 万~90 万 token** 的超长请求（`#pending-token` 峰值 735,933，单请求吃掉 KV 池近 80%）；
- 单个这样的 prefill 要 **20–30 分钟**（prefill 实测 460–1000 tok/s），期间队列 2–4，**普通请求被堵在后面** → 用户体感"非常卡"；
- 排队 + 超时重试把**同时在跑的请求数**推高到 5、6 → 命中 `bs > 4` → 崩。
- 也就是说：**"卡"和"崩"是同一条因果链上的两环**，不是两件事。

## 4. 修复（方案 1：把并发收到 decode graph 的 bs）

| 参数 | 旧 | 新 | 说明 |
|---|---|---|---|
| `--max-running-requests` | 8 | **4** | 与 `--cuda-graph-max-bs-decode 4` 对齐 → `bs ≥ 5` 不可能出现 |
| `--max-mamba-cache-size` | 96 | **48** | 项目脚本配方：并发 × 12（DFLASH 用 `extra_buffer`） |

其余**全部不动**：镜像、三个挂载、env、`--memory/--memory-swap 100g`、`--cpuset-cpus 5-9,15-19`、`--ipc host`、host 网络、`--restart unless-stopped`、
`--context-length 1010000`、`--mem-fraction-static 0.70`、`--disable-flashinfer-autotune`、`--kv-cache-dtype fp8_e4m3`、DFLASH 各项、`--torch-compile-max-bs 4`、`--cuda-graph-max-bs-decode 4`。

**执行方式（重建容器才生效；`docker restart` 不行）**：从 `docker inspect` 原样导出 `docker run`（保证镜像/挂载/环境/上限一字不差），只改上表两个值。

```bash
cd /home/<user>/projects/qwen38-sglang
docker rm -f qwen38-27b-sglang
bash recreate-concurrency4-20260918.sh          # 见 §8 复现材料
# 等待冷启动（本机 ≈7.5 分钟）：curl -fsS http://127.0.0.1:8888/health
```

> 为什么不用"把 `--cuda-graph-max-bs-decode` 提到 8"这条对称解法（备选 2）？
> 它同样能修（缓冲区随之变 8 行），但 **bs 5–8 的 decode 路径在本机从未验证过**（现有 2 天稳定期全在 bs≤4 的图捕获路径上）。
> 保守起见先选"不进入未验证路径"的方案 1；想恢复 8 并发时，在维护窗口按备选 2 重建并压测。

## 5. 验证（修复后，2026-09-18 16:23–16:24）

| 测试 | 结果 |
|---|---|
| 冷启动 | 16:15:52 重建 → 16:23:17 `/health` 200（**7 分 25 秒**） |
| 6 并发短请求 | 6/6 `HTTP 200`（1.5–3.6 s），`Scheduler hit an exception` = 0 |
| **8 并发请求**（旧配置下必崩的场景） | **`running` 峰值恰好 4、`queue` 峰值 4**，8/8 `HTTP 200`（8.5–18 s，排在后面的等前 4 个让位） |
| 压测后状态 | 异常计数 0、`RestartCount=0`、`StartedAt` 未变、`/health` 200、watchdog 判定 `healthy` |

原始返回：`<user>/projects/qwen38-sglang/.concurrency-test/stress1..8.json`。

**补充观察（修复后 6 分钟，2026-09-18 16:29–16:32）**：同一类超长请求又来了（`#pending-token` 从 71,865 递减，prefill ≈440–520 tok/s，KV 用到 0.21）。
表现与修复前一致的是"繁忙"现象：`/health` 探活 **10s 超时**、引擎日志出现 `Health check failed. Server couldn't get a response from detokenizer for last 20 seconds`、部分客户端拿到错误响应（`create_error_response`）。
**不同的是结果**：`Scheduler hit an exception` 计数 **0**、容器 `RestartCount=0`、探针随后仍返回 200 —— 即 **长上下文负载本身造成的"卡"依旧存在（这是另一件事，见 §9.5 与 §10），但它不再把服务打崩**。

**副作用（已知、可接受）**：并发上限 4，第 5 个起排队（FCFS，实测排队 9–17 秒返回）。
好处是所有 decode 批次都走"已捕获 CUDA graph（bs≤4）"这条最熟的路径，不再有 eager 混跑。

## 6. 回滚

```bash
# 方式一：把并发改回 8（编辑重建脚本里两个值后重跑）
cd /home/<user>/projects/qwen38-sglang
sed -i 's/"--max-running-requests" "4"/"--max-running-requests" "8"/; s/"--max-mamba-cache-size" "48"/"--max-mamba-cache-size" "96"/' recreate-concurrency4-20260918.sh
docker rm -f qwen38-27b-sglang && bash recreate-concurrency4-20260918.sh

# 方式二：按崩溃前的完整定义重建（回滚依据，未脱敏原件在本机）
#   /home/<user>/projects/qwen38-sglang/container-inspect-20260917-pre-fix.json
```

> 注意：回滚 = **把已知会崩的配置装回去**。要恢复高并发，建议走备选 2（`--cuda-graph-max-bs-decode 8`）而不是直接回到 8/4 的错配。

## 7. 未采纳的备选（备查）

| 方案 | 做法 | 不采纳的原因 |
|---|---|---|
| 备选 2 | `--cuda-graph-max-bs-decode 4 → 8`（保留 8 并发） | 缓冲区随之为 8 行能修，但 bs 5–8 路径本机从未验证；留给维护窗口 + 压测 |
| 备选 3 | 关投机解码（`--speculative-algorithm NONE`） | 最稳但吞吐损失大（DFlash2 accept len 实测 1.4–3.8），且放弃本机已有调优 |
| 备选 4 | 给容器内 Python 打补丁（把 `max_bs` 直接放大） | 改动落在容器层，重建即丢；且绕过上游语义，不适合长期形态 |
| 备选 5 | 回退 DSpark 506K / MTP 262K（`LONG-CONTEXT-CUTOVER.md` §8） | 代价是上下文规格下降，先不动（本次根因已明确且修复代价极小） |

## 8. 复现材料（留档）

| 文件 | 说明 |
|---|---|
| `qwen38-sglang/recreate-concurrency4-20260918.sh`（本仓库，脱敏副本） | 修复用的重建脚本：从 inspect 导出 + 只改两个值，头部写明根因与回滚 |
| `<user>/projects/qwen38-sglang/container-inspect-20260917-pre-fix.json`（本机） | 修复前完整容器定义（回滚依据） |
| `<user>/projects/qwen38-sglang/recreate-concurrency4-20260918.sh`（本机） | 实际执行的脚本（路径未脱敏，可直接跑） |
| `<user>/projects/qwen38-sglang/.concurrency-test/stress1..8.json` | 8 并发压测原始返回 |
| `~/.local/state/qwen38-watchdog/alerts.log` | watchdog 侧时间线（探活抖动、策略拉起、恢复应答） |

## 9. 运维要点（本次新增认知）

1. **重建/重启的语义要分清**：`docker restart` 只是重启（参数不变，本次证明**治不了这个 bug**）；**改参数必须重建**（`docker rm -f` + `docker run`）。
2. **重建期间不必冻结 watchdog**：冷启动宽限（`STARTUP_GRACE` 1200s）已覆盖，实测全程判为 `warming` 未插手。
   （想更稳可先 `--maintenance on`，但要能写 `~/.local/state/qwen38-watchdog/`——DSH 沙箱只允许写工作区时会失败，本次即如此，未影响结果。）
3. **`/health` 6s 超时 ≠ 挂了**：长上下文 prefill 期间 HTTP 事件循环迟滞是常态，与 09-15 复盘 §9 的"繁忙误判"同源。看 `Scheduler hit an exception` 计数与 `RestartCount` 才能区分"忙"和"崩"。
4. **别只看健康度**：判断"卡"要看 `#pending-token`/`#queue-req`/`num_running_reqs`；判断"崩"要看 `Scheduler hit an exception` + `RestartCount` + `StartedAt`。
5. **长上下文风暴是本机最脆弱的负载形态**：单请求 70 万 token ≈ 20–30 分钟独占 GPU，还会把并发推高触发本 bug。真正的护栏（`--context-length` 下调 / `--max-queued-requests` 限流 / `--enable-priority-scheduling`）本次**未动**，留待与用户确认后单独做。

## 10. 待办

- [ ] 向 sgl-project/sglang 报 issue（`dflash_worker_v2.py stage_sampling_params`：缓冲区按 decode graph max bs 分配但未对 `bs > max_bs` 做保护/扩容），附本次 traceback + 镜像 commit `5f55db35`（09-15 复盘已列，仍未报）。
- [ ] **查 09-18 16:01 那次崩溃为何没有 watchdog 告警条目**（`alerts.log` 在 15:30:25 之后直接跳到 16:16:04）：怀疑 watchdog 进程在那一刻重启/状态丢失，需要复核 `journalctl --user -u qwen38-watchdog` 与 systemd 重启时间。
- [ ] 给 watchdog 增加"检测到 `RestartCount` 增长 / `StartedAt` 变化时自动 dump 崩溃前日志到 `incidents/`"——本次第 3 次崩溃的 traceback 随 `docker rm -f` 一起丢了。
- [ ] 观察 3~5 天：确认 bs 上限 4 之后不再出现 `Scheduler hit an exception`；若 0 次，考虑在维护窗口试备选 2 恢复 8 并发（并压测）。
- [ ] 长上下文客户端护栏（`context-length` / `max-queued-requests` / 优先级调度）——需与用户确认后再动。
- [ ] 备选：把 `recreate-concurrency4-20260918.sh` 的等价逻辑合进 `start-dspark-long.sh`（例如把 `MAX_CONCURRENT_REQUESTS` 默认值改成与 `--cuda-graph-max-bs-decode` 联动并加断言），从源头消除"两个参数各写各的"这类错配。
