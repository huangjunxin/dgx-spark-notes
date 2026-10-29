# 事故复盘：qwen3.8-27b 主服务静默停摆 66 分钟（DFlash2 batch 尺寸 bug）

> 日期: 2026-09-15 · 主机: spark-XXXX（DGX Spark GB10）· 服务: 容器 `qwen38-27b-sglang`
> 镜像: `lmsysorg/sglang:dev-qwen38-27b-dflash2`（dev 构建，commit `5f55db35`）
> 影响: 11:14–12:20（约 66 分钟）所有 `qwen3.8-27b` 请求 502
> 状态: 已恢复；已加持久护栏（restart 策略 + watchdog + 告警）
> 后续: 同一 bug 于 **2026-09-17~18 再复发 3 次**（共 5 次）→ **2026-09-18 源码级定案并修复**（真因是并发上限 8 与 decode CUDA graph 的 bs=4 不匹配；本文 §3 的"缓冲区按上次 batch resize/复用"解释**已更正**）。见 [qwen38-dflash2-crash-20260918.md](qwen38-dflash2-crash-20260918.md)

---

## 1. TL;DR

- **不是卡死、也不是谁手动停的**：11:14:21 SGLang scheduler 抛 `RuntimeError` → 进程树自杀（SIGQUIT）→ 容器 **exit 0**；
- 当时容器 `RestartPolicy=no`，**没有自愈**，服务就那样静默躺了 66 分钟，直到人工发现；
- 根因在 **DFlash2 投机解码草稿采样器**：`dflash_worker_v2.py:231` 的 `self.greedy_mask[:bs].copy_(...)` 在 batch 尺寸变化时张量长度不匹配（4 vs 7）；
- 现已三重护栏：`unless-stopped` 自愈、`qwen38-watchdog` 探活/抓现场/重启、告警推 Discord。

## 2. 时间线

| 时刻 (HKT) | 事件 |
|---|---|
| 11:14:21 | scheduler 抛 `RuntimeError: The size of tensor a (4) must match the size of tensor b (7)`；日志紧邻处有 `UserWarning: An output ... resized ... shape [4] ... required [7]` |
| 11:14:21 | `SIGQUIT received. signum=None, frame=None. It usually means one child failed.` |
| 11:14:26 | `No live scheduler processes found; skipping py-spy and CUDA coredump` → `kill_process_tree called: parent_pid=1` |
| 11:14:27 | 容器退出（`ExitCode=0`、`OOMKilled=false`），127.0.0.1:8888 消失 |
| 11:14–12:20 | switchyard(`:4000`) 对 `qwen3.8-27b` 持续返回 502 `upstream_error: error sending request for url (http://localhost:8888/...)` |
| 12:13:5x | 人工确认后 `docker start qwen38-27b-sglang` |
| 12:20:31 | `/health` 200（冷启动 ≈6.5 分钟，含 CUDA graph 捕获）；`:4000` 端到端 `1+1=?` → `2` |

## 3. 现场与根因

```
[2026-09-15 03:14:21] Scheduler hit an exception: Traceback (most recent call last):
  ...
  File ".../sglang/srt/speculative/dflash_worker_v2.py", line 1898, in forward_batch_generation
    self._draft_sampler.stage_sampling_params(
  File ".../sglang/srt/speculative/dflash_worker_v2.py", line 231, in stage_sampling_params
    self.greedy_mask[:bs].copy_(
RuntimeError: The size of tensor a (4) must match the size of tensor b (7) at non-singleton dimension 0
```

上一行的 torch 警告已经把病灶点名了：

```
dflash_worker_v2.py:226: UserWarning: An output with one or more elements was resized since it had shape [4],
which does not match the required output shape [7]. ...
```

**机理**：DFlash 草稿采样器的 `greedy_mask` 缓冲区按上一次 batch 的尺寸（7）resize/复用，这一轮 batch 变成 4，`greedy_mask[:bs].copy_(src)` 直接张量长度不匹配 → scheduler 异常 → 整个 server 自杀。触发场景是长上下文 + 多并发混合负载：崩溃瞬间 `#running-req: 5`、`#pending-token: 186526`、`full token usage 0.51`。跑了 4 天 18 小时才踩到，属于低概率但致命的路径。

**为什么"优雅退出"反而是坏事**：SGLang 收到子进程失败后走 SIGQUIT → 清理 → `kill_process_tree`，容器以 **exit 0** 结束。这与"正常停止"在 Docker 眼里没区别，所以 `on-failure` 类策略永远不会救它 —— 必须用 `always` / `unless-stopped`，或者应用层探活。

## 4. 排查过程（怎么排除"人为"和"机器故障"）

| 假设 | 证据 | 结论 |
|---|---|---|
| 有人 `docker stop` | `docker events` 在 11:00–11:30 窗口内**只有 `die`，没有 `kill`/`stop` 事件** | 排除 |
| 有人 SSH 进来敲的 | 11:14 前后无 sudo / ssh / logind 记录；`.bash_history` mtime 停在 9/9 | 排除 |
| AI agent 干的 | `~/.hermes` 网关日志 11:14 只有内存心跳；`.dsh`/`.hermes` 会话里无相关命令 | 排除 |
| GPU / 驱动挂了 | nvidia 模块在、`/proc/driver/nvidia/gpus/*/information` 正常、`GPU Firmware 580.173.02` | 排除 |
| 显存/内存 OOM | `OOMKilled=false`、容器 exit 0；崩溃时整机 91/121 GB 仍在正常水位 | 排除 |
| 主机重启 | `uptime` 22 天，cgroup 时间连续 | 排除 |
| **应用自崩** | traceback + SIGQUIT + `kill_process_tree` | **成立** |

> ⚠️ 踩坑：在 DSH 的文件沙箱里执行 `nvidia-smi` 会报 `Failed to initialize NVML: Unknown Error`（沙箱限制导致的**假警报**）。判断 GPU 好坏要看 `/proc/driver/nvidia/`、`lsmod`、`/dev/nvidia*`，不要被 nvidia-smi 误导。

## 5. 恢复动作

```bash
docker start qwen38-27b-sglang        # 复用原容器配置（DFlash2 + 1,010,000 上下文），无需重建
curl -fsS http://127.0.0.1:8888/health            # 200 即就绪（冷启动 ≈6.5 分钟）
curl -s  http://127.0.0.1:8888/v1/models          # 确认 id=qwen3.8-27b, max_model_len=1010000
```

## 6. 护栏（本次新增）

**① 容器级自愈**
```bash
docker update --restart unless-stopped qwen38-27b-sglang
```
注意：必须用 `unless-stopped`/`always`。崩溃是 exit 0，`on-failure` **不会**触发。
`unless-stopped` 的好处是：`./stop.sh`（人工 stop）后不会被自动拉起，不影响维护窗口。
docker 自身对反复崩溃有退避（实测秒级重启 6 次后间隔逐步拉长），配合 watchdog 的限流即可避免崩溃-重启风暴。

**② 应用级 watchdog**（探活 + 抓现场 + 自愈 + 限流）
> 取向：**告警要早，动作要晚，不确定就不动**。冷启动 6.5 分钟，一次误重启就是 7 分钟中断，比"晚发现"贵得多。

按证据分级判定（`classify`），越不确定越不动手：

| 判定 | 依据 | 动作 |
|---|---|---|
| `healthy` | `/health` 通过（超时 20s） | 无 |
| `warming` | 冷启动窗口（`STARTUP_GRACE` 1200s）内且本次启动未健康过 | **只观察**，不计失败、绝不重启 |
| `exited` | 容器已退出（崩溃，exit 0） | 连续 `FAIL_THRESHOLD`(4) 次 → `docker start`（无"杀"风险） |
| `portdown` | 容器在跑但端口没人监听（进程确已死） | 连续 `DEAD_CONFIRM`(6) 次 → `docker restart` |
| `unresponsive` | 端口在监听但探活不应答（繁忙 or 假死，**二者未定**） | 两段式：**`HUNG_GRACE`=10 min** 先发 CRIT **预警**（不重启，给人工抢先窗口）→ **`HUNG_IDLE`=15 min** 日志静默门槛到达才真正重启。两条是 AND；`HUNG_RESTART=0` 则永不自动重启 |

- 判据细节：端口用 TCP 连接 + `ss` 双重复核（探不出来时按"在监听"处理，即不动手）；"在忙"看最近 `ACTIVITY_WINDOW`(45s) 日志是否有输出；`WARMED` 标记保证冷启动宽限只在"本次启动尚未健康"期间生效；
- 抓现场永远在动作之前：`docker inspect` + 端口/uptime 状态 + `docker logs --tail 400`（崩溃 traceback 就在里面）→ `~/.local/state/qwen38-watchdog/incidents/`；
- 限流：1 小时内最多自愈 3 次，超出则退避 30 分钟并告警；
- 检测"外部/策略重启"（docker restart 策略拉起的 = 刚崩过一次），带 `RestartCount` 告警，不重复处理；
- **维护窗口**：`--maintenance on` 后只探活/告警、**不再自愈**（配合 `./stop.sh` 使用）；`--maintenance off` 恢复；
- 随时可查判定依据：`--diag`（打印容器/端口/探活/日志活动/判定/此刻会怎么做，**绝不动作**）。
- 单机自测：
```bash
qwen38-watchdog.sh --status      # 当前状态 + 最近告警
qwen38-watchdog.sh --once        # 单轮检查（不健康就自愈一次）
DRY_RUN=1 HEALTH_URL=http://127.0.0.1:9/health ALERT_CMD=cat STATE_DIR=/tmp/wd qwen38-watchdog.sh --once   # 演练，不碰容器
```

**③ 告警**（复用 hermes 网关现有的 Discord 通道，无需新凭据）
- 消息走 `hermes send --to discord:<频道>`；同时写 `journal`（`journalctl --user -u qwen38-watchdog`）与 `~/.local/state/qwen38-watchdog/alerts.log`；
- 告警点：不健康、自愈动作、恢复成功（含耗时）、自愈失败、触发限流、检测到外部重启；
- **2026-10-29 降噪修订**：上线头 45 天 1117 条告警中 987 条是未达告警线的"恢复 OK"虚惊（长 prefill 期 `/health` 偶发超时自愈），刷屏 Hermes 上下文。已改：恢复播报只补给真发过告警的事件；2min 级繁忙/疑似假死 WARN 降为 journal-only（`STALL_ALERT=0`）。CRIT 预警/自动重启/限流链路不变，详见 [qwen38-watchdog-denoise-20261029.md](qwen38-watchdog-denoise-20261029.md)；
- 换通道/关推送：编辑 `~/.config/qwen38-watchdog.env`（见 `qwen38-watchdog.env.example`）。

**安装/运维**
```bash
install -m644 ~/projects/dgx-spark-notes/qwen38-sglang/qwen38-watchdog.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now qwen38-watchdog.service
systemctl --user status qwen38-watchdog          # 本机 Linger=yes，无需登录常驻
journalctl --user -u qwen38-watchdog -f
```

## 7. 上线验证（2026-09-15 12:29）
- `systemctl --user is-active qwen38-watchdog` → `active`，`is-enabled` → `enabled`（本机 `Linger=yes`，登录/注销都不影响常驻）；
- **重启策略实测**：临时起一个 `exit 0` 就退出的探针容器（同一镜像、`--network none`、无 GPU），加 `--restart unless-stopped` 后 6 秒内 `RestartCount=6` → 证实 **exit 0 的崩溃同样会被自动拉起**（这正是本次事故缺的那一环）；验证完 `docker rm -f` 删除；
- **告警通道实测**：经 `hermes send --to discord:<频道>` 发出测试消息，返回 `sent`（exit 0）；
- **自愈路径演练**：`DRY_RUN=1 ALERT_CMD=cat STATE_DIR=/tmp/wd qwen38-watchdog.sh --once`（指向假 health 地址）→ 正确生成现场文件（含 `docker inspect` + `docker logs --tail 400`）与告警文本，全程未触碰容器。

## 8. 若复发：备选形态

同一镜像的 DFlash2 路径有 bug，若再犯（或自愈循环触发限流告警），按 `LONG-CONTEXT-CUTOVER.md` 回退：

```bash
./stop.sh && ./start.sh                                    # MTP + 262K（最稳）
./stop.sh && CONTEXT_LENGTH=506000 ./start-dspark-long.sh  # DSpark + 506K（原备用形态）
```


## 9. 同日复发 + 护栏实战复盘（2026-09-15 12:38–12:50）

恢复后约 25 分钟、累计约 15 个 chat 请求，**同一个 DFlash2 bug 再次发作**。这次三层护栏都在真实故障里跑了一遍，有两条要记：

| 时刻 | 事件 | 评价 |
|---|---|---|
| ~12:38 | `dflash_worker_v2` 采样器警告再次出现（3 次），探活开始抖动（12:39:14 / 12:39:53 失败，12:40:24 短暂恢复） | 崩溃前兆与上次一致 |
| ~12:40:4x | scheduler 再次异常 → 容器 `die exitCode=0`（第 2 次） | 同一 bug，真实负载下 25 分钟即复发 |
| 12:40:52 | **`unless-stopped` 策略自动拉起容器** ✅（`docker events`: `die exitCode=0` → `start`） | **① 层护栏按设计生效** —— 对比首次事故的"静默 66 分钟" |
| 12:42:55 | watchdog 报"服务不健康"并 `docker restart`，`docker events`: `kill signal=15` → `kill signal=9` → `die exitCode=137` → `start` | ⚠️ **误判**：容器 12:40:52 才重启，此刻**才冷启动 2 分钟**（本机冷启动 ≈6.5 分钟），被 watchdog 掐死 |
| 12:46 | 修复并重启守护 | 见下 |

**修复（watchdog 冷启动宽限）**：新增 `container_uptime_secs()`，主循环与 `--once` 都先判断"容器本轮运行时长 < `STARTUP_GRACE`（默认 1200s）" → 只记日志、**不计失败、绝不自愈**。
日志实测：`容器启动中（已 178s / 冷启动窗口 1200s），探活失败不计入`。
教训：**探活阈值必须显式让位于冷启动时长**，否则"自愈"会变成"反复掐死正在加载的进程"，比不修还糟。

**结论与决定（2026-09-15 12:55）**：DFlash2 形态在本机真实负载下已两次崩溃（间隔 4 天 18 小时 → 25 分钟）。
- 备选方案（§8 MTP + 262K / DSpark + 506K）与"崩溃循环自动回退开关"都已准备好，**当前决定：暂不切形态，先靠 自愈 + 告警 扛着**，观察复发频率；
- 接受每次崩溃 ≈7 分钟中断（冷启动 6.5 min）的代价；若进入崩溃循环（Discord 收到"限流退避"CRIT 告警），再按 §8 切形态。
- 为配合"扛着"策略做的两处加强：① watchdog 冷启动宽限（避免掐死正在加载的实例）；② 策略触发的重启从 `[OK]` 升为 `[WARN]` 并带上 `RestartCount`（>0 即代表刚崩过一次），确保每次崩溃你都收得到提示。
- 另：`start.sh` / `start-dspark-long.sh` 的 `docker run` 已补 `--restart unless-stopped`（原来重建容器会丢掉这条护栏）。

**第二处修复（12:53，繁忙误判）**：12:51 巡检时 `/health` 返回 000（curl 6s 超时），但 8888 在监听、容器日志 1 分钟前才返回过 `200 OK`、且正在处理 8192-token 分块的超长 prefill —— 即**长上下文负载下 HTTP 事件循环会迟迟不响应探活**，而不是服务故障。若按旧逻辑（8s 超时 + 连续失败即自愈），这类繁忙会被误判成"挂了"并触发一次 7 分钟假中断。修复：
- 探活超时 8s → **`HEALTH_TIMEOUT`（默认 20s）**；
- 新增**忙判定** `container_busy()`：最近 `ACTIVITY_WINDOW`（默认 45s）内日志仍有输出 → 判为繁忙，不计失败；
- 新增**冷启动只认"健康前"**：`WARMED` 标记本次启动是否已健康过，宽限只在"本次启动尚未健康"期间生效（否则"容器在跑但服务已死"会被宽限掩盖 20 分钟）。

日志实测：`冷启动完成，进入正常监控（/health 通过）`；修复后 `/health` 1.0s 返回 200。

**第三处修复（12:58，整体保守化 v2）**：复核时发现"8s 超时 + 连续失败即自愈"仍是"宁可错杀"的思路——长上下文 prefill 期间 `/health` 实测 6s 超时属正常繁忙，误杀一次 = 7 分钟中断。按 **"告警要早，动作要晚，不确定就不动"** 重写判定：
- 引入 `classify` 五态（`healthy/warming/exited/portdown/unresponsive`）与**按证据分级**的动作：容器已退出→只 `start`；端口无人监听（进程确已死）→ 3 分钟后 `restart`；**端口在监听但不应答（繁忙 or 假死，二者未定）→ 默认只告警**，并在 10 min 时先发 CRIT 预警、15 min 日志静默门槛到达才重启（`HUNG_RESTART=0` 可设为永不自动重启）；
- **阈值用实测标定**：取容器日志 19 小时共 287 个时间戳，统计正常运行时的日志空档 —— 中位 **11.2s**、P95 **179s**、最长 **466s(7.8 min)**；因此"日志静默"门槛取 **15 min ≈ 2× 实测上限**（取 10 min 只有 1.3× 余量，偏紧），且必须与"不应答"同时成立才动手；
- 端口判定用 TCP 连接 + `ss` 双重复核，**探不出来时按"在监听"处理**（即不动手）；
- `--diag` 子命令：随时打印"容器/端口/探活/日志活动/判定/此刻会怎么做"，只读不动作；
- 实测（模拟三种故障态）：`unresponsive` → `判定: unresponsive / 此刻动作: 只告警不重启`；`portdown` → `连续 6 次后 docker restart`；`--once` 在 unresponsive 下**不动作**。
- **两段式（预警→执行）实测**（用一个"端口在监听、日志静默"的哑容器模拟假死，DRY_RUN 不真动）：
  ```
  [WARN] 探活连续 2 次失败，端口在监听但日志近 45s 无输出 → 疑似假死；先观察不重启（门槛：不应答 ≥40s 且日志静默 ≥60s）
  [CRIT] 预警：已连续 41s 不应答，且最近 40s 日志完全静默。再过约 19s（静默满 60s）仍无改善将自动重启；要人工介入请现在处理，或先跑 --maintenance on 冻结自愈
  [CRIT] 连续不应答已达 41s（≥40s）且日志静默 ≥60s，判定为真死，执行重启  → [dry-run] 跳过 docker restart
  ```
- **动作时刻的语义**（易误读，记一下）：AND 里那条"日志静默"会被活动日志不断刷新，所以——
  **空闲中挂死 → 到 `HUNG_GRACE`(10 min) 就动作**；**繁忙后挂死 → 要等日志也静默满 `HUNG_IDLE`(15 min)** 才动作。

**恢复确认（2026-09-15 12:50）**：容器 `StartedAt 12:43:26` → `/health` 于 **12:49:49** 返回 200（冷启动 6 分 23 秒）；`:4000` 端到端 `9+9=?` → `18`；`RestartCount=0`、`watchdog=active`、内存 available 25 GB。期间共发出 2 条 Discord 告警（12:40:54 策略重启 / 12:42:55 误判自愈）。

## 10. 待办

- [x] ~~观察是否复发~~ → **2026-09-15 12:40 即复发**（真实负载约 15 请求 / 25 分钟），见 §10；下一步应按 §8 切备用形态或开自动回退开关；
- [x] ~~继续复发~~ → **2026-09-17 15:14 / 15:24、09-18 约 16:01 再崩 3 次**（同一 bug，共 5 次）。**2026-09-18 已定案并修复**：真因是**并发上限 8 与 decode CUDA graph 的 bs=4 不匹配**（本文 §3 里"缓冲区按上次 batch 尺寸 resize/复用"的解释**已更正**）→ 改 `--max-running-requests 4` 后 8 并发压测通过。完整复盘：[qwen38-dflash2-crash-20260918.md](qwen38-dflash2-crash-20260918.md)；
- [ ] 向 sgl-project/sglang 报 issue（`dflash_worker_v2.py stage_sampling_params` 的 `greedy_mask` 未按 `bs` 重建），带上本次 traceback 与镜像 commit `5f55db35`；
- [ ] 考虑给 watchdog 增加"崩溃自动回退到备用形态"的开关（当前只重启同形态，避免无人值守时擅自改变服务规格）。
