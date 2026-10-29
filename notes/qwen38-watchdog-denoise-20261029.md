# 告警降噪改造：Hermes 通道"沉默即健康"（2026-10-29）

> 状态：已上线（watchdog 11:56 重启生效；dgx-health-monitor timer 每轮拉新进程自动生效）
> 关联：[qwen38-dflash2-crash-20260915.md](qwen38-dflash2-crash-20260915.md)（watchdog 诞生记）· [dgx-spark-model-service.md §9](../dgx-spark-model-service.md)（护栏总表）· `~/.config/systemd/user/dgx-health-monitor.{service,timer}`（9/18 断网事故后加装的系统级巡检）

## 1. 问题：没断线也在一直发

用户观察：Hermes/Discord（`#zunni-world`）天天收到监控消息，"没事情也一直发"，污染 hermes agent 的上下文（hermes 的模型就是本机 qwen3.8-27b，消息都在抢同一个上下文窗口）。

实锤（`~/.local/state/qwen38-watchdog/alerts.log`，2026-09-15~10-29 共 45 天）：

| 级别 | 条数 | 构成 |
|---|---|---|
| OK | **987** | 几乎全是"此前连续 **1~3 次**探活失败、观察了 50~150s，未做任何重启"的恢复播报 |
| WARN | 129 | 114 条 2min 级"疑似假死"（同为长 prefill 假警报）+ 9 条"繁忙" + 5 条外部重启 + 1 条真自愈 |
| CRIT | 1 | 唯一的真事件（10/08 预警后自愈） |

日均 ~10 条，其中 **99% 是虚惊**：sglang 长 prefill 期间 `/health` 偶发超时（30~50s），下一轮探活自愈，watchdog 却每次都发一条"服务恢复"。

## 2. 根因（三处实现偏差，设计本身没错）

两个监控的**主体设计都是事件驱动**（healthy/在线时零发送，"告警要早，动作要晚"哲学），噪声来自：

1. **watchdog 恢复播报没有门槛**：healthy 分支只判 `hung_since` 非空就发 OK，不判"本事件是否真的发过告警"→ 没到告警线（FAIL_THRESHOLD=4）的虚惊也播报恢复；
2. **health-monitor 的 hung task 按增量日志行告警**：挂起持续期间内核每轮都写新 `blocked for more than` 行 → 每 5 分钟重复一条 CRIT；
3. **health-monitor 断网重复催办**：`RE_ALERT_EVERY=12`，持续断网时每小时重复一条（虽然算"还在断"，但对只想要"断/恢复"两个点的场景太吵）。

## 3. 改动清单

| 文件 | 改动 | 默认行为 |
|---|---|---|
| `~/projects/dgx-spark-notes/qwen38-sglang/qwen38-watchdog.sh` | ① 恢复播报(OK)加门槛：新增 `alerted_incident` 状态，只补给"真发过告警"的事件；② 新增 `STALL_ALERT=0`：2min 级"繁忙/疑似假死"WARN 降为 journal-only | 虚惊完全静默 |
| `~/bin/dgx-health-monitor.sh` | ① hung task 同事件去重：同任务只告警一次，连续 `HUNG_CLEAN_CONFIRM=2` 轮（≈10min）无新记录才发恢复；② `RE_ALERT_EVERY` 默认 12→0（关重复催办，保留首条 CRIT + 恢复 RECOVER） | 断网只发"断"和"恢复"两条 |

**保持原样不动**：HUNG_GRACE(10min) CRIT 预警（人工窗口）、HUNG_IDLE(15min)+日志静默 AND 才自动重启、限流 3 次/小时、维护窗口、外部重启检测（带 RestartCount）。

备份：`qwen38-watchdog.sh.bak-pre-denoise-20261029` · `dgx-health-monitor.sh.bak-pre-dedup-20261029`。`qwen38-watchdog.env.example` 已同步新开关文档。

**效果预估**：按 45 天历史回放，告警量从日均 ~10 条降到 **~0 条**，只保留真事故（CRIT 预警 / 自愈动作 / 恢复收尾）。

## 4. 验证方法（可复用）

隔离桩测试：stub 掉 `docker`/`curl`/`logger`（放临时 PATH 最前面），`ALERT_CMD=cat`（告警载荷落到测试 alerts.log，不碰真 hermes），`HERMES_PY=/bin/false`。覆盖四条路径：

| 场景 | 构造 | 期望 | 实测 |
|---|---|---|---|
| 虚惊自愈 | 失败 2 次（达 FAIL_THRESHOLD）后恢复 | 零告警、alerts.log 不创建、journal 有"不上报"行 | ✅ |
| 预警后自愈 | 失败 ≥HUNG_GRACE 后恢复（用"8s 窗口日志忙"桩抑制自动重启） | CRIT 预警 + 恢复 OK 各 1 条 | ✅ |
| 真死自愈 | 失败 ≥HUNG_GRACE 且 IDLE 窗口日志静默 | CRIT×2 + WARN + docker restart 动作执行 | ✅ |
| 全程健康 | 无任何故障 | 零输出 | ✅ |

上线核验：`ps -o lstart` 确认新进程启动时间晚于脚本 mtime（重启由 dyl 执行：`systemctl --user restart qwen38-watchdog`）。

## 5. 踩坑与教训（本次实踩）

1. **bash 对运行中的脚本是增量读取**：改完常驻脚本必须重启服务（`systemctl --user restart`），否则旧进程按字节偏移继续读已变更的文件，行为不可预期（轻则跑旧逻辑，重则解析错乱）。改完必重启，`ps -o lstart` 核验。
2. **`exec 9>"$LOCKFILE" 2>/dev/null` 会把整个脚本的 stderr 永久重定向**（重定向型 exec 的副作用）：之后所有报错、`bash -x` trace 全部静默消失。调试这类脚本先把该行 sed 成 `2>>/tmp/xxx.trace`。
3. **测试桩必须忠实模拟被测接口的输出契约**：真 `curl -o /dev/null` stdout 为空，桩却 `echo ok` → `healthy()` 命令替换返回非空 → `st='ok\nhealthy'` 匹配不上任何 case → 主循环进入无输出空转，表象是"脚本无声死掉"，排查半天其实是桩的问题。**先怀疑桩，再怀疑被测代码。**
4. **`systemctl --user` 是每用户独立的**：unit 装在 dyl 的用户管理器（`~dyl/.config/systemd/user/`），hjx 执行 `systemctl --user ...` 查自己的管理器，报 `Unit not found`。跨用户操作：`sudo -u dyl XDG_RUNTIME_DIR=/run/user/<dyl-uid> systemctl --user ...`。
5. **ACL 会被进程自己的 chmod 压掉**：给 `~/.hermes` 授过 ACL，hermes 网关后续 chmod 它的 `cron/` 目录时把 mask 重置为 `---`，effective 权限清零。授权要连 mask 一起设：`setfacl -m u:hjx:rwX,m::rwX`；对活跃进程管理的目录要预期"权限被改回去"。
6. **日志先行，定性靠数据**：一开始以为"一直发"是 hermes 网关自身的周期心跳，翻完 `alerts.log` 按级别/按天聚合才定位到是 watchdog 的 987 条 OK。先量化（`grep -o '\[OK\]' | wc -l`）再动手，避免改错对象。

## 6. 回滚

- watchdog：`STALL_ALERT=1`（恢复 2min 级 WARN 上报）或用备份覆盖脚本后 `systemctl --user restart qwen38-watchdog`；
- health-monitor：env 里 `RE_ALERT_EVERY=12`，或用备份覆盖（timer 下一轮自动生效）。

## 7. 遗留观察项

- 观察两三天 `journalctl --user -u qwen38-watchdog`：应能看到被降级的"未达告警线，不上报"行（保住排障证据），Discord 侧安静；
- 若真事故出现且告警链路正常（CRIT → 自愈 → OK），任务闭环；
- notes 仓库 `git init` 后**至今零提交**——本次所有改动只有文件备份、没有版本锚点，建议尽快做初始提交。
