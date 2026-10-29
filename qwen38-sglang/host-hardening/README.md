# 主机侧稳定性加固（DGX Spark / GB10）

> 产出日期：2026-09-30 · 背景调研：`../../research-qwen38-20260930/`（C 线 §稳定性、D 线 §主机审计）
> 目的：把"防止整机被拖到需要断电"和"区分真死 / 繁忙"这两件事从口头约定变成可执行的东西。

## 为什么必须做

| 事实 | 来源 |
|---|---|
| cgroup 对 CUDA 统一内存**基本失明**：实测 10.7GB 显存分配只被计到 377MB | C 线实测 |
| `systemd-run MemoryMax` 会被 Docker 逃逸；`docker --memory 100g` 也拦不住 GPU 侧分配 | C 线实测 |
| mem-fraction >0.80 已实测崩（3 并发剩 2GiB）；cookbook 的 0.85 被 earlyoom 杀掉 15/48 | C 线/社区 |
| `/health` 在引擎 wedged 时**仍返回 200**，会把假活判成健康 | C 线 |
| 容器 `--restart unless-stopped` 既救偶发崩溃，也可能造成"OOM→重启→再 OOM"砖循环 | 09-18 复盘 |
| 宿主当前 `vm.swappiness=60`、swap 仅 15GB | 本机实测 |

**唯一真护栏 = mem-fraction（已有，0.70）+ PSI 看门狗 + earlyoom。**

## 文件

| 文件 | 作用 | 何时调用 |
|---|---|---|
| `earlyoom.default` | `/etc/default/earlyoom` 内容：空闲内存 <3% 触发，优先杀 python/sglang/triton，绝不碰 systemd/sshd/dockerd/switchyard | 常驻 |
| `canary-check.sh` | 真实生成式探活（替代 `/health`），叠加 PSI 内存压力判断 | 定时/watchdog，每 30s 一次 |
| `restart-storm-guard.sh` | 检测窗口内重启风暴，写 `STORM` 标记并给处置建议 | 定时，每 5 分钟 |
| `install.sh` | 应用上述配置（默认 dry-run，`--apply` 才执行，`--rollback` 回滚） | 一次性 |

## canary-check.sh 的退出码约定

| 码 | 含义 | 建议动作 |
|---|---|---|
| 0 | 健康（生成成功） | 无 |
| 1 | 死亡（生成失败 **且** 容器近 45s 无日志） | 可重启 |
| 2 | 内存压力临界（PSI `full avg10` > 10%） | 告警，别急着重启 |
| 3 | 繁忙（生成超时/失败但容器仍在输出日志） | **只告警**，长 prefill 期间属正常 |

> 这套码是照着现有 `qwen38-watchdog.sh` 的设计哲学来的：**告警要早，动作要晚，不确定就不动**
> （冷启动 7-10 分钟，一次误重启比晚发现更贵）。

## 安装（需要你在宿主机执行 —— DSH 沙箱内 sudo 被禁用）

```bash
cd ~/projects/qwen38-sglang/host-hardening
./install.sh                                    # 先看 dry-run 输出
./install.sh --apply --install-earlyoom         # 确认后执行
```

另外两条建议手工做（脚本不自动改）：

```bash
# 1) 降低 swap 倾向（统一内存下 swap 抖动 = 整机卡死；当前 60）
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-qwen38.conf && sudo sysctl --system

# 2) 试跑 canary（应返回 0 且 1 秒内完成）
./canary-check.sh; echo "exit=$?"
```

## 与现有 watchdog 的关系

`qwen38-watchdog.service`（09-15 上线，`dgx-spark-notes/qwen38-sglang/qwen38-watchdog.sh`）继续负责容器存活与自愈，**不要直接停用**。
建议的接法（择一）：

- **保守**：保持 watchdog 不变，把 `canary-check.sh` 与 `restart-storm-guard.sh` 挂成额外定时任务，只告警。
- **进阶**：在 watchdog 里新增一条分支——"canary 连续失败 ≥3 次 **且** 退出码=1" 才触发重启；退出码=3（繁忙）永远不动作。

未做这件事的原因是：改 18KB 的现场 watchdog 属于有行为风险的改动，应单独做并演练，不宜与本次脚本改动混在一起。

## 附：外部直连代理的持久化（`qwen38-proxy.service`）

`vllm-open-proxy.py`（Tailscale/LAN :8888 → 127.0.0.1:8888）在 2026-09-30 被发现**已静默停掉**
（无进程、无监听、日志停在 8/23），导致外部直连全失效而无人察觉 —— 因为它是用 `nohup` 起的，没有任何监管。
已于 09-30 重新拉起并验证三张网卡（<LAN-IP-1> / <LAN-IP-2> / <Tailscale-IP>）均 200。

要让它以后能自愈 + 开机自启（推荐，需你执行）：

```bash
mkdir -p ~/.config/systemd/user
cp qwen38-proxy.service ~/.config/systemd/user/
# 先把 nohup 那份停掉, 避免抢 8888 端口
pkill -f vllm-open-proxy.py
systemctl --user daemon-reload
systemctl --user enable --now qwen38-proxy.service
systemctl --user status qwen38-proxy.service --no-pager
curl -s -o /dev/null -w '%{http_code}\n' http://<LAN-IP-1>:8888/v1/models   # 期望 200
```

> 注意：该 unit 是**用户级**服务，只在用户登录会话存在时运行。若要开机即起（无需登录），
> 需要 `sudo loginctl enable-linger dyl`，或改成系统级 unit。

## 回滚

```bash
./install.sh --rollback        # 停用 earlyoom 并删除 /etc/default/earlyoom
```

`canary-check.sh` / `restart-storm-guard.sh` 是只读探针，不用了直接不调用即可。

## 已知边界

- `restart-storm-guard.sh` 的状态目录默认 `${HOME}/.local/state/qwen38-watchdog`（与现有 watchdog 一致）。
  在受限沙箱里该目录不可写，脚本会打印 WARN 并 exit 0 —— 这时用 `STATE_DIR=/tmp/...` 指到可写位置。
- earlyoom 的 `--prefer/--avoid` 是正则匹配进程名；若你以后改了进程命名，需要同步更新。
- earlyoom 需要 `CAP_KILL`（Ubuntu 的 systemd unit 已含 `AmbientCapabilities=CAP_KILL`）。
