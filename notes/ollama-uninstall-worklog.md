# Ollama 卸载与清理工作记录

> 生成时间：调查当日
> 目标：卸载 Ollama 本体，并删除其下载的模型
> 状态：**安装信息已查清；卸载/删除需 root 权限，由用户在自身终端执行**（详见文末说明）

---

## 1. 安装调查结果（已确认）

| 项目 | 位置 / 值 |
| --- | --- |
| Ollama 本体（二进制） | `/usr/local/bin/ollama`（版本 `0.32.15`，root 所有） |
| 服务（systemd） | `ollama.service`，单元文件 `/etc/systemd/system/ollama.service`，附带 `/etc/systemd/system/ollama.service.d/` |
| 运行方式 | `ExecStart=/usr/local/bin/ollama serve`，用户 `ollama:ollama`，已 **disabled / inactive** |
| 环境变量 | 未设置 `OLLAMA_HOME`，使用系统默认值 |
| 模型数据目录 | `/usr/share/ollama/.ollama/models/`（**总大小 46 GB**） |
| 模型目录结构 | `blobs/`（真实权重，以 `sha256-` 命名）+ `manifests/registry.ollama.ai/library/...`（各模型清单） |
| 用户级目录 | `~/.ollama/`（仅含 CLI 配置、密钥 `id_ed25519*`、`skills`、`backup`、`cache`；**模型权重不在此处**） |

### 已安装的模型（从 manifest 解析）

| 模型名 | 模型权重 blob（在 /usr/share/ollama/.ollama/models/blobs/） | 大小 |
| --- | --- | --- |
| `qwen3-embedding:8b` | `sha256-3fcd3feb...` | ~4.7 GB |
| `nemotron-3.5-lightning`（源自 `nemotron3.5:30b-a3b-q4_K_M`） | `sha256-5c19f628...` | ~25.4 GB |
| `nomic-embed-text` | `sha256-970aa74c...` | ~274 MB |
| `qwen3.8:27b` (`27b-q4_K_M`) | `sha256-f5f1dd89...`（+ projector `sha256-ac3714bf...` ~0.9 GB） | ~16.8 GB |
| `qwen3.8:27b-fast` | 复用同一份 `sha256-f5f1dd89...` 与 `sha256-ac3714bf...`（不额外占空间） | 同上 |

> 注意：blobs/ 中存在一组未下载完的残片 `sha256-6f85a640...-partial`（及其 `-partial-0`~`-8` 分片），是一次被中断的 pull 留下的，重新 pull 同一模型时会被补齐/覆盖。

---

## 2. 为什么需要用户在 root 终端执行

本次操作所在的 shell 是**无根用户 <user>**（uid 1000），且无法提权：

- sudo 无法提权到 root（报：The "no new privileges" flag is set, which prevents sudo from running as root）
- 即便放开沙箱限制，对 /usr/local/bin、/usr/share/ollama、/etc/systemd/system 的写/删操作依旧返回 权限不够（EACCES）

因此二进制、systemd 单元、模型目录都删不动。请在你能使用 sudo 的终端（桌面终端）里执行下面的命令。

---

## 3. 待执行：卸载 Ollama + 删除模型

### 卸载 Ollama 本体 + 删除模型（手动分步，需 root）

> ⚠️ 本机 Ollama 版本为 0.32.15，**没有 `uninstall` 子命令**（已用 `sudo ollama help` 确认：可用命令里没有 uninstall）。因此不用再试 `ollama uninstall`，直接用下面命令手动移除即可。

```bash
# 1) 停止并禁用服务
sudo systemctl stop ollama
sudo systemctl disable ollama

# 2) 删除 systemd 单元并刷新
sudo rm -f /etc/systemd/system/ollama.service
sudo rm -rf /etc/systemd/system/ollama.service.d
sudo systemctl daemon-reload

# 3) 删除二进制本体
sudo rm -f /usr/local/bin/ollama

# 4) 删除模型及其他 ollama 数据（此处即 46 GB 模型）
sudo rm -rf /usr/share/ollama

# 5)（可选）删除用户级目录 ~/.ollama 里的配置/密钥/缓存
rm -rf ~/.ollama
```

> 若只想清模型、保留安装：只执行第 4 步 `sudo rm -rf /usr/share/ollama/.ollama/models`。

---

## 4. 验证

```bash
which ollama          # 应无输出（命令不存在）
ls /usr/share/ollama  # 应提示：No such file or directory
systemctl status ollama 2>&1 | head -3   # 单元应消失（not-found）
du -sh ~/.ollama ~/.cache 2>/dev/null    # 用户目录体积变小
```

---

## 5. 小结

- ✅ Ollama 本体、模型、服务、每个模型对应的文件位置均已查清（见第 1、2 节）。
- ⚠️ 卸载/删除需 root，本沙箱环境无法执行；请按第 3 节在自己的终端运行。
- 本机 ollama 0.32.15 没有 uninstall 子命令，直接用第 3 节的手动方式卸载即可。

---

## 6. 验证成果（已由用户在终端完成并验证）

在用户自带 sudo 权限的终端执行完第 3 节命令后，用只读命令逐项核对，结果如下：

| 验证项 | 结果 | 状态 |
| --- | --- | --- |
| `which ollama` | 无输出，PATH 上找不到 | ✅ |
| `ollama --version` | 未找到命令（command not found） | ✅ |
| `/usr/local/bin/ollama`（二进制本体） | 没有那个文件或目录 | ✅ 已删 |
| `/usr/share/ollama`（46 GB 模型数据） | 没有那个文件或目录 | ✅ 已删 |
| `/usr/share/ollama/.ollama`（ollama 数据目录） | 没有那个文件或目录 | ✅ 已删 |
| `systemctl status ollama` | `Unit ollama.service could not be found` | ✅ 服务已移除 |
| systemd 单元文件注册 | no ollama unit file | ✅ 已清除 |

**核心卸载 = 100% 成功**：本体、模型、服务、单元文件全部清除到位。

> 仅剩可选项：用户级目录 `~/.ollama`（含 CLI 配置、密钥、skills、backup、cache）仍在。因 ollama 本体已卸载，留着无任何作用，可执行 `rm -rf ~/.ollama` 彻底清理（本沙箱无法写 `/home/<user>`，故未代为删除）。
