# Ornith-1.5-35B-A3B 本机部署记录（DGX Spark / GB10）

> 记录 2026-08-22 在 `spark-XXXX`（NVIDIA DGX Spark）上，按
> [MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark](https://github.com/MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark)
> 部署 Ornith-1.5-35B-A3B（NVFP4 + MTP 投机解码）的完整过程：
> 前置条件、正确部署路径、踩过的坑与解决方案、日常运维。

---

## 0. 结论先行

- ✅ 部署成功，服务运行于 `http://127.0.0.1:8888`（OpenAI 兼容接口）
- 模型名：`ornith-1.5-35b-a3b-nvfp4`，容器：`ornith-b12x-serve`
- 配置：NVFP4（~22 GB）· MTP 投机解码（1 spec token）· b12x MoE 后端
  · 256k 上下文 · fp8_e4m3 KV cache · `--gpu-memory-utilization 0.85`
- 启动耗时：约 6 分钟（模型加载 164 s + torch.compile / CUDA graph 捕获）
- **上游 `start.sh` 在本机（宿主无 vllm/torch）会因 `set -e` 直接崩溃，需打 3 行补丁**（见 §4.4 与 §5）

---

## 1. 本机环境（部署目标机）

| 项目 | 值 |
|---|---|
| 机型 | NVIDIA DGX Spark（GB10 / SM121，128 GB 统一内存） |
| 主机名 / OS | spark-XXXX / Ubuntu 24.04.4 LTS（aarch64，内核 6.17.0-1031-nvidia） |
| GPU 驱动 / CUDA | 580.173.02 / 13.0（nvidia-smi 不显示 GB10 显存数值，属正常） |
| Docker | 29.2.1（daemon 以 root 运行；普通用户需 docker 组或 `sg docker`） |
| NVIDIA Container Toolkit | 1.20.0（`/usr/bin/nvidia-ctk`） |
| 系统 Python | 3.12.3（**无 torch、无 vllm** —— 这是本机与上游作者环境的关键差异） |
| 磁盘 | 3.4 TB 可用（部署占用约 22 GB 权重缓存 + 23 GB 镜像 + ~2 GB venv） |
| 同机竞争服务 | ollama（systemd 服务，曾占用 ~49 GB 显存 + 96% GPU） |

---

## 2. 部署前置条件清单

| # | 前置条件 | 检查命令 | 本机初始状态 |
|---|---|---|---|
| 1 | Docker 可用（当前用户） | `docker ps` | ❌ 不在 docker 组，需 sudo |
| 2 | NVIDIA runtime 注册进 Docker | `cat /etc/docker/daemon.json` | ❌ 无 daemon.json |
| 3 | GPU 显存空闲（建议 ≥ 100 GB） | `nvidia-smi` | ❌ ollama 占 ~49 GB |
| 4 | HF token | `test -f ~/.cache/huggingface/token` | ❌ 缺失 |
| 5 | `hf` CLI（start.sh 宿主侧下载权重用） | `hf --version` | ❌ 缺失 |
| 6 | 磁盘空间 ≥ 60 GB | `df -h /` | ✅ 3.4 TB |
| 7 | 端口 8888 空闲 | `ss -tlnp | grep 8888` | ✅ |

> 模型仓库 `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` 是**公开（非 gated）**仓库，~18.7 GB；
> 但 start.sh 强制要求 token 文件存在（无 token 时裸 `hf download` 会在 401 上静默挂起）。

---

## 3. 正确部署路径（本机完整步骤）

### 3.1 一次性系统配置（需 root，约 1 分钟）

在机器终端执行（会重启 docker，影响当前容器；执行前先停 ollama 释放显存）：

```bash
# 1) 当前用户加入 docker 组
sudo usermod -aG docker <user>

# 2) 注册 nvidia runtime 到 docker（生成 /etc/docker/daemon.json）
sudo nvidia-ctk runtime configure --runtime=docker

# 3) （推荐）停止 ollama，给 vLLM 留出全部 ~128 GB 显存
sudo systemctl stop ollama

# 4) 重启 docker 使 runtime 生效
sudo systemctl restart docker
```

> ⚠️ 加入 docker 组后，**已存在的登录会话不会自动获得权限**；
> 新开终端即可，或使用 `sg docker -c "docker ps"` 在现会话中生效（无需重新登录）。

### 3.2 部署（无 root，约 1 小时，大头是下载）

```bash
# 1) 克隆部署仓库
git clone --depth 1 https://github.com/MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark.git ~/projects/Ornith-1.5-35B-A3B-DGX-Spark
cd ~/projects/Ornith-1.5-35B-A3B-DGX-Spark

# 2) 安装 hf CLI（Ubuntu 24.04 pip 受 PEP 668 管理，用 venv）
python3 -m venv ~/.venvs/hf
~/.venvs/hf/bin/pip install -U pip huggingface_hub
~/.venvs/hf/bin/hf auth whoami        # 验证 token

# 3) 写入 HF token（模型公开但仍需文件；权限收紧到 600）
mkdir -p ~/.cache/huggingface && umask 077
printf '%s' 'hf_你的token' > ~/.cache/huggingface/token

# 4) 打 start.sh 补丁（本机必须！原因见 §4.4；补丁内容见 §5）

# 5) 启动（脚本自动：校验 token → 下载/校验权重 → 拉镜像 → 起容器 → 等 /health）
export PATH="$HOME/.venvs/hf/bin:$PATH"
sg docker -c "./start.sh"        # 已有 docker 组权限的直接 ./start.sh
```

启动成功后脚本打印 `Health OK: http://127.0.0.1:8888/health` 并返回 shell。

### 3.3 验证

```bash
curl -s http://127.0.0.1:8888/health                 # 200 OK
curl -s http://127.0.0.1:8888/v1/models              # 列出模型 ornith-1.5-35b-a3b-nvfp4
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith-1.5-35b-a3b-nvfp4","messages":[{"role":"user","content":"say hi"}],"max_tokens":32}'
# 应返回带 reasoning 字段的正常回复（MTP + qwen3 reasoning parser 生效）
```

---

## 4. 踩坑记录

### 4.1 坑①：当前用户无 Docker 权限（permission denied）

- **现象**：`docker ps` → `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`
- **原因**：docker 套接字属 `root:docker`，而用户 `<user>` 不在 docker 组；`sudo` 又需要密码（交互式环境不可用）
- **解决**：`sudo usermod -aG docker <user>` + 用 `sg docker -c "<命令>"` 在**当前会话**立即生效（无需重新登录）

### 4.2 坑②：NVIDIA runtime 未注册进 Docker

- **现象**：容器工具链已装（`nvidia-ctk --version` 正常），但 `/etc/docker/daemon.json` 不存在，
  `docker run --gpus all` 会报 `could not select device driver "nvidia"`
- **原因**：装了 toolkit ≠ 注册了 runtime；注册动作需要 root
- **解决**：`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`，
  验证：`docker info | grep -A3 Runtimes` 应包含 `nvidia`

### 4.3 坑③：ollama 抢占显存与算力

- **现象**：ollama 常驻两个 llama-server 进程（qwen3.8:27b MTP 草稿 + embedding），占 ~49 GB 显存、GPU 利用率 96%
- **影响**：仓库 README 的 0.85 利用率假设"空闲 Spark"；有竞争服务时 start.sh 会探测剩余显存并调低
  `--gpu-memory-utilization`（KV cache 变小），且推理与 ollama 抢算力
- **解决**：部署期间 `sudo systemctl stop ollama`（服务 disabled，不会自启）；
  之后需要时 `sudo systemctl start ollama` 恢复。两者可共存，只是 KV cache 会缩水

### 4.4 坑④（最隐蔽）：start.sh 三处 `set -e` 崩溃 —— 宿主无 vllm/torch 时必炸

- **现象**：`./start.sh` 在打印 `Using cached weights: ...` 后**立即**退出（exit 1），无任何报错输出
- **原因**：start.sh 用 `set -euo pipefail`，而作者假设宿主机装了 vllm（其 shebang 里的 python 自带 torch）。
  本机没有，导致三处命令替换/读取在失败时触发 `set -e` 直接终止脚本：

  | 行 | 代码 | 失败条件 |
  |---|---|---|
  | ① | `PROBE_BIN="$(command -v vllm)"` | 宿主无 vllm → `command -v` 返回 1 → 赋值语句非零 → **脚本退出** |
  | ② | `DEVINFO="$("$PYBIN" -c 'import torch; ...' 2>/dev/null)"` | 宿主无 torch → python 退出码 1 → 赋值语句非零 → **脚本退出** |
  | ③ | `read -r FREE_MIB TOTAL_MIB <<< "$DEVINFO"` | 空输入 → read 返回 1 → **脚本退出**（即使前两处修好也会死在这） |

- **解决**：三处各加 `|| true`，让脚本走其**原本设计好的 fallback 路径**
  （探测失败 → 打印 `Warning: device memory probe failed; using 0.85.` → 用 0.85 继续）。
  空闲机器上 0.85 恰好就是 README 默认值，无任何性能损失。补丁内容见 §5

> 💡 通用教训：`VAR="$(cmd)"` 在 `set -e` 下，命令替换失败即脚本退出；
> 这类"探测型"命令替换务必 `|| true` 或放进条件上下文。

### 4.5 坑⑤：根路径 404（"为啥404"）

- **现象**：浏览器打开 `http://127.0.0.1:8888/` → 404
- **原因**：vLLM **刻意不提供根路径页面**，这是正常行为，不是服务故障
- **解决**：可视化用 `http://127.0.0.1:8888/docs`（Swagger UI，可 Try it out）；
  编程对接用 `http://127.0.0.1:8888/v1`；健康检查用 `/health`

### 4.6 坑⑥：服务仅绑定 127.0.0.1，局域网其他机器连不上

- **现象**：其他机器访问 `http://<本机IP>:8888` 连接被拒
- **原因**：start.sh 默认 `HOST=127.0.0.1`（`ss -tlnp` 可见仅监听回环）
- **解决**：`HOST=0.0.0.0 sg docker -c "./start.sh"`（环境变量可覆盖）；
  注意公开端口等于暴露服务，建议配合防火墙/认证

### 4.7 坑⑦：aarch64 无官方 CUDA torch 轮子 → 内存探测无法在宿主生效

- **现象**：`pip download torch --index-url https://download.pytorch.org/whl/cu128`
  在 aarch64 上返回 `No matching distribution found`
- **原因**：PyTorch 未给 linux-aarch64 发布带 CUDA 的 pip 轮子（DGX Spark 的 torch 需容器内自带）
- **解决**：不装宿主 torch，接受 start.sh 的 0.85 fallback（空闲机器上正确）；
  若 ollama 常驻需要精确探测，可在容器内（有 torch）取 `torch.cuda.mem_get_info()` 的数值

### 4.8 坑⑧：后台任务输出被管道缓冲，误判"卡死"

- **现象**：`docker pull ... | tail -15` 看不到任何进度，`docker system df` 显示 0B，误以为拉取卡死
- **原因**：管道让输出只在命令结束时出现；部分镜像层未完成时 `docker images` 也不显示
- **解决**：重跑时不加管道直接流式输出；以层状态（`Download complete` 等）判断真实进度

### 4.9 坑⑨：模型自称"Claude"

- **现象**：冒烟测试问"介绍你自己"，模型答"我是 Claude，由 Anthropic 开发"
- **原因**：Ornith 家族基于 Claude 输出蒸馏，身份偏向是模型行为，**不是部署问题**

---

## 5. start.sh 补丁明细（本机必打）

```diff
--- a/start.sh
+++ b/start.sh
@@ -60,7 +60,7 @@
-PROBE_BIN="$(command -v vllm)"
+PROBE_BIN="$(command -v vllm || true)"  # host without vllm: empty probe bin must not trip set -e
 if [[ -n "$PROBE_BIN" ]]; then
@@ -71,8 +71,8 @@
-  DEVINFO="$("$PYBIN" -c 'import torch; f,t=torch.cuda.mem_get_info(); print(int(f//1048576), int(t//1048576))' 2>/dev/null)"
-  read -r FREE_MIB TOTAL_MIB <<< "$DEVINFO"
+  DEVINFO="$("$PYBIN" -c 'import torch; f,t=torch.cuda.mem_get_info(); print(int(f//1048576), int(t//1048576))' 2>/dev/null)" || true  # host without torch: non-zero exit must not trip set -e
+  read -r FREE_MIB TOTAL_MIB <<< "$DEVINFO" || true  # host without torch: fall through to 0.85 below
   [[ -n "$FREE_MIB" && -n "$TOTAL_MIB" ]] && (( FREE_MIB > 64*1024 )) && break
```

> 当前工作副本已打补丁（`git diff` 即上述内容）；上游更新后需重新应用。
> 补丁不改变任何行为逻辑，仅让"探测失败"走 fallback。

---

## 6. 关键路径与产物

| 项目 | 位置 |
|---|---|
| 部署仓库（含补丁后的 start.sh / stop.sh / patches/） | `~/projects/Ornith-1.5-35B-A3B-DGX-Spark/` |
| hf CLI venv | `~/.venvs/hf/bin/hf`（v1.28.0） |
| HF 权重缓存（22 GB） | `~/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-NVFP4` |
| 权重快照 | `snapshots/0f0b1b59b879ccde1353e6ebd0fb10c204d4c544` |
| HF token | `~/.cache/huggingface/token` |
| Docker 镜像 | `eugr/spark-vllm-b12x:latest`（拉取 10.9 GB，解包 23 GB） |
| 容器 | `ornith-b12x-serve`（网络 host 模式，端口 8888） |
| 启动日志备份 | `/tmp/ornith-start.log`（本次部署） |

容器内 patch 挂载（验证：`docker exec ornith-b12x-serve ls -la ...`）：

```
patches/kernel.py     -> /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/kernel.py:ro   (488184 B 补丁版)
patches/route_pack.py -> /usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/route_pack.py:ro (17326 B 补丁版)
```

> 这两个 patch 是 b12x MoE 后端配合 CUDA graph 捕获的必需修复（W4A16 输出排空路径的
> `metadata_row` 未定义 + route-packing workspace 在捕获期间非法 `torch.empty` 分配），
> 镜像原版为同目录 `*.b12x` 文件（kernel.py 488202 B / route_pack.py 16821 B）。

---

## 7. 日常运维

```bash
# 查看日志（vLLM 运行日志）
sg docker -c "docker logs -f ornith-b12x-serve"

# 停止服务
cd ~/projects/Ornith-1.5-35B-A3B-DGX-Spark
sg docker -c "./stop.sh"

# 重启服务（先停 ollama 可获完整 KV cache；HOST 可覆盖监听地址）
export PATH="$HOME/.venvs/hf/bin:$PATH"
sg docker -c "./start.sh"

# 恢复 ollama
sudo systemctl start ollama

# 监控指标（KV cache 用量等 Prometheus 格式）
curl -s http://127.0.0.1:8888/metrics | grep '^vllm:kv_cache'
```

---

## 8. 常见问题排查表

| 现象 | 原因 | 处理 |
|---|---|---|
| `./start.sh` 无输出即退出 exit 1 | §4.4 的 set -e 三连坑 | 打 §5 补丁 |
| `docker: permission denied` | 用户不在 docker 组 / 旧会话 | `sg docker -c` 或新开终端 |
| `could not select device driver "nvidia"` | runtime 未注册 | §3.1 第 2 步 + 重启 docker |
| 根路径 404 | vLLM 无根页面（正常） | 用 `/docs`、`/v1/...`、`/health` |
| 其他机器连不上 | 仅监听 127.0.0.1 | `HOST=0.0.0.0 ./start.sh` |
| /health 绿但首次请求慢 | CUDA graph 预热 / kernel 编译 | 属正常，多等几秒 |
| 显存不足告警 / KV 变小 | 有竞争服务（ollama 等） | 停 ollama 再重启 |
| 模型答非所问/自称 Claude | 模型蒸馏来源特性 | 非故障，正常使用即可 |
| `hf download` 卡住 | 无 token 撞 401（README 原话） | 写 token 文件（模型公开仍建议有） |

---

## 9. 参考数据（本次部署实测 / README）

- 模型权重：3 个 safetensors 分片，共 18,683,860,336 B ≈ 18.7 GB（NVFP4）
- 运行时显存：模型加载 21.97 GiB；KV cache fp8_e4m3，块大小 2112 token
- 下载速度实测 ~8.5 MB/s（权重 18.7 GB 约 16 分钟；镜像 10.9 GB 约 20 分钟）
- 启动链路：容器起 → 权重加载 19 s / 模型总载 164 s → torch.compile + CUDA graph 捕获（数分钟）→ /health 200
- vLLM 版本：0.1.dev20003+gad848fc41.d20260815（容器内 CUDA 13.0.2）
- README 吞吐参考（单 Spark，MTP 开，256k 上下文）：1 流 ~86 tok/s，24 并发 ~440 tok/s 聚合
- 启动参数（日志中 non-default args 摘要）：`--spec-method mtp --spec-tokens 1 --max-num-seqs 24
  --moe-backend b12x --gpu-memory-utilization 0.85 --max-model-len 262144
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --reasoning-parser qwen3`

---

*文档生成时间：2026-08-22 · 部署环境：spark-XXXX (DGX Spark / GB10) · 上游仓库 commit：main 分支（部署当日最新）*
