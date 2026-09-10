# DGX Spark 模型服务记录

> 记录日期：2026-08-23 · 主机：`spark-XXXX`（NVIDIA DGX Spark / GB10，aarch64，Ubuntu 6.17-nvidia 内核）
> 本文件记录：本机当前运行的模型服务、对外暴露方式（本次改动）、访问方式与维护注意事项。

---

## 1. 当前运行的模型服务

| 项目 | 值 |
|---|---|
| 容器 | `ornith-b12x-serve`（镜像 `eugr/spark-vllm-b12x:latest`） |
| 模型 | `ornith-ai/Ornith-1.5-35B-A3B-NVFP4`（35B 总参数 / 3B 激活，MoE，NVFP4 量化，Qwen3 系架构） |
| served 名称 | `ornith-1.5-35b-a3b-nvfp4` |
| 服务框架 | vLLM（OpenAI 兼容 API），容器内 `vllm serve` |
| 监听 | `127.0.0.1:8888`（容器为 **host 网络模式**，直接绑宿主机） |
| 启动方式 | 手动 `docker run`，脚本见 [start.sh](Ornith-1.5-35B-A3B-DGX-Spark/start.sh)（同目录有 `stop.sh`）；容器 **RestartPolicy = no**，重启机器后不会自动拉起 |

### 关键启动参数（vLLM CMD）

```
vllm serve <模型路径> \
  --served-model-name ornith-1.5-35b-a3b-nvfp4 \
  --host 127.0.0.1 --port 8888 \
  --trust-remote-code --enable-prefix-caching \
  --spec-method mtp --spec-tokens 1          # MTP 投机解码（1 个草稿 token）\
  --max-num-seqs 4                            # 并发：24 → 4（2026-08-23 起）\
  --moe-backend b12x                          # Blackwell 张量核优化 MoE 后端（DGX Spark 专属）\
  --gpu-memory-utilization 0.65 \                        # 0.85~0.90 → 0.65（2026-08-24 内存优化，见 ornith-memory-optimization.md）\
  --max-model-len 1048576                     # 上下文：256K → 1M（2026-08-23 起）\
  # 环境变量：VLLM_ALLOW_LONG_MAX_MODEL_LEN=1（模型 max_position_embeddings=262144，超限必须设）\
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

挂载：HF 缓存（`~/.cache/huggingface`）、vLLM 缓存（`~/.cache/vllm`）、b12x kernel 补丁（`patches/kernel.py`、`patches/route_pack.py`，只读）。

### 运行状态（2026-08-23 记录）

- 生成吞吐约 80 tokens/s，prompt 吞吐约 900–1400 tokens/s
- MTP 接受率约 70%，KV cache 占用 <1%，前缀缓存命中率约 83%

---

## 2. 服务依赖关系（重要）

本机推理链路：

```
DSH Web GUI (127.0.0.1:3080)
  └─> switchyard-server (0.0.0.0:4000，配置 /home/<user>/routes.toml)
        └─> routes.smart = stage_router（efficient_first，confidence_threshold 0.05）
              ├─> 本地 ornith   ← http://localhost:8888/v1（llm_client vllm_ornith）
              └─> openrouter deepseek-v4-flash（llm_client openrouter）
```

**当前对话/日常推理默认走本地 ornith**（smart 路由优先选 efficient 目标）。因此：

- 修改 vLLM 绑定、重启容器会**中断正在使用的服务**（包括本对话的后端）；
- 本次对外暴露选择零停机方案（见下），没有动容器。

---

## 3. 本次改动：把 8888 端口放开给其他机器

### 背景

vLLM 只绑 `127.0.0.1:8888`，其他机器无法访问。直接改 `--host 0.0.0.0` 需要重建容器（有停机 + 中断当前会话的风险），故采用**宿主机 TCP 转发代理**：

- 监听宿主机全部外部网卡 IP 的 `8888` 端口（与 `127.0.0.1:8888` 不冲突），
- 将连接原样转发到 `127.0.0.1:8888` 的 vLLM；
- 对 vLLM 进程零影响，不用重载模型。

### 文件

| 文件 | 说明 |
|---|---|
| `vllm-open-proxy.py` | TCP 转发代理脚本（python3，仅标准库） |
| `vllm-proxy.log` | 代理运行日志 |

### 管理命令

```bash
# 启动（当前已在运行，PID 见 pgrep）
cd /home/<user>/projects && setsid nohup python3 vllm-open-proxy.py > vllm-proxy.log 2>&1 &

# 查看状态
pgrep -af vllm-open-proxy.py && tail -f vllm-proxy.log

# 停止
pkill -f vllm-open-proxy.py
```

### 验证结果（2026-08-23）

- `/v1/models` 在 3 个外部 IP 上均返回 HTTP 200
- 通过 Tailscale IP 完成了一次真实 `/v1/chat/completions` 推理（正常返回）

---

## 4. 其他机器的访问方式（OpenAI 兼容 API）

| 网络 | Base URL |
|---|---|
| 本机 | `http://127.0.0.1:8888/v1` |
| 局域网（有线 `<LAN-IP-2>` / 无线 `<LAN-IP-1>`） | `http://<LAN-IP-2>:8888/v1` |
| Tailscale | `http://<Tailscale-IP>:8888/v1` |

模型 ID：`ornith-1.5-35b-a3b-nvfp4`。

---

## 4.5 1M 上下文扩展（2026-08-23）

### 为什么可行

模型是 Qwen3.5 混合架构：40 层中只有 **10 层 full attention**（KV 随长度增长），
其余 30 层是 Gated DeltaNet 线性注意力（KV 为固定大小状态，不随长度增长）。
加上 KV cache 自动使用 **fp8_e4m3**（每 token ~12KB 含 padding 开销），
1M 上下文 × 4 并发仅需 KV 池 ~50GB，而当前 0.85 util 预留 75.95 GiB（实测可存 7.1M token，
1M 并发容量 6.77x）。

### 改动（已生效）

```bash
# start.sh 新增环境变量（均可覆盖）：
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
# docker run 增加：
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# 另：hf CLI 不在宿主机 PATH 时自动跳过下载步骤（权重已在缓存）
```

### 实测验证（2026-08-23）

| 测试 | 结果 |
|---|---|
| 超限拒绝 | 1,048,545 token 请求被准确拒绝（边界正确） |
| 680,059 token 请求 | 完整运行，无报错（远超 256K 训练上限，RoPE 外推正常） |
| 长距离 recall | 从 680K token 之前的文本中正确回忆出秘密数字 `918273` |
| 前缀缓存 | 同前缀二次请求 12s 完成（首次冷 prefill ~14min，主要是 JIT 编译） |
| KV 池 | 7,103,791 token，1M 并发容量 6.77x（4 并发绰绰有余） |

### 注意事项

- 1M 是 4 倍 RoPE 外推（训练上限 256K），长程注意力质量可能略降；实测 680K recall 正常。
- 超长请求首次 prefill 慢（JIT 编译 + 长 prefill），同前缀复用极快。
- 剩余显存 ~40GB，仍可再挂一个 30B 级模型（见第 6 节思路）。
- 回退方案：`MAX_MODEL_LEN=262144 MAX_NUM_SEQS=24 ./start.sh`。

### 4.6 内存优化（2026-08-24）

- **问题**：整机 used 109/121 GiB、swap 挤出 4.2 GiB；vLLM 按 util 0.85 启动时预分配 103.4 GiB，其中 KV 池 75.95 GiB（容量 6.77× 并发 1M 请求），实际 KV 使用率 0%。
- **改动**：`start.sh` 支持 `GPU_UTIL` 环境变量覆盖（显式指定时跳过自动探测，未指定时行为不变），本次以 `GPU_UTIL=0.65 ./start.sh` 重启：

  | 指标 | 之前 (0.85) | 之后 (0.65) |
  |---|---|---|
  | vLLM 总预留 | 103.4 GiB | 79.1 GiB |
  | KV 池 | 75.95 GiB / 7.10M tokens | 51.88 GiB / 4.85M tokens（4.63× 1M 并发 > 实际 4 并发） |
  | 整机 used | 109 GiB | 86 GiB（可用 12→35 GiB） |

- 完整过程、方案对比、回滚方法与后续可选优化见 [ornith-memory-optimization.md](notes/ornith-memory-optimization.md)。

---

## 5. 已知注意事项

1. **无鉴权**：vLLM 未要求 API key（裸请求 200）。局域网内知道 IP 即可调用，如介意可加防火墙限制，例如只放行 Tailscale 网段：
   ```bash
   sudo ufw allow from 100.64.0.0/10 to any port 8888
   ```
2. **ufw 处于 active**：本机测试外部 IP 可达；若其他机器连不上，先检查 ufw 规则（`sudo ufw status`）。
3. **重启后不自动拉起**：代理和容器（RestartPolicy=no）在机器重启后都需要手动启动：
   ```bash
   cd /home/<user>/projects/Ornith-1.5-35B-A3B-DGX-Spark && ./start.sh   # 先起模型
   cd /home/<user>/projects && setsid nohup python3 vllm-open-proxy.py > vllm-proxy.log 2>&1 &  # 再起代理
   ```

---

## 6. 后续可选优化（维护窗口再做）

**Qwen3.8-27B 替代评估（2026-08-24）**：调研后结论为**保持 ornith 现状**（性能/内存/能力均不支持替代，且会破坏判官方案）。完整对比数据与未来试用路径见 [qwen38-27b-alternative-eval.md](notes/qwen38-27b-alternative-eval.md)。

**彻底方案**：重建容器时把 vLLM 绑到 `0.0.0.0`，之后可去掉代理。`start.sh` 已支持 `HOST` 环境变量：

```bash
HOST=0.0.0.0 ./start.sh   # 容器内 vllm 以 --host 0.0.0.0 启动
```

注意：`start.sh` 会 `docker rm -f` 旧容器并重载模型（约 1–2 分钟停机），且有"按空闲显存下调 gpu-memory-utilization"的逻辑，需要机器空闲时执行。

---

## 7. 参考

- 模型部署仓库：`~/projects/Ornith-1.5-35B-A3B-DGX-Spark/`（`README.md`、`DEPLOYMENT.md`、`start.sh`、`stop.sh`）
- switchyard 路由配置：`~/routes.toml`
- 本机 IP：有线 `<LAN-IP-2>`、无线 `<LAN-IP-1>`、Tailscale `<Tailscale-IP>`
