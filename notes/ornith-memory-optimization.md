# Ornith vLLM 内存优化工作记录

> 生成时间：2026-08-24 · 主机：spark-XXXX（DGX Spark / GB10）
> 目标：解决 ornith 容器占满 ~120 GiB 统一内存的问题
> 状态：**已完成并生效** —— vLLM 预留 103.4→79.1 GiB，整机 used 109→86 GiB，能力不变（仍支持 4 并发 × 1M 上下文）

---

## 1. 问题现象

- `free -h`：整机 **used 109 GiB / 121 GiB**，可用仅 12 GiB，**swap 已被挤出 4.2 GiB**
- 模型为 35B-A3B MoE + NVFP4 量化（权重 ~22 GB），推算不应吃这么多内存，怀疑异常

## 2. 诊断：内存去哪了

从 vLLM 启动日志（08-23 09:09）与 /metrics 取数：

| 项目 | 大小 |
|---|---|
| 权重 + 非 torch 开销 | 25.86 GiB |
| 峰值 activation | 2.63 GiB |
| CUDA graph | ~0 GiB |
| **KV cache 池（预留）** | **75.95 GiB** = 7,103,791 tokens = **6.77 个并发满 1M 上下文请求** |
| 0.85 util 总预留 | 103.44 GiB / 121.69 GiB |
| 宿主进程合计 | ~8 GiB |

- /metrics 显示 `kv_cache_usage_perc = 0.0`（KV 池实际使用率 **0%**）
- **结论**：vLLM 按 `--gpu-memory-utilization 0.85` 在启动时**一次性预分配** ~103 GiB（CUDA 预占，非按需增长），其中 76 GiB 是 KV 池，按"6.77 个并发满 1M 上下文"设计，而实际部署只开 4 并发。
  **120G 是预留过大，不是模型本身占内存。**

## 3. 优化方案对比

实际需求：4 并发 × 1M tokens = 4.19M tokens ≈ 45 GiB KV 池（fp8 KV + float32 SSM 状态，~10.7 KiB/token）。

| 方案 | 改动 | 总预留 | KV 池 | 释放 | 说明 |
|---|---|---|---|---|---|
| **A. util 0.85→0.65** ✅已实施 | 1 行 | ~79 GiB | 50.8 GiB（4.75M tok，13% 余量） | **~24 GiB** | 保留 4×1M，最保守 |
| B. 回 256k + util 0.40 | 2 行 | ~43-49 GiB | ~20 GiB（1.9M tok） | **~50-60 GiB** | 1M 是 4 倍 RoPE 外推实验，日常流量用不上 |
| C. `--kv-cache-memory-bytes` | 1 行 | 精确指定 | 精确指定 | 任意 | vLLM 日志会给出换算值 |
| D. `--mamba-ssm-cache-dtype bfloat16` | 1 行 | 固定状态减半 | — | 几个 GiB | 30/40 层线性注意力状态不随上下文增长，收益有限 |
| E. 关 MTP | 1 行 | ~1-2 GiB | — | 小 | 不推荐，吞吐直接减半 |

KV dtype 已是 fp8_e4m3、权重已是 NVFP4，这两块无压缩空间。

## 4. 实施过程（2026-08-24）

### 4.1 start.sh 补丁

`~/projects/Ornith-1.5-35B-A3B-DGX-Spark/start.sh`：

- `GPU_UTIL=0.85` → `GPU_UTIL="${GPU_UTIL:-}"`：支持环境变量覆盖（`GPU_UTIL=0.65 ./start.sh`）
- 自动探测逻辑包进 `if [[ -z "$GPU_UTIL" ]]`：**显式指定时跳过探测**（否则探测会把 0.65 顶回 0.85）
- 未指定时行为与原来完全一致（探测空闲内存 → 0.85 默认 → 空闲时自动 0.85）
- 校验：`bash -n` 通过

### 4.2 重启

```bash
cd ~/projects/Ornith-1.5-35B-A3B-DGX-Spark
GPU_UTIL=0.65 ./start.sh
```

- 日志确认 `Using --gpu-memory-utilization 0.65.`
- 启动耗时 ~3 分钟（权重加载 2:34 + torch.compile / CUDA graph 捕获）
- ⚠️ 重启会中断经过 ornith 的服务（本机 DSH/switchyard 路由可能指向它），需在维护窗口执行

## 5. 效果验证

| 指标 | 之前 (0.85) | 之后 (0.65) |
|---|---|---|
| vLLM 总预留 | 103.4 GiB | **79.1 GiB** |
| KV 池 | 75.95 GiB / 7.10M tokens | **51.88 GiB / 4.85M tokens**（4.63× 1M 并发 > 实际 4 并发） |
| 整机 used | 109 GiB | **86 GiB** |
| 整机 available | 12 GiB | **35 GiB** |
| buff/cache | 8.7 GiB | 25 GiB |
| swap | 4.2 GiB 被挤出 | 逐步回收 |

- 冒烟测试：`/health` 200；`/v1/chat/completions` 正常返回（reasoning 字段正常，MTP 解析器工作）

## 6. 回滚

```bash
cd ~/projects/Ornith-1.5-35B-A3B-DGX-Spark
GPU_UTIL=0.85 ./start.sh    # 或不带 GPU_UTIL（自动探测，空闲机器即 0.85）
```

## 7. 后续可选优化（未实施）

- **方案B**：`MAX_MODEL_LEN=262144 GPU_UTIL=0.40 ./start.sh` —— 整机占用可再降到 ~60 GiB
- **精确控制**：以 `--kv-cache-memory-bytes=<bytes>` 替换 util
- **小项**：`--mamba-ssm-cache-dtype bfloat16`（容器已支持该 flag）
