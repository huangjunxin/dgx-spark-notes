# 判官 PK 最终记录：MiniCPM5-1B vs Qwen3-8B-AWQ（2026-08-24）

> 主机：spark-XXXX（DGX Spark / GB10）· 前置：ornith 内存优化（预留 103.4→79.1 GiB）+ MiniCPM5-1B 试跑（见 minicpm-judge-trial.md）
> 结果：**Qwen3-8B-AWQ 胜出（85% 校准）**，判定为可用的本地路由判官；接入方案见 §5

---

## 1. PK 结论（省流版）

| 模型 | 体积 | 校准命中率（20 任务×2 轮） | 灾难性错误 | crux 理解 | 结论 |
|---|---|---|---|---|---|
| MiniCPM5-1B（BF16） | 1.1 GB | **65%**（13/20） | 随机出现（easy 给 0 / impossible 给 0.99） | 复制示例文本 | ❌ 不可用 |
| **Qwen3-8B-AWQ** | **5.7 GB** | **85%**（17/20） | **0 个** | 真实描述任务 | ✅ **可用** |

- 同套 V3 提示词（难度分级→分数映射 + 5 档校准锚）、同套 20 任务、temperature 0.0、两轮取均值
- 验证了 LESSONS 的旧结论："8B 是判断力分水岭"；也验证了内存优化后 8B 级 judge 可部署（util 0.10 预算 12.2 GiB，实际权重 5.7 GiB）

## 2. Qwen3-8B 判断质量（20 任务梯度）

| 难度档 | 示例任务 | p_solve |
|---|---|---|
| trivial | 7×8 | 0.97 |
| easy | 翻译 / 邮件 / 总结 / todolist | 0.95 |
| medium | SQL / 正则 / LRU / 编辑距离 / 快排 | 0.55-0.85 |
| hard | numpy transformer / 无锁队列 / osdev | 0.35 |
| impossible | 黎曼猜想 / 整本书翻译 | 0.00-0.01 |

- 错题 3 个均为"中等偏难"边界微差（±0.05-0.15）：lockfree 0.35、dpknapsack 0.75、osdev 0.35
- crux 全部真实（'easy: short translation'、'medium: standard SQL pattern'、'hard: complex systems design'）

## 3. 部署记录

### 3.1 下载

```bash
HF_HUB_DISABLE_XET=1 /home/<user>/.venvs/hf/bin/hf download Qwen/Qwen3-8B-AWQ \
  --local-dir /home/<user>/projects/judge-models/Qwen3-8B-AWQ   # 12 文件，59s
```

### 3.2 启动（start-judge-qwen3-8b.sh，:8889）

```bash
export VLLM_CACHE_ROOT="/home/<user>/projects/judge-models/.vllm-cache"
export PATH="/home/<user>/projects/venv-vllm/bin:$PATH"   # ninja for flashinfer JIT
/home/<user>/projects/venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/<user>/projects/judge-models/Qwen3-8B-AWQ \
  --served-model-name qwen3-8b-awq --host 127.0.0.1 --port 8889 \
  --gpu-memory-utilization 0.10 --max-model-len 16384 \
  --quantization awq --trust-remote-code --api-key skip
```

实测：加载 5.68 GiB → torch.compile → flashinfer autotune → CUDA graph 捕获（51+35）→ KV 池 0.26 GiB（25,424 tokens，判官短请求绰绰有余）→ 总计约 3 分钟上线。启动前需停掉 MiniCPM judge 腾空间（或降其 util）。

### 3.3 关键踩坑：进程"无声消失"根因（新教训）

- 现象：两次启动，进程在加载权重阶段无错误日志地消失
- 根因：**DSH bash 前台命令有超时（~90s），vLLM 启动需 2-3 分钟；命令超时被杀时连带杀掉了授权命令启动的进程树**
- 对照：MiniCPM 成功那次命令仅 ~35s（未超时）；两次失败命令均 90s+（超时）
- 解决：**用后台任务（run_in_background）+ GPU 授权启动**——后台任务无超时，启动命令返回后进程稳定存活
- 附带教训：pkill -f 模式匹配自身包装进程（第二次踩坑），用 'xxx[9]' 括号技巧规避

## 4. 当前服务状态（2026-08-24 收尾时）

| 服务 | 端口 | 状态 | 占用 |
|---|---|---|---|
| ornith-1.5-35B-A3B（NVFP4，util 0.65） | :8888 | ✅ 200 | 79.1 GiB 预留 |
| Qwen3-8B-AWQ 判官 | :8889 | ✅ 200 | ~12.2 GiB 预算（权重 5.7 + KV 0.26） |
| MiniCPM5-1B 判官 | 已停 | — | 可随时恢复（start-judge-minicpm.sh） |

整机：used 101 GiB / 121 GiB，available 19 GiB。

## 5. switchyard 接入预案（未执行）

目标：routes.toml 的 smart 路由加 llm_classifier 档，判官按 p_solve 分流：易→本地 ornith，难→deepseek（OpenRouter）。

```bash
# 1) routes.toml 中 llm_classifier 的 target 指向判官
#    模型名 qwen3-8b-awq，base_url http://127.0.0.1:8889/v1
#    （llm_classifier 契约：strict json_schema，4 字段 crux/primary_rule/capability_boundary/p_solve；
#     判官失败 fail-open 到 strong/云端——已由 Qwen3-8B 的 strict 输出兜底）

# 2) 判官请求配置建议
#    extra_body.chat_template_kwargs.enable_thinking=false（Qwen3 系生效）
#    temperature 0.0，max_tokens 400
#    session_affinity=true（每会话只判一次）

# 3) 阈值建议（基于 20 任务实测）
#    p_solve >= 0.7 → 本地 ornith（trivial/easy/多数 medium 全落本地）
#    p_solve < 0.7 → 云端 deepseek（hard/impossible）
#    （V3 提示词下 0.7 分割线在实测中与期望区间吻合度最高）
```

注意事项：
- 接入会改动 switchyard 路由配置，需维护窗口操作（当前对话流量可能经过路由）
- 判官首请求有 CUDA graph 预热延迟（~1s 内）；后续稳定
- 若判官不可用（崩溃/重启），switchyard 应 fail-open 到云端，不影响主链路

## 6. 运维速查

```bash
# Qwen3-8B 判官
bash /home/<user>/projects/start-judge-qwen3-8b.sh   # 需 GPU 授权 + 后台任务方式（见 §3.3）
curl -s http://127.0.0.1:8889/health
# 评测：/home/<user>/projects/venv-vllm/bin/python /tmp/test-judge-v3.py（MODEL 改 qwen3-8b-awq）

# MiniCPM5-1B 判官（备胎）
bash /home/<user>/projects/start-judge-minicpm.sh    # 与 Qwen 判官抢 :8889，二选一

# 停止判官
pkill -f 'vllm.entrypoints.*888[9]'
```

## 7. 文件清单

| 文件 | 说明 |
|---|---|
| /home/<user>/projects/judge-models/Qwen3-8B-AWQ | Qwen3-8B-AWQ 权重（5.7GB） |
| /home/<user>/projects/judge-models/MiniCPM5-1B | MiniCPM5-1B 权重（1.1GB） |
| /home/<user>/projects/judge-models/.vllm-cache | vLLM compile/flashinfer 缓存 |
| /home/<user>/projects/start-judge-qwen3-8b.sh | Qwen 判官启动脚本 |
| /home/<user>/projects/start-judge-minicpm.sh | MiniCPM 判官启动脚本 |
| /tmp/test-judge-v3.py | 20 任务校准测试脚本（改 MODEL 切换被测模型） |
