# MiniCPM5-1B 判官试跑记录（2026-08-24）

> 主机：spark-XXXX（DGX Spark / GB10）· 前置：ornith 内存优化完成（预留 103.4→79.1 GiB）后有余量再试本地判官
> 结果：**部署成功、格式全过、判断力 65% 校准（不可用于生产路由）**，服务保留在 :8889 待命

---

## 1. 背景

- 2026-08-23 判官探索（见 LOCAL-JUDGE-LESSONS.md）结论：显存不够，Qwen3-8B-AWQ（5.7GB）放不下，方案搁置
- 2026-08-24 内存优化后：ornith 预留降至 79.1 GiB，设备 free 9.0 GiB + 可回收页缓存 ~26 GiB，8B 级可以放下
- 用户要求用**最新一代 MiniCPM** 与 Qwen3-8B 做判官 PK

## 2. 型号选型

| 候选 | 结论 |
|---|---|
| **MiniCPM5-1B**（2026 首发，1B 密集） | ✅ 选定：标准 LlamaForCausalLM（vLLM 原生，零 custom code）、BF16 1.08GB、128k 上下文、Apache-2.0、主打 agentic tool use |
| MiniCPM3-4B-GPTQ-Int4（2024，2.2GB） | 备选：自定义架构需 trust-remote-code，旧一代 |
| MiniCPM-V 4.0 / V-4.6 1.3B | 多模态视觉模型，纯文本判官无加成，排除 |

MiniCPM5 系列目前只有 1B（无更大版本）。

## 3. 部署过程

### 3.1 下载

```bash
HF_HUB_DISABLE_XET=1 /home/<user>/.venvs/hf/bin/hf download openbmb/MiniCPM5-1B \
  --local-dir /home/<user>/projects/judge-models/MiniCPM5-1B   # 11 文件，56s
```

（xet 后端写 ~/.cache/huggingface 被沙箱挡，用 HF_HUB_DISABLE_XET=1 + --local-dir 到 workspace 内）

### 3.2 启动脚本

`/home/<user>/projects/start-judge-minicpm.sh`（vLLM 0.27.1 用户态 venv，:8889）：

```bash
export VLLM_CACHE_ROOT="/home/<user>/projects/judge-models/.vllm-cache"   # ~/.cache/vllm 被 ornith 容器以 root 写过，宿主不可写
export PATH="/home/<user>/projects/venv-vllm/bin:$PATH"                    # flashinfer JIT 需要 ninja
/home/<user>/projects/venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/<user>/projects/judge-models/MiniCPM5-1B \
  --served-model-name minicpm5-1b --host 127.0.0.1 --port 8889 \
  --gpu-memory-utilization 0.10 --max-model-len 16384 \
  --trust-remote-code --api-key skip
```

启动实测：加载 2.09 GiB（22.8s）→ torch.compile 9.2s → flashinfer autotune → CUDA graph 捕获（51+35，6s，0.57 GiB）→ KV 池 9.31 GiB（util 0.10 预算 12.17 GiB）→ 总计约 2 分钟上线。显存占用 ~12.2 GiB（预留口径）。

### 3.3 踩坑记录（重要）

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| 1 | `Failed to infer device type` | 沙箱拦 GPU ioctl（Error 304），**每次启动命令都需要 danger-full-access 授权** | 启动命令带 sandbox_permissions 重试 |
| 2 | `PermissionError: ~/.cache/vllm/torch_compile_cache` | 目录被 ornith 容器（root）写过，宿主 <user> 无写权限 | `VLLM_CACHE_ROOT` 指向 workspace 内目录 |
| 3 | `FileNotFoundError: 'ninja'`（flashinfer JIT 编译采样 kernel 时） | ninja 在 venv 里但不在 PATH | 脚本里 `export PATH=venv/bin:$PATH` |
| 4 | API 返回 `Unauthorized` | `--api-key skip` 启用鉴权 | 请求带 `Authorization: Bearer skip` |
| 5 | 测试脚本 SyntaxError `'{' was never closed` | **原版 test-judge-schema.py 的 schema 字典嵌套 5 层但行尾只有 4 个 `}`**（文件自带 bug） | 补一个 `}` |
| 6 | `enable_thinking` 参数无效 | chat_template 在独立 `chat_template.jinja` 文件里，vLLM 0.27.1 未加载（tokenizer_config.json 无 chat_template 字段） | 未处理（判官场景影响不大） |

## 4. 评测（判官契约：strict JSON schema，4 字段 crux/primary_rule/capability_boundary/p_solve）

### 4.1 提示词演进

| 版本 | 说明 | 校准命中率（20 任务，2 轮） |
|---|---|---|
| V1 无示例 | 原始系统提示 | **26%**（4/15 首轮 15 任务） |
| V2 few-shot | 3 个示例（全高分档） | 锚定过头：全判 supported/0.95+，黎曼猜想给 0.99 |
| **V3 难度映射** | 显式"难度分级→分数区间"映射 + 3 档校准锚 | **65%**（13/20） |
| V3.1 微调 | 换 5 个更贴任务的示例 + 防复制指令 | **65%**（13/20），错误模式漂移 |

### 4.2 V3 关键数据（2 轮平均）

- 超难全对：黎曼 0.0 ✅、整本《战争与和平》0.0 ✅、x86 bootloader 0.0 ✅
- 简单大多对：7×8 0.97 ✅、翻译 0.95 ✅、邮件 0.97 ✅
- 错题：一句话总结 0.0 ❌、LRU 缓存 0.0 ❌、numpy 手写 transformer 0.95 ❌、无锁队列 0.85 ❌
- V3.1 修好总结/LRU，但 7×8 掉到 0.6、整本书翻译飙到 0.97 —— **错题随机漂移，不可预测**

### 4.3 结论

- ✅ 格式：strict JSON schema **3/3 全过**（vLLM guided decoding 兜底），1B 模型格式不再翻车（修正了 2026-08-23"4B 都翻车"的旧结论——那是模型+无引导的问题）
- ❌ 判断：65% 校准是提示词天花板；概率判断随机波动大（同题两轮 0↔0.99）、crux 复制示例文本（贴标签而非理解）、few-shot 锚定效应强
- **判官不可用**：路由判官要求 p_solve 可信，65% + 随机错题会把 ~1/3 流量错分
- 印证旧结论："8B 是判断力分水岭"；MiniCPM5 新代际解决的是格式/指令遵循，不是 1B 的判断力上限

## 5. 运维

```bash
# 启动（需 GPU 授权；当前服务在跑，健康检查）
bash /home/<user>/projects/start-judge-minicpm.sh     # 日志 /tmp/vllm-minicpm.log
curl -s http://127.0.0.1:8889/health

# 停止
pkill -f 'vllm.entrypoints.*8889'

# 评测脚本
/home/<user>/projects/venv-vllm/bin/python /tmp/test-judge-v3.py    # 20 任务校准测试
```

模型文件：`/home/<user>/projects/judge-models/MiniCPM5-1B`（1.1GB）· 缓存：`judge-models/.vllm-cache`

## 6. 后续可选

1. ~~**Qwen3-8B-AWQ 对比**（5.7GB，未下载）~~ **已完成（2026-08-24）**：Qwen3-8B-AWQ 同套测试 17/20（85%），判定可用。完整 PK 记录与接入预案见 [judge-pk-result.md](judge-pk-result.md)
2. MiniCPM5-1B 换用途：本地草稿模型 / 低风险分类（非概率路由）
3. 若 Qwen3-8B 达标：接入 switchyard llm_classifier（routes.toml target + session_affinity），judge 失败 fail-open 到云端
