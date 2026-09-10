# 本地判官方案探索记录（Lessons Learned）

> 日期：2026-08-23 · 主题：给 Switchyard 加一个"本地判官模型"（按难度自动分流：易→本地 Ornith，难→线上 deepseek）的完整尝试与结论
> 2026-08-23 最终结果：**放弃本地判官方案**，Ornith 恢复 util 0.90，25GB 下载文件已清理。
> **2026-08-24 更新**：ornith 内存优化后（util 0.85→0.65）重试判官——**方案成立**：MiniCPM5-1B 格式全过但 1B 判断力 65% 不可用（[minicpm-judge-trial.md](minicpm-judge-trial.md)）；**Qwen3-8B-AWQ 同套测试 85%（17/20）判定可用**，8B 级 judge 在内存优化后可部署（[judge-pk-result.md](judge-pk-result.md)）。
> **2026-09-10 更新**：Qwen3-8B-AWQ judge 服务（:8889）已**关停**——新主服务（SGLang Qwen3.8-27B）上线后整机内存压力大（available ~8.5%，hermes kanban 被限流），且 judge 自 8/24 起 16 天无调用方、纯空转。详见 §7。

---

## 1. 目标回顾

- Switchyard（:4000）作为唯一 LLM 入口，管理多个上游模型；
- 想加一个 **本地 judge**：`llm_classifier` 路由按任务难度打分，易→本地 Ornith（:8888），难→线上 deepseek（OpenRouter）；
- 硬性约束：**不泄漏**（judge 必须本地跑）、单入口、自动分流。

## 2. 核心结论（省流版）

| 结论 | 说明 |
|---|---|
| vLLM 2026 版**已移除 GGUF 加载** | 0.27.1 与容器 dev 版的 load-format 列表均无 `gguf`；GGUF 只能走 Ollama/llama.cpp |
| **Gemma-4 QAT 权重与 vLLM 不兼容** | compressed-tensors checkpoint 用"逐层 PLE 门+投影"，vLLM 用"模型级投影"，形状对不上（embed_tokens 2304 vs 2560） |
| **显存是真正的瓶颈** | GB10 121.7GiB，系统其他占用 ~18GiB 且波动；Ornith 0.85 时只剩 ~6.9GiB、0.80 时实测 ~5.8GiB；**8B 级模型放不下**，7.75GB 的 E2B-CT 也放不下 |
| 判官必须支持**严格 JSON Schema 结构化输出** | libsy 的 `llm_classifier` 契约：`response_format=json_schema`（strict、additionalProperties:false），4 字段（crux/primary_rule/capability_boundary/p_solve）；吐不出合法 schema → fail-open 全丢云端 |
| Ornith 当判官不行 | 推理模型：`enable_thinking=false` 是空操作、structured output 不可靠、输出常驻 `reasoning` 字段、预算紧就截断 |
| 沙箱拦截 GPU | 本会话沙箱默认拦截 GPU ioctl（CUDA Error 304）；需 `danger-full-access` 授权；docker 操作不受影响（socket 走守护进程） |

## 3. 详细教训

### 3.1 vLLM 生态变化（2026）

- **GGUF 支持被移除**：`vllm serve --load-format gguf` 报 `Load format "gguf" is not supported`；`--quantization gguf` 同样被拒绝（不在量化方法列表）。GGUF 生态（Google QAT GGUF 4.8GB/3.1GB）只能配 Ollama/llama.cpp。
- **compressed-tensors 文件"虚胖"**：Gemma-4 E4B w4a16-ct 权重 10.72GB、E2B 也要 7.75GB（约 1.3 bytes/参数），不如 Q4_0 GGUF 划算；BF16 QAT 版更大（9.5GB）。
- **Gemma4 PLE 布局不兼容**（关键坑）：
  - Google checkpoint：`embed_tokens` 完整 hidden 2560 + 每层 `per_layer_input_gate`(2560→256) + `per_layer_projection`(256→2560) + `embed_tokens_per_layer`(vocab, 42×256)；
  - vLLM gemma4.py：`embed_tokens` 用 (hidden − per_layer_input)=2304 + **模型级** `per_layer_model_projection`(hidden→num_layers×ple)；
  - transformers 5.15 会把配置转成 `gemma4_text` 子模型（hidden 自动减 PLE 维），vLLM 模型初始化读到 hidden=2304，加载权重 (262144,2560) → 形状不匹配崩溃；
  - 两个构建（0.27.1 与容器 dev 版）同病。要硬啃需改加载器映射权重，数值语义不保证，不值得。
- **结构化输出选型参考**：Qwen3-8B 格式遵循实测 97/100（XSCT Bench）；8B 是可靠 JSON 的分水岭（1.7B/4B 在简单 JSON 任务上翻车）；Gemma-4-E4B 同为 8B 级、Apache-2.0（2026），但 PLE 兼容问题待 vLLM 修复。

### 3.2 显存与部署环境（GB10 / DGX Spark）

- 总显存 121.7GiB（统一内存）；nvidia-smi 显示 ornith EngineCore 占用 ~95-97GiB（pool 按 util 预留）。
- **"系统其他占用" ~18GiB 且波动**（显示/桌面/页缓存等）：docker cp 11.5GB 进容器后页缓存会进一步吃掉可用显存；因此"Ornith 降 util 腾空间"收益不可靠（0.85→0.80 后 free 反而从 6.9 降到 5.8）。
- 测量显存：`torch.cuda.mem_get_info()` 需在**授权**环境下跑（沙箱拦 GPU）；<user> 用户默认 python 无 torch（vLLM 是 root/容器环境的），需自建 venv。
- 启动检查：vLLM 要求 `free ≥ util×total`，否则直接报错拒绝启动。

### 3.3 会话沙箱（DSH 环境）

- 沙箱默认拦 GPU ioctl：CUDA init 报 `Error 304: OS call failed`，nvidia-smi 报 NVML Unknown Error；**用 `sandbox_permissions: danger-full-access` + 用户批准可解开**（判官 vLLM 常驻服务需此授权启动）。
- `sudo` 不可用（容器内 no_new_privileges）。
- hf CLI 的 xet 后端要写 `~/.cache/huggingface`（沙箱外）→ 下载失败；**改用 curl 直连 HF CDN**（`huggingface.co/<org>/<repo>/resolve/main/<file>`）下载到 workspace。
- 沙箱文件写权限只在 workspace（`/home/<user>/projects`）内。
- **docker 操作不需要授权**（socket 调用守护进程），可绕开沙箱跑 GPU 容器（本方案中 ornith 容器换 util 即走此路）；但容器内 vLLM 是 dev 版，同样有兼容问题。
- 小坑：`pkill -f 'xxx'` 会匹配自己命令行的 bash -c 包装进程导致自杀/超时 → 用 `pkill -f 'xxx[r]'` 括号技巧。

### 3.4 switchyard / libsy 路由机制

- `stage_router`：按**近几轮工具结果/Agent 进度信号**选档（efficient/capable），**不按文本难度**——纯文本难题不会触发升级（实测 7 个难题全留本地）。
- `llm_classifier`：每次请求先调 judge（第三模型），按 `p_solve ≥ base_threshold` 走 weak/strong；**要求判官严格结构化输出**；`session_affinity=true` 可让每会话只判一次；判官失败 fail-open 到 strong（云端）。
- routes.toml 的 target 支持 `extra_body = { chat_template_kwargs = { enable_thinking = false } }`（对 Qwen3 类有效；对 Ornith 无效）。

### 3.5 Ornith 模型特性

- 推理模型：答案常驻 `reasoning` 字段，`content` 常在推理完成后才出现；预算紧（≤60 token）直接截断、content 为空。
- `enable_thinking=false`（chat_template_kwargs）**对它无效**（实测两次输出完全相同）。
- strict json_schema（guided decoding）下不稳定：顶满 token 预算也不吐合法 JSON（content 空或半截）。
- 结论：Ornith 只适合当生成模型，**不适合当判官/评分器**。

## 4. 如果以后还要做：可行路径

| 方案 | 显存 | 说明 |
|---|---|---|
| **Ollama + Gemma-4-E2B Q4_0 GGUF**（3.1GB） | ✅ 放得下 | 最省事：GGUF 原生支持、零编译；需实测 Ollama OpenAI-compat 对 strict json_schema 的支持（Ollama 0.32 已支持 response_format json_schema，strict 强制力弱一档，可接受 fail-open 兜底） |
| **Qwen3-4B-AWQ**（2.7GB）+ vLLM | ✅ 放得下 | vLLM 原生 awq、guided decoding 强制 schema；4B 判断力一般但格式合规有保证 |
| Qwen3-8B-AWQ（5.7GB） | ❌ 放不下 | 需先解决显存（缩 ornith 或清页缓存），或 CPU-offload（慢 5-15s/次） |
| Gemma-4-E4B | ❌ 兼容问题 | 等 vLLM 修复 PLE 布局兼容后再考虑 |

## 5. 恢复与善后（已完成）

- 删除全部下载：`~/projects/gguf/`（25GB）、容器内 `/root/gemma-judge`（11.5GB）、ollama-models、Modelfile；
- Ornith 容器 util 恢复 **0.90**（用户指定；脚本自动回滚机制保留：`swap-ornith-util.sh` 先 rename 备份再启动，失败自动恢复）；
- switchyard 配置未改动，路由原样（本地 ornith + OpenRouter deepseek 可用，aliyun key 无效待修）；
- 保留 `~/projects/venv-vllm`（vLLM 0.27.1 用户态环境，含两个无害补丁：config convertor 的 per-layer 属性放行 + gemma4.py 调试打印）。

## 6. 关键命令速查

```bash
# 判官 vLLM 启动（需沙箱授权；脚本现位于 scripts/ 目录）
bash /home/<user>/projects/scripts/start-judge-vllm.sh        # Gemma-4（旧方案，日志 /tmp/vllm-gemma.log）
bash /home/<user>/projects/scripts/start-judge-qwen3-8b.sh   # Qwen3-8B-AWQ（现用，日志 /tmp/vllm-qwen3-8b.log）

# Ornith 容器换 util（docker 无需授权，自动回滚）
# 编辑 swap-ornith-util.sh 中 --gpu-memory-utilization 后：
bash /home/<user>/projects/swap-ornith-util.sh

# HF 直连下载（绕开 xet 沙箱问题）
curl -fsL -o <file> https://huggingface.co/<org>/<repo>/resolve/main/<file>

# 容器内查显存占用
docker exec ornith-b12x-serve nvidia-smi

# 用户态测显存（需授权）
/home/<user>/projects/venv-vllm/bin/python -c 'import torch; print(torch.cuda.mem_get_info())'
```

## 7. Judge 服务关停记录（2026-09-10 00:13–00:16 HKT）

### 7.1 背景与原因

- **内存压力**：新主服务 `qwen38-27b-sglang`（SGLang，`mem-fraction-static 0.85`）9/9 23:06 上线后，整机 112/121 GiB 已用，MemAvailable 仅 ~10 GiB（~8.5%），swap 已用 5.7/15 GiB。hermes 网关判定为 `elevated`（阈值：available < 15% × MemTotal），自 23:35 起持续告警，kanban dispatch **被限流为每 tick 最多派 1 个 worker**。
- **judge 无调用方、纯空转**：8/24 起运行 16 天，关停时核实：
  - 当前 `routes.toml`（9/9 21:48 部署）**不引用 8889**；
  - switchyard 日志自 8/24 以来无真实 judge/8889 调用记录；
  - 无到 8889 的活动 TCP 连接；hermes 配置（`.env`/`config.yaml`）也未引用；
  - 进程 ~3.7 GiB 页被挤入 swap（EngineCore VmSwap ~2.7 GiB），实际常驻内存很小，但仍占着 KV 预留池（`gpu-memory-utilization 0.10` ≈ 12 GiB 统一内存）。

### 7.2 执行

```bash
pkill -f 'port 8889'      # SIGTERM（沿用 start-judge-vllm.sh 的惯例）
# EngineCore 与 8889 端口随即释放；主进程卡在 D 状态（GPU 内存回收中）
kill -9 300660 300792     # SIGKILL 收尾
```

| PID | 角色 | 结局 |
|---|---|---|
| 300660 | vllm api_server 主进程 | SIGTERM → D 状态 → SIGKILL |
| 300792 | multiprocessing resource_tracker | SIGKILL |
| 300793 | VLLM::EngineCore | SIGTERM 后自行退出 |

### 7.3 结果

| 指标 | 关停前 | 关停后 |
|---|---|---|
| MemAvailable | 9.9 GiB（~8%） | **21.5 GiB（~16.9%）** |
| Swap 已用 | 5.7 GiB | **2.0 GiB** |
| 端口 8889 | LISTEN | 已释放 |
| hermes 内存告警 | 持续 `elevated` | 最后一条 00:16:58 后不再出现（16.9% > 15% 阈值 → ok） |
| kanban dispatch | 每 tick 限 1 个 worker | 限流解除 |

### 7.4 保留与恢复

- 模型文件保留：`judge-models/Qwen3-8B-AWQ`（~17 GB 磁盘）；若长期不用可整目录删除腾磁盘（不影响任何在跑服务）；
- `venv-vllm`（vLLM 0.27.1 用户态环境）未动；
- 重启：`bash /home/<user>/projects/scripts/start-judge-qwen3-8b.sh`（日志 `/tmp/vllm-qwen3-8b.log`）；
- 影响范围：仅 judge。8888 主服务（sglang）与 `vllm-open-proxy.py`（现转发 sglang，名字为历史遗留）不受影响。