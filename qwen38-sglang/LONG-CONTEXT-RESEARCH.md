# Qwen3.8-27B 上下文扩展调研（262K → 512K / 1M）

> 调研日期: 2026-09-10 · 主机 spark-XXXX (DGX Spark GB10) · 线上服务: 容器 qwen38-27b-sglang (SGLang + DSpark, 262144)
> 结论: **可以扩展**。官方路线是 YaRN + 原生 MTP；**但社区已有保住 DSpark 的做法（把 YaRN 写进模型 config 而不是走 CLI 覆盖），本机已验证到"能启动"这一层，速度是否真保住需在维护窗口实测接受率**（见 §10）。

---

## 0. 结论速览 (TL;DR)

| 问题 | 答案 |
|---|---|
| 还能不能延长? | **能**。官方模型卡明确写 "262K native context, extendable up to 1M via YaRN" |
| 官方配方 | `--context-length N` + `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` + `--json-model-override-args` 里把 text_config.rope_parameters 改成 yarn |
| 代价 | **必须把投机解码从 DSpark 换成 MTP**：YaRN 覆盖会泄漏进独立草稿模型 config，rope 校验直接崩（本机已复现，见 §5） |
| 显存够吗 | 够。KV ≈ 32.8 KB/token(fp8)：512K 一条 ≈17GB，1M 一条 ≈33GB；现网 KV 池约 130 万 token(≈42GB) → 1M 单条或 512K 两条并发 |
| 推荐档位 | **512K (YaRN factor 2.0)** 常开；1M 按需切（重启 8-10 分钟）。2.0 与 4.0 是模型卡验证点 |
| 性能影响 | 代码/工具类负载吞吐下降（公开同硬件实测 DSpark code 51.5 → MTP 34.5 tok/s），长文/essay 反而更快（18.3 → 24.1） |
| 改动面 | 端口 8888 / 容器名 / served-model-name 全不变 → switchyard、vllm-open-proxy.py **零改动** |
| **想保住 DSpark 速度?** | **有办法（社区做法 + 本机验证）**：不要把 YaRN 走 `--json-model-override-args`，而是**写进 target 的 config.json**（模型卡方案一）→ 草稿不再被污染 → DSpark 原样保留。§10 给出三种模式与实测计划 |

---

## 1. 现状与证据（本机实测）

| 项 | 值 | 来源 |
|---|---|---|
| 线上上下文 | `max_model_len=262144` | `curl 127.0.0.1:8888/v1/models` |
| 引擎 | SGLang `0.0.0.dev0+qwen38.27b.g561c8f3`，镜像 `lmsysorg/sglang:qwen38-27b` | 容器内 `sglang.__version__` |
| 模型架构 | `qwen3_5`（Qwen3.5 混合：**64 层 = 48 层 GDN 线性注意力 + 16 层 full attention**，每 4 层一次全注意力） | 本地 config.json `text_config.layer_types` |
| 注意力形状 | 24 Q heads / **4 KV heads** / head_dim 256 / partial_rotary_factor 0.25 / mrope（原生 VLM） | config.json |
| rope 现状 | `rope_type=default, rope_theta=1e7`，`max_position_embeddings=262144` | config.json |
| **MTP 头** | **在主 checkpoint 内**：`mtp_num_hidden_layers=1`，index.json 里有 15 个 `mtp.*` 张量 → 无需额外下载 | 本地 safetensors index |
| DSpark 草稿 | `max_position_embeddings=262144`，自身 rope 就是 `yarn factor 32 / original 8192`（即"YaRN 到 262K"训练而来），>262K 属外推 | draft config.json |
| 超限开关 | `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN`（默认 False，超限直接 raise） | 容器内 `sglang/srt/environ.py:902`、`configs/model_config.py:786` |
| KV 池容量 | 日志 `#full token: 129848, full token usage: 0.10` → 池 ≈ **130 万 token**(≈42GB, fp8) | `docker logs qwen38-27b-sglang` |

---

## 2. 官方扩展配方（Qwen3.8-27B 模型卡 "Processing Ultra-Long Texts"）

SGLang 官方写法（模型卡原文，仅把 factor/长度按需改）：

```shell
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server ... \
  --json-model-override-args '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --context-length 1000000
```

要点：
- **factor = round(目标长度 / 262144)**：524288→2.0，786432→3.0，1000000→4.0；模型卡验证点是 **2.0 与 4.0**。
- 静态 YaRN：缩放对**所有**请求生效（含短请求），模型卡因此建议"只在确实需要长上下文时才改"，并"典型长度 512K 就用 factor 2.0"。
- 也可直接改 `config.json` 的 `text_config.rope_parameters`；本机缓存是只读软链，走命令行覆盖更干净（不动缓存）。
- 同等写法 vLLM 是 `--hf-overrides '{"text_config": {...}}' --max-model-len 1000000` + `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`（备查，本机现役是 SGLang）。

---

## 3. 关键约束：YaRN 与 DSpark 不兼容（必须换 MTP）

机制：SGLang 把 `--json-model-override-args` 的 `text_config.rope_parameters` **也套到了草稿模型**的扁平 config 上，transformers 的 rope 校验随即取不到 `max_position_embeddings` 而崩：

```
AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'
  (transformers/modeling_rope_utils.py:882 _validate_yarn_rope_parameters)
```

- 本机已在 **不动线上服务** 的前提下复现（`verify-yarn-config.py` 用例 E，见 §5）。
- 同硬件公开实测仓库 MiaAI-Lab 的结论一致：`DSpark cannot use YaRN / context > 262144 on this build (ditto DFlash2 — same draft-config leak)`。
- MTP 头属于主模型，**和主模型共用同一套被覆盖的 rope 参数**，因此不受影响。

MTP 启动参数（替换现网 DSpark 的三行；不再需要 `--speculative-draft-model-path`）：

```
--speculative-algorithm EAGLE \
--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

（MiaAI 同硬件实测 MTP 3/1/4 为最优点；本 build 的 `--speculative-algorithm` 内置 EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM/DFLASH/DSPARK。）

---

## 4. 容量与成本测算（GB10 / 128GB 统一内存）

KV 每 token ≈ **32.8 KB**（16 层 full attention × 4 KV head × 256 head_dim × K+V × fp8；与 MiaAI 实测值一致）。

| 目标上下文 | YaRN factor | 单请求 KV | 现网池(≈130 万 token, 42GB) 可并发 | 冷 prefill 估算* |
|---|---|---|---|---|
| 262144（现状） | — | 8.6 GB | 4-5 条 | ~4 分钟 |
| **524288（推荐）** | **2.0** | **17.2 GB** | **2 条** | ~9 分钟 |
| 786432 | 3.0 | 25.8 GB | 1-2 条 | ~13 分钟 |
| 1048576 | 4.0 | 33.0 GB | 1 条 | ~17 分钟 |

\* 按实测 prefill ≈1000 tok/s（chunked-prefill 8192）估算；首次还要叠加 torch.compile JIT。同前缀二次请求走 radix cache，秒级返回。

其它与长度无关的占用：**GDN/Mamba 状态池 = 并发 × 4 slot × 78.4MB**（现网 32 slot ≈2.5GB，bf16 状态）；投机解码的 verify 窗口是引擎侧独立 buffer，不占状态池。

> ⚠️ 统一内存红线：现网 `--mem-fraction-static 0.85` 是为避免 host swap（0.90 时曾出现 swap 抖动，见 DEPLOYMENT-RECORD §7.4）。MiaAI 的 MTP 配方用 0.95 + `--sleep-on-idle`，**不建议本机照搬**；如需更多池子，先 0.87/0.88 小步试，观察 `free -g` 与 `swapon --show`。

---

## 5. 零停机预检（已执行，全部在本机完成）

`verify-yarn-config.py`：只用 config（不加载权重、不占服务端口），按 launch_server 的同一条代码路径构造 ModelConfig：

| 用例 | 结果 |
|---|---|
| A. 主模型 原生 262144，无覆盖 | ✅ `context_len=262144` |
| B. 主模型 + YaRN factor 2.0，context 524288 | ✅ 通过，rope 被替换为 yarn(factor 2.0, original_max 262144) |
| C. 主模型 + YaRN factor 4.0，context 1048576 | ✅ `context_len=1048576` |
| D. DSpark 草稿 无覆盖 262144（现网） | ✅ 正常（草稿自带 yarn factor 32） |
| E. DSpark 草稿 + YaRN 覆盖 524288 | ❌ **AttributeError: max_position_embeddings**（复现"泄漏"崩溃） |

运行方式：
```bash
docker run --rm --gpus all \
  -v /home/<user>/projects/qwen38-model-cache:/root/.cache/huggingface \
  -v /home/<user>/projects/qwen38-sglang:/work \
  -e HF_HOME=/root/.cache/huggingface -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 -e HF_HUB_OFFLINE=1 \
  lmsysorg/sglang:qwen38-27b python3 /work/verify-yarn-config.py
```

---

## 6. 执行方案（维护窗口，约 10 分钟）

```bash
cd /home/<user>/projects/qwen38-sglang

# 0) 备份 + 记录现状
cp start.sh start.sh.bak-262k-$(date +%m%d)
curl -s http://127.0.0.1:8888/v1/models   # 记录 max_model_len

# 1) 切换（默认 512K + MTP 3/1/4）
./stop.sh
CONTEXT_LENGTH=524288 ./start-long.sh      # 1M: CONTEXT_LENGTH=1048576 ./start-long.sh

# 2) 自检
curl -s http://127.0.0.1:8888/v1/models    # 期望 max_model_len=524288
docker logs qwen38-27b-sglang 2>&1 | grep -E 'context_len|max_running_requests|mamba pool'

# 3) 功能验证：短问烟测 + 长文大海捞针
python3 bench-long-ctx.py --ctx 400000 --depth 0.1     # 期望 命中=是, prompt_tokens≈400K
curl -s http://127.0.0.1:4000/v1/models                # switchyard 侧模型名/端口应无变化

# 4) 回滚（<1 分钟切换 + 8-10 分钟冷启）
./stop.sh && ./start.sh                    # 回到 DSpark + 262K
```

产物：
- `start-long.sh` — 长上下文启动脚本（YaRN + MTP，端口/模型名与现网一致）
- `bench-long-ctx.py` — 大海捞针验证（零依赖，报告 prompt_tokens / TTFT / 命中与否）
- `verify-yarn-config.py` — 上述预检脚本

---

## 7. 风险与权衡

1. **静态 YaRN 影响短文本**：factor 2.0/4.0 对所有请求生效，短对话质量可能有细微变化（模型卡明确提示）。这是"常开 or 按需切"的核心权衡；512K 档因子最小、影响最轻。
2. **投机解码栈的性能交换**：MTP 在代码/工具类负载吞吐低于 DSpark（MiaAI 同硬件 code 51.5→34.5 tok/s；essay 18.3→24.1）。本机现网 DSpark 实测 25.4 tok/s（400 token 中文 essay 探针，与对方探针不同口径），切换后需用同探针自测对比。
3. **1M 档未在本机验证质量**：官方只在 2.0/4.0 两个点上验证过；1M 冷 prefill ~17 分钟，长请求会占用大半 KV 池，其它请求 TTFT 变差。
4. **多模态 rope**：模型是原生 VLM（mrope_section 11/11/10），官方覆盖里带了 mrope 字段，预检 OK；但长上下文 + 图像输入的组合未验证。
5. **不要盲目提 mem-fraction**：统一内存 swap 是这台机器的历史痛点。
6. **DSpark 崩溃的根因是"覆盖泄漏"而非模型能力**；若上游修好该问题，DSpark + YaRN 可直接复用同一个 start.sh 的 YaRN 参数。

---

## 8. 其它思路（未验证 / 不推荐优先做）

| 思路 | 评价 |
|---|---|
| 给 DSpark 草稿打 config 补丁绕过泄漏（复制 draft 目录、改 `max_position_embeddings` 与自己的 rope factor，再 `--speculative-draft-model-path` 指过去） | 理论可行但草稿只训练到 262K，>262K 接受率会崩；输出正确性由 target 验证兜底，但**只会更慢**，不值得 |
| 双实例分工（8888 DSpark/262K + 8890 MTP/512K） | 单卡要放两份权重(21GB×2) + 两个 KV 池，统一内存吃紧，风险高 |
| `--enable-hierarchical-cache` / HiCache（本 build 支持） | 降低重复 prefill 成本、提高多轮命中率，**不提高单请求上限** |
| `--enable-unified-memory`（本 build 支持） | 让 KV 池与 GDN 状态池动态共享同一 buffer，减少长上下文下的池浪费；本机未验证 |
| 跟踪新版镜像 | DSpark/DFlash2 + YaRN 的兼容修复若合入上游，可用现网 start.sh 直接开 YaRN |

---

## 9. 参考

- 官方模型卡（超长文本 / YaRN 配方）: https://huggingface.co/Qwen/Qwen3.8-27B
- 官方仓库（262K native，YaRN 可到 1M）: https://github.com/AlibabaCloud-Official/Qwen3.8-27B
- SGLang cookbook（Qwen3.8-27B / DGX Spark cell，in-checkpoint MTP）: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- 同硬件 GB10 实测仓库（Long context & concurrency 章节、DSpark 不兼容 YaRN 的说明）: https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
- 本机既有文档: `DEPLOYMENT-RECORD.md`（§4 参数 / §7.4 性能 / §12 注意事项）、`MIGRATION-NOTES.md`

---

## 10. 保住 DSpark 速度的长上下文方案（2026-09-10 追加调研）

### 10.1 社区现状（有人踩过同一个坑）

| 来源 | 内容 | 对我们的意义 |
|---|---|---|
| [sgl-project/sglang#33505](https://github.com/sgl-project/sglang/issues/33505) | "[Bug] The --json-model-override-args does not recursively update the model configuration"，**open，0 评论，无修复** | 官方没修：泄漏根因就是"非递归覆盖 + 覆盖被转发给草稿" |
| [drowzeys/keys-Qwen3.8-27B-DSpark-vLLM-YaRN-1M-Context](https://huggingface.co/drowzeys/keys-Qwen3.8-27B-DSpark-vLLM-YaRN-1M-Context) | **config-only fork**：把 1M 的 YaRN rope **直接打在草稿上**（权重不动，`max_position_embeddings: 1048576`, `factor: 4.0`, `original_max_position_embeddings: 262144`）。README 明确复述了我们的失败模式（"YaRN either crashes (text_config / max_position_embeddings) or the draft rope stays at 262k while the target is at 1M"）。**注意：他们自己也没测过接受率**（"Acceptance at 1M is unmeasured"） | 印证"给草稿换绳"是社区共识做法 → 我们的 MODE=draftyarn |
| [drowzeys/keys-1M-CTX-…-Dspark-…-SGLang-SM121…](https://github.com/drowzeys/keys-1M-CTX-Inkling-Small-NVFP4-Dspark-NVFP4-KV-Cache-SGlang-SM121-optimized-on-Two-DGX-Sparks) | 在 **双 DGX Spark (GB10/sm_121a) + SGLang 上把 DSpark 开到 1,048,576 上下文**（另一模型 Inkling-Small）。要点：fp4 KV cache（自研补丁，KV 池 3.12×）、`--triton-attention-reduce-in-fp32`（**接受率 +11%**，因为 bf16 累加扰动 logits 导致草稿对不上）、`--page-size 1`、显式 `--cuda-graph-bs` 列表；草稿用**原版 checkpoint 不改**；草稿微调 5 次全失败 | 证明"DSpark + 1M 在这类硬件上能跑"；他们的加速旋钮可作为本机 A/B 候选 |
| [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) | 与我们同硬件的实测仓库：明确写 "DSpark cannot use YaRN / context > 262144 on this build (draft-config leak)"，官方建议切 MTP | 反证：他们只试了 CLI override 这条路，没试"把 YaRN 写进 config" |

**结论：社区没有现成的"一行开关"，但有两条已被别人走过一半的路——(a) 把 YaRN 从 CLI 挪到 config 里，(b) 顺便把草稿的 rope 也换成同一套 YaRN。我们把两条合并，并在本机验证到"能启动"这一层。**

### 10.2 三种模式（都已离线预检通过，512K 与 1M 都过）

| MODE | target | 草稿 | 预期 |
|---|---|---|---|
| `yarn` | config.json 内联 YaRN | 原样（自带 yarn32/8192，max_pos 被自动抬到新上下文） | 草稿 rope 与 target 不一致，接受率未知 |
| `draftyarn`（默认，社区做法） | 同上 | 软链 variant，rope 换成与 target 同一套 YaRN | 两者位置编码一致，理论上接受率更稳 |
| `extrap` | 原样（不缩放） | 原样 | ≤262K 与今天**逐字节一致**，>262K 纯外推，质量未知但速度零损失 |

离线预检实录（`verify-dspark-modes.py`，只读 config、不起服务）：

```
[OK] yarn(草稿原样) @ 512K     target rope=yarn/f2.0/orig262144 | draft rope=yarn/f32.0/orig8192 max_pos=524288
[OK] yarn(草稿原样) @ 1024K    target rope=yarn/f4.0/orig262144 | draft rope=yarn/f32.0/orig8192 max_pos=1048576
[OK] draftyarn @ 512K          target rope=yarn/f2.0/orig262144 | draft rope=yarn/f2.0/orig262144
[OK] draftyarn @ 1024K         target rope=yarn/f4.0/orig262144 | draft rope=yarn/f4.0/orig262144
[OK] extrap @ 512K / 1024K     target rope=default            | draft rope=yarn/f32.0/orig8192
```

### 10.3 关键实现细节（为什么不用改镜像、不用重新下载）

- **`--json-model-override-args` 保持 `{}`**：这是让草稿不被污染的开关。
- **YaRN 落在 target 的 config.json**：用软链 farm（`make-yarn-variant.py`）生成变体目录，只把 `config.json` 换成真实文件，权重/tokenizer 全是相对软链 → **零拷贝、零下载**；容器内路径 `/root/.cache/huggingface/variants/…`（复用已有挂载）。
- **草稿 rope overlay**：同样用软链 farm（`--draft`），把草稿的 `max_position_embeddings` 与 `rope_parameters` 改成与 target 一致（数值参考社区那份 config-only fork）。
- **`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`** 仍需要；它同时会把草稿的 `max_position_embeddings` 自动抬到新上下文（启动日志有 `Overriding the draft model's max_position_embeddings to N`）。

### 10.4 维护窗口实测计划（**速度是否保住，只能实测**）

单卡无法并存两套配置，所以每次切换 = 8-10 分钟冷启动。建议顺序：

1. `MODE=extrap CONTEXT_LENGTH=524288`：先拿到"零速度损失 + 纯外推质量"的基线（这个模式下 ≤262K 行为与今天完全一致）。
2. `MODE=draftyarn CONTEXT_LENGTH=524288`：社区做法，看接受率是否还接近现网。
3. `MODE=yarn CONTEXT_LENGTH=524288`：草稿不换绳的对照。
4. （可选）`CONTEXT_LENGTH=1048576`：确认 1M 能起、能捞针。
5. 参考对照：`CONTEXT_LENGTH=524288 ./start-long.sh`（MTP）。

每个模式测三项：
```bash
# ① 接受率 (决定是否真"保速")
docker logs qwen38-27b-sglang 2>&1 | grep 'accept len' | tail -20
#    现网 262K 基线: accept len≈2.0-4.9, accept rate≈0.15-0.69
# ② 速度 (与现网 25.4 tok/s 非思考探针对比)
python3 bench.py
# ③ 长文召回 (深度 10% 处埋密码)
python3 bench-long-ctx.py --ctx 400000 --depth 0.1
```

判据（建议）：512K 下 `accept len ≥ ~2.5` 且 `bench.py` 吞吐 ≥ 现网 80% → DSpark 长上下文可用，不必切 MTP；若接受率崩到 1.x（≈无投机），则长上下文仍走 MTP 路线（`start-long.sh`）。

### 10.5 其它可选旋钮（社区给的，风险自评）

| 旋钮 | 说明 | 我们是否已具备 |
|---|---|---|
| `--kv-cache-dtype nvfp4 / fp4_mx_block16 / fp4_e2m1` | KV 再省一半 → 池子翻倍，1M 并发从 1 条到 2 条 | **本镜像的 CLI 已列出这些取值**，但未实测（Keys 那边是自研补丁 + triton 后端；我们走 flashinfer，需实测） |
| `--triton-attention-reduce-in-fp32` | Keys 实测接受率 **+11%**（bf16 累加扰动 logits） | flag 存在；但它是 triton 后端的开关，我们现用 flashinfer，需换后端才有意义 |
| `--page-size 1` | Keys 报告 page-128 会破坏 triton 的 verify 路径 | 可作 A/B；flashinfer 下未知 |
| `--sleep-on-idle` | 空闲时释放池子（GB10 统一内存紧张时有用） | 可用，代价是首请求唤醒延迟 |
| HiCache / `--enable-unified-memory` | 提升多轮/长文档复用率，不提高单请求上限 | 本 build 支持 |

### 10.6 相关文件

- `start-dspark-long.sh` — 三模式 DSpark 长上下文启动脚本（默认 draftyarn + 512K）
- `make-yarn-variant.py` — 生成 target / draft 的软链变体（`--draft` 生成草稿 rope overlay）
- `verify-dspark-modes.py` — 三模式离线预检（只读 config）
- `verify-yarn-inline.py` / `verify-yarn-config.py` — 内联 YaRN 与原始泄漏的对照预检
- 变体目录：`qwen38-model-cache/variants/{qwen38-nvfp4-yarn512,qwen38-nvfp4-yarn1024,qwen38-dspark-yarn512,qwen38-dspark-yarn1024}`

---

## 11. 社区说法逐条裁决（2026-09-10，全部在本机同一 build 上复现/证伪）

> build: `lmsysorg/sglang:qwen38-27b`（SGLang 0.0.0.dev0+qwen38.27b.g561c8f3）

| # | 流传的说法 | 裁决 | 本机证据 |
|---|---|---|---|
| 1 | DSpark 在官方 CLI 传参体系下无法与 YaRN 共存 | ✅ 成立 | 上游 issue [#33505](https://github.com/sgl-project/sglang/issues/33505) open 0 评论；隔离参数 PR [#35583](https://github.com/sgl-project/sglang/pull/35583) 已关闭、[#35841](https://github.com/sgl-project/sglang/pull/35841) 未合并；MiaAI README/`.env.sample` 明文不兼容；HF [RadixArk/…-DSpark discussion #6 "Failed with YaRN"](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark/discussions/6) 无人回复 |
| 2 | 根因是"参数污染"，草稿 config 缺 `max_position_embeddings` | ✅ 成立（位置更精确） | 泄漏点：`speculative/dspark_components/dspark_config.py:35,136`（DSpark）、`arg_groups/speculative_hook.py:217`（DFLASH）把全局 override 传给草稿 `get_config()`；崩溃点在 `utils/hf_transformers/common.py get_hf_text_config()` 把裸 dict 实例化成 `PretrainedConfig` |
| 3 | 方案一：给 SGLang 打"隔离草稿参数"补丁 | ✅ 方向正确，位置需修正 | 贴文点名的 `server_args.py` / `model_loader/loader.py` 不对；正确打点见 #2。上游对应 PR #35583（已关）/#35841（open）—— **任何 release/镜像里都没有这个参数** |
| 4 | 方案二：草稿 config 补 `max_position_embeddings` + `rope_parameters{factor:1.0,type:default}` 就能骗过校验 | ❌ **实测证伪** | 变体 `qwen38-dspark-cfg-ropeplanB` + CLI YaRN 覆盖 → 仍 `AttributeError: max_position_embeddings`。原因：崩的是 **override 新建的那个嵌套 stub text_config**，与草稿顶层写什么无关 |
| 4b | （同思路的正确做法，本次新增）给草稿 config 补一个**完整 text_config** | ✅ **实测可行** | 变体 `qwen38-dspark-cfg-textconfig` + CLI YaRN 覆盖 → **OK**（override 走 merge 分支 `current.update(value)`，因为 current 已是 PretrainedConfig）。→ 可**同时用官方配方（CLI 覆盖）+ DSpark** |
| 5 | "DSpark 草稿本质是固定 1024 左右的滑窗注意力" | ❌ 事实错误 | DSpark 草稿 config：5 层全 `full_attention`、`sliding_window: null`、`use_sliding_window: false`、32 Q/8 KV、max_pos 262144。**滑窗 2048 的是 DFlash2 草稿**（`layer_types` 全 sliding_attention、`sliding_window: 2048`）—— 贴文把两者搞混了 |
| 6 | DFlash2"原生兼容主模型长上下文，不会因全局 RoPE 覆盖崩溃" | ❌ **实测证伪** | 同一 build：DFlash2 草稿(z-lab/Qwen3.8-27B-DFlash2) + CLI YaRN 覆盖 → **同样的 AttributeError**。MiaAI 也写 "same draft-config leak as DSpark" |
| 7 | DFlash2 已有 524K 成功案例（Reddit `r/BlackwellPerformance` 等） | ⚠️ 未找到该来源；**真正存在的案例是另一个** | HF [incoai/Qwen3.8-27B-DFlash2 discussion #8](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/8)：GB10 + **vLLM**，把 YaRN 烧进草稿 config，实测跑到 **539K**（接受率 ~47%、AL ~1.9）。**是 vLLM 不是 SGLang** |
| 8 | DFlash2 在 DGX Spark 上 100~130+ tok/s（单并发） | ❌ 与实测不符 | MiaAI 同硬件：DFlash2 code ≈50.9 / essay ≈25.4 tok/s；Keys 的 1M 报告 decode ≈33 tok/s |
| 9 | 配方用 `--attention-backend fa3` | ❌ 本机不可用 | GB10/SM121：MiaAI 实测必须 flashinfer；Keys 报告 fa4 仅 sm_100。fa3/fa4 出现在本 build 选项里只是全平台枚举 |
| 10 | `--speculative-dflash-draft-window-size` | ❌ 参数名不存在 | 本 build 是 `--speculative-dflash-block-size` 与 `--speculative-draft-window-size` |
| 11 | 方案三"转向 DFlash2" | ⚠️ 作为**性能平替**合理，作为 **YaRN 解法无效** | MiaAI 实测 DFlash2 ≈ DSpark 的 code 速度、essay 更快；但 YaRN 问题与 DSpark 完全一样。不过它同样能用 config 层两条路线解决：本机已验证 `qwen38-dflash2-yarn512`（草稿内烧 YaRN）+ 无 CLI 覆盖 → **OK** |

### 11.1 由此确定的两条可用路线（都不需要打补丁、不需要重下权重）

| 路线 | 做法 | 已验证程度 |
|---|---|---|
| **A. 不用 CLI 覆盖** | 把 YaRN 写进 target 的 config.json（软链变体 `qwen38-nvfp4-yarn{512,1024}`），`--json-model-override-args '{}'`；草稿原样或换同套绳 | config 加载 ✅（`verify-yarn-inline.py` / `verify-dspark-modes.py`）；**速度/质量待维护窗口实测** |
| **B. 保留 CLI 覆盖** | 官方配方照旧 + 草稿 config 补**完整 text_config**（`qwen38-dspark-cfg-textconfig`） | config 加载 ✅（`verify-dspark-planB.py` 用例 C）；草稿 rope 会被强行改成 target 的 YaRN，接受率待实测 |

### 11.2 两个必须在实测里叠加的坑（社区/上游独立发现，已核实）

1. **verify 窗口越界（sglang#33454，open；修复 PR #33963 未合并）**：DSpark 的 verify 窗口不做剩余预算裁剪，`seq_len + block_pos` 越过上下文上界会触发 `fused_q_norm_rope` 的 CUDA illegal address（引擎直接挂，比 config 崩溃更难查）。→ 维护窗口里 **`--context-length` 要比"实际要服务的长度"多留 ≥8 token**（要服务 512K 就配 524296 以上），或严格限制请求长度。
2. **草稿 KV 也要吃预算（HF RadixArk/…-DSpark discussion #8）**：DSpark 草稿 5 层全注意力 GQA，在 32GB 卡上把可用上下文从 187,872 压到 ~56,032。本机换算：草稿 KV ≈ 8 KV头 × 128 head_dim × 2(K/V) × 5 层 × 1B ≈ **10.2 KB/token** → 512K 约 +5.3GB、1M 约 +10.7GB（另加 GDN 状态池）。

### 11.3 本轮新增产物

- `verify-dspark-planB.py` — 裁决"方案二"与"完整 text_config"路线（用例 B/C）
- `verify-dflash2-yarn.py` — 裁决 DFlash2 的 YaRN 说法（5 用例），配套草稿变体 `qwen38-dflash2-yarn512` / `qwen38-dflash2-textconfig`
- `start-dspark-long.sh` 新增 `ALGO=DFLASH` 支线（自动选用烧了 YaRN 的 DFlash2 草稿）
- 草稿权重：`qwen38-model-cache/dflash2-draft`（z-lab/Qwen3.8-27B-DFlash2，2.6GB，已下载）

---

## 12. 参考 hasso5703/dgx-spark-qwen38（GB10 同硬件的现场方案）

> 仓库：https://github.com/hasso5703/dgx-spark-qwen38（MIT，166 stars，2026-08-15 建，2026-09-09 仍在更新）
> 定位：DGX Spark (GB10) 上 Qwen3.8-27B 的每日实测配置（SGLang + NVFP4 + DFlash2），1M 模式自 2026-08-22 起天天在跑。

### 12.1 他们的 1M 方案（关键文件 patch-yarn.py + qwen38-sglang-1m.service.template）

| 做法 | 细节 | 对我们的意义 |
|---|---|---|
| 把 YaRN 写进 config.json，而不是走 CLI 覆盖 | patch-yarn.py 把 max_position_embeddings=1010000 与 rope_type=yarn / factor=4.0 / original_max_position_embeddings=262144 合并进已缓存的 config.json；target 和 DFlash2 草稿两个文件都改（原文：or the draft crashes at load），原文件备份为 config.json.pre-yarn | 就是我们的路线 A，而且印证了草稿也要处理（我们的 MODE=draftmaxpos）；他们观察到不补草稿会在加载时崩，说明不能只依赖引擎自动抬升 |
| 启动参数 | --context-length 1010000 与 SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1，完全不传 --json-model-override-args | 我们脚本传 '{}' 等价 |
| 窗口留余量 | 服务 1,010,000，而 YaRN 理论上界是 4x262144=1,048,576，留约 3.7% 余量 | 与 sglang#33454（verify 窗口越界导致 CUDA illegal address）一致；我们 512K 档也应服务约 506K，不要顶格 524288 |
| 统一内存纪律 | --mem-fraction-static 0.70，docker --memory 100g --memory-swap 100g，并关闭 flashinfer autotune | 他们的教训：GB10 上 SGLang 内存记账看不到 25-40GB 瞬时分配（autotuner / CUDA graph 捕获），高于 0.50 的原生跑法或开着 autotune 都可能把宿主机内存打到 0，只能断电恢复；0.80 在三并发下实测崩过；0.85 只剩约 8GB 触发 earlyoom。我们现网 0.85、free 只剩 18GB，长上下文实验前必须降到 0.70 并加 docker 上限 |
| 稳定性运维 | HF_HUB_OFFLINE=1（防止 Hub 重新解析把打过补丁的 config 覆盖回去）、Restart=always、--sleep-on-idle | 我们已把 HF_HUB_OFFLINE=1、内存上限、长上下文默认 0.70 与关 autotune 写进 start-dspark-long.sh |
| 实测结论 | 真实 690K token 请求跑通（冷 prefill 40 分钟，之后走缓存），DFlash2 接受率不变（acceptance unchanged）；KV 池 863K-1019K token（波动大） | 这是外挂草稿 + YaRN 目前最硬的现场证据，但用的是 DFlash2，不是 DSpark |

### 12.2 与我们场景的三点差异（不能照抄）

1. 算法不同：他们跑 DFlash2（草稿是 2048 滑窗，rope_type 为 default），把草稿 rope 改成 yarn(4.0/262144) 后接受率不变。我们的 DSpark 草稿自带训练过的 rope = yarn(32.0 / orig 8192)，盲目照抄「把 target 的 rope 灌进草稿」有可能反而掉接受率，所以默认用 draftmaxpos（只抬 max_pos，rope 不动），draftyarn（换 rope）作对照。
2. 上下文档位：他们只做 1M（factor 4.0）与原生 262K 两档；我们要的是 512K（factor 2.0）作为日常档，所以变体按上下文生成（qwen38-nvfp4-yarn512 等），不写死。
3. 镜像与权重不同：他们自带 overlay 补丁（含 mrope 修复 sglang#34446，官方 2026-08-22 镜像没有该修复），并用 sha256 固化。我们现网镜像 lmsysorg/sglang:qwen38-27b 是否含该 mrope 修复尚未核实；若要用图片/视频输入，建议先验证（多模态路径的质量问题，与上下文长度无关）。

### 12.3 一个值得重新审视的前提：必须保住 DSpark 吗？

该仓库自己的冻结电池对比（同机同探针）：

| 生成内容 | v1.2（DFlash2，现役） | v1.1（DSpark，旧配置） | 稳定 MTP 引擎 |
|---|---|---|---|
| Agentic coding（代码/补丁/工具调用） | 32-40 tok/s | 28-36 | 24-28 |
| 数学与结构化推理 | 41-44 | 38-42 | 24-33 |
| 技术解释（法文） | 26 | 23-25 | 约 22 |

MiaAI-Lab 的实测同向：DFlash2 code 50.9 vs DSpark 51.5（平手）、长文 25.4 vs 18.3（DFlash2 明显更好）。
结论：DFlash2 在这类硬件上并不比 DSpark 慢（我们本机 DSpark 现网 25.4 tok/s 是 400-token 中文 essay 探针，口径不同）。而 DFlash2 是唯一有 1M + YaRN 现场验证的栈（hasso5703 的 690K，以及 incoai #8 的 539K）。因此计划里应把 ALGO=DFLASH + 1M 当作「已知可行」的参照臂，而不是备选项。

### 12.4 维护窗口 A/B 计划（更新版）

| 顺序 | 命令 | 目的 |
|---|---|---|
| 1 | MODE=extrap CONTEXT_LENGTH=524288 ./start-dspark-long.sh | DSpark 零改动对照（不超 262K 时逐字节不变），拿纯外推基线 |
| 2 | MODE=draftmaxpos CONTEXT_LENGTH=506000 ./start-dspark-long.sh | 主推：target 内联 YaRN + 草稿只抬 max_pos（= hasso 对草稿做的事，但保留 DSpark 训练 rope）+ 窗口留余量 |
| 3 | MODE=draftyarn CONTEXT_LENGTH=506000 ./start-dspark-long.sh | 草稿 rope 换成与 target 同套 YaRN（hasso/社区同款） |
| 4 | ALGO=DFLASH CONTEXT_LENGTH=506000 ./start-dspark-long.sh，再 CONTEXT_LENGTH=1010000 | 复刻 hasso 现场配置，作为「应当成功」的参照 |
| 5 | CONTEXT_LENGTH=506000 ./start-long.sh | MTP 兜底对照 |

每轮取三个数：accept len/rate（对照 262K 现网 2.0-4.9 / 0.15-0.69）、bench.py 吞吐、bench-long-ctx.py 长文召回。
注意：长上下文档必须先降 mem-fraction 到 0.70 并加 docker 内存上限（脚本已默认），否则有把宿主机拖到断电的风险。

### 12.5 本轮采纳进本机脚本的清单

- start-dspark-long.sh：默认 MODE=draftmaxpos；长上下文 MEM_FRACTION 默认 0.70；--memory/--memory-swap 100g 硬上限；-e HF_HUB_OFFLINE=1；长上下文自动加 --disable-flashinfer-autotune；ALGO=DFLASH 支线。
- make-yarn-variant.py：新增 --draft-keep-rope（只抬草稿 max_position_embeddings，保留其训练 rope），产出 qwen38-dspark-maxpos-yarn512 / qwen38-dspark-maxpos-yarn1024。
- 参考：该仓库 patch-yarn.py、qwen38-sglang-1m.service.template、README 的 The 1M context mode 与 The GB10 unified-memory trap 两节。

---

## 13. 维护窗口实测结果（2026-09-10，已上线）

> 完整切换记录见同目录 `LONG-CONTEXT-CUTOVER.md`。

### 13.1 结果

| 项 | 切换前 (DSpark 262K) | 切换后 (DSpark 506K, MODE=draftmaxpos) |
|---|---|---|
| 上线上下文 | 262,144 | **506,000** |
| bench.py 非思考 400tok | 21.5 / 20.2 tok/s | **24.8 / 23.0 tok/s** |
| 大海捞针 | 不可用（超窗口） | **310,427 tok ✅ / 465,855 tok ✅** |
| KV 池 | 约 1.30M token @mem 0.85 | 1,131,401 token @mem 0.70 |
| 宿主机可用内存 | 18 GB（swap 2 GB） | **27 GB（swap 0.3 GB）** |
| accept len (rate) | 2.8-3.95 (0.44-0.57)† | 2.05-3.23 (0.15-0.32) |

† 基线样本取自旧容器日志尾部（同一 bench 窗口），非严格配对；严格配对需再切一次基线（约 20 分钟）。

### 13.2 结论

- **DSpark 保住了**：端到端吞吐与基线持平甚至略好，长上下文与投机解码可以共存——**前提是 YaRN 走 config.json 而不是 CLI 覆盖**。
- 接受率有下降迹象（草稿只抬了 max_pos、rope 仍为训练值），但未反映到墙钟吞吐上；若要精确量化，做一次配对 A/B。
- 现场踩坑两条：① DFLASH 不支持 `--mamba-radix-cache-strategy extra_buffer_lazy`（引擎断言，须 `extra_buffer`）；② 本镜像未注册 `DFlash2DraftModel`，跑 DFlash2 必须用官方 dev 镜像 `lmsysorg/sglang:dev-qwen38-27b-dflash2`（已拉取备用）。
- 首 token 延迟随长度线性增长：310K ≈ 9.5 分钟、466K ≈ 18 分钟冷 prefill；同前缀二次请求走缓存。
