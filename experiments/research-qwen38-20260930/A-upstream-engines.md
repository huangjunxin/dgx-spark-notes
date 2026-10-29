# A — 上游引擎状态调研（SGLang 为主，vLLM / TensorRT-LLM / llama.cpp / LMDeploy）

**调研日期：2026-09-30**　**基线对照日：2026-09-18**（本地现行部署 = `lmsysorg/sglang:dev-qwen38-27b-dflash2`，内部版本 `0.0.0.dev1+g5f55db35e`）

**方法与可信度分级**

- 一手来源 = GitHub API / GitHub Raw / 官方文档站 / Docker Hub Registry API / 官方 release notes。
- 标注约定：**【官方/上游】**= 项目自己的 release / PR / 文档；**【社区】**= 第三方仓库、论坛、个人 Gist，未经上游确认。
- 本次 GitHub REST API 匿名额度（60/h）已耗尽，故部分条目无法二次核对 PR diff；凡不能核实的，下文明确写"未核实"。

---

## 0. 结论速览（相对 2026-09-18 的净变化）

| 问题 | 结论 | 关键依据 |
|---|---|---|
| 有没有比 `dev-qwen38-27b-dflash2` 更新的官方镜像？ | **有，但那个 tag 就是目前"官方指定"的 DFlash2 镜像本身**（2026-08-22 构建，未再更新）。同时 `v0.5.20` 已于 **2026-09-18** 正式发布并有官方镜像。**DFlash2 的 pin 镜像不再是唯一最优解** | 见 §1 |
| 正式版本是否原生支持 Qwen3.8-27B + GB10/sm_121？ | **是。Qwen3.8-27B 在 v0.5.19（2026-09-05）成为正式支持模型；v0.5.20（2026-09-18）新增 Qwen3.8 NVFP4 在 DGX Spark 的启用修复。cookbook 已在 GB10 上验证全部 80 个配置** | 见 §1、§5 |
| 两个阻塞 issue 修了吗？ | **两个都还 open。** #33505 底层根因（浅合并）**其实已在 main 修好**，但 issue 没关；#33454 的修复 PR #33963 **仍未合并** | 见 §2 |
| 有没有 draft 参数隔离？ | **仍没有。** 新增了 `--speculative-draft-load-format` / `--speculative-draft-kv-cache-dtype` 等同族隔离参数，但 **`--speculative-draft-json-model-override-args`（PR #35841）未合并**，target 的 `json_model_override_args` 仍无条件下发 draft | 见 §2.3 |
| FP4 KV cache 可用吗？ | **nvfp4 / fp4_mx_block16 已是文档化选项（实验性），但官方文档明确只在 SM100 说原生；SM120 走 FP8 E4M3 反量化路径。对 GB10 无"正式可用"承诺** | 见 §3.1 |
| DGX Spark 统一内存记账修了吗？ | **没有官方修复。**vLLM 侧有一个 GB10 专用修复（#54048）已进 v0.29.0，但另一个统一内存 CUDA graph 估算修复（#46932）**仍 open 未合并**。SGLang 侧重测口径仍是 concurrency 1 | 见 §4、§5.3 |

---

## 1. 镜像与正式版本（确切 tag + 日期）

### 1.1 SGLang 正式 release

| tag | 类型 | 发布日期 | 说明 |
|---|---|---|---|
| **v0.5.20** | 正式 | **2026-09-18 22:41 UTC** | 713 PR / 237 贡献者；含 Qwen3.8 NVFP4 on DGX Spark 修复 |
| v0.5.19 | 正式 | 2026-09-05 02:27 UTC | 786 PR；**Qwen3.8-27B 首次成为正式支持模型** |
| v0.5.18 | 正式 | 2026-08-22 00:09 UTC | — |
| v0.5.17 | 正式 | 2026-08-08 00:19 UTC | — |

来源：【官方/上游】[GitHub releases API](https://api.github.com/repos/sgl-project/sglang/releases?per_page=20)（查询于 2026-09-30）、[v0.5.20 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)、[v0.5.19 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.19)。

> **v0.5.19 的新增模型表**明确列出：`Qwen3.8-27B | Autoregressive | PR #34859 | cookbook link`。PR #34859「Qwen3.8-27B Model Support」于 **2026-08-19 08:31 UTC 合并**（【官方/上游】[PR #34859](https://github.com/sgl-project/sglang/pull/34859)）。cookbook 页脚亦写"Every cell above … is measured on **v0.5.19**"。

### 1.2 Docker 镜像 tag（Docker Hub Registry API，查询于 2026-09-30）

| tag | last_updated (UTC) | digest（前 16 位） | 备注 |
|---|---|---|---|
| `dev-qwen38-27b-dflash2` | **2026-08-22 03:26** | `sha256:616a3e97f45191af` | ← **本地正在用的 tag**；与 `dev-cu13-qwen38-27b-dflash2` 同 digest |
| `dev-cu12-qwen38-27b-dflash2` | 2026-08-22 03:26 | `sha256:440744edff15a933` | CUDA 12 变体（**v0.5.20 起 CUDA 12 lane 已退役**） |
| `qwen38-27b` | 2026-08-14 15:03 | `sha256:febfb971c7352570` | 见 §5.2 issue #34872 引用的 digest |
| `qwen38` | 2026-08-12 15:03 | `sha256:3f37902a0a4acb50` | — |
| `qwen38-27b-cu129` | 2026-08-14 15:05 | `sha256:e35dfb0beaf6b1fb` | — |
| **`v0.5.20`** | **2026-09-18 09:33** | `sha256:06e4f2ed21afde4f` | ＝ `v0.5.20-cu130`（同 digest），CUDA 13 |
| `v0.5.20-runtime` / `-xpu` / `-rocm*` / `-cann*` | 2026-09-18 | — | — |
| `dev` / `dev-cu13` | **2026-09-30 01:00** | — | 当日滚动 dev |
| `nightly-dev-20260930-cae69be5` | 2026-09-30 01:00 | — | 2026-09-30 最新 nightly |
| `dev-cu13-nightly-0929` / `dev-nightly-0929` | 2026-09-30 04:23 | — | 第三方 nightly 别名 |
| `dev-qwen38flashnext` | 2026-09-03 11:44 | `sha256:5ae5816783d58e2e` | Qwen3.8-**Flash-Next** 专用（不同模型） |
| `dev-qwen38-next-local` | 2026-09-07 11:12 | `sha256:9d2a843c706c74bc` | 同上 |
| `dev-dspark` / `dev-cu13-dspark` | 2026-07-07 03:50 | `sha256:e49bb65c33261280` | DSpark 专用 dev，**2 个多月未更新** |
| `dev-cu13-inkling-small-dgx-spark` | 存在 | — | DGX Spark 专用 tag 已出现 |

来源：【官方/上游】[Docker Hub lmsysorg/sglang tags API](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/?page_size=100&ordering=last_updated)（查询于 2026-09-30；共抓取 500 个 tag）。

**关键判断：**

1. **不存在比 `dev-qwen38-27b-dflash2` 更新的"DFlash2 专用"官方 tag。** 该 tag 最后一次构建就是 2026-08-22，与本地所见一致。滚动更新的是通用 `dev` / `dev-cu13` / `nightly-dev-*`。
2. **`v0.5.20` 是 2026-09-18 之后唯一的新正式镜像 tag，日期恰在基线上。** 但 §5.2 显示 DFlash2 / NVFP4 的官方验证口径写在 **v0.5.19**，而官方 cookbook 的 Docker 指引已改为通用 `lmsysorg/sglang:latest`（见 §5.1）。
3. 社区仓库 MiaAI-Lab 的 CHANGELOG 记录：2026-09-05 起"官方镜像已带 DFlash2 + quantized-lm_head selector"，2026-09-09 把默认 pin 换成 `nightly-cu134-20260909-708f51e`（index digest `sha256:00205b89f746…`），理由是 `dev-qwen38-27b-dflash2` 早于 PR #35255（合并 2026-09-04，修 zombie request），并明确写"**Re-pin to `v0.5.20` when tagged**"。【社区】[MiaAI-Lab CHANGELOG](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/blob/main/CHANGELOG.md)

---

## 2. 两个已知阻塞点的当前状态

### 2.1 issue #33505 — `--json-model-override-args` 不递归

| 字段 | 值 |
|---|---|
| 状态 | **open** |
| 创建 | 2026-08-04 07:39 UTC |
| **最后更新** | **2026-08-04 07:40 UTC** |
| 评论数 | **0** |
| 关闭时间 | 无 |

来源：【官方/上游】[issue #33505](https://github.com/sgl-project/sglang/issues/33505)（GitHub API，查询于 2026-09-30）。

**重要修正（本次调研新发现）：issue 描述的浅合并根因，在 main 分支上其实已经被修好了。**

issue 原文指向 `sglang/srt/utils/hf_transformers/config.py` 里 `config.update(model_override_args)` 只做顶层浅合并。**当前 main 的对应代码已经是深合并**（读取于 2026-09-30）：

```python
# python/sglang/srt/utils/hf_transformers/config.py  (main, 2026-09-30)
if model_override_args:
    if isinstance(config, DeepseekV41Config):
        model_override_args = normalize_deepseek_v41_config(model_override_args)
    # A plain update() setattrs a dict-valued override straight onto the
    # config, so '{"text_config": {...}}' on a VLM would replace the whole
    # sub-config with a dict and break attribute access downstream.
    for key, value in model_override_args.items():
        current = getattr(config, key, None)
        if isinstance(value, dict) and isinstance(current, PretrainedConfig):
            current.update(value)      # ← 递归合并
        else:
            setattr(config, key, value)
```

来源：【官方/上游】[raw config.py @ main](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/utils/hf_transformers/config.py)（第 324–335 行，读取于 2026-09-30）。

对应的修复 PR 是 **[#33351](https://github.com/sgl-project/sglang/pull/33351)**「[misc] Deep-merge nested config overrides and parse request bodies with orjson」，**已于 2026-08-03 06:55 UTC 合并**（即 issue 创建前一天）。

> **结论**：issue #33505 的原始症状（YaRN 的 `rope_parameters` 嵌套 override 触发 `Unrecognized keys in rope_parameters` 而启动失败）**在 v0.5.18 之后的任何构建中应已消失**；issue 本身 stale 未关。**本地镜像 `g5f55db35e`（2026-08-22）已包含该修复。**
> **但**：这不解决第二个问题 —— override 被无条件下发到 draft 模型。

### 2.2 issue #33454 — DSpark verify 窗口不裁剪

| 字段 | 值 |
|---|---|
| 状态 | **open** |
| 创建 | 2026-08-04 01:58 UTC |
| **最后更新** | **2026-08-06 20:57 UTC** |
| 评论数 | 3 |
| 关闭时间 | 无 |

来源：【官方/上游】[issue #33454](https://github.com/sgl-project/sglang/issues/33454)（GitHub API）。

issue 正文要点（原文）："DSpark always allocates a full verification window, even when a request has fewer valid context positions remaining. At the context boundary this creates positions greater than or equal to `max_position_embeddings` and can crash the target model with an illegal RoPE table read."

- 复现场景：DeepSeek-V4-Flash，`context_len=1048576`，DSpark `gamma=5`，`verify_num_draft_tokens=6`；某请求 145,747 输入 + 902,822 输出，到 `full token: 1048576` 时 TP1 崩溃。
- 崩溃点：`fused_q_norm_rope<__nv_bfloat16, 512, 64, long, true>`，`CUDA Exception: Warp Illegal Address`，失败指令是对 `freqs_cis[position]` 的 64-bit load。
- issue 明确写道 bug **在当时的 main commit `572924634b9f18ea86da039929db4337508c89af` 上仍存在**，指向 `python/sglang/srt/speculative/draft_worker_common.py` 的 `build_block_pos_offsets()`。

**3 条评论（全部日期与要点）：**

| 作者 | 日期 | 内容 |
|---|---|---|
| junliu-mde | 2026-08-04 02:06 UTC | 提议"简单 workaround"：启用投机解码时把 app ctx 减小 draft 长度（如 100000 → 99995） |
| haimianxing | 2026-08-04 02:43 UTC | 认领，提出**按请求 clamp** verify 窗口到剩余 context/generation 预算，并列出 4 项回归测试计划；请求 @ch-wan 确认方向 |
| hassellof | **2026-08-06 20:57 UTC** | 支持 clamp 方案，并给出**先前技术**："vLLM handles the related boundary in its speculative config validation (**v0.26.0 speculative.py around L1004**)"；自述在 4× RTX PRO 6000 上以 **1M 配置 context + DSPARK** 跑生产，愿意做负载测试 |

来源：【官方/上游】[issue #33454 comments API](https://api.github.com/repos/sgl-project/sglang/issues/33454/comments)。

### 2.3 修复 PR 与替代方案状态

| PR / issue | 标题 | 状态 | 创建 | 最后更新 | 合并时间 |
|---|---|---|---|---|---|
| **#33963** | [Spec][DSpark] Clamp per-request verify window to context/generation budget (fix #33454) | **open**（未合并） | 2026-08-07 07:08 | **2026-08-07 09:46** | — |
| **#35841** | [Spec][Args] Add `--speculative-draft-json-model-override-args` to override the draft model config independently of the target's | **open**（未合并） | 2026-08-21 09:34 | **2026-08-24 03:55** | — |
| #35583 | feat(spec): allow independent draft model config overrides | **closed，未合并** | 2026-08-19 22:52 | 2026-08-23 16:34（关闭） | — |
| #35753 | [docs] Tell Qwen3.8-27B DFLASH2 users to build from main | closed，**已合并** | 2026-08-20 21:04 | 2026-08-20 23:34 | 2026-08-20 23:34 |
| #33351 | [misc] Deep-merge nested config overrides … | closed，**已合并** | 2026-08-03 06:16 | 2026-08-03 06:55 | 2026-08-03 06:55 |
| #34872 | [Cookbook] Qwen3.8-27B-FP8 validated on one DGX Spark at mem-fraction-static 0.70 | **closed，未转成 PR** | 2026-08-14 18:15 | 2026-08-14 18:22 | — |

来源：【官方/上游】GitHub API 逐个查询（查询于 2026-09-30）。

**#33963 与 #35841 自 2026-08-24 起再无任何活动（>5 周无更新），可判定为 stale。**

**draft 参数隔离的现状 —— 仍未实现（本次直接查源码核实）：**

在 main 的 `python/sglang/srt/arg_groups/fields/spec.py`（265 行，读取于 2026-09-30）中，draft 专属参数已经有一批：

```
speculative_draft_model_path / _revision / _load_format / _attention_backend /
_draft_kv_cache_dtype / _draft_window_size / _draft_sink_size /
_draft_model_quantization / _dsa_topk_backend …
```

**但其中没有 `speculative_draft_json_model_override_args`。** 反过来，`json_model_override_args` 定义在 `arg_groups/fields/model.py:144`，并且在 draft 路径上被无条件使用：

```python
# python/sglang/srt/configs/model_config.py  (main, 2026-09-30, 约 790–820 行)
quantization = (
    cfg.speculative_draft_model_quantization if is_draft_model else cfg.quantization
)
override_config_file = (
    cfg.decrypted_draft_config_file if is_draft_model else cfg.decrypted_config_file
)
return ModelConfig(
    ...
    model_override_args=cfg.json_model_override_args,   # ← 无条件，draft 也吃 target 的 override
    is_draft_model=is_draft_model,
    ...
)
```

即为了让 draft 与 target 隔离，需要写成 `if is_draft_model else` 的三元表达式的地方，`json_model_override_args` **恰恰没有**。

来源：【官方/上游】[raw spec.py @ main](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/fields/spec.py)、[raw model.py @ main](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/fields/model.py)、[raw model_config.py @ main](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/configs/model_config.py)（均读取于 2026-09-30）。

**社区侧独立确认（与 #33505 / #35841 完全一致）：**

NVIDIA DGX Spark 论坛同一线程里，用户 basbunarhasan 的 1M context 现场报告原文：

> "Do not use `--json-model-override-args` (**it leaks into the DFlash2 draft config and crashes at load**). Instead patch `rope_parameters` directly into **BOTH `config.json` files** in the HF cache (**target model AND draft**): `rope_type yarn`, `factor 4.0`, `original_max_position_embeddings 262144`, `max_position_embeddings 1010000`. Then `--context-length 1010000`, `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`, and `--mem-fraction-static 0.70`: about 960K tokens of KV pool on the GB10, DFlash2 acceptance unchanged."

同一帖还报告：单次 opencode 会话跑到 **535,361 tokens 上下文**（两倍原生窗口，零 compaction）、约 360K 生成 token、239 agent steps、274 tool calls。

**⚠️ 但该帖日期是 2026-08-23**，早于基线日 2026-09-18 —— 属于"已知现状的社区来源背书"，**不是 9-18 之后的新变化**。

来源：【社区】[NVIDIA Developer Forums 帖 #380257](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257/83)（post by basbunarhasan，2026-08-23）。

---

## 3. 长上下文（≥500K）相关引擎新特性

### 3.1 KV cache 量化新选项：nvfp4 / fp4_mx_block16

官方文档《Quantized KV Cache》现在明确列出 **FP4 格式**：

> "**FP4 Format** — OCP specifies MXFP4 (Microscaling FP4), a 4-bit floating-point format. SGLang exposes two **experimental** E2M1 KV-cache recipes:
> - **`nvfp4`**: NVIDIA FP4 with 16-value blocks, E4M3 block scales, and a per-tensor global scale.
> - **`fp4_mx_block16`**: An E2M1 block-size-16 compatibility recipe. It is distinct from the standard OCP MXFP4 block-size-32 format."

**容量收益（官方口径）**：BF16 → FP4 支持约 **3.56×** 更多 token（`2 / 0.5625`，含 block-scale 开销）。

**SM100 原生配方与 SM120 的关键差异（对 GB10 直接相关）**：

> "`nvfp4` means that prefill consumes the packed native NVFP4 cache directly without any additional dequantization. The default value, `auto`, selects this native NVFP4 mode on **SM100** and **FP8 E4M3 dequantization on SM120**. Decode consumes native NVFP4 in both recipes."

另有 `--prefill-kv-cache-dequant-dtype` 用于选择在线反量化 dtype。

**精度风险（官方自己的数据，摘录）**：

| 模型 | 数据集 | KV16 | KV8 (FP8 E4M3) | KV4 (FP4 E2M1) |
|---|---|---|---|---|
| Qwen3-235B-A22B | gsm8k | 0.9168 | 0.9181 | 0.9186 |
| Qwen3-235B-A22B | **aime25** | 0.7733 | 0.7333 | **0.6000** |
| Qwen3-235B-A22B | gpqa_diamond | 0.7010 | 0.6899 | 0.6778 |
| GPT-OSS-120B | **aime25** | 0.7533 | 0.7667 | **0.3533** |
| GPT-OSS-120B | gpqa_diamond | 0.5081 | 0.5434 | 0.3202 |

官方"Best Practices"原文亦警示：**"Context length: Accuracy degradation may be more pronounced in long-context scenarios, as the accumulation of the quantization error may become significant."**

来源：【官方/上游】[SGLang docs — Quantized KV Cache](https://docs.sglang.io/docs/advanced_features/quantized_kv_cache)（读取于 2026-09-30）。

**对 GB10 / sm_121 的可用性判定：**

- 文档**通篇只提 SM100（原生）与 SM120（FP8 反量化）**，**完全没有提到 SM121 / GB10**，也没有承诺 NVFP4 KV 在 GB10 上可用。
- **没有找到任何 release note 或 PR 声明"nvfp4 / fp4_mx_block16 KV 在正式镜像中对 SM121 可用"。**
- 因此：**不能声称 FP4 KV 在 GB10 + 正式 v0.5.20 镜像上可用**；它属于文档化但平台未验证的实验特性。FP8 E4M3 KV（本地方案）仍是官方 cookbook 对 Qwen3.8-27B 的唯一指定 KV dtype（见 §5.1）。

### 3.2 分层 / radix cache（v0.5.20）

**Unified radix tree 的两个大项（官方 release notes highlights）：**

- **分支点缓存（Branching-Point Caching）用于 SWA 组件**：在请求从共享前缀分叉的位置保留 sliding-window 状态，分支复用而非重算。**DeepSeek-V4-Flash 上共享 system prompt 场景：token hit rate 43.8% → 60.8%，mean TTFT 1.57 s → 1.07 s**（PR #34565）。已移植到 Rust TreeCore（#37584）。
- **可选外部 linker**：通过 **Mooncake 或 UMBP** 让树寻址共享全局内存池（#37381、#37578），并且 **`[Unified Cache][7/N]` 已支持把 MTP / EAGLE / DSpark 的 draft KV cache 也纳入外部 linker**（#37914）——这是相对 9-18 的新特性之一。

其他相关：`[KV Shard 1/4]` 逻辑页放置（#38356）、HiCache 的 MLA host-dedup（#36800）、segment lock 协议替代 `skip_lock_node_ids`（#36848）、buffer-mode sidecar pool（#37424）、L3 storage prefetch 生命周期指标（#37503）等。

来源：【官方/上游】[v0.5.20 release notes — HiCache & Radix Cache 段](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)。

### 3.3 chunked prefill / prefill 加速 / CUDA graph / PD 分离

**chunked prefill 调优（官方 cookbook 对本模型的明确建议）：**

> "`--chunked-prefill-size 2048`: decode steps stall behind each prefill chunk on hybrid GDN models, and **8192-token chunks stall them ~600ms at a time**. 2048 keeps decode inter-token latency smooth under mixed load and also improves single-wave TTFT."

⚠️ 这与本地/社区常用的 8192 相反 —— **官方 cookbook 现在推荐 2048**（H200 例外，用 32768）。来源：【官方/上游】[Qwen3.8-27B cookbook mdx @ main, L343–346](https://raw.githubusercontent.com/sgl-project/sglang/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx)。

**CUDA graph / 统一内存记账相关（v0.5.20 新条目）：**

- `[Unified Memory] Enable prefill cuda-graph capture`（#37418）—— **统一内存下启用 prefill CUDA graph 捕获**，与 GB10 场景直接相关。
- `[Memory] Size the CUDA graph pool from warmup measurements and fix graph-pool borrowing`（#36911）
- `[Memory] Retire graph borrow pool before updating static runs`（#37966）
- `[Memory] Reuse output storage across full prefill CUDA graphs`（#38038）
- `Fix KV-canary workspace accounting after graph capture`（#38596）
- `[Memory] Borrow CUDA graph pool storage for EAGLE sampling`（#35375，v0.5.19）

来源：【官方/上游】v0.5.20 release notes「Piecewise & Breakable CUDA Graph」「Speculative Decoding」段。

**PD 分离 + DSpark（v0.5.20 highlight）：**

> "**DSpark under PD with decode context parallelism.** A DCP1 prefill can now transfer its DSpark draft KV to a DCP-N decode, so hybrid models such as Kimi-Linear run DSpark in disaggregated, context-parallel serving. **Verified on 8x B300 over NIXL and Mooncake up to 256K input**（#37709）。"

注意：验证平台是 **8× B300，非 GB10**。

**v0.5.19 的其他长上下文相关（基线日之前，列作对照）：**

- **DCP on the default Blackwell MLA backend**：`trtllm_mla` 上支持 decode context parallelism；128K 输入下 8× B200 上"plain TP stops scaling around 680 tokens/s，而 DCP keeps going"（#33926）。
- **Unified radix tree 成为所有模型的默认 cache**（#35081，属 Breaking Changes）。

来源：【官方/上游】[v0.5.19 release notes](https://github.com/sgl-project/sglang/releases/tag/v0.5.19)。

### 3.4 v0.5.20 Breaking Changes 中与本地部署相关的

- **CUDA 12 lane 退役**：v0.5.20 起不再发布 `-cu129` / `-cu12` wheel 和镜像；`v0.5.19` 是最后一个带 CUDA 12 lane 的版本（#38404）。已发布的旧 tag 不受影响。
- **SM120 的 FP8 线性层改走 per-tensor（cudnn/nvjet）** 而非 channelwise CUTLASS（#38006）；**32 GB GPU 上 decode CUDA-graph 上限 24 → 48**（TP≥4 时 160），chunked prefill **2K → 4K**（#37898）。
- **17 个 deprecated CLI flag 被移除，且 `get_global_server_args` 现在会 raise**；配置层 `ServerArgs` 变成 `msgspec.Struct`，`dataclasses.asdict` 不再可用（#38375、#38753、#38958）。→ **升级到 v0.5.20 需要审查启动脚本里是否有被删的 flag。**
- **`/v1/responses` 存储默认关闭**，需 `--enable-response-store`（#39122）。
- **ROCm 10 成为 AMD 默认 lane**（与本地无关）。
- **未压缩贡献者格式变更**：Python 格式化 black → ruff-format（#37210）。

来源：【官方/上游】v0.5.20 release notes「Breaking Changes & Upgrade Notes」段。

### 3.5 投机解码：DFlash2 / DSpark / MTP 的新条目（v0.5.20）

| 条目 | PR | 说明 |
|---|---|---|
| DFlash2: local convolution + candidate selector | #35371 | **v0.5.19 引入**；batch 1 相对 no-spec **3.43×**，concurrency 64 相对初代 DFlash 约 **+24%** |
| Support quantized target `lm_head` in the DFlash2 selector | #35496 | **使 DFlash2 能跑在 NVFP4 packed head 上** |
| DFlash2 优化的 Domino rollout | #36899 | 新增 |
| DFlash2 TP>1 的 Domino rollout | #37069 | 新增 |
| `--speculative-dsa-topk-backend` | #36313 | 新增 flag |
| Support speculative decoding with unified SWA memory | #36403 | 新增 |
| Fix DSpark CUDA graph replay with MegaMoE TP attention | #34919 | 修复 |
| Improve CUDA graph and speculative execution output handling | #37329 | 修复 |
| `[Fix] Load Qwen3.5 MTP embedding under PP` | #37471 | 与 Qwen3.8 同族 |
| 自定义 draft worker classes in DSpark | #35397 | v0.5.19 |
| Make draft attention backends extensible | #35932 | v0.5.19 |

⚠️ **注意**：v0.5.20 的 Speculative Decoding 段中，**没有任何一条明确针对 DSpark / DFlash2 的 verify 窗口越界修复** —— 与 §2.2 的判断（#33963 未合并）一致。

来源：【官方/上游】v0.5.20 / v0.5.19 release notes「Speculative Decoding」段。

---

## 4. 其他引擎

### 4.1 vLLM

| tag | 发布日期 | 备注 |
|---|---|---|
| **v0.30.0** | **2026-09-22 05:20 UTC** | 762 commits / 315 贡献者 |
| **v0.29.0** | **2026-09-09 08:54 UTC** | 594 commits / 277 贡献者 |
| v0.28.0 | 2026-08-26 09:46 UTC | — |

来源：【官方/上游】[vLLM releases API](https://api.github.com/repos/vllm-project/vllm/releases?per_page=12)（查询于 2026-09-30）。

**与本地场景直接相关的条目：**

1. **GB10 已进入 vLLM 的硬件修复清单（v0.29.0）**：
   > "cuBLAS out_dtype router GEMM on **all archs including GB10**（#54048）"
   来源：【官方/上游】v0.29.0 release notes「Hardware & Performance → NVIDIA」段。

2. **投机解码的边界修复已经落地（v0.29.0）**：
   > "spec decode no longer padded up to `max_model_len`（**#53962**）"
   PR #53962「[Bugfix][Scheduler] Don't pad spec decode up to `max_model_len`」于 **2026-08-27 15:30 UTC 合并** —— 早于 v0.29.0（2026-09-09）发布。**这是 SGLang #33454 的对应问题在 vLLM 侧已经修掉的证据**，也印证了 issue #33454 里 hassellof 说的"vLLM 在 speculative config validation 里已经处理了相关边界"。
   来源：【官方/上游】[vLLM PR #53962](https://github.com/vllm-project/vllm/pull/53962)、v0.29.0 release notes「Speculative decoding」段。

3. **DFlash2 已在 vLLM 支持（v0.29.0）**：
   > "DFlash2 loading from speculators format（#53797）"
   基础 PR **#52816「[Spec Decode] DFlash2: local convolution + candidate selector」于 2026-08-21 05:27 UTC 合并**，比 SGLang 的 #35371（2026-08-19 合并）晚 2 天。
   来源：【官方/上游】[vLLM PR #52816](https://github.com/vllm-project/vllm/issues/52816)、v0.29.0 release notes。

4. **v0.30.0 的长上下文 / 投机新特性**：
   - **HiSparse**：为 sparse-MLA decode 增加 host-resident tier，在 GPU 压力下把 KV page 溢写到 pinned host memory，top-k miss 由 per-request GPU hot buffer 服务（#53781），带 Prometheus 指标（#56061）、跨 TP rank 共享 host cache（#56629）。
   - **Model Runner V2** 下 **MTP（#46994）与 EAGLE3 / DFlash / DSpark（#50514）在 pipeline parallelism 下**的投机解码；`extract_hidden_states`（#49811）；**FULL CUDA graphs 下的 dual-batch overlap**（#51700）。
   - **Breaking**：**"YaRN aligned with Transformers so vendor YaRN aliases no longer re-scale `max_model_len`（#56446）"** —— ⚠️ **这一条会影响 1M 上下文配置**：如果有依赖 vendor YaRN alias 顺带把 `max_model_len` 放大的用法，升级后会失效，需要显式设 `--max-model-len`。
   来源：【官方/上游】v0.30.0 release notes（GitHub releases API）。

5. **vLLM 官方 recipe 对 Qwen3.8-27B 的状态**（页面标注 "Updated **2026-09-14**"，**早于基线日**）：
   - 要求 **vLLM 0.17.0+**；DFlash2 要求 **vLLM ≥0.28.0**（对应 #52816）。
   - 支持硬件矩阵中**只有 sm120（2× RTX 5090 / 1× RTX 5090）被验证**，**GB10 / DGX Spark 不在 vLLM recipe 的已验证硬件列表里**。
   - **长上下文（1M）指引**：官方 recipe 说通过 `--hf-overrides` 应用 model card 的 YaRN 配置并把 `--max-model-len` 提到 1,000,000，并提示"**for a ~524K workload, halve the card's `factor` to `2.0`**"，且明确 "**Note the override is nested under `text_config` here**"。
   - DFlash2 用法：`vllm serve Qwen/Qwen3.8-27B --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'`。
   来源：【官方/上游】[vLLM Recipes — Qwen/Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)（读取于 2026-09-30，页脚更新时间 2026-09-14）。

**⚠️ 一个仍未修的 GB10 问题**：PR **#46932「[Bugfix][GB10] Fix negative CUDA graph memory estimate on unified-memory GPUs (#44740)」状态 open，最后更新 2026-07-31**，**未合并**。这正是"统一内存上 CUDA graph 记账"的同类问题。
来源：【官方/上游】[vLLM PR #46932](https://api.github.com/repos/vllm-project/vllm/issues/46932)（查询于 2026-09-30）。

### 4.2 TensorRT-LLM

**截至 2026-09-30，TensorRT-LLM 没有 stable release，仍在 1.3.0rc 系列：**

| tag | 发布日期 | 类型 |
|---|---|---|
| **v1.3.0rc29** | **2026-09-29 10:56 UTC** | prerelease |
| v1.3.0rc28 | 2026-09-23 06:34 UTC | prerelease |
| v1.3.0rc27 | 2026-09-18 03:14 UTC | prerelease |
| v1.3.0rc26 | 2026-09-09 09:37 UTC | prerelease |
| v1.3.0rc25 | 2026-08-31 03:24 UTC | prerelease |

来源：【官方/上游】[TensorRT-LLM releases API](https://api.github.com/repos/NVIDIA/TensorRT-LLM/releases?per_page=10)（查询于 2026-09-30）。

**结论**：**截至 2026-09-30，未找到 TensorRT-LLM 对 Qwen3.8-27B 在 GB10 / sm_121 上跑长上下文 + 投机解码的官方支持声明。** 论坛最后可见的 GB10 相关 TRT-LLM 讨论仍是 `1.3.0rc5`（2026-04-08 帖），**未见更新**。
来源：【社区】[NVIDIA Forums — TensorRT-LLM 1.3.0rc5 (GB10)](https://forums.developer.nvidia.com/t/tensorrt-llm-1-3-0-rc5/366006/3)（帖内日期 2026-04-08）。

> 注：SGLang v0.5.20 引入了 TRT-LLM 的 attention kernel 用于 **DeepSeek-V4 的 SM100/SM103**（#30805），但那是 SGLang 内部集成，**不构成 GB10 上的 TRT-LLM 支持**。

### 4.3 llama.cpp

**状态：Qwen3.8-27B 的 GGUF 与 MTP 投机解码均已有实现，但存在未修的 MTP draft 加载 bug。**

- **MTP draft 模型加载 bug — issue #29044，open，创建 2026-09-17**：
  - 标题：「Eval bug: draft-mtp draft model fails to load with -ngl (vector::_M_range_check 1 >= 1); explicit -ts fixes it — likely NaN split when device reports 0 free」
  - 复现环境：llama.cpp master `81aeaeb74`（2026-09-17），CUDA 12.2，RTX 3080 Laptop + WSL2；draft 用 `ggml-org/Qwen3.8-27B-GGUF` 与 `unsloth/Qwen3.8-27B-GGUF` 的 `mtp-Qwen3.8-27B-Q4_0.gguf`，两者都复现。
  - 症状：`-ngl 99 --spec-type draft-mtp --model-draft` → `llama_model_load: error loading model: vector::_M_range_check`。
  - **有可用 workaround**：加 **`-ts 1`**（单设备显式 split）即可加载并工作，报告者测得 **1.9× decode，acceptance 71–82%**。
  - issue 自述根因怀疑（**作者声明未经 debugger 确认**）：`tensor_split` 全零时用 `ggml_backend_dev_memory_free` 归一化，而 WSL 下主模型占满显存后设备上报 `free=0, total>0`，导致 `split_sum==0` → `splits[0] = NaN` → `upper_bound` 返回 `end` → `devices.at(1)` 越界。
  - 来源：【官方/上游】[llama.cpp issue #29044](https://github.com/ggml-org/llama.cpp/issues/29044)（创建 2026-09-17；读取于 2026-09-30，**未见已合并修复**）。

- **Qwen3.8-27B GGUF 基础模型问题（DeltaNet CUDA）— Discussion #27164**：
  老版本 llama.cpp CUDA 后端在 Qwen3.8 的 **DeltaNet / Gated DeltaNet 层**上算错，导致输出全乱码（"Loads normally / Run at full GPU inference speed / Show no crash" 但输出 garbage）。**修复方式 = 更新到较新 commit**（报告者从 `221f0f6` 换到 `ece963f41` ≈ build 10450）。**MTP 不是原因**（无 MTP 也乱码，Vulkan 后端正确）。报告者最终在 RTX 3090 上得到 **~42.9 tok/s**，MTP 正常工作。
  - 来源：【社区】[llama.cpp Discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164)（**日期在本次抓取中未显示，无法确认是否为 9-18 之后**）。

### 4.4 LMDeploy

**截至 2026-09-30，未找到 LMDeploy 对 Qwen3.8-27B 的支持声明。** 检索到的 LMDeploy 官方文档仍是 v0.12.1（含 `Qwen3-VL / Qwen3.5` 等），未出现 Qwen3.8。来源：【官方/上游】[LMDeploy releases](https://newreleases.io/project/github/InternLM/lmdeploy/release/v0.12.1)、[LMDeploy supported_models](https://lmdeploy.readthedocs.io/)。

> **"截至 2026-09-30 未见更新"** —— 未找到 Qwen3.8-27B 的 LMDeploy 支持证据，也未找到明确否认，故记作"未见"。

---

## 5. 官方 cookbook / release notes / DGX Spark 指引的更新

### 5.1 Qwen3.8-27B cookbook 已经大改（这是相对 9-18 最实质的变化来源）

**cookbook 源文件提交历史**（`docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`，GitHub commits API，查询于 2026-09-30）：

| 日期 | 变更 |
|---|---|
| **2026-09-25** | `[feat] add a system one compatible /v1/systemone route`（#41208） |
| 2026-09-15 | `[NPU][DOC] Rename NPU hardware to Ascend A2/A3 Series`（#39389） |
| **2026-09-11** | `[docs] Add the NVIDIA NVFP4 export to the Qwen3.8-27B cookbook`（**#38611**） |
| 2026-09-08 | `[Config] Retire get_global_server_args…`（#38375） |
| 2026-09-07 | `[NPU] Adapt DFlash2 speculative decoding to Ascend NPUs`（#35629） |
| 2026-08-24 | `[docs] Split the Qwen3.8-27B NVFP4 cells by lm_head precision`（#36020） |
| **2026-08-22** | `[docs] Re-measure the Qwen3.8-27B RTX 5090 / RTX PRO 6000 / DGX Spark grids on 1cf2b8c`（#35825） |
| 2026-08-20 | `[docs] Point the Qwen3.8-27B DFLASH2 note back at the rolling dev image tag`（#35767）/ `Tell … DFLASH2 users to build from main`（#35753）/ `Add DFlash2 speculative cells`（#35663） |
| 2026-08-17 | `docs(cookbook): add Qwen3.8-27B DGX Spark configs`（#35121） |

> **⚠️ 关于"相对 2026-09-18 有哪些新变化"：cookbook 在 9-18 之后只有 2 次改动 —— 2026-09-25（#41208，无关的 systemone 路由）和 2026-09-15（#39389，NPU 更名，早于基线日）。也就是说，cookbook 关于 Qwen3.8-27B 的实质内容（NVFP4 cell、DGX Spark pin、DFlash2）在 9-18 之前就已完成，#38611（2026-09-11）是最后一次实质更新。**

**cookbook 当前的关键内容（【官方/上游】[raw mdx @ main](https://raw.githubusercontent.com/sgl-project/sglang/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx)，L1–L346）：**

1. **checkpoint 清单已换成 5 个**，NVFP4 三个：
   - `RadixArk/Qwen3.8-27B-NVFP4`（FP4 head，lm_head 打包为 FP4）
   - `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead`（dense BF16 head，"every recipe on this page was measured against it"）
   - `nvidia/Qwen3.8-27B-NVFP4`（NVIDIA ModelOpt 导出的同一 W4A4 body + FP4 head，"identical tensor set, identical 21.9 GB on disk"）

2. **DGX Spark 的官方 pin 与容量推理（原文 L225–L246）**：
   > "On DGX Spark the 128GB is **unified memory shared with the host CPU**, so all three checkpoints fit, and its cells reuse the RTX PRO 6000 recipe at **`--mem-fraction-static 0.80`** rather than a separate operating point. The one lower pin is the unified pool pricing the host's memory too: **0.85 of 128GB leaves ~8GB for the OS — exactly DGX OS earlyoom's SIGTERM threshold** — and the first long prefill or boot-time graph capture dips under it and gets the scheduler killed with **`exit code -15` and no traceback** (`journalctl -u earlyoom` shows the kill). **At 0.85, 15 of the 48 cells were killed that way** … at 0.80 every cell served on every attempt."
   >
   > "**Validated on SM121 / aarch64**: all **80 configurations** (5 checkpoints × Speculative Decoding × Serving Strategy × Mamba SSM Dtype, **DFLASH2 included**) served on **GB10 on `v0.5.19` at ISL 8192 / OSL 1024, concurrency 1**, and each scored the full 1319-question GSM8K (**93.18–95.15%**)."
   >
   > 三个 GB10 主机坑（原文）："docker GPU access is **CDI-only** (`--device nvidia.com/gpu=all`, as no `nvidia` runtime is registered); `nvidia-smi` reports **`Not Supported` for memory** because it is unified with the CPU — gate a relaunch on **`MemAvailable` in `/proc/meminfo`** instead; and the BF16 checkpoint takes ~6.5 minutes just to load its 18 shards from NVMe, so **budget ~10 minutes to READY**。"

3. **投机解码的官方推荐（L254–L296）**：
   - **MTP**：`--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（用 in-checkpoint MTP head；原文档写 `NEXTN`，是 `EAGLE` 的别名）。
   - **DSpark**：`--speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark`；**不传 `--speculative-num-draft-tokens`**，verify 窗口 = `--speculative-dspark-block-size`(gamma) **+ 1**，gamma 从 draft checkpoint 自动推断（本 checkpoint 为 7，故 D=8）。
   - **DFlash2**：`--speculative-algorithm DFLASH --speculative-draft-model-path **incoai/Qwen3.8-27B-DFlash2** --speculative-num-draft-tokens 8`。
     > ⚠️ **cookbook 现在指向 `incoai/Qwen3.8-27B-DFlash2`，而本地方案用的是 `z-lab/Qwen3.8-27B-DFlash2`。** 在 GB10 上，社区脚本同时出现过 `z-lab/…`（论坛 run.sh）和 cookbook 的 `incoai/…`。**本次调研未能核实两者是否为同一 checkpoint 的不同 repo**（GitHub API 额度耗尽，且未在官方 release note 中找到改名说明）。

4. **KV cache dtype 明确规定**：
   > "The two RadixArk checkpoints declare `kv_cache_quant_algo: FP8`, so SGLang's default `--kv-cache-dtype auto` already puts their KV pool in `fp8_e4m3`. The NVIDIA export ships no `kv_cache_scheme`, so `auto` would leave its pool in BF16 instead. **Every recipe on this page pins `--kv-cache-dtype fp8_e4m3` explicitly**."
   → **官方 cookbook 对 Qwen3.8-27B 只指定 FP8 KV，没有使用 FP4 KV。**

5. **官方 Docker 指引已改为通用 tag**：
   ```bash
   docker pull lmsysorg/sglang:latest
   ```
   并在 Python 路径给 `uv pip install --prerelease=allow sglang`。
   → 注意：**cookbook 现在不再指向 `dev-qwen38-27b-dflash2`**。MiaAI-Lab 的 CHANGELOG 说"cookbook maps `dev-qwen38-27b-dflash2` to DGX Spark"是 2026-09-05 的记录，之后已被 `latest` 取代。

6. **cookbook 的验证声明口径**：
   > "Every cell above — RTX 5090, RTX PRO 6000 and DGX Spark, across all five checkpoints and every Speculative Decoding / Serving Strategy / SSM dtype combination — is measured on **v0.5.19**. That is **202 cells**, each one served and scored on the full 1319-question GSM8K (93.18–95.15%). **The serving envelope behind the pins is ISL 8192 / OSL 1024 at concurrency 1**; throughput and acceptance-length numbers were not re-taken in that sweep."

7. **`--mamba-full-memory-ratio` 被官方标为"唯一重要的吞吐相关 sizing flag"**；默认 0.9 会 over-provision KV 并静默 clamp 并发。DGX Spark 例子给 `--mamba-full-memory-ratio 4.59`（state/slot 153.9 MB、KV/token 32.8 KB、S=5）。
   > 社区仓库另记一条官方默认值陷阱：`.env.example` 留空 → SGLang 默认 0.9 → KV pool 比文档值低约 40%。【社区】[MiaAI-Lab issue #21](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/21)

### 5.2 issue #34872（2026-08-14，已关但**从未转成文档 PR**）

标题：「[Cookbook] Qwen3.8-27B-FP8 validated on one DGX Spark at mem-fraction-static 0.70」。**状态 closed，`merged=n/a`，关闭于 2026-08-14 18:22，0 评论。**

内容要点（原文）：
- 在 1× GB10 / SM121 上用官方专用镜像测 `Qwen/Qwen3.8-27B-FP8`：
  - 镜像 digest `sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`（＝ `qwen38-27b` tag，2026-08-14）
  - SGLang 版本 `0.0.0.dev0+qwen38.27b.g561c8f3`
  - attention backend FlashInfer，GDN 内核 Triton，prefill chunk 8192，**`--mem-fraction-static 0.70`**
  - 原生 MTP：EAGLE 3 steps / topk 1 / 4 draft tokens
  - 72/72 请求完成；**最大测试 prompt 32,768 tokens；最大并发 8**
  - 0.75 的预跑越过了 512 MiB swap 增长护栏（超 26,406,912 bytes）发生在 CUDA graph capture 期间，被 watchdog 停容器
- 提出文档改动建议：`--mem-fraction-static 0.70 --attention-backend flashinfer --chunked-prefill-size 8192`
- 并称 `lmsysorg/sglang:qwen38-27b` 在 arm64 GB10 上成功拉起

来源：【官方/上游】[issue #34872](https://github.com/sgl-project/sglang/issues/34872)（GitHub API，查询于 2026-09-30）。

> **对比现行 cookbook：官方最终采用的是 `--mem-fraction-static 0.80`（#35825 于 2026-08-22 重测），而不是 #34872 建议的 0.70。** 且 #34872 只测到 32K prompt、并发 8 —— **远未触及 ≥500K 长上下文**。

### 5.3 DGX Spark 统一内存记账 —— 官方未修，社区有实测文档

**社区文档（重要，且明确指出这是引擎缺陷）：**【社区】[hasso5703/dgx-spark-qwen38 — docs/gb10-memory.md](https://github.com/hasso5703/dgx-spark-qwen38/blob/main/docs/gb10-memory.md)（读取于 2026-09-30）：

> "SGLang's memory accounting **does not see 25–40 GB** of transient allocations on GB10 unified memory (the flashinfer fp8 autotuner and CUDA graph capture allocate outside the tracked pool). Running `--mem-fraction-static` above **0.50**, or running SGLang **natively** (outside Docker), can drive host available memory to **zero** … that means a hard freeze only a power cycle fixes."

该仓库的自保措施（社区实践，非官方建议）：
- `--memory 100g --memory-swap 100g`（并注明"**the cgroup does _not_ see CUDA unified allocations**, so the real guard is the fraction"）
- 262K 模式 `--mem-fraction-static 0.50`
- **1m 模式用 0.76**（同 docker cap + 关掉 autotuner）；"It ran **0.70 until v1.14**，the number moved with the image, not with the appetite: the official release claims a smaller static budget for the same fraction, so **0.70 there cost 15% of the pool (770,118 tokens against 906,524)** while leaving 10 GB of GPU memory unclaimed. 0.76 hands that back and still leaves a wider margin than the old pin did, **21.30 GB free after graph capture against 19.38**. **0.80 was measured crashing** under 3 concurrent requests (2 GiB free, Triton `CUDA operation not permitted`)."
- `Restart=always`（不用 `on-failure`，因为"a Triton compile crash measured on 2026-08-22 ended in `SystemExit: 0`, which `on-failure` never relaunches"）
- 与 cookbook 0.80 的关系，社区自己写的调和："Its GB10 cells ran 48 configurations at ISL 8192 / OSL 1024, **concurrency 1**, boot-and-serve only, and 0.80 served every cell on every attempt … This repo's 0.80 failure was measured under **three concurrent requests** on a box that also runs the operator's tools. One number is a **single-stream boot-and-serve bound**, the other is a **multi-client operating point**."

> **结论：截至 2026-09-30，SGLang 侧未见任何"统一内存 mem-fraction 记账修复"的 release note 或已合并 PR。** v0.5.20 只有 `[Unified Memory] Enable prefill cuda-graph capture`（#37418）这一条相关条目，**不构成记账修复**。
>
> **对本地的直接含义：`--mem-fraction-static 0.70` + 1M context 属于在"1m 模式 0.76"附近的激进取值，其安全边界依赖 Docker `--memory 100g` 这个"cgroup 看不见 CUDA 统一分配"的软护栏 —— 社区明确说真正的护栏是 fraction 本身，而不是容器内存上限。**

### 5.4 NVIDIA 官方 DGX Spark 指引

- 检索到的 **NVIDIA Developer Forums DGX Spark / GB10 板块**有同一主题的超长线程（**post by basbunarhasan 后仍有 80+ 回复**，日期覆盖 2026-08-23 ~ 2026-08-27 及以后），但**这是用户论坛的项目帖，不是官方指引**。【社区】[forum thread #380257](https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257/83)
- **截至 2026-09-30，未找到 NVIDIA 官方（Docs / Playbooks）针对 Qwen3.8-27B 在 DGX Spark 上的推荐配置文档。** 明确记作"未见更新"。
- DGX Spark 官方论坛仍在活跃的近期话题是**驱动问题**（「Driver Issue with 580.178.04」，发帖"1 天前"即约 2026-09-29），与本文主题无关但提示平台层仍在变动。【社区】[forum #383859](https://forums.developer.nvidia.com/t/driver-issue-with-580-178-04/383859)

---

## 6. 明确"截至 X 未见更新"的项

| 项 | 状态 |
|---|---|
| 更新的 DFlash2 专用官方镜像 tag | **截至 2026-09-30 未见。** `dev-qwen38-27b-dflash2` 最后一次构建仍是 2026-08-22，digest `sha256:616a3e97…` 未变 |
| #33963（verify 窗口 clamp）合并 | **截至 2026-09-30 未见。** 自 2026-08-07 起无活动 |
| #35841（draft override 隔离）合并 | **截至 2026-09-30 未见。** 自 2026-08-24 起无活动 |
| issue #33505 / #33454 关闭 | **截至 2026-09-30 未见。** 两者仍 open |
| SGLang 统一内存 mem-fraction 记账修复 | **截至 2026-09-30 未见。** v0.5.20 release notes 无对应条目 |
| nvfp4 / fp4_mx_block16 KV 在 SM121/GB10 的正式可用声明 | **截至 2026-09-30 未见。** 官方文档只写 SM100 原生 / SM120 FP8 反量化 |
| TensorRT-LLM stable release | **截至 2026-09-30 未见。** 最新仍是 `v1.3.0rc29`（2026-09-29） |
| TensorRT-LLM 对 GB10 + Qwen3.8-27B 的支持声明 | **截至 2026-09-30 未见** |
| LMDeploy 对 Qwen3.8-27B 的支持 | **截至 2026-09-30 未见** |
| llama.cpp Qwen3.8 MTP draft 加载 bug 修复 | **截至 2026-09-30 未见。** issue #29044（2026-09-17）仍 open，workaround 是 `-ts 1` |
| NVIDIA 官方 DGX Spark + Qwen3.8-27B 推荐配置文档 | **截至 2026-09-30 未见** |
| cookbook 在 2026-09-18 之后的 Qwen3.8-27B 实质更新 | **未见实质更新。** 9-18 后仅 2026-09-25 一次无关改动（#41208）；最后一次实质更新是 2026-09-11（#38611） |

---

## 7. 相对 2026-09-18 的新变化清单（净 diff）

**✅ 9-18 当天或之后的新东西：**

1. **SGLang `v0.5.20` 正式发布（2026-09-18 22:41 UTC）+ 官方镜像 `lmsysorg/sglang:v0.5.20`（2026-09-18 09:33 UTC，digest `sha256:06e4f2ed…`，CUDA 13）** —— 这是唯一一个新的正式 tag/镜像。
2. v0.5.20 的 **Unified radix tree 分支点缓存**（#34565）、**外部 linker 支持 MTP/EAGLE/DSpark draft KV**（#37914）。
3. v0.5.20 的 **`[Unified Memory] Enable prefill cuda-graph capture`（#37418）**。
4. v0.5.20 的 **DSpark under PD with DCP**（#37709，验证于 8× B300）。
5. v0.5.20 的 **CUDA 12 lane 退役**、**SM120 FP8 per-tensor 路由**、**17 个 flag 移除 + `ServerArgs` 改 `msgspec.Struct`** 等 Breaking Changes。
6. 2026-09-30 的滚动 `dev` / `dev-cu13` / `nightly-dev-20260930-cae69be5`（**注意 `cae69be5` 不是下游版本号，是 nightly tag 里的 commit 前缀，与本地 `0.0.0.dev1+g5f55db35e` 不是同一 commit**）。
7. **vLLM v0.30.0（2026-09-22）**：HiSparse host-tier、MRV2 下 MTP/EAGLE3/DFlash/DSpark 的 PP 支持、**YaRN 与 Transformers 对齐（#56446，会影响 1M 配置）**。
8. **llama.cpp issue #29044（2026-09-17）**：Qwen3.8-27B MTP draft `-ngl` 加载失败，需 `-ts 1`。
9. **TensorRT-LLM v1.3.0rc27（2026-09-18）/ rc29（2026-09-29）** —— 仍是 prerelease，无 stable。

**❌ 9-18 之前就已存在（不是新变化，勿误报）：**

- Qwen3.8-27B 正式支持（v0.5.19，2026-09-05）
- DFlash2 local conv + candidate selector（PR #35371 合并 2026-08-19）
- DFlash2 quantized lm_head selector（PR #35496 合并 2026-08-20）
- Qwen3.8 NVFP4 on DGX Spark 修复（PR #39126 合并 2026-09-13）
- cookbook 的 DGX Spark 0.80 pin / 80-config GB10 验证 / NVFP4 cell（#35825 2026-08-22、#38611 2026-09-11）
- `--json-model-override-args` 深合并修复（PR #33351 合并 2026-08-03）
- vLLM CUDA graph 统一内存负估算修复 PR #46932 —— **仍未合并**（最后更新 2026-07-31）
- vLLM DFlash2（PR #52816 合并 2026-08-21）、vLLM spec decode 不再 pad 到 `max_model_len`（PR #53962 合并 2026-08-27）
- 社区 1M context + YaRN 双 config.json 补丁方案（论坛 2026-08-23）

---

## 8. 对本地部署的直接含义（结论性判断）

1. **两个上游阻塞点在 2026-09-30 仍然存在**，其中：
   - **#33505 的"浅合并"根因已修**（本地镜像已含），但 **"override 泄漏进 draft"这个真正的痛点未修**，官方隔离 flag（#35841）未合并，**只能继续用"手工改两份 config.json"的社区补丁**。
   - **#33454（verify 窗口越界）修复 PR #33963 已 stale 5 周以上，未合并**。**安全做法仍是把 `--context-length` 收到 `max_position_embeddings - (verify 窗口)` 以下**（issue 首条评论的 workaround），或复现性降载。
2. **升级到 `v0.5.20` 是可行的**（正式 tag 存在、日期就在基线日），但需处理 Breaking Changes，尤其是 **17 个 removed flag + `ServerArgs` 改 `msgspec.Struct`**。
3. **cookbook 的官方 DGX Spark pin 是 `--mem-fraction-static 0.80` + concurrency 1**；本地用 `0.70` 配 1M context 属于**社区路线**（社区 1m 模式用 0.76，且明确 0.80 在三并发下实测崩溃）。**官方从未验证过 ≥500K 上下文**（cookbook 验证口径是 ISL 8192）。
4. **FP4 KV 不要在本平台上当作"正式可用"**：官方文档只对 SM100 承诺原生、SM120 走 FP8 反量化，**SM121 无任何声明**；且官方自己的精度表显示 FP4 KV 在难数据集上掉分剧烈（aime25: 0.7733 → 0.6000）。
5. **Draft checkpoint 来源已变**：cookbook 现在指向 `incoai/Qwen3.8-27B-DFlash2`，本地用 `z-lab/Qwen3.8-27B-DFlash2`。**建议核实两者关系**（本次因 API 额度耗尽未核实）。

---

## 9. 来源链接表

### 官方 / 上游

| # | 来源 | URL | 类型 | 日期 |
|---|---|---|---|---|
| 1 | SGLang v0.5.20 release | https://github.com/sgl-project/sglang/releases/tag/v0.5.20 | release | 2026-09-18 |
| 2 | SGLang v0.5.19 release | https://github.com/sgl-project/sglang/releases/tag/v0.5.19 | release | 2026-09-05 |
| 3 | SGLang releases API | https://api.github.com/repos/sgl-project/sglang/releases?per_page=20 | API | 查询 2026-09-30 |
| 4 | SGLang issue #33505 | https://github.com/sgl-project/sglang/issues/33505 | issue (open) | 创建 2026-08-04 / 更新 2026-08-04 |
| 5 | SGLang issue #33454 | https://github.com/sgl-project/sglang/issues/33454 | issue (open) | 创建 2026-08-04 / 更新 2026-08-06 |
| 6 | SGLang PR #33963 | https://github.com/sgl-project/sglang/pull/33963 | PR (open) | 2026-08-07 / 更新 2026-08-07 |
| 7 | SGLang PR #35841 | https://github.com/sgl-project/sglang/pull/35841 | PR (open) | 2026-08-21 / 更新 2026-08-24 |
| 8 | SGLang PR #35583 | https://github.com/sgl-project/sglang/pull/35583 | PR (closed) | 关闭 2026-08-23 |
| 9 | SGLang PR #33351 | https://github.com/sgl-project/sglang/pull/33351 | PR (merged) | 2026-08-03 |
| 10 | SGLang PR #35753 | https://github.com/sgl-project/sglang/pull/35753 | PR (merged) | 2026-08-20 |
| 11 | SGLang issue #34872 | https://github.com/sgl-project/sglang/issues/34872 | issue (closed) | 2026-08-14 |
| 12 | SGLang PR #34859 (Qwen3.8-27B support) | https://github.com/sgl-project/sglang/pull/34859 | PR (merged) | 2026-08-19 |
| 13 | SGLang PR #35371 (DFlash2 selector) | https://github.com/sgl-project/sglang/pull/35371 | PR (merged) | 2026-08-19 |
| 14 | SGLang PR #35496 (quantized lm_head) | https://github.com/sgl-project/sglang/pull/35496 | PR (merged) | 2026-08-20 |
| 15 | SGLang PR #39126 (Qwen3.8 NVFP4 on DGX Spark) | https://github.com/sgl-project/sglang/pull/39126 | PR (merged) | 2026-09-13 |
| 16 | Qwen3.8-27B cookbook (raw mdx) | https://raw.githubusercontent.com/sgl-project/sglang/main/docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx | docs | 读取 2026-09-30 |
| 17 | Qwen3.8-27B cookbook (渲染页) | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B | docs | 读取 2026-09-30 |
| 18 | SGLang 《Quantized KV Cache》 | https://docs.sglang.io/docs/advanced_features/quantized_kv_cache | docs | 读取 2026-09-30 |
| 19 | SGLang config.py @ main | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/utils/hf_transformers/config.py | source | 读取 2026-09-30 |
| 20 | SGLang model_config.py @ main | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/configs/model_config.py | source | 读取 2026-09-30 |
| 21 | SGLang spec.py @ main | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/fields/spec.py | source | 读取 2026-09-30 |
| 22 | SGLang model.py @ main | https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/arg_groups/fields/model.py | source | 读取 2026-09-30 |
| 23 | Docker Hub lmsysorg/sglang tags | https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/?page_size=100 | registry API | 查询 2026-09-30 |
| 24 | vLLM releases API | https://api.github.com/repos/vllm-project/vllm/releases?per_page=12 | API | 查询 2026-09-30 |
| 25 | vLLM v0.30.0 | https://github.com/vllm-project/vllm/releases/tag/v0.30.0 | release | 2026-09-22 |
| 26 | vLLM v0.29.0 | https://github.com/vllm-project/vllm/releases/tag/v0.29.0 | release | 2026-09-09 |
| 27 | vLLM PR #52816 (DFlash2) | https://github.com/vllm-project/vllm/pull/52816 | PR (merged) | 2026-08-21 |
| 28 | vLLM PR #53962 (spec decode pad) | https://github.com/vllm-project/vllm/pull/53962 | PR (merged) | 2026-08-27 |
| 29 | vLLM PR #46932 (GB10 CUDA graph mem) | https://github.com/vllm-project/vllm/pull/46932 | PR (**open**) | 创建 2026-06-27 / 更新 2026-07-31 |
| 30 | vLLM Recipes — Qwen3.8-27B | https://recipes.vllm.ai/Qwen/Qwen3.8-27B | docs | 页脚更新 2026-09-14 |
| 31 | TensorRT-LLM releases API | https://api.github.com/repos/NVIDIA/TensorRT-LLM/releases?per_page=10 | API | 查询 2026-09-30 |
| 32 | llama.cpp issue #29044 | https://github.com/ggml-org/llama.cpp/issues/29044 | issue (open) | 2026-09-17 |

### 社区（非上游，仅供参考，勿当官方结论）

| # | 来源 | URL | 日期 |
|---|---|---|---|
| C1 | MiaAI-Lab Qwen3.8-27B-SGLang-DGX-Spark CHANGELOG | https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark/blob/main/CHANGELOG.md | 最新条目 2026-09-09 |
| C2 | NVIDIA Forums #380257（1M context 现场报告） | https://forums.developer.nvidia.com/t/qwen3-8-27b-at-34-38-tok-s-on-dgx-spark-open-source-one-command-setup-sglang-nvfp4-dspark/380257/83 | 2026-08-23 |
| C3 | hasso5703/dgx-spark-qwen38 — docs/gb10-memory.md | https://github.com/hasso5703/dgx-spark-qwen38/blob/main/docs/gb10-memory.md | 读取 2026-09-30 |
| C4 | hasso5703/dgx-spark-qwen38 — v1.14.0 commit | https://github.com/hasso5703/dgx-spark-qwen38/commit/4698b685f27f67b426d3e70a98b443a0e3a2014b | 未显示日期 |
| C5 | llama.cpp Discussion #27164（Qwen3.8 DeltaNet CUDA garbage） | https://github.com/ggml-org/llama.cpp/discussions/27164 | 未显示日期 |
| C6 | MiaAI-Lab issue #21（mamba ratio 默认值陷阱） | https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/21 | 未显示日期 |
| C7 | NVIDIA Forums #383859（驱动 580.178.04 问题） | https://forums.developer.nvidia.com/t/driver-issue-with-580-178-04/383859 | 约 2026-09-29 |
| C8 | drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark | https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark | 未显示日期 |
| C9 | amasu/dgx-spark-qwen38 | https://github.com/amasu/dgx-spark-qwen38 | 未显示日期 |

---

## 10. 本次调研的限制

1. **GitHub REST API 匿名额度（60/h）在调研中途耗尽**，因此以下未能核实：
   - PR #33963 / #35841 的具体 diff（是否只是 stale 而非有技术障碍）
   - `z-lab/Qwen3.8-27B-DFlash2` 与 `incoai/Qwen3.8-27B-DFlash2` 是否为同一 checkpoint
   - PR #33351 的完整改动文件列表
2. **HuggingFace 经 r.jina.ai 被封锁**（`Anonymous access to domain huggingface.co blocked`），因此未能直接读取 `RadixArk/Qwen3.8-27B-DSpark`、`incoai/Qwen3.8-27B-DFlash2`、`z-lab/Qwen3.8-27B-DFlash2` 的 model card，也未能核对 cookbook 中 `incoai/…` 与本地 `z-lab/…` 的对应关系。
3. **GitHub 网页版 HTML 抓取只返回导航栏**，所有 GitHub 数据均改用 REST API 获取；API 额度耗尽后未再做 GitHub 侧核对。
4. `z-lab/…` vs `incoai/…` 的差异是本报告最重要的未解疑点，**建议下游单独核实**。
