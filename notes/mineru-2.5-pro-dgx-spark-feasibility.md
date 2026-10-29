# MinerU2.5-Pro × DGX Spark 可行性（结论：当前栈不可用）

> 主机：`spark-XXXX` · NVIDIA DGX Spark（GB10 / Blackwell sm_121，aarch64，128 GB 统一内存，CUDA 13.0）
> 日期：2026-09-15 · 方法：**真机端到端实测**（不是纸面推演）
> 本机既有栈：vLLM **0.27.1** / torch 2.13.0+cu130 / **transformers 5.15.1**（`venv-vllm`）；主服务 Qwen3.8-27B（SGLang，:8888）；稳态 available 约 **25 GiB**

---

## 0. 省流版结论

| 维度 | 判定 |
|---|---|
| **客户端安装** | ✅ **可行**。`pip install mineru` 基础包在 aarch64 上装得干净：只有 magika / onnxruntime / openai / mineru-vl-utils 等纯依赖，**不含 torch / vLLM / transformers / paddlepaddle**（511 MB）。 |
| **模型加载** | ✅ 可行。MinerU2.5-Pro 是 **Qwen2-VL 架构**（`qwen2_vl`），本机 vLLM 0.27.1 **原生注册**（registry L264/L550），实测加载 2.41 GiB / 15.6 s，HTTP 接口正常返回 200。 |
| **实际文档解析** | ❌ **不可用**。模型输出**退化成连续 `!`**，MinerU 判定 `Layout output does not match expected format`，最终 markdown **0 行**（合成图与真实 PDF 均如此）。 |
| **根因** | **推理侧依赖版本不匹配**：MinerU 官方闸门是 `transformers<5.0.0` + `vllm<0.22.0`；本机是 **transformers 5.15.1 + vLLM 0.27.1**。该模型依赖 4.x 时代 transformers 的 Qwen2-VL 处理行为。 |
| **结论** | **不建议在本机部署**。客户端与模型加载都没问题，卡在推理正确性上；要跑通必须为它单独维护一套旧栈环境（transformers 4.57 + torch，或 vLLM 0.21.x），后者在 aarch64/GB10 上的可用性未知，**性价比低**。 |

---

## 1. 版本与模型事实（已核对）

| 项 | 值 |
|---|---|
| HF 模型 | `opendatalab/MinerU2.5-Pro-2605-1.2B`（88,314 下载）/ `MinerU2.5-Pro-2604-1.2B`（248,503 下载，likes 167） |
| 基座 | `MinerU2.5-2509-1.2B`（likes 373）、`MinerU2.0-2505-0.9B` |
| 架构 | `Qwen2VLForConditionalGeneration` / `model_type: qwen2_vl`（hidden 896 / 24 层 / 14 heads / 2 KV heads） |
| 本机 vLLM 支持 | ✅ **原生注册**：`registry.py` L264 / L550 `Qwen2VLForConditionalGeneration → qwen2_vl`（无需 trust-remote-code） |
| 权重 | `model.safetensors` **2312 MB**（bf16）；视觉塔完整（**391 个 `visual.*` 张量 + merger**，lm_head 省略=绑定词表） |
| 许可 | **Apache-2.0**（HF tag） |
| 上下文 | `max_position_embeddings: 32768`、`sliding_window: 32768`、tokenizer `model_max_length: 16384` |
| 量化变体 | 社区 GGUF/MLX 若干（`mradermacher/*-GGUF` 等），本机不适用（vLLM 0.27.1 已移除 GGUF 加载） |
| MinerU 包 | PyPI `mineru` **3.4.5**（稳定）；GitHub 另有 `v4.0.0a6` alpha |

**依赖闸门（关键）**——`mineru` 的可选依赖钉死了版本：

| extra | 关键依赖 |
|---|---|
| `[vlm]` | torch≥2.6、**transformers<5.0.0,>=4.57.3**、accelerate |
| `[pipeline]` | torch、**transformers<5.0.0**、onnxruntime、shapely… |
| `[vllm]` | **vllm<0.22.0,>=0.10.1.1** |
| 基础包 | 无 torch / 无 vllm（纯客户端 + `openai` + `mineru-vl-utils`） |

→ 本机 `venv-vllm` 是 transformers **5.15.1** + vLLM **0.27.1**，**同时越过了两道闸门**。

---

## 2. 官方路径与本机的差异

官方后端矩阵（README）：`pipeline` / `*-engine` / `*-http-client`，最低显存分别 4GB / 8GB / **2GB**；RAM 最低 16GB；磁盘 http-client 仅需 2GB。

官方文档给的轻量客户端用法（**正是本机该走的路**）：

```bash
uv pip install mineru
mineru -p <input> -o <output> -b vlm-http-client -u http://127.0.0.1:30000
```

但官方模型卡 README 的 vLLM 示例里有一行容易被忽略的关键代码：

```python
from mineru_vl_utils import MinerULogitsProcessor
llm = LLM(model="opendatalab/MinerU2.5-Pro-2605-1.2B",
          logits_processors=[MinerULogitsProcessor])   # 必须
```

即：**该模型依赖 MinerU 自带的 logits processor**（`VllmV1NoRepeatNGramLogitsProcessor`，抑制重复 n-gram）。裸 serve 会退化。

---

## 3. 实测记录

### 3.1 客户端安装（✅ 成功）

```bash
python3 -m venv /home/<user>/projects/venv-mineru
/home/<user>/projects/venv-mineru/bin/pip install mineru    # 511 MB
```

- aarch64 轮子全部解析成功（onnxruntime 1.30.0 / opencv-python 5.0.0.93 / magika 1.0.3 …）
- **没有引入 torch / vLLM / transformers / paddlepaddle** → 与 `venv-vllm` 零冲突
- CLI 支持 `-b vlm-http-client -u <url>`（`--url` 即服务端地址）

### 3.2 服务端（✅ 加载成功，含 2 个坑）

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve opendatalab/MinerU2.5-Pro-2605-1.2B \
  --served-model-name MinerU2.5-Pro-2605-1.2B --port 30000 \
  --gpu-memory-utilization 0.06 --max-model-len 16384 --enforce-eager \
  --limit-mm-per-prompt '{"image": 1}' \
  --logits-processors 'mineru_vl_utils.logits_processor.vllm_v1_no_repeat_ngram:VllmV1NoRepeatNGramLogitsProcessor'
```

- ✅ `Resolved architecture: Qwen2VLForConditionalGeneration`；权重加载 **2.41 GiB / 15.6 s**；KV cache 63,232 tokens（16K 上下文）
- **坑 1**：`--max-model-len` 被模型"派生上限 8192"卡住，必须 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 才能开到 16384/32768
- **坑 2**：`--logits-processors` 格式是 **`module:qualname`（冒号）**，不是点路径；写错会报 `ValueError: not enough values to unpack (expected 2, got 1)`

### 3.3 文档解析（❌ 失败）

| 输入 | 结果 |
|---|---|
| 自造合成图（1000×640，含表格/公式） | markdown **0 行**；日志 2 条警告 |
| 真实 PDF `/home/<user>/.hermes/reports/market_briefing.pdf` | markdown **0 行**；日志 **24 条**警告 |

日志原文：

```
WARNING mineru_vl_utils.vlm_client.http_client: The response was truncated due to length limit.
WARNING mineru_vl_utils.mineru_client.parse_layout_output: Layout output does not match expected format: !!!!!!!!!!!!!!!!...
```

直接探测服务器（绕过 MinerU 客户端）：

| 推理路径 | prompt | 输出 |
|---|---|---|
| vLLM 0.27.1 HTTP | `\nLayout Detection:` | 4096 tokens **全是 `!`**，finish_reason=length |
| vLLM（**已挂** MinerULogitsProcessor） | `\nLayout Detection:` | 仍然**全是 `!`**（1200 tokens） |
| transformers 5.15.1 直推 | `\nLayout Detection:` / `\nText Recognition:` | **多语言乱码**（UART Dtype boothfarm มาlonlon…） |
| 官方 README 的 `MinerUClient(backend="transformers")` | `two_step_extract` | **返回空列表 `[]`** |

→ 挂 logits processor **不能**救回来；官方 transformers 后端同样失败。**问题不在 vLLM 或 MinerU 集成，而在模型与当前 transformers/vLLM 版本的组合。**

### 3.4 根因判定

- MinerU 官方依赖闸门 `transformers<5.0.0` + `vllm<0.22.0` 与本机（**5.15.1** / **0.27.1**）冲突，这不是"建议版本"而是**功能正确性依赖**（实测验证）。
- 该 checkpoint 的 Qwen2-VL 处理在当前 transformers 5.x 下产出退化结果；视觉塔权重本身完整（391 张量），排版图像 token 数正常（prompt 853 tokens 含约 840 图像 token），**排除"权重缺失/图片没送进去"**。

---

## 4. 若要强行跑通的选项（按代价排序）

| 选项 | 做法 | 代价/风险 |
|---|---|---|
| **A. 独立 venv + 本地 transformers 后端** | 新建 venv：`pip install "mineru[core,vlm]"`（会装 transformers 4.57 + torch），用 `-b vlm-engine` 本地推理 | 多一套 ~4–6 GB 环境；与 `venv-vllm` 隔离不冲突；**未验证**（transformers 4.x 是否在本机 aarch64/CUDA13 上可用需实测） |
| **B. 独立 venv + 旧版 vLLM** | `pip install "mineru[core,vllm]"`（vllm 0.21.x） | vLLM 0.21 在 **aarch64/GB10 上有无可用轮子未知**；且与现网 vLLM 栈分裂，维护成本高 |
| **C. 官方 Docker / 云 API** | 用 MinerU 官方镜像或 MinerU 云服务 | 官方镜像需确认是否有 **aarch64** 变体（大概率 x86_64）；走云则数据出机 |
| **D. 放弃**（当前建议） | 继续用已跑通的 **GLM-OCR** 或 **PaddleOCR-VL-1.6** | 零额外成本 |

---

## 5. 与本机其他 OCR/文档解析方案的横向对比（全部真机实测）

| 方案 | 本机状态 | 关键数据 |
|---|---|---|
| **GLM-OCR** | ✅ **已跑通** | vLLM AR，英/中/表格全对，**0.87–1.78 s/图**，占用约 9 GiB；版面走 torch 可 GPU |
| **PaddleOCR-VL-1.6** | ✅ 可行（未端到端实跑） | 两阶段，版面锁 PaddlePaddle→CPU；0.9B |
| **HunyuanOCR-1.5** | ✅ 可行（AR 模式） | DFlash/CUDA13 已验证；**MTP 实测不可用** |
| **MinerU2.5-Pro** | ❌ **当前栈不可行** | 客户端/加载 OK，推理输出退化；根因 transformers<5 + vllm<0.22 |

> 需要 MinerU 那套**版面重排 pipeline**（而非单纯 OCR）时，本机当前的替代是 GLM-OCR + `glmocr[selfhosted]` 的 PP-DocLayout-V3 版面链路（已验证，见 [glm-ocr-dgx-spark-deployment.md](glm-ocr-dgx-spark-deployment.md) §10.8）。

---

## 6. 复现材料（留在本机）

- 客户端环境：`/home/<user>/projects/venv-mineru`（511 MB，轻量，可删）
- 模型缓存：`/home/<user>/projects/mineru-model-cache`（2.2 GB）
- 测试产物：`/home/<user>/projects/mineru-test/`（`out-lp` / `out-real` / `run-*.log`）
- 服务端日志：`/home/<user>/projects/mineru-test-vllm.log`
- 关键日志行：`The response was truncated due to length limit` + `Layout output does not match expected format: !!!...`
