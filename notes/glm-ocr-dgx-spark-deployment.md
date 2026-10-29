# GLM-OCR × DGX Spark（aarch64 GB10）部署方案

> 主机：`spark-XXXX` · NVIDIA DGX Spark（GB10 / Blackwell，sm_121，128 GB 统一内存，aarch64，Ubuntu 24.04，内核 6.17-nvidia，20 核 = 10×Cortex-X925 + 10×A725）
> 日期：2026-09-15 · 主题：**GLM-OCR（zai-org）在本机的部署方案**——两条路径、官方 SDK pipeline、算力端/客户端拓扑、服务化封装
> 姊妹文档：[paddleocr-vl-1.6-dgx-spark-feasibility.md](paddleocr-vl-1.6-dgx-spark-feasibility.md)（PaddleOCR-VL-1.6 可行性 + HunyuanOCR 对比；本文承接其结论：**GLM-OCR 为本机首选 OCR 方案**）
> **客户端接入**：[glm-ocr-client-integration.md](glm-ocr-client-integration.md)（**别的机器怎么调用本服务**：HTTP 契约 / 官方 SDK 的 MaaS 模式 / 上传网关；含"PDF 走客户端会静默返回空结果"的实测根因与对策）
> 本机前置：`venv-vllm`（vLLM 0.27.1 / torch 2.13.0，aarch64，原生注册 `glm_ocr`）；主服务 `qwen38-27b-sglang`（:8888）；`switchyard`（:4000）；**Ollama 已卸载**（见 notes/ollama-uninstall-worklog.md）

---

## 0. 省流版结论

| 维度 | 判定 |
|---|---|
| **是否推荐本机部署** | ✅ **推荐，且为本机 OCR 首选**。0.9B 模型（权重 2.65 GB bf16），128 GB 统一内存下与 Qwen3.8-27B 主服务共存毫无压力。 |
| **许可** | ✅ 最干净：权重 **MIT**（HF `license:mit`）、官方 SDK `glmocr` **Apache-2.0**（pyproject）。无地域限制，优于腾讯混元社区协议。 |
| **本机 vLLM 原生** | ✅ vLLM 0.27.1 已注册 `glm_ocr`（registry L418）与 `glm_ocr_mtp`（L655），**无需 `--trust-remote-code`**，直接 `vllm serve zai-org/GLM-OCR`。 |
| **关键优势（本机特有）** | ★ 版面分析 **PP-DocLayout-V3 走官方 SDK 的 torch/transformers 路径**（`glmocr[selfhosted]`/`[layout]` 依赖 torch+transformers，**不含 paddlepaddle**），且 `GlmOcr(layout_device="cuda")` 可显式指定 GPU → **本机 aarch64 也能全链路 GPU**。对比：PaddleOCR-VL-1.6 的版面锁死在 PaddlePaddle（本机无 aarch64 GPU 构建，只能 CPU）。 |
| **内存（实测）** | ✅ 主服务 1M 上下文 + `mem-fraction-static=0.85` 下**稳态 available 25 GiB**；GLM-OCR 增量约 **5–7 GB** → 余量约 3–4×，**可部署**（详见 §8）。 |
| **两条路径** | A) **vLLM(+MTP) + 官方 SDK**（生产/批量，★ 推荐）；B) **Ollama 一键**（试跑；本机需先重装 Ollama，且只跑 VLM 无版面）。 |
| **实测结论（2026-09-15 真机）** | ✅ **已跑通**：vLLM AR 起服成功，英文/中文/表格 OCR 全部准确、无幻觉，单图 **0.87–1.78 s**；实测占用约 **9 GiB**，停服后 available 回到 25 GiB。详见 §10。 |
| **唯一未通过项** | ❌ **MTP 投机不可用**（checkpoint 与 vLLM 的 MTP 权重命名不匹配，见 §10.4）→ 本机用**纯 AR**。 |
| **客户端接入（实测）** | ✅ 服务端 `python -m glmocr.server`（默认 `0.0.0.0:5002`）+ 客户端 `POST /glmocr/parse`；图片传 `data:image/*;base64` 即可（单图 **2.6–4.9 s**）。⚠️ **PDF 与 http(s) URL 会被静默拒绝**（HTTP 200 + 空结果）→ 用**上传网关**或客户端逐页渲染。详见 [glm-ocr-client-integration.md](glm-ocr-client-integration.md) |
| **服务化现状（收工状态）** | ⚠️ 已在 `venv-vllm` 补装 `glmocr[server]`（Flask 3.1.3 等 4 个包，**未动 torch/vllm**）；新增一键启停 `glm-ocr-test/{start,stop}-service.sh` 与上传网关 `glm-ocr-upload-gateway.py`（:5010）。**服务默认全停**（端口干净、available 26.3 GB），用则起、跑完即停 → 详见 **§12** |
| **版面 pipeline** | ✅ **已端到端跑通**（`glmocr[selfhosted]`）：PP-DocLayoutV3 经 torch/transformers 运行、**不依赖 PaddlePaddle**，CPU/CUDA 均可，检出区域+顺序+markdown 正确，PDF 多页合并正常。建议本机用 `--layout-device cpu`（GPU 仅快 0.4 s/页却多占 3.6 GB）。见 §10.8。 |

---

## 1. 模型事实（已逐项核对）

| 项 | 值 | 来源 |
|---|---|---|
| 架构 | `GlmOcrForConditionalGeneration` / `model_type: glm_ocr` | HF `config.json` |
| 组成 | CogViT 视觉编码器 + 轻量跨模态 connector（token 降采样）+ **GLM-0.5B** 语言解码器；约 **0.9B** 参数 | 官方 README |
| 文本配置 | hidden 1536 / 16 层 / 16 attn heads / 8 KV heads / head_dim 128 / silu / `max_position_embeddings` 131072 | HF `config.json` |
| MTP | `num_nextn_predict_layers: 1`（单层 MTP） | HF `config.json` |
| 权重体积 | `model.safetensors` **2650.6 MB**（bf16） | HF tree API |
| 许可 | 权重 **MIT**；SDK `glmocr` **Apache-2.0** | HF tag / pyproject |
| 精度 | **OmniDocBench V1.5 = 94.62，排名第 1**；公式/表格/信息抽取均 SOTA | 官方 README |
| 速度（官方自测） | 1.86 pages/s（PDF）、0.67 images/s（图像） | 官方 README |
| 流水线 | 两阶段：**PP-DocLayout-V3 版面分析** + 并发区域识别 | 官方 README |
| 部署支持 | **vLLM / SGLang / Ollama / transformers** | 官方 README |
| 生态 | HF `zai-org/GLM-OCR`（1.89M 下载 / 2038 likes）；ModelScope `ZhipuAI/GLM-OCR`；支持 LLaMA-Factory 微调 | HF API |

---

## 2. 本机适配性

### 2.1 本机 vLLM 0.27.1 原生注册（决定性事实）

本机 `venv-vllm` 的 model registry 已原生包含 GLM-OCR：

| registry key | 类 | 行 | 说明 |
|---|---|---|---|
| `glm_ocr` | `GlmOcrForConditionalGeneration` | L418 | 主模型（含视觉塔） |
| `glm_ocr_mtp` | `GlmOcrMTP` | L655 | MTP 草稿模型，**投机解码可用** |

证据：`vllm/model_executor/models/registry.py` L418 / L655；模型实现文件 `glm_ocr.py`、`glm_ocr_mtp.py` 均在 `models/` 目录；`vllm/config/speculative.py` L43 的 `MTPModelTypes` 含 `"glm_ocr_mtp"`，L53 支持 `"mtp"` 作为 speculative method。

### 2.2 版面阶段走 torch，不依赖 PaddlePaddle（本机关键）

官方 SDK `glmocr`（PyPI 0.1.5）的依赖声明：

- 基础依赖：pillow / numpy / requests / pydantic / PyYAML / portalocker / python-dotenv / tqdm / **pypdfium2**（**无 torch**，够用于 MaaS 云模式）
- `[layout]` / `[selfhosted]` extra：opencv-python-headless、**torch≥2.10**、torchvision≥0.25、**transformers≥5.3**、sentencepiece、accelerate（+ pypdfium2）
- `[server]` extra：flask（内置服务）

**即：版面检测（PP-DocLayout-V3）经 torch/transformers 加载，依赖树中不含 paddlepaddle。** 且 API 显式提供 `layout_device: "cpu" | "cuda" | "cuda:N"` 与 `cuda_visible_devices` 参数 → **本机可把版面放到 GPU**。

> 对比 PaddleOCR-VL-1.6：版面 PP-DocLayoutV2 必须走 PaddlePaddle，而本机 aarch64 无 PaddlePaddle GPU 构建（PyPI `paddlepaddle-gpu` 仅 x86_64 轮子），版面只能 CPU。**这是 GLM-OCR 更适配本机的根本原因。**

### 2.3 与其他 OCR 方案的本机适配对比

| 模型 | 参数 | 许可 | 本机 vLLM 原生 | 端到端 | 速度*（H20） | 精度* | 本机备注 |
|---|---|---|---|---|---|---|---|
| **GLM-OCR**（zai-org） | 0.9B | **MIT** / Apache-2.0 | ✅ `glm_ocr`+`glm_ocr_mtp` | ✅ | 1.65 s/img | **OmniDocBench V1.5 #1 (94.62)** | ★ 本机首选：最轻+许可最干净+版面可 GPU |
| chandra-ocr-2（Datalab） | ~1.5–2B | OpenRAIL | ✅ `qwen3_5` | ✅ 自含版面 | — | 85.8% olmOCR、90+ 语言 | `pip install chandra-ocr` + `chandra_vllm` 一条命令起 vLLM，部署最省心 |
| HunyuanOCR-1.5（腾讯） | ~1.5B | 腾讯社区协议（地域排 EU/UK/韩） | ✅ `hunyuan_vision` | ✅ | **1.41 s/img**（+DFlash 2.14×） | 端到端强 | DFlash 与本机主服务同栈、官方 CUDA13 已验证；许可地域受限 |
| Qwen3-VL / Qwen2.5-VL | 2–72B | Apache-2.0 | ✅ `qwen3_vl`/`qwen2_5_vl` | ✅ 通用 VLM | 视尺寸 | 好（非专用 SOTA） | infra 风险最低（本机主服务同系），但更重 |
| PaddleOCR-VL-1.6（百度） | 0.9B+版面 | Apache-2.0 | ✅ `paddleocr_vl` | ❌ 两阶段 | 1.74 s/img | OmniDocBench V1.6 SOTA 96.3%、109 语言 | 精度/多语言/印章古籍最强；版面锁 PaddlePaddle CPU，链路最长 |
| Unlimited-OCR（百度 MoE） | MoE | MIT | ✅ `unlimited_ocr` | ✅ | 3.66 s/img | 多语言强 | custom_code；偏慢 |
| DeepSeek-OCR-2 | ~3B | Apache-2.0 | ✅ `deepseek_ocr2` | ✅ | 5.46 s/img | 强 | 偏大偏慢 |
| dots.ocr | ~1B | 视仓 | ✅ `dots_ocr` | ✅ | 7.15 s/img | 结构化好 | 最慢 |

\* 速度为 HunyuanOCR `docs/benchmark.md`（H20 80GB，concurrency=1，max_tokens=8000）横向值，**本机 GB10 数值需实测校准**。

---

## 3. 方案 A：vLLM(+MTP) + 官方 SDK（生产/批量，★ 推荐）

### 3.1 启动推理引擎（vLLM，开 MTP 投机）

```bash
vllm serve zai-org/GLM-OCR \
  --port 8080 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --served-model-name glm-ocr \
  --gpu-memory-utilization 0.2
```

SGLang 备选：

```bash
python3 -m sglang.launch_server --model-path zai-org/GLM-OCR --port 8080
```

> ⚠️ **MTP 参数更正（重要）**：本机 GLM-OCR `config.json` 显示 `num_nextn_predict_layers: 1`（仅 1 层 MTP），而常见示例写 `num_speculative_tokens: 3`。vLLM 的 MTP 路径通常按 MTP 层数取投机 token 数（DeepSeek 系惯例为 1），**3 是否被 0.27.1 接受需手测**；报错则改为 `num_speculative_tokens: 1`，或先不带 `--speculative-config` 跑纯 AR 基线再逐步加。
> 本机已确认 **vLLM 支持该能力**：`glm_ocr_mtp` 在 `MTPModelTypes` 白名单内、`"mtp"` 是合法 method。
> ❌ **2026-09-15 实测更新：本机 MTP 路径直接启动失败** —— `ValueError: Following weights were not initialized from checkpoint: model.layers.16.mtp_block.*`（checkpoint 里 MTP 层实际名为 `model.language_model.layers.16.*`，命名不匹配，详见 §10.4）。**结论：本机请用纯 AR，不要加 `--speculative-config`。**

### 3.2 安装官方 SDK

```bash
# 仅用智谱云 MaaS（最轻，无 torch，无需 GPU）
pip install glmocr

# 自托管完整 pipeline（版面检测；拉 torch/transformers，非 PaddlePaddle）★ 本机用这个
pip install "glmocr[selfhosted]"

# 内置服务能力
pip install "glmocr[server]"
```

### 3.3 指向本机 vLLM（配置）

SDK 配置文件 `glmocr/config.yaml` 的自托管关键项：

```yaml
pipeline:
  maas:
    enabled: false              # 关闭云模式 → 走本地 vLLM
  ocr_api:
    api_host: 127.0.0.1
    api_port: 8080              # 与 vllm serve --port 一致
    model: glm-ocr              # 对应 vllm --served-model-name
    api_path: /v1/chat/completions
    api_mode: openai            # Ollama 遇 502 可改 ollama_generate + /api/generate
    verify_ssl: false
    connection_pool_size: 128   # 需 ≥ max_workers
  max_workers: 32               # 区域并发识别；服务端 503 多则调低
  page_loader:
    max_tokens: 8192
    temperature: 0.0
    top_k: 1
    pdf_dpi: 200
  result_formatter:
    output_format: both         # markdown + json
```

### 3.4 客户端调用（正确 API）

```python
from glmocr.api import GlmOcr        # 例程使用 glmocr.api；亦可 from glmocr import GlmOcr

# selfhosted 模式：指向本机 vLLM；版面显式放 GPU（本机 aarch64 亦可）
with GlmOcr(mode="selfhosted",
            ocr_api_host="127.0.0.1",
            ocr_api_port=8080,
            model="glm-ocr",
            layout_device="cuda") as parser:
    result = parser.parse("document.pdf")     # 支持 路径 / bytes / 列表
    print(result.markdown_result)
    print(result.to_dict())                   # 结构化、可 JSON 序列化
    result.save(output_dir="./output")        # 落 markdown + json
# parse() 亦可接受列表：results = parser.parse(["a.png", pdf_bytes])
```

**`GlmOcr` 真实签名**（v0.1.5）：`config_path`、`api_key`、`api_url`、`model`、`mode`（`"maas"`/`"selfhosted"`）、`timeout`、`log_level`、`env_file`、`ocr_api_host`、`ocr_api_port`、`cuda_visible_devices`、`layout_device`。

> ⚠️ **API 更正**：流传写法 `GlmOcr(api_base=..., model_name=...)` **不是真实参数名**。正解是上面的 `mode="selfhosted"` + `ocr_api_host`/`ocr_api_port` + `model`（或走 config.yaml 的 `pipeline.ocr_api.*`）。另外 `GlmOcr()` 若检测到 `ZHIPU_API_KEY` 环境变量会**自动切到 MaaS 云模式**——本机自托管时必须显式 `mode="selfhosted"` 或设 `maas.enabled: false`。

---

## 4. 方案 B：Ollama 一键（轻量、最快试跑）

```bash
ollama run glm-ocr
ollama run glm-ocr "Text Recognition: ./invoice.png"
```

- Ollama 官方库**确有 `glm-ocr`**（ollama.com/library/glm-ocr，说明文案与 GLM-OCR 一致）。
- ⚠️ **本机 Ollama 已卸载**（见 `notes/ollama-uninstall-worklog.md`，46 GB 模型清理），走此路需先重装；本机主服务与判官方案曾因内存压力卸载它，**若无必要不建议为 OCR 重新引入 Ollama**。
- ⚠️ Ollama 只托管 **VLM**，**不含 PP-DocLayout-V3 版面 pipeline** → 仅适合单图/单页快速试跑；批量与高精度用方案 A。
- 若用 Ollama 接 SDK：`api_mode: ollama_generate` + `api_path: /api/generate`（官方提示：Ollama 视觉请求遇 502 时切换）。

---

## 5. SDK pipeline 本质：算力端 vs 客户端

`glmocr` SDK 本质是**端到端文档处理流水线调度器**：

```
PDF/长图 → [1 渲染切页 CPU/内存密集] → [2 PP-DocLayout-V3 版面分析]
        → [3 坐标裁剪 & 阅读顺序 CPU 几何] → [4 并发调用 VLM（localhost HTTP）]
        → [5 Markdown/JSON 结构化拼接]
```

| 考量维度 | 方案 A：SDK 放算力端（推荐 ★） | 方案 B：SDK 放客户端 |
|---|---|---|
| 拓扑 | 客户端上传原始文件 → **DGX Spark（SDK + vLLM）** | **客户端（SDK）** → 大量切片图 HTTP → DGX Spark（vLLM） |
| 网络开销 | **极小**：只上传 1 个原始 PDF/图片 | **极大**：一页切十几块，10 页上百张切片频繁往返 |
| 吞吐效率 | **极高**：切片后走 localhost 打满 GPU batch | 受限：客户端上行带宽 + HTTP RTT |
| 客户端负担 | **零计算**：只上传与展示 | 较重：需装 torch/图像依赖跑版面 |
| 部署维护 | 一次性封装进 Spark 容器，统一对外 API | 每个使用端都要配复杂 Python 依赖 |

**结论：生产/高性能场景一律把 SDK 与推理引擎都放算力端（DGX Spark）。** 客户端 SDK 仅适合本地单机调试，或客户端本身是与 Spark 高带宽低延迟直连的专用工作站。

---

## 6. 服务化落地

### 6.1 官方内置服务（最省事）

SDK 自带服务，默认 `:5002`（`glmocr/server` + `glmocr[server]`）：

```yaml
server:
  host: "0.0.0.0"
  port: 5002
```

本机端口规划：vLLM `:8080` + glmocr server `:5002`，避开主服务 `4000/8888`。

### 6.2 自建 FastAPI 包装（需要自定义契约时）

```python
# DGX Spark 上封装为统一解析服务：客户端只 POST 一个 PDF，内部完成切片+并发推理+拼接
from fastapi import FastAPI, UploadFile
from glmocr.api import GlmOcr
import shutil

app = FastAPI()
parser = GlmOcr(mode="selfhosted", ocr_api_host="127.0.0.1",
                ocr_api_port=8080, model="glm-ocr", layout_device="cuda")

@app.post("/parse_doc")
async def parse_doc(file: UploadFile):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = parser.parse(tmp)
    return {"content": result.markdown_result, "json": result.to_dict()}
```

### 6.3 客户端接入（2026-09-15 补充实测）

> ⚠️ 上面 §6.2 的包装思路**方向对但细节要改**：① 全局单例 `parser` 被多请求共用（SDK 的 `Pipeline._current_state` 是单实例），并发下不保险；② 路由直接把上传文件交给 SDK 也仍然绕不开"PDF 不被支持"的问题。
> **已实测可用**的实现见 [glm-ocr-client-integration.md](glm-ocr-client-integration.md)：服务端 `0.0.0.0:5002` 内置服务 + 一个 ~70 行的**上传网关**（multipart → 服务端 `file://` → 内置服务），PDF/图片一个请求搞定；脚本在 `glm-ocr-test/{start,stop}-service.sh`。

**核心结论（先看这条）**：`POST /glmocr/parse` 的 `images` 字段**只认 `data:image:*`、`file://` 和服务端本地路径**；传 `data:application/pdf` 或 `http(s)://` 一律 **200 + 空结果**（异常在 `page_loader.py:160-165` 被吞）。而官方 SDK 的 MaaS 模式恰好把 PDF 包成 `data:application/pdf` —— 所以**"客户端只装 glmocr、把 api_url 指过来"这条路对图片有效、对 PDF 静默失败**。

---

## 7. 本机注意项 + 手测 checklist

> ✅ **本 checklist 已于 2026-09-15 真机逐项执行完毕**，结果与实测数据见 §10（含 MTP 失败结论与版面 pipeline 验证）。以下保留为操作清单。

- [ ] **起 vLLM**：`vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --port 8080 --gpu-memory-utilization 0.2`（先不带 MTP 跑通基线）。
- [ ] **MTP**：加 `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` 试；确认 0.27.1 是否接受 3（见 §3.1 更正）。
- [ ] **SDK**：`pip install "glmocr[selfhosted]"`，确认版面阶段在 aarch64 上跑通（先 `layout_device="cpu"` 保底，再试 `"cuda"`）。
- [ ] **端到端**：官方示例图/一张真实 PDF → markdown 无幻觉、表格/公式/印章正常。
- [ ] **端口/资源**：vLLM :8080、glmocr server :5002，避开 4000/8888；显存预算 ≤0.2，观察 available 内存保持 >10%。
- [ ] **勿空转常驻**：部署后须有真实调用方，否则参考 judge :8889 教训（16 天无调用即关停）——跑完即停。
- [ ] **NGC 容器**（可选）：如需隔离，优先用 NVIDIA NGC 的 Spark/Blackwell aarch64 适配镜像跑 vLLM，避免自编译轮子；本机 `venv-vllm` 亦可直接复用。

---

## 8. 内存预算实测（能否部署，2026-09-15 核查）

| 项 | 实测 | 说明 |
|---|---|---|
| 主服务 | SGLang `qwen38-27b-sglang`，`max_model_len=1010000`（**1M 上下文**），`mem-fraction-static=0.85` → 约 **103 GiB 预留** | `:8888` 已就绪 |
| **稳态 available** | **25 GiB**（used 96 Gi / free 2.8 Gi / buff-cache 24 Gi；swap 仅用 334 Mi） | ← 这才是做预算该看的数 |
| 加载暂态（陷阱） | 容器刚重启、权重加载中时曾见 **available 92 Gi** | KV cache 未分配，**不可作预算依据** |
| GLM-OCR 增量 | 权重 2.65 GB（bf16）→ vLLM 含 KV/CUDA graph **约 4–6 GB**；版面 `layout_device="cpu"` 另约 0.5–1 GB（走 `"cuda"` 约 1–2 GB） | 合计 **约 5–7 GB** |
| 磁盘 | 权重 + 依赖约 6–8 GB（可用 3.3 T） | 无压力 |

**结论：可以部署。** 25 GiB 稳态余量 vs 约 5–7 GB 需求，余量约 3–4 倍（**2026-09-15 实测占用约 9 GiB，仍余 16 GiB，见 §10.3**）。横向对照 [LOCAL-JUDGE-LESSONS.md](LOCAL-JUDGE-LESSONS.md)：同样余量下 **8B 级**（5.7 GB 权重 + KV）曾被判定"放不下"，GLM-OCR 0.9B 约为其 1/6，余量充足。

> ⚠️ GB10 是**统一内存**，CPU 与 GPU 共用同一 128 GB 池：把版面放 CPU 省的是 GPU 侧分配，**不省池子总量**，主服务的 `0.85` 预留照样占着。真正的杠杆是 `MEM_FRACTION` 与给 OCR 侧设上限。

**若要更省（三档，按需选）**：
1. **bf16 + vLLM（推荐）**：约 4–6 GB，原生支持、可开 MTP。
2. **GGUF Q8_0（1.6 GB）/ Q4_K_M（约 1 GB）**：可再省 1–3 GB，但 **vLLM 0.27.1 不能加载 GGUF**（2026 版已移除该 load-format，见 [LOCAL-JUDGE-LESSONS.md](LOCAL-JUDGE-LESSONS.md)），必须走 llama.cpp 或重装 Ollama → 丢掉 vLLM/MTP、多一个运行时。
3. **版面 `layout_device="cpu"`**：把版面从 GPU 侧挪开（池子总量不变）。

**三个收紧杠杆**：① 给 GLM-OCR 侧 vLLM 设 `--gpu-memory-utilization 0.04~0.06`（按 121 GiB 约合 5–7 GB 上限）；② 主服务 `MEM_FRACTION` 0.85→0.80（可多出约 5.8 GiB）；③ 版面 CPU 绑空闲核 **0-4,10-14**（主服务占 `CPUSET=5-9,15-19`）。
**时机**：等主服务加载完成、`:8888` 正常响应后再起 OCR 服务；两者不要同时冷启动，否则争统一内存易触发 swap。

---

## 9. 版本与量化变体现状（2026-09-15 核查）

**模型**

| 项 | 值 |
|---|---|
| HF 主仓 | `zai-org/GLM-OCR`（**无版本号后缀**，滚动更新） |
| 最近更新 | **2026-09-11**（commit `2e85a628`），下载 1.89M / likes 2039 |
| 权重 | 单文件 `model.safetensors` **2650.6 MB**（bf16） |
| 语言 | zh / en / fr / es / ru / de / ja / ko（HF tag） |
| 仓内评测 | `.eval_results/olmocrbench.yaml`、`.eval_results/mdpbench.yaml` |
| 官方版本化模型 | **无**（不像 PaddleOCR-VL 有 1.5 / 1.6 多版本；GLM-OCR 只有一个滚动仓） |

**SDK \(glmocr\)**（PyPI 版本与 GitHub tag 一致）

| 版本 | 发布 |
|---|---|
| **0.1.5** | **2026-04-08（最新）** |
| 0.1.4 | 2026-03-19 |
| 0.1.3 | 2026-03-13 |
| 0.1.2 / 0.1.1 | 2026-03-12 |

**社区量化 / 移植变体（按下载量）**

- `ggml-org/GLM-OCR-GGUF`（29.5k）—— ggml 官方的 GGUF 转换，llama.cpp 路线首选
- `mlx-community/GLM-OCR-4bit`（31.3k）及 5/6/8bit / `bf16` —— Apple MLX，**本机（aarch64+CUDA）不适用**
- `mradermacher/GLM-OCR-GGUF`（3.5k）、`GLM-OCR-i1-GGUF`（1.2k）
- `onnx-community/GLM-OCR-ONNX`（307）—— ONNX / CPU 路线
- Ollama 库：`glm-ocr:bf16`（约 2.2 GB）、`glm-ocr:q8_0`（约 1.6 GB）、`latest`
- 另有若干个人 Q4_K_M / Q8_0 GGUF 仓

> **版本一致性提示**：模型仓 **2026-09-11 有更新**，而本机 vLLM 0.27.1 内的 `glm_ocr` 实现早于该更新。若加载报错（config 字段不匹配等），可先用 `--revision` 固定到旧 commit 验证，再决定是否升级 vLLM。

---

## 10. 实测验证记录（2026-09-15 真机跑通）

**一句话：GLM-OCR 在本机（aarch64 GB10）部署成功、OCR 质量优秀；唯一失败项是 MTP 投机。**

### 10.1 启动命令（AR 模式，实测可用）

```bash
venv-vllm/bin/pip install ninja          # 坑 2，见 10.5
export PATH=/home/<user>/projects/venv-vllm/bin:$PATH
export HF_HOME=/home/<user>/projects/glm-ocr-model-cache
export VLLM_CACHE_ROOT=/home/<user>/projects/glm-ocr-test/.vllm-cache   # 坑 1，见 10.5
vllm serve zai-org/GLM-OCR --served-model-name glm-ocr --host 127.0.0.1 --port 8080 --gpu-memory-utilization 0.05 --max-model-len 16384 --enforce-eager --limit-mm-per-prompt '{"image": 1}'
```

- 权重下载 **2.5 GB**（`hf download zai-org/GLM-OCR`，约 1 分钟）
- 权重加载 **2.21 GiB / 12.6 s**；引擎初始化（profile + KV cache + warmup）**19.0 s**
- **全程无需 `--trust-remote-code`**（vLLM 原生 `glm_ocr`，与本机 registry 吻合）
- FlashAttention 2 后端自动选中（本机 aarch64 可用）；resolved architecture = `GlmOcrForConditionalGeneration`

### 10.2 OCR 质量实测（自造测试图：正文 + 表格 + 公式 + 编号）

| 测试 | 延迟 | 输出 token | 结果 |
|---|---|---|---|
| 英文文档 `Text Recognition:` | **1.78 s** | 201 | ✅ 全文准确，数字/百分比/编号全对，无幻觉 |
| 英文表格（转 markdown/HTML） | **1.48 s** | 192 | ✅ 输出完整 HTML 表格，4 行 3 列数值全对 |
| 中文文档 `Text Recognition:` | **0.87 s** | 102 | ✅ 中文全文准确（1,204,500 / 22.1% / 38.7% / HT-2026-0915 / ¥12,750.00） |

对照官方 H20 基准 1.65 s/img —— **本机 GB10 单图延迟同量级（0.87–1.78 s）**，中文场景更快。

### 10.3 内存实测

| 指标 | 实测值 |
|---|---|
| 权重加载 | **2.21 GiB** |
| KV cache | **1.56 GiB**（25,616 tokens） |
| GPU 侧实际占用 | **3.57 GiB**（预算 0.05 = 6.08 GiB，未用满） |
| 系统 available | **25 GiB → 16 GiB**（净占用约 **9 GiB**，含两个 Python 进程 RSS 约 4.8 GB） |
| 停服后 | available 回到 **25 GiB**，主服务 `:8888` 仍健康 |

**结论：实测占用（约 9 GiB）比 §8 预估（5–7 GB）略高，但 16 GiB 余量下依然安全。**

### 10.4 MTP 投机：实测**不可用**（重要更正）

带 `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` 启动 → 引擎初始化失败：

```
ValueError: Following weights were not initialized from checkpoint:
  model.layers.16.mtp_block.post_attention_layernorm.weight ...
```

**根因（已定位到张量名）**：checkpoint 里 MTP 层**确实存在**，但命名为
`model.language_model.layers.16.{eh_proj,enorm,hnorm,input_layernorm,...}`（DeepSeek 风格），
而 vLLM 0.27.1 的 `glm_ocr_mtp`（复用 GLM-4-MoE-Lite 的 MTP plumbing）期望 `model.layers.16.mtp_block.*`
→ **命名不匹配，草稿模型加载失败**。

- 与 `num_speculative_tokens` 取值无关（日志仅警告 >1 会重复前向），1 或 3 都会同样失败。
- 模型仓 2026-09-11 的两次提交是 `Tag as image-to-text` / `update configs and tokenizer`，**未改权重**；
  故这不是"新版本回归"，而是该 checkpoint 与 vLLM 实现的长期命名差异（即 §9 提示的版本偏斜，方向反了：不是权重变了，是 vLLM 侧映射对不上）。
- **对策：本机就用纯 AR**（实测 0.87–1.78 s/img 已够快）；若必须投机，需等 vLLM 修 `glm_ocr_mtp` 的权重名映射，或自行打 patch。

### 10.5 部署踩坑（均已解决）

1. **`~/.cache/vllm/torch_compile_cache` 属 root**（8/23 某个 root 容器遗留，`dyl` 不可写）→ 主机进程报 `PermissionError: [Errno 13]`。**对策**：`export VLLM_CACHE_ROOT=<工作区内可写目录>`。
2. **缺 `ninja`** → vLLM JIT 内核编译 `FileNotFoundError: 'ninja'`。**对策**：`pip install ninja`，并确保 venv 的 `bin` 在 `PATH` 上（按绝对路径调用 `vllm` 时不会自动带上）。
3. **沙箱拦主机 GPU ioctl**（CUDA Error 304）→ 主机侧跑 vLLM 需提权；docker 路径不受影响（同 [LOCAL-JUDGE-LESSONS.md](LOCAL-JUDGE-LESSONS.md)）。
4. （小坑）`pkill -f "vllm serve"` 会**误杀自己所在的 shell**（命令行里含同样字符串）→ 用 `pkill -f "vllm [s]erve"` 或记 PID。

### 10.6 与原结论的差异

| 项 | 文档原判 | 实测 |
|---|---|---|
| 能否部署 | 可以 | ✅ **确认可以**，且 OCR 质量优秀 |
| 内存增量 | 约 5–7 GB | 实测约 **9 GiB**（GPU 侧 3.57 GiB + 进程开销） |
| MTP 投机 | "registry 支持，需手测" | ❌ **实测不可用**（权重命名不匹配），改纯 AR |
| 版面 / 端到端 SDK pipeline | 需手测 | ⏳ **本次未测**（只验证了 VLM 本体；`glmocr[selfhosted]` 版面链路未跑） |

### 10.7 复现脚本（留在本机）

- 测试图：`/home/<user>/projects/glm-ocr-test/testdoc.png`（英文）、`testdoc_zh.png`（中文）
- 启动日志：`/home/<user>/projects/glm-ocr-test/vllm.log`（AR）、`vllm-mtp3.log`（MTP 失败现场）
- 权重缓存：`/home/<user>/projects/glm-ocr-model-cache`（2.5 GB）

### 10.8 版面 pipeline 实测（`glmocr[selfhosted]`，2026-09-15 补充）

**安装**：`pip install "glmocr[selfhosted]"` 装入 `venv-vllm`。装前 `--dry-run` 确认只新增 `accelerate/glmocr/portalocker/pymupdf/pypdfium2`，**未改动 torch/transformers** → 事后复核 vLLM 仍为 0.27.1 + torch 2.13.0+cu130，未被破坏。

**关键证据（证实 §2.2 的判断）**：`glmocr/layout/layout_detector.py` 的实现是
```python
import torch
from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor
model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors   # 百度产的权重，但经 transformers 加载
```
→ **版面模型用 torch/transformers 运行，不依赖 PaddlePaddle 运行时**。这正是 GLM-OCR 相对 PaddleOCR-VL-1.6（版面锁死 PaddlePaddle，本机只能 CPU）的核心差异，**已在真机证实**。

**实测结果**

| 输入 | 版面设备 | 单文件耗时 | 结果 |
|---|---|---|---|
| 英文文档图 | CPU | 2.79 s | ✅ 检出 **7 个区域**（5 text / 1 paragraph_title / 1 table），bbox+polygon 齐全，markdown 正确 |
| 英文文档图 | **cuda** | 2.39 s | ✅ 日志确认 `PP-DocLayoutV3 loaded on device: cuda` |
| 中文文档图 | CPU | 1.71 s | ✅ 内容正确（小瑕疵见下） |
| 3 页 PDF | CPU | 5.64 s（3 页） | ✅ 3 页按序合并为单一 markdown |

- 端到端 wall time：单图 **5.6–7.1 s**（含版面模型加载）；纯解析部分 1.7–2.8 s
- 版面区域可直接产出带 `native_label` / `bbox_2d` / `polygon` / `content` 的 JSON，并可选输出可视化图（`layout_vis/*.jpg`）

**内存对比（本机关键取舍）**

| 版面设备 | 运行期间最低 available | 相对基线增量 |
|---|---|---|
| CPU | 17,293 MB | **≈0.4 GB** |
| cuda | 14,151 MB | **≈3.6 GB** |

**结论与建议**：版面**确实能跑 GPU**（claim 成立），但本机端到端瓶颈是 VLM 的 HTTP 调用，版面换到 GPU 仅快约 **0.4 s/页**，却多占约 **3.6 GB** 统一内存。→ **在只剩约 25 GiB 余量的本机上，建议默认 `--layout-device cpu`**（省 3.6 GB，延迟几乎无差别）；只有批量高并发、版面成为瓶颈时才切 cuda。

**已知小瑕疵（不影响可用性）**：中文输出会把百分号转成 LaTeX（`22.1%` → `$22.1\%$`），逗号偶被全角化（`486,300` → `486，300`）—— 属模型输出风格，可后处理规整。

**产物**：`/home/<user>/projects/glm-ocr-test/{out-cpu,out-cuda,out-zh,out-pdf}/`（各含 `*.md` / `*.json` 区域标注 / `layout_vis/` 可视化）

---

## 11. 附录：资源与证据

**GLM-OCR**
- 权重：HF `zai-org/GLM-OCR`（MIT；bf16；`model.safetensors` 2650.6 MB）；ModelScope `ZhipuAI/GLM-OCR`
- 仓库 / SDK：`github.com/zai-org/GLM-OCR`；PyPI `glmocr` 0.1.5（Apache-2.0）
- 微调：LLaMA-Factory 教程（`examples/finetune/README.md`）；技术报告 arXiv 2603.10910
- 部署：vLLM recipes（`recipes.vllm.ai/zai-org/GLM-OCR`）、SGLang cookbook（`cookbook.sglang.io/.../GLM-OCR`）、Ollama（`ollama.com/library/glm-ocr`）
- **本机 registry 证据**：`venv-vllm/.../vllm/model_executor/models/registry.py` L418（`glm_ocr`）、L655（`glm_ocr_mtp`）；`vllm/config/speculative.py` L43（MTP 白名单）、L53（`"mtp"` method）
- **SDK 依赖证据**：`glmocr` pyproject —— `[layout]`/`[selfhosted]` = torch≥2.10 + torchvision + transformers≥5.3 + opencv + accelerate（**无 paddlepaddle**）
- **SDK API 证据**：`glmocr/api.py` `GlmOcr.__init__`（`mode`/`ocr_api_host`/`ocr_api_port`/`layout_device`）；`glmocr/config.yaml`（`pipeline.ocr_api.*`、`server.port: 5002`）

**相关本机文档**
- PaddleOCR-VL-1.6 可行性 + HunyuanOCR 对比：[paddleocr-vl-1.6-dgx-spark-feasibility.md](paddleocr-vl-1.6-dgx-spark-feasibility.md)
- 本机 OCR 候选调研结论（GLM-OCR 为首选）同见该文档
- Ollama 卸载背景：[ollama-uninstall-worklog.md](ollama-uninstall-worklog.md)
- 内存/常驻教训：[LOCAL-JUDGE-LESSONS.md](LOCAL-JUDGE-LESSONS.md)

---

## 12. 服务化部署现状 + 第二轮踩坑（2026-09-15：客户端接入）

> 本轮把方案从"**本机能跑**"推进到"**别的机器能调**"。客户端契约、三种接法、代码样例与逐项实测数据见 [glm-ocr-client-integration.md](glm-ocr-client-integration.md)；本节只记**部署侧现状与踩坑**。

### 12.1 现状快照（收工时实际状态）

| 项 | 值 |
|---|---|
| 已安装 | `venv-vllm` 新增 `glmocr[server]`：**Flask 3.1.3 / Werkzeug 3.1.8 / blinker 1.9.0 / itsdangerous 2.2.0**。装前 `--dry-run` 确认只新增这 4 个包 —— vLLM 0.27.1、torch 2.13.0+cu130 **未改动**（延续 §10.8 的纪律） |
| 服务状态 | ⚠️ **全部已停**：`:5002` / `:5010` / `:8080` 均无监听；`available` **26.3 GB**（基线 25 GB）；主服务 `:8888` 正常响应 |
| 端口规划 | vLLM `127.0.0.1:8080` · **SDK 服务 `0.0.0.0:5002`** · 上传网关 `0.0.0.0:5010`（避开主服务 4000/8888） |
| 启动 / 停止 | `bash glm-ocr-test/start-service.sh --with-gateway` / `bash glm-ocr-test/stop-service.sh` |
| 全栈开销 | 约 **10 GB** 统一内存（vLLM GPU 侧 3.57 GiB + SDK 服务进程 RSS 3.3 GB + 版面 CPU 约 0.4 GB） |

**本轮新增文件**

| 文件 | 作用 |
|---|---|
| `glm-ocr-upload-gateway.py` | **上传网关**（:5010）：`POST /parse` 吃 multipart、`POST /glmocr/parse` 吃官方 JSON 契约；`data:application/pdf` 落盘转 `file://` —— 补上内置服务不支持 PDF 的缺口 |
| `glm-ocr-test/glmocr-server.yaml` | SDK 服务端配置（由包内默认 `config.yaml` 复制后定点替换 3 处：maas 关、版面 cpu、max_workers 8） |
| `glm-ocr-test/start-service.sh` / `stop-service.sh` | 一键启停，逐级探活；停止脚本回读端口与内存 |
| `glm-ocr-test/{client-http-test,client-sdk-test,client-path-tests,client-sdk-via-gateway-test}.py` | 客户端侧逐项验证（HTTP 直连 / 官方 SDK / 三种 PDF 路径 / 网关闭环） |
| `glm-ocr-test/{page-loader-probe,patch-config}.py` | 根因单元探针 / 服务端配置生成 |
| `glm-ocr-test/_recording_stub.py`、`_stub_8080.py` | 测试桩：记录网关真实转发载荷 / 顶替 vLLM 验证启动脚本编排（无需 GPU） |

### 12.2 第二轮踩坑（均为本次实际踩到）

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| Q1 | 停止脚本"看起来成功"（打印 ✓ / 未运行），但 `:5010` 仍在监听 | `pkill -f` 的模式少写一个连字符（`glmocr-` vs `glm-ocr-`）；**pkill 不匹配只返回非 0，不会报错** | 停止脚本必须**回读端口**确认（`ss -ltn | grep`），不要只信 pkill 退出码 |
| Q2 | 手敲 `pkill -f "glm-ocr-upload-gateway.py"` 把**自己所在的 shell** 杀了 | `bash -c` 的命令行里含同一个匹配串（与 §10.5 坑 4 同款，本轮再踩一次） | 方括号规避自匹配：`pkill -f "glm-ocr-upload-[g]ateway.py"` |
| Q3 | `curl -d '{"file":"data:application/pdf;base64,<223KB>"}'` 报 **参数列表过长** | 命令行参数超 ARG_MAX | 大 payload 走 `--data-binary @file` 或 multipart（顺带省掉 base64 的 +33%） |
| Q4 | 自写 YAML 起服务报 `pipeline.layout.model_dir is required` | 配置是**分区浅合并**：只写 `layout: {device: cpu}` 会把整节默认值（model_dir / label_task_mapping / id2label …）一并替换掉 | 从包内 `glmocr/config.yaml` 复制后定点替换（`patch-config.py`） |
| Q5 | SDK 服务起来约 30 s 后自杀：`Failed to connect to API server … within 30 seconds` | 启动时会对 `ocr_api` 做一次探活（`ocr_client.py:175-235`），**vLLM 必须先在线** | 启动脚本按 vLLM → SDK 服务 → 网关顺序逐级探活 |
| Q6 | `Failed to infer device type` / `Can't initialize NVML`（复现 §10.5 坑 3） | 沙箱内无 GPU 直通权限 | 用正常 shell 运行；**提权后同一命令立即成功，GPU 直通是唯一变量** |
| Q7 | 读不到 `/etc/ufw/user.rules`（root-only），`sudo` 报 `no new privileges` | 当前环境禁用 sudo | ufw 放行只能由用户自己执行 `sudo ufw status` / `sudo ufw allow …`（命令见客户端文档 §1） |
| Q8 | 用扁平列表桩模拟上游时，客户端报 `'str' object has no attribute 'get'` | 上游 `json_result`/`layout_details` 是 **list[page] → list[region]** 嵌套形状 | 自建包装服务/造桩必须按此嵌套返回（见 `server.py` 的 `_build_response`） |

### 12.3 收尾纪律（本轮每次验证后都执行）

1. **跑完即停**：`bash glm-ocr-test/stop-service.sh`；停服后回读端口与内存（呼应 §7 与 judge `:8889` 的教训）。
2. **起服前先看 `free -g`**：主服务 1M 上下文下稳态 available ≈25 GB，全栈 OCR 要吃约 10 GB。
3. **装包先 `--dry-run`**：本轮 `pip install --dry-run "glmocr[server]"` 确认只新增 4 个包，未触碰 torch/vLLM。
4. **开机不自动拉起**：与主服务/代理一致，三个服务都需要手动起（脚本已做就绪探活，可直接重复执行）。
