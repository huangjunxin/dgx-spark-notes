# PaddleOCR-VL-1.6 × DGX Spark 可行性（附 HunyuanOCR-1.5 对比）

> 本文聚焦 **PaddleOCR-VL-1.6 在本机（aarch64 GB10）的可行性**（§0–§6），并附 **HunyuanOCR-1.5** 的对比与部署（§7–§8）；§9 为其他 OCR 候选的索引。
> **本机 OCR 最终选型（GLM-OCR 为首选）与部署详案已独立成文**：[glm-ocr-dgx-spark-deployment.md](glm-ocr-dgx-spark-deployment.md)。

> 主机：`spark-XXXX` · NVIDIA DGX Spark（GB10 / Blackwell，sm_121，128 GB 统一内存，aarch64，Ubuntu 24.04，内核 6.17-nvidia，CPU 20 核 = 10×Cortex-X925 + 10×A725）
> 日期：2026-09-15 · 主题：结合官方 PaddleOCR-VL 文档 + 本机实测，探求在本机部署 **PaddleOCR-VL-1.6** 的可行性
> 前置背景：现主服务 `qwen38-27b-sglang`（SGLang + Qwen3.8-27B-NVFP4 + DFlash2，:8888）、`venv-vllm`（vLLM 0.27.1 / torch 2.13.0，aarch64）、SGLang dev 镜像均已在本机跑通

---

## 0. 省流版结论

| 维度 | 判定 |
|---|---|
| **官方一键 Docker** | ❌ 不可直接用。官方 Blackwell 教程及配套镜像面向 **x86_64**（sm120 tag、prebuild flash-attn 轮子均为 `linux_x86_64`，官方仅在 RTX 5070 上完成验证），aarch64 的 DGX Spark 拉不到 / 跑不起来；除非自己重编译，不划算。 |
| **手动 aarch64 安装（推荐路线 Path B）** | ✅ 可行，但**版面分析阶段只能走 CPU**（PyPI 无 PaddlePaddle GPU-aarch64 构建）；VLM 识别阶段走 GPU vLLM。 |
| **显存 / 内存** | ✅ 0.9B VLM 权重约 1.8–2.5 GB（FP16），128 GB 统一内存下即便 Qwen3.8-27B 主服务在跑也毫无压力。 |
| **唯一必须手测的关键点** | ⚠️ PaddleOCR-VL-0.9B 的 LLM 头是 **ERNIE-4.5-0.3B**（NaViT 视觉编码器 + ERNIE-4.5-0.3B），原生 vLLM 不直接认识该架构；官方要求用 PaddleOCR 自己的 `genai_server` 封装启动 vLLM 服务（官方标注"直接裸用 vLLM/SGLang 启动：未验证"）。**该封装在本机 aarch64 + vLLM 0.27.1 / torch 2.13 上是否跑得通，是唯一未验证环节，动手前务必实测。** |
| **GitHub 有无现成"开箱即用"** | 无 aarch64 + GB10 的一键仓。官方 PaddleOCR 主仓 Blackwell 教程 = x86_64；社区 GGUF 仓（`PaddlePaddle/PaddleOCR-VL-1.6-GGUF`，45 万+ 下载）生态成熟但同样偏 x86。 |

**★ 本机选型结论（详见 §7）**：两个模型本机的 VLM 侧**都能被现有 `venv-vllm`（vLLM 0.27.1，aarch64）原生加载**（`paddleocr_vl` 与 `hunyuan_vision` 均在 registry）。但**综合部署复杂度、速度、与本机既有技术栈的契合度，推荐 HunyuanOCR-1.5**：单一端到端 VLM（~1.5B，无独立版面模型、无需 PaddlePaddle CPU 编排）、DFlash 投机加速正是本机主服务已在用的技术、官方 CUDA 13 端到端验证（本机 13.0 ✓）、基准更快（1.41s/img vs 1.74s/img）。PaddleOCR-VL-1.6 是**精度/多语言更强**的备选（OmniDocBench v1.6 SOTA 96.3%、109 语言、印章/古籍），代价是两阶段、版面需 PaddlePaddle CPU、链路更长。

**★ 更广候选结论（本机最终选型）**：本机 vLLM 0.27.1 的 registry **原生注册了整个 OCR 模型族**（glm_ocr / qwen3_5 / unlimited_ocr / deepseek_ocr(2) / dots_ocr / hunyuan_vision / paddleocr_vl 等，均无需 trust-remote-code）。综合许可、体积、精度与“版面能否走 GPU”，**GLM-OCR 为本机首选**——其版面经官方 SDK 的 torch 路径可上 GPU，不依赖 PaddlePaddle。详见 [glm-ocr-dgx-spark-deployment.md](glm-ocr-dgx-spark-deployment.md)。

---

## 1. 本机实测环境（写本文时逐项核查）

| 项 | 实测值 | 说明 |
|---|---|---|
| 架构 | `uname -m` = aarch64 | GB10，非 x86_64 |
| GPU | PCI device `0x2e12`（GB10 / Blackwell，sm_121） | lspci 确认 |
| 驱动 | 580.173.02（aarch64 open kernel module） | /proc/driver/nvidia 正常 |
| CUDA toolkit | 13.0（`/usr/local/cuda/bin/nvcc`，V13.0.88） | 满足官方"驱动支持 CUDA ≥12.9"要求 |
| `nvidia-smi` | **失败**：`Failed to initialize NVML: Unknown Error` | GB10 上 NVML 本就不可用（官方已知现象），**不代表 GPU 故障**——现主服务 Qwen3.8-27B 正在该 GPU 上运行。查 GPU 状态用 CUDA 手段（如 torch `cuda.get_device_capability`）。 |
| 内存 | 总 121 Gi，当前 available ~26 Gi（主服务运行中） | 0.9B 增量可忽略 |
| 磁盘 | `/` 可用 3.3 T | 下模型 + 镜像充足 |
| Docker | 29.2.1，runtimes 含 `nvidia`，arch aarch64 | 容器 GPU 直通 OK |
| Python | 系统 3.12.3 | 落在官方支持 3.9–3.13 区间 |
| 现成推理栈 | `venv-vllm`：vLLM 0.27.1 / torch 2.13.0（aarch64） | 可直接复用做 VLM 后端 |

**PaddlePaddle 轮子实测（关键，已逐一验证）**：
- `paddlepaddle`（基础包，**CPU**）3.2.2 → ✅ **有 aarch64 轮子**：`paddlepaddle-3.2.2-cp312-cp312-manylinux2014_aarch64.whl`（88 MB；cp39–313 全有）。
- `paddlepaddle-gpu` → ❌ **PyPI 无任何 aarch64 轮子**（`No matching distribution found`）；官方 `paddlepaddle.org.cn/packages/stable/cu129/` 索引里也**只有 x86_64 的 GPU 轮子**。
- **结论**：本机 PaddlePaddle 只能装 **CPU 版**。版面分析（PP-DocLayoutV2，检测类小模型）在 20 核 CPU（10×X925 大核）上完全可接受，不是瓶颈；性能瓶颈在 VLM，而 VLM 走 GPU。

---

## 2. 对"现成方案"的逐条更正（避免照抄踩坑）

> 此前调研中流传的"vLLM 官方镜像 + PaddleOCR pipeline"写法有多处与官方文档不符，按 PaddleOCR 官方 Blackwell 教程与主教程更正：

| 流传说法 | 更正（以官方为准） |
|---|---|
| `--model paddlepaddle/PaddleOCR-VL-1.6-0.9B` 直接喂给 vLLM | HF 仓库真实名是 **`PaddlePaddle/PaddleOCR-VL-1.6`**（34,742 下载；另有 GGUF 版 452,876 下载）。"0.9B"是**完整流程里的 VLM 组件名**（PaddleOCR-VL v1 的 VLM 即 PaddleOCR-VL-0.9B），不是独立 HF 仓库；且不能裸喂 vLLM——官方明确：直接请求 vLLM/SGLang/FastDeploy 服务**≠完整 PaddleOCR-VL 流程**，会导致"无法复现论文精度 / 大量幻觉文本"。 |
| "0.9B 结构 = PP-DocLayout + 0.9B VLM" | 基本对，但 VLM 内部 = **NaViT 风格动态分辨率视觉编码器 + ERNIE-4.5-0.3B 语言模型**（不是 Qwen 系！这决定了 vLLM 加载方式）。版面模型为 **PP-DocLayoutV2**。支持 109 种语言。 |
| 构造参数 `PaddleOCR(pipeline_name=..., model_version="1.6", backend="vllm", vllm_server_url=...)` | 真实 API 是 **`from paddleocr import PaddleOCRVL`**，`pipeline = PaddleOCRVL(device=...)`，`pipeline.predict(input)` 后 `res.print() / res.save_to_markdown() / res.save_to_json()`。CLI：`paddleocr doc_parser -i xxx.png --save_path ./output`。安装：`pip install -U "paddleocr[doc-parser]"`。上一版那个构造参数不存在。 |
| 官方 vLLM 镜像 `vllm/vllm-openai:latest` | 本机**不需要也不该**另拉 x86 的 vllm-openai。VLM 侧走本机 aarch64 的 vLLM（`venv-vllm` 已有 0.27.1/torch 2.13），且须通过 PaddleOCR `genai_server` 封装启动（`paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --backend vllm --port 8118`），由封装层处理 ERNIE 架构与 NaViT 动态分辨率。 |
| "1.6 与 1.5 结构一致，零成本迁移" | 官方属实：2026-05-28 发布，OmniDocBench v1.6 达 96.3%，与 1.5 结构完全一致，可无缝迁移。 |

**正确的两阶段结构（官方定义）**：
1. **版面分析**：PP-DocLayoutV2，整图输入 → 检测定位各类版面元素（表格/公式/文本…）+ 阅读顺序 → 裁出元素子图。PaddlePaddle 跑（本机 = CPU）。
2. **VLM 识别**：PaddleOCR-VL-0.9B，逐子图独立推理生成 Markdown/文本 → 按版面顺序合并。GPU vLLM（经 PaddleOCR `genai_server` 封装）跑。
- 完整能力**必须两阶段协同**，仅跑 VLM 会丢精度。

---

## 3. 推荐路线：手动 aarch64 安装（Path B）

> 官方两条路：A) 官方 Docker 镜像（**x86_64 专用**，❌ 本机不适用）；B) 手动安装推理引擎 + PaddleOCR（✅ 本机走这条，官方支持 3.9–3.13，本机 3.12.3）。

### 3.1 步骤

```bash
# (1) 版面/客户端侧：PaddlePaddle CPU + PaddleOCR doc-parser（aarch64）
python3 -m venv /home/<user>/projects/venv-paddleocr
source /home/<user>/projects/venv-paddleocr/bin/activate
pip install -U paddlepaddle            # aarch64 CPU 轮子（已验证存在，3.2.2）
pip install -U "paddleocr[doc-parser]" # 当前 PyPI 最新 3.7.0

# (2) 兜底验证：纯 CPU 全流程（不接 GPU）
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png --save_path ./output
# 0.9B 在 20 核 CPU 上可出结果，速度较慢但一定跑通 → 先确认功能链路

# (3) VLM 侧：装 PaddleOCR 的 vLLM 服务封装依赖（建议在独立 venv，避免与 venv-vllm 现有栈冲突）
python3 -m venv /home/<user>/projects/venv-paddle-vlm
source /home/<user>/projects/venv-paddle-vlm/bin/activate
pip install -U paddlepaddle "paddleocr[doc-parser]"
paddleocr install_genai_server_deps vllm

# (4) 启动 VLM 推理服务（PaddleOCR 封装，非裸 vLLM；0.9B 显存占用极低）
paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --backend vllm --host 0.0.0.0 --port 8118
# 如需限显存：加 --backend_config vllm_config.yml（参考官方 §3.3.1 服务端参数调整）
```

```python
# (5) 客户端调用（完整流程；VLM 指向本机 :8118）
from paddleocr import PaddleOCRVL
pipeline = PaddleOCRVL()   # 版面侧本机 CPU；VLM 走本地 genai_server
out = pipeline.predict("document.pdf")  # 支持 PDF / 图片
for res in out:
    res.print()
    res.save_to_markdown(save_path="./output")
```

### 3.2 关键待验证点（务必先手测再谈服务化）

`paddleocr genai_server --backend vllm` 能否在 **aarch64 + GB10(sm_121) + 本机 torch 2.13/vLLM** 上把 **ERNIE-4.5-0.3B + NaViT 编码器** 正确加载并出图。官方对"直接裸用 vLLM/SGLang 启动"标注**未验证**，对本机 aarch64 同样无官方声明。验证方法：
1. 拉官方 demo 图（§3.1 第 2 步同图），经 `genai_server`(:8118) + 客户端 `predict` 跑一遍，核对 markdown：**表格/公式/正文正常、无幻觉** → 判定通过。
2. 若 `install_genai_server_deps vllm` 拉到的 vLLM/flash-attn 与 aarch64 不匹配（官方 prebuild flash-attn 轮子是 `linux_x86_64`）→ 在本 aarch64 venv 里 `pip install flash-attn` 现编（需 nvcc + CUDA 13.0，本机具备），或退回纯 CPU 兜底路线（§3.1 第 2 步已验证可通）。

### 3.3 资源预算

- 0.9B VLM：FP16 权重 ~1.8 GB + KV cache（`--max-model-len` 4K 级）→ 实际 **3–4 GB**，128 GB 统一内存可忽略；与 Qwen3.8-27B 主服务**可共存**（内存吃紧时给 VLM 限 `gpu-memory-utilization ≤ 0.2`）。
- 版面 PP-DocLayoutV2（CPU）：数百 MB，单核级占用。
- 磁盘：模型 + 依赖约 5–10 GB，当前可用 3.3 T，无压力。

#### 3.3.1 内存预算实测（2026-09-15 补，回答"现在还剩这么少内存还能部署吗"）

| 项 | 实测 | 说明 |
|---|---|---|
| 主服务稳态 | **实际运行值 `--mem-fraction-static 0.70`**（2026-09-15 从进程命令行核对；`start-long.sh` 脚本默认是 0.85）→ **约 85 GiB 预留** | 权重 + KV cache 一次性按该比例吃满统一内存 |
| 稳态 available | 会话初期实测 **used 95 Gi / available 26 Gi**（swap 仅用 395 Mi） | 这才是做预算该看的数 |
| 暂态陷阱 | 本次实测（容器 `Up 2 minutes` 重新加载中）：used 29 Gi / **available 92 Gi** | 加载中 KV cache 尚未分配，**不可作为预算依据** |
| PaddleOCR-VL-1.6 增量 | VLM 0.9B（权重 1.92 GB）经 vLLM **~3–5 GB** + PaddlePaddle CPU 版面 **~0.5–2 GB** → 合计 **~5–8 GB** | 磁盘另需 ~5–10 GB（可用 3.3 T，无压力） |

**结论：仍然可部署。** 稳态 ~20–26 GiB 余量 vs 需求 ~5–8 GB，够用但属"紧凑"预算。横向对照 [LOCAL-JUDGE-LESSONS.md](LOCAL-JUDGE-LESSONS.md)：同一余量下 **8B 级**（5.7 GB 权重 + KV）曾被判定"放不下"，而 0.9B 约为其 1/6，余量充足。

**三个收紧杠杆**：
1. 给 Paddle 侧 VLM 显式设上限：`--gpu-memory-utilization 0.04~0.06`（按 121 GiB 总量约合 5–7 GB）。
2. 版面阶段（CPU）固定到**空闲核**：主服务已用 `CPUSET=5-9,15-19`（10 核），空闲 **0-4,10-14**（10 核）可留给 PP-DocLayoutV2。
3. 必要时把主服务 `MEM_FRACTION` 由 0.85 降到 0.80（[LOCAL-JUDGE-LESSONS.md](LOCAL-JUDGE-LESSONS.md) 实测可多出约 5.8 GiB）。

**时机建议**：等主服务加载完成、`:8888` 正常响应后再起 OCR 服务；两者不要同时冷启动（同时加载会争统一内存，易触发 swap）。

> 若改选 GLM-OCR（本机首选，见 [glm-ocr-dgx-spark-deployment.md](glm-ocr-dgx-spark-deployment.md)）：权重 2.65 GB → 增量约 **4–6 GB**，与 Paddle 路线同级预算。

---

## 4. 备选路线（Path B 手测不通过时的退路）

| 路线 | 说明 | 适合 |
|---|---|---|
| **GGUF + 现有 vLLM** | `PaddlePaddle/PaddleOCR-VL-1.6-GGUF`（45 万下载，社区生态最成熟）；本机 vLLM 0.27.1 支持 GGUF load-format（见 `LOCAL-JUDGE-LESSONS` 记录：2026 版 vLLM 移除 GGUF 加载——**需先验证 0.27.1 是否仍保留**，若已移除则走 Ollama/llama.cpp）。可快速验证 VLM 侧兼容性。 | 快速试错；但属"只跑 VLM"，完整流程精度仍需客户端版面编排。 |
| **纯 CPU 全流程** | `paddleocr doc_parser` 不接 GPU。0.9B 在 10×X925 大核上可出结果，**慢但一定通**。 | 兜底 / 低并发验证功能 / 无 GPU 依赖的离线场景。 |
| **换模型** | **Qwen2.5-VL-7B / GOT-OCR-2.0**：aarch64 + Blackwell 的 Dockerfile、SGLang/vLLM 一键脚本、HF 支持都成熟；本机 SGLang dev 镜像现成。 | 工程成熟度最高；代价是放弃 PaddleOCR-VL 的版面-阅读顺序编排与 109 语言能力。 |

> 取舍原则：要"必须是 PaddleOCR-VL-1.6 的完整能力（版面顺序 + 复杂元素 + 109 语言）" → Path B；要的是"文档→高质量 Markdown 的通用服务"、不强绑百度模型 → Qwen2.5-VL 路线本机最省心。

---

## 5. 风险与注意事项

1. **官方 Blackwell 教程 ≠ 本机**：`PaddleOCR-VL-NVIDIA-Blackwell.md` 面向 x86_64（RTX 50xx / sm120），Docker 镜像与 prebuild flash-attn 轮子均为 `linux_x86_64`，官方仅在 RTX 5070 完成验证。本机 GB10 是 aarch64，**不能直接套用其 Docker 步骤**，只能参考"手动安装 + genai_server"逻辑。
2. **PaddlePaddle 无 aarch64 GPU 构建** → 版面阶段注定 CPU。非致命（检测模型 CPU 够快），但要纳入预期：全链路 GPU 加速在本机做不到。
3. **ERNIE 头 + vLLM 未验证** → 唯一可能卡住的点；先做 §3.2 单图手测，通了再谈服务化。
4. **不要"只喂 VLM"** → 否则精度不可复现 / 幻觉（官方反复强调）。务必经 PaddleOCR 客户端完整流程（版面 + VLM 协同）。
5. **nvidia-smi 失效是正常现象**（GB10 不支持 NVML），别当故障信号；GPU 状态用 CUDA/torch 探测。
6. **内存共存 + 避免空转**：主服务 Qwen3.8-27B 占用不小，VLM 服务显存预算保守；部署后必须有真实调用方，否则参考 judge :8889 教训（16 天无调用即被关停，9/10）——跑完即停，勿常驻空转。

---

## 6. 落地 checklist（动手前逐项勾）

- [ ] 建 `venv-paddleocr`：装 `paddlepaddle`(CPU) + `paddleocr[doc-parser]`，跑官方 demo 图纯 CPU 全流程，确认 markdown 正常（兜底链路）。
- [ ] 建独立 `venv-paddle-vlm`：`paddleocr install_genai_server_deps vllm`，确认 aarch64 下 vLLM 版本 / flash-attn 可用（必要时 nvcc 现编）。
- [ ] `paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --backend vllm --port 8118` 启动成功，客户端 `predict` 官方 demo 图：表格/公式/正文正常、无幻觉 → **关键手测通过**。
- [ ] 端口避开 4000/8888；VLM 服务 `gpu-memory-utilization` ≤ 0.2，与主服务共存观察内存（available 保持 >10%）。
- [ ] 有真实调用方再考虑常驻；否则跑完即停。
- [ ] 若 ERNIE 头在 aarch64 vLLM 卡住 → 转 §4 备选（GGUF 试跑 / 纯 CPU / 换 Qwen2.5-VL）。

---

## 7. 本机选型对比：HunyuanOCR-1.5 vs PaddleOCR-VL-1.6

> 已逐项核对（本机 vLLM registry、HF 权重、官方推理/基准/许可文档）。**关键前提**：本机 vLLM 0.27.1（aarch64）的 model registry **同时原生注册**了两个架构——`HunYuanVLForConditionalGeneration`(registry L432) 与 `PaddleOCRVLForConditionalGeneration`(L534)，都不需要 `--trust-remote-code`、也不需要 PaddleOCR 的 `genai_server` 封装来"让 vLLM 认识模型"（后者只在 PaddleOCR 客户端编排层面有价值）。

### 7.1 逐项对比

| 维度 | HunyuanOCR-1.5（腾讯） | PaddleOCR-VL-1.6（百度） |
|---|---|---|
| 架构 | **单一端到端 OCR VLM**（~1.5B，`hunyuan_vl`），版面/阅读顺序/正文/表格/公式/spotting/IE/文图翻译**全内建** | **两阶段**：PP-DocLayoutV2（版面）+ PaddleOCR-VL-0.9B（VLM，ERNIE-4.5-0.3B 头，`paddleocr_vl`） |
| 本机 vLLM 原生支持 | ✅ `hunyuan_vision.py`（registry L432） | ✅ `paddleocr_vl.py`（registry L534） |
| 版面分析 | 内建（单模型直接出，走 GPU） | 需 PaddlePaddle 跑 PP-DocLayoutV2；**本机 PaddlePaddle 只有 CPU 版**（无 aarch64 GPU 构建）→ 版面走 CPU |
| 部署件数 | **1**（单 vLLM 服务即可端到端出结果） | **3**（PaddlePaddle CPU 版面 + vLLM VLM + PaddleOCR 客户端编排/裁剪/合并） |
| 权重体积（实测） | `model.safetensors` **2.24 GB**(bf16) + `dflash/` 草稿模型 | VLM 头 `model.safetensors` **1.92 GB** + 版面模型（Paddle 自下载，数百 MB） |
| 显存/内存（本机） | ~4–5 GB（含 KV） | ~3–4 GB（VLM）+ 版面 CPU 数百 MB |
| 加速 | **DFlash 投机解码 2.14× 提速**；本机主服务 Qwen3.8-27B 正在用 DFlash2 → 技术已验证 | 无对应投机加速；版面 CPU 是速度短板 |
| 基准速度（官方 H20，同条件） | **1.41 s/img**（+DFlash）、0.706 page/s | 1.74 s/img、0.562 page/s（两阶段 0.9B） |
| 精度 | 端到端强；官方称 AR/DFlash/transformers 三者精度已对齐 | **OmniDocBench v1.6 SOTA 96.3%**（更高）、1.5 精度 94.5% |
| 语言覆盖 | 多语言（以 en/zh 为主） | **109 种语言** |
| 特色能力 | 文档解析/spotting/IE/文图翻译统一；CHAOS-Bench 字符级抗幻觉 | 印章识别、古籍/生僻字、spotting、图表、异形框定位 |
| 官方硬件验证 | **CUDA 13 端到端测试通过**（本机 CUDA 13.0 ✓）；有 llama.cpp/GGUF PC 端路线 | Blackwell 教程面向 **x86_64**（RTX 5070 验证）；aarch64 未官方验证 |
| License | **腾讯混元社区协议**（HF `license:other`）：地域**排除 EU/UK/韩国**，本机 HK 在授权范围内；须守 Acceptable Use Policy；输出不构成模型派生 | **Apache 2.0**（HF tag `license:apache-2.0`，更宽松） |
| 本机契合度 | **★ 高**：单模型 + 现有 vLLM + DFlash（已验证）+ CUDA13 | 中：需额外 PaddlePaddle(CPU) 编排，链路更长 |

### 7.2 结论与适用场景

- **本机默认推荐：HunyuanOCR-1.5。** 理由：单模型端到端、vLLM 原生支持、**无需 PaddlePaddle 依赖**（避开"本机只有 CPU 版 Paddle"的短板）、**DFlash 加速与本机主服务同栈**（已在 GB10 验证）、官方 **CUDA 13 端到端验证**、基准速度更快、显存 4–5 GB 与主服务共存轻松。**若你的诉求是"文档/图片 → 高质量 Markdown 的通用 OCR 服务"，选它最省心。**
- **选 PaddleOCR-VL-1.6 当且仅当**：需要**最高版面精度**（OmniDocBench v1.6 SOTA 96.3%）、**109 语言**、或**印章/古籍/生僻字**等 PaddleOCR 强项，且能接受两阶段 + PaddlePaddle CPU 编排 + 更长的部署链路。
- **License 提醒**：HunyuanOCR 为腾讯混元社区协议，**地域排除 EU/UK/韩国**（本机 HK 合规），商用前须读 Acceptable Use Policy；PaddleOCR-VL-1.6 为 Apache 2.0，对外分发/商用更无负担。若服务要面向 EU/UK/韩国用户，PaddleOCR-VL-1.6 的许可更干净。

---

## 8. HunyuanOCR-1.5 本机部署步骤（推荐路线）

> 官方统一推理环境要求 **CUDA 13**（本机 13.0 ✓）、Python 3.12（本机 3.12.3 ✓）、`vllm>=0.25.1`（本机 0.27.1 ✓）、`flash-attn==2.8.3`（aarch64 需 nvcc 现编，本机 CUDA 13 + nvcc 具备）。三条配置：vLLM AR / vLLM+DFlash / 原生 transformers，官方已验证三者精度对齐。**推荐 DFlash**（本机主服务同款加速）。

```bash
# (1) 统一环境（uv + CUDA13）
pip install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1" runai-model-streamer
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"   # aarch64 现编（本机 nvcc+CUDA13 具备）

# (2) 下载权重（base + dflash/ 草稿，共 ~2.2 GB + 草稿）
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"

# (3) 启动服务（AR 或 DFlash；端口避开 4000/8888，用 8000/8001）
# AR:
MODEL_PATH=./HunyuanOCR PORT=8000 bash inference/vLLM/serve.sh
# DFlash（推荐，复用本机已验证的投机解码）:
MODEL_PATH=./HunyuanOCR PORT=8000 bash inference/DFlash/serve_DFlash.sh
curl -sf http://127.0.0.1:8000/v1/models   # 就绪检查

# (4) 单图 / 批量（OpenAI 兼容客户端；12 种 task-type，doc_parse 为文档解析）
python inference/vLLM/infer_vllm_client.py --image doc.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
python inference/vLLM/batch_infer.py --image-dir ./imgs --out-dir ./out \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

**兜底路线**：无 CUDA 13 或只要 CPU/PC 端 → 官方 `llama.cpp` + GGUF（`docs/llama_cpp.md`），OpenAI 兼容 `llama-server`，含 DFlash 适配 fork；1.5B 在 20 核 CPU 上亦可出结果。

**资源/共存**：1.5B 权重 2.24 GB + KV → 4–5 GB；`gpu-memory-utilization` 设 0.15 即可与 Qwen3.8-27B 主服务共存；部署后须有真实调用方（勿空转，参考 judge :8889 教训）。

**待手测（动手前）**：① aarch64 下 `flash-attn==2.8.3` 现编能否通过（不过则让 vLLM 走非 flash 的 attention backend 兜底）；② `serve_DFlash.sh` 在本 GB10 上 DFlash 草稿模型（`dflash/`）加载与 2.14× 提速是否成立；③ 单图 doc_parse 出 markdown 无幻觉。

---

## 9. 其他 OCR 候选调研（已独立成文）

> 本节原有内容（OCR 候选全景 + 本机 vLLM 原生支持表 + GLM-OCR 部署详案）已拆分到独立文档：
> **[glm-ocr-dgx-spark-deployment.md](glm-ocr-dgx-spark-deployment.md)**

结论摘要：本机 vLLM 0.27.1（aarch64）的 model registry **原生注册了整个 OCR 模型族**（glm_ocr / glm_ocr_mtp / qwen3_5 / unlimited_ocr / deepseek_ocr / deepseek_ocr2 / dots_ocr / hunyuan_vision / paddleocr_vl / qwen3_vl 等），均无需 --trust-remote-code。综合许可、体积、精度、以及“版面能否走 GPU”，**GLM-OCR 为本机 OCR 首选**（0.9B、权重 MIT / SDK Apache-2.0、OmniDocBench V1.5 排名第 1、版面经 SDK 的 torch 路径可上 GPU —— 不依赖 PaddlePaddle），chandra-ocr-2 次之。完整对比与部署见上述独立文档。

---

## 附：官方资源（已逐项核对）

**PaddleOCR-VL-1.6**
- 主教程：`PaddlePaddle/PaddleOCR` → `docs/version3.x/pipeline_usage/PaddleOCR-VL.md`（推理方式与硬件支持矩阵、快速开始、genai_server、客户端用法、性能调优）
- Blackwell 专章（**x86_64 专用**）：`.../PaddleOCR-VL-NVIDIA-Blackwell.md`
- Docker Compose 完整 API 服务（x86_64）：`deploy/paddleocr_vl_docker/accelerators/nvidia-gpu-sm120/`（:8080，双容器）
- HF 模型：`PaddlePaddle/PaddleOCR-VL-1.6`（主仓，Apache 2.0）、`PaddlePaddle/PaddleOCR-VL-1.6-GGUF`（452.9k 下载）
- demo 图：`https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png`

**HunyuanOCR-1.5**
- 仓库：`Tencent-Hunyuan/HunyuanOCR`（2.0k star，Python，许可=腾讯混元社区协议）
- 推理文档：`docs/inference/inference.md`（统一环境 / CUDA13 / vLLM AR / DFlash / transformers）
- 基准：`docs/benchmark.md`（AR vs DFlash 2.14×；vs PaddleOCR-VL 1.6 等 7 模型横向对比，H20 80GB）
- PC/llama.cpp：`docs/llama_cpp.md`（GGUF + DFlash 适配 fork）
- HF 模型：`tencent/HunyuanOCR`（655k 下载，`hunyuan_vl`，bf16，base 2.24 GB + `dflash/` 草稿 + `v1.0/`）
- 论文：arXiv 2607.04884（1.5）/ 2511.19575（1.0）
- 本机 vLLM registry 证据：`venv-vllm/.../vllm/model_executor/models/registry.py` L432（HunYuanVLForConditionalGeneration）、L534（PaddleOCRVLForConditionalGeneration）

**其他候选（§9）**
- GLM-OCR：`zai-org/GLM-OCR`（MIT，0.9B，OmniDocBench V1.5 #1；SDK `zai-org/GLM-OCR`；本机 registry `glm_ocr`+`glm_ocr_mtp`；vLLM recipes + SGLang cookbook）
- chandra-ocr-2 / surya-ocr-2：`datalab-to/*`（OpenRAIL，`qwen3_5` arch；`pip install chandra-ocr` + `chandra_vllm`；85.8% olmOCR、90+ 语言）
- Unlimited-OCR：`baidu/Unlimited-OCR`（MIT，MoE，`unlimited_ocr`）
- DeepSeek-OCR / OCR-2：`deepseek-ai/DeepSeek-OCR*`（MIT / Apache-2.0，`deepseek_ocr`/`deepseek_ocr2`）
- Qwen-VL 系：`qwen2_5_vl` / `qwen3_vl` / `qwen3_vl_moe`（Apache-2.0，通用 VLM 兼 OCR，本机主服务同系）
