---
name: glm-ocr-remote
description: >-
  调用自托管的 GLM-OCR 文档识别服务（本机或内网 DGX Spark 上的 glmocr SDK + vLLM），
  把图片、截图、扫描件、PDF 转成文字与 Markdown，支持中文、英文、表格、公式、多页 PDF 与版面阅读顺序。
  当用户要 OCR、提取图片或 PDF 中的文字、截图识字、扫描件转文字、表格识别、公式识别、PDF 转 Markdown 时使用。
  Trigger: OCR, 文字识别, 图片转文字, 截图识字, 提取图中文字, 扫描件识别, PDF 转文字, PDF 转 Markdown,
  表格识别, 公式识别, GLM-OCR, image to text, extract text from image, screenshot OCR, table recognition, document parsing。
  不用于：询问图片内容或含义（用模型自身视觉能力即可）；纯文本处理。
metadata:
  version: "1.0"
---

# GLM-OCR 远程识别

通过 HTTP 调用自托管的 **GLM-OCR** 文档识别服务（DGX Spark 上的 `glmocr` SDK 服务 + vLLM），
把图片 / 截图 / 扫描件 / PDF 转成文字与 Markdown。**数据不出内网，无云端 token，无按次计费。**

- 服务地址：默认 `http://127.0.0.1:5010`（上传网关），可用 `--server`、环境变量 `GLMOCR_URL`
  或 skill 目录下 `config.env` 覆盖（优先级依次降低）
- 脚本纯标准库，**无需 pip 安装任何东西**；只有 `--pages` / `--dpi` 需要 PyMuPDF
- 单图约 **3–8 秒**；PDF 每页约 3–5 秒（服务端渲染 + 版面分析 + 逐区域识别）

## 何时使用 / 不使用

**使用**：提取图片或 PDF 中的文字；截图识字；扫描件数字化；表格转结构化文本；公式转 LaTeX；
需要保留版面阅读顺序的整篇文档转 Markdown。

**不使用**：
- 问图片"是什么 / 画了什么"这类视觉问答 → 用模型自身的看图能力，不要走 OCR
- 只需要读一张小图上的几个字 → 模型自身视觉能力更快

## 第一步：检查服务

```bash
python3 ~/.agents/skills/glm-ocr-remote/scripts/check_server.py
```

不可达时**不要反复重试**，直接告知用户并给出启动命令（见下方"服务端启停"）。
冷启动约 **2–4 分钟**（vLLM 加载权重 + 版面模型），期间 `/health` 会失败。

## 识别图片

```bash
S=~/.agents/skills/glm-ocr-remote/scripts

# 单张（中文截图、照片、扫描件都可以）
python3 $S/ocr.py 截图.png

# 多张
python3 $S/ocr.py a.png b.jpg

# 只要结构化结果（含 bbox 区域坐标与标签）
python3 $S/ocr.py 表格.png --json

# 落盘
python3 $S/ocr.py 扫描件.jpg -o out.txt
```

表格与公式**不需要指定提示词**：服务的版面分析会自动识别区域类型，
表格走 `Table Recognition:`、公式走 `Formula Recognition:`、正文走 `Text Recognition:`。
（这是本 skill 用"上传网关 + 完整 pipeline"而不是直接打 vLLM 的原因。）

## 识别 PDF

默认把整个 PDF 交给服务端：服务端用 PyMuPDF 按 200 DPI 渲染每一页，
逐页做版面分析，最后把各页 Markdown **按顺序拼接在同一段结果里**（单个 PDF 是一个"单元"，
**不会**插入 `---` 页分隔符；`---` 只在一次上传多个文件、形成多个单元时出现）。

需要**页级切分**时：用 `--json`（`json_result` 按页给出，每页一组区域），
或用 `--pages` 逐页渲染（每页单独成一段输出）。

```bash
# 整本（多页自动合并）
python3 $S/ocr.py 报告.pdf

# 只要某几页 —— 这时改由客户端渲染，需要 PyMuPDF（pip install PyMuPDF）
python3 $S/ocr.py 报告.pdf --pages 1-3,7
python3 $S/ocr.py 报告.pdf --pages 1-3 --dpi 300 -o 前3页.txt
```

## 结果处理

- **完整展示识别结果，不要截断**；超过约 1 万字再考虑落盘 + 摘要
- 表格在 Markdown 里已是表格结构；如需单元格原文可加 `--json`
- 已知输出风格（模型特性，不是错误，可提示用户）：
  - 中文百分号会变 LaTeX：`22.1%` → `$22.1\%$`
  - 千分位逗号偶尔被全角化：`486,300` → `486，300`

## 服务端启停（仅在服务跑在本机时）

```bash
bash /home/<user>/projects/glm-ocr-test/start-service.sh --with-gateway   # 起 vLLM + SDK 服务 + 网关
bash /home/<user>/projects/glm-ocr-test/stop-service.sh                  # 全停，归还内存
```

- 全栈常占约 **10 GB** 统一内存，与主服务（`:8888`）共享 128 GB 池 → **用完即停**
- vLLM 需要 GPU 直通权限：**不要在沙箱/受限 shell 里启动**（会报 `Can't initialize NVML`）

## 故障排查

| 现象 | 原因与处理 |
|---|---|
| `服务不可达` / Connection refused | 服务没起 → 用上面的 start 脚本；冷启动要等 2–4 分钟 |
| **返回 200 但内容是空的** | 地址指向了内置服务 `:5002` 而不是网关 `:5010`，且输入是 PDF。内置服务只认 `data:image` / `file://`，**对 PDF 与 http(s) URL 会静默返回空**。脚本已内置该检查并会提示；改用 `--server http://127.0.0.1:5010` |
| HTTP 404 / 405 | 地址不是上传网关（没有 `/parse`）；脚本会自动回退到内置服务的 JSON 契约，但 PDF 仍需网关 |
| 首次请求很慢 | vLLM 冷启动 + 大图；之后同尺寸图片约 3–5 秒 |
| PDF 报错缺 PyMuPDF | 只有 `--pages` / `--dpi` 需要：`pip install PyMuPDF`；不加这两个参数无需任何依赖 |
| 结果乱码 | 图片倒置或严重形变，换正向扫描图重试 |

## 技术参考

- **调用链**：本 skill → 网关 `:5010`（multipart 上传）→ SDK 服务 `:5002`（版面分析 + 区域裁剪 + 并发调度）→ vLLM `:8080`（GLM-OCR 0.9B 推理）。对外**只有一来一回**，内部几十次区域调用全走 localhost
- 上传网关接受 multipart，也接受官方 SDK/MaaS 的 JSON 契约（`{"file": "data:application/pdf;base64,…"}`）
- 官方 Python SDK 客户端也可直接把 `api_url` 指到网关：`http://<host>:5010/glmocr/parse`
- 完整部署与实测记录：`/home/<user>/projects/dgx-spark-notes/notes/glm-ocr-client-integration.md`
