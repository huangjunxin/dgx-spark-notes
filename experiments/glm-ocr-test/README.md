# glm-ocr-test — GLM-OCR 本机验证 / 复现目录

本目录是 **GLM-OCR 在 DGX Spark（aarch64 GB10）上的实测复现包**：服务端一键启停、
客户端各条调用路径的验证脚本、根因探针、测试素材与产物。

配套文档在 `dgx-spark-notes` 仓库：

| 文档 | 内容 |
|---|---|
| `notes/glm-ocr-dgx-spark-deployment.md` | 部署方案 + §10 真机实测（含 MTP 不可用、内存实测、§12 服务化现状与踩坑） |
| `notes/glm-ocr-client-integration.md` | 客户端接入：调用链拆解、三种接法、PDF 的坑、§12 终端 agent skill |

---

## 一键启停

```bash
bash start-service.sh --with-gateway   # vLLM :8080 + SDK 服务 :5002 + 上传网关 :5010
bash stop-service.sh                   # 全停，归还约 10 GB 统一内存
```

- 服务地址：`:5002`（SDK 服务，图片）、**`:5010`（上传网关，图片 + PDF，推荐）**
- 冷启动约 **2–4 分钟**（vLLM 加载权重 + 版面模型）；启动脚本逐级探活

## 前置条件

| 项 | 值 |
|---|---|
| Python 环境 | `/home/<user>/projects/venv-vllm`（vLLM 0.27.1 / torch 2.13.0+cu130 / glmocr 0.1.5 + `[selfhosted,server]`） |
| 模型缓存 | `/home/<user>/projects/glm-ocr-model-cache`（`HF_HOME`，含 GLM-OCR 权重 2.5 GB 与 PP-DocLayoutV3） |
| GPU | 主机侧跑 vLLM **需要 GPU 直通权限**；受限 shell / 沙箱内会报 `Can't initialize NVML` |
| 缓存目录 | `VLLM_CACHE_ROOT=.vllm-cache`（`~/.cache/vllm` 曾被 root 占位不可写，见部署文档 §10.5 坑 1） |

---

## 文件说明

### 服务端

| 文件 | 说明 |
|---|---|
| `glmocr-server.yaml` | SDK 服务端配置（由包内默认 `config.yaml` 复制后定点替换：maas 关、版面 cpu、max_workers 8） |
| `start-service.sh` / `stop-service.sh` | 一键启停，逐级探活；停止脚本回读端口与内存 |
| `patch-config.py` | 从包内默认配置生成 `glmocr-server.yaml`（**注意：配置是分区浅合并**，直接手写局部键会覆盖整节默认值） |

### 客户端验证脚本

| 文件 | 验证内容 |
|---|---|
| `client-http-test.py` | HTTP 直连 7 项：健康检查 / 英文图 / 中文图 / MaaS 格式 / **PDF 三种路径** / 错误分支 |
| `client-sdk-test.py` | 官方 Python SDK MaaS 模式：PNG ✅ / **PDF 静默返回空** ❌ |
| `client-path-tests.py` | `http(s)` URL、直连 vLLM（无版面）、客户端逐页渲染 PDF |
| `client-sdk-via-gateway-test.py` | 官方 SDK 客户端 → 上传网关：PNG/PDF 均能被正确解析 |
| `page-loader-probe.py` | 根因单元探针：证明 `data:application/pdf` 与 `http(s)` 不被支持 |

### 测试桩（无需 GPU / 无需真模型）

| 文件 | 用途 |
|---|---|
| `_recording_stub.py` | 顶替 vLLM，记录网关真实转发给上游的载荷（含 `file://` 路径存活性与 magic） |
| `_stub_8080.py` | 顶替 :8080，验证 `start-service.sh` 的编排与探活逻辑 |

### 测试素材

| 文件 | 说明 |
|---|---|
| `testdoc.png` | 英文样例：正文 + 表格 + 公式 + 编号（版面检出 7 个区域） |
| `testdoc_zh.png` | 中文样例 |
| `multi.pdf` | 3 页 PDF（客户端 PDF 路径验证用） |

### 产物与日志

| 路径 | 说明 |
|---|---|
| `out-cpu/ out-cuda/ out-zh/ out-pdf/` | 部署文档 §10.8 的版面 pipeline 产物：`*.md` / `*.json`（区域标注）/ `layout_vis/`（可视化） |
| `vllm.log` | 9/15 AR 模式起服日志 |
| `vllm-mtp3.log` | **MTP 投机失败现场**（权重命名不匹配，部署文档 §10.4） |
| `vllm-serve.log` | 由 `start-service.sh` 产生的 vLLM 日志 |
| `glmocr-server.log` / `gateway.log` | SDK 服务与网关日志（**体积较大**：失败 warning 会把整个 base64 塞进日志行） |
| `uploads/` | 上传网关的临时落盘目录（请求结束即清空） |
| `.vllm-cache/` | vLLM 编译/autotune 缓存（可丢弃，会自动重建） |

---

## 关键结论速查（详见配套文档）

1. **图片**：`POST :5010/parse`（multipart）或 `:5002/glmocr/parse`（`data:image/*;base64`）→ 单图 2.6–4.9 s ✅
2. **PDF**：**必须**走上传网关 `:5010`。内置服务 `:5002` 只认 `data:image` / `file://`，对 `data:application/pdf` 与 `http(s)` **静默返回 200 + 空结果**（异常在 `page_loader.py:160-165` 被吞）
3. **MTP 投机不可用**：checkpoint 与 vLLM 的 MTP 权重命名不匹配 → 本机用纯 AR
4. **版面**：PP-DocLayoutV3 走 torch/transformers（**不依赖 PaddlePaddle**），建议 `layout.device=cpu`（省 3.6 GB，仅慢约 0.4 s/页）
5. **内存**：全栈约 10 GB，与主服务 `:8888` 共享 128 GB 统一内存池 → **跑完即停**
