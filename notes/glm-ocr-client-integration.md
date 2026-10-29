# GLM-OCR 客户端接入指南（spark-XXXX 实测）

> 主机：`spark-XXXX` · NVIDIA DGX Spark（GB10 / aarch64） · 日期：**2026-09-15**
> 姊妹文档：[glm-ocr-dgx-spark-deployment.md](glm-ocr-dgx-spark-deployment.md)（那份解决"**本机怎么把 GLM-OCR 跑起来**"，本文解决"**别的机器怎么调用它**"）
> 本文所有结论均为**本机真机实测**；标注 ⚠️ 的条目为源码判定（附行号/文件），未标注的均为跑过的实测数据。
> 实测环境：vLLM 0.27.1 + torch 2.13.0（AR 模式，无 MTP，见部署文档 §10.4）· glmocr 0.1.5 · PP-DocLayoutV3 版面走 **CPU**

---

## 0. 省流版结论

| 问题 | 结论 |
|---|---|
| 客户端装什么？ | **什么都不用装**（HTTP 直连），或只装 `pip install glmocr`（**不含 torch**） |
| 第一推荐接法 | **HTTP 直连 `POST /glmocr/parse`**，图片用 `data:image/png;base64,...` 传 ✅实测 |
| 最大的坑 | ⚠️ **内置服务只认 `data:image:*` 和 `file://`**。PDF 若按官方 README 用 MaaS 模式传，会 **HTTP 200 + 空结果**（不报错！）—— 见 §4.3 |
| PDF 怎么办 | ① **上传网关（推荐）**：`curl -F "file=@doc.pdf" :5010/parse`；官方 Python SDK 客户端把 `api_url` 指到 `:5010/glmocr/parse` 即零改动可用 ✅实测；② 客户端把 PDF 逐页渲染成 PNG 再发 ✅实测；③ PDF 已在服务端时直接传 `file://` 路径 ✅实测 |
| 服务端要改什么 | 装 `glmocr[server]`（本机原本只有 `[selfhosted]`，**flask 缺失**）+ 一份 config + 一条启动命令 |
| 网络 | 服务默认绑 `0.0.0.0:5002`，局域网/Tailscale 三地址实测全通；**但无鉴权**（见 §9） |
| 成本 | 全栈起来后 available 25 GB → **14.9 GB**（净占用约 10 GB）；**跑完即停**，停服后回到 26.3 GB |
| 单张图延迟 | **2.6–4.9 s**（中文更快）；3 页 PDF 约 **8–9.5 s** |
| 部署现状 | ⚠️ 服务**默认全停**（用则起、跑完即停）；安装清单 / 文件清单 / 端口 / 补充踩坑见部署文档 [§12](glm-ocr-dgx-spark-deployment.md) |

---

## 1. 拓扑与访问地址

```
┌──────────────────────┐   HTTP    ┌──────────────── spark-XXXX ─────────────────┐
│ 客户端（任何机器）    │ ────────► │  :5002  glmocr SDK 服务（版面+调度）         │
│ 无需 GPU / 无需 torch │           │    └── localhost HTTP ──► :8080 vLLM(glm-ocr)│
└──────────────────────┘           │  :5010  上传网关（可选，吃 multipart/PDF）   │
                                   └──────────────────────────────────────────────┘
```

| 服务 | 监听 | 本机 | 局域网 | Tailscale |
|---|---|---|---|---|
| **glmocr SDK 服务**（客户端主入口） | `0.0.0.0:5002` | `http://127.0.0.1:5002` | `http://<LAN-IP-2>:5002`（有线）/ `http://<LAN-IP-1>:5002`（无线） | `http://<Tailscale-IP>:5002` |
| 上传网关（可选） | `0.0.0.0:5010` | `http://127.0.0.1:5010` | 同上换端口 | 同上换端口 |
| vLLM 引擎（**别直接对外**） | `127.0.0.1:8080` | `http://127.0.0.1:8080/v1` | 需另加转发 | 需另加转发 |

三个地址在 §7 均实测返回 200。避开主服务 `:8888`（SGLang 主服务）、`:4000`（switchyard）。

> ⚠️ **ufw**：本机 ufw 为 active（规则文件 root-only，无法免密读取）。客户端连不上时先查 `sudo ufw status`，需要放行：
> ```bash
> sudo ufw allow from 192.168.1.0/24 to any port 5002 proto tcp
> sudo ufw allow from 100.64.0.0/10  to any port 5002 proto tcp   # Tailscale
> ```

### 1.1 调用链拆解：网络上只有 2 次 HTTP（一来一回）

最常被问的是"是不是要经过本机的 SDK 客户端、再进模型、再一路返回？"。**准确说法：中间那层是"服务端"不是"客户端"**，而且内部的多次调用都不出网卡。

```
你的 client 机器（笔记本 / 另一台服务器）
   │
   │  ① 1 次 HTTP POST —— 发的是【整个 PDF / 整张图】，不是切片
   ▼
┌──────────────── spark-XXXX ──────────────────────────────┐
│  :5010  上传网关（可选，只干一件事：把上传的东西存成文件） │
│   │  ①.5  本机 file:// 路径交给下一跳（不出网卡）          │
│   ▼                                                       │
│  :5002  glmocr SDK 服务进程   ← 中间层在这里（是服务端）   │
│   │  ② 版面分析 PP-DocLayoutV3（torch，CPU）找区域+阅读顺序│
│   │  ③ PDF 渲染成页 → 按区域裁剪成 N 张小图                │
│   │  ④ N 次 HTTP，走 localhost，并发  ───────┐             │
│   ▼                                          ▼             │
│  :8080  vLLM 引擎（GLM-OCR，GPU）   ⑤ 每个区域返回文本     │
│   └──────── ⑥ 拼成完整 markdown / JSON ◄─────┘             │
└───┼───────────────────────────────────────────────────────┘
    │  ⑦ 1 次 HTTP 响应：完整 markdown + json_result
    ▼
你的 client 机器
```

**关键点：网络上只有 ① 和 ⑦（一来一回）。②–⑥ 全在 Spark 内部走 localhost，永远不经过网线。**
英文样张内部对 vLLM 发 **7 次**请求（版面检出 7 个区域，每区域一次 VLM 调用，见部署文档 §10.8），3 页 PDF 约 **21 次** —— 这些若都放到网络上，就是"一页切十几块、10 页上百张切片频繁往返"。

| 环节 | 在哪台机器 | 干什么 | 客户端要装吗 |
|---|---|---|---|
| 发请求 | **客户端** | 发 1 个文件 | **不用装任何东西** |
| PDF 渲染（200 DPI） | Spark `:5002` | PyMuPDF 拆页 | 不用 |
| 版面分析 | Spark `:5002` | PP-DocLayoutV3 找区域 + 阅读顺序 | 不用（**要 torch，所以必须放 Spark**） |
| 区域裁剪 + 并发调度 | Spark `:5002` | 切 N 块、并发打 vLLM | 不用 |
| VLM 推理 | Spark `:8080` | 逐区域识别 | 不用 |
| 拼装 markdown/JSON | Spark `:5002` | 合成一个结果 | 不用 |

**"SDK 客户端"的两种含义**（混淆的根源）：

- **glmocr 当库装在客户端**（§4/§5 里的 `GlmOcr(mode="maas", …)`）→ 只是帮你拼 HTTP 请求、解析响应的代码，跑在客户端自己的进程里，**不增加任何跳数**；
- **glmocr 当服务跑在 Spark 上**（`python -m glmocr.server`）→ 这才是中间层，也是**唯一**该放 Spark 的东西。

所以 **客户端装不装 glmocr，链路拓扑不变**：装了只是省得自己拼 base64 / 解析返回，不装就直接 `curl`。

**能不能少一跳？** 可以 —— 直连 vLLM（§6，实测 4.26 s），代价是没有版面分析、PDF 多页、阅读顺序、markdown 拼装，等于把 SDK 的活搬到客户端重写一遍还要装 torch。只在"单张图整页 OCR"时划算。

**反过来放（SDK 装客户端）为什么不行**：网络上行变成几十上百次切片往返，且每台客户端都要装 torch + 版面模型（部署文档 §5 已论证）—— 只有客户端与 Spark 同机、或高带宽直连的专用工作站才考虑。

---

## 2. 服务端启动（客户端接入的前提）

### 2.1 一次性前置：补装 `[server]` extra

本机 `venv-vllm` 原先只装了 `glmocr[selfhosted]`（部署文档 §10.8），**没有 flask**，直接起服务会报：
`Error: Flask server support requires the optional server extra.` ✅实测

```bash
cd /home/<user>/projects && ./venv-vllm/bin/pip install "glmocr[server]"
# dry-run 实测只新增 4 个包：Flask-3.1.3 / Werkzeug / blinker / itsdangerous
# 不触碰 torch / transformers / vllm —— 与 §10.8 的安装纪律一致
```

### 2.2 一键启停（本文新增）

```bash
bash /home/<user>/projects/glm-ocr-test/start-service.sh --with-gateway  # 起 vLLM + SDK 服务 + 网关
bash /home/<user>/projects/glm-ocr-test/stop-service.sh                  # 全停，归还内存
```

- 启动脚本会**逐级探活**（`/v1/models` → `/health`），已就绪的组件自动跳过；✅实测编排正确。
- 配置文件：[glm-ocr-test/glmocr-server.yaml](../../glm-ocr-test/glmocr-server.yaml)（基于 SDK 默认 config.yaml 改 3 处：`maas.enabled=false`、`layout.device=cpu`、`max_workers=8`）。
- 日志：`glm-ocr-test/{vllm-serve,glmocr-server,gateway}.log`。

> ⚠️ **两个启动顺序陷阱**（均已实测）：
> 1. **vLLM 必须先起**。SDK 服务启动时会向 `ocr_api` 发一次探活请求，连不上就 **30 秒后自杀**（`ocr_client.connect()`，`ocr_client.py:175-235`）。实测日志：`Failed to connect to API server at http://127.0.0.1:8080/v1/chat/completions within 30 seconds`。
> 2. **沙箱内起不了 vLLM**：宿主机进程跑 vLLM 需要 GPU/NVML 直通权限（部署文档 §10.5 坑 3，本次再次复现 `Can't initialize NVML` /`Failed to infer device type`）。正常登录 shell 里跑没这个问题。

### 2.3 内存实测

| 时点 | available |
|---|---|
| 基线（仅主服务 :8888） | 25.0 GB |
| vLLM 起来后 | 17.7 GB |
| SDK 服务起来后（版面已加载） | 14.9 GB |
| `stop-service.sh` 之后 | **26.3 GB** |

glmocr 服务进程 RSS 约 **3.3 GB**；vLLM GPU 侧约 3.57 GiB。全栈净占用约 **10 GB**，与部署文档 §10.3 的结论一致。

---

## 3. 【接法 A · 推荐】HTTP 直连，客户端零依赖

任何语言、任何机器都能调，不需要装 glmocr / torch。

### 3.1 接口契约

| 项 | 值 |
|---|---|
| 端点 | `POST /glmocr/parse`（探活：`GET /health` → `{"status":"ok"}`） |
| `Content-Type` | **必须精确等于 `application/json`** ⚠️ 与源码 `server.py:91` 的字符串相等判断一致 |
| 请求体（SDK 原生） | `{"images": ["<url>", ...]}` —— 也可以是单个字符串（实测同样 200 ✅） |
| 请求体（MaaS 兼容） | `{"file": "<url>", "model": "glm-ocr"}` |
| 响应 | `json_result` / `markdown_result` + MaaS 别名 `layout_details` / `md_results` / `data_info` / `usage` / `model` / `id` / `created` |
| 多单元语义 | `images` **数组 = 同一文档的多页**，`markdown_result` 用 `"\n\n---\n\n"` 拼接，`json_result` 变成按单元的数组（实测 3 页 PDF → 3 个单元） |

**`images` 里到底能放什么**（逐项实测，⚠️ 与 `server.py` docstring 声称的 "http/https/file/data" **不符**）：

| 取值 | 结果 |
|---|---|
| `data:image/png;base64,...` / jpeg | ✅ 唯一推荐用法的**图片**路径 |
| `file:///abs/path.png` / `.pdf` | ✅ 可用，但**路径必须在服务端本机存在**（含 PDF 多页，走 PyMuPDF 按 `pdf_dpi=200` 渲染） |
| 服务端本地绝对路径（不带 `file://`） | ✅ 可用（相对路径按服务端 CWD 解析，客户端别用） |
| `data:application/pdf;base64,...` | ❌ **200 + 空结果**（见 §4.3 根因） |
| `http(s)://...` | ❌ **200 + 空结果**：`Invalid image source`，**代码里根本没有 http 抓取分支** |

### 3.2 客户端代码

**curl（图片）**
```bash
B64=$(base64 -w0 invoice.png)
curl -s http://<LAN-IP-2>:5002/glmocr/parse \
  -H 'Content-Type: application/json' \
  -d "{\"images\":[\"data:image/png;base64,$B64\"]}" | jq -r .markdown_result
```

**Python（`requests`，多页 = 多张图）**
```python
import base64, requests
API = "http://<LAN-IP-2>:5002/glmocr/parse"

def enc(path, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()

r = requests.post(API, json={"images": [enc("page1.png"), enc("page2.jpg")]}, timeout=600)
r.raise_for_status()
data = r.json()
print(data["markdown_result"])
print(len(data["json_result"]), "个区域")     # 带 bbox_2d(0-1000 归一化) / label / content
```

**JS / TS（浏览器或 Node 18+）**
```ts
async function ocr(file: File, api = "http://<LAN-IP-2>:5002/glmocr/parse") {
  const b64 = await new Promise<string>((res) => {
    const fr = new FileReader();
    fr.onload = () => res((fr.result as string).split(",")[1]);
    fr.readAsDataURL(file);
  });
  const r = await fetch(api, {
    method: "POST",
    headers: { "Content-Type": "application/json" },   // 不能加 charset
    body: JSON.stringify({ images: [`data:${file.type};base64,${b64}`] }),
    signal: AbortSignal.timeout(600_000),               // 单页 3-10 s，务必给足超时
  });
  if (!r.ok) throw new Error(await r.text());
  return (await r.json()) as { markdown_result: string; json_result: any[] };
}
```

> **客户端必读三条**：① 超时给到 **300–600 s**（官方集成测试默认 300 s）；② base64 体积约 **+33%**（实测 49 KB PNG → 64 KiB 请求体）；③ 拿到 200 **不代表成功**——务必检查 `markdown_result`/`json_result` 非空（见 §8 坑 1）。

---

## 4. 【接法 B】官方 SDK：把 MaaS 模式指向本机服务

官方 `examples/self-host/README.md` 就是这么设计的：**客户端只装基础包，复用 MaaS 模式把 `api_url` 指到自建服务**。

```bash
pip install glmocr        # 基础包即可，无 torch / 无版面模型
```
```python
from glmocr import GlmOcr
# api_key 随便填：自建服务不校验（server.py 里没有任何鉴权代码）
with GlmOcr(mode="maas", api_url="http://<LAN-IP-2>:5002/glmocr/parse",
            api_key="any-string", timeout=600) as parser:
    result = parser.parse("document.png")
    print(result.markdown_result)
```

### 4.1 ✅ 图片：实测通过
`client-sdk-test.py` 实测 PNG → **4.74 s / 603 字**，与 HTTP 直连等价。

### 4.2 客户端会发什么
`MaaSClient._prepare_file()`（`maas_client.py:209-299`）把本地文件转成 data URI 后 POST `{"model": "glm-ocr", "file": <dataURI>}`，头是 `Authorization: Bearer <api_key>`：
- 图片（JPEG/PNG）→ `data:image/png;base64,...`
- 非 JPEG/PNG 图 → 客户端先用 PIL 转成 PNG（超 10 MB 转 JPEG）
- **PDF → `data:application/pdf;base64,...`** ← 问题就出在这

### 4.3 ❌ PDF：实测**静默返回空**（本文最重要的发现）

| 实测 | 结果 |
|---|---|
| `GlmOcr(mode="maas", api_url=…).parse("multi.pdf")` | **0.03 s 返回，md_len = 0**，不抛异常、HTTP 200 |
| 手工 POST `{"images":["data:application/pdf;base64,…"]}` | **200，`json_result: []`，`markdown_result: ""`** |

**根因链**（已用单元探针 `page-loader-probe.py` 逐段证实）：
```
客户端 _prepare_file(): PDF → "data:application/pdf;base64,…"
  → POST {"file": …} → server.py 塞进 messages[].image_url.url
    → pipeline.extract_image_sources() 原样取出
      → PageLoader._load_source()  : 不是 bytes、不是 file://、os.path.isfile()=False
        → PageLoader._load_image() : 只判 source.startswith("data:image")  ← PDF 在这里掉队
          → ValueError: Invalid image source
            → iter_pages_with_unit_indices() 里 except 吞掉，只留一条 warning
              → 0 个 unit → _build_response(None, "") → HTTP 200 + 空  ❌
```
单元探针原始输出：
```
OK    data:image/png;base64,<49034B>        -> 1 page(s)
FAIL  data:application/pdf;base64,<167263B> -> RuntimeError: Error loading image 'data:application/pdf;base64,JVBERi0xLjQK…
FAIL  https://example.com/a.png             -> RuntimeError: Invalid image source
OK    file:///home/<user>/…/multi.pdf          -> 3 page(s)
OK    file:///home/<user>/…/testdoc.png        -> 1 page(s)
```
> 附带发现：这条 warning 会把**整个 base64 塞进日志**（一条几百 KB～几 MB 的日志行）。生产环境建议屏蔽该 logger 或改用网关。
> 结论：**图片用接法 B 没问题；PDF 必须用接法 A+客户端渲染，或接法 C。**

---

## 5. 【接法 C】上传网关：客户端一个 multipart 搞定 PDF

内置服务做不了"客户端上传 PDF"，加一个 ~70 行的适配层即可（**新增文件，未改动 SDK**）：
[`/home/<user>/projects/glm-ocr-upload-gateway.py`](../../glm-ocr-upload-gateway.py)（FastAPI + uvicorn + httpx —— 本机 vLLM 已带这些依赖，**零新增安装**）

原理：客户端给的东西先落盘成服务端临时文件 → 转成 `file://` → 交给已验证的 `:5002` 上游 → 用完即删。

**两个端点，覆盖所有客户端形态**：

| 端点 | 输入 | 适合谁 |
|---|---|---|
| `POST /parse` | multipart 文件上传 | curl / JS / Go / 任何语言；**推荐** |
| `POST /glmocr/parse` | 官方 SDK/MaaS 的 JSON 契约（`{"images":[…]}` 或 `{"file":…}`） | **官方 Python SDK 客户端零改动**（只把 `api_url` 指过来） |

`/glmocr/parse` 内部做的事：`data:image:*` **原样透传**（上游本来就支持，不浪费磁盘），**非图片的 `data:` URI（如 `data:application/pdf`）自动解码落盘成 `file://…pdf`** —— 正好补上 §4.3 那个断点；落盘文件名按 `%PDF-` magic 判定后缀，因为上游靠 `.pdf` 后缀走 PyMuPDF 多页渲染。

```bash
# ① multipart（任意语言，推荐）
curl -F "file=@doc.pdf" http://<LAN-IP-2>:5010/parse                     # {markdown_result, json_result, elapsed_s}
curl -F "file=@doc.pdf" "http://<LAN-IP-2>:5010/parse?format=markdown"   # 只要纯 markdown

# ② 官方 JSON 契约（客户端自己 base64；大文件务必用 --data-binary @file，别塞进命令行）
curl -X POST http://<LAN-IP-2>:5010/glmocr/parse -H 'Content-Type: application/json' \
     --data-binary @body.json        # body.json = {"file":"data:application/pdf;base64,..."}
```

```python
# ③ 官方 Python SDK：只改 api_url 指向网关，PDF 就能用了（§4.3 的坑被网关吃掉）
from glmocr import GlmOcr
with GlmOcr(mode="maas", api_url="http://<LAN-IP-2>:5010/glmocr/parse",
            api_key="any-string", timeout=600) as parser:
    print(parser.parse("document.pdf").markdown_result)   # ✅ 不再静默为空
```

**验证结果**

| 项 | 结果 |
|---|---|
| multipart 上传 PNG（49 KB）→ **真模型** | **5.85 s**，603 字 ✅ |
| multipart 上传 PDF（167 KB / 3 页）→ **真模型** | **9.52 s**，1367 字 / 3 个单元 ✅ ← 内置服务做不到的那条路 |
| `?format=markdown` | 纯文本直出 ✅ |
| 临时文件 | 请求结束即清理（`uploads/` 恒为空）✅ |
| `data:application/pdf` 转发给上游的形态（桩验证） | `file:///…/xxx.pdf`、`exists=True`、`size=167263`、magic `%PDF-` ✅ |
| `data:image/png` 转发形态（桩验证） | 原样透传、不落盘 ✅ |
| 官方 Python SDK 客户端 → 网关（桩验证） | PNG / PDF **均无错返回**（`error=None`），PDF 确实以 `.pdf` 临时文件形态到达上游 ✅ |
| 错误分支 | 空 `images` → 400；非法 JSON → 400 ✅ |

> **关于"桩验证"**：带此行标注的三项，用**记录型上游桩**（[glm-ocr-test/_recording_stub.py](../../glm-ocr-test/_recording_stub.py)）替换 vLLM，目的是在不重启模型的前提下检查"网关到底把什么转给了上游"和"客户端能否正确解析响应"。桩的响应字段与 `server.py` 的 `_build_response` 完全一致（`json_result`/`layout_details` 是 **list[page] → list[region]** 的嵌套形状——**客户端按这个形状解析，写成扁平列表会报 `'str' object has no attribute 'get'`**）。真实 OCR 文本质量由上面两行真模型结果背书。

> 踩坑：`curl -d "{\"file\":\"data:application/pdf;base64,<223KB>\"}"` 会报 `参数列表过长`（ARG_MAX），**大 payload 必须走 `--data-binary @file` 或 multipart** —— 这也是推荐 multipart 的原因之一（还能省掉 base64 的 +33%）。

**接法 A+ 与接法 C 的取舍**：客户端逐页渲染（A+）**零服务端改动**，但 3 页 PDF 要上行 3 张 PNG（约 6 MB，比原始 PDF 大一个量级）；网关（C）只上行 1 个原始 PDF，渲染放服务端 —— **批量/大 PDF 用 C，零改动场景用 A+**。

---

## 6. 【接法 D】直连 vLLM（绕过 SDK 与版面）

适合"客户端自己会切块"或只想做**单图整页 OCR**的场景。请求就是标准 OpenAI 接口：

```bash
curl -s http://<vllm-host>:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-ocr",
  "messages": [{"role":"user","content":[
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
      {"type":"text","text":"Text Recognition:"}]}],
  "max_tokens": 2048, "temperature": 0.0}'
```

| 实测（本机 127.0.0.1:8080） | 结果 |
|---|---|
| `Text Recognition:` | 4.26 s / 439 字，正文准确 ✅ |
| `Table Recognition:` | 2.90 s / 296 字，输出 HTML 表格 ✅ |

- **只有 3 个官方 prompt**：`Text Recognition:` / `Table Recognition:` / `Formula Recognition:`（HF 模型卡 + SDK `config.yaml` 一致）。**不要用 "Figure Recognition:"**（查无此物，SDK 对 chart/image 标签是 `skip`）。
- **图片必须排在文本 prompt 之前**（SDK 自己的构造顺序也是先图后文）。
- 代价：**没有版面分析**，整图当一块识别，长文档/多栏排版会掉结构；且 vLLM 目前只绑 `127.0.0.1`，对外要么改成 `--host 0.0.0.0`，要么照 `vllm-open-proxy.py` 的路子加一层 TCP 转发（**不建议**：裸模型无鉴权、且等于把 §8 的所有坑丢给客户端）。

---

## 7. 实测数据总表（2026-09-15 真机）

配置：vLLM AR + `max-model-len 16384`；SDK `layout.device=cpu`、`max_workers=8`；版面 PP-DocLayoutV3。

| 场景 | 耗时 | 结果 |
|---|---|---|
| `GET /health`（三个地址） | <10 ms | 200 ✅ |
| 单张英文 PNG 49 KB → `data:image/png` | **4.86 s** | 200 / 603 字 ✅ |
| 单张中文 PNG → `data:image/png` | **2.58 s** | 200 / 157 字 ✅ |
| MaaS 格式 `{"file": data:image/png}` | 4.66 s | 200 / 603 字 ✅ |
| `images` 传**字符串**而非数组 | — | 200 / 603 字 ✅ |
| 3 页 PDF → `file://`（服务端本地） | 8.25 s | 200 / 1367 字 ✅ |
| 3 页 PDF → 客户端渲染成 3 张 `data:image/png` | 9.22 s | 200 / 1377 字 / 3 单元 ✅ |
| **3 页 PDF → `data:application/pdf`** | 0.01 s | ❌ 200 但空 |
| **`https://…png`** | 0.03 s | ❌ 200 但空 |
| 空 `images` | — | 400 `{"error":"No images provided"}` |
| `Content-Type: application/json; charset=utf-8` | — | **400** |
| 3 并发同图 | 各 4.7–4.8 s | 全部 200、结果正确（=单发延迟，未排队） |
| 8 并发同图 | 各 25.7 s（wall 26.0 s） | 全部 200、结果正确（吞吐↑ / 单发延迟↑） |
| 官方 SDK MaaS 模式 · PNG | 4.74 s | ✅ 603 字 |
| **官方 SDK MaaS 模式 · PDF** | 0.03 s | ❌ **静默 md_len=0** |
| 上传网关 · PNG | 5.85 s | ✅ 603 字 |
| 上传网关 · PDF | 9.52 s | ✅ 1367 字 |
| 直连 vLLM · `Text Recognition:` | 4.26 s | ✅ 439 字 |
| 直连 vLLM · `Table Recognition:` | 2.90 s | ✅ HTML 表格 |

并发结论：3 路并发与单发延迟持平；8 路并发（本机用小 KV cache + `max_workers=8`）会排队到 ~26 s。**结果在 8 路并发下依然正确**（同一 Pipeline 实例被多线程复用未见串扰），但更重的并发未验证，建议客户端侧自己限流（如每客户端 ≤ 3 并发）。

---

## 8. 坑清单（客户端最容易踩的）

| # | 现象 | 根因 | 对策 |
|---|---|---|---|
| 1 | **HTTP 200 但 `markdown_result` 为空** | `images` 不是 `data:image`/`file://` 时，异常被 `iter_pages_with_unit_indices` 静默吞掉，服务端仍返回 200 | 客户端**必须校验结果非空**；服务端日志看 `Skipping source (unit N)` |
| 2 | **PDF 客户端一传就"成功"却啥也没有** | 官方 SDK 把 PDF 包成 `data:application/pdf`，而 `_load_image` 只认 `data:image` | 逐页渲染（A+）或用上传网关（C） |
| 3 | `Invalid Content-Type` 400 | 源码是**字符串全等**比较，带 `charset` 也挂 | 头只写 `application/json` |
| 4 | 客户端请求超时 | 单页 3–5 s、PDF 每页累加，冷启动首次含版面模型加载（+3 s） | 超时设 300–600 s，别用默认 30 s |
| 5 | SDK 服务起不来 | ① 没装 `[server]`（flask 缺失）；② vLLM 没先起（30 s 探活失败自杀）；③ YAML 里只写 `pipeline.layout` 的部分键会**覆盖整节**导致 `model_dir is required` | 见 §2 |
| 6 | 沙箱里 vLLM 起不来 | `Can't initialize NVML` / `Failed to infer device type` | 用正常 shell 跑（§10.5 坑 3） |
| 7 | 中文百分号变 LaTeX | `22.1%` → `$22.1\%$`、千分位逗号被全角化 | 客户端后处理（部署文档 §10.8 已记录） |
| 8 | 日志被 base64 刷爆 | 失败 warning 内嵌整个 data URI | 用网关侧落盘路径，别把大 base64 直接喂内置服务 |
| 9 | 服务"停成功了"但端口还在听 | 停止脚本里 `pkill -f` 模式写错时**只返回非 0、不报错** | 停服后回读 `ss -ltn`，别只信 pkill 的退出码 |
| 10 | `curl -d` 塞大 base64 报 **参数列表过长** | 命令行参数超 ARG_MAX（223 KB 就够触发） | 用 `--data-binary @file` 或 multipart |
| 11 | 自建包装服务返回的 `json_result` 用扁平列表，客户端报 `'str' object has no attribute 'get'` | 客户端按 **list[page] → list[region]** 嵌套解析 | 照 `server.py` 的 `_build_response` 形状返回 |

---

## 9. 安全与运维

| 项 | 现状 / 建议 |
|---|---|
| **鉴权** | ⚠️ **完全没有**。`server.py` 无任何鉴权代码，`Authorization` 头被忽略（官方 README 明说 `api_key: any-string`）。局域网内知道 IP 就能用。 |
| 暴露面 | 默认 `0.0.0.0:5002`，实测局域网双 IP + Tailscale 全部 200。**若不希望被扫到**：改成只绑 Tailscale IP，或 ufw 只放行指定网段。 |
| DoS 面 | `/glmocr/parse` 是 CPU/GPU 重活，任意人可并发打满（8 并发已能把单发延迟推到 26 s）。要长期对外，建议在网关层加 token + 客户端限流。 |
| **跑完即停** | 呼应部署文档 §7 与 judge `:8889` 的教训（16 天无调用即关停）：**没有真实调用方就不要常驻**。全栈常占约 10 GB 统一内存。 |
| 与主服务共存 | vLLM 用 `--gpu-memory-utilization 0.05`；先确认 `:8888` 正常响应再起 OCR；两者不要同时冷启动（部署文档 §8）。 |
| 超时/重试 | SDK 服务侧 `ocr_api.request_timeout=120`、`retry_max_attempts=2`（对 429/5xx 重试）；客户端侧另设 300–600 s。 |

---

## 10. 复现脚本与产物清单

| 文件 | 用途 |
|---|---|
| [glm-ocr-test/start-service.sh](../../glm-ocr-test/start-service.sh) | 一键起 vLLM + SDK 服务（+ `--with-gateway` 带上网关），逐级探活 |
| [glm-ocr-test/stop-service.sh](../../glm-ocr-test/stop-service.sh) | 一键停全栈并打印内存 |
| [glm-ocr-test/glmocr-server.yaml](../../glm-ocr-test/glmocr-server.yaml) | SDK 服务端配置（maas 关、版面 cpu、max_workers 8） |
| [glm-ocr-upload-gateway.py](../../glm-ocr-upload-gateway.py) | 上传网关（multipart → `file://`），FastAPI |
| [glm-ocr-test/client-http-test.py](../../glm-ocr-test/client-http-test.py) | 客户端 HTTP 直连 7 项验证（健康检查 / 图片 / 中文 / MaaS 格式 / PDF 三种路径 / 错误分支） |
| [glm-ocr-test/client-sdk-test.py](../../glm-ocr-test/client-sdk-test.py) | 官方 SDK MaaS 模式验证（PNG ✅ / PDF ❌） |
| [glm-ocr-test/client-path-tests.py](../../glm-ocr-test/client-path-tests.py) | http URL、直连 vLLM、客户端逐页渲染 PDF 三条路径 |
| [glm-ocr-test/page-loader-probe.py](../../glm-ocr-test/page-loader-probe.py) | 单元探针：证明 `data:application/pdf` 与 `http(s)` 不被支持（根因证据） |
| [glm-ocr-test/client-sdk-via-gateway-test.py](../../glm-ocr-test/client-sdk-via-gateway-test.py) | 官方 Python SDK 客户端 → 网关：验证 PNG/PDF 两条路都能被客户端正确解析 |
| [glm-ocr-test/_recording_stub.py](../../glm-ocr-test/_recording_stub.py) | 记录型上游桩：记录网关转发给上游的真实载荷（含 `file://` 路径存活性与 magic） |

---

## 11. 附录：源码证据索引

| 结论 | 证据位置（`venv-vllm/lib/python3.12/site-packages/glmocr/`） |
|---|---|
| 端点与请求体、MaaS 兼容字段 | `server.py:75-153`（`/glmocr/parse`）、`server.py:34-48`（`_build_response`）、`server.py:155-158`（`/health`） |
| Content-Type 字符串全等 | `server.py:91` |
| 只认 data:image / file:// / 本地路径 | `dataloader/page_loader.py:261-281`（`_load_image`）、`:239-259`（`_load_source`） |
| 无 http(s) 抓取分支 | `dataloader/page_loader.py` 与 `utils/image_utils.py` 全文无 `requests.get`/`urlopen` |
| 异常被吞 → 静默空结果 | `dataloader/page_loader.py:160-165`（`iter_pages_with_unit_indices` 的 `except`） |
| 客户端把 PDF 包成 data:application/pdf | `maas_client.py:259-262`（`_prepare_file`） |
| 客户端 MaaS 请求体与 Bearer 头 | `maas_client.py:400-403`、`:423-440` |
| SDK 服务启动探活（30 s 自杀） | `ocr_client.py:175-235`（`connect()`） |
| 发往 vLLM 的载荷与三个 prompt | `dataloader/page_loader.py:340-383`、`config.yaml` `task_prompt_mapping` |
| 默认监听 0.0.0.0:5002、server extra = flask | `config.yaml:7-10`、`glmocr-0.1.5.dist-info` |
| 官方自托管客户端说明 | GitHub `zai-org/GLM-OCR` → `examples/self-host/README.md` |

---

## 12. 终端 agent 直接调用：`glm-ocr-remote` skill

> 目标：在终端里对 agent 说"识别这张截图 / 把这个 PDF 转成文字"，而不是每次手写 curl。
> 参考实现：[huangjunxin/PaddleOCR-Skill](https://github.com/huangjunxin/PaddleOCR-Skill)（同款思路：自托管服务 + 零依赖脚本 + SKILL.md 触发词）。
> 本节产物已入仓：[`skills/glm-ocr-remote/`](../skills/glm-ocr-remote/) + [`skills/install.sh`](../skills/install.sh)。

### 12.1 装在哪、怎么被 agent 发现

DSH / Claude Code 等 agent 的 skill 发现根目录（按 rank 优先级）：

| Rank | 来源 | 路径 |
|---|---|---|
| 100 | 项目 | `<projectRoot>/.dsh/skills` |
| 200 | 项目 | `<projectRoot>/.agents/skills` |
| 400 | 用户 | `~/.dsh/skills` |
| 500 | 用户（**本 skill 装这里**） | `~/.agents/skills` |

- skill 必须是**被扫描根目录顶层的目录 bundle**（`<name>/SKILL.md`）；**不支持嵌套**的 `**/SKILL.md`
- 根目录有 watcher：新增 / 改名 / 改 frontmatter 会在**下一个模型步骤**进入会话目录，**不用重启**（本次实测：写完 SKILL.md 后会话目录立刻出现 `glm-ocr-remote`）
- 本机 `~/.agents/skills/wind-mcp-skill` 就是这套机制，新 skill 与它并列；`~/.claude/skills/` 放软链做 Claude Code 兼容

```bash
bash /home/<user>/projects/dgx-spark-notes/skills/install.sh   # 装到 ~/.agents/skills + 自动软链
```

### 12.2 目录结构

```
skills/glm-ocr-remote/
├── SKILL.md               # 主体：触发词 / 何时用不用 / 用法 / 故障排查
├── config.env.example     # 个人配置模板（复制为 config.env，install.sh 会保留它）
└── scripts/
    ├── _common.py         # 地址解析（--server > GLMOCR_URL > config.env > 默认）+ HTTP
    ├── ocr.py             # 图片 / PDF -> 文字与 Markdown（纯标准库）
    └── check_server.py    # 健康检查 + 起服务提示
```

### 12.3 设计要点（为什么这么做）

| 决策 | 原因 |
|---|---|
| **默认打网关 `:5010`，不是内置服务 `:5002`** | `:5002` 对 PDF 静默返回空（§4.3）；网关一个接口同时吃图片与 PDF |
| **脚本纯标准库（urllib）** | 客户端零依赖，agent 直接 `python3` 就能跑；只有 `--pages/--dpi` 才要 PyMuPDF |
| **主动做"空结果守卫"** | 内置服务的静默陷阱是这套方案最容易骗过 agent 的地方：脚本识别到"200 但内容为空"就给出**具体原因与改法**，并以退出码 2 结束 |
| **404/405 自动回退 JSON 契约** | 地址误指 `:5002` 时，图片仍能走 `{"images":["data:image/..."]}` 正常工作，只有 PDF 会命中上面的守卫 |
| `config.env` 三级覆盖 | 一台机器一次配置；团队/多机部署只改这一个文件 |
| **不提供 `--prompt`** | 版面分析会按区域类型自动选 `Text/Table/Formula Recognition:`，比手工指定更准 |
| 只暴露 `ocr.py` / `check_server.py` 两个入口 | SKILL.md 里的命令越少，agent 选错的概率越低 |

### 12.4 实测（真服务端到端）

| 场景 | 结果 |
|---|---|
| `check_server.py` | 网关 + 上游 SDK 服务 + `:5002` 三级全绿 ✅ |
| 英文 PNG（系统 python3，零依赖） | **4.9 s**，正文 + HTML 表格 + 公式全对 ✅ |
| 中文 PNG | 2.6 s ✅ |
| 3 页 PDF（服务端渲染） | **8.4 s**，三页按序拼在同一段 Markdown ✅ |
| `--json` | `mode=multipart`，检出 **7 个区域**，带 `bbox_2d`(0–1000) / `label` / `native_label` / `polygon` ✅ |
| `--pages 3`（客户端渲染） | 只返回第 3 页内容 ✅（需 PyMuPDF） |
| 地址误指 `:5002` + 图片 | 自动回退 JSON 契约，正常返回 ✅ |
| 地址误指 `:5002` + PDF | 命中空结果守卫，打印原因与改法，**退出码 2** ✅ |
| 服务不可达 | 打印启动命令与冷启动时长提示，**退出码 2** ✅ |
| `install.sh` 重复安装 | 保留已有 `config.env`，重建脚本权限，自动软链 ✅ |

### 12.5 最大的坑：SKILL.md 的 frontmatter 被 YAML 悄悄判死

`description:` 如果用**不加引号的普通标量**，正文里出现 `Trigger: `（冒号 + 空格）会直接破坏 YAML 解析：

```
yaml.scanner.ScannerError: mapping values are not allowed here
  in "<unicode string>", line 3, column 205:  ... 时使用。Trigger: OCR, 文字识别, ...
```

**后果最坑的地方在于它是静默的**：DSH 只是把该 skill 随警告丢弃，会话目录里根本看不到它，
报错只在宿主日志里 —— 表现为"明明写了 SKILL.md，agent 就是发现不了"。

**正解**：用折叠块标量 `>-`（本机 `wind-mcp-skill` 也是这么写的），或给整串加引号。
自查一行命令：
```bash
python3 -c "import yaml,pathlib;t=pathlib.Path('SKILL.md').read_text();print(yaml.safe_load(t.split('---',2)[1]))"
```

### 12.6 用法速查

```bash
S=~/.agents/skills/glm-ocr-remote/scripts
python3 $S/check_server.py                   # 先看服务在不在
python3 $S/ocr.py 截图.png                    # 图片
python3 $S/ocr.py a.png b.jpg --json          # 多张 + 结构化（含 bbox）
python3 $S/ocr.py 报告.pdf                    # PDF（服务端渲染，零依赖）
python3 $S/ocr.py 报告.pdf --pages 1-3 --dpi 300 -o out.txt   # 指定页（需 PyMuPDF）
python3 $S/ocr.py 图.png --server http://<LAN-IP-2>:5010    # 临时换服务地址
```

---

## 13. 一句话建议

**图片**：客户端直接 `POST /glmocr/parse` 发 `data:image/png;base64,…`，零依赖、2.6–4.9 s 出结果。
**PDF**：别把客户端直接指向内置服务——**会静默返回空**。把客户端指向**上传网关 `:5010`**：multipart 用 `-F "file=@doc.pdf"`，官方 Python SDK 只改 `api_url` 到 `http://<spark>:5010/glmocr/parse` 即可；不想加服务就在客户端逐页渲染成 PNG。
**运维**：无鉴权，别裸奔在公网；用完 `stop-service.sh`，把 10 GB 内存还给主服务。
