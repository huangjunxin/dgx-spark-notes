#!/usr/bin/env python3
"""客户端侧验证：只走 HTTP，不依赖 torch/版面模型。

模拟"另一台机器上的客户端"如何调用 spark-XXXX 上的 GLM-OCR 服务。
"""
import base64, json, sys, time, urllib.request, urllib.error, pathlib

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5002"
HERE = pathlib.Path("/home/<user>/projects/glm-ocr-test")


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode())
            return r.status, body, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:400], time.perf_counter() - t0


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def data_uri(p: pathlib.Path) -> str:
    mime = "application/pdf" if p.suffix.lower() == ".pdf" else f"image/{p.suffix[1:].lower()}"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


print(f"### target = {BASE}")
st, body = get("/health")
print(f"[1] GET /health -> {st} {body}")

# --- 2. 单张 PNG（data: URL，官方推荐的自托管客户端路径）
png = HERE / "testdoc.png"
uri = data_uri(png)
st, body, dt = post("/glmocr/parse", {"images": [uri]})
md = body.get("markdown_result", "") if isinstance(body, dict) else str(body)
print(f"[2] POST /glmocr/parse  images=[data:image/png]  {st}  {dt:.2f}s  "
      f"payload={len(uri)/1024:.0f} KiB  md_len={len(md)}")
print("    md[:160]:", repr(md[:160]))

# --- 3. 中文图
png_zh = HERE / "testdoc_zh.png"
st, body, dt = post("/glmocr/parse", {"images": [data_uri(png_zh)]})
md = body.get("markdown_result", "") if isinstance(body, dict) else str(body)
print(f"[3] POST 中文图 data:URL  {st}  {dt:.2f}s  md_len={len(md)}")
print("    md[:160]:", repr(md[:160]))

# --- 4. MaaS 兼容格式：{"file": data-uri}
st, body, dt = post("/glmocr/parse", {"file": uri, "model": "glm-ocr"})
md = body.get("markdown_result", "") if isinstance(body, dict) else str(body)
print(f"[4] POST MaaS 格式 {{'file': data:image/png}}  {st}  {dt:.2f}s  md_len={len(md)}")

# --- 5. PDF 通过 data:application/pdf（官方 README 宣称的客户端 PDF 路径）
pdf = HERE / "multi.pdf"
st, body, dt = post("/glmocr/parse", {"images": [data_uri(pdf)]})
md = body.get("markdown_result", "") if isinstance(body, dict) else str(body)
print(f"[5] POST PDF as data:application/pdf  {st}  {dt:.2f}s  "
      f"json_result={body.get('json_result') if isinstance(body,dict) else '?'!r}  md_len={len(md)}")

# --- 6. PDF 以服务端 file:// 路径（文件必须在服务器上）
st, body, dt = post("/glmocr/parse", {"images": [f"file://{pdf}"]})
md = body.get("markdown_result", "") if isinstance(body, dict) else str(body)
print(f"[6] POST PDF via file:// (服务端本地路径)  {st}  {dt:.2f}s  md_len={len(md)}")
print("    md[:200]:", repr(md[:200]))

# --- 7. 错误路径
st, body, dt = post("/glmocr/parse", {"images": []})
print(f"[7] 空 images -> {st}  {str(body)[:80]}")
