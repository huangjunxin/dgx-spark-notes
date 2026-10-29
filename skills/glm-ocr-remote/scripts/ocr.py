#!/usr/bin/env python3
"""GLM-OCR 远程识别：图片 / 截图 / 扫描件 / PDF -> 文字与 Markdown。

默认调用【上传网关】:5010/parse（multipart 上传）：图片与 PDF 同一个接法，
PDF 由服务端用 PyMuPDF 逐页渲染并在服务端跑版面分析（客户端零依赖）。

纯标准库；只有 --pages / --dpi（改为客户端渲染）才需要 PyMuPDF。

用法：
  python3 ocr.py 截图.png
  python3 ocr.py a.png b.jpg --json -o out.json
  python3 ocr.py 报告.pdf
  python3 ocr.py 报告.pdf --pages 1-3,7 --dpi 300 -o 报告.txt
"""
import argparse
import base64
import json
import mimetypes
import pathlib
import sys
import urllib.error
import urllib.request
import uuid

from _common import http_json, resolve_base, start_hint

EMPTY_HINT = """服务返回空结果（HTTP 200 但没有内容）。
  · 最常见原因：地址指向了【内置服务 :5002】而不是【上传网关 :5010】，而输入是 PDF。
    内置服务的 images 字段只认 data:image / file://，对 data:application/pdf
    与 http(s) URL 一律静默返回空（200 + 空结果，不报错）。
  · 处理：改用网关地址，例如
        python3 ocr.py --server http://127.0.0.1:5010 <文件>
  · 或确认输入文件本身可读、未损坏。"""


def post_multipart(base: str, filename: str, data: bytes, timeout: float):
    boundary = "----glmocr" + uuid.uuid4().hex
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        base + "/parse", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8", "replace"))


def ocr_bytes(base: str, filename: str, data: bytes, timeout: float):
    """返回 (markdown, json_result, elapsed_s, mode)。"""
    # 1) 首选：网关 multipart（图片 + PDF 通吃）
    try:
        _, body = post_multipart(base, filename, data, timeout)
        return (body.get("markdown_result") or "", body.get("json_result"),
                body.get("elapsed_s"), "multipart")
    except urllib.error.HTTPError as e:
        if e.code not in (404, 405):        # 404/405 => 该地址没有 /parse（不是网关）
            raise
    # 2) 回退：内置服务的 JSON 契约（仅图片有效；PDF 会落入“空结果”检查）
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    uri = f"data:{ctype};base64," + base64.b64encode(data).decode()
    status, body = http_json(base + "/glmocr/parse", {"images": [uri]}, timeout=timeout)
    if status != 200 or not isinstance(body, dict):
        raise RuntimeError(f"HTTP {status}: {body}")
    return (body.get("markdown_result") or "", body.get("json_result"), None, "json")


def parse_page_spec(spec: str, total: int):
    idx = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            idx += list(range(int(a), int(b) + 1))
        else:
            idx.append(int(part))
    return sorted({i - 1 for i in idx if 1 <= i <= total})


def render_pdf(path: pathlib.Path, spec, dpi: int):
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24（优先，避免旧 fitz 别名告警）
    except ImportError:
        try:
            import fitz
        except ImportError:
            sys.exit("错误：--pages / --dpi 需要 PyMuPDF（pip install PyMuPDF）。\n"
                     "不加这两个参数时 PDF 由服务端渲染，客户端无需任何依赖。")
    doc = fitz.open(path)
    pages = parse_page_spec(spec, doc.page_count) if spec else list(range(doc.page_count))
    return [(i + 1, doc[i].get_pixmap(dpi=dpi).tobytes("png")) for i in pages]


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-OCR 远程识别（图片 / PDF -> 文字与 Markdown）")
    ap.add_argument("files", nargs="+", help="图片或 PDF 路径（可多个）")
    ap.add_argument("--server", help="服务地址，默认 127.0.0.1:5010（网关）")
    ap.add_argument("--json", action="store_true", help="输出完整 JSON（含 bbox 区域）")
    ap.add_argument("-o", "--output", help="写入文件（默认 stdout）")
    ap.add_argument("--pages", help="PDF 页码范围，如 1-3,7（触发客户端渲染，需要 PyMuPDF）")
    ap.add_argument("--dpi", type=int, default=200, help="配合 --pages 的渲染 DPI（默认 200）")
    ap.add_argument("--timeout", type=float, default=600, help="单文件超时秒数（默认 600）")
    ap.add_argument("--quiet", action="store_true", help="多文件时不打印分隔标题")
    args = ap.parse_args()

    base = resolve_base(args.server)
    status, _ = http_json(f"{base}/health", timeout=8)
    if status != 200:
        print(start_hint(base), file=sys.stderr)
        return 2

    results, empty = [], False
    for f in args.files:
        path = pathlib.Path(f).expanduser()
        if not path.is_file():
            print(f"跳过（不是文件）：{f}", file=sys.stderr)
            continue

        if path.suffix.lower() == ".pdf" and (args.pages or args.dpi != 200):
            units = [(f"{path.stem}-p{n}.png", png, n) for n, png in render_pdf(path, args.pages, args.dpi)]
        else:
            units = [(path.name, path.read_bytes(), None)]

        for name, data, page in units:
            try:
                md, js, elapsed, mode = ocr_bytes(base, name, data, args.timeout)
            except Exception as e:                                  # noqa: BLE001
                print(f"✗ {path} 识别失败：{e}", file=sys.stderr)
                empty = True
                continue
            if not md.strip() and not js:
                empty = True
            results.append({"file": str(path), "page": page, "mode": mode,
                            "elapsed_s": elapsed, "markdown_result": md, "json_result": js})

    if args.json:
        out = json.dumps({"server": base, "results": results}, ensure_ascii=False, indent=2)
    else:
        chunks = []
        for r in results:
            md = r["markdown_result"] or ""
            if len(results) > 1 and not args.quiet:
                label = r["file"] + (f"（第 {r['page']} 页）" if r["page"] else "")
                chunks.append(f"===== {label} =====\n{md}")
            else:
                chunks.append(md)
        out = "\n\n".join(chunks)

    if args.output:
        pathlib.Path(args.output).write_text(out, encoding="utf-8")
        print(f"已写入 {args.output}（{len(out)} 字符）", file=sys.stderr)
    else:
        print(out)

    if empty:
        print("\n" + EMPTY_HINT, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
