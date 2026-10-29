"""记录型上游桩：镜像 server.py 的 _build_response 超集字段。

关键形状（源码实证）：json_result / layout_details 是 **list[page]，每个 page 是 list[region]**，
markdown 走 md_results；客户端 _maas_response_to_pipeline_result() 按此解析。
"""
import json, pathlib, http.server, time, uuid

class H(http.server.BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"status": "ok"})

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(n) or b'{}')
        pages = []
        for u in body.get('images', []):
            if isinstance(u, str) and u.startswith('file://'):
                p = pathlib.Path(u[7:])
                head = p.read_bytes()[:5] if p.exists() else b''
                desc = {"kind": "file", "path": str(p), "exists": p.exists(),
                        "size": p.stat().st_size if p.exists() else 0,
                        "magic": head.decode('latin1')}
            else:
                desc = {"kind": "passthrough", "prefix": str(u)[:32]}
            pages.append([{"index": 0, "label": "text", "content": json.dumps(desc), "bbox_2d": [0, 0, 10, 10]}])
        md = "\n\n---\n\n".join(json.dumps(p[0]["content"]) for p in pages)
        self._send({"json_result": pages, "markdown_result": md,
                    "layout_details": pages, "md_results": md,
                    "data_info": {"pages": []}, "usage": {}, "model": "glm-ocr",
                    "id": f"chatcmpl-{uuid.uuid4().hex[:29]}", "created": int(time.time())})

    def log_message(self, *a): pass

http.server.ThreadingHTTPServer(('127.0.0.1', 5099), H).serve_forever()
