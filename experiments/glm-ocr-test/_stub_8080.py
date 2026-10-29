import http.server, json
class H(http.server.BaseHTTPRequestHandler):
    def _ok(self):
        body = json.dumps({"id": "stub", "object": "chat.completion", "created": 0,
                           "model": "glm-ocr",
                           "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                        "finish_reason": "stop"}]}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    do_GET = _ok
    do_POST = _ok
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', 8080), H).serve_forever()
