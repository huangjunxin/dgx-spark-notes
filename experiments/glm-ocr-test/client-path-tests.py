
import base64, io, json, pathlib, time, urllib.request, urllib.error
HERE = pathlib.Path('/home/<user>/projects/glm-ocr-test')

def post(url, payload, timeout=600, ctype='application/json'):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': ctype}, method='POST')
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.perf_counter()-t0
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300], time.perf_counter()-t0

# --- C. http(s) URL in "images"（server.py docstring 声称支持）---
pdf = HERE/'multi.pdf'
st, body, dt = post('http://<LAN-IP-2>:5002/glmocr/parse',
                    {'images': ['https://www.baidu.com/img/flexible/logo/pc/result.png']})
print(f"[C] images=[https://...]  -> {st}  {dt:.2f}s  json_result={body.get('json_result') if isinstance(body,dict) else body!r}")

# --- D. 绕过 SDK，直接打 vLLM 的 OpenAI 接口（无版面，单图）---
from PIL import Image
img = Image.open(HERE/'testdoc.png').convert('RGB')
buf = io.BytesIO(); img.save(buf, format='JPEG', quality=95)
b64 = base64.b64encode(buf.getvalue()).decode()
for prompt in ['Text Recognition:', 'Table Recognition:']:
    st, body, dt = post('http://127.0.0.1:8080/v1/chat/completions', {
        'model': 'glm-ocr',
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
            {'type': 'text', 'text': prompt}]}],
        'max_tokens': 2048, 'temperature': 0.0})
    txt = body['choices'][0]['message']['content'] if isinstance(body, dict) and 'choices' in body else str(body)[:200]
    print(f"[D] 直连 vLLM prompt={prompt!r} -> {st} {dt:.2f}s len={len(txt)} head={txt[:90]!r}")

# --- E. 客户端侧把 PDF 渲染成逐页 PNG，再按页发 data: URL ---
import pypdfium2 as pdfium
pdfdoc = pdfium.PdfDocument(str(pdf))
pages = [p.render(scale=200/72).to_pil().convert('RGB') for p in pdfdoc]
print(f"[E] 客户端渲染 multi.pdf -> {len(pages)} 页")
uris = []
for p in pages:
    b = io.BytesIO(); p.save(b, format='PNG')
    uris.append('data:image/png;base64,' + base64.b64encode(b.getvalue()).decode())
st, body, dt = post('http://<LAN-IP-2>:5002/glmocr/parse', {'images': uris})
md = body.get('markdown_result', '') if isinstance(body, dict) else str(body)
print(f"[E] 逐页 POST -> {st}  {dt:.2f}s  总长={len(md)}  页分隔符={'---' in md}  head={md[:100]!r}")
print(f"    JSON 结果类型: {type(body.get('json_result')).__name__}, 单元数={len(body['json_result']) if isinstance(body.get('json_result'), list) else 'n/a'}")
