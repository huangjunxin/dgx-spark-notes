
import sys, time
from glmocr import GlmOcr

URL = "http://<LAN-IP-2>:5002/glmocr/parse"
print("client SDK version test ->", URL)

with GlmOcr(mode="maas", api_url=URL, api_key="any-string", timeout=600) as p:
    t0 = time.perf_counter(); r = p.parse("/home/<user>/projects/glm-ocr-test/testdoc.png")
    md = r.markdown_result if hasattr(r, "markdown_result") else str(r)
    print(f"[A] PNG  -> {time.perf_counter()-t0:.2f}s  md_len={len(md)}  head={md[:80]!r}")
    t0 = time.perf_counter(); r = p.parse("/home/<user>/projects/glm-ocr-test/multi.pdf")
    md = r.markdown_result if hasattr(r, "markdown_result") else str(r)
    print(f"[B] PDF  -> {time.perf_counter()-t0:.2f}s  md_len={len(md)}  head={md[:80]!r}")
