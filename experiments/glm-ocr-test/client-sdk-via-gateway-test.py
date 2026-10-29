"""官方 Python SDK 客户端 → 上传网关 → 上游：验证 PDF 路径闭环。"""
from glmocr import GlmOcr

URL = "http://127.0.0.1:5011/glmocr/parse"      # 客户端指向【网关】，而不是内置服务
for name, path in [("PNG", "/home/<user>/projects/glm-ocr-test/testdoc.png"),
                   ("PDF", "/home/<user>/projects/glm-ocr-test/multi.pdf")]:
    with GlmOcr(mode="maas", api_url=URL, api_key="any-string", timeout=300) as p:
        r = p.parse(path)
    md = r.markdown_result or ""
    err = getattr(r, "_error", None)
    print(f"  [{name}] md_len={len(md)} error={err}")
    print(f"        {md[:150]}")
