
import base64, pathlib, traceback
from glmocr.config import load_config
from glmocr.dataloader.page_loader import PageLoader

cfg = load_config()
pl = PageLoader(cfg.pipeline.page_loader)

png = pathlib.Path('/home/<user>/projects/glm-ocr-test/testdoc.png').read_bytes()
pdf = pathlib.Path('/home/<user>/projects/glm-ocr-test/multi.pdf').read_bytes()

cases = [
    ('data:image/png;base64,<' + str(len(png)) + 'B>', 'data:image/png;base64,' + base64.b64encode(png).decode()),
    ('data:application/pdf;base64,<' + str(len(pdf)) + 'B>', 'data:application/pdf;base64,' + base64.b64encode(pdf).decode()),
    ('https://example.com/a.png', 'https://example.com/a.png'),
    ('file:///home/<user>/projects/glm-ocr-test/multi.pdf', 'file:///home/<user>/projects/glm-ocr-test/multi.pdf'),
    ('file:///home/<user>/projects/glm-ocr-test/testdoc.png', 'file:///home/<user>/projects/glm-ocr-test/testdoc.png'),
]
for label, src in cases:
    try:
        pages = pl._load_source(src)
        print(f"  OK    {label:55s} -> {len(pages)} page(s)")
    except Exception as e:
        print(f"  FAIL  {label:55s} -> {type(e).__name__}: {str(e)[:90]}")
