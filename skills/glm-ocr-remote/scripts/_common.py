"""glm-ocr-remote 公共部分：配置解析与 HTTP。

地址解析优先级（高 → 低）：
  1. 命令行 --server
  2. 环境变量 GLMOCR_URL
  3. skill 目录下的 config.env
  4. 默认 http://127.0.0.1:5010（上传网关）
"""
import json
import os
import pathlib
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:5010"
SKILL_DIR = pathlib.Path(__file__).resolve().parent.parent
CONFIG_ENV = SKILL_DIR / "config.env"


def load_config_env() -> dict:
    vals = {}
    if CONFIG_ENV.is_file():
        for line in CONFIG_ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def resolve_base(cli_url=None) -> str:
    """返回服务基地址，如 http://127.0.0.1:5010（不带 /parse）。"""
    url = cli_url or os.environ.get("GLMOCR_URL") or load_config_env().get("GLMOCR_URL") or DEFAULT_URL
    url = url.strip().rstrip("/")
    for suffix in ("/glmocr/parse", "/parse"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    return url.rstrip("/")


def http_json(url: str, payload=None, timeout: float = 30.0, method=None):
    """发 JSON 或 GET；返回 (status, body)。body 为解析后的对象或原始文本。"""
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(raw)
            except json.JSONDecodeError:
                return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:500]
    except urllib.error.URLError as e:
        return 0, f"无法连接 {url}：{e.reason}"


def start_hint(base: str) -> str:
    return (
        f"服务不可达：{base}\n"
        "  · 在本机（DGX Spark）上启动：\n"
        "      bash /home/<user>/projects/glm-ocr-test/start-service.sh --with-gateway\n"
        "    停止：bash /home/<user>/projects/glm-ocr-test/stop-service.sh\n"
        "  · 冷启动约 2-4 分钟（vLLM 加载权重 + 版面模型），期间 /health 会失败\n"
        f"  · 服务在别的机器时，改 {CONFIG_ENV} 里的 GLMOCR_URL"
    )
