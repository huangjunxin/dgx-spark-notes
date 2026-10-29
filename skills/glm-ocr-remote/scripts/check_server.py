#!/usr/bin/env python3
"""GLM-OCR 服务健康检查。

退出码：0 = 可用；2 = 不可达（含排查提示）。
用法：python3 check_server.py [--server URL]
"""
import sys

from _common import http_json, resolve_base, start_hint


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    base = resolve_base(args[0] if args else None)
    print(f"服务地址: {base}")

    status, body = http_json(f"{base}/health", timeout=8)
    if status != 200:
        print(f"  x GET {base}/health -> {status if status else '连接失败'}")
        print(start_hint(base))
        return 2

    upstream = body.get("upstream") if isinstance(body, dict) else None
    print(f"  ok 网关 /health -> 200")
    if isinstance(upstream, dict):
        print(f"  ok 上游 SDK 服务 -> {upstream}")

    if ":5010" in base:
        s, _ = http_json(base.replace(":5010", ":5002") + "/health", timeout=5)
        print(f"  {'ok' if s == 200 else 'x '} SDK 服务(:5002) -> {s if s else '连接失败'}")

    print("\n可用。示例：python3 ocr.py 图片.png   /   python3 ocr.py 文档.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
