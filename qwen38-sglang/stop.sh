#!/usr/bin/env bash
set -euo pipefail
docker stop qwen38-27b-sglang
echo "已停止 (保留容器, 可用 docker start qwen38-27b-sglang 恢复)"
