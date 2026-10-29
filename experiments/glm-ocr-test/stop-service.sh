#!/usr/bin/env bash
# 停止 GLM-OCR 服务栈，归还统一内存（主服务共用 128 GB 池）
set -uo pipefail
echo "停止中..."
pkill -f "glm-ocr-upload-gateway.py"                               && echo "  ✓ 上传网关" || echo "  · 上传网关未运行"
pkill -f "glmocr[.]server"                                        && echo "  ✓ SDK HTTP 服务" || echo "  · SDK 服务未运行"
pkill -f "vllm [s]erve"                                           && echo "  ✓ vLLM 引擎" || echo "  · vLLM 未运行"
sleep 5
echo
echo "剩余监听："; ss -ltn | grep -E ':(5002|5010|8080)\b' || echo "  (无)"
free -m | head -2
