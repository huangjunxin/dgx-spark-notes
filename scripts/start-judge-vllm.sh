#!/usr/bin/env bash
# Start Gemma-4-E4B (QAT Q4_0 GGUF) as switchyard judge on :8889 (<user> user vLLM)
set -euo pipefail
VLLM_PY="/home/<user>/projects/venv-vllm/bin/python"
MODEL_DIR="/home/<user>/projects/gguf/qwen3-8b-awq"
LOG="/tmp/vllm-gemma.log"
pkill -f "port 8889" 2>/dev/null || true
sleep 1
nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name qwen3-8b-awq \
  --host 127.0.0.1 --port 8889 \
  --gpu-memory-utilization 0.10 \
  --max-model-len 16384 \
  --quantization awq \
  --trust-remote-code \
  --api-key skip \
  > "$LOG" 2>&1 &
echo "vLLM judge (GGUF) launching on 127.0.0.1:8889 (log: $LOG)"
