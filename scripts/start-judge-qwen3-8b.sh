#!/usr/bin/env bash
# Start Qwen3-8B-AWQ as switchyard judge on :8889 (<user> user vLLM)
# 8B-class judge candidate - the memory-freed re-test of the original plan
set -euo pipefail
export VLLM_CACHE_ROOT="/home/<user>/projects/judge-models/.vllm-cache"
export PATH="/home/<user>/projects/venv-vllm/bin:$PATH"
VLLM_PY="/home/<user>/projects/venv-vllm/bin/python"
MODEL_DIR="/home/<user>/projects/judge-models/Qwen3-8B-AWQ"
LOG="/tmp/vllm-qwen3-8b.log"
pkill -f 'vllm.entrypoints.*888[9]' 2>/dev/null || true
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
echo "vLLM judge (Qwen3-8B-AWQ) launching on 127.0.0.1:8889 (log: $LOG)"
