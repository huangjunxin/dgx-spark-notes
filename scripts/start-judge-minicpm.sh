#!/usr/bin/env bash
# Start MiniCPM5-1B as switchyard judge on :8889 (<user> user vLLM)
# Latest-generation 1B dense LLM (LlamaForCausalLM, 128k ctx) - candidate judge
set -euo pipefail
export VLLM_CACHE_ROOT="/home/<user>/projects/judge-models/.vllm-cache"   # ~/.cache/vllm is root-owned by ornith container
export PATH="/home/<user>/projects/venv-vllm/bin:$PATH"   # ninja for flashinfer JIT
VLLM_PY="/home/<user>/projects/venv-vllm/bin/python"
MODEL_DIR="/home/<user>/projects/judge-models/MiniCPM5-1B"
LOG="/tmp/vllm-minicpm.log"
pkill -f 'port 888[89]' 2>/dev/null || true
sleep 1
nohup "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name minicpm5-1b \
  --host 127.0.0.1 --port 8889 \
  --gpu-memory-utilization 0.10 \
  --max-model-len 16384 \
  --trust-remote-code \
  --api-key skip \
  > "$LOG" 2>&1 &
echo "vLLM judge (MiniCPM5-1B) launching on 127.0.0.1:8889 (log: $LOG)"
