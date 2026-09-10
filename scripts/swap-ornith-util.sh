#!/usr/bin/env bash
# Swap ornith container to gpu-memory-utilization 0.80 (backup kept for rollback)
set -euo pipefail
SNAP="/root/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-NVFP4/snapshots/0f0b1b59b879ccde1353e6ebd0fb10c204d4c544"
IMAGE="eugr/spark-vllm-b12x:latest"
PATCHES="/home/<user>/projects/Ornith-1.5-35B-A3B-DGX-Spark/patches"
echo "[1/3] stopping old container..."
docker stop ornith-b12x-serve || true
echo "[2/3] renaming old to ornith-b12x-bak (rollback point)..."
docker rename ornith-b12x-serve ornith-b12x-bak || true
echo "[3/3] launching new container with util 0.80..."
docker run -d --name ornith-b12x-serve --gpus all --network host --ipc=host \
  -v /home/<user>/.cache/huggingface:/root/.cache/huggingface \
  -v /home/<user>/.cache/vllm:/root/.cache/vllm \
  -v "$PATCHES/kernel.py:/usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/kernel.py:ro" \
  -v "$PATCHES/route_pack.py:/usr/local/lib/python3.12/dist-packages/b12x/moe/_shared/kernels/w4a16/route_pack.py:ro" \
  "$IMAGE" \
  vllm serve "$SNAP" \
    --served-model-name ornith-1.5-35b-a3b-nvfp4 \
    --host 127.0.0.1 --port 8888 \
    --trust-remote-code --enable-prefix-caching \
    --spec-method mtp --spec-tokens 1 \
    --max-num-seqs 24 --moe-backend b12x \
    --gpu-memory-utilization 0.90 \
    --max-model-len 262144 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3
echo "new container launched; waiting for health..."
for i in $(seq 1 150); do
  if ! docker ps --format '{{.Names}}' | grep -qx ornith-b12x-serve; then
    echo "ERROR: container exited early"
    docker logs --tail 40 ornith-b12x-serve 2>&1 | tail -40 || true
    echo "ROLLBACK: restoring backup..."
    docker stop ornith-b12x-serve 2>/dev/null || true
    docker rename ornith-b12x-serve ornith-b12x-failed 2>/dev/null || true
    docker rename ornith-b12x-bak ornith-b12x-serve 2>/dev/null || true
    docker start ornith-b12x-serve 2>/dev/null || true
    exit 1
  fi
  if curl -fs --max-time 4 http://127.0.0.1:8888/health >/dev/null 2>&1; then
    echo "HEALTHY: ornith is back on 8888 (util 0.80)"
    docker rm -f ornith-b12x-bak >/dev/null 2>&1 || true
    echo "backup container removed."
    exit 0
  fi
  sleep 4
done
echo "TIMEOUT waiting for health; leaving backup in place (ornith-b12x-bak)."
exit 2
