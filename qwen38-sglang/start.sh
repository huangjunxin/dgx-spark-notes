#!/usr/bin/env bash
# Qwen3.8-27B (SGLang + DSpark) — DGX Spark / GB10 (aarch64)
#
# 依据:
#   官方 cookbook (DGX Spark cell): https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
#   同硬件实测仓库: MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark (start-dspark.sh 的 DSPARK 参数栈)
#
# 关键适配(与现网 vLLM ornith 无缝对接):
#   - host 网络 + 绑定 127.0.0.1:8888 —— 与现网 vLLM 完全同位置:
#     * switchyard (routes.toml base_url http://localhost:8888/v1) 不用改
#     * vllm-open-proxy.py (外部 IP 8888 -> 127.0.0.1:8888) 不用改
#     * 切勿用 0.0.0.0:8888: 会和代理脚本抢端口, 导致外部直连挂掉
#   - 模型预下载在 /home/<user>/projects/qwen38-model-cache (准备阶段), 首次启动不再在线下载
#   - served-model-name = qwen3.8-27b, 与 routes.toml 中 target id 严格一致
#
# 用法: ./start.sh   停止: ./stop.sh
# 性能调优(2026-09-09): --mem-fraction-static 0.85 (释放 GB10 统一内存压力, 0.90 曾致 host swap) + NVFP4 packed-FP4-head 模型

set -euo pipefail

IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen38-27b-sglang"
MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"   # packed FP4 lm_head (网友同款, 略快/略省显存)
# 备选: BF16 lm_head 版 -> RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead (cookbook 默认)
DRAFT_ID="RadixArk/Qwen3.8-27B-DSpark"              # DSpark 草稿 (~2.7GB, 已预下载)
SERVED_MODEL_NAME="qwen3.8-27b"
HOST="127.0.0.1"
PORT="8888"

# 并发与 GDN 状态池: 池 = 并发 x S; extra_buffer_lazy + 默认锁 => S=4
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * 4 ))

HF_CACHE="/home/<user>/projects/qwen38-model-cache"                # 预下载缓存(<user> 可写, 不影响 ornith 的旧缓存)
TRITON_CACHE="/home/<user>/projects/qwen38-model-cache/.triton"
CPUSET="5-9,15-19"   # GB10 的 10 个 Cortex-X5 大核(实测 +2~7% decode)

LOG_FILE="$(cd $(dirname $0) && pwd)/.sglang.log"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "容器 ${CONTAINER_NAME} 已在运行 (${LOG_FILE})"
  exit 0
fi
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
mkdir -p "${HF_CACHE}" "${TRITON_CACHE}"

echo "启动 SGLang: ${MODEL_ID} + DSpark(${DRAFT_ID})"
echo "监听 ${HOST}:${PORT}  served-model-name=${SERVED_MODEL_NAME}  并发=${MAX_CONCURRENT_REQUESTS} (mamba池=${MAMBA_CACHE_SIZE})"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  --shm-size 32g \
  --cpuset-cpus "${CPUSET}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.torch-cache \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE}:/root/.triton" \
  -v "${TRITON_CACHE}/../.torch-cache:/root/.torch-cache" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  --context-length 262144 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${DRAFT_ID}" \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 8 \
  --enable-torch-compile --torch-compile-max-bs 4 \
  --cuda-graph-max-bs-decode 4 \
  --num-continuous-decode-steps 2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  >/dev/null

echo "等待 HTTP 就绪 (http://127.0.0.1:${PORT}/v1/models) ..."
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "!! 容器启动失败, 日志最后 50 行:"; docker logs --tail 50 "${CONTAINER_NAME}"
    exit 1
  fi
  sleep 5
done
echo "SGLang 就绪。日志: ${LOG_FILE}  (docker logs -f ${CONTAINER_NAME})"