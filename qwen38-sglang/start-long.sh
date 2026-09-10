#!/usr/bin/env bash
# Qwen3.8-27B 长上下文启动脚本 (SGLang + 原生 MTP 投机解码 + YaRN)
# ——— 由 start.sh (DSpark/262K) 改造, 2026-09-10 调研产出, 默认不自动执行 ———
#
# 用途: 把上下文从原生 262144 扩到 512K / 768K / 1M。
#
# 为什么必须换掉 DSpark:
#   DSpark/DFlash2 的草稿模型是独立 checkpoint, YaRN 覆盖会"泄漏"进草稿的
#   config, 触发 transformers rope 校验崩溃 (AttributeError: max_position_embeddings)。
#   官方模型卡 & 同硬件实测仓库 (MiaAI-Lab) 均确认: YaRN 只能配 MTP。
#   MTP 头是主 checkpoint 自带的 (mtp.layers.0.*, 15 个张量), 无需额外下载,
#   与主模型共享同一套 (已被 YaRN 覆盖的) rope 参数。
#
# 用法:
#   ./start-long.sh                      # 默认 512K (YaRN factor 2.0) + MTP 3/1/4
#   CONTEXT_LENGTH=1048576 ./start-long.sh   # 1M (factor 4.0)
#   CONTEXT_LENGTH=786432 ./start-long.sh    # 768K (factor 3.0)
#   MAX_CONCURRENT_REQUESTS=4 ./start-long.sh
#   MEM_FRACTION=0.90 ./start-long.sh
#   ./start.sh                           # 回滚到 DSpark + 262K
#
# 端口/容器名/served-model-name 与 start.sh 完全一致 -> switchyard、vllm-open-proxy.py 零改动。

set -euo pipefail

IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen38-27b-sglang"
MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4"
SERVED_MODEL_NAME="qwen3.8-27b"
HOST="127.0.0.1"
PORT="8888"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-524288}"      # 262144 .. 1048576
MEM_FRACTION="${MEM_FRACTION:-0.85}"            # 0.85 沿用现网(避免统一内存 swap)
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"

# MTP 投机解码 (MiaAI 同硬件实测最优: steps 3 / topk 1 / draft 4)
SPEC_STEPS="${SPEC_STEPS:-3}"
SPEC_TOPK="${SPEC_TOPK:-1}"
SPEC_DRAFT="${SPEC_DRAFT:-4}"

HF_CACHE="/home/<user>/projects/qwen38-model-cache"
TRITON_CACHE="/home/<user>/projects/qwen38-model-cache/.triton"
TORCH_CACHE="/home/<user>/projects/qwen38-model-cache/.torch-cache"
CPUSET="5-9,15-19"

if (( CONTEXT_LENGTH < 262144 || CONTEXT_LENGTH > 1048576 )); then
  echo "CONTEXT_LENGTH=${CONTEXT_LENGTH} 超出范围 (262144..1048576)"; exit 1
fi
if (( SPEC_TOPK != 1 || SPEC_DRAFT != SPEC_STEPS + 1 )); then
  echo "MTP 链式投机要求 topk=1 且 draft = steps+1"; exit 1
fi

NEED_YARN=0
YARN_FACTOR=1
YARN_OVERRIDE=""
CONTEXT_ARGS=(--context-length "${CONTEXT_LENGTH}")
ALLOW_LONGER_ARGS=()
if (( CONTEXT_LENGTH > 262144 )); then
  NEED_YARN=1
  # 官方模型卡: factor = round(CONTEXT_LENGTH / 262144), 262144 -> 1.0 为基准
  YARN_FACTOR="$(awk -v n="${CONTEXT_LENGTH}" 'BEGIN{printf "%.0f", n/262144}')"
  (( YARN_FACTOR < 1 )) && YARN_FACTOR=1
  YARN_OVERRIDE="$(printf '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": %s, "original_max_position_embeddings": 262144}}}' "${YARN_FACTOR}")"
  CONTEXT_ARGS=(--json-model-override-args "${YARN_OVERRIDE}" --context-length "${CONTEXT_LENGTH}")
  ALLOW_LONGER_ARGS=(-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
fi

MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * 4 ))   # extra_buffer_lazy + overlap => S=4

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "容器 ${CONTAINER_NAME} 已在运行; 需先 ./stop.sh"
  exit 1
fi
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
mkdir -p "${HF_CACHE}" "${TRITON_CACHE}" "${TORCH_CACHE}"

echo "启动 SGLang+YaRN: ${MODEL_ID}  上下文=${CONTEXT_LENGTH} (YaRN factor ${YARN_FACTOR})"
echo "投机解码: MTP steps=${SPEC_STEPS} topk=${SPEC_TOPK} draft=${SPEC_DRAFT}  并发=${MAX_CONCURRENT_REQUESTS} (mamba池=${MAMBA_CACHE_SIZE})"

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
  "${ALLOW_LONGER_ARGS[@]}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE}:/root/.triton" \
  -v "${TORCH_CACHE}:/root/.torch-cache" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  "${CONTEXT_ARGS[@]}" \
  --speculative-algorithm EAGLE \
  --speculative-num-steps "${SPEC_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_DRAFT}" \
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

echo "等待 HTTP 就绪 (http://127.0.0.1:${PORT}/v1/models, 冷启动约 8-10 分钟) ..."
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "!! 容器启动失败, 日志最后 60 行:"; docker logs --tail 60 "${CONTAINER_NAME}"; exit 1
  fi
  sleep 5
done
echo "就绪。自检:"
curl -s "http://127.0.0.1:${PORT}/v1/models"
echo
echo "预期 max_model_len=${CONTEXT_LENGTH}; 若仍是 262144 说明 YaRN 覆盖未生效。"
echo "日志: docker logs -f ${CONTAINER_NAME}"
