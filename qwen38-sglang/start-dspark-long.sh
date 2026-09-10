#!/usr/bin/env bash
# DSpark + 长上下文启动脚本 —— 保住 DSpark 速度的路线 (2026-09-10 调研产出, 默认不自动执行)
#
# 为什么这样能绕过 (根因, 已在本机复现):
#   SGLang 的 --json-model-override-args 会被转发给 *独立草稿模型* 的 config 加载
#   (speculative/dspark_components/dspark_config.py: model_override_args=json.loads(...)),
#   它把 {"text_config": {...}} 当扁平 key 盖到草稿 config 上 → 草稿多出一个 dict 型
#   text_config → transformers rope 校验取不到 max_position_embeddings → 启动崩溃
#   (AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings')。
#   上游 issue sgl-project/sglang#33505 记录了同一根因, 至今 open。
#   ⇒ 不走 CLI override, 改用 Qwen 模型卡的"方案一": 把 YaRN 写进 target 自己的 config.json。
#     草稿完全不受污染; SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 会自动把草稿的
#     max_position_embeddings 抬到新上下文 (日志: "Overriding the draft model's
#     max_position_embeddings to N")。
#
# MODE 三选一 (维护窗口 A/B, 详见 LONG-CONTEXT-RESEARCH.md §10):
#   draftyarn (默认, 社区做法): target 内联 YaRN + 草稿也换成同一套 YaRN rope
#   yarn    : target 内联 YaRN, 草稿保持自带 rope (yarn32/8192) 不动
#   extrap  : 谁都不缩放, 纯 RoPE 外推 (≤262K 行为与今天逐 token 一致, 草稿零变化)
# MTP 对照: ./stop.sh && CONTEXT_LENGTH=524288 ./start-long.sh
#
# 用法:
#   ./start-dspark-long.sh                            # draftyarn + 512K
#   MODE=yarn ./start-dspark-long.sh
#   MODE=extrap CONTEXT_LENGTH=524288 ./start-dspark-long.sh
#   KV_DTYPE=fp4_mx_block16 ./start-dspark-long.sh    # 更大 KV 池 (本镜像支持, 需实测)
#   ./start.sh                                        # 回滚 (DSpark + 262K)

set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:qwen38-27b}"   # DFlash2 需官方 dev 镜像: IMAGE=lmsysorg/sglang:dev-qwen38-27b-dflash2
CONTAINER_NAME="qwen38-27b-sglang"
SERVED_MODEL_NAME="qwen3.8-27b"
HOST="127.0.0.1"
PORT="8888"
DRAFT_ID="RadixArk/Qwen3.8-27B-DSpark"

MODE="${MODE:-draftmaxpos}"                     # draftmaxpos | draftyarn | yarn | extrap
#   draftmaxpos: target 内联 YaRN + 草稿只抬 max_position_embeddings(rope 保持训练值)  ← 默认, 最保守
#   draftyarn  : target 内联 YaRN + 草稿换成同套 YaRN rope
#   yarn       : target 内联 YaRN, 草稿完全不动
#   extrap     : 都不缩放, 纯外推
CONTEXT_LENGTH="${CONTEXT_LENGTH:-524288}"
MEM_FRACTION="${MEM_FRACTION:-}"                 # 留空=长上下文用 0.70, 262K 用 0.85
MEM_CAP="${MEM_CAP:-100g}"                      # docker 硬上限, 防止把宿主机拖死(GB10 统一内存陷阱)
AUTOTUNE="${AUTOTUNE:-auto}"                    # auto=长上下文关闭 flashinfer autotune; 1=强制开
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"                # 可试 nvfp4 / fp4_mx_block16 换更大 KV 池
EXTRA_ARGS="${EXTRA_ARGS:-}"                    # 例: EXTRA_ARGS="--page-size 1"
ALGO="${ALGO:-DSPARK}"                          # DSPARK | DFLASH (DFlash2, 本镜像内含 dflash_worker_v2)

HF_CACHE="/home/<user>/projects/qwen38-model-cache"
TRITON_CACHE="${HF_CACHE}/.triton"
TORCH_CACHE="${HF_CACHE}/.torch-cache"
CPUSET="5-9,15-19"

if (( CONTEXT_LENGTH < 262144 || CONTEXT_LENGTH > 1048576 )); then
  echo "CONTEXT_LENGTH=${CONTEXT_LENGTH} 超出范围 (262144..1048576)"; exit 1
fi
case "${MODE}" in draftmaxpos|draftyarn|yarn|extrap) ;; *) echo "MODE 必须是 draftmaxpos|draftyarn|yarn|extrap"; exit 1 ;; esac

# 借鉴 hasso5703/dgx-spark-qwen38 的 1M 现场经验: 长上下文下 mem-fraction 别超 0.70
if [[ -z "${MEM_FRACTION}" ]]; then
  if (( CONTEXT_LENGTH > 262144 )); then MEM_FRACTION=0.70; else MEM_FRACTION=0.85; fi
fi
SAFETY_ARGS=()
if (( CONTEXT_LENGTH > 262144 )) && [[ "${AUTOTUNE}" != "1" ]]; then
  SAFETY_ARGS+=(--disable-flashinfer-autotune)   # autotuner 在 GB10 上有 25-40GB 不可见瞬时分配
fi

# YaRN 变体按"factor x 262144"取整: 例如服务 506000 用 factor 2 -> 变体 524288
YARN_FACTOR=$(( (CONTEXT_LENGTH + 131072) / 262144 ))
(( YARN_FACTOR < 1 )) && YARN_FACTOR=1
VARIANT_CTX=$(( YARN_FACTOR * 262144 ))
SUFFIX=$(( VARIANT_CTX / 1024 ))

MODEL_PATH="RadixArk/Qwen3.8-27B-NVFP4"
DRAFT_PATH="${DRAFT_ID}"
YARN_NOTE="无缩放 (纯外推)"
if [[ "${MODE}" != "extrap" ]] && (( CONTEXT_LENGTH > 262144 )); then
  T_VARIANT="${HF_CACHE}/variants/qwen38-nvfp4-yarn${SUFFIX}"
  if [[ ! -f "${T_VARIANT}/config.json" ]]; then
    echo "缺少 target 变体 ${T_VARIANT}; 先运行: python3 make-yarn-variant.py --context ${VARIANT_CTX}"
    exit 1
  fi
  MODEL_PATH="/root/.cache/huggingface/variants/qwen38-nvfp4-yarn${SUFFIX}"   # 容器内路径
  YARN_NOTE="target 内联 YaRN factor=$(( VARIANT_CTX / 262144 ))"
  if [[ "${MODE}" == "draftmaxpos" ]]; then
    D_VARIANT="${HF_CACHE}/variants/qwen38-dspark-maxpos-yarn${SUFFIX}"
    if [[ ! -f "${D_VARIANT}/config.json" ]]; then
      echo "缺少 draft 变体 ${D_VARIANT}; 先运行: python3 make-yarn-variant.py --context ${VARIANT_CTX} --draft-keep-rope"
      exit 1
    fi
    DRAFT_PATH="/root/.cache/huggingface/variants/qwen38-dspark-maxpos-yarn${SUFFIX}"
    YARN_NOTE="${YARN_NOTE} + draft 只抬 max_pos(rope 不动)"
  fi
  if [[ "${MODE}" == "draftyarn" ]]; then
    D_VARIANT="${HF_CACHE}/variants/qwen38-dspark-yarn${SUFFIX}"
    if [[ ! -f "${D_VARIANT}/config.json" ]]; then
      echo "缺少 draft 变体 ${D_VARIANT}; 先运行: python3 make-yarn-variant.py --context ${VARIANT_CTX} --draft"
      exit 1
    fi
    DRAFT_PATH="/root/.cache/huggingface/variants/qwen38-dspark-yarn${SUFFIX}"
    YARN_NOTE="${YARN_NOTE} + draft rope overlay"
  fi
fi

# DFlash2 路线: 草稿是 2048 滑窗块扩散; 社区做法(incoai #8)是把 YaRN 烧进草稿自己的 config
if [[ "${ALGO}" == "DFLASH" ]]; then
  if [[ "${MODE}" == "extrap" ]] || (( CONTEXT_LENGTH <= 262144 )); then
    DRAFT_PATH="/root/.cache/huggingface/dflash2-draft"
  else
    D_VARIANT="${HF_CACHE}/variants/qwen38-dflash2-yarn${SUFFIX}"
    if [[ ! -f "${D_VARIANT}/config.json" ]]; then
      echo "缺少 DFlash2 变体 ${D_VARIANT} (draft config 已烧入 YaRN)"; exit 1
    fi
    DRAFT_PATH="/root/.cache/huggingface/variants/qwen38-dflash2-yarn${SUFFIX}"
    YARN_NOTE="${YARN_NOTE} + DFlash2 draft 内烧 YaRN"
  fi
fi

# 投机解码参数组
if [[ "${ALGO}" == "DFLASH" ]]; then
  SPEC_ARGS=(--speculative-algorithm DFLASH
             --speculative-dflash-block-size 8
             --speculative-draft-model-quantization unquant
             --speculative-num-draft-tokens 8)
  MAMBA_STRATEGY="extra_buffer"      # DFLASH 不支持 extra_buffer_lazy (引擎断言)
  MAMBA_SLOTS="${MAMBA_SLOTS:-12}"     # hasso 现场: 并发 x 12 = 96 槽
else
  SPEC_ARGS=(--speculative-algorithm DSPARK
             --speculative-dspark-block-size 7
             --speculative-draft-model-quantization unquant
             --speculative-num-draft-tokens 8)
  MAMBA_STRATEGY="extra_buffer_lazy"
  MAMBA_SLOTS=4
fi

MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * MAMBA_SLOTS ))

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "容器 ${CONTAINER_NAME} 已在运行; 需先 ./stop.sh"; exit 1
fi
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
mkdir -p "${HF_CACHE}" "${TRITON_CACHE}" "${TORCH_CACHE}"

echo "启动 SGLang + 投机解码(Long): algo=${ALGO} mode=${MODE}  上下文=${CONTEXT_LENGTH}  (${YARN_NOTE})"
echo "安全: mem-fraction=${MEM_FRACTION}  docker 内存上限=${MEM_CAP}  autotune=${AUTOTUNE}"
echo "模型: ${MODEL_PATH}"
echo "草稿: ${DRAFT_PATH}  KV=${KV_DTYPE}  并发=${MAX_CONCURRENT_REQUESTS} (mamba池=${MAMBA_CACHE_SIZE})"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  --shm-size 32g \
  --memory "${MEM_CAP}" --memory-swap "${MEM_CAP}" \
  --cpuset-cpus "${CPUSET}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.torch-cache \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE}:/root/.triton" \
  -v "${TORCH_CACHE}:/root/.torch-cache" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  "${SAFETY_ARGS[@]}" \
  --kv-cache-dtype "${KV_DTYPE}" \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy "${MAMBA_STRATEGY}" \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  --context-length "${CONTEXT_LENGTH}" \
  --json-model-override-args '{}' \
  "${SPEC_ARGS[@]}" \
  --speculative-draft-model-path "${DRAFT_PATH}" \
  --enable-torch-compile --torch-compile-max-bs 4 \
  --cuda-graph-max-bs-decode 4 \
  --num-continuous-decode-steps 2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  ${EXTRA_ARGS} \
  >/dev/null

echo "等待 HTTP 就绪 (冷启动约 8-10 分钟) ..."
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "!! 容器启动失败, 日志最后 60 行:"; docker logs --tail 60 "${CONTAINER_NAME}"; exit 1
  fi
  sleep 5
done
echo "就绪:"; curl -s "http://127.0.0.1:${PORT}/v1/models"; echo
echo "启动自检 (草稿 max_position_embeddings 是否被抬到 ${CONTEXT_LENGTH}):"
docker logs "${CONTAINER_NAME}" 2>&1 | grep -E "Overriding the draft model|context_length" | tail -3
echo
echo "★ 关键验收: DSpark 遇到 YaRN 后接受率有没有掉? (决定这条路是否真的保速)"
echo "  docker logs ${CONTAINER_NAME} 2>&1 | grep 'accept len' | tail -20"
echo "  对照: 262K 现网 accept len≈2.0-4.9 / accept rate≈0.15-0.69"
echo "  速度: python3 bench.py    召回: python3 bench-long-ctx.py --ctx 400000"