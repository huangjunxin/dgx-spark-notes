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
# ⚠️ 并发上限与 decode CUDA graph 必须对齐 (2026-09-18 源码级定案):
#   DFlash2 草稿采样器的缓冲区行数取自 --cuda-graph-max-bs-decode (本脚本默认 4),
#   一旦 decode batch ≥5 (即同时跑 5 个请求) 就必然越界:
#     RuntimeError: The size of tensor a (4) must match the size of tensor b (5)
#     → Scheduler hit an exception → kill_process_tree → 容器 exit 0
#   本机 09-15/09-17/09-18 因此崩过 5 次。脚本第 ~180 行有断言强制 MAX_CONCURRENT_REQUESTS
#   ≤ CUDA_GRAPH_MAX_BS_DECODE, 不要再靠"记得手改"来保证这件事。
#
# 用法:
#   ./start-dspark-long.sh                            # 现役形态: DFLASH + 1010000
#   CHECK=1 ./start-dspark-long.sh                    # 预检: 只打印将要执行的配置与命令, 不碰容器
#   MODE=yarn ./start-dspark-long.sh
#   MODE=extrap CONTEXT_LENGTH=524288 ./start-dspark-long.sh
#   DRAFT_FLAVOR=nvfp4 ./start-dspark-long.sh         # 草稿换 NVFP4 量化版 (A/B 用)
#   MAX_CONCURRENT_REQUESTS=8 CUDA_GRAPH_MAX_BS_DECODE=8 ./start-dspark-long.sh   # 恢复 8 并发(需压测)
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
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-4}"
#   默认 4 = 与 CUDA_GRAPH_MAX_BS_DECODE 对齐。这是 2026-09-18 事故的修复值,
#   不是保守值: 设成 8 会让 DFlash2 在 decode batch ≥5 时必崩 (见文件头说明)。
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-4}"
TORCH_COMPILE_MAX_BS="${TORCH_COMPILE_MAX_BS:-4}"
#   要恢复更高并发, 必须"同一变更单"里把这两个值一起提到目标并发数, 并做并发压测。
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"                # 可试 nvfp4 / fp4_mx_block16 换更大 KV 池
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"   # fp4 KV 必须配 triton (见下方断言)
EXTRA_ARGS="${EXTRA_ARGS:-}"                    # 例: EXTRA_ARGS="--page-size 1"
ALGO="${ALGO:-DSPARK}"                          # DSPARK | DFLASH (DFlash2, 本镜像内含 dflash_worker_v2)
# DFlash2 草稿代次: zlab = BF16 原版 (3.5GB, 现役); nvfp4 = maurienne-ai modelopt NVFP4 (1.4GB, A/B 候选)
DRAFT_FLAVOR="${DRAFT_FLAVOR:-zlab}"
# 崩溃取证: 把 SGLang 的 crash dump 落到已挂载的 HF 缓存目录下 (容器外持久化) —— 09-18 那次
# traceback 就是随 docker rm -f 一起丢的。CRASH_DUMP=0 关闭。
CRASH_DUMP="${CRASH_DUMP:-1}"
CHECK="${CHECK:-0}"                             # CHECK=1 = 只预检打印, 不 rm/run 容器

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

# DFlash2 草稿代次 -> 源目录 / 容器内路径 / 变体前缀 / 量化方式
DFLASH_SRC_DIR="${HF_CACHE}/dflash2-draft"
DFLASH_CTR_DIR="/root/.cache/huggingface/dflash2-draft"
DFLASH_VARIANT_PREFIX="qwen38-dflash2"
DFLASH_QUANT="unquant"
case "${DRAFT_FLAVOR}" in
  zlab)  ;;   # BF16 原版 (z-lab / incoai 同一 checkpoint), 现役
  nvfp4) DFLASH_SRC_DIR="${HF_CACHE}/dflash2-nvfp4-draft"
         DFLASH_CTR_DIR="/root/.cache/huggingface/dflash2-nvfp4-draft"
         DFLASH_VARIANT_PREFIX="qwen38-dflash2nvfp4"
         DFLASH_QUANT="modelopt_fp4" ;;   # maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal
  *) echo "DRAFT_FLAVOR 必须是 zlab|nvfp4 (当前: ${DRAFT_FLAVOR})"; exit 1 ;;
esac

# DFlash2 路线: 草稿是 2048 滑窗块扩散; 社区做法(incoai #8)是把 YaRN 烧进草稿自己的 config
if [[ "${ALGO}" == "DFLASH" ]]; then
  if [[ "${MODE}" == "extrap" ]] || (( CONTEXT_LENGTH <= 262144 )); then
    DRAFT_PATH="${DFLASH_CTR_DIR}"
  else
    D_VARIANT="${HF_CACHE}/variants/${DFLASH_VARIANT_PREFIX}-yarn${SUFFIX}"
    if [[ ! -f "${D_VARIANT}/config.json" ]]; then
      echo "缺少 DFlash2 变体 ${D_VARIANT} (draft config 已烧入 YaRN)"
      echo "生成: python3 make-yarn-variant.py --context ${VARIANT_CTX} --dflash2 \\"
      echo "        --dflash2-dir ${DFLASH_SRC_DIR} --dflash2-prefix ${DFLASH_VARIANT_PREFIX}"
      exit 1
    fi
    DRAFT_PATH="/root/.cache/huggingface/variants/${DFLASH_VARIANT_PREFIX}-yarn${SUFFIX}"
    YARN_NOTE="${YARN_NOTE} + DFlash2 draft(${DRAFT_FLAVOR}) 内烧 YaRN"
  fi
fi

# 投机解码参数组
if [[ "${ALGO}" == "DFLASH" ]]; then
  SPEC_ARGS=(--speculative-algorithm DFLASH
             --speculative-dflash-block-size 8
             --speculative-draft-model-quantization "${DFLASH_QUANT}"
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

# ================= 参数联动断言 (2026-09-18 事故护栏, 不要删) =================
if (( MAX_CONCURRENT_REQUESTS < 1 )); then
  echo "!! MAX_CONCURRENT_REQUESTS 必须 ≥1 (当前 ${MAX_CONCURRENT_REQUESTS})"; exit 1
fi
if (( MAMBA_CACHE_SIZE < MAX_CONCURRENT_REQUESTS )); then
  echo "!! max-mamba-cache-size (${MAMBA_CACHE_SIZE}) < 并发 (${MAX_CONCURRENT_REQUESTS}); 并发会被 mamba 状态池卡住"; exit 1
fi
if [[ "${ALGO}" == "DFLASH" ]] && (( MAX_CONCURRENT_REQUESTS > CUDA_GRAPH_MAX_BS_DECODE )); then
  cat >&2 <<EOF
!! 拒绝启动: 并发 ${MAX_CONCURRENT_REQUESTS} > --cuda-graph-max-bs-decode ${CUDA_GRAPH_MAX_BS_DECODE}
   DFlash2 草稿采样器缓冲区只有 ${CUDA_GRAPH_MAX_BS_DECODE} 行, decode batch ≥$(( CUDA_GRAPH_MAX_BS_DECODE + 1 )) 时必然越界:
     RuntimeError: The size of tensor a (${CUDA_GRAPH_MAX_BS_DECODE}) must match the size of tensor b (bs)
   (本机 2026-09-15/09-17/09-18 已因此崩 5 次; 根因见 ../notes/qwen38-dflash2-crash-20260918.md)

   两种正确做法:
     a) 保持当前稳定形态 (降并发):
          MAX_CONCURRENT_REQUESTS=${CUDA_GRAPH_MAX_BS_DECODE} ./start-dspark-long.sh
     b) 真要更高并发 —— 必须把两个值一起改, 并做并发压测 (bs>4 的图捕获路径本机从未验证):
          MAX_CONCURRENT_REQUESTS=${MAX_CONCURRENT_REQUESTS} CUDA_GRAPH_MAX_BS_DECODE=${MAX_CONCURRENT_REQUESTS} ./start-dspark-long.sh
          压测素材: .concurrency-test/stress*.json (8 并发, 期望 running 峰值=并发数、全部 200、容器不重启)
EOF
  exit 1
fi
if (( TORCH_COMPILE_MAX_BS < CUDA_GRAPH_MAX_BS_DECODE )); then
  echo "!! --torch-compile-max-bs (${TORCH_COMPILE_MAX_BS}) < --cuda-graph-max-bs-decode (${CUDA_GRAPH_MAX_BS_DECODE}); 建议两者取同值"; exit 1
fi
if [[ "${KV_DTYPE}" == fp4* || "${KV_DTYPE}" == nvfp4 ]]; then
  if [[ "${ATTENTION_BACKEND}" != "triton" ]]; then
    cat >&2 <<EOF
!! 拒绝启动: KV_DTYPE=${KV_DTYPE} 需要 --attention-backend triton (当前 ${ATTENTION_BACKEND})
   - fp4_mx_block16 只有 triton 后端能读; nvfp4 连 triton 也读不了 (别用)
   - 换 triton 会离开现役 flashinfer 的已验证路径 ⇒ 必须重跑长文召回 (bench-long-ctx.py) 再判定
   用法: ATTENTION_BACKEND=triton KV_DTYPE=fp4_mx_block16 ./start-dspark-long.sh
   建议先按 LONG-CONTEXT-RESEARCH/调研结论做 N=300K 的 bf16-vs-fp8 判别实验再决定是否值得。
EOF
    exit 1
  fi
fi

if [[ "${CHECK}" != "1" ]] && docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "容器 ${CONTAINER_NAME} 已在运行; 需先 ./stop.sh"; exit 1
fi
CRASH_DUMP_ARGS=()
CRASH_DUMP_DIR="/root/.cache/huggingface/crash-dumps"
if [[ "${CRASH_DUMP}" == "1" ]]; then
  CRASH_DUMP_ARGS=(--crash-dump-folder "${CRASH_DUMP_DIR}")
fi

RUN_ARGS=(
  -d
  --name "${CONTAINER_NAME}"
  --restart unless-stopped          # scheduler 异常时容器以 exit 0 结束, on-failure 不会触发
  --network host
  --ipc host
  --privileged
  --gpus all
  --shm-size 32g
  --memory "${MEM_CAP}" --memory-swap "${MEM_CAP}"
  --cpuset-cpus "${CPUSET}"
  -e HF_HOME=/root/.cache/huggingface
  -e TRITON_CACHE_DIR=/root/.triton
  -e TORCHINDUCTOR_CACHE_DIR=/root/.torch-cache
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
  -e HF_HUB_OFFLINE=1
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -v "${TRITON_CACHE}:/root/.triton"
  -v "${TORCH_CACHE}:/root/.torch-cache"
  "${IMAGE}"
  python3 -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --mem-fraction-static "${MEM_FRACTION}"
  --attention-backend "${ATTENTION_BACKEND}"
  --chunked-prefill-size 8192
  --disable-prefill-cuda-graph
  ${SAFETY_ARGS[@]+"${SAFETY_ARGS[@]}"}
  --kv-cache-dtype "${KV_DTYPE}"
  --mamba-ssm-dtype bfloat16
  --mamba-full-memory-ratio 4.21
  --mamba-radix-cache-strategy "${MAMBA_STRATEGY}"
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}"
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}"
  --context-length "${CONTEXT_LENGTH}"
  --json-model-override-args '{}'
  "${SPEC_ARGS[@]}"
  --speculative-draft-model-path "${DRAFT_PATH}"
  --enable-torch-compile --torch-compile-max-bs "${TORCH_COMPILE_MAX_BS}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
  --num-continuous-decode-steps 2
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --sampling-defaults model
  --enable-metrics
  --host "${HOST}"
  --port "${PORT}"
  ${CRASH_DUMP_ARGS[@]+"${CRASH_DUMP_ARGS[@]}"}
  ${EXTRA_ARGS}
)

echo "启动 SGLang + 投机解码(Long): algo=${ALGO} mode=${MODE}  上下文=${CONTEXT_LENGTH}  (${YARN_NOTE})"
echo "安全: mem-fraction=${MEM_FRACTION}  docker 内存上限=${MEM_CAP}  autotune=${AUTOTUNE}"
echo "模型: ${MODEL_PATH}"
echo "草稿: ${DRAFT_PATH}  KV=${KV_DTYPE}  注意力后端=${ATTENTION_BACKEND}"
echo "并发=${MAX_CONCURRENT_REQUESTS} (mamba池=${MAMBA_CACHE_SIZE}, decode graph bs=${CUDA_GRAPH_MAX_BS_DECODE})  DRAFT_FLAVOR=${DRAFT_FLAVOR}"

if [[ "${CHECK}" == "1" ]]; then
  echo
  echo "===== CHECK=1 预检模式: 不会 rm/启动任何容器 ====="
  printf '  docker run '
  printf '%q ' "${RUN_ARGS[@]}"
  printf '\n\n'
  if [[ -n "${CHECK_ARGS_OUT:-}" ]]; then
    printf '%s\n' "${RUN_ARGS[@]}" > "${CHECK_ARGS_OUT}"
    echo "  参数已写到 ${CHECK_ARGS_OUT} (每行一个, 可用于与 docker inspect .Args 逐条对比)"
  fi
  echo "参数联动: 并发(${MAX_CONCURRENT_REQUESTS}) ≤ decode graph bs(${CUDA_GRAPH_MAX_BS_DECODE}) ✓"
  echo "如需切实执行, 去掉 CHECK=1 重跑 (注意: 会停掉当前容器, 冷启动约 8-10 分钟)"
  exit 0
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
mkdir -p "${HF_CACHE}" "${TRITON_CACHE}" "${TORCH_CACHE}" "${HF_CACHE}/crash-dumps"

docker run "${RUN_ARGS[@]}" >/dev/null

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