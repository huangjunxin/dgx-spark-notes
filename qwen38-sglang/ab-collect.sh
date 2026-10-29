#!/usr/bin/env bash
# ab-collect.sh — 给"当前正在跑的实例"打一份档位快照, 用于 A/B 对比
#
# 为什么需要: 本机单卡无法并存两套配置, 每次对比都要停机重建 (冷启动 7-10 分钟)。
# 所以每个档位跑完必须立刻把数据固定下来, 否则换档后就再也拿不到基线。
#
# 用法:
#   ./ab-collect.sh baseline                 # 只采快照 (不动服务)
#   ./ab-collect.sh nvfp4-draft --bench      # 快照 + 跑 bench.py 吞吐
#   ./ab-collect.sh nvfp4-draft --bench --needle
#
# 输出: ./ab-records/<label>-<时间戳>.md  (同时打印到屏幕)
#
# 对比时至少看这四项: decode graph bs / 并发、KV 池 (max_total_num_tokens)、
# accept len & rate、以及"未出现 Scheduler hit an exception 且容器未重启"。

set -uo pipefail
cd "$(dirname "$0")"

LABEL="${1:-}"; shift || true
if [[ -z "${LABEL}" ]]; then
  echo "用法: ./ab-collect.sh <label> [--bench] [--needle]"; exit 1
fi
RUN_BENCH=0; RUN_NEEDLE=0
for a in "$@"; do
  case "$a" in
    --bench) RUN_BENCH=1 ;;
    --needle) RUN_NEEDLE=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

CONTAINER="${CONTAINER:-qwen38-27b-sglang}"
PORT="${PORT:-8888}"
OUTDIR="ab-records"
mkdir -p "${OUTDIR}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="${OUTDIR}/${LABEL}-${TS}.md"

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "!! 容器 ${CONTAINER} 未在运行, 无法采集"; exit 1
fi

# ---------- 1) 容器定义与当前档位 ----------
STARTED_AT="$(docker inspect "${CONTAINER}" --format '{{.State.StartedAt}}')"
RESTARTS="$(docker inspect "${CONTAINER}" --format '{{.RestartCount}}')"
IMAGE="$(docker inspect "${CONTAINER}" --format '{{.Config.Image}}')"
IMAGE_ID="$(docker inspect "${CONTAINER}" --format '{{.Image}}' | cut -c8-19)"
ARGS_JSON="$(docker inspect "${CONTAINER}" --format '{{json .Args}}')"
getarg() { printf '%s' "${ARGS_JSON}" | python3 -c "
import sys, json
a = json.load(sys.stdin)
k = sys.argv[1]
if k in a:
    i = a.index(k)
    print(a[i+1] if i+1 < len(a) else '')
else:
    print('-')
" "$1"; }

MEM_FRACTION="$(getarg --mem-fraction-static)"
CONCURRENCY="$(getarg --max-running-requests)"
GRAPH_BS="$(getarg --cuda-graph-max-bs-decode)"
KV_DTYPE="$(getarg --kv-cache-dtype)"
ATTN="$(getarg --attention-backend)"
CTX="$(getarg --context-length)"
DRAFT_PATH="$(getarg --speculative-draft-model-path)"
DRAFT_QUANT="$(getarg --speculative-draft-model-quantization)"
DRAFT_TOKENS="$(getarg --speculative-num-draft-tokens)"
ALGO="$(getarg --speculative-algorithm)"
MAMBA_CACHE="$(getarg --max-mamba-cache-size)"
CHUNK="$(getarg --chunked-prefill-size)"

# 草稿实际代次: 从变体名/路径推断 (zlab vs nvfp4)
case "${DRAFT_PATH}" in
  *dflash2nvfp4*) DRAFT_FLAVOR="nvfp4" ;;
  *dflash2*)      DRAFT_FLAVOR="zlab" ;;
  *)              DRAFT_FLAVOR="(非 DFlash2 变体)" ;;
esac

# ---------- 2) 服务端自述 ----------
MODELS_JSON="$(curl -fsS "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || echo '{}')"
MAX_LEN="$(printf '%s' "${MODELS_JSON}" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin); print(d['data'][0].get('max_model_len', '-'))
except Exception: print('-')" 2>/dev/null)"

# ---------- 3) 启动期容量数字 (KV 池) ----------
POOL_LINE="$(docker logs "${CONTAINER}" 2>&1 | grep 'max_total_num_tokens' | tail -1)"
POOL_TOKENS="$(printf '%s' "${POOL_LINE}" | grep -oE 'max_total_num_tokens=[0-9]+' | cut -d= -f2)"
AVAIL_GPU="$(printf '%s' "${POOL_LINE}" | grep -oE 'available_gpu_mem=[0-9.]+ GB' | cut -d= -f2)"
DRAFT_OVERRIDE="$(docker logs "${CONTAINER}" 2>&1 | grep -oE "Overriding the draft model's max_position_embeddings to [0-9]+" | tail -1)"

# ---------- 4) 健康度: 崩溃计数 + 宿主内存 ----------
CRASH_COUNT="$(docker logs "${CONTAINER}" 2>&1 | grep -c 'Scheduler hit an exception' || true)"
HOST_MEM="$(free -h | awk '/Mem|内存/ {print "total="$2" used="$3" avail="$7}')"
SWAP="$(free -h | awk '/Swap|交换/ {print $3}')"

# ---------- 5) 接受率/吞吐统计 (最近 20000 行 decode batch 日志) ----------
STATS="$(docker logs --tail 20000 "${CONTAINER}" 2>&1 | grep 'accept len' | awk '
  function med(arr, cnt,   m, i, k, s) {
    if (cnt == 0) return 0
    for (i = 0; i < cnt; i++) s[i] = arr[i]
    for (i = 1; i < cnt; i++) for (k = i; k > 0 && s[k-1] > s[k]; k--) { m = s[k]; s[k] = s[k-1]; s[k-1] = m }
    return s[int(cnt/2)]
  }
  {
    for (i = 1; i <= NF; i++) {
      if ($i == "len:") { gsub(/,/, "", $(i+1)); al[n] = $(i+1) + 0 }
      if ($i == "rate:") { gsub(/,/, "", $(i+1)); ar[n] = $(i+1) + 0 }
      if ($i == "throughput") { gsub(/[(),]/, "", $(i+2)); tp[n] = $(i+2) + 0 }
      if ($i == "#running-req:") { r = $(i+1) + 0; if (r > maxrq) maxrq = r }
    }
    n++
  }
  END {
    if (n == 0) { print "0 0 0 0 0 0 0"; exit }
    sal = sar = stp = 0
    for (i = 0; i < n; i++) { sal += al[i]; sar += ar[i]; stp += tp[i] }
    printf "%d %.2f %.2f %.2f %.2f %.2f %d\n", n, sal/n, med(al, n), sar/n, med(ar, n), stp/n, maxrq
  }')"
read -r N_SAMPLES ACC_LEN_MEAN ACC_LEN_P50 ACC_RATE_MEAN ACC_RATE_P50 GEN_TP_MEAN MAX_RUNNING <<<"${STATS}"

# ---------- 6) 可选: 吞吐 / 长文召回 ----------
BENCH_OUT="-"; NEEDLE_OUT="-"
if (( RUN_BENCH )); then
  echo "[ab-collect] 跑 bench.py (约 1-2 分钟) ..." >&2
  BENCH_OUT="$(python3 bench.py 2>&1 | tail -20 || echo '(bench.py 失败)')"
fi
if (( RUN_NEEDLE )); then
  NEEDLE_CTX="${NEEDLE_CTX:-300000}"
  echo "[ab-collect] 跑 bench-long-ctx.py --ctx ${NEEDLE_CTX} (可能很久) ..." >&2
  NEEDLE_OUT="$(python3 bench-long-ctx.py --ctx "${NEEDLE_CTX}" --depth 0.1 --no-thinking 2>&1 | tail -15 || echo '(needle 失败)')"
fi

# ---------- 7) 输出 ----------
{
  echo "# A/B 档位快照: ${LABEL}"
  echo
  echo "> 采集时间: $(date '+%Y-%m-%d %H:%M:%S %Z') · 容器 \`${CONTAINER}\` · 主机 $(hostname)"
  echo
  echo "## 档位定义"
  echo
  echo "| 项 | 值 |"
  echo "|---|---|"
  echo "| 镜像 | \`${IMAGE}\` (@${IMAGE_ID}) |"
  echo "| 投机算法 | ${ALGO} / draft tokens ${DRAFT_TOKENS} |"
  echo "| 草稿代次 | **${DRAFT_FLAVOR}** (\`${DRAFT_PATH}\`) |"
  echo "| 草稿量化 | ${DRAFT_QUANT} |"
  echo "| 上下文 | ${CTX} (服务端 max_model_len=${MAX_LEN}) |"
  echo "| mem-fraction / KV dtype / 注意力后端 | ${MEM_FRACTION} / ${KV_DTYPE} / ${ATTN} |"
  echo "| 并发 / decode graph bs / mamba 池 / chunked prefill | ${CONCURRENCY} / ${GRAPH_BS} / ${MAMBA_CACHE} / ${CHUNK} |"
  echo
  echo "## 容量"
  echo
  echo "| 项 | 值 |"
  echo "|---|---|"
  echo "| KV 池 | ${POOL_TOKENS:-'-'} token |"
  echo "| 启动时可用显存 | ${AVAIL_GPU:-'-'} |"
  echo "| 草稿 max_pos 覆盖 | ${DRAFT_OVERRIDE:-'（无, 走变体 config）'} |"
  echo
  echo "## 运行态"
  echo
  echo "| 项 | 值 |"
  echo "|---|---|"
  echo "| 容器启动于 | ${STARTED_AT} |"
  echo "| RestartCount | ${RESTARTS} |"
  echo "| \`Scheduler hit an exception\` 计数 | ${CRASH_COUNT} |"
  echo "| 宿主内存 / swap 已用 | ${HOST_MEM} / ${SWAP} |"
  echo
  echo "## 投机解码表现 (最近 ${N_SAMPLES} 条 decode batch 样本)"
  echo
  echo "| 指标 | 值 |"
  echo "|---|---|"
  echo "| accept len 均值 / 中位数 | ${ACC_LEN_MEAN} / **${ACC_LEN_P50}** |"
  echo "| accept rate 均值 / 中位数 | ${ACC_RATE_MEAN} / **${ACC_RATE_P50}** |"
  echo "| gen throughput 均值 | ${GEN_TP_MEAN} tok/s |"
  echo "| 峰值 #running-req | ${MAX_RUNNING} |"
  echo
  if [[ "${BENCH_OUT}" != "-" ]]; then
    echo "## bench.py (400-token 中文 essay)"
    echo
    echo '```'
    printf '%s\n' "${BENCH_OUT}"
    echo '```'
    echo
  fi
  if [[ "${NEEDLE_OUT}" != "-" ]]; then
    echo "## bench-long-ctx.py (大海捞针, ctx=${NEEDLE_CTX:-300000})"
    echo
    echo '```'
    printf '%s\n' "${NEEDLE_OUT}"
    echo '```'
    echo
  fi
  echo "## 判定参考"
  echo
  echo "- 稳态基线 (2026-09-18 ~ 09-30, 11d19h): accept len p50 **3.38** / rate p50 **0.34**, 0 崩溃, 0 重启"
  echo "- 换档后必须满足: \`Scheduler hit an exception\` 仍为 0、RestartCount 不增长、长文召回仍命中"
} | tee "${OUT}"

echo
echo "已写入 ${OUT}"
