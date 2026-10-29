#!/usr/bin/env bash
# switch-draft-flavor.sh — 在停机窗口内切换 DFlash2 草稿代次 (zlab <-> nvfp4), 失败自动回滚
#
# 为什么单独写一个脚本而不是手敲几条命令:
#   1) stop.sh 只停不删容器, 而 watchdog 判定 exited 后会在 ~2 分钟内 docker start 旧容器,
#      跟新容器抢名字 ⇒ 必须"停 + 立即删", 这一步不能忘。
#   2) 冷启动 8-10 分钟, 中途失败要能自动回到上一个已验证形态 (11 天稳定期不拿来赌)。
#   3) 每个档位的数据要固化 (ab-collect), 否则换档后就拿不到基线了。
#
# 用法:
#   ./switch-draft-flavor.sh nvfp4            # 切到 NVFP4 草稿
#   ./switch-draft-flavor.sh zlab             # 切回原版 (回滚)
#   BENCH=1 ./switch-draft-flavor.sh nvfp4    # 就绪后顺便跑 bench.py
#
# 退出码: 0=切换成功; 1=切换失败(已尝试回滚, 回滚结果见日志)
set -uo pipefail
cd "$(dirname "$0")"

TARGET_FLAVOR="${1:-}"
case "${TARGET_FLAVOR}" in
  zlab|nvfp4) ;;
  *) echo "用法: $0 <zlab|nvfp4>   (BENCH=1 可顺带跑吞吐)"; exit 1 ;;
esac

CONTAINER="qwen38-27b-sglang"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-qwen38-27b-dflash2}"
ALGO="${ALGO:-DFLASH}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-1010000}"
BENCH="${BENCH:-0}"
OTHER_FLAVOR=$([[ "${TARGET_FLAVOR}" == "nvfp4" ]] && echo zlab || echo nvfp4)

TS="$(date +%Y%m%d-%H%M%S)"
LOG="ab-records/switch-${TARGET_FLAVOR}-${TS}.log"
mkdir -p ab-records
# 全程留档, 同时保留真实退出码 (不要用 | tee, 否则 exit 会被管道子 shell 吞掉)
exec > >(tee "${LOG}") 2>&1

flavor_of_container() {
  docker inspect "${CONTAINER}" --format '{{json .Args}}' 2>/dev/null | python3 -c "
import sys, json
try:
    a = json.load(sys.stdin)
    p = a[a.index('--speculative-draft-model-path') + 1] if '--speculative-draft-model-path' in a else ''
    print('nvfp4' if 'dflash2nvfp4' in p else ('zlab' if 'dflash2' in p else 'unknown'))
except Exception:
    print('none')"
}

CURRENT_FLAVOR="$(flavor_of_container)"
echo "=== $(date '+%F %T') 草稿切换: ${CURRENT_FLAVOR} -> ${TARGET_FLAVOR} ==="
echo "镜像=${IMAGE} 算法=${ALGO} 上下文=${CONTEXT_LENGTH}"
if [[ "${CURRENT_FLAVOR}" == "${TARGET_FLAVOR}" ]]; then
  echo "当前草稿已经是 ${TARGET_FLAVOR}, 无需切换"; exit 0
fi

echo "--- 0) 换档前采集当前档位数据 (若服务在跑) ---"
if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  ./ab-collect.sh "pre-${CURRENT_FLAVOR}" || true
else
  echo "  容器未运行, 跳过"
fi

echo "--- 1) 停并删除旧容器 (删除是为了不让 watchdog 2 分钟后把它 start 回来) ---"
docker stop "${CONTAINER}" >/dev/null 2>&1 || true
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
  echo "!! 容器未删除干净, 终止"; exit 1
fi
echo "  已删除"

echo "--- 2) 启动 ${TARGET_FLAVOR} 档位 (冷启动 8-10 分钟) ---"
IMAGE="${IMAGE}" ALGO="${ALGO}" CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  DRAFT_FLAVOR="${TARGET_FLAVOR}" ./start-dspark-long.sh
RC=$?
echo "=== start 退出码=${RC}  $(date '+%F %T') ==="

if (( RC == 0 )); then
  echo "--- 3) 采集新档位数据 ---"
  if [[ "${BENCH}" == "1" ]]; then
    ./ab-collect.sh "${TARGET_FLAVOR}-draft" --bench
  else
    ./ab-collect.sh "${TARGET_FLAVOR}-draft"
  fi
  echo "=== $(date '+%F %T') 切换成功 ==="
  exit 0
fi

echo "!!! 启动失败, 自动回滚到 ${OTHER_FLAVOR} ---"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
IMAGE="${IMAGE}" ALGO="${ALGO}" CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  DRAFT_FLAVOR="${OTHER_FLAVOR}" ./start-dspark-long.sh
RC2=$?
echo "=== 回滚退出码=${RC2}  $(date '+%F %T') ==="
if (( RC2 == 0 )); then
  ./ab-collect.sh "rollback-${OTHER_FLAVOR}" || true
  echo "已回滚到 ${OTHER_FLAVOR}; 本次切换失败, 失败原因见上方日志"
else
  echo "!! 回滚也失败了, 需要人工介入: 手动执行 DRAFT_FLAVOR=${OTHER_FLAVOR} ./start-dspark-long.sh"
fi
exit 1
