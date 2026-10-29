#!/usr/bin/env bash
# canary-check.sh — 真实生成式健康检查 (替代 /health)
#
# 为什么不能用 /health:
#   SGLang 的 /health 在"引擎 wedged / 事件循环卡死但仍能应答 HTTP"时依然返回 200,
#   会把"假活"判成健康。本脚本改为真正打一次极小的生成请求, 并叠加 PSI 内存压力判断。
#
# 退出码 (供 watchdog / systemd 判定):
#   0 = 健康 (生成成功)
#   1 = 死亡 (生成失败 且 最近 ACTIVITY_WINDOW 秒内容器无任何日志输出)
#   2 = 内存压力临界 (PSI full avg10 超阈值; 与生成是否成功无关)
#   3 = 繁忙 (生成超时/失败, 但容器仍在持续输出日志 —— 长 prefill 常见, 不等于故障)
#
# 用法:
#   ./canary-check.sh              # 人类可读
#   ./canary-check.sh --quiet      # 只在非 0 时输出一行 (给 watchdog 用)
set -uo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8888}"
MODEL="${MODEL:-qwen3.8-27b}"
CONTAINER="${CONTAINER:-qwen38-27b-sglang}"
TIMEOUT="${TIMEOUT:-30}"                 # 单次生成探活超时(秒); 长 prefill 期间应放宽
ACTIVITY_WINDOW="${ACTIVITY_WINDOW:-45}" # 判断"在干活"的日志静默窗口(秒)
PSI_FULL_MAX="${PSI_FULL_MAX:-10}"       # /proc/pressure/memory full avg10 上限(%)
QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1
say() { (( QUIET )) || echo "$@"; }

# ---- 1) 内存压力 (先测, 因为它比生成失败更早出现) ----
psi_full="$(awk -F'avg10=' '/^full/ {split($2, a, " "); print a[1]; exit}' /proc/pressure/memory 2>/dev/null || echo 0)"
psi_full="${psi_full:-0}"
if awk -v v="${psi_full}" -v m="${PSI_FULL_MAX}" 'BEGIN { exit !(v > m) }'; then
  echo "CRITICAL memory-pressure full.avg10=${psi_full}% > ${PSI_FULL_MAX}%  (earlyoom 若未装, 整机有 livelock 风险)"
  exit 2
fi

# ---- 2) 真实生成 ----
BODY="$(printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"max_tokens":4,"temperature":0,"stream":false}' "${MODEL}")"
TMP="$(mktemp)"; trap 'rm -f "${TMP}"' EXIT
START_MS="$(date +%s%3N)"
HTTP_CODE="$(curl -sS -o "${TMP}" -w '%{http_code}' \
  --max-time "${TIMEOUT}" \
  -H 'Content-Type: application/json' \
  -X POST "${BASE_URL}/v1/chat/completions" \
  -d "${BODY}" 2>/dev/null || echo 000)"
ELAPSED_MS=$(( $(date +%s%3N) - START_MS ))

if [[ "${HTTP_CODE}" == "200" ]] && grep -q '"choices"' "${TMP}" 2>/dev/null; then
  say "OK healthy: 生成成功 HTTP 200, 耗时 ${ELAPSED_MS}ms (PSI full avg10=${psi_full}%)"
  exit 0
fi

# ---- 3) 区分"忙"与"死" ----
RECENT_LOGS="$(docker logs --since "${ACTIVITY_WINDOW}s" "${CONTAINER}" 2>&1 | wc -l || echo 0)"
if (( RECENT_LOGS > 0 )); then
  say "BUSY 生成未成功 (HTTP ${HTTP_CODE}, 耗时 ${ELAPSED_MS}ms), 但容器近 ${ACTIVITY_WINDOW}s 有 ${RECENT_LOGS} 行日志 ⇒ 判定为繁忙(长 prefill/排队), 非故障"
  exit 3
fi

echo "DEAD 生成未成功 (HTTP ${HTTP_CODE}) 且容器近 ${ACTIVITY_WINDOW}s 无任何日志输出 ⇒ 疑似真死"
exit 1
