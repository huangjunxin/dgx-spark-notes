#!/usr/bin/env bash
# restart-storm-guard.sh — 检测"OOM → 重启 → 再 OOM"的砖循环
#
# 背景: 容器用 --restart unless-stopped 自愈, 这既救了"偶发崩溃"(09-15 那次静默停摆 66 分钟),
#       也带来风险: 如果崩溃根因是内存, 它会无限重启, 每次冷启动 7-10 分钟, 机器永远起不来。
#       systemd 管不到 docker 容器的重启, 所以用这个脚本自己记账。
#
# 判定: WINDOW 秒内 >= MAX_RESTARTS 次重启 (按 .State.StartedAt 变化计) ⇒ 写标记 + exit 1
#
# 用法:
#   ./restart-storm-guard.sh            # 记账并判定 (只读 docker, 不改任何状态)
#   MAX_RESTARTS=2 ./restart-storm-guard.sh
#
# 退出码: 0 = 正常; 1 = 检测到重启风暴 (建议暂停 watchdog 自愈并人工介入)
set -uo pipefail

CONTAINER="${CONTAINER:-qwen38-27b-sglang}"
WINDOW="${WINDOW:-1800}"          # 观察窗(秒)
MAX_RESTARTS="${MAX_RESTARTS:-3}"
STATE_DIR="${STATE_DIR:-${HOME}/.local/state/qwen38-watchdog}"
HISTORY="${STATE_DIR}/restart-history"
STORM_MARKER="${STATE_DIR}/STORM"
mkdir -p "${STATE_DIR}" 2>/dev/null || true
# 用"真的写一次"来探测可写性: 受限沙箱里 access(-w) 会假阳性, 必须实际落盘才知道
probe="${STATE_DIR}/.write-probe.$$"
if ! { : > "${probe}"; } 2>/dev/null; then
  echo "WARN 状态目录不可写: ${STATE_DIR}"
  echo "     受限沙箱里请指定可写目录, 例如: STATE_DIR=/tmp/qwen38-state $0"
  exit 0
fi
rm -f "${probe}" 2>/dev/null || true

if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "容器 ${CONTAINER} 不存在 (可能刚被删除或正在重建)"; exit 0
fi

STARTED_AT="$(docker inspect "${CONTAINER}" --format '{{.State.StartedAt}}')"
NOW="$(date +%s)"
[[ -f "${HISTORY}" ]] || : > "${HISTORY}"

# StartedAt 变化 = 发生过一次重启; 记一行 "epoch<TAB>StartedAt"
last_line="$(tail -1 "${HISTORY}" 2>/dev/null || true)"
last_started="${last_line#*$'\t'}"
if [[ "${last_started}" != "${STARTED_AT}" ]]; then
  printf '%s\t%s\n' "${NOW}" "${STARTED_AT}" >> "${HISTORY}"
fi

# 只保留窗口内的记录
cutoff=$(( NOW - WINDOW ))
awk -v c="${cutoff}" -F'\t' '$1 >= c' "${HISTORY}" > "${HISTORY}.tmp" 2>/dev/null && mv "${HISTORY}.tmp" "${HISTORY}"
COUNT="$(wc -l < "${HISTORY}" | tr -d ' ')"

if (( COUNT >= MAX_RESTARTS )); then
  {
    echo "STORM $(date '+%F %T') 容器 ${CONTAINER} 在 ${WINDOW}s 内重启了 ${COUNT} 次 (阈值 ${MAX_RESTARTS})"
    echo "最近记录:"; tail -"${COUNT}" "${HISTORY}" | sed 's/^/  /'
    echo "建议:"
    echo "  1) 暂停自愈, 避免砖循环:  systemctl --user stop qwen38-watchdog   (或 ./qwen38-watchdog.sh --maintenance on)"
    echo "  2) 看上次崩溃原因:        docker logs --tail 200 ${CONTAINER} | grep -A20 'Scheduler hit an exception'"
    echo "  3) 看宿主内存/schema:     cat /proc/pressure/memory ; free -h"
    echo "  4) 恢复: 修好根因后再 systemctl --user start qwen38-watchdog"
  } | tee "${STORM_MARKER}"
  exit 1
fi

rm -f "${STORM_MARKER}" 2>/dev/null || true
echo "OK 窗口 ${WINDOW}s 内重启 ${COUNT} 次 (< ${MAX_RESTARTS}); 当前 StartedAt=${STARTED_AT}"
exit 0
