#!/usr/bin/env bash
# qwen38-watchdog.sh — 守护主服务容器 qwen38-27b-sglang（DGX Spark / GB10）
#
# 背景: 2026-09-15 11:14 SGLang scheduler 因 DFlash2 batch 尺寸 bug 抛 RuntimeError，
#       进程树自杀 → 容器 exit 0。当时 RestartPolicy=no，服务静默停摆 66 分钟无人知。
#       事故复盘见 ../notes/qwen38-dflash2-crash-20260915.md
#
# 设计原则（2026-09-15 第二版起）:
#   **告警要早，动作要晚，不确定就不动。** 宁可多观察半小时，也不要误杀一个正在干活的实例
#   —— 冷启动 6.5 分钟，一次误重启就是 7 分钟中断，比"晚发现"贵得多。
#
# 分级判定（classify）与对应动作：
#   healthy       探活通过                      → 什么都不做（顺便标记冷启动结束）
#   warming       冷启动窗口内且本次启动未健康过  → 只记日志，不计失败、绝不动手
#   exited        容器已退出（崩溃）             → 连续 FAIL_THRESHOLD 次 → docker start（无"杀"的风险）
#   portdown      容器在跑但端口没人监听         → 连续 DEAD_CONFIRM 次 → docker restart（进程确已死）
#   unresponsive  端口在监听但探活不应答         → 默认**只告警、不重启**；仅当连续不应答 ≥ HUNG_GRACE
#                                                且日志静默 ≥ HUNG_IDLE 时才重启（HUNG_RESTART=1 时）
#   （维护模式 / 限流 一律压过上面所有动作）
#
# 用法:
#   ./qwen38-watchdog.sh                 # 常驻（systemd user 服务拉起，见 qwen38-watchdog.service）
#   ./qwen38-watchdog.sh --status        # 当前状态 + 最近告警
#   ./qwen38-watchdog.sh --diag          # 诊断：打印判定依据与"此刻会怎么做"，绝不动作
#   ./qwen38-watchdog.sh --once          # 单轮（按策略执行，可能自愈）
#   ./qwen38-watchdog.sh --maintenance on|off   # 维护窗口：只探活/告警，不自愈
#
# 配置: ~/.config/qwen38-watchdog.env（可选，见 qwen38-watchdog.env.example）

set -uo pipefail

CONTAINER="${CONTAINER:-qwen38-27b-sglang}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8888/health}"
HEALTH_HOST="${HEALTH_HOST:-127.0.0.1}"
HEALTH_PORT="${HEALTH_PORT:-8888}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-20}"      # 单次探活超时（秒）。长上下文 prefill 时 /health 可能 6s+ 不响应
ACTIVITY_WINDOW="${ACTIVITY_WINDOW:-45}"    # 忙判定窗口（秒）：日志近 N 秒有输出 = 在干活
INTERVAL="${INTERVAL:-30}"                  # 探活间隔（秒）
FAIL_THRESHOLD="${FAIL_THRESHOLD:-4}"       # 连续失败几次开始告警 / 判定容器已退出（4×30s ≈ 2 min）
STALL_ALERT="${STALL_ALERT:-0}"             # 0=2min级"繁忙/疑似假死"只记 journal 不发通道（默认，降噪：本机多为长 prefill 假警报）；1=也发通道
DEAD_CONFIRM="${DEAD_CONFIRM:-6}"           # 端口无人监听：连续几次才重启（6×30s ≈ 3 min）
HUNG_GRACE="${HUNG_GRACE:-600}"             # 不应答多久后发 CRIT 预警（默认 10 min，给人工抢先窗口）
HUNG_IDLE="${HUNG_IDLE:-900}"               # AND 条件：还需日志静默这么久才真重启（默认 15 min）
                                            # 依据实测：正常运行最长日志空档 466s(7.8min)、P95 179s → 15min ≈ 2× 余量
                                            # 实际重启时刻 = max(HUNG_GRACE, HUNG_IDLE)
HUNG_RESTART="${HUNG_RESTART:-1}"           # 0 = 这类"疑似假死"永不自动重启，只告警（最保守）
STARTUP_GRACE="${STARTUP_GRACE:-1200}"      # 冷启动宽限（秒；本机冷启动实测 ≈6.5 min）
MAX_RESTARTS_PER_HOUR="${MAX_RESTARTS_PER_HOUR:-3}"
BACKOFF="${BACKOFF:-1800}"                  # 触发限流后的退避（秒）
STATE_DIR="${STATE_DIR:-$HOME/.local/state/qwen38-watchdog}"
ALERT_TARGET="${ALERT_TARGET:-}"            # 例: discord:#ops（留空且无 ALERT_CMD 时只写 journal + 文件）
HERMES_PY="${HERMES_PY:-$HOME/.hermes/hermes-agent/venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

# 告警通道：设好 ALERT_TARGET 后，默认复用 hermes 网关已配置的消息平台（消息从 stdin 读）
if [ -z "${ALERT_CMD:-}" ] && [ -n "$ALERT_TARGET" ] && [ -x "$HERMES_PY" ]; then
  ALERT_CMD="timeout 90 $HERMES_PY -m hermes_cli.main send --to $ALERT_TARGET -q"
fi

mkdir -p "$STATE_DIR/incidents" 2>/dev/null || true
LOCKFILE="$STATE_DIR/watchdog.lock"
MAINT_FLAG="$STATE_DIR/maintenance"   # 存在此文件 = 维护窗口，watchdog 只记录不自愈
RESTART_LOG=()   # 自愈时间戳（用于限流）
WARMED=0         # 本次启动后是否已经健康过（把"冷启动中"和"跑着跑着死了"区分开）

# ---------------- 基础设施 ----------------
log()   { printf '%s %s\n' "$(date '+%F %T')" "$*"; }

alert() {  # alert <LEVEL> <message...>
  local level="$1"; shift
  local body="$*" line prio
  line="$(date '+%F %T') [$level] $body"
  printf '%s\n' "$line"
  case "$level" in
    OK)   prio=info ;;
    WARN) prio=warning ;;
    *)    prio=err ;;
  esac
  logger -t qwen38-watchdog -p "daemon.$prio" -- "[$level] $body" 2>/dev/null || true
  printf '%s\n' "$line" >> "$STATE_DIR/alerts.log" 2>/dev/null || true
  if [ -n "$ALERT_CMD" ]; then
    if printf '[%s][qwen3.8-27b] %s\n%s\n' "$(hostname -s)" "$level" "$body" | bash -c "$ALERT_CMD" >>"$STATE_DIR/alerts.log" 2>&1; then
      :
    else
      printf '%s [ALERT_CMD-FAILED] %s\n' "$(date '+%F %T')" "$body" >> "$STATE_DIR/alerts.log" 2>/dev/null || true
    fi
  fi
}

healthy() {  # /health 返回 2xx 即健康
  curl -fsS -m "$HEALTH_TIMEOUT" -o /dev/null "$HEALTH_URL" 2>/dev/null
}

container_running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]
}

container_state() {
  docker inspect -f 'running={{.State.Running}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} restarts={{.RestartCount}}' "$CONTAINER" 2>/dev/null || echo "inspect-failed"
}

container_uptime_secs() {  # 本轮已运行秒数；未知返回 -1
  local started cleaned epoch
  started="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null)"
  [ -z "$started" ] && { echo -1; return; }
  cleaned="$(printf '%s' "$started" | sed 's/\.[0-9]\{1,\}//')"
  epoch="$(date -d "$cleaned" +%s 2>/dev/null)" || { echo -1; return; }
  [ -z "$epoch" ] && { echo -1; return; }
  echo $(( $(date +%s) - epoch ))
}

log_active() {  # log_active <窗口秒> → 0 = 窗口内有日志输出（在干活）
  docker logs --since "${1}s" "$CONTAINER" 2>/dev/null | grep -q .
}

container_busy() { log_active "$ACTIVITY_WINDOW"; }

port_state() {  # yes=端口在监听 / no=没人监听 / unknown=探测不了（按 yes 处理，宁可不动作）
  if timeout 4 bash -c "exec 3<>/dev/tcp/${HEALTH_HOST}/${HEALTH_PORT}" 2>/dev/null; then echo yes; return 0; fi
  if command -v ss >/dev/null 2>&1; then
    if ss -ltn 2>/dev/null | grep -qE "[:.]${HEALTH_PORT}[[:space:]]"; then echo yes; return 0; fi
    echo no; return 0
  fi
  echo unknown
}

warming() {  # 冷启动窗口内、且本次启动还没健康过
  [ "$WARMED" = 1 ] && return 1
  local up; up="$(container_uptime_secs)"
  container_running && [ "$up" -ge 0 ] && [ "$up" -lt "$STARTUP_GRACE" ]
}

seed_warmed() {  # watchdog 自身重启时：此刻探活通过 → 直接认定已完成冷启动，避免把在跑的服务当"启动中"糊弄 20 分钟
  [ "$WARMED" = 1 ] && return 0
  if healthy; then WARMED=1; return 0; fi
  return 1
}

classify() {  # healthy | warming | exited | portdown | unresponsive
  if healthy; then echo healthy; return; fi
  if warming; then echo warming; return; fi
  if ! container_running; then echo exited; return; fi
  if [ "$(port_state)" = no ]; then echo portdown; return; fi
  echo unresponsive
}

capture() {  # capture <reason> → 现场文件路径（崩溃后日志仍在 json-file 里，务必先抓再重启）
  local reason="$1" ts file
  ts="$(date +%Y%m%d-%H%M%S)"
  file="$STATE_DIR/incidents/${ts}-${reason}.log"
  {
    echo "### qwen38 事故现场  $(date '+%F %T')  reason=$reason"
    echo "## 容器状态: $(container_state)"
    echo "## 端口状态: $(port_state)   uptime=$(container_uptime_secs)s   warmed=$WARMED"
    echo "## docker inspect --format json .State"
    docker inspect "$CONTAINER" --format '{{json .State}}' 2>&1
    echo "## docker logs --tail 400（含崩溃 traceback）"
    docker logs --tail 400 "$CONTAINER" 2>&1
  } > "$file" 2>&1
  echo "$file"
}

wait_healthy() {  # wait_healthy <timeout_sec> → 0 健康 / 1 超时
  local deadline=$(( $(date +%s) + $1 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    healthy && return 0
    sleep 15
  done
  return 1
}

rate_limited() {  # 0 = 允许自愈；1 = 限流中（已告警+退避）
  local now kept t
  now=$(date +%s); kept=()
  for t in ${RESTART_LOG[@]+"${RESTART_LOG[@]}"}; do
    [ $((now - t)) -lt 3600 ] && kept+=("$t")
  done
  RESTART_LOG=(${kept[@]+"${kept[@]}"})
  if [ "${#RESTART_LOG[@]}" -ge "$MAX_RESTARTS_PER_HOUR" ]; then
    alert CRIT "1 小时内已自愈 ${#RESTART_LOG[@]} 次（上限 $MAX_RESTARTS_PER_HOUR），疑似崩溃循环；退避 ${BACKOFF}s 不再重启，需人工介入。"
    sleep "$BACKOFF"
    RESTART_LOG=()
    return 1
  fi
  RESTART_LOG+=("$now")
  return 0
}

# ---------------- 自愈（只有"确已死"才会走到这里） ----------------
remediate() {  # remediate <reason>
  local reason="$1" file started elapsed action
  if [ -e "$MAINT_FLAG" ]; then
    alert WARN "需要自愈（$reason），但存在维护标志 $MAINT_FLAG，按维护窗口跳过。"
    return 1
  fi
  rate_limited || return 1
  started=$(date +%s)
  file="$(capture "$reason")"
  if container_running; then action="docker restart"; else action="docker start (容器已退出)"; fi
  alert WARN "判定服务确已不可用（$reason）。现场已存: $file。执行自愈: $action"
  if [ "$DRY_RUN" = 1 ]; then
    log "[dry-run] 跳过 $action"
    return 0
  fi
  if container_running; then
    docker restart -t 30 "$CONTAINER" >/dev/null 2>&1
  else
    docker start "$CONTAINER" >/dev/null 2>&1
  fi
  if wait_healthy "$STARTUP_GRACE"; then
    elapsed=$(( $(date +%s) - started ))
    alert OK "已恢复（自愈耗时 ${elapsed}s，action=$action）"
    return 0
  fi
  alert CRIT "自愈失败：等待 ${STARTUP_GRACE}s 仍未通过 $HEALTH_URL 探活（action=$action）。需要人工介入。"
  return 1
}

# ---------------- 诊断 ----------------
diagnose() {
  local st up ps busy
  seed_warmed
  st="$(classify)"; up="$(container_uptime_secs)"; ps="$(port_state)"
  busy="no"; log_active "$ACTIVITY_WINDOW" && busy="yes"
  echo "时间:        $(date '+%F %T')"
  echo "容器:        $(container_state)"
  echo "运行时长:    ${up}s（冷启动宽限 ${STARTUP_GRACE}s，warmed=$WARMED）"
  echo "端口 ${HEALTH_PORT}:     $ps"
  echo "探活:        $(healthy && echo OK || echo FAIL)  ($HEALTH_URL, 超时 ${HEALTH_TIMEOUT}s)"
  echo "日志活动:    最近 ${ACTIVITY_WINDOW}s 有输出 = $busy"
  echo "维护标志:    $([ -e "$MAINT_FLAG" ] && echo 存在（不自愈） || echo 无)"
  echo "判定:        $st"
  case "$st" in
    healthy)      echo "此刻动作:    无" ;;
    warming)      echo "此刻动作:    无（冷启动中，只观察）" ;;
    exited)       echo "此刻动作:    连续 ${FAIL_THRESHOLD} 次后 docker start（容器已退出，无杀风险）" ;;
    portdown)     echo "此刻动作:    连续 ${DEAD_CONFIRM} 次后 docker restart（端口无人监听 = 进程确已死）" ;;
    unresponsive) echo "此刻动作:    只告警不重启；需连续不应答 ≥${HUNG_GRACE}s 且日志静默 ≥${HUNG_IDLE}s 才考虑重启（HUNG_RESTART=${HUNG_RESTART}）" ;;
  esac
  echo "最近告警:"; tail -n 3 "$STATE_DIR/alerts.log" 2>/dev/null | sed 's/^/  /' || echo "  (无)"
}

check_once() {  # 单轮：按策略执行一次（可能自愈）
  local st up
  seed_warmed
  st="$(classify)"; up="$(container_uptime_secs)"
  case "$st" in
    healthy)  log "健康: $HEALTH_URL 正常"; return 0 ;;
    warming)  log "容器启动中（已 ${up}s / 冷启动窗口 ${STARTUP_GRACE}s），跳过"; return 0 ;;
    unresponsive)
      log "端口在监听但探活不应答 → 判为繁忙/疑似假死，单轮模式不重启"
      return 0 ;;
    exited|portdown) remediate "$st" ;;
  esac
}

# ---------------- 主循环 ----------------
main() {
  exec 9>"$LOCKFILE" 2>/dev/null && flock -n 9 || { echo "已有 watchdog 实例在跑（$LOCKFILE），退出"; exit 0; }

  log "watchdog 启动: container=$CONTAINER health=$HEALTH_URL interval=${INTERVAL}s timeout=${HEALTH_TIMEOUT}s dry_run=$DRY_RUN"
  log "动作阈值: exited→${FAIL_THRESHOLD}次 / portdown→${DEAD_CONFIRM}次 / unresponsive→${HUNG_GRACE}s 先预警、日志静默 ${HUNG_IDLE}s 才重启 (HUNG_RESTART=${HUNG_RESTART}) / 限流 ${MAX_RESTARTS_PER_HOUR} 次每小时"
  log "告警通道: ${ALERT_CMD:-<仅 journal + $STATE_DIR/alerts.log>}"

  local fails=0 deadfails=0 hung_since="" alerted_unresp=0 alerted_port=0 alerted_incident=0 stall_msg=""
  local last_started cur now up st hung remain
  last_started="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null || echo '')"
  [ -z "$last_started" ] && alert WARN "容器 $CONTAINER 不存在，无法守护"
  seed_warmed && log "启动自检: /health 通过 → 认定为已完成冷启动（避免把在跑的服务当启动中）"

  while true; do
    now=$(date +%s); up="$(container_uptime_secs)"; st="$(classify)"

    cur="$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null || echo '')"
    if [ -n "$cur" ] && [ -n "$last_started" ] && [ "$cur" != "$last_started" ]; then
      alert WARN "容器被外部/策略重启（StartedAt $last_started → $cur，RestartCount=$(docker inspect -f '{{.RestartCount}}' "$CONTAINER" 2>/dev/null)）。RestartCount>0 = docker restart 策略拉起，即刚崩溃过一次；watchdog 不重复处理。"
      fails=0; deadfails=0; hung_since=""; alerted_unresp=0; alerted_port=0; alerted_incident=0; WARMED=0
    fi
    last_started="$cur"

    case "$st" in
      healthy)
        if [ "$WARMED" = 0 ]; then WARMED=1; log "冷启动完成，进入正常监控（/health 通过）"; fi
        if [ -n "$hung_since" ] && [ "$alerted_incident" = 1 ]; then
          # 只给"真的发过告警"的事件报恢复；没达告警线的虚惊（1~3 次失败自愈）静默，不再刷屏
          alert OK "服务恢复应答（此前连续 ${fails} 次探活失败、观察了 $((now - hung_since))s，未做任何重启）"
        elif [ -n "$hung_since" ]; then
          log "探活自行恢复（连续 ${fails} 次失败、观察了 $((now - hung_since))s，未达告警线，不上报）"
        elif [ "$fails" -gt 0 ]; then
          log "探活恢复正常"
        fi
        fails=0; deadfails=0; hung_since=""; alerted_unresp=0; alerted_port=0; alerted_incident=0
        ;;

      warming)
        log "容器启动中（已 ${up}s / 冷启动窗口 ${STARTUP_GRACE}s），探活失败不计入、绝不重启"
        fails=0; deadfails=0
        ;;

      exited)
        fails=$((fails + 1))
        log "容器已退出（exit 状态）$fails/$FAIL_THRESHOLD"
        if [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
          remediate "exited" && fails=0
        fi
        ;;

      portdown)
        fails=$((fails + 1)); deadfails=$((deadfails + 1))
        log "容器在跑但 ${HEALTH_PORT} 端口没人监听 $deadfails/$DEAD_CONFIRM（服务进程已死）"
        if [ "$deadfails" -ge "$DEAD_CONFIRM" ]; then
          remediate "portdown" && { fails=0; deadfails=0; }
        elif [ "$deadfails" -ge "$FAIL_THRESHOLD" ] && [ "$alerted_port" = 0 ]; then
          alert WARN "容器在跑但 ${HEALTH_PORT} 端口无人监听（连续 $deadfails 次）——服务进程已死；连续 ${DEAD_CONFIRM} 次后自动重启。"
          alerted_port=1; alerted_incident=1
        fi
        ;;

      unresponsive)
        fails=$((fails + 1))
        [ -z "$hung_since" ] && hung_since=$now
        hung=$((now - hung_since))
        log "端口在监听但探活不应答（连续 $fails 次，已观察 ${hung}s）——繁忙 或 假死，二者未定"
        # ① 2 分钟：首次发现（默认只记 journal 不上报；STALL_ALERT=1 可改回发通道）
        if [ "$alerted_unresp" = 0 ] && [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
          if log_active "$ACTIVITY_WINDOW"; then
            stall_msg="探活连续 $fails 次失败，但端口仍在监听、日志仍在推进 → 判为**繁忙**，不重启，继续观察。"
          else
            stall_msg="探活连续 $fails 次失败，端口在监听但日志近 ${ACTIVITY_WINDOW}s 无输出 → 疑似假死；按保守策略**先观察不重启**（门槛：不应答 ≥${HUNG_GRACE}s 且日志静默 ≥${HUNG_IDLE}s，两条同时满足）。"
          fi
          if [ "$STALL_ALERT" = 1 ]; then
            alert WARN "$stall_msg"; alerted_incident=1
          else
            log "$stall_msg（STALL_ALERT=0：只记录，不上报）"
          fi
          alerted_unresp=1
        fi
        # ② HUNG_GRACE：CRIT 预警，给人工抢先窗口（此刻仍不重启）
        if [ "$HUNG_RESTART" = 1 ] && [ "$hung" -ge "$HUNG_GRACE" ] && [ "$alerted_unresp" = 1 ] && ! log_active "$HUNG_GRACE"; then
          remain=$(( HUNG_IDLE > hung ? HUNG_IDLE - hung : 0 ))
          alert CRIT "预警：已连续 ${hung}s 探活不应答，且最近 ${HUNG_GRACE}s 日志完全静默。再过约 ${remain}s（静默满 ${HUNG_IDLE}s）仍无改善将自动重启；要人工介入请现在处理，或先跑 qwen38-watchdog.sh --maintenance on 冻结自愈。"
          alerted_unresp=2; alerted_incident=1
        fi
        # ③ 两条门槛都满足 → 执行重启
        if [ "$HUNG_RESTART" = 1 ] && [ "$hung" -ge "$HUNG_GRACE" ] && ! log_active "$HUNG_IDLE"; then
          alert CRIT "连续不应答已达 ${hung}s（≥${HUNG_GRACE}s）且日志静默 ≥${HUNG_IDLE}s，判定为真死，执行重启。"
          remediate "hung" && { fails=0; deadfails=0; hung_since=""; alerted_unresp=0; }
        elif [ "$HUNG_RESTART" != 1 ] && [ "$hung" -ge "$HUNG_GRACE" ] && [ "$alerted_unresp" != 3 ]; then
          alert CRIT "连续不应答已达 ${hung}s，但 HUNG_RESTART=0（只告警不自愈）→ 需人工介入。"
          alerted_unresp=3
        fi
        ;;
    esac

    sleep "$INTERVAL"
  done
}

case "${1:-}" in
  --status)
    echo "容器: $(container_state)"
    if healthy; then echo "健康: OK  ($HEALTH_URL)"; else echo "健康: FAIL  ($HEALTH_URL)"; fi
    echo "判定: $(classify)   端口: $(port_state)   维护标志: $([ -e "$MAINT_FLAG" ] && echo 存在 || echo 无)"
    echo "最近告警:"; tail -n 5 "$STATE_DIR/alerts.log" 2>/dev/null || echo "  (无)"
    ;;
  --diag) diagnose ;;
  --once) check_once ;;
  --maintenance)
    case "${2:-}" in
      on)  ( mkdir -p "$STATE_DIR" && : > "$MAINT_FLAG" ) && echo "维护模式已开启：watchdog 只探活/记录，不再自愈（$MAINT_FLAG）" || { echo "无法创建 $MAINT_FLAG（权限或路径问题）"; exit 1; } ;;
      off) rm -f "$MAINT_FLAG" && echo "维护模式已关闭：恢复自愈（已删除 $MAINT_FLAG）" || { echo "无法删除 $MAINT_FLAG"; exit 1; } ;;
      *)   echo "用法: $0 --maintenance on|off"; exit 2 ;;
    esac
    ;;
  ""|--run) main ;;
  *) echo "用法: $0 [--run|--status|--diag|--once|--maintenance on|off]"; exit 2 ;;
esac
