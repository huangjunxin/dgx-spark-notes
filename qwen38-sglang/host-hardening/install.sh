#!/usr/bin/env bash
# install.sh — 应用主机侧 OOM / 健康检查加固 (默认 dry-run, 加 --apply 才真正执行)
#
# 为什么需要 sudo: 装包里软件 (earlyoom) 与改 /etc 下的配置。DSH 沙箱内 sudo 被禁用,
# 所以这一步必须由你手动执行; 脚本只做可审查的动作, 不碰 docker 与服务配置。
#
# 用法:
#   ./install.sh                 # 只打印将要执行的动作 (默认, 安全)
#   ./install.sh --apply         # 真正执行 (需要 sudo 密码)
#   ./install.sh --apply --install-earlyoom   # 连 apt 安装一起做
#
# 回滚: ./install.sh --rollback
set -euo pipefail
cd "$(dirname "$0")"

APPLY=0; DO_INSTALL=0; ROLLBACK=0
for a in "$@"; do
  case "$a" in
    --apply) APPLY=1 ;;
    --install-earlyoom) DO_INSTALL=1 ;;
    --rollback) ROLLBACK=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

run() {
  if (( APPLY )); then
    echo "+ $*"; "$@"
  else
    printf '  [dry-run] %s\n' "$*"
  fi
}

if (( ROLLBACK )); then
  echo "== 回滚 =="
  run sudo systemctl disable --now earlyoom
  run sudo rm -f /etc/default/earlyoom
  echo "已回滚 (earlyoom 包本身未卸载; 如需: sudo apt-get remove -y earlyoom)"
  exit 0
fi

echo "== 1) earlyoom 配置 =="
if [[ -f /etc/default/earlyoom ]]; then
  echo "  已存在 /etc/default/earlyoom (应用时会先备份为 .bak)"
else
  echo "  未安装 (将新建)"
fi
if (( APPLY )); then
  sudo test -f /etc/default/earlyoom && sudo cp -a /etc/default/earlyoom /etc/default/earlyoom.bak || true
  sudo install -m 0644 earlyoom.default /etc/default/earlyoom
  echo "+ 已写入 /etc/default/earlyoom"
else
  echo "  [dry-run] sudo install -m 0644 earlyoom.default /etc/default/earlyoom"
fi

if (( DO_INSTALL )); then
  echo "== 2) 安装 earlyoom =="
  run sudo apt-get update
  run sudo apt-get install -y earlyoom
fi

echo "== 3) 启用 earlyoom =="
run sudo systemctl enable --now earlyoom
run sudo systemctl status earlyoom --no-pager

echo
echo "== 4) 验证 =="
run systemctl is-active earlyoom
run cat /proc/pressure/memory
echo
echo "== 5) 建议同时做 (手工, 不在本脚本内自动改) =="
cat <<'EOF'
  a) 降低 swap 倾向 (GB10 统一内存下 swap 抖动会拖死整机; 当前 vm.swappiness=60):
       echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-qwen38.conf && sudo sysctl --system
  b) 把健康检查从 /health 换成 canary:
       ./canary-check.sh            # 先手工验证; 退出码 0=健康 1=死 3=繁忙
     现有 watchdog 使用 HEALTH_URL(:8888/health)。若要接 canary, 建议在 watchdog 侧
     新增"canary 失败 ≥N 次且退出码=1"才动作的分支, 而不是直接替换 URL。
  c) 崩溃取证目录已由 start-dspark-long.sh 挂好:
       ~/projects/qwen38-model-cache/crash-dumps/  (容器内 --crash-dump-folder)
  d) 重启风暴护栏 (供 watchdog/定时任务调用, 只读):
       ./restart-storm-guard.sh
EOF
echo
(( APPLY )) || echo "以上均为 dry-run。确认无误后重跑: ./install.sh --apply --install-earlyoom"
