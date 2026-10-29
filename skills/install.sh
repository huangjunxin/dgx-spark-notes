#!/usr/bin/env bash
# 把本目录下的 skill 安装到 agent 的 skill 根目录。
#   bash skills/install.sh                # 装到 ~/.agents/skills（DSH / Claude Code / OpenClaw 通用）
#   bash skills/install.sh /path/to/root  # 装到自定义根目录
#
# 重复安装会保留已有的 config.env（个人服务地址不丢）。
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$HOME/.agents/skills}"
CLAUDE_ROOT="$HOME/.claude/skills"

mkdir -p "$ROOT"
for skill in "$SRC"/*/; do
  name="$(basename "$skill")"
  [ -f "$skill/SKILL.md" ] || continue
  target="$ROOT/$name"

  if [ -f "$target/config.env" ]; then
    tmp="$(mktemp)"
    cp "$target/config.env" "$tmp"
  fi

  rm -rf "$target"
  cp -r "$skill" "$target"
  [ -n "${tmp:-}" ] && mv "$tmp" "$target/config.env" && echo "  保留已有 config.env"

  chmod +x "$target/scripts/"*.py 2>/dev/null || true
  echo "  ✓ 安装 $name -> $target"

  # Claude Code 兼容软链（与 wind-mcp-skill 相同的做法）
  if [ -d "$CLAUDE_ROOT" ] && [ "$ROOT" != "$CLAUDE_ROOT" ]; then
    ln -sfn "$target" "$CLAUDE_ROOT/$name" 2>/dev/null && echo "    ↳ 软链 $CLAUDE_ROOT/$name" || true
  fi
done

echo
echo "装好了。下一步："
echo "  1) 让 agent 新开一个会话（或等目录热加载）即可自动发现"
echo "  2) 检查服务：python3 $ROOT/glm-ocr-remote/scripts/check_server.py"
echo "  3) 服务在别的机器：cp $ROOT/glm-ocr-remote/config.env.example $ROOT/glm-ocr-remote/config.env 后改 GLMOCR_URL"
