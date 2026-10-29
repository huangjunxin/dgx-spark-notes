#!/usr/bin/env bash
# ab-compare.sh — 并排对比两份 ab-collect.sh 快照, 直接给结论
#
# 用法:
#   ./ab-compare.sh ab-records/baseline-bench-*.md ab-records/nvfp4-draft-*.md
#
# 为什么需要: 换档后判断"值不值得留"要看四类数字 —— 容量(KV 池)、
# 投机效率(accept len/rate)、端到端吞吐(bench.py)、以及健康(Scheduler 异常/重启计数)。
set -uo pipefail

A="${1:-}"; B="${2:-}"
if [[ -z "${A}" || -z "${B}" ]]; then
  echo "用法: $0 <快照A.md> <快照B.md>"; exit 1
fi
for f in "${A}" "${B}"; do
  [[ -f "${f}" ]] || { echo "找不到文件: ${f}"; exit 1; }
done

# 取表格行第 3 列 (| 指标 | 值 |)
cell() { awk -F'|' -v k="$2" '$2 ~ k {gsub(/^ +| +$/, "", $3); print $3; exit}' "$1"; }

val() { # file, label -> 去掉 markdown 强调与反引号
  cell "$1" "$2" | sed 's/\*//g; s/`//g'
}

row() { # label, key
  local label="$1" key="$2"
  printf '| %s | %s | %s |\n' "${label}" "$(val "${A}" "${key}")" "$(val "${B}" "${key}")"
}

bench_of() {
  awk '/^## bench.py/{f=1} f && /^```/{c++; next} f && c==1{print} f && c==2{exit}' "$1" \
    | grep -oE '[0-9]+\.[0-9]+ tok/s' | sed 's/ tok\/s//' | paste -sd'|' - | sed 's/|/ \/ /g'
}

samples_of() {
  grep -m1 '^## 投机解码表现' "$1" | grep -oE '[0-9]+ 条' | grep -oE '[0-9]+' || echo '-'
}

echo "# A/B 对比"
echo
echo "- A: \`${A}\`"
echo "- B: \`${B}\`"
echo
echo "| 指标 | A | B |"
echo "|---|---|---|"
row "草稿代次" "草稿代次"
row "草稿量化" "草稿量化"
row "上下文 / max_model_len" "上下文"
row "mem-fraction / KV / 后端" "mem-fraction"
row "并发 / graph bs / mamba / chunk" "并发"
row "KV 池 (token)" "KV 池"
row "启动可用显存" "启动时可用显存"
row "accept len 均值 / 中位数" "accept len"
row "accept rate 均值 / 中位数" "accept rate"
row "gen throughput 均值" "gen throughput"
row "峰值 #running-req" "峰值"
printf '| %s | %s | %s |\n' "采样条数 (最近)" "$(samples_of "${A}")" "$(samples_of "${B}")"
row "Scheduler 异常计数" "Scheduler hit an exception"
row "RestartCount" "RestartCount"
row "宿主内存" "宿主内存"
echo
echo "**bench.py (tok/s, 各次运行)**: A = $(bench_of "${A}") | B = $(bench_of "${B}")"
echo
echo "## 判定标准 (换档后 B 应满足)"
echo
echo '1. `Scheduler hit an exception` = 0, RestartCount 不增长 (硬门槛)'
echo '2. accept len 中位数 不低于 A (现役基线 p50 ≈ 3.38-3.40)'
echo '3. bench.py 吞吐不低于 A (现役基线 ≈ 16.9 / 16.9 / 19.1 tok/s, 含 TTFT)'
echo '4. 长文召回仍需单独验证: `python3 bench-long-ctx.py --ctx 300000 --depth 0.1`'
echo
echo "> 注意: 长 prefill / 并发水位不同会显著影响 accept rate 与 gen throughput,"
echo "> 所以只有在**相近负载**下对比才有意义 (ab-collect 记录的是最近 20000 行 decode 日志的统计)。"
