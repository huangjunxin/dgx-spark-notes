# A/B 结论：DFlash2 草稿换代 zlab(BF16) → nvfp4(modelopt NVFP4)

> 执行时间：2026-09-30 22:09 – 22:23 (HKT) · 主机 spark-XXXX (DGX Spark / GB10)
> 切换方式：`./switch-draft-flavor.sh nvfp4`（停+删容器 → 冷启动 6m55s → 自动采集）
> **结论：保留 nvfp4。** 全部硬门槛通过，端到端吞吐稳态提升约 50%（含冷启那轮 +31%）。

---

## 1. 变更内容

| 项 | 换档前 | 换档后 |
|---|---|---|
| 草稿权重 | `z-lab/Qwen3.8-27B-DFlash2`（BF16，3.6 GB） | `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal@bd7a934`（1.5 GB） |
| 草稿变体 | `variants/qwen38-dflash2-yarn1024` | `variants/qwen38-dflash2nvfp4-yarn1024`（YaRN f4.0 已烧入） |
| `--speculative-draft-model-quantization` | `unquant` | `modelopt_fp4` |
| 其余（镜像/算法/上下文/并发/KV/mem-fraction） | 完全不变 | 完全不变 |

## 2. 结果对比

| 指标 | zlab（基线） | nvfp4 | 变化 |
|---|---|---|---|
| **bench.py 稳态 tok/s** | 16.9 / 16.9 / 19.1（均值 **17.6**） | 27.3 / 26.3 / 26.2 与 24.1 / 28.3 / 29.7（稳态均值 **27.0**） | **+53%** |
| bench.py 首轮（含冷启/JIT） | — | 18.2 / 26.6 / 24.6（均值 23.1） | +31% |
| 400-token 墙钟时间 | 21.0 – 23.6 s | 13.5 – 16.6 s | **−33%** |
| KV 池 | 1,205,656 token | **1,236,510 token** | +30,854 (+2.6%) |
| 宿主可用内存 | 19 GiB | **29 GiB** | +10 GiB |
| accept len 中位数 | 3.40（4,094 样本 / 12 天） | 2.08（25 样本 / 本轮负载） | **−39%** ⚠️ |
| accept rate 中位数 | 0.34 | 0.15 | −56% ⚠️ |
| decode gen throughput 均值 | 34.93 tok/s | 41.90 tok/s | +20% |
| `Scheduler hit an exception` | 0 | **0** | ✅ |
| RestartCount | 0 | **0** | ✅ |

## 3. 怎么解读"接受率降了、吞吐反而升了"

NVFP4 草稿每一步更快（权重 1/2.4、dequant 开销低），但**一步接受的 token 更少**（2.08 vs 3.40）。
两者相乘的净效果是正的 —— bench.py 与 decode gen throughput 两个口径都指向更快。

⚠️ **口径警告**：accept len 两侧样本不可比。基线是 12 天混合负载的 4,094 条；
新档位是本轮 25 条（中文散文 + 长文召回）。所以 2.08 这个数**不能**直接判定为永久退化，
但方向明确，值得持续观察：`docker logs qwen38-27b-sglang | grep 'accept len' | tail -50`。

原始记录（可复核）：`baseline-bench-20260930-220714.md`（zlab）、`nvfp4-final-20260930-222313.md`（nvfp4）、
`pre-zlab-20260930-220905.md`（换档前最后一采）、`needle-nvfp4-300k.log`。

## 4. 验收项逐条

| 验收项 | 结果 |
|---|---|
| `Scheduler hit an exception` = 0 | ✅ 0 |
| RestartCount 不增长 | ✅ 0 |
| 满并发（4 = decode graph bs 边界）压测 | ✅ 4 并发 × 1200 token，59 s 全部 HTTP 200，容器未重启 |
| 长文召回 | ✅ `prompt_tokens=178,919` **命中**（TTFT 281.8 s ≈ 635 tok/s prefill） |
| 上下文未缩水 | ✅ `max_model_len=1010000` |
| YaRN 生效方式正确 | ✅ 无 `Overriding the draft model's max_position_embeddings`（走变体 config，非 CLI 覆盖） |
| 宿主内存未恶化 | ✅ avail 19 → 29 GiB |
| 外部直连不受影响 | ✅ 换容器后三张网卡仍 200 |
| 停机窗口未被 watchdog 误判 | ✅ 全程判 `warming`，日志明确"未做任何重启" |

## 5. 回滚（8 分钟内）

```bash
cd ~/projects/qwen38-sglang
./switch-draft-flavor.sh zlab          # 自动停删+重建+采集, 失败还会自动兜回
```
zlab 的权重与 YaRN 变体都还在原位，**零重新下载**。

## 6. 下一步可做（都在原调研结论里）

1. **accept len 长期观察**：若稳定在 2.0 附近，再考虑 `--speculative-num-draft-tokens` 8→16 的 A/B（社区证据冲突，必须自测）。
2. **mem-fraction 0.70→0.72** 试第一档（现在宿主多出 10 GiB，回收 KV 池的收益/风险比更好）。
3. **FP4 KV 判别实验**：先做 N=300K 的 bf16-vs-fp8 判别，再决定是否上 `fp4_mx_block16` + triton。
4. 主机加固（earloop/canary）仍待你用 sudo 执行：`host-hardening/install.sh --apply --install-earlyoom`。
