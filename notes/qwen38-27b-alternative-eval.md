# Qwen3.8-27B 替代 Ornith 评估记录（2026-08-24）

> 主机：spark-XXXX（DGX Spark / GB10）· 起因：用户看到 MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark（286 stars，2026-08-24 仍更新）的部署脚本，询问是否用 Qwen3.8-27B 替代 ornith
> 结论：**保持现状，不建议替代**（性能全面落后 + 内存独占破坏判官方案 + 能力丢失）

---

## 1. 调研对象

- 仓库：[MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark)
- 模型：RadixArk/Qwen3.8-27B-NVFP4（NVFP4 ~22GB，含 MTP head；LM 权重 ~16.5GB）
- 引擎：SGLang（Docker，镜像 lmsysorg/sglang:qwen38-27b），三种可切换模式：
  - `start.sh`：EAGLE/MTP 投机（3/1/4），mem 0.95
  - `start-dspark.sh`：DSpark（block-7，草稿 RadixArk/Qwen3.8-27B-DSpark ~2.7GB，8 draft tokens），mem 0.90
  - `start-dflash.sh`：DFlash2（NVFP4 目标），mem 0.90
- 默认配置：262K 上下文、YaRN 关、10 并发、FP8 KV、GDN bf16、flashinfer、torch.compile + decode graph caps 4、连续 2 步 decode、chunk 8192

## 2. 实测速度数据（仓库 README "Measured on this box"，2026-08-18/19，本机同型号 GB10）

| Probe | DSpark | MTP (EAGLE 3/1/4) | DFlash2 |
|---|---|---|---|
| 代码 LRUCache | **51.5 tok/s** | 34.5 tok/s | 50.9 tok/s |
| 长文 essay | 18.3 tok/s | 24.1 tok/s | **25.4 tok/s** |
| 日常 chat | 慢于 DFlash2 | ~21 | **最快**（按 completion_tokens 计） |

- DSpark：代码/数学/工具调用快，自由文本比 MTP 慢
- DFlash2：曾因 mem 0.95 把机器搞挂（硬重启），当前 pin 0.90 + 16 并发
- 注意：这些数字是"net decode tok/s"口径，不可与本机 ornith 的 README 数字（86/440）直接同口径对比，但量级差异是明确的

## 3. 双方能力对比表

| 项目 | Qwen3.8-27B（SGLang） | Ornith-1.5-35B-A3B（vLLM，现状） |
|---|---|---|
| 类型 | 密集 27B（每 token 全参数计算） | MoE 35B-A3B（3B 激活/token） |
| 权重 | NVFP4 22GB（+DSpark 草稿 2.7GB） | NVFP4 ~22GB |
| 内存 | **mem 0.90-0.95 静态 = 109.5-115.6 GiB 独占** | util 0.65 = 79.1 GiB |
| 单流速度 | 18-51.5 tok/s（看模式/场景） | ~86-88 tok/s |
| 聚合吞吐 | 10 并发（无聚合数据） | 24 并发 ≈ 440 tok/s |
| 上下文 | 262K 原生；YaRN 1M 仅 MTP 模式（DSpark/DFlash2 不支持） | **1M 实测可用**（680K recall 通过） |
| sixcat 总分 | 82.3（stock Q4 数据） | **85.0** |
| sixcat 数学 | 40 | **95** |
| sixcat 代码 | **100** | 85 |
| 行为特性 | Qwen3 系：结构化输出/工具调用强 | 推理模型：reasoning 字段、自称 Claude 倾向、结构化输出不可靠（判官由其补位） |
| 稳定性 | 2026-08-18 起验证；DFlash2 出过硬重启事故 | 本机 21+ 天稳定（b12x patches 修复后） |
| 部署 | 脚本开箱即用（无需 patch） | 需 2 个 b12x kernel patches（已就位） |

## 4. 不替代的三个硬理由

1. **性能全面落后**：单流慢 40-70%（51.5 vs 86）；聚合吞吐量级差异（440 vs 无数据）；sixcat 总分低 2.7、数学差 55 分。Qwen3.8 唯一强项是代码（100 vs 85）。
2. **内存独占破坏现有分工**：mem 0.90 静态预留后整机只剩 ~12 GiB，**Qwen3-8B 判官（85% 校准，刚验证可用）必然被挤掉**。当前"ornith 生成（快/1M/高并发）+ Qwen3-8B 判断"是互补最优解。
3. **能力丢失**：1M 上下文、24 并发、数学强项。DSpark 模式连 YaRN 都不支持（262K 封顶）。

## 5. 结论

**保持现状**。ornith 作为主生成模型 + Qwen3-8B-AWQ 判官的分工不变。

**若未来要试 Qwen3.8-27B 的前提条件**（满足才值得）：
- 代码/agentic 流量占比极高（Qwen3.8 代码分 100 vs 85）
- 愿意放弃判官方案和 1M 上下文
- 有维护窗口（必须停 ornith：内存无法共存）

**试用/回滚路径**（两个仓库同源 MiaAI-Lab，回滚成本低）：
```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
cd Qwen3.8-27B-SGLang-DGX-Spark && cp .env.sample .env
# 维护窗口内：停 ornith（~/projects/Ornith-1.5-35B-A3B-DGX-Spark/stop.sh）
./start-dspark.sh          # 或 ./start.sh（MTP，支持 YaRN 1M）
# 不满意回滚：./stop.sh && 恢复 ornith（./start.sh）
```

## 6. 附：仓库配置要点（备忘，若将来部署）

- DSpark 草稿训练于 FP8；QUANT=fp8 不提升接受率；spec 解码报错时 `--attention-backend triton`
- YaRN/上下文 >262144 与 DSpark/DFlash2 不兼容（draft config 泄漏 rope override 崩溃）——YaRN 仅 MTP
- MTP 3/1/4 为实测最优（steps 扫描 2→12.8, 3→17.2, 4→16.8）；NGRAM 慢 ~30%，prefill CUDA graphs 被拒
- GDN 状态池：max-mamba-cache-size = 并发 × S（S=4，extra_buffer_lazy + overlap scheduler）；10 并发 → 40 slots ≈ 3.1GB
- `--mamba-full-memory-ratio 0.9` 过度预留 KV 会压并发，pin 4.21
- 首启动拉权重 ~22GB 进容器内 ./.cache/huggingface
