# Qwen3.8-27B 长上下文：262K → 1M（DFlash2 / DSpark 双路线）+ 踩坑记录

> 日期：2026-09-10 · 主机：`spark-XXXX`（DGX Spark / GB10, sm_121a, 128GB 统一内存）
> 目标：**不牺牲投机解码速度**，把主服务上下文从原生 262,144 扩到 1M。
> 结果：✅ 现役 **DFlash2 + 1,010,000**（实测 652,137 token 召回正确）；✅ 备用 **DSpark + 506,000**（310K/466K 召回正确）。
> ⚠️ **2026-09-15 更新：DFlash2 现役形态在真实负载下已两次崩溃**（`dflash_worker_v2.py` batch 尺寸 bug → 容器 exit 0；间隔从 4 天 18 小时缩短到 25 分钟），
> 详见 [qwen38-dflash2-crash-20260915.md](qwen38-dflash2-crash-20260915.md)。当前决定：靠自愈 + 告警扛着；复发到崩溃循环再切 DSpark 506K / MTP 262K。

---

## 0. 结论速览

| 路线 | 上下文 | 投机解码 | 镜像 | 实测召回 | 吞吐 (bench.py 400tok) |
|---|---|---|---|---|---|
| **A. 现役（推荐）** | **1,010,000** | DFlash2 (block 8) | `lmsysorg/sglang:dev-qwen38-27b-dflash2` | **652,137 ✅** | 23.4 / 24.0 / 24.8 tok/s |
| B. 备用（保 DSpark） | 506,000 | DSpark (block 7) | `lmsysorg/sglang:qwen38-27b` | 310,427 ✅ / 465,855 ✅ | 24.8 / 23.0 / 22.8 tok/s |
| （原基线） | 262,144 | DSpark | `lmsysorg/sglang:qwen38-27b` | 不可用 | 21.5 / 20.2 / 24.8 tok/s |

| 其它指标 | 基线 262K | DSpark 506K | DFlash2 1M |
|---|---|---|---|
| KV 池 | 约 1.30M token @0.85 | 1,131,401 @0.70 | 1,031,239 @0.70 |
| 宿主机可用内存 | 18 GB（swap 2 GB） | 27 GB | 26 GB |
| accept len (rate) | 2.8-3.95 (0.44-0.57) | 2.05-3.23 (0.15-0.32) | 2.27-2.80 (0.18-0.26) |
| 冷 prefill 首 token | — | 568s / 1094s | 568s / **1941s** |

**一句话**：长上下文与投机解码可以共存，但 **YaRN 绝不能走 `--json-model-override-args`**——那个参数会被 SGLang 转发给独立草稿模型并把它打崩（§3）。把 YaRN 写进**模型自己的 config.json** 即可，两条投机栈都适用。

---

## 1. 现役方案 A：DFlash2 + 1,010,000

### 1.1 一次性准备

```bash
# 官方 DFlash2 镜像（老镜像 lmsysorg/sglang:qwen38-27b 未注册 DFlash2DraftModel）
docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2        # SGLang 0.0.0.dev1+g5f55db35e

# DFlash2 草稿权重（z-lab 发布，约 2.6 GB）
HF_HUB_DISABLE_XET=1 /home/<user>/.venvs/hf/bin/hf download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir /home/<user>/projects/qwen38-model-cache/dflash2-draft
```

### 1.2 生成 config 变体（软链 farm，零拷贝）

```bash
cd /home/<user>/projects/qwen38-sglang
python3 make-yarn-variant.py --context 1048576            # target：YaRN factor 4.0 内联
python3 make-yarn-variant.py --context 1048576 --dflash2  # 草稿：把同一套 YaRN 烧进它的 config
```

| 变体（`qwen38-model-cache/variants/`） | 改了什么 | 用途 |
|---|---|---|
| `qwen38-nvfp4-yarn1024/` | target `text_config`：`max_position_embeddings=1048576` + `rope_parameters{yarn, factor 4.0, original_max 262144}` | `--model-path` |
| `qwen38-dflash2-yarn1024/` | DFlash2 草稿（z-lab 原 rope_type=default）：`max_position_embeddings=1048576` + 同一套 YaRN | `--speculative-draft-model-path` |

只有 `config.json` 是真实文件，权重/tokenizer 全是指向 HF 缓存的**相对软链**：不复制 21GB、不动缓存里被校验的 blob、无需重新下载。容器内复用已有挂载：`/root/.cache/huggingface/variants/<name>`。

### 1.3 启动（当前生效命令）

```bash
cd /home/<user>/projects/qwen38-sglang
IMAGE=lmsysorg/sglang:dev-qwen38-27b-dflash2 ALGO=DFLASH CONTEXT_LENGTH=1010000 ./start-dspark-long.sh
```

| 参数 | 值 | 为什么 |
|---|---|---|
| 镜像 | `...dev-qwen38-27b-dflash2` | 老镜像启动报 `DFlash2DraftModel is not a registered model` |
| `--model-path` | `.../variants/qwen38-nvfp4-yarn1024` | YaRN 内联在 config.json |
| `--speculative-draft-model-path` | `.../variants/qwen38-dflash2-yarn1024` | 草稿自带同一套 YaRN |
| `--json-model-override-args` | `'{}'` | **关键**：一传 YaRN 覆盖，草稿就崩（§3） |
| `--speculative-algorithm` | `DFLASH`，block 8 / draft 8 / `--speculative-draft-model-quantization unquant` | DFlash2 块扩散草稿 |
| `--context-length` | `1010000` | YaRN factor 4 上界 1048576，故意留 3.7% 余量（verify 窗口不做边界裁剪，顶格会 CUDA illegal address） |
| `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | 是 | 允许超原生长度 |
| `--mem-fraction-static` | `0.70`（旧 0.85） | GB10 统一内存陷阱：记账看不到 25-40GB 瞬时分配 |
| docker `--memory/--memory-swap` | `100g` | 硬上限，防宿主机 livelock（要断电那种） |
| `--disable-flashinfer-autotune` | 长上下文自动加 | autotuner 是瞬时分配大户 |
| `HF_HUB_OFFLINE=1` | 是 | 防 Hub 刷新把打过补丁的 config 覆盖回去 |
| `--mamba-radix-cache-strategy` | DFLASH → `extra_buffer` | **DFLASH 不支持 extra_buffer_lazy**（引擎断言） |
| `--max-mamba-cache-size` | 96（并发 8 x 12） | hasso 现场值 |

### 1.4 就绪自检

```bash
curl -s http://127.0.0.1:8888/v1/models        # 期望 max_model_len=1010000
docker logs qwen38-27b-sglang 2>&1 | grep -oE "speculative_algorithm='[A-Z]+'|max_total_num_tokens=[0-9]+|context_len=[0-9]+|available_gpu_mem=[0-9.]+ GB" | tail -5
```

本机实测：`max_model_len=1010000`、`speculative_algorithm='DFLASH'`、`max_total_num_tokens=1,031,239`、`available_gpu_mem=30.39 GB`。

### 1.5 回滚

```bash
./stop.sh && ./start.sh                                    # 回最初：DSpark + 262K + mem 0.85
./stop.sh && CONTEXT_LENGTH=506000 ./start-dspark-long.sh  # 回备用：DSpark + 506K
```

---

## 2. 备用方案 B：DSpark + 506,000（保留原算法）

```bash
cd /home/<user>/projects/qwen38-sglang
python3 make-yarn-variant.py --context 524288                    # target：YaRN factor 2.0
python3 make-yarn-variant.py --context 524288 --draft-keep-rope  # DSpark 草稿：只抬 max_pos，rope 保持训练值
CONTEXT_LENGTH=506000 ./start-dspark-long.sh                     # MODE=draftmaxpos + ALGO=DSPARK（默认）
```

| 项 | 方案 A (DFlash2) | 方案 B (DSpark) |
|---|---|---|
| 镜像 | dev-qwen38-27b-dflash2 | qwen38-27b |
| target 变体 | yarn1024（factor 4.0） | yarn512（factor 2.0） |
| 草稿变体 | dflash2-yarn1024（草稿换绳） | **dspark-maxpos-yarn512（只抬 max_pos，rope 不动）** |
| context-length | 1010000 | 506000（factor 2 上界 524288，留 3.5% 余量） |
| mamba 策略 | extra_buffer / 96 槽 | extra_buffer_lazy / 32 槽 |
| 实测召回 | 652,137 ✅ | 310,427 ✅ / 465,855 ✅ |

为什么 DSpark 草稿只抬 max_pos：它自带训练过的 `rope_parameters{yarn, factor 32.0, original_max 8192}`，改绳有掉接受率的风险（脚本另留 `MODE=draftyarn` 作对照）。

---

## 3. 共同机制：为什么 YaRN 不能走 CLI 覆盖

SGLang 把 `--json-model-override-args` **同时用于 target 和 draft**：

| 位置 | 代码 |
|---|---|
| DSpark 草稿 | `speculative/dspark_components/dspark_config.py:35,136` |
| DFlash/DFlash2 草稿 | `arg_groups/speculative_hook.py:217` |
| 根因（覆盖语义） | `utils/hf_transformers/config.py`：只有目标属性已是 `PretrainedConfig` 才 merge，否则 `setattr` |

草稿是**扁平文本 config**（没有 `text_config` 键），`{"text_config":{...}}` 被整体 setattr 成裸 dict；随后 `get_hf_text_config()` 用 `PretrainedConfig(**_sub)` 实例化它，YaRN 校验去取 `max_position_embeddings` → **AttributeError，启动即崩**。

上游状态：issue [#33505](https://github.com/sgl-project/sglang/issues/33505) **open、0 评论**；隔离草稿参数的 PR [#35583](https://github.com/sgl-project/sglang/pull/35583) **已关闭**、[#35841](https://github.com/sgl-project/sglang/pull/35841) **未合并** ⇒ 任何镜像里都没有这个参数。

| 做法 | 本机裁决 |
|---|---|
| **A. YaRN 写进 target 的 config.json（采用）** | ✅ 通过（512K / 1M 均验证） |
| **B. 保留 CLI 覆盖 + 给草稿补完整 `text_config`** | ✅ 通过（覆盖走 merge 分支） |
| C. 给引擎打补丁隔离草稿参数 | 未采用（要挂载/改镜像） |
| ❌ 「草稿顶层补 max_position_embeddings + rope_parameters{factor:1.0,type:default} 就能骗过校验」 | **实测证伪**：仍崩（崩的是覆盖新建的嵌套 stub） |
| ❌ 「DFlash2 原生兼容长上下文，不受全局 RoPE 覆盖影响」 | **实测证伪**：DFlash2 草稿 + CLI 覆盖 = 同样崩 |
| ❌ 「DSpark 草稿本质是 1024 滑窗注意力」 | 事实错误：DSpark 草稿是 5 层 `full_attention`（`sliding_window: null`）；**2048 滑窗的是 DFlash2 草稿** |

---

## 4. 踩坑清单（本轮全部实际踩过）

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| P1 | 加 YaRN 覆盖后启动即崩：`AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'` | CLI 覆盖被转发给草稿并在草稿上建出裸 dict `text_config`（§3） | 别走 CLI；YaRN 写进 target config.json |
| P2 | 按「草稿顶层补 max_pos + rope_parameters」改仍崩 | 崩的是覆盖新建的嵌套 stub，与草稿顶层无关 | 同 P1（该说法已证伪） |
| P3 | 切 DFLASH 后启动断言失败 | DFLASH 不支持 `extra_buffer_lazy` | 改 `extra_buffer`（脚本按算法自动切） |
| P4 | 老镜像跑 DFlash2 报 `DFlash2DraftModel is not a registered model` | 镜像不含 DFlash2 架构注册 | 用官方 `dev-qwen38-27b-dflash2` |
| P5 | 长上下文下宿主机内存可能被打满（极端情况需断电恢复） | GB10 统一内存：记账看不到 autotuner / CUDA graph 的 25-40GB 瞬时分配（社区实测：0.80 三并发崩过，0.85 只剩约 8GB 触发 earlyoom） | mem-fraction ≤0.70 + docker 内存硬上限 + 关 autotune |
| P6 | 接近窗口上界的请求可能触发 CUDA illegal address（引擎挂掉） | verify 窗口不做剩余预算裁剪（上游 issue #33454，修复 PR #33963 未合并） | `--context-length` 留 ≥8 token / 约 3.7% 余量 |
| P7 | 长上下文 KV 预算比估算紧 | 外挂草稿自身也吃 KV（DSpark 草稿 5 层全注意力约 10.2 KB/token；DFlash2 草稿 2048 滑窗开销小） | 规划并发时把草稿 KV 算进去 |
| P8 | 担心 Hub 刷新把打过补丁的 config 覆盖回去 | HF 缓存可能被重新解析 | 容器加 `HF_HUB_OFFLINE=1` |
| P9 | 变体目录找不到 | YaRN factor = round(len/262144)，服务长度与变体长度不是一对一 | 变体按 factor x 262144 取整命名 + 脚本同规则解析 |
| P10 | 大海捞针脚本请求的 token 数与预期差很多 | 脚本按字符估算（中文约 0.6 token/字） | 以服务端 `usage.prompt_tokens` 为准 |
| P11 | 服务端已是 1M，但 DSH（harness）界面仍显示 262K | DSH 对自定义 OpenAI 兼容 provider **不探测 `max_model_len`**，落到内置默认 `DEFAULT_CONTEXT_WINDOW = 262144`（`@deepseek-ai/dsh-llm-pi-ai`） | 在 `~/.dsh/settings.yaml` 的模型条目里显式写 `contextWindow: 1010000`；该文件被 chokidar 监听可热加载，但**旧会话缓存了旧值**，需开新会话或重启 harness |

---

## 5. 验证方法与实测数据

1. **零停机预检**（只读 config、不加载权重、不占端口）：`verify-*.py` 系列跑 `ModelConfig.from_server_args`，可复现 P1 崩溃、也可验证各变体能否加载。
2. **吞吐**：`bench.py`（400-token 中文 essay，非思考 x2 + 思考 x1，含 TTFT）。
3. **长文召回**：`bench-long-ctx.py`（10% 深度埋密码，报告真实 `prompt_tokens` / TTFT / 是否命中）。实测：71,383 ✅ / 310,427 ✅ / 465,855 ✅（DSpark 506K）/ **652,137 ✅**（DFlash2 1M）。
4. **接受率**：`docker logs qwen38-27b-sglang | grep 'accept len'`，与基线对照。
5. **冷 prefill**：310K ≈ 9.5 分钟、466K ≈ 18 分钟、652K ≈ 32 分钟（之后同前缀走 radix cache 秒回）。

---

## 6. 参考资料

- 官方模型卡（超长文本 / YaRN 配方）：https://huggingface.co/Qwen/Qwen3.8-27B
- 官方仓库（262K native，YaRN 可到 1M）：https://github.com/AlibabaCloud-Official/Qwen3.8-27B
- SGLang cookbook（Qwen3.8-27B / DGX Spark cell）：https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- 同硬件实测仓库：https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
- **1M 现场方案（DFlash2 + patch-yarn.py 改两个 config.json）**：https://github.com/hasso5703/dgx-spark-qwen38
- 社区 DSpark 草稿 YaRN fork（vLLM）：https://huggingface.co/drowzeys/keys-Qwen3.8-27B-DSpark-vLLM-YaRN-1M-Context
- DFlash2 539K 实测讨论：https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/8
- 上游 issue：#33505（覆盖不递归）、#33454（verify 窗口越界）

---

## 7. 可继续做的事

1. **配对接受率 A/B**：基线与长上下文档用同一探针各跑一次，量化 YaRN 对接受率的影响（本轮基线样本非严格配对）。
2. **DSpark 1M**：变体（`qwen38-nvfp4-yarn1024` + `qwen38-dspark-maxpos-yarn1024`）已就绪，但 1M 单请求约需 33GB KV + 草稿 KV，当前池只是刚好；建议压并发后实测。
3. **KV dtype 实验**：本镜像 CLI 支持 `nvfp4` / `fp4_mx_block16` / `fp4_e2m1`，若可用可把 KV 池再翻倍（换来 2 条 1M 并发）。
4. **视觉输入**：DFlash2 官方 dev 镜像不含社区那份 mrope 修复（#34446），图片/视频输入位置编码可能有偏差；纯文本无影响。
