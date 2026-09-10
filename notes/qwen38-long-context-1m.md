# Qwen3.8-27B 长上下文：262K → 1M（YaRN + DFlash2/DSpark）

> 日期：2026-09-10 · 主机：`spark-XXXX`（DGX Spark / GB10, sm_121a, 128GB 统一内存）
> 目标：在**不牺牲投机解码速度**的前提下，把主服务的上下文从原生 262,144 扩到 1M。
> 结果：✅ 上线 **DFlash2 + 1,010,000 token**；另有一条 **DSpark + 506,000** 的等价路线（也实测通过）。

---

## 1. 结论速览

| 项 | 切换前 | DSpark+YaRN 506K | **DFlash2 1M（现役）** |
|---|---|---|---|
| 上下文 | 262,144 | 506,000 | **1,010,000** |
| 投机解码 | DSpark | DSpark（保留） | DFlash2 |
| bench 非思考 400tok | 21.5 / 20.2 tok/s | 24.8 / 23.0 | 23.4 / 24.0 |
| bench 思考 400tok | 24.8 | 22.8 | 24.8 |
| 大海捞针 | 不可用 | 310K ✅ / 466K ✅ | 310K ✅ / **652K ✅** |
| KV 池 | 约 1.30M token @mem 0.85 | 1,131,401 @0.70 | 1,031,239 @0.70 |
| 宿主机可用内存 | 18 GB（swap 用 2 GB） | 27 GB | 26 GB |
| accept len (rate) | 2.8-3.95 (0.44-0.57) | 2.05-3.23 (0.15-0.32) | 2.27-2.80 (0.18-0.26) |

**一句话结论**：长上下文与投机解码可以共存，但**YaRN 绝不能走 `--json-model-override-args`**——那个参数会被 SGLang 转发给独立草稿模型并把它打崩。把 YaRN 写进**模型自己的 config.json** 即可，两条投机栈（DSpark / DFlash2）都适用。

---

## 2. 核心机制：参数污染（本仓库最重要的一条经验）

SGLang 把 `--json-model-override-args` 同时用在 **target 和 draft** 上：

- `speculative/dspark_components/dspark_config.py:35,136`（DSpark 草稿）
- `arg_groups/speculative_hook.py:217`（DFlash/DFlash2 草稿）
- 根因在 `utils/hf_transformers/config.py`：override 只在目标属性已是 `PretrainedConfig` 时才 merge，否则 `setattr`

草稿是**扁平文本 config**（没有 `text_config` 键），于是 `{"text_config": {...}}` 被整体 setattr 成一个**裸 dict**；之后 `get_hf_text_config()` 用 `PretrainedConfig(**_sub)` 实例化它，transformers 的 YaRN 校验去取 `max_position_embeddings` → **AttributeError，引擎启动即崩**。

上游状态（截至记录时）：
| 项 | 状态 |
|---|---|
| issue [sgl-project/sglang#33505](https://github.com/sgl-project/sglang/issues/33505)（同一条报错） | **open，0 评论** |
| PR [#33351](https://github.com/sgl-project/sglang/pull/33351)（deep-merge 覆盖，已合并） | 解释了「target 不崩、只崩 draft」 |
| PR [#35583](https://github.com/sgl-project/sglang/pull/35583) `--speculative-draft-model-override-args` | **已关闭，未合并** |
| PR [#35841](https://github.com/sgl-project/sglang/pull/35841) `--speculative-draft-json-model-override-args` | **open，未合并** |
| 官方模型卡（Qwen/Qwen3.8-27B） | 只给了 CLI override 的做法，未覆盖草稿场景 |

---

## 3. 三条路线与实测裁决

| 路线 | 做法 | 实测结果（本机同一镜像） |
|---|---|---|
| **A（本机采用）** | 不走 CLI 覆盖，把 YaRN 写进 target 的 `config.json`（= 模型卡「方案一」） | ✅ 512K / 1M 均通过，草稿完全不受污染 |
| **B** | 保留官方 CLI 覆盖，但给草稿 config 补一个**完整的 `text_config`**（含 `num_attention_heads` / `max_position_embeddings`），让覆盖走 merge 分支 | ✅ 通过（可作为「必须用 CLI」场景的解法） |
| C | 给 SGLang 打补丁隔离草稿参数（上游 PR 未合并） | 未采用（要挂载/改镜像） |
| ❌ 谣言 | 「在草稿 config 顶层补 `max_position_embeddings` + `rope_parameters{factor:1.0,type:default}` 就能骗过校验」 | **实测证伪**：仍然 AttributeError（崩的是覆盖新建的嵌套 stub） |
| ❌ 谣言 | 「DFlash2 原生兼容长上下文，不受全局 RoPE 覆盖影响」 | **实测证伪**：DFlash2 草稿 + CLI 覆盖 = 同样崩溃 |
| ❌ 谣言 | 「DSpark 草稿本质是 1024 滑窗注意力」 | 事实错误：DSpark 草稿是 5 层 `full_attention`（`sliding_window: null`）；**2048 滑窗的是 DFlash2 草稿** |

实现细节（路线 A）：

- 用**软链 farm** 生成变体目录：只把 `config.json` 换成真实文件，权重/tokenizer 全部相对软链 → **零拷贝、不动 HF 缓存 blob、不需要重新下载**。
- 变体按 `factor x 262144` 取整命名（512K→`yarn512`、1M→`yarn1024`），脚本按同一规则解析，服务长度（如 506000 / 1010000）与变体长度解耦。
- 容器内路径复用已有挂载：`/root/.cache/huggingface/variants/<name>`。

---

## 4. 最终方案（DFlash2 + 1M）与备用方案（DSpark + 506K）

```bash
cd /home/<user>/projects/qwen38-sglang

# 生成 / 重建 config 变体
python3 make-yarn-variant.py --context 1048576                      # target（内联 YaRN factor 4.0）
python3 make-yarn-variant.py --context 1048576 --dflash2            # DFlash2 草稿（烧 YaRN）
python3 make-yarn-variant.py --context 1048576 --draft-keep-rope    # DSpark 草稿（只抬 max_pos，rope 保持训练值）

# 现役：DFlash2 + 1M（需要官方 dev 镜像）
IMAGE=lmsysorg/sglang:dev-qwen38-27b-dflash2 ALGO=DFLASH CONTEXT_LENGTH=1010000 ./start-dspark-long.sh

# 等价路线：DSpark + 506K（同一个镜像 lmsysorg/sglang:qwen38-27b 即可）
CONTEXT_LENGTH=506000 ./start-dspark-long.sh

# 回滚到最初形态（DSpark + 262K）
./stop.sh && ./start.sh
```

关键参数（现役）：

| 参数 | 值 | 为什么 |
|---|---|---|
| `--json-model-override-args` | `'{}'` | 见 §2，一旦传 YaRN 覆盖草稿就崩 |
| `--context-length` | `1010000` | YaRN 上界 1048576，留 3.7% 余量避开 verify 窗口越界（见 P6） |
| `--mem-fraction-static` | `0.70` | GB10 统一内存陷阱（见 P5） |
| docker `--memory/--memory-swap` | `100g` | 防止把宿主机拖到需要断电的 livelock |
| `--disable-flashinfer-autotune` | 长上下文自动加 | autotuner 有 25-40GB 不可见瞬时分配 |
| `HF_HUB_OFFLINE=1` | 是 | 防止 Hub 重新解析覆盖打过补丁的 config（P8） |
| `--mamba-radix-cache-strategy` | DFLASH→`extra_buffer` / DSPARK→`extra_buffer_lazy` | DFLASH 不支持 lazy（P3） |

---

## 5. 踩坑清单（本轮全部实际踩过）

| # | 现象 | 根因 | 解决 |
|---|---|---|---|
| P1 | 加 YaRN 覆盖后启动即崩：`AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'` | CLI 覆盖被转发给草稿并在草稿上建出裸 dict `text_config`（§2） | 别走 CLI；YaRN 写进 target config.json（或路线 B） |
| P2 | 按「草稿顶层补 max_pos + rope_parameters」改仍崩 | 崩的是覆盖新建的嵌套 stub，与草稿顶层无关 | 同 P1；该说法已证伪 |
| P3 | 切 DFLASH 后启动断言失败 | DFLASH 不支持 `extra_buffer_lazy` | 改 `extra_buffer`（脚本按算法自动切） |
| P4 | 用 `lmsysorg/sglang:qwen38-27b` 跑 DFlash2 报 `DFlash2DraftModel is not a registered model` | 该镜像不含 DFlash2 架构注册 | 用官方 `lmsysorg/sglang:dev-qwen38-27b-dflash2` |
| P5 | 长上下文下宿主机内存可能被打满（极端情况需断电恢复） | GB10 统一内存：SGLang 记账看不到 autotuner / CUDA graph 的 25-40GB 瞬时分配（社区实测：0.80 在三并发下崩过，0.85 只剩约 8GB 触发 earlyoom） | mem-fraction 不高于 0.70 + docker 内存硬上限 + 关 autotune |
| P6 | 接近窗口上界的请求可能触发 CUDA illegal address（引擎挂掉） | verify 窗口不做剩余预算裁剪（上游 issue #33454，修复 PR #33963 未合并） | `--context-length` 留 ≥8 token / 约 3.7% 余量 |
| P7 | 长上下文 KV 预算比估算紧 | 外挂草稿自身也吃 KV（DSpark 草稿 5 层全注意力约 10.2 KB/token；DFlash2 草稿 2048 滑窗） | 规划并发时把草稿 KV 算进去 |
| P8 | 担心 Hub 刷新把打过补丁的 config 覆盖回去 | HF 缓存可能被重新解析 | 容器加 `HF_HUB_OFFLINE=1` |
| P9 | 变体目录找不到 | YaRN factor = round(len/262144)，服务长度与变体长度不是一对一 | 变体按 factor x 262144 取整命名 + 脚本同规则解析 |
| P10 | 大海捞针脚本请求的 token 数与预期差很多 | 脚本按字符估算（中文约 0.6 token/字） | 以服务端 `usage.prompt_tokens` 为准 |
| P11 | 服务端已是 1M，但 DSH（harness）界面仍显示 262K | DSH 对自定义 OpenAI 兼容 provider **不探测 `max_model_len`**，落到内置默认 `DEFAULT_CONTEXT_WINDOW = 262144`（`@deepseek-ai/dsh-llm-pi-ai`） | 在 `~/.dsh/settings.yaml` 的模型条目里显式写 `contextWindow: 1010000`；该文件被 chokidar 监听可热加载，但**旧会话缓存了旧值**，需开新会话或重启 harness |

---

## 6. 验证方法与实测数据

1. **零停机预检**（只读 config、不加载权重、不占端口）：`verify-*.py` 系列——直接跑 `ModelConfig.from_server_args`，可复现 P1 崩溃、也可验证各变体能否加载。
2. **吞吐**：`bench.py`（400-token 中文 essay，非思考 x2 + 思考 x1，含 TTFT）。
3. **长文召回**：`bench-long-ctx.py`（在 10% 深度埋密码，报告真实 `prompt_tokens` / TTFT / 是否命中）。实测：310,427 token ✅；465,855 token ✅（DSpark 506K 档）；652,137 token ✅（DFlash2 1M 档）。
4. **接受率**：`docker logs qwen38-27b-sglang | grep 'accept len'`，与基线对照。
5. **冷 prefill 时间**：310K ≈ 9.5 分钟、466K ≈ 18 分钟、652K ≈ 32 分钟（之后同前缀请求走 radix cache 秒回）。

---

## 7. 参考资料

- 官方模型卡（超长文本 / YaRN 配方）：https://huggingface.co/Qwen/Qwen3.8-27B
- 官方仓库（262K native，YaRN 可到 1M）：https://github.com/AlibabaCloud-Official/Qwen3.8-27B
- SGLang cookbook（Qwen3.8-27B / DGX Spark cell）：https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- 同硬件实测仓库：https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
- **1M 现场方案（DFlash2，配 patch-yarn.py 改两个 config.json）**：https://github.com/hasso5703/dgx-spark-qwen38
- 社区 DSpark 草稿 YaRN fork（vLLM）：https://huggingface.co/drowzeys/keys-Qwen3.8-27B-DSpark-vLLM-YaRN-1M-Context
- DFlash2 539K 实测讨论：https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/discussions/8
- 上游 issue：#33505（覆盖不递归）、#33454（verify 窗口越界）

---

## 8. 可继续做的事

1. **配对接受率 A/B**：基线（DSpark 262K）与长上下文档用同一探针各跑一次，量化 YaRN 对接受率的影响（本轮基线样本非严格配对）。
2. **DSpark 1M**：变体（`qwen38-nvfp4-yarn1024` + `qwen38-dspark-maxpos-yarn1024`）已就绪，但 1M 单请求约需 33GB KV + 草稿 KV，当前池只是刚好；建议压并发后实测。
3. **kV dtype 实验**：本镜像 CLI 已支持 `nvfp4` / `fp4_mx_block16` / `fp4_e2m1`，若可用可把 KV 池再翻倍（换来 2 条 1M 并发）。
4. **视觉输入**：DFlash2 的官方 dev 镜像不含社区那份 mrope 修复（sglang#34446），图片/视频输入的位置编码可能有偏差，纯文本无影响——要用视觉输入需单独验证。
