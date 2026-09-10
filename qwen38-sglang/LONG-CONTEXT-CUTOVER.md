# 长上下文切换记录：262,144 → **1,010,000**（DFlash2 现役 / DSpark 506K 备用）

> 执行时间: 2026-09-10 09:24 - 12:0x (HKT) · 主机 spark-XXXX (DGX Spark GB10)
> 现役结果: ✅ **DFlash2 + 上下文 1,010,000**，实测 **652,137 token 召回正确**，吞吐与 262K 基线持平
> 备用结果: ✅ **DSpark + 506,000**（不换算法，只换 config），实测 310K / 466K 召回正确

---

## 0. 先看这个：两条可用路线

| 路线 | 上下文 | 投机解码 | 镜像 | 实测召回 | 吞吐 (bench.py 400tok) |
|---|---|---|---|---|---|
| **A. 现役（推荐）** | **1,010,000** | DFlash2 (block 8) | `lmsysorg/sglang:dev-qwen38-27b-dflash2` | **652,137 ✅** | 23.4 / 24.0 / 24.8 tok/s |
| B. 备用（保 DSpark） | 506,000 | DSpark (block 7) | `lmsysorg/sglang:qwen38-27b` | 310,427 ✅ / 465,855 ✅ | 24.8 / 23.0 / 22.8 tok/s |
| （原基线） | 262,144 | DSpark | `lmsysorg/sglang:qwen38-27b` | 不可用 | 21.5 / 20.2 / 24.8 tok/s |

**两条路线的共同前提（也是本轮唯一的关键技术点）**：YaRN 只能写进**模型自己的 config.json**，绝不能走 `--json-model-override-args`——那个参数会被 SGLang 转发给独立草稿模型并把它打崩（§3 有机制与上游 issue）。

---

## 1. 现役方案 A：DFlash2 + 1,010,000

### 1.1 一次性准备

```bash
# 1) 官方 DFlash2 镜像（DFlash2 只在 dev 镜像里；qwen38-27b 镜像未注册 DFlash2DraftModel）
docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2      # SGLang 0.0.0.dev1+g5f55db35e

# 2) DFlash2 草稿权重（z-lab 发布，约 2.6 GB）
HF_HUB_DISABLE_XET=1 /home/<user>/.venvs/hf/bin/hf download z-lab/Qwen3.8-27B-DFlash2 \
  --local-dir /home/<user>/projects/qwen38-model-cache/dflash2-draft
```

### 1.2 生成 config 变体（软链 farm，零拷贝）

```bash
cd /home/<user>/projects/qwen38-sglang
python3 make-yarn-variant.py --context 1048576                 # target: YaRN factor 4.0 内联
python3 make-yarn-variant.py --context 1048576 --dflash2       # 草稿: YaRN 烧进它的 config
```

产出与作用：

| 变体目录（`qwen38-model-cache/variants/`） | 改了什么 | 谁用 |
|---|---|---|
| `qwen38-nvfp4-yarn1024/` | target 的 `text_config`：`max_position_embeddings=1048576` + `rope_parameters{rope_type:yarn, factor:4.0, original_max_position_embeddings:262144}` | `--model-path` |
| `qwen38-dflash2-yarn1024/` | DFlash2 草稿（z-lab，原 rope_type=default）：`max_position_embeddings=1048576` + 同一套 YaRN | `--speculative-draft-model-path` |

实现要点：只有 `config.json` 是真实文件，权重/tokenizer 全是指向 HF 缓存的**相对软链**——不复制 21GB、不改动缓存里被校验的 blob、不需要重新下载。容器内路径复用已有挂载：`/root/.cache/huggingface/variants/<name>`。

### 1.3 启动（当前生效命令）

```bash
cd /home/<user>/projects/qwen38-sglang
IMAGE=lmsysorg/sglang:dev-qwen38-27b-dflash2 ALGO=DFLASH CONTEXT_LENGTH=1010000 ./start-dspark-long.sh
```

| 参数 | 值 | 为什么 |
|---|---|---|
| 镜像 | `lmsysorg/sglang:dev-qwen38-27b-dflash2` | 老镜像没有 DFlash2 架构注册，启动报 `DFlash2DraftModel is not a registered model` |
| `--model-path` | `/root/.cache/huggingface/variants/qwen38-nvfp4-yarn1024` | YaRN 内联在 config.json |
| `--speculative-draft-model-path` | `.../variants/qwen38-dflash2-yarn1024` | 草稿自带同一套 YaRN |
| `--json-model-override-args` | `'{}'` | **关键**：一传 YaRN 覆盖草稿就崩（§3） |
| `--speculative-algorithm` | `DFLASH`，block 8 / 8 draft tokens，`--speculative-draft-model-quantization unquant` | DFlash2 块扩散草稿 |
| `--context-length` | `1010000` | YaRN factor 4 上界是 1048576，**故意留 3.7% 余量**：verify 窗口不做边界裁剪（上游 #33454），顶格会触发 CUDA illegal address |
| `-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | 是 | 允许超过原生长度；引擎会把草稿 max_pos 一并抬到新上下文 |
| `--mem-fraction-static` | `0.70`（旧 0.85） | GB10 统一内存陷阱：SGLang 记账看不到 autotuner / CUDA graph 的 25-40GB 瞬时分配 |
| docker `--memory/--memory-swap` | `100g` | 硬上限，防止把宿主机拖到需要断电的 livelock |
| `--disable-flashinfer-autotune` | 长上下文自动加 | 同上，autotuner 是瞬时分配大户 |
| `-e HF_HUB_OFFLINE=1` | 是 | 防止 Hub 重新解析把打过补丁的 config 覆盖回去 |
| `--mamba-radix-cache-strategy` | DFLASH → `extra_buffer`（DSPARK → `extra_buffer_lazy`） | **DFLASH 不支持 lazy**，引擎会断言失败；脚本按算法自动切 |
| `--max-mamba-cache-size` | 96（并发 8 x 12） | hasso 现场值 |
| 其余（flashinfer / fp8 KV / mamba bf16 / torch.compile / cpuset / reasoning+tool parser） | 与旧配置一致 | 行为不变 |

### 1.4 就绪自检

```bash
curl -s http://127.0.0.1:8888/v1/models            # 期望 max_model_len=1010000
docker logs qwen38-27b-sglang 2>&1 | grep -oE "speculative_algorithm='[A-Z]+'|max_total_num_tokens=[0-9]+|context_len=[0-9]+|available_gpu_mem=[0-9.]+ GB" | tail -5
```

本机实测：`max_model_len=1010000`、`speculative_algorithm='DFLASH'`、`context_len=1010000`、`max_total_num_tokens=1,031,239`、`available_gpu_mem=30.39 GB`。

### 1.5 实测结果

| 项目 | 结果 |
|---|---|
| bench.py（400tok 中文 essay） | 非思考 **23.4 / 24.0** tok/s，思考 **24.8** tok/s |
| 长文召回 | **310,427 token ✅**（TTFT 568s）、**652,137 token ✅**（TTFT 1941s ≈ 32 分钟冷 prefill） |
| 接受率 | accept len 2.27-2.80 / rate 0.18-0.26 |
| KV 池 | 1,031,239 token（fp8_e4m3），mem-fraction 0.70 |
| 宿主机内存 | 可用 26 GB，swap 基本不用 |
| 全链路 | switchyard :4000 → HTTP 200，chat 正常 |

### 1.6 回滚

```bash
./stop.sh && ./start.sh                                   # 回最初形态：DSpark + 262K + mem 0.85
./stop.sh && CONTEXT_LENGTH=506000 ./start-dspark-long.sh # 回备用形态：DSpark + 506K
```

YaRN 变体都是独立目录，回滚不需要还原任何缓存文件。

---

## 2. 备用方案 B：DSpark + 506,000（保留原算法）

适合：不想换镜像/算法，或 DFlash2 出问题时的等价回退。**同一个 `lmsysorg/sglang:qwen38-27b` 镜像即可**。

```bash
cd /home/<user>/projects/qwen38-sglang
python3 make-yarn-variant.py --context 524288                    # target: YaRN factor 2.0
python3 make-yarn-variant.py --context 524288 --draft-keep-rope  # DSpark 草稿: 只抬 max_pos，rope 保持训练值
CONTEXT_LENGTH=506000 ./start-dspark-long.sh                     # MODE 默认 draftmaxpos + ALGO 默认 DSPARK
```

与方案 A 的差异：

| 项 | 方案 A (DFlash2) | 方案 B (DSpark) |
|---|---|---|
| 镜像 | dev-qwen38-27b-dflash2 | qwen38-27b |
| target 变体 | yarn1024（factor 4.0） | yarn512（factor 2.0） |
| 草稿变体 | dflash2-yarn1024（草稿换绳） | **dspark-maxpos-yarn512（草稿只抬 max_pos，rope 不动）** |
| context-length | 1010000 | 506000（factor 2 上界 524288，留 3.5% 余量） |
| mamba 策略 | extra_buffer / 96 槽 | extra_buffer_lazy / 32 槽 |
| 实测召回 | 652,137 ✅ | 310,427 ✅ / 465,855 ✅ |
| 吞吐 | 23.4 / 24.0 / 24.8 | 24.8 / 23.0 / 22.8 |

为什么 DSpark 草稿只抬 max_pos、不换 rope：DSpark 草稿自带训练过的 `rope_parameters{yarn, factor 32.0, original_max_position_embeddings 8192}`，改绳有掉接受率的风险（脚本另留 `MODE=draftyarn` 作对照）。

---

## 3. 共同关键：为什么 YaRN 不能走 `--json-model-override-args`

SGLang 会把该参数**同时用于 target 和 draft**：

| 位置 | 代码 |
|---|---|
| DSpark 草稿 | `speculative/dspark_components/dspark_config.py:35,136` |
| DFlash/DFlash2 草稿 | `arg_groups/speculative_hook.py:217` |
| 根因（覆盖语义） | `utils/hf_transformers/config.py`：只有目标属性已是 `PretrainedConfig` 才 merge，否则 `setattr` |

草稿是**扁平文本 config**（没有 `text_config` 键），于是 `{"text_config":{...}}` 被整体 setattr 成裸 dict；随后 `get_hf_text_config()` 用 `PretrainedConfig(**_sub)` 实例化它，transformers 的 YaRN 校验去取 `max_position_embeddings` → **AttributeError，引擎启动即崩**。

上游状态：issue [#33505](https://github.com/sgl-project/sglang/issues/33505) **open、0 评论**；隔离草稿参数的 PR [#35583](https://github.com/sgl-project/sglang/pull/35583) **已关闭**、[#35841](https://github.com/sgl-project/sglang/pull/35841) **未合并** ⇒ 任何镜像里都没有这个参数。

三种修法与本机裁决：

| 做法 | 结果 |
|---|---|
| **A. YaRN 写进 target 的 config.json（本机采用）** | ✅ 通过（512K / 1M 均验证） |
| **B. 保留 CLI 覆盖 + 给草稿补完整 `text_config`** | ✅ 通过（让覆盖走 merge 分支，见 `variants/qwen38-dspark-cfg-textconfig`） |
| C. 给引擎打补丁隔离草稿参数 | 未采用（需挂载/改镜像） |
| ❌ 「草稿顶层补 `max_position_embeddings` + `rope_parameters{factor:1.0,type:default}` 就能骗过校验」 | **实测证伪**：仍崩（崩的是覆盖新建的嵌套 stub） |
| ❌ 「DFlash2 原生兼容长上下文，不受全局 RoPE 覆盖影响」 | **实测证伪**：DFlash2 草稿 + CLI 覆盖 = 同样崩 |

---

## 4. 验证实录（时间线）

| 时间 | 项目 | 结果 |
|---|---|---|
| 09:24 | `ALGO=DFLASH CONTEXT_LENGTH=1010000`（老镜像） | ❌ 失败两次：(a) DFLASH 不支持 `extra_buffer_lazy`；(b) 老镜像未注册 `DFlash2DraftModel` |
| 09:30 | `CONTEXT_LENGTH=506000 ./start-dspark-long.sh` | ✅ 就绪（DSpark 路线） |
| 09:33 | bench.py ×3 | 24.8 / 22.8 / 23.0 tok/s |
| 09:41 | 大海捞针 71,383 / 310,427 tok | ✅ / ✅（TTFT 52.7s / 568s） |
| 10:2x | 大海捞针 465,855 tok | ✅（TTFT 1094s） |
| 11:0x | `docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2` | ✅（0.0.0.dev1+g5f55db35e） |
| 11:1x | `IMAGE=...dev-qwen38-27b-dflash2 ALGO=DFLASH CONTEXT_LENGTH=1010000` | ✅ 就绪，`max_model_len=1010000`，池 1,031,239 token |
| 11:2x | bench.py ×3（DFlash2） | 23.4 / 24.8 / 24.0 tok/s |
| 11:3x | 大海捞针 310,427 tok | ✅ |
| 11:5x | 大海捞针 **652,137 tok** | ✅（TTFT 1941s） |
| 12:0x | switchyard :4000 | ✅ HTTP 200 + chat 正常 |

---

## 5. 限制与注意事项

1. **冷 prefill 很慢**：310K ≈ 9.5 分钟、466K ≈ 18 分钟、652K ≈ 32 分钟（约 400-550 tok/s）。同前缀二次请求走 radix cache 秒回——长文档场景应固定前缀。
2. **KV 预算要算草稿**：target fp8 约 32.8 KB/token；DFlash2 草稿是 2048 滑窗（开销小），DSpark 草稿 5 层全注意力约 +10.2 KB/token。1M 单请求约需 33GB KV，当前池 1.03M token 只能同时容纳 1 条长请求。
3. **接受率非严格配对**：三档的 accept len/rate 样本来自不同探针窗口，端到端吞吐没有退化，但精确差值需配对 A/B（约 20 分钟、2 次重启）。
4. **视觉输入**：DFlash2 官方 dev 镜像不含社区那份 mrope 修复（上游 #34446），图片/视频输入的位置编码可能有偏差；纯文本无影响，需要用视觉输入时单独验证。
5. **`--json-model-override-args` 必须保持空**：一旦传 YaRN 覆盖，草稿会再次崩溃（除非走 §3 的路线 B）。
6. **DSH（harness）侧显示**：DSH 对自定义 OpenAI 兼容 provider 不探测 `max_model_len`，落到内置默认 262144；需在 `~/.dsh/settings.yaml` 的模型条目里声明 `contextWindow: 1010000`（已改；旧会话有缓存，需开新会话或重启 harness）。

---

## 6. 运维速查

```bash
cd /home/<user>/projects/qwen38-sglang

# 变体生成（按需重建；factor = round(目标长度 / 262144)）
python3 make-yarn-variant.py --context 524288                    # target 512K
python3 make-yarn-variant.py --context 524288 --draft-keep-rope  # DSpark 草稿（只抬 max_pos）
python3 make-yarn-variant.py --context 524288 --draft            # DSpark 草稿（换同套 YaRN，对照）
python3 make-yarn-variant.py --context 1048576 --dflash2         # DFlash2 草稿（烧 YaRN）

# 启动
IMAGE=lmsysorg/sglang:dev-qwen38-27b-dflash2 ALGO=DFLASH CONTEXT_LENGTH=1010000 ./start-dspark-long.sh   # 现役
CONTEXT_LENGTH=506000 ./start-dspark-long.sh                                                             # 备用
CONTEXT_LENGTH=506000 ./start-long.sh                                                                    # MTP 对照
./stop.sh && ./start.sh                                                                                  # 回最初

# 验证
python3 bench.py                                            # 吞吐（对基线 21.5 / 20.2 / 24.8）
python3 bench-long-ctx.py --ctx 1090000 --depth 0.1 --no-thinking   # 长文召回（看返回的 prompt_tokens）
docker logs qwen38-27b-sglang 2>&1 | grep 'accept len' | tail -20   # 接受率

# 零停机预检（只读 config，不起服务）
docker run --rm --gpus all -v /home/<user>/projects/qwen38-model-cache:/root/.cache/huggingface \
  -v /home/<user>/projects/qwen38-sglang:/work -e HF_HOME=/root/.cache/huggingface \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 -e HF_HUB_OFFLINE=1 \
  lmsysorg/sglang:qwen38-27b python3 /work/verify-dspark-modes.py
```

---

## 7. 相关文件

| 文件 | 说明 |
|---|---|
| `start-dspark-long.sh` | 长上下文启动脚本：`MODE=extrap\|draftmaxpos\|draftyarn\|yarn` + `ALGO=DSPARK\|DFLASH` + `KV_DTYPE` / `MEM_FRACTION` / `AUTOTUNE` |
| `start-long.sh` | MTP 路线（对照组） |
| `make-yarn-variant.py` | 生成 target / draft 变体（`--draft` / `--draft-keep-rope` / `--dflash2`） |
| `bench-long-ctx.py` | 大海捞针（报告真实 prompt_tokens / TTFT / 命中） |
| `verify-*.py` | 零停机预检：复现参数污染崩溃 + 验证各变体能否加载 |
| `LONG-CONTEXT-RESEARCH.md` | 调研与社区说法逐条裁决（含 hasso5703 / MiaAI / 上游 issue 链接） |
| `../notes/qwen38-long-context-1m.md` | 完整 worklog + 11 条踩坑清单 |
