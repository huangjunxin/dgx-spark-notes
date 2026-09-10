# 长上下文切换记录：262,144 → 506,000（保留 DSpark）

> 执行时间: 2026-09-10 09:24 - 10:35 (HKT) · 主机 spark-XXXX (DGX Spark GB10)
> 结果: ✅ 已上线运行 · 上下文 **506,000** · 投机解码仍是 **DSpark** · 实测吞吐与 262K 基线持平

## 1. 结论（先看这个）

| 项 | 切换前 | 切换后 |
|---|---|---|
| 上线上下文 | 262,144 | **506,000** |
| 投机解码 | DSpark (block 7 / 8 draft) | **DSpark（不变）** |
| bench.py 非思考 400tok | 21.5 / 20.2 tok/s | **24.8 / 23.0 tok/s** |
| bench.py 思考 400tok | 24.8 tok/s | 22.8 tok/s（该轮 reasoning 403 tok vs 基线 76 tok，不可直接比） |
| 长文召回 | 未验证（>262K 不可用） | **310,427 tok 命中 ✅ / 465,855 tok 命中 ✅** |
| 接受率 accept len (rate) | 约 2.8-3.95 (0.44-0.57)* | 2.05-3.23 (0.15-0.32) |
| KV 池 | 约 1.30M token @mem 0.85 | **1,131,401 token @mem 0.70** |
| 宿主机可用内存 | 18 GB（swap 用 2 GB） | **27 GB（swap 用 0.3 GB）** |

\* 基线接受率来自切换前旧容器日志尾部样本（与 bench.py 同窗口），非严格配对测量。

## 2. 关键设计：为什么不用 `--json-model-override-args`

SGLang 会把该参数**转发给独立的 DSpark 草稿模型**（`sglang/srt/speculative/dspark_components/dspark_config.py:35,136`），
把 `{"text_config": {...}}` 当扁平 key 盖到草稿 config 上 → 草稿多出一个 dict 型 `text_config` →
transformers 的 YaRN 校验取不到 `max_position_embeddings` → 启动崩溃（上游 issue sglang#33505，至今 open）。

**本机采用的绕法（= Qwen 模型卡的方案一 + hasso5703/dgx-spark-qwen38 的现场做法）**：
把 YaRN 直接写进模型自己的 `config.json`，`--json-model-override-args` 保持 `{}`，草稿就不再被污染。

实现方式为**软链 farm**（零拷贝，不动 HF 缓存里的 blob，不需要重新下载）：

```
qwen38-model-cache/variants/
├── qwen38-nvfp4-yarn512/        ← target: text_config.max_position_embeddings=524288 + YaRN factor 2.0
├── qwen38-nvfp4-yarn1024/       ← target: 1048576 + factor 4.0（1M 档备用）
├── qwen38-dspark-maxpos-yarn512/← draft: 只抬 max_position_embeddings=524288, **rope 保持训练值** ← 本次使用
├── qwen38-dspark-yarn512/       ← draft: rope 换成与 target 同套 YaRN（对照用）
└── qwen38-dflash2-yarn{512,1024}← DFlash2 草稿（hasso/incoai 做法, 备用）
```

容器内路径：`/root/.cache/huggingface/variants/<name>`（复用已有挂载，无需额外 -v）。

## 3. 当前生效的启动命令（可复现）

```bash
cd /home/<user>/projects/qwen38-sglang
CONTEXT_LENGTH=506000 ./start-dspark-long.sh      # MODE 默认 draftmaxpos, ALGO 默认 DSPARK
```

等价要点（与旧 start.sh 的差异）：

| 参数 | 值 | 原因 |
|---|---|---|
| `--model-path` | `/root/.cache/huggingface/variants/qwen38-nvfp4-yarn512` | YaRN 内联在 config.json |
| `--speculative-draft-model-path` | `/root/.cache/huggingface/variants/qwen38-dspark-maxpos-yarn512` | 草稿只抬 max_pos，rope 不动 |
| `--json-model-override-args` | `{}` | 关键：不污染草稿 |
| `--context-length` | `506000` | YaRN factor 2.0 上界 524288，留 ~3.5% 余量（规避 sglang#33454 verify 窗口越界） |
| `-e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` | 是 | 允许超原生长度 |
| `--mem-fraction-static` | `0.70`（旧 0.85） | GB10 统一内存陷阱；旧配置 swap 已用 2GB |
| docker `--memory/--memory-swap` | `100g` | 防止把宿主机拖到 OOM 断电 |
| `--disable-flashinfer-autotune` | 是（长上下文自动加） | autotuner 在 GB10 上有 25-40GB 不可见瞬时分配 |
| `-e HF_HUB_OFFLINE=1` | 是 | 防止 Hub 重新解析把打过补丁的 config 覆盖回去 |
| 其余（flashinfer / kv fp8 / mamba bf16 / torch.compile / cpuset / tool-call-parser） | 与旧配置一致 | 行为不变 |

## 4. 验证实录

| 时间 | 项目 | 结果 |
|---|---|---|
| 09:24 | 首次尝试 `ALGO=DFLASH CONTEXT_LENGTH=1010000` | ❌ 失败两次：(a) DFLASH 不支持 `extra_buffer_lazy`（引擎断言，须 `extra_buffer`）；(b) 本镜像未注册 `DFlash2DraftModel` 架构 → DFlash2 需要官方 dev 镜像（已 `docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2` 备用） |
| 09:30 | `CONTEXT_LENGTH=506000 ./start-dspark-long.sh` | ✅ 就绪；`/v1/models` → `max_model_len=506000`；`max_total_num_tokens=1131401`；可用 GPU 内存 31.12 GB |
| 09:33 | bench.py ×3 | 24.8 / 22.8 / 23.0 tok/s（基线 21.5 / 24.8 / 20.2） |
| 09:41 | 大海捞针 71,383 tok | ✅ 命中，TTFT 52.7s |
| 09:49 | 大海捞针 **310,427 tok**（超原生窗口） | ✅ 命中，TTFT 568s（≈546 tok/s prefill） |
| 10:2x | 大海捞针 **465,855 tok**（贴近 506K 上界） | ✅ 命中，TTFT 1094s（≈426 tok/s prefill） |
| 10:3x | switchyard 4000 全链路 | ✅ `/v1/models` 正常、chat 正常返回 |

## 5. 回滚（<1 分钟切换 + 8-10 分钟冷启）

```bash
cd /home/<user>/projects/qwen38-sglang
./stop.sh && ./start.sh      # 回到 DSpark + 262144 + mem 0.85（原配置）
```

YaRN 变体是独立目录，回滚不需要还原任何缓存文件。

## 6. 已知限制与后续可选项

1. **接受率略有下降**（len 2.05-3.23 vs 基线 2.8-3.95）：草稿只抬了 max_pos，rope 与 target 的 YaRN 缩放不一致；但端到端吞吐没有退化。想要严格配对数据需要再切一次基线跑同一探针（约 20 分钟，2 次重启）。
2. **首 token 延迟与长度成正比**：466K 冷 prefill 18 分钟（约 426 tok/s）；同前缀二次请求走 radix cache 秒回。
3. **1M 档未上线**：变体（`*-yarn1024`）已就绪，但 1M 单请求约需 33GB KV + 草稿 KV，当前池 1.13M token 只是刚好；要上 1M 建议 `CONTEXT_LENGTH=1010000` 并按 hasso 配方压 mem-fraction/并发后再实测。
4. **DFlash2 备用路线**：官方 dev 镜像已拉取（`lmsysorg/sglang:dev-qwen38-27b-dflash2`），变体与脚本支线就绪；本镜像跑不了 DFlash2 的草稿架构。
5. `--json-model-override-args` **必须保持空**：一旦传 YaRN 覆盖，DSpark/DFlash2 草稿会再次崩溃（除非草稿 config 自带完整 `text_config`，见 `qwen38-dspark-cfg-textconfig`）。

---

## 7. 最终状态更新（2026-09-10 11:5x）：改为 **DFlash2 + 1M**

按用户要求切到 DFlash2（hasso5703 的现役栈）。DFlash2 需要官方 dev 镜像，本机已 `docker pull lmsysorg/sglang:dev-qwen38-27b-dflash2`（`0.0.0.dev1+g5f55db35e`，与 MiaAI 记录一致）。

### 7.1 当前生效配置

```bash
cd /home/<user>/projects/qwen38-sglang
IMAGE=lmsysorg/sglang:dev-qwen38-27b-dflash2 ALGO=DFLASH CONTEXT_LENGTH=1010000 ./start-dspark-long.sh
```

| 参数 | 值 |
|---|---|
| 镜像 | `lmsysorg/sglang:dev-qwen38-27b-dflash2` |
| target | `variants/qwen38-nvfp4-yarn1024`（YaRN factor 4.0, max_pos 1048576） |
| 草稿 | `variants/qwen38-dflash2-yarn1024`（z-lab/Qwen3.8-27B-DFlash2 + YaRN 烧进草稿 config） |
| 算法 | `DFLASH`，block 8，8 draft tokens，`--speculative-draft-model-quantization unquant` |
| 上下文 | `--context-length 1010000`（factor 4 上界 1048576，留 3.7% 余量） |
| KV | fp8_e4m3，池 **1,031,239 token**；mem-fraction 0.70；mamba `extra_buffer` 96 槽；可用 GPU 显存 30.4 GB |
| 其它 | `--json-model-override-args '{}'`、`SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`、`HF_HUB_OFFLINE=1`、docker 内存上限 100g、关 flashinfer autotune |

### 7.2 实测（同机同探针）

| 项 | DSpark 262K（原基线） | DSpark+YaRN 506K（中间臂） | **DFlash2 1M（现役）** |
|---|---|---|---|
| bench.py 非思考 400tok | 21.5 / 20.2 tok/s | 24.8 / 23.0 | **23.4 / 24.0** |
| bench.py 思考 400tok | 24.8 | 22.8 | **24.8** |
| 大海捞针 | 不可用 | 310,427 ✅ / 465,855 ✅ | **310,427 ✅ / 652,137 ✅** |
| 冷 prefill 首 token | — | 568s / 1094s | 568s / **1941s** |
| accept len (rate) | 2.8-3.95 (0.44-0.57)* | 2.05-3.23 (0.15-0.32) | 2.27-2.80 (0.18-0.26) |

结论：**DFlash2 吞吐与 DSpark 同级**（与 hasso5703、MiaAI 的公开实测一致），而长上下文只到 DFlash2 侧做到 1M + 652K 召回。

### 7.3 回滚

```bash
./stop.sh && ./start.sh                                            # 回到 DSpark + 262K + mem 0.85（最初配置）
./stop.sh && CONTEXT_LENGTH=506000 ./start-dspark-long.sh          # 回到 DSpark + 506K
```
