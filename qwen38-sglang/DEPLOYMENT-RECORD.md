# Qwen3.8-27B 部署全流程记录（SGLang + DSpark, DGX Spark GB10）
> 部署时间: 2026-09-09 21:36 - 21:50 (HKT)  状态: ✅ 已上线运行
> 本文档为完整部署记录: 环境 -> 准备 -> 执行 -> 验证 -> 运维 -> 回滚, 含全部命令与实测结果。
>
> ⚠️ **2026-09-18 更新（照抄前必读）**：本文是 **2026-09-09 首版部署**（DSpark + 262K）的现场记录。现役形态已变：**DFlash2 + 1,010,000 上下文**（参数见 [dgx-spark-model-service.md](../dgx-spark-model-service.md) §8.1 / [LONG-CONTEXT-CUTOVER.md](LONG-CONTEXT-CUTOVER.md)），
> 且**并发上限已从 8 收到 `--max-running-requests 4`**（+ mamba 池按 并发×12；DFlash2 草稿采样器缓冲区按 decode CUDA graph 的 bs=4 分配，并发 8 时 bs≥5 必崩）——本文中的 `MAX_CONCURRENT_REQUESTS=8` 与 `并发 x 4` 的池算法都是首版值，**不要直接照抄**。定案与验证：[notes/qwen38-dflash2-crash-20260918.md](../notes/qwen38-dflash2-crash-20260918.md)。

## 1. 目标
将现网 ornith 模型（vLLM, Ornith-1.5-35B-A3B-NVFP4）切换为 Qwen3.8-27B（SGLang + DSpark 投机解码），
**对用户最小影响: 请求地址不变, 仅模型名改变**。

## 2. 环境
| 项 | 值 |
|---|---|
| 主机 | spark-XXXX, NVIDIA DGX Spark (GB10), aarch64 |
| GPU | 单卡 nvidia0, 128GB 统一内存, driver 580.173.02 |
| CPU | 20 核 (10x Cortex-X5 大核 5-9,15-19 + 10x A725 小核) |
| 内存/磁盘 | 121GB RAM / 3.4TB 空闲 |
| Docker | 29.2.1 (NVIDIA Container Toolkit 可用) |
| 网络 | DockerHub / HuggingFace 可达 |

## 3. 架构（切换后, 与切换前对齐）
```
用户 ──► A. 直连: <LAN-IP-1>/<LAN-IP-2>:8888, <Tailscale-IP>:8888 (Tailscale)
             │  vllm-open-proxy.py (纯TCP转发, 未改动)
             └──► 127.0.0.1:8888 ── 容器 [qwen38-27b-sglang] (host网络)
用户 ──► B. switchyard 0.0.0.0:4000 (systemd user 服务) ──► 同上 8888 (本地 Qwen3.8)
                                                          └─► OpenRouter/DeepSeek (云, 未改动)
```

## 4. 最终部署参数（当前生效）
### 4.1 镜像与模型
- 镜像: lmsysorg/sglang:qwen38-27b (digest sha256:febfb971... , arm64, 与 MiaAI 实测仓库 pin 一致)
- 主模型: RadixArk/Qwen3.8-27B-NVFP4 (packed FP4 lm_head, 21GB, snapshot 319f741c) — 2026-09-09 22:50 由 BF16-LMHead 版更换(性能优化, 见 7.4)
- 草稿模型: RadixArk/Qwen3.8-27B-DSpark (3.5GB, snapshot b9a5dbdf), DSPARK 投机解码 (block 7 / 验证窗口 8 tokens)
- 缓存目录: /home/<user>/projects/qwen38-model-cache (预下载, 容器内映射 /root/.cache/huggingface)

### 4.2 启动脚本（start.sh 全文, 即当前容器配置）
```bash
#!/usr/bin/env bash
# Qwen3.8-27B (SGLang + DSpark) — DGX Spark / GB10 (aarch64)
#
# 依据:
#   官方 cookbook (DGX Spark cell): https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
#   同硬件实测仓库: MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark (start-dspark.sh 的 DSPARK 参数栈)
#
# 关键适配(与现网 vLLM ornith 无缝对接):
#   - host 网络 + 绑定 127.0.0.1:8888 —— 与现网 vLLM 完全同位置:
#     * switchyard (routes.toml base_url http://localhost:8888/v1) 不用改
#     * vllm-open-proxy.py (外部 IP 8888 -> 127.0.0.1:8888) 不用改
#     * 切勿用 0.0.0.0:8888: 会和代理脚本抢端口, 导致外部直连挂掉
#   - 模型预下载在 /home/<user>/projects/qwen38-model-cache (准备阶段), 首次启动不再在线下载
#   - served-model-name = qwen3.8-27b, 与 routes.toml 中 target id 严格一致
#
# 用法: ./start.sh   停止: ./stop.sh

set -euo pipefail

IMAGE="lmsysorg/sglang:qwen38-27b"
CONTAINER_NAME="qwen38-27b-sglang"
MODEL_ID="RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead"   # BF16 lm_head 版(官方 cookbook 默认)
DRAFT_ID="RadixArk/Qwen3.8-27B-DSpark"              # DSpark 草稿 (~2.7GB, 已预下载)
SERVED_MODEL_NAME="qwen3.8-27b"
HOST="127.0.0.1"
PORT="8888"

# 并发与 GDN 状态池: 池 = 并发 x S; extra_buffer_lazy + 默认锁 => S=4
MAX_CONCURRENT_REQUESTS="${MAX_CONCURRENT_REQUESTS:-8}"
MAMBA_CACHE_SIZE=$(( MAX_CONCURRENT_REQUESTS * 4 ))

HF_CACHE="/home/<user>/projects/qwen38-model-cache"                # 预下载缓存(<user> 可写, 不影响 ornith 的旧缓存)
TRITON_CACHE="/home/<user>/projects/qwen38-model-cache/.triton"
CPUSET="5-9,15-19"   # GB10 的 10 个 Cortex-X5 大核(实测 +2~7% decode)

LOG_FILE="$(cd $(dirname $0) && pwd)/.sglang.log"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "容器 ${CONTAINER_NAME} 已在运行 (${LOG_FILE})"
  exit 0
fi
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
mkdir -p "${HF_CACHE}" "${TRITON_CACHE}"

echo "启动 SGLang: ${MODEL_ID} + DSpark(${DRAFT_ID})"
echo "监听 ${HOST}:${PORT}  served-model-name=${SERVED_MODEL_NAME}  并发=${MAX_CONCURRENT_REQUESTS} (mamba池=${MAMBA_CACHE_SIZE})"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --privileged \
  --gpus all \
  --shm-size 32g \
  --cpuset-cpus "${CPUSET}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${TRITON_CACHE}:/root/.triton" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --mem-fraction-static 0.85   # 2026-09-09: 0.90 曾致统一内存 swap 抖动, 降为 0.85 \
  --attention-backend flashinfer \
  --chunked-prefill-size 8192 \
  --disable-prefill-cuda-graph \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-dtype bfloat16 \
  --mamba-full-memory-ratio 4.21 \
  --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-mamba-cache-size "${MAMBA_CACHE_SIZE}" \
  --max-running-requests "${MAX_CONCURRENT_REQUESTS}" \
  --context-length 262144 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "${DRAFT_ID}" \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --speculative-num-draft-tokens 8 \
  --enable-torch-compile --torch-compile-max-bs 4 \
  --cuda-graph-max-bs-decode 4 \
  --num-continuous-decode-steps 2 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --sampling-defaults model \
  --enable-metrics \
  --host "${HOST}" \
  --port "${PORT}" \
  >/dev/null

echo "等待 HTTP 就绪 (http://127.0.0.1:${PORT}/v1/models) ..."
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "!! 容器启动失败, 日志最后 50 行:"; docker logs --tail 50 "${CONTAINER_NAME}"
    exit 1
  fi
  sleep 5
done
echo "SGLang 就绪。日志: ${LOG_FILE}  (docker logs -f ${CONTAINER_NAME})"

### 4.3 关键参数说明
| 参数 | 值 | 来源/原因 |
|---|---|---|
| --network host, --ipc host, --privileged, --gpus all | - | 与旧 vLLM 容器同模式; GB10 需 privileged |
| --cpuset-cpus 5-9,15-19 | - | 绑定 X5 大核, 实测 +2~7% decode |
| --host 127.0.0.1 --port 8888 | - | 与旧 vLLM 同位置, TCP 代理/switchyard 零改动 (勿用 0.0.0.0, 会抢代理端口) |
| --served-model-name qwen3.8-27b | - | 与 routes.toml target id 严格一致 |
| --mem-fraction-static 0.90 | - | MiaAI DSPARK 实测值 (配 torch.compile) |
| --attention-backend flashinfer | - | SM120/SM121 必须 (trtllm_mha 仅 SM100) |
| --speculative-algorithm DSPARK | - | 官方特性; 代码/工具调用快, 长文比 MTP 慢 |
| --speculative-dspark-block-size 7 / --speculative-num-draft-tokens 8 | - | 草稿自带 block_size=7, 验证窗口 8 |
| --speculative-draft-model-quantization unquant | - | 草稿为 BF16, 必须 unquant |
| --mamba-full-memory-ratio 4.21 + --max-mamba-cache-size 32 | - | GDN 状态池 = 并发8 x S4; 比默认0.9更准 (并发不 clamp) |
| --mamba-radix-cache-strategy extra_buffer_lazy | - | 同硬件实测优于 extra_buffer |
| --kv-cache-dtype fp8_e4m3 | - | NVFP4 权重自带校准 scale, KV 省一半 |
| --mamba-ssm-dtype bfloat16 | - | 状态槽 78.4MB (fp32 会翻倍) |
| --context-length 262144 | - | 原生上下文; YaRN >262K 与 DSpark 不兼容 |
| --reasoning-parser qwen3 / --tool-call-parser qwen3_coder | - | reasoning_content + 结构化 tool_calls |
| --enable-torch-compile --torch-compile-max-bs 4 --cuda-graph-max-bs-decode 4 --num-continuous-decode-steps 2 | - | MiaAI 实测提参 |

## 5. 准备阶段（零影响）
1. docker pull lmsysorg/sglang:qwen38-27b (~19GB 内容, 38.6GB 磁盘)
2. hf 预下载: 主 23GB (22 文件) + 草稿 3.5GB (6 文件), 未认证下载 (公开仓库)
3. 校验: routes.toml.next 经 switchyard-server --dry-run (需 export VLLM_API_KEY/OPENROUTER_API_KEY)
4. 生产文件: start.sh / stop.sh / routes.toml.next / CUTOVER.md / MIGRATION-NOTES.md

## 6. 切换执行实录
| 时间(HKT) | 命令/动作 | 结果 |
|---|---|---|
| 21:36 | cp /home/<user>/routes.toml /home/<user>/routes.toml.pre-migrate-09092136 | OK |
| 21:37 | docker stop ornith-b12x-serve (容器保留) | Exited(0) |
| 21:37 | ./start.sh (后台, 容器名为 qwen38-27b-sglang) | - |
| 21:45 | HTTP 就绪 | 启动共 ~8 分钟 (权重 23s, FlashInfer autotune, torch.compile 首编译) |
| 21:46 | 直连冒烟 4 项 | 全部通过 |
| 21:47 | switchyard-server --dry-run routes.toml.next | server OK: cloud-test, ornith-local, qwen3.8-27b, switchyard |
| 21:48 | cp routes.toml.next /home/<user>/routes.toml && systemctl --user restart switchyard | active (running) |
| 21:49 | 全链路验证 4 项 | 全部通过 |
| 21:52 | routes.toml 过时注释头清理 | OK (内容未变) |

执行中踩坑与修复:
1. TRITON_CACHE 默认 /home/<user>/.cache/triton 在工作区外被沙箱拒 -> 改 /home/<user>/projects/qwen38-model-cache/.triton
2. dry-run 需 export 两个 API key (systemd 单元内才有) -> 校验时 export

## 7. 验证结果
### 7.1 直连 8888 冒烟
- /v1/models: [('qwen3.8-27b', 262144)] ✅ (旧 vLLM 的 1M 上下文 -> 262K)
- chat: content="2", reasoning_content 有值, finish=stop ✅
- 工具调用: get_weather(city=北京) 解析为结构化 tool_calls ✅
- 并发钳制: 日志 max_running_requests=8 未被 clamp; max_mamba_cache_size=32 ✅
### 7.2 全链路 (switchyard 4000 / 外部直连)
- 4000 /v1/models: cloud-test, ornith-local, qwen3.8-27b, switchyard ✅
- 路由日志: requested qwen3.8-27b / ornith-local / switchyard -> selected 均为 qwen3.8-27b ✅
- 外部直连 <LAN-IP-1>:8888 (经 TCP 代理): 返回 qwen3.8-27b ✅
- 内容输出: "北京是中国的首都, 一座融合千年皇家历史与现代都市活力的北方文化名城。" (finish=stop) ✅
### 7.3 运行时指标 (部署后 ~15min 采样)
- 已处理请求: 非流式 7 个 + 流式若干; TTFT 中位 <1s (0.8s 桶内 2 个)
- 可用 GPU 内存: 9.01GB; KV 池 fp8 ~135万 token (20.69GB+6.47GB); Mamba 池 32 slots


### 7.4 性能优化 (2026-09-09 22:50~23:20): NVFP4 + mem 0.85
- 变更: 模型 BF16-LMHead -> RadixArk/Qwen3.8-27B-NVFP4 (packed FP4 head), mem-fraction 0.90 -> 0.85, 增加 torch inductor 编译缓存挂载
- 实测 (同协议 400 token 单流):
| 场景 | BF16 head + 0.90 (旧) | NVFP4 + 0.85 (新) |
|---|---|---|
| 非思考 | 14.9 tok/s | **25.4 tok/s** (+70%) |
| 思考 | 13.9 tok/s | **21.9 tok/s** (+58%) |
| 非思考(复测) | 22.3 tok/s (稳定, 无离群) |
- 同硬件参考基线: DSpark 16.2 / MTP 21.0 (MiaAI) -> 新配置已超过基线
- 内存: 0.90 时 swap 6.9GB -> 0.85 时 4.3GB 且 available 回升
- 教训: start.sh 参数区严禁插入行内注释 (\ 续行 + # 会吞掉后续全部参数, 曾导致容器以默认参数启动) — 注释只放头部

## 8. 当前运行状态 (2026-09-09 23:20)
| 项 | 值 |
|---|---|
| 容器 | qwen38-27b-sglang, Up (lmsysorg/sglang:qwen38-27b), host 网络 |
| 旧容器 | ornith-b12x-serve, Exited (保留, 未删) |
| switchyard | active (running), 0.0.0.0:4000 |
| TCP 代理 | vllm-open-proxy.py PID 25344, 未改动 |
| 评测模型 | Qwen3-8B-AWQ 127.0.0.1:8889, 未触碰 |

## 9. 用户可见变化
- **仅一处**: model 字段 ornith-local -> qwen3.8-27b
- 灰度期: ornith-local 仍可用 (同样打到 Qwen3.8), 不着急改配置; 1-2 周后删除该 route
- 地址/端口/auth/OpenAI+Anthropic 协议/streaming/工具调用 全部不变

## 10. 日常运维
```bash
cd /home/<user>/projects/qwen38-sglang
./start.sh                 # 启动 (已就绪自动跳过); ./stop.sh 停止
docker logs -f qwen38-27b-sglang        # 看日志 (/.sglang.log 也在)
curl -s http://127.0.0.1:8888/metrics   # Prometheus 指标 (--enable-metrics)
curl -s http://127.0.0.1:8888/v1/models # 健康检查
# 调整并发: MAX_CONCURRENT_REQUESTS=16 ./start.sh (池 = 并发 x 4 自动算)
# 检查调度上限是否被 clamp: docker logs qwen38-27b-sglang | grep max_running_requests
systemctl --user restart switchyard    # 路由配置改动后需重启 (30s drain)
```

## 11. 回滚 (任一步失败, <1 分钟)
```bash
docker stop qwen38-27b-sglang
docker start ornith-b12x-serve
cp /home/<user>/routes.toml.bak /home/<user>/routes.toml   # 8/23 全量旧配置
systemctl --user restart switchyard
```

## 12. 注意事项
- 上下文: 262K 原生; >262K 需 YaRN + 切 MTP 栈 (YaRN 与 DSpark 不兼容)
- DSpark 特长: 代码/数学/工具调用; 长文 prose 比 MTP 慢 (同镜像可切: EXTRA 参数栈见 CUTOVER.md)
- 会话内 tool call 格式: vLLM 曾用 qwen3_xml, 现为 qwen3_coder -> 都是标准 OpenAI tool_calls, 客户端无感
- 首次/重启冷启动 ~8 分钟 (torch.compile); 如需更快可去掉 --enable-torch-compile (损失吞吐)

## 附录 A: 当前生效 routes.toml
```toml
# ===== 当前生效配置 (2026-09-09 21:48 部署, 迁移自 vLLM/ornith -> SGLang/Qwen3.8-27B) =====
# 用户侧: 模型名 ornith-local -> qwen3.8-27b (地址/端口/auth/协议全不变)
# 灰度: ornith-local 暂时也指向新模型 (待 1-2 周后删除该 route)
schema_version = 1

[llm_clients.vllm_ornith]            # 保留: 回滚时旧后端就绪即可秒切
format = "openai_chat"
base_url = "http://localhost:8888/v1"
api_key_env = "VLLM_API_KEY"

[llm_clients.sglang_qwen38]
format = "openai_chat"
base_url = "http://localhost:8888/v1"   # 端口不变, 后端从 vLLM 换成 SGLang
api_key_env = "VLLM_API_KEY"

[llm_clients.openrouter]             # 云路由不动
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.ornith]                     # 保留(回滚用)
id = "ornith-1.5-35b-a3b-nvfp4"
llm_client = "vllm_ornith"

[targets.qwen38]
id = "qwen3.8-27b"                   # 必须与 SGLang --served-model-name 完全一致
llm_client = "sglang_qwen38"

[targets.openrouter_strong]
id = "deepseek/deepseek-v4-flash"
llm_client = "openrouter"

[routes.ornith_local]                # 灰度期: 旧模型名也打到新模型, 让没改配置的客户端无感
id = "ornith-local"
type = "passthrough"
target = "qwen38"                    # ← 灰度期指向新后端; 1-2 周后删掉整个 route

[routes.qwen_local]
id = "qwen3.8-27b"                   # 新模型名
type = "passthrough"
target = "qwen38"

[routes.cloud_test]
id = "cloud-test"
type = "passthrough"
target = "openrouter_strong"

[routes.smart]                       # 智能路由的"便宜档"换成新模型, 云路由保留
id = "switchyard"
type = "stage_router"
efficient_target = "qwen38"
capable_target = "openrouter_strong"
picker = "efficient_first"
confidence_threshold = 0.05```