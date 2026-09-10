# Qwen3.8-27B (SGLang + DSpark) 切换记录与调研备忘

> 记录日期: 2026-09-09
> 目标: 将现网 ornith (vLLM, Ornith-1.5-35B-A3B-NVFP4) 切换到 Qwen3.8-27B (SGLang + DSpark 投机解码),
> 并对用户做到最小影响: **请求地址不变, 仅模型名变化**。
> 状态: ✅ **已切换完成 (2026-09-09 21:48)** —— SGLang+Qwen3.8-27B+DSpark 已在 8888 服务, switchyard 新路由已生效, 全链路验证通过。
> 📄 完整部署记录(含全部命令/参数/验证/运维/回滚): 同目录 **DEPLOYMENT-RECORD.md**

---

## 一、当前系统现状（实测核查记录）

### 1.1 硬件与网络
- 机器: NVIDIA **DGX Spark (GB10)**, 架构 **aarch64**, 单 GPU (**/dev/nvidia0**, 128GB 统一内存), dirver 580.173.02
- CPU: 20 核 (10x Cortex-X5 大核 + 10x A725 小核)
- 内存 121GB (常驻约 101GB), 磁盘 3.7TB (空闲 3.4TB)
- 网络: DockerHub / HuggingFace / ModelScope 均可达
- 主机名: spark-XXXX

### 1.2 服务架构（三层）
```
用户 ──► A. 直连: <LAN-IP-1>:8888 / <LAN-IP-2>:8888 / <Tailscale-IP>:8888 (Tailscale)
             │  vllm-open-proxy.py (纯TCP转发, PID 25344)      ← 零改动
             └──► 127.0.0.1:8888 ── vLLM 容器 [ornith-b12x-serve]
                  (host网络, 1M上下文, MTP spec, GPU util 0.65)
用户 ──► B. switchyard: 0.0.0.0:4000 (systemd user服务, routes.toml) ─┼──► 同上 8888 (本地 ornith)
                                                                      └──► OpenRouter (DeepSeek 云)
```

### 1.3 关键进程与服务
| 组件 | PID | 说明 |
|---|---|---|
| vLLM (容器 ornith-b12x-serve, root) | 231436 (+EngineCore 231870) | vllm serve, host网络, 127.0.0.1:8888, GPU util 0.65 |
| vllm-open-proxy.py (<user>, 8/23 启动) | 25344 | 外部IP:8888 -> 127.0.0.1:8888 纯TCP转发 |
| switchyard-server (<user>, 8/23 12:27 启动) | 20716 | 0.0.0.0:4000, systemd user 服务 |
| Qwen3-8B-AWQ (<user>, 评测用, 别动) | 300660 | 127.0.0.1:8889 |

### 1.4 vLLM 当前启动参数（实录）
```
vllm serve /root/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-NVFP4/snapshots/0f0b1b59...
  --served-model-name ornith-1.5-35b-a3b-nvfp4
  --host 127.0.0.1 --port 8888 --trust-remote-code --enable-prefix-caching
  --spec-method mtp --spec-tokens 1 --max-num-seqs 4 --moe-backend b12x
  --gpu-memory-utilization 0.65 --max-model-len 1048576
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
```
容器: eugr/spark-vllm-b12x:latest, Network=host, Ipc=host, --gpus all
挂载: /home/<user>/.cache/vllm, /home/<user>/.cache/huggingface -> /root/.cache/huggingface, 以及
      /home/<user>/projects/Ornith-1.5-35B-A3B-DGX-Spark/patches/ 下的 b12x 内核补丁

### 1.5 switchyard 配置状态（重要发现）
- **磁盘上的 /home/<user>/routes.toml (8/27 17:13 修改) 与运行中配置不一致**:
  磁盘版只有 ornith-local 一条路由; 运行中进程(8/23)仍是旧全量配置。
  用户当前可用模型: `cloud-test`(OpenRouter/DeepSeek), `ornith-local`(本地vLLM), `switchyard`(智能路由: 本地优先, 云兜底)
- **全量旧配置完好保存在 /home/<user>/routes.toml.bak (8/27 17:11)**, 内容见附录 A。
- /tmp/routes_clean.toml (8/23) 是残缺草稿, 无参考价值。
- systemd user 服务: ~/.config/systemd/user/switchyard.service
  (环境变量: VLLM_API_KEY=<已脱敏>, OPENROUTER_API_KEY=sk-or-v1-…<已脱敏>)
- ⚠️ 若直接重启 switchyard 而不先恢复全量配置, cloud-test / switchyard 两个模型会消失。

### 1.6 健康检查（切换前基线）
- 直连 8888: `/v1/models` 返回 ornith-1.5-35b-a3b-nvfp4 ✅
- chat completion 冒烟: 正常返回 ✅
- ⚠️ 2026-09-09 日志出现 600s 超时(状态500) 与 510s 长响应 —— 现网偶发很慢, 是切换动机之一

---

## 二、目标方案（网友提供）与验证结论

### 2.1 原文命令（备查）
```
docker run -d --gpus all --name qwen38-27b --shm-size 32g -p 30000:30000
  -v /home/<原作者>/.cache/modelscope/hub/models/RadixArk/Qwen3.8-27B-NVFP4:/models/main-model
  -v /home/<原作者>/.cache/modelscope/hub/models/RadixArk/Qwen3.8-27B-DSpark:/models/draft-model
  --ipc=host
  lmsysorg/sglang:qwen38-27b
  sglang serve --trust-remote-code --served-model-name Qwen3.8-27B
  --model-path /models/main-model --mem-fraction-static 0.85
  --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer
  --context-length 262144 --max-running-requests 8 --chunked-prefill-size 4096
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --mamba-full-memory-ratio 0.4 --host 0.0.0.0 --port 30000
  --speculative-algorithm DSPARK
  --speculative-draft-model-path /models/draft-model
  --speculative-num-steps 5
  --speculative-draft-attention-backend flashinfer
  --mamba-radix-cache-strategy extra_buffer
  --mamba-ssm-dtype bfloat16
```

### 2.2 验证结论
| 项目 | 结论 |
|---|---|
| 镜像 lmsysorg/sglang:qwen38-27b | ✅ **真实存在**, 官方 cookbook 专用 build, 多架构(amd64+arm64), 2026-08-14 更新 |
| DSPARK 投机解码 | ✅ SGLang 正式特性 (官方 cookbook 有 Qwen3.8-27B 专章 + DGX Spark cell) |
| 主模型 HF | ✅ RadixArk/Qwen3.8-27B-NVFP4 (packed FP4 head), 公开, 约 22GB |
| 主模型 HF (推荐) | ✅ RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead (BF16 lm_head, cookbook默认), 公开, 约 24GB |
| 草稿模型 HF | ✅ RadixArk/Qwen3.8-27B-DSpark, 公开, 约 2.7GB |
| 同硬件参考 | ✅ **MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark** (GB10 实测仓库, start.sh / start-dspark.sh) |

### 2.3 网友命令的问题清单（逐条）
| 参数 | 判定 | 说明 |
|---|---|---|
| `--speculative-num-steps 5` | ❌ 错误 | 这是 EAGLE/MTP 的参数; DSPARK 用 `--speculative-num-draft-tokens 8` + `--speculative-dspark-block-size 7` (草稿自带 block_size=7, 可省略) |
| 缺 `--speculative-draft-model-quantization unquant` | ❌ 风险 | 草稿是 BF16(unquant), 不 pin 可能加载失败 |
| `--mamba-full-memory-ratio 0.4` | ⚠️ 风险 | GDN 状态池/KV 池分配失真, 会静默 clamp 并发。同硬件实测: 4.21 + 显式 `--max-mamba-cache-size = 并发 x 4` |
| `--mamba-radix-cache-strategy extra_buffer`(S=5) | ⚠️ 可改 | 同硬件实测更优: extra_buffer_lazy (S=4) |
| `--speculative-draft-attention-backend flashinfer` | ⚠️ 可能报错 | 该旗标一般用于 EAGLE 系; 主 `--attention-backend flashinfer` 已覆盖 (SM121 上 flashinfer 是必须的, ✅) |
| 端口 30000 + `-p` 映射 | ❌ 增改动 | 会迫使改 switchyard base_url 和 TCP 代理; 应 host网络 + 127.0.0.1:8888 |
| 挂载 /home/<原作者>/... | ❌ 本机无此路径 | 需改本机预下载缓存 (最终用 /home/<user>/projects/qwen38-model-cache) |
| `--mem-fraction-static 0.85` | ⚠️ 非实测值 | DSPARK 实测配方 0.90 |
| `--chunked-prefill-size 4096` | ⚠️ 非实测值 | 实测配方 8192 |
| 无 torch.compile / cuda-graph / 连续decode步数 | ⚠️ 缺优化 | 同硬件实测会加: --enable-torch-compile --torch-compile-max-bs 4 --cuda-graph-max-bs-decode 4 --num-continuous-decode-steps 2 |
| 无 cpuset 大核绑定 | ⚠️ 缺优化 | GB10 上绑定 5-9,15-19 (X5大核): +2~7% decode |
| `--reasoning-parser qwen3` / `--tool-call-parser qwen3_coder` | ✅ 正确 | 注意与现网 vLLM 的 qwen3_xml 不同: 工具调用内部格式变, 但客户端拿到的仍是标准 OpenAI tool_calls, 一般无感 |
| `--kv-cache-dtype fp8_e4m3` | ✅ | 与 cookbook 一致 |
| `--context-length 262144` | ✅ | 原生262K; ⚠️ YaRN(>262K)与 DSpark 不兼容, 若需长上下文须切 MTP |

### 2.4 参考来源
- SGLang 官方 cookbook (Qwen3.8-27B + DGX Spark): https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- 同款 GB10 实测仓库 (DSpark/MTP/DFlash2): https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark
- 镜像: https://hub.docker.com/r/lmsysorg/sglang/tags/qwen38-27b
- 模型: https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead , .../Qwen3.8-27B-DSpark
- DSPARK 特性发布: https://github.com/sgl-project/sglang/releases (DSpark: confidence-driven speculative decoding)

---

## 三、最小影响迁移设计

**原则: 换底下的引擎, 外面全不动。**

### 3.1 不变清单 (零改动)
- switchyard 服务: 0.0.0.0:4000, 地址/auth/OpenAI+Anthropic 协议, systemd user 服务
- vllm-open-proxy.py: 外部IP:8888 -> 127.0.0.1:8888 纯转发 (sglang 同样只绑 127.0.0.1:8888, 就不冲突)
- switchyard llm_client base_url: http://localhost:8888/v1
- 云路由: cloud-test (OpenRouter/DeepSeek), switchyard 智能路由的云兜底档
- 旧 vLLM 容器 ornith-b12x-serve: **只 stop 不删**, 权重保留, 随时回滚

### 3.2 只变清单
- 用户 model 字段: **ornith-local / ornith-1.5-35b-a3b-nvfp4 -> qwen3.8-27b**
- 灰度期: ornith-local 路由临时也指向新模型 (没改配置的客户端无感), 1-2 周后删除
- stage_router (switchyard 智能路由) 的 efficient_target: ornith -> qwen38

### 3.3 停机窗口与回滚
- 停机 = 模型加载(预计 5-10 分钟, 含首次 torch.compile) + switchyard 重启(数秒)
- 单 GPU 无法并存新旧模型, 无 A/B 并行; 通过"准备-切换-验证-可控回滚"保障
- 回滚 <1 分钟: docker stop qwen38-27b-sglang; docker start ornith-b12x-serve; 恢复 routes.toml.bak; restart switchyard

---

## 四、准备阶段执行记录（2026-09-09）

### 4.1 时间线
1. 核查现状 (进程/端口/容器/配置/健康)
2. 验证网友命令 (Docker Hub tag API、HF API、官方 cookbook、MiaAI 仓库)
3. 用户决策: ① 现在开始零影响准备 ② 主模型用 -BF16-LMHead 版
4. 启动后台任务: bash-1 = docker pull 镜像(~15GB); bash-5 = hf 下载主+草稿模型(~27GB)
5. 产出配置文件 (见 4.3)

### 4.2 踩坑记录
- huggingface-cli 旧命令 `huggingface-cli download` 已改为 `hf download` (新CLI) —— 曾报帮助文本
- DSH 文件沙箱仅允许写 /home/<user>/projects (工作区): `/home/<user>/.cache/hfdl-venv`、`/home/<user>/models` 均被拒,
  模型缓存最终放 /home/<user>/projects/qwen38-model-cache
- 模型缓存未再复用 /home/<user>/.cache/huggingface (该目录与 ornith 容器共用, 保持隔离)

### 4.3 产出文件清单
| 文件 | 说明 |
|---|---|
| /home/<user>/projects/qwen38-sglang/start.sh | 修正后启动脚本 (host网络+127.0.0.1:8888+DSPARK参数栈+cpuset) |
| /home/<user>/projects/qwen38-sglang/stop.sh | 停止脚本 |
| /home/<user>/projects/qwen38-sglang/routes.toml.next | 新 switchyard 路由草稿 (未启用) |
| /home/<user>/projects/qwen38-sglang/CUTOVER.md | 切换/回滚 runbook |
| /home/<user>/projects/qwen38-sglang/MIGRATION-NOTES.md | 本文档 |
| /home/<user>/projects/qwen38-model-cache/ | 模型预下载缓存 (进行中) |
| /home/<user>/projects/.hfdl-venv/ | HF 下载 CLI 环境 |

### 4.4 进度跟踪
- bash-1 拉镜像: ✅ **完成 (2026-09-09)**: lmsysorg/sglang:qwen38-27b, arm64, exit 0
- bash-5 下模型: ✅ **完成 (2026-09-09)**: 主模型 23G (RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead, 22文件, snapshot 009632fe) + 草稿 3.5G (RadixArk/Qwen3.8-27B-DSpark, 6文件, snapshot b9a5dbdf), 共 26G, 均含 config.json + safetensors, 离线启动无需再下载
- 线上服务: 未受影响

---

## 五、切换步骤摘要（详见 CUTOVER.md）
1. 备份 routes.toml + 记录旧容器状态
2. docker stop ornith-b12x-serve
3. cd /home/<user>/projects/qwen38-sglang && ./start.sh (自动等就绪)
4. 直连冒烟: /v1/models + chat + tools + reasoning_content + 确认日志 max_running_requests>=8
5. cp routes.toml.next /home/<user>/routes.toml && systemctl --user restart switchyard
6. 全链路验证: 4000 四个模型 + 外部IP直连
7. 通知用户改 model 名 (灰度期 ornith-local 仍可用)



---

## 七、切换执行实录 (2026-09-09 21:36~21:50)

### 7.1 时间线
| 时间(HKT) | 动作 | 结果 |
|---|---|---|
| 21:36 | 备份 `/home/<user>/routes.toml` -> `routes.toml.pre-migrate-09092136` | OK |
| 21:37 | docker stop ornith-b12x-serve (容器保留) | Exited(0) |
| 21:37 | 启动 qwen38-27b-sglang (start.sh) | 含环境变量/沙箱路径修正后成功 |
| 21:45 | HTTP 就绪 (127.0.0.1:8888/v1/models) | **启动~8分钟** (含 FlashInfer autotune + torch.compile 首编译) |
| 21:46 | 直连冒烟: chat / tool_call / reasoning_content / 并发 | 全部通过; max_running_requests=8 未 clamp |
| 21:47 | 新配置 dry-run 校验 | server OK: cloud-test, ornith-local, qwen3.8-27b, switchyard |
| 21:48 | 部署 routes.toml + restart switchyard | active (running) |
| 21:49 | 全链路验证 | 4 模型列表 / 三路径 chat / 外部直连 全部通过 |

### 7.2 验证结果(实录)
- 4000 模型列表: cloud-test, ornith-local, **qwen3.8-27b**, switchyard
- switchyard 日志确认路由: requested qwen3.8-27b/ornith-local/switchyard -> selected **qwen3.8-27b** (智能路由本地档、灰度别名均生效)
- 直连 8888(经TCP代理): model=qwen3.8-27b 正常
- 工具调用: get_weather 结构化 tool_calls 解析 OK
- reasoning_content 字段: OK
- 启动细节: 主模型 23s 加载(命中本地snapshot), KV池 fp8 ~135万token, Mamba池 32 slots, 就绪时可用 GPU 内存 9.01GB
- 老 vLLM 容器或nith 模型: 保留未删 (rollback 随时 `docker start ornith-b12x-serve`)

### 7.3 执行中的两个小坑(已解决)
- start.sh 的 TRITON_CACHE 默认 /home/<user>/.cache/triton (工作区外 → 沙箱拒写), 已改为 /home/<user>/projects/qwen38-model-cache/.triton
- 配置校验需环境变量: switchyard --dry-run 检查 OPENROUTER_API_KEY/VLLM_API_KEY (systemd 单元里有, shell 里没有) — 校验时需 export

## 六、待办 / 开放问题
- [x] 模型下载完成 (主 23G + 草稿 3.5G, snapshot 校验 OK)
- [x] 镜像拉取完成 (lmsysorg/sglang:qwen38-27b, arm64)
- [x] 选择低峰切换窗口 (2026-09-09 21:36, 用户指令'现在跑')
- [x] 正式切换 + 全链路验证 (2026-09-09 21:48)
- [ ] 灰度期结束后删除 ornith-local 旧路由
- [ ] 长上下文需求? >262144 需 YaRN + 切 MTP 栈

---

## 附录 A: /home/<user>/routes.toml.bak (运行中的全量配置, 迁移基础)
```toml
schema_version = 1

[llm_clients.vllm_ornith]
format = "openai_chat"
base_url = "http://localhost:8888/v1"
api_key_env = "VLLM_API_KEY"

[llm_clients.openrouter]
format = "openai_chat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"

[targets.ornith]
id = "ornith-1.5-35b-a3b-nvfp4"
llm_client = "vllm_ornith"

[targets.openrouter_strong]
id = "deepseek/deepseek-v4-flash"
llm_client = "openrouter"

[routes.ornith_local]
id = "ornith-local"
type = "passthrough"
target = "ornith"

[routes.cloud_test]
id = "cloud-test"
type = "passthrough"
target = "openrouter_strong"

[routes.smart]
id = "switchyard"
type = "stage_router"
efficient_target = "ornith"
capable_target = "openrouter_strong"
picker = "efficient_first"
confidence_threshold = 0.05
```

## 附录 B: 当前磁盘 /home/<user>/routes.toml (简化版, 8/27 17:13, 未生效)
```toml
schema_version = 1

[llm_clients.vllm_ornith]
format = "openai_chat"
base_url = "http://localhost:8888/v1"
api_key_env = "VLLM_API_KEY"

[targets.ornith]
id = "ornith-1.5-35b-a3b-nvfp4"
llm_client = "vllm_ornith"

[routes.ornith_local]
id = "ornith-local"
type = "passthrough"
target = "ornith"
```


---

## 八、性能排查记录 (2026-09-09 22:50~23:00)

### 8.1 实测环境
- 单流基准(400 token 中文 essay prompt, 无并发): 
  - 非思考: **14.9 tok/s** (26.9s)
  - 思考(默认): **13.9 tok/s** (400/400 token 全部为 reasoning!)
  - 非思考(复测): 5.0 tok/s — **原因查明: 撞上并发大请求**(下方)
- 同硬件参考基线 (MiaAI-Lab 实测, 同款 GB10, 单流 400 tok essay):
  - DSpark+NVFP4: 16.2 tok/s (非思考) / 16.9 (思考)
  - MTP 3/1/4 + NVFP4: 21.0 tok/s (非思考) / 16.9 (思考)
  - DSpark+FP8: 11.4~11.5 tok/s
- 结论: 我们 14.9 vs 基线 16.2 (-8%), **配置并非明显劣化, 属于同档**; -8% 主要来自 BF16 lm_head 每次 decode 的额外 matmul 与负担。

### 8.2 并发调度证据 (sglang 日志 14:58)
- 存在真实大上下文请求: Prefill 8192x5=~24K tokens (~1000 tok/s), 期间 decode 与基准请求共享
- 调度数据: accept rate 0.26~0.47, accept len 2.85~4.28, 多请求合并 decode gen throughput 8~59 tok/s
- 结论: 5.0 tok/s 的离群值 = 并发大请求竞争, 非配置故障

### 8.3 关键风险: 统一内存压力 (GB10)
- 主机: 121GB 中 **113GB 已用, 8GB available, swap 6.9GB 已用** (16G swap)
- 成因: --mem-fraction-static 0.90 静态预留 ~115GB + judge(Qwen3-8B-AWQ) + 宿主系统
- 后果: 内存页换出/换入抖动, 造成间歇性卡顿(比算法慢更影响体感)
- ✅ 建议: mem-fraction-static 0.90 -> 0.85 (网友同值), 可选停 judge 释放 2~4GB

### 8.4 与网友命令逐条差异(性能相关)及判定
| 项 | 网友 | 我们 | 对速度影响 |
|---|---|---|---|
| 模型 | RadixArk/...-NVFP4 (packed FP4 head, ~22GB) | ...-NVFP4-BF16-LMHead (BF16 head, ~24GB) | BF16 head 每 token 略慢(<10%), 显存多占 ~3GB |
| mem-fraction | 0.85 | 0.90 | ⚠️ 0.90 挤爆统一内存/swap, 间歇卡顿主因候选 |
| mamba ratio | 0.4 (无显式 pin) | 4.21 + max-mamba-cache-size 32 | 池子分配差异, 不影响单流; 我们有显式 pin 更可控 |
| chunked-prefill | 4096 | 8192 | 4096 大输入 TTFT 略平滑; 8192 吞吐略高 (实测 prefill ~1000 tok/s) |
| speculative-num-steps 5 | 有(无效参数,EAGLE用) | N/A | 无差异(被忽略; 双方均按草稿 block 7/验证8) |
| draft-attention-backend | 显式 flashinfer | 继承主 backend (日志: Overriding... flashinfer) | 等价 |
| torch.compile / cuda graph / 连续decode | 无 | 有 | 我们应更快(抵消了 BF16 head 的损失) |
| cpuset 大核绑定 | 无 | 有 (5-9,15-19) | 我们应更快 (+2~7%) |
| 思考模式 | 模板默认开 | 模板默认开 | 相同; 但**用户体感慢的主因**: 每个请求先烧几百 reasoning token |
| DSpark vs MTP | DSpark | DSpark | 同; 聊天/长文 MTP +30%, 代码/工具 DSpark 快(MiaAI: code 51.5 vs 34.5) |

### 8.5 结论与建议(按收益排序)
1. **mem-fraction-static 0.85** (+可选停 judge): 解除 swap 抖动 — 间歇性卡顿改善最直接 (需重启 ~8min)
2. **默认关闭思考**: --default-chat-template-kwargs '{"enable_thinking": false}' 或客户端传参; 简单对话的 token 浪费立减 (行为变更, 需拍板)
3. **MTP A/B 对比**: 若主要负载是聊天/长文, 切 MTP(同镜像改参数) 预期 +20~30%; 代码/工具负载保持 DSpark
4. 可选: 换网友同款 FP4 head 模型(下载 ~22GB) — 收益小(<10%), 不建议优先

### 8.6 优化执行结果 (2026-09-09 23:20): NVFP4 + mem 0.85
- 新配置: RadixArk/Qwen3.8-27B-NVFP4 (packed FP4 head, 21GB) + --mem-fraction-static 0.85 + torch 编译缓存持久化
- 实测: 非思考 **25.4 tok/s** (旧 14.9, +70%) / 思考 **21.9** (旧 13.9, +58%) / 复测稳定 22.3 (无离群)
- 结论: ①打包FP4 head + ②解除统一内存 swap 压力后, 吞吐超过 MiaAI 同硬件 DSpark 基线(16.2) 与 MTP(21.0)
- 全链路 (switchyard 4000) 确认无影响; 路由层零改动
