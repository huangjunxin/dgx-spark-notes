# EmbeddingGemma 2 部署记录 (2026-10-16)

## 1. 目标

在 DGX Spark 上部署 Google **EmbeddingGemma 2** 作为本地 embedding 服务,
OpenAI 兼容 API,与现有 qwen3.8-27b 主服务(4000/8888)完全隔离。

## 2. 环境架构(部署前盘点,只读确认)

```
用户(本机 Claude Code / 其他客户端)
  └─→ switchyard-server (Rust, 0.0.0.0:4000, systemd user 服务)
        配置: /home/<user>/routes.toml
        routes: qwen3.8-27b / ornith-local(灰度) / cloud-test / switchyard(智能路由)
        └─→ SGLang 容器 qwen38-27b-sglang (host 网络, 127.0.0.1:8888)
              模型: RadixArk/Qwen3.8-27B-NVFP4 + DFlash2 草稿
              root 运行, mem-fraction-static 0.70
        + qwen38-watchdog.sh (systemd user, 探活/自愈/告警)
        + vllm-open-proxy.py (外部 IP → 127.0.0.1:8888 TCP 转发)
```

**关键结论**: 4000 是路由网关,`routes.toml` 只声明 `openai_chat` 客户端,
没有 embeddings 路由能力。新模型不会"自动"出现在 4000 上。

资源: 121GiB 统一内存, qwen 占 ~94GiB, 可用 ~26GiB; 磁盘 3.3T 可用。
EmbeddingGemma 2 (text-only 270M) 运行占用 < 2GiB, 无冲突。

## 3. 部署方案

| 项 | 值 |
|---|---|
| 目录 | `/home/<user>/projects/embeddinggemma-2/` |
| 模型 | `google/embeddinggemma-2`, **text-only 变体** (270M, `vision_config/audio_config=None`) |
| 权重 | `hf-home/gemma-model/model.safetensors` (1.49GB, 含全部 740M 参数, 运行时只加载文本编码器) |
| 运行时 | `venv/` (Python 3.12, uv 创建): sentence-transformers 6.1.0, transformers 5.20.0, torch 2.14.1+cu130, torchvision 0.29.1+cu130, pillow |
| 服务 | `serve_embeddings.py` (标准库 http.server + ST), 端口 **8890** |
| 启动 | `./start.sh` → nohup, 日志 `logs/serve.log` |
| 停止 | `./stop.sh` (pkill -f serve_embeddings.py) |

端口选择理由: 8889 历史为"判官"(已停用但有记录), 8890 全新且与主服务 8888 相邻便于记忆。

### API

- `POST /v1/embeddings` — OpenAI 兼容
  - `input`: string 或 list[string], 支持中文/英文/代码
  - `model`: 任意字符串, 默认 `embeddinggemma-2`
  - `dimensions`: 可选, MRL 截断 (128/256/512/768, 默认 768)
- `GET /health` — 加载状态
- `GET /v1/models` — 模型列表

调用示例:

```bash
curl -X POST http://spark-XXXX:8890/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["主权基金 ESG 尽调", "ESG due diligence"], "model": "embeddinggemma-2"}'
```

## 4. 实测记录

- 模型加载: 4.1s (GPU, aarch64 + CUDA 13.0)
- 单文本 "你好,世界": dim=768, norm=0.9982, usage=5 tokens, 200 OK
- 批量 3 条: 语义合理 — cos(天气/雨伞)=0.7669 > cos(天气/量子纠错)=0.5892
- MRL 512d: 返回 512 维, 200 OK
- 命脉验证: 4000 switchyard (pid 2346) / 8888 SGLang (pid 1309633) /
  qwen38-watchdog — 进程、端口、配置均零改动; 4000 /v1/models 正常响应
- 内存占用: 服务 < 2GiB (统一内存 26Gi 可用, 无压力)

## 5. 踩坑

1. `huggingface_hub` 新版: `snapshot_download(local_dir_use_best_practice=...)` 已移除, 用 `local_dir=`
2. 即使只用文本, `EmbeddingGemma2Processor` 初始化仍会构建 vision/video/audio
   processor → 需要 **pillow + torchvision**, 否则 ImportError
3. `prompt_name="query"` 为查询侧 prompt; 若做双向检索, 文档侧应设
   `prompt_name="passage"` (ST 6.x 按 repo 配置自动区分, 本仓库配置需确认)

## 6. 可选后续(未做)

- **挂到 switchyard 4000**: 需要改 `routes.toml` + 可能需 switchyard(Rust)
  增加 embeddings 透传类型 (现只支持 openai_chat)。属于独立工程, 且
  reload 有闪断风险, 暂不做。
- systemd user 服务化: 仿 `qwen38-watchdog`, 开机自启 + 探活。当前
  start.sh 为手动; 重启 Spark 后需重跑。
- 768d vs 256d 选型: 文本/code 场景 256d 保留 ~95% 质量且省 3x 存储。

## 7. 与 4000 的关系(用户最关心的问题)

**新模型不在 4000 上, 也不应该自动在。** 4000 = switchyard 路由(只透传
chat/completions), embeddings 走独立端口 8890。以后调用:
`http://spark-XXXX:8890/v1/embeddings`。
若确实想统一入口, 走 §6 第一条 (改 switchyard)。
