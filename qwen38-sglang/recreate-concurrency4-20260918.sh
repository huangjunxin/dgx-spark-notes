#!/usr/bin/env bash
# 重建 Qwen3.8-27B (qwen3.8-27b) 容器 —— 修复 DFlash2 batch 尺寸崩溃  2026-09-18
#
# 根因 (镜像源码 dflash_worker_v2.py, 已确认):
#     max_bs = max(get_exec().graph.cuda_graph_config.decode.bs)   # = --cuda-graph-max-bs-decode = 4
#     self.greedy_mask = torch.ones((max_bs,), ...)                # 只有 4 行 buffer
#   decode batch 达到 5 时:
#     self.greedy_mask[:bs].copy_(...)  ->  RuntimeError: tensor a (4) vs b (5)
#   -> SGLang scheduler 自杀 (kill_process_tree) -> 容器 exit 0
#   2026-09-17 15:14:11 与 15:24:18 两次崩溃均由此触发。
#
# 修复: --max-running-requests 8 -> 4  (与 decode CUDA graph 的 bs=4 对齐, bs>=5 不可能出现)
#       --max-mamba-cache-size  96 -> 48 (项目配方: 并发 x 12)
# 其余一切从 docker inspect 原样复制: 镜像/挂载/env/内存上限/cpuset/ipc/网络。
#
# 回滚: 编辑本文件把上面两个值改回 8 / 96 重跑, 或参见
#       container-inspect-20260917-pre-fix.json (崩溃前完整定义)
set -euo pipefail

docker run -d --name qwen38-27b-sglang --restart unless-stopped --network host --ipc host --privileged --gpus all --shm-size 34359738368 --memory 107374182400 --memory-swap 107374182400 --cpuset-cpus 5-9,15-19 -e HF_HOME=/root/.cache/huggingface -e TRITON_CACHE_DIR=/root/.triton -e TORCHINDUCTOR_CACHE_DIR=/root/.torch-cache -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 -e HF_HUB_OFFLINE=1 -v /home/<user>/projects/qwen38-model-cache/.triton:/root/.triton -v /home/<user>/projects/qwen38-model-cache/.torch-cache:/root/.torch-cache -v /home/<user>/projects/qwen38-model-cache:/root/.cache/huggingface lmsysorg/sglang:dev-qwen38-27b-dflash2 python3 -m sglang.launch_server --model-path /root/.cache/huggingface/variants/qwen38-nvfp4-yarn1024 --served-model-name qwen3.8-27b --trust-remote-code --mem-fraction-static 0.70 --attention-backend flashinfer --chunked-prefill-size 8192 --disable-prefill-cuda-graph --disable-flashinfer-autotune --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mamba-full-memory-ratio 4.21 --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 48 --max-running-requests 4 --context-length 1010000 --json-model-override-args '{}' --speculative-algorithm DFLASH --speculative-dflash-block-size 8 --speculative-draft-model-quantization unquant --speculative-num-draft-tokens 8 --speculative-draft-model-path /root/.cache/huggingface/variants/qwen38-dflash2-yarn1024 --enable-torch-compile --torch-compile-max-bs 4 --cuda-graph-max-bs-decode 4 --num-continuous-decode-steps 2 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --sampling-defaults model --enable-metrics --host 127.0.0.1 --port 8888
