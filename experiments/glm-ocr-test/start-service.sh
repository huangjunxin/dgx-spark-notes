#!/usr/bin/env bash
# 一键启动 GLM-OCR 服务栈（vLLM 引擎 + SDK HTTP 服务 + 上传网关）
# 用法：bash /home/<user>/projects/glm-ocr-test/start-service.sh [--with-gateway]
set -euo pipefail

ROOT=/home/<user>/projects
VENV=$ROOT/venv-vllm
TEST=$ROOT/glm-ocr-test
WITH_GATEWAY=${1:-}

export PATH=$VENV/bin:$PATH
export HF_HOME=$ROOT/glm-ocr-model-cache
export VLLM_CACHE_ROOT=$TEST/.vllm-cache          # 坑：~/.cache/vllm 属 root 不可写

wait_for () { # url, name, tries
  for _ in $(seq 1 "$3"); do
    if curl -sf --max-time 3 "$1" >/dev/null 2>&1; then echo "  ✓ $2 ready"; return 0; fi
    sleep 2
  done
  echo "  ✗ $2 未就绪：$1" >&2; return 1
}

# 1) 起 vLLM（本机需 GPU 直通权限，见部署文档 §10.5 坑 3）
if curl -sf --max-time 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
  echo "  · vLLM :8080 已在运行"
else
  echo "  · 启动 vLLM :8080 ..."
  cd "$TEST"
  setsid nohup vllm serve zai-org/GLM-OCR --served-model-name glm-ocr \
    --host 127.0.0.1 --port 8080 --gpu-memory-utilization 0.05 \
    --max-model-len 16384 --enforce-eager --limit-mm-per-prompt '{"image": 1}' \
    > "$TEST/vllm-serve.log" 2>&1 &
  # 实测：冷启动可能超过 2 分钟（vLLM 加载权重 + 编译），窗口给到 300s
  wait_for http://127.0.0.1:8080/v1/models "vLLM" 150
fi

# 2) 起 SDK HTTP 服务（版面 CPU，占统一内存约 0.4 GB）
if curl -sf --max-time 2 http://127.0.0.1:5002/health >/dev/null 2>&1; then
  echo "  · glmocr 服务 :5002 已在运行"
else
  echo "  · 启动 glmocr 服务 :5002 ..."
  cd "$TEST"
  setsid nohup "$VENV/bin/python" -m glmocr.server --config "$TEST/glmocr-server.yaml" \
    > "$TEST/glmocr-server.log" 2>&1 &
  wait_for http://127.0.0.1:5002/health "glmocr 服务" 90   # 含版面模型加载 + 对 vLLM 的 30s 探活
fi

# 3) 可选：上传网关（multipart → file://，解决内置服务不能传 PDF 的问题）
if [ "$WITH_GATEWAY" = "--with-gateway" ]; then
  if curl -sf --max-time 2 http://127.0.0.1:5010/health >/dev/null 2>&1; then
    echo "  · 上传网关 :5010 已在运行"
  else
    echo "  · 启动上传网关 :5010 ..."
    cd "$ROOT"
    setsid nohup "$VENV/bin/python" glm-ocr-upload-gateway.py --host 0.0.0.0 --port 5010 \
      > "$TEST/gateway.log" 2>&1 &
    wait_for http://127.0.0.1:5010/health "上传网关" 30
  fi
fi

echo
echo "服务地址（局域网/ Tailscale 客户端可用）："
echo "  SDK HTTP 服务 : http://<LAN-IP-2>:5002/glmocr/parse   (或 <Tailscale-IP> / <LAN-IP-1>)"
echo "  上传网关      : http://<LAN-IP-2>:5010/parse         (multipart，支持 PDF)"
echo "  vLLM 原生 API : http://127.0.0.1:8080/v1                (仅本机；对外需转发)"
echo
free -m | head -2
