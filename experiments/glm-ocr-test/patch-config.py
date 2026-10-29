
import pathlib, re
src = pathlib.Path('/home/<user>/projects/venv-vllm/lib/python3.12/site-packages/glmocr/config.yaml')
dst = pathlib.Path('/home/<user>/projects/glm-ocr-test/glmocr-server.yaml')
t = src.read_text()
before = t
# 1) MaaS off -> 自托管
t = t.replace('    enabled: true # Set to true to use MaaS mode',
              '    enabled: false # Set to true to use MaaS mode')
# 2) 版面显式 CPU（§10.8 建议）
t = t.replace('    # device: null', '    device: cpu')
# 3) 并发调低（本次验证的 KV cache 只有 ~25k token）
t = t.replace('  max_workers: 32', '  max_workers: 8')
assert t != before
dst.write_text(t)
for k in ['enabled: false # Set to true', 'device: cpu', 'max_workers: 8',
          'host: "0.0.0.0"', 'port: 5002', 'api_port: 8080', 'model_dir: PaddlePaddle']:
    print(('OK   ' if k in t else 'MISS '), k)
