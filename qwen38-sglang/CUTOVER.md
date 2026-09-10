# Qwen3.8-27B(SGLang+DSpark) 迁移 Runbook —— 对用户最小影响
日期: 2026-09-09  硬件: DGX Spark GB10 (aarch64, 单卡 128GB 统一内存)

## 用户可见变化(全部)
- model 字段: ornith-local / ornith-1.5-35b-a3b-nvfp4  ->  qwen3.8-27b
- 其余一律不变: base URL(4000 / 8888)、auth、OpenAI/Anthropic 协议、steaming、工具调用

## 不变的组件(零改动)
- switchyard 服务: 0.0.0.0:4000, systemd user 服务
- vllm-open-proxy.py: <LAN-IP-1>/<LAN-IP-2> + <Tailscale-IP>:8888 -> 127.0.0.1:8888
- OpenRouter/DeepSeek 云路由 (cloud-test / switchyard 智能路由的云档)
- 旧 vLLM 容器 ornith-b12x-serve 权重保留 (rollback 用)

## 准备阶段(已大体完成, 零影响)
- [x] 拉镜像 lmsysorg/sglang:qwen38-27b (约15GB, 后台)
- [x] 预下载模型到 /home/<user>/projects/qwen38-model-cache (主+草稿, 后台)
- [x] start.sh / stop.sh (本目录)
- [x] routes.toml.next (经烘焙好的新路由配置)
- [ ] 切换前最后校验: 镜像+模型就位

## 切换步骤(选低峰, 总停机 ≈ 模型加载时间 + 几秒)
1) 备份与旧配置快照
   cp /home/<user>/routes.toml /home/<user>/routes.toml.bak2-$(date +%m%d%H%M)
   docker ps | grep ornith   # 记录运行状态
2) 停旧后端(不删容器!)
   docker stop ornith-b12x-serve
3) 起新后端
   cd /home/<user>/projects/qwen38-sglang && ./start.sh
   # 就绪轮询在脚本内; 首次含 torch.compile, 可能 5-10 分钟
4) 直连冒烟
   curl http://127.0.0.1:8888/v1/models
   curl http://127.0.0.1:8888/v1/chat/completions -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":32}'
   # 再测 tools(带 tools 参数) + reasoning_content 字段
   docker logs qwen38-27b-sglang 2>&1 | grep -i 'max_running_requests'  # 确认没被 clamp 到 < 8
5) 切路由
   cp routes.toml.next /home/<user>/routes.toml
   systemctl --user restart switchyard     # 30s drain, 重启 <10s
6) 全链路验证
   curl http://127.0.0.1:4000/v1/models            # 应含 qwen3.8-27b / ornith-local / cloud-test / switchyard
   curl http://127.0.0.1:4000/v1/chat/completions -d '{"model":"qwen3.8-27b",...}'
   curl http://127.0.0.1:4000/v1/chat/completions -d '{"model":"switchyard",...}'  # 智能路由→本地新模型
   curl http://<LAN-IP-1>:8888/v1/models         # 外部直连(经TCP代理)
7) 通知: model 改为 qwen3.8-27b (灰度期内 ornith-local 也指向新模型, 不慌)

## 回滚(任一步失败, <1min)
docker stop qwen38-27b-sglang
docker start ornith-b12x-serve
cp /home/<user>/routes.toml.bak /home/<user>/routes.toml
systemctl --user restart switchyard

## 已知取舍/备注
- DSPARK 在 code/math/tool 上快, 长文 prose 比 MTP 慢; 想切 MTP: 把 start.sh 里
  --speculative-algorithm DSPARK 段换成 EAGLE 系(见 MiaAI start.sh), 权重同镜像。
- YaRN 与 DSPARK 不兼容: 长上下文(>262144)需要 YARN=1 + CONTEXT_LENGTH, 会切到 MTP。
- vs 网友原始命令的修正: 见调研正文 (speculative-num-steps 5→dspark 栈;
  mamba ratio 0.4→4.21+显式 pin; 30000 端口→8888 host网络; etc.)