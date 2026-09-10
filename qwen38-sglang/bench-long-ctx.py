#!/usr/bin/env python3
"""长上下文"大海捞针"验证脚本 (零依赖, 只用标准库).

用法:
  python3 bench-long-ctx.py --base-url http://127.0.0.1:8888/v1 --model qwen3.8-27b --ctx 400000
  python3 bench-long-ctx.py --ctx 900000 --secret 918273 --depth 0.1

做的事: 拼一个约 --ctx token 的填充文本, 在约 --depth 比例深度插入一个秘密数字,
然后提问, 检查回答里是否包含该数字; 同时报告 prompt_tokens (服务端真实计数)、
TTFT、总耗时、生成速度。
注意: 512K 首次冷 prefill 可能十几分钟 (chunked prefill 8192 + JIT), 默认超时 1 小时。
"""
import argparse, json, sys, time, urllib.request

FILLER = (
    "第 {i} 段: 这是一段用于填充上下文的中性文本, 内容没有特殊含义, 仅用于测试长上下文检索能力。"
    "深度学习推理服务的性能取决于显存带宽、批处理大小与 KV 缓存命中率。"
)


def build_prompt(target_tokens: int, secret: str, depth: float):
    # 中文约 1 token/字, 经验系数 1.0; 英文另计。实际 token 数由服务端 usage 报告。
    chunks, approx, i, filled = [], 0, 0, False
    need_at = int(target_tokens * depth)
    while approx < target_tokens:
        i += 1
        if not filled and approx >= need_at:
            chunks.append(f"\n【重要记录】保险柜密码是 {secret}, 请记住。\n")
            filled = True
        line = FILLER.format(i=i)
        chunks.append(line)
        approx += len(line)
    if not filled:
        chunks.append(f"\n【重要记录】保险柜密码是 {secret}, 请记住。\n")
    chunks.append("\n\n问题: 上文【重要记录】里写的保险柜密码是多少? 只回答数字。")
    return "".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--ctx", type=int, default=400000)
    ap.add_argument("--depth", type=float, default=0.1)
    ap.add_argument("--secret", default="918273")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    prompt = build_prompt(args.ctx, args.secret, args.depth)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": not args.no_thinking},
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    print(f"请求 {args.ctx} 近似 token (深度 {args.depth}, 秘密 {args.secret}) -> {args.base_url}", flush=True)
    t0 = time.time()
    ttft = None
    text, usage = "", {}
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                        print(f"  TTFT {ttft:.1f}s", flush=True)
                    text += piece
            if ttft is None and time.time() - t0 > 60:
                print(f"  ...仍在 prefill ({time.time()-t0:.0f}s)", flush=True)
                t0b = time.time()
                while time.time() - t0b < 60 and ttft is None:
                    time.sleep(5)
    total = time.time() - t0
    pt = usage.get("prompt_tokens")
    print(f"结果: 命中={'是' if args.secret in text else '否'}  prompt_tokens={pt}  completion={usage.get('completion_tokens')}")
    print(f"耗时: 总 {total:.1f}s  TTFT {ttft if ttft is None else round(ttft,1)}s")
    print("回答:", text.strip()[:200])


if __name__ == "__main__":
    sys.exit(main())
