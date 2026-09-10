#!/usr/bin/env python3
"""方案A 离线验证: 把 YaRN 直接写进 target 的 config.json (= 模型卡方案1),
这样 --json-model-override-args 保持 "{}", DSpark 草稿就不会被污染。"""
import json, os
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig

DRAFT = "RadixArk/Qwen3.8-27B-DSpark"
# 容器内路径 (宿主 /home/<user>/projects/qwen38-model-cache -> 容器 /root/.cache/huggingface)
BASE = os.environ.get("VARIANT_BASE", "/root/.cache/huggingface/variants")
VARIANTS = {
    "512K": (f"{BASE}/qwen38-nvfp4-yarn512", 524288),
    "1M":   (f"{BASE}/qwen38-nvfp4-yarn1024", 1048576),
}


def probe(title, *, model_path, is_draft, ctx, override="{}"):
    sa = ServerArgs(model_path=model_path, trust_remote_code=True, context_length=ctx,
                    json_model_override_args=override, device="cuda")
    try:
        mc = ModelConfig.from_server_args(sa, model_path=model_path, is_draft_model=is_draft, context_length=ctx)
        tc = mc.hf_text_config
        rope = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
        print(f"[OK]   {title}: ctx={mc.context_len} rope_type={rope.get('rope_type')} factor={rope.get('factor')} "
              f"orig_max={rope.get('original_max_position_embeddings')}")
        return True
    except Exception as e:
        print(f"[FAIL] {title}: {type(e).__name__}: {e}")
        return False


for label, (path, ctx) in VARIANTS.items():
    probe(f"target(config内联YaRN {label})", model_path=path, is_draft=False, ctx=ctx)
    probe(f"draft(无 CLI override) + {label}", model_path=DRAFT, is_draft=True, ctx=ctx)
