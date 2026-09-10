#!/usr/bin/env python3
"""裁决"DFlash2 原生兼容 YaRN"这一说法: 在同一 build 上跑真实 ServerArgs 路径."""
import json, os
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig

T512 = "/root/.cache/huggingface/variants/qwen38-nvfp4-yarn512"
DFLASH_STOCK = "/root/.cache/huggingface/dflash2-draft"
V = "/root/.cache/huggingface/variants"
CTX = 524288
YARN2 = json.dumps({"text_config": {"rope_parameters": {
    "mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn",
    "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 2.0,
    "original_max_position_embeddings": 262144}}})

CASES = [
    ("1 DFlash2 草稿原样 + CLI YaRN 覆盖", DFLASH_STOCK, YARN2),
    ("2 DFlash2 草稿原样, 无覆盖", DFLASH_STOCK, "{}"),
    ("3 DFlash2 草稿烧入 YaRN(incoai #8 做法), 无覆盖", f"{V}/qwen38-dflash2-yarn512", "{}"),
    ("4 DFlash2 草稿烧入 YaRN + CLI 覆盖 (= 官方配方叠加)", f"{V}/qwen38-dflash2-yarn512", YARN2),
    ("5 DFlash2 草稿带完整 text_config + CLI 覆盖", f"{V}/qwen38-dflash2-textconfig", YARN2),
]

for title, draft, override in CASES:
    try:
        sa = ServerArgs(model_path=T512, trust_remote_code=True, context_length=CTX,
                        json_model_override_args=override, device="cuda",
                        speculative_algorithm="DFLASH",
                        speculative_draft_model_path=draft,
                        speculative_num_draft_tokens=8)
        dc = ModelConfig.from_server_args(sa, model_path=draft, is_draft_model=True, context_length=CTX)
        tc = dc.hf_text_config
        rope = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
        print(f"[OK]   {title}")
        print(f"       draft max_pos={getattr(tc,'max_position_embeddings',None)} rope={rope} "
              f"sliding={getattr(tc,'sliding_window',None)} layer0={ (getattr(tc,'layer_types',None) or [None])[0] }")
    except Exception as e:
        print(f"[FAIL] {title}: {type(e).__name__}: {e}")
