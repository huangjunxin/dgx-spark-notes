#!/usr/bin/env python3
"""裁决用户贴的"方案二"以及变体C: 在 CLI override 仍然存在(= 社区/官方配方)的前提下,
   草稿 config 要怎么改才能不崩。"""
import json, os
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig

STOCK_T = "RadixArk/Qwen3.8-27B-NVFP4"
T512 = "/root/.cache/huggingface/variants/qwen38-nvfp4-yarn512"   # target 用内联 YaRN
VAR = "/root/.cache/huggingface/variants"
CTX = 524288
YARN2 = json.dumps({"text_config": {"rope_parameters": {
    "mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn",
    "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 2.0,
    "original_max_position_embeddings": 262144}}})

CASES = [
    ("A 草稿原样 + CLI YaRN 覆盖 (基线崩溃)", STOCK_T, YARN2),
    ("B 草稿顶层补 max_pos+rope_parameters(方案二字面版) + 覆盖", f"{VAR}/qwen38-dspark-cfg-ropeplanB", YARN2),
    ("C 草稿自带完整 text_config + 覆盖", f"{VAR}/qwen38-dspark-cfg-textconfig", YARN2),
    ("D 草稿自带 rope overlay(与 target 同套 YaRN) + 覆盖", f"{VAR}/qwen38-dspark-yarn512", YARN2),
    ("E 同 D 但不传 CLI 覆盖 (= 我们脚本的做法)", f"{VAR}/qwen38-dspark-yarn512", "{}"),
]

for title, draft_path, override in CASES:
    sa = ServerArgs(model_path=T512, trust_remote_code=True, context_length=CTX,
                    json_model_override_args=override, device="cuda",
                    speculative_draft_model_path=draft_path)
    try:
        dc = ModelConfig.from_server_args(sa, model_path=draft_path, is_draft_model=True, context_length=CTX)
        tc = dc.hf_text_config
        rope = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
        print(f"[OK]   {title}")
        print(f"       draft text_config type={type(tc).__name__} max_pos={getattr(tc,'max_position_embeddings',None)} "
              f"heads={getattr(tc,'num_attention_heads',None)} rope={rope}")
    except Exception as e:
        print(f"[FAIL] {title}: {type(e).__name__}: {e}")
