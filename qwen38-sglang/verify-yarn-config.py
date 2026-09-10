#!/usr/bin/env python3
"""Pre-flight check: does this SGLang build accept the model-card YaRN override?

Zero-downtime: loads only configs (no weights, no GPU), then builds ModelConfig
for the target and the DSpark draft exactly the way launch_server does.
Run:  see run-verify-yarn.sh
"""
import json, os, sys, traceback

os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig

TARGET = "RadixArk/Qwen3.8-27B-NVFP4"
DRAFT = "RadixArk/Qwen3.8-27B-DSpark"


def yarn_override(factor: float) -> str:
    return json.dumps({
        "text_config": {
            "rope_parameters": {
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_type": "yarn",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
                "factor": factor,
                "original_max_position_embeddings": 262144,
            }
        }
    })


def probe(title, *, model_path, is_draft, context_length, override):
    sa = ServerArgs(
        model_path=model_path,
        trust_remote_code=True,
        context_length=context_length,
        json_model_override_args=override,
        device="cuda",  # 跳过 get_device() 探测, 让本脚本无需 GPU 即可跑
    )
    try:
        mc = ModelConfig.from_server_args(
            sa, model_path=model_path, is_draft_model=is_draft,
            context_length=context_length,
        )
        rope = getattr(mc.hf_text_config, "rope_parameters", None) or getattr(mc.hf_text_config, "rope_scaling", None)
        print(f"[OK]   {title}: context_len={mc.context_len} max_pos={getattr(mc.hf_text_config,'max_position_embeddings',None)} rope={rope}")
        return True
    except Exception as e:
        print(f"[FAIL] {title}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("A. target model, native 262144, no override (today's config)")
    probe("target native", model_path=TARGET, is_draft=False, context_length=262144, override="{}")

    print("=" * 70)
    print("B. target model, YaRN factor 2.0, context 524288")
    probe("target 512K yarn2", model_path=TARGET, is_draft=False, context_length=524288, override=yarn_override(2.0))

    print("=" * 70)
    print("C. target model, YaRN factor 4.0, context 1048576")
    probe("target 1M yarn4", model_path=TARGET, is_draft=False, context_length=1048576, override=yarn_override(4.0))

    print("=" * 70)
    print("D. DSpark draft, no override, context 262144 (today's config)")
    probe("draft native", model_path=DRAFT, is_draft=True, context_length=262144, override="{}")

    print("=" * 70)
    print("E. DSpark draft + YaRN override leaked in, context 524288  (expected: crash)")
    probe("draft 512K yarn2 (leak)", model_path=DRAFT, is_draft=True, context_length=524288, override=yarn_override(2.0))
    print("=" * 70)
    print("done")
