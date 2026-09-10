#!/usr/bin/env python3
"""DSpark + YaRN 绕过方案离线预检: 哪种草稿配置能让 512K 上下文启动时不崩?

只读 config, 不加载权重, 不占用 8888, 可与线上服务共存。
"""
import json, os, traceback

os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig

TARGET = "RadixArk/Qwen3.8-27B-NVFP4"
DRAFT = "RadixArk/Qwen3.8-27B-DSpark"
CTX = 524288
YARN2 = json.dumps({"text_config": {"rope_parameters": {
    "mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn",
    "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 2.0,
    "original_max_position_embeddings": 262144}}})


def probe(title, *, override, draft_config_file=None, ctx=CTX):
    sa = ServerArgs(model_path=TARGET, trust_remote_code=True, context_length=ctx,
                    json_model_override_args=override, device="cuda",
                    decrypted_draft_config_file=draft_config_file)
    try:
        mc = ModelConfig.from_server_args(sa, model_path=DRAFT, is_draft_model=True,
                                          context_length=ctx)
        tc = mc.hf_text_config
        rope = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
        print(f"[OK]   {title}")
        print(f"       draft model_type={getattr(tc,'model_type',None)} max_pos={getattr(tc,'max_position_embeddings',None)} "
              f"num_heads={getattr(tc,'num_attention_heads',None)} ctx={mc.context_len}")
        print(f"       rope={rope}")
        top = getattr(mc.hf_config, "rope_parameters", None)
        print(f"       hf_config(top-level) rope={top}")
        return True
    except Exception as e:
        print(f"[FAIL] {title}: {type(e).__name__}: {e}")
        return False


print("基线: 现网草稿, 262144, 无 YaRN 覆盖")
probe("baseline", override="{}", ctx=262144)
print("\n复现: 目标 YaRN 覆盖 + 草稿原样 (泄漏)")
probe("leak (预期崩)", override=YARN2)
print("\n方案C: 草稿 config 补完整 text_config + --decrypted-draft-config-file")
probe("C text_config mirror", override=YARN2, draft_config_file="/home/<user>/projects/qwen38-sglang/dspark-draft-config-textcfg.json")
print("\n方案D: 草稿 config 只抬 max_position_embeddings")
probe("D maxpos only (预期崩)", override=YARN2, draft_config_file="/home/<user>/projects/qwen38-sglang/dspark-draft-config-maxpos.json")
print("\n方案E: 引擎侧不给草稿传 override (= is-draft 用 '{}' 加载)")
probe("E no override for draft", override="{}")
