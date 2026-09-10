#!/usr/bin/env python3
"""三种"保住 DSpark"的长上下文模式的离线预检 (只读 config, 不加载权重)."""
import os
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig

VAR = "/root/.cache/huggingface/variants"
STOCK_T = "RadixArk/Qwen3.8-27B-NVFP4"
STOCK_D = "RadixArk/Qwen3.8-27B-DSpark"

MODES = {
    "yarn(草稿原样)":      (f"{VAR}/qwen38-nvfp4-yarn{{s}}", STOCK_D),
    "draftyarn(草稿也换绳)": (f"{VAR}/qwen38-nvfp4-yarn{{s}}", f"{VAR}/qwen38-dspark-yarn{{s}}"),
    "extrap(不缩放)":       (STOCK_T, STOCK_D),
}
CTXS = {512: 524288, 1024: 1048576}


def probe(title, model_path, draft_path, ctx):
    sa = ServerArgs(model_path=model_path, trust_remote_code=True, context_length=ctx,
                    json_model_override_args="{}", device="cuda",
                    speculative_draft_model_path=draft_path)
    try:
        mc = ModelConfig.from_server_args(sa, model_path=model_path, is_draft_model=False, context_length=ctx)
        dc = ModelConfig.from_server_args(sa, model_path=draft_path, is_draft_model=True, context_length=ctx)
        tr = getattr(mc.hf_text_config, "rope_parameters", {})
        dr = getattr(dc.hf_text_config, "rope_parameters", {})
        print(f"[OK]   {title}")
        print(f"       target ctx={mc.context_len} rope={tr.get('rope_type')}/f{tr.get('factor')}/orig{tr.get('original_max_position_embeddings')}")
        print(f"       draft  ctx={dc.context_len} max_pos={getattr(dc.hf_text_config,'max_position_embeddings',None)} "
              f"rope={dr.get('rope_type')}/f{dr.get('factor')}/orig{dr.get('original_max_position_embeddings')}")
    except Exception as e:
        print(f"[FAIL] {title}: {type(e).__name__}: {e}")


for label, (t_tpl, d_tpl) in MODES.items():
    for suffix, ctx in CTXS.items():
        probe(f"{label} @ {ctx//1024}K", t_tpl.format(s=suffix), d_tpl.format(s=suffix), ctx)
