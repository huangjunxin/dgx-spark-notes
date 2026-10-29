#!/usr/bin/env python3
"""生成"内联 YaRN"的 target 模型目录 (软链farm, 不复制 21GB 权重).

背景: SGLang 的 --json-model-override-args 会被转发给独立的 DSpark 草稿模型,
      把 text_config.rope_parameters 覆盖塞进草稿的扁平 config → rope 校验崩溃
      (AttributeError: max_position_embeddings)。把 YaRN 直接写进 target 自己的
      config.json (Qwen 模型卡的方案一) 就不需要该 CLI 参数, 草稿不受影响。

用法:
  python3 make-yarn-variant.py --context 524288
  python3 make-yarn-variant.py --context 1048576 --name qwen38-nvfp4-yarn1024

产出: <cache>/variants/<name>/  其中 config.json 是真实文件, 其余为相对软链。
容器内路径: /root/.cache/huggingface/variants/<name>
"""
import argparse, glob, json, os

CACHE = "/home/<user>/projects/qwen38-model-cache"
SRC_TARGET = "RadixArk/Qwen3.8-27B-NVFP4"
SRC_DRAFT = "RadixArk/Qwen3.8-27B-DSpark"
DFLASH2_DIR = CACHE + "/dflash2-draft"          # z-lab/Qwen3.8-27B-DFlash2 (本地目录, 非 hub 缓存)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, required=True)
    ap.add_argument("--base-factor-at", type=int, default=262144)
    ap.add_argument("--name", default=None)
    ap.add_argument("--draft", action="store_true",
                    help="给 DSpark 草稿做 rope overlay (草稿也换成与 target 一致的 YaRN)")
    ap.add_argument("--dflash2", action="store_true",
                    help="针对 DFlash2 草稿 (本地 dflash2-draft 目录): 把 YaRN 烧进它自己的 config "
                         "(= hasso5703 patch-yarn.py / incoai #8 的做法)")
    ap.add_argument("--dflash2-dir", default=DFLASH2_DIR,
                    help=f"DFlash2 草稿源目录 (默认 {DFLASH2_DIR})。换草稿换代时指向新目录, "
                         "如 <cache>/dflash2-nvfp4-draft")
    ap.add_argument("--dflash2-prefix", default="qwen38-dflash2",
                    help="DFlash2 变体名前缀 (默认 qwen38-dflash2)。换草稿时用不同前缀, "
                         "以便与旧草稿的变体并存、便于回滚")
    ap.add_argument("--draft-keep-rope", action="store_true",
                    help="只把草稿的 max_position_embeddings 抬到目标长度, rope 保持训练值不动 "
                         "(= hasso5703/dgx-spark-qwen38 patch-yarn.py 对草稿做的事的保守版)")
    args = ap.parse_args()

    factor = round(args.context / args.base_factor_at)
    factor = max(1, factor)
    if args.dflash2:
        args.draft = True
    if args.draft_keep_rope:
        args.draft = True
    repo_id = SRC_DRAFT if args.draft else SRC_TARGET
    if args.dflash2:
        prefix = args.dflash2_prefix
    elif args.draft_keep_rope:
        prefix = "qwen38-dspark-maxpos"
    else:
        prefix = "qwen38-dspark" if args.draft else "qwen38-nvfp4"
    name = args.name or f"{prefix}-yarn{round(args.context/1024)}"
    if args.dflash2:
        snap = args.dflash2_dir
        if not os.path.isfile(os.path.join(snap, "config.json")):
            raise SystemExit(f"DFlash2 草稿目录无效 (缺 config.json): {snap}")
    else:
        repo_dir = os.path.join(CACHE, "hub", "models--" + repo_id.replace("/", "--"))
        snap = glob.glob(os.path.join(repo_dir, "snapshots", "*"))[0]
    out = os.path.join(CACHE, "variants", name)
    os.makedirs(out, exist_ok=True)

    for entry in os.listdir(out):
        p = os.path.join(out, entry)
        if os.path.islink(p) or os.path.isfile(p):
            os.remove(p)
    for entry in os.listdir(snap):
        if entry == "config.json":
            continue
        os.symlink(os.path.relpath(os.path.join(snap, entry), out), os.path.join(out, entry))

    cfg = json.load(open(os.path.join(snap, "config.json")))
    if args.draft:
        # 草稿是扁平 qwen3 config: rope 直接在顶层
        cfg["max_position_embeddings"] = args.context
        if args.draft_keep_rope:
            json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2, ensure_ascii=False)
            print(f"built {out}  (draft max_position_embeddings={args.context}, rope 保持训练值不变)")
            print(f"容器内 --speculative-draft-model-path: /root/.cache/huggingface/variants/{name}")
            return
        cfg["rope_parameters"] = {
            "rope_type": "yarn",
            "rope_theta": 10000000,
            "factor": float(factor),
            "original_max_position_embeddings": args.base_factor_at,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }
        cfg["rope_scaling"] = dict(cfg["rope_parameters"])
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2, ensure_ascii=False)
        print(f"built {out}  (draft rope overlay: ctx={args.context}, YaRN factor={factor})")
        print(f"容器内 --speculative-draft-model-path: /root/.cache/huggingface/variants/{name}")
        return
    tc = cfg["text_config"]
    tc["max_position_embeddings"] = args.context
    tc["rope_parameters"] = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "yarn",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "factor": float(factor),
        "original_max_position_embeddings": args.base_factor_at,
    }
    tc.pop("rope_scaling", None)  # 避免与 rope_parameters 重复/冲突
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2, ensure_ascii=False)
    print(f"built {out}  (context={args.context}, YaRN factor={factor})")
    print(f"容器内 --model-path: /root/.cache/huggingface/variants/{name}")


if __name__ == "__main__":
    main()