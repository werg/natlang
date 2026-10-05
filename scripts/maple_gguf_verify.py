#!/usr/bin/env python3
"""maple-qat M0.2: the official Maple GGUF equals the published (already ternary) BF16 checkpoint.

For each checked tensor: dequantize the GGUF tensor (TQ2_0 / F16 / F32) and compare with the checkpoint tensor
(experts stacked as the converter does). Ternary tensors must match exactly (the TQ2_0 block scale is the row's
BF16 alpha in FP16); F16 tensors must equal the BF16 values rounded to FP16.

    python scripts/maple_gguf_verify.py --gguf .../maple-preview-TQ2_0-head-F16.gguf \
        --checkpoint /home/werg/data/models/maple-preview-bf16 --layers 0,11,23 --out runs/.../m0.2.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GGUF_PY = ROOT / "runs/maple-preview-evaluation-20261005/llama.cpp/gguf-py"

NAMES = {
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj", "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "ffn_gate_inp": "mlp.gate",
}
EXPERTS = {"ffn_gate_exps": "gate_proj", "ffn_up_exps": "up_proj", "ffn_down_exps": "down_proj"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layers", default="0,11,23", help="comma list, or 'all'")
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.path.insert(0, str(GGUF_PY))
    import gguf
    from gguf.quants import dequantize
    from safetensors import safe_open

    reader = gguf.GGUFReader(args.gguf)
    tensors = {t.name: t for t in reader.tensors}
    checkpoint = Path(args.checkpoint)
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    handles = {}

    def load(name):
        file = weight_map[name]
        if file not in handles:
            handles[file] = safe_open(checkpoint / file, framework="pt")
        return handles[file].get_tensor(name)

    n_layers = 1 + max(int(n.split(".")[1]) for n in tensors if n.startswith("blk."))
    layers = range(n_layers) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    checks = [("token_embd.weight", lambda: load("model.word_embeddings.weight")),
              ("output.weight", lambda: load("lm_head.weight")),
              ("output_norm.weight", lambda: load("model.norm.weight"))]
    for i in layers:
        for short, hf in NAMES.items():
            checks.append((f"blk.{i}.{short}.weight", lambda i=i, hf=hf: load(f"model.layers.{i}.{hf}.weight")))
        for short, hf in EXPERTS.items():
            def stacked(i=i, hf=hf):
                count = sum(1 for n in weight_map if n.startswith(f"model.layers.{i}.mlp.experts.") and n.endswith(f"{hf}.weight"))
                return torch.stack([load(f"model.layers.{i}.mlp.experts.{e}.{hf}.weight") for e in range(count)])
            checks.append((f"blk.{i}.{short}.weight", stacked))

    results = []
    for gname, loader in checks:
        t = tensors[gname]
        qtype = gguf.GGMLQuantizationType(t.tensor_type)
        values = dequantize(np.asarray(t.data), qtype).reshape(tuple(reversed([int(x) for x in t.shape])))
        reference = loader().float()
        if qtype == gguf.GGMLQuantizationType.F16:
            expected = reference.to(torch.float16).float().numpy()
        else:
            expected = reference.numpy()
        expected = expected.reshape(values.shape)
        diff = np.abs(values.astype(np.float32) - expected)
        record = {"tensor": gname, "type": qtype.name, "shape": list(values.shape), "exact": bool((diff == 0).all()),
                  "max_abs_diff": float(diff.max()), "mismatched_fraction": float((diff > 0).mean())}
        results.append(record)
        print(json.dumps(record), flush=True)
    summary = {"schema": "natlang.maple_gguf_verify/1", "gguf": args.gguf, "checkpoint": args.checkpoint,
               "layers": list(layers), "tensors": len(results), "all_exact": all(r["exact"] for r in results),
               "results": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
