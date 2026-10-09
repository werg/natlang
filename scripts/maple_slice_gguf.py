#!/usr/bin/env python3
"""Write a nested Maple GGUF (plans/neuralese/MAPLE_NESTED.md §6): the first n experts of every layer, in the order
from N0, cut byte-for-byte out of a full Maple GGUF. Router rows are reordered and cut alike; expert_count is set to
n (and expert_used_count to --top-k if given). Everything else is copied unchanged.

Embedding/head re-quantization is a separate step with llama-quantize, e.g.
    llama-quantize --allow-requantize --token-embedding-type q4_k --output-tensor-type q6_k in.gguf out.gguf TQ2_0
(TQ2_0 re-quantization of already-ternary tensors is exact).

    python scripts/maple_slice_gguf.py --gguf full.gguf --order runs/.../expert-order.pt --experts 32 --out small.gguf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
from natlang_neuralese.common.paths import resolve  # noqa: E402

sys.path.insert(0, str(resolve("llama_cpp", "gguf-py")))
import gguf  # noqa: E402

EXPERT_TENSORS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--order", required=True, help="expert-order.pt (orders: per-layer permutations)")
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.gguf)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    orders = [o.numpy() for o in torch.load(args.order)["orders"]]
    keep = [o[:args.experts] for o in orders]
    writer = gguf.GGUFWriter(args.out, arch=arch, endianess=reader.endianess)
    expert_count = gguf.Keys.LLM.EXPERT_COUNT.format(arch=arch)
    expert_used = gguf.Keys.LLM.EXPERT_USED_COUNT.format(arch=arch)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == expert_count:
            value = args.experts
        elif field.name == expert_used and args.top_k:
            value = args.top_k
        writer.add_key_value(field.name, value, kind, sub_type=sub)

    def sliced(tensor):
        parts = tensor.name.split(".")
        if parts[0] == "blk" and parts[2] in EXPERT_TENSORS + ("ffn_gate_inp",):
            return np.ascontiguousarray(tensor.data[keep[int(parts[1])]])
        return tensor.data

    datas = [(t, sliced(t)) for t in reader.tensors]
    for tensor, data in datas:
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data in datas:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {args.out}: {Path(args.out).stat().st_size / 1e9:.2f} GB, {args.experts} experts per layer")


if __name__ == "__main__":
    main()
