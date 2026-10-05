"""Export a trained nested-family member (or the trained full model) to a llama.cpp GGUF, exactly as trained.

Starts from the official Maple GGUF (whose tensors equal the published checkpoint: maple-qat §2.0) and replaces:

- attention Q/K/V/O: ``codes(W + ΔW_shared + ΔW_member) × scale`` with the learned FP16 block scales (or Maple's
  row rule without them), re-quantized to TQ2_0 (exact: every 256-block holds {−s, 0, +s});
- experts: the member's first n experts in N0 order, their fixed codes times the learned FP16 block scales (TQ2_0);
- router: the member's private router rows (F32), or the full router reordered;
- norms: weight + the member's private gain;
- depth members: layers past the exit dropped, per-layer metadata arrays truncated.

Embedding and head are copied (F16); re-quantize them with llama-quantize afterwards (Q4_K / Q6_K), which leaves
TQ2_0 tensors exact.

    python -m natlang_neuralese.maple.export_member --gguf official.gguf --state runs/.../nested-state.pt \\
        --member 24x32 --out member.gguf        (--member full: the trained full model)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from .ternary import ternary_codes

GGUF_PY = Path("/home/werg/llama.cpp-neuralese/gguf-py")
ATTENTION = {"attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj", "attn_output": "o_proj"}
NORMS = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}


def _expand(scale: torch.Tensor, columns: int, block: int = 256) -> torch.Tensor:
    return scale.repeat_interleave(block, dim=-1)[..., :columns]


def member_attention(base: torch.Tensor, state: dict, prefix: str, key: str | None, scale: float,
                     private_scale: float) -> torch.Tensor:
    """The exported attention matrix (FP32 {−s, 0, +s} per 256-block) for ``key`` (None: the full model)."""
    merged = base.float()
    a, b = state.get(f"{prefix}.lora_A"), state.get(f"{prefix}.lora_B")
    if a is not None:
        merged = merged + scale * (b.float() @ a.float())
    if key is not None and f"{prefix}.private.{key}.lora_A" in state:
        merged = merged + private_scale * (state[f"{prefix}.private.{key}.lora_B"].float()
                                           @ state[f"{prefix}.private.{key}.lora_A"].float())
    codes, alpha = ternary_codes(merged)
    learned = state.get(f"{prefix}.learned_scale")
    if learned is None:
        return codes.float() * alpha.float()
    return codes.float() * _expand(learned.half().float(), merged.shape[-1])


def member_experts(values: torch.Tensor, blocks: torch.Tensor | None) -> torch.Tensor:
    """Experts with their fixed codes re-scaled by learned FP16 block scales ([E, rows, cols] FP32)."""
    if blocks is None:
        return values
    return torch.sign(values) * _expand(blocks.half().float(), values.shape[-1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--state", required=True, help="nested-state.pt from nested_train")
    ap.add_argument("--member", required=True, help="LxE key (e.g. 24x32, 8x16) or 'full'")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(GGUF_PY))
    import gguf
    from gguf.quants import dequantize, quantize

    saved = torch.load(args.state, map_location="cpu")
    state = {k.removeprefix("model."): v for k, v in saved["trainable"].items()}
    train_args = saved.get("args", {})
    scale = 2.0  # alpha = 2 * rank in nested_train
    private_scale = 2.0
    orders = [o.numpy() for o in torch.load(saved["order"])["orders"]] if saved.get("order") else None
    reader = gguf.GGUFReader(args.gguf)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    total = 1 + max(int(t.name.split(".")[1]) for t in reader.tensors if t.name.startswith("blk."))
    n_expert = int(reader.fields[gguf.Keys.LLM.EXPERT_COUNT.format(arch=arch)].contents())
    if args.member == "full":
        key, layers, experts = None, total, n_expert
    else:
        depth, experts = (int(x) for x in args.member.split("x"))
        key, layers = args.member, depth
    keep = [o[:experts] for o in orders] if orders is not None else [np.arange(experts)] * total

    writer = gguf.GGUFWriter(args.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        kind = field.types[0]
        sub = field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name == gguf.Keys.LLM.EXPERT_COUNT.format(arch=arch):
            value = experts
        elif field.name == gguf.Keys.LLM.BLOCK_COUNT.format(arch=arch):
            value = layers
        elif kind == gguf.GGUFValueType.ARRAY and field.name.startswith(f"{arch}.") and isinstance(value, list) \
                and len(value) == total:
            value = value[:layers]  # per-layer arrays (sliding-window pattern, SwiGLU clamp)
        writer.add_key_value(field.name, value, kind, sub_type=sub)

    tensors = []
    for tensor in reader.tensors:
        parts = tensor.name.split(".")
        qtype = gguf.GGMLQuantizationType(tensor.tensor_type)
        if parts[0] == "blk":
            i = int(parts[1])
            if i >= layers:
                continue
            kind = parts[2]
            if kind in ATTENTION:
                base = torch.from_numpy(dequantize(np.asarray(tensor.data), qtype).copy())
                prefix = f"layers.{i}.self_attn.{ATTENTION[kind]}.parametrizations.weight.0"
                value = member_attention(base, state, prefix, key, scale, private_scale)
                tensors.append((tensor.name, quantize(value.numpy(), qtype), qtype))
                continue
            if kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                values = torch.from_numpy(dequantize(np.asarray(tensor.data), qtype).copy())[keep[i]]
                blocks = None
                if kind == "ffn_down_exps":
                    blocks = state.get(f"layers.{i}.mlp.experts.down_blocks")
                else:
                    gate_up = state.get(f"layers.{i}.mlp.experts.gate_up_blocks")
                    if gate_up is not None:
                        ff = values.shape[1]
                        blocks = gate_up[:, :ff] if kind == "ffn_gate_exps" else gate_up[:, ff:]
                if blocks is not None:
                    blocks = blocks[:experts]
                tensors.append((tensor.name, quantize(member_experts(values, blocks).numpy(), qtype), qtype))
                continue
            if kind == "ffn_gate_inp":
                private = state.get(f"layers.{i}.mlp.private_gate.{key}") if key else None
                rows = private.float().numpy() if private is not None else np.asarray(tensor.data)[keep[i]]
                tensors.append((tensor.name, np.ascontiguousarray(rows, dtype=np.float32), qtype))
                continue
            if kind in NORMS:
                gain = state.get(f"layers.{i}.{NORMS[kind]}.private.{key}") if key else None
                weight = np.asarray(tensor.data, dtype=np.float32)
                tensors.append((tensor.name, weight + gain.float().numpy() if gain is not None else weight, qtype))
                continue
        if tensor.name == "output_norm.weight" and key and f"norm.private.{key}" in state:
            weight = np.asarray(tensor.data, dtype=np.float32) + state[f"norm.private.{key}"].float().numpy()
            tensors.append((tensor.name, weight, qtype))
            continue
        tensors.append((tensor.name, np.asarray(tensor.data), qtype))

    for name, data, qtype in tensors:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, data, _ in tensors:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {args.out}: {Path(args.out).stat().st_size / 1e9:.2f} GB ({args.member}: {layers} layers, {experts} experts)")


if __name__ == "__main__":
    main()
