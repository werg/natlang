"""`python -m natlang_neuralese.export --out DIR [--heads checkpoint.pt] [--cutoff 6] [--max-block 64]`

Writes `DIR/model.gguf` (the backbone with the control rows merged) and `DIR/neuralese.gguf` (the port heads) for the
llama.cpp fork (training/neuralese/llama-cpp-fork.json): `llama-neuralese-server -m DIR/model.gguf --nz DIR/neuralese.gguf`.
Without `--heads` the heads are untrained (as in the reference server), which is enough to test plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a Neuralese port to llama.cpp GGUF files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--heads", default=None, help="S3 trainer checkpoint with port heads and control rows")
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--max-block", type=int, default=None)
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--outtype", default="f32", choices=["f32", "f16", "bf16"])
    args = parser.parse_args(argv)

    import torch

    from ..model.dialect import DIALECT
    from ..model.heads import PortHeads
    from ..model.lfm2_port import ControlTokens, PortBackbone, load_backbone
    from .gguf import export_heads_gguf, export_model_gguf, export_model_hf

    out = Path(args.out)
    # Never replace a previous export, including an interrupted one.
    out.mkdir(parents=True, exist_ok=False)
    # Export needs model tensors, not the checkpoint's optimizer allocations.
    state = torch.load(args.heads, map_location="cpu", weights_only=False, mmap=True) if args.heads else None
    metadata = (state or {}).get("port_config", {})
    cutoff = args.cutoff if args.cutoff is not None else metadata.get("cutoff")
    if state is not None and cutoff is None:
        raise ValueError("Legacy checkpoint has no cutoff metadata; supply its actual --cutoff")
    cutoff = cutoff if cutoff is not None else 6
    saved_length = int(state["heads"]["stop.position.weight"].shape[0]) - 1 if state else None
    max_block = args.max_block if args.max_block is not None else saved_length or 64
    if saved_length is not None and max_block != saved_length:
        raise ValueError(f"--max-block must match checkpoint length {saved_length}")
    if metadata.get("cutoff") is not None and cutoff != metadata["cutoff"]:
        raise ValueError("--cutoff differs from the trained checkpoint")
    torch.manual_seed(0)
    model, tokenizer = load_backbone(args.base, args.lora, dtype=torch.float32, device="cpu")
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    heads = PortHeads(backbone, cutoff=cutoff, max_length=max_block).eval()
    if state is not None:
        heads.load_state_dict(state["heads"])
        with torch.no_grad():
            backbone.control_rows.copy_(state["control_rows"].to(backbone.control_rows))
        if state.get("lora"):
            from ..train.adapters import inject_lora, lora_state
            from peft.tuners.tuners_utils import BaseTunerLayer

            inject_lora(backbone, state["lora_layers"], rank=state["lora_rank"],
                        alpha=metadata.get("lora_alpha", 2 * state["lora_rank"]))
            if set(lora_state(backbone)) != set(state["lora"]):
                raise ValueError("Checkpoint backbone adapter names do not match the model")
            parameters = dict(backbone.hf.named_parameters())
            with torch.no_grad():
                for name, value in state["lora"].items():
                    parameters[name].copy_(value.to(parameters[name]))
            # Export ordinary merged HF weights, not unrecognised PEFT tensor names.
            for name, module in list(backbone.hf.named_modules()):
                if isinstance(module, BaseTunerLayer):
                    module.merge(safe_merge=True)
                    parent, _, child = name.rpartition(".")
                    setattr(backbone.hf.get_submodule(parent), child, module.get_base_layer())
    with tempfile.TemporaryDirectory() as tmp:
        hf_dir = export_model_hf(backbone, tokenizer, Path(tmp) / "hf")
        model_gguf = export_model_gguf(hf_dir, out / "model.gguf", outtype=args.outtype)
    heads_gguf = export_heads_gguf(heads, backbone, out / "neuralese.gguf", dialect=args.dialect or DIALECT)
    receipt = {"model": str(model_gguf), "heads": str(heads_gguf), "cutoff": heads.cutoff,
               "max_block": heads.max_length, "checkpoint": args.heads,
               "backbone_adapters_merged": bool((state or {}).get("lora"))}
    (out / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
