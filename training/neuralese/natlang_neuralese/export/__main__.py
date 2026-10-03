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
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--max-block", type=int, default=64)
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--outtype", default="f32", choices=["f32", "f16", "bf16"])
    args = parser.parse_args(argv)

    import torch

    from ..model.dialect import DIALECT
    from ..model.heads import PortHeads
    from ..model.lfm2_port import ControlTokens, PortBackbone, load_backbone
    from .gguf import export_heads_gguf, export_model_gguf, export_model_hf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model, tokenizer = load_backbone(args.base, args.lora, dtype=torch.float32, device="cpu")
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    heads = PortHeads(backbone, cutoff=args.cutoff, max_length=args.max_block).eval()
    if args.heads:
        state = torch.load(args.heads, map_location="cpu", weights_only=False)
        heads.load_state_dict(state["heads"])
        with torch.no_grad():
            backbone.control_rows.copy_(state["control_rows"].to(backbone.control_rows))
    with tempfile.TemporaryDirectory() as tmp:
        hf_dir = export_model_hf(backbone, tokenizer, Path(tmp) / "hf")
        model_gguf = export_model_gguf(hf_dir, out / "model.gguf", outtype=args.outtype)
    heads_gguf = export_heads_gguf(heads, backbone, out / "neuralese.gguf", dialect=args.dialect or DIALECT)
    print(json.dumps({"model": str(model_gguf), "heads": str(heads_gguf), "cutoff": heads.cutoff,
                      "max_block": heads.max_length}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
