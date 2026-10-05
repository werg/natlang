"""Weight adapters as LoRA files for other servers (LEARNING_CONTINUUM.md §6.5; M4 item 10).

`tiny` and `xs` adapters live in the top-r singular subspace of each adapted matrix, so a decoded adapter is exactly a
rank-r LoRA: ΔW = U_r R V_rᵀ, A = R V_rᵀ (r × in), B = U_r (out × r) (`AdapterBank.lora`). This module writes one as
a PEFT LoRA directory (vLLM serves it directly) and converts that to a GGUF LoRA with the fork's unchanged
`convert_lora_to_gguf.py` (llama.cpp applies it with a per-request scale). The adapter's scale is folded into B, so
the files apply the adapter at scale 1 (alpha = r).

    python -m natlang_neuralese.export.adapters --nz adapter.nz [--export NAME] --out DIR [--scale 1.0] [--base HF]

writes DIR/peft/ and DIR/lora.gguf for an Adapter export of a `.nz` file (`learning.save`). The fork loads the GGUF
with `PUT /v1/neuralese/adapters/{block id}/lora`; vLLM serves DIR/peft.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from .gguf import fork_root


def export_peft(bank, spec, coefficients: torch.Tensor, out_dir: str | Path, scale: float = 1.0,
                base_model: str | None = None) -> Path:
    """The adapter as a PEFT LoRA directory (`adapter_config.json`, `adapter_model.safetensors`)."""
    from safetensors.torch import save_file

    bank.check(spec, coefficients)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, modules = {}, set()
    with torch.no_grad():
        for module, (a, b) in bank.lora(spec, coefficients.detach().float()).items():
            tensors[f"base_model.model.{module}.lora_A.weight"] = a.float().cpu().contiguous()
            tensors[f"base_model.model.{module}.lora_B.weight"] = (b.float() * scale).cpu().contiguous()
            modules.add(module.rsplit(".", 1)[-1])
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": spec.rank, "lora_alpha": spec.rank,
              "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "target_modules": sorted(modules),
              "layers_to_transform": list(spec.layers), "base_model_name_or_path": base_model,
              "natlang_adapter": {"dialect": spec.dialect(), "scale": scale}}
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


def export_lora_gguf(peft_dir: str | Path, base_hf_dir: str | Path, out_file: str | Path, outtype: str = "f32") -> Path:
    """A PEFT LoRA directory as a GGUF LoRA for the fork (`--base` is the HF base model directory)."""
    out_file = Path(out_file)
    subprocess.run([sys.executable, str(fork_root() / "convert_lora_to_gguf.py"), str(peft_dir), "--base",
                    str(base_hf_dir), "--outfile", str(out_file), "--outtype", outtype], check=True)
    return out_file


def main(argv=None) -> int:
    import argparse

    from ..model.lfm2_port import resolve_base
    from ..model.tiny_adapters import AdapterSpec, is_adapter_dialect
    from ..nz import read_nz
    from ..serve import load_engine

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nz", required=True)
    parser.add_argument("--export", default=None, help="the export's name (default: the file's only adapter)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--base", default=None, help="the HF base model (default: the served base)")
    args = parser.parse_args(argv)
    _, exports = read_nz(args.nz)
    adapters = {name: e for name, e in exports.items() if e.dialect and is_adapter_dialect(e.dialect)}
    if not adapters:
        raise SystemExit(f"{args.nz} has no adapter export")
    if args.export is None and len(adapters) > 1:
        raise SystemExit(f"several adapters ({', '.join(sorted(adapters))}): pass --export")
    entry = adapters[args.export] if args.export else next(iter(adapters.values()))
    engine = load_engine(args.base, device="cpu", dtype=torch.float32)
    spec = AdapterSpec.parse(entry.dialect)
    if spec.version < 2:
        print("warning: an adapter/1 block's bases have device-dependent signs; this export matches the adapter only "
              "if its SVD on this CPU agrees with the device that trained it", file=sys.stderr)
    out = Path(args.out)
    peft = export_peft(engine.adapter_bank, spec, entry.payload, out / "peft", scale=args.scale, base_model=args.base)
    gguf = export_lora_gguf(peft, resolve_base(args.base), out / "lora.gguf")
    print(json.dumps({"adapter": entry.block_id, "dialect": entry.dialect, "peft": str(peft), "gguf": str(gguf)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
