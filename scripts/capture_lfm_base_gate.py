#!/usr/bin/env python3
"""Measure the pinned 48-probe behavior gate for an LFM base and clean heads."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training/neuralese"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--probes-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    if digest(args.probes) != args.probes_sha256:
        raise ValueError("probe bytes differ from the pinned SHA-256")
    probes = [json.loads(line) for line in args.probes.read_text().splitlines() if line.strip()]
    if len(probes) != 48 or len({probe.get("id") for probe in probes}) != 48:
        raise ValueError("the behavior reference requires 48 unique probes")
    if args.out.exists():
        raise FileExistsError(args.out)

    import torch
    from natlang_neuralese.serve import load_engine
    from natlang_neuralese.maple.generation_gate import run_gate
    from natlang_neuralese.maple.distill_data import render

    start = time.monotonic()
    engine = load_engine(base=str(args.base.resolve(strict=True)),
                         heads_checkpoint=str(args.heads.resolve(strict=True)), device=args.device)
    with torch.inference_mode():
        result = run_gate(engine.backbone.hf, engine.tokenizer, probes, render,
                          engine.tokenizer.convert_tokens_to_ids("<|im_end|>"), keep_text=True)
    result.update(schema="natlang.base-behavior-gate-reference/1",
                  seconds=time.monotonic() - start,
                  base=str(args.base.resolve()),
                  heads_sha256=digest(args.heads), probe_sha256=digest(args.probes),
                  training_admission=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"gate_pass": result["gate_pass"], "seconds": result["seconds"],
                      "report": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
