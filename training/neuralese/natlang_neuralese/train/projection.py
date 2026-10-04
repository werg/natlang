"""Stage 1 of the adapter projection P (LEARNING_CONTINUUM.md §6.4): fit P to directly trained adapters.

Each trained adapter gets a free code (a short block in the model's dialect), optimised jointly with the shared P to
reconstruct the adapter's coefficients; the error is relative to the adapter's norm, so large and small adapters
count alike. Zero adapters (starting states) are left out. The round-trip check of §6.4 runs on held-out adapters: with P fixed, only a fresh code is optimised for
each one. A projection whose held-out error is near its training error has learned a code space that covers new
adapters rather than memorising the training set; the summary reports the same round trip through an untrained
projection as the control (with long codes almost any projection can reach any adapter). Codes need several vectors:
with one, every output row pools the same vector.

Usage: python -m natlang_neuralese.train.projection --adapters A.nz [B.nz ...] --out DIR [--code-length 8
         --hidden 256 --steps 2000 --lr 3e-3 --heldout 0.2 --code-steps 500]

Inputs are `.nz` files with `Adapter` exports (the method-arm and memetic runners write them); every adapter must
share one spec (its dialect). Writes `projection.pt`, `codes.pt` (training codes by adapter ID) and `summary.json`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from ..model.projections import AdapterProjection
from ..model.tiny_adapters import AdapterSpec
from ..nz import read_nz


def load_adapters(paths: list[str]) -> tuple[str, dict[str, torch.Tensor]]:
    adapters, dialects = {}, set()
    for path in paths:
        _, exports = read_nz(path)
        for export in exports.values():
            # Zero adapters are starting states (the steps' `before`), not trained adapters to reconstruct.
            if export.type == "Adapter" and export.payload is not None and float(export.payload.abs().max()) > 0:
                adapters[export.block_id] = export.payload
                dialects.add(export.dialect)
    if len(dialects) != 1:
        raise SystemExit(f"adapters must share one spec; found {sorted(dialects)}")
    return dialects.pop(), adapters


def relative_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-adapter ‖P(code) − a‖² / ‖a‖², averaged."""
    dims = tuple(range(1, target.dim()))
    return ((predicted - target).pow(2).sum(dims) / target.pow(2).sum(dims).clamp_min(1e-12)).mean()


def fit_codes(projection, targets: torch.Tensor, length: int, width: int, steps: int, lr: float, seed: int) -> tuple:
    """Codes for `targets` with the projection fixed (the round-trip check)."""
    generator = torch.Generator().manual_seed(seed)
    codes = (0.02 * torch.randn(targets.shape[0], length, width, generator=generator)).requires_grad_(True)
    optimizer = torch.optim.Adam([codes], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = relative_error(projection(codes), targets)
        loss.backward()
        optimizer.step()
    return codes.detach(), float(loss.detach())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-dialect", default="nd:natlang@1")
    parser.add_argument("--width", type=int, default=1024, help="the model's hidden size (code vector width)")
    parser.add_argument("--code-length", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--code-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--heldout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    target_dialect, adapters = load_adapters(args.adapters)
    spec = AdapterSpec.parse(target_dialect)
    ids = sorted(adapters)
    random.Random(args.seed).shuffle(ids)
    held = ids[:max(1, int(len(ids) * args.heldout))] if len(ids) > 2 else []
    train = [i for i in ids if i not in held]
    rows, width = adapters[ids[0]].shape
    if width != spec.width:
        raise SystemExit(f"coefficients are {width} wide, the spec says {spec.width}")
    torch.manual_seed(args.seed)
    projection = AdapterProjection(args.source_dialect, target_dialect, args.width, rows, width, args.hidden)
    # The output layer starts at zero (identity update); a small start lets the joint fit leave the saddle.
    with torch.no_grad():
        projection.project.out.weight.normal_(0, 1e-3)
    targets = torch.stack([adapters[i] for i in train])
    codes = (0.02 * torch.randn(len(train), args.code_length, args.width)).requires_grad_(True)
    optimizer = torch.optim.Adam([{"params": projection.parameters()}, {"params": [codes]}], lr=args.lr)
    log = open(out / "train.jsonl", "w")
    for step in range(args.steps):
        optimizer.zero_grad()
        loss = relative_error(projection(codes), targets)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            log.write(json.dumps({"step": step, "relative_error": float(loss.detach())}) + "\n")
            log.flush()
    for p in projection.parameters():
        p.requires_grad_(False)
    train_error = float(relative_error(projection(codes.detach()), targets))
    held_error = random_error = None
    if held:
        held_targets = torch.stack([adapters[i] for i in held])
        _, held_error = fit_codes(projection, held_targets, args.code_length, args.width, args.code_steps, args.lr, args.seed + 1)
        # Control: the same round trip through an untrained projection (random queries and maps, small random
        # output). A long code can reach any adapter through almost any projection; the fit is worth something only
        # where it beats this.
        control = AdapterProjection(args.source_dialect, target_dialect, args.width, rows, width, args.hidden)
        with torch.no_grad():
            control.project.out.weight.normal_(0, 0.05)
        for p in control.parameters():
            p.requires_grad_(False)
        _, random_error = fit_codes(control, held_targets, args.code_length, args.width, args.code_steps, args.lr, args.seed + 1)
    projection.save(out / "projection.pt", adapters=len(ids), train=train, heldout=held)
    torch.save({i: c for i, c in zip(train, codes.detach())}, out / "codes.pt")
    summary = {"schema": "natlang.projection-fit/1", "target": target_dialect, "source": args.source_dialect,
               "adapters": len(ids), "train": len(train), "heldout": len(held), "rows": rows, "width": width,
               "train_relative_error": train_error, "heldout_relative_error": held_error,
               "heldout_relative_error_random_projection": random_error,
               "identity": projection.identity(), "options": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("adapters", "train_relative_error", "heldout_relative_error",
                                             "heldout_relative_error_random_projection")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
