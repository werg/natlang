"""Stage 1 of the delta projection D (LEARNING_CONTINUUM.md §5, §13a): fit D to recorded soft-skill deltas.

A soft delta is `after − before` of an improvement step that changed a soft skill in place (gradient refinement or a
learned merge; same length). Each delta gets a free written block (a short code in the model's dialect), optimised
jointly with the shared D so that D(code, before) reconstructs the delta; the error is relative to the delta's norm.
Steps that replace the block (re-embedding a text, even at the same length) are not deltas and are left out.

The round-trip check runs on held-out deltas: with D fixed, only a fresh code is optimised for each. The summary
reports the same round trip through an untrained D (control), as for P: a long code can reach much through any D, so the
fit is worth something only where it beats that.

Usage: python -m natlang_neuralese.train.delta_projection --runs STEPS.jsonl=ARTIFACTS.nz [...] --out DIR
         [--code-length 8 --hidden 256 --steps 2000 --lr 3e-3 --heldout 0.2 --code-steps 500]

Writes `delta_projection.pt`, `codes.pt` (training codes by step ID) and `summary.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from ..model.projections import DeltaProjection
from ..nz import read_nz
from .projection import relative_error

SCHEMA = "natlang.delta-projection/1"


def load_deltas(runs: list[str]) -> tuple[str, list[dict]]:
    """[{id, operator, base [L, d], delta [L, d]}] from `steps.jsonl=artifacts.nz` pairs."""
    deltas, dialects = [], set()
    for run in runs:
        steps_path, artifacts_path = run.split("=", 1)
        _, exports = read_nz(artifacts_path)
        blocks = {e.block_id: (e.payload, e.dialect) for e in exports.values() if e.payload is not None and e.type != "Adapter"}
        with open(steps_path) as stream:
            for line in stream:
                step = json.loads(line)
                before = [r["id"] for r in step["before"] if r["kind"] == "soft-skill"]
                after = [r["id"] for r in step["after"] if r["kind"] == "soft-skill"]
                if len(before) != 1 or len(after) != 1 or before[0] == after[0] or "embed" in step["operator"]["kind"]:
                    continue
                if before[0] not in blocks or after[0] not in blocks:
                    continue
                (base, dialect), (changed, _) = blocks[before[0]], blocks[after[0]]
                if base.shape != changed.shape:
                    continue
                deltas.append({"id": step["id"], "operator": step["operator"]["kind"], "base": base, "delta": changed - base})
                dialects.add(dialect)
    if len(dialects) != 1:
        raise SystemExit(f"deltas must share one dialect; found {sorted(dialects)}")
    return dialects.pop(), deltas


def round_trip(projection: DeltaProjection, items: list[dict], length: int, steps: int, lr: float, seed: int) -> list[float]:
    """Per-delta relative error after fitting a fresh code for each with D fixed."""
    generator = torch.Generator().manual_seed(seed)
    errors = []
    for item in items:
        code = (0.02 * torch.randn(length, projection.dim, generator=generator)).requires_grad_(True)
        optimizer = torch.optim.Adam([code], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = relative_error(projection(code, item["base"])[None], item["delta"][None])
            loss.backward()
            optimizer.step()
        errors.append(float(loss.detach()))
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", nargs="+", required=True, help="STEPS.jsonl=ARTIFACTS.nz")
    parser.add_argument("--out", required=True)
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
    dialect, deltas = load_deltas(args.runs)
    if len(deltas) < 3:
        raise SystemExit(f"only {len(deltas)} soft deltas found")
    rng = random.Random(args.seed)
    rng.shuffle(deltas)
    held, train = deltas[:max(1, int(len(deltas) * args.heldout))], deltas[max(1, int(len(deltas) * args.heldout)):]
    width = deltas[0]["base"].shape[1]
    torch.manual_seed(args.seed)
    projection = DeltaProjection(dialect, width, args.hidden)
    # The output starts at zero (identity update); a small start lets the joint fit leave the saddle.
    with torch.no_grad():
        projection.out.weight.normal_(0, 1e-3)
    codes = [(0.02 * torch.randn(args.code_length, width)).requires_grad_(True) for _ in train]
    optimizer = torch.optim.Adam([{"params": projection.parameters()}, {"params": codes}], lr=args.lr)
    log = open(out / "train.jsonl", "w")

    def train_error():
        return sum(relative_error(projection(c, d["base"])[None], d["delta"][None]) for c, d in zip(codes, train)) / len(train)

    for step in range(args.steps):
        optimizer.zero_grad()
        loss = train_error()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            log.write(json.dumps({"step": step, "relative_error": float(loss.detach())}) + "\n")
            log.flush()
    for p in projection.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        fitted = float(train_error())
    held_errors = round_trip(projection, held, args.code_length, args.code_steps, args.lr, args.seed + 1)
    control = DeltaProjection(dialect, width, args.hidden)
    with torch.no_grad():
        control.out.weight.normal_(0, 0.05)
    for p in control.parameters():
        p.requires_grad_(False)
    control_errors = round_trip(control, held, args.code_length, args.code_steps, args.lr, args.seed + 1)

    digest = hashlib.sha256(f"{SCHEMA}\0{dialect}".encode())
    for name, tensor in sorted(projection.state_dict().items()):
        digest.update(name.encode() + tensor.float().numpy().tobytes())
    identity = digest.hexdigest()[:16]
    torch.save({"schema": SCHEMA, "source_dialect": dialect, "shape": [width, args.hidden], "state": projection.state_dict(),
                "identity": identity, "train": [d["id"] for d in train], "heldout": [d["id"] for d in held]},
               out / "delta_projection.pt")
    torch.save({d["id"]: c.detach() for c, d in zip(codes, train)}, out / "codes.pt")
    mean = lambda xs: sum(xs) / len(xs)
    summary = {"schema": "natlang.delta-projection-fit/1", "source": dialect, "deltas": len(deltas), "train": len(train),
               "heldout": len(held), "operators": sorted({d["operator"] for d in deltas}),
               "train_relative_error": fitted, "heldout_relative_error": mean(held_errors),
               "heldout_relative_error_random_projection": mean(control_errors),
               "heldout_deltas": [{"id": d["id"], "operator": d["operator"], "error": e, "random": r}
                           for d, e, r in zip(held, held_errors, control_errors)],
               "identity": identity, "options": vars(args)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("deltas", "train_relative_error", "heldout_relative_error",
                                             "heldout_relative_error_random_projection")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
