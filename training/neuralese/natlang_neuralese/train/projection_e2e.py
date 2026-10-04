"""Stage 2 of the adapter projection P (LEARNING_CONTINUUM.md §6.4): train P end to end on downstream decision loss.

Each training family gets a code (a free block in the model's dialect); its adapter is P(code); the decision readout's
cross-entropy on the family's support cases is backpropagated through the adapted model into P and the codes. P is
shared across families, so it learns a code space of useful adapters.

Evaluation on held-out families (never seen by P), each adapted with the same number of optimiser steps on its own
support cases and scored on its held-out query cases (readout quality, 1 − Brier/2 or 1 − RPS):

  code       a fresh code through the trained P (P fixed): adapters written as Neuralese;
  direct     an `xs` adapter of the same spec trained directly (the method-arm baseline);
  random-P   a fresh code through an untrained P (control: how much of the code result is P's structure);
  none       no adapter.

Usage: python -m natlang_neuralese.train.projection_e2e --prompts prompts.jsonl --out DIR --families a,b,...
         --heldout-families c,d [--projection stage1/projection.pt --heads CKPT --support 16 --query 24
         --code-length 8 --steps 300 --batch 4 --lr 1e-3 --code-lr 1e-2 --adapter-lr 0.05 --adapt-steps 16 --rank 4]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from ..model.projections import AdapterProjection
from ..model.tiny_adapters import AdapterSpec, active
from .decision import scores, target_distribution


def quality(kind: str, predicted: list[float], gold: list[float]) -> float:
    s = scores(kind, predicted, gold)
    return 1 - s["rps"] if "rps" in s else 1 - s["brier"] / 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--heldout-families", required=True)
    parser.add_argument("--projection", default=None, help="stage-1 projection to start from (else a fresh one)")
    parser.add_argument("--heads", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=6)
    parser.add_argument("--support", type=int, default=16)
    parser.add_argument("--query", type=int, default=24)
    parser.add_argument("--max-options", type=int, default=12)
    parser.add_argument("--code-length", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--code-lr", type=float, default=1e-2)
    parser.add_argument("--adapter-lr", type=float, default=0.05)
    parser.add_argument("--adapt-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from ..serve import load_engine
    from ..serve.grad import GradSession

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    families, heldout = args.families.split(","), args.heldout_families.split(",")
    rows = defaultdict(lambda: {"train": [], "heldout": []})
    with open(args.prompts) as stream:
        for line in stream:
            row = json.loads(line)
            if row["family"] in families + heldout and len(row["options"]) <= args.max_options:
                rows[row["family"]][row["role"]].append(row)
    for f in families + heldout:
        if len(rows[f]["train"]) < args.support or len(rows[f]["heldout"]) < args.query:
            raise SystemExit(f"{f}: not enough cases")
    split = {f: {"support": rows[f]["train"][:args.support], "query": rows[f]["heldout"][:args.query]} for f in families + heldout}

    engine = load_engine(heads_checkpoint=args.heads, device=args.device)
    session, bank, device = GradSession(engine), engine.adapter_bank, engine.device
    if args.projection:
        projection = AdapterProjection.load(args.projection, map_location=device)
        spec = AdapterSpec.parse(projection.target)
        if spec.base != bank.base_hash():
            raise SystemExit(f"the projection decodes adapters for base {spec.base}, this model is {bank.base_hash()}")
    else:
        spec = bank.spec(kind="xs", rank=args.rank, cutoff=engine.heads.cutoff)
        projection = AdapterProjection(engine.dialect, spec.dialect(), engine.width, len(bank.matrices(spec)), spec.width, args.hidden)
        with torch.no_grad():
            projection.project.out.weight.normal_(0, 1e-3)
    projection = projection.to(device)
    bank.install(spec)
    for p in engine.backbone.parameters():
        p.requires_grad_(False)

    def logp(row, coefficients):
        with active([[(spec, coefficients, 1.0)]] if coefficients is not None else None):
            scores_, _ = session.decision_logprobs(row["messages"], None, row["options"], {})
        return torch.log_softmax(scores_.float(), 0)

    def loss_on(batch, coefficients_of):
        total = 0.0
        for row in batch:
            goal = torch.tensor(target_distribution(row, "gold"), device=device)
            goal = goal / goal.sum()
            value = -(goal * logp(row, coefficients_of())).sum() / len(batch)
            value.backward()
            total += float(value.detach())
        return total

    def evaluate(cases, coefficients):
        with torch.no_grad():
            return sum(quality(r["kind"], logp(r, coefficients).exp().tolist(), r["gold"]) for r in cases) / len(cases)

    # Training: P and one code per training family.
    codes = {f: (0.02 * torch.randn(args.code_length, engine.width, device=device)).requires_grad_(True) for f in families}
    optimizer = torch.optim.Adam([{"params": projection.parameters(), "lr": args.lr},
                                  {"params": list(codes.values()), "lr": args.code_lr}])
    log = open(out / "train.jsonl", "w")
    started = time.time()
    with torch.enable_grad():
        for step in range(args.steps):
            family = families[step % len(families)]
            batch = random.sample(split[family]["support"], min(args.batch, args.support))
            optimizer.zero_grad(set_to_none=True)
            value = loss_on(batch, lambda: projection(codes[family]))
            torch.nn.utils.clip_grad_norm_(list(projection.parameters()) + list(codes.values()), 1.0)
            optimizer.step()
            if step % 10 == 0 or step == args.steps - 1:
                log.write(json.dumps({"step": step, "family": family, "loss": value, "seconds": round(time.time() - started)}) + "\n")
                log.flush()
    projection.save(out / "projection.pt", stage=2, families=families)
    for p in projection.parameters():
        p.requires_grad_(False)

    def adapt(make, lr, family):
        """`make()` → (parameters, coefficient function); Adam for adapt-steps on the family's support."""
        params, coefficients_of = make()
        opt = torch.optim.Adam(params, lr=lr)
        with torch.enable_grad():
            for step in range(args.adapt_steps):
                opt.zero_grad(set_to_none=True)
                loss_on(random.sample(split[family]["support"], min(args.batch, args.support)), coefficients_of)
                opt.step()
        with torch.no_grad():
            return coefficients_of().detach()

    control = AdapterProjection(engine.dialect, spec.dialect(), engine.width, projection.project.rows, spec.width, args.hidden).to(device)
    with torch.no_grad():
        control.project.out.weight.normal_(0, 1e-3)
    for p in control.parameters():
        p.requires_grad_(False)

    def code_arm(p):
        def make():
            code = (0.02 * torch.randn(args.code_length, engine.width, device=device)).requires_grad_(True)
            return [code], lambda: p(code)
        return make

    def direct_arm():
        coefficients = torch.zeros(projection.project.rows, spec.width, device=device, requires_grad=True)
        return [coefficients], lambda: coefficients

    results = {"train_families": {}, "heldout_families": {}}
    for f in families:
        with torch.no_grad():
            results["train_families"][f] = {"code": evaluate(split[f]["query"], projection(codes[f])),
                                            "none": evaluate(split[f]["query"], None)}
    for f in heldout:
        query = split[f]["query"]
        arms = {"none": evaluate(query, None),
                "code": evaluate(query, adapt(code_arm(projection), args.code_lr, f)),
                "direct": evaluate(query, adapt(direct_arm, args.adapter_lr, f)),
                "random-P": evaluate(query, adapt(code_arm(control), args.code_lr, f))}
        results["heldout_families"][f] = arms
        print(json.dumps({"family": f, **{k: round(v, 4) for k, v in arms.items()}}), flush=True)
    summary = {"schema": "natlang.projection-e2e/1", "spec": spec.dialect(), "identity": projection.identity(),
               "options": vars(args), **results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
