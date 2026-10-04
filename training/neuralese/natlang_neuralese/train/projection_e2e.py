"""Stage 2 of the adapter projection P (LEARNING_CONTINUUM.md §6.4): train P end to end on downstream decision loss.

Each training family gets a code (a free block in the model's dialect); its adapter is P(code); the decision readout's
cross-entropy on the family's support cases is backpropagated through the adapted model into P and the codes. P is
shared across families, so it learns a code space of useful adapters.

With `--codes written` the codes are not free: they are blocks the model writes itself (a forced Neuralese write after
a prompt showing the family's instructions and a few support cases with their answers; `--writes` per family, each
from different cases), and only P is trained. A held-out family then gets its adapter zero-shot from its own written
note. Controls: `--codes embedded` uses the token embeddings of the same prompt in place of the written block, and
`--codes encoded` its one-pass encoding through the port (does the model's writing add over reading the raw text?).

Evaluation on held-out families (never seen by P), each adapted with the same number of optimiser steps on its own
support cases and scored on its held-out query cases (readout quality, 1 − Brier/2 or 1 − RPS):

  code       a fresh code through the trained P (P fixed): adapters written as Neuralese;
  direct     an `xs` adapter of the same spec trained directly (the method-arm baseline);
  random-P   a fresh code through an untrained P (control: how much of the code result is P's structure);
  none       no adapter;

and with written or embedded codes also

  zero-shot  P applied to the family's own code (mean over its writes), no optimisation;
  shuffled   P applied to another held-out family's code (control: does the code's content matter);
  code-init  the family's code as the start of the code adaptation.

Usage: python -m natlang_neuralese.train.projection_e2e --prompts prompts.jsonl --out DIR --families a,b,...
         --heldout-families c,d [--projection stage1/projection.pt --heads CKPT --support 16 --query 24
         --code-length 8 --steps 300 --batch 4 --lr 1e-3 --code-lr 1e-2 --adapter-lr 0.05 --adapt-steps 16 --rank 4
         --codes free|written|embedded --writes 4 --write-examples 4]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from ..model.projections import AdapterProjection
from ..model.tiny_adapters import AdapterSpec, active
from .decision import scores, target_distribution

_INSTRUCTIONS = re.compile(r"Instructions:\n(.*?)\n\nIn eval", re.S)
_STATE = re.compile(r"^state: string = (.*)\nDeclared state", re.S)


def write_prompt(cases: list[dict]) -> str:
    """The family's instructions and a few cases with their answers, asking for a note on how to decide."""
    def text(row, role):
        return next(m["content"] for m in row["messages"] if m["role"] == role and isinstance(m["content"], str))
    instructions = _INSTRUCTIONS.search(text(cases[0], "user"))
    examples = []
    for row in cases:
        state = _STATE.search(text(row, "tool"))
        answer = row["options"][max(range(len(row["gold"])), key=lambda i: row["gold"][i])]
        examples.append(f"Input: {json.loads(state.group(1)) if state else '?'}\nAnswer: {answer}")
    return (f"A function decides: {instructions.group(1).strip() if instructions else '?'}\n"
            f"Allowed answers: {', '.join(cases[0]['options'])}\n\nExamples:\n" + "\n\n".join(examples) +
            "\n\nWrite a short note for whoever answers this function next: how to decide well.")


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
    parser.add_argument("--codes", choices=["free", "written", "embedded", "encoded"], default="free")
    parser.add_argument("--writes", type=int, default=4, help="written or embedded codes per family")
    parser.add_argument("--write-examples", type=int, default=4, help="support cases shown in each write prompt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from ..serve import load_engine
    from ..serve.grad import GradSession, encode_text

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

    def given_codes(family) -> list[torch.Tensor]:
        """The family's written (or embedded) codes, each from a different window of its support cases."""
        from ..serve.engine import GenerationRequest
        from ..serve.grad import embed_text

        support, k = split[family]["support"], args.write_examples
        result = []
        for i in range(args.writes):
            prompt = write_prompt([support[(i * k + j) % len(support)] for j in range(k)])
            if args.codes == "embedded":
                block = embed_text(engine, prompt)
            elif args.codes == "encoded":
                block = encode_text(engine, prompt)
            else:
                response = engine.generate(GenerationRequest(messages=[{"role": "user", "content": prompt}],
                                                             forced=["Note: ", {"neuralese": "write"}], max_tokens=engine.max_block + 8))
                block = engine.lookup(response["neuralese"]["blocks"][0]["id"])
            result.append(block.payload.float().to(device))
        return result

    # Training: P and one code per training family (free), or P alone on the family's given codes.
    if args.codes == "free":
        codes = {f: [(0.02 * torch.randn(args.code_length, engine.width, device=device)).requires_grad_(True)] for f in families}
        groups = [{"params": projection.parameters(), "lr": args.lr}, {"params": [c for cs in codes.values() for c in cs], "lr": args.code_lr}]
    else:
        codes = {f: given_codes(f) for f in families + heldout}
        (out / "codes.json").write_text(json.dumps({f: [list(c.shape) for c in cs] for f, cs in codes.items()}) + "\n")
        groups = [{"params": projection.parameters(), "lr": args.lr}]
    optimizer = torch.optim.Adam(groups)
    log = open(out / "train.jsonl", "w")
    started = time.time()
    with torch.enable_grad():
        for step in range(args.steps):
            family = families[step % len(families)]
            batch = random.sample(split[family]["support"], min(args.batch, args.support))
            optimizer.zero_grad(set_to_none=True)
            code = random.choice(codes[family])
            value = loss_on(batch, lambda: projection(code))
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
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

    def code_arm(p, start=None):
        def make():
            code = (start.clone() if start is not None else 0.02 * torch.randn(args.code_length, engine.width, device=device)).requires_grad_(True)
            return [code], lambda: p(code)
        return make

    def zero_shot(cases, family):
        with torch.no_grad():
            return sum(evaluate(cases, projection(c)) for c in codes[family]) / len(codes[family])

    def direct_arm():
        coefficients = torch.zeros(projection.project.rows, spec.width, device=device, requires_grad=True)
        return [coefficients], lambda: coefficients

    results = {"train_families": {}, "heldout_families": {}}
    for f in families:
        with torch.no_grad():
            results["train_families"][f] = {"code": sum(evaluate(split[f]["query"], projection(c)) for c in codes[f]) / len(codes[f]),
                                            "none": evaluate(split[f]["query"], None)}
    for index, f in enumerate(heldout):
        query = split[f]["query"]
        arms = {"none": evaluate(query, None),
                "code": evaluate(query, adapt(code_arm(projection), args.code_lr, f)),
                "direct": evaluate(query, adapt(direct_arm, args.adapter_lr, f)),
                "random-P": evaluate(query, adapt(code_arm(control), args.code_lr, f))}
        if args.codes != "free":
            arms["zero-shot"] = zero_shot(query, f)
            arms["shuffled"] = zero_shot(query, heldout[(index + 1) % len(heldout)])
            arms["code-init"] = evaluate(query, adapt(code_arm(projection, codes[f][0]), args.code_lr, f))
        results["heldout_families"][f] = arms
        print(json.dumps({"family": f, **{k: round(v, 4) for k, v in arms.items()}}), flush=True)
    summary = {"schema": "natlang.projection-e2e/1", "spec": spec.dialect(), "identity": projection.identity(),
               "options": vars(args), **results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
