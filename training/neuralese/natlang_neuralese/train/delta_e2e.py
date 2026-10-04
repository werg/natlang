"""Stage 2 of the delta projection D (LEARNING_CONTINUUM.md §5, §13a): train D on what a delta does, not on its shape.

Stage 1 (delta_projection.py) showed that recorded soft-skill deltas (Adam updates) are close to sign noise and do not
reconstruct through a shared D. Here D is trained functionally: each training family's skill is
`base + D(code, base)`, where `base` is the generic text-initialised soft skill; the skill block is shown before the
decision question, and the decision readout's cross-entropy on the family's support cases is backpropagated into D
and the codes. Codes are free, written by the model (`--codes written`, as in projection_e2e.py) or the token
embeddings of the same prompt (`--codes embedded`).

Held-out families (never seen by D), scored on their query cases (readout quality):

  none        no skill;
  base        the generic skill, unchanged;
  code        a fresh code through the trained D, optimised for adapt-steps on the family's support cases;
  direct      the skill itself optimised for the same steps (soft-gold: a delta written freely);
  random-D    a fresh code through an untrained D (control);

and with written or embedded codes also zero-shot (D applied to the family's own code) and shuffled (another
held-out family's code).

Usage: python -m natlang_neuralese.train.delta_e2e --prompts prompts.jsonl --out DIR --families a,b,...
         --heldout-families c,d [--heads CKPT --codes free|written|embedded --steps 1500 --code-length 8]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from ..model.projections import DeltaProjection
from .decision import target_distribution
from .projection_e2e import quality, write_prompt

BASE_TEXT = "Read the input closely, weigh the evidence for each allowed answer, and give the answer the evidence supports."
SKILL_ID = "nz1_" + "a" * 52  # the leaf name the skill block is shown under; its rows always come from the leaves


def with_skill(messages: list[dict]) -> list[dict]:
    """The decision prompt with the skill block before the call's user turn."""
    out, placed = [], False
    for message in messages:
        if not placed and message["role"] == "user" and isinstance(message["content"], str):
            out.append({**message, "content": [{"type": "text", "text": "Skill: "}, {"type": "neuralese", "id": SKILL_ID},
                                               {"type": "text", "text": "\n\n" + message["content"]}]})
            placed = True
        else:
            out.append(message)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--heldout-families", required=True)
    parser.add_argument("--heads", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=6)
    parser.add_argument("--support", type=int, default=16)
    parser.add_argument("--query", type=int, default=24)
    parser.add_argument("--max-options", type=int, default=12)
    parser.add_argument("--code-length", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--code-lr", type=float, default=1e-2)
    parser.add_argument("--skill-lr", type=float, default=0.02, help="the direct arm (as the method arms' soft-gold)")
    parser.add_argument("--adapt-steps", type=int, default=16)
    parser.add_argument("--codes", choices=["free", "written", "embedded"], default="free")
    parser.add_argument("--writes", type=int, default=4)
    parser.add_argument("--write-examples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from ..serve import load_engine
    from ..serve.engine import GenerationRequest
    from ..serve.grad import GradSession, embed_text

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
    for p in engine.backbone.parameters():
        p.requires_grad_(False)
    session, device = GradSession(engine), engine.device
    base = embed_text(engine, BASE_TEXT).payload.float().to(device)
    projection = DeltaProjection(engine.dialect, engine.width, args.hidden).to(device)
    with torch.no_grad():
        projection.out.weight.normal_(0, 1e-3)

    def logp(row, skill):
        messages = with_skill(row["messages"]) if skill is not None else row["messages"]
        scores, _ = session.decision_logprobs(messages, None, row["options"], {SKILL_ID: skill} if skill is not None else {})
        return torch.log_softmax(scores.float(), 0)

    def loss_on(batch, skill_of):
        total = 0.0
        for row in batch:
            goal = torch.tensor(target_distribution(row, "gold"), device=device)
            value = -(goal / goal.sum() * logp(row, skill_of())).sum() / len(batch)
            value.backward()
            total += float(value.detach())
        return total

    def evaluate(cases, skill):
        with torch.no_grad():
            return sum(quality(r["kind"], logp(r, skill).exp().tolist(), r["gold"]) for r in cases) / len(cases)

    def given_codes(family):
        support, k = split[family]["support"], args.write_examples
        result = []
        for i in range(args.writes):
            prompt = write_prompt([support[(i * k + j) % len(support)] for j in range(k)])
            if args.codes == "embedded":
                block = embed_text(engine, prompt)
            else:
                response = engine.generate(GenerationRequest(messages=[{"role": "user", "content": prompt}],
                                                             forced=["Note: ", {"neuralese": "write"}], max_tokens=engine.max_block + 8))
                block = engine.lookup(response["neuralese"]["blocks"][0]["id"])
            result.append(block.payload.float().to(device))
        return result

    if args.codes == "free":
        codes = {f: [(0.02 * torch.randn(args.code_length, engine.width, device=device)).requires_grad_(True)] for f in families}
        groups = [{"params": projection.parameters(), "lr": args.lr}, {"params": [c for cs in codes.values() for c in cs], "lr": args.code_lr}]
    else:
        codes = {f: given_codes(f) for f in families + heldout}
        groups = [{"params": projection.parameters(), "lr": args.lr}]
    optimizer = torch.optim.Adam(groups)
    log = open(out / "train.jsonl", "w")
    started = time.time()
    with torch.enable_grad():
        for step in range(args.steps):
            family = families[step % len(families)]
            code = random.choice(codes[family])
            optimizer.zero_grad(set_to_none=True)
            value = loss_on(random.sample(split[family]["support"], min(args.batch, args.support)), lambda: base + projection(code, base))
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            optimizer.step()
            if step % 10 == 0 or step == args.steps - 1:
                log.write(json.dumps({"step": step, "family": family, "loss": value, "seconds": round(time.time() - started)}) + "\n")
                log.flush()
    torch.save({"source_dialect": engine.dialect, "shape": [engine.width, args.hidden], "state": projection.state_dict(),
                "stage": 2, "families": families, "base_text": BASE_TEXT}, out / "delta_projection.pt")
    for p in projection.parameters():
        p.requires_grad_(False)
    control = DeltaProjection(engine.dialect, engine.width, args.hidden).to(device)
    with torch.no_grad():
        control.out.weight.normal_(0, 1e-3)
    for p in control.parameters():
        p.requires_grad_(False)

    def adapt(params, skill_of, lr, family):
        opt = torch.optim.Adam(params, lr=lr)
        with torch.enable_grad():
            for _ in range(args.adapt_steps):
                opt.zero_grad(set_to_none=True)
                loss_on(random.sample(split[family]["support"], min(args.batch, args.support)), skill_of)
                opt.step()
        with torch.no_grad():
            return skill_of().detach()

    def code_arm(d, family, start=None):
        code = (start.clone() if start is not None else 0.02 * torch.randn(args.code_length, engine.width, device=device)).requires_grad_(True)
        return adapt([code], lambda: base + d(code, base), args.code_lr, family)

    def direct_arm(family):
        skill = base.clone().requires_grad_(True)
        return adapt([skill], lambda: skill, args.skill_lr, family)

    def zero_shot(cases, code_family):
        with torch.no_grad():
            return sum(evaluate(cases, base + projection(c, base)) for c in codes[code_family]) / len(codes[code_family])

    results = {"train_families": {}, "heldout_families": {}}
    for f in families:
        results["train_families"][f] = {"code": zero_shot(split[f]["query"], f), "base": evaluate(split[f]["query"], base),
                                        "none": evaluate(split[f]["query"], None)}
    for index, f in enumerate(heldout):
        query = split[f]["query"]
        arms = {"none": evaluate(query, None), "base": evaluate(query, base), "code": evaluate(query, code_arm(projection, f)),
                "direct": evaluate(query, direct_arm(f)), "random-D": evaluate(query, code_arm(control, f))}
        if args.codes != "free":
            arms["zero-shot"] = zero_shot(query, f)
            arms["shuffled"] = zero_shot(query, heldout[(index + 1) % len(heldout)])
        results["heldout_families"][f] = arms
        print(json.dumps({"family": f, **{k: round(v, 4) for k, v in arms.items()}}), flush=True)
    (out / "summary.json").write_text(json.dumps({"schema": "natlang.delta-e2e/1", "options": vars(args), **results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
