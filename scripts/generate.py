#!/usr/bin/env python3
"""Generate synthetic programs, run the reference policy through the real harness, write training samples.

One sample = one assistant turn: (messages so far, the turn's tools) -> target turn.
Usage: scripts/generate.py --n 200 --seed 0 --out data/ref-v0.jsonl
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.gen.policy import ReferenceAgent
from natlang.gen.programs import BLOCKED, FAMILIES
from natlang.gen.synth import SHAPES
FAMILIES = {**FAMILIES, **SHAPES}
from natlang.runtime import Runtime
from natlang.types import TypeEnv
from natlang.values import coerce, dump, load_program


def run_program(prog, check_grammar=True):
    samples = []

    def factory(lam):
        plan = prog.plans.get(lam.fn_name) if lam.fn_name else None         # a function of the code base
        plan = plan or prog.plans.get((lam.original_body or lam.body).strip()) or prog.plans.get(lam.body)
        if plan is None:
            plan = next((p for k, p in prog.plans.items() if k.strip() == lam.body.strip()), None)
        assert plan is not None, f"no plan for: {lam.body!r}"
        return ReferenceAgent(plan, samples, check_grammar=check_grammar)

    root = load_program(prog.root)
    env = root.env(TypeEnv())
    for name, value in prog.inputs.items():
        root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
    rt = Runtime(factory)
    out, value = rt.run_root(root)
    if prog.expected == BLOCKED:                 # undetermined on purpose: the right outcome is a blocker note
        assert out.kind == "quiesced" and out.detail.startswith("blocked: "), (out.kind, out.detail)
        return samples, rt.episodes_started
    assert out.kind == "done", out.detail
    assert dump(value) == prog.expected, (dump(value), prog.expected)
    return samples, rt.episodes_started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/ref-v0.jsonl"))
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stats, episodes = Counter(), 0
    with a.out.open("w") as f:
        for i in range(a.n):
            fam = a.families[i % len(a.families)]
            prog = FAMILIES[fam](rng)
            samples, eps = run_program(prog)
            episodes += eps
            for j, s in enumerate(samples):
                f.write(json.dumps({"id": f"{fam}-{i}-{j}", "family": fam, **s}, default=str) + "\n")
                stats[(fam, s["skill"])] += 1
    total = sum(stats.values())
    print(f"{a.n} programs, {episodes} episodes, {total} samples -> {a.out}")
    for (fam, skill), n in sorted(stats.items()):
        print(f"  {fam:<16} {skill:<22} {n}")


if __name__ == "__main__":
    main()
