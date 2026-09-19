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
from natlang.gen.codebases import CODEBASES
FAMILIES = {**FAMILIES, **SHAPES, **CODEBASES}
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

    if prog.loader is not None:
        root = prog.loader()
    else:
        root = load_program(prog.root)
        env = root.env(TypeEnv())
        for name, value in prog.inputs.items():
            root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
    rt = Runtime(factory, max_episodes=2000, capabilities=prog.capabilities)
    out, value = rt.run_root(root)
    if prog.expected == BLOCKED:                 # undetermined on purpose: the right outcome is a blocker note
        assert out.kind == "quiesced" and out.detail.startswith("blocked: "), (out.kind, out.detail)
        return samples, rt.episodes_started
    assert out.kind == "done", out.detail
    if callable(prog.expected):
        assert prog.expected(dump(value)), dump(value)
    else:
        assert dump(value) == prog.expected, (dump(value), prog.expected)
    return samples, rt.episodes_started


MIXES = {
    # share of programs per family. Leaves stay about a fifth; synthesized structure is the bulk; code bases the rest.
    "v7": {"judge": 3, "classify": 4, "extract": 4, "crisp_scalar": 3, "tiny": 8,
           "map_leaf": 2, "map_then_count": 2, "ticket_report": 3, "review_digest": 3, "expense_audit": 3,
           "nested_assessment": 3, "per_item_condition": 7, "fold_with_steps": 6, "composed": 30,
           "cb_legal_move": 3, "cb_moderation": 3, "cb_nlprolog": 3, "cb_shopkeeper": 3, "cb_webserver": 3,
           "cb_highlighter": 2, "cb_mail_rules": 3},
}


def make_program(seed: int, i: int, families):
    """Program number i of a run: a function of (seed, i) only, so that runs can be sharded, parallel, resumed, and
    revisited (scripts/teacher_leaves.py) without generating everything before it. `families` is a list (round robin)
    or a dict of weights."""
    rng = random.Random(f"{seed}:{i}")
    if isinstance(families, dict):
        fam = rng.choices(list(families), weights=list(families.values()))[0]
    else:
        fam = families[i % len(families)]
    return fam, FAMILIES[fam](rng)


def _shard(job):
    seed, first, last, families, keep_template, keep_alternatives = job
    from natlang.native import _strip_private
    lines, stats, episodes = [], Counter(), 0
    for i in range(first, last):
        fam, prog = make_program(seed, i, families)
        samples, eps = run_program(prog)
        episodes += eps
        for j, s in enumerate(samples):
            if s.get("template") and not keep_template:
                continue                      # generative leaves wait for teacher-written references
            if not keep_alternatives:         # the per-turn grammar alternatives are most of a sample's size and are
                s = {**s, "tools": _strip_private(s["tools"])}     # only needed to evaluate under the grammar
            lines.append(json.dumps({"id": f"{fam}-{i}-{j}", "program_id": f"{seed}:{fam}:{i}", "family": fam, **s}, default=str))
            stats[(fam, s["skill"])] += 1
    return first, lines, stats, episodes


def main():
    import gzip, multiprocessing, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/ref-v0.jsonl"),
                    help="a .jsonl / .jsonl.gz file, or a directory: then shards are written and finished shards are kept on a rerun")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--mix", choices=sorted(MIXES), default=None, help="weighted family mix instead of --families")
    ap.add_argument("--keep-template-leaves", action="store_true")
    ap.add_argument("--keep-alternatives", action="store_true", help="keep the grammar alternatives (needed by eval_turns.py)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--shard", type=int, default=250, help="programs per shard")
    a = ap.parse_args()
    families = MIXES[a.mix] if a.mix else (a.families or list(FAMILIES))
    sharded = a.out.suffix not in (".jsonl", ".gz")
    (a.out if sharded else a.out.parent).mkdir(parents=True, exist_ok=True)
    name = lambda first: a.out / f"part-{first:07d}.jsonl.gz"
    jobs = [(a.seed, first, min(first + a.shard, a.n), families, a.keep_template_leaves, a.keep_alternatives)
            for first in range(0, a.n, a.shard) if not (sharded and name(first).exists())]
    stats, episodes, total = Counter(), 0, 0
    single = None if sharded else (gzip.open(a.out, "wt") if a.out.suffix == ".gz" else a.out.open("w"))
    with multiprocessing.Pool(a.workers) as pool:
        results = pool.imap(_shard, jobs) if not sharded else pool.imap_unordered(_shard, jobs)
        for first, lines, st, eps in results:
            if sharded:
                tmp = name(first).with_suffix(".tmp")
                with gzip.open(tmp, "wt") as f:
                    f.write("\n".join(lines) + "\n")
                tmp.rename(name(first))              # a shard exists only when it is complete
            else:
                single.write("\n".join(lines) + "\n")
            stats.update(st); episodes += eps; total += len(lines)
            print(f"  programs {first}..{first + a.shard - 1}: {len(lines)} samples", flush=True)
    if single:
        single.close()
    print(f"{a.n} programs ({len(jobs)} shards run now), {episodes} episodes, {total} samples -> {a.out}")
    groups = Counter()
    for (fam, skill), n in stats.items():
        groups[fam] += n
    for fam, n in sorted(groups.items()):
        print(f"  {fam:<20} {n}")


if __name__ == "__main__":
    main()
