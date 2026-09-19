#!/usr/bin/env python3
"""Per-turn evaluation against the reference corpus (teacher-forced): the model sees the reference history of an
episode and produces the next turn under the turn's own grammar; does it match the reference turn?

  scripts/eval_turns.py data/ref-v5.jsonl --per-cell 12 --server http://127.0.0.1:8080
Reports, per family group and kind of turn: exact match, right tool, right function (for call).
A value written by a leaf is compared exactly, so leaf accuracy is a lower bound where wording may vary.
"""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.native import NativeCallDecoder


def group(family):
    return "codebase" if family.startswith("cb_") else "composed" if family == "composed" else \
        "leaf" if family in ("judge", "classify", "extract", "crisp_scalar") else "shape"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan", type=int, default=60000, help="how many lines of the file to sample from")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    cells = defaultdict(list)
    with a.data.open() as f:
        for i, line in enumerate(f):
            if i >= a.scan:
                break
            if rng.random() < 0.15:
                s = json.loads(line)
                cells[(group(s["family"]), s["skill"])].append(s)
    dec = NativeCallDecoder(a.server, timeout=300)
    rows = {}
    for cell, samples in sorted(cells.items()):
        rng.shuffle(samples)
        n = exact = tool = fn = 0
        for s in samples[: a.per_cell]:
            turn = dec.chat(s["messages"], s["tools"], temperature=0.0, seed=1, max_tokens=500)
            want = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in s["target"].get("tool_calls") or []]
            got = turn.calls
            n += 1
            if not want:
                exact += not got; tool += not got
                continue
            exact += got == want
            tool += bool(got) and got[0][0] == want[0][0]
            if want[0][0] == "call":
                fn += bool(got) and got[0][0] == "call" and got[0][1].get("function") == want[0][1].get("function")
        rows[cell] = (n, exact, tool, fn)
        print(f"{cell[0]:<9} {cell[1]:<15} n={n:<3} exact={exact / n:4.0%}  right tool={tool / n:4.0%}"
              + (f"  right function={fn / n:4.0%}" if cell[1] == "call" else ""), flush=True)
    total = sum(r[0] for r in rows.values())
    print(f"\nall turns: exact={sum(r[1] for r in rows.values()) / total:.0%}  right tool={sum(r[2] for r in rows.values()) / total:.0%}  (n={total})")


if __name__ == "__main__":
    main()
