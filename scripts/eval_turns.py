#!/usr/bin/env python3
"""Paired, teacher-forced next-turn evaluation. Reuses a manifest of exact samples.

The first invocation selects a deterministic sample across the entire corpus;
subsequent models use the same manifest. Exact match measures the canonical
reference, not all possible correct continuations. Program evaluation is baseline.py.
"""
import argparse
import heapq
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.corpus import digest, records
from natlang.native import NativeCallDecoder


def group(family):
    return "codebase" if family.startswith("cb_") else "composed" if family == "composed" else \
        "leaf" if family in ("judge", "classify", "extract", "crisp_scalar") else "shape"


def cell(row):
    return group(row["family"]), row["skill"]


def select_samples(data, manifest_path, per_cell=12, seed=0):
    """Bounded-memory sampling; existing manifests also verify sample contents."""
    if per_cell < 1:
        raise ValueError("--per-cell must be positive")
    path = Path(manifest_path)
    if path.exists():
        manifest = json.loads(path.read_text())
        wanted = {x["id"]: x["sha256"] for x in manifest["samples"]}
        found = {}
        for row in records(data):
            key = row["id"]
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"Duplicate evaluation ID: {key}")
            if digest(row) != wanted[key]:
                raise ValueError(f"Evaluation sample changed: {key}; use a new manifest for a new corpus")
            found[key] = row
        missing = wanted.keys() - found.keys()
        if missing:
            raise ValueError(f"Manifest samples missing: {sorted(missing)[:5]}")
        return [found[x["id"]] for x in manifest["samples"]], manifest
    pools, seen = defaultdict(list), set()
    for row in records(data):
        key = row["id"]
        if key in seen:
            raise ValueError(f"Duplicate evaluation ID: {key}")
        seen.add(key)
        rank = int(digest([seed, key]), 16)
        pool = pools[cell(row)]
        item = (-rank, key, row)
        if len(pool) < per_cell:
            heapq.heappush(pool, item)
        elif item > pool[0]:
            heapq.heapreplace(pool, item)
    samples = [r for c in sorted(pools) for _, _, r in sorted(pools[c], reverse=True)]
    if not samples:
        raise ValueError("Evaluation corpus is empty")
    manifest = {"version": "turn-manifest/1", "seed": seed, "per_cell": per_cell,
                "corpus": str(data), "samples": [{"id": s["id"], "sha256": digest(s)} for s in samples]}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not replace a manifest another evaluator created concurrently.
    with path.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return samples, manifest


def score(sample, turn):
    want = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for c in sample["target"].get("tool_calls") or []]
    got = turn.calls
    fns = lambda calls: [a.get("function") for n, a in calls if n == "call"]
    return {"id": sample["id"], "cell": list(cell(sample)), "reply": not want,
            "exact": got == want, "right_tool": [n for n, _ in got] == [n for n, _ in want],
            "right_function": fns(got) == fns(want) if fns(want) else None,
            "expected": want, "actual": got, "text": turn.text}


def summary(rows):
    return {"n": len(rows), "exact": sum(r["exact"] for r in rows),
            "right_tool": sum(r["right_tool"] for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--per-cell", type=int, default=12)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", type=Path, help="reused sample IDs and content hashes")
    ap.add_argument("--out", type=Path, help="machine-readable scores (default: runs/eval-<timestamp>.json)")
    ap.add_argument("--model-label", default="unspecified", help="checkpoint/model identifier for this run")
    a = ap.parse_args()
    manifest_path = a.manifest or a.data.with_suffix(".eval-manifest.json")
    samples, manifest = select_samples(a.data, manifest_path, a.per_cell, a.seed)
    dec = NativeCallDecoder(a.server, timeout=300)
    results = []
    for s in samples:
        turn = dec.chat(s["messages"], s["tools"], temperature=0.0, seed=1, max_tokens=500)
        results.append(score(s, turn))
    cells = defaultdict(list)
    for r in results:
        cells[tuple(r["cell"])].append(r)
    for c, rows in sorted(cells.items()):
        st = summary(rows)
        print(f"{c[0]:<9} {c[1]:<22} n={st['n']:<3} exact={st['exact']/st['n']:.0%} "
              f"right tool={st['right_tool']/st['n']:.0%}")
    totals = {"all": summary(results), "actions": summary([r for r in results if not r["reply"]]),
              "replies": summary([r for r in results if r["reply"]])}
    for name, st in totals.items():
        print(f"{name}: {st['exact']}/{st['n']} exact; {st['right_tool']}/{st['n']} right tool")
    out = a.out or Path(f"runs/eval-{time.time_ns()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": a.model_label, "server": a.server, "manifest": str(manifest_path),
                              "manifest_sha256": digest(manifest), "summary": totals, "rows": results,
                              "usage": dec.usage}, indent=2) + "\n")
    print(f"manifest: {manifest_path}; results: {out}")


if __name__ == "__main__":
    main()
