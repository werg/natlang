"""Expand reviewed semantic-merge scenarios without treating agreement as a quality oracle.

Every transport variant retains its group so a train/test split can keep related
histories together. The expected outcome and rubric describe semantic admission;
teacher trajectories still have to be collected and reviewed separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEEDS = ROOT / "codebases/semantic_merge/scenarios/seed_cases.jsonl"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def expand(seed: dict, root_seed: int = 43) -> list[dict]:
    changes = seed["changes"]
    updates = [{"id": change["id"], "parents": change.get("parents", []),
                "base_revision": seed["base"]["revision"], "author": change["author"],
                "text": change["text"]} for change in changes]
    variants = {
        "canonical": updates,
        "reversed": list(reversed(updates)),
        "redelivery-first": [updates[0], *updates],
        "redelivery-last": [*updates, updates[-1]],
    }
    rows = []
    for delivery, presented in variants.items():
        inputs = {"base": seed["base"], "updates": presented, "policy": seed["policy"]}
        rows.append({"schema": "semantic-merge-case/v1", "case_id": f"{seed['group']}:{delivery}",
                     "group": seed["group"], "program": seed["program"], "delivery": delivery,
                     "inputs": inputs, "root_seed": root_seed, "expected": seed["expected"],
                     "rubric": seed["rubric"],
                     "input_sha256": hashlib.sha256(canonical(inputs).encode()).hexdigest()})
    return rows


def generate(seed_path: Path = SEEDS, root_seed: int = 43) -> list[dict]:
    seeds = [json.loads(line) for line in seed_path.read_text().splitlines() if line.strip()]
    groups = [seed["group"] for seed in seeds]
    if len(groups) != len(set(groups)):
        raise ValueError("semantic merge scenario groups must be unique")
    rows = []
    for seed in seeds:
        if len(seed["changes"]) < 2 or seed["expected"] not in ("merged", "unresolved"):
            raise ValueError(f"invalid scenario {seed['group']}")
        ids = [change["id"] for change in seed["changes"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate source ID in scenario {seed['group']}")
        rows.extend(expand(seed, root_seed))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, default=SEEDS)
    parser.add_argument("--root-seed", type=int, default=43)
    parser.add_argument("--out", type=Path, default=ROOT / "codebases/semantic_merge/scenarios/cases.jsonl")
    args = parser.parse_args()
    rows = generate(args.seeds, args.root_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"wrote {len(rows)} cases from {len(rows) // 4} groups to {args.out}")


if __name__ == "__main__":
    main()
