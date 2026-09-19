#!/usr/bin/env python3
"""Check IR uniqueness, split isolation, and program mix across JSONL files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from program_ir import read_jsonl


def audit(paths):
    ids = set()
    group_splits = {}
    counts = Counter()
    operators = Counter()
    for path in paths:
        for record in read_jsonl(path):
            key = record["id"]
            if key in ids:
                raise ValueError(f"duplicate program: {key}")
            ids.add(key)
            for group in record["source_groups"]:
                group_key = (record["source"], group)
                prior = group_splits.setdefault(group_key, record["split"])
                if prior != record["split"]:
                    raise ValueError(f"source group crosses splits: {group_key}")
            counts[(record["source"], record["split"], record["kind"])] += 1
            sem = record["semantics"]
            if record["kind"] == "scene_program":
                operators.update(node["function"] for node in sem["nodes"])
            elif record["kind"] == "numeric_program":
                operators.update(step["op"] for step in sem["steps"])
            elif record["kind"] == "state_sequence":
                operators["state_transition"] += len(sem["steps"])
            elif record["kind"] == "lambda_source":
                operators[sem["operation"]] += 1
            elif record["kind"] == "proposal_review":
                operators["proposal:" + sem["verdict"]] += 1
            elif record["kind"] in {"lambda_graph", "lambda_scenario"}:
                def count_ops(steps):
                    for step in steps:
                        operators[step["op"]] += 1
                        if step["op"] == "branch":
                            count_ops(step["then"])
                            count_ops(step["else"])
                count_ops(sem["operations"])
                for steps in sem.get("functions", {}).values():
                    count_ops(steps)
                if sem.get("nested"):
                    operators["nested:" + sem["nested"]["kind"]] += 1
            else:
                operators["typed_decision"] += len(sem["decisions"])
    return {"programs": len(ids), "groups": len(group_splits),
            "mix": [{"source": s, "split": p, "kind": k, "programs": n}
                    for (s, p, k), n in sorted(counts.items())],
            "operators": dict(sorted(operators.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.paths)
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
