#!/usr/bin/env python3
"""Verify compact classifier batch provenance and summarize accepted labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit(path: Path) -> dict:
    batch_path = path.with_suffix(path.suffix + ".batches.jsonl")
    batches = {}
    if batch_path.exists():
        for line in batch_path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                batches[row["batch_id"]] = row
    latest = {}
    for number, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        rid = row["id"]
        jev = row["jev"]
        batch_id = jev.get("batch_id")
        if batch_id is not None:
            batch = batches.get(batch_id)
            if batch is None:
                raise ValueError(f"line {number}: missing batch {batch_id}")
            index = jev["response_index"]
            if batch["ids"][index] != rid or batch["request"]["inputs"][index] != row["state"]:
                raise ValueError(f"line {number}: batch input mismatch")
            response = batch["response"]
            results = response.get("results") if isinstance(response, dict) else None
            result = results[index] if isinstance(results, list) and index < len(results) else None
            if result != jev["result"]:
                raise ValueError(f"line {number}: batch result mismatch")
        latest[rid] = row
    status = Counter()
    models = Counter()
    splits = Counter()
    labels = defaultdict(Counter)
    groups = set()
    for row in latest.values():
        jev = row["jev"]
        status[jev["status"]] += 1
        models[str(jev.get("model"))] += 1
        splits[row.get("split", "unknown")] += 1
        groups.add(row.get("group_id"))
        if jev["status"] == "accepted":
            if row.get("gold_source") != "jev" or row.get("gold") != jev["label"]:
                raise ValueError(f"{row['id']}: accepted result does not match gold")
            labels[row["instruction"]][row["gold"]] += 1
    return {"unique_items": len(latest), "batches": len(batches),
            "status": dict(status), "models": dict(models),
            "splits": dict(splits), "groups": len(groups),
            "labels_by_question": {question: dict(counts) for question, counts in labels.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = audit(args.input)
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
