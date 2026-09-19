#!/usr/bin/env python3
"""Migrate frozen proposal labels into records linked to executable base IR."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.corpus import file_digest
from program_ir import VERSION, digest, read_jsonl, validate


def review_record(old, base, source_revision):
    contrast = old["contrast_group"]
    case, turn = contrast.rsplit(":", 1)
    base_id = "scenario:" + case
    if base_id != base["id"]:
        raise ValueError(f"review/base mismatch: {old['id']}")
    args = json.loads(old["target"]["tool_calls"][0]["function"]["arguments"])
    if args["decision"] != old["expected"]:
        raise ValueError(f"review verdict mismatch: {old['id']}")
    sem = {"base_program_id": base_id, "base_program_digest": digest(base),
           "turn_index": int(turn), "proposal": old["proposal"],
           "check_call_index": old.get("check_call_index", 0),
           "verdict": old["expected"], "reason": args["reason"],
           "proposal_justified": old["proposal_justified"],
           "task_feasible": old["task_feasible"],
           "lesson_ids": old["lesson_ids"], "contrast_group": contrast,
           "prompt_variant": old.get("prompt_variant", "baseline"),
           "challenge_index": int(case.split(":")[2]) % 3}
    return validate({"version": VERSION, "id": "review:" + old["id"],
                     "kind": "proposal_review", "source": "synthetic-proposal-review",
                     "split": base["split"], "source_ids": [old["id"]],
                     "source_groups": base["source_groups"],
                     "source_revisions": [source_revision, digest(base)],
                     "license": base["license"], "gold_sources": ["constructed-contrast"],
                     "semantics": sem})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_reviews", type=Path)
    ap.add_argument("base_ir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-groups", type=int)
    args = ap.parse_args()
    bases = {r["id"]: r for r in read_jsonl(args.base_ir)}
    source_revision = file_digest(args.source_reviews)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    staging = args.out.with_suffix(args.out.suffix + ".building")
    counts = Counter()
    groups = set()
    contrast_labels = {}
    contrast_feasibility = {}
    with args.source_reviews.open() as src, staging.open("w") as out:
        for line in src:
            old = json.loads(line)
            base_id = "scenario:" + old["contrast_group"].rsplit(":", 1)[0]
            if args.max_groups is not None and base_id.split(":")[3] not in groups:
                if len(groups) >= args.max_groups:
                    continue
                groups.add(base_id.split(":")[3])
            record = review_record(old, bases[base_id], source_revision)
            contrast = record["semantics"]["contrast_group"]
            contrast_labels.setdefault(contrast, set()).add(record["semantics"]["verdict"])
            feasibility = record["semantics"]["task_feasible"]
            if contrast in contrast_feasibility and contrast_feasibility[contrast] != feasibility:
                raise ValueError(f"inconsistent task feasibility: {contrast}")
            contrast_feasibility[contrast] = feasibility
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[record["semantics"]["verdict"]] += 1
    if any("approve" not in labels or len(labels) < 2 for labels in contrast_labels.values()):
        raise ValueError("proposal contrast lacks an approved and a challenged candidate")
    staging.replace(args.out)
    manifest = {"version": VERSION, "base_ir_sha256": file_digest(args.base_ir),
                "source_reviews_sha256": source_revision, "reviews": sum(counts.values()),
                "verdicts": dict(counts), "ir_sha256": file_digest(args.out)}
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
