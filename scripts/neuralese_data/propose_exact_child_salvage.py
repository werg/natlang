#!/usr/bin/env python3
"""Bind exact child decisions from failed directory-reducer roots for semantic review.

This is discovery only. The proposal grants no training or trace admission;
source quality and each selected child's meaning still need review.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path):
    return json.loads(path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cases", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate_bytes = args.candidate_manifest.read_bytes()
    candidates = json.loads(candidate_bytes)
    source_bytes = args.source_cases.read_bytes()
    source_sha = digest(source_bytes)
    if candidates.get("schema") != "natlang.provider-child-action-candidates/2" or (
        candidates.get("source_cases", {}).get("sha256") != source_sha
    ):
        raise ValueError("candidate discovery does not bind these source cases")
    sources = {}
    for line in source_bytes.splitlines():
        row = json.loads(line)
        if row["id"] in sources:
            raise ValueError(f"duplicate source ID: {row['id']}")
        sources[row["id"]] = row

    results = {}
    for path in sorted(args.results_dir.rglob("results.jsonl")):
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            if row["id"] in results:
                raise ValueError(f"duplicate trajectory ID: {row['id']}")
            results[row["id"]] = (row, path, digest(line))

    eligible = []
    skipped = Counter()
    for candidate in candidates["candidates"]:
        source = sources.get(candidate["item_id"])
        result = results.get(candidate["trajectory_id"])
        if source is None or result is None:
            raise ValueError(f"missing source or result for {candidate['candidate_id']}")
        row, path, row_sha = result
        if (candidate["source_role"] != "train" or source.get("role") != "train" or
                source["group"] != candidate["source_group"] or
                source["answer"] != candidate["expected_value"] or
                candidate["observed_value"] != candidate["expected_value"] or
                candidate["source_reference_review"].get("accepted") is not True):
            raise ValueError(f"candidate lacks exact source binding: {candidate['candidate_id']}")
        if row["outcome"].get("accepted") is not False:
            skipped["accepted_root"] += 1
            continue
        eligible.append({
            "candidate_id": candidate["candidate_id"],
            "program_id": candidate["program_id"],
            "trajectory_id": candidate["trajectory_id"],
            "trajectory_sha256": candidate["trajectory_sha256"],
            "raw_result_path": str(path),
            "raw_result_line_sha256": row_sha,
            "item_id": candidate["item_id"],
            "item_sha256": candidate["item_sha256"],
            "source_group": candidate["source_group"],
            "state": source["state"],
            "question": source["question"],
            "options": source["options"],
            "expected_value": candidate["expected_value"],
            "observed_value": candidate["observed_value"],
            "parent_observed_value": candidate["parent_observed_value"],
            "invocation_id": candidate["invocation_id"],
            "decision_index": candidate["decision_index"],
            "target_sha256": candidate["target_sha256"],
            "provider_rendered_request_sha256": candidate["provider_rendered_request_sha256"],
            "visible_fields": candidate["visible_fields"],
        })

    selected = {}
    for row in sorted(eligible, key=lambda item: (item["item_id"], item["decision_index"], item["candidate_id"])):
        if row["item_id"] in selected:
            skipped["duplicate_correct_child"] += 1
        else:
            selected[row["item_id"]] = row
    proposal = {
        "schema": "natlang.exact-child-salvage-proposal/1",
        "training_admission": False,
        "trace_admission": False,
        "source_cases": {"path": str(args.source_cases), "sha256": source_sha},
        "candidate_manifest": {"path": str(args.candidate_manifest), "sha256": digest(candidate_bytes)},
        "selection": "failed parent; exact source label; exact provider child; one child per source item",
        "semantic_review": "pending for every selected item",
        "counts": {"candidates": len(candidates["candidates"]), "failed_root_exact_candidates": len(eligible),
                   "unique_items_for_review": len(selected), **skipped},
        "rows": [selected[item_id] for item_id in sorted(selected)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "sha256": digest(args.out.read_bytes()),
                      "counts": proposal["counts"]}))


if __name__ == "__main__":
    main()
