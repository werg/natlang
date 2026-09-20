#!/usr/bin/env python3
"""Make a template-neutral teacher IR view of manually admitted leaf attempts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def project(audit_path: Path, trajectory_path: Path, review_path: Path,
            destination: Path) -> int:
    if destination.exists():
        raise FileExistsError(destination)
    review = json.loads(review_path.read_text())
    audit_bytes = audit_path.read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != review["source_sha256"]:
        raise ValueError("teacher audit differs from reviewed snapshot")
    audit = [json.loads(line) for line in audit_bytes.splitlines() if line.strip()]
    trajectories = [json.loads(line) for line in trajectory_path.read_text().splitlines()
                    if line.strip()]
    if len(audit) != len(trajectories):
        raise ValueError("audit and trajectory IR row counts differ")
    indices = [index for group in review["approved"].values() for index in group]
    if len(indices) != len(set(indices)):
        raise ValueError("review approves an audit row more than once")
    stage = destination.with_suffix(destination.suffix + ".building")
    if stage.exists():
        raise FileExistsError(stage)
    try:
        with stage.open("x") as target:
            for index in sorted(indices):
                source, row = audit[index], copy.deepcopy(trajectories[index])
                if row["task"]["reference_key"] != source["key"]:
                    raise ValueError(f"trajectory {index} does not match the audit key")
                if source["status"] != "done" or row["outcome"]["status"] != "done":
                    raise ValueError(f"reviewed row {index} did not complete")
                original = row["outcome"]["accepted"]
                if original != source["accepted"]:
                    raise ValueError(f"trajectory {index} changed the judge decision")
                if not original and str(index) not in review.get("notes", {}):
                    raise ValueError(f"judge override {index} lacks a review note")
                row["outcome"]["accepted"] = True
                row["outcome"]["admitted"] = True
                row["training_admission"] = {
                    "kind": "manual_leaf_review", "review": str(review_path),
                    "source_sha256": review["source_sha256"], "audit_index": index,
                    "original_judge_accepted": original}
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        stage.replace(destination)
    finally:
        stage.unlink(missing_ok=True)
    return len(indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(f"{project(args.audit, args.trajectories, args.review, args.destination)} "
          f"reviewed teacher trajectories -> {args.destination}")


if __name__ == "__main__":
    main()
