#!/usr/bin/env python3
"""Admit explicitly reviewed teacher leaf outputs to the reference bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.gen.codebases import ref_key


def admit(audit_path: Path, review_path: Path, references: Path) -> dict:
    review = json.loads(review_path.read_text())
    audit_bytes = audit_path.read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest() != review["source_sha256"]:
        raise ValueError("teacher audit differs from reviewed snapshot")
    rows = [json.loads(line) for line in audit_bytes.splitlines() if line.strip()]
    approved = review["approved"]
    indices = [index for group in approved.values() for index in group]
    if len(indices) != len(set(indices)):
        raise ValueError("review approves an audit row more than once")
    selected = []
    for group, group_indices in approved.items():
        for index in group_indices:
            row = rows[index]
            if row["function"] != "say" or row["args"]["action"]["code"] != group:
                raise ValueError(f"review group does not match audit row {index}")
            if row["status"] != "done" or not isinstance(row["value"], str):
                raise ValueError(f"reviewed row {index} has no completed text result")
            if not row["accepted"] and str(index) not in review.get("notes", {}):
                raise ValueError(f"judge-rejected row {index} needs an override note")
            if row["key"] != ref_key(row["function"], row["args"]):
                raise ValueError(f"reviewed row {index} has a mismatched reference key")
            selected.append({"key": row["key"], "function": row["function"],
                             "args": row["args"], "value": row["value"]})
    if len({row["key"] for row in selected}) != len(selected):
        raise ValueError("reviewed rows have duplicate reference keys")

    existing_bytes = references.read_bytes()
    existing = {row["key"]: row for row in
                (json.loads(line) for line in existing_bytes.splitlines() if line.strip())}
    additions = []
    for row in selected:
        prior = existing.get(row["key"])
        if prior is not None:
            if prior != row:
                raise ValueError(f"reference key already has a different value: {row['key']}")
            continue
        additions.append(row)
    if additions:
        stage = references.with_name(references.name + f".reviewed.{os.getpid()}.tmp")
        if stage.exists():
            raise FileExistsError(stage)
        payload = existing_bytes + b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in additions)
        try:
            stage.write_bytes(payload)
            stage.chmod(references.stat().st_mode)
            stage.replace(references)
        finally:
            stage.unlink(missing_ok=True)
    return {"reviewed": len(selected), "appended": len(additions),
            "already_present": len(selected) - len(additions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("references", type=Path)
    args = parser.parse_args()
    print(json.dumps(admit(args.audit, args.review, args.references), indent=2))


if __name__ == "__main__":
    main()
