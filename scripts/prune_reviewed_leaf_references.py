#!/usr/bin/env python3
"""Quarantine reviewed bad leaf references while preserving an exact backup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def prune(bank: Path, review_path: Path, backup: Path) -> dict:
    if backup.exists():
        raise FileExistsError(f"backup already exists: {backup}")
    review = json.loads(review_path.read_text())
    source = bank.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    lines = source.splitlines(keepends=True)
    prefix_count = review["prefix_lines"]
    if len(lines) < prefix_count or hashlib.sha256(b"".join(lines[:prefix_count])).hexdigest() != review["prefix_sha256"]:
        raise ValueError("reference bank prefix differs from reviewed snapshot")
    rows = [json.loads(line) for line in lines]
    scoped = {index for index, row in enumerate(rows[:prefix_count])
              if row["function"] == "say" and row["args"]["action"]["code"].startswith("sell_")}
    approved = set(review["approved_indices"])
    if (len(scoped) != review["expected_scoped_rows"] or
            len(approved) != len(review["approved_indices"]) or
            not approved <= scoped):
        raise ValueError("reviewed sell-reference scope or approved indices differ")
    removed = scoped - approved
    filtered = b"".join(line for index, line in enumerate(lines) if index not in removed)
    if hashlib.sha256(bank.read_bytes()).hexdigest() != source_hash:
        raise ValueError("reference bank changed during review; retry")
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(source)
    stage = bank.with_name(bank.name + f".reviewed.{os.getpid()}.tmp")
    if stage.exists():
        raise FileExistsError(stage)
    try:
        stage.write_bytes(filtered)
        stage.chmod(bank.stat().st_mode)
        if hashlib.sha256(bank.read_bytes()).hexdigest() != source_hash:
            raise ValueError("reference bank changed before publication; retry")
        stage.replace(bank)
    finally:
        stage.unlink(missing_ok=True)
    return {"reviewed_sell_references": len(scoped), "retained": len(approved),
            "quarantined": len(removed), "bank_rows": len(rows) - len(removed),
            "backup": str(backup)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("backup", type=Path)
    args = parser.parse_args()
    print(json.dumps(prune(args.bank, args.review, args.backup), indent=2))


if __name__ == "__main__":
    main()
