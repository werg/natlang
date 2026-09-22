#!/usr/bin/env python3
"""Select reviewed teacher IR against a reference bank and replay it as training turns.

The source IR remains untouched. Review files may contain original collector
decisions, later rejudgments, or manual decisions. A manual decision without a
source location inherits the location of an earlier rejudgment of the same key.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.materialize_teacher_trajectory_ir import materialize
from scripts.program_ir import digest as program_digest
from scripts.project_teacher_trajectory_ir import project
from scripts.teacher_trajectory_ir import digest, leaf_programs


def location(source, line):
    return (str(Path(source).resolve()), int(line))


def file_sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_reviews(paths):
    decisions = {}
    by_key = {}
    for path in paths:
        for number, raw in enumerate(path.open(), 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            source = item.get("source_audit") or item.get("source")
            line = item.get("line")
            if source is None:
                if item["key"] not in by_key:
                    raise ValueError(f"{path}:{number}: unlocated review needs an earlier located review")
                loc = by_key[item["key"]]
            else:
                loc = location(source, line)
                by_key[item["key"]] = loc
            approved = (item["manual_keep"] if "manual_keep" in item else
                        bool(item.get("accepted") and item.get("admitted")))
            kind = "manual" if "manual_keep" in item else "rejudged"
            if decisions.get(loc, {}).get("kind") == "manual" and kind != "manual":
                continue
            decisions[loc] = {"key": item["key"], "approved": approved,
                              "kind": kind, "file": str(path), "line": number,
                              "reason": item.get("reason")}
    return decisions


def require_surface(row, target_surface, path):
    actual = row.get("provenance", {}).get("tool_schema")
    if target_surface is not None and actual != target_surface:
        raise ValueError(f"{path}: trajectory surface {actual!r} does not match target {target_surface!r}; project it explicitly")


def select(paths, bank, reviews, target_surface=None):
    candidates = defaultdict(list)
    stats = Counter()
    for path in paths:
        for raw in path.open():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row["task"]["kind"] != "generative_leaf":
                continue
            require_surface(row, target_surface, path)
            key = row["task"]["reference_key"]
            loc = location(row["provenance"]["audit_file"], row["provenance"]["audit_line"])
            review = reviews.get(loc)
            if review and review["key"] != key:
                raise ValueError(f"review key does not match trajectory at {loc}")
            if (key not in bank or row["outcome"]["status"] != "done" or
                    row["outcome"]["value"] != bank[key]):
                stats["nonmatching_or_failed"] += 1
                continue
            if review and not review["approved"]:
                stats["review_rejected"] += 1
                continue
            rank = (3 if review and review["kind"] == "manual" else
                    2 if review else 1 if row["outcome"].get("accepted") and
                    row["outcome"].get("admitted") else 0)
            if rank == 0:
                stats["unreviewed"] += 1
                continue
            candidates[key].append((rank, row, review))
    selected = []
    for key, choices in candidates.items():
        _, row, review = max(choices, key=lambda item: (item[0], item[1]["id"]))
        # Admission is a disposable training projection, with its decision source.
        row = json.loads(json.dumps(row))
        row["training_admission"] = (review or {"kind": "collector", "approved": True})
        row["outcome"]["accepted"] = True
        row["outcome"]["admitted"] = True
        selected.append(row)
    return sorted(selected, key=lambda row: row["task"]["reference_key"]), stats


def iter_programs(paths, stats, target_surface=None):
    """Take independently admitted whole-program attempts from any frozen IR batch."""
    chosen = set()
    revisions = {}
    for path in paths:
        for raw in path.open():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row["task"]["kind"] != "whole_program":
                raise ValueError(f"{path}: expected whole_program trajectory")
            require_surface(row, target_surface, path)
            if not row["outcome"].get("accepted") or not (
                    row["outcome"].get("admission") or {}).get("admitted"):
                stats["not_admitted"] += 1
                continue
            if program_digest(row["task"]["program_ir"]) != row["provenance"]["program_ir_sha256"]:
                raise ValueError(f"{path}: teacher program IR digest mismatch")
            identity = (row["task"]["program_ir"]["id"],
                        row["provenance"]["program_ir_sha256"])
            old = revisions.setdefault(identity[0], identity[1])
            if old != identity[1]:
                raise ValueError(f"conflicting frozen revisions for {identity[0]}")
            if identity in chosen:
                stats["duplicate_program"] += 1
                continue
            row["training_admission"] = {"kind": "program-trace", "approved": True,
                                          "source_ir": str(path)}
            chosen.add(identity)
            yield row


def select_programs(paths, target_surface=None):
    stats = Counter()
    return list(iter_programs(paths, stats, target_surface)), stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--ir", type=Path, action="append", default=[],
                    help="generative-leaf teacher trajectory IR; may repeat")
    ap.add_argument("--whole-ir", type=Path, action="append", default=[],
                    help="admitted whole-program teacher trajectory IR; may repeat")
    ap.add_argument("--review", type=Path, action="append", default=[])
    ap.add_argument("--reference-bank", type=Path, default=Path("data/leaf_references.jsonl"))
    ap.add_argument("--program-ir", type=Path,
                    help="frozen programs used to recover leaf definitions absent from older audits")
    ap.add_argument("--system-file", type=Path, default=Path("natlang/prompts/tools_small.md"))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--target-surface", default="scope-eval-v1",
                    help="reject trajectories from any other model-facing surface")
    args = ap.parse_args()
    if not args.ir and not args.whole_ir:
        ap.error("supply --ir or --whole-ir")
    if args.out.exists():
        ap.error("output already exists")
    bank = ({item["key"]: item["value"] for item in
             map(json.loads, args.reference_bank.read_text().splitlines())} if args.ir else {})
    selected, stats = select(args.ir, bank, read_reviews(args.review), args.target_surface)
    program_stats = Counter()
    programs = iter_programs(args.whole_ir, program_stats, args.target_surface)
    missing_programs = [row for row in selected if not row["task"].get("leaf_program")]
    if missing_programs:
        if not args.program_ir:
            raise ValueError(f"{len(missing_programs)} trajectories lack a leaf program; supply --program-ir")
        frozen_leaves = leaf_programs(args.program_ir)
        for row in missing_programs:
            key = row["task"]["reference_key"]
            if key not in frozen_leaves:
                raise ValueError(f"frozen program IR has no definition for {key}")
            row["task"]["leaf_program"] = frozen_leaves[key]
            row["training_admission"]["recovered_program_ir"] = str(args.program_ir)
    missing = sorted(bank.keys() - {row["task"]["reference_key"] for row in selected})
    if missing and not args.allow_incomplete:
        raise ValueError(f"{len(missing)} bank keys have no reviewed matching trajectory; "
                         "use --allow-incomplete to export the available subset")
    system_prompt = args.system_file.read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = args.out.with_suffix(args.out.suffix + f".building.{os.getpid()}")
    turns = 0
    program_count = 0
    try:
        with stage.open("x") as target:
            for row in chain(selected, programs):
                program_count += row["task"]["kind"] == "whole_program"
                view = project(row, tool_map={"end_turn": None}, accepted_only=True,
                               empty_success_reply=True)
                try:
                    samples = materialize(view, system_prompt=system_prompt)
                except Exception as exc:
                    key = row["task"].get("reference_key") or row["task"]["program_ir"]["id"]
                    raise ValueError(f"teacher replay failed for {row['id']} ({key}): {exc}") from exc
                for sample in samples:
                    target.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    turns += 1
        stage.replace(args.out)
    finally:
        stage.unlink(missing_ok=True)
    manifest = {"version": "natlang.teacher_training/1",
                "source_ir": {str(x): file_sha256(x) for x in args.ir},
                "whole_program_ir": {str(x): file_sha256(x) for x in args.whole_ir},
                "reviews": {str(x): file_sha256(x) for x in args.review},
                "reference_bank": str(args.reference_bank),
                "reference_bank_sha256": file_sha256(args.reference_bank) if args.ir else None,
                "program_ir": {str(args.program_ir): file_sha256(args.program_ir)} if args.program_ir else None,
                "target_surface": args.target_surface,
                "trajectories": len(selected) + program_count,
                "leaf_trajectories": len(selected), "whole_program_trajectories": program_count,
                "turns": turns, "missing_keys": missing, "selection_stats": dict(stats),
                "program_selection_stats": dict(program_stats)}
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"{len(selected)} reviewed leaves, {program_count} admitted programs, "
          f"{turns} turns; {len(missing)} bank keys unavailable")


if __name__ == "__main__":
    main()
