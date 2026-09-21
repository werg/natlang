#!/usr/bin/env python3
"""Deduplicate and behavior-balance rendered SFT rows without splitting programs.

The source files remain immutable.  Conflicting targets for the same exact
prompt are rejected because they teach an ambiguous next action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def difficulty(row: dict) -> tuple[int, tuple[str, ...]]:
    """Estimate execution difficulty from the target and available state.

    This deliberately uses only stable, renderer-independent metadata and the
    canonical native target.  It is a curriculum signal, not a correctness
    judgment: higher-scoring turns are the state transitions most likely to
    exercise orchestration rather than surface formatting.
    """
    skill = row.get("skill") or "unknown"
    parts = set(skill.split("+"))
    target = row.get("native_target") or ""
    context_items = int(row.get("context_items") or 0)
    features = []
    score = 0
    if parts & {"report_error", "report_blocker", "edit"}:
        score += 100; features.append("failure_or_repair")
    if any(marker in target for marker in ("over=", "init=", "until=")):
        score += 80; features.append("iteration")
    if "call" in parts and "inputs=" in target and "'let/" in target:
        score += 60; features.append("dependent_call")
    if "+" in skill:
        score += 45; features.append("multi_action")
    if "run_code" in parts:
        score += 40; features.append("algorithmic_glue")
    if "call" in parts and "to='let/" in target:
        score += 25; features.append("local_result")
    if context_items >= 8 and parts & {"call", "write", "mark_done", "read"}:
        score += min(30, 10 + context_items); features.append("later_state")
    return score, tuple(features)


def rank(row: dict) -> tuple:
    parts = set((row.get("skill") or "unknown").split("+"))
    priority = (0 if parts & {"report_error", "report_blocker", "edit"} else
                1 if "run_code" in parts else 2 if parts & {"read", "checkpoint"} else
                3 if "call" in parts else 4 if "mark_done" in parts else
                5 if "write" in parts else 6 if "reply" in parts else 4)
    stable = hashlib.sha256(row["id"].encode()).hexdigest()
    score, _ = difficulty(row)
    return -score, priority, stable


def select(rows: list[dict], max_per_program: int, max_writes: int, max_terminals: int):
    by_program = defaultdict(list)
    for row in rows:
        by_program[row.get("program_id") or row["id"]].append(row)
    kept = []
    dropped = Counter()
    for program in sorted(by_program):
        bucket = by_program[program]
        writes = sorted((r for r in bucket if "write" in (r.get("skill") or "").split("+")), key=rank)
        terminals = sorted((r for r in bucket if r.get("skill") == "reply"), key=rank)
        reads = sorted((r for r in bucket if "read" in (r.get("skill") or "").split("+")), key=rank)
        rare = [r for r in bucket if
                set((r.get("skill") or "").split("+")) & {"report_error", "report_blocker", "edit"}
                or (r.get("family") or "").startswith("failure_")]
        # Preserve the minimum contrast and state-access examples even when a
        # program has more hard transitions than its ordinary cap.  In that
        # uncommon case essential coverage may exceed max_per_program.
        mandatory = ({r["id"] for r in writes[:max_writes]} |
                     {r["id"] for r in terminals[:max_terminals]} |
                     {r["id"] for r in reads[:1]} |
                     {r["id"] for r in rare})
        allowed = mandatory
        candidates = [r for r in bucket if ("write" not in (r.get("skill") or "").split("+")
                                             and r.get("skill") != "reply") or r["id"] in allowed]
        required = [r for r in bucket if r["id"] in mandatory]
        optional = [r for r in sorted(candidates, key=rank) if r["id"] not in mandatory]
        chosen = required + optional[:max(0, max_per_program - len(required))]
        chosen_ids = {r["id"] for r in chosen}
        kept.extend(r for r in bucket if r["id"] in chosen_ids)  # retain source trajectory order
        for row in bucket:
            if row["id"] not in chosen_ids:
                dropped[row.get("skill") or "unknown"] += 1
    return kept, dropped


def distribution(rows):
    families = Counter(r.get("family") or "unknown" for r in rows)
    completion_lengths = sorted(len(r["completion"]) for r in rows)
    def percentile(q):
        if not completion_lengths:
            return 0
        return completion_lengths[round(q * (len(completion_lengths) - 1))]
    algorithmic = sum((r.get("family") or "").startswith("algo_") or
                      ":array_kernel:" in r["id"] or ":staged_ranking:" in r["id"] or
                      ":algorithm_pipeline:" in r["id"] for r in rows)
    difficulty_features = Counter(feature for row in rows for feature in difficulty(row)[1])
    scores = sorted(difficulty(row)[0] for row in rows)
    return {"families": dict(families), "algorithmic_rows": algorithmic,
            "algorithmic_fraction": algorithmic / len(rows) if rows else 0,
            "reasoning_rows": sum("<think>" in r["completion"] for r in rows),
            "difficulty_features": dict(difficulty_features),
            "difficulty_score": {"p50": scores[len(scores) // 2] if scores else 0,
                                 "p90": scores[min(len(scores) - 1, len(scores) * 9 // 10)] if scores else 0,
                                 "max": scores[-1] if scores else 0},
            "completion_chars": {"p10": percentile(.1), "p50": percentile(.5),
                                 "p90": percentile(.9), "max": percentile(1)}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dst", type=Path)
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--max-per-program", type=int, default=32)
    ap.add_argument("--max-writes-per-program", type=int, default=4)
    ap.add_argument("--max-terminals-per-program", type=int, default=1)
    args = ap.parse_args()
    if min(args.max_per_program, args.max_writes_per_program, args.max_terminals_per_program) < 1:
        ap.error("selection caps must be positive")
    if args.dst.exists() or args.dst.with_suffix(args.dst.suffix + ".manifest.json").exists():
        ap.error("destination already exists")

    rows, source_info, ids = [], [], set()
    renderer = None
    exact = {}
    duplicates = reasoning_variants = 0
    before = Counter()
    for path in args.sources:
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        identity = manifest.get("renderer")
        if identity:
            if renderer is not None and identity != renderer:
                raise ValueError(f"renderer mismatch: {path}")
            renderer = identity
        count = 0
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row["id"] in ids:
                    raise ValueError(f"duplicate id {row['id']} in {path}:{number}")
                ids.add(row["id"])
                before[row.get("skill") or "unknown"] += 1
                key = row["prompt"]
                prior = exact.get(key)
                if prior is not None:
                    if prior["completion"] != row["completion"]:
                        if prior.get("native_target") != row.get("native_target"):
                            raise ValueError(f"conflicting actions for an identical prompt: {prior['id']} and {row['id']}")
                        # The action choice is identical; only free reasoning
                        # wording differs. Keep one deterministic exemplar so
                        # repeated teacher samples do not overweight a state.
                        reasoning_variants += 1
                        continue
                    duplicates += 1
                    continue
                exact[key] = row
                rows.append(row)
                count += 1
        source_info.append({"path": str(path), "sha256": digest(path), "unique_rows": count})

    selected, dropped = select(rows, args.max_per_program, args.max_writes_per_program,
                               args.max_terminals_per_program)
    after = Counter(r.get("skill") or "unknown" for r in selected)
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    staged = args.dst.with_suffix(args.dst.suffix + ".building")
    h = hashlib.sha256()
    with staged.open("x") as out:
        for row in selected:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            out.write(line)
            h.update(line.encode())
    staged.replace(args.dst)
    result = {"schema": "natlang.sft_selection/1", "sources": source_info,
              "renderer": renderer, "source_rows": sum(before.values()),
              "exact_duplicates_removed": duplicates, "unique_prompts": len(rows),
              "reasoning_variants_removed": reasoning_variants,
              "selected_rows": len(selected), "programs": len({r.get('program_id') or r['id'] for r in selected}),
              "skill_counts_before": dict(before), "skill_counts_after": dict(after),
              "distribution_before": distribution(rows),
              "distribution_after": distribution(selected),
              "dropped_by_cap": dict(dropped),
              "caps": {"per_program": args.max_per_program,
                       "writes_per_program": args.max_writes_per_program,
                       "terminals_per_program": args.max_terminals_per_program},
              "sha256": h.hexdigest()}
    args.dst.with_suffix(args.dst.suffix + ".manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
