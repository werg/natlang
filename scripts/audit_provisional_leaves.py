#!/usr/bin/env python3
"""Audit generated-text leaves in a frozen synthetic IR.

This is an offline audit.  It does not run the generator or ask a teacher model
for outputs.  A case is identified with the same function-and-arguments key
used by ``natlang.gen.codebases``.  Reference rows are read directly so that
duplicate and conflicting rows are visible instead of being silently hidden by
``load_references``' last-row-wins behavior.

Example::

    python scripts/audit_provisional_leaves.py \
      --ir data/external_pilot/synthetic-all-current.ir.jsonl \
      --references data/leaf_references.jsonl \
      --out data/external_pilot/provisional-leaf-audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from natlang.gen.codebases import ref_key  # noqa: E402

DEFAULT_IR = ROOT / "data/external_pilot/synthetic-all-current.ir.jsonl"
DEFAULT_REFERENCES = ROOT / "data/leaf_references.jsonl"
DEFAULT_FAMILIES = ("cb_shopkeeper", "cb_webserver")


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                record = json.loads(line)
                record["_line_number"] = line_number
                yield record


def _load_references(path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Return rows grouped by declared key and malformed/key-mismatch rows."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid: list[dict[str, Any]] = []
    for row in _records(path):
        fn, args, key = row.get("function"), row.get("args"), row.get("key")
        expected = ref_key(fn, args) if isinstance(fn, str) and isinstance(args, dict) else None
        if not isinstance(key, str) or expected != key:
            invalid.append({"line_number": row["_line_number"], "key": key,
                            "function": fn, "expected_key": expected})
        if isinstance(key, str):
            grouped[key].append(row)
    return dict(grouped), invalid


def _cases(path: Path, families: set[str]) -> tuple[dict[str, dict[str, Any]], set[str], int]:
    """Collect template leaf cases, grouped by their exact reference key."""
    cases: dict[str, dict[str, Any]] = {}
    provisional_ids: set[str] = set()
    program_count = 0
    for record in _records(path):
        if record.get("family") not in families:
            continue
        oracles = record.get("semantics", {}).get("leaf_oracles", {})
        if not isinstance(oracles, dict):
            continue
        record_had_template = False
        for function, oracle in oracles.items():
            if not isinstance(oracle, dict):
                continue
            for case in oracle.get("cases", []):
                if not isinstance(case, dict) or not case.get("template"):
                    continue
                args = case.get("input")
                if not isinstance(args, dict):
                    continue
                key = ref_key(function, args)
                item = cases.setdefault(key, {"key": key, "function": function, "args": args,
                                               "occurrences": 0, "program_ids": []})
                item["occurrences"] += 1
                item["program_ids"].append(record.get("id"))
                record_had_template = True
        if record_had_template:
            program_count += 1
            provisional_ids.add(record.get("id"))
    return cases, provisional_ids, program_count


def audit(ir: Path = DEFAULT_IR, references: Path = DEFAULT_REFERENCES,
          families: Iterable[str] = DEFAULT_FAMILIES) -> dict[str, Any]:
    families = set(families)
    cases, provisional_ids, provisional_program_count = _cases(ir, families)
    grouped, invalid = _load_references(references)
    covered = 0
    missing = 0
    affected_ids: set[str] = set()
    for key, item in cases.items():
        rows = grouped.get(key, [])
        item["covered"] = bool(rows)
        item["reference_rows"] = [r["_line_number"] for r in rows]
        item["reference_values"] = [r.get("value") for r in rows]
        if rows:
            covered += 1
        else:
            missing += 1
            affected_ids.update(item["program_ids"])
        item["program_ids"] = sorted(set(item["program_ids"]))

    duplicate_keys = []
    conflicting_keys = []
    for key, rows in grouped.items():
        if len(rows) > 1:
            duplicate_keys.append({"key": key, "lines": [r["_line_number"] for r in rows],
                                   "values": [r.get("value") for r in rows]})
            if len({json.dumps(r.get("value"), sort_keys=True, ensure_ascii=False) for r in rows}) > 1:
                conflicting_keys.append(duplicate_keys[-1])

    return {
        "ir": str(ir), "references_file": str(references), "families": sorted(families),
        "summary": {"provisional_programs": provisional_program_count,
                     "exact_case_occurrences": sum(x["occurrences"] for x in cases.values()),
                     "distinct_keys": len(cases), "covered_keys": covered, "missing_keys": missing,
                     "affected_programs": len(affected_ids),
                     "fully_covered_programs": provisional_program_count - len(affected_ids)},
        "keys": sorted(cases.values(), key=lambda x: x["key"]),
        "affected_program_ids": sorted(affected_ids),
        "references": {"rows": sum(len(v) for v in grouped.values()),
                       "unique_keys": len(grouped), "duplicate_keys": duplicate_keys,
                       "conflicting_keys": conflicting_keys, "invalid_key_rows": invalid},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.ir, args.references, args.families)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(json.dumps({"out": str(args.out), "summary": report["summary"],
                          "reference_issues": {key: len(report["references"][key])
                                               for key in ("duplicate_keys", "conflicting_keys", "invalid_key_rows")}},
                         indent=2))
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
