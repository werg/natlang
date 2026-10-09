#!/usr/bin/env python3
"""Build and verify source-row identities for reviewed Luna dispatch plans.

The inventory binds each dispatcher index and seed to the exact JSONL row,
program identity, split, root code, and embedded program-bound skill text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_hex as sha256  # noqa: E402


def read_rows(source: Path) -> tuple[bytes, list[tuple[int, bytes, dict[str, Any]]]]:
    source_bytes = source.read_bytes()
    rows = []
    for line_no, line in enumerate(source_bytes.splitlines(keepends=True)):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"source row {line_no} must be an object")
        rows.append((line_no, line, value))
    return source_bytes, rows


def inventory(source: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    source = source.resolve(strict=True)
    source_bytes, rows = read_rows(source)
    indexed = {line_no: (line, row) for line_no, line, row in rows}
    entries = []
    seen = set()
    for case in cases:
        index, seed = case.get("index"), case.get("seed")
        if type(index) is not int or type(seed) is not int or seed < 0:
            raise ValueError("each case must have a nonnegative integer index and seed")
        if index in seen or index not in indexed:
            raise ValueError(f"duplicate or missing source row index {index}")
        seen.add(index)
        row_bytes, row = indexed[index]
        source_groups = row.get("source_groups")
        if not isinstance(source_groups, list) or len(source_groups) != 1 or not isinstance(source_groups[0], str):
            raise ValueError(f"source row {index} must identify exactly one source group")
        program = row.get("curriculum", {}).get("reference", {}).get("root")
        if not isinstance(program, list) or not program or not isinstance(program[0], list) or len(program[0]) < 2:
            raise ValueError(f"source row {index} has no canonical root program")
        root_code = program[0][1].get("code")
        if not isinstance(root_code, str):
            raise ValueError(f"source row {index} root program has no code")
        files = row.get("semantics", {}).get("files", {})
        skills = []
        for file_path, body in sorted(files.items()):
            if file_path.endswith("/SKILL.md"):
                if not isinstance(body, str):
                    raise ValueError(f"skill body is not text: {file_path}")
                body_bytes = body.encode("utf-8")
                skills.append({
                    "path": file_path,
                    "sha256": sha256(body_bytes),
                    "bytes": len(body_bytes),
                    "body": body,
                })
        entries.append({
            "index": index,
            "seed": seed,
            "program_id": row.get("id"),
            "source_ids": row.get("source_ids"),
            "source_groups": source_groups,
            "split": row.get("split"),
            "source_revisions": row.get("source_revisions"),
            "row_sha256_including_lf": sha256(row_bytes),
            "root_code_sha256": sha256(root_code.encode("utf-8")),
            "skill_bindings": skills,
        })
    return {
        "schema": "natlang.luna-source-row-inventory/1",
        "source_path": str(source),
        "source_sha256": sha256(source_bytes),
        "entries": entries,
    }


def verify(inventory_path: Path, source: Path, cases: list[dict[str, Any]]) -> dict[str, Any]:
    pinned = json.loads(inventory_path.read_text(encoding="utf-8"))
    current = inventory(source, cases)
    if pinned != current:
        raise ValueError("source inventory does not match current source rows and dispatcher case seeds")
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--cases", type=Path, required=True, help="JSON file containing the dispatcher's cases array")
    build.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("verify")
    check.add_argument("--source", type=Path, required=True)
    check.add_argument("--cases", type=Path, required=True)
    check.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    case_value = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = case_value.get("cases") if isinstance(case_value, dict) else case_value
    if not isinstance(cases, list):
        raise ValueError("cases input must be a JSON array or object with a cases array")
    if args.command == "build":
        result = inventory(args.source, cases)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    else:
        result = verify(args.inventory, args.source, cases)
    print(json.dumps({
        "status": "verified",
        "source": result["source_path"],
        "source_sha256": result["source_sha256"],
        "entries": len(result["entries"]),
        "inventory_sha256": sha256(args.output.read_bytes()) if args.command == "build" else sha256(args.inventory.read_bytes()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
