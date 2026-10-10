#!/usr/bin/env python3
"""Ratchet: duplicate helper definitions outside natlang_neuralese/common and tests/ may only fall.

Counts `def` lines named sha, sha<digits>*, sha_*, canonical*, write_json* or utc_now (not e.g. shaped_*) per tracked
.py file and compares with scripts/duplicate-helpers-baseline.json ({"files": {path: count}}).
A file's count growing, or a new file appearing, fails. Falling counts print "shrink the baseline".
--update rewrites the baseline from the current tree.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "duplicate-helpers-baseline.json"
DEF = re.compile(r"^\s*(?:async\s+)?def\s+(?:sha(?:[\d_]\w*)?|canonical\w*|write_json\w*|utc_now)\b", re.M)
EXCLUDED_PREFIXES = ("training/neuralese/natlang_neuralese/common/", "tests/")


def counts(root: Path = ROOT) -> dict[str, int]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True, check=True).stdout
    result = {}
    for rel in filter(None, out.splitlines()):
        if rel.startswith(EXCLUDED_PREFIXES):
            continue
        try:
            n = len(DEF.findall((root / rel).read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
        if n:
            result[rel] = n
    return dict(sorted(result.items()))


def load_baseline(path: Path = BASELINE) -> dict[str, int]:
    return json.loads(path.read_text(encoding="utf-8"))["files"]


def compare(current: dict[str, int], baseline: dict[str, int]) -> tuple[list[str], list[str]]:
    grown = [f"{f}: {baseline.get(f, 0)} -> {n}" for f, n in current.items() if n > baseline.get(f, 0)]
    shrunk = [f"{f}: {n} -> {current.get(f, 0)}" for f, n in baseline.items() if current.get(f, 0) < n]
    return grown, shrunk


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    current = counts()
    if "--update" in args:
        BASELINE.write_text(json.dumps({"files": current}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {sum(current.values())} definitions in {len(current)} files")
        return 0
    grown, shrunk = compare(current, load_baseline())
    for line in grown:
        print(f"DUPLICATE helper count grew (import from natlang_neuralese.common): {line}")
    for line in shrunk:
        print(f"shrink the baseline: {line}")
    return 1 if grown else 0


if __name__ == "__main__":
    sys.exit(main())
