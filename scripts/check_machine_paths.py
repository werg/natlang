#!/usr/bin/env python3
"""Ratchet: tracked .py files may not gain absolute machine paths.

Lists tracked .py files containing `/home/werg`, `/mnt/external`, or the hostnames
`pop-os`/`mltick` and compares against scripts/machine-paths-baseline.json.
A file not in the baseline fails; a baseline file that is now clean prints
"shrink the baseline" (and fails only with --strict). Use --update to rewrite the
baseline (only ever to remove entries).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "machine-paths-baseline.json"
PATTERN = re.compile(r"/home/werg|/mnt/external|\bpop-os\b|\bmltick\b")
# The sanctioned home of the default roots, and the ratchet's own sources.
EXEMPT = {
    "training/neuralese/natlang_neuralese/common/paths.py",
    "tests/test_common_paths.py",
    "scripts/check_machine_paths.py",
    "tests/test_machine_paths_ratchet.py",
}


def tracked_py(root: Path = ROOT) -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def offenders(root: Path = ROOT) -> list[str]:
    found = []
    for rel in tracked_py(root):
        if rel in EXEMPT:
            continue
        try:
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if PATTERN.search(text):
            found.append(rel)
    return sorted(found)


def load_baseline(path: Path = BASELINE) -> list[str]:
    return sorted(json.loads(path.read_text(encoding="utf-8"))["files"])


def compare(current: list[str], baseline: list[str]) -> tuple[list[str], list[str]]:
    return sorted(set(current) - set(baseline)), sorted(set(baseline) - set(current))


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    current = offenders()
    if "--update" in args:
        BASELINE.write_text(json.dumps({"files": current}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(current)} files to {BASELINE}")
        return 0
    new, gone = compare(current, load_baseline())
    for rel in new:
        print(f"NEW machine path (use natlang_neuralese.common.paths): {rel}")
    for rel in gone:
        print(f"shrink the baseline: {rel} no longer has machine paths")
    if new:
        return 1
    return 1 if gone and "--strict" in args else 0


if __name__ == "__main__":
    sys.exit(main())
