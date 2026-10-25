#!/usr/bin/env python3
"""Run the exact external-asset preflight before the historical custody tests.

Use this command for the historical/integration gate. Ordinary portable tests
may collect the history module and report explicit skips, but this command
cannot report success unless every pinned historical input is present and the
custody tests pass.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    preflight = subprocess.run(
        [sys.executable, str(ROOT / "scripts/coreweave/check_teacher_helper_history.py"), "--require"],
        cwd=ROOT, check=False)
    if preflight.returncode:
        return preflight.returncode
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_teacher_helper_continuation.py"], cwd=ROOT, check=False)
    return tests.returncode


if __name__ == "__main__":
    raise SystemExit(main())
