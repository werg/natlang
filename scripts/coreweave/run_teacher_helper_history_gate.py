"""Run the exact external-asset preflight before historical custody and stage tests.

Use this command for the historical/integration gate. Ordinary portable tests
may collect the history module and report explicit skips, but this command
cannot report success unless every pinned historical input is present and the
custody and real-asset stage tests pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    preflight = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/coreweave/check_teacher_helper_history.py"),
            "--require",
        ],
        cwd=ROOT,
        check=False,
    )
    if preflight.returncode:
        return preflight.returncode
    env = {**os.environ, "NATLANG_REQUIRE_TEACHER_HELPER_HISTORY": "1"}
    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_teacher_helper_continuation.py",
        ],
        cwd=ROOT,
        check=False,
        env=env,
    )
    if tests.returncode:
        return tests.returncode
    stage_tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "historical",
            "tests/test_stage_teacher_helper_continuation.py",
        ],
        cwd=ROOT,
        check=False,
        env=env,
    )
    return stage_tests.returncode


if __name__ == "__main__":
    raise SystemExit(main())
