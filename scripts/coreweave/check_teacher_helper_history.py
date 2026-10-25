#!/usr/bin/env python3
"""Preflight the external immutable inputs required by historical custody tests.

This check is deliberately separate from ordinary unit-test collection: the
historical bytes are not in Git and must be provisioned from their preserved
authority source before the exact integrity gate can run.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "training" / "neuralese"))
from scripts.coreweave import teacher_helper_continuation as custody


def check_reference(reference: dict[str, object]) -> str | None:
    """Return why one exact path/SHA/byte pin is unavailable or mismatched."""
    path = Path(str(reference["path"]))
    if path.is_symlink() or not path.is_file():
        return f"missing regular file: {path}"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != reference["sha256"] or len(raw) != reference["bytes"]:
        return f"identity mismatch: {path} (sha256={digest}, bytes={len(raw)})"
    return None


def check(references: tuple[dict[str, object], ...] | None = None) -> list[str]:
    pins = references or (custody.LEGACY_OBSERVATION, custody.LEGACY_AUTHORITY,
                          custody.SOURCE_CLOSURE_RECEIPT, custody.SOURCE_CLOSURE_TRANSITION)
    return [problem for reference in pins if (problem := check_reference(reference))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", action="store_true",
                        help="exit nonzero when historical inputs are unavailable")
    args = parser.parse_args()
    problems = check()
    if not problems:
        print("PASS exact historical helper inputs are present and match pinned path/SHA/byte identities")
        return 0
    print("BLOCKED exact historical helper integrity gate; no substitute assets are accepted:")
    for problem in problems:
        print(f"  - {problem}")
    print("Provision the preserved files at the exact pinned paths from the immutable historical authority, "
          "then rerun this preflight and tests/test_teacher_helper_continuation.py.")
    return 1 if args.require else 0


if __name__ == "__main__":
    raise SystemExit(main())
