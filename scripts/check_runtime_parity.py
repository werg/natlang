"""Require a cross-runtime counterpart and paired fixture for semantic runtime changes."""
from __future__ import annotations

import argparse
import re
import subprocess

PYTHON_RUNTIME = re.compile(
    r"^natlang/(?:runtime|tool_agent|model_agent|codebase|types|values|invocation|"
    r"meta|surface|slots|refs|paths|trace|scenario|execution|nodes|checks)\.py$"
    r"|^natlang/prompts/"
)
TYPESCRIPT_RUNTIME = re.compile(r"^ts-host/src/native/|^ts-host/src/contracts\.ts$")
PAIRED_FIXTURE = re.compile(
    r"^ts-host/test/native-parity\.test\.mjs$|^conformance/(?:programs/|infrastructure_baseline\.json$)"
)


def violations(paths: list[str]) -> list[str]:
    python = [path for path in paths if PYTHON_RUNTIME.search(path)]
    typescript = [path for path in paths if TYPESCRIPT_RUNTIME.search(path)]
    fixtures = [path for path in paths if PAIRED_FIXTURE.search(path)]
    errors = []
    if python and not typescript:
        errors.append("Python runtime changed without a TypeScript runtime counterpart: " + ", ".join(python))
    if typescript and not python and not fixtures:
        errors.append("TypeScript runtime changed without a Python counterpart or paired fixture: " +
                      ", ".join(typescript))
    if (python or typescript) and not fixtures:
        errors.append("Runtime change needs a paired Python/TypeScript fixture in ts-host/test/native-parity.test.mjs or conformance/")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base commit or ref")
    parser.add_argument("--head", default="HEAD", help="head commit or ref")
    args = parser.parse_args()
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "-z", f"{args.base}...{args.head}"]
    ).decode().split("\0")
    errors = violations([path for path in changed if path])
    if errors:
        for error in errors:
            print("PARITY: " + error)
        return 1
    print("Runtime parity change gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
