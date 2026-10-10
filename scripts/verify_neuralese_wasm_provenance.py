#!/usr/bin/env python3
"""Check the vendored browser Neuralese service (ts-host/vendor/neuralese-wasm) against its provenance.json.

- every vendored module (`*.mjs`, `*.wasm`) is listed under `sha256`, every listed file exists, and its bytes hash to
  the recorded digest;
- `fork_commit` is the fork commit training/neuralese/llama-cpp-fork.json pins (`commit`), so the browser runtime
  serves the same protocol as the native server;
- with `--fork DIR` (a checkout of werg/llama.cpp-neuralese), that commit exists there, and with `--builds` the build
  outputs in DIR (build-wasm, build-wasm-mt, build-wasm-gpu) are byte-identical to the vendored files.

Exit status 0 when everything checks out; otherwise each problem is printed and the status is 1.

    python3 scripts/verify_neuralese_wasm_provenance.py [--fork ../llama.cpp-neuralese [--builds]]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "ts-host" / "vendor" / "neuralese-wasm"
PIN = ROOT / "training" / "neuralese" / "llama-cpp-fork.json"
BUILD_DIRS = {"neuralese-wasm": "build-wasm", "neuralese-wasm-mt": "build-wasm-mt", "neuralese-wasm-gpu": "build-wasm-gpu"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def problems(vendor: Path = VENDOR, pin_file: Path = PIN, fork: Path | None = None, builds: bool = False) -> list[str]:
    found: list[str] = []
    provenance = json.loads((vendor / "provenance.json").read_text())
    recorded: dict = provenance.get("sha256") or {}
    modules = sorted(p.name for p in vendor.iterdir() if p.suffix in (".mjs", ".wasm"))
    for name in modules:
        if name not in recorded:
            found.append(f"{name} is vendored but has no sha256 in provenance.json")
    for name, digest in sorted(recorded.items()):
        path = vendor / name
        if not path.exists():
            found.append(f"{name} is listed in provenance.json but not vendored")
        elif sha256(path) != digest:
            found.append(f"{name}: sha256 {sha256(path)} does not match the recorded {digest}")
    built = str(provenance.get("fork_commit") or "")
    pin = str(json.loads(pin_file.read_text()).get("commit") or "")
    if not built or not pin or not (pin.startswith(built) or built.startswith(pin)):
        found.append(f"built from fork {built or '?'}, but llama-cpp-fork.json pins {pin or '?'}")
    if fork is not None:
        verified = subprocess.run(["git", "-C", str(fork), "cat-file", "-e", f"{built}^{{commit}}"], capture_output=True)
        if verified.returncode != 0:
            found.append(f"fork commit {built} is not in {fork}")
        if builds:
            for name in recorded:
                stem = name.rsplit(".", 1)[0]
                output = fork / BUILD_DIRS.get(stem, "?") / "bin" / name
                if not output.exists():
                    found.append(f"{name}: no build output at {output}")
                elif sha256(output) != recorded[name]:
                    found.append(f"{name}: the build output {output} differs from the vendored file")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fork", type=Path, help="a checkout of the fork, to check the commit (and --builds)")
    parser.add_argument("--builds", action="store_true", help="also compare the fork's build outputs byte for byte")
    args = parser.parse_args()
    found = problems(fork=args.fork, builds=args.builds)
    for problem in found:
        print(problem, file=sys.stderr)
    if not found:
        print("ts-host/vendor/neuralese-wasm matches its provenance")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
