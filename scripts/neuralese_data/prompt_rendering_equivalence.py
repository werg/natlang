#!/usr/bin/env python3
"""Equivalence table of superseded prompt-rendering identities (data/rendering_contract.py).

For every state of the /1 contract sources (data/prompt_rendering.py, serve/engine.py, serve/chat.py and
training/prompt-pieces/pieces.json) since /1 was introduced (c1de43eb), this computes the recorded /1 identity from
the git blobs and the /2 behavioural identity by running the CURRENT fingerprint (rendering_contract.py and its
fixtures) over that commit's blobs in a materialised copy of the package. The table maps each /1 identity to that
/2 identity; rendering_contract.binding_accepted accepts a /1 binding only when its /2 equals the current one.

  prompt_rendering_equivalence.py [--write]     # print, or write training/prompt-pieces/rendering-identity-equivalence.json

When the fixture set changes, rerun with --write (every /2 value is recomputed under the new fixtures) and add the
superseded /2 identities as entries too. tests/neuralese/test_rendering_contract.py re-verifies every entry from git.
Runs under the memory ledger (imports torch through serve/engine; about 1 GB).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "training/neuralese/natlang_neuralese"
V1_SOURCES = ("data/prompt_rendering.py", "serve/engine.py", "serve/chat.py")
PIECES = "training/prompt-pieces/pieces.json"
TABLE = ROOT / "training/prompt-pieces/rendering-identity-equivalence.json"
V1_INTRODUCED = "c1de43eb"
# The /1 state live from 058a5269 until /2 (also restored by 54067476): evaluation comparison keys recorded under it
# keep comparing while the behaviour is unchanged (rendering_contract.comparison_binding).
COMPARISON_CANONICAL_COMMIT = "058a5269"
MODES = ("full", "program-header")
# Declared rendering changes that apply to artifacts bound before them: (commit, mode, decision). The table records
# the /2 identity at the commit's parent (from_v2) and at the commit (to_v2).
ADOPTIONS = (
    ("95bc6eca", "program-header",
     "Pop 2026-10-12: program-header serving keeps externally supplied tool schemas; native natlang requests send "
     "none in this mode, so existing program-header exports keep their native prompts"),
)


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], check=True,
                          capture_output=True).stdout


def v1_identity(blobs: dict, pieces: bytes, mode: str) -> dict:
    """The /1 formula (natlang.prompt-rendering-identity/1) over source blobs."""
    implementation = {name: hashlib.sha256(blobs[name]).hexdigest() for name in V1_SOURCES}
    pieces_sha = hashlib.sha256(pieces).hexdigest()
    identity = {"schema": "natlang.prompt-rendering-identity/1", "mode": mode,
                "implementation_sha256": implementation, "pieces_sha256": pieces_sha}
    identity["identity_sha256"] = hashlib.sha256(
        (identity["schema"] + "\0" + mode + "\0" + json.dumps(implementation, sort_keys=True) +
         "\0" + pieces_sha).encode()).hexdigest()
    return identity


def v2_identities(blobs: dict, pieces: bytes, edit=None) -> dict:
    """/2 identities per mode of the working-tree package with ``blobs`` (package-relative) and ``pieces`` swapped in.

    ``edit(package_dir)`` may change the materialised copy further (tests)."""
    with tempfile.TemporaryDirectory(prefix="prompt-identity-") as tmp:
        tmp = Path(tmp)
        package = tmp / PACKAGE
        shutil.copytree(ROOT / PACKAGE, package, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "training/prompt-pieces", tmp / "training/prompt-pieces")
        for name, data in blobs.items():
            (package / name).write_bytes(data)
        (tmp / PIECES).write_bytes(pieces)
        if edit is not None:
            edit(package)
        code = ("import json, sys\nsys.path.insert(0, sys.argv[1])\n"
                "from natlang_neuralese.data.rendering_contract import compute_identity\n"
                f"print(json.dumps({{m: compute_identity(m) for m in {MODES!r}}}))\n")
        out = subprocess.run([sys.executable, "-c", code, str(tmp / "training/neuralese")], check=True,
                             capture_output=True, text=True, cwd=tmp)
        return json.loads(out.stdout.strip().splitlines()[-1])


def blobs_at(commit: str) -> tuple[dict, bytes]:
    return ({name: git_blob(commit, f"{PACKAGE}/{name}") for name in V1_SOURCES}, git_blob(commit, PIECES))


def adoption_entries() -> list:
    out = []
    for commit, mode, decision in ADOPTIONS:
        before = v2_identities(*blobs_at(commit + "^"))[mode]["identity_sha256"]
        after = v2_identities(*blobs_at(commit))[mode]["identity_sha256"]
        out.append({"commit": commit, "mode": mode, "decision": decision, "from_v2": before, "to_v2": after})
    return out


def v1_states() -> list[str]:
    """Commits (oldest first) that changed a /1 source since /1 was introduced, plus the introducing commit."""
    paths = [f"{PACKAGE}/{name}" for name in V1_SOURCES] + [PIECES]
    revs = subprocess.run(["git", "-C", str(ROOT), "rev-list", "--reverse", f"{V1_INTRODUCED}^..HEAD", "--", *paths],
                          check=True, capture_output=True, text=True).stdout.split()
    return [rev[:8] for rev in revs]


def entries_at(commit: str) -> dict:
    blobs, pieces = blobs_at(commit)
    if b"prompt-rendering-identity/1" not in blobs["data/prompt_rendering.py"]:
        return {}  # this state computes /2 itself
    v2 = v2_identities(blobs, pieces)
    out = {}
    for mode in MODES:
        identity = v1_identity(blobs, pieces, mode)
        out[identity["identity_sha256"]] = {
            "mode": mode, "commit": commit, "identity": identity,
            "v2_identity_sha256": v2[mode]["identity_sha256"],
            "comparison_canonical": commit.startswith(COMPARISON_CANONICAL_COMMIT)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    identities = {}
    for commit in v1_states():
        for sha, entry in entries_at(commit).items():
            identities.setdefault(sha, entry)  # a state restored later keeps its first commit
    table = {"schema": "natlang.prompt-rendering-identity-equivalence/1",
             "generated_by": "scripts/neuralese_data/prompt_rendering_equivalence.py",
             "identities": identities, "adoptions": adoption_entries()}
    text = json.dumps(table, indent=1, sort_keys=True) + "\n"
    if args.write:
        TABLE.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
