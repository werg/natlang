"""Losslessly archive superseded rendered SFT JSONL files.

The archive is published only after its decompressed bytes match the source.
Source removal happens last, so an interrupted run can be rerun safely.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def archive(path: Path, level: int) -> tuple[int, int, str]:
    if not path.name.endswith(".sft.jsonl") or not path.is_file():
        raise ValueError(f"expected an existing .sft.jsonl file: {path}")
    target = path.with_name(path.name + ".zst")
    temporary = target.with_name(target.name + ".tmp")
    if target.exists() or temporary.exists():
        raise FileExistsError(target if target.exists() else temporary)

    original_size = path.stat().st_size
    original_digest = digest(path)
    try:
        subprocess.run(
            ["zstd", "-T2", f"-{level}", "-q", "-o", str(temporary), str(path)],
            check=True,
        )
        hasher = hashlib.sha256()
        with subprocess.Popen(["zstd", "-dc", str(temporary)], stdout=subprocess.PIPE) as proc:
            assert proc.stdout is not None
            for chunk in iter(lambda: proc.stdout.read(1024 * 1024), b""):
                hasher.update(chunk)
            if proc.wait() != 0:
                raise RuntimeError(f"cannot decompress {temporary}")
        if hasher.hexdigest() != original_digest:
            raise RuntimeError(f"archive digest mismatch for {path}")
        temporary.replace(target)
        archived_size = target.stat().st_size
        path.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return original_size, archived_size, original_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--level", type=int, default=3, choices=range(1, 10))
    args = parser.parse_args()
    for path in args.paths:
        original, archived, sha = archive(path, args.level)
        print(f"{path}\t{original}\t{archived}\t{sha}", flush=True)


if __name__ == "__main__":
    main()
