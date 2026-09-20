#!/usr/bin/env python3
"""Bundle rendered SFT JSONL files with duplicate-ID and template checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


RENDERER_FIELDS = ("template_sha256", "end_token", "terminal_tool_policy",
                   "teacher_reasoning_policy")


def combine(destination: Path, sources: list[Path]) -> dict:
    if destination.exists():
        raise ValueError(f"refusing overwrite: {destination}")
    if not sources or len({p.resolve() for p in sources}) != len(sources):
        raise ValueError("supply distinct source files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_suffix(destination.suffix + f".building.{os.getpid()}")
    identities = set()
    renderer = None
    inputs = []
    pairs = reasoning_pairs = 0
    output_hash = hashlib.sha256()
    try:
        with stage.open("xb") as target:
            for path in sources:
                manifest_path = path.with_suffix(path.suffix + ".manifest.json")
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
                identity = ({key: manifest["renderer"].get(key) for key in RENDERER_FIELDS}
                            if manifest and manifest.get("renderer") else None)
                if identity is not None:
                    if renderer is not None and identity != renderer:
                        raise ValueError(f"incompatible SFT renderer: {path}")
                    renderer = identity
                source_hash = hashlib.sha256()
                count = 0
                with path.open("rb") as stream:
                    for number, line in enumerate(stream, 1):
                        source_hash.update(line)
                        if not line.endswith(b"\n"):
                            raise ValueError(f"{path}:{number}: missing JSONL newline")
                        row = json.loads(line)
                        if row["id"] in identities:
                            raise ValueError(f"duplicate SFT id: {row['id']}")
                        identities.add(row["id"])
                        target.write(line)
                        output_hash.update(line)
                        count += 1
                        reasoning_pairs += "<think>" in row["completion"]
                inputs.append({"path": str(path), "sha256": source_hash.hexdigest(),
                               "pairs": count, "manifest": str(manifest_path) if manifest else None})
                pairs += count
        stage.replace(destination)
    finally:
        stage.unlink(missing_ok=True)
    result = {"schema": "natlang.sft_bundle/1", "sources": inputs,
              "renderer": renderer, "pairs": pairs, "reasoning_pairs": reasoning_pairs,
              "sha256": output_hash.hexdigest()}
    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    result = combine(args.destination, args.sources)
    print(f"{result['pairs']} pairs, {result['reasoning_pairs']} with reasoning -> {args.destination}")


if __name__ == "__main__":
    main()
