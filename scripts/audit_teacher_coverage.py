#!/usr/bin/env python3
"""Fail when a project target is undiscovered or lacks varied frozen teacher inputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def discovered(root: Path) -> set[str]:
    codebases = {f"codebase:{path.name}" for path in (root / "codebases").iterdir()
                 if path.is_dir() and any(path.glob("*.nl"))}
    applications = {f"application:{path.name}" for path in (root / "applications").iterdir()
                    if path.suffix in (".py", ".mjs")}
    return codebases | applications


def audit(root: Path, config_path: Path, studio_path: Path, programs_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    targets = config["targets"]
    owned = {item for target in targets for item in target["owns"]}
    excluded = set(config.get("excluded", []))
    actual = discovered(root)
    unknown = sorted(actual - owned - excluded)
    missing = sorted(owned - actual)
    counts = Counter()
    for raw in studio_path.read_text().splitlines():
        if raw.strip():
            row = json.loads(raw)
            counts[(row["target"], row["split"])] += 1
    for raw in programs_path.read_text().splitlines():
        if raw.strip():
            row = json.loads(raw)
            counts[("family:" + row.get("family", row["kind"]), row.get("split", "train"))] += 1
    below = []
    for target in targets:
        for split, minimum in config["minimums"][target["provider"]].items():
            found = counts[(target["id"], split)]
            if found < minimum:
                below.append({"target": target["id"], "split": split,
                              "found": found, "minimum": minimum})
    return {"schema": "natlang.teacher_coverage_audit/1", "discovered": len(actual),
            "registered": len(owned), "targets": len(targets), "unknown": unknown,
            "missing": missing, "below_minimum": below,
            "ready": not (unknown or missing or below)}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "training/teacher_coverage.json")
    parser.add_argument("--studio-cases", type=Path, default=root / "data/teacher/studio-cases-v1.jsonl")
    parser.add_argument("--program-ir", type=Path,
                        default=root / "data/external_pilot/teacher-selection-balanced-s909.ir.jsonl")
    args = parser.parse_args()
    report = audit(root, args.config, args.studio_cases, args.program_ir)
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
