#!/usr/bin/env python3
"""Refresh inventory from an existing pipeline config and write a pinned readiness receipt."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from inventory_training_data import catalog


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".pending", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        Path(name).replace(path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--ready-out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config = json.loads(args.pipeline.read_text())
    source_report_path, report = catalog(repo, config)
    report_bytes = source_report_path.read_bytes()
    report_sha = hashlib.sha256(report_bytes).hexdigest()
    # Keep the inventory's canonical full report in data-inventory; this run copy
    # is the immutable pipeline-stage output used by readiness validation.
    atomic_bytes(args.report_out, report_bytes)
    ready = not (report.get("missing_required_default_inputs") or report.get("included_quality_blockers"))
    receipt = {"schema": "natlang.training_inventory_readiness/1", "ready": ready,
               "report": str(args.report_out.resolve()), "sha256": report_sha,
               "catalog_report": str(source_report_path.resolve()),
               "catalog_report_sha256": report_sha,
               "policy_sha256": hashlib.sha256((repo / "training/data_sources.json").read_bytes()).hexdigest(),
               "missing_required_default_inputs": report.get("missing_required_default_inputs", []),
               "included_quality_blockers": report.get("included_quality_blockers", [])}
    atomic_json(args.ready_out, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if ready else 75


if __name__ == "__main__":
    raise SystemExit(main())
