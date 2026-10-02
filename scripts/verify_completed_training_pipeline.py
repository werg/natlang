#!/usr/bin/env python3
"""Verify an immutable completed pipeline run before reusing its artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest_path(path: Path) -> str:
    path = Path(path)
    if path.is_dir():
        files = {str(item.relative_to(path)): digest_path(item)
                 for item in sorted(path.rglob("*")) if item.is_file()}
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verify(config_path: Path, run_path: Path, required_stages) -> dict:
    config_path, run_path = Path(config_path).resolve(), Path(run_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if Path(config.get("run_directory", "")).resolve() != run_path:
        raise ValueError("parent recipe run_directory does not match the requested run")
    state_path = run_path / "pipeline-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_config = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if state.get("config_sha256") != expected_config or state.get("status") != "complete":
        raise ValueError("parent pipeline state is not complete or does not match its recipe")
    verified = {}
    historical_code_deltas = {}
    code_suffixes = {".py", ".mjs", ".js", ".ts", ".cjs", ".jsonc"}
    for stage_id in required_stages:
        stage = state.get("stages", {}).get(stage_id)
        if not stage or stage.get("status") != "complete" or stage.get("exit_code") != 0:
            raise ValueError(f"required parent stage is not complete: {stage_id}")
        for raw_path, expected in stage.get("inputs", {}).items():
            path = Path(raw_path)
            if not path.exists():
                raise ValueError(f"parent input is missing: {path}")
            actual = digest_path(path)
            if actual != expected:
                if path.suffix in code_suffixes:
                    historical_code_deltas[str(path)] = {
                        "recorded_sha256": expected, "current_sha256": actual}
                else:
                    raise ValueError(f"parent input changed: {path}")
        for raw_path, expected in stage.get("outputs", {}).items():
            path = Path(raw_path)
            if not path.exists() or digest_path(path) != expected:
                raise ValueError(f"parent output changed or is missing: {path}")
        verified[stage_id] = {
            "status": stage["status"], "exit_code": stage["exit_code"],
            "inputs": stage.get("inputs", {}), "outputs": stage.get("outputs", {})}
    return {
        "schema": "natlang.completed_training_pipeline_proof/1",
        "config_path": str(config_path),
        "config_file_sha256": digest_path(config_path),
        "config_identity_sha256": expected_config,
        "run_path": str(run_path),
        "state_sha256": digest_path(state_path),
        "required_stages": verified,
        "historical_code_input_deltas": historical_code_deltas,
        "verified": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-stage", action="append", required=True)
    args = parser.parse_args()
    proof = verify(args.config, args.run, args.require_stage)
    atomic_json(args.output, proof)
    print(json.dumps({"verified": True, "stages": list(proof["required_stages"]),
                      "state_sha256": proof["state_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
