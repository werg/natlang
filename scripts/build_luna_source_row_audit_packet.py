#!/usr/bin/env python3
"""Make a source-bound copy of a Luna source-row audit packet.

Identity fields are derived from the exact physical JSONL row and the packet's
pinned raw result. The builder never rewrites its input packet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def object_sha(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def source_row(path: Path, physical_index: int, expected_file_sha: str) -> tuple[dict[str, Any], bytes]:
    file_bytes = path.read_bytes()
    actual_file_sha = sha256(file_bytes)
    if actual_file_sha != expected_file_sha:
        raise ValueError("source file bytes do not match packet source pin")
    lines = file_bytes.splitlines(keepends=True)
    if physical_index < 0 or physical_index >= len(lines):
        raise ValueError("source physical row index is outside the pinned JSONL file")
    row_bytes = lines[physical_index]
    body = row_bytes.removesuffix(b"\n").removesuffix(b"\r")
    row = json.loads(body)
    if not isinstance(row, dict):
        raise ValueError("selected physical source row is not a JSON object")
    return row, row_bytes


def derive_binding(row: dict[str, Any], row_bytes: bytes, file_sha: str,
                   source_path: str, physical_index: int) -> dict[str, Any]:
    source_id = row.get("id", row.get("case_id"))
    groups = row.get("source_groups", row.get("groups"))
    if not isinstance(source_id, str) or not isinstance(groups, list) or not groups:
        raise ValueError("source row lacks a stable ID or source group")
    split = row.get("split")
    if not isinstance(split, str) or not split:
        raise ValueError("source row lacks a split")
    try:
        root_code = row["curriculum"]["reference"]["root"][0][1]["code"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("source row lacks authored root code") from None
    if not isinstance(root_code, str):
        raise ValueError("authored root code is not a string")
    return {
        "path": source_path,
        "file_sha256": file_sha,
        "physical_index": physical_index,
        "source_case_id": source_id,
        "source_group": groups[0],
        "split": split,
        "row_sha256_including_lf": sha256(row_bytes),
        "authored_root_code_sha256": sha256(root_code.encode("utf-8")),
    }


def bind_packet(packet_path: Path, repo_root: Path) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    source = packet.get("source") or packet.get("source_binding")
    if not isinstance(source, dict):
        raise ValueError("audit packet has no source binding")
    source_path = source.get("path")
    file_sha = source.get("sha256", source.get("file_sha256"))
    index = source.get("physical_index", source.get("index"))
    if not isinstance(source_path, str) or not isinstance(file_sha, str) or not isinstance(index, int):
        raise ValueError("audit packet source binding lacks path/hash/physical index")
    resolved_source = (repo_root / source_path).resolve()
    if not resolved_source.is_relative_to(repo_root.resolve()):
        raise ValueError("source path escapes repository root")
    row, row_bytes = source_row(resolved_source, index, file_sha)
    binding = derive_binding(row, row_bytes, file_sha, source_path, index)
    old_source_id = source.get("source_case_id", source.get("id", source.get("program_id")))
    old_group = source.get("source_group")
    if old_group is not None and old_group != binding["source_group"]:
        raise ValueError("packet source group differs from the exact physical source row")

    result_ref = packet.get("input_and_output", {}).get("result")
    result_path = packet.get("run", {}).get("result_path")
    result_sha = packet.get("run", {}).get("result_sha256")
    if isinstance(result_ref, dict):
        result_path = result_ref.get("path")
        result_sha = result_ref.get("sha256")
    if not isinstance(result_path, str) or not isinstance(result_sha, str):
        raise ValueError("audit packet lacks a pinned raw result path/hash")
    resolved_result = (repo_root / result_path).resolve()
    if not resolved_result.is_relative_to(repo_root.resolve()):
        raise ValueError("raw result path escapes repository root")
    result_bytes = resolved_result.read_bytes()
    if sha256(result_bytes) != result_sha:
        raise ValueError("raw result bytes do not match audit packet pin")
    result = json.loads(result_bytes)
    if not isinstance(result.get("id"), str) or not isinstance(result.get("execution_identity"), dict):
        raise ValueError("raw result lacks materialized program or execution identity")
    if packet.get("input_and_output", {}).get("result_id") not in (None, result["id"]):
        raise ValueError("packet materialized result ID differs from pinned raw result")
    if packet.get("run", {}).get("result_sha256") not in (None, result_sha):
        raise ValueError("packet run result hash differs from pinned raw result")

    result_id = result["id"]
    old_ids = {source.get("id"), source.get("program_id"), source.get("source_case_id")}
    if old_source_id not in (None, binding["source_case_id"], result_id):
        raise ValueError("packet source ID is neither the physical source row ID nor the pinned result ID")
    if result_id not in old_ids and packet.get("input_and_output", {}).get("result_id") != result_id:
        raise ValueError("packet does not identify the same materialized program as the pinned result")

    trace_ref = packet.get("input_and_output", {}).get("trace")
    trace_path = packet.get("run", {}).get("trace_path")
    trace_sha = packet.get("run", {}).get("trace_sha256")
    if isinstance(trace_ref, dict):
        trace_path = trace_ref.get("path")
        trace_sha = trace_ref.get("sha256")
    if not isinstance(trace_path, str) or not isinstance(trace_sha, str):
        raise ValueError("audit packet lacks a pinned trace path/hash")
    resolved_trace = (repo_root / trace_path).resolve()
    if not resolved_trace.is_relative_to(repo_root.resolve()) or sha256(resolved_trace.read_bytes()) != trace_sha:
        raise ValueError("raw trace path escapes repository or bytes do not match packet pin")
    packet["source_binding"] = binding
    if isinstance(packet.get("source"), dict):
        packet["source"] = {
            **packet["source"],
            "id": binding["source_case_id"],
            "source_case_id": binding["source_case_id"],
            "materialized_program_id": result_id,
        }
    packet["materialized_program_id"] = result_id
    packet["execution_identity"] = result["execution_identity"]
    packet["trajectory_id"] = result["execution_identity"].get("execution_run_id")
    packet["source_binding_provenance"] = {
        "method": "exact-physical-jsonl-row-plus-pinned-raw-result-v1",
        "raw_result_sha256": result_sha,
        "source_row_sha256_including_lf": binding["row_sha256_including_lf"],
        "context_digest_convention": packet.get("digest_conventions", {}).get("context_sha256_canonical_json"),
    }
    packet["schema"] = packet["schema"].replace("/1", "/2")
    return packet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    packet = bind_packet(args.input, args.repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with output.open("xb") as f:
        f.write(data)
    print(json.dumps({"output": str(output), "bytes": len(data), "sha256": sha256(data),
                      "source_case_id": packet["source_binding"]["source_case_id"],
                      "source_group": packet["source_binding"]["source_group"],
                      "materialized_program_id": packet["materialized_program_id"],
                      "trajectory_id": packet["trajectory_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
