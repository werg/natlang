#!/usr/bin/env python3
"""Prepare exact-oracle Luna source rows for the native episode capturer.

This is a source-preserving conversion for already accepted exact results. It
does not judge, regrade, or change their outcome. One row per output keeps the
capturer's complete invocation graph and decision chain independently bound.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training/neuralese"))

from scripts.neuralese_data.portable_paths import node_executable

from natlang_neuralese.common.hashing import sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import canonical_json_bytes
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.provider_campaign_source import source_binding


SCHEMA = "natlang.luna-provider-native-exact-ingestion-audit/1"


def _rows(path: Path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def _binding(path: Path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return {"path": str(path), "sha256": sha256_hex(data), "bytes": len(data)}


def _runtime_row_digest(runtime: Path, row: dict, decision_receipt: dict) -> str:
    module = Path(runtime).resolve(strict=True) / "dist/teacher/native-materializer.js"
    expected = (decision_receipt.get("identity", {}).get("runtime") or {}).get(
        "ts-host/dist/teacher/native-materializer.js")
    if not isinstance(expected, str) or sha256_file_hex(module) != expected:
        raise ValueError("decision cache materializer differs from the campaign's exact frozen runtime")
    node = node_executable()
    code = ('import fs from "node:fs";'
            'const {nativeRowDigest}=await import(process.argv[1]);'
            'process.stdout.write(nativeRowDigest(JSON.parse(fs.readFileSync(0,"utf8"))));')
    completed = subprocess.run([node, "--input-type=module", "-e", code, module.as_uri()],
        input=json.dumps(row, ensure_ascii=False), text=True, capture_output=True,
        check=True, timeout=30)
    value = completed.stdout.strip()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("frozen materializer returned an invalid source row digest")
    return value


def build(*, provider_source: Path, plan_path: Path, decisions_path: Path,
          decision_receipt_path: Path, protection_path: Path, task_id: str,
          output_dir: Path):
    provider_source = Path(provider_source).resolve(strict=True)
    declaration = json.loads(provider_source.read_bytes())
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_bytes())
    decisions_path = Path(decisions_path).resolve(strict=True)
    decision_receipt_path = Path(decision_receipt_path).resolve(strict=True)
    protection_path = Path(protection_path).resolve(strict=True)
    if (declaration.get("schema") != "natlang.provider-campaign-source/1"
            or declaration.get("training_admission") is not False
            or declaration.get("training_exposure") is not False
            or declaration.get("oracle_regrades") != []
            or declaration.get("model") != "gpt-6-luna"
            or declaration.get("provider") != "openai-codex"
            or declaration.get("plan", {}).get("path") != str(plan_path)
            or declaration.get("plan", {}).get("sha256") != sha256_file_hex(plan_path)
            or plan.get("root_approved") is not True or plan.get("model_id") != "gpt-6-luna"
            or plan.get("provider") != "openai-codex"):
        raise ValueError("exact native conversion requires this closed reviewed Luna source and plan")
    raw_candidates = [entry for entry in declaration.get("results", [])
                      if entry.get("result", {}).get("path")]
    match = []
    for entry in raw_candidates:
        result_path = Path(entry["result"]["path"]).resolve(strict=True)
        rows = _rows(result_path)
        if len(rows) == 1 and rows[0].get("id") == task_id:
            match.append((entry, result_path, rows[0]))
    if len(match) != 1:
        raise ValueError("exact source task must match one unique collector result in the closed snapshot")
    result_entry, raw_path, raw = match[0]
    if _binding(raw_path) != result_entry["result"]:
        raise ValueError("collector result bytes differ from closed snapshot binding")
    outcome = raw.get("outcome") or {}
    if (raw.get("version") != "natlang.teacher_trajectory.native/1"
            or outcome.get("status") != "done" or outcome.get("accepted") is not True
            or (outcome.get("oracle") or {}).get("accepted") is not True
            or (outcome.get("oracle") or {}).get("level") != "exact"
            or (raw.get("provenance") or {}).get("model") != "gpt-6-luna"
            or (raw.get("provenance") or {}).get("provider") != "openai-codex"
            or (raw.get("provenance") or {}).get("transport") != "pi-provider"):
        raise ValueError("native conversion only accepts the exact original Luna result")

    # Re-run the complete provider-source verifier for the snapshot (including
    # closed claim, collector, actor, outcome and current protection bindings).
    class Args:
        pass
    args = Args()
    args.provider_source_manifest = provider_source
    args.source = Path(declaration["source"]["path"])
    args.protection_manifest = protection_path
    binding = source_binding(args, protection_reader=current_protection)
    if raw.get("task", {}).get("program_ir", {}).get("id") not in binding["selected_ids"]:
        raise ValueError("selected exact result program is absent from authenticated source binding")

    cache_receipt = json.loads(decision_receipt_path.read_bytes())
    if (cache_receipt.get("schema") != "natlang.verified-teacher-decision-materialization/1"
            or cache_receipt.get("summary", {}).get("rejected_rows") != 0
            or cache_receipt.get("summary", {}).get("eligible_decisions", 0) < 1):
        raise ValueError("decision cache lacks a complete successful materialization receipt")
    row_sha = _runtime_row_digest(Path(plan["runtime"]), raw, cache_receipt)
    raw_trajectory = raw.get("trajectory") or []
    cached = [row for row in _rows(decisions_path)
              if (row.get("source_ref") or {}).get("trajectory_id") == task_id]
    if (not raw_trajectory or len(cached) != len(raw_trajectory)
            or [((row.get("decision") or {}).get("index")) for row in cached] != list(range(len(raw_trajectory)))):
        raise ValueError("decision cache does not cover the task's complete contiguous trajectory")
    converted = []
    for index, original in enumerate(cached):
        turn = raw_trajectory[index]
        source_ref = original.get("source_ref") or {}
        context = source_ref.get("provider_input_context") or {}
        if (source_ref.get("source_row_sha256") != row_sha
                or source_ref.get("invocation_id") != turn.get("invocation_id")
                or context.get("source_trajectory_index") != index
                or context.get("source_request_sha256") != turn.get("request_sha256")
                or context.get("source_response_sha256") != turn.get("raw_response_sha256")
                or sha256_hex(canonical_json_bytes(original.get("messages"))) != context.get("student_context_sha256")
                or original.get("target") is None or original.get("tools") != turn.get("tools_offered")):
            raise ValueError(f"decision cache row {index} differs from its exact provider turn")
        row = json.loads(json.dumps(original, ensure_ascii=False))
        row["native_source_conversion"] = {
            "source_messages": row["messages"],
            "source_messages_sha256": sha256_hex(canonical_json_bytes(row["messages"])),
            "source_target_present": True,
            "source_target": row["target"],
            "source_neuralese_conversion_present": False,
        }
        receipts = source_ref.get("provider_expanded_read_contexts") or []
        if receipts:
            row["neuralese_conversion"] = {"external_context_inputs": [{
                "schema": "natlang.external-context-input/1",
                "origin": "same-run-producer",
                "learner_representation": "typed-read-from-authenticated-runtime-writer-event-context-only",
                "block_id": (receipt.get("block") or {}).get("id"),
                "type": (receipt.get("block") or {}).get("type"),
                "body_sha256": (receipt.get("block") or {}).get("body_sha256"),
                "invocation_id": receipt.get("invocation_id"),
                "parent_invocation_id": receipt.get("parent_invocation_id"),
                "source_row_sha256": receipt.get("source_row_sha256"),
                "trace_sha256": receipt.get("trace_sha256"),
                "transport_provenance_sha256": receipt.get("transport_provenance_sha256"),
                "raw_request_sha256": receipt.get("raw_request_sha256"),
                "rendered_request_sha256": receipt.get("rendered_request_sha256"),
                "source_request_sha256": receipt.get("source_request_sha256"),
                "source_response_sha256": receipt.get("source_response_sha256"),
                "read_node": (receipt.get("block_read") or {}).get("node"),
                "model_turn_node": (receipt.get("model_turn") or {}).get("node"),
                "producer_write_node": (receipt.get("producer_write") or {}).get("node"),
                "producer_call_id": (receipt.get("producer_write") or {}).get("call_id"),
                "writer_source_class": receipt.get("writer_source_class"),
                "writer_witness": receipt.get("writer_witness"),
                "writer_target_selected": False,
                "learned_vectors": False,
                "qualification_certificate": False,
                "training_admission": False,
                "context_occurrences": receipt.get("context_occurrences"),
                "source_trajectory_index": receipt.get("source_trajectory_index"),
                "source_action_target_sha256": receipt.get("source_action_target_sha256"),
                "target_binding_adapter": receipt.get("target_binding_adapter"),
                "teacher_text": (receipt.get("block") or {}).get("teacher_text") is True,
                "format": (receipt.get("block") or {}).get("format"),
                "tokenizer_ids": (receipt.get("block") or {}).get("tokenizer_ids"),
                "additional_read_nodes": [x.get("block_read", {}).get("node") for x in
                    receipt.get("additional_read_turn_pairs") or []],
                "additional_model_turn_nodes": [x.get("model_turn", {}).get("node") for x in
                    receipt.get("additional_read_turn_pairs") or []],
            } for receipt in receipts]}
        converted.append(row)

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("exact native conversion output is immutable; choose a fresh directory")
    output_dir.mkdir(parents=True)
    raw_copy = output_dir / "raw-result.jsonl"
    raw_data = raw_path.read_bytes()
    raw_copy.write_bytes(raw_data)
    decisions_out = output_dir / "converted-decisions.jsonl"
    decision_data = b"".join(canonical_json_bytes(row) + b"\n" for row in converted)
    decisions_out.write_bytes(decision_data)
    repo_code = [
        Path(__file__).resolve(),
        ROOT / "scripts/neuralese_data/capture_luna_native_episode.py",
        ROOT / "training/neuralese/natlang_neuralese/data/provider_teacher_source.py",
        ROOT / "training/neuralese/natlang_neuralese/data/provider_native_graph.py",
        ROOT / "training/neuralese/natlang_neuralese/data/text_corpus.py",
        ROOT / "training/neuralese/natlang_neuralese/serve/teacher_source_receipt.py",
        ROOT / "training/neuralese/natlang_neuralese/serve/engine.py",
    ]
    audit = {
        "schema": SCHEMA,
        "task_id": task_id,
        "raw_results": _binding(raw_copy),
        "original_result": _binding(raw_path),
        "python_corpus_conversion": _binding(decisions_out),
        "plan": _binding(plan_path),
        "runtime": {"path": plan["runtime"],
                    "receipt_sha256": sha256_file_hex(Path(plan["runtime"]) / "frozen-runtime.json")},
        "provider_source_manifest": _binding(provider_source),
        "decision_cache": _binding(decisions_path),
        "decision_cache_receipt": _binding(decision_receipt_path),
        "decision_cache_output_sha256": cache_receipt["output_sha256"],
        "protection": _binding(protection_path),
        "conversion_builder": _binding(Path(__file__)),
        "source_code": [{"path": _binding(path)["path"], "sha256": _binding(path)["sha256"]}
                        for path in repo_code],
        "trajectory_count": len(raw_trajectory),
        "converted_decision_count": len(converted),
        "approved_decision_count": sum(bool((row.get("training_admission") or {}).get("approved")) for row in converted),
        "oracle_level": "exact",
        "exact_oracle_claimed": True,
        "training_admission": False,
        "current_protection_digest": current_protection(protection_path)["proof"],
    }
    audit_path = output_dir / "ingestion-audit.json"
    audit_data = canonical_json_bytes(audit) + b"\n"
    audit_path.write_bytes(audit_data)
    return {"audit": str(audit_path), "audit_sha256": sha256_hex(audit_data),
            "converted": str(decisions_out), "converted_sha256": sha256_hex(decision_data),
            "task_id": task_id, "turns": len(converted),
            "approved_decisions": audit["approved_decision_count"],
            "native_text_receipt_rows": sum(bool((row.get("source_ref") or {}).get(
                "provider_expanded_read_contexts")) for row in converted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("provider-source", "plan", "decisions", "decision-receipt", "protection", "task-id", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    result = build(provider_source=Path(args.provider_source), plan_path=Path(args.plan),
        decisions_path=Path(args.decisions), decision_receipt_path=Path(args.decision_receipt),
        protection_path=Path(args.protection), task_id=args.task_id, output_dir=Path(args.output_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
