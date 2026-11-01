#!/usr/bin/env python3
"""Build a hash-bound native-capture input from one judged Luna source episode.

This conversion preserves the provider trace and normalized decisions. It does
not rerun the teacher, promote a judged oracle to exact, or claim admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training/neuralese"))
from natlang_neuralese.common.hashing import sha256_hex
from natlang_neuralese.common.jsonio import canonical_json_bytes


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line.strip()]


def bound(path):
    data = Path(path).resolve(strict=True).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": sha256_hex(data), "bytes": len(data)}, data


def build(raw_path, decisions_path, provider_source_path, plan_path, runtime_path,
          protection_path, output_dir):
    raw_binding, raw_bytes = bound(raw_path)
    decision_binding, decision_bytes = bound(decisions_path)
    source_binding, source_bytes = bound(provider_source_path)
    plan_binding, plan_bytes = bound(plan_path)
    protection_binding, protection_bytes = bound(protection_path)
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    if len(raw_rows) != 1:
        raise ValueError("native conversion requires one exact closed provider source row")
    raw = raw_rows[0]
    source_manifest = json.loads(source_bytes)
    source_artifact = source_manifest.get("source") or {}
    if (source_manifest.get("schema") != "natlang.provider-campaign-source/1"
            or source_manifest.get("training_admission") is not False
            or source_manifest.get("training_exposure") is not False
            or source_manifest.get("oracle_regrades") != []
            or source_artifact.get("path") != raw_binding["path"]
            or source_artifact.get("sha256") != raw_binding["sha256"]
            or source_artifact.get("bytes") != raw_binding["bytes"]
            or source_manifest.get("model") != "gpt-6-luna"
            or source_manifest.get("provider") != "openai-codex"):
        raise ValueError("closed provider-source manifest does not bind this exact Luna artifact")
    provenance = raw.get("provenance") or {}
    oracle = (raw.get("outcome") or {}).get("oracle") or {}
    if (raw.get("version") != "natlang.teacher_trajectory.native/1"
            or provenance.get("model") != "gpt-6-luna"
            or provenance.get("provider") != "openai-codex"
            or provenance.get("transport") != "pi-provider"
            or provenance.get("collection_role") != "teacher"
            or provenance.get("neuralese_server") is not False
            or (raw.get("outcome") or {}).get("status") != "done"
            or (raw.get("outcome") or {}).get("accepted") is not True
            or oracle.get("accepted") is not True or oracle.get("level") != "judged"):
        raise ValueError("source row is not an accepted actual Luna teacher result with a judged oracle")
    plan = json.loads(plan_bytes)
    if (plan.get("root_approved") is not True or plan.get("model_id") != "gpt-6-luna"
            or plan.get("provider") != "openai-codex"
            or plan.get("judge_model") != {"model_id": "gpt-6-sol", "provider": "openai-codex"}
            or plan.get("runtime_receipt_sha256") is None):
        raise ValueError("reviewed campaign plan does not bind Luna collection and independent Sol judgment")
    if (source_manifest.get("plan", {}).get("path") != plan_binding["path"]
            or source_manifest.get("plan", {}).get("sha256") != plan_binding["sha256"]):
        raise ValueError("closed provider snapshot is not bound to this exact reviewed plan")
    selected_path = plan.get("source")
    selected_digest = (plan.get("artifact_hashes") or {}).get(selected_path)
    selected_bytes = Path(selected_path).resolve(strict=True).read_bytes()
    selected_rows = [json.loads(line) for line in selected_bytes.splitlines() if line.strip()]
    if (not selected_digest or sha256_hex(selected_bytes) != selected_digest
            or len(selected_rows) != 1 or selected_rows[0] != (raw.get("task") or {}).get("program_ir")):
        raise ValueError("reviewed selected train program differs from the raw source task")
    judge_starts = [item for item in (raw.get("request_telemetry") or {}).get("starts") or []
                    if item.get("role") == "judge" and item.get("purpose") == "judge"]
    if len(judge_starts) != 1 or judge_starts[0].get("status") != "completed":
        raise ValueError("raw collector telemetry does not prove one completed independent Sol judgment")
    plan_protection_sha = (plan.get("artifact_hashes") or {}).get(protection_binding["path"])
    if plan_protection_sha != protection_binding["sha256"]:
        raise ValueError("protection projection is not the exact artifact reviewed by the plan")
    protection = json.loads(protection_bytes)
    if (protection.get("schema") != "natlang.luna-codebase-protection-preflight/1"
            or protection.get("selected_train_rows") != 1):
        raise ValueError("protection projection does not establish one selected train world")

    plan_runtime = Path(plan["runtime"]).resolve(strict=True)
    runtime_receipt_path = plan_runtime / "frozen-runtime.json"
    runtime_binding, _ = bound(runtime_receipt_path)
    if runtime_binding["sha256"] != plan["runtime_receipt_sha256"]:
        raise ValueError("runtime receipt differs from the reviewed plan")
    if (plan.get("artifact_hashes") or {}).get(str(runtime_receipt_path)) != runtime_binding["sha256"]:
        raise ValueError("reviewed plan does not hash-bind its frozen runtime receipt")
    if Path(runtime_path).resolve(strict=True) != plan_runtime:
        raise ValueError("runtime argument differs from the reviewed campaign runtime")

    decisions = [json.loads(line) for line in decision_bytes.splitlines() if line.strip()]
    trajectory = raw.get("trajectory") or []
    if not trajectory or len(decisions) != len(trajectory):
        raise ValueError("decision materialization must cover the complete source trajectory")
    converted, seen = [], set()
    for item in decisions:
        source_ref = item.get("source_ref") or {}
        context_ref = source_ref.get("provider_input_context") or {}
        index = context_ref.get("source_trajectory_index")
        if (type(index) is not int or not 0 <= index < len(trajectory) or index in seen
                or source_ref.get("trajectory_id") != raw.get("id")):
            raise ValueError("materialized decision does not bind one unique turn of the source episode")
        seen.add(index)
        messages = (trajectory[index] or {}).get("context")
        if (not isinstance(messages, list)
                or sha256_hex(canonical_json_bytes(messages)) != context_ref.get("student_context_sha256")):
            raise ValueError("materialized prompt differs from the exact captured provider context")
        row = json.loads(json.dumps(item, ensure_ascii=False))
        row["messages"] = messages
        row["native_source_conversion"] = {
            "source_messages": messages,
            "source_messages_sha256": sha256_hex(canonical_json_bytes(messages)),
            "source_target_present": True,
            "source_target": item.get("target"),
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
        else:
            row.pop("neuralese_conversion", None)
        converted.append(row)
    if seen != set(range(len(trajectory))):
        raise ValueError("decision conversion omits source turns")
    converted.sort(key=lambda row: row["source_ref"]["provider_input_context"]["source_trajectory_index"])

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_path = output_dir / "converted-decisions.jsonl"
    converted_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in converted)
    converted_path.write_bytes(converted_bytes)
    repo = Path(__file__).resolve().parents[2]
    code_paths = [
        Path(__file__).resolve(),
        repo / "scripts/neuralese_data/capture_luna_native_episode.py",
        repo / "training/neuralese/natlang_neuralese/data/provider_teacher_source.py",
        repo / "training/neuralese/natlang_neuralese/data/provider_native_graph.py",
        repo / "training/neuralese/natlang_neuralese/data/text_corpus.py",
        repo / "training/neuralese/natlang_neuralese/serve/teacher_source_receipt.py",
        repo / "training/neuralese/natlang_neuralese/serve/engine.py",
    ]
    audit = {
        "schema": "natlang.luna-provider-native-ingestion-audit/1",
        "task_id": raw["id"],
        "raw_results": {"path": raw_binding["path"], "sha256": raw_binding["sha256"]},
        "python_corpus_conversion": {"path": str(converted_path), "sha256": sha256_hex(converted_bytes)},
        "plan": {"path": plan_binding["path"], "sha256": plan_binding["sha256"]},
        "runtime": {"path": str(plan_runtime), "receipt_sha256": runtime_binding["sha256"]},
        "provider_source_manifest": {"path": source_binding["path"], "sha256": source_binding["sha256"]},
        "decision_cache": {"path": decision_binding["path"], "sha256": decision_binding["sha256"]},
        "protection": {"path": protection_binding["path"], "sha256": protection_binding["sha256"]},
        "conversion_builder": {"path": str(Path(__file__).resolve()), "sha256": sha256_hex(Path(__file__).read_bytes())},
        "source_code": [{"path": str(path.resolve()), "sha256": sha256_hex(path.read_bytes())} for path in code_paths],
        "trajectory_count": len(trajectory),
        "converted_decision_count": len(converted),
        "approved_decision_count": sum(bool((row.get("training_admission") or {}).get("approved")) for row in converted),
        "oracle_level": "judged",
        "oracle_judge": plan["judge_model"],
        "exact_oracle_claimed": False,
        "training_admission": False,
    }
    audit_path = output_dir / "ingestion-audit.json"
    audit_bytes = canonical_json_bytes(audit) + b"\n"
    audit_path.write_bytes(audit_bytes)
    print(json.dumps({"audit": str(audit_path), "audit_sha256": sha256_hex(audit_bytes),
        "converted": str(converted_path), "converted_sha256": sha256_hex(converted_bytes),
        "turns": len(trajectory), "approved_decisions": audit["approved_decision_count"]}, indent=2))


def main():
    from scripts.neuralese_data.provider_execution_runtime import require_frozen_provider_runtime
    require_frozen_provider_runtime(Path(__file__).resolve().parents[2])
    parser = argparse.ArgumentParser()
    for name in ("raw", "decisions", "provider-source", "plan", "runtime", "protection", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    build(args.raw, args.decisions, args.provider_source, args.plan, args.runtime, args.protection, args.output_dir)


if __name__ == "__main__":
    main()
