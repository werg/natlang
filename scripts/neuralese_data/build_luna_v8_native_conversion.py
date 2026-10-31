#!/usr/bin/env python3
"""Build a hash-bound converted view of the v8 Luna LogHub episode.

This is a source-preserving bridge from the closed provider snapshot and its
decision cache into the native teacher-turn loader. It does not rewrite the
raw Luna trajectory or promote its judged oracle to exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def external_input_metadata(record, receipt):
    block = receipt.get("block") or {}
    read = receipt.get("block_read") or {}
    turn = receipt.get("model_turn") or {}
    write = receipt.get("producer_write") or {}
    pairs = receipt.get("additional_read_turn_pairs") or []
    return {"schema": "natlang.external-context-input/1", "origin": "same-run-producer",
        "learner_representation": "typed-read-from-authenticated-runtime-writer-event-context-only",
        "block_id": block.get("id"), "type": block.get("type"), "body_sha256": block.get("body_sha256"),
        "invocation_id": receipt.get("invocation_id"), "parent_invocation_id": receipt.get("parent_invocation_id"),
        "source_row_sha256": receipt.get("source_row_sha256"), "trace_sha256": receipt.get("trace_sha256"),
        "transport_provenance_sha256": receipt.get("transport_provenance_sha256"),
        "raw_request_sha256": receipt.get("raw_request_sha256"),
        "rendered_request_sha256": receipt.get("rendered_request_sha256"),
        "source_request_sha256": receipt.get("source_request_sha256"),
        "source_response_sha256": receipt.get("source_response_sha256"),
        "read_node": read.get("node"), "model_turn_node": turn.get("node"),
        "producer_write_node": write.get("node"), "producer_call_id": write.get("call_id"),
        "writer_source_class": receipt.get("writer_source_class"),
        "writer_witness": receipt.get("writer_witness"), "writer_target_selected": False,
        "learned_vectors": False, "qualification_certificate": False, "training_admission": False,
        "context_occurrences": receipt.get("context_occurrences"),
        "source_trajectory_index": receipt.get("source_trajectory_index"),
        "source_action_target_sha256": receipt.get("source_action_target_sha256"),
        "target_binding_adapter": receipt.get("target_binding_adapter"),
        "teacher_text": block.get("teacher_text") is True, "format": block.get("format"),
        "tokenizer_ids": block.get("tokenizer_ids"),
        "additional_read_nodes": [item.get("block_read", {}).get("node") for item in pairs],
        "additional_model_turn_nodes": [item.get("model_turn", {}).get("node") for item in pairs]}


def build(raw_path, decision_path, package_source_path, plan_path, runtime_path, output_dir):
    raw_bytes = Path(raw_path).read_bytes()
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    if len(raw_rows) != 1:
        raise ValueError("v8 conversion requires the one exact closed provider row")
    raw = raw_rows[0]
    decisions = read_rows(decision_path)
    if len(decisions) != 33:
        raise ValueError("v8 conversion requires all 33 materialized action decisions")
    seen = set()
    converted = []
    for record in decisions:
        source_ref = record.get("source_ref") or {}
        provider = source_ref.get("provider_input_context") or {}
        index = provider.get("source_trajectory_index")
        if type(index) is not int or not 0 <= index < len(raw.get("trajectory") or []) or index in seen:
            raise ValueError("decision cache trajectory indices are absent, duplicate, or outside the source trace")
        seen.add(index)
        turn = raw["trajectory"][index]
        messages = turn.get("context")
        if not isinstance(messages, list) or sha(canonical(messages)) != provider.get("student_context_sha256"):
            raise ValueError("source turn context differs from the captured student prompt digest")
        item = json.loads(json.dumps(record, ensure_ascii=False))
        item["messages"] = messages
        item["native_source_conversion"] = {"source_messages": messages,
            "source_messages_sha256": sha(canonical(messages)), "source_target_present": True,
            "source_target": record.get("target"), "source_neuralese_conversion_present": False}
        receipts = source_ref.get("provider_expanded_read_contexts") or []
        if receipts:
            item["neuralese_conversion"] = {"external_context_inputs": [
                external_input_metadata(item, receipt) for receipt in receipts]}
        else:
            item.pop("neuralese_conversion", None)
        converted.append(item)
    if seen != set(range(33)):
        raise ValueError("converted view does not cover all 33 source trajectory turns")
    converted.sort(key=lambda item: item["source_ref"]["provider_input_context"]["source_trajectory_index"])

    source_meta = read_json(package_source_path)
    regrade = (source_meta.get("oracle_regrades") or [])[0]
    plan = read_json(plan_path)
    repo_root = Path(__file__).resolve().parents[2]
    source_code_files = [
        "training/neuralese/natlang_neuralese/data/provider_teacher_source.py",
        "training/neuralese/natlang_neuralese/data/provider_native_graph.py",
        "training/neuralese/natlang_neuralese/data/text_corpus.py",
        "training/neuralese/natlang_neuralese/serve/teacher_source_receipt.py",
        "training/neuralese/natlang_neuralese/serve/engine.py",
        "scripts/neuralese_data/capture_luna_v8_native_episode.py",
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converted_path = output_dir / "converted-decisions.jsonl"
    data = b"".join(canonical(item) + b"\n" for item in converted)
    converted_path.write_bytes(data)
    audit = {"schema": "natlang.luna-v8-provider-native-ingestion-audit/1",
        "raw_results": {"path": str(Path(raw_path).resolve()), "sha256": sha(raw_bytes)},
        "python_corpus_conversion": {"path": str(converted_path.resolve()), "sha256": sha(data)},
        "plan": {"path": str(Path(plan_path).resolve()), "sha256": sha(Path(plan_path).read_bytes())},
        "runtime": {"path": str(Path(runtime_path).resolve()),
            "receipt_sha256": sha((Path(runtime_path) / "frozen-runtime.json").read_bytes())},
        "provider_source_manifest": {"path": str(Path(package_source_path).resolve()),
            "sha256": sha(Path(package_source_path).read_bytes())},
        "source_regrade": {"path": regrade["projection"]["path"],
            "sha256": regrade["projection"]["sha256"],
            "sol_qualification_path": regrade["proof"]["path"],
            "sol_qualification_sha256": regrade["proof"]["sha256"]},
        "decision_cache": {"path": str(Path(decision_path).resolve()),
            "sha256": sha(Path(decision_path).read_bytes())},
        "conversion_builder": {"path": str(Path(__file__).resolve()),
            "sha256": sha(Path(__file__).read_bytes())},
        "source_code": [{"path": str((repo_root / path).resolve()),
            "sha256": sha((repo_root / path).read_bytes())} for path in source_code_files],
        "task_id": raw["id"], "trajectory_count": len(raw["trajectory"]),
        "converted_decision_count": len(converted),
        "approved_decision_count": sum(bool((item.get("training_admission") or {}).get("approved"))
                                         for item in converted),
        "oracle_level": "judged", "exact_oracle_claimed": False}
    audit_path = output_dir / "ingestion-audit.json"
    audit_path.write_bytes(canonical(audit) + b"\n")
    print(json.dumps({"audit": str(audit_path), "converted": str(converted_path),
        "audit_sha256": sha(audit_path.read_bytes()), "converted_sha256": sha(data),
        "trajectory_count": len(raw["trajectory"]), "converted_decisions": len(converted),
        "approved_decisions": audit["approved_decision_count"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--provider-source", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    build(args.raw, args.decisions, args.provider_source, args.plan, args.runtime, args.output_dir)


if __name__ == "__main__":
    main()
