#!/usr/bin/env python3
"""Assemble a verified native episode graph from a complete fsynced turn capture."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/neuralese"))
from natlang_neuralese.common.hashing import sha256_hex
from natlang_neuralese.common.jsonio import canonical_json_bytes
from natlang_neuralese.data.provider_native_graph import native_graph_from_luna_episode, _replace_refs
from natlang_neuralese.data.provider_teacher_source import load_verified_luna_teacher_turn
from natlang_neuralese.serve.store import decode_native_state, native_state_id
from natlang_neuralese.serve.teacher_source_receipt import validate_provider_teacher_source


def _read_json(path):
    return json.loads(Path(path).read_bytes())


def assemble(audit_path, capture_audit_path, source_capture_dir, output_dir):
    audit_path = Path(audit_path).resolve(strict=True)
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("schema") != "natlang.luna-v8-provider-native-ingestion-audit/1":
        raise ValueError("assembly requires the exact v8 Luna source audit")
    capture_audit_path = Path(capture_audit_path).resolve(strict=True)
    capture_audit_bytes = capture_audit_path.read_bytes()
    capture_audit = json.loads(capture_audit_bytes)
    if capture_audit.get("schema") != audit.get("schema"):
        raise ValueError("capture and validation audits use different source schemas")
    source_capture_dir = Path(source_capture_dir).resolve(strict=True)
    source_partial = source_capture_dir / "partial-turns.jsonl"
    partial_bytes = source_partial.read_bytes()
    source_lines = [line for line in partial_bytes.splitlines() if line.strip()]
    source_events = [json.loads(line) for line in source_lines]
    completed = [row for row in source_events if row.get("event") == "turn_complete"]
    terminal = [row for row in source_events if row.get("event") == "error"]
    count = audit.get("trajectory_count")
    if (type(count) is not int or count != 33 or len(completed) != count
            or len(terminal) != 1 or source_events[-1] is not terminal[0]
            or terminal[0].get("source_index") is not None):
        raise ValueError("source capture must contain all 33 completed turns and only a terminal assembly error")
    if any(row.get("audit_sha256") != sha256_hex(capture_audit_bytes) for row in completed):
        raise ValueError("captured turns do not bind the supplied capture audit")
    for key in ("raw_results", "plan", "runtime", "provider_source_manifest", "source_regrade", "decision_cache"):
        if capture_audit.get(key) != audit.get(key):
            raise ValueError(f"capture and validation audits disagree about {key}")
    if (capture_audit.get("python_corpus_conversion", {}).get("sha256")
            != audit.get("python_corpus_conversion", {}).get("sha256")
            or capture_audit.get("task_id") != audit.get("task_id")
            or capture_audit.get("trajectory_count") != audit.get("trajectory_count")):
        raise ValueError("capture and validation audits do not bind the same converted episode")

    raw_path = Path(audit["raw_results"]["path"]).resolve(strict=True)
    converted_path = Path(audit["python_corpus_conversion"]["path"]).resolve(strict=True)
    raw_bytes = raw_path.read_bytes()
    converted_bytes = converted_path.read_bytes()
    if (sha256_hex(raw_bytes) != audit["raw_results"]["sha256"]
            or sha256_hex(converted_bytes) != audit["python_corpus_conversion"]["sha256"]):
        raise ValueError("source trajectory or converted decisions changed after audit")
    decisions = [json.loads(line) for line in converted_bytes.splitlines() if line.strip()]
    if len(decisions) != count:
        raise ValueError("audited converted decisions do not contain all episode turns")
    decisions.sort(key=lambda row: row["source_ref"]["provider_input_context"]["source_trajectory_index"])
    completed.sort(key=lambda row: row.get("source_index", -1))
    if [row.get("source_index") for row in completed] != list(range(count)):
        raise ValueError("captured turn indices are not the exact contiguous source episode")

    captures = []
    actor_identity = None
    for index, (partial, decision) in enumerate(zip(completed, decisions)):
        record_id = decision.get("id")
        if partial.get("record_id") != record_id:
            raise ValueError("captured source row identity differs from audited decision order")
        loaded = partial.get("loaded")
        receipt = partial.get("source_receipt")
        response = partial.get("response")
        capture = partial.get("capture")
        if (not isinstance(loaded, dict) or not isinstance(receipt, dict)
                or not isinstance(response, dict) or not isinstance(capture, dict)
                or response.get("x_natlang_rl_capture") != capture
                or capture.get("teacher_source_receipt") != receipt):
            raise ValueError("captured response, native execution graph, and source receipt disagree")
        if (loaded.get("record", {}).get("id") != record_id
                or loaded.get("source_receipt", {}).get("source_row_id") != receipt.get("source_row_id")
                or receipt.get("source_turn_index") != index
                or receipt.get("source_row_id") != audit.get("task_id")):
            raise ValueError("captured source receipt is not bound to its exact audited trajectory turn")
        expected_context = decision.get("native_source_conversion", {}).get("source_messages")
        if (not isinstance(expected_context, list)
                or partial.get("source_opening") != expected_context
                or partial.get("loaded", {}).get("record", {}).get("id") != record_id):
            raise ValueError("captured turn changed its audited source prompt or record")
        current_loaded = load_verified_luna_teacher_turn(audit_path, record_id,
            replay_messages=decision["messages"], replay_tools=decision.get("tools"),
            allow_context_only=True)
        expected_receipt = dict(current_loaded["source_receipt"])
        expected_receipt["prompt_sha256"] = sha256_hex(canonical_json_bytes({
            "messages": partial["rewritten_messages"], "tools": decision.get("tools") or []}))
        if (expected_receipt != receipt
                or current_loaded.get("native_text_spans") != loaded.get("native_text_spans")
                or current_loaded.get("typed_text_bodies") != loaded.get("typed_text_bodies")
                or current_loaded.get("record", {}).get("target") != loaded.get("record", {}).get("target")):
            raise ValueError("captured source receipt or typed target differs from current authenticated source")
        binding_map = {item["teacher_block_id"]: item for item in partial.get("encoder_bindings") or []}
        expected_messages = _replace_refs(decision["messages"], binding_map,
            {item["block_id"]: item["context_occurrences"]
             for item in current_loaded.get("typed_text_bodies") or []})
        if partial["rewritten_messages"] != expected_messages:
            raise ValueError("captured prompt differs from the exact authenticated text-reference rewrite")
        validate_provider_teacher_source(receipt, loaded["record"]["target"],
            partial["rewritten_messages"], loaded["record"].get("tools") or [],
            loaded.get("native_text_spans"), source_opening=partial["source_opening"],
            native_text_bindings=partial.get("encoder_bindings") or [])
        identity = partial.get("native_identity")
        runtime_identity = partial.get("actor_runtime_identity")
        serving = partial.get("serving_config")
        if (not isinstance(identity, dict) or not isinstance(runtime_identity, dict)
                or not isinstance(serving, dict)
                or partial.get("serving_config_identity_sha256") != serving.get("identity_sha256")
                or partial.get("prompt_rendering_identity_sha256") is None):
            raise ValueError("turn is missing its actual actor, prompt, runtime, or serving identity")
        turn_identity = {"native_identity": identity, "runtime_identity": runtime_identity,
            "serving_config": serving,
            "serving_config_identity_sha256": partial["serving_config_identity_sha256"],
            "prompt_rendering_identity_sha256": partial["prompt_rendering_identity_sha256"]}
        if actor_identity is None:
            actor_identity = turn_identity
        elif actor_identity != turn_identity:
            raise ValueError("captured turns use different actor, prompt, runtime, or serving identities")

        states = []
        state_files = partial.get("encoder_state_files") or []
        bindings = partial.get("encoder_bindings") or []
        if len(state_files) != len(bindings):
            raise ValueError("captured encoder state files do not cover exact encoder bindings")
        for state_file, binding in zip(state_files, bindings):
            relative = Path(state_file.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("captured encoder state path is unsafe")
            state_path = (source_capture_dir / relative).resolve(strict=True)
            if source_capture_dir not in state_path.parents:
                raise ValueError("captured encoder state escapes its source directory")
            encoded = state_path.read_bytes()
            if sha256_hex(encoded) != state_file.get("sha256"):
                raise ValueError("captured encoder state bytes changed after their turn was fsynced")
            block, state_id = decode_native_state(encoded)
            if state_id != binding.get("native_block_id") or native_state_id(block) != state_id:
                raise ValueError("captured encoder state differs from its authenticated content address")
            states.append(block)
        captures.append({"record": loaded["record"], "source_receipt": receipt,
            "source_opening": partial["source_opening"],
            "rewritten_messages": partial["rewritten_messages"],
            "encoder_binding": bindings[0] if bindings else None,
            "encoder_state": states[0] if states else None,
            "encoder_bindings": bindings, "encoder_states": states,
            "encoder_occurrence_counts": partial.get("encoder_occurrence_counts") or {},
            "capture": capture, "response": response, "loaded": loaded,
            "native_identity": identity})

    graph, example = native_graph_from_luna_episode(captures)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError("assembled episode directory must be empty; failed directories are immutable")
    state_dir = output_dir / "encoder-states"
    state_dir.mkdir()
    for row in completed:
        for state_file in row.get("encoder_state_files") or []:
            relative = Path(state_file["path"])
            source = (source_capture_dir / relative).resolve(strict=True)
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(source, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
    partial_path = output_dir / "partial-turns.jsonl"
    partial_output = b"".join(canonical_json_bytes(row) + b"\n" for row in completed)
    with partial_path.open("xb") as stream:
        stream.write(partial_output)
        stream.flush()
        os.fsync(stream.fileno())
    graph_path = output_dir / "native-episode.jsonl"
    graph_bytes = canonical_json_bytes(graph) + b"\n"
    with graph_path.open("xb") as stream:
        stream.write(graph_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema": "natlang.luna-v8-native-episode-capture/1",
        "assembly_mode": "verified-fsynced-turn-prefix",
        "validation_audit_path": str(audit_path), "validation_audit_sha256": sha256_hex(audit_bytes),
        "capture_audit_path": str(capture_audit_path),
        "capture_audit_sha256": sha256_hex(capture_audit_bytes),
        "source_capture_path": str(source_capture_dir), "source_capture_partial_sha256": sha256_hex(partial_bytes),
        "source_capture_completed_turns": len(completed),
        "source_capture_terminal_error": terminal[0].get("error"),
        "raw_source_path": str(raw_path), "raw_source_sha256": sha256_hex(raw_bytes),
        "converted_decisions_path": str(converted_path),
        "converted_decisions_sha256": sha256_hex(converted_bytes),
        "source_turn_count": len(captures),
        "target_supervised_turns": sum(bool(item["source_receipt"]["target_supervision"]) for item in captures),
        "context_only_turns": sum(not item["source_receipt"]["target_supervision"] for item in captures),
        "supervised_tokens": sum(weight > 0 for weight in example.target_weights),
        "inline_producer_count": len(graph.get("teacher_text_inline_producers") or []),
        "consumer_action_count": sum(bool(item.get("consumer_native_block_ids"))
                                      for item in graph.get("teacher_episode_actions") or []),
        "native_checkpoint_sha256": actor_identity["native_identity"].get("checkpoint_sha256"),
        "actor_runtime_identity": actor_identity["runtime_identity"],
        "prompt_rendering_identity_sha256": actor_identity["prompt_rendering_identity_sha256"],
        "serving_config": actor_identity["serving_config"],
        "serving_config_identity_sha256": actor_identity["serving_config_identity_sha256"],
        "tokenizer_fingerprint": actor_identity["native_identity"].get("tokenizer_fingerprint"),
        "graph_path": str(graph_path), "graph_sha256": sha256_hex(graph_bytes),
        "native_graph_invocations": len(example.plan.invocations),
        "source_groups": graph["source_groups"], "oracle_level": "judged",
        "exact_oracle_claimed": False, "training_admission": False,
        "graph_assembler_source_path": str(Path(__file__).resolve()),
        "graph_assembler_source_sha256": sha256_hex(Path(__file__).read_bytes()),
        "partial_turns_path": str(partial_path), "partial_turns_sha256": sha256_hex(partial_output),
        "partial_turn_count": len(completed), "partial_turns_complete": True}
    receipt_path = output_dir / "receipt.json"
    with receipt_path.open("xb") as stream:
        stream.write(canonical_json_bytes(receipt) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(json.dumps({"receipt": str(receipt_path), **receipt}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--capture-audit", required=True)
    parser.add_argument("--source-capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assemble(args.audit, args.capture_audit, args.source_capture, args.output)


if __name__ == "__main__":
    main()
