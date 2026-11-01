#!/usr/bin/env python3
"""Verify and assemble a complete fsynced Luna native-capture episode.

This accepts both Neuralese-bearing episodes and ordinary multi-turn call trees.
It cannot launch a model request or grant training admission.
"""
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


SUPPORTED = {"natlang.luna-provider-native-ingestion-audit/1",
             "natlang.luna-provider-native-exact-ingestion-audit/1"}


def _read_json(path):
    return json.loads(Path(path).read_bytes())


def _same_source_receipt(actual: dict, expected: dict) -> bool:
    """Allow only a validation-copy path change when the source bytes match."""
    if actual.get("source_file_sha256") != expected.get("source_file_sha256"):
        return False
    normalized = dict(expected)
    normalized["source_path"] = actual.get("source_path")
    return actual == normalized


def assemble(audit_path: Path, source_capture_dir: Path, output_dir: Path,
             capture_audit_path: Path | None = None) -> dict:
    audit_path = Path(audit_path).resolve(strict=True)
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("schema") not in SUPPORTED:
        raise ValueError("assembly requires a current authenticated Luna source audit")
    source_capture_dir = Path(source_capture_dir).resolve(strict=True)
    capture_audit_path = (Path(capture_audit_path).resolve(strict=True)
                          if capture_audit_path is not None else audit_path)
    capture_audit_bytes = capture_audit_path.read_bytes()
    capture_audit = json.loads(capture_audit_bytes)
    if capture_audit.get("schema") != audit.get("schema"):
        raise ValueError("capture and validation audits use different source schemas")
    stable_fields = ("original_result", "plan", "runtime", "provider_source_manifest",
                     "decision_cache", "decision_cache_receipt", "decision_cache_output_sha256",
                     "protection", "task_id", "trajectory_count", "converted_decision_count",
                     "oracle_level", "exact_oracle_claimed", "training_admission")
    for key in stable_fields:
        if capture_audit.get(key) != audit.get(key):
            raise ValueError(f"capture and validation audits disagree about {key}")
    for key in ("raw_results", "python_corpus_conversion"):
        old_binding, new_binding = capture_audit.get(key) or {}, audit.get(key) or {}
        if (old_binding.get("sha256") != new_binding.get("sha256")
                or old_binding.get("bytes") != new_binding.get("bytes")):
            raise ValueError(f"capture and validation audits disagree about the bytes for {key}")
    captured_code = {item.get("path"): item.get("sha256") for item in capture_audit.get("source_code") or []}
    current_code = {item.get("path"): item.get("sha256") for item in audit.get("source_code") or []}
    if set(captured_code) != set(current_code):
        raise ValueError("capture and validation audits do not bind the same source-code files")
    changed_code = {path for path in captured_code if captured_code[path] != current_code[path]}
    allowed_builder = str(ROOT / "training/neuralese/natlang_neuralese/data/provider_native_graph.py")
    if changed_code - {allowed_builder}:
        raise ValueError("capture and validation code differ beyond the authenticated episode graph assembler")
    source_partial = source_capture_dir / "partial-turns.jsonl"
    partial_bytes = source_partial.read_bytes()
    events = [json.loads(line) for line in partial_bytes.splitlines() if line.strip()]
    completed = [event for event in events if event.get("event") == "turn_complete"]
    terminal = [event for event in events if event.get("event") == "error"]
    count = audit.get("trajectory_count")
    if (type(count) is not int or count < 1 or len(completed) != count
            or len(terminal) > 1 or (terminal and (events[-1] is not terminal[0]
                or terminal[0].get("source_index") is not None))):
        raise ValueError("source capture must contain the whole episode and only an optional terminal assembly error")
    if any(event.get("audit_sha256") != sha256_hex(capture_audit_bytes) for event in completed):
        raise ValueError("captured turns do not bind the exact audit used at generation time")

    raw_binding = audit["raw_results"]
    converted_binding = audit["python_corpus_conversion"]
    raw_path = Path(raw_binding["path"]).resolve(strict=True)
    converted_path = Path(converted_binding["path"]).resolve(strict=True)
    raw_bytes = raw_path.read_bytes()
    converted_bytes = converted_path.read_bytes()
    if (sha256_hex(raw_bytes) != raw_binding["sha256"]
            or sha256_hex(converted_bytes) != converted_binding["sha256"]):
        raise ValueError("audited raw task or materialized decision chain changed")
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    decisions = [json.loads(line) for line in converted_bytes.splitlines() if line.strip()]
    if len(raw_rows) != 1 or len(decisions) != count or raw_rows[0].get("id") != audit.get("task_id"):
        raise ValueError("audit does not describe one whole exact source episode")
    decisions.sort(key=lambda row: row["source_ref"]["provider_input_context"]["source_trajectory_index"])
    completed.sort(key=lambda row: row.get("source_index", -1))
    if [row.get("source_index") for row in completed] != list(range(count)):
        raise ValueError("captured turns are not the complete contiguous source episode")

    captures = []
    actor_identity = None
    for index, (partial, decision) in enumerate(zip(completed, decisions)):
        record_id = decision.get("id")
        if partial.get("record_id") != record_id:
            raise ValueError("captured source row differs from the audited decision order")
        loaded, receipt = partial.get("loaded"), partial.get("source_receipt")
        response, capture = partial.get("response"), partial.get("capture")
        if (not isinstance(loaded, dict) or not isinstance(receipt, dict)
                or not isinstance(response, dict) or not isinstance(capture, dict)
                or response.get("x_natlang_rl_capture") != capture
                or capture.get("teacher_source_receipt") != receipt):
            raise ValueError("captured response, source receipt, or execution graph is incomplete")
        source_opening = decision.get("native_source_conversion", {}).get("source_messages")
        if not isinstance(source_opening, list) or partial.get("source_opening") != source_opening:
            raise ValueError("captured prompt opening differs from its authenticated source messages")
        tools = decision.get("tools") or []
        current_loaded = load_verified_luna_teacher_turn(audit_path, record_id,
            replay_messages=decision["messages"], replay_tools=tools, allow_context_only=True)
        expected_receipt = dict(current_loaded["source_receipt"])
        if not _same_source_receipt(loaded.get("source_receipt") or {}, expected_receipt):
            raise ValueError("captured source receipt differs from the current verified receipt")
        expected_receipt["source_path"] = receipt.get("source_path")
        expected_receipt["prompt_sha256"] = sha256_hex(canonical_json_bytes({
            "messages": partial.get("rewritten_messages"), "tools": tools}))
        if (expected_receipt != receipt
                or current_loaded.get("native_text_spans") != loaded.get("native_text_spans")
                or current_loaded.get("typed_text_bodies") != loaded.get("typed_text_bodies")
                or current_loaded.get("record", {}).get("target") != loaded.get("record", {}).get("target")):
            raise ValueError("capture does not match the current authenticated turn and source result")
        bindings = partial.get("encoder_bindings") or []
        expected_messages = _replace_refs(decision["messages"],
            {item["teacher_block_id"]: item for item in bindings},
            {item["block_id"]: item["context_occurrences"]
             for item in current_loaded.get("typed_text_bodies") or []})
        if partial.get("rewritten_messages") != expected_messages:
            raise ValueError("captured prompt differs from the authenticated typed-reference rewrite")
        validate_provider_teacher_source(receipt, loaded["record"]["target"],
            partial["rewritten_messages"], loaded["record"].get("tools") or [],
            loaded.get("native_text_spans"), source_opening=source_opening,
            native_text_bindings=bindings)

        identity = partial.get("native_identity")
        runtime_identity = partial.get("actor_runtime_identity")
        serving = partial.get("serving_config")
        if (not isinstance(identity, dict) or not isinstance(runtime_identity, dict)
                or not isinstance(serving, dict)
                or partial.get("serving_config_identity_sha256") != serving.get("identity_sha256")
                or not partial.get("prompt_rendering_identity_sha256")):
            raise ValueError("captured turn lacks exact actor, prompt, runtime or serving identity")
        turn_identity = {"native_identity": identity, "runtime_identity": runtime_identity,
            "serving_config": serving,
            "serving_config_identity_sha256": partial["serving_config_identity_sha256"],
            "prompt_rendering_identity_sha256": partial["prompt_rendering_identity_sha256"]}
        if actor_identity is None:
            actor_identity = turn_identity
        elif actor_identity != turn_identity:
            raise ValueError("episode turns use different native actors or serving identities")

        states = []
        state_files = partial.get("encoder_state_files") or []
        if len(state_files) != len(bindings):
            raise ValueError("captured state files do not cover exact encoder bindings")
        for state_file, binding in zip(state_files, bindings):
            relative = Path(state_file.get("path", ""))
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
                raise ValueError("captured encoder state path is unsafe")
            state_path = (source_capture_dir / relative).resolve(strict=True)
            if source_capture_dir not in state_path.parents:
                raise ValueError("captured encoder state escapes its source directory")
            encoded = state_path.read_bytes()
            if sha256_hex(encoded) != state_file.get("sha256"):
                raise ValueError("captured encoder state changed after its turn was fsynced")
            block, state_id = decode_native_state(encoded)
            if state_id != binding.get("native_block_id") or native_state_id(block) != state_id:
                raise ValueError("captured encoder state differs from its content address")
            if (sha256_hex(block.text.encode("utf-8")) != binding.get("body_sha256")
                    or block.checkpoint_sha256 != identity.get("checkpoint_sha256")
                    or block.tokenizer_fingerprint != identity.get("tokenizer_fingerprint")):
                raise ValueError("captured encoder body or model identity differs from its exact binding")
            states.append(block)
        captures.append({"record": loaded["record"], "source_receipt": receipt,
            "source_opening": source_opening, "rewritten_messages": partial["rewritten_messages"],
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
        raise ValueError("assembled episode output must be new and empty")
    state_paths = []
    state_dirs = set()
    for partial in completed:
        for state_file in partial.get("encoder_state_files") or []:
            relative = Path(state_file["path"])
            source_path = (source_capture_dir / relative).resolve(strict=True)
            target_path = output_dir / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target_path)
            with target_path.open("rb") as stream:
                os.fsync(stream.fileno())
            state_dirs.add(target_path.parent)
            state_paths.append({**state_file, "path": str(relative)})
    partial_out = output_dir / "partial-turns.jsonl"
    partial_data = b"".join(canonical_json_bytes(row) + b"\n" for row in completed)
    partial_out.write_bytes(partial_data)
    graph_path = output_dir / "native-episode.jsonl"
    graph_bytes = canonical_json_bytes(graph) + b"\n"
    graph_path.write_bytes(graph_bytes)
    exact = audit.get("schema") == "natlang.luna-provider-native-exact-ingestion-audit/1"
    receipt = {"schema": "natlang.luna-native-episode-assembly/1",
        "assembly_mode": "verified-fsynced-turns", "audit_path": str(audit_path),
        "audit_sha256": sha256_hex(audit_bytes), "source_capture_dir": str(source_capture_dir),
        "capture_audit_path": str(capture_audit_path),
        "capture_audit_sha256": sha256_hex(capture_audit_bytes),
        "capture_to_validation_changed_code": sorted(changed_code),
        "source_capture_partial_sha256": sha256_hex(partial_bytes),
        "source_capture_completed_turns": len(completed),
        "source_capture_terminal_error": terminal[0].get("error") if terminal else None,
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
        "teacher_text_encoder_count": len(graph.get("teacher_text_encoders") or []),
        "encoder_state_files": state_paths,
        "native_checkpoint_sha256": actor_identity["native_identity"].get("checkpoint_sha256"),
        "actor_runtime_identity": actor_identity["runtime_identity"],
        "prompt_rendering_identity_sha256": actor_identity["prompt_rendering_identity_sha256"],
        "serving_config": actor_identity["serving_config"],
        "serving_config_identity_sha256": actor_identity["serving_config_identity_sha256"],
        "tokenizer_fingerprint": actor_identity["native_identity"].get("tokenizer_fingerprint"),
        "graph_path": str(graph_path), "graph_sha256": sha256_hex(graph_bytes),
        "native_graph_invocations": len(example.plan.invocations),
        "source_groups": graph["source_groups"], "oracle_level": "exact" if exact else "judged",
        "exact_oracle_claimed": exact, "training_admission": False,
        "graph_assembler_source_path": str(Path(__file__).resolve()),
        "graph_assembler_source_sha256": sha256_hex(Path(__file__).read_bytes()),
        "partial_turns_path": str(partial_out), "partial_turns_sha256": sha256_hex(partial_data),
        "partial_turn_count": len(completed), "partial_turns_complete": True}
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    for path in (partial_out, graph_path, receipt_path):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    for directory in sorted(state_dirs | {output_dir}):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--capture-audit")
    parser.add_argument("--source-capture", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.audit, args.source_capture, args.output,
                              capture_audit_path=args.capture_audit), indent=2))


if __name__ == "__main__":
    main()
