#!/usr/bin/env python3
"""Run a source-bound, train-only history-reasoning profile comparison.

This is an evaluation runner only. It does not change a checkpoint, recipe,
training source, or admission state. Existing train-admission receipts bind
each selected target before the same 48 cases run under both profiles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


SCHEMA = "natlang.history-reasoning-live-window/1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_bound_file(reference: dict, description: str, cache: dict[Path, str]) -> Path:
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str) or not isinstance(reference.get("sha256"), str):
        raise ValueError(f"{description} lacks a path and SHA-256 binding")
    path = Path(reference["path"]).resolve(strict=True)
    actual = cache.get(path)
    if actual is None:
        actual = _sha256(path)
        cache[path] = actual
    if actual != reference["sha256"]:
        raise ValueError(f"{description} changed: {path}")
    return path


def _canonical_sha(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_output_binding(proof_output: dict, receipt_output: dict, *, description: str) -> None:
    if not isinstance(proof_output, dict) or not isinstance(receipt_output, dict):
        raise ValueError(f"{description} is missing from the validation proof or conversion receipt")
    fields = ("path", "sha256", "bytes", "rows", "tokens", "max_row_tokens")
    if any(proof_output.get(field) != receipt_output.get(field) for field in fields):
        raise ValueError(f"{description} proof/receipt bindings disagree")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def prepare_case_packet(proof_paths: list[Path], exclusion_path: Path, output_path: Path, *,
                        exclusion_sha256: str, current_protection_sha256: str,
                        expected_count: int = 48) -> dict:
    """Select distinct train programs whose exact target is admitted for both students."""
    import io
    import zstandard

    exclusion_path = exclusion_path.resolve(strict=True)
    if _sha256(exclusion_path) != exclusion_sha256:
        raise ValueError("the current SST context exclusion proof changed")
    excluded = {entry["program_id"] for entry in json.loads(exclusion_path.read_text())
                .get("excluded_programs", [])}
    candidates: dict[str, dict] = {}
    proof_bindings = []
    artifact_hashes: dict[Path, str] = {}
    for proof_path in proof_paths:
        proof_path = proof_path.resolve(strict=True)
        proof_bytes = proof_path.read_bytes()
        proof_sha = hashlib.sha256(proof_bytes).hexdigest()
        proof = json.loads(proof_bytes)
        if proof.get("status") != "passed" or proof.get("training_admission") is not True or proof.get("exposure") != 0:
            continue
        if proof.get("current_protection", {}).get("sha256") != current_protection_sha256:
            continue
        _verify_bound_file(proof["current_protection"], "current protection manifest", artifact_hashes)
        _verify_bound_file(proof["current_global_source_closure"], "current global source closure", artifact_hashes)
        source_ref = proof["source"]
        source_path = Path(source_ref["path"]).resolve(strict=True)
        if _sha256(source_path) != source_ref["sha256"]:
            raise ValueError(f"source artifact changed: {source_path}")
        raw_rows = {}
        with source_path.open("rb") as source_stream:
            for raw_line in source_stream:
                if not raw_line.strip():
                    continue
                source_row = json.loads(raw_line)
                raw_rows[source_row["id"]] = (hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest(), source_row)
        output_bindings = {}
        for model_name in ("lfm25", "mellum21"):
            model_proof = proof["models"][model_name]
            receipt_ref = model_proof["receipt"]
            receipt_path = Path(receipt_ref["path"]).resolve(strict=True)
            if _sha256(receipt_path) != receipt_ref["sha256"]:
                raise ValueError(f"{model_name} receipt changed: {receipt_path}")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "passed" or receipt.get("source", {}).get("sha256") != source_ref["sha256"]:
                raise ValueError(f"{model_name} receipt is not bound to the validated source")
            output_ref = model_proof["outputs"]["train"]
            graph_path = Path(output_ref["path"]).resolve(strict=True)
            if _sha256(graph_path) != output_ref["sha256"]:
                raise ValueError(f"{model_name} train graph output changed: {graph_path}")
            row_bindings = {item["id"]: item for item in receipt["source"]["row_bindings"]}
            allowed = set()
            with graph_path.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as reader, io.TextIOWrapper(reader, encoding="utf-8") as graph_stream:
                for line in graph_stream:
                    graph = json.loads(line)
                    if graph.get("split") != "train":
                        continue
                    source = (graph.get("conversion") or {}).get("teacher_source") or {}
                    if source.get("decision_admission", {}).get("approved") is True:
                        allowed.add((source.get("source_row_id"), source.get("teacher_turn_index"),
                                     source.get("raw_response_sha256"), source.get("source_program_id"),
                                     source.get("source_row_sha256")))
            output_bindings[model_name] = (row_bindings, allowed)
        for row_id, (actual_row_sha, raw_row) in raw_rows.items():
            binding = output_bindings["lfm25"][0].get(row_id)
            if binding is None or binding.get("sha256") != actual_row_sha:
                continue
            program_id = binding.get("program_id")
            if not isinstance(program_id, str) or program_id in excluded:
                continue
            # A candidate is one source-authenticated teacher turn which both
            # existing model views admitted. Keep its full earlier trajectory.
            for turn_index, turn in enumerate(raw_row.get("trajectory") or []):
                assistant = turn.get("assistant") or {}
                context = turn.get("context") or []
                raw_response_sha = turn.get("raw_response_sha256")
                key = (row_id, turn_index, raw_response_sha, program_id, actual_row_sha)
                if key not in output_bindings["lfm25"][1] or key not in output_bindings["mellum21"][1]:
                    continue
                if raw_row.get("outcome", {}).get("accepted") is not True or raw_row.get("outcome", {}).get("status") != "done":
                    continue
                if not any(message.get("role") == "assistant" and
                           isinstance(message.get("reasoning_content"), str) and
                           message["reasoning_content"].strip() for message in context):
                    continue
                response = turn.get("model_response") or {}
                if not (assistant.get("reasoning") or assistant.get("content") or response.get("raw_calls")):
                    continue
                candidate = {"case_id": _canonical_sha([program_id, row_id, turn_index])[:24],
                    "task_id": program_id, "program_id": program_id, "source_row_id": row_id,
                    "source_row_sha256": actual_row_sha, "trajectory_index": turn_index,
                    "raw_response_sha256": raw_response_sha, "validation_proof_path": str(proof_path),
                    "validation_proof_sha256": proof_sha}
                existing = candidates.get(program_id)
                if existing is None or turn_index > existing["trajectory_index"]:
                    candidates[program_id] = candidate
        proof_bindings.append({"path": str(proof_path), "sha256": proof_sha,
            "source_path": str(source_path), "source_sha256": source_ref["sha256"]})
    selected = [candidates[key] for key in sorted(candidates)]
    if len(selected) < expected_count:
        raise ValueError(f"only {len(selected)} distinct safe admitted programs; need {expected_count}")
    selected = selected[:expected_count]
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing case packet: {output_path}")
    with output_path.open("x", encoding="utf-8") as stream:
        for case in selected:
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {"schema": "natlang.history-reasoning-live-case-packet/1",
        "case_packet": {"path": str(output_path), "sha256": _sha256(output_path), "bytes": output_path.stat().st_size},
        "cases": len(selected), "unique_programs": len({case["program_id"] for case in selected}),
        "current_protection_sha256": current_protection_sha256,
        "exclusion_proof": {"path": str(exclusion_path), "sha256": exclusion_sha256},
        "validation_proofs": proof_bindings,
        "selected_program_ids_sha256": _canonical_sha(sorted(case["program_id"] for case in selected)),
        "training_admission": False}
    _atomic_json(output_path.with_name("case-packet-receipt.json"), receipt)
    return receipt


def validate_case_packet(case_path: Path, exclusion_path: Path, *, exclusion_sha256: str,
                         current_protection_sha256: str, expected_count: int = 48) -> tuple[list[dict], dict]:
    """Resolve cases directly through existing conversion/admission receipts."""
    case_path = case_path.resolve(strict=True)
    exclusion_path = exclusion_path.resolve(strict=True)
    if _sha256(exclusion_path) != exclusion_sha256:
        raise ValueError("the admitted SST context exclusion proof changed")
    exclusions_doc = json.loads(exclusion_path.read_text(encoding="utf-8"))
    excluded_programs = {item["program_id"] for item in exclusions_doc.get("excluded_programs", [])}
    rows = _load_jsonl(case_path)
    if len(rows) != expected_count:
        raise ValueError(f"expected exactly {expected_count} cases, received {len(rows)}")
    source_cache: dict[Path, dict] = {}
    verified_proofs: dict[Path, dict] = {}
    verified_artifact_hashes: dict[Path, str] = {}
    verified_graphs: dict[Path, set[tuple[str, int, str, str, str]]] = {}
    seen = set()
    for index, row in enumerate(rows, 1):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"case row {index} has a missing or duplicate case_id")
        if row.get("task_id") != row.get("program_id") or not isinstance(row.get("program_id"), str):
            raise ValueError(f"case {case_id} task/program identity is inconsistent")
        seen.add(case_id)
        proof_path = Path(row["validation_proof_path"]).resolve(strict=True)
        proof_sha = row["validation_proof_sha256"]
        if proof_path not in verified_proofs:
            if _sha256(proof_path) != proof_sha:
                raise ValueError(f"case {case_id} validation-proof bytes changed")
            proof = json.loads(proof_path.read_text(encoding="utf-8"))
            if proof.get("status") != "passed" or proof.get("training_admission") is not True:
                raise ValueError(f"case {case_id} lacks passing native training admission")
            if proof.get("exposure") != 0:
                raise ValueError(f"case {case_id} source proof is not an unexposed train source")
            if proof.get("current_protection", {}).get("sha256") != current_protection_sha256:
                raise ValueError(f"case {case_id} was not validated against the requested current protection")
            _verify_bound_file(proof["current_protection"], "current protection manifest", verified_artifact_hashes)
            _verify_bound_file(proof["current_global_source_closure"], "current global source closure",
                               verified_artifact_hashes)
            verified_proofs[proof_path] = proof
        proof = verified_proofs[proof_path]
        model_receipts = {}
        for model_name in ("lfm25", "mellum21"):
            model_proof = proof["models"][model_name]
            receipt_ref = model_proof["receipt"]
            receipt_path = Path(receipt_ref["path"]).resolve(strict=True)
            if _sha256(receipt_path) != receipt_ref["sha256"]:
                raise ValueError(f"case {case_id} {model_name} receipt changed")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "passed":
                raise ValueError(f"case {case_id} {model_name} conversion receipt is not admitted")
            if receipt.get("source", {}).get("sha256") != proof.get("source", {}).get("sha256"):
                raise ValueError(f"case {case_id} {model_name} receipt differs from the validation source")
            for split in ("train", "test"):
                proof_output = model_proof.get("outputs", {}).get(split)
                receipt_output = receipt.get("outputs", {}).get(split)
                _assert_output_binding(proof_output, receipt_output,
                                       description=f"case {case_id} {model_name} {split} output")
            model_receipts[model_name] = (model_proof, receipt, receipt_path)
        source_ref = proof.get("source") or {}
        source_path = Path(source_ref.get("path", "")).resolve(strict=True)
        if source_path not in source_cache:
            if _sha256(source_path) != source_ref.get("sha256"):
                raise ValueError(f"case {case_id} source artifact changed")
            source_cache[source_path] = {}
            with source_path.open("rb") as stream:
                for raw_line in stream:
                    if not raw_line.strip():
                        continue
                    raw_row = json.loads(raw_line)
                    source_cache[source_path][raw_row.get("id")] = (
                        hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest(), raw_row)
        if row.get("program_id") in excluded_programs:
            raise ValueError(f"case {case_id} is covered by the held SST context exclusion")
        source_row_id = row.get("source_row_id")
        found = source_cache[source_path].get(source_row_id)
        if found is None or found[0] != row.get("source_row_sha256"):
            raise ValueError(f"case {case_id} does not resolve to its exact raw trajectory row")
        raw_row = found[1]
        if raw_row.get("outcome", {}).get("accepted") is not True or raw_row.get("outcome", {}).get("status") != "done":
            raise ValueError(f"case {case_id} raw trajectory is not an accepted completed execution")
        for model_name, (_, receipt, _) in model_receipts.items():
            row_binding = next((binding for binding in receipt["source"]["row_bindings"]
                                if binding.get("id") == source_row_id), None)
            if (row_binding is None or row_binding.get("sha256") != found[0] or
                    row_binding.get("program_id") != row.get("program_id")):
                raise ValueError(f"case {case_id} raw row is absent from {model_name} admitted source bindings")
        trajectory_index = row.get("trajectory_index")
        trajectory = raw_row.get("trajectory") or []
        if type(trajectory_index) is not int or trajectory_index < 0 or trajectory_index >= len(trajectory):
            raise ValueError(f"case {case_id} has an invalid trajectory index")
        selected_turn = trajectory[trajectory_index]
        if selected_turn.get("raw_response_sha256") != row.get("raw_response_sha256"):
            raise ValueError(f"case {case_id} selected teacher response does not match its raw trajectory")
        messages = selected_turn.get("context")
        assistant = selected_turn.get("assistant") or {}
        response = selected_turn.get("model_response") or {}
        target = {"content": assistant.get("content") or response.get("text") or "",
                  "reasoning_content": assistant.get("reasoning") or ""}
        raw_calls = response.get("raw_calls") or []
        if raw_calls:
            target["tool_calls"] = raw_calls
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"case {case_id} needs nonempty source-bound prior messages")
        # Confirm this exact emitted turn occurs in a train graph for both student views.
        for model_name, (model_proof, receipt, receipt_path) in model_receipts.items():
            train_ref = model_proof["outputs"]["train"]
            graph_path = Path(train_ref["path"]).resolve(strict=True)
            if graph_path not in verified_graphs:
                if _sha256(graph_path) != train_ref["sha256"]:
                    raise ValueError(f"case {case_id} {model_name} admitted train graph bytes changed")
                import io
                import zstandard
                matches = set()
                with graph_path.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as reader, io.TextIOWrapper(reader, encoding="utf-8") as stream:
                    for line in stream:
                        graph = json.loads(line)
                        source = (graph.get("conversion") or {}).get("teacher_source") or {}
                        if graph.get("split") == "train" and source.get("decision_admission", {}).get("approved") is True:
                            matches.add((source.get("source_row_id"), source.get("teacher_turn_index"),
                                         source.get("raw_response_sha256"), source.get("source_program_id"),
                                         source.get("source_row_sha256")))
                verified_graphs[graph_path] = matches
            if (source_row_id, trajectory_index, row.get("raw_response_sha256"), row.get("program_id"),
                    row.get("source_row_sha256")) not in verified_graphs[graph_path]:
                raise ValueError(f"case {case_id} selected action is not an admitted {model_name} train graph")
        if not isinstance(target.get("content"), str):
            raise ValueError(f"case {case_id} target content is not text")
        if not any(message.get("role") == "assistant" and
                   isinstance(message.get("reasoning_content"), str) and
                   message["reasoning_content"].strip() for message in messages):
            raise ValueError(f"case {case_id} has no earlier assistant reasoning to measure")
        if any(message.get("role") == "assistant" and
               message.get("content") == target["content"] and
               message.get("reasoning_content") == target.get("reasoning_content")
               for message in messages):
            raise ValueError(f"case {case_id} target is already present in its context")
        row["messages"] = messages
        row["target"] = target
        row["tools"] = selected_turn.get("tools_offered")
    if len(seen) != expected_count:
        raise ValueError("case IDs do not cover the requested comparison count")
    return rows, {"case_packet_path": str(case_path), "case_packet_sha256": _sha256(case_path),
                  "exclusion_proof_path": str(exclusion_path), "exclusion_proof_sha256": exclusion_sha256,
                  "current_protection_sha256": current_protection_sha256, "case_count": len(rows),
                  "validation_proofs": [{"path": str(path), "sha256": _sha256(path)} for path in sorted(verified_proofs)]}


def _tokenize_rendered(engine, rendered) -> list[int]:
    tokens = []
    for segment in rendered.segments:
        if not isinstance(segment, str):
            raise ValueError("live-window cases must not contain unresolved soft-state blocks")
        tokens.extend(engine._template_tokens(segment, rendered.escape_nonce))
    return tokens


def _render_messages(engine, messages, tools, *, add_generation_prompt: bool):
    from natlang_neuralese.serve.chat import render_messages, render_with_empty_thought

    tokenizer = engine.tokenizer

    def apply(turns, schemas):
        def render(current):
            return tokenizer.apply_chat_template(current, tools=schemas or None, tokenize=False,
                                                 add_generation_prompt=add_generation_prompt)
        return render_with_empty_thought(render, turns) if not add_generation_prompt else render(turns)

    return engine.render_request(messages, tools, template=apply)


def target_log_likelihood(engine, messages, target: dict, tools=None) -> dict:
    """Score the actual target continuation in one causal native forward."""
    import torch
    import torch.nn.functional as F

    prompt_render = engine.render_request(messages, tools)
    prompt_ids = _tokenize_rendered(engine, prompt_render)
    full_messages = [*messages, {"role": "assistant", **target}]
    complete_render = _render_messages(engine, full_messages, tools, add_generation_prompt=False)
    full_ids = _tokenize_rendered(engine, complete_render)
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("generation prompt token IDs are not an exact prefix of the teacher-forced target")
    target_ids = full_ids[len(prompt_ids):]
    if not target_ids:
        raise ValueError("target renders to no scored tokens")
    if len(full_ids) > engine.context:
        raise ValueError(f"case requires {len(full_ids)} tokens, exceeding context {engine.context}")
    device = engine.shared_model.input_rows.device
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden = engine._native_forward(input_ids[0].tolist(), None)["h_final"]
        labels = torch.tensor(target_ids, dtype=torch.long, device=device)
        scored = []
        first_position = len(prompt_ids) - 1
        for start in range(0, len(target_ids), 32):
            stop = min(start + 32, len(target_ids))
            logits = engine._native_run(lambda start=start, stop=stop: engine.shared_model.logits(
                hidden[:, first_position + start:first_position + stop]))
            scored.append(F.log_softmax(logits[0].float(), dim=-1).gather(
                1, labels[start:stop, None]).squeeze(1))
        token_log_probs = torch.cat(scored)
    return {"target_token_count": len(target_ids), "target_log_likelihood": float(token_log_probs.sum().item()),
            "target_mean_log_likelihood": float(token_log_probs.mean().item()),
            "target_token_ids_sha256": hashlib.sha256(json.dumps(target_ids, separators=(",", ":")).encode()).hexdigest(),
            "prompt_token_count": len(prompt_ids), "full_token_count": len(full_ids),
            "earlier_reasoning_in_prompt": True}


def _atomic_json(path: Path, value: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(args) -> dict:
    import torch
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from natlang_neuralese.serve.engine import GenerationRequest

    cases, source_binding = validate_case_packet(Path(args.cases), Path(args.exclusion_proof),
        exclusion_sha256=args.exclusion_proof_sha256, current_protection_sha256=args.current_protection_sha256,
        expected_count=args.cases_per_variant)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    checkpoint_sha = _sha256(checkpoint)
    output = Path(args.output).resolve()
    engine = load_native_engine(str(checkpoint), base=args.base, device=args.device,
                                prompt_rendering=args.prompt_rendering)
    baseline = engine.tokenizer.natlang_history_reasoning["policy"]
    if baseline not in ("keep", "last_turn_only"):
        raise ValueError(f"checkpoint history profile is not a supported comparison baseline: {baseline!r}")
    if args.baseline is not None and args.baseline != baseline:
        raise ValueError(f"requested baseline {args.baseline!r} differs from checkpoint-bound {baseline!r}")
    if args.cases_per_variant != 48:
        raise ValueError("the adopted live window requires exactly 48 train-owned cases per profile")
    output.mkdir(parents=True, exist_ok=False)
    profiles = (baseline, "keep")
    engine.shared_model.eval()
    for parameter in engine.shared_model.parameters():
        parameter.requires_grad_(False)
    metadata = {"schema": SCHEMA, "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                "native_identity": engine.native_identity, "baseline_profile": baseline,
                "candidate_profile": "keep", "cases": source_binding,
                "prompt_rendering": args.prompt_rendering, "device": args.device,
                "max_tokens": args.max_tokens, "temperature": 0.0,
                "weights_mutated": False, "training_admission": False}
    _atomic_json(output / "run.json", metadata)
    rows_path = output / "results.jsonl"
    with rows_path.open("x", encoding="utf-8") as results:
        for profile in profiles:
            bind_history_reasoning(engine.tokenizer, policy=profile, allow_transition=True)
            for row in cases:
                context = row["messages"]
                target = row["target"]
                likelihood = target_log_likelihood(engine, context, target, row.get("tools"))
                completion = engine.generate(GenerationRequest(messages=context, tools=row.get("tools"),
                    max_tokens=args.max_tokens, temperature=0.0, seed=args.seed))
                message = completion["choices"][0]["message"]
                record = {"case_id": row["case_id"], "task_id": row["task_id"], "profile": profile,
                          "source_row_id": row["source_row_id"], "source_row_sha256": row["source_row_sha256"],
                          "generation": message, "finish_reason": completion["choices"][0]["finish_reason"],
                          "usage": completion.get("usage"), "likelihood": likelihood}
                results.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                results.flush()
                os.fsync(results.fileno())
    summary = {**metadata, "results_path": str(rows_path), "results_sha256": _sha256(rows_path),
               "result_rows": len(cases) * len(profiles), "profiles": list(profiles),
               "completed_at_unix": time.time(), "weights_mutated": False,
               "training_admission": False}
    _atomic_json(output / "summary.json", summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="select exact currently admitted train turns")
    prepare.add_argument("--validation-proof", action="append", type=Path, required=True)
    prepare.add_argument("--exclusion-proof", required=True)
    prepare.add_argument("--exclusion-proof-sha256", required=True)
    prepare.add_argument("--current-protection-sha256", required=True)
    prepare.add_argument("--output", required=True, help="new immutable case JSONL path")
    prepare.add_argument("--cases-per-variant", type=int, default=48)
    verify = subparsers.add_parser("verify", help="verify packet and all existing admission/source bindings")
    verify.add_argument("--cases", required=True)
    verify.add_argument("--exclusion-proof", required=True)
    verify.add_argument("--exclusion-proof-sha256", required=True)
    verify.add_argument("--current-protection-sha256", required=True)
    verify.add_argument("--cases-per-variant", type=int, default=48)
    compare = subparsers.add_parser("compare", help="run same cases under checkpoint and KEEP profiles")
    compare.add_argument("--checkpoint", required=True)
    compare.add_argument("--base")
    compare.add_argument("--cases", required=True, help="48-row JSONL of train-only source-bound cases")
    compare.add_argument("--exclusion-proof", required=True, help="current train-source context exclusion proof")
    compare.add_argument("--exclusion-proof-sha256", required=True)
    compare.add_argument("--current-protection-sha256", required=True)
    compare.add_argument("--output", required=True, help="new immutable output directory")
    compare.add_argument("--device", default="cuda")
    compare.add_argument("--prompt-rendering", choices=("full", "program-header"), default="full")
    compare.add_argument("--baseline", choices=("keep", "last_turn_only"))
    compare.add_argument("--cases-per-variant", type=int, default=48)
    compare.add_argument("--max-tokens", type=int, default=256)
    compare.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_case_packet(args.validation_proof, Path(args.exclusion_proof), Path(args.output),
            exclusion_sha256=args.exclusion_proof_sha256,
            current_protection_sha256=args.current_protection_sha256,
            expected_count=args.cases_per_variant)
    elif args.command == "verify":
        _, result = validate_case_packet(Path(args.cases), Path(args.exclusion_proof),
            exclusion_sha256=args.exclusion_proof_sha256,
            current_protection_sha256=args.current_protection_sha256,
            expected_count=args.cases_per_variant)
    else:
        result = run(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
