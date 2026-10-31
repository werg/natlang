"""Load Luna provider turns only through their bound campaign and context receipts."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_value(value) -> str:
    return _sha(_canonical(value))


def _read_bound(path, digest):
    if not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("provider teacher source artifact binding is incomplete")
    data = Path(path).read_bytes()
    if _sha(data) != digest:
        raise ValueError(f"provider teacher source artifact digest mismatch: {path}")
    return data


def _jsonl_by_id(data, ident):
    matches = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("id") == ident:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError("provider teacher source row id is absent or ambiguous")
    return matches[0]


_TEACHER_TEXT_WRAPPER = re.compile(r"<\|neuralese\|>([\s\S]+?)<\|/neuralese\|>")


def _rewrite_target_text_spans(target, source_turn):
    """Rewrite exact teacher Neuralese return values to forced native spans.

    The raw provider target stays immutable. The replay target replaces only a
    whole wrapped return_result.value with a collision-free marker; the body is
    supervised by the engine's current produce-role span path.
    """
    assistant = (source_turn or {}).get("assistant") or {}
    calls = assistant.get("calls") or []
    if not isinstance(target, dict) or not isinstance(target.get("tool_calls"), list):
        return copy_json(target), [], None
    rewritten = copy_json(target)
    spans = []
    for ordinal, source_call in enumerate(calls):
        if not isinstance(source_call, dict) or source_call.get("tool") != "return_result":
            continue
        arguments = source_call.get("arguments")
        if not isinstance(arguments, dict) or not isinstance(arguments.get("value"), str):
            continue
        value = arguments["value"]
        match = _TEACHER_TEXT_WRAPPER.fullmatch(value)
        if match is None:
            continue
        body = match.group(1)
        if not body:
            raise ValueError("provider Neuralese result body is empty")
        targets = []
        for index, call in enumerate(rewritten["tool_calls"]):
            function = call.get("function") or {}
            if function.get("name") != "return_result":
                continue
            try:
                parsed = json.loads(function.get("arguments"))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("provider target return_result arguments are not exact JSON") from error
            if parsed.get("value") == value:
                targets.append((index, parsed))
        if len(targets) != 1:
            raise ValueError("provider wrapped result does not map to one normalized return_result target")
        index, parsed = targets[0]
        marker = "NATLANG_TEACHER_TEXT_" + _sha_value({
            "target": target, "turn": source_turn.get("invocation_id"), "ordinal": ordinal,
            "body_sha256": _sha(body.encode("utf-8"))})[:32]
        if marker in _canonical(target).decode("utf-8"):
            raise ValueError("provider native span marker collides with its source target")
        parsed["value"] = marker
        rewritten["tool_calls"][index]["function"]["arguments"] = json.dumps(
            parsed, ensure_ascii=False, separators=(",", ":"))
        path = ["tool_calls", index, "arguments", "value"]
        spans.append({"path": path, "marker": marker, "text": body})
    if len(spans) > 1:
        raise ValueError("provider turn has multiple wrapped results; explicit multi-span provenance is required")
    if not spans:
        return rewritten, [], None
    original_sha = _sha_value(target)
    rewritten_sha = _sha_value(rewritten)
    span = spans[0]
    binding = {"schema": "natlang.provider-return-result-native-span-rewrite/1",
        "source_turn_id": source_turn.get("invocation_id"),
        "source_turn_sha256": _sha_value(source_turn),
        "source_response_sha256": source_turn.get("raw_response_sha256"),
        "raw_target_sha256": original_sha, "rewritten_target_sha256": rewritten_sha,
        "path": span["path"], "marker": span["marker"],
        "body_sha256": _sha(span["text"].encode("utf-8"))}
    return rewritten, spans, binding


def copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def load_verified_luna_teacher_turn(audit_path, record_id, *, replay_messages, replay_tools=None):
    """Return an exact forced-turn receipt and authenticated context bodies.

    The audit, reviewed plan, frozen runtime, full provider trace, and converted
    source row are re-read and hash-checked. No teacher answers are synthesized
    or regraded. The caller supplies the actual prompt after any native-state
    rewrite; the resulting receipt binds those exact replay bytes.
    """
    audit_bytes = Path(audit_path).read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get("schema") != "natlang.luna-v6-teacher-text-python-ingestion-audit/1":
        raise ValueError("provider teacher source audit schema is unsupported")
    raw_binding = audit.get("raw_results") or {}
    converted_binding = audit.get("python_corpus_conversion") or {}
    plan_binding = audit.get("plan") or {}
    runtime_binding = audit.get("runtime") or {}
    raw_bytes = _read_bound(raw_binding.get("path"), raw_binding.get("sha256"))
    converted_bytes = _read_bound(converted_binding.get("path"), converted_binding.get("sha256"))
    plan_path = plan_binding.get("path")
    plan = json.loads(_read_bound(plan_path, plan_binding.get("sha256")))
    runtime_path = str(runtime_binding.get("path", ""))
    runtime_file = str(Path(runtime_path) / "frozen-runtime.json")
    runtime_bytes = _read_bound(runtime_file, runtime_binding.get("receipt_sha256"))
    raw_row_id = record_id.rsplit(":decision:", 1)[0]
    raw = _jsonl_by_id(raw_bytes, raw_row_id)
    record = _jsonl_by_id(converted_bytes, record_id)
    if (raw.get("version") != "natlang.teacher_trajectory.native/1"
            or (raw.get("provenance") or {}).get("model") != "gpt-6-luna"
            or (raw.get("provenance") or {}).get("provider") != "openai-codex"
            or (raw.get("provenance") or {}).get("transport") != "pi-provider"
            or (raw.get("provenance") or {}).get("collection_role") != "teacher"
            or (raw.get("provenance") or {}).get("neuralese_server") is not False):
        raise ValueError("provider teacher source row does not preserve the actual Luna provider identity")
    if (record.get("training_admission", {}).get("approved") is not True
            or record.get("outcome", {}).get("accepted") is not True
            or record.get("split") != "train"):
        raise ValueError("only accepted training turns may be forced from this Luna source")
    source_ref = record.get("source_ref") or {}
    provider_context = source_ref.get("provider_input_context") or {}
    source_conversion = record.get("native_source_conversion") or {}
    messages = source_conversion.get("source_messages")
    if (provider_context.get("schema") != "natlang.captured-provider-input-context/1"
            or not isinstance(messages, list)
            or _sha(_canonical(messages)) != source_conversion.get("source_messages_sha256")
            or source_conversion.get("source_messages_sha256") != provider_context.get("student_context_sha256")):
        raise ValueError("provider teacher turn lacks its authenticated production-context transformation")
    trajectory = raw.get("trajectory")
    index = provider_context.get("source_trajectory_index")
    if not isinstance(trajectory, list) or type(index) is not int or not 0 <= index < len(trajectory):
        raise ValueError("provider teacher turn index is outside its exact source trajectory")
    source_turn = trajectory[index]
    action_target = record.get("target")
    target_sha = _sha_value(action_target)
    target_adapter = None
    for receipt in source_ref.get("provider_expanded_read_contexts") or []:
        candidate = receipt.get("target_binding_adapter") if isinstance(receipt, dict) else None
        if isinstance(candidate, dict):
            if target_adapter is not None and target_adapter != candidate:
                raise ValueError("provider teacher target has multiple normalization bindings")
            target_adapter = candidate
    if target_adapter is None:
        # Most decisions in a captured episode have no Neuralese read, so their
        # provider-action adapter lives only in the raw trajectory. Rebuild the
        # adapter deterministically from that exact raw action and validate the
        # same call/argument projection the corpus converter emitted.
        model_response = source_turn.get("model_response") if isinstance(source_turn, dict) else None
        raw_calls = model_response.get("raw_calls") if isinstance(model_response, dict) else None
        if isinstance(raw_calls, list):
            raw_action = {"role": "assistant", "content": (model_response.get("content") or ""),
                          "raw_calls": raw_calls}
            normalized_target = record.get("target")
            normalized_sha = _sha(_canonical(normalized_target))
            source_row_sha = source_ref.get("source_row_sha256")
            normalized_calls = []
            try:
                for ordinal, raw_call in enumerate(raw_calls):
                    function = (raw_call or {}).get("function") or {}
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        raise ValueError("provider raw action has no exact JSON arguments")
                    parsed_arguments = json.loads(arguments)
                    normalized_calls.append({"id": f"teacher_{index}_{ordinal}", "type": "function",
                        "function": {"name": function.get("name"),
                                    "arguments": json.dumps(parsed_arguments, ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":"))}})
                reconstructed = {"role": "assistant", "content": raw_action["content"],
                                 "tool_calls": normalized_calls}
            except (TypeError, json.JSONDecodeError, ValueError):
                reconstructed = None
            target_adapter = {"schema": "natlang.provider-action-target-normalization/1",
                "kind": "exact-raw-provider-calls-to-materialized-tool-calls",
                "source_row_sha256": source_row_sha, "source_trajectory_index": index,
                "raw_action": raw_action, "raw_action_sha256": _sha_value(raw_action),
                "normalized_target": normalized_target, "normalized_target_sha256": normalized_sha,
                "call_count": len((normalized_target or {}).get("tool_calls") or [])}
            if reconstructed is None or _canonical(reconstructed) != _canonical(normalized_target):
                target_adapter = None
    if (not isinstance(target_adapter, dict)
            or target_adapter.get("schema") != "natlang.provider-action-target-normalization/1"
            or target_adapter.get("source_row_sha256") != source_ref.get("source_row_sha256")
            or target_adapter.get("source_trajectory_index") != index
            or target_adapter.get("normalized_target_sha256") != target_sha):
        raise ValueError("provider teacher target is not bound to its exact normalized action")
    rewritten_target, native_text_spans, target_span_binding = _rewrite_target_text_spans(
        action_target, source_turn)
    if target_span_binding is not None:
        target_adapter = {**target_adapter,
            "native_span_rewrite": target_span_binding,
            "normalized_target_sha256": _sha_value(rewritten_target)}
        action_target = rewritten_target
        target_sha = _sha_value(action_target)
        record = copy_json(record)
        record["target"] = action_target
    source_groups = record.get("source_groups")
    if not isinstance(source_groups, list) or not source_groups:
        raise ValueError("provider teacher turn lacks its upstream source groups")
    protection_path = next((path for path, digest in (plan.get("artifact_hashes") or {}).items()
                            if path.endswith("/protection-v2.json")), None)
    if protection_path is None:
        raise ValueError("reviewed provider campaign plan omits its protection artifact")
    protection_digest = (plan.get("artifact_hashes") or {}).get(protection_path)
    protection = json.loads(_read_bound(protection_path, protection_digest))
    program_ir = ((record.get("task") or {}).get("program_ir") or {})
    case_id = program_ir.get("id")
    upstream_groups = program_ir.get("source_groups")
    selected = [item for item in protection.get("selected_receipts") or []
                if item.get("case_id") == case_id and item.get("source_groups") == upstream_groups]
    if len(selected) != 1 or selected[0].get("receipt", {}).get("split") != "train":
        raise ValueError("provider teacher source groups lack one exact protected-world train disposition")
    if set(source_groups) != set(selected[0]["source_groups"]) | {selected[0]["case_id"]}:
        raise ValueError("provider teacher source groups add or omit an upstream protection group")
    read_receipts = source_ref.get("provider_expanded_read_contexts") or []
    trace_sha = (raw.get("provenance") or {}).get("trace_sha256")
    if (source_turn.get("request_sha256") != provider_context.get("source_request_sha256")
            or source_turn.get("raw_response_sha256") != provider_context.get("source_response_sha256")
            or not isinstance(trace_sha, str)
            or any(item.get("trace_sha256") != trace_sha for item in read_receipts if isinstance(item, dict))):
        raise ValueError("provider teacher request/response differs from its exact source trajectory turn")
    from .text_corpus import (_attested_provider_expanded_reads,
                              _provider_expanded_writer_source_valid)
    authentication_record = dict(record)
    authentication_record["messages"] = messages
    if source_conversion.get("source_target_present") is True:
        authentication_record["target"] = source_conversion.get("source_target")
    else:
        authentication_record.pop("target", None)
    if source_conversion.get("source_neuralese_conversion_present") is True:
        authentication_record["neuralese_conversion"] = source_conversion.get("source_neuralese_conversion")
    else:
        authentication_record.pop("neuralese_conversion", None)
    current_conversion = record.get("neuralese_conversion") or {}
    if current_conversion.get("external_context_inputs"):
        authentication_record["neuralese_conversion"] = current_conversion
    context_attestations = _attested_provider_expanded_reads(
        authentication_record, {}, split="train", source_groups=source_groups)
    typed_writes = []
    for context in context_attestations:
        block_id = context.get("block_id")
        body = context.get("body")
        if (not isinstance(block_id, str) or not block_id.startswith("nzt1_") or not isinstance(body, str)
                or _sha(body.encode("utf-8")) != context.get("body_sha256")):
            continue
        provider_receipts = [item for item in source_ref.get("provider_expanded_read_contexts") or []
                             if isinstance(item, dict) and (item.get("block") or {}).get("id") == block_id]
        if len(provider_receipts) != 1:
            raise ValueError("provider teacher typed result lacks one exact host read/write receipt")
        provider_receipt = provider_receipts[0]
        write = provider_receipt.get("producer_write") or {}
        witness = provider_receipt.get("writer_witness") or {}
        source_class = provider_receipt.get("writer_source_class")
        # The corpus attestor has already checked the whole consumer-side graph
        # edge and provider expansion. Recover the producer proof from that same
        # original receipt: its normalized attestation intentionally omits the
        # witness, so reading it from the normalized object loses the host result.
        if (not _provider_expanded_writer_source_valid(source_class, write, witness,
                                                       provider_receipt.get("block", {}).get("type"))
                or write.get("block") != block_id
                or write.get("node") != context.get("producer_write_node")
                or write.get("text_body_sha256") != context.get("body_sha256")
                or witness.get("writer_node") != write.get("node")
                or witness.get("host_result_call_id") != write.get("call_id")
                or witness.get("body_sha256") != context.get("body_sha256")
                or witness.get("raw_response_sha256") is None):
            raise ValueError("provider teacher typed result lacks a bound producer host-result witness")
        typed_writes.append({"block_id": block_id, "body": body,
            "body_sha256": context["body_sha256"], "source": write.get("source"),
            "writer_call_id": write.get("call_id"), "writer_node": write.get("node"),
            "host_result_receipt_sha256": _sha_value(witness),
            "raw_response_sha256": witness.get("raw_response_sha256"),
            "context_receipt_sha256": _sha_value(provider_receipt), "target_span_path": None,
            "context_occurrences": context.get("context_occurrences", 1)})
    receipt = {"schema": "natlang.provider-teacher-turn-source/1",
        "source_path": raw_binding["path"], "source_file_sha256": raw_binding["sha256"],
        "source_row_id": raw_row_id, "source_row_sha256": source_ref.get("source_row_sha256"),
        "source_turn_index": index, "source_turn_sha256": _sha_value(source_turn),
        "task_sha256": _sha_value(raw.get("task")),
        "program_sha256": _sha_value((raw.get("task") or {}).get("program_ir")),
        "prompt_sha256": _sha_value({"messages": replay_messages, "tools": replay_tools or []}),
        "source_prompt_sha256": _sha_value({"messages": messages, "tools": replay_tools or []}),
        "target_sha256": target_sha, "raw_target_sha256": target_adapter.get("raw_action_sha256"),
        "target_binding_sha256": _sha_value(target_adapter),
        "plan_path": plan_path, "plan_sha256": plan_binding["sha256"],
        "runtime_path": runtime_file, "runtime_receipt_sha256": runtime_binding["receipt_sha256"],
        "split": "train", "source_groups": source_groups,
        "trace_sha256": trace_sha,
        "request_sha256": provider_context.get("source_request_sha256"),
        "response_sha256": provider_context.get("source_response_sha256"),
        "transport_provenance_sha256": provider_context.get("transport_provenance_sha256"),
        "actor": {"model": "gpt-6-luna", "provider": "openai-codex", "role": "teacher",
                  "transport": "pi-provider"},
        "target_text_spans": [{"path": item["path"], "marker": item["marker"],
            "body_sha256": _sha(item["text"].encode("utf-8")),
            "rewrite_sha256": _sha_value(target_span_binding),
            "source_turn_sha256": target_span_binding["source_turn_sha256"],
            "source_response_sha256": target_span_binding["source_response_sha256"],
            "raw_target_sha256": target_span_binding["raw_target_sha256"]}
            for item in native_text_spans],
            "typed_text_writes": [{key: item[key] for key in ("block_id", "body_sha256", "source", "writer_call_id",
            "writer_node", "host_result_receipt_sha256", "raw_response_sha256", "context_receipt_sha256",
            "target_span_path", "context_occurrences")} for item in typed_writes]}
    return {"record": record, "source_receipt": receipt, "native_text_spans": native_text_spans,
            "typed_text_bodies": [{key: item[key] for key in ("block_id", "body", "body_sha256", "source",
                "writer_call_id", "writer_node", "host_result_receipt_sha256", "context_receipt_sha256",
                "context_occurrences")}
                for item in typed_writes],
            "audit_sha256": _sha(audit_bytes), "raw_row_sha256": _sha_value(raw),
            "runtime_manifest_sha256": _sha(runtime_bytes)}
