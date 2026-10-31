"""Authenticated Luna teacher text -> native student encoder -> consumer graph.

The host-evaluated text is an input to a real native text encoder. It is never
relabelled as a model-sampled writer turn. The returned graph contains the
student encoder invocation and the actual teacher-forced consumer capture.
"""
from __future__ import annotations

import copy
import hashlib
import json

from ..model.native_graph_plan import ImportSpan, Invocation, compile_graph, invocation_dict


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _replace_refs(value, bindings, expected_occurrences=None):
    """Replace authenticated teacher-text refs and preserve every recorded use."""
    found = {ident: 0 for ident in bindings}
    def visit(item):
        if isinstance(item, list):
            return [visit(part) for part in item]
        if isinstance(item, dict):
            ident = item.get("id")
            if item.get("type") == "teacher_text" and ident in bindings:
                found[ident] += 1
                binding = bindings[ident]
                return {"type": "native_state", "id": binding["native_block_id"],
                        "source_occurrence": binding["source_occurrence"]}
            return {key: visit(part) for key, part in item.items()}
        return item
    result = visit(copy.deepcopy(value))
    expected_occurrences = expected_occurrences or {ident: 1 for ident in bindings}
    if set(expected_occurrences) != set(bindings) or any(
            type(expected_occurrences[ident]) is not int or expected_occurrences[ident] < 1
            or found[ident] != expected_occurrences[ident] for ident in bindings):
        raise ValueError("authenticated Luna text references differ from their exact recorded consumer occurrences")
    return result


def capture_luna_native_turn(engine, audit_path, record_id, *, seed=0):
    """Encode a verified host result with this student, then force the exact Luna action.

    This is a teacher-forced native capture, never a sampled-policy rollout. The
    source loader reopens and hashes the raw trajectory, conversion, reviewed
    plan, runtime, and protection artifacts before this function calls the model.
    """
    from .provider_teacher_source import load_verified_luna_teacher_turn
    from ..serve.engine import GenerationRequest

    # The original source messages are carried through the audit-bound source row.
    # The loader validates them against the conversion's source-message hash.
    from .provider_teacher_source import _jsonl_by_id, _read_bound
    raw_source = json.loads(__import__("pathlib").Path(audit_path).read_bytes())
    converted_binding = raw_source.get("python_corpus_conversion") or {}
    converted_bytes = _read_bound(converted_binding.get("path"), converted_binding.get("sha256"))
    record = _jsonl_by_id(converted_bytes, record_id)
    source_opening = (record.get("native_source_conversion") or {}).get("source_messages")
    if not isinstance(source_opening, list):
        raise ValueError("Luna teacher row lacks its exact source prompt")
    tools = record.get("tools") or []
    loaded = load_verified_luna_teacher_turn(audit_path, record_id,
        replay_messages=source_opening, replay_tools=tools)
    if not loaded["typed_text_bodies"]:
        raise ValueError("native Luna graph requires at least one authenticated typed text input")
    bindings, encoder_states, replacements, occurrences = [], [], {}, {}
    for typed in loaded["typed_text_bodies"]:
        written = engine.native_write_text(typed["body"], value_type="Neuralese<string>")
        native_record = engine.store.get_native_state(written["id"])
        if native_record is None or native_record.block.text != typed["body"]:
            raise ValueError("student native text encoder did not archive the exact authenticated host body")
        source_occurrence = "teacher-text-encoder:" + _sha((record_id + typed["block_id"]).encode())[:24]
        binding = {"teacher_block_id": typed["block_id"], "body_sha256": typed["body_sha256"],
            "native_block_id": written["id"], "source_occurrence": source_occurrence,
            "source_capture": written["source_capture"]}
        bindings.append(binding)
        replacements[typed["block_id"]] = binding
        occurrences[typed["block_id"]] = typed["context_occurrences"]
        encoder_states.append(native_record.block)
    rewritten = _replace_refs(source_opening, replacements, occurrences)
    source_receipt = copy.deepcopy(loaded["source_receipt"])
    source_receipt["prompt_sha256"] = _sha(_canonical({"messages": rewritten, "tools": tools}))
    forced = {"schema": "natlang.native-teacher-forced-turn/1", "assistant": record["target"],
        "source_receipt": source_receipt, "native_text_spans": [], "source_opening": source_opening,
        "native_text_bindings": bindings}
    response = engine.generate(GenerationRequest(messages=rewritten, tools=tools, max_tokens=4096,
        temperature=0.0, seed=seed, forced_message=forced, rl_capture=True, native_recording=True))
    capture = response.get("x_natlang_rl_capture")
    if (not isinstance(capture, dict) or capture.get("schema") != "natlang.native-teacher-forced-capture/1"
            or capture.get("teacher_source_receipt") != source_receipt):
        raise ValueError("native student did not return its exact teacher-forced capture and source receipt")
    return {"record": record, "source_receipt": source_receipt, "source_opening": source_opening,
        "rewritten_messages": rewritten, "encoder_binding": bindings[0], "encoder_state": encoder_states[0],
        "encoder_bindings": bindings, "encoder_states": encoder_states,
        "encoder_occurrence_counts": occurrences,
        "capture": capture, "response": response, "loaded": loaded,
        "native_identity": copy.deepcopy(engine.native_identity)}


def native_graph_from_luna_capture(captured):
    """Create an authenticated multi-writer consumer graph with differentiable encoder KV."""
    bindings = captured.get("encoder_bindings") or [captured["encoder_binding"]]
    states = captured.get("encoder_states") or [captured["encoder_state"]]
    if len(bindings) != len(states) or not bindings:
        raise ValueError("Luna graph needs one archived native state per authenticated text binding")
    capture = captured["capture"]
    receipt = captured["source_receipt"]
    student_identity = captured.get("native_identity")
    if (not isinstance(student_identity, dict) or not student_identity.get("checkpoint_sha256")
            or not student_identity.get("tokenizer_fingerprint")
            or capture.get("policy") != student_identity):
        raise ValueError("Luna native graph needs its exact captured student checkpoint/tokenizer identity")
    tokenizer_fingerprint = student_identity["tokenizer_fingerprint"]
    if (capture.get("schema") != "natlang.native-teacher-forced-capture/1"
            or capture.get("teacher_source_receipt") != receipt):
        raise ValueError("Luna native graph capture is not bound to its exact authenticated source")
    from ..serve.store import native_state_id
    reader_name = "luna-consumer:" + _sha((receipt["source_row_id"] + str(receipt["source_turn_index"])).encode())[:24]
    own, imports = [], []
    bindings_by_native_id = {binding["native_block_id"]: (binding, block)
                             for binding, block in zip(bindings, states)}
    if len(bindings_by_native_id) != len(bindings):
        raise ValueError("Luna native encoder bindings reuse a native block ID")
    writer_nodes = []
    writer_invocations = []
    body_hashes = {}
    for binding, block in zip(bindings, states):
        if (_sha(block.text.encode("utf-8")) != binding["body_sha256"]
                or native_state_id(block) != binding["native_block_id"]):
            raise ValueError("Luna native encoder block changed its authenticated host result")
        writer_name = binding["source_occurrence"]
        writer_source = binding["source_capture"]
        writer_tokens = [token for segment in writer_source.get("segments", [])
                         for token in segment.get("token_ids", [])]
        lo, hi = writer_source.get("payload_start"), writer_source.get("payload_stop")
        if (not isinstance(lo, int) or not isinstance(hi, int)
                or writer_tokens[lo:hi] != block.token_ids.tolist()
                or lo != 1 or hi != len(writer_tokens)-1):
            raise ValueError("Luna student encoder source capture differs from its archived native text state")
        writer_nodes.append({"name": writer_name, "token_ids": writer_tokens,
            "target_weights": [0.0] * len(writer_tokens), "producing": [[lo, hi]]})
        writer_invocations.append(Invocation(writer_name, tuple(writer_tokens), producing=((lo, hi),)))
        body_hashes[binding["native_block_id"]] = _sha(_canonical(writer_source))
    import_counts = {binding["native_block_id"]: 0 for binding in bindings}
    for segment in capture.get("prompt") or []:
        if segment.get("kind") == "tokens":
            ids = segment.get("token_ids")
            if not isinstance(ids, list) or any(type(token) is not int for token in ids):
                raise ValueError("teacher-forced Luna prompt has malformed exact token inputs")
            own.extend(ids)
        elif segment.get("kind") == "native_state":
            pair = bindings_by_native_id.get(segment.get("id"))
            if pair is None or segment.get("source_occurrence") != pair[0]["source_occurrence"]:
                raise ValueError("teacher-forced Luna consumer changed its exact native encoder import")
            binding, block = pair
            # The graph source coordinate includes the explicit produce-role
            # opener; import only the producing payload span, never framing.
            imports.append(ImportSpan(at=len(own), source=binding["source_occurrence"], start=1,
                                      stop=1 + len(block.token_ids)))
            import_counts[binding["native_block_id"]] += 1
        else:
            raise ValueError("teacher-forced Luna prompt contains an unknown input segment")
    receipt_counts = {item.get("block_id"): item.get("context_occurrences")
                      for item in receipt.get("typed_text_writes", []) if isinstance(item, dict)}
    if (len(receipt_counts) != len(receipt.get("typed_text_writes", []))
            or set(receipt_counts) != {binding["teacher_block_id"] for binding in bindings}):
        raise ValueError("Luna graph encoder bindings differ from the authenticated source write inventory")
    expected_counts = {binding["native_block_id"]: receipt_counts[binding["teacher_block_id"]]
                       for binding in bindings}
    if (not own or any(type(expected_counts.get(ident)) is not int or
                       import_counts[ident] != expected_counts[ident] for ident in import_counts)):
        raise ValueError(f"Luna native graph imports do not cover every authenticated text use: "
                         f"actual={import_counts}, expected={expected_counts}")
    prompt_length = len(own)
    generated = capture.get("generated_inputs") or []
    if not generated or any(type(item.get("token_id")) is not int for item in generated):
        raise ValueError("teacher-forced Luna capture lacks the exact forced target token sequence")
    target_ids = [item["token_id"] for item in generated]
    consumer_ids = [*own, *target_ids]
    nodes = [*writer_nodes,
        {"name": reader_name, "token_ids": consumer_ids,
         "target_weights": [0.0] * prompt_length + [1.0] * len(target_ids),
         "imports": [{"at": edge.at, "source": edge.source, "start": edge.start,
                      "stop": edge.stop} for edge in imports]},
    ]
    plan = compile_graph([*writer_invocations,
        Invocation(reader_name, tuple(consumer_ids), imports=tuple(imports))])
    row = {"schema": "natlang.neuralese-native-graph/1",
        "id": "luna-native-forced:" + receipt["source_row_id"] + ":" + str(receipt["source_turn_index"]),
        "split": "train", "source_groups": list(receipt["source_groups"]),
        "tokenizer_fingerprint": tokenizer_fingerprint, "nodes": nodes,
        "student_native_identity": student_identity,
        "teacher_source_receipt": receipt,
        "teacher_text_encoders": [{"schema": "natlang.authenticated-teacher-text-encoder/1",
            "teacher_block_id": binding["teacher_block_id"], "native_block_id": binding["native_block_id"],
            "body_sha256": binding["body_sha256"], "source_occurrence": binding["source_occurrence"],
            "source_capture_sha256": body_hashes[binding["native_block_id"]],
            "host_result_receipt_sha256": next(item["host_result_receipt_sha256"]
                for item in captured["loaded"]["source_receipt"]["typed_text_writes"]
                if item["block_id"] == binding["teacher_block_id"])} for binding in bindings],
        "teacher_text_encoder": {"schema": "natlang.authenticated-teacher-text-encoder/1",
            "teacher_block_id": bindings[0]["teacher_block_id"], "native_block_id": bindings[0]["native_block_id"],
            "body_sha256": bindings[0]["body_sha256"], "source_occurrence": bindings[0]["source_occurrence"],
            "source_capture_sha256": body_hashes[bindings[0]["native_block_id"]],
            "host_result_receipt_sha256": next(item["host_result_receipt_sha256"]
                for item in captured["loaded"]["source_receipt"]["typed_text_writes"]
                if item["block_id"] == bindings[0]["teacher_block_id"])}}
    # Compile the serialized form through the production parser before exposing it.
    from .native_graph_schema import parse_graph
    example = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
    if (len(example.plan.invocations) != len(bindings) + 1
            or sum(len(node.imports) for node in example.plan.invocations) != len(imports)):
        raise ValueError("Luna native graph failed its complete producer-consumer shape check")
    return row, example
