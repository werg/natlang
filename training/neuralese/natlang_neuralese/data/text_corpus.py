"""Deterministic ordinary gold-text rows for neuralese warm-up."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def tokenizer_fingerprint(tokenizer) -> str:
    """Bind pretokenized text to exact token coordinates and chat formatting."""
    return _sha(_canonical({"vocab": tokenizer.get_vocab(),
                            "chat_template": tokenizer.chat_template,
                            "special_tokens": sorted(tokenizer.all_special_tokens),
                            "tokenizer_backend": json.loads(tokenizer.backend_tokenizer.to_str())}).encode("utf-8"))


def _native_gold_render(tokenizer, turns, tools):
    """Use the serving renderer; content markers remain ordinary content tokens."""
    from ..serve.chat import render_messages, render_with_empty_thought, split_escaped
    def template(turns, schemas):
        return render_with_empty_thought(lambda m: tokenizer.apply_chat_template(
            m, tools=schemas or None, tokenize=False, add_generation_prompt=False), turns)
    rendered = render_messages(turns, tools, template,
                               specials=(*tokenizer.all_special_tokens, "<|neuralese|>", "<|/neuralese|>"))
    if rendered.blocks:
        raise ValueError("ordinary gold text contains unresolved neuralese blocks")
    text, ids = [], []
    for segment in rendered.segments:
        for run, escaped in split_escaped(segment, rendered.escape_nonce):
            text.append(run)
            ids.extend(tokenizer(run, add_special_tokens=False,
                                 split_special_tokens=escaped)["input_ids"])
    return "".join(text), ids


def _message_neuralese_ids(value):
    if isinstance(value, dict):
        if value.get("type") == "neuralese" and isinstance(value.get("id"), str):
            yield value["id"]
        for key, child in value.items():
            if key == "arguments" and isinstance(child, str):
                try:
                    child = json.loads(child)
                except json.JSONDecodeError:
                    pass
            yield from _message_neuralese_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _message_neuralese_ids(child)


def _message_soft_reads(value):
    """Yield explicit soft-state read parts without interpreting free text."""
    if isinstance(value, dict):
        if value.get("type") == "read" and isinstance(value.get("name"), str):
            yield value
        for key, child in value.items():
            if key == "arguments" and isinstance(child, str):
                try:
                    child = json.loads(child)
                except json.JSONDecodeError:
                    pass
            yield from _message_soft_reads(child)
    elif isinstance(value, list):
        for child in value:
            yield from _message_soft_reads(child)


def _soft_writer_sources(records, source_hashes):
    """Index only admitted, successful string writes as possible *context* bodies.

    These bodies are sourced from the exact `$write.source` in an approved
    producer target. They hydrate matching reader context; they do not create
    extra target rows or independent gold labels.
    """
    records = list(records)
    sources = {}

    def add_source(block_id, attestation):
        same_writer = [item for item in sources.setdefault(block_id, [])
                       if item.get("writer_record_id") == attestation.get("writer_record_id")
                       and item.get("body_sha256") == attestation.get("body_sha256")
                       and item.get("body") == attestation.get("body")]
        occurrence = {key: attestation[key] for key in
                      ("writer_write_node", "result_path", "write_name", "source_kind") if key in attestation}
        if len(same_writer) == 1:
            same_writer[0].setdefault("write_occurrences", []).append(occurrence)
        else:
            attestation["write_occurrences"] = [occurrence]
            sources[block_id].append(attestation)

    def add_write(write, record):
        if (isinstance(write, dict) and isinstance(write.get("name"), str)
                and write["name"].startswith("soft-state:")
                and write.get("type") == "Neuralese<string>"
                and isinstance(write.get("source"), str)):
            block_id = write.get("block_id") or write["name"].split(":", 1)[1]
            if not isinstance(block_id, str) or not block_id.startswith("nz1_"):
                return
            attestation = {"body": write["source"], "writer_record_id": record.get("id"),
                           "writer_record_sha256": source_hashes.get(record.get("id")),
                           "writer_source_row_sha256": ((record.get("source_ref") or {}).get("source_row_sha256")),
                           "writer_split": record.get("split"),
                           "writer_source_groups": sorted(set(g for g in (record.get("source_groups") or [])
                                                                if isinstance(g, str) and g)),
                           "write_name": write["name"], "body_sha256": _sha(write["source"].encode("utf-8"))}
            add_source(block_id, attestation)

    def visit(value, record):
        if isinstance(value, dict):
            write = value.get("$write")
            add_write(write, record)
            for child in value.values():
                visit(child, record)
        elif isinstance(value, list):
            for child in value:
                visit(child, record)

    for record in records:
        decision = record.get("decision") or {}
        if (record.get("training_admission", {}).get("approved") is not True
                or decision.get("training_approved") is not True
                or decision.get("failed_action") is not False):
            continue
        for call in (record.get("target") or {}).get("tool_calls") or []:
            raw = (call.get("function") or {}).get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                continue
            visit(args, record)
            sidecar = call.get("neuralese_code")
            if sidecar is not None:
                from ..train.inline_instructions import validate_inline_instruction_code
                checked = validate_inline_instruction_code(raw, sidecar)
                if checked.valid:
                    for write in checked.value.writes:
                        add_write({"name": write.name, "type": write.type, "source": write.source}, record)

        source_ref = record.get("source_ref") or {}
        for call in ((decision.get("assistant") or {}).get("calls") or []):
            for receipt in ((call.get("outcome") or {}).get("typed_result_writes") or []):
                if (not isinstance(receipt, dict)
                        or receipt.get("schema") != "natlang.typed-result-write/1"
                        or receipt.get("invocation_id") != source_ref.get("invocation_id")
                        or receipt.get("source_row_sha256") != source_ref.get("source_row_sha256")
                        or receipt.get("source_kind") not in {"typed-text-result", "typed-text-result-field"}
                        or receipt.get("result_type") != "Neuralese<string>"
                        or not isinstance(receipt.get("writer_node"), str)
                        or not isinstance(receipt.get("block_id"), str)
                        or not receipt["block_id"].startswith("nz1_")
                        or not isinstance(receipt.get("body_source"), str)
                        or not isinstance(receipt.get("body_sha256"), str)
                        or _sha(receipt["body_source"].encode("utf-8")) != receipt["body_sha256"]
                        or not isinstance(receipt.get("result_path"), list)
                        or not receipt["result_path"] or receipt["result_path"][0] != "return"):
                    continue
                add_source(receipt["block_id"], {"body": receipt["body_source"],
                    "writer_record_id": record.get("id"), "writer_record_sha256": source_hashes.get(record.get("id")),
                    "writer_source_row_sha256": source_ref.get("source_row_sha256"),
                    "writer_split": record.get("split"),
                    "writer_source_groups": sorted(set(g for g in (record.get("source_groups") or [])
                                                         if isinstance(g, str) and g)),
                    "write_name": "soft-state:" + receipt["block_id"], "body_sha256": receipt["body_sha256"],
                    "writer_invocation_id": receipt["invocation_id"], "writer_write_node": receipt["writer_node"],
                    "result_path": receipt["result_path"], "source_kind": "runtime-typed-result-write"})

    rows_by_invocation = {}
    for record in records:
        source_ref = record.get("source_ref") or {}
        rows_by_invocation.setdefault((source_ref.get("trajectory_id"), source_ref.get("invocation_id")), []).append(record)
    for reader in records:
        reader_ref = reader.get("source_ref") or {}
        for context in reader_ref.get("provider_expanded_read_contexts") or []:
            if not isinstance(context, dict) or context.get("origin") != "same-run-producer":
                continue
            writer_event, block = context.get("producer_write"), context.get("block")
            if not isinstance(writer_event, dict) or not isinstance(block, dict):
                continue
            writer_call, writer_node = writer_event.get("call_id"), writer_event.get("node")
            block_id, body, body_sha = block.get("id"), block.get("body"), block.get("body_sha256")
            if (not isinstance(writer_call, str) or not isinstance(writer_node, str)
                    or not isinstance(block_id, str) or not isinstance(body, str)
                    or not isinstance(body_sha, str) or _sha(body.encode("utf-8")) != body_sha
                    or writer_event.get("block") != block_id or writer_event.get("text_body_sha256") != body_sha):
                continue
            for writer_record in rows_by_invocation.get((reader_ref.get("trajectory_id"), writer_call), []):
                writer_ref = writer_record.get("source_ref") or {}
                if writer_ref.get("source_row_sha256") != reader_ref.get("source_row_sha256"):
                    continue
                receipts = [receipt for call in ((((writer_record.get("decision") or {}).get("assistant") or {}).get("calls")) or [])
                            for receipt in ((call.get("outcome") or {}).get("typed_result_writes") or [])
                            if isinstance(receipt, dict) and receipt.get("writer_node") == writer_node
                            and receipt.get("block_id") == block_id and receipt.get("body_sha256") == body_sha]
                if len(receipts) != 1 or receipts[0].get("body_source") is not None:
                    continue
                receipt = receipts[0]
                add_source(block_id, {"body": body, "writer_record_id": writer_record.get("id"),
                    "writer_record_sha256": source_hashes.get(writer_record.get("id")),
                    "writer_source_row_sha256": writer_ref.get("source_row_sha256"),
                    "writer_split": writer_record.get("split"),
                    "writer_source_groups": sorted(set(g for g in (writer_record.get("source_groups") or [])
                                                         if isinstance(g, str) and g)),
                    "write_name": "soft-state:" + block_id, "body_sha256": body_sha,
                    "writer_invocation_id": writer_call, "writer_write_node": writer_node,
                    "result_path": receipt.get("result_path"),
                    "source_kind": "same-run-provider-expanded-writer-context"})
    return sources


def _attested_neuralese_message_bodies(record, writer_sources=None, *, split, source_groups,
                                       authenticated_context_bodies=None):
    """Map exact message blocks to hash-bound writer or creation source text."""
    messages = record.get("messages") or []
    message_block_ids = list(_message_neuralese_ids(messages))
    block_ids = sorted(set(message_block_ids))
    if not block_ids:
        return {}, []
    site = (((record.get("source_ref") or {}).get("inline_instruction_site") or {}).get("site") or {})
    body_id, body = site.get("soft_body_id"), site.get("raw_body_source")
    bodies, attestations = {}, []
    for block_id in block_ids:
        reader_source_row = ((record.get("source_ref") or {}).get("source_row_sha256"))
        eligible_writers = [writer for writer in (writer_sources or {}).get(block_id, [])
                            if writer.get("writer_split") == split
                            and (set(writer.get("writer_source_groups") or []) & set(source_groups)
                                 or (isinstance(reader_source_row, str) and reader_source_row
                                     and writer.get("writer_source_row_sha256") == reader_source_row))]
        writer = eligible_writers[0] if len(eligible_writers) == 1 else None
        if (block_id == body_id and message_block_ids.count(block_id) == 1
                and isinstance(body, str) and body.strip()
                and "<|neuralese|>" not in body and "<|/neuralese|>" not in body
                and site.get("raw_body_source_sha256") == _sha(body.encode("utf-8"))):
            attestation = {"body": body, "writer_record_id": record.get("id"),
                           "writer_record_sha256": record.get("_source_record_sha256"),
                           "writer_source_row_sha256": ((record.get("source_ref") or {}).get("source_row_sha256")),
                           "write_name": "inline_instruction_site", "body_sha256": _sha(body.encode("utf-8")),
                           "source_kind": "hash_bound_creation_body"}
        elif writer is not None:
            attestation = {**writer, "source_kind": "approved_writer_target_source"}
        elif isinstance(authenticated_context_bodies, dict) and block_id in authenticated_context_bodies:
            context = authenticated_context_bodies[block_id]
            attestation = {key: value for key, value in context.items() if key != "body"} | {
                "body": context["body"], "source_kind": "provider_expanded_context_only_input"}
        else:
            raise ValueError(f"message neuralese body has no unique same-split, same-source hash-bound source: {block_id}")
        bodies[block_id] = f"<|neuralese|>{attestation['body']}<|/neuralese|>"
        attestations.append({key: value for key, value in attestation.items() if key != "body"}
                            | {"block_id": block_id})
    return bodies, attestations


def _attested_provider_expanded_reads(record, writer_sources=None, *, split, source_groups):
    """Attest exact crisp stand-in reads against their authenticated same-run writer.

    The reader stays a typed ``read`` in native/R data, while ordinary gold text
    renders its exact source string. This receipt makes the grounding explicit
    without claiming the body was a learned vector or a new target.
    """
    conversion = record.get("neuralese_conversion") or {}
    metadata = [item for item in conversion.get("external_context_inputs", [])
                if isinstance(item, dict)
                and item.get("schema") == "natlang.external-context-input/1"
                and item.get("origin") == "same-run-producer"
                and item.get("learner_representation") in {
                    "typed-read-linked-to-existing-writer",
                    "typed-read-from-authenticated-runtime-writer-event-context-only"}]
    if not metadata:
        return []
    reads = list(_message_soft_reads(record.get("messages") or []))
    def context_ref_count(value, block_id):
        if isinstance(value, dict):
            own = ((value.get("type") == "neuralese" and value.get("id") == block_id) or
                   (value.get("type") == "read" and value.get("name") == "soft-state:" + block_id))
            count = int(own)
            for key, child in value.items():
                if key == "arguments" and isinstance(child, str):
                    try:
                        count += context_ref_count(json.loads(child), block_id)
                    except json.JSONDecodeError:
                        pass
                else:
                    count += context_ref_count(child, block_id)
            return count
        if isinstance(value, list):
            return sum(context_ref_count(child, block_id) for child in value)
        return 0
    attestations = []
    reader_source_row = ((record.get("source_ref") or {}).get("source_row_sha256"))
    reader_invocation = ((record.get("source_ref") or {}).get("invocation_id"))
    for item in metadata:
        block_id = item.get("block_id")
        body_sha256 = item.get("body_sha256")
        if (item.get("type") != "Neuralese<string>" or not isinstance(block_id, str)
                or not block_id.startswith("nz1_") or not isinstance(body_sha256, str)
                or item.get("learned_vectors") is not False
                or item.get("qualification_certificate") is not False
                or item.get("training_admission") is not False
                or item.get("invocation_id") != reader_invocation
                or item.get("source_row_sha256") != reader_source_row
                or not isinstance(item.get("read_node"), str)
                or not isinstance(item.get("model_turn_node"), str)
                or not isinstance(item.get("producer_write_node"), str)
                or not isinstance(item.get("transport_provenance_sha256"), str)):
            raise ValueError("provider-expanded read context metadata is incomplete")
        name = "soft-state:" + block_id
        matches = [part for part in reads if part.get("name") == name]
        if not matches or any(not isinstance(part.get("source"), str) for part in matches):
            raise ValueError("provider-expanded read does not have an exact typed source")
        body = matches[0]["source"]
        if any(part["source"] != body for part in matches) or _sha(body.encode("utf-8")) != body_sha256:
            raise ValueError("provider-expanded read source digest mismatch")
        eligible = [writer for writer in (writer_sources or {}).get(block_id, [])
                    if writer.get("writer_split") == split
                    and (set(writer.get("writer_source_groups") or []) & set(source_groups)
                         or (isinstance(reader_source_row, str) and reader_source_row
                             and writer.get("writer_source_row_sha256") == reader_source_row))
                    and ((writer.get("writer_write_node") is None and
                          not any(isinstance(item, dict) and item.get("writer_write_node") is not None
                                  for item in writer.get("write_occurrences", []))) or
                         writer.get("writer_write_node") == item.get("producer_write_node") or
                         any(isinstance(occurrence, dict) and
                             occurrence.get("writer_write_node") == item.get("producer_write_node")
                             for occurrence in writer.get("write_occurrences", [])))
                    and writer.get("body_sha256") == body_sha256
                    and writer.get("body") == body]
        if len(eligible) == 1:
            writer = eligible[0]
            attestations.append({key: value for key, value in writer.items() if key != "body"} | {
                "block_id": block_id, "reader_record_id": record.get("id"),
                "read_node": item["read_node"], "model_turn_node": item["model_turn_node"],
                "producer_write_node": item["producer_write_node"],
                "transport_provenance_sha256": item["transport_provenance_sha256"],
                "writer_target_selected": item.get("writer_target_selected"), "body": body,
                "source_kind": "provider-expanded-same-run-read"})
            continue
        if eligible:
            raise ValueError("provider-expanded read has ambiguous same-split, same-source writers")
        if item.get("writer_target_selected") is not False:
            raise ValueError("provider-expanded read has no unique same-split, same-source writer")
        source_ref = record.get("source_ref") or {}
        receipts = [receipt for receipt in source_ref.get("provider_expanded_read_contexts", [])
                    if isinstance(receipt, dict) and receipt.get("schema") == "natlang.provider-expanded-read-context/2"
                    and receipt.get("origin") == "same-run-producer" and receipt.get("block", {}).get("id") == block_id]
        if len(receipts) != 1:
            raise ValueError("context-only read lacks one hash-bound provider receipt")
        receipt = receipts[0]
        block = receipt.get("block") or {}
        write = receipt.get("producer_write") or {}
        read = receipt.get("block_read") or {}
        turn = receipt.get("model_turn") or {}
        source_binding_fields = (receipt.get("source_action_target_sha256"),
                                 receipt.get("source_trajectory_index"),
                                 receipt.get("source_request_sha256"),
                                 receipt.get("source_response_sha256"))
        source_binding_present = any(value is not None for value in source_binding_fields)
        writer_source_class = receipt.get("writer_source_class")
        writer_source_valid = (
            writer_source_class in (None, "modern-typed-text-result")
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
        ) or (
            writer_source_class == "legacy-text-marker-standin-eval-code"
            and write.get("emulation_version") == "text-marker-standin/2"
            and write.get("marker_context") == "eval-code"
            and write.get("learned_vectors") is False
        ) or (
            writer_source_class == "legacy-text-marker-standin-return-result"
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
            and write.get("source") == "return_result"
            and write.get("marker_context") == "return-result"
            and isinstance(receipt.get("writer_witness"), dict)
            and receipt["writer_witness"].get("kind") == "raw-return-result-value-equals-expanded-body"
            and receipt["writer_witness"].get("source") == "return_result"
            and receipt["writer_witness"].get("host_result_call_id") == write.get("call_id")
            and receipt["writer_witness"].get("host_result_type") == block.get("type")
            and _sha256_hex(receipt["writer_witness"].get("host_result_value_sha256"))
            and _sha256_hex(receipt["writer_witness"].get("raw_response_sha256"))
        ) or (
            writer_source_class == "legacy-text-marker-standin-eval-finish"
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
            and write.get("source") == "eval-finish"
            and write.get("marker_context") == "return-result"
            and isinstance(receipt.get("writer_witness"), dict)
            and receipt["writer_witness"].get("kind") == "completed-eval-finish-host-reference"
            and receipt["writer_witness"].get("source") == "eval-finish"
            and receipt["writer_witness"].get("host_result_call_id") == write.get("call_id")
            and receipt["writer_witness"].get("host_result_type") == block.get("type")
            and _sha256_hex(receipt["writer_witness"].get("host_result_value_sha256"))
        ) or (
            writer_source_class == "legacy-text-marker-standin-eval-return"
            and write.get("producer") == "text-marker-emulation"
            and write.get("source_kind") == "typed-text-result"
            and write.get("source") == "eval-return"
            and write.get("marker_context") == "return-result"
            and isinstance(receipt.get("writer_witness"), dict)
            and receipt["writer_witness"].get("kind") == "completed-eval-return-host-reference"
            and receipt["writer_witness"].get("source") == "eval-return"
            and receipt["writer_witness"].get("host_result_call_id") == write.get("call_id")
            and receipt["writer_witness"].get("host_result_type") == block.get("type")
            and receipt["writer_witness"].get("completion_status") == "done"
            and receipt["writer_witness"].get("completion_source") == "execution_graph"
            and receipt["writer_witness"].get("completion_detail") == f"\uE000{block_id}\uE001"
            and _sha256_hex(receipt["writer_witness"].get("host_result_value_sha256"))
        )
        repeated_pairs = receipt.get("additional_read_turn_pairs") or []
        body = matches[0].get("source")
        required_hashes = (body_sha256, item.get("source_row_sha256"), item.get("trace_sha256"),
                           item.get("transport_provenance_sha256"), item.get("raw_request_sha256"),
                           item.get("rendered_request_sha256"), receipt.get("source_row_sha256"),
                           receipt.get("trace_sha256"), receipt.get("transport_provenance_sha256"),
                           receipt.get("raw_request_sha256"), receipt.get("rendered_request_sha256"),
                           block.get("body_sha256"), write.get("text_body_sha256"))
        if source_binding_present:
            required_hashes += (receipt.get("source_action_target_sha256"), receipt.get("source_request_sha256"),
                                receipt.get("source_response_sha256"))
        valid_repeated_pairs = isinstance(repeated_pairs, list) and all(
            isinstance(pair, dict) and isinstance(pair.get("block_read"), dict)
            and isinstance(pair.get("model_turn"), dict)
            and pair["block_read"].get("kind") == "block_read"
            and pair["model_turn"].get("kind") == "model_turn"
            and _nonempty_string(pair["block_read"].get("node"))
            and _nonempty_string(pair["model_turn"].get("node"))
            and pair["block_read"].get("turn") == pair["model_turn"].get("node")
            and pair["block_read"].get("call_id") == reader_invocation
            and pair["block_read"].get("block") == block_id
            and pair["model_turn"].get("call_id") == reader_invocation
            and any(isinstance(inp, dict) and inp.get("node") == write.get("node")
                    and inp.get("block") == block_id for inp in pair["block_read"].get("inputs", []))
            and any(isinstance(inp, dict) and inp.get("node") == pair["block_read"].get("node")
                    and inp.get("port") == "read" and inp.get("block") == block_id
                    for inp in pair["model_turn"].get("inputs", []))
            for pair in repeated_pairs)
        if (receipt.get("invocation_id") != reader_invocation
                or receipt.get("source_row_sha256") != reader_source_row
                or item.get("trace_sha256") != ((record.get("provenance") or {}).get("trace_sha256"))
                or receipt.get("trace_sha256") != ((record.get("provenance") or {}).get("trace_sha256"))
                or receipt.get("writer_target_selected") is not False
                or receipt.get("learned_vectors") is not False
                or receipt.get("qualification_certificate") is not False
                or receipt.get("training_admission") is not False
                or receipt.get("parent_invocation_id") != item.get("parent_invocation_id")
                or item.get("writer_source_class") != writer_source_class
                or item.get("writer_witness") != receipt.get("writer_witness")
                or block.get("type") != item.get("type") or block.get("body") != body
                or block.get("body_sha256") != body_sha256 or _sha(body.encode("utf-8")) != body_sha256
                or read.get("kind") != "block_read" or turn.get("kind") != "model_turn"
                or read.get("turn") != turn.get("node")
                or not _nonempty_string(write.get("node")) or not _nonempty_string(write.get("call_id"))
                or not _nonempty_string(read.get("node")) or not _nonempty_string(turn.get("node"))
                or not _nonempty_string(item.get("producer_write_node"))
                or not _nonempty_string(item.get("producer_call_id"))
                or not all(_sha256_hex(value) for value in required_hashes)
                or read.get("node") != item.get("read_node") or read.get("call_id") != reader_invocation
                or read.get("block") != block_id or turn.get("node") != item.get("model_turn_node")
                or turn.get("call_id") != reader_invocation
                or not any(inp.get("node") == read.get("node") and inp.get("port") == "read"
                           and inp.get("block") == block_id for inp in turn.get("inputs", []))
                or write.get("kind") != "block_write" or write.get("block") != block_id
                or write.get("call_id") != item.get("producer_call_id")
                or write.get("node") != item.get("producer_write_node")
                or write.get("result_type") != block.get("type") or write.get("truncated") is not False
                or not writer_source_valid
                or write.get("text_body_sha256") != body_sha256
                or not any(inp.get("node") == write.get("node") and inp.get("block") == block_id
                           for inp in read.get("inputs", []))
                or not valid_repeated_pairs
                or item.get("additional_read_nodes", []) != [pair.get("block_read", {}).get("node") for pair in repeated_pairs]
                or item.get("additional_model_turn_nodes", []) != [pair.get("model_turn", {}).get("node") for pair in repeated_pairs]
                or any(pair.get("block_read", {}).get("call_id") != reader_invocation
                       or pair.get("block_read", {}).get("block") != block_id
                       or pair.get("model_turn", {}).get("call_id") != reader_invocation
                       or not any(inp.get("node") == write.get("node") and inp.get("block") == block_id
                                  for inp in pair.get("block_read", {}).get("inputs", []))
                       or not any(inp.get("node") == pair.get("block_read", {}).get("node")
                                  and inp.get("port") == "read" and inp.get("block") == block_id
                                  for inp in pair.get("model_turn", {}).get("inputs", []))
                       for pair in repeated_pairs)
                or not isinstance(receipt.get("transport_provenance_sha256"), str)
                or not isinstance(receipt.get("raw_request_sha256"), str)
                or not isinstance(receipt.get("rendered_request_sha256"), str)
                or item.get("raw_request_sha256") != receipt.get("raw_request_sha256")
                or item.get("rendered_request_sha256") != receipt.get("rendered_request_sha256")
                or item.get("transport_provenance_sha256") != receipt.get("transport_provenance_sha256")
                or item.get("parent_invocation_id") != receipt.get("parent_invocation_id")):
            raise ValueError("context-only provider read receipt does not authenticate exact body and graph")
        if source_binding_present:
            if (type(receipt.get("source_trajectory_index")) is not int
                    or receipt.get("source_trajectory_index") != (record.get("decision") or {}).get("index")
                    or receipt.get("source_action_target_sha256") != _sha(_canonical(record.get("target")).encode("utf-8"))
                    or item.get("source_action_target_sha256") != receipt.get("source_action_target_sha256")
                    or item.get("source_trajectory_index") != receipt.get("source_trajectory_index")
                    or item.get("source_request_sha256") != receipt.get("source_request_sha256")
                    or item.get("source_response_sha256") != receipt.get("source_response_sha256")):
                raise ValueError("provider-expanded context selected-action binding mismatch")
        actual_context_occurrences = context_ref_count(record.get("messages") or [], block_id)
        receipt_context_occurrences = receipt.get("context_occurrences")
        metadata_context_occurrences = item.get("context_occurrences")
        if type(receipt_context_occurrences) is not int or receipt_context_occurrences < 1:
            raise ValueError("provider-expanded context occurrence count is not an exact positive integer")
        if metadata_context_occurrences is None:
            # Older authenticated schema/2 receipts predate this converter
            # metadata field. Recover only the unambiguous single-reference
            # case from the exact typed messages and provider receipt.
            if receipt_context_occurrences != 1 or actual_context_occurrences != 1:
                raise ValueError("legacy provider context occurrence count is not a single exact typed reference")
        elif (type(metadata_context_occurrences) is not int or
              metadata_context_occurrences != receipt_context_occurrences or
              actual_context_occurrences != receipt_context_occurrences):
            raise ValueError("provider-expanded context occurrence count does not match exact typed references")
        attestations.append({"block_id": block_id, "reader_record_id": record.get("id"),
            "writer_record_id": None, "writer_source_row_sha256": reader_source_row,
            "writer_invocation_id": write.get("call_id"), "writer_write_node": write.get("node"),
            "writer_target_selected": False, "read_node": item["read_node"],
            "model_turn_node": item["model_turn_node"], "producer_write_node": item["producer_write_node"],
            "transport_provenance_sha256": item["transport_provenance_sha256"],
            "raw_request_sha256": item.get("raw_request_sha256"),
            "rendered_request_sha256": item.get("rendered_request_sha256"),
            "parent_invocation_id": item.get("parent_invocation_id"),
            "body_sha256": body_sha256, "body": body,
            "source_kind": "provider-expanded-context-only-same-run-read"})
    return attestations


def _hydrate_tool_argument_blocks(messages, bodies):
    """Turn attested block parts nested in tool arguments into ordinary text parts."""
    hydrated = json.loads(json.dumps(messages, ensure_ascii=False))
    used = []
    def replace(value):
        if isinstance(value, dict):
            if value.get("type") == "neuralese" and isinstance(value.get("id"), str):
                block_id = value["id"]
                if block_id not in bodies:
                    raise ValueError(f"tool argument Neuralese block has no hash-bound source: {block_id}")
                used.append(block_id)
                return {"type": "text", "text": bodies[block_id]}
            return {k: replace(v) for k, v in value.items()}
        if isinstance(value, list):
            return [replace(v) for v in value]
        return value

    for message in hydrated:
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, list):
                fn["arguments"] = replace(args)
            elif isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    continue
                changed = replace(parsed)
                if changed != parsed:
                    fn["arguments"] = json.dumps(changed, ensure_ascii=False, separators=(",", ":"))
    return hydrated, sorted(set(used))


def native_gold_document(tokenizer, messages, target, tools):
    return _native_gold_render(tokenizer,[*messages,target],tools)


def native_gold_packet(tokenizer, messages, target, tools):
    """Bind the observed assistant suffix to exact native token coordinates.

    The prefix divergence includes any template boundary tokens affected by
    appending the target. No decode/re-encode or character-to-token guess is used.
    """
    text,ids=native_gold_document(tokenizer,messages,target,tools)
    _,prefix=_native_gold_render(tokenizer,messages,tools)
    boundary=0
    for original,complete in zip(prefix,ids):
        if original!=complete:break
        boundary+=1
    if boundary>=len(ids):
        raise ValueError('gold target adds no native token suffix')
    return text,ids,boundary


def gold_text_rows(records: Iterable[Mapping[str, Any]], pieces: Mapping[str, str] | Iterable[Mapping[str, Any]], *, tokenizer):
    """Return ``(rows, receipt, omissions, provenance)`` for admitted SFT records.

    Each row has the shared ``text_warmup.load_text_rows`` schema. Text is a
    native chat rendering of crisp messages plus the actual gold target,
    preserving roles and tool schemas/calls. Pretokenized IDs preserve the
    serving distinction between template structure and quoted content markers. Tool
    calls are never executed. Pieces resolve soft references; only explicit
    handover/read source values resolve handovers.
    """
    from ..train.trajectories import (authenticated_capture_context_augmentation, crisp_messages,
                                      handover_notes, write_sites)

    fingerprint = tokenizer_fingerprint(tokenizer)
    record_rows = list(records)
    piece_map = (dict(pieces) if isinstance(pieces, Mapping) else
                 {p["name"]: p["text"] for p in pieces
                  if isinstance(p.get("name"), str) and isinstance(p.get("text"), str)})
    prepared, omitted = [], []
    source_hashes = {}
    for record in record_rows:
        rid = record.get("id", "")
        source_hashes[rid] = record.get("_source_record_sha256") or _sha(_canonical(dict(record)).encode("utf-8"))
    writer_sources = _soft_writer_sources(record_rows, source_hashes)
    for record in record_rows:
        rid = record.get("id", "")
        if record.get("training_admission", {}).get("approved") is not True:
            omitted.append({"id": rid, "reason": "not_approved_sft_record"})
            continue
        split = record.get("split")
        if split not in ("train", "test"):
            omitted.append({"id": rid, "reason": "invalid_or_missing_split"})
            continue
        groups = sorted(set(g for g in (record.get("source_groups") or []) if isinstance(g, str) and g))
        if not groups:
            omitted.append({"id": rid, "reason": "missing_factual_source_groups"})
            continue
        try:
            notes = handover_notes(record)
            provider_context_attestations = _attested_provider_expanded_reads(
                record, writer_sources, split=split, source_groups=groups)
            provider_context_bodies = {item["block_id"]: item for item in provider_context_attestations
                                       if isinstance(item.get("body"), str)}
            neuralese_bodies, context_attestations = _attested_neuralese_message_bodies(
                record, writer_sources, split=split, source_groups=groups,
                authenticated_context_bodies=provider_context_bodies)
            for item in provider_context_attestations:
                item.pop("body", None)
            context_attestations.extend(provider_context_attestations)
            message_inputs, _ = _hydrate_tool_argument_blocks(record.get("messages") or [], neuralese_bodies)
            messages = crisp_messages(message_inputs, piece_map, notes,
                                      neuralese_bodies=neuralese_bodies)
            capture_augmentations_by_digest = {}
            for _, _, _, writer_name in write_sites(record):
                augmentation, digest = authenticated_capture_context_augmentation(record, writer_name)
                if augmentation is None:
                    continue
                if not isinstance(digest, str):
                    raise ValueError("capture context augmentation lacks its digest")
                content = augmentation.get("content")
                if not isinstance(content, str):
                    raise ValueError("capture context augmentation digest mismatch")
                prefix, separator, trailer = content.rpartition("\ncontext_augmentation_sha256=")
                if not separator or trailer != digest or _sha(prefix.encode("utf-8")) != digest:
                    raise ValueError("capture context augmentation digest mismatch")
                item = capture_augmentations_by_digest.setdefault(digest, {
                    "message": augmentation, "writer_names": [],
                    "message_sha256": _sha(_canonical(augmentation).encode("utf-8")),
                })
                if writer_name not in item["writer_names"]:
                    item["writer_names"].append(writer_name)
            messages.extend(item["message"] for item in capture_augmentations_by_digest.values())
            capture_augmentation_attestations = [
                {"writer_names": item["writer_names"], "augmentation_sha256": digest,
                 "message_sha256": item["message_sha256"]}
                for digest, item in capture_augmentations_by_digest.items()]
            target = crisp_messages([record["target"]], piece_map, notes)[0]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            omitted.append({"id": rid, "reason": "unresolved_or_malformed_crisp_reference", "detail": str(exc)[:240]})
            continue
        try:
            text, token_ids, suffix_start = native_gold_packet(tokenizer, messages, target, record.get("tools") or [])
        except (KeyError, TypeError, ValueError) as exc:
            omitted.append({"id": rid, "reason": "native_chat_render_failed", "detail": str(exc)[:240]})
            continue
        if not text.strip():
            omitted.append({"id": rid, "reason": "empty_native_turn"})
            continue
        prepared.append({"text": text, "token_ids": token_ids, "tokenizer_sha256": fingerprint,
                         "supervised_suffix_start": suffix_start,
                         "split": split, "source_groups": groups, "id": rid,
                         "neuralese_context_attestations": [dict(item, reader_record_id=rid)
                                                            for item in context_attestations],
                         "capture_context_augmentation_attestations": capture_augmentation_attestations})

    train_groups = {g for row in prepared if row["split"] == "train" for g in row["source_groups"]}
    test_groups = {g for row in prepared if row["split"] == "test" for g in row["source_groups"]}
    overlap = sorted(train_groups & test_groups)
    if overlap:
        raise ValueError(f"factual source groups cross train/test: {overlap[:20]}")
    test_texts = {row["text"] for row in prepared if row["split"] == "test"}
    n_before = len(prepared)
    prepared = [row for row in prepared if not (row["split"] == "train" and row["text"] in test_texts)]
    excluded_train_held = n_before - len(prepared)
    dedup = {}
    for row in prepared:
        key = row["split"], row["text"], tuple(row["token_ids"]),row['supervised_suffix_start']
        if key not in dedup:
            dedup[key] = {**row, "source_record_ids": [row["id"]]}
        else:
            representative = dedup[key]
            representative["source_groups"] = sorted(set(representative["source_groups"] + row["source_groups"]))
            representative["source_record_ids"].append(row["id"])
            representative["neuralese_context_attestations"].extend(row["neuralese_context_attestations"])
            representative["capture_context_augmentation_attestations"].extend(
                row["capture_context_augmentation_attestations"])
    rows = list(dedup.values())
    same_split_dupes = n_before - excluded_train_held - len(rows)
    if not any(row["split"] == "train" for row in rows) or not any(row["split"] == "test" for row in rows):
        raise ValueError("nonempty independent train and held text required")

    provenance = [{"id": row["id"], "split": row["split"], "source_groups": row["source_groups"],
                   "source_record_sha256": source_hashes[row["id"]],
                   "source_records": [{"id": rid, "sha256": source_hashes[rid]} for rid in row["source_record_ids"]],
                   "text_sha256": _sha(row["text"].encode("utf-8")),
                   "token_ids_sha256": _sha(_canonical(row["token_ids"]).encode("utf-8")),
                   "tokenizer_sha256": fingerprint,
                   "neuralese_context_attestations": row["neuralese_context_attestations"],
                   "capture_context_augmentation_attestations": row["capture_context_augmentation_attestations"]}
                  for row in rows]
    for row,entry in zip(rows,provenance):
        entry['supervised_suffix_start']=row['supervised_suffix_start']
    omissions_bytes = "".join(_canonical(row) + "\n" for row in omitted).encode("utf-8")
    provenance_bytes = "".join(_canonical(row) + "\n" for row in provenance).encode("utf-8")
    receipt = {
        "format": "natlang.gold_text_packet_receipt/1",
        "policy": "approved SFT records only; deterministic crisp rendering from supplied pieces and explicit handover notes; exact named Neuralese reader-context blocks hydrate only from a unique approved same-split writer source sharing a source group or source-row hash, or from an explicit exact provider-expanded runtime read receipt bound to body, type, request, graph and source row; context-only receipts never create a writer target or recurrence edge; typed eval-finish marker outputs are rendered from validated exact code sidecars; hash-bound full capture snapshots omitted from historical previews are supplied only as separately labeled same-invocation context augmentations; hydrated context and capture augmentation never create separate target rows; duplicate identical augmentations are emitted once per target; complete source-group split retained; train copies of held complete documents excluded; target turn rendered through native chat template with serving content escaping; no tools executed",
        "rendering": "natlang.native_gold_chat/2", "tokenizer_sha256": fingerprint,
        "supervision": "all tokens plus the actual assistant suffix beginning at native prefix token divergence; boundary tokens may be included; no fabricated targets",
        "ordinary_text_stage_only": True, "task_or_trajectory_admission_granted": False,
        "documents": len(rows), "train_documents": sum(row["split"] == "train" for row in rows),
        "test_documents": sum(row["split"] == "test" for row in rows),
        "omitted_records": len(omitted), "excluded_train_exact_held_complete_documents": excluded_train_held,
        "duplicate_same_split_documents_deduplicated": same_split_dupes,
        "hash_bound_reader_context_blocks": sum(len(row["neuralese_context_attestations"]) for row in rows),
        "unresolved_omissions": omitted,
        "omissions_jsonl_sha256": _sha(omissions_bytes),
        "provenance_jsonl_sha256": _sha(provenance_bytes),
    }
    return rows, receipt, omitted, provenance
