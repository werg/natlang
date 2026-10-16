"""Role-aware static graphs for `natlang.port-record/1` rows.

The caller owns source admission. This module only performs a declared static
transformation: it puts the exact source text in a producer span, removes that
same text from the consumer's own token stream, and imports the producer state
at that point. It never infers success from a source label or from matching
text elsewhere.
"""
from __future__ import annotations

import hashlib
import json
import math
import copy
from typing import Mapping

from ..common.hashing import sha256_hex as sha256_bytes
from .native_graph_conversion import witnessed_pair_graph
from .records import PortRecordRow

def consumer_target_authorization(row: PortRecordRow, *, reviewed_targets: Mapping[str, Mapping] | None = None):
    """Return positive-target evidence, requiring exact occurrence bindings.

    The source row's `gold`/`checked` label follows the port-record contract.
    Other labels stay in the complete consumer context unless a separate
    reviewed-target receipt binds the exact raw row, exact target, decision and
    proof. A broad case/program admission is not an occurrence-level approval.
    """
    raw_sha = sha256_bytes(row.raw_line)
    target_sha = hashlib.sha256(
        json.dumps(row.raw.get("target"), ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode("utf-8")).hexdigest()
    if row.record.is_imitation_target:
        return {"kind": "port-record-outcome", "label": row.record.outcome_label,
                "source_row_sha256": raw_sha, "target_sha256": target_sha}
    proof = (reviewed_targets or {}).get(raw_sha)
    if proof is None:
        return None
    required_hashes = (proof.get("source_row_sha256"), proof.get("target_sha256"),
                       proof.get("proof_sha256"))
    if (proof.get("status") != "approved" or proof.get("id") != row.record.id
            or required_hashes[0] != raw_sha or required_hashes[1] != target_sha
            or any(not isinstance(value, str) or len(value) != 64
                   or any(ch not in "0123456789abcdef" for ch in value)
                   for value in required_hashes)
            or not isinstance(proof.get("decision_id"), str) or not proof["decision_id"]):
        raise ValueError(f"reviewed consumer target does not bind exact source occurrence: {row.record.id}")
    return {"kind": proof.get("review_kind", "source-reviewed-target"), "source_row_sha256": raw_sha,
            "target_sha256": target_sha, "decision_id": proof["decision_id"],
            "proof_sha256": proof["proof_sha256"],
            **({"review_sha256": proof["review_sha256"],
                "oracle_verified": proof["oracle_verified"],
                "native_execution_verified": proof["native_execution_verified"]}
               if proof.get("review_kind") == "source-curated-third-party-SFT" else {})}


def _source_payload_text(raw: Mapping) -> str:
    """Keep every field in message-shaped sources, including reasoning and calls."""
    chunks = []
    for source in raw.get("sources", []):
        if isinstance(source.get("text"), str):
            chunks.append(source["text"])
        elif isinstance(source.get("messages"), list):
            chunks.append(json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        else:
            raise ValueError("source needs exact text or a full message trajectory")
    payload = "\n\n".join(chunks)
    if not payload:
        raise ValueError("port-record source payload is empty")
    return payload


def _serving_messages(messages):
    """Copy message fields into the shared serving-template spelling."""
    result = copy.deepcopy(list(messages))
    for message in result:
        if "reasoning" in message and "reasoning_content" not in message:
            message["reasoning_content"] = message.pop("reasoning")
    return result


def _template(tokenizer, turns, tools, *, generation):
    return tokenizer.apply_chat_template(turns, tools=tools or None, tokenize=False,
                                         add_generation_prompt=generation)


def _render_with_payload(tokenizer, messages, tools, payload_ids, role_open, role_close, *, generation=False):
    """Render structured serving messages while replacing one explicit block part with a native span."""
    from ..serve.chat import bind_history_reasoning, render_messages, split_escaped

    bind_history_reasoning(tokenizer)
    declared = [part.get("id") for message in messages
                for part in (message.get("content") if isinstance(message.get("content"), list) else [])
                if isinstance(part, Mapping) and part.get("type") == "neuralese"]
    if len(declared) != 1 or not isinstance(declared[0], str):
        raise ValueError("one exact static neuralese payload part must be declared")
    block_id = declared[0]
    specials = (*tokenizer.all_special_tokens, "<|neuralese|>", "<|/neuralese|>",
                "<|neuralese_produce|>", "<|/neuralese_produce|>",
                "<|neuralese_consume|>", "<|/neuralese_consume|>")
    rendered = render_messages(messages, tools, lambda turns, schemas: _template(
        tokenizer, turns, schemas, generation=generation), specials=specials)
    if rendered.blocks != [block_id] or rendered.block_kinds != ["neuralese"]:
        raise ValueError("serving template dropped, duplicated or changed the declared static payload")
    ids = []
    container = None
    for segment in rendered.segments:
        if isinstance(segment, int):
            if container is not None or segment != 0:
                raise ValueError("serving template produced an ambiguous static payload position")
            start = len(ids)
            ids.extend((role_open, *payload_ids, role_close))
            container = (start, len(ids))
        else:
            for run, escaped in split_escaped(segment, rendered.escape_nonce):
                ids.extend(tokenizer(run, add_special_tokens=False,
                                     split_special_tokens=escaped)["input_ids"])
    if container is None:
        raise ValueError("serving template omitted the static payload")
    return ids, container


def _target_message(row: PortRecordRow) -> dict:
    target = row.raw["target"]
    kind, value = target["kind"], target["value"]
    if kind == "message":
        if not isinstance(value, dict) or value.get("role") != "assistant":
            raise ValueError("message target must be one complete assistant message")
        message = copy.deepcopy(value)
    elif kind in {"text", "choice", "sql"}:
        if not isinstance(value, str):
            raise ValueError(f"{kind} target must be exact text")
        message = {"role": "assistant", "content": value}
    elif kind == "calls":
        if not isinstance(value, list) or not value:
            raise ValueError("calls target must be a nonempty complete tool-call list")
        message = {"role": "assistant", "content": "", "tool_calls": copy.deepcopy(value)}
    elif kind == "json":
        message = {"role": "assistant", "content": json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}
    else:
        raise ValueError(f"unsupported exact target kind {kind!r}")
    # Port-record keeps the upstream field name; the serving template consumes
    # reasoning_content. The raw target remains unchanged in graph provenance.
    if "reasoning" in message and "reasoning_content" not in message:
        message["reasoning_content"] = message.pop("reasoning")
    return message


def _target_reply(tokenizer, target, tools) -> list[int]:
    from ..serve.chat import assistant_reply_segments
    specials = (*tokenizer.all_special_tokens, "<|neuralese|>", "<|/neuralese|>")
    segments = assistant_reply_segments(lambda turns, generation: _template(
        tokenizer, turns, tools, generation=generation), target, specials=specials)
    if segments is None:
        raise ValueError("assistant target does not continue the native serving generation prompt")
    ids = []
    for text, escaped in segments:
        ids.extend(tokenizer(text, add_special_tokens=False,
                             split_special_tokens=escaped)["input_ids"])
    if not ids:
        raise ValueError("assistant target rendered to no tokens")
    return ids


def port_record_graph(row: PortRecordRow, *, source_tokenizer,
                      source_tokenizer_fingerprint: str, expanded_tokenizer_fingerprint: str,
                      role_ids: tuple[int, int, int, int], corpus_id: str,
                      source_file_sha256: str, consumer_admission: Mapping | None = None,
                      context_weight: float = 0.05, effective_split: str | None = None,
                      decision_hold_exclusions: tuple[Mapping, ...] = ()) -> dict:
    """Convert one source occurrence into a producer/consumer graph without truncation."""
    if not math.isfinite(context_weight) or not 0 <= context_weight <= 1:
        raise ValueError("context weight must be finite between zero and one")
    record = row.record
    if record.split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported source split {record.split!r}")
    groups = sorted(set(record.split_groups))
    if not groups or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError("port-record needs nonempty source groups")
    raw_sha = sha256_bytes(row.raw_line)
    source_text = _source_payload_text(row.raw)
    if not source_text:
        raise ValueError("port-record source payload must not be empty")
    source_ids = list(source_tokenizer(source_text, add_special_tokens=False)["input_ids"])
    if not source_ids or source_tokenizer.decode(source_ids, skip_special_tokens=False) != source_text:
        raise ValueError("source payload must tokenize and decode exactly as a standalone span")

    # This is an explicit static transformation: original writer context and
    # instructions condition an assistant-produced payload. The source is not
    # mislabeled as user input, and all source message fields are retained.
    writer_messages = _serving_messages(record.writer_context)
    writer_messages.append({"role": "user", "content": record.writer_instructions})
    writer_messages.append({"role": "assistant", "content": [
        {"type": "neuralese", "id": "nz1_static_" + hashlib.sha256(
            json.dumps(writer_messages, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()[:24]}]})
    writer_ids, writer_container = _render_with_payload(
        source_tokenizer, writer_messages, None, source_ids, role_ids[0], role_ids[1])
    writer_container_start, writer_container_stop = writer_container
    writer_row = {"id": "writer:" + raw_sha, "split": "train",
        "source_groups": groups,
        "tokenizer_sha256": source_tokenizer_fingerprint,
        "token_ids": writer_ids,
        "text": source_tokenizer.decode(writer_ids, skip_special_tokens=False),
        "supervised_suffix_start": writer_container_start}

    # The declared source-withheld transformation inserts one exact block part
    # at the beginning of the first consumer request. Existing assistant
    # reasoning, tool calls, tool results and their order remain in the rendered
    # serving conversation around that imported state.
    consumer = row.raw["consumer"]
    consumer_messages = _serving_messages(record.consumer_context)
    if not consumer_messages:
        raise ValueError("consumer context must retain its complete original message sequence")
    first_user = next((i for i, message in enumerate(consumer_messages)
                       if message.get("role") == "user"), None)
    if first_user is None or "sources" not in record.withheld:
        raise ValueError("native producer/consumer view requires a withheld source and an original user request")
    first = consumer_messages[first_user]
    old_content = first.get("content")
    block_id = "nz1_static_" + hashlib.sha256(json.dumps(consumer_messages, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    if isinstance(old_content, str):
        first["content"] = [{"type": "neuralese", "id": block_id},
                            {"type": "text", "text": "\n\n" + old_content}]
    elif old_content is None:
        first["content"] = [{"type": "neuralese", "id": block_id}]
    elif isinstance(old_content, list) and all(isinstance(part, dict) for part in old_content):
        first["content"] = ([{"type": "neuralese", "id": block_id},
                             {"type": "text", "text": "\n\n"}] + copy.deepcopy(old_content))
    else:
        raise ValueError("consumer request content cannot be preserved as serving text/parts")
    tools = consumer.get("tools") or None
    reader_ids, reader_container = _render_with_payload(
        source_tokenizer, consumer_messages, tools, source_ids, role_ids[2], role_ids[3], generation=True)
    reader_container_start, reader_container_stop = reader_container
    target = _target_message(row)
    target_ids = _target_reply(source_tokenizer, target, tools)
    target_start = len(reader_ids)
    reader_ids.extend(target_ids)
    if consumer_admission is None:
        authorization = consumer_target_authorization(row)
    else:
        # A caller-provided review receipt is only a candidate until it is
        # rechecked against this row's current ID and exact target bytes.
        raw_sha = sha256_bytes(row.raw_line)
        authorization = consumer_target_authorization(
            row, reviewed_targets={raw_sha: consumer_admission})
        if authorization is None:
            raise ValueError("consumer target admission did not authorize this exact source row")
    native_split = effective_split or ("test" if record.split in {"test", "validation"} else "train")
    if native_split not in {"train", "test"}:
        raise ValueError("effective native split must be train or test")
    writer_row["split"] = native_split
    reader_row = {"id": "reader:" + raw_sha,
        "split": native_split,
        "source_groups": groups, "tokenizer_sha256": source_tokenizer_fingerprint,
        "token_ids": reader_ids,
        "text": source_tokenizer.decode(reader_ids, skip_special_tokens=False),
        "supervised_suffix_start": target_start}

    payload_sha = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    edge_id = hashlib.sha256((corpus_id + "\0" + raw_sha + "\0" + payload_sha).encode()).hexdigest()
    writer_span = {"tokenize_payload_alone": True, "container_start": writer_container_start,
        "container_stop": writer_container_stop, "body_text": source_text, "payload_token_ids": source_ids}
    reader_span = {"tokenize_payload_alone": True, "container_start": reader_container_start,
        "container_stop": reader_container_stop, "body_text": source_text, "payload_token_ids": source_ids}
    graph = witnessed_pair_graph(writer_row, reader_row, source_tokenizer=source_tokenizer,
        source_tokenizer_fingerprint=source_tokenizer_fingerprint,
        expanded_tokenizer_fingerprint=expanded_tokenizer_fingerprint, role_ids=role_ids,
        writer_span=writer_span, reader_span=reader_span, payload_sha256=payload_sha,
        source_span_id="source:" + raw_sha, edge_id=edge_id, chain_id="port-row:" + raw_sha,
        context_weight=context_weight)
    consumer_node = graph["nodes"][1]
    if not authorization or decision_hold_exclusions:
        consumer_node["target_weights"][-len(target_ids):] = [0.0] * len(target_ids)
    # The authorizing record outcome is exact and separate from the synthetic
    # producer/consumer relationship; this is not an observed actor capture.
    graph["id"] = "port-record:" + raw_sha
    graph["conversion"] = {"kind": "static-port-record-producer-consumer/1",
        "corpus_id": corpus_id, "source_file_sha256": source_file_sha256,
        "source_path": str(row.path), "source_line_number": row.line_number,
        "source_byte_offset": row.byte_offset, "source_row_sha256": raw_sha,
        "original_source_split": record.split, "effective_native_split": native_split,
        "original_id": record.id, "source_groups": groups,
        "outcome_label": record.outcome_label, "outcome_checked": row.raw["outcome"].get("checked"),
        "consumer_target_authorization": authorization,
        "consumer_target_positive_tokens": len(target_ids) if authorization and not decision_hold_exclusions else 0,
        "consumer_target_context_tokens": 0 if authorization and not decision_hold_exclusions else len(target_ids),
        "decision_hold_target_exclusions": [dict(item) for item in decision_hold_exclusions],
        "source_payload_sha256": payload_sha,
        "source_payload_tokens": len(source_ids),
        "source_tokenizer_fingerprint": source_tokenizer_fingerprint,
        "declared_transformation": "static source-message serialization emitted in an assistant producer span and imported at the beginning of the original first user request; original source row remains bound by exact path, offset and line SHA",
        "original_source_messages_sha256": hashlib.sha256(json.dumps(
            [source.get("messages") for source in row.raw.get("sources", []) if "messages" in source],
            ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "original_target_sha256": hashlib.sha256(json.dumps(row.raw["target"], ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "original_consumer_messages_sha256": hashlib.sha256(json.dumps(
            consumer.get("context", []), ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "lineage": record.lineage, "license": record.license,
        "source_count": len(record.sources),
        "exact_ref_count": sum(len(source.exact_refs) for source in record.sources)}
    graph["source_pairing"].update({"schema": "natlang.port-record-static-communication/1",
        "source_corpus_id": corpus_id, "source_row_sha256": raw_sha,
        "source_file_sha256": source_file_sha256,
        "consumer_target_authorization": authorization,
        "decision_hold_target_exclusions": [dict(item) for item in decision_hold_exclusions],
        "relationship": "static-transformation; not observed actor execution"})
    return graph
