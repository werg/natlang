"""Deterministic ordinary gold-text rows for neuralese warm-up."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _soft_writer_sources(records, source_hashes):
    """Index only admitted, successful string writes as possible *context* bodies.

    These bodies are sourced from the exact `$write.source` in an approved
    producer target. They hydrate matching reader context; they do not create
    extra target rows or independent gold labels.
    """
    sources = {}
    def visit(value, record):
        if isinstance(value, dict):
            write = value.get("$write")
            if (isinstance(write, dict) and isinstance(write.get("name"), str)
                    and write["name"].startswith("soft-state:")
                    and write.get("type") == "Neuralese<string>"
                    and isinstance(write.get("source"), str)):
                block_id = write["name"].split(":", 1)[1]
                attestation = {"body": write["source"], "writer_record_id": record.get("id"),
                               "writer_record_sha256": source_hashes.get(record.get("id")),
                               "writer_source_row_sha256": ((record.get("source_ref") or {}).get("source_row_sha256")),
                               "writer_split": record.get("split"),
                               "writer_source_groups": sorted(set(g for g in (record.get("source_groups") or [])
                                                                    if isinstance(g, str) and g)),
                               "write_name": write["name"], "body_sha256": _sha(write["source"].encode("utf-8"))}
                sources.setdefault(block_id, []).append(attestation)
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
    return sources


def _attested_neuralese_message_bodies(record, writer_sources=None, *, split, source_groups):
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
        else:
            raise ValueError(f"message neuralese body has no unique same-split, same-source hash-bound source: {block_id}")
        bodies[block_id] = f"<|neuralese|>{attestation['body']}<|/neuralese|>"
        attestations.append({key: value for key, value in attestation.items() if key != "body"}
                            | {"block_id": block_id})
    return bodies, attestations


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
    from ..train.trajectories import crisp_messages, handover_notes

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
            neuralese_bodies, context_attestations = _attested_neuralese_message_bodies(
                record, writer_sources, split=split, source_groups=groups)
            message_inputs, _ = _hydrate_tool_argument_blocks(record.get("messages") or [], neuralese_bodies)
            messages = crisp_messages(message_inputs, piece_map, notes,
                                      neuralese_bodies=neuralese_bodies)
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
                                                            for item in context_attestations]})

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
                   "neuralese_context_attestations": row["neuralese_context_attestations"]} for row in rows]
    for row,entry in zip(rows,provenance):
        entry['supervised_suffix_start']=row['supervised_suffix_start']
    omissions_bytes = "".join(_canonical(row) + "\n" for row in omitted).encode("utf-8")
    provenance_bytes = "".join(_canonical(row) + "\n" for row in provenance).encode("utf-8")
    receipt = {
        "format": "natlang.gold_text_packet_receipt/1",
        "policy": "approved SFT records only; deterministic crisp rendering from supplied pieces and explicit handover notes; exact named Neuralese reader-context blocks may be hydrated only from one approved, successful, hash-bound writer target source in the same split and with a shared source group or matching source-row hash, with attestations in provenance; hydrated context never creates a separate target row; complete source-group split retained; train copies of held complete documents excluded; target turn rendered through native chat template with serving content escaping; no tools executed",
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
