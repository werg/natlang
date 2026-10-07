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


def native_gold_document(tokenizer, messages, target, tools):
    """Use the serving renderer; content markers remain ordinary content tokens."""
    from ..serve.chat import render_messages, render_with_empty_thought, split_escaped
    def template(turns, schemas):
        return render_with_empty_thought(lambda m: tokenizer.apply_chat_template(
            m, tools=schemas or None, tokenize=False, add_generation_prompt=False), turns)
    rendered = render_messages([*messages, target], tools, template,
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
            messages = crisp_messages(record.get("messages") or [], piece_map, notes)
            target = crisp_messages([record["target"]], piece_map, notes)[0]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            omitted.append({"id": rid, "reason": "unresolved_or_malformed_crisp_reference", "detail": str(exc)[:240]})
            continue
        try:
            text, token_ids = native_gold_document(tokenizer, messages, target, record.get("tools") or [])
        except (KeyError, TypeError, ValueError) as exc:
            omitted.append({"id": rid, "reason": "native_chat_render_failed", "detail": str(exc)[:240]})
            continue
        if not text.strip():
            omitted.append({"id": rid, "reason": "empty_native_turn"})
            continue
        prepared.append({"text": text, "token_ids": token_ids, "tokenizer_sha256": fingerprint,
                         "split": split, "source_groups": groups, "id": rid})

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
        key = row["split"], row["text"], tuple(row["token_ids"])
        if key not in dedup:
            dedup[key] = {**row, "source_record_ids": [row["id"]]}
        else:
            representative = dedup[key]
            representative["source_groups"] = sorted(set(representative["source_groups"] + row["source_groups"]))
            representative["source_record_ids"].append(row["id"])
    rows = list(dedup.values())
    same_split_dupes = n_before - excluded_train_held - len(rows)
    if not any(row["split"] == "train" for row in rows) or not any(row["split"] == "test" for row in rows):
        raise ValueError("nonempty independent train and held text required")

    provenance = [{"id": row["id"], "split": row["split"], "source_groups": row["source_groups"],
                   "source_record_sha256": source_hashes[row["id"]],
                   "source_records": [{"id": rid, "sha256": source_hashes[rid]} for rid in row["source_record_ids"]],
                   "text_sha256": _sha(row["text"].encode("utf-8")),
                   "token_ids_sha256": _sha(_canonical(row["token_ids"]).encode("utf-8")),
                   "tokenizer_sha256": fingerprint} for row in rows]
    omissions_bytes = "".join(_canonical(row) + "\n" for row in omitted).encode("utf-8")
    provenance_bytes = "".join(_canonical(row) + "\n" for row in provenance).encode("utf-8")
    receipt = {
        "format": "natlang.gold_text_packet_receipt/1",
        "policy": "approved SFT records only; deterministic crisp rendering from supplied pieces and explicit handover notes; complete source-group split retained; train copies of held complete documents excluded; target turn rendered through native chat template with serving content escaping; no tools executed",
        "rendering": "natlang.native_gold_chat/1", "tokenizer_sha256": fingerprint,
        "ordinary_text_stage_only": True, "task_or_trajectory_admission_granted": False,
        "documents": len(rows), "train_documents": sum(row["split"] == "train" for row in rows),
        "test_documents": sum(row["split"] == "test" for row in rows),
        "omitted_records": len(omitted), "excluded_train_exact_held_complete_documents": excluded_train_held,
        "duplicate_same_split_documents_deduplicated": same_split_dupes,
        "unresolved_omissions": omitted,
        "omissions_jsonl_sha256": _sha(omissions_bytes),
        "provenance_jsonl_sha256": _sha(provenance_bytes),
    }
    return rows, receipt, omitted, provenance
