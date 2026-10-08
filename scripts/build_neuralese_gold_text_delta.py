#!/usr/bin/env python3
"""Derive source-reviewed native-action additions to an existing gold-text packet.

The base packet is copied byte-for-byte. A hash-bound source selection receipt
specifies the exact delta IDs; it does not itself grant cohort admission. Delta records are rendered with the
shared ``gold_text_rows`` contract and the caller-selected tokenizer/package
snapshot; exact existing documents are accounted for without rewriting the base.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def read_records(path):
    out = []
    with path.open("rb") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            row["_source_record_sha256"] = sha(line.rstrip(b"\r\n"))
            out.append(row)
    return out

def signature(row):
    value = [row["split"], row["text"], row["token_ids"], row["supervised_suffix_start"]]
    return sha(canonical(value).encode())

def content_signature(row):
    value = [row["text"], row["token_ids"], row["supervised_suffix_start"]]
    return sha(canonical(value).encode())

def anchor_qualification(anchor, rendered, omissions, helper_receipt):
    """Qualify the held base anchor independently from selected delta rows."""
    anchor_id = anchor.get("id")
    anchor_omission = next((item for item in omissions if item.get("id") == anchor_id), None)
    if helper_receipt.get("excluded_train_exact_held_complete_documents") != 0:
        return {"qualified": False, "reason": "train_document_collides_with_held_content",
                "excluded_train_exact_held_complete_documents": helper_receipt.get(
                    "excluded_train_exact_held_complete_documents")}
    if anchor_omission is not None:
        return {"qualified": False, "reason": "anchor_omitted_by_shared_text_renderer",
                "omission": anchor_omission}
    rendered_ids = {source_id for item in rendered
                    for source_id in item.get("source_record_ids", [item.get("id")])}
    if anchor_id not in rendered_ids:
        return {"qualified": False, "reason": "anchor_not_present_in_rendered_rows"}
    return {"qualified": True, "reason": "rendered_as_held_test_anchor"}

def anchor_complexity(record):
    """Try self-contained ordinary held records before reference-heavy anchors."""
    def refs(value):
        if isinstance(value, dict):
            own = isinstance(value.get("type"), str) and value["type"] in {"read", "soft", "neuralese"}
            return int(own) + sum(refs(child) for child in value.values())
        if isinstance(value, list): return sum(refs(child) for child in value)
        return 0
    messages, target = record.get("messages") or [], record.get("target") or {}
    return (refs(messages) + refs(target), len(canonical(messages)) + len(canonical(target)), record.get("id", ""))

def selected_delta_omissions(omissions, delta_ids):
    return [item for item in omissions if item.get("id") in delta_ids]

def append_prefix(prefix: Path, output: Path, additions: list[dict]):
    with output.open("xb") as f:
        with prefix.open("rb") as src:
            last = b""
            for block in iter(lambda: src.read(1 << 20), b""):
                f.write(block); last = block[-1:]
        if prefix.stat().st_size and last != b"\n": f.write(b"\n")
        for row in additions: f.write((canonical(row) + "\n").encode())
    if sha_prefix(output, prefix.stat().st_size) != sha_file(prefix):
        raise ValueError(f"base prefix changed: {prefix}")

def sha_prefix(path: Path, length: int) -> str:
    h = hashlib.sha256(); remaining = length
    with path.open("rb") as f:
        while remaining:
            block = f.read(min(1 << 20, remaining))
            if not block: raise ValueError("unexpected EOF during prefix check")
            h.update(block); remaining -= len(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base-text", "base-provenance", "base-records", "base-root-receipt",
                 "delta-records", "pieces", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--base-pieces", type=Path,
                   help="approved base native piece catalog (defaults to native-pieces.jsonl beside --base-records)")
    p.add_argument("--source-selection", "--source-approval", dest="source_approval", required=True, type=Path,
                   help="hash-bound source/action selection; this is not root cohort admission")
    p.add_argument("--tokenizer", required=True, help="Pinned tokenizer ID or local snapshot")
    p.add_argument("--twin-of-text", type=Path,
                   help="the root-approved text whose tokenizer twin --base-text is: the receipt must bind it, and "
                        "--base-text must carry the identical document ID/split/source-group/source-record sequence")
    p.add_argument("--renderer-package-root", type=Path,
                   help="Path containing natlang_neuralese/; use the reviewed renderer snapshot")
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()): raise ValueError(f"output is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    package = (args.renderer_package_root or (ROOT / "training/neuralese")).resolve()
    sys.path.insert(0, str(package))
    from natlang_neuralese.data.text_corpus import gold_text_rows
    from transformers import AutoTokenizer

    base_records = read_records(args.base_records)
    delta_records = read_records(args.delta_records)
    base_approval = json.loads(args.base_root_receipt.read_text())
    approved_text = args.twin_of_text or args.base_text
    def manifest_binds(text):  # a root admission that binds the packet's output manifest, which binds the text
        manifest = text.parent / "output-manifest.json"
        return (manifest.is_file() and base_approval.get("output_manifest_sha256") == sha_file(manifest) and
                json.loads(manifest.read_text())["outputs"]["text.jsonl"]["sha256"] == sha_file(text))
    if base_approval.get("approved") is not True or (base_approval.get("text_sha256") != sha_file(approved_text)
                                                     and not manifest_binds(approved_text)):
        raise ValueError("base text root receipt does not approve/bind the exact text prefix")
    if args.twin_of_text:
        def identity(path):  # tokenizer-independent document identity: ID, split and the admitted source records
            with path.open("r", encoding="utf-8") as f:
                return [(r["id"], r["split"], r.get("source_groups"), r.get("source_record_ids"))
                        for r in map(json.loads, filter(str.strip, f))]
        if identity(args.twin_of_text) != identity(args.base_text):
            raise ValueError("tokenizer twin does not carry the approved text's document IDs, splits, source groups "
                             "and source record IDs in order")
    source_approval = json.loads(args.source_approval.read_text())
    approved_ids = source_approval.get("approved_row_ids")
    delta_ids = {r.get("id") for r in delta_records}
    if not isinstance(approved_ids, list) or set(approved_ids) != delta_ids or len(approved_ids) != len(delta_ids):
        raise ValueError("delta records do not equal source approval IDs")
    for rel, expected in source_approval.get("artifact_hashes", {}).items():
        bound = (ROOT / rel).resolve()
        if not bound.is_file() or sha_file(bound) != expected:
            raise ValueError(f"source approval artifact missing/hash mismatch: {rel}")
    def iter_jsonl(path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): yield json.loads(line)
    base_rows = iter_jsonl(args.base_text)
    base_keys, by_key, held_content = set(), {}, {}
    for row in base_rows:
        key = signature(row)
        base_keys.add(key)
        by_key.setdefault(key, {"id": row.get("id"), "split": row["split"],
                                "source_record_ids": row.get("source_record_ids", [row.get("id")])})
        if row["split"] == "test": held_content.setdefault(content_signature(row), by_key[key])
    anchors = [r for r in base_records if r.get("split") == "test" and
               r.get("training_admission", {}).get("approved") is True]
    if not anchors: raise ValueError("base native records have no approved test anchor")
    anchors.sort(key=anchor_complexity)
    base_pieces_path = args.base_pieces or (args.base_records.parent / "native-pieces.jsonl")
    base_pieces = list(iter_jsonl(base_pieces_path)) if base_pieces_path.is_file() else []
    delta_pieces = list(iter_jsonl(args.pieces))
    pieces_by_name = {}
    for piece in [*base_pieces, *delta_pieces]:
        name, text = piece.get("name"), piece.get("text")
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError("base or delta piece catalog contains a malformed piece")
        if name in pieces_by_name and pieces_by_name[name] != text:
            raise ValueError(f"base and delta piece text conflicts for {name}")
        pieces_by_name[name] = text
    pieces = [{"name": name, "text": text} for name, text in pieces_by_name.items()]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rendered = omissions = provenance = helper_receipt = None
    anchor_attempts = []
    # A delta document may exactly match the test anchor. Select a deterministic
    # different held record so the shared helper's held-duplicate filter applies
    # only to the real, previously derived base packet.
    for anchor in anchors:
        rendered, helper_receipt, omissions, provenance = gold_text_rows(
            [*delta_records, anchor], pieces, tokenizer=tokenizer)
        qualification = anchor_qualification(anchor, rendered, omissions, helper_receipt)
        anchor_attempts.append({"id": anchor.get("id"),
                                "source_record_sha256": anchor.get("_source_record_sha256"),
                                **qualification})
        if qualification["qualified"]:
            break
    else:
        raise ValueError("no approved base test anchor survived shared rendering; see anchor-selection diagnostics")
    if helper_receipt["tokenizer_sha256"] != json.loads((args.base_text.parent / "receipt.json").read_text())["tokenizer_sha256"]:
        raise ValueError("tokenizer fingerprint differs from base packet")

    additions, coverage = [], []
    delta_omissions = selected_delta_omissions(omissions, delta_ids)
    omission_by_id = {r["id"]: r for r in delta_omissions}
    # Rows are already de-duplicated by the shared gold-text helper.
    for row in rendered:
        source_ids = [sid for sid in row.get("source_record_ids", [row.get("id")]) if sid in delta_ids]
        if not source_ids: continue  # the held-test anchor exists to validate split context only
        key = signature(row)
        if key in base_keys:
            existing = by_key[key]
            status = "matches_existing_v15_document"
            destination = existing
        elif row["split"] == "train" and content_signature(row) in held_content:
            status = "excluded_exact_held_document"
            destination = held_content[content_signature(row)]
        else:
            additions.append(row)
            destination = {"id": row["id"], "source_record_ids": row["source_record_ids"], "split": row["split"]}
            status = "appended_new_document"
        for rid in source_ids:
            original = next(r for r in delta_records if r["id"] == rid)
            coverage.append({"record_id": rid, "source_ids": original.get("source_ids", []),
                             "status": status, "document_signature_sha256": key,
                             "destination": destination})
    covered = {c["record_id"] for c in coverage}
    for rid, omission in omission_by_id.items():
        if rid not in delta_ids:
            continue  # the temporary held-test anchor is not part of this delta
        original = next(r for r in delta_records if r["id"] == rid)
        coverage.append({"record_id": rid, "source_ids": original.get("source_ids", []),
                         "status": "omitted_by_shared_text_renderer", "reason": omission,
                         "source_record_sha256": original["_source_record_sha256"]})
        covered.add(rid)
    if covered != delta_ids:
        raise ValueError(f"delta record coverage incomplete: {sorted(delta_ids-covered)}")
    coverage.sort(key=lambda x: x["record_id"])
    anchor_selection = {"schema": "natlang.gold-text-delta-anchor-selection/1",
                        "selected_anchor_id": anchor.get("id"),
                        "attempts": anchor_attempts,
                        "omissions_are_not_delta_coverage": True}
    (args.out / "anchor-selection.json").write_text(json.dumps(anchor_selection, indent=2,
        ensure_ascii=False, sort_keys=True)+"\n")
    append_prefix(args.base_text, args.out / "text.jsonl", additions)
    add_provenance = [x for x in provenance if x.get("id") in delta_ids]
    append_prefix(args.base_provenance, args.out / "provenance.jsonl", add_provenance)
    base_omissions = args.base_text.parent / "omissions.jsonl"
    append_prefix(base_omissions, args.out / "omissions.jsonl", delta_omissions)
    coverage_bytes = "".join(canonical(x)+"\n" for x in coverage).encode()
    (args.out / "source-coverage.jsonl").write_bytes(coverage_bytes)
    old_receipt = json.loads((args.base_text.parent / "receipt.json").read_text())
    receipt = dict(old_receipt)
    base_same_split = sum(c["status"] == "matches_existing_v15_document" for c in coverage)
    base_held = sum(c["status"] == "excluded_exact_held_document" for c in coverage)
    receipt.update({"status": "held cumulative ordinary-text proposal; root review pending",
                    "policy": helper_receipt["policy"],
                    "rendering": helper_receipt["rendering"],
                    "supervision": helper_receipt["supervision"],
                    "ordinary_text_stage_only": True,
                    "base_renderer_code": old_receipt.get("renderer_code", {}),
                    "documents": old_receipt["documents"] + len(additions),
                    "train_documents": old_receipt["train_documents"] + sum(x["split"] == "train" for x in additions),
                    "test_documents": old_receipt["test_documents"] + sum(x["split"] == "test" for x in additions),
                    "source_text_prefix_sha256": sha_file(args.base_text),
                    "source_text_prefix_bytes": args.base_text.stat().st_size,
                    "delta_source_run": str(args.delta_records.parent),
                    "delta_record_count": len(delta_records), "delta_appended_document_count": len(additions),
                    "delta_hash_bound_reader_context_blocks": helper_receipt.get("hash_bound_reader_context_blocks", 0),
                    "delta_capture_context_augmentation_records": sum(
                        bool(row.get("capture_context_augmentation_attestations")) for row in provenance
                        if row.get("id") in delta_ids),
                    "delta_capture_context_augmentation_messages": sum(
                        len(row.get("capture_context_augmentation_attestations", [])) for row in provenance
                        if row.get("id") in delta_ids),
                    "delta_typed_eval_finish_marker_calls": sum(
                        call.get("neuralese_code", {}).get("schema") == "natlang.neuralese-code/1"
                        for record in delta_records for call in (record.get("target") or {}).get("tool_calls", [])),
                    "omitted_records": old_receipt.get("omitted_records", 0) + len(delta_omissions),
                    "unresolved_omissions": old_receipt.get("unresolved_omissions", []) + delta_omissions,
                    "duplicate_same_split_documents_deduplicated": old_receipt.get("duplicate_same_split_documents_deduplicated", 0) +
                        helper_receipt["duplicate_same_split_documents_deduplicated"] + base_same_split,
                    "excluded_train_exact_held_complete_documents": old_receipt.get("excluded_train_exact_held_complete_documents", 0) +
                        helper_receipt["excluded_train_exact_held_complete_documents"] + base_held,
                    "delta_source_coverage_sha256": sha(coverage_bytes),
                    "anchor_selection": anchor_selection,
                    "text_jsonl_sha256": sha_file(args.out / "text.jsonl"),
                    "provenance_jsonl_sha256": sha_file(args.out / "provenance.jsonl"),
                    "omissions_jsonl_sha256": sha_file(args.out / "omissions.jsonl"),
                    "source_files": {"delta_records": sha_file(args.delta_records),
                                     "base_pieces": sha_file(base_pieces_path) if base_pieces_path.is_file() else None,
                                     "pieces": sha_file(args.pieces),
                                     "source_approval": sha_file(args.source_approval),
                                     "base_text_root_receipt": sha_file(args.base_root_receipt)},
                    "renderer_code": {"text_corpus.py": sha((package / "natlang_neuralese/data/text_corpus.py").read_bytes()),
                                      "chat.py": sha((package / "natlang_neuralese/serve/chat.py").read_bytes()),
                                      "trajectories.py": sha((package / "natlang_neuralese/train/trajectories.py").read_bytes()),
                                      "inline_instructions.py": sha((package / "natlang_neuralese/train/inline_instructions.py").read_bytes()),
                                      "delta_builder.py": sha(Path(__file__).read_bytes())},
                    "task_or_trajectory_admission_granted": False})
    if args.twin_of_text:
        receipt["tokenizer_twin_of"] = {"text_sha256": sha_file(args.twin_of_text), "path": str(args.twin_of_text)}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True)+"\n")
    packet = {"schema": "natlang.native-gold-text-delta-proposal/1",
              "status": "held; ordinary text only; independent review required",
              "base_packet": {"text_sha256": receipt["source_text_prefix_sha256"],
                              "text_bytes": receipt["source_text_prefix_bytes"],
                              "document_count": old_receipt["documents"]},
              "delta_source": {"records_sha256": sha_file(args.delta_records),
                               "records": len(delta_records), "piece_sha256": sha_file(args.pieces),
                               "source_selection_sha256": sha_file(args.source_approval)},
              "coverage": {s: sum(c["status"] == s for c in coverage)
                           for s in sorted({c["status"] for c in coverage})},
              "anchor_selection_sha256": sha_file(args.out/"anchor-selection.json"),
              "outputs": {name: {"sha256": sha_file(args.out/name), "bytes": (args.out/name).stat().st_size}
                          for name in ("text.jsonl", "provenance.jsonl", "omissions.jsonl", "source-coverage.jsonl",
                                       "receipt.json", "anchor-selection.json")},
              "training_admission": False, "task_or_trajectory_admission": False}
    (args.out / "packet-manifest.json").write_text(json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True)+"\n")
    hydrated_contexts = sum(len(item.get("neuralese_context_attestations", [])) for item in provenance)
    (args.out / "README.md").write_text(
        "# Held ordinary gold-text proposal\n\n"
        "This is a text-only proposal for independent root review. Its `text.jsonl` and `provenance.jsonl` preserve the supplied admitted text packet as exact byte prefixes. "
        "The selected delta was rendered with the shared `gold_text_rows` helper and the pinned tokenizer fingerprint recorded in `receipt.json`; no model generation or tools were used.\n\n"
        f"The base prefix contains {old_receipt['documents']} documents; this proposal appends {len(additions)} new documents from {len(delta_records)} selected source-reviewed native records. "
        f"{hydrated_contexts} exact named Neuralese reader-context blocks were hydrated from successful, hash-bound writer target sources in the same split and source group; attestations are recorded in `provenance.jsonl`. "
        "The current delta renderer records typed eval-finish marker sidecars and separately labeled full capture context augmentations. Those augmentations come from authenticated same-invocation snapshots and do not claim the omitted text was historically provider-visible. "
        "Hydrated blocks and capture augmentations add context only; they do not create separate target rows. "
        f"{receipt['omitted_records']} records remain unresolved omissions; see `source-coverage.jsonl` and `omissions.jsonl`. "
        "The base held-test anchor qualification is recorded separately in `anchor-selection.json`; anchor omissions do not appear as selected-delta omissions. "
        "The proposal does not grant text packet admission, task/trajectory admission, model qualification, or training authorization.\n"
    )
    output_names = ["text.jsonl", "provenance.jsonl", "omissions.jsonl", "source-coverage.jsonl",
                    "receipt.json", "packet-manifest.json", "README.md", "anchor-selection.json"]
    output_manifest = {"schema": "natlang.gold-text-delta-output-manifest/1",
                       "packet_manifest_sha256": sha_file(args.out/"packet-manifest.json"),
                       "outputs": {name: {"sha256": sha_file(args.out/name), "bytes": (args.out/name).stat().st_size}
                                   for name in output_names}}
    (args.out/"output-manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"base_documents": old_receipt["documents"], "appended_documents": len(additions),
                      "cumulative_documents": receipt["documents"], "delta_records": len(delta_records),
                      "coverage_statuses": {s: sum(c["status"] == s for c in coverage)
                                            for s in sorted({c["status"] for c in coverage})},
                      "tokenizer_sha256": helper_receipt["tokenizer_sha256"],
                      "text_sha256": receipt["text_jsonl_sha256"]}, indent=2))

if __name__ == "__main__": main()
