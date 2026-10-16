#!/usr/bin/env python3
"""Bounded conversion of selected Search/OWL port-record occurrences.

This is a static structural transform, not actor replay. Exact raw JSONL
occurrences are joined to the extraction membership ledger before either
tokenizer view is written. See plans/neuralese/SEARCH_OWL_PORT_RECORD_GRAPH_DECOMPOSITION_20261016.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
sys.path.insert(0, str(ROOT))

from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter  # noqa: E402
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision  # noqa: E402
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection  # noqa: E402
from scripts.neuralese_data.decision_holds import load_decision_holds  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file, sha256_hex as sha256_bytes  # noqa: E402
from natlang_neuralese.data.port_record_graph import port_record_graph  # noqa: E402
from natlang_neuralese.data.records import RecordError, read_record_rows  # noqa: E402
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint  # noqa: E402


def jsonl(path):
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield row


def adapter_code_binding():
    files = (Path(__file__).resolve(),
        Path(__import__("natlang_neuralese.data.port_record_graph", fromlist=["__file__"]).__file__).resolve(),
        Path(__import__("natlang_neuralese.data.native_graph_conversion", fromlist=["__file__"]).__file__).resolve(),
        Path(__import__("natlang_neuralese.serve.chat", fromlist=["__file__"]).__file__).resolve(),
        Path(__import__("natlang_neuralese.data.records", fromlist=["__file__"]).__file__).resolve(),
        Path(__file__).resolve().with_name("build_native_graph_text_corpus.py"),
        Path(__file__).resolve().with_name("build_native_semantic_pairs_e2e.py"),
        Path(__file__).resolve().with_name("decision_holds.py"))
    return [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files]


def load_membership(path):
    members = {}
    id_counts = {}
    group_splits = {}
    rows = 0
    for row in jsonl(path):
        key = (row.get("id"), row.get("sha256"))
        if (not isinstance(key[0], str) or not isinstance(key[1], str)
                or len(key[1]) != 64 or key in members
                or type(row.get("bytes")) is not int
                or not isinstance(row.get("groups"), list)
                or row.get("split") not in {"train", "validation", "test"}):
            raise ValueError("membership rows need unique exact id/content bindings")
        members[key] = row
        id_counts[key[0]] = id_counts.get(key[0], 0) + 1
        split = "test" if row["split"] in {"validation", "test"} else "train"
        for group in row["groups"]:
            if not isinstance(group, str) or not group:
                raise ValueError("membership row contains an invalid source group")
            group_splits.setdefault(group, set()).add(split)
        rows += 1
    return members, id_counts, group_splits, rows


def load_omissions(path):
    counts = {}
    rows = 0
    for row in jsonl(path):
        identifier, reason = row.get("id"), row.get("reason")
        if not isinstance(identifier, str) or not isinstance(reason, str) or not reason:
            raise ValueError("omission row needs exact source ID and reason")
        counts[identifier] = counts.get(identifier, 0) + 1
        rows += 1
    return counts, rows


def load_reviewed_targets(path, review_path, *, source_sha256):
    if path is None:
        if review_path is not None:
            raise ValueError("target review cannot be supplied without its row bindings")
        return {}
    if review_path is None:
        raise ValueError("source-curated targets require their immutable review binding")
    review = json.loads(Path(review_path).read_text())
    review_sha = sha256_file(review_path)
    if (review.get("schema") != "natlang.external-curated-sft-target-review/1"
            or review.get("source_sha256") != source_sha256
            or review.get("status") != "approved"
            or review.get("admission_class") != "source-curated-third-party-SFT"
            or review.get("oracle_verified") is not False
            or review.get("native_execution_verified") is not False
            or review.get("observed_actor_state_exchange") is not False):
        raise ValueError("curated target review must bind this source and declare its non-oracle status")
    out = {}
    for row in jsonl(path):
        raw_sha, target_sha, proof_sha = row.get("source_row_sha256"), row.get("target_sha256"), row.get("proof_sha256")
        if (not all(isinstance(value, str) and len(value) == 64 for value in (raw_sha, target_sha, proof_sha))
                or row.get("status") != "approved" or not row.get("decision_id")
                or row.get("review_kind") != review["admission_class"]
                or row.get("oracle_verified") is not False
                or row.get("native_execution_verified") is not False
                or proof_sha != review_sha):
            raise ValueError("reviewed target must bind exact row, target and immutable review decision")
        if raw_sha in out:
            raise ValueError("duplicate reviewed source-row admission")
        out[raw_sha] = {**row, "review_sha256": review_sha}
    if review.get("source_rows") != len(out):
        raise ValueError("curated target review row count differs from its exact row bindings")
    return out


def load_approved_program_groups(admission_path, index_dir):
    admission = json.loads(Path(admission_path).read_text())
    decisions = admission.get("native_decisions", {})
    if (admission.get("schema") != "natlang.search-owl-native-admission-review/1"
            or admission.get("approved") is not True
            or decisions.get("admitted") != decisions.get("cases")
            or decisions.get("approved") != decisions.get("decisions")):
        raise ValueError("Search/OWL source-group join needs the exact passed native admission review")
    groups = set()
    bindings = []
    paths = sorted(Path(index_dir).glob("*.json"))
    if not paths:
        raise ValueError("admitted Search/OWL program-group index is empty")
    for path in paths:
        index_digest = sha256_file(path)
        index = json.loads(path.read_text())
        source = index.get("source")
        if (index.get("schema") != "natlang.native-program-groups/1"
                or not isinstance(source, dict) or source.get("sha256") != path.stem
                or not isinstance(source.get("path"), str)
                or type(source.get("bytes")) is not int):
            raise ValueError(f"program-group index lacks its source binding: {path}")
        source_path = Path(source["path"])
        if (not source_path.is_file() or source_path.stat().st_size != source["bytes"]
                or sha256_file(source_path) != source["sha256"]):
            raise ValueError(f"program-group index source bytes fail their immutable binding: {source_path}")
        for values in index.get("programs", {}).values():
            for group in values:
                if isinstance(group, str) and group.startswith("expansion-d02-task:"):
                    groups.add(group)
        bindings.append({"path": str(path), "sha256": index_digest,
            "source_path": str(source_path), "source_sha256": source["sha256"],
            "source_bytes": source["bytes"]})
    return groups, {"path": str(admission_path), "sha256": sha256_file(admission_path),
        "schema": admission["schema"], "approved_cases": decisions["admitted"],
        "approved_decisions": decisions["approved"], "program_group_index": bindings,
        "matched_expansion_task_groups": len(groups)}


def load_hold_details(paths):
    """Use the shared hold-ID loader and retain its exact reason/source bindings."""
    identities, bindings = load_decision_holds(paths)
    details = {}
    for path in paths:
        for row in jsonl(path):
            if row["id"] in details:
                raise ValueError(f"duplicate current decision-hold ID across bound files: {row['id']}")
            details[row["id"]] = {"id": row["id"], "reason": row["reason"],
                "path": str(path), "source_row_sha256": row.get("source_row_sha256"),
                "source_corpus": row.get("source_corpus")}
    if set(details) != identities:
        raise ValueError("decision hold details differ from shared exact ID loader")
    for binding in bindings:
        if sha256_file(binding["path"]) != binding["sha256"]:
            raise ValueError(f"decision hold file changed while loading details: {binding['path']}")
    return identities, details, bindings


def _hold_references(value):
    source_ids, target_ids = set(), set()
    def visit(item, *, top=False):
        if isinstance(item, dict):
            for key, child in item.items():
                if isinstance(child, str):
                    if (top and key == "id") or key == "teacher_trajectory_id":
                        source_ids.add(child)
                    elif key == "decision_id":
                        target_ids.add(child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value, top=True)
    return source_ids, target_ids


def decision_hold_policy(row, authorization, hold_ids, hold_details):
    """Bind current source/trajectory holds and exact consumer-target holds."""
    raw_sha = sha256_bytes(row.raw_line)
    source_ids, raw_target_ids = _hold_references(row.raw)
    source_hits = sorted(({row.record.id} | source_ids) & hold_ids)
    target_ids = raw_target_ids & hold_ids
    reviewed_decision = authorization.get("decision_id") if authorization else None
    if isinstance(reviewed_decision, str) and reviewed_decision in hold_ids:
        target_ids.add(reviewed_decision)
    source_hash_ids = {identifier for identifier, detail in hold_details.items()
                       if detail.get("source_row_sha256") == raw_sha}
    target_ids.update(source_hash_ids)
    exclusions = tuple(hold_details[item] for item in sorted(target_ids))
    return source_hits, exclusions, sorted(target_ids - source_hash_ids), sorted(source_hash_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--membership-sha256", required=True)
    parser.add_argument("--omissions", type=Path, required=True)
    parser.add_argument("--omissions-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--protection-manifest", type=Path, required=True)
    parser.add_argument("--admission-review", type=Path, required=True)
    parser.add_argument("--program-group-index", type=Path, required=True)
    parser.add_argument("--reviewed-targets", type=Path)
    parser.add_argument("--target-review", type=Path)
    parser.add_argument("--decision-hold", type=Path, action="append",
        help="current exact decision-hold JSONL; repeat for each bound hold file (default: sorted current files)")
    parser.add_argument("--context-weight", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code_binding = adapter_code_binding()

    if sha256_file(args.source) != args.source_sha256:
        raise ValueError("selected port-record source SHA differs from its immutable binding")
    for path, expected, name in ((args.membership, args.membership_sha256, "membership"),
                                 (args.omissions, args.omissions_sha256, "omissions")):
        if sha256_file(path) != expected:
            raise ValueError(f"{name} sidecar SHA differs from its immutable binding")
    protection = current_protection(args.protection_manifest)
    hold_paths = args.decision_hold or sorted((ROOT / "training" / "decision-holds").glob("*.jsonl"))
    decision_hold_ids, decision_hold_details, decision_hold_bindings = load_hold_details(hold_paths)
    admitted_program_groups, program_group_binding = load_approved_program_groups(
        args.admission_review, args.program_group_index)
    reviewed = load_reviewed_targets(args.reviewed_targets, args.target_review,
                                     source_sha256=args.source_sha256)
    members, id_counts, group_splits, member_rows = load_membership(args.membership)
    omission_ids, omission_rows = load_omissions(args.omissions)
    if member_rows != 7027 or omission_rows != 7002 or len(omission_ids) != 7001:
        raise ValueError("selected over-cap membership changed from the exact 7,027/7,002/7,001 receipt")
    if set(omission_ids) != set(id_counts):
        raise ValueError("original over-cap omission IDs do not exactly match selected source-row IDs")
    member_hashes = {source_sha for _, source_sha in members}
    if reviewed and set(reviewed) != member_hashes:
        raise ValueError("curated target review does not bind every selected exact source occurrence")

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_tokenizer), trust_remote_code=False,
                                               local_files_only=True)
    source_fingerprint = tokenizer_fingerprint(tokenizer)
    role_ids = provision(tokenizer)
    expanded_fingerprint = tokenizer_fingerprint(tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    writers = {}
    omissions_writer = args.output / "omissions.jsonl"
    omissions_stream = None
    seen, train_groups, test_groups = set(), set(), set()
    totals = {"rows": 0, "producer_payload_tokens": 0, "consumer_positive_rows": 0,
              "consumer_positive_tokens": 0, "consumer_context_only_rows": 0,
              "consumer_context_only_tokens": 0, "source_group_admitted_rows": 0,
              "source_program_group_joined_rows": 0, "source_program_group_unmatched_rows": 0,
              "protected_train_rows_moved_to_test": 0,
              "decision_hold_source_id_intersections": 0,
              "decision_hold_target_decision_intersections": 0,
              "decision_hold_raw_source_sha_intersections": 0,
              "decision_hold_zero_weight_target_rows": 0,
              "protected_quality_or_pending_omissions": 0,
              "static_graph_transform_omissions": 0}
    outcome_labels = {}
    omissions_count = 0
    try:
        for split in ("train", "test"):
            writers[split] = GraphWriter(args.output / split / "native-graphs.jsonl.zst")
        omissions_stream = omissions_writer.open("x", encoding="utf-8")
        for row in read_record_rows([args.source], strict=True):
            raw_sha = sha256_bytes(row.raw_line)
            key = (row.record.id, raw_sha)
            member = members.get(key)
            if member is None:
                raise ValueError(f"source occurrence is absent from extraction membership: {row.record.id}")
            if key in seen:
                raise ValueError(f"selected source occurrence repeated unexpectedly: {row.record.id}")
            seen.add(key)
            if (len(row.raw_line) != member["bytes"]
                    or row.record.split != member["split"]
                    or sorted(set(row.record.split_groups)) != sorted(set(member["groups"]))):
                raise ValueError(f"selected occurrence disagrees with exact extraction membership: {row.record.id}")
            if row.record.id not in omission_ids:
                raise ValueError(f"selected occurrence has no original over-cap omission: {row.record.id}")
            totals["rows"] += 1
            outcome_labels[row.record.outcome_label] = outcome_labels.get(row.record.outcome_label, 0) + 1
            expansion_groups = {group for group in row.record.split_groups
                                if group.startswith("expansion-d02-task:")}
            if not expansion_groups & admitted_program_groups:
                omissions_stream.write(json.dumps({"id": row.record.id, "source_row_sha256": raw_sha,
                    "reason": "no-source-group-join-to-approved-searchowl-native-program",
                    "source_groups": sorted(expansion_groups)}, ensure_ascii=False, separators=(",", ":")) + "\n")
                omissions_count += 1
                totals["source_program_group_unmatched_rows"] += 1
                continue
            totals["source_program_group_joined_rows"] += 1
            groups = set(row.record.split_groups)
            quality_or_pending = groups & (protection["quality"] | protection["pending"])
            if quality_or_pending:
                omissions_stream.write(json.dumps({"id": row.record.id, "source_row_sha256": raw_sha,
                    "reason": "current-quality-or-pending-source-review-protection",
                    "source_groups": sorted(quality_or_pending)}, ensure_ascii=False, separators=(",", ":")) + "\n")
                omissions_count += 1
                totals["protected_quality_or_pending_omissions"] += 1
                continue
            effective_split = "test" if row.record.split in {"validation", "test"} else "train"
            if effective_split == "train" and (groups & protection["held"] or any(
                    group_splits.get(group) == {"train", "test"} for group in groups)):
                effective_split = "test"
                totals["protected_train_rows_moved_to_test"] += 1
            authorization = reviewed.get(raw_sha)
            if authorization is None and row.record.is_imitation_target:
                from natlang_neuralese.data.port_record_graph import consumer_target_authorization
                authorization = consumer_target_authorization(row)
            source_hold_ids, exact_target_exclusions, target_decision_ids, raw_hash_hold_ids = decision_hold_policy(
                row, authorization, decision_hold_ids, decision_hold_details)
            totals["decision_hold_source_id_intersections"] += len(source_hold_ids)
            totals["decision_hold_target_decision_intersections"] += len(target_decision_ids)
            totals["decision_hold_raw_source_sha_intersections"] += len(raw_hash_hold_ids)
            if exact_target_exclusions:
                totals["decision_hold_zero_weight_target_rows"] += 1
            if source_hold_ids:
                effective_split = "test"
            try:
                graph = port_record_graph(row, source_tokenizer=tokenizer,
                    source_tokenizer_fingerprint=source_fingerprint,
                    expanded_tokenizer_fingerprint=expanded_fingerprint, role_ids=role_ids,
                    corpus_id=args.corpus_id, source_file_sha256=args.source_sha256,
                    consumer_admission=authorization, context_weight=args.context_weight,
                    effective_split=effective_split,
                    decision_hold_exclusions=exact_target_exclusions)
            except (ValueError, RecordError) as exc:
                omissions_stream.write(json.dumps({"id": row.record.id, "source_row_sha256": raw_sha,
                    "reason": "static-native-graph-transform-failed",
                    "error_type": type(exc).__name__, "detail": str(exc),
                    "source_groups": sorted(groups)}, ensure_ascii=False, separators=(",", ":")) + "\n")
                omissions_count += 1
                totals["static_graph_transform_omissions"] += 1
                continue
            if source_hold_ids:
                graph["conversion"]["decision_hold_source_exclusions"] = [
                    decision_hold_details[identifier] for identifier in source_hold_ids]
            positive = graph["conversion"]["consumer_target_positive_tokens"] > 0
            if positive:
                totals["consumer_positive_rows"] += 1
                totals["consumer_positive_tokens"] += graph["conversion"]["consumer_target_positive_tokens"]
            else:
                totals["consumer_context_only_rows"] += 1
                totals["consumer_context_only_tokens"] += graph["conversion"]["consumer_target_context_tokens"]
            totals["producer_payload_tokens"] += graph["conversion"]["source_payload_tokens"]
            writer = writers[effective_split]
            writer.write(graph)
            target_groups = train_groups if effective_split == "train" else test_groups
            target_groups.update(graph["source_groups"])
            if set(graph["source_groups"]) & protection["held"] and effective_split == "train":
                raise ValueError("training graph source groups intersect current held protection")
            if set(graph["source_groups"]) & (protection["quality"] | protection["pending"]):
                raise ValueError("unexpected protected group survived the quality/pending exclusion")
        if seen != set(members):
            raise ValueError(f"source occurrence coverage differs from membership: seen={len(seen)} expected={len(members)}")
        if train_groups & test_groups:
            raise ValueError("native train and held source groups overlap")
        summaries = [writers[split].finish() for split in ("train", "test")]
        omissions_stream.flush()
        omissions_stream.close()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        if omissions_stream is not None and not omissions_stream.closed:
            omissions_stream.close()
        raise
    receipt = {"schema": "natlang.searchowl-port-record-native-build/1", "model_id": args.model_id,
        "adapter_code": code_binding,
        "source": {"path": str(args.source), "sha256": args.source_sha256, "rows": totals["rows"]},
        "membership": {"path": str(args.membership), "sha256": args.membership_sha256,
                       "rows": member_rows, "unique_ids": len(id_counts), "duplicate_id_occurrences": member_rows-len(id_counts)},
        "original_omissions": {"path": str(args.omissions), "sha256": args.omissions_sha256,
                               "rows": omission_rows, "unique_ids": len(omission_ids)},
        "source_tokenizer_fingerprint": source_fingerprint,
        "expanded_tokenizer_fingerprint": expanded_fingerprint, "role_token_ids": list(role_ids),
        "current_protection": protection["proof"], "outcomes": outcome_labels,
        "current_decision_holds": {"bindings": decision_hold_bindings,
            "unique_ids": len(decision_hold_ids),
            "matched_source_ids": totals["decision_hold_source_id_intersections"],
            "matched_target_decision_ids": totals["decision_hold_target_decision_intersections"],
            "matched_raw_source_hashes": totals["decision_hold_raw_source_sha_intersections"],
            "zero_weighted_target_rows": totals["decision_hold_zero_weight_target_rows"]},
        "approved_searchowl_source_group_join": program_group_binding,
        "consumer_target_policy": "S1 gold/checked labels or exact source-row/target/decision/review binding; source-curated targets are static third-party SFT, not oracle/native-verified",
        "target_review": (None if args.target_review is None else {
            "path": str(args.target_review), "sha256": sha256_file(args.target_review),
            "reviewed_targets_path": str(args.reviewed_targets),
            "reviewed_targets_sha256": sha256_file(args.reviewed_targets),
            "schema": "natlang.external-curated-sft-target-review/1",
            "oracle_verified": False, "native_execution_verified": False}),
        "totals": totals, "omission_rows": omissions_count,
        "omissions": {"path": str(omissions_writer), "sha256": sha256_file(omissions_writer),
                      "rows": omissions_count, "bytes": omissions_writer.stat().st_size},
        "outputs": summaries,
        "train_groups": len(train_groups), "held_groups": len(test_groups),
        "static_transformation": "source text is a synthetic producer payload imported by the consumer; no observed actor edge is claimed"}
    args.output.mkdir(parents=True, exist_ok=True)
    temp = args.output / ".receipt.json.tmp"
    temp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    with temp.open("rb") as stream:
        import os
        os.fsync(stream.fileno())
    temp.replace(args.output / "receipt.json")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
