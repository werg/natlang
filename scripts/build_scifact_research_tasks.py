#!/usr/bin/env python3
"""Prepare provenance-pinned SciFact claim/evidence candidates; this does not admit training rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

REPOSITORY = "https://github.com/allenai/scifact"
DATA_ARCHIVE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_SCHEMA_URL = "https://github.com/allenai/scifact/blob/master/doc/data.md"
LICENSE_URL = "https://github.com/allenai/scifact/blob/master/LICENSE.md"
ORACLE_LABEL_REFERENCE = "https://github.com/allenai/scifact/blob/master/verisci/inference/label_prediction/oracle.py"
CLAIMS_LICENSE = "CC-BY-4.0"
ABSTRACTS_LICENSE = "ODC-By-1.0"
DATASET_CITATION = "Wadden et al., Fact or Fiction: Verifying Scientific Claims, EMNLP 2020"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256(payload)


def _read_jsonl(raw: bytes, label: str) -> list[tuple[int, dict[str, Any], str]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    rows: list[tuple[int, dict[str, Any], str]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label}:{line_number}: expected a JSON object")
        rows.append((line_number, value, sha256(line.encode("utf-8"))))
    return rows


def _valid_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            lo, hi = sorted((a, b))
            self.parent[hi] = lo

    def components(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for item in sorted(self.parent):
            result.setdefault(self.find(item), []).append(item)
        return result


def prepare_scifact_bytes(
    corpus_bytes: bytes,
    claims_train_bytes: bytes,
    *,
    source_revision: str,
    source_archive_sha256: str | None = None,
    claims_dev_bytes: bytes | None = None,
    include_dev_validation: bool = False,
    corpus_path: str = "corpus.jsonl",
    claims_train_path: str = "claims_train.jsonl",
    claims_dev_path: str | None = None,
) -> dict[str, Any]:
    """Validate local SciFact JSONL bytes and return candidate and held rows plus manifest."""
    if not source_revision.strip():
        raise ValueError("an explicit upstream source revision is required")
    if source_archive_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", source_archive_sha256):
        raise ValueError("source_archive_sha256 must be a lowercase 64-character SHA-256")
    if claims_dev_bytes is not None and not include_dev_validation:
        raise ValueError("dev input was supplied; pass include_dev_validation=True to use it as validation")
    if include_dev_validation and claims_dev_bytes is None:
        raise ValueError("include_dev_validation requires a claims_dev input")

    corpus_rows = _read_jsonl(corpus_bytes, corpus_path)
    train_rows = _read_jsonl(claims_train_bytes, claims_train_path)
    dev_rows = _read_jsonl(claims_dev_bytes, claims_dev_path or "claims_dev.jsonl") if include_dev_validation and claims_dev_bytes is not None else []

    corpus: dict[str, dict[str, Any]] = {}
    conflicting_doc_ids: set[str] = set()
    duplicate_doc_rows = 0
    for line_number, doc, line_sha in corpus_rows:
        if not _valid_int(doc.get("doc_id")) or doc["doc_id"] < 0 or not isinstance(doc.get("title"), str) or not isinstance(doc.get("abstract"), list) or \
                any(not isinstance(sentence, str) for sentence in doc["abstract"]) or not isinstance(doc.get("structured"), bool):
            raise ValueError(f"{corpus_path}:{line_number}: invalid corpus document schema")
        doc_id = str(doc["doc_id"])
        normalized = {"doc_id": doc["doc_id"], "title": doc["title"], "abstract": doc["abstract"], "structured": doc["structured"]}
        if doc_id in corpus:
            duplicate_doc_rows += 1
            corpus[doc_id]["source_rows"].append({"corpus_line": line_number, "corpus_line_sha256": line_sha})
            if canonical_sha(corpus[doc_id]["document"]) != canonical_sha(normalized):
                conflicting_doc_ids.add(doc_id)
        else:
            corpus[doc_id] = {"document": normalized,
                              "source_rows": [{"corpus_line": line_number, "corpus_line_sha256": line_sha}]}

    input_hashes = {"corpus": sha256(corpus_bytes), "claims_train": sha256(claims_train_bytes)}
    if include_dev_validation and claims_dev_bytes is not None:
        input_hashes["claims_dev"] = sha256(claims_dev_bytes)
    sources: list[dict[str, Any]] = []
    for role, rows in (("train", train_rows), ("validation", dev_rows)):
        for line_number, claim, line_sha in rows:
            sources.append({"role": role, "line": line_number, "line_sha256": line_sha, "claim": claim,
                            "claim_sha256": canonical_sha(claim), "issues": []})

    # Keep every source row. Repeated identical claims share identity/groups and get a stable row ordinal.
    claim_ids: dict[str, list[dict[str, Any]]] = {}
    union = _UnionFind()
    for row_index, source in enumerate(sources):
        claim = source["claim"]
        raw_id = claim.get("id")
        if not _valid_int(raw_id):
            source["issues"].append("invalid_claim_id")
            source["claim_node"] = f"invalid-claim-row:{source['role']}:{source['line']}:{row_index}"
        else:
            source["claim_id"] = raw_id
            source["claim_node"] = f"claim:{raw_id}"
            claim_ids.setdefault(str(raw_id), []).append(source)
        nodes = [source["claim_node"]]
        claim_text = claim.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            source["issues"].append("missing_claim_text")
        cited = claim.get("cited_doc_ids")
        if not isinstance(cited, list) or any(not _valid_int(doc_id) for doc_id in cited):
            source["issues"].append("invalid_cited_doc_ids")
            cited_ids: list[str] = []
        else:
            cited_ids = [str(doc_id) for doc_id in cited]
            if len(set(cited_ids)) != len(cited_ids):
                source["issues"].append("duplicate_cited_doc_id")
            nodes.extend(f"doc:{doc_id}" for doc_id in cited_ids)
        source["cited_doc_ids"] = cited_ids
        evidence = claim.get("evidence")
        if not isinstance(evidence, dict):
            source["issues"].append("invalid_evidence_map")
            evidence = {}
        source["evidence"] = evidence
        for raw_doc_id in evidence:
            try:
                parsed = int(raw_doc_id)
            except (TypeError, ValueError):
                source["issues"].append("invalid_evidence_doc_id")
                continue
            if str(parsed) != str(raw_doc_id) or parsed < 0:
                source["issues"].append("invalid_evidence_doc_id")
                continue
            nodes.append(f"doc:{parsed}")
            if str(parsed) not in cited_ids:
                source["issues"].append("evidence_document_not_cited")
        for node in nodes[1:]:
            union.union(nodes[0], node)
        union.find(nodes[0])

    for claim_id, occurrences in claim_ids.items():
        if len(occurrences) > 1:
            fingerprints = {row["claim_sha256"] for row in occurrences}
            if len(fingerprints) > 1:
                for row in occurrences:
                    row["issues"].append("conflicting_duplicate_claim_id")
            else:
                for duplicate_number, row in enumerate(occurrences, 1):
                    row["exact_duplicate_ordinal"] = duplicate_number

    components = union.components()
    component_for: dict[str, list[str]] = {node: members for members in components.values() for node in members}
    claims_by_component: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        members = component_for.get(source["claim_node"], [source["claim_node"]])
        source["component_members"] = members
        component_id = "scifact:component:" + canonical_sha(members)
        source["component_id"] = component_id
        claims_by_component.setdefault(component_id, []).append(source)
    role_collision = {
        component_id for component_id, rows in claims_by_component.items()
        if {row["role"] for row in rows} == {"train", "validation"}
    }

    candidates: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    held_reason_counts: dict[str, int] = {}
    for source in sources:
        claim = source["claim"]
        issues = list(dict.fromkeys(source["issues"]))
        if source["component_id"] in role_collision:
            issues.append("connected_component_crosses_train_validation")
        label_values: list[str] = []
        accepted_sets: list[dict[str, Any]] = []
        evidence = source["evidence"]
        for raw_doc_id, rationales in evidence.items():
            doc_id = str(int(raw_doc_id)) if str(raw_doc_id).lstrip("+").isdigit() else str(raw_doc_id)
            doc_entry = corpus.get(doc_id)
            if doc_id in conflicting_doc_ids:
                issues.append("conflicting_corpus_document_id")
            if doc_entry is None:
                issues.append("missing_evidence_document")
            if not isinstance(rationales, list) or not rationales:
                issues.append("missing_rationales")
                continue
            for rationale_index, rationale in enumerate(rationales):
                if not isinstance(rationale, dict):
                    issues.append("invalid_rationale")
                    continue
                label = rationale.get("label")
                sentences = rationale.get("sentences")
                if label not in ("SUPPORT", "CONTRADICT"):
                    issues.append("invalid_rationale_label")
                else:
                    label_values.append(label)
                if not isinstance(sentences, list) or not sentences or any(not _valid_int(index) for index in sentences):
                    issues.append("invalid_evidence_sentence_ids")
                    continue
                if len(set(sentences)) != len(sentences):
                    issues.append("duplicate_evidence_sentence_id")
                if doc_entry is not None and any(index < 0 or index >= len(doc_entry["document"]["abstract"]) for index in sentences):
                    issues.append("evidence_sentence_id_out_of_range")
                accepted_sets.append({"doc_id": int(doc_id) if doc_id.isdigit() else doc_id,
                                      "rationale_index": rationale_index,
                                      "label": label,
                                      "sentence_ids": list(sentences) if isinstance(sentences, list) else []})
        empty_evidence_nei = not evidence and bool(source["cited_doc_ids"])
        # SciFact has no top-level label field. Its official oracle maps a retrieved cited
        # document with no gold rationale to NOT_ENOUGH_INFO; with an empty evidence map,
        # all cited documents therefore have that label. Keep this derivation explicit.
        if empty_evidence_nei:
            label_values.append("NOT_ENOUGH_INFO")
        elif not evidence:
            issues.append("no_annotated_evidence_and_no_cited_document")
        if len(set(label_values)) > 1:
            issues.append("conflicting_rationale_labels")
        if not label_values:
            issues.append("missing_supported_label")
        for doc_id in source["cited_doc_ids"]:
            if doc_id not in corpus:
                issues.append("missing_cited_document")
            if doc_id in conflicting_doc_ids:
                issues.append("conflicting_corpus_document_id")
        issues = list(dict.fromkeys(issues))
        if issues:
            for reason in issues:
                held_reason_counts[reason] = held_reason_counts.get(reason, 0) + 1
            held.append({"schema": "natlang.scifact-held-candidate/1", "role": source["role"],
                         "source_claim_id": source.get("claim_id"), "source_line": source["line"],
                         "source_file": (claims_dev_path or "claims_dev.jsonl") if source["role"] == "validation" else claims_train_path,
                         "source_revision": source_revision, "input_sha256": input_hashes,
                         "source_line_sha256": source["line_sha256"], "claim_sha256": source["claim_sha256"],
                         "component_id": source["component_id"], "source_groups": source["component_members"],
                         "reasons": issues})
            continue

        documents = []
        document_rows = []
        for doc_id in source["cited_doc_ids"]:
            document = corpus[doc_id]["document"]
            documents.append({"doc_id": document["doc_id"], "title": document["title"],
                              "abstract_sentences": [{"sentence_id": index, "text": text}
                                                     for index, text in enumerate(document["abstract"])],
                              "structured": document["structured"]})
            document_rows.append({"doc_id": document["doc_id"], "corpus_rows": corpus[doc_id]["source_rows"]})
        candidate_id = f"scifact:{source['role']}:{source['claim_id']}:row{source['line']}"
        candidates.append({
            "schema": "natlang.scifact-research-candidate/1",
            "candidate_id": candidate_id,
            "collection_status": "candidate_not_admitted",
            "role": source["role"],
            "task": {"instruction": "Assess the claim using only the supplied cited abstracts. Return SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO and cite the document ID plus sentence IDs that justify a supported or contradicted decision. For NOT_ENOUGH_INFO, return no evidence citations.",
                     "claim": claim["claim"], "documents": documents},
            "host_only_oracle": {"label": label_values[0], "accepted_evidence_sets": accepted_sets},
            "source_groups": [f"scifact:claim:{source['claim_id']}",
                              *(f"scifact:document:{doc_id}" for doc_id in source["cited_doc_ids"]),
                              source["component_id"]],
            "provenance": {"dataset": "SciFact", "repository": REPOSITORY,
                           "source_revision": source_revision, "source_archive_sha256": source_archive_sha256,
                           "source_role": source["role"],
                           "source_row": source["line"], "source_line_sha256": source["line_sha256"],
                           "source_file": (claims_dev_path or "claims_dev.jsonl") if source["role"] == "validation" else claims_train_path,
                           "input_sha256": input_hashes, "document_source_rows": document_rows,
                           "claim_sha256": source["claim_sha256"], "component_id": source["component_id"],
                           "component_members": source["component_members"],
                           "exact_duplicate_ordinal": source.get("exact_duplicate_ordinal"),
                           "label_derivation": "empty_evidence_to_NOT_ENOUGH_INFO_via_official_oracle" if empty_evidence_nei else "evidence_rationale_labels",
                           "claim_license": CLAIMS_LICENSE, "abstract_license": ABSTRACTS_LICENSE,
                           "schema_reference": DATA_SCHEMA_URL, "license_reference": LICENSE_URL,
                           "dataset_citation": DATASET_CITATION,
                           "label_semantics_reference": ORACLE_LABEL_REFERENCE,
                           "input_paths": {"corpus": corpus_path, "claims": (claims_dev_path or "claims_dev.jsonl") if source["role"] == "validation" else claims_train_path}},
        })

    inputs = {"corpus": {"path": corpus_path, "sha256": input_hashes["corpus"], "rows": len(corpus_rows)},
              "claims_train": {"path": claims_train_path, "sha256": input_hashes["claims_train"], "rows": len(train_rows)}}
    if include_dev_validation and claims_dev_bytes is not None:
        inputs["claims_dev"] = {"path": claims_dev_path or "claims_dev.jsonl", "sha256": input_hashes["claims_dev"], "rows": len(dev_rows)}
    candidate_lines = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in candidates)
    held_lines = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in held)
    manifest = {"schema": "natlang.scifact-research-preparation/1", "dataset": "SciFact",
                "repository": REPOSITORY, "source_revision": source_revision,
                "source_archive_url": DATA_ARCHIVE_URL, "source_archive_sha256": source_archive_sha256,
                "builder_sha256": sha256(Path(__file__).read_bytes()),
                "dataset_citation": DATASET_CITATION,
                "schema_reference": DATA_SCHEMA_URL, "license_reference": LICENSE_URL,
                "licenses": {"claims_and_evidence_annotations": CLAIMS_LICENSE, "abstracts": ABSTRACTS_LICENSE},
                "inputs": inputs,
                "split_policy": {"train": "claims_train only", "validation": "claims_dev only when explicitly enabled",
                                 "test": "not read or imported", "cross_role_connected_components": "held in full"},
                "counts": {"candidate_rows": len(candidates), "held_rows": len(held),
                           "candidate_train": sum(row["role"] == "train" for row in candidates),
                           "candidate_validation": sum(row["role"] == "validation" for row in candidates),
                           "candidate_labels": {label: sum(row["host_only_oracle"]["label"] == label for row in candidates)
                                                for label in ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")},
                           "candidate_components": len({row["provenance"]["component_id"] for row in candidates}),
                           "exact_duplicate_document_rows": duplicate_doc_rows,
                           "conflicting_document_ids": len(conflicting_doc_ids),
                           "claim_rows_in_exact_duplicate_groups": sum("exact_duplicate_ordinal" in row for row in sources),
                           "held_reasons": dict(sorted(held_reason_counts.items()))},
                "label_derivation": {"rationale_labels": "preserved as supplied; conflicting labels are held",
                                     "empty_evidence": "NOT_ENOUGH_INFO, following the official repository label-prediction oracle behavior",
                                     "semantics_reference": ORACLE_LABEL_REFERENCE},
                "admission": "candidate preparation only; no rows are admitted or marked training-ready",
                "model_calls": 0, "provider_calls": 0,
                "outputs": {"candidates.jsonl": {"sha256": sha256(candidate_lines), "bytes": len(candidate_lines)},
                            "held.jsonl": {"sha256": sha256(held_lines), "bytes": len(held_lines)}}}
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return {"candidates": candidates, "held": held, "manifest": manifest,
            "candidate_bytes": candidate_lines, "held_bytes": held_lines, "manifest_bytes": manifest_bytes}


def write_packet(output: str | Path, packet: dict[str, Any]) -> None:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.tmp-", dir=destination.parent) as tmp_name:
        tmp = Path(tmp_name)
        for name, data in (("candidates.jsonl", packet["candidate_bytes"]),
                           ("held.jsonl", packet["held_bytes"]),
                           ("manifest.json", packet["manifest_bytes"])):
            path = tmp / name
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        directory_fd = os.open(tmp, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(tmp, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--claims-train", required=True, type=Path)
    parser.add_argument("--claims-dev", type=Path)
    parser.add_argument("--include-dev-validation", action="store_true",
                        help="explicitly include claims_dev.jsonl as validation; test split is never read")
    parser.add_argument("--source-revision", required=True, help="pinned upstream SciFact repository revision")
    parser.add_argument("--source-archive-sha256", help="optional exact archive checksum from acquisition metadata")
    parser.add_argument("--out", required=True, type=Path, help="new output directory; must not already exist")
    args = parser.parse_args()
    if args.claims_dev and not args.include_dev_validation:
        parser.error("--claims-dev requires --include-dev-validation")
    if args.include_dev_validation and not args.claims_dev:
        parser.error("--include-dev-validation requires --claims-dev")
    packet = prepare_scifact_bytes(args.corpus.read_bytes(), args.claims_train.read_bytes(),
                                   source_revision=args.source_revision,
                                   source_archive_sha256=args.source_archive_sha256,
                                   claims_dev_bytes=args.claims_dev.read_bytes() if args.claims_dev else None,
                                   include_dev_validation=args.include_dev_validation,
                                   corpus_path=str(args.corpus), claims_train_path=str(args.claims_train),
                                   claims_dev_path=str(args.claims_dev) if args.claims_dev else None)
    write_packet(args.out, packet)
    print(json.dumps(packet["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
