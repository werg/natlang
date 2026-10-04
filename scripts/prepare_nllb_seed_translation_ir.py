#!/usr/bin/env python3
"""Build a pinned, held NLLB-Seed Program/2 translation candidate.

The adapter reads only the official NLLB-Seed archive. Target translations remain in a separate
host-only sidecar. It does not emit native results, admit training rows, or use FLORES evaluation
data. Grouping is by connected exact sentence identity across selected source and reference text;
the archive has no article IDs and does not document a cross-direction row-ID schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from prepare_nusax_translation_ir import canonical_bytes, sha256_bytes, sha256_file, translation_source


SOURCE_REPOSITORY = "https://github.com/facebookresearch/flores"
SOURCE_REVISION = "a6c830c6e1051fb4ac1a44b32358f00463f332bd"
ARCHIVE_URL = "https://dl.fbaipublicfiles.com/nllb/NLLB-Seed.zip"
ARCHIVE_VERSION_ID = "0.Jk.Pd_WL3RiByHqI025SMXTPjTY2LP"
ARCHIVE_SHA256 = "d269fa2bebba88c85de8912a5c1e5ddd9fd2086f8432b29f868627918b131c6d"
README_SHA256 = "b712325f3cf90f748b4cfa56ba29c0a160d192142c81eadf4bc009d3d15d5335"
LICENSE_SHA256 = "fe38377ee0e52dc6c9c78554055a4e5e78781d73022df4508c9cf284e0f462e9"
LICENSE = "CC-BY-SA-4.0"
PROGRAM_VERSION = "natlang.program/2"
ADAPTER_VERSION = "natlang.nllb_seed_translation_static_adapter/1"
DEFAULT_GROUP_LIMIT = 512

# Codes from the pinned NLLB-Seed README. The archive supplies only the direction folders listed
# in its own ZIP; this list is not used to synthesize missing reverse directions.
LANGUAGES = {
    "ace_Arab": "Acehnese (Arabic script)", "ace_Latn": "Acehnese (Latin script)",
    "ary_Arab": "Moroccan Arabic", "arz_Arab": "Egyptian Arabic", "bam_Latn": "Bambara",
    "ban_Latn": "Balinese", "bho_Deva": "Bhojpuri", "bjn_Arab": "Banjar (Arabic script)",
    "bjn_Latn": "Banjar (Latin script)", "bug_Latn": "Buginese", "crh_Latn": "Crimean Tatar",
    "dik_Latn": "Southwestern Dinka", "dzo_Tibt": "Dzongkha", "fur_Latn": "Friulian",
    "fuv_Latn": "Nigerian Fulfulde", "grn_Latn": "Guarani", "hne_Deva": "Chhattisgarhi",
    "kas_Arab": "Kashmiri (Arabic script)", "kas_Deva": "Kashmiri (Devanagari script)",
    "knc_Arab": "Central Kanuri (Arabic script)", "knc_Latn": "Central Kanuri (Latin script)",
    "lij_Latn": "Ligurian", "lim_Latn": "Limburgish", "lmo_Latn": "Lombard",
    "ltg_Latn": "Latgalian", "mag_Deva": "Magahi", "mni_Beng": "Meitei (Bengali script)",
    "mri_Latn": "Maori", "nus_Latn": "Nuer", "pbt_Arab": "Southern Pashto",
    "prs_Arab": "Dari", "scn_Latn": "Sicilian", "shn_Mymr": "Shan",
    "srd_Latn": "Sardinian", "szl_Latn": "Silesian", "taq_Latn": "Tamasheq (Latin script)",
    "taq_Tfng": "Tamasheq (Tifinagh script)", "tzm_Tfng": "Central Atlas Tamazight",
    "vec_Latn": "Venetian", "eng_Latn": "English",
}


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            if a < b:
                self.parent[b] = a
            else:
                self.parent[a] = b


def _hash_text_identity(text: str) -> str:
    # Used only for duplicate/group detection. Actual source bytes are retained unchanged.
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def _zip_lines(data: bytes, member: str) -> list[tuple[str, str]]:
    # Split physical rows on LF only: Unicode line separators are valid text content.
    pieces = data.split(b"\n")
    if pieces and pieces[-1] == b"":
        pieces.pop()
    out = []
    for index, raw in enumerate(pieces):
        line_bytes = raw[:-1] if raw.endswith(b"\r") else raw
        try:
            text = line_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid_utf8:{member}:{index}") from exc
        out.append((text, sha256_bytes(line_bytes)))
    return out


def load_archive(archive: Path, *, verify_pin: bool = True, expected_sha256: str = ARCHIVE_SHA256,
                 allowed_languages: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive_sha = sha256_file(archive)
    if verify_pin and archive_sha != expected_sha256:
        raise ValueError(f"nllb_seed_archive_hash_mismatch:{archive_sha}")
    codes = set(LANGUAGES if allowed_languages is None else allowed_languages)
    if "eng_Latn" not in codes or not codes.issubset(LANGUAGES):
        raise ValueError("invalid_language_allowlist")
    if not archive.is_file():
        raise ValueError("nllb_seed_archive_missing")

    directions: list[dict[str, Any]] = []
    member_receipts: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive) as zf:
        seen_members: set[str] = set()
        for info in zf.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "NLLB-Seed":
                raise ValueError(f"unsafe_or_unexpected_archive_path:{info.filename}")
            if info.is_dir():
                continue
            if len(path.parts) != 3:
                raise ValueError(f"unexpected_archive_member:{info.filename}")
            folder, leaf = path.parts[1], path.parts[2]
            match = re.fullmatch(r"([A-Za-z0-9_]+)-([A-Za-z0-9_]+)", folder)
            if not match:
                raise ValueError(f"invalid_direction_folder:{folder}")
            source_code, target_code = match.groups()
            if source_code not in codes or target_code not in codes or source_code == target_code:
                raise ValueError(f"unrecognized_direction:{folder}")
            if leaf not in (source_code, target_code):
                raise ValueError(f"unexpected_language_member:{info.filename}")
            if info.filename in seen_members:
                raise ValueError(f"duplicate_archive_member:{info.filename}")
            seen_members.add(info.filename)

        folders = sorted({name.split("/")[1] for name in seen_members})
        for folder in folders:
            source_code, target_code = folder.split("-")
            source_member = f"NLLB-Seed/{folder}/{source_code}"
            target_member = f"NLLB-Seed/{folder}/{target_code}"
            if source_member not in seen_members or target_member not in seen_members:
                raise ValueError(f"incomplete_direction_members:{folder}")
            source_raw, target_raw = zf.read(source_member), zf.read(target_member)
            source_lines, target_lines = _zip_lines(source_raw, source_member), _zip_lines(target_raw, target_member)
            if len(source_lines) != len(target_lines):
                raise ValueError(f"parallel_file_line_count_mismatch:{folder}")
            if not source_lines:
                raise ValueError(f"empty_direction:{folder}")
            directions.append({"source_code": source_code, "target_code": target_code, "folder": folder,
                               "source_member": source_member, "target_member": target_member,
                               "source_lines": source_lines, "target_lines": target_lines})
            member_receipts[source_member] = {"sha256": sha256_bytes(source_raw), "bytes": len(source_raw), "rows": len(source_lines)}
            member_receipts[target_member] = {"sha256": sha256_bytes(target_raw), "bytes": len(target_raw), "rows": len(target_lines)}
    if not directions:
        raise ValueError("no_nllb_seed_directions_found")
    dirs_by_name = {row["folder"]: row for row in directions}
    if len(dirs_by_name) != len(directions):
        raise ValueError("duplicate_direction_folder")
    # Require each supplied language to be represented in a real archive direction; do not invent
    # inverse paths. Production default uses exactly the 39 source directions in the archive.
    represented = {d["source_code"] for d in directions} | {d["target_code"] for d in directions}
    if not codes.issubset(represented):
        raise ValueError(f"language_missing_from_archive:{sorted(codes - represented)}")
    directions.sort(key=lambda d: (d["source_code"], d["target_code"]))
    metadata = {"archive_sha256": archive_sha, "archive_bytes": archive.stat().st_size,
                "direction_count": len(directions), "directions": [
                    {"source": d["source_code"], "target": d["target_code"], "rows": len(d["source_lines"]),
                     "source_member": d["source_member"], "target_member": d["target_member"]}
                    for d in directions], "pinned_members": member_receipts}
    return directions, metadata


def prepare_candidate(archive: Path, *, max_parallel_rows: int | None = DEFAULT_GROUP_LIMIT,
                      verify_pin: bool = True, expected_sha256: str = ARCHIVE_SHA256,
                      allowed_languages: set[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if max_parallel_rows is not None and (not isinstance(max_parallel_rows, int) or max_parallel_rows < 1):
        raise ValueError("max_parallel_rows_must_be_positive_or_none")
    directions, source_meta = load_archive(archive, verify_pin=verify_pin, expected_sha256=expected_sha256,
                                            allowed_languages=allowed_languages)
    records: list[dict[str, Any]] = []
    selection_ledger: list[dict[str, Any]] = []
    record_text_hashes: dict[str, set[str]] = defaultdict(set)
    for d in directions:
        complete_indices = [i for i, (source_row, target_row) in enumerate(zip(d["source_lines"], d["target_lines"]))
                            if source_row[0].strip() and target_row[0].strip()]
        ranked_indices = sorted(complete_indices,
                                key=lambda i: hashlib.sha256(
                                    f"nllb-seed-sample-v1:{d['folder']}:{i}".encode()).digest())
        selected_indices = ranked_indices if max_parallel_rows is None else ranked_indices[:max_parallel_rows]
        selected_indices.sort()
        if not selected_indices:
            raise ValueError(f"no_complete_parallel_rows:{d['folder']}")
        d["selected_indices"] = selected_indices
        selected_set = set(selected_indices)
        for index, (source_row, target_row) in enumerate(zip(d["source_lines"], d["target_lines"])):
            source_text, source_line_sha = source_row
            target_text, target_line_sha = target_row
            if index in selected_set:
                status, reason = "selected", None
            elif not source_text.strip() and not target_text.strip():
                status, reason = "excluded", "empty_source_and_target"
            elif not source_text.strip():
                status, reason = "excluded", "empty_source"
            elif not target_text.strip():
                status, reason = "excluded", "empty_target"
            else:
                status, reason = "excluded", "per_direction_sampling_limit"
            selection_ledger.append({"schema": "natlang.nllb_seed_row_selection/1",
                                     "source_pair": d["folder"], "source_code": d["source_code"],
                                     "target_code": d["target_code"], "row_index_zero_based": index,
                                     "status": status, **({"reason": reason} if reason else {}),
                                     "source_line_content_sha256": source_line_sha,
                                     "target_line_content_sha256": target_line_sha})
        for index in selected_indices:
            source_text, source_line_sha = d["source_lines"][index]
            target_text, target_line_sha = d["target_lines"][index]
            task_id = f"nllb-seed:{d['source_code']}->{d['target_code']}:line-{index + 1}"
            rec = {"task_id": task_id, "source_pair": d["folder"], "row_index_zero_based": index,
                   "source_code": d["source_code"], "target_code": d["target_code"],
                   "source_text": source_text, "target_text": target_text,
                   "source_line_content_sha256": source_line_sha, "target_line_content_sha256": target_line_sha,
                   "source_member": d["source_member"], "target_member": d["target_member"]}
            records.append(rec)
            for text in (source_text, target_text):
                key = _hash_text_identity(text)
                if key:
                    record_text_hashes[key].add(task_id)

    union = UnionFind([rec["task_id"] for rec in records])
    for tasks in record_text_hashes.values():
        first = next(iter(tasks))
        for task_id in tasks:
            union.union(first, task_id)
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        components[union.find(record["task_id"])].append(record)
    normalized_components: list[dict[str, Any]] = []
    for members in components.values():
        task_ids = sorted(row["task_id"] for row in members)
        component_digest = sha256_bytes("\n".join(task_ids).encode("utf-8"))
        group_id = f"nllb-seed/text-component/{component_digest[:24]}"
        normalized_components.append({"id": group_id, "digest": component_digest, "members": members,
                                      "row_indices": sorted({row["row_index_zero_based"] for row in members})})
    normalized_components.sort(key=lambda c: c["id"])
    component_for_task = {member["task_id"]: component["id"]
                          for component in normalized_components for member in component["members"]}

    # Create a deterministic, weighted 20% validation split over exact-text components.
    # Components are indivisible; the algorithm chooses a set closest to the target case count.
    total_cases = len(records)
    validation_target = round(total_cases * 0.20)
    ranked = sorted(normalized_components,
                    key=lambda c: hashlib.sha256(f"nllb-seed-validation-v1:{c['digest']}".encode()).digest())
    validation_ids: set[str] = set()
    validation_cases = 0
    for comp in ranked:
        weight = len(comp["members"])
        if abs(validation_target - (validation_cases + weight)) < abs(validation_target - validation_cases):
            validation_ids.add(comp["id"])
            validation_cases += weight
    roles: dict[str, str] = {}
    for comp in normalized_components:
        if comp["id"] in validation_ids:
            roles[comp["id"]] = "validation"
        else:
            bucket = int(hashlib.sha256(f"nllb-seed-train-role-v1:{comp['digest']}".encode()).hexdigest()[:8], 16) % 5
            roles[comp["id"]] = "train_query" if bucket == 0 else "train_support"

    ir_rows, host_rows = [], []
    by_direction: Counter[str] = Counter()
    by_direction_role: Counter[str] = Counter()
    component_row_indices: dict[str, set[int]] = {}
    for comp in normalized_components:
        role = roles[comp["id"]]
        split = "validation" if role == "validation" else "train"
        component_row_indices[comp["id"]] = set(comp["row_indices"])
        for record in comp["members"]:
            src_code, tgt_code = record["source_code"], record["target_code"]
            src_label, tgt_label = LANGUAGES[src_code], LANGUAGES[tgt_code]
            ir_id = f"nllb-seed-translation-ir:{record['task_id']}"
            task = {
                "version": PROGRAM_VERSION, "id": ir_id, "kind": "lambda_source",
                "family": "translation_source_candidate", "source": "NLLB-Seed",
                "split": split, "source_ids": [record["task_id"]], "source_groups": [comp["id"]],
                "source_revisions": [SOURCE_REVISION, source_meta["archive_sha256"]],
                "license": LICENSE,
                "gold_sources": ["NLLB-Seed professional translation; target retained host-side"],
                "generation": {"generator": ADAPTER_VERSION, "original_split": "train",
                               "derived_role": role, "parallel_row_index_zero_based": record["row_index_zero_based"],
                               "source_identity_scope": "exact-text-connected-component"},
                "semantics": {"root": "translate.nl",
                              "files": {"translate.nl": translation_source(src_label, tgt_label)},
                              "inputs": {"source_language": src_label, "source_text": record["source_text"],
                                         "target_language": tgt_label}},
                "external_source": {
                    "repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
                    "archive_url": ARCHIVE_URL, "archive_sha256": source_meta["archive_sha256"],
                    "source_pair_folder": record["source_pair"],
                    "source_member": record["source_member"], "target_member": record["target_member"],
                    "source_language_code": src_code, "target_language_code": tgt_code,
                    "parallel_row_index_zero_based": record["row_index_zero_based"],
                    "source_line_content_sha256": record["source_line_content_sha256"],
                    "target_line_content_sha256": record["target_line_content_sha256"],
                    "quality": {"version": "natlang.source_quality/1", "status": "held",
                                "checks": ["archive_and_license_pins_verified", "parallel_pair_line_counts_match",
                                           "source_visible_target_host_separated",
                                           "cross_split_exact_sentence_identities_closed"],
                                "hold_reasons": ["native_program2_source_replay_not_performed",
                                                 "current_source_policy_and_training_admission_not_reviewed",
                                                 "article_ids_absent_topic_level_separation_unproven"]},
                    "source_identity_note": "The upstream archive has no article IDs or declared shared row IDs across direction folders. Candidate split groups use connected components of exact normalized sentence text across selected pairs; no article/topic-disjoint claim is made.",
                },
            }
            ir_rows.append(task)
            host_rows.append({"schema": "natlang.translation-static-host-reference/1", "ir_id": ir_id,
                              "task_id": record["task_id"], "source_group": comp["id"], "split": split,
                              "role": role, "original_split": "train", "source_language": src_label,
                              "target_language": tgt_label, "target_text": record["target_text"],
                              "target_line_content_sha256": record["target_line_content_sha256"],
                              "visibility": "host-only", "license": LICENSE, "admission": "held"})
            by_direction[f"{src_code}->{tgt_code}:{split}"] += 1
            by_direction_role[f"{src_code}->{tgt_code}:{role}"] += 1

    # The exact-text components assigned above are also the split groups. Verify that no exact
    # normalized source/reference string crosses train and validation among emitted records.
    text_roles: dict[str, set[str]] = defaultdict(set)
    for task in records:
        role = roles[component_for_task[task["task_id"]]]
        for text in (task["source_text"], task["target_text"]):
            key = _hash_text_identity(text)
            if key:
                text_roles[key].add(role)
    if any(len(values) > 1 for values in text_roles.values()):
        raise AssertionError("exact_text_identity_crosses_candidate_splits")

    source_lines = {d["folder"]: len(d["source_lines"]) for d in directions}
    component_summary = {"components": len(normalized_components),
                         "component_sizes_task_rows": sorted((len(c["members"]) for c in normalized_components), reverse=True),
                         "validation_components": sum(roles[c["id"]] == "validation" for c in normalized_components),
                         "validation_task_rows": sum(len(c["members"]) for c in normalized_components if roles[c["id"]] == "validation"),
                         "train_support_components": sum(roles[c["id"]] == "train_support" for c in normalized_components),
                         "train_support_task_rows": sum(len(c["members"]) for c in normalized_components if roles[c["id"]] == "train_support"),
                         "train_query_components": sum(roles[c["id"]] == "train_query" for c in normalized_components),
                         "train_query_task_rows": sum(len(c["members"]) for c in normalized_components if roles[c["id"]] == "train_query")}
    ledger_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in selection_ledger:
        key = row["source_pair"]
        ledger_counts[key][row.get("reason") or row["status"]] += 1
    metadata = {
        "schema": "natlang.translation_static_sft_candidate/1", "adapter": ADAPTER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"name": "NLLB-Seed", "repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION,
                   "archive_url": ARCHIVE_URL, "archive_sha256": source_meta["archive_sha256"],
                   "archive_bytes": source_meta["archive_bytes"], "archive_version_id": ARCHIVE_VERSION_ID,
                   "license": LICENSE, "license_file_sha256": LICENSE_SHA256,
                   "license_url": "https://github.com/facebookresearch/flores/blob/main/LICENSE_CC-BY-SA-4.0",
                   "license_file": "vendor/datasets/nllb-seed-20261004/LICENSE_CC-BY-SA-4.0",
                   "readme_url": "https://github.com/facebookresearch/flores/blob/main/nllb_seed/README.md",
                   "readme_file": "vendor/datasets/nllb-seed-20261004/NLLB-Seed-README.md",
                   "readme_sha256": README_SHA256, "component_attribution": "NLLB Team; source material sampled from Wikimedia's List of articles every Wikipedia should have.",
                   "license_scope": "The pinned upstream NLLB-Seed README declares the dataset CC-BY-SA 4.0. Review attribution/share-alike obligations before downstream distribution.",
                   "direction_count": len(directions), "directions": source_meta["directions"],
                   "member_receipts": source_meta["pinned_members"], "raw_rows_by_direction": source_lines,
                   "source_split": "All source rows come from NLLB-Seed's training-intended release; the candidate validation role is deterministically derived from the selected NLLB-Seed rows."},
        "selection": {"maximum_rows_per_direction": max_parallel_rows,
                      "selected_direction_row_indices_sha256": sha256_bytes(canonical_bytes({
                          d["folder"]: d["selected_indices"] for d in directions})),
                      "selection_rule": (("All nonempty parallel rows in each actual archive direction; row indices "
                                          "are local to that direction and are not cross-direction identities.")
                                         if max_parallel_rows is None else
                                         (f"Within each actual archive direction independently, select up to "
                                          f"{max_parallel_rows} nonempty parallel rows by SHA-256 rank of "
                                          "nllb-seed-sample-v1:<folder>:<zero-based-row-index>; row indices "
                                          "are local to that direction and are not cross-direction identities.")),
                      "available_nonempty_rows_by_direction": {
                          d["folder"]: sum(bool(src[0].strip() and tgt[0].strip())
                                            for src, tgt in zip(d["source_lines"], d["target_lines"]))
                          for d in directions},
                      "row_disposition_counts_by_direction": {
                          folder: dict(sorted(counts.items())) for folder, counts in sorted(ledger_counts.items())},
                      "parallel_linkage_rule": "Connected components of exact Unicode-whitespace-normalized source or target sentence text among selected direction rows; no cross-folder line-index equality assumption.",
                      "article_or_topic_ids_available": False,
                      "exact_text_cross_split_conflicts": 0},
        "candidate_counts": {"task_count": len(ir_rows), "ir_rows": len(ir_rows),
                             "host_reference_rows": len(host_rows),
                             "source_rows_in_selection_ledger": len(selection_ledger),
                             "by_direction_and_split": dict(sorted(by_direction.items())),
                             "by_direction_and_role": dict(sorted(by_direction_role.items())),
                             "distinct_text_components": len(normalized_components), **component_summary},
        "protected_evaluation": {"flores_dev_devtest_test_rows_acquired": 0,
                                 "flores_evaluation_used": False,
                                 "note": "Only the NLLB-Seed archive was acquired; separate FLORES evaluation files were not downloaded or read."},
        "readiness": {
            "static_sft": {"status": "held_before_admission_native_ir_policy_review",
                            "source_basis": "The upstream describes NLLB-Seed as professionally translated and intended for training; the selected target text is kept host-only.",
                            "native_program2_source_replay": "not_performed", "source_policy_review": "pending",
                            "training_admission": "held"},
            "self_improvement_reward": {"status": "held_semantic_evaluator_pending",
                                        "exact_reference_string_grading": False,
                                        "semantic_alternative_policy": "not defined; translation quality needs reviewed meaning-preservation scoring and alternatives",
                                        "admission": "held"},
        },
        "quality_limitations": ["NLLB-Seed does not receive FLORES-200's human quality assurance.",
                                "No article or source-document identifiers are included, so topic/article-disjoint evaluation cannot be proven.",
                                "Source pairs are not bidirectional unless both folders exist."],
        "pipeline_registration": {"candidate_contract": "natlang.translation_static_sft_candidate/1",
                                  "existing_static_bundle_contract": "natlang.source_static_bundle/1 requires native static results, source conversion provenance, and current admission",
                                  "default_recipe_inclusion": False,
                                  "registry_entry": ("training/self_improvement_tasks.json: nllb-seed-static-full-v2"
                                                     if max_parallel_rows is None else
                                                     "training/self_improvement_tasks.json: nllb-seed-static-translation-v1"),
                                  "reason": "Candidate references are not native replay results or an admitted source_static_bundle."},
        "model_calls": 0, "admission": "none",
        "artifacts": {},
    }
    return ir_rows, host_rows, selection_ledger, metadata


def write_candidate(out: Path, ir_rows: list[dict[str, Any]], host_rows: list[dict[str, Any]],
                    selection_ledger: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for filename, rows in (("translation-source-ir.jsonl", ir_rows),
                           ("translation-references.host-only.jsonl", host_rows),
                           ("row-selection-ledger.jsonl", selection_ledger)):
        raw = b"".join(canonical_bytes(row) + b"\n" for row in rows)
        path = out / filename
        path.write_bytes(raw)
        artifacts[filename] = {"rows": len(rows), "bytes": len(raw), "sha256": sha256_bytes(raw)}
    metadata["artifacts"] = artifacts
    (out / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("vendor/datasets/nllb-seed-20261004"),
                        help="directory with the pinned archive, upstream README and license text")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-parallel-rows", type=int, default=DEFAULT_GROUP_LIMIT,
                           help="maximum rows sampled independently in each actual archive direction")
    selection.add_argument("--all-rows", action="store_true",
                           help="include every nonempty parallel row from every actual archive direction")
    parser.add_argument("--out", type=Path, required=True, help="new candidate output directory (must not exist)")
    args = parser.parse_args()
    root = args.source_root.resolve()
    archive = root / "download/NLLB-Seed.zip"
    if sha256_file(root / "NLLB-Seed-README.md") != README_SHA256:
        raise ValueError("pinned_nllb_seed_readme_changed")
    if sha256_file(root / "LICENSE_CC-BY-SA-4.0") != LICENSE_SHA256:
        raise ValueError("pinned_nllb_seed_license_changed")
    limit = None if args.all_rows else args.max_parallel_rows
    ir_rows, host_rows, selection_ledger, metadata = prepare_candidate(archive, max_parallel_rows=limit)
    write_candidate(args.out.resolve(), ir_rows, host_rows, selection_ledger, metadata)
    print(json.dumps({"output": str(args.out.resolve()), **metadata["candidate_counts"],
                      "training_admission": "held", "admission": "none"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
