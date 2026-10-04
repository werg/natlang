#!/usr/bin/env python3
"""Prepare held, source-pinned NusaX-MT task records without inventing a translation score.

The task file contains only source-side model inputs. Exact dataset translations are written to a
separate host-reference file. Neither file is a ready-to-train episode: semantic evaluation and
rights-chain review are explicit gates for a later adapter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

NUSAX_REVISION = "75b0be5d982348332509f296e10d9bf9cb66d60f"
REPO_URL = "https://github.com/IndoNLP/nusax"
LANGUAGES = (
    "indonesian", "acehnese", "banjarese", "english", "madurese", "ngaju",
    "sundanese", "balinese", "buginese", "javanese", "minangkabau", "toba_batak",
)
ALIASES = {"en": "english", "id": "indonesian"}
REGIONAL = tuple(name for name in LANGUAGES if name not in {"english", "indonesian"})
DEFAULT_PAIRS = tuple(
    [(a, b) for a, b in (("english", "indonesian"), ("indonesian", "english"))]
    + [(src, dst) for regional in REGIONAL for src, dst in (
        ("english", regional), (regional, "english"),
        ("indonesian", regional), (regional, "indonesian"),
    )]
)
PINNED_FILES = {
    "README.md": "71bd3123a251987b278fb27ddeda18e554b680dde18737b4fefba240f0f45ad2",
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "datasets/LICENSE": "5e436ff8ffbb77d8607220e9bce20c8915d860010feeb6c1ebef5a85688e9b39",
    "datasets/mt/README.md": "4819bc4a4d7c0295c86f33eb6c082f9922b7bf3acd39ac32ad96dcd7079af39f",
    "datasets/mt/train.csv": "cf4276730de04f0cb3369c48b67d454f4aae52646315f19eb71b908f5dfaae0e",
    "datasets/mt/valid.csv": "fd148ce931676460bb89ce94205f5cea3fb0fbd1fc4b561a016dc25ceb8e049a",
    "datasets/mt/test.csv": "0082161de28a0a17a85fc6e02fc07a2e2514d8dd7f344ead95a88aa86dc706cc",
}
DATA_LICENSE = "CC-BY-SA-4.0"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)).encode("utf-8")


def parse_pairs(value: str | None) -> list[tuple[str, str]]:
    if value is None or value.strip().lower() == "default":
        return list(DEFAULT_PAIRS)
    if value.strip().lower() == "all":
        return [(src, dst) for src in LANGUAGES for dst in LANGUAGES if src != dst]
    out: list[tuple[str, str]] = []
    for item in value.split(","):
        pieces = item.strip().lower().replace("→", "->").split("->")
        if len(pieces) != 2:
            raise ValueError(f"language pair must use source->target: {item!r}")
        src, dst = (ALIASES.get(piece.strip(), piece.strip()) for piece in pieces)
        if src not in LANGUAGES or dst not in LANGUAGES or src == dst:
            raise ValueError(f"unsupported or identical language pair: {item!r}")
        pair = (src, dst)
        if pair in out:
            raise ValueError(f"duplicate language pair: {item!r}")
        out.append(pair)
    return out


def pinned_checkout(repo: Path, *, verify_pin: bool = True) -> dict[str, Any]:
    if verify_pin:
        try:
            revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"not a readable NusaX git checkout: {repo}") from exc
        if revision != NUSAX_REVISION:
            raise ValueError(f"NusaX checkout revision mismatch: expected {NUSAX_REVISION}, got {revision}")
    else:
        revision = NUSAX_REVISION
    files: dict[str, dict[str, Any]] = {}
    for rel, expected in PINNED_FILES.items():
        path = repo / rel
        if not path.is_file():
            raise ValueError(f"required pinned NusaX source file is missing: {rel}")
        actual = sha256_file(path)
        if verify_pin and actual != expected:
            raise ValueError(f"NusaX file hash mismatch for {rel}: expected {expected}, got {actual}")
        files[rel] = {"sha256": actual, "bytes": path.stat().st_size}
    return {"repository": REPO_URL, "revision": revision, "files": files}


def load_split(repo: Path, split: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rel = f"datasets/mt/{split}.csv"
    path = repo / rel
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["", *LANGUAGES]:
            raise ValueError(f"unexpected header for {rel}: {reader.fieldnames!r}")
        seen: set[str] = set()
        for line_number, raw in enumerate(reader, start=2):
            row_id = raw.get("")
            if row_id is None or not row_id.isdecimal() or row_id in seen:
                raise ValueError(f"invalid or duplicate row identity at {rel}:{line_number}")
            seen.add(row_id)
            clean = {name: raw.get(name, "") for name in LANGUAGES}
            if any(not isinstance(value, str) for value in clean.values()):
                raise ValueError(f"nontext cell at {rel}:{line_number}")
            rows.append({"row_id": row_id, **clean})
    return rows, {"path": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size, "rows": len(rows)}


def _sample_key(split: str, row_id: str) -> str:
    return hashlib.sha256(f"nusax-mt-row-v1\0{split}\0{row_id}".encode()).hexdigest()


def _role(group: str) -> str:
    # One assignment per parallel row, shared across every direction of translation.
    bucket = int(hashlib.sha256(("nusax-role-v1\0" + group).encode()).hexdigest()[:8], 16) % 5
    return "train_query" if bucket == 0 else "train_support"


def build_translation_tasks(
    repo: Path,
    *,
    pairs: list[tuple[str, str]] | None = None,
    max_train_rows: int = 64,
    max_valid_rows: int = 16,
    verify_pin: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= max_train_rows <= 500 or not 0 <= max_valid_rows <= 100:
        raise ValueError("row caps must be 1..500 train and 0..100 validation")
    chosen_pairs = pairs or list(DEFAULT_PAIRS)
    if not chosen_pairs or len(chosen_pairs) != len(set(chosen_pairs)):
        raise ValueError("at least one unique direction is required")
    if any(src not in LANGUAGES or dst not in LANGUAGES or src == dst for src, dst in chosen_pairs):
        raise ValueError("unsupported or identical language direction")
    source = pinned_checkout(repo, verify_pin=verify_pin)
    loaded: dict[str, list[dict[str, str]]] = {}
    split_receipts: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        loaded[split], split_receipts[split] = load_split(repo, split)
    selected_ids: dict[str, set[str]] = {
        "train": {r["row_id"] for r in sorted(loaded["train"], key=lambda r: _sample_key("train", r["row_id"]))[:max_train_rows]},
        "valid": {r["row_id"] for r in sorted(loaded["valid"], key=lambda r: _sample_key("valid", r["row_id"]))[:max_valid_rows]},
    }
    tasks: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped_empty: Counter[str] = Counter()
    selected_parallel_groups: dict[str, str] = {}
    for split in ("train", "valid"):
        for row in loaded[split]:
            row_id = row["row_id"]
            if row_id not in selected_ids[split]:
                continue
            group = f"nusax/mt/{split}/{row_id}"
            role = _role(group) if split == "train" else "validation"
            selected_parallel_groups[group] = role
            for src, dst in chosen_pairs:
                pair_name = f"{src}->{dst}"
                source_text, target_text = row[src], row[dst]
                if not source_text.strip() or not target_text.strip():
                    skipped_empty[pair_name] += 1
                    continue
                task_id = "nusax_mt_" + hashlib.sha256(
                    f"{NUSAX_REVISION}\0{group}\0{src}\0{dst}".encode()
                ).hexdigest()[:24]
                tasks.append({
                    "schema": "natlang.translation-source-task/1",
                    "id": task_id,
                    "source_group": group,
                    "role": role,
                    "original_split": split,
                    "source_language": src,
                    "target_language": dst,
                    "input": {"text": source_text},
                    "candidate_status": "held_semantic_evaluator_pending",
                    "model_input_visibility": "input-only; do not expose the reference record",
                    "provenance": {"project": "NusaX", "revision": NUSAX_REVISION, "source_path": f"datasets/mt/{split}.csv", "row_id": row_id},
                })
                references.append({
                    "schema": "natlang.translation-source-reference/1",
                    "task_id": task_id,
                    "source_group": group,
                    "reference_target": target_text,
                    "visibility": "host-only",
                    "license": DATA_LICENSE,
                    "evaluation_status": "reference text is not an exact-match score; semantic evaluator pending",
                })
                counts[f"{split}:{role}:{pair_name}"] += 1
    task_group_roles: dict[str, set[str]] = {}
    for task in tasks:
        task_group_roles.setdefault(task["source_group"], set()).add(task["role"])
    if any(len(roles) != 1 for roles in task_group_roles.values()):
        raise AssertionError("one parallel sentence was assigned to multiple roles")
    test_groups = [f"nusax/mt/test/{row['row_id']}" for row in loaded["test"]]
    metadata = {
        "schema": "natlang.translation-source-manifest/1",
        "source": source,
        "dataset_license": DATA_LICENSE,
        "license_evidence": {
            "dataset_license_file": "datasets/LICENSE",
            "dataset_license_file_sha256": source["files"]["datasets/LICENSE"]["sha256"],
            "readme_statement": "CC-BY-SA, version-unspecified",
            "software_license": "Apache-2.0 (repository code only; not applied to the dataset)",
            "rights_scope_note": "NusaX README says the corpus translates an existing sentiment dataset; this adapter preserves the NusaX dataset license evidence but does not resolve item-level upstream rights beyond the repository license file.",
        },
        "language_columns": list(LANGUAGES),
        "selected_directions": [f"{src}->{dst}" for src, dst in chosen_pairs],
        "default_direction_count": len(DEFAULT_PAIRS),
        "selection": {"max_train_parallel_rows": max_train_rows, "max_validation_parallel_rows": max_valid_rows,
                      "deterministic_shared_row_sample": True, "train_query_fraction": 0.2,
                      "role_assignment": "stable SHA-256 bucket of the split-qualified parallel source group; shared by all directions"},
        "source_splits": split_receipts,
        "selected_parallel_rows": {split: len(selected_ids[split]) for split in selected_ids},
        "selected_parallel_group_role_counts": dict(Counter(selected_parallel_groups.values())),
        "protected_original_test": {"row_count": len(test_groups), "groups_sha256": hashlib.sha256("\n".join(test_groups).encode()).hexdigest(),
                                    "exported_as_tasks": False, "policy": "original NusaX test split remains protected and is not emitted"},
        "emitted_task_counts": dict(sorted(counts.items())),
        "empty_cell_skips": dict(sorted(skipped_empty.items())),
        "task_count": len(tasks),
        "host_reference_count": len(references),
        "semantic_evaluator": {"status": "pending", "exact_reference_string_grading": False, "admission": "held"},
        "rights_review": {"dataset_license_version_explicit_in_dataset_license_file": True,
                          "item_level_upstream_rights_review": "pending", "training_admission": "held"},
        "citation": "Winata et al. (2023), NusaX: Multilingual Parallel Sentiment Dataset for 10 Indonesian Local Languages, EACL 2023, https://aclanthology.org/2023.eacl-main.57/",
    }
    return tasks, references, metadata


def write_packet(out: Path, tasks: list[dict[str, Any]], references: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    task_blob, reference_blob = jsonl_bytes(tasks), jsonl_bytes(references)
    (out / "translation-source-tasks.jsonl").write_bytes(task_blob)
    (out / "translation-source-references.host-only.jsonl").write_bytes(reference_blob)
    manifest = {**manifest,
        "artifacts": {
            "translation-source-tasks.jsonl": {"sha256": hashlib.sha256(task_blob).hexdigest(), "bytes": len(task_blob)},
            "translation-source-references.host-only.jsonl": {"sha256": hashlib.sha256(reference_blob).hexdigest(), "bytes": len(reference_blob)},
        },
        "publication": "Held source task candidates only; no semantic labels, provider outputs, or admitted training examples",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=root / "vendor/datasets/nusax-20261004")
    parser.add_argument("--out", type=Path, required=True, help="new output directory; must not already exist")
    parser.add_argument("--pairs", default="default", help="comma-separated source->target pairs; use 'all' for all 132 ordered directions")
    parser.add_argument("--max-train-rows", type=int, default=64, help="bounded number of shared parallel row groups")
    parser.add_argument("--max-valid-rows", type=int, default=16, help="bounded number of original validation row groups")
    args = parser.parse_args()
    tasks, references, manifest = build_translation_tasks(args.repo.resolve(), pairs=parse_pairs(args.pairs),
                                                          max_train_rows=args.max_train_rows, max_valid_rows=args.max_valid_rows)
    print(json.dumps(write_packet(args.out.resolve(), tasks, references, manifest), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
