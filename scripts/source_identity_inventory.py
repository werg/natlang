#!/usr/bin/env python3
"""Build a provenance-pinned inventory of source IDs and groups in case artifacts.

The inventory deliberately includes registered files and unregistered canonical
case artifacts under runs/, data/, and training/. Registration and availability
do not imply admission; this is only a deduplication aid.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402


CASE_NAMES = ("source.cases.jsonl", "cases.jsonl", "cases.ir.jsonl", "*.ir.jsonl")
PRUNED_DIRS = {".git", "node_modules", ".venv", "venv", "vendor", "dist", "target",
               "__pycache__", "toolchain", "runtime", "runtime-build", "deps", ".cache"}
ID_KEYS = {"source_id", "sourceId", "source_record_id", "sourceRecordId",
           "independent_world", "independentWorld", "world_id", "worldId"}
GROUP_KEYS = {"source_group", "sourceGroup", "source_group_id", "sourceGroupId",
              "group_id", "groupId", "split_group", "splitGroup", "pair_group", "pairGroup"}


def string_values(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list):
        return [item for child in value for item in string_values(child)]
    return []


def extract(obj: Any, ids: set[str], groups: set[str]) -> None:
    if isinstance(obj, dict):
        for key in ("source_ids", "sourceIds", "source_record_ids", "sourceRecordIds"):
            ids.update(string_values(obj.get(key)))
        for key in ("source_groups", "sourceGroups", "group_ids", "groupIds"):
            groups.update(string_values(obj.get(key)))
        for key in ID_KEYS:
            ids.update(string_values(obj.get(key)))
        for key in GROUP_KEYS:
            groups.update(string_values(obj.get(key)))
        # Canonical builder rows often carry identity in nested records as {id, group}.
        records = (obj.get("dataset_records") or obj.get("datasetRecords")
                   or obj.get("source_records") or obj.get("sourceRecords"))
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    ids.update(string_values(record.get("id")))
                    groups.update(string_values(record.get("group")))
        for value in obj.values():
            if isinstance(value, (dict, list)):
                extract(value, ids, groups)
    elif isinstance(obj, list):
        for value in obj:
            extract(value, ids, groups)


def is_case_artifact(path: Path) -> bool:
    name = path.name
    return (name == "source.cases.jsonl" or name == "cases.jsonl"
            or name == "cases.ir.jsonl" or name.endswith(".ir.jsonl"))


def include_matches(relative: str, pattern: str) -> bool:
    # pathlib glob treats **/ as zero or more path components; fnmatch does not.
    return fnmatch.fnmatchcase(relative, pattern) or (
        pattern.startswith("**/") and fnmatch.fnmatchcase(relative, pattern[3:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", type=Path, default=Path("training/neuralese_corpora.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    registry = json.loads(registry_path.read_text())

    # Registry snapshot manifests are a second authority for explicitly listed
    # files. Read them separately so canonical source bundles beneath pruned build
    # or vendor trees are not silently omitted.
    manifest_root = root / "training/corpus-manifests"
    manifests_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    if manifest_root.exists():
        for manifest_path in manifest_root.glob("*.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(manifest.get("id"), str):
                manifests_by_id[manifest["id"]] = (manifest_path.resolve(), manifest)

    # Discover canonical case artifacts globally so held proposals and old bundles
    # omitted from the registry still participate in source deduplication.
    files: set[Path] = set()
    for base_name in ("runs", "data", "training"):
        base = root / base_name
        if base.exists():
            before = len(files)
            # os.walk avoids repeated pathlib glob expansion and does not follow
            # directory symlinks into duplicate or external trees. Build/runtime
            # dependency trees cannot contain source corpus artifacts and are pruned.
            for directory, dirnames, filenames in os.walk(base, followlinks=False):
                dirnames[:] = [name for name in dirnames if name not in PRUNED_DIRS
                               and not (Path(directory) / name).is_symlink()]
                for name in filenames:
                    if name in CASE_NAMES[:-1] or name.endswith(".ir.jsonl"):
                        candidate = Path(directory) / name
                        if candidate.is_file():
                            resolved = candidate.resolve()
                            if resolved == root or root in resolved.parents:
                                files.add(resolved)
            print(f"discovered {len(files) - before} canonical case artifacts under {base_name}", file=sys.stderr, flush=True)

    # Resolve explicit registry include patterns and include recognized case files.
    corpus_rows = []
    registry_manifest_rows = []
    manifest_ids: set[str] = set()
    manifest_groups: set[str] = set()
    coverage_omissions = []
    for corpus in registry.get("corpora", []):
        base = (root / corpus.get("path", "")).resolve()
        include = corpus.get("include", ["**/*"])
        matched: list[Path] = []
        manifest_entry = manifests_by_id.get(corpus.get("id"))
        manifest_case_paths: list[str] = []
        if manifest_entry:
            manifest_path, manifest = manifest_entry
            manifest_source_ids: set[str] = set()
            manifest_source_groups: set[str] = set()
            extract(manifest, manifest_source_ids, manifest_source_groups)
            manifest_ids.update(manifest_source_ids)
            manifest_groups.update(manifest_source_groups)
            for listed in manifest.get("files", []):
                if not isinstance(listed, dict) or not isinstance(listed.get("path"), str):
                    continue
                rel_path = listed["path"]
                if not is_case_artifact(Path(rel_path)):
                    continue
                target = (base / rel_path).resolve()
                if target.exists() and (target == root or root in target.parents):
                    files.add(target)
                    manifest_case_paths.append(str(target.relative_to(root)))
            registry_manifest_rows.append({
                "corpus_id": corpus.get("id"),
                "path": str(manifest_path.relative_to(root)),
                "sha256": sha256_file(manifest_path),
                "source_ids": sorted(manifest_source_ids),
                "source_groups": sorted(manifest_source_groups),
                "listed_case_artifacts": sorted(set(manifest_case_paths)),
            })
        for candidate in files:
            try:
                relative = candidate.relative_to(base).as_posix()
            except ValueError:
                continue
            if is_case_artifact(candidate) and any(include_matches(relative, pattern) for pattern in include):
                matched.append(candidate)
        corpus_rows.append({
            "id": corpus.get("id"),
            "owner": corpus.get("owner"),
            "path": corpus.get("path"),
            "admission": corpus.get("admission"),
            "case_artifacts": sorted(str(path.relative_to(root)) for path in set(matched)),
        })
        if not matched and not manifest_case_paths:
            coverage_omissions.append({
                "corpus_id": corpus.get("id"),
                "path": corpus.get("path"),
                "admission": corpus.get("admission"),
                "reason": "No canonical source/case/IR JSONL listed by its manifest or matched under the registry include patterns; this inventory cannot establish its source identities.",
            })

    artifacts = []
    all_ids: set[str] = set()
    all_groups: set[str] = set()
    inode_cache: dict[tuple[int, int, int, int], tuple[str, int, int, list[str], list[str]]] = {}
    reused_hardlinks = 0
    ordered_files = sorted(files)
    for index, path in enumerate(ordered_files, 1):
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = inode_cache.get(signature)
        if cached:
            digest, size, row_count, cached_ids, cached_groups = cached
            ids, groups = set(cached_ids), set(cached_groups)
            malformed = 0
            reused_hardlinks += 1
        else:
            digest_builder = hashlib.sha256()
            ids: set[str] = set()
            groups: set[str] = set()
            row_count = 0
            malformed = 0
            with path.open("rb") as stream:
                for raw in stream:
                    digest_builder.update(raw)
                    if not raw.strip():
                        continue
                    row_count += 1
                    try:
                        extract(json.loads(raw), ids, groups)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        malformed += 1
            digest = digest_builder.hexdigest()
            size = stat.st_size
            inode_cache[signature] = (digest, size, row_count, sorted(ids), sorted(groups))
        all_ids.update(ids)
        all_groups.update(groups)
        artifacts.append({
            "path": str(path.relative_to(root)),
            "sha256": digest,
            "bytes": size,
            "rows": row_count,
            "malformed_rows": malformed,
            "source_ids": sorted(ids),
            "source_groups": sorted(groups),
        })
        if index % 25 == 0 or index == len(ordered_files):
            print(f"[{index}/{len(ordered_files)}] {path.relative_to(root)} ({row_count} rows)", file=sys.stderr, flush=True)

    payload = {
        "schema": "natlang.source_identity_inventory/1",
        "builder": {"path": str(Path(__file__).resolve().relative_to(root)), "sha256": sha256_file(Path(__file__))},
        "registry": {"path": str(registry_path.relative_to(root)), "sha256": sha256_file(registry_path)},
        "scope": "registered case artifacts plus canonical case JSONL under runs/, data/, and training/",
        "admission": "inventory only; does not imply source quality or training admission",
        "counts": {"registry_corpora": len(corpus_rows), "case_artifacts": len(artifacts),
                   "source_ids": len(all_ids), "source_groups": len(all_groups),
                   "reused_hardlink_artifacts": reused_hardlinks},
        "source_ids": sorted(all_ids | manifest_ids),
        "source_groups": sorted(all_groups | manifest_groups),
        "corpora": corpus_rows,
        "registry_manifests": registry_manifest_rows,
        "registry_coverage_omissions": coverage_omissions,
        "artifacts": artifacts,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
