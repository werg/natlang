#!/usr/bin/env python3
"""Create a held, run-scoped CallStore snapshot from exact invocation evidence.

The source store remains live and untouched. SQLite's online backup API provides
a transactionally consistent image; this script then retains only calls reached
by explicitly pinned invocation IDs plus their descendant calls and blob links.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def invocation_ids(spec: dict[str, Any], base: Path) -> tuple[set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    pins: list[dict[str, Any]] = []
    for evidence in spec["evidence"]:
        path = (base / evidence["path"]).resolve()
        if not path.is_file():
            raise ValueError(f"missing evidence file: {path}")
        digest = sha256(path)
        expected = evidence.get("sha256")
        if expected and digest != expected:
            raise ValueError(f"evidence hash mismatch for {path}: {digest} != {expected}")
        pins.append({"path": evidence["path"], "sha256": digest, "id_field": evidence.get("id_field"),
                     "format": evidence["format"], "case_id": evidence.get("case_id")})
        if evidence.get("select", True) is False:
            continue
        if evidence["format"] == "jsonl":
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    value = row
                    for part in evidence["id_field"].split("."):
                        value = value[part]
                    if not isinstance(value, str) or not value:
                        raise ValueError(f"invalid invocation id at {path}:{line_no}")
                    ids.add(value)
                    if evidence.get("first_line"):
                        break
        elif evidence["format"] == "json":
            value = json.loads(path.read_text(encoding="utf-8"))
            for part in evidence["id_field"].split("."):
                value = value[part]
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"invalid invocation ids in {path}")
            ids.update(values)
        else:
            raise ValueError(f"unsupported evidence format {evidence['format']!r}")
    if not ids:
        raise ValueError(f"run {spec['run_id']} selected no invocation IDs")
    return ids, pins


def table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def snapshot(spec_path: Path, store_root: Path, out_root: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema") != "natlang.call_store_snapshot_spec/1":
        raise ValueError("unsupported snapshot spec schema")
    source_db = store_root / "calls.sqlite"
    source_blobs = store_root / "blobs"
    if not source_db.is_file() or not source_blobs.is_dir():
        raise ValueError(f"CallStore not found at {store_root}")
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    selected_ids: set[str] = set()
    run_pins: list[dict[str, Any]] = []
    ids_by_run: dict[str, set[str]] = {}
    run_origin: dict[str, str] = {}
    spec_base = spec_path.parent
    for run in spec["runs"]:
        if run.get("origin_class") not in ("scripted", "provider"):
            raise ValueError(f"run {run.get('run_id')} needs explicit origin_class")
        if not run.get("classification_evidence"):
            raise ValueError(f"run {run.get('run_id')} needs classification evidence")
        ids, evidence_pins = invocation_ids(run, spec_base)
        selected_ids.update(ids)
        ids_by_run[run["run_id"]] = ids
        run_origin[run["run_id"]] = run["origin_class"]
        run_pins.append({"run_id": run["run_id"], "origin_class": run["origin_class"],
                         "campaign_id": run.get("campaign_id"), "case_id": run.get("case_id"),
                         "source_program_id": run.get("source_program_id"),
                         "plan_sha256": run.get("plan_sha256"), "source_sha256": run.get("source_sha256"),
                         "runtime_sha256": run.get("runtime_sha256"),
                         "classification_evidence": run["classification_evidence"],
                         "evidence": evidence_pins, "invocation_id_count": len(ids),
                         "invocation_ids_sha256": canonical_hash("\n".join(sorted(ids)).encode()),
                         "invocation_ids": sorted(ids)})
    evidence_id_owners: dict[str, list[str]] = {}
    for run_id, ids in ids_by_run.items():
        for evidence_id in ids:
            evidence_id_owners.setdefault(evidence_id, []).append(run_id)
    cross_origin_collisions = [{"evidence_id": evidence_id, "run_ids": sorted(owners),
                                "origin_classes": sorted({run_origin[owner] for owner in owners})}
                               for evidence_id, owners in evidence_id_owners.items()
                               if len({run_origin[owner] for owner in owners}) > 1]
    if cross_origin_collisions:
        raise ValueError(f"invocation ID collision crosses origin classes: {cross_origin_collisions[:5]}")

    out_db = out_root / "calls.sqlite"
    # Online backup reads a coherent SQLite snapshot while writers may still be open.
    src = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=30)
    backup = sqlite3.connect(out_db)
    try:
        src.backup(backup, pages=256, sleep=0.05)
    finally:
        backup.close()
        src.close()

    db = sqlite3.connect(out_db)
    db.execute("PRAGMA foreign_keys=OFF")
    db.execute("BEGIN IMMEDIATE")
    known_tables = {"blobs", "calls", "call_blobs", "annotations", "compilations", "compilation_files",
                    "cases", "case_calls", "declines", "jobs", "meta", "iteration_statistics"}
    present_tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'") if not r[0].startswith("sqlite_")}
    unknown_tables = sorted(present_tables - known_tables)
    if unknown_tables:
        db.rollback()
        db.close()
        raise ValueError(f"snapshot refuses unknown CallStore tables: {unknown_tables}")
    db.execute("CREATE TEMP TABLE selected_call_ids (id TEXT PRIMARY KEY)")
    before_rows = db.execute("SELECT call_id,task_id,parent_call_id,started_at,executor,model_id,model_requests,record_hash,events_hash FROM calls").fetchall()
    by_call = {r[0]: r for r in before_rows}
    chosen: set[str] = set()
    roots_found = {r[0] for r in before_rows if r[0] in selected_ids or r[2] in selected_ids}
    chosen.update(roots_found)
    # Preserve exact descendants whose parent is a selected recorded call.
    while True:
        more = {r[0] for r in before_rows if r[2] in chosen} - chosen
        if not more:
            break
        chosen.update(more)
    db.executemany("INSERT INTO selected_call_ids VALUES (?)", ((value,) for value in sorted(chosen)))

    run_call_ids: dict[str, set[str]] = {}
    run_link_details: dict[str, dict[str, dict[str, set[str]]]] = {}
    for run_id, ids in ids_by_run.items():
        details: dict[str, dict[str, set[str]]] = {}
        for row in before_rows:
            call_id, parent_id = row[0], row[2]
            if call_id in ids:
                details.setdefault(call_id, {"evidence_roots": set(), "paths": set()})
                details[call_id]["evidence_roots"].add(call_id)
                details[call_id]["paths"].add("exact_call_id")
            if parent_id in ids:
                details.setdefault(call_id, {"evidence_roots": set(), "paths": set()})
                details[call_id]["evidence_roots"].add(parent_id)
                details[call_id]["paths"].add("exact_parent_call_id")
        while True:
            added = False
            for row in before_rows:
                call_id, parent_id = row[0], row[2]
                if parent_id in details:
                    parent_roots = details[parent_id]["evidence_roots"]
                    if call_id not in details:
                        details[call_id] = {"evidence_roots": set(parent_roots), "paths": {"descendant_parent_call_id"}}
                        added = True
                    else:
                        before = len(details[call_id]["evidence_roots"])
                        details[call_id]["evidence_roots"].update(parent_roots)
                        if len(details[call_id]["evidence_roots"]) != before:
                            added = True
                        details[call_id]["paths"].add("descendant_parent_call_id")
            if not added:
                break
        run_link_details[run_id] = details
        run_call_ids[run_id] = set(details) & chosen
    call_run_links: dict[str, list[dict[str, Any]]] = {}
    for run_id, details in run_link_details.items():
        for call_id in run_call_ids[run_id]:
            call_run_links.setdefault(call_id, []).append({"run_id": run_id, "origin_class": run_origin[run_id],
                "root_evidence_ids": sorted(details[call_id]["evidence_roots"]),
                "matching_paths": sorted(details[call_id]["paths"])})

    selected_rows = [r for r in before_rows if r[0] in chosen]
    matched_evidence_ids = {item for item in selected_ids if item in by_call or any(r[2] == item for r in before_rows)}
    call_hashes = {r[7] for r in selected_rows if r[7]}
    call_hashes.update(r[8] for r in selected_rows if r[8])
    if table_exists(db, "call_blobs"):
        call_hashes.update(r[0] for r in db.execute(
            "SELECT DISTINCT cb.hash FROM call_blobs cb JOIN selected_call_ids s ON s.id=cb.call_id") if r[0])

    case_hashes: set[str] = set()
    compilation_ids: set[str] = set()
    if chosen and table_exists(db, "case_calls"):
        links = db.execute("SELECT DISTINCT cc.case_hash FROM case_calls cc JOIN selected_call_ids s ON s.id=cc.call_id").fetchall()
        case_hashes = {r[0] for r in links}
        db.execute("DELETE FROM case_calls WHERE call_id NOT IN (SELECT id FROM selected_call_ids)")
        if case_hashes and table_exists(db, "cases"):
            db.execute("CREATE TEMP TABLE selected_case_hashes (id TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO selected_case_hashes VALUES (?)", ((value,) for value in sorted(case_hashes)))
            compilation_ids = {r[0] for r in db.execute("SELECT DISTINCT c.compilation_id FROM cases c JOIN selected_case_hashes s ON s.id=c.hash")}
            db.execute("DELETE FROM cases WHERE hash NOT IN (SELECT id FROM selected_case_hashes)")
        else:
            db.execute("DELETE FROM cases")
    else:
        for table in ("case_calls", "cases"):
            if table_exists(db, table): db.execute(f"DELETE FROM {table}")
    if table_exists(db, "compilations"):
        if compilation_ids:
            db.execute("CREATE TEMP TABLE selected_compilation_ids (id TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO selected_compilation_ids VALUES (?)", ((value,) for value in sorted(compilation_ids)))
            db.execute("DELETE FROM compilations WHERE id NOT IN (SELECT id FROM selected_compilation_ids)")
            db.execute("DELETE FROM compilation_files WHERE compilation_id NOT IN (SELECT id FROM selected_compilation_ids)")
            call_hashes.update(r[0] for r in db.execute(
                "SELECT hash FROM compilation_files WHERE compilation_id IN (SELECT id FROM selected_compilation_ids)") if r[0])
        else:
            db.execute("DELETE FROM compilations")
            if table_exists(db, "compilation_files"): db.execute("DELETE FROM compilation_files")
    for table, key in (("calls", "call_id"), ("annotations", "call_id"), ("jobs", "call_id")):
        if table_exists(db, table):
            db.execute(f"DELETE FROM {table} WHERE {key} NOT IN (SELECT id FROM selected_call_ids)")
    if table_exists(db, "blobs"):
        if call_hashes:
            db.execute("CREATE TEMP TABLE selected_blob_hashes (id TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO selected_blob_hashes VALUES (?)", ((value,) for value in sorted(call_hashes)))
            db.execute("DELETE FROM blobs WHERE hash NOT IN (SELECT id FROM selected_blob_hashes)")
        else:
            db.execute("DELETE FROM blobs")
    if table_exists(db, "call_blobs"):
        db.execute("DELETE FROM call_blobs WHERE call_id NOT IN (SELECT id FROM selected_call_ids)")
    # Declines have no call/run relation and cannot be safely attributed.
    if table_exists(db, "declines"):
        db.execute("DELETE FROM declines")
    # Keep operational singleton metadata untouched; clear global aggregates that
    # cannot be assigned to the selected runs.
    for table in ("iteration_statistics",):
        if table_exists(db, table): db.execute(f"DELETE FROM {table}")
    db.commit()
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.execute("VACUUM")
    integrity = [r[0] for r in db.execute("PRAGMA integrity_check")]
    foreign_key_violations = [list(r) for r in db.execute("PRAGMA foreign_key_check")]
    foreign_key_tables = sorted({r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")})
    declared_foreign_keys = {table: [list(row) for row in db.execute(f"PRAGMA foreign_key_list('{table}')")]
                             for table in foreign_key_tables}
    declared_foreign_keys = {table: rows for table, rows in declared_foreign_keys.items() if rows}
    if integrity != ["ok"] or foreign_key_violations:
        db.close()
        raise ValueError(f"snapshot SQLite integrity failure: integrity={integrity}, foreign_keys={foreign_key_violations}")
    db.close()

    blob_out = out_root / "blobs"
    missing: list[dict[str, str]] = []
    copied: list[dict[str, Any]] = []
    src_ro = sqlite3.connect(f"file:{out_db}?mode=ro", uri=True)
    blob_meta = {r[0]: r[1] for r in src_ro.execute("SELECT hash,bytes FROM blobs")}
    src_ro.close()
    for digest in sorted(call_hashes):
        source = source_blobs / digest[:2] / digest[2:]
        target = blob_out / digest[:2] / digest[2:]
        if not source.is_file():
            missing.append({"hash": digest, "reason": "referenced blob absent from source store"})
            continue
        actual = sha256(source)
        if actual != digest:
            missing.append({"hash": digest, "reason": f"source blob bytes hash to {actual}"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != digest:
            missing.append({"hash": digest, "reason": "copied blob failed destination hash verification"})
            target.unlink()
            continue
        if digest in blob_meta and target.stat().st_size != blob_meta[digest]:
            missing.append({"hash": digest, "reason": "blob size differs from CallStore blobs table"})
            target.unlink()
            continue
        copied.append({"hash": digest, "bytes": target.stat().st_size,
                       "store_bytes": blob_meta.get(digest), "path": target.relative_to(out_root).as_posix()})

    # Preserve all non-selected calls as IDs/metadata only, so ambiguous attribution is visible.
    unresolved = [{"call_id": r[0], "task_id": r[1], "parent_call_id": r[2], "started_at": r[3],
                   "executor": r[4], "model_id": r[5], "model_requests": r[6],
                   "selection": "no exact invocation evidence matched"}
                  for r in before_rows if r[0] not in chosen]
    unresolved_path = out_root / "unresolved-calls.json"
    unresolved_path.write_text(json.dumps(unresolved, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    db_hash = sha256(out_db)
    match_path_counts: dict[str, int] = {}
    for call_link in call_run_links.values():
        for run_link in call_link:
            for path in run_link["matching_paths"]:
                match_path_counts[path] = match_path_counts.get(path, 0) + 1
    result = {
        "schema": "natlang.call_store_run_snapshot/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_store_root": str(store_root.resolve()),
        "source_database_bytes": source_db.stat().st_size,
        "source_main_db_sha256_non_snapshot": sha256(source_db),
        "source_wal_present_at_capture": (store_root / "calls.sqlite-wal").exists(),
        "snapshot_database": {"path": "calls.sqlite", "sha256": db_hash, "bytes": out_db.stat().st_size,
                               "method": "Python sqlite3 online backup API"},
        "runs": [{**run, "selected_call_ids": sorted(run_call_ids[run["run_id"]]),
                  "selected_call_count": len(run_call_ids[run["run_id"]])} for run in run_pins],
        "cross_origin_invocation_id_collisions": cross_origin_collisions,
        "call_run_links": [{"call_id": call_id, "parent_call_id": by_call[call_id][2],
                            "task_id": by_call[call_id][1], "started_at": by_call[call_id][3],
                            "executor": by_call[call_id][4], "model_id": by_call[call_id][5],
                            "model_requests": by_call[call_id][6],
                            "record_hash": by_call[call_id][7], "events_hash": by_call[call_id][8],
                            "runs": sorted(run_links, key=lambda item: item["run_id"])}
                           for call_id, run_links in sorted(call_run_links.items())],
        "external_parent_links": [{"call_id": row[0], "parent_call_id": row[2],
                                    "runs": call_run_links.get(row[0], [])}
                                   for row in selected_rows if row[2] and row[2] not in chosen],
        "unknown_source_tables": unknown_tables,
        "selection": {"exact_evidence_id_count": len(selected_ids), "matched_evidence_id_count": len(matched_evidence_ids),
                      "unmatched_evidence_ids": sorted(selected_ids - matched_evidence_ids),
                      "selection_rule": "Exact invocation ID match to calls.call_id or calls.parent_call_id, then transitive parent_call_id closure; no timestamp, task-name, or substring matching.",
                      "matched_root_call_count": len(roots_found),
                      "selected_call_count": len(chosen), "unresolved_global_call_count": len(unresolved),
                      "matching_path_counts": match_path_counts,
                      "selected_call_ids": sorted(chosen),
                      "selected_call_ids_sha256": canonical_hash("\n".join(sorted(chosen)).encode()),
                      "unresolved_index": "unresolved-calls.json"},
        "sqlite_validation": {"integrity_check": integrity, "foreign_key_violations": foreign_key_violations,
                              "declared_foreign_keys": declared_foreign_keys,
                              "note": "No foreign-key constraints are declared in the current CallStore schema; external parent_call_id links are retained as strings."},
        "blobs": {"referenced_hash_count": len(call_hashes), "copied_verified_count": len(copied),
                  "missing_or_mismatched": missing, "files": copied},
        "linked_compilation_ids": sorted(compilation_ids),
        "linked_case_hashes": sorted(case_hashes),
        "training_admission": False,
        "trace_admission": False,
        "limitations": ["CallStore has no campaign/run column; run linkage is supplied only by pinned invocation evidence.",
                        "Unmatched calls remain unresolved and are not included in the snapshot database.",
                        "Origin class is inherited from explicit run evidence; it is not inferred from executor kind alone."],
    }
    provenance_path = out_root / "provenance.json"
    provenance_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path, help="pinned run/evidence specification JSON")
    parser.add_argument("--store", type=Path, default=Path.home() / ".local/share/natlang/calls")
    parser.add_argument("--out", required=True, type=Path, help="new or empty output directory")
    args = parser.parse_args()
    try:
        result = snapshot(args.spec.resolve(), args.store.resolve(), args.out.resolve())
    except Exception as exc:
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "created", "out": str(args.out.resolve()),
                      "selected_calls": result["selection"]["selected_call_count"],
                      "unresolved_calls": result["selection"]["unresolved_global_call_count"],
                      "verified_blobs": result["blobs"]["copied_verified_count"],
                      "missing_blobs": len(result["blobs"]["missing_or_mismatched"]),
                      "snapshot_sha256": result["snapshot_database"]["sha256"]}, sort_keys=True))
    return 0 if not result["blobs"]["missing_or_mismatched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
