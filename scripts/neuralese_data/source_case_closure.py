"""Project authored semantic source cases through the shared port-record closure API.

This is a provenance/closure adapter, not a trainer or semantic grader. It accepts source-case
JSONL rows with ``semantics.expected``, ``semantics.expected_files`` (or ``folder_files``),
``semantics.root``/``semantics.files``, plus native target rows whose
``task.program_ir.id`` names the source case. The native action target remains provenance only;
the port target always comes from the authored source case's semantic expected value.

Example::

  python3 -m scripts.neuralese_data.source_case_closure \\
    --source cases.jsonl --candidate-rows selected-native-turns.jsonl \\
    --source-manifest source-manifest.json \\
    --index cross-corpus-index-s1-full-final-20261003-v1 \\
    --index cross-corpus-index-harness-bench-swe-rebench-openhands-pi-records-20261010-v4-v1 \\
    --protected data/neuralese/corpora/protected-bgkit-20261003/bgkit-benchmarks.protected.json \\
    --out runs/source-case-closure-v1

It writes immutable-input hashes, the validated port projection, candidate-to-case bindings,
the apply result, and check receipts. It never changes its input records or grants training
admission. Use a fresh output directory for each run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from . import cross_corpus, cross_corpus_registry, records
from .splits import check_closed, protected_hit


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dotted(value: dict, path: str, *, where: str):
    cur = value
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise ValueError(f"{where}: missing {path}")
        cur = cur[key]
    return cur


def canonical_row_sha256(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_source_manifest(source: Path, manifest_path: Path) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("source")
    if not isinstance(declared, dict) or not isinstance(declared.get("sha256"), str):
        raise ValueError("source manifest must declare source.sha256")
    actual = sha256(source)
    if actual != declared["sha256"]:
        raise ValueError(f"source manifest SHA mismatch: expected {declared['sha256']}, got {actual}")
    declared_path = declared.get("path")
    receipt = {"path": str(manifest_path), "sha256": sha256(manifest_path),
               "source_path_declared": declared_path, "source_path_used": str(source),
               "source_sha256_declared": declared["sha256"], "source_sha256_used": actual,
               "source_binding_matches": True}
    return manifest, receipt


def verify_raw_lineage(manifest: dict, manifest_path: Path) -> tuple[dict[str, dict], dict | None]:
    """Load a manifest's raw trajectory rows when they are explicitly declared."""
    raw_spec = manifest.get("raw_results")
    if raw_spec is None:
        return {}, None
    if not isinstance(raw_spec, dict) or not isinstance(raw_spec.get("path"), str) or not isinstance(raw_spec.get("sha256"), str):
        raise ValueError("source manifest raw_results must declare path and sha256")
    raw_path = manifest_path.parent / raw_spec["path"]
    if not raw_path.is_file():
        raise ValueError(f"manifest raw_results file is missing: {raw_path}")
    raw_sha = sha256(raw_path)
    if raw_sha != raw_spec["sha256"]:
        raise ValueError(f"source manifest raw_results SHA mismatch: expected {raw_spec['sha256']}, got {raw_sha}")
    rows = read_jsonl(raw_path)
    expected_rows = raw_spec.get("rows")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"source manifest raw_results row count mismatch: expected {expected_rows}, got {len(rows)}")
    by_lineage = {}
    for row in rows:
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("raw_results row requires a nonempty id")
        binding = (row_id, canonical_row_sha256(row))
        if binding in by_lineage:
            raise ValueError(f"duplicate raw_results lineage binding: {row_id}")
        by_lineage[binding] = row
    return by_lineage, {"path": str(raw_path), "sha256": raw_sha, "rows": len(rows),
                        "manifest_sha256": raw_spec["sha256"], "manifest_rows": expected_rows}


def verify_registered_index(reference: str, index_root: Path) -> dict:
    """Verify registered index manifest file hashes when an index is registered locally."""
    registry = json.loads((ROOT / "training/neuralese_corpora.json").read_text(encoding="utf-8"))
    entry = next((item for item in registry["corpora"]
                  if item.get("kind") == "cross-corpus-index" and
                  (item.get("id") == reference or (ROOT / item.get("path", "")) == index_root)), None)
    if entry is None:
        meta = json.loads((index_root / "index.json").read_text(encoding="utf-8"))
        return {"reference": reference, "path": str(index_root), "registered": False,
                "manifest_verified": False, "index_sha256": sha256(index_root / "index.json"),
                "format": meta.get("format"), "records": meta.get("records"), "groups": meta.get("groups")}
    manifest_path = ROOT / f"training/corpus-manifests/{entry['id']}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_files = []
    for item in manifest["files"]:
        path = index_root / item["path"]
        actual = sha256(path)
        matches = path.stat().st_size == item["bytes"] and actual == item["sha256"]
        verified_files.append({"path": str(path), "bytes": path.stat().st_size,
                               "sha256": actual, "manifest_bytes": item["bytes"],
                               "manifest_sha256": item["sha256"], "matches": matches})
    if not all(item["matches"] for item in verified_files):
        raise ValueError(f"registered cross-corpus index manifest mismatch for {entry['id']}")
    meta = json.loads((index_root / "index.json").read_text(encoding="utf-8"))
    return {"reference": reference, "registered_id": entry["id"], "path": str(index_root),
            "registered": True, "manifest_path": str(manifest_path),
            "manifest_sha256": sha256(manifest_path), "manifest_verified": True,
            "index_sha256": sha256(index_root / "index.json"), "format": meta.get("format"),
            "records": meta.get("records"), "groups": meta.get("groups"),
            "files": verified_files}


def source_case_projection(case: dict, bound_candidates: list[dict], raw_lineage: dict[tuple[str, str], dict]) -> dict:
    """Create one auditable port component per source case, not per native action."""
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("source case requires nonempty id")
    semantics = case.get("semantics")
    if not isinstance(semantics, dict) or "expected" not in semantics:
        raise ValueError(f"{case_id}: semantics.expected is required")
    expected = semantics["expected"]
    file_map = semantics.get("expected_files") or semantics.get("folder_files")
    if not isinstance(file_map, dict) or "task.json" not in file_map:
        raise ValueError(f"{case_id}: semantics expected_files/folder_files must contain task.json")
    task_text = file_map["task.json"]
    if not isinstance(task_text, str):
        raise ValueError(f"{case_id}: task.json must be text")
    task = json.loads(task_text)
    output_path = task.get("output_path")
    if not isinstance(output_path, str) or output_path not in file_map:
        raise ValueError(f"{case_id}: task.json output_path must name a declared output file")
    root_name = semantics.get("root")
    root_text = (semantics.get("files") or {}).get(root_name)
    if not isinstance(root_name, str) or not isinstance(root_text, str) or not root_text:
        raise ValueError(f"{case_id}: semantics.files must contain the exact callable root")
    groups = set(case.get("source_groups") or [])
    if not groups:
        raise ValueError(f"{case_id}: source_groups are required")
    if not bound_candidates:
        raise ValueError(f"{case_id}: no candidate rows bind to this source case")
    candidate_ids = [candidate.get("id") for candidate in bound_candidates]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        raise ValueError(f"{case_id}: each candidate row requires a nonempty id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError(f"{case_id}: duplicate candidate row IDs")
    if any(not groups.intersection(candidate.get("source_ids") or []) for candidate in bound_candidates):
        raise ValueError(f"{case_id}: candidate row lacks a matching inherited source_id")
    candidate_groups = set().union(*(set(c.get("source_groups") or []) for c in bound_candidates))
    if case_id not in candidate_groups:
        raise ValueError(f"{case_id}: candidate rows do not carry their source-case program group")
    splits = {candidate.get("split") for candidate in bound_candidates}
    case_split = case.get("split")
    if splits != {case_split} or case_split not in {"train", "validation", "test"}:
        raise ValueError(f"{case_id}: source-case split disagrees with candidate rows: {splits}")

    raw_rows = []
    for candidate in bound_candidates:
        source_ref = candidate.get("source_ref")
        if not isinstance(source_ref, dict):
            if raw_lineage:
                raise ValueError(f"{case_id}: candidate {candidate['id']} has no source_ref for manifest-declared raw lineage")
            continue
        trajectory_id = source_ref.get("trajectory_id")
        source_row_sha = source_ref.get("source_row_sha256")
        if raw_lineage:
            if not isinstance(trajectory_id, str) or not isinstance(source_row_sha, str):
                raise ValueError(f"{case_id}: candidate {candidate['id']} lacks trajectory_id/source_row_sha256")
            raw = raw_lineage.get((trajectory_id, source_row_sha))
            if raw is None:
                raise ValueError(f"{case_id}: candidate {candidate['id']} source_ref does not match manifest raw_results")
            raw_program_id = dotted(raw, "task.program_ir.id", where=f"raw trajectory {trajectory_id}")
            if raw_program_id != case_id or source_ref.get("program_ir_id") != raw_program_id:
                raise ValueError(f"{case_id}: candidate {candidate['id']} program ID disagrees with raw source lineage")
            raw_rows.append((trajectory_id, source_row_sha, raw))

    if raw_lineage:
        raw_bindings = {(trajectory_id, row_sha) for trajectory_id, row_sha, _ in raw_rows}
        if len(raw_bindings) != 1:
            raise ValueError(f"{case_id}: candidates bind to multiple raw source trajectories")

    def first_user_request(candidate: dict):
        messages = [message for message in candidate.get("messages", []) if message.get("role") == "user"]
        return messages[0].get("content") if messages else None

    # The first request for the parent invocation is the consumer question. Child
    # invocations naturally have their own user prompts, so select the root result
    # through its explicit call/parent metadata and verify the raw lineage above.
    root_candidates = []
    for candidate in bound_candidates:
        source_ref = candidate.get("source_ref") or {}
        capture = (source_ref.get("host_result_capture") or {}).get("capture") or {}
        if (capture.get("capture_kind") == "invocation_output" and not capture.get("parent_call_id") and
                capture.get("call_id") == source_ref.get("invocation_id")):
            root_candidates.append(candidate)
    if raw_lineage and not root_candidates:
        raise ValueError(f"{case_id}: no root invocation result identifies the actual consumer request")
    if root_candidates:
        root_requests = [first_user_request(candidate) for candidate in root_candidates]
        request_fingerprints = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in root_requests}
        if None in root_requests or len(request_fingerprints) != 1:
            raise ValueError(f"{case_id}: root invocation candidates do not share one actual first user request")
        request = root_requests[0]
    else:
        # Without raw lineage, require the candidate rows to attest the same first
        # user request rather than silently taking whichever row sorts first.
        requests = [first_user_request(candidate) for candidate in bound_candidates]
        request_fingerprints = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in requests}
        if None in requests or len(request_fingerprints) != 1:
            raise ValueError(f"{case_id}: candidate rows do not share one actual first user request")
        request = requests[0]
    if not isinstance(request, str):
        raise ValueError(f"{case_id}: root first user request must be text")

    visible_sources = [records.source("file", root_text, title=root_name,
                                      meta={"path": root_name, "role": "callable-root"}, exact_refs=[])]
    for path, body in sorted(file_map.items()):
        if path == output_path:
            continue
        if not isinstance(body, str) or not body:
            raise ValueError(f"{case_id}: declared visible file {path} is empty or non-text")
        role = "record" if path.startswith("records/") else "file"
        visible_sources.append(records.source(role, body, title=path,
                                              meta={"path": path, "role": "visible-input"}, exact_refs=[]))

    question = task.get("instruction") or task.get("criterion") or json.dumps(task, ensure_ascii=False)
    case_sha = hashlib.sha256(json.dumps(case, sort_keys=True, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    candidate_bindings = [{"id": candidate["id"], "source_ref": candidate.get("source_ref"),
                           "source_ids": candidate.get("source_ids", []),
                           "source_groups": candidate.get("source_groups", [])}
                          for candidate in bound_candidates]
    kind = "text" if isinstance(expected, str) else "json"
    record = {
        "version": records.VERSION,
        "id": "natlang:source-case-" + hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:24],
        "family": "authored_source_case_closure",
        "task": "consume",
        "sources": visible_sources,
        "writer": {"instructions": "Resolve the caller's task from the visible evidence files.",
                   "result_type": "string" if kind == "text" else "JSON", "context": []},
        "consumer": {"context": [{"role": "user", "content": request}], "withheld": ["sources"]},
        "target": {"kind": kind, "value": expected},
        "outcome": {"label": "checked", "checked": "source-case closure projection only; source semantics and training admission are separate"},
        "lineage": {"project": "natlang", "store": "authored-source-case-closure-projection", "row": case_id,
                    "upstream": "authored-source-case-jsonl", "upstream_id": case_id,
                    "upstream_revision": (case.get("source_revisions") or [None])[0], "teacher": None,
                    "converter": "scripts/neuralese_data/source_case_closure.py",
                    "notes": {"question": question, "source_case_id": case_id,
                              "source_case_sha256": case_sha,
                              "semantic_target_origin": "source_case.semantics.expected",
                              "candidate_native_rows_are_provenance_only": True,
                              "candidate_rows_bound_to_component": candidate_bindings}},
        "license": {"spdx": "LicenseRef-Natlang-Project-Generated", "noncommercial": False,
                    "notes": "Analysis-only port projection from authored source cases."},
        "split": case_split,
        "split_groups": sorted(groups | candidate_groups),
    }
    records.seal(record)
    errors = records.validate_with_schema(record)
    if errors:
        raise ValueError(f"{case_id}: invalid projected port record: {errors[:10]}")
    return record


def run(args: argparse.Namespace) -> dict:
    source_path = args.source.resolve()
    candidate_path = args.candidate_rows.resolve()
    manifest_path = args.source_manifest.resolve()
    protected_path = args.protected.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is nonempty; choose a fresh --out: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source_manifest, source_pin = verify_source_manifest(source_path, manifest_path)
    raw_lineage, raw_lineage_pin = verify_raw_lineage(source_manifest, manifest_path)
    cases = read_jsonl(source_path)
    candidates = read_jsonl(candidate_path)
    if not cases or not candidates:
        raise ValueError("source and candidate JSONL inputs must be nonempty")

    case_ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("every source case requires a nonempty id")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("source case IDs are not unique")
    candidate_ids = [candidate.get("id") for candidate in candidates]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        raise ValueError("every candidate row requires a nonempty id")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate row IDs are not unique")
    candidates_by_case: dict[str, list[dict]] = {case_id: [] for case_id in case_ids}
    unbound_candidates = []
    for candidate in candidates:
        case_id = dotted(candidate, "task.program_ir.id", where="candidate row")
        if case_id not in candidates_by_case:
            unbound_candidates.append(candidate.get("id"))
        else:
            candidates_by_case[case_id].append(candidate)
    if unbound_candidates:
        raise ValueError(f"candidate rows refer to unknown source cases: {unbound_candidates[:10]}")
    if any(not values for values in candidates_by_case.values()):
        missing = [case_id for case_id, values in candidates_by_case.items() if not values]
        raise ValueError(f"source cases have no candidate rows: {missing}")

    index_roots = []
    index_receipts = []
    for reference in args.index:
        root, _registered = cross_corpus_registry.resolve(reference)
        root = root.resolve()
        index_receipts.append(verify_registered_index(reference, root))
        index_roots.append(root)
    indexes = [cross_corpus.Index(root) for root in index_roots]
    protected = json.loads(protected_path.read_text(encoding="utf-8"))

    projection = [source_case_projection(case, candidates_by_case[case["id"]], raw_lineage) for case in cases]
    projection_path = out / "source-case-port-projection.jsonl"
    projection_path.write_text("".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                                                    for record in projection), encoding="utf-8")
    bindings = []
    for record in projection:
        notes = record["lineage"]["notes"]
        bindings.append({"source_case_id": notes["source_case_id"], "projection_id": record["id"],
                         "candidate_native_row_ids": [item["id"] for item in notes["candidate_rows_bound_to_component"]],
                         "candidate_native_row_count": len(notes["candidate_rows_bound_to_component"]),
                         "semantic_target_origin": notes["semantic_target_origin"],
                         "candidate_actions_used_as_semantic_target": False,
                         "split_before": record["split"]})
    binding_path = out / "candidate-source-case-bindings.json"
    binding_path.write_text(json.dumps({"schema": "natlang.source-case-candidate-bindings/1",
                                        "candidate_rows_sha256": sha256(candidate_path),
                                        "candidate_rows": len(candidates), "bindings": bindings,
                                        "unbound_candidates": []}, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    protected_hits = {record["lineage"]["notes"]["source_case_id"]: protected_hit(record, protected)
                      for record in projection}
    kept, apply_report = cross_corpus.apply(copy.deepcopy(projection), indexes, protected=protected,
                                            log=lambda message: print(message, flush=True))
    cross_errors = cross_corpus.check(kept, indexes)
    internal_errors = check_closed(kept)
    kept_ids = {record["id"] for record in kept}
    kept_path = out / "closed-source-case-port-projection.jsonl"
    kept_path.write_text("".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                                                for record in kept), encoding="utf-8")
    report = {
        "schema": "natlang.authored-source-case-closure-report/1",
        "status": "held-closed-no-cross-corpus-violations" if not cross_errors and not internal_errors else "held-closure-violations",
        "training_admission": False,
        "source": {"path": str(source_path), "sha256": sha256(source_path), "rows": len(cases),
                   "manifest": source_pin, "raw_lineage": raw_lineage_pin},
        "candidate_rows": {"path": str(candidate_path), "sha256": sha256(candidate_path),
                           "rows": len(candidates), "binding_path": str(binding_path),
                           "binding_sha256": sha256(binding_path)},
        "projection": {"path": str(projection_path), "sha256": sha256(projection_path),
                       "records": len(projection), "basis": "one port record per source case; exact callable root and visible input files as sources; actual user request as consumer; authored semantics.expected as target"},
        "closed_projection": {"path": str(kept_path), "sha256": sha256(kept_path), "records": len(kept),
                               "omitted_projection_ids": sorted(record["id"] for record in projection if record["id"] not in kept_ids)},
        "indexes": index_receipts,
        "protected": {"path": str(protected_path), "sha256": sha256(protected_path),
                       "hits_by_source_case": protected_hits,
                       "scope": "the protected map indexes question hashes and IDs, not source text"},
        "apply": apply_report,
        "cross_corpus_check_errors": cross_errors,
        "internal_split_group_errors": internal_errors,
        "limits": ["Candidate native action targets remain provenance only; they are not interpreted as source-case answers.",
                   "Closure does not certify source semantic correctness, trajectory quality, or training/native qualification."]
    }
    report_path = out / "closure-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_sha256"] = sha256(report_path)
    print(json.dumps({"status": report["status"], "records_in": len(projection), "records_out": len(kept),
                      "candidate_rows": len(candidates), "cross_corpus_errors": len(cross_errors),
                      "internal_group_errors": len(internal_errors), "report_path": str(report_path),
                      "report_sha256": report["report_sha256"]}, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="authored source-case JSONL")
    parser.add_argument("--candidate-rows", type=Path, required=True, help="selected native rows linked to source cases")
    parser.add_argument("--source-manifest", type=Path, required=True, help="manifest containing source.sha256")
    parser.add_argument("--index", action="append", required=True, help="registered cross-corpus index id; repeatable")
    parser.add_argument("--protected", type=Path, required=True, help="protected question-hash/ID map JSON")
    parser.add_argument("--out", type=Path, required=True, help="fresh, empty output directory")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
