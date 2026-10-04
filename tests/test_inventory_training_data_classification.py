import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.inventory_training_data import catalog


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _put_input_manifest(repo, rows):
    manifest = {"schema": "natlang.training_existing_data_input_manifest/1", "inputs": rows,
                "input_count": len(rows)}
    manifest["sha256"] = _sha(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode())
    path = repo / "runs/input-manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    return path


def _row(repo, relative, include, role, notes):
    path = repo / relative
    data = path.read_bytes()
    return {"path": relative, "bytes": len(data), "sha256": _sha(data),
            "include_for_training": include, "role": role, "notes": notes}


def _snapshot(repo, result_path):
    directory = repo / "data/teacher/generated-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "selected.manifest.json"
    path.write_text(json.dumps({
        "version": "natlang.generated_training_snapshot/2", "time": "2026-10-02T00:00:00Z",
        "results": {"path": str(result_path), "sha256": _sha(result_path.read_bytes())},
        "summary": {"selected_trajectories": 1, "unique_programs": 1, "completed_files": 1},
    }))
    return path


def _config(repo, manifest, training, *, evidence=(), provenance=()):
    return {
        "stages": [{"id": "consume", "inputs": [str(repo / x) for x in evidence], "outputs": []}],
        "data_inventory_training_inputs": [str(repo / x) for x in training],
        "data_inventory_positive_carriers": [],
        "data_inventory_review_inputs": [],
        "data_inventory_generated_snapshot_manifests": [],
        "data_inventory_provenance_manifest": {
            "path": str(manifest), "sha256": _sha(manifest.read_bytes())},
    }


def test_training_inventory_separates_positive_provenance_and_failure_lanes(tmp_path):
    repo = tmp_path / "repo"
    (repo / "training").mkdir(parents=True)
    (repo / "data/teacher/generated-snapshots").mkdir(parents=True)
    (repo / "runs/snapshot").mkdir(parents=True)
    (repo / "data/teacher/positive.jsonl").write_text('{"id":"positive"}\n')
    (repo / "data/teacher/held.jsonl").write_text('{"id":"held"}\n')
    positive_result = repo / "runs/snapshot/selected.results.jsonl.gz"
    with gzip.open(positive_result, "wb") as stream:
        stream.write(b'{"id":"selected"}\n')
    failure = repo / "runs/snapshot/failures.jsonl.gz"
    with gzip.open(failure, "wb") as stream:
        stream.write(b'{"id":"failure"}\n')

    entries = []
    for path, include, role, notes in [
        ("data/teacher/positive.jsonl", True, "positive_sft", "Current positive input."),
        ("data/teacher/held.jsonl", False, "source_hold", "Held evidence; never a positive training row."),
        ("runs/snapshot/selected.results.jsonl.gz", True, "current_selected_teacher_positive_snapshot", "Selected positive results."),
    ]:
        path = repo / path
        data = path.read_bytes()
        entries.append({"path": str(path.relative_to(repo)), "bytes": len(data), "sha256": _sha(data),
                        "include_for_training": include, "role": role, "notes": notes})
    manifest = {"schema": "natlang.training_existing_data_input_manifest/1", "inputs": entries,
                "input_count": len(entries)}
    manifest["sha256"] = _sha(json.dumps(entries, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode())
    manifest_path = repo / "runs/input-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest))

    success_manifest = {
        "version": "natlang.generated_training_snapshot/2",
        "time": "2026-10-02T00:00:00Z",
        "results": {"path": str(positive_result), "sha256": _sha(positive_result.read_bytes())},
        "summary": {"selected_trajectories": 1, "unique_programs": 1, "completed_files": 1},
    }
    success_path = repo / "data/teacher/generated-snapshots/selected.manifest.json"
    success_path.write_text(json.dumps(success_manifest))
    failure_manifest = {
        "version": "natlang.teacher_failure_inventory/1",
        "created_at": "2026-10-02T00:00:00Z",
        "artifact": {"path": str(failure), "sha256": _sha(failure.read_bytes())},
        "candidates": 1,
    }
    failure_manifest_path = repo / "runs/snapshot/failures.manifest.json"
    failure_manifest_path.write_text(json.dumps(failure_manifest))

    policy = {"required_default_inputs": ["data/teacher/positive.jsonl"], "decisions": [], "replacements": {}}
    (repo / "training/data_sources.json").write_text(json.dumps(policy))
    (repo / "data/teacher/data-inventory").mkdir(parents=True)
    # Simulate a prior catalog entry: the held row was once seen as a recipe input.
    (repo / "data/teacher/data-inventory/current.json").write_text(json.dumps({"artifacts": [
        {"path": "data/teacher/held.jsonl", "ever_recipe_input": True, "direct_recipe_input": True}
    ]}))

    config = {
        "stages": [{"id": "consume", "inputs": [str(repo / "data/teacher/positive.jsonl"),
                                                       str(positive_result), str(repo / "data/teacher/held.jsonl")],
                   "outputs": []}],
        "data_inventory_training_inputs": [str(repo / "data/teacher/positive.jsonl"), str(positive_result)],
        "data_inventory_review_inputs": [str(failure), str(failure_manifest_path)],
        "data_inventory_generated_snapshot_manifests": [str(success_path), str(failure_manifest_path)],
        "data_inventory_provenance_manifest": {"path": str(manifest_path), "sha256": _sha(manifest_path.read_bytes())},
    }
    _, report = catalog(repo, config)
    assert len(report["generated_snapshots"]) == 1
    assert report["generated_snapshots"][0]["summary"]["selected_trajectories"] == 1
    assert len(report["generated_failure_inventories"]) == 1
    assert report["generated_failure_inventories"][0]["candidates"] == 1
    held = next(item for item in report["not_carried_forward"] if item["path"] == "data/teacher/held.jsonl")
    assert held["resolution"] == "explicit_provenance_only"
    assert report["missing_required_default_inputs"] == []


def test_hashed_stage_metadata_cannot_satisfy_required_training_input(tmp_path):
    repo = tmp_path / "repo"
    (repo / "training").mkdir(parents=True)
    (repo / "data/teacher").mkdir(parents=True)
    (repo / "runs/snapshot").mkdir(parents=True)
    (repo / "data/teacher/held.jsonl").write_text('{"id":"held"}\n')
    result = repo / "runs/snapshot/selected.results.jsonl"
    result.write_text('{"id":"selected"}\n')
    manifest = _put_input_manifest(repo, [
        _row(repo, "data/teacher/held.jsonl", False, "held_required_input", "Held; not a positive training input."),
        _row(repo, "runs/snapshot/selected.results.jsonl", True, "selected_snapshot", "Positive selected result."),
    ])
    snapshot = _snapshot(repo, result)
    policy = {"required_default_inputs": ["data/teacher/held.jsonl"], "decisions": [], "replacements": {}}
    (repo / "training/data_sources.json").write_text(json.dumps(policy))
    config = _config(repo, manifest, ["runs/snapshot/selected.results.jsonl"],
                     evidence=["data/teacher/held.jsonl"], provenance=["data/teacher/held.jsonl"])
    config["data_inventory_generated_snapshot_manifests"] = [str(snapshot)]
    try:
        catalog(repo, config)
    except ValueError as error:
        assert "data/teacher/held.jsonl" in str(error)
    else:
        raise AssertionError("A hashed stage input incorrectly satisfied the required positive lane")


def test_held_required_input_resolves_only_through_approved_positive_replacement(tmp_path):
    repo = tmp_path / "repo"
    (repo / "training").mkdir(parents=True)
    (repo / "data/teacher").mkdir(parents=True)
    (repo / "runs/snapshot").mkdir(parents=True)
    original_rel = "data/teacher/original.jsonl"
    replacement_rel = "data/teacher/replacement.jsonl"
    row = {"id": "row-1", "teacher_trajectory_digest": "trace-1"}
    row_bytes = (json.dumps(row, sort_keys=True) + "\n").encode()
    (repo / original_rel).write_bytes(row_bytes)
    (repo / replacement_rel).write_bytes(row_bytes)
    result = repo / "runs/snapshot/selected.results.jsonl"
    result.write_text('{"id":"selected"}\n')
    snapshot = _snapshot(repo, result)
    review = {
        "status": "source_reviewed_replacement_approved", "original_artifact": original_rel,
        "original_sha256": _sha(row_bytes), "replacement_artifact": replacement_rel,
        "replacement_sha256": _sha(row_bytes), "original_turn_count": 1, "replacement_turn_count": 1,
        "retained_turns": [{"original_line": 1, "original_line_sha256": _sha(row_bytes.rstrip(b"\n")),
                             "id": "row-1", "trajectory_digest": "trace-1"}], "excluded_turns": [],
    }
    review_path = repo / "data/teacher/review.json"
    review_path.write_text(json.dumps(review, sort_keys=True))
    review_sha = _sha(review_path.read_bytes())
    manifest = _put_input_manifest(repo, [
        _row(repo, original_rel, False, "held_original", "Held original, retained as lineage only."),
        _row(repo, replacement_rel, True, "approved_replacement", "Approved reviewed positive replacement."),
        _row(repo, "runs/snapshot/selected.results.jsonl", True, "selected_snapshot", "Positive selected result."),
    ])
    policy = {
        "required_default_inputs": [original_rel],
            "decisions": [{"glob": original_rel, "replacement": replacement_rel, "status": "source_review_approved",
                       "review_manifest": "data/teacher/review.json", "review_manifest_sha256": review_sha}],
        "replacements": {original_rel: replacement_rel},
    }
    (repo / "training/data_sources.json").write_text(json.dumps(policy))
    config = _config(repo, manifest, [replacement_rel, "runs/snapshot/selected.results.jsonl"],
                     evidence=[original_rel])
    config["data_inventory_generated_snapshot_manifests"] = [str(snapshot)]
    _, report = catalog(repo, config)
    resolution = next(x for x in report["required_input_resolutions"] if x["required"] == original_rel)
    assert resolution["resolved"] == replacement_rel
    assert resolution["mode"] == "catalog_pinned_approved_replacement"
    assert report["missing_required_default_inputs"] == []


def test_inventory_can_pin_completed_parent_policy_for_a_new_phase(tmp_path):
    import pytest
    (tmp_path / 'training').mkdir()
    current = tmp_path / 'training/data_sources.json'
    current.write_text(json.dumps({'required_default_inputs': [], 'decisions': [], 'replacements': {}}))
    old = tmp_path / 'parent-policy.json'
    old.write_bytes(current.read_bytes())
    config = {'stages': [], 'data_inventory_explicit_input_override': True,
              'data_inventory_policy_manifest': {'path': str(old), 'sha256': _sha(old.read_bytes()),
                  'reason': 'Preserve completed parent curriculum policy during post-training.'}}
    _, report = catalog(tmp_path, config)
    assert report['policy'] == str(old)
    assert report['pinned_policy_reason']
    old.write_text('{}')
    with pytest.raises(ValueError, match='identity changed'):
        catalog(tmp_path, config)
