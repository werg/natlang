import hashlib
import json

from scripts.project_reviewed_leaf_pass import project


def test_reviewed_projection_keeps_original_judge_decision(tmp_path):
    audit = tmp_path / "audit.jsonl"
    audit.write_text(json.dumps({"key": "say:key", "status": "done", "accepted": False}) + "\n")
    trajectories = tmp_path / "trajectory.jsonl"
    trajectories.write_text(json.dumps({
        "task": {"reference_key": "say:key"},
        "outcome": {"status": "done", "accepted": False, "admitted": False},
    }) + "\n")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({
        "source_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
        "approved": {"counter_offer": [0]},
        "notes": {"0": "Manually checked counter-offer."},
    }))
    destination = tmp_path / "reviewed.jsonl"

    assert project(audit, trajectories, review, destination) == 1
    row = json.loads(destination.read_text())
    assert row["outcome"]["accepted"] is True
    assert row["outcome"]["admitted"] is True
    assert row["training_admission"]["original_judge_accepted"] is False
    assert row["training_admission"]["audit_index"] == 0
