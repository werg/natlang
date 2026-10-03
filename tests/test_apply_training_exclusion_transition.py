import hashlib
import json
from pathlib import Path

import pytest

from scripts.apply_training_exclusion_transition import (
    apply_transition,
    reject_overlapping_run_paths,
)


def test_target_must_not_overlap_source_run(tmp_path):
    source = tmp_path / "active"
    source.mkdir()
    child = source / "next"
    outside = tmp_path / "new-run"
    reject_overlapping_run_paths(outside, source)
    with pytest.raises(ValueError, match="disjoint"):
        reject_overlapping_run_paths(child, source)
    with pytest.raises(ValueError, match="disjoint"):
        reject_overlapping_run_paths(tmp_path, source)


def test_candidate_manifest_cannot_be_applied(tmp_path):
    manifest = tmp_path / "candidate.json"
    manifest.write_text(json.dumps({
        "schema": "natlang.training_source_exclusion/1",
        "status": "review_candidate_not_applied",
    }))
    actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
    target = tmp_path / "future-run"
    with pytest.raises(ValueError, match="not root-approved"):
        apply_transition(
            manifest_path=manifest.resolve(),
            manifest_sha256=actual,
            quiescence_path=tmp_path / "not-read.json",
            target_run=target,
        )
    assert not target.exists()
