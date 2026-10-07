"""Small deterministic helpers for authored semantic source-world fixtures.

The builder intentionally emits source references and proof receipts only. It
does not create teacher trajectories, admitted rows, or training targets.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


def canonical_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_world_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Check source identity, split isolation, schema links, and proof shape."""
    rows = list(rows)
    ids = [row.get("id") for row in rows]
    groups = [group for row in rows for group in row.get("source_groups", [])]
    assert len(rows) >= 4, "at least four independent worlds are required"
    assert len(ids) == len(set(ids)) and None not in ids
    assert len(groups) == len(set(groups)), "each world must have a distinct factual group"
    assert {row.get("split") for row in rows} == {"train", "test"}
    assert all(row.get("kind") == "lambda_source" for row in rows)
    assert all(row.get("source_ids") == row.get("source_groups") for row in rows)
    for row in rows:
        semantics = row["semantics"]
        curriculum = row["curriculum"]
        assert semantics["root"] in semantics["files"]
        assert curriculum["reference"]["root"]
        assert curriculum["split_group"] == row["source_groups"][0]
        assert curriculum["reference"].get("children")
        assert set(semantics["expected_files"]) == set(semantics["folder_files"])
    return {
        "world_count": len(rows),
        "train_count": sum(row["split"] == "train" for row in rows),
        "test_count": sum(row["split"] == "test" for row in rows),
        "unique_groups": len(set(groups)),
    }


def build_proof(rows: list[dict[str, Any]], source_bytes: bytes) -> dict[str, Any]:
    """Emit a source-consistency receipt; it makes no runtime/teacher claim."""
    counts = validate_world_rows(rows)
    return {
        "schema": "natlang.neuralese-authored-world-source-proof/1",
        "source_cases_sha256": canonical_sha256(source_bytes),
        **counts,
        "model_calls": 0,
        "provider_calls": 0,
        "teacher_trajectories": 0,
        "admission_granted": False,
        "proof_scope": "source facts, expected values, visible instructions, and scripted reference traces only",
    }
