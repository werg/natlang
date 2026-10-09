#!/usr/bin/env python3
"""Validate that a reviewed Luna plan's prior-attempt artifacts describe one run."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from luna_source_inventory import verify as verify_luna_source_inventory


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(path: str, plan_dir: Path) -> Path:
    item = Path(path)
    return (item if item.is_absolute() else plan_dir / item).resolve()


def require_file(path: Path, expected: str | None, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    if expected and sha(path) != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    schema = plan.get("schema")
    if schema not in {"natlang.reviewed_luna_dispatch_plan/1", "natlang.reviewed_luna_dispatch_plan/2"}:
        raise ValueError("unsupported dispatch plan schema")
    for path_text, expected in (plan.get("artifact_hashes") or {}).items():
        require_file(Path(path_text), expected, "artifact pin")

    prior = plan.get("prior_healthy_attempt")
    if not isinstance(prior, dict):
        raise ValueError("plan lacks prior_healthy_attempt")
    root = next((parent for parent in plan_path.parents
                 if (parent / "training/neuralese_corpora.json").is_file()), None)
    if root is None:
        raise ValueError("cannot locate repository root from the dispatch plan")
    if schema == "natlang.reviewed_luna_dispatch_plan/2":
        source_inventory = plan.get("source_inventory")
        if not isinstance(source_inventory, dict):
            raise ValueError("plan v2 lacks a source inventory binding")
        inventory_path = resolve(source_inventory.get("path", ""), plan_path.parent)
        expected = source_inventory.get("sha256")
        require_file(inventory_path, expected, "source inventory")
        source_path = resolve(plan.get("source", ""), plan_path.parent)
        verify_luna_source_inventory(inventory_path, source_path, plan.get("cases") or [])
    def linked_path(name: str, hash_name: str | None = None) -> Path:
        value = prior.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"prior attempt lacks {name}")
        path = resolve(value, plan_path.parent)
        require_file(path, prior.get(hash_name or f"{name}_sha256"), name)
        return path

    result = linked_path("result")
    trace = linked_path("trace")
    semantic_review_path = linked_path("semantic_review")
    semantic_review = json.loads(semantic_review_path.read_text())
    expected_result_rel = result.relative_to(root).as_posix()
    if (semantic_review.get("raw_result_path") != expected_result_rel
            or semantic_review.get("raw_result_sha256") != sha(result)):
        raise ValueError("semantic review does not describe the exact pinned prior result")
    trace_digest = semantic_review.get("raw_trace_canonical_sha256")
    if not isinstance(trace_digest, str) or len(trace_digest) != 64:
        raise ValueError("semantic review lacks its canonical trace digest")
    if trace.parent != result.parent or trace.name != result.name.replace(".result.json", ".trace.jsonl"):
        raise ValueError("prior trace and result paths are not the paired job outputs")

    normalization_path = prior.get("raw_native_normalization_review")
    if normalization_path:
        normalization = json.loads(linked_path("raw_native_normalization_review").read_text())
        if (normalization.get("raw_result_path") != expected_result_rel
                or normalization.get("raw_result_sha256") != sha(result)
                or normalization.get("raw_trace_canonical_sha256") != trace_digest):
            raise ValueError("normalization review does not join the exact result and semantic-review trace")
        if prior.get("materializer_manifest"):
            materializer = linked_path("materializer_manifest")
            materializer_rel = materializer.relative_to(root).as_posix()
            if (normalization.get("frozen_runtime_manifest_path") != materializer_rel
                    or normalization.get("frozen_runtime_manifest_sha256") != sha(materializer)):
                raise ValueError("normalization review does not join the pinned prior materializer manifest")

    manifest_name = "case_results_manifest" if prior.get("case_results_manifest") else "result_manifest"
    manifest_path = linked_path(manifest_name)
    if prior.get("result_manifest") and prior.get("case_results_manifest"):
        second = linked_path("result_manifest")
        if second != manifest_path or sha(second) != sha(manifest_path):
            raise ValueError("duplicate result-manifest references disagree")
    manifest = json.loads(manifest_path.read_text())
    output = Path(str(manifest_path)[:-len(".manifest.json")])
    require_file(output, manifest.get("output_sha256"), "manifest output")
    if sha(output) != sha(result):
        raise ValueError("prior case-results manifest output differs from the single pinned raw result")

    print(json.dumps({
        "schema": "natlang.luna-prior-attempt-binding-verification/1",
        "plan": str(plan_path), "plan_sha256": sha(plan_path),
        "artifact_pins_verified": len(plan.get("artifact_hashes") or {}),
        "campaign_id": prior.get("campaign_id"),
        "result_sha256": sha(result), "trace_file_sha256": sha(trace),
        "semantic_review_sha256": sha(semantic_review_path),
        "canonical_trace_sha256": trace_digest,
        "result_manifest_sha256": sha(manifest_path), "manifest_output_sha256": sha(output),
        "status": "all prior-attempt references join one exact invocation"
    }, indent=2))


if __name__ == "__main__":
    main()
