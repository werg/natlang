"""Verify root-approved integration adoption receipts and their exact artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARTIFACTS = {
    "native": "assembled_native_records",
    "recurrence": "assembled_recurrence_records",
    "native_pieces": "assembled_native_pieces",
    "recurrence_pieces": "assembled_recurrence_pieces",
    "text": "cumulative_text",
    "provenance": "cumulative_text_provenance",
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(root: Path, rel: str, label: str) -> Path:
    path = Path(rel)
    if path.is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the repository root")
    return resolved


def root_integration_adoption_bindings(receipt: dict, *, root: Path | None = None) -> dict | None:
    """Return verified prefix artifact pins, or None for a different receipt schema.

    The root receipt approves scope; its pinned integration review supplies the
    cohort-specific artifact and count facts. Those facts are validated for
    internal consistency instead of assuming one campaign's IDs or totals.
    """
    if receipt.get("schema") != "natlang.root-integration-adoption/1":
        return None
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    scope = receipt.get("scope", {})
    if (receipt.get("decision") != "approve-exact-native-and-text-assembly-for-next-reviewed-run"
            or scope.get("native_action_SFT") is not True
            or scope.get("ordinary_text_derivation") is not True
            or scope.get("active_GPU_inputs_changed") is not False
            or any(scope.get(flag) is not False for flag in (
                "new_recurrence_admission", "learned_writer_qualification",
                "whole_trajectory_admission", "runtime_qualification", "DPO_admission"))):
        raise ValueError("root integration adoption does not approve the exact native/text-only scope")
    rel = receipt.get("integration_receipt")
    expected = receipt.get("integration_receipt_sha256")
    if not isinstance(rel, str) or not isinstance(expected, str):
        raise ValueError("root integration adoption lacks its integration receipt pin")
    integration_path = _repo_path(root, rel, "integration receipt")
    if not integration_path.is_file() or _sha_file(integration_path) != expected:
        raise ValueError("root integration receipt is missing or has a hash mismatch")
    integration = json.loads(integration_path.read_text())
    if (not isinstance(integration.get("schema"), str)
            or not isinstance(integration.get("status"), str)
            or "verified" not in integration["status"].lower()):
        raise ValueError("pinned integration receipt is not a verified review")
    counts = integration.get("counts", {})
    native, recurrence, text = (counts.get(k, {}) for k in ("native", "recurrence", "text"))
    selected = native.get("selected_delta")
    if (not isinstance(selected, int) or selected <= 0
            or not all(isinstance(native.get(k), int) for k in ("total", "train", "test"))
            or native["total"] != native["train"] + native["test"]
            or recurrence.get("added_rows") != 0
            or not isinstance(recurrence.get("total"), int)
            or text.get("selected_delta_actions_covered") != selected
            or text.get("omissions_for_delta") != 0
            or not all(isinstance(text.get(k), int) for k in ("total", "train", "test"))
            or text["total"] != text["train"] + text["test"]):
        raise ValueError("pinned integration receipt has inconsistent action, split, recurrence, or text counts")
    limits = integration.get("limits", {})
    admission = integration.get("admission", {})
    if (limits.get("no_child_or_parent_trajectory_admission") is not True
            or limits.get("no_qualification_claim") is not True
            or limits.get("no_gpu_launch_or_input_change") is not True
            or admission.get("active_training_inputs_changed") is not False
            or admission.get("learned_writer_qualification") is not False
            or admission.get("recurrence_admission") is not False
            or admission.get("whole_trajectories") != 0):
        raise ValueError("pinned integration receipt exceeds the approved native/text scope")
    checks = receipt.get("independent_checks", {})
    if (checks.get("artifact_pins_verified", 0) < len(ARTIFACTS)
            or checks.get("native_base_prefix_byte_identical") is not True
            or checks.get("text_base_prefix_byte_identical") is not True
            or checks.get("text_provenance_base_prefix_byte_identical") is not True
            or checks.get("delta_omissions") != 0
            or checks.get("suffix_decode_mismatches") != 0
            or checks.get("exact_admitted_target_message_split_group_matches") != selected
            or checks.get("native_rows") != native.get("total")
            or checks.get("native_train") != native.get("train")
            or checks.get("native_test") != native.get("test")
            or checks.get("text_rows") != text.get("total")
            or checks.get("text_train") != text.get("train")
            or checks.get("text_test") != text.get("test")
            or checks.get("delta_omissions") != text.get("omissions_for_delta")):
        raise ValueError("root adoption review did not verify exact prefixes and complete text coverage")
    artifact_rows = integration.get("artifacts", {})
    normalized = {}
    for name, key in ARTIFACTS.items():
        artifact = artifact_rows.get(key, {})
        rel_path, digest = artifact.get("path"), artifact.get("sha256")
        if not isinstance(rel_path, str) or not isinstance(digest, str):
            raise ValueError(f"root integration receipt lacks {key} pin")
        path = _repo_path(root, rel_path, key)
        if not path.is_file() or _sha_file(path) != digest:
            raise ValueError(f"root integration artifact missing or hash mismatch: {key}")
        normalized[name] = {"path": path, "sha256": digest, "bytes": path.stat().st_size}
    return {"schema": receipt["schema"], "integration_receipt": integration_path,
            "integration_receipt_sha256": expected, "counts": counts,
            "artifacts": normalized}
