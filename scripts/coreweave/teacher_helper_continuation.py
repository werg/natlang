"""Exact historical/successor custody for teacher admission helper execution.

Callers must pin ``transition_binding`` and ``successor_helper_binding`` in the
immutable operator closure. This module deliberately does not interpret either
as a policy decision or an admission result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from natlang_neuralese.common import paths as machine_paths
from typing import Any


LEGACY_OBSERVATION = {
    "path": str(machine_paths.resolve('data_hdd', 'natlang-development-data/teacher198-current-admission-20261019/teacher-runtime-alias-legacy-default-source-closure-compatibility-0aad212b5-v1.json')),
    "sha256": "7a3fa38e0c0251469caa5b9322ee4f8d09b12a0301188982b815023ee4c14cbe",
    "bytes": 1437,
}
LEGACY_AUTHORITY = {
    "path": str(machine_paths.resolve('data_hdd', 'natlang-development-data/teacher198-current-admission-20261019/teacher-runtime-alias-legacy-0aad212b5-immutable-helper-authority/authority.json')),
    "sha256": "fbf5ba315825fd333e5ee27b6051f94eef20c16aab70b51a8179ba177cd1660d",
    "bytes": 2568,
}
LEGACY_API_RELATIVE = "scripts/coreweave/teacher_tranche_admission.py"
SUCCESSOR_COMMIT = "8c3358fac0355d462652f1b3352970ff2f5ca0f9"
SOURCE_CLOSURE_RECEIPT = {
    "path": str(machine_paths.resolve('data_nvme', 'dgxops/acquired-connected-source-protection-20261017-d13-d14-d32-v3/closure-receipt-full-20261020-content-facade-helper-v1.json')),
    "sha256": "f1b9be730845c0c498d9084f051949e5523e6f2a38fcfaf0bc632a6aca3d45c9",
    "bytes": 14025086,
}
SOURCE_CLOSURE_TRANSITION = {
    "path": str(machine_paths.resolve('data_nvme', 'dgxops/acquired-connected-source-protection-20261017-d13-d14-d32-v3/closure-code-transition-content-facade-helper-v1.json')),
    "sha256": "a597bdfe4b2eb717091cf9e8ca7bb35b754d5dde76b3856015669a5cb28c2b57",
    "bytes": 9071,
}


def binding(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"helper continuation artifact is not a regular file: {target}")
    raw = target.read_bytes()
    return {"path": str(target.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_bound(reference: dict[str, Any], *, expected: dict[str, Any] | None = None) -> tuple[dict[str, Any], bytes]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}:
        raise ValueError("helper continuation requires a complete path/SHA/byte binding")
    if expected is not None and any(reference.get(key) != expected.get(key) for key in ("path", "sha256", "bytes")):
        raise ValueError("helper continuation authority differs from its pinned binding")
    actual = binding(reference["path"])
    if actual != reference:
        raise ValueError("helper continuation artifact changed after it was bound")
    return actual, Path(reference["path"]).read_bytes()


def verify_helper_continuation(
    execution: dict[str, Any],
    *,
    transition_binding: dict[str, Any],
    successor_helper_binding: dict[str, Any],
    successor_verifier_binding: dict[str, Any],
    loaded_helper_path: str | Path,
    loaded_verifier_path: str | Path,
    repository_root: str | Path,
    expected_executor: dict[str, Any] | None = None,
) -> str:
    """Verify one exact historical or successor helper receipt.

    ``transition_binding`` and ``successor_helper_binding`` must be constants
    from the operator source closure, whose own digest is carried by the
    operator declaration. A proof cannot authorize a different helper simply
    by setting ``passed``.
    """
    if not isinstance(execution, dict):
        raise ValueError("helper continuation execution receipt is missing")
    actual_receipt = execution.get("helper_continuation")
    old_observation, old_raw = _read_bound(LEGACY_OBSERVATION, expected=LEGACY_OBSERVATION)
    old = json.loads(old_raw)
    if not (old.get("passed") is True and old.get("source_closure_implementation_unchanged") is True and
            old.get("normal_full_artifact_verifier_passed") is True and old.get("source_rows_parsed") == 0 and
            old.get("training_admission") is False):
        raise ValueError("preserved historical helper observation is not an exact non-admission proof")
    authority_binding, authority_raw = _read_bound(LEGACY_AUTHORITY, expected=LEGACY_AUTHORITY)
    authority = json.loads(authority_raw)
    if (authority.get("schema") != "natlang.immutable-source-helper-authority/1" or
            authority.get("commit") != "0aad212b5" or authority.get("training_admission") is not False):
        raise ValueError("preserved helper authority has an unexpected identity")
    old_api = authority.get("files", {}).get(LEGACY_API_RELATIVE)
    observed_old_api = old.get("reviewed_external_helpers", {}).get("files", {}).get(LEGACY_API_RELATIVE)
    if (not isinstance(old_api, dict) or not isinstance(observed_old_api, dict) or
            any(old_api.get(key) != observed_old_api.get(key) for key in ("sha256", "bytes"))):
        raise ValueError("historical helper observation and immutable authority disagree")
    for relative, old_ref in authority["files"].items():
        observed = old.get("reviewed_external_helpers", {}).get("files", {}).get(relative)
        actual_old = binding(old_ref["path"])
        if (not isinstance(observed, dict) or any(old_ref.get(k) != observed.get(k) for k in ("sha256", "bytes")) or
                actual_old.get("sha256") != old_ref["sha256"] or actual_old.get("bytes") != old_ref["bytes"]):
            raise ValueError(f"historical helper authority changed: {relative}")

    builder_relative = "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    builder_ref = old.get("reviewed_external_helpers", {}).get("files", {}).get(builder_relative)
    builder_path = Path(repository_root).resolve(strict=True) / builder_relative
    builder_actual = binding(builder_path)
    recorded_builder = execution.get("converter")
    if (not isinstance(builder_ref, dict) or builder_actual.get("sha256") != builder_ref.get("sha256") or
            builder_actual.get("bytes") != builder_ref.get("bytes") or not isinstance(recorded_builder, dict) or
            recorded_builder.get("path") != str(builder_path) or recorded_builder.get("sha256") != builder_ref.get("sha256")):
        raise ValueError("conversion execution does not bind the reviewed trajectory graph builder")

    if actual_receipt == old_observation:
        if expected_executor is not None and execution.get("converter") != expected_executor:
            raise ValueError("historical execution does not match the exact legacy converter")
        return "historical"

    transition_ref, transition_raw = _read_bound(transition_binding, expected=transition_binding)
    proof = json.loads(transition_raw)
    if (not isinstance(actual_receipt, dict) or actual_receipt != transition_ref or
            proof.get("schema") != "natlang.teacher-helper-continuation-transition/1"):
        raise ValueError("execution does not name the one pinned successor helper transition")
    if (proof.get("legacy_observation") != old_observation or proof.get("legacy_authority") != authority_binding or
            proof.get("training_admission") is not False or proof.get("exposure") != 0):
        raise ValueError("successor proof does not preserve the exact historical authority and non-admission scope")
    api_binding = binding(loaded_helper_path)
    if (api_binding != successor_helper_binding or proof.get("successor_helper") != api_binding):
        raise ValueError("loaded successor helper differs from the exact proof and operator pin")
    verifier_binding = binding(loaded_verifier_path)
    if (verifier_binding != successor_verifier_binding or
            proof.get("successor_verifier") != verifier_binding):
        raise ValueError("loaded successor verifier differs from the exact proof and operator pin")

    closure = proof.get("source_closure")
    if not isinstance(closure, dict):
        raise ValueError("successor proof omits unchanged source-closure evidence")
    for name, fixed in (("receipt", SOURCE_CLOSURE_RECEIPT), ("transition", SOURCE_CLOSURE_TRANSITION)):
        ref = closure.get(name)
        if not isinstance(ref, dict) or any(ref.get(key) != fixed[key] for key in ("path", "sha256")):
            raise ValueError(f"successor proof does not bind the exact historical source-closure {name}")
        _read_bound(ref)
    closure_receipt = json.loads(Path(closure["receipt"]["path"]).read_bytes())
    closure_transition = json.loads(Path(closure["transition"]["path"]).read_bytes())
    if (old.get("original_receipt_sha256") != closure["receipt"]["sha256"] or
            closure_receipt.get("schema") != "natlang.acquired-source-closure-receipt/1" or
            closure_receipt.get("code_transition") != {k: closure["transition"][k] for k in ("path", "sha256")} or
            closure_transition.get("schema") != "natlang.acquired-source-closure-code-transition/6" or
            closure_receipt.get("training_admission") is not False or
            closure_transition.get("training_admission") is not False or
            closure_transition.get("closure_verifier_sha256") != closure_receipt.get("verifier_sha256") or
            closure_transition.get("protection_sha256") != closure_receipt.get("protection_sha256") or
            closure_transition.get("accounting_sha256") != _canonical_sha256(closure_receipt.get("accounting")) or
            closure_transition.get("input_artifacts_sha256") != _canonical_sha256(closure_receipt.get("input_artifacts")) or
            closure.get("accounting") != closure_receipt.get("accounting")):
        raise ValueError("successor helper changed the independently recorded source-closure accounting")
    inventory = proof.get("implementation_files")
    old_inventory = old.get("reviewed_external_helpers", {}).get("files", {})
    if not isinstance(inventory, dict) or set(inventory) != set(old_inventory):
        raise ValueError("successor proof has an incomplete implementation inventory")
    root = Path(repository_root).resolve(strict=True)
    for relative, new_ref in inventory.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("successor implementation inventory escapes the repository")
        target = Path(loaded_helper_path).resolve(strict=True) if relative == LEGACY_API_RELATIVE else (root / relative).resolve(strict=True)
        if relative != LEGACY_API_RELATIVE and root not in target.parents and target != root:
            raise ValueError("successor implementation inventory escapes the repository")
        actual = binding(target)
        if (not isinstance(new_ref, dict) or actual.get("sha256") != new_ref.get("sha256") or
                actual.get("bytes") != new_ref.get("bytes")):
            raise ValueError(f"successor implementation file changed: {relative}")
        if relative != LEGACY_API_RELATIVE and any(
                new_ref.get(k) != old_inventory[relative].get(k) for k in ("sha256", "bytes")):
            raise ValueError(f"unreviewed helper changed outside the selected API: {relative}")
    review_ref = proof.get("review")
    if not isinstance(review_ref, dict):
        raise ValueError("successor proof omits independent source/test review")
    _, review_raw = _read_bound(review_ref)
    review = json.loads(review_raw)
    if (review.get("schema") != "natlang.teacher-helper-transition-review/1" or
            review.get("passed") is not True or review.get("full_check") is not True or
            review.get("source_commit") != SUCCESSOR_COMMIT or
            review.get("training_admission") is not False or review.get("exposure") != 0 or
            review.get("successor_helper_sha256") != api_binding["sha256"]):
        raise ValueError("successor review does not prove the exact helper and full checks")
    for field in ("focused_evidence", "full_check_evidence"):
        evidence_ref = review.get(field)
        _, evidence_raw = _read_bound(evidence_ref)
        evidence = json.loads(evidence_raw)
        if (evidence.get("schema") != "natlang.teacher-helper-transition-check-result/1" or
                evidence.get("status") != "passed" or evidence.get("exit_code") != 0 or
                evidence.get("source_commit") != SUCCESSOR_COMMIT or not evidence.get("unit")):
            raise ValueError(f"successor review has no terminal-green {field}")
    if expected_executor is not None and execution.get("converter") != expected_executor:
        raise ValueError("successor execution does not match the exact active converter")
    return "successor"


def verify_successor_bundle(
    *,
    transition_binding: dict[str, Any],
    successor_helper_binding: dict[str, Any],
    successor_verifier_binding: dict[str, Any],
    loaded_helper_path: str | Path,
    loaded_verifier_path: str | Path,
    repository_root: str | Path,
) -> None:
    """Check the successor closure before conversion or strict validation starts."""
    builder_path = Path(repository_root).resolve(strict=True) / "scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py"
    builder_ref = binding(builder_path)
    verify_helper_continuation(
        {"helper_continuation": transition_binding,
         "converter": {"path": builder_ref["path"], "sha256": builder_ref["sha256"]}},
        transition_binding=transition_binding,
        successor_helper_binding=successor_helper_binding,
        successor_verifier_binding=successor_verifier_binding,
        loaded_helper_path=loaded_helper_path,
        loaded_verifier_path=loaded_verifier_path,
        repository_root=repository_root,
    )
