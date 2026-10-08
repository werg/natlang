import importlib.util
from pathlib import Path
import hashlib
import json


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_neuralese_gold_text_delta.py"
SPEC = importlib.util.spec_from_file_location("gold_text_delta", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_anchor_omission_is_not_selected_delta_coverage():
    anchor = {"id": "held-anchor"}
    omissions = [{"id": "held-anchor", "reason": "unresolved_reference"},
                 {"id": "delta-row", "reason": "separate_delta_omission"}]
    receipt = {"excluded_train_exact_held_complete_documents": 0}
    result = MODULE.anchor_qualification(anchor, [], omissions, receipt)
    assert result["qualified"] is False
    assert result["reason"] == "anchor_omitted_by_shared_text_renderer"
    assert result["omission"]["id"] == "held-anchor"


def test_anchor_must_be_present_in_rendered_rows():
    anchor = {"id": "held-anchor"}
    result = MODULE.anchor_qualification(anchor, [{"id": "delta-row", "source_record_ids": ["delta-row"]}],
                                         [], {"excluded_train_exact_held_complete_documents": 0})
    assert result == {"qualified": False, "reason": "anchor_not_present_in_rendered_rows"}


def test_base_anchor_omission_is_separate_from_delta_omissions():
    omissions = [{"id": "held-anchor", "reason": "anchor_only"},
                 {"id": "delta-row", "reason": "selected_delta"}]
    assert MODULE.selected_delta_omissions(omissions, {"delta-row"}) == [
        {"id": "delta-row", "reason": "selected_delta"}]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _integration_adoption_fixture(tmp_path):
    artifacts = {}
    for key, name in {
        "assembled_native_records": "native.jsonl",
        "assembled_recurrence_records": "recurrence.jsonl",
        "assembled_native_pieces": "native-pieces.jsonl",
        "assembled_recurrence_pieces": "recurrence-pieces.jsonl",
        "cumulative_text": "text.jsonl",
        "cumulative_text_provenance": "provenance.jsonl",
    }.items():
        path = tmp_path / name
        path.write_text(name + "\n")
        artifacts[key] = {"path": str(path.relative_to(tmp_path)), "sha256": _sha(path)}
    integration = tmp_path / "integration-review.json"
    integration.write_text(json.dumps({
        "schema": "natlang.v18-action45-admitted84-native-text-integration-review/1",
        "status": "verified exact admitted native/text delta; ready for registry publication and DGX sync",
        "scope": "Exact84 root-admitted native SFT actions plus faithful shared-renderer ordinary text only; no new recurrence rows and no active GPU input selection.",
        "counts": {
            "native": {"selected_delta": 7, "total": 5374, "train": 3339, "test": 2035},
            "recurrence": {"added_rows": 0, "total": 4961},
            "text": {"selected_delta_actions_covered": 7, "omissions_for_delta": 0,
                     "total": 4612, "train": 2908, "test": 1704},
        },
        "limits": {"no_child_or_parent_trajectory_admission": True,
                   "no_qualification_claim": True, "no_gpu_launch_or_input_change": True},
        "admission": {"active_training_inputs_changed": False,
                      "learned_writer_qualification": False, "recurrence_admission": False,
                      "whole_trajectories": 0},
        "artifacts": artifacts,
    }))
    adoption = {
        "schema": "natlang.root-integration-adoption/1",
        "decision": "approve-exact-native-and-text-assembly-for-next-reviewed-run",
        "scope": {"native_action_SFT": True, "ordinary_text_derivation": True,
                  "active_GPU_inputs_changed": False, "new_recurrence_admission": False,
                  "learned_writer_qualification": False, "whole_trajectory_admission": False,
                  "runtime_qualification": False, "DPO_admission": False},
        "integration_receipt": integration.name,
        "integration_receipt_sha256": _sha(integration),
        "independent_checks": {"artifact_pins_verified": 6,
                               "native_base_prefix_byte_identical": True,
                               "text_base_prefix_byte_identical": True,
                               "text_provenance_base_prefix_byte_identical": True,
                               "delta_omissions": 0,
                               "suffix_decode_mismatches": 0,
                               "exact_admitted_target_message_split_group_matches": 7,
                               "native_rows": 5374, "native_train": 3339, "native_test": 2035,
                               "text_rows": 4612, "text_train": 2908, "text_test": 1704},
    }
    return adoption, artifacts


def test_root_integration_adoption_normalizes_exact_prefix_pins(tmp_path):
    adoption, artifacts = _integration_adoption_fixture(tmp_path)
    normalized = MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    assert normalized["schema"] == "natlang.root-integration-adoption/1"
    assert normalized["artifacts"]["text"]["sha256"] == artifacts["cumulative_text"]["sha256"]
    assert normalized["artifacts"]["provenance"]["sha256"] == artifacts["cumulative_text_provenance"]["sha256"]
    assert normalized["artifacts"]["recurrence"]["sha256"] == artifacts["assembled_recurrence_records"]["sha256"]


def test_root_integration_adoption_rejects_mismatched_artifact(tmp_path):
    adoption, artifacts = _integration_adoption_fixture(tmp_path)
    artifacts["cumulative_text"]["sha256"] = "0" * 64
    integration = tmp_path / adoption["integration_receipt"]
    review = json.loads(integration.read_text())
    review["artifacts"] = artifacts
    integration.write_text(json.dumps(review))
    adoption["integration_receipt_sha256"] = _sha(integration)
    try:
        MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    except ValueError as exc:
        assert "cumulative_text" in str(exc)
    else:
        raise AssertionError("mismatched cumulative text must be rejected")


def test_root_integration_adoption_rejects_unverified_review_counts(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    integration = tmp_path / adoption["integration_receipt"]
    review = json.loads(integration.read_text())
    review["counts"]["recurrence"]["added_rows"] = 1
    integration.write_text(json.dumps(review))
    adoption["integration_receipt_sha256"] = _sha(integration)
    try:
        MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    except ValueError as exc:
        assert "inconsistent action, split, recurrence, or text counts" in str(exc)
    else:
        raise AssertionError("an integration review adding recurrence rows must be rejected")


def test_root_integration_adoption_rejects_receipt_path_escape(tmp_path):
    adoption, _ = _integration_adoption_fixture(tmp_path)
    adoption["integration_receipt"] = "../outside-review.json"
    try:
        MODULE.root_integration_adoption_bindings(adoption, root=tmp_path)
    except ValueError as exc:
        assert "escapes the repository root" in str(exc)
    else:
        raise AssertionError("integration receipts must stay inside the canonical repository")
