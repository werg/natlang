import importlib.util
from pathlib import Path


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
