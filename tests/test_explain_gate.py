import importlib.util
import json
from pathlib import Path

import pytest


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load("explain_gate")
audit = load("audit_student_projection_failures")

THRESHOLDS = {"min_argmax_agreement": .99, "max_kl_nats": .02, "max_quality_ce_gap_nats": .05}


def stratum(agreement=1.0, kl=0.0, gap=0.0):
    return {"argmax_agreement": agreement, "kl_plain_to_projected_nats": kl, "quality_ce_gap": gap}


def self_feedback(strata, passed, windows=()):
    return {"thresholds": THRESHOLDS, "strata": strata, "windows": list(windows), "proposed_gate_passed": passed}


def test_self_feedback_facts_follow_the_gates_arithmetic():
    report = {"nested": self_feedback({"first": stratum(), "last": stratum(), "length_band:long": stratum(.97, .05, .0)}, False, [
        {"document_sha256": "ab" * 32, "offset": 3, "proposed_gate_passed": False, "kl_plain_to_projected_nats": .09},
        {"document_sha256": "cd" * 32, "offset": 0, "proposed_gate_passed": True, "kl_plain_to_projected_nats": .01}])}
    facts = gate.gate_facts(report, run="r")
    assert facts["passed"] is False
    assert [(e["stratum"], e["metric"], e["direction"]) for e in facts["failed"]] == [
        ("length_band:long", "argmax_agreement", "min"), ("length_band:long", "kl_plain_to_projected_nats", "max")]
    assert facts["failed"][0]["margin"] == pytest.approx(.02)
    assert facts["failed"][1]["margin"] == pytest.approx(.03)
    assert facts["passed_strata"] == ["first", "last"]
    assert [w["id"] for w in facts["worst_windows"]] == ["abababababab@3"]


def test_a_recomputed_decision_that_differs_from_the_report_is_refused():
    report = self_feedback({"first": stratum(.5, .5, .5)}, True)
    with pytest.raises(ValueError, match="differs from the report"):
        gate.gate_facts(report)
    with pytest.raises(ValueError, match="not a known gate report"):
        gate.gate_facts({"something": 1})


def test_identity_control_requires_exact_zeros():
    rows = [{"source_sha256": "a" * 64, "raw_transport_max_abs": 0.0, "identity_readback_logits_max_abs": 0.0, "full_output_reference_max_abs": 0.0},
            {"source_sha256": "b" * 64, "raw_transport_max_abs": 0.0, "identity_readback_logits_max_abs": 1e-7, "full_output_reference_max_abs": 3e-6}]
    facts = gate.gate_facts({"token_aligned_reference_passed": False, "rows": rows})
    assert [(e["stratum"], e["metric"]) for e in facts["failed"]] == [
        ("row:1:bbbbbbbbbbbb", "identity_readback_logits_max_abs"), ("row:1:bbbbbbbbbbbb", "full_output_reference_max_abs")]
    assert facts["passed_strata"] == ["row:0:aaaaaaaaaaaa"]
    assert facts["worst_windows"][0]["observed"] == 3e-6


def test_causal_bootstrap_uses_the_gate_flags_defaults_and_overrides():
    report = {"step": 5, "feedback_gate_passed": False, "strata": {"a": {"agreement": .95, "kl": .4}, "b": {"agreement": .95, "kl": .1}}}
    facts = gate.gate_facts(report)
    assert [(e["stratum"], e["metric"]) for e in facts["failed"]] == [("a", "kl")]
    assert facts["context"]["step"] == 5
    with pytest.raises(ValueError):
        gate.gate_facts(report, kl_gate=.5)  # then it would pass: the report says it failed


def test_cli_writes_facts_once_and_skips_passed_gates(tmp_path, capsys):
    failed = tmp_path / "report.json"
    failed.write_text(json.dumps(self_feedback({"first": stratum(.9, 0, 0)}, False)))
    assert gate.main([str(failed)]) == 0
    facts = json.loads((tmp_path / "report.json.facts.json").read_text())
    assert facts["schema"] == "natlang.gate_facts/1" and facts["facts"]["failed"][0]["metric"] == "argmax_agreement"
    with pytest.raises(FileExistsError):
        gate.main([str(failed)])
    passed = tmp_path / "ok.json"
    passed.write_text(json.dumps(self_feedback({"first": stratum()}, True)))
    assert gate.main([str(passed)]) == 0
    assert not (tmp_path / "ok.json.facts.json").exists()
    # The report itself is untouched.
    assert json.loads(failed.read_text())["proposed_gate_passed"] is False


def receipt(error="", tool="", checks=None, calls=3):
    turns = [{"response": {"calls": [["eval", {"code": f"step {i}"}]]},
              "request": {"messages": [{"role": "tool", "content": tool + str(i)}]}} for i in range(calls)]
    return {"error": {"message": error}, "row": {"outcome": {"checks": checks or {}, "detail": "d"}}, "turns": turns}


def test_failure_card_is_bounded_and_audit_counts_are_unchanged(tmp_path):
    unclassified = receipt(tool="a surprising reply ")
    classified = receipt(tool="file not found: x ")
    assert audit.classify(unclassified) == ["unclassified_needs_review"]
    assert audit.classify(classified) == ["state_recovery"]
    card = audit.failure_card(receipt(calls=20, tool="x" * 5000), "fam", "p1")
    assert len(card["actions"]) == 6 and len(card["feedback"]) == 6
    assert all(len(text) < audit.CARD_LIMIT + 40 for text in card["feedback"])
    assert card["turn_count"] == 20 and card["family"] == "fam"
    case = tmp_path / "case"
    case.mkdir()
    (case / "initial.json").write_text(json.dumps({"row": {"task": {"program_ir": {"family": "fam", "id": "p1"}}}}))
    for name, candidate in (("search-1.json", unclassified), ("search-2.json", classified)):
        (case / name).write_text(json.dumps({"accepted": False, "candidate": candidate}))
    plain = audit.audit(tmp_path)
    cards = []
    explained = audit.audit(tmp_path, cards)
    assert plain == explained, "collecting cards changes no count and no report field"
    assert [c["id"] for c in cards] == ["case/search-1.json"] and cards[0]["crisp_tags"] == ["unclassified_needs_review"]
    assert explained["training_rows_created"] == 0 and explained["heuristic_labels_require_review"] is True
