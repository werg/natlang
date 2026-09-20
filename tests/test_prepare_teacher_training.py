import json
import pytest

from scripts.prepare_teacher_training import read_reviews, select, select_programs
from scripts.program_ir import digest


def test_reviewed_selection_uses_exact_audit_and_bank_value(tmp_path):
    audit = tmp_path / "audit.jsonl"
    trajectory = tmp_path / "teacher.trajectory.ir.jsonl"
    def row(line, value, admitted=False):
        return {"id": f"teacher:{line}", "task": {"kind": "generative_leaf",
                "reference_key": "say:key"},
                "provenance": {"audit_file": str(audit), "audit_line": line},
                "outcome": {"status": "done", "value": value, "accepted": admitted,
                            "admitted": admitted}}
    trajectory.write_text("\n".join(json.dumps(x) for x in (
        row(1, "right", True), row(2, "right"), row(3, "wrong"))) + "\n")
    review = tmp_path / "manual.jsonl"
    review.write_text("\n".join(json.dumps(x) for x in (
        {"source_audit": str(audit), "line": 1, "key": "say:key", "manual_keep": False},
        {"source_audit": str(audit), "line": 2, "key": "say:key", "manual_keep": True},
        {"source_audit": str(audit), "line": 3, "key": "say:key", "manual_keep": True})) + "\n")
    chosen, stats = select([trajectory], {"say:key": "right"}, read_reviews([review]))
    assert [x["id"] for x in chosen] == ["teacher:2"]
    assert chosen[0]["training_admission"]["kind"] == "manual"
    assert stats["review_rejected"] == 1
    assert stats["nonmatching_or_failed"] == 1


def test_unlocated_manual_review_inherits_rejudgment_source(tmp_path):
    audit = tmp_path / "audit.jsonl"
    automated = tmp_path / "automated.jsonl"
    manual = tmp_path / "manual.jsonl"
    automated.write_text(json.dumps({"source": str(audit), "line": 7, "key": "say:key",
                                     "accepted": True, "admitted": True}) + "\n")
    manual.write_text(json.dumps({"key": "say:key", "manual_keep": False}) + "\n")
    decision = read_reviews([automated, manual])[(str(audit.resolve()), 7)]
    assert decision["kind"] == "manual"
    assert decision["approved"] is False
    located = tmp_path / "located.jsonl"
    located.write_text(json.dumps({"source_audit": str(audit), "line": 7,
                                   "key": "say:key", "manual_keep": False}) + "\n")
    assert read_reviews([located, automated])[(str(audit.resolve()), 7)]["approved"] is False


def test_whole_program_selection_requires_trace_admission_and_deduplicates(tmp_path):
    path = tmp_path / "programs.jsonl"
    def row(name, admitted):
        return {"id": name, "task": {"kind": "whole_program", "program_ir": {"id": "p"}},
                "provenance": {"program_ir_sha256": digest({"id": "p"})},
                "outcome": {"accepted": admitted, "admission": {"admitted": admitted}}}
    path.write_text("\n".join(json.dumps(x) for x in
                              (row("failed", False), row("first", True), row("duplicate", True))) + "\n")
    selected, stats = select_programs([path])
    assert [x["id"] for x in selected] == ["first"]
    assert stats == {"not_admitted": 1, "duplicate_program": 1}
    tampered = tmp_path / "tampered.jsonl"
    bad = row("bad", True)
    bad["provenance"]["program_ir_sha256"] = "incorrect"
    tampered.write_text(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        select_programs([tampered])
