import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_provisional_leaves import audit, ref_key


def test_audit_reports_missing_coverage_and_programs(tmp_path):
    args_a = {"purpose": "A"}
    args_b = {"purpose": "B"}
    records = [
        {"id": "73:cb_webserver:1", "family": "cb_webserver", "semantics": {
            "leaf_oracles": {"page_content": {"cases": [
                {"input": args_a, "template": True},
                {"input": args_b, "template": True},
                {"input": {"purpose": "concrete"}, "template": False},
            ]}}}},
        {"id": "73:cb_shopkeeper:2", "family": "cb_shopkeeper", "semantics": {
            "leaf_oracles": {"say": {"cases": [{"input": args_a, "template": True}]}}}},
        {"id": "other:3", "family": "other", "semantics": {
            "leaf_oracles": {"say": {"cases": [{"input": args_b, "template": True}]}}}},
    ]
    ir = tmp_path / "ir.jsonl"
    ir.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    refs = tmp_path / "refs.jsonl"
    refs.write_text(json.dumps({"key": ref_key("page_content", args_a), "function": "page_content",
                                "args": args_a, "value": "ok"}) + "\n")
    report = audit(ir, refs)
    assert report["summary"] == {"provisional_programs": 2, "exact_case_occurrences": 3,
                                 "distinct_keys": 3, "covered_keys": 1, "missing_keys": 2,
                                 "affected_programs": 2, "fully_covered_programs": 0}
    assert report["affected_program_ids"] == ["73:cb_shopkeeper:2", "73:cb_webserver:1"]


def test_audit_exposes_duplicate_conflicts_and_bad_keys(tmp_path):
    args = {"purpose": "A"}
    key = ref_key("page_content", args)
    ir = tmp_path / "ir.jsonl"
    ir.write_text(json.dumps({"id": "p", "family": "cb_webserver", "semantics": {
        "leaf_oracles": {"page_content": {"cases": [{"input": args, "template": True}]}}}}) + "\n")
    refs = tmp_path / "refs.jsonl"
    refs.write_text("\n".join(json.dumps(row) for row in [
        {"key": key, "function": "page_content", "args": args, "value": "one"},
        {"key": key, "function": "page_content", "args": args, "value": "two"},
        {"key": "wrong", "function": "page_content", "args": args, "value": "bad"},
    ]) + "\n")
    refs_report = audit(ir, refs)["references"]
    assert refs_report["rows"] == 3
    assert refs_report["unique_keys"] == 2
    assert len(refs_report["duplicate_keys"]) == 1
    assert len(refs_report["conflicting_keys"]) == 1
    assert refs_report["invalid_key_rows"] == [{"line_number": 3, "key": "wrong",
                                                  "function": "page_content", "expected_key": key}]
