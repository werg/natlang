import json

from natlang.gen import codebases as C
from scripts.teacher_leaves import _leaf_audit_status, missing_from_ir, retry_keys


def test_missing_from_ir_uses_exact_template_cases_and_deduplicates(tmp_path):
    missing_args = {"purpose": "test page"}
    known_args = {"purpose": "known page"}
    known_key = C.ref_key("page_content", known_args)
    records = [
        {"semantics": {"leaf_oracles": {
            "page_content": {"cases": [
                {"input": missing_args, "template": True},
                {"input": missing_args, "template": True},
                {"input": known_args, "template": True},
                {"input": {"purpose": "already concrete"}, "template": False},
            ]},
        }}},
        {"semantics": {"leaf_oracles": []}},
    ]
    path = tmp_path / "programs.ir.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    original = C.REFERENCES.get(known_key)
    C.REFERENCES[known_key] = "known reference"
    try:
        assert missing_from_ir(path) == [(C.ref_key("page_content", missing_args),
                                          "page_content", missing_args)]
    finally:
        if original is None:
            del C.REFERENCES[known_key]
        else:
            C.REFERENCES[known_key] = original


def test_missing_from_ir_prioritizes_keys_that_finish_programs(tmp_path):
    x = {"purpose": "x"}
    y = {"purpose": "y"}
    z = {"purpose": "z"}
    records = [
        {"semantics": {"leaf_oracles": {"page_content": {"cases": [
            {"input": x, "template": True},
        ]}}}},
        {"semantics": {"leaf_oracles": {"page_content": {"cases": [
            {"input": x, "template": True},
            {"input": y, "template": True},
        ]}}}},
        {"semantics": {"leaf_oracles": {"page_content": {"cases": [
            {"input": z, "template": True},
        ]}}}},
    ]
    path = tmp_path / "programs.ir.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    original = dict(C.REFERENCES)
    C.REFERENCES.clear()
    try:
        todo = missing_from_ir(path)
        assert [item[1:] for item in todo] == [
            ("page_content", x), ("page_content", z), ("page_content", y)
        ]
    finally:
        C.REFERENCES.clear()
        C.REFERENCES.update(original)


def test_leaf_exception_is_serializable_as_an_audit_outcome():
    error = {"type": "TimeoutError", "message": "judge timed out"}
    assert _leaf_audit_status(None, error) == ("exception", error)


def test_retry_keys_selects_only_rejected_statuses(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in [
        {"key": "a", "accepted": True, "status": "done"},
        {"key": "b", "accepted": False, "status": "done"},
        {"key": "c", "accepted": False, "status": "quiesced"},
    ]) + "\n")
    assert retry_keys(path) == {"b", "c"}
    assert retry_keys(path, "quiesced") == {"c"}
