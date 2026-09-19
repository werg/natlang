import json

from natlang.gen import codebases as C
from scripts.teacher_leaves import missing_from_ir


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
