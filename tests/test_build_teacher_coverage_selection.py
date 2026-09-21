import json

import pytest

from scripts.build_teacher_coverage_selection import build


def row(identity, family):
    return {"id": identity, "kind": "lambda_source", "family": family, "split": "train",
            "semantics": {}}


def test_coverage_selection_combines_new_corpora_and_balances_families(tmp_path):
    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    one.write_text("".join(json.dumps(row(f"a{i}", "a")) + "\n" for i in range(3)))
    two.write_text("".join(json.dumps(item) + "\n" for item in
                           [row("a3", "a"), row("b0", "b"), row("b1", "b")]))
    rows, manifest = build([one, two], per_family=2, seed=7)
    assert len(rows) == 4 and manifest["families"] == 2
    assert {item["family"] for item in rows} == {"a", "b"}


def test_coverage_selection_fails_when_new_family_lacks_variants(tmp_path):
    source = tmp_path / "one.jsonl"
    source.write_text(json.dumps(row("only", "new")) + "\n")
    with pytest.raises(ValueError, match="new"):
        build([source], per_family=2, seed=7)
