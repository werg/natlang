"""Frozen source programs gain checked leaf answers without changing their inputs."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refresh_synthetic_references import load_references, replace_oracles  # noqa: E402
from natlang.gen.codebases import ref_key  # noqa: E402


def test_replacement_preserves_source_and_marks_remaining_templates():
    first, second = {"heard": "hello"}, {"heard": "goodbye"}
    record = {"id": "fixture", "semantics": {"root": {"source": "unchanged"},
              "contains_templates": True, "leaf_oracles": {"say": {"cases": [
                  {"input": first, "output": "canned", "template": True},
                  {"input": second, "output": "canned", "template": True}]}}}}
    updated, replaced = replace_oracles(record, {ref_key("say", first): "Teacher answer"})
    assert replaced == 1
    assert updated["semantics"]["root"] == record["semantics"]["root"]
    assert updated["semantics"]["leaf_oracles"]["say"]["cases"][0]["output"] == "Teacher answer"
    assert updated["semantics"]["contains_templates"]
    assert record["semantics"]["leaf_oracles"]["say"]["cases"][0]["template"]


def test_reference_bank_rejects_conflicting_or_miskeyed_rows(tmp_path):
    args = {"heard": "hello"}
    key = ref_key("say", args)
    path = tmp_path / "references.jsonl"
    row = {"key": key, "function": "say", "args": args, "value": "First"}
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "value": "Different"}) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        load_references(path)
    path.write_text(json.dumps({**row, "key": "wrong"}) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        load_references(path)
    path.write_text(json.dumps({**row, "value": 7}) + "\n")
    with pytest.raises(ValueError, match="must be text"):
        load_references(path)
