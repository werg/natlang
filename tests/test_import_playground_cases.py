"""Browser case admission preserves exact source, outcome, and action obligations."""
import sys
import json
import subprocess
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_playground_cases import convert  # noqa: E402
from program_ir import lower  # noqa: E402


def case():
    events = [
        {"version": "reduction-trace/1", "seq": 0, "kind": "manifest", "run_id": "run-1"},
        {"version": "reduction-trace/1", "seq": 1, "kind": "state", "phase": "initial", "value": None},
        {"version": "reduction-trace/1", "seq": 2, "kind": "action", "call_id": "$root@1",
         "name": "write", "arguments": {"path": "return", "type": "Num", "value": 7}, "outcome": "ok"},
        {"version": "reduction-trace/1", "seq": 3, "kind": "state", "phase": "final", "value": 7, "outcome": "done"},
    ]
    return {"schema": "natlang.playground.case/1", "id": "case-1", "projectId": "project-1",
            "groupId": "project-1", "revision": "revision-1", "split": "train",
            "reviewStatus": "accepted", "admission": {"admitted": True},
            "source": {"root": "tasks/answer.nl", "files": {
                "tasks/answer.nl": "---\nreturns: Num\n---\nWrite seven to return.\n"}},
            "inputs": {}, "expected": {"kind": "done", "value": 7},
            "requiredActions": [{"name": "write", "arguments": {"path": "return"}}],
            "effects": [], "constrainedCalls": [], "trace": events}


def test_accepted_leaf_becomes_shared_lambda_scenario(tmp_path):
    record = convert(case(), tmp_path)
    assert record["version"] == "natlang.program/1"
    assert record["kind"] == "lambda_scenario"
    assert record["source_groups"] == ["project-1"]
    assert record["semantics"]["operations"] == [
        {"op": "assign", "target": "return", "value_type": "Num", "value": 7}]
    assert lower(record).expected == 7


def test_unreviewed_or_complex_case_is_rejected(tmp_path):
    unreviewed = case()
    unreviewed["reviewStatus"] = "draft"
    with pytest.raises(ValueError, match="reviewed"):
        convert(unreviewed, tmp_path)
    nested = case()
    nested["trace"][2]["call_id"] = "child@1"
    with pytest.raises(ValueError, match="richer adapter"):
        convert(nested, tmp_path)


def test_cli_case_to_materialized_episode_round_trip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "cases.jsonl"
    ir = tmp_path / "programs.jsonl"
    episodes = tmp_path / "episodes.jsonl"
    source.write_text(json.dumps(case()) + "\n")
    imported = subprocess.run([sys.executable, str(root / "scripts/import_playground_cases.py"),
                               str(source), str(ir)], capture_output=True, text=True, check=True)
    assert json.loads(imported.stdout)["accepted"] == 1
    assert ir.read_text().count("\n") == 1
    subprocess.run([sys.executable, str(root / "scripts/materialize_ir.py"),
                    str(ir), str(episodes)], capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in episodes.read_text().splitlines()]
    assert len(rows) >= 1
