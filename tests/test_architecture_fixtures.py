"""Small architectural fixtures keep the reference policy's action shape visible."""
import json
import random

from natlang.gen.architectures import dependency_plan, order_saga, reconciliation
from scripts.generate import run_program


def _calls(samples):
    return [(c["function"]["name"], json.loads(c["function"]["arguments"]))
            for sample in samples
            for c in sample["target"].get("tool_calls", [])
            if c["function"]["name"] in ("run_function", "for_each", "fold", "repeat", "resume")]


def test_reconciliation_fixture_passes_paths_and_dependent_locals():
    samples, _ = run_program(reconciliation(random.Random(1)))
    calls = _calls(samples)
    assert calls[0] == ("run_function", {"function": "join_events",
                         "inputs": ["args/customers", "args/events"], "save_as": "let/joined"})
    assert ("for_each", {"function": "assess", "items": "let/joined/rows",
                          "save_as": "let/labels"}) in calls
    assert ("run_function", {"function": "summarize",
            "inputs": ["let/joined", "let/labels"], "save_as": "return"}) in calls


def test_dependency_fixture_teaches_repeat_and_taken_branch_skips():
    samples, _ = run_program(dependency_plan(random.Random(1)))
    calls = _calls(samples)
    assert ("run_function", {"function": "prepare", "inputs": ["args/tasks"],
                              "save_as": "let/initial"}) in calls
    assert ("repeat", {"function": "step", "initial": "let/initial",
                        "until": "finished", "at_most": 16, "save_as": "return"}) in calls
    assert any(c["function"]["name"] == "mark_lines" and
               json.loads(c["function"]["arguments"]).get("skipped") is True and
               json.loads(c["function"]["arguments"]).get("start") == 4
               for sample in samples for c in sample["target"].get("tool_calls", []))


def test_dependency_empty_fixture_marks_repeat_branch_skipped():
    samples, _ = run_program(dependency_plan(random.Random(2)))
    assert any(json.loads(c["function"]["arguments"]) ==
               {"start": 5, "end": 6, "skipped": True}
               for sample in samples for c in sample["target"].get("tool_calls", [])
               if c["function"]["name"] == "mark_lines")


def test_semantic_leaf_reads_a_collection_hidden_by_the_preview():
    samples, _ = run_program(dependency_plan(random.Random("926:541")))
    assert any(sample["native_target"] ==
               "<|tool_call_start|>[read(path='args/ready')]"
               for sample in samples)


def test_saga_fixture_covers_duplicate_skip_and_resumable_dispatch():
    samples, _ = run_program(order_saga(random.Random(1)))
    calls = _calls(samples)
    assert (("run_function", {"function": "dispatch", "save_as": "let/sent"}) in calls or
            ("resume", {"computation": "let/sent"}) in calls)
    assert any(c["function"]["name"] == "copy_value" and
               json.loads(c["function"]["arguments"]).get("source") == "args/acc"
               for sample in samples for c in sample["target"].get("tool_calls", []))
    assert any(json.loads(c["function"]["arguments"]) ==
               {"start": 5, "end": 11, "skipped": True}
               for sample in samples for c in sample["target"].get("tool_calls", [])
               if c["function"]["name"] == "mark_lines")
