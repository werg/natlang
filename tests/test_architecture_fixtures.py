"""Small architectural fixtures keep the reference policy's action shape visible."""
import json
import random

from natlang.gen.architectures import dependency_plan, order_saga, reconciliation
from scripts.generate import run_program


def _calls(samples):
    return [json.loads(c["function"]["arguments"])
            for sample in samples
            for c in sample["target"].get("tool_calls", [])
            if c["function"]["name"] == "call"]


def test_reconciliation_fixture_passes_paths_and_dependent_locals():
    samples, _ = run_program(reconciliation(random.Random(1)))
    calls = _calls(samples)
    assert calls[0] == {"function": "join_events", "to": "let/joined",
                        "inputs": {"customers": "args/customers", "events": "args/events"}}
    assert {"function": "assess", "to": "let/labels", "over": "let/joined/rows"} in calls
    assert {"function": "summarize", "to": "return",
            "inputs": {"joined": "let/joined", "labels": "let/labels"}} in calls


def test_dependency_fixture_teaches_repeat_and_taken_branch_skips():
    samples, _ = run_program(dependency_plan(random.Random(1)))
    calls = _calls(samples)
    assert {"function": "prepare", "to": "let/initial",
            "inputs": {"tasks": "args/tasks"}} in calls
    assert {"function": "step", "to": "return", "init": "let/initial",
            "until": "finished", "max": 16} in calls
    assert any(c["function"]["name"] == "mark_done" and
               json.loads(c["function"]["arguments"]).get("skipped") is True and
               json.loads(c["function"]["arguments"]).get("start") == 4
               for sample in samples for c in sample["target"].get("tool_calls", []))


def test_dependency_empty_fixture_marks_repeat_branch_skipped():
    samples, _ = run_program(dependency_plan(random.Random(2)))
    assert any(json.loads(c["function"]["arguments"]) ==
               {"start": 5, "end": 6, "skipped": True}
               for sample in samples for c in sample["target"].get("tool_calls", [])
               if c["function"]["name"] == "mark_done")


def test_semantic_leaf_reads_a_collection_hidden_by_the_preview():
    samples, _ = run_program(dependency_plan(random.Random("926:541")))
    assert any(sample["native_target"] ==
               "<|tool_call_start|>[read(path='args/ready')]"
               for sample in samples)


def test_saga_fixture_covers_duplicate_skip_and_resumable_dispatch():
    samples, _ = run_program(order_saga(random.Random(1)))
    calls = _calls(samples)
    assert {"function": "dispatch", "to": "let/sent"} in calls
    assert any(c["function"]["name"] == "write" and
               json.loads(c["function"]["arguments"]).get("source") == "args/acc"
               for sample in samples for c in sample["target"].get("tool_calls", []))
    assert any(json.loads(c["function"]["arguments"]) ==
               {"start": 5, "end": 11, "skipped": True}
               for sample in samples for c in sample["target"].get("tool_calls", [])
               if c["function"]["name"] == "mark_done")
