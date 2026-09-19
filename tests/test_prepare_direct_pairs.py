import json
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "prepare_direct_pairs", Path(__file__).resolve().parents[1] / "scripts" / "prepare_direct_pairs.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
convert, jeff, nanojev, typed_decisions = (
    module.convert, module.jeff, module.nanojev, module.typed_decisions)


def test_nanojev_expands_questions_and_skips_tied_action():
    row = {
        "id": "grid:train:1", "state_id": "physical-1", "family_id": "grid",
        "split": "train", "state": "Agent at (0,0).",
        "questions": {
            "action": {"type": "choice", "instructions": "Move next.",
                       "criteria": {"east": "Go east", "south": "Go south"}},
            "solvable": {"type": "boolean", "instructions": "Can it reach the goal?"},
            "value": {"type": "score", "instructions": "How far?", "criteria": ["Near", "Far"]},
        },
        "gold": {"action": "east", "solvable": True, "value": 1},
        "gold_label_kind": {"action": "reference_argmax_compatibility", "solvable": "deterministic_truth", "value": "deterministic_truth"},
        "optimal_actions": {"action": ["east", "south"]},
        "metadata": {"source": "self_authored_programmatic", "license": "CC0-1.0", "source_group_id": "same-board"},
    }
    tasks = list(nanojev(row))
    assert [task["id"].split("/")[-1] for task in tasks] == ["solvable", "value"]
    assert [task["gold"] for task in tasks] == ["true", "1"]
    assert tasks[0]["group_id"] == tasks[1]["group_id"] == "nanojev/same-board"
    assert tasks[1]["criteria"] == {"0": "Near", "1": "Far"}


def test_jeff_benchmark_remains_test_and_retains_rubric():
    row = {"id": "sms_spam-16", "state": "A message", "questions": {
        "q": {"type": "noul", "instructions": "Is this spam?",
              "criteria": {"true": None, "false": "Personal"}}}, "gold": False}
    task, = jeff(row, filename="sms_spam")
    assert task["split"] == "test"
    assert task["kind"] == "boolean"
    assert task["labels"] == ["false", "true"]
    assert task["gold"] == "false"
    assert task["criteria"]["false"] == "Personal"
    assert task["criteria"]["true"] == "true"


def test_typed_decisions_decodes_json_columns_and_teacher_provenance():
    row = {"id": "tr_customer_service_000001", "workflow": "customer_service", "split": "train",
           "state": '{"message":"Can I get a refund?"}',
           "questions": json.dumps({"refund": {"type": "choice", "instructions": "Route request.",
                                              "criteria": {"yes": None, "no": "No refund"}}}),
           "gold": json.dumps({"refund": {"label": "yes", "confidence": 0.7}})}
    task, = typed_decisions(row)
    assert task["gold"] == "yes"
    assert task["gold_source"] == "synthetic"
    assert task["criteria"]["yes"] == "yes"
    assert task["license"] == "Apache-2.0"
    assert task["source_meta"]["teacher_confidence"] == 0.7


def test_convert_deduplicates_repeated_configs(tmp_path):
    row = {"id": "tr_a_000001", "workflow": "a", "split": "train", "state": "state",
           "questions": {"q": {"type": "boolean", "instructions": "Is it true?"}},
           "gold": {"q": {"label": "true"}}}
    source = tmp_path / "rows.jsonl"
    source.write_text((json.dumps(row) + "\n") * 2)
    output = tmp_path / "out.jsonl"
    counts = convert("typed-decisions", [source], output)
    assert counts == {"train": 1, "duplicate_tasks": 1}
    assert len(output.read_text().splitlines()) == 1


def test_nanojev_family_filter_skips_algorithmic_rows(tmp_path):
    row = {"id": "grid:train:1", "state_id": "s1", "family_id": "grid_navigation_bfs_v1",
           "split": "train", "state": "A board", "questions": {
               "solvable": {"type": "boolean", "instructions": "Can it reach the goal?"}},
           "gold": {"solvable": True}, "metadata": {"source": "self_authored_programmatic",
                                               "license": "CC0-1.0"}}
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "out.jsonl"
    counts = convert("nanojev", [source], output, nanojev_families={"support_decisions_v1"})
    assert counts == {"excluded_family": 1}
    assert output.read_text() == ""
