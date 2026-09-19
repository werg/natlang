"""The classifier adapter never promotes fallback or malformed answers to gold."""

import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "label_classifier", Path(__file__).resolve().parents[1] / "scripts" / "label_classifier.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def task(rid, state="hello", gold=None):
    row = {"id": rid, "state": state, "instruction": "Is this a greeting?",
           "kind": "boolean", "labels": ["false", "true"]}
    if gold is not None:
        row.update(gold=gold, gold_source="human")
    return row


def prepared(row):
    rid, labels, rubric, text = module.validate(row, 1)
    return {"row": row, "id": rid, "labels": labels, "instructions": rubric,
            "text": text, "hash": module.task_hash(row)}


def test_same_rubric_batch_and_existing_gold():
    rows = [prepared(task("a")), prepared(task("b", gold="false"))]
    chunk, request = next(module.groups(rows, 20))
    assert [x["id"] for x in chunk] == ["a", "b"]
    assert request == {"inputs": ["hello", "hello"], "labels": ["false", "true"],
                       "instructions": "Is this a greeting?", "tier": "fast"}
    response = {"model": "jev-1.13.0", "results": [
        {"label": "true", "model": "jev-1.13.0", "scores": {"false": .1, "true": .9}},
        {"label": "true", "model": "jev-1.13.0", "scores": {"false": .1, "true": .9}},
    ]}
    first = module.make_result(chunk[0], request, response, 0, 200, {"x-api-version": "v1"}, [])
    second = module.make_result(chunk[1], request, response, 1, 200, {}, [])
    assert (first["gold"], first["gold_source"], first["jev"]["status"]) == ("true", "jev", "accepted")
    assert (second["gold"], second["gold_source"]) == ("false", "human")
    assert first["jev"]["response"] == response
    assert first["jev"]["api_version"] == "v1"


def test_fallback_invalid_and_unscored_are_not_gold():
    item = prepared(task("a"))
    request = {"inputs": ["hello"], "labels": ["false", "true"], "tier": "fast"}
    for result in (
        {"label": "true", "model": "openrouter/foo"},
        {"label": "maybe", "model": "jev-1"},
        {"label": "true", "model": "jev-1", "scores": None},
        {"label": "true", "model": "jev-1", "unscored": "not natural language"},
    ):
        output = module.make_result(item, request, {"model": "mixed", "results": [result]}, 0, 200, {}, [])
        assert output["jev"]["status"] == "rejected"
        assert "gold" not in output


def test_resume_detects_changed_tasks(tmp_path):
    src = tmp_path / "tasks.jsonl"
    dst = tmp_path / "labels.jsonl"
    row = task("a")
    src.write_text(json.dumps(row) + "\n", encoding="utf-8")
    saved = module.make_result(prepared(row), {}, {"model": "jev-1", "results": [{"label": "true"}]}, 0, 200, {}, [])
    dst.write_text(json.dumps(saved) + "\n", encoding="utf-8")
    assert module.read_tasks(src, module.read_existing(dst), False, 10) == []
    src.write_text(json.dumps(task("a", state="changed")) + "\n", encoding="utf-8")
    try:
        module.read_tasks(src, module.read_existing(dst), False, 10)
    except ValueError as exc:
        assert "changed since output" in str(exc)
    else:
        raise AssertionError("changed task was silently skipped")


def test_compact_audit_keeps_item_result_and_batch_reference():
    item = prepared(task("a"))
    request = {"inputs": ["hello"], "labels": ["false", "true"], "tier": "fast"}
    response = {"model": "jev-1", "results": [{"label": "true", "scores": {"false": .1, "true": .9}}]}
    output = module.make_result(item, request, response, 0, 200, {}, [], batch_id="batch-1")
    assert output["gold"] == "true"
    assert output["jev"]["batch_id"] == "batch-1"
    assert output["jev"]["result"] == response["results"][0]
    assert "request" not in output["jev"] and "response" not in output["jev"]
