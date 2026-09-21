import json

from scripts.collect_teacher_batch import (
    expected_provenance, job_key, merge_completed, result_matches, write_atomic,
)


def record(name="one"):
    return {"version": "natlang.program/1", "id": name, "kind": "lambda_source",
            "source": "fixture", "split": "train", "source_ids": [name],
            "source_groups": [name], "license": "project-generated",
            "semantics": {"operation": "literal", "root": {"$lambda": {
                "type": "Lambda<{}, Text>", "instructions": "Return ok."}},
                "inputs": {}, "expected": "ok", "leaf_oracles": []}}


def result(row, provenance):
    return {"task": {"program_ir": row}, "provenance": provenance,
            "outcome": {"status": "done", "accepted": True}}


def test_atomic_jobs_resume_only_when_every_input_matches(tmp_path):
    row = record()
    expected = expected_provenance(row, model_id="teacher", root_seed=9,
                                   system_prompt="prompt", segment_turns=6,
                                   segment_messages=12)
    path = tmp_path / (job_key(3, row) + ".result.json")
    write_atomic(path, result(row, expected))
    assert result_matches(path, row, expected)
    changed = {**expected, "segment_turns": 7}
    assert not result_matches(path, row, changed)
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_merge_is_source_ordered_and_reports_missing(tmp_path):
    rows = [(9, record("nine")), (2, record("two"))]
    expected = lambda row: expected_provenance(
        row, model_id="teacher", root_seed=4, system_prompt="p",
        segment_turns=6, segment_messages=12)
    for index, row in rows:
        path = tmp_path / (job_key(index, row) + ".result.json")
        write_atomic(path, result(row, expected(row)))
    out = tmp_path / "merged.jsonl"
    count, missing = merge_completed(rows, tmp_path, out, expected)
    assert count == 2 and missing == []
    assert [json.loads(line)["task"]["program_ir"]["id"]
            for line in out.read_text().splitlines()] == ["nine", "two"]
    (tmp_path / (job_key(2, rows[1][1]) + ".result.json")).unlink()
    count, missing = merge_completed(rows, tmp_path, out, expected)
    assert count == 1 and missing == [2]
