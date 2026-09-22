import json

from scripts.collect_teacher_batch import (
    expected_provenance, import_completed, job_key, merge_completed, result_matches, run_job, write_atomic,
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


def test_import_adopts_compact_rows_and_rejects_full_conversation_legacy(tmp_path):
    rows = [(0, record("compact")), (1, record("legacy"))]
    expected = lambda row: expected_provenance(row, model_id="teacher", root_seed=4,
        system_prompt="p", segment_turns=6, segment_messages=12)
    source = tmp_path / "old.jsonl"
    old = []
    for index, row in rows:
        provenance = {key: value for key, value in expected(row).items()
                      if key not in ("segment_turns", "segment_messages")}
        old.append({"task": {"program_ir": row}, "provenance": provenance,
                    "trajectory": [{"context": [{}] * (8 if index == 0 else 40)}]})
    source.write_text("".join(json.dumps(row) + "\n" for row in old))
    imported, rejected = import_completed([source], rows, tmp_path, expected,
        segment_turns=6, segment_messages=12)
    assert (imported, rejected) == (1, 1)
    assert result_matches(tmp_path / (job_key(0, rows[0][1]) + ".result.json"),
                          rows[0][1], expected(rows[0][1]))


def test_retry_reuses_canonical_trace_and_defers_obsolete_cleanup(tmp_path, monkeypatch):
    row = record()
    key = job_key(2, row)
    (tmp_path / f"{key}.trace.jsonl").write_text("old")
    (tmp_path / f"{key}.retry1.trace.jsonl").write_text("older")
    observed = {}
    def fake_collect(record, decoder, **kwargs):
        observed["trace"] = kwargs["trace_path"]
        assert (tmp_path / f"{key}.retry1.trace.jsonl").exists()
        return ({"task": {"program_ir": record}, "provenance": {},
                 "outcome": {"status": "done", "accepted": True}}, None)
    monkeypatch.setattr("scripts.collect_teacher_batch.collect", fake_collect)
    monkeypatch.setattr("scripts.collect_teacher_batch.decoder_for", lambda args: object())
    args = type("Args", (), {"model_id": "teacher", "root_seed": 4,
        "segment_turns": 6, "segment_messages": 12})()
    expected = expected_provenance(row, model_id="teacher", root_seed=4,
        system_prompt="p", segment_turns=6, segment_messages=12)
    run_job(2, row, args, "p", tmp_path, expected)
    assert observed["trace"] == tmp_path / f"{key}.trace.jsonl"
