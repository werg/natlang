import json

import pytest

from scripts import finalize_synthetic_backfill as finalize


def test_wait_for_pass_counts_the_full_unique_audit(tmp_path, monkeypatch):
    audit_path = tmp_path / "teacher.jsonl"
    calls = iter([2, 5])
    monkeypatch.setattr(finalize, "audit_rows", lambda path: next(calls))
    starts = iter(["tick", "tick", None])
    monkeypatch.setattr(finalize, "process_start", lambda pid: next(starts))
    monkeypatch.setattr(finalize.time, "sleep", lambda seconds: None)

    finalize.wait_for_pass(9, "tick", audit_path, expected_attempts=5, poll_seconds=1)


def test_wait_for_pass_rejects_recycled_pid(tmp_path, monkeypatch):
    monkeypatch.setattr(finalize, "audit_rows", lambda path: 0)
    monkeypatch.setattr(finalize, "process_start", lambda pid: "new-tick")
    with pytest.raises(RuntimeError, match="PID was reused"):
        finalize.wait_for_pass(9, "old-tick", tmp_path / "teacher.jsonl", 1, 1)


def test_file_hash_is_sha256(tmp_path):
    path = tmp_path / "bank.jsonl"
    path.write_text("reference\n")
    import hashlib
    assert finalize.file_hash(path) == hashlib.sha256(b"reference\n").hexdigest()


def test_retry_missing_omits_per_turn_cap(tmp_path, monkeypatch):
    audit_path = tmp_path / "retry.jsonl"
    monkeypatch.setattr(finalize, "audit", lambda src, references: {
        "summary": {"missing_keys": 2}})
    commands = []

    def run(command):
        commands.append(command)
        audit_path.write_text("{}\n{}\n")

    monkeypatch.setattr(finalize, "run", run)
    attempts = finalize.retry_missing_without_turn_cap(
        tmp_path / "frozen.ir.jsonl", tmp_path / "references.jsonl", audit_path)
    assert attempts == 2
    assert len(commands) == 1
    assert "--turn-tokens" not in commands[0]
    assert commands[0][-2:] == ["--audit-out", str(audit_path)]


def test_render_sft_checks_count_and_template_before_publishing(tmp_path, monkeypatch):
    result = {"ir": str(tmp_path / "snapshot.ir.jsonl"),
              "shards": str(tmp_path / "shards"),
              "trace_counts": {"eligible_turns": 2}}

    def render(command):
        stage = tmp_path / "snapshot.sft.jsonl.building"
        assert command[2:4] == [str(tmp_path / "shards"), str(stage)]
        stage.write_text('{"id":"a"}\n{"id":"b"}\n')
        stage.with_suffix(stage.suffix + ".manifest.json").write_text(json.dumps({
            "pairs": 2, "renderer": {"template_sha256": "expected"}}))

    monkeypatch.setattr(finalize, "run", render)
    finalized = finalize.render_sft(result, "http://localhost:8080", "student",
                                    "expected", 4)
    assert finalized["sft_pairs"] == 2
    assert (tmp_path / "snapshot.sft.jsonl").exists()
    assert (tmp_path / "snapshot.sft.jsonl.manifest.json").exists()
    assert json.loads((tmp_path / "snapshot.summary.json").read_text())["sft_pairs"] == 2
