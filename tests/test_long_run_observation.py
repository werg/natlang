from applications.long_run import inspect_run, inspect_trace
from natlang.trace import TraceRecorder


def test_live_trace_shows_inflight_request_without_stuck_verdict(tmp_path):
    path = tmp_path / "running.jsonl"
    recorder = TraceRecorder({"run_id": "a", "source_sha256": "b"}, path)
    recorder.emit("model_request", call_id="$root@1", phase="start", turn=1)
    live = inspect_trace(path)
    assert live["model_requests_in_flight"][0]["call_id"] == "$root@1"
    assert live["seconds_since_event"] is not None
    recorder.emit("model_request", call_id="$root@1", phase="end", turn=1,
                  duration_ms=12, completion_tokens=5)
    recorder.emit("action", call_id="$root@1", name="read", arguments={"path": "args/x"})
    recorder.emit("action", call_id="$root@1", name="read", arguments={"path": "args/x"})
    recorder.close()
    done = inspect_trace(path)
    assert done["model_requests_in_flight"] == []
    assert done["completion_tokens_reported"] == 5
    assert done["consecutive_identical_actions"] == 2


def test_live_run_can_be_inspected_before_journal_starts(tmp_path):
    result = inspect_run(journal=tmp_path / "journal.jsonl", trace_dir=tmp_path)
    assert result["journal"]["status"] == "not_started"
