from scripts.roll_teacher_runtime import queue_exhausted


def test_finished_rejected_and_timeout_attempts_are_boundaries():
    entries = {"accepted": {}, "rejected": {}, "timeout": {}}
    rows = [{"event": "finish", "key": key, "status": status} for key, status in
            [("accepted", "complete"), ("rejected", "complete"), ("timeout", "timeout")]]
    assert queue_exhausted(entries, rows)


def test_started_but_unfinished_attempt_requires_inspection():
    assert not queue_exhausted({"unfinished": {}}, [{"event": "start", "key": "unfinished"}])


def test_other_queue_completion_cannot_authorize_resume():
    assert not queue_exhausted({"expected": {}}, [{"event": "finish", "key": "other"}])
