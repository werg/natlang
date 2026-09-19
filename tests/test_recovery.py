import pytest

from hosts.recovery import RecoveryStore


class Remote:
    def __init__(self):
        self.receipts = {}
        self.dispatches = 0

    def dispatch(self, key, request):
        self.dispatches += 1
        return self.receipts.setdefault(key, {"remote_id": len(self.receipts) + 1,
                                               "request": request})

    def lookup(self, key):
        return self.receipts.get(key)


def crash_at(boundary):
    def crash(phase):
        if phase == boundary:
            raise RuntimeError("crash injection")
    return crash


def test_crash_before_dispatch_retries_planned_operation(tmp_path):
    path = tmp_path / "workflow.sqlite"
    remote = Remote()
    with RecoveryStore(path) as store:
        with pytest.raises(RuntimeError):
            store.perform("wf", "send-1", {"to": "Dana"}, remote,
                          crash_at("before_dispatch"))
        assert store.load("wf")["operations"][0]["status"] == "planned"
    with RecoveryStore(path) as store:
        receipt = store.perform("wf", "send-1", {"to": "Dana"}, remote)
        assert receipt["remote_id"] == 1 and remote.dispatches == 1


def test_remote_commit_before_ack_reconciles_without_duplicate(tmp_path):
    path = tmp_path / "workflow.sqlite"
    remote = Remote()
    with RecoveryStore(path) as store:
        with pytest.raises(RuntimeError):
            store.perform("wf", "send-1", {"to": "Dana"}, remote,
                          crash_at("after_external_commit_before_ack"))
        assert store.load("wf")["operations"][0]["status"] == "dispatched"
    with RecoveryStore(path) as store:
        receipt = store.perform("wf", "send-1", {"to": "Dana"}, remote)
        assert receipt["remote_id"] == 1 and remote.dispatches == 1
        assert store.load("wf")["operations"][0]["status"] == "confirmed"


def test_unresolved_dispatch_stays_uncertain_and_checkpoint_restores_position(tmp_path):
    path = tmp_path / "workflow.sqlite"
    remote = Remote()
    with RecoveryStore(path) as store:
        store.checkpoint("wf", 1, 12, {"acc": 7, "pending": ["job-1"]})
        store.plan("wf", "notify-1", {"message": "done"})
        store._status("wf", "notify-1", "dispatched")
    with RecoveryStore(path) as store:
        assert store.load("wf")["checkpoint"] == {
            "revision": 1, "position": 12, "state": {"acc": 7, "pending": ["job-1"]}}
        assert store.reconcile("wf", "notify-1", remote) is None
        assert store.load("wf")["operations"][0]["status"] == "dispatched"
        with pytest.raises(ValueError):
            store.checkpoint("wf", 2, 11, {"acc": 8})
