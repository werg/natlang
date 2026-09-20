import pytest

from hosts.recovery import RecoveryStore
from natlang.runtime import Runtime
from natlang.streams import QueueSource, StreamBuffer
from natlang.values import dump_state, load_program


def test_map_state_snapshot_restores_completed_and_pending_slots():
    document = {"$map": {"type": "Map<Num, Num>", "over": [1, 3],
        "fn": {"$lambda": {"type": "Lambda<{ item: Num }, Num>",
                           "code": "return args.item * 2;"}},
        "slots": [2, {"$lambda": {"type": "Lambda<{ item: Num }, Num>",
                                  "code": "return args.item * 2;", "args": {"item": 3}}}]}}
    restored = load_program(dump_state(load_program(document)))
    outcome, value = Runtime(None).run_root(restored)
    assert outcome.kind == "done" and value == [2, 6]


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


def test_fold_checkpoint_restores_at_completed_step_boundary(tmp_path):
    def fold(source):
        root = load_program({"$fold": {"type": "Fold<Num, Num>", "init": 0,
            "step": {"$lambda": {"type": "Lambda<{ acc: Num, item: Num }, Num>",
                                "code": "return args.acc + args.item;"}}}})
        root.over = StreamBuffer(source)
        return root
    before = QueueSource()
    before.put(2); before.put(3)
    root = fold(before)
    out, _ = Runtime(None).run_root(root)
    assert out.kind == "waiting"
    path = tmp_path / "fold.sqlite"
    with RecoveryStore(path) as store:
        store.checkpoint_fold("wf", 1, root)
    after = QueueSource()
    after.put(4); after.close()
    with RecoveryStore(path) as store:
        restored = store.restore_fold("wf", fold(QueueSource()), after)
        assert restored.over.position == 2 and restored.acc == 5
        out, value = Runtime(None).run_root(restored)
        assert (out.kind, value) == ("done", 9)
