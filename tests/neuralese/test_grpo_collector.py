"""CPU checks of grpo.py's collector slots: an episode killed by a signal (a unit stop, the OOM killer) abandons its group;
it is never scored with the killed episodes as unresolved."""
import collections
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402


class FakeProcess:
    def __init__(self, returncode, on_exit=None):
        self.returncode, self.pid, self.on_exit = returncode, 0, on_exit

    def poll(self):
        if self.on_exit:
            self.on_exit()
        return self.returncode

    def wait(self):
        return self.returncode


def collector(tmp_path, size=3):
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = SimpleNamespace(actor='http://actor')
    c.work, c.runtime, c.env = tmp_path, tmp_path / 'ts-host', {}
    c.lock, c.stop, c.paused = threading.Lock(), threading.Event(), threading.Event()
    c.active_slots, c.version, c.outstanding = 1, 0, 2
    for g in (0, 1):
        (tmp_path / f'g{g}').mkdir()
    c.groups = {g: {'dir': tmp_path / f'g{g}', 'size': size, 'done': 0, 'versions': [], 'task_id': f't{g}', 'created': 0}
                for g in (0, 1)}
    c.pending = collections.deque((g, {'key': f'k{g}-{i}'}) for g in (0, 1) for i in range(size))
    c.scored = []
    c._score = c.scored.append
    return c


def run_slot(c, monkeypatch, returncodes, stop_on_exit=()):
    """One slot over the pending entries; the n-th episode exits with ``returncodes[n]`` (the stop event set when it
    exits if n is in ``stop_on_exit``, as when a unit stop reaches child and parent together)."""
    started = []

    def popen(*_args, **_kwargs):
        n = len(started)
        started.append(n)
        return FakeProcess(returncodes[n], c.stop.set if n in stop_on_exit else None)

    def take():
        if c.pending and len(started) < len(returncodes):
            return c.pending.popleft()
        c.stop.set()
        return None
    monkeypatch.setattr(grpo.subprocess, 'Popen', popen)
    monkeypatch.setattr(c, '_take', take)
    c.slot()
    return started


def test_finished_episodes_complete_and_score_their_group(tmp_path, monkeypatch):
    c = collector(tmp_path)
    run_slot(c, monkeypatch, [0, 2, 0])   # 2: a budget stop or wall timeout still has its verdict
    assert c.scored == [0] and c.groups[0]['done'] == 3 and c.outstanding == 2


def test_a_signal_killed_episode_abandons_its_group(tmp_path, monkeypatch):
    c = collector(tmp_path)
    started = run_slot(c, monkeypatch, [0, -9, 0, 0, 0, 0])
    assert c.scored == [1]                     # group 0 is never scored; the slot carries on with group 1
    assert 0 not in c.groups and c.outstanding == 1
    assert len(started) == 5                   # group 0's unstarted entry was withdrawn


def test_a_running_sibling_of_an_abandoned_group_is_discarded(tmp_path, monkeypatch):
    c = collector(tmp_path)
    info = c.groups[0]
    c._abandon(0, 'k0-1', -2)
    c.pending.appendleft((0, {'key': 'k0-0'}))   # a sibling already running in another slot ends after the abandonment
    c.groups[0] = info
    run_slot(c, monkeypatch, [0])
    assert c.scored == [] and info['done'] == 0


def test_an_episode_ending_as_the_run_stops_is_not_counted(tmp_path, monkeypatch):
    """The unit stop reached the child first: its exit code is ordinary but the stop event is set."""
    c = collector(tmp_path, size=1)
    run_slot(c, monkeypatch, [2], stop_on_exit={0})
    assert c.scored == [] and 0 not in c.groups and c.outstanding == 1


def test_a_resume_reuses_untrained_scored_groups_of_policies_it_has_served(tmp_path):
    import json
    c = grpo.Collector.__new__(grpo.Collector)
    c.work, c.lock, c.groups, c.outstanding = tmp_path, threading.Lock(), {}, 0
    import queue
    c.ready = queue.Queue()

    def group(n, versions, files=('rollouts.jsonl', 'score.json'), retired=False):
        d = tmp_path / 'groups' / f'g{n:06d}'
        d.mkdir(parents=True)
        for name in files:
            (d / name).write_text('{}\n')
        (d / 'versions.json').write_text(json.dumps(versions))
        (d / 'rewards.jsonl').write_text(json.dumps({'task_id': f't{n}', 'reward': 1.0}) + '\n')
        if retired:
            (d / grpo.RETIRED).touch()
        return d
    group(1, [5, 5, 6])                     # served by this run's history: reused
    group(2, [4, 6])                        # 6 - 4 = 2 <= max staleness: reused
    group(3, [6, 7])                        # sampled by version 7, past the saved state: retired
    group(4, [3, 6])                        # too stale: retired
    group(5, [6], retired=True)             # already trained
    group(6, [6], files=('score.json',))    # never finished scoring
    reused, left = c.reuse_scored(6, 2)
    assert reused == ['g000001', 'g000002'] and left == ['g000003', 'g000004']
    assert c.outstanding == 2 and c.take_groups(2, 2) == [1, 2] and c.groups[2]['task_id'] == 't2'
    assert (tmp_path / 'groups/g000003' / grpo.RETIRED).exists()
    c2 = grpo.Collector.__new__(grpo.Collector)
    c2.work, c2.lock, c2.groups, c2.outstanding, c2.ready = tmp_path, threading.Lock(), {}, 0, queue.Queue()
    assert c2.reuse_scored(7, 2) == (['g000001'], ['g000002'])   # g3 stays retired though 7 <= 7 now; g2 is now stale
    grpo.prune_group(c, 1, keep=True)
    assert (tmp_path / 'groups/g000001' / grpo.RETIRED).exists()
