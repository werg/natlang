"""CPU checks of grpo.py's collector slots: an episode killed by a signal (a unit stop, the OOM killer) is never scored
with the killed episodes as unresolved: it runs again from scratch (``--episode-reruns``), then its group is abandoned;
a stop leaves the group for the restart's ``resume_unfinished``."""
import collections
import json
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


def collector(tmp_path, size=3, **args):
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = SimpleNamespace(actor='http://actor', **args)
    c.work, c.runtime, c.env = tmp_path, tmp_path / 'ts-host', {}
    c.lock, c.stop, c.paused = threading.Lock(), threading.Event(), threading.Event()
    c.active_slots, c.version, c.outstanding, c.running = 1, 0, 2, 0
    c.policy_sha, c.dynamic, c.STOP_GRACE = 'p' * 64, collections.Counter(), 0
    for g in (0, 1):
        (tmp_path / f'g{g}').mkdir(parents=True)
    c.groups = {g: {'dir': tmp_path / f'g{g}', 'size': size, 'done': 0, 'versions': [], 'task_id': f't{g}', 'created': 0}
                for g in (0, 1)}
    c.pending = collections.deque((g, {'key': f'k{g}-{i}', 'output': str(tmp_path / f'g{g}/row-000/sample-{i:02d}/results.jsonl')})
                                  for g in (0, 1) for i in range(size))
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


def test_episodes_in_flight_are_counted_while_they_run(tmp_path, monkeypatch):
    """``running`` (logged as ``episodes_in_flight`` when the learner starts) counts an episode while its process runs."""
    c = collector(tmp_path)
    seen = []

    def popen(*_args, **_kwargs):
        return FakeProcess(0 if len(seen) else -9, lambda: seen.append(c.running))

    entries = [c.pending.popleft()]
    monkeypatch.setattr(grpo.subprocess, 'Popen', popen)
    monkeypatch.setattr(c, '_take', lambda: entries.pop() if entries else c.stop.set())
    c.slot()
    assert seen == [1] and c.running == 0     # also back to 0 after a signal-killed episode


def test_a_signal_killed_episode_runs_again_from_scratch(tmp_path, monkeypatch):
    c = collector(tmp_path)
    sample = tmp_path / 'g0/row-000/sample-01'
    (sample / 'jobs').mkdir(parents=True)
    (sample / 'jobs/000000-x.partial.json').write_text('{}')
    started = run_slot(c, monkeypatch, [0, -9, 0, 0, 0, 0, 0])
    assert c.scored == [0, 1] and c.groups[0]['done'] == 3 and c.dynamic['episodes_rerun'] == 1
    assert (tmp_path / 'g0/row-000/sample-01.lost-1/jobs/000000-x.partial.json').exists() and not sample.exists()
    assert sorted(p.name for p in (tmp_path / 'g0/workers').glob('*.done.json')) == [
        'k0-0.done.json', 'k0-1.done.json', 'k0-2.done.json']
    assert json.loads((tmp_path / 'g0/workers/k0-1.done.json').read_text())['policy_sha256'] == 'p' * 64
    assert len(started) == 7


def test_a_signal_killed_episode_abandons_its_group_once_its_reruns_are_spent(tmp_path, monkeypatch):
    c = collector(tmp_path, episode_reruns=0)
    started = run_slot(c, monkeypatch, [0, -9, 0, 0, 0, 0])
    assert c.scored == [1]                     # group 0 is never scored; the slot carries on with group 1
    assert 0 not in c.groups and c.outstanding == 1
    assert len(started) == 5                   # group 0's unstarted entry was withdrawn
    assert json.loads((tmp_path / 'g0' / grpo.GROUP_LOST).read_text())['reason'] == 'abandoned'   # never resumed


def test_a_running_sibling_of_an_abandoned_group_is_discarded(tmp_path, monkeypatch):
    c = collector(tmp_path)
    info = c.groups[0]
    c._abandon(0, 'k0-1', -2)
    c.pending.appendleft((0, {'key': 'k0-0'}))   # a sibling already running in another slot ends after the abandonment
    c.groups[0] = info
    run_slot(c, monkeypatch, [0])
    assert c.scored == [] and info['done'] == 0


def test_an_episode_ending_as_the_run_stops_is_not_counted(tmp_path, monkeypatch):
    """The unit stop reached the child first: its exit code is ordinary but the stop event is set. Its group stays
    (no done marker, not abandoned) for the restart's resume; with resume off it is abandoned as before."""
    c = collector(tmp_path, size=1)
    run_slot(c, monkeypatch, [2], stop_on_exit={0})
    assert c.scored == [] and c.groups[0]['done'] == 0 and not (tmp_path / 'g0' / grpo.GROUP_LOST).exists()
    assert not list(tmp_path.glob('g0/workers/*.done.json'))
    c = collector(tmp_path / 'off', size=1, resume_unfinished=False)
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


# ---- resume_unfinished (DECISIONS 2026-11-25 "collectors resume their unfinished groups") -----------------------------

def capture_turn(policy='p' * 64, top_p=0.95, writes=()):
    return {'request_sha256': 'r', 'response': {'raw_response': {
        'native_state': {'blocks': [], 'imports': []},
        'x_natlang_rl_capture': {'policy': {'checkpoint_sha256': policy}, 'samples': [[1, -0.5]], 'writes': list(writes),
                                 'sampling_law': {'temperature': 1, 'top_p': top_p, 'guidance': False}}}}}


def stopped_group(work, number, task='t0', policy='p' * 64, k=4):
    """A group a stopped run left: k entries; the caller adds done markers, journals and partials."""
    d = work / 'groups' / f'g{number:06d}'
    (d / 'workers').mkdir(parents=True)
    (d / 'cases.jsonl').write_text(json.dumps({'id': task}) + '\n')
    (d / 'preparation.json').write_text(json.dumps({'policy_checkpoint_sha256': policy}))
    entries = [{'key': f's7-task-000-{i:02d}', 'index': 0, 'sample': i, 'seed': 1000 * number + i,
                'output': str(d / f'row-000/sample-{i:02d}/results.jsonl'), 'jobs': str(d / f'row-000/sample-{i:02d}/jobs')}
               for i in range(k)]
    (d / 'queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    return d, entries


def finished(d, entry, version=0, policy='p' * 64, marker=True):
    sample = Path(entry['output']).parent
    (sample / 'jobs').mkdir(parents=True, exist_ok=True)
    Path(entry['output']).write_text(json.dumps({'trajectory': [], 'capture': {'checkpoint_sha256': policy}}) + '\n')
    (d / 'workers' / f"{entry['key']}.journal.jsonl").write_text(json.dumps({'event': 'finish', 'key': entry['key'],
                                                                          'status': 'complete', 'exit_code': 0}) + '\n')
    if marker:
        (d / 'workers' / f"{entry['key']}.done.json").write_text(json.dumps({'version': version, 'policy_sha256': policy}))


def killed(d, entry, turns=None):
    sample = Path(entry['output']).parent
    (sample / 'jobs').mkdir(parents=True, exist_ok=True)
    (d / 'workers' / f"{entry['key']}.journal.jsonl").write_text(json.dumps({'event': 'start', 'key': entry['key']}) + '\n')
    if turns is not None:
        (sample / 'jobs' / '000000-abc.partial.json').write_text(json.dumps({'turns': turns}))


def resumer(work, cls=grpo.Collector, version=0, **args):
    c = cls.__new__(cls)
    c.args = SimpleNamespace(actor='http://actor', top_p=0.95, max_staleness=2, **args)
    c.work, c.policy_sha, c.version = work, 'p' * 64, version
    c.lock, c.stop = threading.Lock(), threading.Event()
    c.groups, c.pending, c.outstanding, c.next_group, c.dynamic = {}, collections.deque(), 0, 0, collections.Counter()
    c.scored = []
    c._score = c.scored.append
    return c


def test_a_restart_resumes_a_stopped_group_keeping_its_accepted_episodes(tmp_path):
    d, entries = stopped_group(tmp_path, 7)
    finished(d, entries[0])
    killed(d, entries[1], turns=[capture_turn()])        # grpo: versions change in a run, so it reruns from scratch
    finished(d, entries[2], marker=False)                 # finished, not accepted (the stop raced it): runs again
    c = resumer(tmp_path)
    resumed, left = c.resume_unfinished()
    assert resumed == [7] and left == {}
    info = c.groups[7]
    assert info['done'] == 1 and info['size'] == 4 and info['versions'] == [0] and info['resumed']
    assert [e['key'] for _, e in c.pending] == ['s7-task-000-01', 's7-task-000-02', 's7-task-000-03']
    assert (d / 'row-000/sample-01.stopped-1/jobs/000000-abc.partial.json').exists()          # set aside, kept
    assert (d / 'row-000/sample-01.stopped-1/worker.journal.jsonl').exists()
    assert not (d / 'row-000/sample-01').exists() and not (d / 'row-000/sample-03.stopped-1').exists()   # unstarted
    record = json.loads((d / grpo.GROUP_RESUMED).read_text())
    assert record['kept'] == ['s7-task-000-00'] and record['rerun'] == ['s7-task-000-01', 's7-task-000-02']
    assert c.outstanding == 1 and c.next_group == 8 and c.dynamic['groups_resumed'] == 1
    assert c.resume_unfinished() == ([], {})              # already held: never twice


def test_a_resume_never_keeps_an_episode_of_a_policy_version_it_no_longer_serves(tmp_path):
    d, entries = stopped_group(tmp_path, 1, k=3)
    finished(d, entries[0], version=8)                    # served by this run (step 8)
    finished(d, entries[1], version=9)                    # a version the stopped run reached past its saved state
    finished(d, entries[2], version=5)                    # too stale for --max-staleness 2
    c = resumer(tmp_path, version=8)
    c.resume_unfinished()
    assert c.groups[1]['versions'] == [8] and [e['key'] for _, e in c.pending] == ['s7-task-000-01', 's7-task-000-02']
    assert not (d / 'workers/s7-task-000-01.done.json').exists()     # its marker goes with the set-aside attempt


def test_a_group_of_another_policy_or_a_memetic_group_is_left(tmp_path):
    other, _ = stopped_group(tmp_path, 1, policy='q' * 64)
    memetic, _ = stopped_group(tmp_path, 2)
    (memetic / 'memetic-spec.json').write_text('{}')
    retired, _ = stopped_group(tmp_path, 3)
    (retired / grpo.RETIRED).touch()
    c = resumer(tmp_path)
    assert c.resume_unfinished() == ([], {'policy changed': 1, 'memetic': 1})
    assert json.loads((other / grpo.GROUP_LOST).read_text())['reason'] == 'policy changed'
    assert c.resume_unfinished() == ([], {})              # left once: never looked at again
    c = resumer(tmp_path, resume_unfinished=False)
    stopped_group(tmp_path, 4)
    assert c.resume_unfinished() == ([], {})              # opted out


def test_a_group_whose_episodes_all_finished_is_scored_at_once(tmp_path):
    d, entries = stopped_group(tmp_path, 2, k=2)
    for e in entries:
        finished(d, e)
    c = resumer(tmp_path)
    assert c.resume_unfinished()[0] == [2] and c.scored == [2] and not c.pending


def test_a_scored_group_resumes_its_stopped_teacher_attempts(tmp_path):
    d, entries = stopped_group(tmp_path, 3, k=2)
    for e in entries:
        finished(d, e)
    teacher = {**entries[0], 'key': 's7-task-000-t00', 'jolt_role': 'teacher',
               'output': str(d / 'row-000/sample-t00/results.jsonl')}
    with (d / 'queue.jsonl').open('a') as handle:
        handle.write(json.dumps(teacher) + '\n')
    (d / 'rewards.jsonl').write_text(json.dumps({'task_id': 't0'}) + '\n')
    c = resumer(tmp_path)
    assert c.resume_unfinished()[0] == [3]
    assert c.groups[3]['teachers'] == 1 and [e['key'] for _, e in c.pending] == ['s7-task-000-t00']


def frontier(work, order=('t0', 't1'), own=()):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
    import expert_iteration as ei
    cls = ei.frontier_collector(grpo.Collector)
    c = resumer(work, cls)
    c.by_id = {t: {'id': t} for t in ('t0', 't1', 't2')}
    c.own = {t: (1, 4) for t in own}
    c.order = collections.deque(order)
    return c


def test_expert_iteration_continues_a_partial_of_its_policy_and_keeps_pre_marker_finishes(tmp_path):
    d, entries = stopped_group(tmp_path, 0)
    finished(d, entries[0], marker=False)                 # a run before done markers: its finish row and captures
    killed(d, entries[1], turns=[capture_turn(), capture_turn()])
    killed(d, entries[2], turns=[capture_turn(policy='q' * 64)])     # a replaced session served another policy
    killed(d, entries[3], turns=[capture_turn(top_p=1.0)])           # another sampling law
    c = frontier(tmp_path)
    assert c.resume_unfinished()[0] == [0]
    record = json.loads((d / grpo.GROUP_RESUMED).read_text())
    assert record['kept'] == ['s7-task-000-00'] and record['continued'] == ['s7-task-000-01']
    assert record['rerun'] == ['s7-task-000-02', 's7-task-000-03']
    assert (d / 'row-000/sample-01/jobs/000000-abc.partial.json').exists()     # continued in place
    assert list(c.order) == ['t1']                        # its task is not started again as a new group


def test_expert_iteration_resumes_one_group_per_task_and_none_of_a_measured_task(tmp_path):
    a, ea = stopped_group(tmp_path, 0, task='t0')
    b, eb = stopped_group(tmp_path, 1, task='t0')
    finished(b, eb[0])
    m, _ = stopped_group(tmp_path, 2, task='t1')
    x, _ = stopped_group(tmp_path, 3, task='t9')          # no longer an eligible task
    c = frontier(tmp_path, own=('t1',))
    resumed, left = c.resume_unfinished()
    assert resumed == [1]                                 # the group with the most accepted episodes
    assert left == {'superseded: task measured in another group': 2, 'task not eligible': 1}
    assert json.loads((a / grpo.GROUP_LOST).read_text())['reason'].startswith('superseded')


def test_partial_continuation_checks_every_recorded_turn(tmp_path):
    from natlang_neuralese.rl import episode_files
    jobs = tmp_path / 'sample-00/jobs'
    jobs.mkdir(parents=True)
    entry = {'jobs': str(jobs)}
    assert episode_files.partial_continuable(entry, 'p' * 64, 0.95) == (False, 'no partial')
    partial = jobs / '000000-abc.partial.json'
    partial.write_text(json.dumps({'turns': [capture_turn(), capture_turn()]}) + ' ' * 5000)
    episode_files.compress_file(partial)                  # compacted by an earlier collect start: restored raw
    assert episode_files.partial_continuable(entry, 'p' * 64, 0.95) == (True, '2 recorded turns')
    assert partial.exists() and not (jobs / '000000-abc.partial.json.zst').exists()
    for turns, reason in (([capture_turn(policy='q' * 64)], 'a turn of another policy'),
                          ([capture_turn(top_p=0.9)], 'a turn of another sampling law'),
                          ([capture_turn(writes=[{'id': 'n1'}])], 'a turn wrote or read Neuralese state'),
                          ([{'request_sha256': 'r', 'response': {'raw_response': {}}}], 'a turn without its sampling capture')):
        partial.write_text(json.dumps({'turns': turns}))
        assert episode_files.partial_continuable(entry, 'p' * 64, 0.95) == (False, reason)
    partial.write_text(json.dumps({'turns': [capture_turn()]}))
    (jobs / '000000.error.json').write_text('{}')
    assert episode_files.partial_continuable(entry, 'p' * 64, 0.95) == (False, 'a job has ended')


def test_attempts_set_aside_by_label(tmp_path):
    from natlang_neuralese.rl import episode_files
    entry = {'key': 'k', 'output': str(tmp_path / 'row-000/sample-00/results.jsonl')}
    assert episode_files.set_attempt_aside(tmp_path, entry, 'stopped') == 0 and not list(tmp_path.glob('row-000/*'))
    (tmp_path / 'row-000/sample-00').mkdir(parents=True)
    (tmp_path / 'workers').mkdir()
    (tmp_path / 'workers/k.journal.jsonl').write_text('{"event": "finish"}\n{"half')
    assert episode_files.finish_of(tmp_path / 'workers/k.journal.jsonl') == {'event': 'finish'}
    assert episode_files.set_attempt_aside(tmp_path, entry, 'stopped') == 1
    assert (tmp_path / 'row-000/sample-00.stopped-1/worker.journal.jsonl').exists()
    assert episode_files.lost_attempts(entry) == 0        # a stop is never counted against the entry
    assert episode_files.set_attempt_aside(tmp_path, entry) == 1 and episode_files.lost_attempts(entry) == 1
