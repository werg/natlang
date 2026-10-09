import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('memory_ledger', Path(__file__).resolve().parents[1] / 'scripts/memory_ledger.py')
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)
GIB = 2**30


def claim(cls, budget, used, admitted):
    return {'class': cls, 'budget': budget * GIB, 'used': used * GIB, 'admitted': admitted}


def test_over_budget_unit_is_stopped_first_only_under_memory_pressure():
    live = {'a.service': claim('experiment', 8, 4, 1), 'b.service': claim('collection', 4, 6, 2)}
    assert ledger.victim(live, False, 1.15, pressure=True) == ('b.service', 'over budget under memory pressure')
    assert ledger.victim(live, False, 1.15, pressure=False) == (None, None)  # plenty of memory: keep the work
    assert ledger.victim(live, True, 1.15) == ('b.service', 'over budget under memory pressure')


def test_admission_learns_each_job_familys_measured_peak():
    assert ledger.family('natlang-mellum-qat-test-210556.service') == 'natlang-mellum-qat-test'
    assert ledger.family('natlang-maple-reg-c224705.service') == 'natlang-maple-reg'
    state = {'family_peaks': {'natlang-mellum-qat-test': 10 * GIB}}
    assert ledger.learned_budget(state, 'natlang-mellum-qat-test-999999.service', 4 * GIB) == 11 * GIB
    assert ledger.learned_budget(state, 'natlang-other.service', 4 * GIB) == 4 * GIB


def test_floor_breach_stops_the_lowest_priority_newest_unit():
    live = {'old.service': claim('experiment', 8, 4, 1), 'new.service': claim('experiment', 8, 4, 5),
            'svc.service': claim('service', 8, 4, 9)}
    assert ledger.victim(live, True, 1.15) == ('new.service', 'free memory below floor')
    assert ledger.victim(live, False, 1.15) == (None, None)


def test_outstanding_demand_is_unspent_budget(monkeypatch):
    monkeypatch.setattr(ledger, 'mem_available', lambda: 40 * GIB)
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu, command=None: {'a.service': 2 * GIB, 'b.service': 12 * GIB}[unit])
    now = ledger.time.time()
    state = {'claims': {'a.service': claim('experiment', 10, 0, now), 'b.service': claim('experiment', 10, 0, now)},
             'events': []}
    free, live = ledger.headroom(state, {})
    assert free == 32 * GIB  # a still ramping counts 8 GiB more; b is past its budget and counts nothing
    assert live['b.service']['used'] == 12 * GIB


def test_claims_of_finished_units_are_released(monkeypatch):
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu, command=None: None)
    state = {'claims': {'done.service': claim('experiment', 4, 0, 0)}, 'events': []}
    assert ledger.live_claims(state, {}) == {}
    assert state['claims'] == {} and state['events'][-1]['event'] == 'released'


def test_attached_container_is_charged_to_its_unit(monkeypatch, tmp_path):
    # A `docker start -a NAME` unit holds only the client; the container's cgroup and its CUDA processes count too.
    unit_cg, container_cg = tmp_path / 'unit', tmp_path / 'container'
    for path, current, procs in ((unit_cg, GIB, '10\n'), (container_cg, 6 * GIB, '20\n21\n')):
        path.mkdir()
        (path / 'memory.current').write_text(str(current))
        (path / 'cgroup.procs').write_text(procs)
    monkeypatch.setattr(ledger, 'unit_state', lambda unit: ('active', str(unit_cg).removeprefix('/sys/fs/cgroup')))
    monkeypatch.setattr(ledger, 'container_cgroup', lambda command: str(container_cg) if 'docker' in ' '.join(command) else None)
    real = ledger.cgroup_usage  # the unit's cgroup lives under /sys/fs/cgroup; point it at the fixture
    monkeypatch.setattr(ledger, 'cgroup_usage', lambda root, gpu: real(str(unit_cg) if root.endswith('/unit') else root, gpu))
    gpu = {21: 50 * GIB}
    assert ledger.unit_usage('t.service', gpu, ['docker', 'start', '-a', 'srv']) == GIB + 6 * GIB + 50 * GIB
    assert ledger.unit_usage('t.service', gpu, ['(adopted)']) == GIB


def test_a_settled_claim_counts_up_to_its_measured_peak(monkeypatch):
    # The teacher has run for hours at 56 of its 62 GiB: it is not about to take the other 6.
    monkeypatch.setattr(ledger, 'mem_available', lambda: 30 * GIB)
    use = {'teacher.service': 56 * GIB, 'young.service': 2 * GIB, 'unmeasured.service': 20 * GIB}
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu, command=None: use[unit])
    now = ledger.time.time()
    state = {'claims': {'teacher.service': dict(claim('service', 62, 0, now - 7200), peak=57 * GIB, peak_since=now - 3600),
                        'unmeasured.service': claim('service', 30, 0, now - 7200),
                        'young.service': claim('experiment', 10, 0, now - 60)}, 'events': []}
    free, live = ledger.headroom(state, {})
    # 1 GiB up to the teacher's peak; the young job's whole rest; a claim only now measured counts its whole rest too
    assert free == 30 * GIB - 1 * GIB - 8 * GIB - 10 * GIB
    assert state['claims']['teacher.service']['peak'] == 57 * GIB and state['claims']['young.service']['peak'] == 2 * GIB


def test_one_cache_release_at_a_time(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    holder = open(str(tmp_path / 'ledger.json.release.lock'), 'w')
    ledger.fcntl.flock(holder, ledger.fcntl.LOCK_EX)
    assert ledger.release_cache([str(tmp_path)]) == {'skipped': 'another cache release is running'}
    holder.close()
    assert ledger.release_cache([str(tmp_path)])['files'] == 0


def test_the_guard_runs_one_walk_at_a_time_backs_off_after_a_poor_one_and_skips_without_cache_to_drop(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'mem_free', lambda: 4 * GIB)
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    reclaim = {'bytes': 5 * GIB}
    monkeypatch.setattr(ledger, 'reclaimable', lambda: reclaim['bytes'])
    monkeypatch.setattr(ledger, 'mem_available', lambda: 4 * GIB + reclaim['bytes'])
    clock = {'now': 1000.0}
    monkeypatch.setattr(ledger.time, 'time', lambda: clock['now'])
    started = []

    walks = []

    class Walk:
        def __init__(self, command, **_):
            started.append(command)
            walks.append(self)
            self.done = False
        def poll(self):
            return 0 if self.done else None
        def communicate(self):
            return '{"files": 3, "free_before_gb": 4.0, "free_after_gb": 4.2}', None
    monkeypatch.setattr(ledger.subprocess, 'Popen', Walk)
    ticks = {'n': 0}

    def sleep(_):
        clock['now'] += 120
        ticks['n'] += 1
        if ticks['n'] == 3:
            walks[0].done = True  # it freed 0.2 GiB: the next walk waits half an hour
        if ticks['n'] == 10:
            reclaim['bytes'] = GIB  # nothing left worth dropping
        if ticks['n'] == 30:
            raise StopIteration
    monkeypatch.setattr(ledger.time, 'sleep', sleep)
    args = type('Args', (), {'floor_gb': 8, 'overshoot': 1.15, 'interval': 5, 'once': False})()
    try:
        ledger.guard(args)
    except StopIteration:
        pass
    # one walk while it ran; then a pause of 1800 s (15 ticks of 120 s); by then nothing was left worth dropping
    assert len(started) == 1 and started[0][-1] == 'release-cache'


def test_the_guard_walks_again_after_a_pause_when_the_last_walk_freed_little(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'mem_free', lambda: 4 * GIB)
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    monkeypatch.setattr(ledger, 'reclaimable', lambda: 5 * GIB)
    monkeypatch.setattr(ledger, 'mem_available', lambda: 9 * GIB)
    clock = {'now': 1000.0}
    monkeypatch.setattr(ledger.time, 'time', lambda: clock['now'])
    started = []

    class Walk:
        def __init__(self, command, **_):
            started.append(clock['now'])
        def poll(self):
            return 0
        def communicate(self):
            return '{"free_before_gb": 4.0, "free_after_gb": 4.1}', None
    monkeypatch.setattr(ledger.subprocess, 'Popen', Walk)

    def sleep(_):
        clock['now'] += 300
        if clock['now'] > 1000 + 4000:
            raise StopIteration
    monkeypatch.setattr(ledger.time, 'sleep', sleep)
    args = type('Args', (), {'floor_gb': 8, 'overshoot': 1.15, 'interval': 5, 'once': False})()
    try:
        ledger.guard(args)
    except StopIteration:
        pass
    assert [round(t - started[0]) for t in started] == [0, 2100]  # done at the next check, then 1800 s


def test_a_held_budget_counts_in_full_however_long_the_job_has_run(monkeypatch):
    # The warm-up grows to more sequence passes on its own schedule and checks MemFree before each update.
    monkeypatch.setattr(ledger, 'mem_available', lambda: 30 * GIB)
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu, command=None: 38 * GIB)
    now = ledger.time.time()
    state = {'claims': {'warmup.service': dict(claim('experiment', 42, 0, now - 7200), peak=38 * GIB,
                                               peak_since=now - 7200, hold_budget=True)}, 'events': []}
    free, _ = ledger.headroom(state, {})
    assert free == 26 * GIB


def test_re_adoption_keeps_the_admission_time_and_peak_and_can_hold_the_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    procs = tmp_path / 'cg'
    procs.mkdir()
    (procs / 'cgroup.procs').write_text('')
    monkeypatch.setattr(ledger, 'unit_state', lambda unit: ('active', '/../../..' + str(procs)))
    with ledger.ledger() as state:
        state['claims']['warmup.service'] = dict(claim('experiment', 40, 0, 123.0), command=['docker', 'start', '-a', 'w'],
                                                 peak=38 * GIB, peak_since=200.0)
    args = type('Args', (), {'unit': 'warmup', 'budget_gb': 42, 'cls': 'experiment', 'hold_budget': True})()
    ledger.adopt(args)
    with ledger.ledger() as state:
        held = state['claims']['warmup.service']
    assert (held['admitted'], held['budget'], held['peak'], held['peak_since'], held['hold_budget'], held['command']) == \
        (123.0, 42 * GIB, 38 * GIB, 200.0, True, ['docker', 'start', '-a', 'w'])


def test_container_names_come_from_attached_and_named_runs():
    assert ledger.container_name(['docker', 'start', '-a', 'natlang-x']) == 'natlang-x'
    assert ledger.container_name('/usr/bin/docker run --rm --name pi-executor --gpus all img') == 'pi-executor'
    assert ledger.container_name(['python3', 'train.py']) is None
