import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('memory_ledger', Path(__file__).resolve().parents[1] / 'scripts/memory_ledger.py')
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)
GIB = 2**30


def claim(cls, budget, used, admitted):
    return {'class': cls, 'budget': budget * GIB, 'used': used * GIB, 'admitted': admitted}


def test_over_budget_unit_is_stopped_first_even_above_the_floor():
    live = {'a.service': claim('experiment', 8, 4, 1), 'b.service': claim('collection', 4, 6, 2)}
    assert ledger.victim(live, False, 1.15) == ('b.service', 'over budget')


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


def test_the_guard_starts_no_second_walk_while_one_runs_nor_one_without_cache_to_drop(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'mem_free', lambda: 4 * GIB)
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    reclaim = {'bytes': 5 * GIB}
    monkeypatch.setattr(ledger, 'reclaimable', lambda: reclaim['bytes'])
    monkeypatch.setattr(ledger, 'mem_available', lambda: 4 * GIB + reclaim['bytes'])
    clock = {'now': 1000.0}
    monkeypatch.setattr(ledger.time, 'time', lambda: clock['now'])
    started = []

    class Walk:
        def __init__(self, command, **_):
            started.append(command)
            self.done = False
        def poll(self):
            return 0 if self.done else None
    monkeypatch.setattr(ledger.subprocess, 'Popen', Walk)
    ticks = {'n': 0}

    def sleep(_):
        clock['now'] += 120
        ticks['n'] += 1
        if ticks['n'] == 3:
            reclaim['bytes'] = GIB  # nothing left worth dropping
        if ticks['n'] == 5:
            raise StopIteration
    monkeypatch.setattr(ledger.time, 'sleep', sleep)
    args = type('Args', (), {'floor_gb': 8, 'overshoot': 1.15, 'interval': 5, 'once': False})()
    try:
        ledger.guard(args)
    except StopIteration:
        pass
    assert len(started) == 1 and started[0][-1] == 'release-cache'
