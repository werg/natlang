import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('memory_ledger', Path(__file__).resolve().parents[1] / 'scripts/memory_ledger.py')
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)
GIB = 2**30


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_machine_effects(monkeypatch):
    """Tests never touch real memory state: the balloon and systemd/docker calls fail loudly unless a test stubs them."""
    def refuse(*_args, **_kwargs):
        raise AssertionError('a ledger test reached the real machine (balloon_reclaim)')
    monkeypatch.setattr(ledger, 'balloon_reclaim', refuse)


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
    monkeypatch.setattr(ledger, 'container_cgroup', lambda command, unit=None: str(container_cg) if 'docker' in ' '.join(command) else None)
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
    # Never the real balloon: it touches up to 24 GB of anonymous memory (a ledgered suite was OOM-killed by it).
    monkeypatch.setattr(ledger, 'balloon_reclaim', lambda *a, **k: 0)
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


def test_cgroup_usage_leaves_out_reclaimable_page_cache_but_keeps_shared_memory(tmp_path):
    (tmp_path / 'memory.current').write_text(str(30 * GIB))
    (tmp_path / 'cgroup.procs').write_text('7\n')
    (tmp_path / 'memory.stat').write_text(f'anon {18 * GIB}\nfile {12 * GIB}\nshmem {2 * GIB}\n')
    assert ledger.cgroup_usage(str(tmp_path), {7: 5 * GIB}) == 30 * GIB - 10 * GIB + 5 * GIB


def test_a_guard_stop_waits_out_the_victims_grace_and_kills_only_in_an_emergency():
    pending = {'deadline': 1000.0, 'container': 'c'}
    emergency = 3 * GIB
    assert ledger.escalation(pending, 500.0, 20 * GIB, emergency, active=True) == 'wait'  # writing its checkpoint
    assert ledger.escalation(pending, 500.0, 5 * GIB, emergency, active=True) == 'wait'  # below the floor, not yet critical
    assert ledger.escalation(pending, 500.0, 2 * GIB, emergency, active=True) == 'kill'  # emergency: no grace
    assert ledger.escalation(pending, 1000.0, 20 * GIB, emergency, active=True) == 'kill'  # grace over
    assert ledger.escalation(pending, 500.0, 2 * GIB, emergency, active=False) == 'done'  # it exited


def test_the_guard_signals_one_victim_then_watches_it_without_stopping_another(monkeypatch, tmp_path):
    import json
    import types
    state = {'claims': {'big.service': {'budget': 8 * GIB, 'class': 'experiment', 'admitted': 1, 'command': [],
                                        'stop_seconds': 600, 'stop_seconds_explicit': True},
                        'other.service': {'budget': 8 * GIB, 'class': 'experiment', 'admitted': 2, 'command': []}},
             'events': []}
    path = tmp_path / 'ledger.json'
    path.write_text(json.dumps(state))
    monkeypatch.setattr(ledger, 'STATE', str(path))
    calls = []
    monkeypatch.setattr(ledger, 'checkpoint_writes', lambda path=None, limit=2000: [])
    monkeypatch.setattr(ledger, 'terminate', lambda unit, container: calls.append(('term', unit)))
    monkeypatch.setattr(ledger, 'kill', lambda unit, container: calls.append(('kill', unit)))
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    monkeypatch.setattr(ledger, 'reclaimable', lambda: 0)
    monkeypatch.setattr(ledger, 'mem_free', lambda: 100 * GIB)
    live = {'big.service': {'class': 'experiment', 'budget': 8 * GIB, 'used': 4 * GIB, 'admitted': 1},
            'other.service': {'class': 'experiment', 'budget': 8 * GIB, 'used': 4 * GIB, 'admitted': 2}}
    monkeypatch.setattr(ledger, 'live_claims', lambda state, gpu: dict(live))
    available = [5 * GIB]
    monkeypatch.setattr(ledger, 'mem_available', lambda: available[0])
    monkeypatch.setattr(ledger, 'unit_state', lambda unit: ('active', ''))
    passes = []

    def sleep(_seconds):
        passes.append(list(calls))
        if len(passes) == 1:
            available[0] = 5 * GIB  # still below the floor: the victim keeps its grace, no second victim
        elif len(passes) == 2:
            available[0] = 2 * GIB  # below the emergency level: kill it now
        else:
            raise StopIteration
    monkeypatch.setattr(ledger.time, 'sleep', sleep)
    args = types.SimpleNamespace(floor_gb=8, overshoot=1.15, interval=5, once=False, emergency_gb=3)
    try:
        ledger.guard(args)
    except StopIteration:
        pass
    assert passes[0] == [('term', 'other.service')]  # below the floor: the newest gets SIGTERM, not a kill
    assert passes[1] == [('term', 'other.service')]  # checkpointing within its grace: nothing new
    assert passes[2] == [('term', 'other.service'), ('kill', 'other.service')]
    events = json.loads(path.read_text())['events']
    assert [e['event'] for e in events] == ['stopped', 'killed after grace'] and events[1]['reason'] == 'emergency'


def test_stop_grace_scales_with_the_familys_measured_checkpoint():
    writes = [{'unit': 'natlang-mellum-qat-foundation-113800.service', 'bytes': 46_800_000_000, 'seconds': 250.0},
              {'unit': 'natlang-lfm-warmup-0900', 'bytes': 1_400_000_000, 'seconds': 7.0},
              {'unit': 'natlang-lfm-warmup-0901', 'bytes': 100 << 20, 'seconds': 5.0}]  # small: latency, not rate
    sizes = ledger.family_state_bytes(writes)
    assert sizes == {'natlang-mellum-qat-foundation': 46_800_000_000, 'natlang-lfm-warmup': 1_400_000_000}
    throughput = ledger.write_throughput(writes)
    assert throughput == 200e6  # the median of the large writes (187 and 200 MB/s)
    mellum = ledger.grace_seconds('natlang-mellum-qat-foundation-150000.service', {}, sizes, throughput)
    lfm = ledger.grace_seconds('natlang-lfm-warmup-1200.service', {}, sizes, throughput)
    probe = ledger.grace_seconds('pytest-quant-1200.service', {}, sizes, throughput)
    assert 450 < mellum < 500 and 20 < lfm < 40 and probe == ledger.GRACE_MIN
    explicit = {'stop_seconds': 600, 'stop_seconds_explicit': True}
    assert ledger.grace_seconds('natlang-lfm-warmup-1200.service', explicit, sizes, throughput) == 600
    assert ledger.write_throughput([]) == ledger.DEFAULT_THROUGHPUT


def test_under_the_floor_the_cheapest_stop_per_grace_goes_first_within_a_class():
    live = {'trainer.service': claim('experiment', 60, 50, 1), 'probe.service': claim('experiment', 8, 6, 2),
            'svc.service': claim('service', 40, 30, 3)}
    grace = {'trainer.service': 480, 'probe.service': 5, 'svc.service': 5}
    # the probe frees 6 GB in ~5 s; the trainer 50 GB over 8 min; the service class is protected regardless
    assert ledger.victim(live, True, 1.15, grace=grace) == ('probe.service', 'free memory below floor')
    del live['probe.service']
    assert ledger.victim(live, True, 1.15, grace=grace)[0] == 'trainer.service'
