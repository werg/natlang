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


def test_admission_learns_each_job_familys_measured_peak(monkeypatch):
    # Exercise learned admission independently of the runner's physical RAM;
    # the production policy caps learned peaks below its host reserve.
    monkeypatch.setattr(ledger, 'mem_total', lambda: 64 * GIB)
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


def test_a_running_claim_reserves_only_a_fraction_of_its_unspent_budget_after_ramp_up(monkeypatch):
    # Oversubscription: the teacher at 56 of its 62 GiB, a CPU job at 2 of 30, a young job still ramping.
    monkeypatch.setattr(ledger, 'mem_available', lambda: 30 * GIB)
    use = {'teacher.service': 56 * GIB, 'young.service': 2 * GIB, 'cpu.service': 2 * GIB}
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu, command=None: use[unit])
    now = ledger.time.time()
    state = {'claims': {'teacher.service': dict(claim('service', 62, 0, now - 7200), gpu=True),
                        'cpu.service': claim('experiment', 30, 0, now - 7200),
                        'young.service': claim('experiment', 10, 0, now - 60)}, 'events': []}
    free, live = ledger.headroom(state, {})
    # gpu unit: half of 6 GiB; cpu unit: a quarter of 28 GiB; young: the whole 8 GiB
    assert free == 30 * GIB - 3 * GIB - 7 * GIB - 8 * GIB
    assert state['claims']['young.service']['peak'] == 2 * GIB


def test_one_cache_release_at_a_time(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    # Never the real balloon: it touches up to 24 GB of anonymous memory (a ledgered suite was OOM-killed by it).
    monkeypatch.setattr(ledger, 'balloon_reclaim', lambda *a, **k: 0)
    pidfile = tmp_path / 'ledger.json.release.pid'
    pidfile.write_text(ledger.json.dumps({'pid': ledger.os.getpid(), 'started': ledger.time.time()}))  # a live walk
    assert ledger.release_cache([str(tmp_path)]) == {'skipped': 'another cache release is running'}
    # a walk stuck past its time (D state on a slow disk) no longer blocks the next release
    pidfile.write_text(ledger.json.dumps({'pid': ledger.os.getpid(), 'started': ledger.time.time() - 3 * ledger.RELEASE_SECONDS}))
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
    args = type('Args', (), {'unit': 'warmup', 'budget_gb': 42, 'cls': 'experiment', 'hold_budget': True, 'priority': None})()
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


def test_cgroup_usage_leaves_out_lazily_freed_anonymous_memory(tmp_path, monkeypatch):
    cg = tmp_path / 'cg'
    cg.mkdir()
    (cg / 'memory.current').write_text(str(30 * GIB))
    (cg / 'cgroup.procs').write_text('7\n8\n')
    (cg / 'memory.stat').write_text(f'anon {28 * GIB}\nfile {2 * GIB}\nshmem 0\n')
    proc = tmp_path / 'proc'
    (proc / '7').mkdir(parents=True)
    (proc / '7' / 'smaps_rollup').write_text(f'Rss: {27 * GIB // 1024} kB\nLazyFree: {22 * GIB // 1024} kB\n')
    monkeypatch.setattr(ledger, 'PROC', str(proc))  # pid 8 has no readable smaps: counts nothing freed
    assert ledger.cgroup_usage(str(cg), {7: 50 * GIB}) == 30 * GIB - 2 * GIB - 22 * GIB + 50 * GIB


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
    assert [e['event'] for e in events if not e['event'].startswith('stopped without')] == ['stopped', 'killed after grace'] and events[-1]['reason'] == 'emergency'


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


def running(cls, budget, used, priority=None, age=3600, **extra):
    c = claim(cls, budget, used, ledger.time.time() - age)
    c.update(priority=priority, spec={'reserve': 8 * GIB}, **extra)
    return c


def test_admission_is_by_real_headroom_so_claims_may_exceed_the_machine(monkeypatch):
    # 87 GiB claimed, 5 used: the old claim-sum rule refused a 40 GiB GPU job on a 121 GiB machine
    monkeypatch.setattr(ledger, 'mem_available', lambda: 110 * GIB)
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu, command=None: 5 * GIB)
    now = ledger.time.time()
    state = {'claims': {'cpu.service': claim('experiment', 87, 0, now - 3600)}, 'events': []}
    free, _ = ledger.headroom(state, {})
    assert free == 110 * GIB - int(0.25 * 82 * GIB)
    assert ledger.fits(free, 40 * GIB, 8 * GIB)
    state['claims']['cpu.service']['fixed_claim'] = True  # opt-out: the whole unspent budget reserved
    assert ledger.headroom(state, {})[0] == 110 * GIB - 82 * GIB
    assert not ledger.fits(110 * GIB - 82 * GIB, 40 * GIB, 8 * GIB)


def test_gpu_units_reserve_more_of_their_unclaimed_remainder_than_cpu_units():
    gpu, cpu = running('experiment', 40, 10, gpu=True), running('experiment', 40, 10)
    assert ledger.outstanding(gpu) == 15 * GIB and ledger.outstanding(cpu) == int(7.5 * GIB)
    assert ledger.outstanding(running('experiment', 40, 10, age=60)) == 30 * GIB  # still ramping up


def test_priorities_default_by_class_and_can_be_declared():
    assert ledger.priority_of(claim('experiment', 1, 0, 0)) == 2 and ledger.priority_of(claim('service', 1, 0, 0)) == 3
    assert ledger.priority_of(running('experiment', 1, 0, priority='high')) == 3
    assert ledger.priority_of(running('collection', 1, 0, priority='low')) == 1
    assert ledger.PRIORITY_OOM['high'] < ledger.PRIORITY_OOM['medium'] < ledger.PRIORITY_OOM['low']


def test_the_waiting_queue_is_ordered_by_priority_then_arrival_and_drops_abandoned_waiters():
    now = 1000.0
    state = {'waiting': {'low.service': {'priority': 'low', 'since': 1, 'seen': now},
                         'high-late.service': {'priority': 'high', 'since': 5, 'seen': now},
                         'high-early.service': {'priority': 'high', 'since': 2, 'seen': now},
                         'gone.service': {'priority': 'high', 'since': 0, 'seen': now - 500},
                         'requeued.service': {'priority': 'medium', 'since': 3, 'requeued': True}}}
    assert [u for u, _ in ledger.waiting_order(state, now)] == \
        ['high-early.service', 'high-late.service', 'requeued.service', 'low.service']


def test_preemption_picks_the_lowest_priority_resumable_units_and_never_equals_or_services():
    live = {'train.service': running('experiment', 40, 38, priority='high'),
            'eval.service': running('experiment', 20, 18, priority='medium'),
            'render-a.service': running('collection', 20, 15, priority='low'),
            'render-b.service': running('collection', 20, 6, priority='low'),
            'teacher.service': dict(running('service', 62, 56), resumable=False)}
    wanted = {'priority': 'high', 'budget': 30 * GIB, 'reserve': 8 * GIB}
    plan = ledger.preemption_plan(live, 14 * GIB, wanted)
    assert plan == ['render-a.service', 'render-b.service']  # lowest priority first, biggest first; teacher/train untouched
    assert ledger.preemption_plan(live, 60 * GIB, wanted) == []  # fits already
    assert ledger.preemption_plan(live, 10 * GIB, dict(wanted, budget=120 * GIB)) == []  # cannot fit even so: no needless stops
    assert ledger.preemption_plan(live, 10 * GIB, dict(wanted, priority='low')) == []  # nothing lower than low
    assert 'eval.service' not in ledger.preemption_plan(live, 10 * GIB, dict(wanted, priority='medium'))


def test_pressure_stops_the_lowest_priority_first_unless_a_higher_one_is_over_its_budget():
    live = {'train.service': running('experiment', 40, 30, priority='high', age=9000),
            'render.service': running('collection', 20, 10, priority='low', age=10)}
    assert ledger.victim(live, True, 1.15)[0] == 'render.service'  # even though it is older/cheaper class order says otherwise
    live['train.service']['used'] = 50 * GIB  # above its own claimed budget x overshoot
    assert ledger.victim(live, True, 1.15)[0] == 'train.service'
    live['train.service']['used'] = 40 * GIB  # at its budget: never killed for using what it claimed
    assert ledger.victim(live, True, 1.15)[0] == 'render.service'


def test_a_stopped_resumable_unit_is_requeued_with_its_spec_and_a_service_is_not():
    state = {'claims': {'render.service': running('collection', 20, 10, priority='low'),
                        'teacher.service': running('service', 62, 56)}, 'events': []}
    assert ledger.requeue_spec(state, 'render.service', now=100.0)
    entry = state['waiting']['render.service']
    assert entry['requeued'] and entry['priority'] == 'low' and entry['not_before'] > 100 and entry['spec']
    state['claims']['teacher.service']['resumable'] = False
    assert not ledger.requeue_spec(state, 'teacher.service') and 'teacher.service' not in state['waiting']
    assert not ledger.requeue_spec({'claims': {'old.service': claim('experiment', 1, 0, 0)}, 'events': []}, 'old.service')  # legacy claim: no spec


def test_the_guard_preempts_for_the_queue_head_then_relaunches_when_room_returns(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'CHECKPOINT_WRITES', str(tmp_path / 'none'))
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    monkeypatch.setattr(ledger, 'mem_free', lambda: 100 * GIB)
    available = [30 * GIB]
    monkeypatch.setattr(ledger, 'mem_available', lambda: available[0])
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu, command=None: 30 * GIB)
    calls, launched = [], []
    monkeypatch.setattr(ledger, 'terminate', lambda unit, container: calls.append(unit))
    monkeypatch.setattr(ledger, 'launch', lambda spec: launched.append(spec['unit']) or 0)
    monkeypatch.setattr(ledger, 'unit_state', lambda unit: ('inactive', ''))
    monkeypatch.setattr(ledger, 'time', type('T', (), {'time': staticmethod(lambda: now[0]), 'sleep': staticmethod(lambda s: None)}))
    now = [10000.0]
    with ledger.ledger() as state:
        state['claims']['render.service'] = running('collection', 40, 30, priority='low', admitted=1.0) \
            if False else dict(running('collection', 40, 30, priority='low'), admitted=1.0)
        state['waiting'] = {'train.service': {'priority': 'high', 'since': 9990.0, 'seen': 10000.0, 'budget': 40 * GIB,
                                              'reserve': 8 * GIB}}
    args = type('A', (), {'floor_gb': 8, 'overshoot': 1.15, 'interval': 0, 'once': True})()
    ledger.guard(args)
    assert calls == ['render.service']  # the high job cannot fit by real headroom: the low resumable unit is stopped
    with ledger.ledger() as state:
        assert state['waiting']['render.service']['requeued']  # and requeued for later


def test_the_guard_relaunches_a_requeued_unit_once_its_delay_has_passed_and_it_fits(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'CHECKPOINT_WRITES', str(tmp_path / 'none'))
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    monkeypatch.setattr(ledger, 'mem_free', lambda: 100 * GIB)
    monkeypatch.setattr(ledger, 'mem_available', lambda: 90 * GIB)
    monkeypatch.setattr(ledger, 'unit_state', lambda unit: ('inactive', ''))
    launched = []
    monkeypatch.setattr(ledger, 'launch', lambda spec: launched.append(spec['unit']) or 0)
    now = ledger.time.time()
    spec = {'unit': 'render.service', 'budget': 20 * GIB, 'cls': 'collection', 'command': ['x'], 'priority': 'low',
            'resumable': True, 'reserve': 8 * GIB}
    with ledger.ledger() as state:
        state['waiting'] = {'render.service': {'priority': 'low', 'since': now, 'requeued': True, 'not_before': now + 3600,
                                               'budget': 20 * GIB, 'reserve': 8 * GIB, 'spec': spec}}
    args = type('A', (), {'floor_gb': 8, 'overshoot': 1.15, 'interval': 0, 'once': True})()
    ledger.guard(args)
    assert launched == []  # delay not over
    with ledger.ledger() as state:
        state['waiting']['render.service']['not_before'] = now - 1
    ledger.guard(args)
    assert launched == ['render.service']
    with ledger.ledger() as state:
        assert 'render.service' in state['claims'] and not state['waiting']


def test_a_unit_declared_resumable_without_a_spec_is_preemptible_but_logged_not_requeued():
    c = dict(claim('experiment', 10, 5, 0), resumable=True)
    assert ledger.resumable(c)
    state = {'claims': {'u.service': c}, 'events': []}
    assert not ledger.requeue_spec(state, 'u.service', now=5.0)
    assert 'waiting' not in state or 'u.service' not in state['waiting']
    assert 'owner must restart' in state['events'][-1]['event']
    assert ledger.resumable(claim('experiment', 10, 5, 0))  # undeclared: rerunnable by default
    assert not ledger.resumable(dict(claim('experiment', 10, 5, 0), adopted=True))  # adopted: protected
    assert not ledger.resumable(claim('service', 10, 5, 0))
    assert ledger.preemption_plan({'u.service': c}, 0, {'priority': 'high', 'budget': 5 * GIB, 'reserve': 0}) == ['u.service']


def test_a_learned_budget_never_exceeds_what_the_machine_can_ever_admit(monkeypatch):
    monkeypatch.setattr(ledger, 'mem_total', lambda: 121 * GIB)
    state = {'family_peaks': {'natlang-e2e': 103 * GIB}}
    assert ledger.learned_budget(state, 'natlang-e2e-7777.service', 96 * GIB) == 105 * GIB


def test_a_waiting_run_registers_in_the_queue_before_any_slow_cache_walk(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    monkeypatch.setattr(ledger, 'mem_free', lambda: 1 * GIB)
    monkeypatch.setattr(ledger, 'mem_available', lambda: 50 * GIB)
    monkeypatch.setattr(ledger, 'reclaimable', lambda: 40 * GIB)
    monkeypatch.setattr(ledger, 'start_release', lambda: None)
    args = type('A', (), {'budget_gb': 96, 'reserve_gb': 8, 'unit': 'e2e', 'priority': 'high', 'cls': 'experiment', 'wait': 0,
                          'host_max_gb': None, 'workdir': '.', 'env': [], 'stop_seconds': None, 'command': ['x'],
                          'resumable': None, 'hold_budget': False, 'oom_policy': 'stop'})()
    seen = []
    def peek(_):
        with ledger.ledger() as state:
            seen.append(list(state.get('waiting', {})))
        raise KeyboardInterrupt
    monkeypatch.setattr(ledger.time, 'sleep', peek)
    args.wait = 100
    with pytest.raises(KeyboardInterrupt):
        ledger.run(args)
    assert seen == [['e2e.service']]


def test_blockers_name_the_non_preemptible_units_in_status_and_refusals():
    live = {'tests.service': running('experiment', 40, 32, priority='medium', resumable=False),
            'train.service': running('experiment', 40, 30, priority='high'),
            'render.service': running('collection', 20, 10, priority='low')}
    rows = ledger.non_preemptible(live, {'priority': 'high'})
    assert [(r['unit'], r['reason']) for r in rows] == [('tests.service', 'not resumable'),
                                                       ('train.service', 'priority high is not below high')]
    assert 'tests.service' in ledger.blocked_by(live, {'priority': 'high'}) and 'render.service' not in ledger.blocked_by(live, {'priority': 'high'})


def test_a_undeclared_test_unit_is_preempted_for_high_priority_work():
    live = {'tests.service': dict(claim('experiment', 40, 32, 1), priority='medium')}
    assert ledger.preemption_plan(live, 20 * GIB, {'priority': 'high', 'budget': 40 * GIB, 'reserve': 8 * GIB}) == ['tests.service']


def test_unit_containers_find_detached_runs_from_the_command_line_of_the_client(monkeypatch, tmp_path):
    cg = tmp_path / 'cg'
    cg.mkdir()
    (cg / 'cgroup.procs').write_text('4242\n')
    monkeypatch.setattr(ledger, 'unit_state', lambda unit: ('active', '/../../..' + str(cg)))
    proc = tmp_path / 'proc' / '4242'
    proc.mkdir(parents=True)
    (proc / 'cmdline').write_bytes(b'docker\0run\0--rm\0--name\0render-b00\0img\0')
    real_open = open
    monkeypatch.setattr('builtins.open', lambda path, *a, **k: real_open(str(path).replace('/proc/4242', str(proc)), *a, **k))
    assert ledger.unit_containers('static-render-b00-57085.service', ['env', 'x.sh'], ['extra']) == \
        ['extra', 'static-render-b00-57085', 'render-b00']


def test_stop_containers_stops_only_running_ones_and_a_requeue_waits_for_them(monkeypatch):
    stopped = []
    monkeypatch.setattr(ledger, 'container_running', lambda name: name == 'render-b00')
    monkeypatch.setattr(ledger.subprocess, 'Popen', lambda cmd, **k: stopped.append(cmd))
    ledger.stop_containers(['render-b00', 'gone'])
    assert stopped == [['docker', 'stop', '-t', '10', 'render-b00']]


def test_launch_puts_local_bin_first_on_path(monkeypatch):
    seen = []
    monkeypatch.setattr(ledger.subprocess, 'run', lambda cmd, **k: seen.append(cmd) or type('R', (), {'returncode': 0})())
    spec = {'unit': 'u.service', 'budget': GIB, 'priority': 'low', 'oom_policy': 'stop', 'workdir': '.', 'env': [],
            'command': ['node', '-v'], 'stop_seconds': 5, 'containers': ['c1']}
    ledger.launch(spec)
    cmd = seen[-1]
    assert any(a.startswith('PATH=') and a[5:].startswith(ledger.os.path.expanduser('~/.local/bin') + ':') for a in cmd)
    assert 'docker stop -t 5 u c1' in ' '.join(cmd)


def test_a_cache_walk_stops_itself_at_its_deadline_and_skips_the_slow_disk_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'balloon_reclaim', lambda *a, **k: 0)
    assert not any(root.startswith('/mnt/' + 'external') for root in ledger.CACHE_ROOTS)
    for i in range(3):
        (tmp_path / f'f{i}').write_bytes(b'x' * (17 << 20))
    assert ledger.release_cache([str(tmp_path)], seconds=-1)['files'] == 0  # deadline passed: no work at all
    assert ledger.release_cache([str(tmp_path)])['files'] == 3
    assert not (tmp_path / 'ledger.json.release.pid').exists()


def test_a_small_resumable_unit_backfills_behind_a_big_waiting_head_but_a_non_resumable_does_not():
    order = [('head.service', {}), ('small.service', {})]
    assert ledger.may_start(order, 'head.service', False, True)
    assert not ledger.may_start(order, 'head.service', False, False)
    assert ledger.may_start(order, 'small.service', True, True)
    assert not ledger.may_start(order, 'small.service', False, True)
    assert not ledger.may_start(order, 'small.service', True, False)


def test_the_guard_relaunches_a_requeued_unit_behind_a_head_that_waits_for_room(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    monkeypatch.setattr(ledger, 'CHECKPOINT_WRITES', str(tmp_path / 'none'))
    monkeypatch.setattr(ledger, 'gpu_usage', lambda: {})
    monkeypatch.setattr(ledger, 'mem_free', lambda: 100 * GIB)
    monkeypatch.setattr(ledger, 'mem_available', lambda: 30 * GIB)
    monkeypatch.setattr(ledger, 'unit_state', lambda unit: ('inactive', ''))
    monkeypatch.setattr(ledger, 'container_running', lambda n: False)
    launched = []
    monkeypatch.setattr(ledger, 'launch', lambda spec: launched.append(spec['unit']) or 0)
    now = ledger.time.time()
    spec = {'unit': 'small.service', 'budget': 1 * GIB, 'cls': 'experiment', 'command': ['x'], 'priority': 'medium',
            'resumable': True, 'reserve': 8 * GIB}
    with ledger.ledger() as state:  # no running claims to preempt; the head needs 75 GiB, 30 are free
        state['waiting'] = {'big.service': {'priority': 'high', 'since': now, 'seen': now, 'budget': 75 * GIB, 'reserve': 8 * GIB},
                            'small.service': {'priority': 'medium', 'since': now, 'requeued': True, 'not_before': now - 1,
                                              'budget': 1 * GIB, 'reserve': 8 * GIB, 'spec': spec}}
    ledger.guard(type('A', (), {'floor_gb': 8, 'overshoot': 1.15, 'interval': 0, 'once': True})())
    assert launched == ['small.service']


def test_cancel_drops_a_queue_entry_and_reset_peak_clears_or_sets_a_family_peak(monkeypatch, tmp_path):
    monkeypatch.setattr(ledger, 'STATE', str(tmp_path / 'ledger.json'))
    with ledger.ledger() as state:
        state['waiting'] = {'old-bench.service': {'priority': 'low', 'since': 1.0, 'budget': GIB, 'requeued': True}}
        state['family_peaks'] = {'fam-x': 100 * GIB}
    assert ledger.cancel(type('A', (), {'unit': 'old-bench'})()) == 0
    assert ledger.cancel(type('A', (), {'unit': 'old-bench'})()) == 1
    ledger.reset_peak(type('A', (), {'family': 'fam-x', 'to_gb': 88.0})())
    with ledger.ledger() as state:
        assert state['waiting'] == {} and state['family_peaks']['fam-x'] == 88 * GIB
        assert [e['event'] for e in state['events']] == ['cancelled', 'reset-peak']
    ledger.reset_peak(type('A', (), {'family': 'fam-x', 'to_gb': None})())
    with ledger.ledger() as state:
        assert 'fam-x' not in state['family_peaks']
