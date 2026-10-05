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
    state = {'claims': {'a.service': claim('experiment', 10, 0, 0), 'b.service': claim('experiment', 10, 0, 0)},
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
