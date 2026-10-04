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
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu: {'a.service': 2 * GIB, 'b.service': 12 * GIB}[unit])
    state = {'claims': {'a.service': claim('experiment', 10, 0, 0), 'b.service': claim('experiment', 10, 0, 0)},
             'events': []}
    free, live = ledger.headroom(state, {})
    assert free == 32 * GIB  # a still ramping counts 8 GiB more; b is past its budget and counts nothing
    assert live['b.service']['used'] == 12 * GIB


def test_claims_of_finished_units_are_released(monkeypatch):
    monkeypatch.setattr(ledger, 'unit_usage', lambda unit, gpu: None)
    state = {'claims': {'done.service': claim('experiment', 4, 0, 0)}, 'events': []}
    assert ledger.live_claims(state, {}) == {}
    assert state['claims'] == {} and state['events'][-1]['event'] == 'released'
