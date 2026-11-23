"""Guard the real extraction/replay boundaries affected by combined held/memetic changes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import sft_sources as sources
from natlang_neuralese.rl import held_eval


def test_combined_held_closure_retains_every_registry_reservation(tmp_path, monkeypatch):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    monkeypatch.setattr(held_eval, 'held_manifests', lambda: [('held-eval', second)])
    monkeypatch.setattr(sources, '_held_closure_one', lambda path: (
        {Path(path).stem}, {f'group:{Path(path).stem}'}, {f'source:{Path(path).stem}'},
        [f'reservation:{Path(path).stem}']))
    closure = sources.held_closure(first)
    assert closure[0] == {'first', 'second'}
    assert closure.reservations == ('reservation:first', 'reservation:second')


def test_memetic_supervisor_cannot_admit_a_held_support_or_query_world(monkeypatch):
    monkeypatch.setattr(held_eval, 'training_exclusion_ids', lambda *_: set())
    tasks = {'supervisor': {'id': 'supervisor', 'curriculum': {'world_ids': ['held-query']}},
             'independent': {'id': 'independent', 'curriculum': {'world_ids': ['train-world']}}}
    assert sources.held_excluded_ids(None, tasks, ({'held-query'}, set(), set())) == {'supervisor'}
