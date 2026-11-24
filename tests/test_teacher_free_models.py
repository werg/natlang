import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import teacher_free_models as t  # noqa: E402


def test_first_listing_is_a_baseline_and_later_additions_are_open_candidates(tmp_path):
    catalogue = {'a': {'tools': True}}

    def failing():
        raise RuntimeError('down')
    providers = {'or': lambda: dict(catalogue), 'down': failing}
    first = t.check(tmp_path, providers, now='T1')
    assert first['new'] == [] and first['providers']['or']['baseline'] and 'error' in first['providers']['down']
    catalogue['b:free'] = {'tools': True, 'reasoning': True}
    second = t.check(tmp_path, providers, now='T2')
    assert second['new'] == ['or:b:free']
    assert [r['model'] for r in t.open_candidates(tmp_path)] == ['b:free']
    del catalogue['a']
    third = t.check(tmp_path, providers, now='T3')
    assert third['new'] == [] and third['providers']['or']['gone'] == ['a']
    t.mark('b:free', 'qualified', note='3/20', out=tmp_path)
    assert t.open_candidates(tmp_path) == []
    rows = [json.loads(line) for line in (tmp_path / 'candidates.jsonl').open()]
    assert rows[0]['status'] == 'qualified' and rows[0]['note'] == '3/20'


def test_openrouter_free_means_the_free_suffix_or_zero_prices():
    assert t.openrouter_free({'id': 'x/y:free', 'pricing': {'prompt': '0.1'}})
    assert t.openrouter_free({'id': 'x/y', 'pricing': {'prompt': '0', 'completion': '0'}})
    assert not t.openrouter_free({'id': 'x/y', 'pricing': {'prompt': '0', 'completion': '0.2'}})
