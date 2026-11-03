import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'rl_tasks'))
import assemble_selection_pool as pool  # noqa: E402


def case(ident, family, generator, *, body=None, effect=False, level=1):
    semantics = {'root': 'f.nl', 'files': {'f.nl': '---\nargs: {}\nreturns: "string"\n---\n' + (body or ident) + '\n'},
                 'inputs': {}, 'expected': 'x', 'oracle': 'exact'}
    if effect:
        semantics['effects'] = {}
    return {'id': ident, 'split': 'train', 'family': family, 'generation': {'generator': generator},
            'curriculum': {'mode': 'single_call', 'rl': {'level': level, 'family': family}}, 'semantics': semantics}


def test_caps_water_fill_and_refuse_infeasible_share():
    assert pool.caps({'a': 100, 'b': 10, 'c': 10, 'd': 10, 'e': 10}, .2) == {'a': 10, 'b': 10, 'c': 10, 'd': 10, 'e': 10}
    try:
        pool.caps({'a': 5, 'b': 5}, .2)
    except SystemExit as error:
        assert 'cannot satisfy' in str(error)
    else:
        raise AssertionError('infeasible share accepted')


def test_pool_caps_generators_and_families_and_prefers_new_programs(tmp_path):
    rows = [case(f'big-{i}', 'template', 'g/big', body='same text', effect=True) for i in range(200)]
    rows += [case(f'big-other-{i}', f'fam{i % 10}', 'g/big', body=f'program {i} {"x" * (i % 7)} variant') for i in range(100)]
    for name in 'cdef':
        rows += [case(f'{name}-{i}', f'{name}-fam{i % 4}', 'g/' + name, body=f'{name} task {chr(97 + i % 26)}')
                 for i in range(30)]
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt = pool.assemble([str(source)], tmp_path / 'pool', share=.2, domains={}, family_share=.1, protected=Stub())
    kept = [json.loads(line) for line in (tmp_path / 'pool' / 'cases.jsonl').read_text().splitlines()]
    by_generator = {}
    for row in kept:
        by_generator[row['generation']['generator']] = by_generator.get(row['generation']['generator'], 0) + 1
    assert max(by_generator.values()) <= .2 * len(kept) + 1
    # The single effect-checked template cannot dominate even though effect oracles are preferred.
    assert sum(row['family'] == 'template' for row in kept) <= receipt['family_cap_rows']
    assert receipt['cases']['rows'] == len(kept)


def test_effect_state_quota_shrinks_pool_without_duplicating_rows(tmp_path):
    rows = []
    for name in 'abcd':
        rows += [case(f'{name}-plain-{i}', f'{name}-plain{i % 12}', 'g/' + name, body=f'{name} plain {chr(97 + i % 26)}{i}')
                 for i in range(200)]
        rows += [case(f'{name}-eff-{i}', f'{name}-eff{i % 12}', 'g/' + name, body=f'{name} effect {i} step', effect=True)
                 for i in range(10)]
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    free = pool.assemble([str(source)], tmp_path / 'free', share=.3, domains={}, family_share=.2, protected=Stub())
    receipt = pool.assemble([str(source)], tmp_path / 'quota', share=.3, domains={}, family_share=.2, min_effect_share=.3,
                            protected=Stub())
    kept = [json.loads(line) for line in (tmp_path / 'quota' / 'cases.jsonl').read_text().splitlines()]
    ids = [row['id'] for row in kept]
    assert len(ids) == len(set(ids))
    effects = sum('effects' in row['semantics'] for row in kept)
    assert effects == 40 and effects >= .3 * len(kept) - 1
    assert receipt['quota']['applied'] and receipt['cases']['rows'] < free['cases']['rows']
    assert not free['quota']['applied']


def test_program_caps_and_excluded_ids_trim_program_poor_sources_without_touching_rows(tmp_path):
    rows = []
    for i in range(30):
        row = case(f'poor-{i}', 'poor_fam', 'g/poor', body='one template over many worlds')
        row['curriculum']['external'] = {'source': 'bgkit'}
        rows.append(row)
    rows += [case(f'rich-{i}', f'rich{i % 3}', 'g/rich', body=f'rich program {chr(97 + i)}x{i}') for i in range(20)]
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt = pool.assemble([str(source)], tmp_path / 'capped', share=.9, domains={}, family_share=1.0, protected=Stub(),
                            exclude_ids=['poor-3', 'rich-0'], program_caps=[{'source': 'bgkit', 'max_rows_per_program': 8}])
    kept = {json.loads(line)['id'] for line in (tmp_path / 'capped' / 'cases.jsonl').read_text().splitlines()}
    assert sum(i.startswith('poor-') for i in kept) == 8 and 'poor-3' not in kept and 'rich-0' not in kept
    assert sum(i.startswith('rich-') for i in kept) == 19
    assert receipt['dropped_before_selection'] == {'excluded_ids': 2, 'bgkit': 21}


def test_returned_state_without_a_world_is_not_effect_state_and_min_rows_bounds_the_shrinking(tmp_path):
    rows = []
    for name in 'abcd':
        rows += [case(f'{name}-plain-{i}', f'{name}-plain{i % 12}', 'g/' + name, body=f'{name} plain {chr(97 + i % 26)}{i}')
                 for i in range(200)]
        rows += [case(f'{name}-eff-{i}', f'{name}-eff{i % 12}', 'g/' + name, body=f'{name} effect {i} step', effect=True)
                 for i in range(10)]
        for i in range(10):  # an author-declared "complete state" with no folder, service or effect check
            row = case(f'{name}-ret-{i}', f'{name}-ret{i % 12}', 'g/' + name, body=f'{name} returned {i} state')
            row['curriculum']['external'] = {'oracle_depth': 'effect-state'}
            rows.append(row)
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    strict = pool.assemble([str(source)], tmp_path / 'strict', share=.3, domains={}, family_share=.2, min_effect_share=.3,
                           protected=Stub())
    assert strict['quota']['effect_rows'] == 40 and strict['quota']['met']
    floor = pool.assemble([str(source)], tmp_path / 'floor', share=.3, domains={}, family_share=.2, min_effect_share=.3,
                          min_pool_rows=300, protected=Stub())
    assert floor['cases']['rows'] >= 300 > strict['cases']['rows']
    assert floor['quota']['effect_rows'] == 40 and not floor['quota']['met']
    assert floor['quota']['effect_share'] < .3


class Stub:
    digest = 'stub'

    def __init__(self, related=()):
        self.ids = set(related)

    def related(self, row):
        return ['group'] if row['id'] in self.ids else []


def test_train_rows_related_to_protected_worlds_are_never_selectable(tmp_path):
    rows = [case(f'r-{i}', f'fam{i % 5}', 'g/r', body=f'program {i} {"y" * (i % 9)}') for i in range(40)]
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt = pool.assemble([str(source)], tmp_path / 'guarded', share=1.0, domains={}, family_share=1.0,
                            protected=Stub({'r-3', 'r-4'}))
    kept = {json.loads(line)['id'] for line in (tmp_path / 'guarded' / 'cases.jsonl').read_text().splitlines()}
    assert not kept & {'r-3', 'r-4'} and receipt['dropped_before_selection']['protected_world_related'] == 2
    assert receipt['protected_worlds_digest'] == 'stub'
