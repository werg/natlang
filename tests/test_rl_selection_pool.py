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


def test_effect_state_share_keeps_every_candidate_and_tags_world_rows(tmp_path):
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
    # The share is not met by dropping candidates: the pool is exactly the unconstrained pool, effect rows tagged.
    assert receipt['cases']['rows'] == free['cases']['rows']
    tagged = {row['id'] for row in kept if row['curriculum']['rl'].get('mix_class') == 'effect_state'}
    assert tagged == {row['id'] for row in kept if 'effects' in row['semantics']} and len(tagged) == 40
    assert receipt['quota']['mode'] == 'selection-time' and receipt['quota']['class_quota'] == {'effect_state': .34}
    assert not any('mix_class' in row['curriculum']['rl'] for row in
                   map(json.loads, (tmp_path / 'free' / 'cases.jsonl').read_text().splitlines()))


def test_repeated_effect_template_rows_stay_in_the_pool_while_repeated_plain_rows_wait(tmp_path):
    rows = [case(f'eff-{i}', 'repair', 'g/code', body='one repair template', effect=True) for i in range(30)]
    rows += [case(f'plain-{i}', 'questions', 'g/code', body='one question template') for i in range(30)]
    rows += [case(f'other-{i}', f'o{i % 5}', 'g/other', body=f'other program {chr(97 + i)}{i}') for i in range(20)]
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    pool.assemble([str(source)], tmp_path / 'pool', share=.5, domains={}, family_share=1.0, min_effect_share=.3, protected=Stub())
    kept = [json.loads(line)['id'] for line in (tmp_path / 'pool' / 'cases.jsonl').read_text().splitlines()]
    assert sum(i.startswith('eff-') for i in kept) == 30
    assert sum(i.startswith('plain-') for i in kept) < 30


def test_selector_honours_the_class_quota_every_round_on_a_low_effect_pool(tmp_path):
    from natlang_neuralese.rl import production
    from natlang_neuralese.rl.curriculum import DEFAULT
    rows = []
    for name in 'abcd':
        rows += [case(f'{name}-plain-{i}', f'{name}-plain{i % 12}', 'g/' + name, body=f'{name} plain {chr(97 + i % 26)}{i}')
                 for i in range(200)]
        rows += [case(f'{name}-eff-{i}', f'{name}-eff{i % 12}', 'g/' + name, body=f'{name} effect {i} step', effect=True)
                 for i in range(10)]
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    pool.assemble([str(source)], tmp_path / 'quota', share=.3, domains={}, family_share=.2, min_effect_share=.3,
                  protected=Stub())
    cases = [json.loads(line) for line in (tmp_path / 'quota' / 'cases.jsonl').read_text().splitlines()]
    assert sum(c['curriculum']['rl'].get('mix_class') == 'effect_state' for c in cases) < .1 * len(cases)
    offset = 0
    for _ in range(12):
        chosen = production.select_tasks(cases, [], {'rows': 16, 'group': 8, 'offset': offset}, curriculum=DEFAULT)
        effect = sum(cases[item['row']]['curriculum']['rl'].get('mix_class') == 'effect_state' for item in chosen)
        assert effect >= .34 * 16 and len({item['row'] for item in chosen}) == 16
        offset = (chosen[-1]['row'] + 1) % len(cases)


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


def test_program_cap_rotation_changes_which_worlds_survive(tmp_path):
    rows = []
    for i in range(40):
        row = case(f'poor-{i}', 'poor_fam', 'g/poor', body='one template over many worlds')
        row['curriculum']['external'] = {'source': 'bgkit'}
        rows.append(row)
    rows += [case(f'rich-{i}', f'rich{i % 3}', 'g/rich', body=f'rich program {chr(97 + i)}x{i}') for i in range(20)]
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    kept = []
    for name, rotation in (('a', 'r1'), ('a2', 'r1'), ('b', 'r2')):
        pool.assemble([str(source)], tmp_path / name, share=.9, domains={}, family_share=1.0, protected=Stub(),
                      program_caps=[{'source': 'bgkit', 'max_rows_per_program': 8, 'rotation': rotation}])
        kept.append({json.loads(line)['id'] for line in (tmp_path / name / 'cases.jsonl').read_text().splitlines()
                     if json.loads(line)['id'].startswith('poor-')})
    assert len(kept[0]) == 8 and kept[0] == kept[1] and kept[0] != kept[2]


def test_returned_state_without_a_world_is_not_tagged_effect_state(tmp_path):
    rows = []
    for name in 'ab':
        rows += [case(f'{name}-eff-{i}', f'{name}-eff{i % 12}', 'g/' + name, body=f'{name} effect {i} step', effect=True)
                 for i in range(10)]
        for i in range(10):  # an author-declared "complete state" with no folder, service or effect check
            row = case(f'{name}-ret-{i}', f'{name}-ret{i % 12}', 'g/' + name, body=f'{name} returned {i} state')
            row['curriculum']['external'] = {'oracle_depth': 'effect-state'}
            rows.append(row)
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    receipt = pool.assemble([str(source)], tmp_path / 'strict', share=.9, domains={}, family_share=1.0, min_effect_share=.3,
                            protected=Stub())
    assert receipt['quota']['effect_rows'] == 20 and receipt['quota']['returned_state_rows'] == 20
    kept = [json.loads(line) for line in (tmp_path / 'strict' / 'cases.jsonl').read_text().splitlines()]
    assert sum(row['curriculum']['rl'].get('mix_class') == 'effect_state' for row in kept) == 20


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
