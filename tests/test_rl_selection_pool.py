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
    receipt = pool.assemble([str(source)], tmp_path / 'pool', share=.2, domains={}, family_share=.1)
    kept = [json.loads(line) for line in (tmp_path / 'pool' / 'cases.jsonl').read_text().splitlines()]
    by_generator = {}
    for row in kept:
        by_generator[row['generation']['generator']] = by_generator.get(row['generation']['generator'], 0) + 1
    assert max(by_generator.values()) <= .2 * len(kept) + 1
    # The single effect-checked template cannot dominate even though effect oracles are preferred.
    assert sum(row['family'] == 'template' for row in kept) <= receipt['family_cap_rows']
    assert receipt['cases']['rows'] == len(kept)
