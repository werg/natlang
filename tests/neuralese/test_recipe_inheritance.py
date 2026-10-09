import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.recipe import (deep_merge, load_recipe, recipe_identity_sha256,
                                            resolve_recipe_data)

ROOT = Path(__file__).parents[2]
RECIPES = ROOT / 'training/neuralese/recipes'
CONVERTED = ['c12-ctx512', 'c12-ctx1k', 'c18', 'c18-ctx512', 'c18-ctx1k', 'c21-ctx512', 'c23-ctx512',
             'c23-ctx1k', 'c23-calibrated-v1']
# Verbatim pre-conversion copies of the original files (taken at commit 8a96d758).
ORIGINALS = Path(__file__).parent / 'recipe_originals'


def base_recipe():
    return {
        'schema': 'natlang.neuralese-training-recipe/1', 'id': 'base-r', 'description': 'base',
        'stages': [
            {'id': 'token_identity', 'kind': 'token_identity', 'requires': [],
             'parameters': {'limit': 32, 'max_tokens': 512}},
            {'id': 'embedding_distillation', 'kind': 'causal_embedding_distillation',
             'requires': ['token_identity'], 'parameters': {'cutoff': 12, 'steps': 8, 'batch': 2}},
        ]}


def write(directory, name, value):
    path = Path(directory) / (name + '.json')
    path.write_text(json.dumps(value))
    return path


def child(**overrides):
    return {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'child-r', 'extends': 'base-r',
            'description': 'child', 'overrides': overrides}


def test_deep_merge_objects_merge_lists_and_scalars_replace_and_delete_marker_removes():
    base = {'a': {'x': 1, 'y': [1, 2]}, 'b': 1, 'c': 2}
    merged = deep_merge(base, {'a': {'y': [3], 'z': None}, 'b': {'$delete': True}})
    assert merged == {'a': {'x': 1, 'y': [3], 'z': None}, 'c': 2}
    assert base == {'a': {'x': 1, 'y': [1, 2]}, 'b': 1, 'c': 2}  # inputs untouched
    with pytest.raises(ValueError):
        deep_merge({}, {'missing': {'$delete': True}})


def test_stage_overrides_merge_by_stage_id_and_lists_replace(tmp_path):
    write(tmp_path, 'base', base_recipe())
    path = write(tmp_path, 'child', child(stages={
        'embedding_distillation': {'parameters': {'cutoff': 18, 'seed': 3}, 'requires': ['token_identity', 'token_identity']},
        'token_identity': {'parameters': {'limit': {'$delete': True}}}}))
    resolved = resolve_recipe_data(path)
    stages = {s['id']: s for s in resolved['stages']}
    assert stages['embedding_distillation']['parameters'] == {'cutoff': 18, 'steps': 8, 'batch': 2, 'seed': 3}
    assert stages['embedding_distillation']['requires'] == ['token_identity', 'token_identity']  # replaced
    assert stages['token_identity']['parameters'] == {'max_tokens': 512}
    assert [s['id'] for s in resolved['stages']] == ['token_identity', 'embedding_distillation']
    assert resolved['id'] == 'child-r' and resolved['description'] == 'child'
    assert 'extends' not in resolved and 'overrides' not in resolved


def test_top_level_override_and_unknown_stage_and_forbidden_keys(tmp_path):
    write(tmp_path, 'base', base_recipe())
    resolved = resolve_recipe_data(write(tmp_path, 'c1', child(scope={'a': 1})))
    assert resolved['scope'] == {'a': 1}
    for bad in (child(stages={'nope': {'parameters': {}}}), child(id='x'), child(schema='y'),
                child(stages={'token_identity': {'id': 'renamed'}})):
        with pytest.raises(ValueError):
            resolve_recipe_data(write(tmp_path, 'bad', bad))
    with pytest.raises(ValueError):
        resolve_recipe_data(write(tmp_path, 'orphan', {**base_recipe(), 'id': 'o', 'overrides': {}}))


def test_unknown_parent_and_cycles_fail(tmp_path):
    write(tmp_path, 'child', child())
    with pytest.raises(ValueError, match='exactly one'):
        resolve_recipe_data(tmp_path / 'child.json')
    a = {**base_recipe(), 'id': 'a', 'extends': 'b'}
    b = {**base_recipe(), 'id': 'b', 'extends': 'c'}
    c = {**base_recipe(), 'id': 'c', 'extends': 'a'}
    for name, value in (('a', a), ('b', b), ('c', c)):
        write(tmp_path, name, value)
    with pytest.raises(ValueError, match='cycle'):
        resolve_recipe_data(tmp_path / 'a.json')
    write(tmp_path, 'self', {**base_recipe(), 'id': 'self', 'extends': 'self'})
    with pytest.raises(ValueError, match='cycle'):
        resolve_recipe_data(tmp_path / 'self.json')


def test_handler_whitelist_applies_after_resolution(tmp_path):
    write(tmp_path, 'base', base_recipe())
    good = write(tmp_path, 'good', child(stages={'embedding_distillation': {'parameters': {'seed': 1}}}))
    assert load_recipe(good)['stages'][1]['parameters']['seed'] == 1
    bad = write(tmp_path, 'bad', child(stages={'embedding_distillation': {'parameters': {'not_a_parameter': 1}}}))
    with pytest.raises(ValueError, match='unknown stage parameters'):
        load_recipe(bad)


def test_hash_is_canonical_over_resolved_content_and_stable(tmp_path):
    write(tmp_path, 'base', base_recipe())
    path = write(tmp_path, 'child', child(stages={'embedding_distillation': {'parameters': {'seed': 1}}}))
    first = recipe_identity_sha256(path)
    assert first == canonical_json_sha256_hex(resolve_recipe_data(path))
    # key order and whitespace in the file do not matter; resolved content does
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data, indent=4, sort_keys=True))
    assert recipe_identity_sha256(path) == first
    data['overrides']['stages']['embedding_distillation']['parameters']['seed'] = 2
    path.write_text(json.dumps(data))
    assert recipe_identity_sha256(path) != first
    # a plain recipe keeps its file-bytes hash so earlier receipts stay comparable
    plain = write(tmp_path, 'plain', {**base_recipe(), 'id': 'plain'})
    import hashlib
    assert recipe_identity_sha256(plain) == hashlib.sha256(plain.read_bytes()).hexdigest()


def test_resolve_cli_prints_recipe_and_hash(tmp_path):
    out = subprocess.run([sys.executable, '-m', 'natlang_neuralese.train.recipe', 'resolve',
                          'token-preserving-foundation-maple-c18-ctx512'],
                         capture_output=True, text=True, check=True, cwd=ROOT,
                         env={**__import__('os').environ, 'PYTHONPATH': str(ROOT / 'training/neuralese')}).stdout
    lines = out.strip().splitlines()
    summary = json.loads(lines[-1])
    recipe = json.loads('\n'.join(lines[:-1]))
    assert summary['recipe_sha256'] == canonical_json_sha256_hex(recipe) and summary['inherits'] is True
    assert recipe['stages'][1]['parameters']['cutoff'] == 18


@pytest.mark.parametrize('name', CONVERTED)
def test_converted_recipe_resolves_to_the_original_content(name):
    original = json.loads((ORIGINALS / f'foundation-maple-{name}.json').read_text())
    path = RECIPES / f'foundation-maple-{name}.json'
    assert 'extends' in json.loads(path.read_text())
    resolved = load_recipe(path)
    ignore = {'extends', 'overrides', 'description'}
    assert {k: v for k, v in resolved.items() if k not in ignore} == {k: v for k, v in original.items() if k not in ignore}
    assert resolved['id'] == original['id']


def test_history_keeps_every_original_description():
    history = (RECIPES / 'HISTORY.md').read_text()
    for name in CONVERTED:
        original = json.loads((ORIGINALS / f'foundation-maple-{name}.json').read_text())
        assert original['description'] in history and original['id'] in history


def test_harness_bench_cohort_is_declared_held_and_shared_by_both_lines():
    """raw-recurrence-v3 / raw-recurrence-mellum-v2 add only the held harness_bench cohort to v2 / mellum-v1."""
    ignore = {'id', 'description', 'cohorts'}
    for new, old in (('raw-recurrence-v3', 'raw-recurrence-v2'), ('raw-recurrence-mellum-v2', 'raw-recurrence-mellum-v1')):
        resolved, previous = load_recipe(RECIPES / f'{new}.json'), load_recipe(RECIPES / f'{old}.json')
        assert {k: v for k, v in resolved.items() if k not in ignore} == \
            {k: v for k, v in previous.items() if k not in ignore}
        assert 'cohorts' not in previous
    lfm, mellum = load_recipe(RECIPES / 'raw-recurrence-v3.json'), load_recipe(RECIPES / 'raw-recurrence-mellum-v2.json')
    cohort = lfm['cohorts']['harness_bench']
    assert mellum['cohorts'] == lfm['cohorts']
    assert cohort['admitted'] is False and cohort['recurrence']['admitted'] is False
    assert cohort['recurrence']['trajectory_trainer']['view'] == 'written'
    registry = {c['id']: c for c in json.loads((ROOT / 'training/neuralese_corpora.json').read_text())['corpora']}
    for twin in cohort['twins'].values():
        entry = registry[twin['corpus']]
        assert entry['training_admission'] is False
        assert entry['build']['text_jsonl_sha256'] == twin['text_jsonl_sha256']
        assert cohort['source']['corpus'] in entry['derived_from']
