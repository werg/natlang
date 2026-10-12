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
                          'neuralese-direct-mellum-v2'],
                         capture_output=True, text=True, check=True, cwd=ROOT,
                         env={**__import__('os').environ, 'PYTHONPATH': str(ROOT / 'training/neuralese')}).stdout
    lines = out.strip().splitlines()
    summary = json.loads(lines[-1])
    recipe = json.loads('\n'.join(lines[:-1]))
    assert summary['recipe_sha256'] == canonical_json_sha256_hex(recipe) and summary['inherits'] is True
    assert recipe['id'] == 'neuralese-direct-mellum-v2' and 'extends' not in recipe
    assert recipe['stages'] == load_recipe(RECIPES / 'neuralese-direct-mellum-v2.json')['stages']


def test_stages_added_insert_after_a_named_stage(tmp_path):
    (tmp_path / 'base.json').write_text(json.dumps(base_recipe()))
    extra = {'id': 'token_identity_again', 'kind': 'token_identity', 'requires': ['token_identity'],
             'parameters': {'limit': 8, 'max_tokens': 64}}
    child = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'child-r', 'extends': 'base-r',
             'overrides': {'stages_added': [{'after': 'token_identity', 'stage': extra}]}}
    (tmp_path / 'child.json').write_text(json.dumps(child))
    resolved = resolve_recipe_data(tmp_path / 'child.json')
    assert [s['id'] for s in resolved['stages']] == ['token_identity', 'token_identity_again', 'embedding_distillation']
    for bad in ({'after': 'nowhere', 'stage': extra}, {'after': 'token_identity', 'stage': {**extra, 'id': 'token_identity'}},
                {'stage': extra}):
        child['overrides']['stages_added'] = [bad]
        (tmp_path / 'child.json').write_text(json.dumps(child))
        with pytest.raises(ValueError):
            resolve_recipe_data(tmp_path / 'child.json')


def legacy_lineage(tmp_path, name='legacy'):
    import legacy_recipes
    path = write(tmp_path, name, legacy_recipes.raw_recurrence())
    return resolve_recipe_data(path)


def test_operational_stage_parameters_do_not_change_the_training_semantics(tmp_path):
    from natlang_neuralese.train.recipe import _without_operational
    old = legacy_lineage(tmp_path)
    new = copy.deepcopy(old)
    new['id'], new['description'] = 'legacy-operational', 'an operational handoff'
    new['stage_parameter_defaults']['core_text_warmup'].update(moe_kernel='grouped', eval_every=256)
    next(s for s in new['stages'] if s['id'] == 'recurrence_warmup')['parameters']['checkpoint_every'] = 50
    assert old != new and _without_operational(old) == _without_operational(new)
    changed = copy.deepcopy(new)
    changed['stage_parameter_defaults']['core_text_warmup']['lr'] = 1e-4
    assert _without_operational(changed) != _without_operational(old)


def test_a_handoff_extends_an_unqualified_warmup_with_its_quantization_range_held(tmp_path):
    from natlang_neuralese.train.recipe import _without_extensions, _without_operational, stage_extensions
    old = legacy_lineage(tmp_path)
    new = copy.deepcopy(old)
    warm = next(stage for stage in new['stages'] if stage['id'] == 'core_text_warmup')
    warm['parameters'].update(steps=6144, quantization_steps=4096)
    new['stage_parameter_defaults']['core_text_warmup']['report_column_fraction'] = 0.25
    out = tmp_path / 'lineage'
    out.mkdir()
    extended = stage_extensions(old, new, out)
    assert extended == {'core_text_warmup': [4096, 6144]}
    assert (_without_operational(_without_extensions(old, extended)) ==
            _without_operational(_without_extensions(new, extended)))
    del warm['parameters']['quantization_steps']
    with pytest.raises(ValueError, match='quantization ramp'):
        stage_extensions(old, new, out)
    warm['parameters']['quantization_steps'] = 4096
    (out / 'core_text_warmup-report.json').write_text('{}')
    with pytest.raises(ValueError, match='already qualified'):
        stage_extensions(old, new, out)
    shorter = copy.deepcopy(old)
    next(stage for stage in shorter['stages'] if stage['id'] == 'core_text_warmup')['parameters']['steps'] = 2048
    with pytest.raises(ValueError, match='only a mapped text warm-up'):
        stage_extensions(old, shorter, tmp_path / 'other')
