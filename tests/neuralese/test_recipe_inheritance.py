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
        'identity_qualification': 'native_state_runtime_qualified',
        'input_bindings': {'train': {'path': '/tmp/train.jsonl', 'sha256': 'a' * 64},
                           'held': {'path': '/tmp/held.jsonl', 'sha256': 'b' * 64}},
        'stages': [
            {'id': 'graphs', 'kind': 'native_state_sft', 'requires': [],
             'inputs': {'native_graphs': 'train', 'held_native_graphs': 'held'},
             'parameters': {'epochs': 1, 'batch_graphs': 2}},
            {'id': 'graphs_again', 'kind': 'native_state_sft', 'requires': ['graphs'],
             'inputs': {'native_graphs': 'train', 'held_native_graphs': 'held'},
             'parameters': {'epochs': 1, 'batch_graphs': 2}},
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
        'graphs_again': {'parameters': {'epochs': 2, 'seed': 3}, 'requires': ['graphs', 'graphs']},
        'graphs': {'parameters': {'batch_graphs': {'$delete': True}}}}))
    resolved = resolve_recipe_data(path)
    stages = {s['id']: s for s in resolved['stages']}
    assert stages['graphs_again']['parameters'] == {'epochs': 2, 'batch_graphs': 2, 'seed': 3}
    assert stages['graphs_again']['requires'] == ['graphs', 'graphs']  # replaced
    assert stages['graphs']['parameters'] == {'epochs': 1}
    assert [s['id'] for s in resolved['stages']] == ['graphs', 'graphs_again']
    assert resolved['id'] == 'child-r' and resolved['description'] == 'child'
    assert 'extends' not in resolved and 'overrides' not in resolved


def test_top_level_override_and_unknown_stage_and_forbidden_keys(tmp_path):
    write(tmp_path, 'base', base_recipe())
    resolved = resolve_recipe_data(write(tmp_path, 'c1', child(scope={'a': 1})))
    assert resolved['scope'] == {'a': 1}
    for bad in (child(stages={'nope': {'parameters': {}}}), child(id='x'), child(schema='y'),
                child(stages={'graphs': {'id': 'renamed'}})):
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
    good = write(tmp_path, 'good', child(stages={'graphs_again': {'parameters': {'seed': 1}}}))
    assert load_recipe(good)['stages'][1]['parameters']['seed'] == 1
    bad = write(tmp_path, 'bad', child(stages={'graphs_again': {'parameters': {'not_a_parameter': 1}}}))
    with pytest.raises(ValueError, match='unknown stage parameters'):
        load_recipe(bad)


def test_hash_is_canonical_over_resolved_content_and_stable(tmp_path):
    write(tmp_path, 'base', base_recipe())
    path = write(tmp_path, 'child', child(stages={'graphs_again': {'parameters': {'seed': 1}}}))
    first = recipe_identity_sha256(path)
    assert first == canonical_json_sha256_hex(resolve_recipe_data(path))
    # key order and whitespace in the file do not matter; resolved content does
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data, indent=4, sort_keys=True))
    assert recipe_identity_sha256(path) == first
    data['overrides']['stages']['graphs_again']['parameters']['seed'] = 2
    path.write_text(json.dumps(data))
    assert recipe_identity_sha256(path) != first
    # a plain recipe keeps its file-bytes hash so earlier receipts stay comparable
    plain = write(tmp_path, 'plain', {**base_recipe(), 'id': 'plain'})
    import hashlib
    assert recipe_identity_sha256(plain) == hashlib.sha256(plain.read_bytes()).hexdigest()


def test_resolve_cli_prints_recipe_and_hash(tmp_path):
    out = subprocess.run([sys.executable, '-m', 'natlang_neuralese.train.recipe', 'resolve',
                          'native-state-production-mellum-20261015-v1'],
                         capture_output=True, text=True, check=True, cwd=ROOT,
                         env={**__import__('os').environ, 'PYTHONPATH': str(ROOT / 'training/neuralese')}).stdout
    lines = out.strip().splitlines()
    summary = json.loads(lines[-1])
    recipe = json.loads('\n'.join(lines[:-1]))
    assert summary['recipe_sha256'] == canonical_json_sha256_hex(recipe) and summary['inherits'] is True
    assert recipe['id'] == 'native-state-production-mellum-20261015-v1'
    assert recipe['stages'] == load_recipe(RECIPES / 'native-state-production-mellum-20261015-v1.json')['stages']


def test_stages_added_insert_after_a_named_stage(tmp_path):
    (tmp_path / 'base.json').write_text(json.dumps(base_recipe()))
    extra = {'id': 'graphs_extra', 'kind': 'native_state_sft', 'requires': ['graphs'],
             'inputs': {'native_graphs': 'train', 'held_native_graphs': 'held'}, 'parameters': {'epochs': 1}}
    child = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'child-r', 'extends': 'base-r',
             'overrides': {'stages_added': [{'after': 'graphs', 'stage': extra}]}}
    (tmp_path / 'child.json').write_text(json.dumps(child))
    resolved = resolve_recipe_data(tmp_path / 'child.json')
    assert [s['id'] for s in resolved['stages']] == ['graphs', 'graphs_extra', 'graphs_again']
    for bad in ({'after': 'nowhere', 'stage': extra}, {'after': 'graphs', 'stage': {**extra, 'id': 'graphs'}},
                {'stage': extra}):
        child['overrides']['stages_added'] = [bad]
        (tmp_path / 'child.json').write_text(json.dumps(child))
        with pytest.raises(ValueError):
            resolve_recipe_data(tmp_path / 'child.json')



def test_native_parameter_handoff_preserves_epoch_extension_and_operational_updates(tmp_path):
    from natlang_neuralese.train.recipe import _without_operational, native_stage_parameter_handoffs
    old = base_recipe()
    new = copy.deepcopy(old)
    new['stages'][0]['parameters'].update(epochs=3, eval_every=128, checkpoint_every=64,
                                          sdpa_backend='flash', deterministic_training=False)
    changes, old_contract, new_contract = native_stage_parameter_handoffs(old, new, tmp_path)
    assert changes['graphs']['epochs'] == [1, 3]
    assert _without_operational(old_contract) == _without_operational(new_contract)
    changed = copy.deepcopy(new)
    changed['stages'][0]['parameters']['batch_graphs'] = 4
    assert _without_operational(old) != _without_operational(changed)
