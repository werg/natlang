"""Regression for the production stage-ID/kind controller failure."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.coreweave.native_controller_continuation import prepare_drained_supervised_continuation


@pytest.fixture
def context(tmp_path):
    def binding(path):
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    def write(path, value):
        path.write_text(json.dumps(value))
    initial = tmp_path / 'initial.json'
    capacity = {'path': str(tmp_path / 'capacity.json'), 'sha256': 'capacity'}
    initial.write_text(json.dumps({'geometry_qualification': capacity, 'geometry_identity': {'profile': 'actual'}}))
    plan = {'code': {'model.py': 'unchanged'}, 'recipe': {'id': 'production', 'stages': [
        {'id': 'native_full_sft', 'kind': 'native_state_sft', 'parameters': {
            'refresh_after_updates': 256, 'memory_qualification': ['old'],
            'task_eval': {'packet': 'protected'}, 'learning_rate': 3e-6}},
        {'id': 'next', 'kind': 'policy', 'parameters': {'debt': 'preserved'}}],
        'input_bindings': {'train': {'sha256': 'all-sources'}, 'held': {'sha256': 'protected'}}}}
    def geometry(ref, *, identity, code):
        assert ref == capacity and identity == {'profile': 'actual'} and code == plan['code']
        return {'forward_tokens': 100, 'kv_slots': 200}
    resident = SimpleNamespace(binding=binding, bound=lambda ref: __import__('pathlib').Path(ref['path']),
        code_inventory=lambda package: {'model.py': 'unchanged'}, geometry_capacity=geometry,
        digest=lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
    record = {'resident': {'initial': binding(initial), 'pending_uncredited': {'sha256': 'pending'}}}
    drain = SimpleNamespace(_drained_plan_authority=lambda record, saved: (None, plan, 'native_full_sft'))
    return dict(record=record, saved=object(), drain=drain, resident=resident,
                package=tmp_path, directory=tmp_path, quantum_updates=1000, write=write), plan


def test_actual_production_id_and_distinct_kind_preserve_full_recipe(context):
    args, plan = context
    before = copy.deepcopy(plan)
    ref = prepare_drained_supervised_continuation(**args)
    recipe = json.loads(args['resident'].bound(ref).read_bytes())
    expected = copy.deepcopy(before['recipe'])
    p = expected['stages'][0]['parameters']
    p.update(refresh_after_updates=1000, resident_geometry_qualification=str(args['directory'] / 'capacity.json'))
    p.pop('memory_qualification')
    assert recipe == expected and plan == before
    assert prepare_drained_supervised_continuation(**args) == ref
    proof = json.loads((args['directory'] / 'drained-continuation-proof.json').read_bytes())
    assert proof['whole_recipe_completed'] is False and proof['exposure_credit'] == 0
    assert proof['source_resident'] == args['record']['resident']


def test_instance_id_cannot_stand_in_for_implementation_kind(context):
    args, plan = context
    plan['recipe']['stages'][0]['kind'] = 'native_full_sft'
    with pytest.raises(ValueError, match='stage kind'):
        prepare_drained_supervised_continuation(**args)


@pytest.mark.parametrize('quantum', [True, False, 0, -1, 1.5, None])
def test_invalid_declared_quantum_never_writes(context, quantum):
    args, _ = context
    args['quantum_updates'] = quantum
    with pytest.raises(ValueError, match='positive declared'):
        prepare_drained_supervised_continuation(**args)
    assert not (args['directory'] / 'drained-continuing-recipe.json').exists()


def test_runtime_change_is_not_capacity_qualification(context):
    args, plan = context
    plan['code'] = {'model.py': 'changed'}
    with pytest.raises(ValueError, match='runtime changed'):
        prepare_drained_supervised_continuation(**args)


def test_retry_rejects_changed_policy(context):
    args, _ = context
    prepare_drained_supervised_continuation(**args)
    args['quantum_updates'] = 2000
    with pytest.raises(ValueError, match='changed on retry'):
        prepare_drained_supervised_continuation(**args)
