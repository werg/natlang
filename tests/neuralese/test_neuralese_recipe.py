import json
from pathlib import Path

import pytest

from natlang_neuralese.train.recipe import load_recipe, require_gate, stage_parameter_args


def declared():
    return json.loads((Path(__file__).parents[2] / 'training/neuralese/recipes/foundation-v1.json').read_text())


def test_shared_recipe_declares_warmup_and_identity_dependency(tmp_path):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(declared()))
    recipe = load_recipe(path)
    assert recipe['stages'][1]['kind'] == 'causal_embedding_distillation'
    assert recipe['stages'][1]['requires'] == ['token_identity']


@pytest.mark.parametrize('change', ['missing_identity', 'forward_dependency', 'unknown_parameter', 'unsafe_id', 'unknown_kind'])
def test_invalid_recipe_is_rejected_before_launch(tmp_path, change):
    recipe = declared()
    if change == 'missing_identity':
        recipe['stages'][1]['requires'] = []
    elif change == 'forward_dependency':
        recipe['stages'][0]['requires'] = ['embedding_distillation']
    elif change == 'unknown_parameter':
        recipe['stages'][1]['parameters']['shell_command'] = 'unused'
    elif change == 'unsafe_id':
        recipe['stages'][0]['id'] = '../escape'
    else:
        recipe['stages'][0]['kind'] = 'arbitrary_module'
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError):
        load_recipe(path)


def test_finishing_steps_or_partial_agreement_does_not_pass_gate():
    with pytest.raises(ValueError, match='distillation gate failed'):
        require_gate({'step': 2048, 'agreement': .62, 'feedback_gate_passed': False}, 'causal_embedding_distillation')
    with pytest.raises(ValueError):
        require_gate({}, 'token_identity')
    require_gate({'feedback_gate_passed': True}, 'causal_embedding_distillation')


def test_runtime_stage_requires_the_embedding_gate(tmp_path):
    recipe = declared()
    assert recipe['stages'][2]['kind'] == 'raw_runtime_qualification'
    recipe['stages'][2]['requires'] = ['token_identity']
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='embedding foundation'):
        load_recipe(path)
    with pytest.raises(ValueError, match='runtime transport gate failed'):
        require_gate({}, 'raw_runtime_qualification')


def test_stage_specific_named_input_bindings_are_hash_pinned_and_role_scoped(tmp_path):
    from natlang_neuralese.train.recipe import resolve_stage_inputs, stage_input_args
    from natlang_neuralese.train.output_embedding_projection import sha

    records = tmp_path / 'native-records.jsonl'
    pieces = tmp_path / 'native-pieces.jsonl'
    text = tmp_path / 'ordinary-text.jsonl'
    records.write_text('{"id":"n1"}\n')
    pieces.write_text('{"id":"p1"}\n')
    text.write_text('{"id":"t1"}\n')
    recipe = declared()
    recipe['input_bindings'] = {
        'native.records': {'path': str(records), 'sha256': sha(records)},
        'native.pieces': {'path': str(pieces), 'sha256': sha(pieces)},
        'text.corpus': {'path': str(text), 'sha256': sha(text)},
    }
    recipe['stages'][0]['inputs'] = {'records': 'native.records'}
    recipe['stages'][1]['inputs'] = {
        'records': 'native.records', 'pieces': 'native.pieces'}
    recipe['stages'][2]['inputs'] = {'records': 'native.records'}
    recipe['stages'].append({
        'id': 'core_text_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification'],
        'parameters': {'neuralese_input': 'map'},
        'inputs': {'records': 'native.records', 'pieces': 'native.pieces',
                   'text_data': 'text.corpus'},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    loaded = load_recipe(path)

    identity = resolve_stage_inputs(loaded, loaded['stages'][0], {})
    assert set(identity) == {'records'}
    assert identity['records']['sha256'] == sha(records)
    assert stage_input_args(identity, 'token_identity') == ['--records', str(records.resolve())]

    distill = resolve_stage_inputs(loaded, loaded['stages'][1], {})
    assert set(distill) == {'records', 'pieces'}
    assert stage_input_args(distill, 'causal_embedding_distillation') == [
        '--records', str(records.resolve()), '--pieces', str(pieces.resolve())]

    warmup = resolve_stage_inputs(loaded, loaded['stages'][3], {})
    assert set(warmup) == {'records', 'pieces', 'text_data'}
    assert stage_input_args(warmup, 'core_text_warmup') == [
        '--records', str(records.resolve()), '--pieces', str(pieces.resolve()),
        '--text-data', str(text.resolve())]


def test_named_input_binding_relocation_keeps_expected_content_identity(tmp_path):
    from natlang_neuralese.train.recipe import parse_input_binding_overrides, resolve_stage_inputs
    from natlang_neuralese.train.output_embedding_projection import sha

    original = tmp_path / 'source.jsonl'
    relocated = tmp_path / 'relocated.jsonl'
    mismatch = tmp_path / 'different.jsonl'
    original.write_text('{"same":true}\n')
    relocated.write_bytes(original.read_bytes())
    mismatch.write_text('{"same":false}\n')
    recipe = declared()
    recipe['input_bindings'] = {
        'corpus.native': {'path': str(original), 'sha256': sha(original)}}
    recipe['stages'][0]['inputs'] = {'records': 'corpus.native'}
    stage = recipe['stages'][0]
    overrides = parse_input_binding_overrides([f'corpus.native={relocated}'])
    resolved = resolve_stage_inputs(recipe, stage, {}, overrides)
    assert resolved['records']['path'] == str(relocated.resolve())
    assert resolved['records']['sha256'] == sha(original)
    with pytest.raises(ValueError, match='content hash mismatch'):
        resolve_stage_inputs(recipe, stage, {}, {'corpus.native': str(mismatch)})
    with pytest.raises(ValueError, match='undeclared input binding'):
        resolve_stage_inputs(recipe, stage, {}, {'other.input': str(relocated)})


def test_recipe_runner_routes_each_stage_its_declared_input_files(tmp_path, monkeypatch):
    from natlang_neuralese.train import recipe as recipe_runner
    from natlang_neuralese.train.output_embedding_projection import sha

    heads = tmp_path / 'heads.pt'
    native = tmp_path / 'native-records.jsonl'
    fallback_records = tmp_path / 'fallback-records.jsonl'
    fallback_pieces = tmp_path / 'fallback-pieces.jsonl'
    for path, body in ((heads, b'head'), (native, b'{"source":"native"}\n'),
                       (fallback_records, b'{"source":"fallback"}\n'),
                       (fallback_pieces, b'{"piece":"fallback"}\n')):
        path.write_bytes(body)
    recipe = declared()
    recipe['input_bindings'] = {
        'native.records': {'path': str(native), 'sha256': sha(native)}}
    recipe['stages'][0]['inputs'] = {'records': 'native.records'}
    recipe_path = tmp_path / 'recipe.json'
    recipe_path.write_text(json.dumps(recipe))
    commands = []

    class CompletedStage:
        def __init__(self, command, env):
            commands.append(command)
            output = Path(command[command.index('--out') + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({'token_aligned_reference_passed': True}))

        def wait(self):
            return 0

    monkeypatch.setattr(recipe_runner.subprocess, 'Popen', CompletedStage)
    out = tmp_path / 'run'
    recipe_runner.main([
        '--recipe', str(recipe_path), '--heads', str(heads),
        '--records', str(fallback_records), '--pieces', str(fallback_pieces),
        '--out', str(out), '--device', 'cpu', '--until', 'token_identity'])

    assert commands
    command = commands[0]
    assert command[command.index('--records') + 1] == str(native.resolve())
    plan = json.loads((out / 'recipe-plan.json').read_text())
    stage_inputs = plan['stage_inputs']
    assert stage_inputs['token_identity']['records']['binding'] == 'native.records'
    assert stage_inputs['token_identity']['records']['sha256'] == sha(native)
    assert stage_inputs['embedding_distillation']['records']['path'] == str(fallback_records.resolve())


def test_named_input_bindings_reject_incomplete_stage_roles(tmp_path):
    recipe = declared()
    recipe['input_bindings'] = {
        'native.records': {'path': 'records.jsonl', 'sha256': '0' * 64}}
    recipe['stages'].append({
        'id': 'core_text_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification'],
        'parameters': {'neuralese_input': 'map'},
        'inputs': {'records': 'native.records'},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='stage input roles'):
        load_recipe(path)


def test_foundation_certificate_survives_verified_directory_relocation(tmp_path):
    from natlang_neuralese.train.recipe import require_foundation
    from natlang_neuralese.train.output_embedding_projection import sha
    heads, checkpoint = tmp_path / 'heads.pt', tmp_path / 'feedback.pt'
    heads.write_bytes(b'head')
    checkpoint.write_bytes(b'feedback')
    stages = []
    for name, kind, gate in [('identity', 'token_identity', {'token_aligned_reference_passed': True}),
                             ('embedding', 'causal_embedding_distillation', {'feedback_gate_passed': True})]:
        report = tmp_path / (name + '-report.json')
        report.write_text(json.dumps({'kind': kind, 'gate': gate}))
        stages.append({'kind': kind, 'report': '/old/machine/' + report.name, 'report_sha256': sha(report)})
    certificate = tmp_path / 'foundation-certificate.json'
    certificate.write_text(json.dumps({'schema': 'natlang.neuralese-foundation-certificate/1', 'qualified': True,
                                       'heads_sha256': sha(heads), 'feedback_checkpoint_sha256': sha(checkpoint),
                                       'stages': stages}))
    assert require_foundation(certificate, heads=heads, checkpoint=checkpoint)['qualified']
    (tmp_path / 'identity-report.json').write_text('{}')
    with pytest.raises(ValueError, match='report changed'):
        require_foundation(certificate, heads=heads, checkpoint=checkpoint)


def test_recipe_declares_runtime_without_hardcoded_order_guards_and_completion_is_not_admission(tmp_path):
    recipe = json.loads((Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v1.json').read_text())
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    assert load_recipe(path)['stages'][3]['requires'] == ['runtime_qualification']
    recipe['stages'][3]['requires'] = ['embedding_distillation']
    path.write_text(json.dumps(recipe))
    # The shared declared recipe carries the policy; loader checks graph shape,
    # not a second hardcoded implementation of the training order.
    assert load_recipe(path)['stages'][3]['requires'] == ['embedding_distillation']
    with pytest.raises(ValueError):
        require_gate({'training_stage_completed': True, 'errors': 1}, 'raw_recurrence_training')
    require_gate({'training_stage_completed': True, 'errors': 0, 'semantic_channel_qualified': False}, 'raw_recurrence_training')


def test_raw_recipe_declares_full_depth_output_and_native_value_sizing():
    recipe = load_recipe(Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v1.json')
    parameters = recipe['stages'][-1]['parameters']
    assert parameters['content_transport'] == 'top-state'
    assert parameters['writer_length_policy'] == 'native-value'
    assert parameters['max_write_vectors'] >= 163
    assert parameters['writer_supervision'] == 'native-value'
    assert parameters['stop_supervision'] == 'gold-native-boundary'


def test_shared_text_recipe_trains_both_projections_to_plateau_then_sequence_passes():
    recipe=load_recipe(Path(__file__).parents[2]/'training/neuralese/recipes/raw-recurrence-v1.json')
    p=next(stage['parameters'] for stage in recipe['stages'] if stage['kind']=='core_text_warmup')
    assert 'aligned_steps' not in p and 'ramp_steps' not in p
    assert p['projection_patience']==3 and p['projection_min_evals']==2
    assert p['sketch_weight']==p['embedding_weight']==1.
    assert p['sketch_lr']>p['lr']
    assert (p['neuralese_input'], p['input_map_kernel'], p['input_map_rank'], p['rollout_passes']) == ('map', 4, 64, 0)


def test_core_text_recipe_requires_and_propagates_explicit_map_mode(tmp_path):
    recipe = declared()
    recipe['stages'].append({
        'id': 'core_text_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification'],
        'parameters': {'neuralese_input': 'map', 'input_map_kernel': 4,
                       'input_map_rank': 64, 'rollout_passes': 0},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    assert load_recipe(path)['stages'][-1]['parameters']['neuralese_input'] == 'map'
    assert stage_parameter_args(recipe['stages'][-1]['parameters']) == [
        '--neuralese-input', 'map', '--input-map-kernel', '4',
        '--input-map-rank', '64', '--rollout-passes', '0',
    ]

    recipe['stages'][-1]['parameters'] = {'rollout_passes': 0}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='must declare neuralese_input'):
        load_recipe(path)


def test_core_text_recipe_keeps_explicit_sketch_reproduction_and_rejects_map_rollout(tmp_path):
    recipe = declared()
    recipe['stages'].append({
        'id': 'core_text_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification'],
        'parameters': {'neuralese_input': 'sketch', 'rollout_passes': 4},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    assert load_recipe(path)['stages'][-1]['parameters']['neuralese_input'] == 'sketch'
    recipe['stages'][-1]['parameters'] = {'neuralese_input': 'map', 'rollout_passes': 4}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='requires rollout_passes=0'):
        load_recipe(path)


def test_every_declared_shared_core_warmup_names_its_input_mode():
    recipes = Path(__file__).parents[2] / 'training/neuralese/recipes'
    declared_recipes = [json.loads(path.read_text()) for path in recipes.glob('*.json')]
    for recipe in declared_recipes:
        if recipe.get('schema') != 'natlang.neuralese-training-recipe/1':
            continue
        for stage in recipe['stages']:
            if stage['kind'] == 'core_text_warmup':
                assert stage['parameters']['neuralese_input'] in {'map', 'sketch'}
