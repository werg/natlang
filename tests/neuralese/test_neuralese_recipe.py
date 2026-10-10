import json
import os
import signal
import subprocess
import sys
import threading
import time
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


def test_ar_feedback_fixup_is_a_declared_map_continuation_mode(tmp_path):
    recipe=declared()
    recipe['stages'].append({
        'id':'ar_feedback_fixup','kind':'core_text_warmup',
        'requires':['runtime_qualification'],
        'parameters':{'ar_feedback_fixup':True},
    })
    path=tmp_path/'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='preceding mapped-input warm-up'):
        load_recipe(path)
    assert stage_parameter_args({'ar_feedback_fixup':True})==['--ar-feedback-fixup']


def test_text_warmup_recipe_propagates_explicit_named_optimizer_restore_controls(tmp_path):
    from argparse import ArgumentParser
    from natlang_neuralese.train.optim_restore import add_optimizer_restore_arguments
    from natlang_neuralese.train.recipe import HANDLERS

    parser = ArgumentParser()
    add_optimizer_restore_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.optimizer_state == 'restore'
    assert defaults.optimizer_added == []

    recipe = declared()
    recipe['stages'].extend([
        {'id': 'mapped_warmup', 'kind': 'core_text_warmup',
         'requires': ['runtime_qualification'], 'parameters': {'steps': 128}},
        {'id': 'mapped_ar_fixup', 'kind': 'core_text_warmup',
         'requires': ['mapped_warmup'], 'parameters': {
             'ar_feedback_fixup': True,
             'optimizer_state': 'restore',
             'optimizer_added': ['heads.new_projection.', 'backbone.new_adapter.'],
         }},
    ])
    path = tmp_path / 'mapped-ar.json'
    path.write_text(json.dumps(recipe))
    loaded = load_recipe(path)
    parameters = loaded['stages'][-1]['parameters']
    assert {'optimizer_state', 'optimizer_added'} <= HANDLERS['core_text_warmup']['parameters']
    assert stage_parameter_args(parameters) == [
        '--ar-feedback-fixup', '--optimizer-state', 'restore',
        '--optimizer-added', 'heads.new_projection.',
        '--optimizer-added', 'backbone.new_adapter.',
    ]

    parameters['optimizer_state'] = 'fresh'
    assert stage_parameter_args(parameters)[1:3] == ['--optimizer-state', 'fresh']


@pytest.mark.parametrize('parameters', [
    {'optimizer_state': 'silent'},
    {'optimizer_added': 'heads.new.'},
    {'optimizer_added': ['heads.valid.', 7]},
])
def test_text_warmup_recipe_rejects_malformed_optimizer_restore_controls(tmp_path, parameters):
    recipe = declared()
    recipe['stages'].append({
        'id': 'mapped_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification'], 'parameters': parameters,
    })
    path = tmp_path / 'invalid-mapped.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='optimizer_'):
        load_recipe(path)


def test_raw_recurrence_recipe_propagates_named_optimizer_restore_controls(tmp_path):
    from natlang_neuralese.train.recipe import HANDLERS

    recipe = load_recipe(Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v1.json')
    stage = next(stage for stage in recipe['stages'] if stage['kind'] == 'raw_recurrence_training')
    stage['parameters']['optimizer_state'] = 'restore'
    stage['parameters']['optimizer_added'] = ['heads.read_adapter.', 'backbone.control_rows']
    path = tmp_path / 'recurrence-optimizer-controls.json'
    path.write_text(json.dumps(recipe))

    loaded = load_recipe(path)
    assert {'optimizer_state', 'optimizer_added'} <= HANDLERS['raw_recurrence_training']['parameters']
    parameters = next(stage['parameters'] for stage in loaded['stages']
                      if stage['kind'] == 'raw_recurrence_training')
    assert stage_parameter_args({key: parameters[key] for key in ('optimizer_state', 'optimizer_added')}) == [
        '--optimizer-state', 'restore',
        '--optimizer-added', 'heads.read_adapter.',
        '--optimizer-added', 'backbone.control_rows',
    ]


HARNESS_RECURRENCE = {'view': 'written', 'view_window': 4096, 'distill': 1.0, 'context_weight': 1.0,
                      'feedback_weight': 0.25, 'context_coverage': 'records',
                      'cohort_weights': {'native': 0.75, 'harness_bench': 0.25}, 'qualification_cohort': 'native'}


def recurrence_recipe_with(tmp_path, parameters):
    recipe = load_recipe(Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v1.json')
    stage = next(stage for stage in recipe['stages'] if stage['kind'] == 'raw_recurrence_training')
    stage['parameters'].update(parameters)
    path = tmp_path / 'recurrence-cohort.json'
    path.write_text(json.dumps(recipe))
    return path


def test_raw_recurrence_recipe_carries_the_harness_cohort_trainer_options(tmp_path):
    loaded = load_recipe(recurrence_recipe_with(tmp_path, HARNESS_RECURRENCE))
    parameters = next(stage['parameters'] for stage in loaded['stages'] if stage['kind'] == 'raw_recurrence_training')
    assert stage_parameter_args({key: parameters[key] for key in HARNESS_RECURRENCE}) == [
        '--view', 'written', '--view-window', '4096', '--distill', '1.0', '--context-weight', '1.0',
        '--feedback-weight', '0.25', '--context-coverage', 'records',
        '--cohort-weights', '{"harness_bench":0.25,"native":0.75}', '--qualification-cohort', 'native']
    declared = load_recipe(Path(__file__).parents[2] / 'training/neuralese/recipes/raw-recurrence-v3.json')
    assert declared['cohorts']['harness_bench']['recurrence']['trajectory_trainer'] == HARNESS_RECURRENCE


@pytest.mark.parametrize('bad', [{'view': 'digest'}, {'view_window': 0}, {'view_window': 4096.0},
                                 {'context_weight': -1}, {'context_weight': float('nan')}, {'distill': 'yes'},
                                 {'feedback_weight': 1.5}, {'context_coverage': 'all'},
                                 {'cohort_weights': {'native': 0.5, 'harness_bench': 0.25}},
                                 {'cohort_weights': {'native': 1.0, '': 0.0}}, {'cohort_weights': [1]},
                                 {'cohort_weights': {'native': 1.0}, 'qualification_cohort': 'harness_bench'},
                                 {'qualification_cohort': ''}])
def test_raw_recurrence_recipe_rejects_malformed_cohort_trainer_options(tmp_path, bad):
    with pytest.raises(ValueError):
        load_recipe(recurrence_recipe_with(tmp_path, bad))


def test_raw_recurrence_recipe_hands_fixup_heads_to_runtime_and_recurrence():
    recipe=load_recipe(Path(__file__).parents[2]/'training/neuralese/recipes/raw-recurrence-v1.json')
    by_id={stage['id']:stage for stage in recipe['stages']}
    assert by_id['autoregressive_text_fixup']['requires']==['core_text_warmup']
    assert by_id['autoregressive_text_fixup']['parameters']['ar_feedback_fixup'] is True
    assert by_id['autoregressive_text_fixup']['parameters']['tokens']==16384
    assert by_id['autoregressive_text_fixup']['parameters']['target_tokens']==256
    from natlang_neuralese.train.recipe import stage_parameter_args
    assert '--target-tokens' in stage_parameter_args({'target_tokens':256})
    assert by_id['adapted_runtime']['requires']==['autoregressive_text_fixup']
    assert 'autoregressive_text_fixup' in by_id['recurrence_warmup']['requires']
    from natlang_neuralese.train.recipe import mapped_fixup_continuation, effective_stage_parameters
    predecessor={'id':'core_text_warmup','kind':'core_text_warmup','artifact':'/run/core/heads.pt',
                 'gate':{'step':22272}}
    checkpoint,heads=mapped_fixup_continuation(by_id['autoregressive_text_fixup'],[predecessor])
    assert str(checkpoint)=='/run/core/checkpoint.pt'
    assert str(heads)=='/run/core/heads.pt'
    assert effective_stage_parameters(by_id['autoregressive_text_fixup'],[predecessor],recipe)['steps']==23296


def test_shared_recipe_serializes_mellum_qat_and_member_controls_to_shared_handlers():
    from natlang_neuralese.train.recipe import HANDLERS, stage_parameter_args
    text={'qat_latent_lr':1e-5,'member_weight':0.2,'member_tokens':256,'member_eval_windows':3}
    recurrence={'qat_latent_lr':2e-5,'member_weight':0.3,'member_tokens':512,
                'member_eval':5,'member_mask_system':False,'member_full_weight':0.4}
    assert set(text)<=HANDLERS['core_text_warmup']['parameters']
    assert set(recurrence)<=HANDLERS['raw_recurrence_training']['parameters']
    assert stage_parameter_args(text)==[
        '--qat-latent-lr','1e-05','--member-weight','0.2','--member-tokens','256',
        '--member-eval-windows','3']
    assert stage_parameter_args(recurrence)==[
        '--qat-latent-lr','2e-05','--member-weight','0.3','--member-tokens','512',
        '--member-eval','5','--no-member-mask-system','--member-full-weight','0.4']


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
        'parameters': {},
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


def test_direct_shared_stage_recipe_pins_contract_and_builds_frozen_launch(tmp_path, monkeypatch):
    from natlang_neuralese.train import recipe as runner
    from natlang_neuralese.train.output_embedding_projection import sha

    records = tmp_path / 'records.jsonl'
    pieces = tmp_path / 'pieces.jsonl'
    text = tmp_path / 'text.jsonl'
    heads = tmp_path / 'heads.pt'
    checkpoint = tmp_path / 'checkpoint.pt'
    for path, data in ((records, b'{}\n'), (pieces, b'{}\n'), (text, b'{}\n'),
                       (heads, b'heads'), (checkpoint, b'checkpoint')):
        path.write_bytes(data)
    package = Path(runner.__file__).parents[1]
    code = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*.py'))}
    output = tmp_path / 'run'
    metadata = tmp_path / 'launch-intent.json'
    recipe = {
        'schema': runner.DIRECT_STAGE_SCHEMA,
        'id': 'fixture-direct-warmup',
        'execution': 'direct_shared_stage',
        'stage': {'id': 'warmup', 'kind': 'core_text_warmup'},
        'runtime': {'image': 'sha256:' + 'a' * 64, 'package_code': code},
        'inputs': {role: {'path': str(path), 'sha256': sha(path)} for role, path in
                   (('records', records), ('pieces', pieces), ('text_data', text),
                    ('heads', heads), ('student_checkpoint', checkpoint))},
        'lineage': {'scope': 'test only'},
        'parameters': {'steps': 4},
        'outputs': {'directory': str(output)},
    }
    recipe_path = tmp_path / 'recipe.json'
    recipe_path.write_text(json.dumps(recipe))
    loaded = load_recipe(recipe_path)
    assert loaded['schema'] == runner.DIRECT_STAGE_SCHEMA

    class FakeChild:
        def __init__(self, command, env):
            del env
            stage_out = Path(command[command.index('--out') + 1])
            (stage_out / 'report.json').write_text(json.dumps({'qualified': True}))
            (stage_out / 'heads.pt').write_bytes(b'head artifact')

        def wait(self):
            return 0

    monkeypatch.setattr(runner.subprocess, 'Popen', FakeChild)
    assert runner.run_declared_direct_stage(loaded, recipe_path, output, metadata, 'cpu') == 0
    plan = json.loads((output / 'direct-stage-plan.json').read_text())
    result = json.loads((output / 'direct-stage-result.json').read_text())
    assert plan['inputs']['student_checkpoint']['sha256'] == sha(checkpoint)
    assert '--student-checkpoint' in plan['child_argv']
    assert result['completed'] and result['qualified']
    assert metadata.is_file()


def test_direct_stage_rejects_unregistered_parameters_and_unpinned_roles(tmp_path):
    from natlang_neuralese.train.recipe import validate_direct_stage_recipe, DIRECT_STAGE_SCHEMA
    recipe = {
        'schema': DIRECT_STAGE_SCHEMA, 'id': 'fixture', 'execution': 'direct_shared_stage',
        'stage': {'id': 'warmup', 'kind': 'core_text_warmup'},
        'runtime': {'image': 'sha256:' + 'a' * 64, 'package_code': {'module.py': 'b' * 64}},
        'inputs': {'records': {'path': 'r', 'sha256': 'c' * 64},
                   'pieces': {'path': 'p', 'sha256': 'd' * 64}},
        'lineage': {'scope': 'test'}, 'parameters': {'shell_command': 'unsafe'},
        'outputs': {'directory': '/tmp/out'},
    }
    with pytest.raises(ValueError, match='unsupported handler parameters'):
        validate_direct_stage_recipe(recipe)
    recipe['parameters'] = {}
    recipe['inputs']['unregistered'] = {'path': 'x', 'sha256': 'e' * 64}
    with pytest.raises(ValueError, match='input roles'):
        validate_direct_stage_recipe(recipe)


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
        'parameters': {},
        'inputs': {'records': 'native.records'},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='stage input roles'):
        load_recipe(path)


def test_named_input_catalog_rejects_unreferenced_artifacts(tmp_path):
    recipe = declared()
    recipe['input_bindings'] = {
        'unused.records': {'path': 'records.jsonl', 'sha256': '0' * 64}}
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='unused named input'):
        load_recipe(path)


def test_trained_map_continuation_requires_handoff_and_uses_same_full_checkpoint_binding(tmp_path):
    from natlang_neuralese.train.recipe import resolve_stage_inputs, stage_input_args
    from natlang_neuralese.train.output_embedding_projection import sha

    artifacts = {}
    for name in ('native-pieces', 'raw-records', 'full-checkpoint',
                 'checkpoint-manifest', 'serving-heads', 'trained-runtime'):
        path = tmp_path / name
        path.write_text(name)
        artifacts[name] = path
    bindings = {name: {'path': str(path), 'sha256': sha(path)} for name, path in artifacts.items()}
    recipe = declared()
    recipe['input_bindings'] = bindings
    recipe['stages'][0]['inputs'] = {'records': 'raw-records'}
    recipe['stages'][1]['inputs'] = {'records': 'raw-records', 'pieces': 'native-pieces'}
    recipe['stages'][2]['inputs'] = {'records': 'raw-records'}
    checkpoint = 'full-checkpoint'
    recipe['stages'].append({
        'id': 'trained_heads_handoff', 'kind': 'verified_heads_handoff',
        'requires': ['runtime_qualification'], 'parameters': {},
        'inputs': {'warmup_checkpoint': checkpoint,
                   'warmup_manifest': 'checkpoint-manifest',
                   'warmup_heads': 'serving-heads',
                   'warmup_runtime_report': 'trained-runtime'},
    })
    recipe['stages'].append({
        'id': 'core_text_warmup', 'kind': 'core_text_warmup',
        'requires': ['runtime_qualification', 'trained_heads_handoff'],
        'parameters': {},
        'inputs': {'records': 'raw-records', 'pieces': 'native-pieces',
                   'heads': 'serving-heads', 'continue_from': checkpoint},
    })
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    loaded = load_recipe(path)
    warmup = resolve_stage_inputs(loaded, loaded['stages'][-1], {})
    assert warmup['continue_from']['sha256'] == sha(artifacts['full-checkpoint'])
    assert stage_input_args(warmup, 'core_text_warmup')[-2:] == [
        '--continue-from', str(artifacts['full-checkpoint'].resolve())]

    recipe['stages'][-1]['inputs']['heads'] = 'raw-records'
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='exact files validated'):
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
    assert 'memory_gb' not in parameters and 'graph_memory_gb' not in parameters
    assert parameters['sketch_gradient'] == 'local_stage'
    assert parameters['sketch_target_weight'] == 0.1
    assert parameters['sketch_target_backbone_scale'] == 0.05


def test_shared_text_recipe_uses_one_mapped_graph_and_inherits_common_parameters():
    from natlang_neuralese.train.recipe import effective_stage_parameters
    recipe=load_recipe(Path(__file__).parents[2]/'training/neuralese/recipes/raw-recurrence-v1.json')
    by_id={stage['id']:stage for stage in recipe['stages']}
    core=effective_stage_parameters(by_id['core_text_warmup'],[],recipe)
    predecessor={'id':'core_text_warmup','kind':'core_text_warmup','artifact':'/run/core/heads.pt',
                 'gate':{'step':22272}}
    ar=effective_stage_parameters(by_id['autoregressive_text_fixup'],[predecessor],recipe)
    assert core['steps']==4096 and ar['steps']==23296
    assert ar['ar_feedback_fixup'] is True
    assert core['input_map_kernel']==ar['input_map_kernel']==4
    assert core['input_map_rank']==ar['input_map_rank']==64
    assert core['sketch_weight']==core['embedding_weight']==1.
    assert core['sketch_lr']>core['lr']
    assert 'neuralese_input' not in core and 'rollout_passes' not in core
    assert by_id['core_text_warmup']['parameters']=={'steps':4096}
    assert by_id['autoregressive_text_fixup']['parameters']=={
        'steps':1024,'ar_feedback_fixup':True,'channel_consistency_weight':1.0,'tokens':16384,'target_tokens':256}


def test_active_core_text_recipe_has_no_objective_choice_or_sketch_rollout(tmp_path):
    recipe=declared()
    recipe['stages'].append({'id':'core_text_warmup','kind':'core_text_warmup',
        'requires':['runtime_qualification'],'parameters':{'input_map_kernel':4,'input_map_rank':64}})
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe))
    assert load_recipe(path)['stages'][-1]['parameters']['input_map_kernel']==4
    recipe['stages'][-1]['parameters']={'neuralese_input':'sketch'}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='unknown stage parameters'):
        load_recipe(path)
    recipe['stages'][-1]['parameters']={'rollout_passes':4}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='unknown stage parameters'):
        load_recipe(path)


def test_active_text_warmup_cli_does_not_expose_legacy_sketch_objectives():
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path(__file__).parents[2]/'training/neuralese')
    help_text=subprocess.check_output([sys.executable,'-m','natlang_neuralese.train.text_warmup','--help'],env=env,text=True)
    for obsolete in ('--neuralese-input','--rollout-passes','--rollout-start-passes','--max-sequence-passes'):
        assert obsolete not in help_text
    assert '--ar-feedback-fixup' in help_text


def test_raw_recipe_declares_only_mapped_and_ar_feedback_text_stages():
    recipe=load_recipe(Path(__file__).parents[2]/'training/neuralese/recipes/raw-recurrence-v1.json')
    warmups=[stage for stage in recipe['stages'] if stage['kind']=='core_text_warmup']
    assert [stage['id'] for stage in warmups]==['core_text_warmup','autoregressive_text_fixup']
    assert all('neuralese_input' not in stage['parameters'] for stage in warmups)
    assert all('rollout_passes' not in stage['parameters'] for stage in warmups)


def test_child_signal_forwarder_delivers_sigterm_and_restores_parent_handler(tmp_path):
    from natlang_neuralese.train.recipe import _ChildSignalForwarder

    ready, received = tmp_path / 'ready', tmp_path / 'received'
    child_code = (
        'import pathlib,signal,sys,time; '
        'ready=pathlib.Path(sys.argv[1]); received=pathlib.Path(sys.argv[2]); '
        'signal.signal(signal.SIGTERM, lambda *_: (received.write_text("term"), sys.exit(0))); '
        'ready.write_text("ready"); time.sleep(30)'
    )
    forwarder = _ChildSignalForwarder()
    previous = signal.getsignal(signal.SIGTERM)
    forwarder.install()

    def send_after_child_ready():
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGTERM)

    sender = threading.Thread(target=send_after_child_ready)
    try:
        sender.start()
        result = forwarder.run([sys.executable, '-c', child_code, str(ready), str(received)])
        sender.join(timeout=3)
        assert not sender.is_alive()
        assert forwarder.stopped == [True]
        assert result.returncode == 0
        assert received.read_text() == 'term'
    finally:
        forwarder.close()
    assert signal.getsignal(signal.SIGTERM) == previous


def test_child_signal_forwarder_handles_signal_before_child_attachment(tmp_path):
    from natlang_neuralese.train.recipe import _ChildSignalForwarder

    ready, received = tmp_path / 'ready', tmp_path / 'received'
    child_code = (
        'import pathlib,signal,sys,time; '
        'ready=pathlib.Path(sys.argv[1]); received=pathlib.Path(sys.argv[2]); '
        'signal.signal(signal.SIGTERM, lambda *_: (received.write_text("term"), sys.exit(0))); '
        'ready.write_text("ready"); time.sleep(30)'
    )
    forwarder = _ChildSignalForwarder()
    forwarder.install()
    child = None
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        child = subprocess.Popen([sys.executable, '-c', child_code, str(ready), str(received)])
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        forwarder.attach(child)
        assert child.wait(timeout=3) == 0
        assert received.read_text() == 'term'
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=3)
        forwarder.close()


def test_child_signal_forwarder_restores_handlers_on_exception():
    from natlang_neuralese.train.recipe import _ChildSignalForwarder

    previous = signal.getsignal(signal.SIGTERM)
    forwarder = _ChildSignalForwarder()
    with pytest.raises(RuntimeError, match='stage failed'):
        with forwarder:
            assert signal.getsignal(signal.SIGTERM) == forwarder._handle
            raise RuntimeError('stage failed')
    assert signal.getsignal(signal.SIGTERM) == previous


def test_child_signal_forwarder_tolerates_exit_between_poll_and_signal():
    from natlang_neuralese.train.recipe import _ChildSignalForwarder

    class ExitsDuringSignal:
        def poll(self):
            return None

        def send_signal(self, _signal):
            raise ProcessLookupError

    forwarder = _ChildSignalForwarder()
    forwarder.child[0] = ExitsDuringSignal()
    forwarder._handle(signal.SIGTERM, None)
    assert forwarder.stopped == [True]
    forwarder.attach(ExitsDuringSignal())


def test_registered_artifacts_bind_by_id_and_are_hash_checked(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import save_file
    from natlang_neuralese import artifacts
    from natlang_neuralese.train.recipe import resolve_stage_inputs, validate_input_bindings

    (tmp_path / 'training').mkdir()
    source = tmp_path / 'bank.nz'
    header = {'exports': {'p': {'type': 'Neuralese<string>', 'value': {'$neuralese': {'id': 'b'}}}},
              'blocks': {'b': {'dialect': 'nd:test@1'}}}
    save_file({'b': torch.zeros(2, 4)}, str(source), metadata={'natlang': json.dumps(header)})
    artifacts.register(tmp_path, {'id': 'bank-recipe-test-v1', 'kind': 'prompt-bank', 'dialect': 'nd:test@1',
                                  'backbone': {'model': 'm', 'revision': 'r'}, 'init': {'method': 'text'},
                                  'qualification': {'status': 'unqualified'}}, {'bank.nz': source})
    monkeypatch.setattr(artifacts, 'REPO', tmp_path)
    recipe = declared()
    recipe['input_bindings'] = {'bank': {'artifact': 'bank-recipe-test-v1'}}
    recipe['stages'][0]['inputs'] = {'records': 'bank'}
    validate_input_bindings(recipe)
    resolved = resolve_stage_inputs(recipe, recipe['stages'][0], {})['records']
    assert resolved['artifact'] == 'bank-recipe-test-v1' and resolved['path'].endswith('bank-recipe-test-v1/bank.nz')
    assert resolved['sha256'] == artifacts.digest(source)
    recipe['input_bindings'] = {'bank': {'artifact': 'bank-recipe-test-v1', 'sha256': 'x'}}
    with pytest.raises(ValueError, match='only artifact'):
        validate_input_bindings(recipe)
    recipe['input_bindings'] = {'bank': {'artifact': 'missing-artifact-v1'}}
    with pytest.raises(ValueError, match='unknown artifact'):
        resolve_stage_inputs(recipe, recipe['stages'][0], {})


def test_relative_binding_paths_resolve_against_the_repository_root_not_the_working_directory(tmp_path, monkeypatch):
    import hashlib
    from natlang_neuralese.common import paths
    from natlang_neuralese.train import recipe as runner
    data = tmp_path / 'repo' / 'data' / 'records.jsonl'
    data.parent.mkdir(parents=True)
    data.write_text('{}\n')
    monkeypatch.setattr(paths, 'root', lambda name: tmp_path / 'repo')
    monkeypatch.chdir(tmp_path)  # a runner started elsewhere (launch.sh: training/neuralese)
    assert runner.binding_path('data/records.jsonl') == data.resolve()
    assert runner.binding_path(str(data)) == data.resolve()
    declared = {'input_bindings': {'records': {'path': 'data/records.jsonl',
                                               'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}},
                'stages': []}
    stage = {'id': 'warm', 'kind': 'core_text_warmup', 'inputs': {'records': 'records', 'pieces': 'records'}}
    resolved = runner.resolve_stage_inputs(declared, stage, {})
    assert resolved['records']['path'] == str(data.resolve())
