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

import legacy_recipes


def test_started_native_append_adds_bound_finite_foundation_and_keeps_predecessor_guards(tmp_path):
    from natlang_neuralese.train import recipe
    from copy import deepcopy
    bound=lambda name,digest:{'path':'/data/'+name,'sha256':digest*64,'binding':name}
    old={'native_graphs':bound('old','a'),'held_native_graphs':bound('held','b')}
    new={**old,'native_graphs':bound('new','c'),'continued_native_graphs':old['native_graphs'],
         'foundation_native_graphs':bound('complete-pair-cohort','d')}
    assert recipe._data_continuation(old,new)
    wrong={**new,'continued_native_graphs':bound('wrong','e')}
    assert not recipe._data_continuation(old,wrong)
    later={**new,'native_graphs':bound('next','f'),'continued_native_graphs':new['native_graphs']}
    assert recipe._data_continuation(new,later)
    assert not recipe._data_continuation(new,{**later,'foundation_native_graphs':bound('weaker','e')})
    before={'stages':[{'id':'full','kind':'native_state_sft','parameters':{'epochs':1}}]}
    after=deepcopy(before);after['stages'][0]['parameters']['communication_foundation']=True
    changes,left,right=recipe.direct_windowing_handoffs(before,after,tmp_path)
    assert changes['full']['communication_foundation']==[None,True]
    assert left==right


def test_compressed_recipe_input_checks_decoder_before_launch(monkeypatch):
    from natlang_neuralese.train import recipe

    inputs = {'direct_warmup': {'text_data': {'path': '/data/train.jsonl.zst', 'sha256': 'a' * 64}}}
    monkeypatch.setattr(recipe.shutil, 'which', lambda name: None)
    with pytest.raises(RuntimeError, match='zstd CLI is required.*train.jsonl.zst'):
        recipe.require_input_decoders(inputs)
    recipe.require_input_decoders({'held': {}})
    monkeypatch.setattr(recipe.shutil, 'which', lambda name: '/usr/bin/zstd')
    recipe.require_input_decoders(inputs)


def test_explicit_reasoning_row_upgrade_is_a_supported_native_continuation(tmp_path):
    from copy import deepcopy
    from natlang_neuralese.train import recipe
    before={'stages':[{'id':'full','kind':'native_state_sft',
        'parameters':{'epochs':1,'communication_foundation':True}}]}
    after=deepcopy(before)
    after['stages'][0]['parameters']['learn_reasoning_boundaries']=True
    changes,left,right=recipe.direct_windowing_handoffs(before,after,tmp_path)
    assert changes=={'full':{'learn_reasoning_boundaries':[None,True]}}
    assert left==right
    assert 'learn_reasoning_boundaries' in recipe.HANDLERS['native_state_sft']['parameters']
    assert '--learn-reasoning-boundaries' in recipe.stage_parameter_args(after['stages'][0]['parameters'])
    (tmp_path/'full-report.json').write_text('{}')
    with pytest.raises(ValueError,match='finished stage'):
        recipe.direct_windowing_handoffs(before,after,tmp_path)


@pytest.mark.parametrize('enabled',[False,True])
def test_native_boolean_recipe_flags_parse_both_declared_values(tmp_path,enabled):
    from natlang_neuralese.train.native_state_sft import build_parser
    parameters={'refresh_at_checkpoint':enabled,'communication_foundation':enabled,
                'learn_reasoning_boundaries':enabled}
    argv=['--native-graphs',str(tmp_path/'train'),'--held-native-graphs',str(tmp_path/'held'),
          '--out',str(tmp_path/'out'),*stage_parameter_args(parameters)]
    parsed=build_parser().parse_args(argv)
    assert all(getattr(parsed,name) is enabled for name in parameters)


@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl'])
def test_guarded_native_arithmetic_is_operational_tuning_not_a_new_lineage(tmp_path,kind):
    from copy import deepcopy
    from natlang_neuralese.train import recipe
    before={'stages':[{'id':'full','kind':kind,'parameters':{
        'native_linear_precision':'ieee','base':'/pinned-base'}}]}
    after=deepcopy(before)
    after['stages'][0]['parameters']['native_linear_precision']='bf16x6'
    changes,left,right=recipe.direct_windowing_handoffs(before,after,tmp_path)
    assert changes=={'full':{'native_linear_precision':['ieee','bf16x6']}}
    assert left==right
    after['stages'][0]['parameters']['base']='/different-base'
    _,left,right=recipe.direct_windowing_handoffs(before,after,tmp_path)
    assert left!=right  # Arithmetic permission never licenses a backbone replacement.


def test_native_shared_controller_is_a_headless_recipe_stage(tmp_path):
    from natlang_neuralese.train.recipe import (HANDLERS, HEADLESS_NATIVE_KINDS,
        NATIVE_IDENTITY_QUALIFICATION, SHARED_NATIVE_KINDS, stage_input_args,
        validate_shared_controller_output)
    assert 'native_shared_stage_controller' in HEADLESS_NATIVE_KINDS
    assert 'native_shared_stage_controller' in SHARED_NATIVE_KINDS
    assert HANDLERS['native_shared_stage_controller']['required_inputs'] == {'controller_config'}
    assert HANDLERS['native_shared_stage_controller']['module'] == 'natlang_neuralese.train.native_readiness'
    assert NATIVE_IDENTITY_QUALIFICATION == 'native_state_runtime_qualified'
    assert stage_input_args({'controller_config': {'path': '/bound/controller.json',
        'sha256': 'a' * 64}}, 'native_shared_stage_controller') == [
            '--config', '/bound/controller.json']
    path = tmp_path/'controller.json'
    output = tmp_path/'shared-stage'
    path.write_text(json.dumps({'schema':'natlang.native-shared-stage-controller/1', 'out':str(output)}))
    validate_shared_controller_output(path, output)
    with pytest.raises(ValueError, match='output must equal'):
        validate_shared_controller_output(path, tmp_path/'other-stage')


def declared():
    return legacy_recipes.foundation()


def test_direct_training_clock_is_declarable_in_shared_recipes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    recipe = load_recipe(root / 'training/neuralese/recipes/diag-mixed-lfm-u10006-p25.json')
    next(s for s in recipe['stages'] if s['id'] == 'direct_sft')['parameters']['stop_after_training_seconds'] = 120.0
    path = tmp_path / 'clock.json'
    path.write_text(json.dumps(recipe))
    loaded = load_recipe(path)
    parameters = next(s for s in loaded['stages'] if s['id'] == 'direct_sft')['parameters']
    argv = stage_parameter_args(parameters)
    index = argv.index('--stop-after-training-seconds')
    assert argv[index + 1] == '120.0'


def legacy_raw_recurrence(tmp_path):
    """The inline legacy raw-recurrence lineage, loaded through a file (load_recipe reads paths)."""
    path = tmp_path / 'legacy-raw-recurrence.json'
    path.write_text(json.dumps(legacy_recipes.raw_recurrence()))
    return load_recipe(path)


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

    recipe = legacy_raw_recurrence(tmp_path)
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
    recipe = legacy_raw_recurrence(tmp_path)
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


def test_raw_recurrence_recipe_hands_fixup_heads_to_runtime_and_recurrence(tmp_path):
    recipe=legacy_raw_recurrence(tmp_path)
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
            stage_out.mkdir(parents=True, exist_ok=True)
            (stage_out / 'report.json').write_text(json.dumps({'qualified': True}))
            (stage_out / 'heads.pt').write_bytes(b'head artifact')

        def wait(self, timeout=None):
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

        def wait(self, timeout=None):
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
    recipe = legacy_recipes.raw_recurrence()
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


def test_shared_text_recipe_uses_one_mapped_graph_and_inherits_common_parameters(tmp_path):
    from natlang_neuralese.train.recipe import effective_stage_parameters
    recipe=legacy_raw_recurrence(tmp_path)
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
        # To the main thread: a process-directed kill may land on any thread, and Python runs the handler only when
        # the main thread next executes bytecode, which it does not while blocked in the child's waitpid (strace,
        # 2026-10-13: delivered to a worker thread, forwarded only after the child slept its 30 s and exited 0).
        signal.pthread_kill(threading.main_thread().ident, signal.SIGTERM)

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


def test_runner_forwards_an_outside_sigterm_to_its_child_within_two_seconds(tmp_path):
    """docker stop's SIGTERM reaches the recipe runner (container PID 1) while its main thread waits on the stage
    child. When the kernel hands the signal to another thread, a blocking wait() ran the handler only after the child
    exited, so the trainer never checkpointed (Mellum v4, 2026-10-13). The runner here sends every SIGTERM to a worker
    thread (the main thread blocks it, as it may when busy in native code) and must still forward within 2 s."""
    ready, received = tmp_path / 'ready', tmp_path / 'received'
    child_code = (
        'import pathlib,signal,sys,time; '
        'ready=pathlib.Path(sys.argv[1]); received=pathlib.Path(sys.argv[2]); '
        'signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM}); '  # a child inherits the runner's mask
        'signal.signal(signal.SIGTERM, lambda *_: (received.write_text(repr(time.time())), sys.exit(0))); '
        'ready.write_text("ready"); time.sleep(60)'
    )
    runner_code = (
        'import signal,sys,threading,time\n'
        'from natlang_neuralese.train.recipe import _ChildSignalForwarder\n'
        'threading.Thread(target=time.sleep,args=(120,),daemon=True).start()\n'
        'signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})\n'
        'with _ChildSignalForwarder() as forwarder:\n'
        '    result=forwarder.run([sys.executable,"-c",sys.argv[1],sys.argv[2],sys.argv[3]])\n'
        'sys.exit(0 if forwarder.stopped==[True] and result.returncode==0 else 3)\n'
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [
        str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'), os.environ.get('PYTHONPATH')])))
    runner = subprocess.Popen([sys.executable, '-c', runner_code, child_code, str(ready), str(received)], env=env)
    try:
        deadline = time.monotonic() + 60
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        sent = time.time()
        os.kill(runner.pid, signal.SIGTERM)
        while not received.exists() and time.time() < sent + 2:
            time.sleep(0.02)
        assert received.exists(), 'SIGTERM was not forwarded to the child within 2 s'
        assert float(received.read_text()) - sent < 2
        assert runner.wait(timeout=10) == 0
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()


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


@pytest.mark.parametrize('hourly_checkpoint', [False, True])
def test_lineage_handoff_takes_newest_code_and_operational_recipe_changes_only(tmp_path, hourly_checkpoint):
    import json as _json

    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('old = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('new = 1\n')
    (package / 'b.py').write_text('added = 1\n')
    old_recipe = {'id': 'r-v10', 'description': 'v10', 'stages': [{'id': 's', 'parameters': {'lr': 1}}],
                  'stage_parameter_defaults': {'core_text_warmup': {'latent_step_in_backward': True}}}
    new_recipe = {'id': 'r-v11', 'description': 'v11', 'extends': 'r-v10',
                  'stages': [{'id': 's', 'parameters': {'lr': 1}}],
                  'stage_parameter_defaults': {'core_text_warmup': {'latent_step_in_backward': True,
                                                                    'moe_kernel': 'grouped'},
                                               'raw_recurrence_training': {'moe_kernel': 'grouped'}}}
    existing = {'recipe': old_recipe, 'recipe_sha256': 'lineage', 'inputs': {'x': '1'}, 'device': 'cuda',
                'code': {'a.py': runner.sha(frozen / 'a.py')}}
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(_json.dumps(existing))
    plan = {'recipe': new_recipe, 'recipe_sha256': 'v11', 'inputs': {'x': '1'}, 'device': 'cuda'}
    if hourly_checkpoint:
        old_recipe['stages'][0]['kind'] = 'native_state_sft'
        new_recipe['stages'][0]['kind'] = 'native_state_sft'
        new_recipe['stages'][0]['parameters']['checkpoint_seconds'] = 3600
    updated = runner.lineage_handoff(existing, plan, frozen, package, plan_path)
    assert updated['recipe_sha256'] == 'lineage'  # stage reports keep matching the lineage
    assert updated['recipe'] == new_recipe and (frozen / 'b.py').read_text() == 'added = 1\n'
    assert set(updated['code']) == {'a.py', 'b.py'}
    record = updated['handoffs'][-1]
    assert record['recipe_id'] == 'r-v11' and record['code_changed'] == ['a.py'] and record['code_added'] == ['b.py']
    assert (out / record['superseded_runtime'] / 'natlang_neuralese' / 'a.py').read_text() == 'old = 1\n'
    assert _json.loads(plan_path.read_text()) == updated
    changed = dict(new_recipe, stages=[{'id': 's', 'parameters': {'lr': 2}}])
    with pytest.raises(ValueError, match='more than operational'):
        runner.lineage_handoff(updated, dict(plan, recipe=changed), frozen, package, plan_path)
    with pytest.raises(ValueError, match='inputs changed'):
        runner.lineage_handoff(updated, dict(plan, inputs={'x': '2'}), frozen, package, plan_path)


def test_lineage_handoff_allows_pinned_code_memory_and_retired_readiness_change(tmp_path):
    import json as _json

    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('old = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('new = 1\n')
    (package / 'b.py').write_text('added = 1\n')
    old_code = {str(path.relative_to(frozen)): runner.sha(path) for path in frozen.rglob('*.py')}
    new_code = {str(path.relative_to(package)): runner.sha(path) for path in package.rglob('*.py')}
    old_recipe = {'id': 'r-v10', 'runtime': {'image': 'sha256:' + '1' * 64, 'package_code': old_code},
                  'stages': [{'id': 's', 'parameters': {'lr': 1, 'memory_gb': 9}}]}
    old_recipe['parent_readiness'] = {'held_ce_plateau': {'evaluations': 3}}
    new_recipe = {'id': 'r-v10', 'runtime': {'image': 'sha256:' + '1' * 64, 'package_code': new_code},
                  'stages': [{'id': 's', 'parameters': {'lr': 1, 'memory_gb': 16}}]}
    existing = {'recipe': old_recipe, 'recipe_sha256': 'lineage', 'inputs': {'x': '1'}, 'device': 'cuda',
                'code': old_code}
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(_json.dumps(existing))
    plan = {'recipe': new_recipe, 'recipe_sha256': 'new-recipe-hash', 'inputs': {'x': '1'}, 'device': 'cuda'}

    updated = runner.lineage_handoff(existing, plan, frozen, package, plan_path)

    assert updated['recipe_sha256'] == 'lineage'
    assert updated['recipe']['runtime']['package_code'] == new_code
    assert updated['recipe']['stages'][0]['parameters']['memory_gb'] == 16
    assert updated['code'] == new_code


def test_active_native_data_handoff_binds_both_exact_predecessors(tmp_path):
    import copy
    from natlang_neuralese.train import recipe as runner
    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('old = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('new = 1\n')
    (out / 'native').mkdir()

    def plan(appended=False):
        roles = {'native_graphs': 'train', 'held_native_graphs': 'held'}
        bindings = {name: {'path': '/' + name + ('-union' if appended else ''),
                           'sha256': name + ('-union' if appended else '')} for name in ('train', 'held')}
        if appended:
            for role, name in list(roles.items()):
                roles['continued_' + role] = 'prior-' + name
                bindings['prior-' + name] = {'path': '/' + name, 'sha256': name}
        resolved = {role: {**bindings[name], 'binding': name} for role, name in roles.items()}
        return {'recipe': {'id': 'native', 'input_bindings': bindings, 'stages': [
            {'id': 'native', 'kind': 'native_state_sft', 'inputs': roles, 'parameters': {'epochs': 1}}]},
            'recipe_sha256': 'current', 'device': 'cuda', 'stage_inputs': {'native': resolved},
            'inputs': {item['path']: item['sha256'] for item in resolved.values()}}

    existing = {**plan(), 'code': {'a.py': runner.sha(frozen / 'a.py')}}
    path = out / 'recipe-plan.json'
    candidate = plan(True)
    candidate['recipe']['stages'][0]['parameters'].update(
        epochs=2, batch_tokens=32768, chunk_tokens=1024, learning_rate=6e-6, weight_decay=.05, latent_ns='cubic5')
    result = runner.lineage_handoff(copy.deepcopy(existing), candidate, frozen, package, path, out, dry_run=True)
    assert result['handoffs'][-1]['stage_input_changes']['native']['kind'] == 'data_continuation'
    assert result['handoffs'][-1]['windowing_changes']['native']['epochs'] == [1, 2]
    assert result['handoffs'][-1]['windowing_changes']['native']['chunk_tokens'] == [None, 1024]
    invalid = copy.deepcopy(candidate)
    invalid['stage_inputs']['native']['continued_held_native_graphs']['sha256'] = 'unrelated-held'
    with pytest.raises(ValueError, match='started stage'):
        runner.lineage_handoff(copy.deepcopy(existing), invalid, frozen, package, path, out, dry_run=True)


def test_shared_lineage_handoff_allows_direct_stage_epoch_change_with_other_parameters_pinned(tmp_path):
    import json as _json

    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('old = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('new = 1\n')
    old_code = {str(path.relative_to(frozen)): runner.sha(path) for path in frozen.rglob('*.py')}
    new_code = {str(path.relative_to(package)): runner.sha(path) for path in package.rglob('*.py')}
    old_recipe = {
        'id': 'direct-lineage', 'runtime': {'image': 'sha256:' + '1' * 64, 'package_code': old_code},
        'stage_parameter_defaults': {'neuralese_direct': {'epochs': 2.0}},
        'stages': [{'id': 'header', 'kind': 'neuralese_direct',
                    'parameters': {'batch': 4, 'eval_every': 100}}],
    }
    new_recipe = {
        'id': 'direct-lineage', 'runtime': {'image': 'sha256:' + '1' * 64, 'package_code': new_code},
        'stage_parameter_defaults': {'neuralese_direct': {'epochs': 1.0}},
        'stages': [{'id': 'header', 'kind': 'neuralese_direct',
                    'parameters': {'batch': 4, 'eval_every': 50}}],
    }
    existing = {'recipe': old_recipe, 'recipe_sha256': 'lineage', 'inputs': {'x': '1'}, 'device': 'cuda',
                'code': old_code}
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(_json.dumps(existing))
    plan = {'recipe': new_recipe, 'recipe_sha256': 'new-recipe-hash', 'inputs': {'x': '1'}, 'device': 'cuda'}

    updated = runner.lineage_handoff(existing, plan, frozen, package, plan_path)
    assert updated['recipe_sha256'] == 'lineage'
    assert updated['recipe']['stage_parameter_defaults']['neuralese_direct']['epochs'] == 1.0
    assert updated['handoffs'][-1]['epoch_budget_changes'] == {
        'header': {
            'old_epochs': 2.0,
            'new_epochs': 1.0,
            'verification': 'trainer-reconstructs-exact-schedule-prefix-and-checkpoint-step',
        }
    }

    changed = _json.loads(_json.dumps(updated['recipe']))
    changed['stages'][0]['parameters']['batch'] = 8
    with pytest.raises(ValueError, match='changes more than operational stage parameters'):
        runner.recipe_handoff_epoch_changes(updated['recipe'], changed)


def test_lineage_handoff_rejects_changed_objective_or_stale_frozen_code(tmp_path):
    import json as _json

    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('old = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('new = 1\n')
    old_code = {str(path.relative_to(frozen)): runner.sha(path) for path in frozen.rglob('*.py')}
    new_code = {str(path.relative_to(package)): runner.sha(path) for path in package.rglob('*.py')}
    old_recipe = {'id': 'r', 'runtime': {'image': 'sha256:' + '1' * 64, 'package_code': old_code},
                  'stages': [{'id': 's', 'parameters': {'lr': 1}}]}
    new_recipe = {'id': 'r', 'runtime': {'image': 'sha256:' + '1' * 64, 'package_code': new_code},
                  'stages': [{'id': 's', 'parameters': {'lr': 1}}]}
    existing = {'recipe': old_recipe, 'recipe_sha256': 'lineage', 'inputs': {'x': '1'}, 'device': 'cuda',
                'code': old_code}
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(_json.dumps(existing))
    plan = {'recipe': new_recipe, 'recipe_sha256': 'new-recipe-hash', 'inputs': {'x': '1'}, 'device': 'cuda'}
    changed = _json.loads(_json.dumps(plan))
    changed['recipe']['stages'][0]['parameters']['lr'] = 2
    with pytest.raises(ValueError, match='changes more than operational stage parameters'):
        runner.lineage_handoff(existing, changed, frozen, package, plan_path)

    (frozen / 'a.py').write_text('tampered = 1\n')
    with pytest.raises(ValueError, match='old frozen runtime differs'):
        runner.lineage_handoff(existing, plan, frozen, package, plan_path)


def test_lineage_handoff_extends_an_unqualified_text_warmup_with_its_quantization_ramp_held(tmp_path):
    """A handoff may raise an unqualified mapped warm-up's steps when its quantization
    update range stays (quantization_steps = the old steps), so the stage continues from its state at the stage-end
    mixes; a moved ramp, a shorter stage or an extended qualified stage is refused."""
    import json as _json

    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('x = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('x = 2\n')

    def recipe(**parameters):
        return {'id': 'r', 'stages': [{'id': 'warm', 'kind': 'core_text_warmup', 'parameters': parameters}],
                'quantization': {'stage_progress': {'warm': [0.0, 0.35]}}}
    existing = {'recipe': recipe(steps=4096), 'recipe_sha256': 'lineage', 'inputs': {}, 'device': 'cuda',
                'code': {'a.py': runner.sha(frozen / 'a.py')}}
    plan_path = out / 'recipe-plan.json'
    plan = {'recipe_sha256': 'v12', 'inputs': {}, 'device': 'cuda'}
    for refused, match in ((recipe(steps=6144), 'moves its quantization ramp'),
                           (recipe(steps=2048, quantization_steps=2048), 'only a mapped text warm-up'),
                           (recipe(steps=6144, quantization_steps=4096, lr=2), 'more than operational')):
        with pytest.raises(ValueError, match=match):
            runner.lineage_handoff(existing, dict(plan, recipe=refused), frozen, package, plan_path)
    extended = recipe(steps=6144, quantization_steps=4096)
    updated = runner.lineage_handoff(existing, dict(plan, recipe=extended), frozen, package, plan_path)
    assert updated['handoffs'][-1]['extended'] == {'warm': [4096, 6144]} and updated['recipe_sha256'] == 'lineage'
    # The stage's quantization component keeps its update range, so the trainer's resume identity holds.
    stage = extended['stages'][0]
    spec = {'stage_progress': {'warm': [0.0, 0.35]}, 'points': []}
    component = runner.stage_quantization({'quantization': spec}, stage, stage['parameters'], [])
    assert component['stage'] == {'id': 'warm', 'progress': [0.0, 0.35], 'start': 0, 'end': 4096}
    assert '--quantization-steps' not in runner.stage_parameter_args(
        {k: v for k, v in stage['parameters'].items() if k not in runner.RUNNER_STAGE_PARAMETERS})
    (out / 'warm-report.json').write_text('{}')
    with pytest.raises(ValueError, match='already qualified'):
        runner.lineage_handoff(_json.loads(_json.dumps(existing)), dict(plan, recipe=extended), frozen, package,
                               plan_path)


def test_lineage_handoff_lets_only_an_unstarted_stage_take_new_inputs(tmp_path):
    """Mellum v4 (2026-10-12): a later phase that has not started may read new content for its declared inputs on a
    handoff (union r3 for direct_trajectory while direct_sft runs); the active or a finished phase's may not, a binding
    shared with a started phase may not change, and roles/bindings stay. The previous input map is recorded so the
    reports of earlier phases keep matching the lineage."""
    import copy
    import json as _json

    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('x = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('x = 2\n')
    (out / 'sft').mkdir()  # the active phase has started

    def recipe(trajectory_sha, header_sha='h'):
        return {'id': 'r', 'input_bindings': {'header': {'path': 'h.jsonl', 'sha256': header_sha},
                                              'trajectory': {'path': 't.jsonl', 'sha256': trajectory_sha}},
                'stages': [{'id': 'sft', 'kind': 'neuralese_direct', 'inputs': {'text_data': 'header'},
                            'parameters': {'epochs': 1.0}},
                           {'id': 'traj', 'kind': 'neuralese_direct', 'requires': ['sft'],
                            'inputs': {'text_data': 'trajectory'}, 'parameters': {'epochs': 1.0}}]}

    def plan_for(trajectory_sha, header_sha='h'):
        stage_inputs = {'sft': {'text_data': {'binding': 'header', 'path': '/d/h' + header_sha, 'sha256': header_sha}},
                        'traj': {'text_data': {'binding': 'trajectory', 'path': '/d/t' + trajectory_sha,
                                               'sha256': trajectory_sha}}}
        inputs = {'/heads': 'H', '/d/h' + header_sha: header_sha, '/d/t' + trajectory_sha: trajectory_sha}
        return {'recipe': recipe(trajectory_sha, header_sha), 'recipe_sha256': 'new', 'inputs': inputs,
                'device': 'cuda', 'stage_inputs': stage_inputs}

    existing = dict(plan_for('r2'), recipe_sha256='lineage', code={'a.py': runner.sha(frozen / 'a.py')})
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(_json.dumps(existing))

    dry = runner.lineage_handoff(copy.deepcopy(existing), plan_for('r3'), frozen, package, plan_path, out,
                                 dry_run=True)
    assert dry['stage_inputs']['traj']['text_data']['sha256'] == 'r3'
    assert (frozen / 'a.py').read_text() == 'x = 1\n' and _json.loads(plan_path.read_text()) == existing

    updated = runner.lineage_handoff(copy.deepcopy(existing), plan_for('r3'), frozen, package, plan_path, out)
    record = updated['handoffs'][-1]
    assert updated['recipe_sha256'] == 'lineage' and updated['inputs'] == plan_for('r3')['inputs']
    assert record['stage_input_changes']['traj']['new']['text_data']['sha256'] == 'r3'
    assert record['previous_inputs'] == existing['inputs'] and list(record['stage_input_changes']) == ['traj']
    assert _json.loads(plan_path.read_text()) == updated

    # The active phase's inputs stay fixed.
    (frozen / 'a.py').write_text('x = 2\n')
    (package / 'a.py').write_text('x = 3\n')
    with pytest.raises(ValueError, match='inputs of started stage sft'):
        runner.lineage_handoff(updated, plan_for('r3', header_sha='h2'), frozen, package, plan_path, out)
    # Once the later phase has started, its inputs are fixed too.
    (out / 'traj').mkdir()
    with pytest.raises(ValueError, match='inputs of started stage traj'):
        runner.lineage_handoff(updated, plan_for('r4'), frozen, package, plan_path, out)
    (out / 'traj').rmdir()
    # A binding shared with a started phase cannot change under it.
    shared = plan_for('r4')
    shared['recipe']['stages'][0]['inputs']['trajectory_too'] = 'trajectory'
    old_shared = copy.deepcopy(updated)
    old_shared['recipe']['stages'][0]['inputs']['trajectory_too'] = 'trajectory'
    with pytest.raises(ValueError, match='also feed a started or unchanged stage'):
        runner.lineage_handoff(old_shared, shared, frozen, package, plan_path, out)
    # Roles and bindings of the unstarted phase stay; only their content changes.
    renamed = plan_for('r4')
    renamed['stage_inputs']['traj']['text_data']['binding'] = 'other'
    with pytest.raises(ValueError, match='changes its input roles or bindings'):
        runner.lineage_handoff(updated, renamed, frozen, package, plan_path, out)
    # Any other recipe change is still refused.
    objective = plan_for('r4')
    objective['recipe']['stages'][1]['parameters']['lr'] = 2
    with pytest.raises(ValueError, match='more than operational'):
        runner.lineage_handoff(updated, objective, frozen, package, plan_path, out)


@pytest.mark.parametrize('previous_quantization', [None, {'stage_progress': {}, 'points': ['unchanged']}])
def test_running_recipe_can_append_future_training_stages_without_restarting(tmp_path, previous_quantization):
    import copy
    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('x = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('x = 1\n')
    (out / 'warmup').mkdir()
    old_recipe = {'id': 'r', 'stages': [{'id': 'warmup', 'kind': 'neuralese_direct',
                                        'parameters': {'epochs': 1}}]}
    old = {'recipe': old_recipe, 'recipe_sha256': 'lineage', 'device': 'cuda',
           'inputs': {'/heads': 'H'}, 'stage_inputs': {'warmup': {}},
           'code': {'a.py': runner.sha(frozen / 'a.py')}}
    plan = copy.deepcopy(old)
    plan['recipe']['input_bindings'] = {'header': {'path': 'h.jsonl', 'sha256': 'h'}}
    # Absence versus empty catalogs isn't a semantic change to an old stage.
    old['recipe']['input_bindings'] = {}
    plan['recipe']['stages'] += [
        {'id': 'sft', 'kind': 'neuralese_direct', 'requires': ['warmup'],
         'inputs': {'text_data': 'header'}, 'parameters': {'epochs': 1}},
        {'id': 'trajectory', 'kind': 'neuralese_direct', 'requires': ['sft'],
         'parameters': {'epochs': 1}}]
    plan['stage_inputs'].update(sft={'text_data': {'binding': 'header', 'path': '/d/h', 'sha256': 'h'}},
                                trajectory={})
    plan['inputs']['/d/h'] = 'h'
    if previous_quantization is not None:
        old['recipe']['quantization'] = previous_quantization
    plan['recipe']['quantization'] = {'stage_progress': {'sft': [0, 0.75], 'trajectory': [0.75, 1]},
                                      'points': ['unchanged']}
    plan['recipe']['family'] = {'stages': ['sft', 'trajectory'], 'members': {'small': 0.1}}
    path = out / 'recipe-plan.json'
    path.write_text(json.dumps(old))
    result = runner.lineage_handoff(copy.deepcopy(old), plan, frozen, package, path, out, dry_run=True)
    assert result['recipe_sha256'] == 'lineage'
    assert result['recipe']['stages'] == plan['recipe']['stages']
    assert set(result['handoffs'][-1]['appended_stages']) == {'sft', 'trajectory'}
    assert result['stage_inputs'] == plan['stage_inputs']
    assert json.loads(path.read_text()) == old
    assert not (out / 'sft').exists()
    mutated = copy.deepcopy(plan)
    mutated['recipe']['stages'][0]['parameters']['lr'] = 2
    with pytest.raises(ValueError, match='more than operational'):
        runner.lineage_handoff(copy.deepcopy(old), mutated, frozen, package, path, out, dry_run=True)
    active_family = copy.deepcopy(plan)
    active_family['recipe']['family']['stages'].append('warmup')
    with pytest.raises(ValueError, match='objective_continuation'):
        runner.lineage_handoff(copy.deepcopy(old), active_family, frozen, package, path, out, dry_run=True)
    active_quantization = copy.deepcopy(plan)
    active_quantization['recipe']['quantization']['stage_progress']['warmup'] = [0, 0.5]
    with pytest.raises(ValueError, match='objective_continuation'):
        runner.lineage_handoff(copy.deepcopy(old), active_quantization, frozen, package, path, out, dry_run=True)
    for stages in [plan['recipe']['stages'][1:], list(reversed(plan['recipe']['stages']))]:
        invalid = copy.deepcopy(plan)
        invalid['recipe']['stages'] = stages
        with pytest.raises(ValueError, match='only append'):
            runner.lineage_handoff(copy.deepcopy(old), invalid, frozen, package, path, out, dry_run=True)
    invalid = copy.deepcopy(plan)
    invalid['recipe']['stages'][1]['requires'] = ['trajectory']
    with pytest.raises(ValueError, match='must depend'):
        runner.lineage_handoff(copy.deepcopy(old), invalid, frozen, package, path, out, dry_run=True)
    (out / 'sft').mkdir()
    with pytest.raises(ValueError, match='already has output'):
        runner.lineage_handoff(copy.deepcopy(old), plan, frozen, package, path, out, dry_run=True)


def test_lineage_handoff_can_activate_only_an_unstarted_declared_stage(tmp_path):
    import copy
    import json

    from natlang_neuralese.train import recipe as runner

    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('x = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('x = 2\n')
    (out / 'sft').mkdir()

    def make_plan(admitted, trajectory_sha='new'):
        recipe = {
            'id': 'r',
            'input_bindings': {'trajectory': {'path': '/data/trajectory.jsonl', 'sha256': trajectory_sha}},
            'stages': [
                {'id': 'sft', 'kind': 'neuralese_direct', 'inputs': {'text_data': 'sft_data'},
                 'parameters': {'epochs': 1}},
                {'id': 'trajectory', 'kind': 'neuralese_direct', 'admitted': admitted,
                 'inputs': {'text_data': 'trajectory'}, 'parameters': {'epochs': 1}},
            ],
            'stage_parameter_defaults': {'neuralese_direct': {'lr': 1e-5}},
        }
        stage_inputs = {'sft': {}, 'trajectory': {}}
        inputs = {'/heads': 'H'}
        if admitted:
            recipe['quantization'] = {'stage_progress': {'trajectory': [0, 1]}, 'points': ['new']}
            stage_inputs['trajectory'] = {
                'text_data': {'binding': 'trajectory', 'path': '/data/trajectory-' + trajectory_sha,
                              'sha256': trajectory_sha}}
            inputs[stage_inputs['trajectory']['text_data']['path']] = trajectory_sha
        return {'recipe': recipe, 'recipe_sha256': 'new', 'device': 'cuda', 'inputs': inputs,
                'stage_inputs': stage_inputs}

    old = make_plan(False, 'old')
    old['recipe']['input_bindings']['trajectory']['sha256'] = 'old'
    old.update(recipe_sha256='lineage', code={'a.py': runner.sha(frozen / 'a.py')})
    plan = make_plan(True, 'new')
    path = out / 'recipe-plan.json'
    path.write_text(json.dumps(old))

    result = runner.lineage_handoff(copy.deepcopy(old), plan, frozen, package, path, out, dry_run=True)
    record = result['handoffs'][-1]
    assert result['recipe_sha256'] == 'lineage'
    assert record['activated_stages'] == ['trajectory']
    assert record['stage_input_changes']['trajectory']['kind'] == 'admitted_stage'
    assert record['stage_input_changes']['trajectory']['new']['text_data']['sha256'] == 'new'
    assert record['previous_inputs'] == old['inputs']
    assert json.loads(path.read_text()) == old
    assert not (out / 'trajectory').exists()

    # An existing run directory means the stage has crossed its safe activation boundary.
    (out / 'trajectory').mkdir()
    with pytest.raises(ValueError, match='cannot activate already-started stage trajectory'):
        runner.lineage_handoff(copy.deepcopy(old), copy.deepcopy(plan), frozen, package, path, out, dry_run=True)
    (out / 'trajectory').rmdir()

    # Activation cannot smuggle unrelated edits to an existing stage's recipe.
    changed = copy.deepcopy(plan)
    changed['recipe']['stages'][1]['parameters']['lr'] = 2e-5
    with pytest.raises(ValueError, match='more than operational'):
        runner.lineage_handoff(copy.deepcopy(old), changed, frozen, package, path, out, dry_run=True)


def test_absent_prepared_memory_request_is_omitted_from_actual_sft_cli():
    # Runtime/recipe handoffs clear this optional request after qualifying the
    # resident envelope. Neither "None" nor JSON null is a request binding.
    from natlang_neuralese.train.native_state_sft import build_parser
    argv=stage_parameter_args({'memory_qualification':None,'epochs':1})
    assert argv==['--epochs','1']
    # Parser's optional memory request must retain its actual None default.
    assert build_parser().get_default('memory_qualification') is None
