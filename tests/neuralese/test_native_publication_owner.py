import json
from pathlib import Path

import pytest

from natlang_neuralese.model.native_checkpoint_control import publication_selection
from natlang_neuralese.rl import native_publication_owner as owner


D24_ROOTS = [
    '/work/data-preparation/acquired-d24-native-20261016-v2/preference-graphs-v6/capacity-envelope-candidate-v2',
    '/work/data-preparation/d24-whole-pending-20261017-v1',
]


def _fixture(tmp_path, *, pointer=None):
    runtime = Path(__file__).resolve().parents[2]
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'complete optimizer parent')
    output = tmp_path / 'rounds' / 'lfm' / 'cycles' / '000004' / 'supervised' / '001'
    output.mkdir(parents=True)
    selection = publication_selection(namespace='/work/native-production', readonly_input_roots=D24_ROOTS)
    command = ['/usr/bin/python', '-u', '-m', owner.TRAINER_MODULE,
        '--out', str(output / 'learner'), '--continue-from', str(parent),
        '--checkpoint-publication', '--checkpoint-publication-namespace', selection['namespace'],
        '--checkpoint-publication-readonly-input-roots', json.dumps(selection['readonly_input_roots'])]
    config = {'native_participant': 'lfm', 'runtime': str(runtime), 'checkpoint': str(parent),
        'native_checkpoint_publication': selection,
        'native_checkpoint_publication_owner_pointer': str(pointer or tmp_path / 'control' / 'lfm-current-owner.json'),
        'production_stage': 'native_full_sft', 'recipe_lineage_sha256': 'a' * 64,
        'request_id': 'round-4-supervised-1'}
    return config, command, output


def _raw_fixture(tmp_path):
    runtime = Path(__file__).resolve().parents[2]
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'complete optimizer parent')
    output = tmp_path / 'rounds' / 'lfm' / 'cycles' / '000004' / 'retired-attempt-000001' / 'multi-rft' / '000'
    writer = output / 'learner-kind'
    writer.mkdir(parents=True)
    selection = publication_selection(namespace='/work/native-production', readonly_input_roots=D24_ROOTS)
    command = ['env', f'PYTHONPATH={runtime}/training/neuralese:{runtime}', f'NATLANG_REPO={runtime}',
        '/usr/bin/python3.11', '-m', owner.TRAINER_MODULE,
        '--out', str(writer), '--continue-from', str(parent), '--checkpoint-publication',
        '--checkpoint-publication-namespace', selection['namespace'],
        '--checkpoint-publication-readonly-input-roots', json.dumps(selection['readonly_input_roots'])]
    config = {'model': 'lfm', 'native_actor_mode': 'raw-state', 'runtime': '/work/native-production/task-runtimes/ts-host',
        'native_checkpoint_publication_role': 'lfm', 'native_checkpoint_publication_runtime_root': str(runtime),
        'checkpoint': str(parent), 'native_checkpoint_publication': selection,
        'native_checkpoint_publication_owner_pointer': str(tmp_path / 'control' / 'lfm-current-owner.json'),
        'production_stage': 'native_full_sft', 'recipe_lineage_sha256': 'b' * 64,
        'request_id': 'raw-round-4-supervised-1'}
    return config, command, output, writer, runtime


def test_raw_owner_binds_explicit_role_python_runtime_and_nested_writer(tmp_path):
    config, command, output, writer, runtime = _raw_fixture(tmp_path)
    published = owner.emit_current(config, command, out=output,
        selected_checkpoint=config['checkpoint'],
        runtime_root=config['native_checkpoint_publication_runtime_root'])
    manifest, _ = owner.read_owner(config['native_checkpoint_publication_owner_pointer'], role='lfm')
    spec = manifest['publication_spec']
    assert published['generation_sha256'] == manifest['generation_sha256']
    assert manifest['operation']['model_role'] == 'lfm'
    assert 'participant' not in manifest['operation']
    assert manifest['operation']['orchestration_runtime'] == config['runtime']
    assert spec['runtime'] == str(runtime.resolve())
    assert spec['effective_argv']['writer_output'] == str(writer.resolve())
    assert spec['effective_argv']['runtime_from_command_env'] is True
    assert spec['effective_argv']['allowed_output_root'] == str(output.resolve())
    assert published['qualification_credit'] is False


@pytest.mark.parametrize('change,pattern', [
    ('role', 'differs from the configured model'),
    ('runtime', 'differs from the expanded trainer environment'),
])
def test_raw_owner_rejects_misbound_operational_metadata(tmp_path, change, pattern):
    config, command, output, _, runtime = _raw_fixture(tmp_path)
    if change == 'role':
        config['native_checkpoint_publication_role'] = 'mellum'
    else:
        config['native_checkpoint_publication_runtime_root'] = str(tmp_path)
    with pytest.raises(ValueError, match=pattern):
        owner.emit_current(config, command, out=output,
            selected_checkpoint=config['checkpoint'],
            runtime_root=config['native_checkpoint_publication_runtime_root'])


def test_export_verifier_accepts_actual_env_prefixed_raw_command_and_rejects_escape(tmp_path):
    from scripts.coreweave.owned_publication_export import verify_owned_publication_command
    config, command, output, writer, runtime = _raw_fixture(tmp_path)
    owner.emit_current(config, command, out=output, selected_checkpoint=config['checkpoint'],
        runtime_root=config['native_checkpoint_publication_runtime_root'])
    manifest, _ = owner.read_owner(config['native_checkpoint_publication_owner_pointer'], role='lfm')
    spec = manifest['publication_spec']
    assert verify_owned_publication_command(command, spec, owner._trainer_invocation) == owner.TRAINER_MODULE
    bad = list(command)
    bad[bad.index('--out') + 1] = str(tmp_path / 'escape' / 'learner')
    with pytest.raises(ValueError, match='bounded writer output'):
        verify_owned_publication_command(bad, spec, owner._trainer_invocation)
    bad = list(command)
    env_index = next(i for i, value in enumerate(bad) if value.startswith('NATLANG_REPO='))
    bad[env_index] = f'NATLANG_REPO={tmp_path}'
    with pytest.raises(ValueError, match='actual trainer environment'):
        verify_owned_publication_command(bad, spec, owner._trainer_invocation)


def test_emits_and_reads_exact_owner_generation_with_narrow_external_roots(tmp_path):
    config, command, output = _fixture(tmp_path)
    published = owner.emit_current(config, command, out=output, selected_checkpoint=config['checkpoint'])
    manifest, reference = owner.read_owner(config['native_checkpoint_publication_owner_pointer'], role='lfm')
    assert published['generation_sha256'] == manifest['generation_sha256']
    assert reference == published['manifest']
    assert manifest['publication_spec']['publication']['readonly_input_roots'] == D24_ROOTS
    assert manifest['publication_spec']['parent']['path'] == config['checkpoint']
    assert manifest['operation']['request_id'] == config['request_id']
    assert manifest['operation']['out'] == str(output.resolve())
    assert manifest['publication_spec']['effective_argv']['allowed_output_root'] == str(output.resolve())
    assert published['qualification_credit'] is False


@pytest.mark.parametrize('existing', [
    {'schema': 'wrong/schema', 'role': 'lfm'},
    {'schema': owner.POINTER_SCHEMA, 'role': 'mellum'},
])
def test_emitter_refuses_to_overwrite_pointer_for_another_role_or_schema(tmp_path, existing):
    pointer = tmp_path / 'control' / 'current-owner.json'
    pointer.parent.mkdir()
    original = json.dumps(existing).encode()
    pointer.write_bytes(original)
    config, command, output = _fixture(tmp_path, pointer=pointer)
    with pytest.raises(ValueError, match='another role/schema'):
        owner.emit_current(config, command, out=output, selected_checkpoint=config['checkpoint'])
    assert pointer.read_bytes() == original


def test_each_supervised_quantum_gets_a_fresh_immutable_generation(tmp_path):
    config, command, output = _fixture(tmp_path)
    first = owner.emit_current(config, command, out=output, selected_checkpoint=config['checkpoint'])
    second_output = tmp_path / 'rounds' / 'lfm' / 'cycles' / '000004' / 'supervised' / '002'
    second_output.mkdir(parents=True)
    second_parent = tmp_path / 'second-parent.pt'
    second_parent.write_bytes(b'next complete optimizer parent')
    second_config = {**config, 'checkpoint': str(second_parent), 'request_id': 'round-4-supervised-2'}
    second_command = list(command)
    second_command[second_command.index('--continue-from') + 1] = str(second_parent)
    second_command[second_command.index('--out') + 1] = str(second_output / 'learner')
    second = owner.emit_current(second_config, second_command, out=second_output, selected_checkpoint=second_parent)
    assert first['generation_sha256'] != second['generation_sha256']
    assert Path(first['manifest']['path']).is_file()
    current, _ = owner.read_owner(config['native_checkpoint_publication_owner_pointer'], role='lfm')
    assert current['generation_sha256'] == second['generation_sha256']
    assert json.loads(Path(first['manifest']['path']).read_text())['generation_sha256'] == first['generation_sha256']


def test_rejects_parent_and_readonly_selection_divergence(tmp_path):
    config, command, output = _fixture(tmp_path)
    bad = list(command)
    bad[bad.index('--continue-from') + 1] = str(tmp_path / 'other.pt')
    with pytest.raises(ValueError, match='parent differs'):
        owner.emit_current(config, bad, out=output, selected_checkpoint=config['checkpoint'])
    bad = list(command)
    bad[bad.index('--checkpoint-publication-readonly-input-roots') + 1] = '[]'
    with pytest.raises(ValueError, match='selection differs'):
        owner.emit_current(config, bad, out=output, selected_checkpoint=config['checkpoint'])


def test_owner_pointer_replacement_race_is_rejected(tmp_path, monkeypatch):
    config, command, output = _fixture(tmp_path)
    owner.emit_current(config, command, out=output, selected_checkpoint=config['checkpoint'])
    pointer = Path(config['native_checkpoint_publication_owner_pointer'])
    old = pointer.read_bytes()
    original = owner.Path.read_bytes
    done = {'value': False}

    def replace_after_manifest(self):
        value = original(self)
        if self == pointer and not done['value']:
            done['value'] = True
            pointer.write_bytes(old + b' ')
        return value

    monkeypatch.setattr(owner.Path, 'read_bytes', replace_after_manifest)
    with pytest.raises(ValueError, match='pointer changed'):
        owner.read_owner(pointer, role='lfm')


def test_retry_uses_live_selected_parent_and_content_addressed_command_in_same_output(tmp_path):
    config, command, output = _fixture(tmp_path)
    selected = tmp_path / 'selected-parent.pt'
    selected.write_bytes(b'parent selected from durable owner state')
    changed = list(command)
    changed[changed.index('--continue-from')+1] = str(selected)
    first = owner.emit_current(config, changed, out=output, selected_checkpoint=selected)
    first_manifest = json.loads(Path(first['manifest']['path']).read_text())
    first_command = first_manifest['publication_spec']['effective_argv']['path']
    assert json.loads(Path(first_command).read_text()) == changed
    assert first_manifest['publication_spec']['parent']['path'] == str(selected)

    next_selected = tmp_path / 'next-selected-parent.pt'
    next_selected.write_bytes(b'next parent selected from durable owner state')
    changed[changed.index('--continue-from')+1] = str(next_selected)
    second = owner.emit_current(config, changed, out=output, selected_checkpoint=next_selected)
    second_manifest = json.loads(Path(second['manifest']['path']).read_text())
    second_command = second_manifest['publication_spec']['effective_argv']['path']
    assert first['generation_sha256'] != second['generation_sha256']
    assert first_command != second_command
    assert Path(first_command).is_file()
    assert json.loads(Path(second_command).read_text()) == changed
    assert second_manifest['publication_spec']['parent']['path'] == str(next_selected)
    mismatched = list(changed)
    mismatched[mismatched.index('--continue-from')+1] = str(selected)
    with pytest.raises(ValueError, match='selected parent'):
        owner.emit_current(config, mismatched, out=output, selected_checkpoint=next_selected)


def test_supervised_execute_publishes_the_exact_expanded_trainer_command(tmp_path, monkeypatch):
    import ast
    import importlib.util
    import sys
    from natlang_neuralese.rl import native_supervised as supervised
    from natlang_neuralese.train import native_operation_report
    from natlang_neuralese.train import recipe
    from natlang_neuralese.model.native_checkpoint_control import publication_trainer_parameters

    runtime = Path(__file__).resolve().parents[2]
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'complete optimizer parent')
    output = tmp_path / 'rounds' / 'lfm' / 'cycles' / '000004' / 'supervised' / '001'
    selection = publication_selection(namespace='/work/native-production', readonly_input_roots=D24_ROOTS)
    args = publication_trainer_parameters(selection)
    trainer_args = ['--checkpoint-publication' if args['checkpoint_publication'] else '',
        '--checkpoint-publication-namespace', args['checkpoint_publication_namespace'],
        '--checkpoint-publication-readonly-input-roots', args['checkpoint_publication_readonly_input_roots']]
    config = {'native_participant':'lfm','runtime':str(runtime),'checkpoint':str(parent),
        'native_checkpoint_publication':selection,
        'native_checkpoint_publication_owner_pointer':str(tmp_path/'control'/'lfm-current-owner.json'),
        'native_checkpoint_publication_reserve_bytes':20*1024**3,
        'production_stage':'native_full_sft','recipe_lineage_sha256':'a'*64,
        'request_id':'round-4-supervised-1','out':str(output),
        'trainer_parameters':{},
        'seed_command':['/usr/bin/python','-u','-m',owner.TRAINER_MODULE,
            '--out','{learner_out}','--continue-from','{checkpoint}','{trainer_args}']}
    monkeypatch.setattr(recipe,'stage_parameter_args',lambda _:trainer_args)
    external_dir=tmp_path/'external-control';external_dir.mkdir()
    helper_path=external_dir/'native_publication_owner.py';helper_path.write_bytes(Path(owner.__file__).read_bytes())
    helper_name='natlang_neuralese.rl.native_publication_owner'
    helper_spec=importlib.util.spec_from_file_location(helper_name,helper_path)
    external_helper=importlib.util.module_from_spec(helper_spec);external_helper.__package__='natlang_neuralese.rl'
    helper_spec.loader.exec_module(external_helper)
    monkeypatch.setitem(sys.modules,helper_name,external_helper)
    supervised_source=Path(supervised.__file__).read_bytes()
    execute_node=next(node for node in ast.parse(supervised_source).body
        if isinstance(node,ast.FunctionDef) and node.name=='execute')
    hook_path=external_dir/'native_supervised_hook.py'
    hook_path.write_bytes(supervised_source)
    namespace=dict(supervised.__dict__)
    exec(compile(ast.Module(body=[execute_node],type_ignores=[]),str(hook_path),'exec'),namespace)
    monkeypatch.setattr(supervised,'execute',namespace['execute'])
    assert Path(supervised.__file__).resolve()!=hook_path.resolve()
    assert supervised.execute.__code__.co_filename==str(hook_path)
    assert external_helper.emit_current.__code__.co_filename==str(helper_path)
    captured={}
    def run(command, values):
        from natlang_neuralese.rl.production import expand_command
        captured['command']=expand_command(command,values)
        learner=Path(values['learner_out']);learner.mkdir(parents=True)
        (learner/'checkpoint.pt').write_bytes(b'updated full checkpoint')
        (learner/'report.json').write_text(json.dumps({'completed':True,'numerics_ok':True}))
    monkeypatch.setattr('natlang_neuralese.rl.production.run_command',run)
    monkeypatch.setattr(native_operation_report,'control_progress',lambda report,path:{'mandatory_sft':{'complete':True}})
    selected_parent=output/'selected-parent.pt'
    selected_parent.parent.mkdir(parents=True,exist_ok=True)
    selected_parent.write_bytes(b'parent from current supervised state')
    state={'checkpoint':str(selected_parent)}
    state_path=tmp_path/'state.json'
    result=supervised.execute(config,state,state_path,None)
    assert result['phase']=='awaiting_semantic_refresh'
    manifest,_=external_helper.read_owner(config['native_checkpoint_publication_owner_pointer'],role='lfm')
    argv_ref=manifest['publication_spec']['effective_argv']
    assert json.loads(Path(argv_ref['path']).read_text())==captured['command']
    assert captured['command'][captured['command'].index('--continue-from')+1]==str(selected_parent)
    assert captured['command'][captured['command'].index('--out')+1]==str(output/'learner')


def test_external_helper_keeps_sidecar_file_and_uses_explicit_runtime_root(tmp_path):
    import importlib.util

    source = Path(owner.__file__).read_bytes()
    sidecar = tmp_path / 'external-control' / 'native_publication_owner.py'
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(source)
    name = 'natlang_neuralese.rl.external_publication_owner_fixture'
    spec = importlib.util.spec_from_file_location(name, sidecar)
    external = importlib.util.module_from_spec(spec)
    external.__package__ = 'natlang_neuralese.rl'
    spec.loader.exec_module(external)
    config, command, output = _fixture(tmp_path)
    published = external.emit_current(config, command, out=output,
        selected_checkpoint=config['checkpoint'], runtime_root=config['runtime'])
    manifest, _ = external.read_owner(config['native_checkpoint_publication_owner_pointer'], role='lfm')
    assert Path(external.__file__).resolve() == sidecar.resolve()
    assert manifest['runtime'] == str(Path(config['runtime']).resolve())
    assert manifest['package']['path'] == str(Path(config['runtime']).resolve() / 'training/neuralese/natlang_neuralese')
    assert published['runtime_code_sha256'] == manifest['runtime_code_sha256']


def test_external_helper_rejects_runtime_root_different_from_bound_config(tmp_path):
    config, command, output = _fixture(tmp_path)
    with pytest.raises(ValueError, match='bound operation config'):
        owner.emit_current(config, command, out=output,
            selected_checkpoint=config['checkpoint'], runtime_root=tmp_path)


def test_raw_rl_invokes_owner_publication_at_the_unchanged_inner_sft_boundary(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.rl import native_stage

    parent=tmp_path/'parent.pt';parent.write_bytes(b'actual selected RL parent')
    job=tmp_path/'rounds'/'lfm'/'learner-job';learner=job/'learner'
    learner.mkdir(parents=True)
    pointer=tmp_path/'control'/'lfm-current-owner.json'
    selection=publication_selection(namespace='/work/native-production',readonly_input_roots=D24_ROOTS)
    forwarded=['--checkpoint-publication','--checkpoint-publication-namespace',selection['namespace'],
        '--checkpoint-publication-readonly-input-roots',json.dumps(selection['readonly_input_roots'])]
    actual=['--native-graphs',str(tmp_path/'rl-native-graphs.jsonl'),
        '--held-native-graphs',str(tmp_path/'held.jsonl'),'--objective-spec',str(tmp_path/'objective.json'),
        '--continue-from',str(parent),'--objective-continuation','--out',str(learner),*forwarded]
    args=SimpleNamespace(publication_owner_pointer=pointer,publication_role='lfm',out=learner,
        continue_from=parent,coefficient_policy=None)
    called={}
    def actual_writer(argv):
        called['argv']=list(argv)
        manifest,_=owner.read_owner(pointer,role='lfm')
        called['manifest']=manifest
        return 17
    assert native_stage._invoke_raw_trainer(args,forwarded,actual,actual_writer)==17
    assert called['argv']==actual
    manifest=called['manifest']
    command=json.loads(Path(manifest['publication_spec']['effective_argv']['path']).read_text())
    assert command[-len(actual):]==actual
    assert command[command.index('--continue-from')+1]==str(parent)
    assert command[command.index('--out')+1]==str(learner)
    assert manifest['publication_spec']['parent']['sha256']==__import__('hashlib').sha256(parent.read_bytes()).hexdigest()
    assert manifest['publication_spec']['effective_argv']['allowed_output_root']==str(job.resolve())
    assert manifest['operation']['production_stage']=='native_rl'
    assert manifest['publication_spec']['runtime_code_sha256']==manifest['runtime_code_sha256']


def test_raw_rl_publication_requires_explicit_real_trainer_controls(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.rl import native_stage

    args=SimpleNamespace(publication_owner_pointer=tmp_path/'pointer.json',publication_role='lfm')
    with pytest.raises(ValueError,match='disabled checkpoint publication'):
        native_stage._publication_values(args,[])
    args.publication_owner_pointer=None
    with pytest.raises(ValueError,match='requires an owner pointer'):
        native_stage._publication_values(args,['--publication-role','lfm'])


def test_accepts_native_preference_adapter_that_forwards_publication_args(tmp_path):
    config, command, output = _fixture(tmp_path)
    command[3] = 'natlang_neuralese.train.native_preference'
    command.extend(['--objective-spec', str(tmp_path / 'objective.json')])
    published = owner.emit_current(config, command, out=output, selected_checkpoint=config['checkpoint'])
    manifest, _ = owner.read_owner(config['native_checkpoint_publication_owner_pointer'], role='lfm')
    assert published['generation_sha256'] == manifest['generation_sha256']
    assert manifest['publication_spec']['effective_argv']['path']


@pytest.mark.parametrize('role', ['lfm', 'mellum'])
def test_raw_rl_actual_emit_preserves_unsorted_declared_roots(tmp_path, role):
    from types import SimpleNamespace
    from natlang_neuralese.rl import native_stage
    parent=tmp_path/'parent.pt';parent.write_bytes(b'exact optimizer parent')
    learner=tmp_path/'rounds'/role/'job'/'learner';learner.mkdir(parents=True)
    pointer=tmp_path/'control'/'current-owner.json'
    roots=list(reversed(D24_ROOTS))
    assert roots != sorted(roots)
    forwarded=['--checkpoint-publication','--checkpoint-publication-namespace','/work/native-production',
        '--checkpoint-publication-readonly-input-roots',json.dumps(roots)]
    actual=['--continue-from',str(parent),'--out',str(learner),*forwarded]
    args=SimpleNamespace(publication_owner_pointer=pointer,publication_role=role,out=learner,
        continue_from=parent,coefficient_policy=None)
    observed={}
    def writer(argv):
        manifest,_=owner.read_owner(pointer,role=role)
        observed.update(argv=list(argv),manifest=manifest)
        return 23
    assert native_stage._invoke_raw_trainer(args,forwarded,actual,writer)==23
    assert observed['argv']==actual
    spec=observed['manifest']['publication_spec']
    assert spec['publication']['readonly_input_roots']==roots
    command=json.loads(Path(spec['effective_argv']['path']).read_bytes())
    assert command[-len(actual):]==actual


def test_raw_rl_unsorted_root_transport_still_rejects_relative_roots(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.rl import native_stage
    args=SimpleNamespace(publication_owner_pointer=tmp_path/'owner.json',publication_role='lfm')
    forwarded=['--checkpoint-publication','--checkpoint-publication-namespace','/work/native-production',
        '--checkpoint-publication-readonly-input-roots',json.dumps(['/absolute','relative'])]
    with pytest.raises(ValueError,match='absolute declared roots'):
        native_stage._publication_values(args,forwarded)
