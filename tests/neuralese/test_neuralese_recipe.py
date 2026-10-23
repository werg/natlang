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


@pytest.mark.parametrize('kind',['native_state_sft','native_offline_preference','native_state_rl'])
def test_history_reasoning_profile_is_an_explicit_native_recipe_handoff(tmp_path,kind):
    from copy import deepcopy
    from natlang_neuralese.train import recipe
    before={'stages':[{'id':'full','kind':kind,'parameters':{'epochs':1}}]}
    after=deepcopy(before)
    after['stages'][0]['parameters']['history_reasoning_policy']='keep'
    changes,left,right=recipe.direct_windowing_handoffs(before,after,tmp_path)
    assert changes=={'full':{'history_reasoning_policy':[None,'keep']}}
    assert left==right
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


def native_declared():
    return {'schema':'natlang.neuralese-training-recipe/1','id':'native-fixture',
        'identity_qualification':'native_state_runtime_qualified',
        'stages':[{'id':'sft','kind':'native_state_sft','requires':[],'parameters':{}}]}


def test_removed_projected_recipe_kinds_are_not_loadable(tmp_path):
    recipe=native_declared();recipe['stages'][0]['kind']='neuralese_direct'
    path=tmp_path/'retired.json';path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='unknown stage implementation'):
        load_recipe(path)


def test_native_stage_specific_bindings_are_hash_pinned_and_role_scoped(tmp_path):
    from natlang_neuralese.train.recipe import resolve_stage_inputs,stage_input_args
    from natlang_neuralese.train.output_embedding_projection import sha
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text('train\n');held.write_text('held\n')
    recipe=native_declared()
    recipe['input_bindings']={'graphs.train':{'path':str(train),'sha256':sha(train)},
                              'graphs.held':{'path':str(held),'sha256':sha(held)}}
    recipe['stages'][0]['inputs']={'native_graphs':'graphs.train','held_native_graphs':'graphs.held'}
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe))
    loaded=load_recipe(path);inputs=resolve_stage_inputs(loaded,loaded['stages'][0],{})
    assert set(inputs)=={'native_graphs','held_native_graphs'}
    assert stage_input_args(inputs,'native_state_sft')==[
        '--native-graphs',str(train.resolve()),'--held-native-graphs',str(held.resolve())]


def test_native_input_binding_relocation_keeps_content_identity(tmp_path):
    from natlang_neuralese.train.recipe import parse_input_binding_overrides,resolve_stage_inputs
    from natlang_neuralese.train.output_embedding_projection import sha
    original=tmp_path/'source.jsonl';relocated=tmp_path/'relocated.jsonl';mismatch=tmp_path/'different.jsonl'
    original.write_text('same\n');relocated.write_bytes(original.read_bytes());mismatch.write_text('different\n')
    recipe=native_declared();recipe['input_bindings']={'graphs.train':{'path':str(original),'sha256':sha(original)}}
    recipe['stages'][0]['inputs']={'native_graphs':'graphs.train','held_native_graphs':'graphs.train'}
    stage=recipe['stages'][0];overrides=parse_input_binding_overrides([f'graphs.train={relocated}'])
    resolved=resolve_stage_inputs(recipe,stage,{},overrides)
    assert resolved['native_graphs']['path']==str(relocated.resolve())
    assert resolved['native_graphs']['sha256']==sha(original)
    with pytest.raises(ValueError,match='content hash mismatch'):
        resolve_stage_inputs(recipe,stage,{}, {'graphs.train':str(mismatch)})
    with pytest.raises(ValueError,match='undeclared input binding'):
        resolve_stage_inputs(recipe,stage,{}, {'other':str(relocated)})


def test_native_recipe_runner_routes_only_declared_controller_input(tmp_path,monkeypatch):
    from natlang_neuralese.train import recipe as runner
    from natlang_neuralese.train.output_embedding_projection import sha
    out=tmp_path/'run';directory=out/'readiness';config=tmp_path/'controller.json'
    config.write_text(json.dumps({'schema':'natlang.native-shared-stage-controller/1','out':str(directory)}))
    recipe={'schema':'natlang.neuralese-training-recipe/1','id':'native-controller-fixture',
        'identity_qualification':'native_state_runtime_qualified',
        'input_bindings':{'controller':{'path':str(config),'sha256':sha(config)}},
        'stages':[{'id':'readiness','kind':'native_shared_stage_controller','requires':[],
                   'inputs':{'controller_config':'controller'},'parameters':{}}]}
    recipe_path=tmp_path/'recipe.json';recipe_path.write_text(json.dumps(recipe));commands=[]
    class CompletedStage:
        def __init__(self,command,env):
            commands.append(command);directory.mkdir(parents=True,exist_ok=True)
            (directory/'shared-stage-completion.json').write_text(json.dumps({
                'schema':'natlang.native-shared-stage-completion/1','completed':True,'production_ready':True}))
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(runner.subprocess,'Popen',CompletedStage)
    runner.main(['--recipe',str(recipe_path),'--out',str(out),'--device','cpu','--until','readiness'])
    assert commands and commands[0][commands[0].index('--config')+1]==str(config.resolve())
    assert json.loads((out/'recipe-plan.json').read_text())['stage_inputs']['readiness']['controller_config']['sha256']==sha(config)


def test_native_incomplete_stage_roles_and_unused_catalog_are_rejected(tmp_path):
    recipe=native_declared();recipe['input_bindings']={'graphs':{'path':'graphs.jsonl','sha256':'0'*64}}
    recipe['stages'][0]['inputs']={'native_graphs':'graphs'}
    path=tmp_path/'incomplete.json';path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='stage input roles'):
        load_recipe(path)
    recipe=native_declared();recipe['input_bindings']={'unused':{'path':'x','sha256':'0'*64}}
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError,match='unused named input'):
        load_recipe(path)


def test_registered_artifacts_bind_by_id_and_are_hash_checked(tmp_path,monkeypatch):
    import torch
    from safetensors.torch import save_file
    from natlang_neuralese import artifacts
    from natlang_neuralese.train.recipe import resolve_stage_inputs,validate_input_bindings
    (tmp_path/'training').mkdir();source=tmp_path/'bank.nz'
    header={'exports':{'p':{'type':'Neuralese<string>','value':{'$neuralese':{'id':'b'}}}},'blocks':{'b':{'dialect':'nd:test@1'}}}
    save_file({'b':torch.zeros(2,4)},str(source),metadata={'natlang':json.dumps(header)})
    artifacts.register(tmp_path,{'id':'bank-recipe-test-v1','kind':'prompt-bank','dialect':'nd:test@1',
        'backbone':{'model':'m','revision':'r'},'init':{'method':'text'},
        'qualification':{'status':'unqualified'}},{'bank.nz':source})
    monkeypatch.setattr(artifacts,'REPO',tmp_path)
    recipe=native_declared();recipe['input_bindings']={'bank':{'artifact':'bank-recipe-test-v1'}}
    recipe['stages'][0]['inputs']={'native_graphs':'bank','held_native_graphs':'bank'}
    validate_input_bindings(recipe)
    resolved=resolve_stage_inputs(recipe,recipe['stages'][0],{})['native_graphs']
    assert resolved['artifact']=='bank-recipe-test-v1' and resolved['path'].endswith('bank-recipe-test-v1/bank.nz')
    assert resolved['sha256']==artifacts.digest(source)
    recipe['input_bindings']={'bank':{'artifact':'bank-recipe-test-v1','sha256':'x'}}
    with pytest.raises(ValueError,match='only artifact'):
        validate_input_bindings(recipe)
    recipe['input_bindings']={'bank':{'artifact':'missing-artifact-v1'}}
    with pytest.raises(ValueError,match='unknown artifact'):
        resolve_stage_inputs(recipe,recipe['stages'][0],{})
