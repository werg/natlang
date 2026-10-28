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
    changes,left,right=recipe.native_stage_parameter_handoffs(before,after,tmp_path)
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
    changes,left,right=recipe.native_stage_parameter_handoffs(before,after,tmp_path)
    assert changes=={'full':{'learn_reasoning_boundaries':[None,True]}}
    assert left==right
    assert 'learn_reasoning_boundaries' in recipe.HANDLERS['native_state_sft']['parameters']
    assert '--learn-reasoning-boundaries' in recipe.stage_parameter_args(after['stages'][0]['parameters'])
    (tmp_path/'full-report.json').write_text('{}')
    with pytest.raises(ValueError,match='finished native stage'):
        recipe.native_stage_parameter_handoffs(before,after,tmp_path)


@pytest.mark.parametrize('kind',['native_state_sft','native_offline_preference','native_state_rl'])
def test_history_reasoning_profile_is_an_explicit_native_recipe_handoff(tmp_path,kind):
    from copy import deepcopy
    from natlang_neuralese.train import recipe
    before={'stages':[{'id':'full','kind':kind,'parameters':{'epochs':1}}]}
    after=deepcopy(before)
    after['stages'][0]['parameters']['history_reasoning_policy']='keep'
    changes,left,right=recipe.native_stage_parameter_handoffs(before,after,tmp_path)
    assert changes=={'full':{'history_reasoning_policy':[None,'keep']}}
    assert left==right
    (tmp_path/'full-report.json').write_text('{}')
    with pytest.raises(ValueError,match='finished native stage'):
        recipe.native_stage_parameter_handoffs(before,after,tmp_path)


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
    changes,left,right=recipe.native_stage_parameter_handoffs(before,after,tmp_path)
    assert changes=={'full':{'native_linear_precision':['ieee','bf16x6']}}
    assert left==right
    after['stages'][0]['parameters']['base']='/different-base'
    _,left,right=recipe.native_stage_parameter_handoffs(before,after,tmp_path)
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
    stage = {'id': 'native', 'kind': 'native_state_sft',
             'inputs': {'native_graphs': 'records', 'held_native_graphs': 'records'}}
    resolved = runner.resolve_stage_inputs(declared, stage, {})
    assert resolved['native_graphs']['path'] == str(data.resolve())
    assert resolved['held_native_graphs']['path'] == str(data.resolve())


@pytest.mark.parametrize('hourly_checkpoint', [False, True])
def test_lineage_handoff_takes_newest_code_and_operational_recipe_changes_only(tmp_path, hourly_checkpoint):
    import json
    from natlang_neuralese.train import recipe as runner
    out=tmp_path/'run'; frozen=out/'runtime'/'natlang_neuralese'; frozen.mkdir(parents=True)
    (frozen/'a.py').write_text('old = 1\n')
    package=tmp_path/'package'; package.mkdir(); (package/'a.py').write_text('new = 1\n'); (package/'b.py').write_text('added = 1\n')
    old_recipe={'id':'r-v10','description':'v10','stages':[{'id':'s','kind':'native_state_sft','parameters':{'epochs':1}}],
                'stage_parameter_defaults':{'native_state_sft':{'latent_ns':'cubic5'}}}
    new_recipe={'id':'r-v11','description':'v11','extends':'r-v10',
                'stages':[{'id':'s','kind':'native_state_sft','parameters':{'epochs':1,**({'checkpoint_seconds':3600} if hourly_checkpoint else {})}}],
                'stage_parameter_defaults':{'native_state_sft':{'latent_ns':'cubic5'}}}
    existing={'recipe':old_recipe,'recipe_sha256':'lineage','inputs':{'x':'1'},'device':'cuda',
              'code':{'a.py':runner.sha(frozen/'a.py')}}
    path=out/'recipe-plan.json';path.write_text(json.dumps(existing))
    plan={'recipe':new_recipe,'recipe_sha256':'v11','inputs':{'x':'1'},'device':'cuda'}
    updated=runner.lineage_handoff(existing,plan,frozen,package,path)
    assert updated['recipe_sha256']=='lineage' and updated['recipe']==new_recipe
    assert (frozen/'b.py').read_text()=='added = 1\n' and set(updated['code'])=={'a.py','b.py'}
    record=updated['handoffs'][-1]
    assert record['recipe_id']=='r-v11' and record['code_changed']==['a.py'] and record['code_added']==['b.py']
    assert (out/record['superseded_runtime']/'natlang_neuralese'/'a.py').read_text()=='old = 1\n'
    changed=json.loads(json.dumps(new_recipe));changed['stages'][0]['parameters']['rank']=64
    with pytest.raises(ValueError,match='recipe changes more than operational'):
        runner.lineage_handoff(updated,dict(plan,recipe=changed),frozen,package,path,dry_run=True)
    with pytest.raises(ValueError,match='inputs changed'):
        runner.lineage_handoff(updated,dict(plan,inputs={'x':'2'}),frozen,package,path,dry_run=True)


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
                  'stages': [{'id': 's', 'kind': 'native_state_sft', 'parameters': {'epochs': 1}}]}
    new_recipe = {'id': 'r-v10', 'runtime': {'image': 'sha256:' + '1' * 64, 'package_code': new_code},
                  'stages': [{'id': 's', 'kind': 'native_state_sft', 'parameters': {'epochs': 1,
                      'memory_qualification': {'path': '/qualification.json', 'sha256': 'c' * 64}}}]}
    existing = {'recipe': old_recipe, 'recipe_sha256': 'lineage', 'inputs': {'x': '1'}, 'device': 'cuda',
                'code': old_code}
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(_json.dumps(existing))
    plan = {'recipe': new_recipe, 'recipe_sha256': 'new-recipe-hash', 'inputs': {'x': '1'}, 'device': 'cuda'}

    updated = runner.lineage_handoff(existing, plan, frozen, package, plan_path)

    assert updated['recipe_sha256'] == 'lineage'
    assert updated['recipe']['runtime']['package_code'] == new_code
    assert updated['recipe']['stages'][0]['parameters']['memory_qualification']['sha256'] == 'c' * 64
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
    result = runner.lineage_handoff(copy.deepcopy(existing), candidate, frozen, package, path, out, dry_run=True, started_stage_ids=('native',))
    assert result['handoffs'][-1]['stage_input_changes']['native']['kind'] == 'data_continuation'
    assert result['handoffs'][-1]['native_parameter_changes']['native']['epochs'] == [1, 2]
    assert result['handoffs'][-1]['native_parameter_changes']['native']['chunk_tokens'] == [None, 1024]
    invalid = copy.deepcopy(candidate)
    invalid['stage_inputs']['native']['continued_held_native_graphs']['sha256'] = 'unrelated-held'
    with pytest.raises(ValueError, match='inputs of started stage'):
        runner.lineage_handoff(copy.deepcopy(existing), invalid, frozen, package, path, out, dry_run=True, started_stage_ids=('native',))


def test_running_recipe_can_append_future_native_graph_stage_without_restarting(tmp_path):
    import copy
    import json
    from natlang_neuralese.train import recipe as runner

    out=tmp_path/'run'; frozen=out/'runtime'/'natlang_neuralese'; frozen.mkdir(parents=True)
    (frozen/'a.py').write_text('old=1\n')
    package=tmp_path/'package'; package.mkdir(); (package/'a.py').write_text('new=1\n')
    (out/'warmup').mkdir()
    old_recipe={'id':'r','stages':[{'id':'warmup','kind':'native_state_sft','parameters':{'epochs':1}}]}
    old={'recipe':old_recipe,'recipe_sha256':'lineage','device':'cuda','inputs':{},
         'stage_inputs':{'warmup':{}},'code':{'a.py':runner.sha(frozen/'a.py')}}
    plan=copy.deepcopy(old)
    plan['recipe']['input_bindings']={'graphs':{'path':'/data/graphs','sha256':'a'*64},
                                      'held':{'path':'/data/held','sha256':'b'*64}}
    plan['recipe']['stages'].append({'id':'graphs','kind':'native_state_sft','requires':['warmup'],
        'inputs':{'native_graphs':'graphs','held_native_graphs':'held'},'parameters':{'epochs':1}})
    plan['stage_inputs']['graphs']={
        'native_graphs':{'binding':'graphs','path':'/data/graphs','sha256':'a'*64},
        'held_native_graphs':{'binding':'held','path':'/data/held','sha256':'b'*64}}
    plan['inputs']={'/data/graphs':'a'*64,'/data/held':'b'*64}
    path=out/'recipe-plan.json';path.write_text(json.dumps(old))
    updated=runner.lineage_handoff(copy.deepcopy(old),plan,frozen,package,path,out,dry_run=True)
    assert updated['recipe_sha256']=='lineage'
    assert updated['handoffs'][-1]['appended_stages']=={'graphs':plan['recipe']['stages'][1]}
    assert updated['stage_inputs']==plan['stage_inputs']
    assert json.loads(path.read_text())==old
    (out/'graphs').mkdir()
    with pytest.raises(ValueError,match='already has output'):
        runner.lineage_handoff(copy.deepcopy(old),plan,frozen,package,path,out,dry_run=True)


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
    from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
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
    from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
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
    from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
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
    state='nzs1_'+'a'*52
    header={'format':'natlang.neuralese-file/1','dialect':'native-state/test','types':'','native_tensors':{},
        'exports':{'p':{'type':'Neuralese<string>','value':{'$neuralese':{'type':'Neuralese<string>','id':state}}}},
        'native_states':{state:{'bytes':4}}}
    save_file({state:torch.zeros(4,dtype=torch.uint8)},str(source),metadata={'natlang':json.dumps(header)})
    artifacts.register(tmp_path,{'id':'bank-recipe-test-v1','kind':'prompt-bank','dialect':'native-state/test',
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


@pytest.mark.parametrize('kind',['native_state_sft','native_offline_preference','native_state_rl'])
def test_native_optimizer_history_storage_is_explicit_and_validated(kind):
    from natlang_neuralese.train import recipe
    recipe.validate_stage_parameters(kind,{'optimizer_state_storage':'cpu'})
    assert recipe.stage_parameter_args({'optimizer_state_storage':'cpu'})==['--optimizer-state-storage','cpu']
    with pytest.raises(ValueError,match='optimizer_state_storage must be device or cpu'):
        recipe.validate_stage_parameters(kind,{'optimizer_state_storage':'automatic'})
    assert 'optimizer_state_storage' in recipe.OPERATIONAL_STAGE_PARAMETERS
    assert 'optimizer_state_storage' in recipe.HANDLERS['native_state_sft']['parameters']


def test_started_append_moves_only_the_declared_shared_train_foundation_input():
    from natlang_neuralese.train.recipe import _data_continuation
    def ref(name,sha,binding):return {'path':'/data/'+name,'sha256':sha*64,'binding':binding}
    old_train=ref('old','a','shared');new_train=ref('new','b','shared');held=ref('held','c','held')
    old={'native_graphs':old_train,'foundation_native_graphs':old_train,'held_native_graphs':held}
    new={**old,'native_graphs':new_train,'foundation_native_graphs':new_train,
         'continued_native_graphs':{**old_train,'binding':'prior'}}
    assert _data_continuation(old,new)
    assert not _data_continuation(old,{**new,'foundation_native_graphs':old_train})
    assert not _data_continuation(old,{**new,'foundation_native_graphs':{**new_train,'binding':'separate'}})
    assert not _data_continuation(old,{**new,'continued_native_graphs':ref('wrong','d','prior')})
    distinct={**old,'foundation_native_graphs':{**old_train,'binding':'separate'}}
    assert not _data_continuation(distinct,new)
