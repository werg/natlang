"""Operational boundaries: collection provenance and useful zero-spread fallback."""
import json
from pathlib import Path

import pytest

from natlang_neuralese.rl import production
from natlang_neuralese.common.jsonio import write_bytes_durable, write_json_durable


def test_native_actor_replica_attestation_binds_actual_embedding_path():
    from copy import deepcopy
    identity = {'checkpoint_sha256': '1' * 64, 'tokenizer_fingerprint': '2' * 64,
                'embedding_weight_sha256': '3' * 64, 'architecture': 'lfm2'}
    binding = {'schema': 'natlang.rl-policy-binding/2', 'checkpoint_sha256': '1' * 64,
               'communication': 'native-state', 'native_identity': identity}
    production.validate_actor_receipts({'a': binding, 'b': deepcopy(binding)},
                                      policy_id='1' * 64, neuralese_server=True)
    different = deepcopy(binding)
    different['native_identity']['embedding_weight_sha256'] = '4' * 64
    with pytest.raises(ValueError, match='exact policy identity'):
        production.validate_actor_receipts({'a': binding, 'b': different},
                                          policy_id='1' * 64, neuralese_server=True)
    del different['native_identity']['tokenizer_fingerprint']
    with pytest.raises(ValueError, match='provenance'):
        production.validate_actor_receipts({'b': different}, policy_id='1' * 64, neuralese_server=True)


def test_actor_receipts_reject_retired_aliases_and_keep_ordinary_text_eligibility():
    text = {'checkpoint_sha256': '1' * 64}
    assert production.validate_actor_receipts({'text': text}, policy_id='1' * 64,
                                               neuralese_server=False) == {'text': text}
    with pytest.raises(ValueError, match='retired Neuralese communication alias'):
        production.validate_actor_receipts({'old': {**text, 'neuralese': True}}, policy_id='1' * 64,
                                            neuralese_server=True)
    with pytest.raises(ValueError, match='unsupported communication mode'):
        production.validate_actor_receipts({'old': {**text, 'communication': 'legacy-projected'}},
                                            policy_id='1' * 64, neuralese_server=False)
    with pytest.raises(ValueError, match='did not restore'):
        production.validate_actor_receipts({'missing': text}, policy_id='1' * 64, neuralese_server=True)


def test_collect_never_reuses_unbound_completed_results(tmp_path, monkeypatch):
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps({'id': 'train', 'split': 'train'}) + '\n')
    out = tmp_path / 'round'
    out.mkdir()
    (out / 'score.json').write_text('{}')
    monkeypatch.setattr(production, 'root', lambda name: Path(__file__).resolve().parents[2])
    with pytest.raises(ValueError, match='no provenance'):
        production.collect(cases=source, out=out, endpoints=['http://unused'], policy_id='p', rows=1)


def test_command_arguments_preserve_literal_json(monkeypatch):
    calls = []
    monkeypatch.setattr(production.subprocess, 'run', lambda argv, **kw: calls.append(argv))
    production.run_command(['python', '--settings', '{"mode":"rl"}', '--out', '{out}'], {'out': '/some dir'})
    assert calls == [['python', '--settings', '{"mode":"rl"}', '--out', '/some dir']]


@pytest.mark.parametrize('publish_fails_once', [False, True])
@pytest.mark.parametrize('native_boundary',[True])
def test_synthetic_seed_result_cannot_commit_rl_credit(tmp_path, monkeypatch, publish_fails_once,native_boundary):
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps({'id': f't{i}', 'split': 'train'}) + '\n' for i in range(4)))
    checkpoint = tmp_path / 'parent.pt'
    checkpoint.write_bytes(b'full state')
    out = tmp_path / 'production'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': str(checkpoint), 'cases': str(source),
        'endpoints': ['http://unused'], 'rows_per_round': 2, 'seed_command': ['seed'],
        'learner_command': ['learn'], 'after_update': ['publish'],
        'native_state':native_boundary,'base':'unused'}))
    if native_boundary:
        from natlang_neuralese.rl import native_raw_policy
        monkeypatch.setattr(native_raw_policy,'prepare_raw_policy',lambda *a,**kw:{'tokenizer':'bound-parent'})
    def collect(**kwargs):
        dest = kwargs['out']
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'rollouts.jsonl').write_text(''.join(json.dumps({'task_id': f't{i}',
            'split': 'train', 'reward': {'quality': 0}, 'turns': []}) + '\n' for i in range(2)))
        return {'groups_with_spread': 0, 'rollouts': 2, 'scored': 2, 'successes': 0}
    monkeypatch.setattr(production, 'collect', collect)
    calls = []
    def command(argv, values):
        calls.append(argv[0])
        if argv[0] == 'publish' and publish_fails_once and calls.count('publish') == 1:
            raise RuntimeError('transient publication failure')
        if argv[0] == 'seed':
            stage = Path(values['learner_out'])
            stage.mkdir(parents=True)
            (stage / 'checkpoint.pt').write_bytes(b'updated full state')
            from natlang_neuralese.common.hashing import sha256_file_hex
            (stage / 'report.json').write_text(json.dumps({'completed': not native_boundary,
                'schema':'natlang.neuralese-native-sft-report/1','numerics_ok':True,
                'mandatory_sft':{'complete':False,'remaining_graph_visits':999},
                'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':250,
                    'checkpoint_sha256':sha256_file_hex(stage/'checkpoint.pt')},
                'this_run': {'supervised_tokens': 120, 'target_tokens': 80}}))
    monkeypatch.setattr(production, 'run_command', command)
    if publish_fails_once:
        with pytest.raises(RuntimeError, match='publication'):
            production.run(config, rounds=1)
    with pytest.raises(ValueError,match='seed SFT report|native training receipt'):
        production.run(config, rounds=1)
    state = json.loads((out / 'state.json').read_text())
    assert calls == (['seed', 'publish', 'publish'] if publish_fails_once else ['seed', 'publish'])
    assert state['round'] == 0 and state['trained_rounds'] == 0 and state['collected_task_fraction'] == 0.
    assert state['trained_task_ids'] == []  # general SFT seed did not train these RL tasks
    assert state['trained_target_tokens'] == 80
    assert state['production_complete'] is False
    assert state['phase']==('updating' if publish_fails_once else 'publishing')
    assert 'active_update_round' in state
    assert 'last_completed_update' not in state
    if native_boundary:
        assert state['last_learner_progress']['completed'] is False
        assert state['last_learner_progress']['mandatory_sft']['remaining_graph_visits']==999


@pytest.mark.parametrize('tamper',['schema','numerics','kind','reached','updates','checkpoint','mode'])
def test_seed_checkpoint_boundary_rejects_unqualified_or_wrong_checkpoint(tmp_path,tamper):
    from copy import deepcopy
    from natlang_neuralese.common.hashing import sha256_file_hex
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'exact-quantum')
    report={'schema':'natlang.neuralese-native-sft-report/1','completed':False,'numerics_ok':True,
        'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':250,
                                'checkpoint_sha256':sha256_file_hex(checkpoint)}}
    assert production.qualified_seed_boundary({'native_state':True},'seed',report,checkpoint)
    bad=deepcopy(report);mode='seed'
    if tamper=='schema':bad['schema']='other'
    elif tamper=='numerics':bad['numerics_ok']=False
    elif tamper=='kind':bad['continuation_boundary']['kind']='other'
    elif tamper=='reached':bad['continuation_boundary']['reached']=False
    elif tamper=='updates':bad['continuation_boundary']['new_optimizer_updates']=0
    elif tamper=='checkpoint':checkpoint.write_bytes(b'different-checkpoint')
    else:mode='rft'
    assert not production.qualified_seed_boundary({'native_state':True},mode,bad,checkpoint)


def test_task_selection_keeps_solved_mixed_and_failed_rows():
    cases = [{'id': 'easy'}, {'id': 'mixed'}, {'id': 'hard'}]
    stats = [{'task_id': 'easy', 'scored': 4, 'mean': 1.},
             {'task_id': 'mixed', 'scored': 4, 'mean': .5}]
    assert production.select_tasks(cases, stats, {'rows': 3, 'group': 4, 'offset': 0}) == [
        {'row': 0, 'samples': 4}, {'row': 1, 'samples': 4}, {'row': 2, 'samples': 4}]


def test_infrastructure_failure_never_runs_seeding_and_retries_fresh_attempt(tmp_path, monkeypatch):
    from natlang_neuralese.rl import native_raw_policy
    monkeypatch.setattr(native_raw_policy, 'prepare_raw_policy', lambda *a, **kw: {'tokenizer': 'bound'})
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps({'id': 'task', 'split': 'train'}) + '\n')
    checkpoint = tmp_path / 'parent.pt'; checkpoint.write_bytes(b'full state')
    out = tmp_path / 'production'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': str(checkpoint), 'cases': str(source),
        'endpoints': ['unused'], 'seed_command': ['seed'], 'learner_command': ['learn'], 'after_update': ['publish'],
        'native_state': True, 'base': 'unused'}))
    attempts = []
    def broken(**kwargs):
        attempts.append(kwargs['out'].name)
        (kwargs['out'] / 'rollouts.jsonl').write_text(json.dumps({'task_id': 'task', 'reward': None}) + '\n')
        return {'scored': 0, 'successes': 0, 'groups_with_spread': 0}
    monkeypatch.setattr(production, 'collect', broken)
    monkeypatch.setattr(production, 'run_command', lambda *args: pytest.fail('infra must not train'))
    for _ in range(2):
        with pytest.raises(RuntimeError, match='infrastructure'):
            production.run(config, rounds=1)
    assert attempts == ['round-000000', 'round-000000-attempt-1']
    state = json.loads((out / 'state.json').read_text())
    assert state['phase'] == 'infrastructure_repair' and state['collection_attempt'] == 2
    assert state['trained_target_tokens'] == 0 and state['trained_rounds'] == 0


def test_incomplete_score_cannot_skip_actor_identity_verification(tmp_path, monkeypatch):
    from natlang_neuralese.rl.actor_client import NativeActorClient
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps({'id': 'task', 'split': 'train'}) + '\n')
    out = tmp_path / 'round'
    monkeypatch.setattr(production, 'root', lambda name: Path(__file__).resolve().parents[2])
    monkeypatch.setattr(NativeActorClient, 'get_bytes', lambda *args: b'{"rl_policy_binding":{"checkpoint_sha256":"wrong"}}')
    with pytest.raises(ValueError, match='immutable checkpoint'):
        production.collect(cases=source, out=out, endpoints=['http://unused'], policy_id='p', rows=1)
    (out / 'score.json').write_text('{"scored":4}')
    with pytest.raises(ValueError, match='immutable checkpoint'):
        production.collect(cases=source, out=out, endpoints=['http://unused'], policy_id='p', rows=1)


def test_training_exposure_survives_noop_resume_and_uncommitted_log_tail(tmp_path):
    log = tmp_path / 'train.jsonl'
    log.write_text(''.join(json.dumps({'step': step, 'target_tokens': 3, 'supervised_tokens': 4}) + '\n'
                           for step in [1, 2, 2, 3]))
    assert production.trained_counts(tmp_path, {'step': 2, 'this_run': {'target_tokens': 0}}) == {
        'target_tokens': 6, 'supervised_tokens': 8}


@pytest.mark.parametrize('legacy_flag', ['--heads', '--held-text-data', '--native-state'])
def test_rl_production_configure_rejects_retired_projected_selector(tmp_path, legacy_flag):
    with pytest.raises(SystemExit) as error:
        production.main(['configure', '--model', 'lfm', '--checkpoint', str(tmp_path/'c'),
            '--base', str(tmp_path/'b'), '--cases', str(tmp_path/'cases'), '--runtime', str(tmp_path/'r'),
            '--out', str(tmp_path/'out'), '--config', str(tmp_path/'config'),
            '--production-recipe', str(tmp_path/'recipe'), '--production-stage', 'native',
            legacy_flag, str(tmp_path/'legacy')])
    assert error.value.code == 2


def test_empty_source_waits_without_freezing_selection(tmp_path):
    source = tmp_path / 'cases.jsonl'; source.write_text('')
    config = tmp_path / 'config.json'; out = tmp_path / 'out'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': 'unused', 'cases': str(source),
        'endpoints': ['unused'], 'after_update': ['publish'], 'native_state': True}))
    assert production.run(config, rounds=1)['phase'] == 'await_new_train_tasks'
    assert not list(out.glob('*selection.json'))



@pytest.mark.parametrize('hook_error',[False,True])
def test_managed_ready_hook_uses_same_live_actor_before_collection(tmp_path,monkeypatch,hook_error):
    from natlang_neuralese.rl import actor_client
    events=[]
    from os import environ
    actor_env={'NATLANG_REPO':'/frozen/actor','PYTHONPATH':'/frozen/actor/training/neuralese'}
    parent_repo=environ.get('NATLANG_REPO')
    class Process:
        pid=123
        def __init__(self,*a,**kw):
            events.append('spawn');self.stopped=False
            assert all(kw['env'][key]==value for key,value in actor_env.items())
            assert environ.get('NATLANG_REPO')==parent_repo
        def poll(self):return 0 if self.stopped else None
        def terminate(self):events.append('terminate');self.stopped=True
        def wait(self,*a,**kw):return 0
    class Server:
        def __init__(self,*a,**kw):pass
        def get_bytes(self,path):events.append('ready');return json.dumps({'rl_policy_binding':{'checkpoint_sha256':'exact-current'}}).encode()
    monkeypatch.setattr(production.subprocess,'Popen',Process)
    from natlang_neuralese.rl import actor_client
    monkeypatch.setattr(actor_client,'NativeActorClient',Server)
    def command(argv,values):
        assert argv==['held-only'] and values['policy_id']=='exact-current'
        events.append('held')
        if hook_error:raise RuntimeError('held monitor error')
    monkeypatch.setattr(production,'run_command',command)
    monkeypatch.setattr(production,'collect',lambda **kw:events.append('train')or {'done':True})
    config={'actor_command':['actor'],'after_actor_ready':['held-only'],'actor_environment':actor_env}
    values={'out':str(tmp_path),'policy_id':'exact-current'}
    if hook_error:
        with pytest.raises(RuntimeError,match='held monitor'):production.collect_with_actor(config,values,out=tmp_path,endpoints=['http://same'])
    else:assert production.collect_with_actor(config,values,out=tmp_path,endpoints=['http://same'])=={'done':True}
    assert events==(['spawn','ready','held','terminate']if hook_error else ['spawn','ready','held','train','terminate'])
    events.clear();(tmp_path/'score.json').write_text(json.dumps({'collection_complete':True}))
    assert production.collect_with_actor(config,values,out=tmp_path,endpoints=['http://same'])=={'done':True}
    assert events==['train']


def test_python_actor_runtime_is_bound_and_mixed_replicas_rejected():
    from copy import deepcopy
    policy = '1' * 64
    native = {'checkpoint_sha256': policy, 'tokenizer_fingerprint': '2' * 64,
              'embedding_weight_sha256': '3' * 64, 'architecture': 'lfm2'}
    runtime = {'schema': 'natlang.python-runtime/1', 'sha256': '4' * 64, 'files': 5}
    receipt = {'schema': 'natlang.rl-policy-binding/2', 'checkpoint_sha256': policy,
        'communication': 'native-state', 'native_identity': native, 'python_runtime': runtime}
    expected = {'native_identity': native, 'native_participant': None,
                'semantic_adapter_sha256': None, 'python_runtime': runtime}
    # External fleets have no local command or repository environment. They
    # remain bound by the actual receipts returned from every actor endpoint.
    assert production.validate_actor_runtime_configuration({
        'native_actor_python_runtime':runtime,'endpoints':['https://actor-fleet.example']}) is True
    production.validate_actor_receipts({'a': receipt}, policy_id=policy, neuralese_server=True,
                                       expected_native=expected)
    changed = deepcopy(receipt)
    changed['python_runtime']['sha256'] = '5' * 64
    with pytest.raises(ValueError, match='requested Python runtime'):
        production.validate_actor_receipts({'a': changed}, policy_id=policy, neuralese_server=True,
                                           expected_native=expected)
    for replacement in (None, changed['python_runtime']):
        with pytest.raises(ValueError, match='replicas disagree on their Python runtime'):
            production.validate_actor_receipts({'a': receipt, 'b': {**receipt, 'python_runtime': replacement}},
                                               policy_id=policy, neuralese_server=True)
    for invalid in ('not a receipt', {**runtime, 'files': True}, {**runtime, 'sha256': 'bad'}):
        with pytest.raises(ValueError, match='Python runtime provenance'):
            production.validate_actor_receipts({'a': {**receipt, 'python_runtime': invalid}},
                                               policy_id=policy, neuralese_server=True)


def test_python_runtime_reuses_resident_inventory_and_is_path_independent(tmp_path):
    from natlang_neuralese.common.runtime_inventory import python_runtime_binding
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.train.native_resident import code_inventory
    packages = [tmp_path/'first', tmp_path/'relocated']
    for package in packages:
        package.mkdir()
        (package/'actor.py').write_text('version = 1\n')
        (package/'ignored.txt').write_text('not executable Python source')
    initial = python_runtime_binding(packages[0])
    assert initial == python_runtime_binding(packages[1])
    assert initial['sha256'] == canonical_json_sha256_hex(code_inventory(packages[0]))
    (packages[1]/'actor.py').write_text('version = 2\n')
    assert python_runtime_binding(packages[1]) != initial


def test_managed_actor_readiness_timeout_stops_the_exact_child(tmp_path,monkeypatch):
    import time
    from natlang_neuralese.rl import actor_client
    stopped=[]
    class Process:
        pid=741;done=False
        def __init__(self,*args,**kwargs):pass
        def poll(self):return 0 if self.done else None
        def terminate(self):self.done=True;stopped.append(self.pid)
        def wait(self,*args,**kwargs):return 0
    class Server:
        def __init__(self,*args,**kwargs):pass
        def get_bytes(self,path):raise OSError('unhealthy info route')
    monkeypatch.setattr(production.subprocess,'Popen',Process)
    monkeypatch.setattr(actor_client,'NativeActorClient',Server)
    clock=iter([0.,0.,2.]);monkeypatch.setattr(time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='did not attest'):
        production.collect_with_actor({'actor_command':['actor'],'actor_ready_seconds':1},
            {'out':str(tmp_path),'policy_id':'current'},out=tmp_path,endpoints=['http://same'])
    assert stopped==[741]


def test_actor_runtime_requires_and_binds_rendering_assets_before_launch(tmp_path):
    import sys
    entry=tmp_path/'scripts/neuralese_data/serve_native_checkpoint.py';entry.parent.mkdir(parents=True);entry.write_text('# actor')
    package=tmp_path/'training/neuralese/natlang_neuralese';package.mkdir(parents=True);(package/'__init__.py').write_text('# source')
    with pytest.raises(ValueError,match='required rendering asset'):
        production.actor_runtime_configuration(tmp_path)
    assets=tmp_path/'training/prompt-pieces';assets.mkdir(parents=True)
    for name in ('pieces.json','rendering-contract-fixtures.json'):(assets/name).write_text('{}')
    config={**production.actor_runtime_configuration(tmp_path),'actor_command':[sys.executable,str(entry)]}
    assert production.validate_actor_runtime_configuration(config)
    (assets/'pieces.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='rendering assets'):
        production.validate_actor_runtime_configuration(config)
