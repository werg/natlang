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
@pytest.mark.parametrize('native_boundary',[False,True])
def test_all_fail_groups_continue_seed_stage_without_claiming_rl_training_exposure(tmp_path, monkeypatch, publish_fails_once,native_boundary):
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
    result = production.run(config, rounds=1)
    assert result['round'] == 1
    state = json.loads((out / 'state.json').read_text())
    assert calls == (['seed', 'publish', 'publish'] if publish_fails_once else ['seed', 'publish'])
    assert state['round'] == 1 and state['collected_task_fraction'] == 0.
    assert state['trained_task_ids'] == []  # general SFT seed did not train these RL tasks
    assert state['trained_target_tokens'] == 80
    assert state['production_complete'] is False
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
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps({'id': 'task', 'split': 'train'}) + '\n')
    checkpoint = tmp_path / 'parent.pt'; checkpoint.write_bytes(b'full state')
    out = tmp_path / 'production'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': str(checkpoint), 'cases': str(source),
        'endpoints': ['unused'], 'seed_command': ['seed'], 'learner_command': ['learn'], 'after_update': ['publish']}))
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
    from natlang_neuralese.rl.grpo import Server
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps({'id': 'task', 'split': 'train'}) + '\n')
    out = tmp_path / 'round'
    monkeypatch.setattr(production, 'root', lambda name: Path(__file__).resolve().parents[2])
    monkeypatch.setattr(Server, 'get_bytes', lambda *args: b'{"rl_policy_binding":{"checkpoint_sha256":"wrong"}}')
    with pytest.raises(ValueError, match='immutable checkpoint'):
        production.collect(cases=source, out=out, endpoints=['http://unused'], policy_id='p', rows=1)
    (out / 'score.json').write_text('{"scored":4}')
    with pytest.raises(ValueError, match='immutable checkpoint'):
        production.collect(cases=source, out=out, endpoints=['http://unused'], policy_id='p', rows=1)


def test_mixed_and_solved_group_selection_is_frozen_across_publish_restart(tmp_path, monkeypatch):
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps({'id': name, 'split': 'train'}) + '\n' for name in ['mixed', 'solved']))
    parent = tmp_path / 'parent.pt'; parent.write_bytes(b'parent')
    config = tmp_path / 'config.json'
    out = tmp_path / 'out'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': str(parent), 'cases': str(source),
        'endpoints': ['unused'], 'rows_per_round': 2, 'learner_command': ['learn'], 'after_update': ['publish']}))
    selections, calls = [], []
    def collect(**kwargs):
        selections.append(Path(kwargs['cases']).read_bytes())
        (kwargs['out'] / 'rollouts.jsonl').write_text('')
        return {'scored': 4, 'successes': 3, 'groups_with_spread': 1, 'groups': [
            {'task_id': 'mixed', 'scored': 2, 'mean': .5}, {'task_id': 'solved', 'scored': 2, 'mean': 1.}]}
    def command(argv, values):
        calls.append(argv[0])
        if argv[0] == 'learn':
            stage = Path(values['learner_out']); stage.mkdir()
            (stage / 'checkpoint.pt').write_bytes(b'new-full-state')
            (stage / 'report.json').write_text(json.dumps({'completed': True, 'this_run': {'target_tokens': 10}}))
        elif calls.count('publish') == 1:
            raise RuntimeError('publication')
    monkeypatch.setattr(production, 'collect', collect)
    monkeypatch.setattr(production, 'run_command', command)
    with pytest.raises(RuntimeError, match='publication'):
        production.run(config, rounds=1)
    production.run(config, rounds=1)
    assert selections[0] == selections[1] and b'solved' in selections[1]
    assert calls == ['learn', 'publish', 'publish']
    state = json.loads((out / 'state.json').read_text())
    assert state['trained_target_tokens'] == 10 and state['group_stats'] == []


def test_rft_text_and_value_full_stages_continue_exact_state_and_resume_partial_value(tmp_path, monkeypatch):
    from natlang_neuralese.rl import stage
    source = tmp_path / 'cases.jsonl'; source.write_text('{"id":"task","split":"train"}\n')
    parent = tmp_path / 'parent.pt'; parent.write_bytes(b'parent')
    config = tmp_path / 'config.json'; out = tmp_path / 'out'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': str(parent), 'cases': str(source),
        'endpoints': ['unused'], 'held_text_data': 'registered-held', 'learner_command': ['learn'], 'after_update': ['publish']}))
    block = 'nz1_' + 'a' * 52
    docs = [{'split': 'train', 'task_id': 'text-task', 'target': {'content': 'teacher text'}},
            {'split': 'train', 'task_id': 'value-task', 'target': {'content': [{'type': 'neuralese', 'id': block}]}}]
    monkeypatch.setattr(stage, 'select_rft_documents', lambda *args, **kwargs: (docs, {}))
    def collect(**kwargs):
        (kwargs['out'] / 'rollouts.jsonl').write_text('')
        return {'scored': 2, 'successes': 0, 'groups_with_spread': 0}
    monkeypatch.setattr(production, 'collect', collect)
    calls = []
    def command(argv, values):
        if argv[0] == 'publish':
            calls.append(('publish', values['checkpoint'])); return
        kind = values['rft_document_kind']
        calls.append((kind, values['checkpoint']))
        directory = Path(values['learner_out']); directory.mkdir(exist_ok=True)
        (directory / 'checkpoint.pt').write_bytes(kind.encode())
        (directory / 'identity-check.json').write_text('{}')
        (directory / 'monitor-reference.pt').write_bytes(b'monitor')
        value_attempt = sum(call[0] == 'value' for call in calls)
        partial = kind == 'value' and value_attempt == 1
        step = value_attempt if kind == 'value' else 1
        (directory / 'report.json').write_text(json.dumps({'completed': not partial, 'step': step}))
        with (directory / 'train.jsonl').open('a') as log:
            log.write(json.dumps({'step': step, 'target_tokens': 3, 'supervised_tokens': 4}) + '\n')
        (directory / 'rl-data-receipt.json').write_text(json.dumps({'tasks_selected': [
            {'split': 'train', 'task_id': kind + '-task'}]}))
    monkeypatch.setattr(production, 'run_command', command)
    result = production.run(config, rounds=1)
    first = out / 'round-000000/learner-text/checkpoint.pt'
    last = out / 'round-000000/learner-value/checkpoint.pt'
    assert calls == [('text', str(parent)), ('value', str(first)), ('value', str(first)), ('publish', str(last))]
    assert result['trained_target_tokens'] == 9  # partial value update was retained, not lost or counted twice
    assert result['trained_task_ids'] == ['text-task', 'value-task']
    assert result['checkpoint'] == str(last)


def test_training_exposure_survives_noop_resume_and_uncommitted_log_tail(tmp_path):
    log = tmp_path / 'train.jsonl'
    log.write_text(''.join(json.dumps({'step': step, 'target_tokens': 3, 'supervised_tokens': 4}) + '\n'
                           for step in [1, 2, 2, 3]))
    assert production.trained_counts(tmp_path, {'step': 2, 'this_run': {'target_tokens': 0}}) == {
        'target_tokens': 6, 'supervised_tokens': 8}


def test_configure_retains_declared_task_quality_cadence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.train import checkpoint_policy, direct_sft, recipe
    parent = tmp_path / 'checkpoint.pt'; parent.write_bytes(b'full')
    declared = tmp_path / 'recipe.json'; declared.write_text('{}')
    state = {'schema': direct_sft.SCHEMA, 'optimizer': {}, 'neuralese_projection': {},
             'identity': {'options': {'optimizer': 'muon', 'text_data': 'registered-train'}}}
    monkeypatch.setattr(checkpoint_policy.CheckpointPolicy, 'load', lambda path: state)
    parameters = {'task_eval': {'cases': 'registered-held-tasks'}, 'task_eval_every': 250,
                  'checkpoint_every': 100, 'checkpoint_layers': True}
    monkeypatch.setattr(recipe, 'resolve_recipe_data', lambda path: {'stages': [
        {'id': 'mixed', 'kind': 'neuralese_direct', 'parameters': parameters}]})
    args = SimpleNamespace(checkpoint=parent, production_recipe=declared, production_stage='mixed',
        config=tmp_path / 'config.json', model='lfm', port=8097, **{key: tmp_path / key
        for key in ('heads', 'base', 'cases', 'held_text_data', 'runtime', 'out')})
    production.configure(args)
    saved = json.loads(args.config.read_text())
    assert saved['trainer_parameters']['task_eval_every'] == 250
    assert saved['trainer_parameters']['task_eval'] == parameters['task_eval']
    assert saved['trainer_parameters']['checkpoint_layers'] is True
    assert saved['production_stage'] == 'mixed'
    actor, learner = saved['actor_command'], saved['learner_command']
    assert actor[actor.index('--prompt-rendering') + 1] == 'full'
    assert learner[learner.index('--rl-prompt-rendering') + 1] == 'full'
    state['optimizer'] = None
    with pytest.raises(ValueError, match='optimizer state'):
        production.configure(args)


@pytest.mark.parametrize('legacy_empty_snapshot', [False, True])
def test_solved_tasks_revisit_and_admit_new_task_on_resume(tmp_path, monkeypatch, legacy_empty_snapshot):
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps({'id': 'solved', 'split': 'train'}) + '\n')
    parent = tmp_path / 'checkpoint.pt'; parent.write_bytes(b'parent')
    out = tmp_path / 'out'
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': str(parent), 'cases': str(source),
        'endpoints': ['unused'], 'after_update': ['publish']}))
    collected = []
    def collect(**kwargs):
        tasks = [json.loads(line)['id'] for line in Path(kwargs['cases']).read_text().splitlines()]
        collected.extend(tasks)
        task=tasks[-1]
        (kwargs['out'] / 'rollouts.jsonl').write_text('')
        return {'scored': 2, 'successes': 2, 'groups': [
            {'task_id': task, 'scored': 2, 'mean': 1.}]}
    monkeypatch.setattr(production, 'collect', collect)
    production.run(config, rounds=1)
    assert production.run(config, rounds=1)['phase'] == 'collect'
    snapshot = out / 'round-000002-selection.json'
    assert not snapshot.exists()
    if legacy_empty_snapshot:
        snapshot.write_text(json.dumps({'selection': [], 'cases': []}))
    with source.open('a') as stream:
        stream.write(json.dumps({'id': 'new', 'split': 'train'}) + '\n')
    result=production.run(config, rounds=1)
    assert collected == ['solved', 'solved', 'solved', 'new']
    assert set(result['task_curriculum_history'])=={'solved','new'}
    assert result['task_curriculum_history']['solved']['observed_round']==1
    assert result['task_curriculum_history']['new']['observed_round']==2


def test_empty_source_waits_without_freezing_selection(tmp_path):
    source = tmp_path / 'cases.jsonl'; source.write_text('')
    config = tmp_path / 'config.json'; out = tmp_path / 'out'
    config.write_text(json.dumps({'out': str(out), 'checkpoint': 'unused', 'cases': str(source),
        'endpoints': ['unused'], 'after_update': ['publish']}))
    assert production.run(config, rounds=1)['phase'] == 'await_new_train_tasks'
    assert not list(out.glob('*selection.json'))
