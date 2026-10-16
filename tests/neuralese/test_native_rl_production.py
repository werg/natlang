"""Native admission/coordination keeps actual policies and both learner updates."""
import copy
import json
from pathlib import Path
import pytest
import torch

from natlang_neuralese.rl import native_rounds, production
from natlang_neuralese.common.hashing import sha256_file_hex


def test_joint_publication_finalizes_each_pending_update_once(tmp_path):
    checkpoint = tmp_path/'updated.pt'; checkpoint.write_bytes(b'updated')
    out = tmp_path/'model'; out.mkdir()
    state = {'checkpoint': 'old-joint', 'round': 3, 'trained_rounds': 2,
             'pending_native_checkpoint': str(checkpoint), 'pending_native_round_dir': 'round-dir',
             'group_stats': [{'task_id': 'easy', 'mean': 1.}]}
    path = out/'state.json'; path.write_text(json.dumps(state))
    deployment = {'checkpoint': {'path': 'new-joint'}, 'semantic_adapter': {'path': 'new-adapter'},
                  'updated_participants': {'lfm': {'sha256': sha256_file_hex(checkpoint)}}}
    config = {'out': str(out), 'native_participant': 'lfm'}
    native_rounds.apply_deployment(config, deployment)
    native_rounds.apply_deployment(config, deployment)
    result = json.loads(path.read_text())
    assert result['round'] == 4 and result['trained_rounds'] == 3
    assert result['checkpoint'] == 'new-joint' and result['semantic_adapter'] == 'new-adapter'
    assert 'pending_native_checkpoint' not in result and result['group_stats'] == []


@pytest.mark.parametrize('pending_updates',[True,False])
def test_native_parallel_round_merges_both_updates_before_refresh(tmp_path, monkeypatch,pending_updates):
    import threading
    from natlang_neuralese.rl import native_refresh
    parent = tmp_path/'parent.pt'; parent.write_bytes(b'joint-parent')
    adapter = tmp_path/'adapter.json'; adapter.write_text(json.dumps({'semantic_obligations':{'complete':False}}))
    configs = []
    updates = {}
    for name in ('lfm', 'mellum'):
        config = {'native_state': True, 'native_participant': name, 'checkpoint': str(parent),
            'semantic_adapter': str(adapter), 'semantic_refresh_config': str(tmp_path/'curriculum.json'),
            'out': str(tmp_path/name)}
        path = tmp_path/(name+'.json'); path.write_text(json.dumps(config)); configs.append(path)
        update = tmp_path/(name+'.pt'); update.write_bytes(name.encode()); updates[name] = str(update)
    barrier = threading.Barrier(2)
    def learner(path, **kwargs):
        config = json.loads(path.read_text())
        assert kwargs['rounds'] == 1 and kwargs['defer_native_refresh'] is True
        assert kwargs['stop'].requested is False
        barrier.wait(timeout=10)  # proves independent learner jobs overlap
        directory = Path(config['out']); directory.mkdir()
        state = {'checkpoint': str(parent), 'round': 0, 'trained_rounds': 0}
        if pending_updates:state['pending_native_checkpoint']=updates[config['native_participant']]
        (directory/'state.json').write_text(json.dumps(state))
        return state
    monkeypatch.setattr(native_rounds, 'run', learner)
    calls = []
    def refresh(**kwargs):
        assert kwargs['updates'] == (updates if pending_updates else {})
        calls.append(kwargs)
        return {'checkpoint': {'path': 'refreshed-shared', 'sha256': 'new'},
            'semantic_adapter': {'path': 'refreshed-adapter', 'sha256': 'adapter'},
            'updated_participants': {name: {'sha256': sha256_file_hex(path)} for name, path in updates.items()} if pending_updates else {}}
    monkeypatch.setattr(native_refresh, 'refresh', refresh)
    result = native_rounds.run_rounds(configs, tmp_path/'joint', rounds=1, device='cpu')
    assert len(calls) == 1 and result['deployment']['checkpoint']['path'] == 'refreshed-shared'
    for name in updates:
        state = json.loads((tmp_path/name/'state.json').read_text())
        assert state['round'] == int(pending_updates) and state['checkpoint'] == 'refreshed-shared'


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('mode', ['rft', 'grpo'])
def test_actual_shared_optimizer_updates_native_policy_graph_for_both_ports(tmp_path, family, mode):
    from test_native_rl_replay import actor_rollout
    from natlang_neuralese.rl.native_replay import prepare_objective, NativePolicyObjective
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, packed_graph_loss
    model, roles, identity, _, block, turns = actor_rollout(family)
    if family == 'mellum':
        model.port.hf.natlang_precision = 'bf16'  # tiny CPU dense fixture, not FP8 qualification
    raw = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'],
            'sampling': turn['sampling'], 'replay': {'raw_calls_retained': True}} for turn in turns]
    episodes = [{'task_id': 'fixture-task', 'sample': 0, 'split': 'train', 'group': 'g',
        'reward': {'quality': 1}, 'policy': {'checkpoint_sha256': identity['checkpoint_sha256']}, 'turns': raw}]
    if mode == 'grpo':
        negative = copy.deepcopy(episodes[0]); negative.update(sample=1, reward={'quality': 0})
        episodes.append(negative)
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    graphs, spec = prepare_objective(episodes, identity=identity, roles=roles,
                                    fetch_block=lambda key: block, mode=mode)
    path = tmp_path/'objective.json'; path.write_text(json.dumps(spec))
    examples = [parse_graph(row, tokenizer_fingerprint=identity['tokenizer_fingerprint']) for row in graphs]
    optimizer, named = make_native_optimizer(model, lr=1e-3, in_backward=False)
    before = {name: value.detach().clone() for name, value in named}
    for example in examples:  # every selected episode receives its actual update
        optimizer.zero_grad(set_to_none=True)
        loss, details = packed_graph_loss(model, [example], objective=NativePolicyObjective(path))
        loss.backward(); optimizer.step()
    assert details['sampled_tokens'] == 7 and torch.isfinite(loss)
    assert any(not torch.equal(before[name], value) for name, value in named if name.startswith('backbone.'))
    assert any(not torch.equal(before[name], value) for name, value in named if name.startswith('interface.'))


@pytest.mark.parametrize('shared_parent',[True,False])
def test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path,shared_parent):
    """Actual tiny LFM full trainer; fixture reward is NOT an environment result."""
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model, model_state, native_identity, SHARED_CHECKPOINT_SCHEMA
    from natlang_neuralese.model.shared_state import native_step
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.loop import capture_training_rng_state
    from natlang_neuralese.data.native_graph import SCHEMA
    from natlang_neuralese.rl.native_stage import main
    base = tmp_path/'base'; tiny_source(base)
    model, tokenizer, roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    optimizer, _ = make_native_optimizer(model, lr=1e-3, weight_decay=0, in_backward=False)
    parent = tmp_path/'shared.pt'
    saved = {'schema': SHARED_CHECKPOINT_SCHEMA,
        'participants': {'lfm': {'model_config': config, 'model_state': model_state(model),
                                  'optimizer': optimizer.state_dict(), 'fp8': None}},
        'training': {'step': 1, 'rng': capture_training_rng_state('cpu'),
                     'config_history': [{'config': {'weight_decay': 0, 'latent_ns': 'quintic'}}]}}
    if not shared_parent:
        from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
        saved = {'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model),
            'training': {'schema':'natlang.neuralese-native-sft-training/1',
                'step': 1, 'global_step': 1, 'optimizer': optimizer.state_dict(),
                'identity': {'schema':'natlang.neuralese-native-sft-training/1',
                    'config':config, 'optimizer':'shared-muon-sr-in-backward-v1',
                    'schedule':'row-visit-deficit/2','seed':1,'rank':4,
                    'fp8':{'kind':'off','recipe':'delayed','amax_history_len':16},
                    'objective':None},
                'source_row_bindings':{'prior-foundation-row':'exact-prior-binding'},
                'source_row_order':['prior-foundation-row'],
                'rng': capture_training_rng_state('cpu'),
                'options': {'weight_decay':0,'latent_ns':'quintic','stage_id':'full','epochs':4},
                'exposure_counts': {'prior-foundation-row':3}}}
    held = tmp_path/'held.jsonl'
    held.write_text(json.dumps({'schema': SCHEMA, 'id': 'held-fixture', 'split': 'test',
        'source_groups': ['held'], 'tokenizer_fingerprint': config['tokenizer_fingerprint'],
        'nodes': [{'name': 'held', 'token_ids': [1, 6, 7], 'target_weights': [0, 0, 1]}]})+'\n')
    if not shared_parent:
        from natlang_neuralese.data.native_sources import NativeGraphSource
        saved['training']['held_row_bindings']=NativeGraphSource(held,config['tokenizer_fingerprint'],
            required_split='test',role_ids=roles.ids).bindings
    torch.save(saved, parent)
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(parent))
    with torch.no_grad():
        out = native_step(model, torch.tensor([[1, 4]]))
        first = float(model.logits(out['h_final'][:, -1]).float().log_softmax(-1)[0, 5])
        out = native_step(model, torch.tensor([[5]]), out['cache'])
        stop = float(model.logits(out['h_final'][:, -1]).float().log_softmax(-1)[0, 2])
    capture = {'schema': 'natlang.native-rl-sampling/1', 'policy': identity,
        'prompt': [{'kind': 'tokens', 'token_ids': [1, 4], 'producing': [False, False]}],
        'samples': [{'index': 0, 'token_id': 5, 'logp': first, 'input_role': 'ordinary', 'terminal': False},
                    {'index': 1, 'token_id': 2, 'logp': stop, 'input_role': 'ordinary', 'terminal': True}],
        'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
        'sampling_law': {'temperature': 1., 'seed': 3, 'guidance': False, 'probabilities': 'post-processing-normalized'}}
    rollout = {'task_id': 'fixture-only', 'sample': 0, 'group': 0, 'split': 'train',
        'policy': {'checkpoint_sha256': identity['checkpoint_sha256']}, 'reward': {'quality': 1},
        'turns': [{'messages': [], 'target': {}, 'invocation_id':'ordinary-policy',
                   'sampling': capture, 'replay': {'raw_calls_retained': True}}]}
    source = tmp_path/'rollouts.jsonl'; source.write_text(json.dumps(rollout)+'\n')
    saved_tokenizer = tmp_path/'tokenizer'; tokenizer.save_pretrained(saved_tokenizer)
    argv = ['--rollouts', str(source), '--rl-tokenizer', str(saved_tokenizer), '--rl-mode', 'rft',
        '--continue-from', str(parent), *(['--native-participant', 'lfm'] if shared_parent else []), '--held-native-graphs', str(held),
        '--out', str(tmp_path/'learner'), '--device', 'cpu', '--epochs', '1', '--rank', '4',
        '--learning-rate', '1e-3', '--weight-decay', '0', '--eval-every', '1',
        '--communication-foundation', '--foundation-native-graphs', str(tmp_path/'parent-only-source')]
    assert main(argv) == 0
    slot = tmp_path/'learner/checkpoint.pt'
    state = torch.load(slot, map_location='cpu', weights_only=False)
    assert state['training']['step'] == 1 and state['training']['global_step'] == 2
    assert state['training']['options']['continuation']['checkpoint_sha256'] == sha256_file_hex(parent)
    assert state['training']['optimizer']
    if not shared_parent:
        from natlang_neuralese.train.native_continuation import progress_obligations, restore_sft_progress
        frame=state['training']['sft_progress']['full']
        assert frame['exposure_counts']['prior-foundation-row']==3
        assert progress_obligations({'full':frame})['remaining_graph_visits']==1
        restored=restore_sft_progress(state['training'],'full')
        assert restored['exposure_counts']['prior-foundation-row']==3
        assert restored['global_step']==2
        assert restored['optimizer'] is state['training']['optimizer']
    assert not state['training']['options']['communication_foundation']
    prior_sha = sha256_file_hex(slot)
    assert main(argv) == 0
    assert sha256_file_hex(slot) == prior_sha  # no repeated update/checkpoint rewrite


@pytest.mark.parametrize('foundation',[False,True])
def test_native_factory_retains_full_recipe_budget_and_uses_headless_actor(tmp_path, monkeypatch,foundation):
    from types import SimpleNamespace
    from natlang_neuralese.train import recipe
    parent = tmp_path/'shared.pt'; parent.write_bytes(b'actual-bound-parent')
    adapter = tmp_path/'semantic-adapter.json'
    adapter.write_text(json.dumps({'schema': 'natlang.native-semantic-adapter/1',
        'checkpoint': {'sha256': sha256_file_hex(parent)}, 'models': {'lfm': {}}}))
    monkeypatch.setattr(recipe, 'resolve_recipe_data', lambda path: {'stages': [
        {'id': 'native_full', 'kind': 'native_state_sft', 'parameters': {
            'epochs': 3, 'dtype': 'float32', 'batch_tokens': 8192,'communication_foundation':foundation,
            'task_eval': {'cases': 'registered-held'}, 'task_eval_every': 5000}}]})
    args = SimpleNamespace(model='lfm', native_participant='lfm', checkpoint=parent, semantic_adapter=adapter,
        production_stage='native_full', port=8097, **{key: tmp_path/key for key in (
            'semantic_refresh_config', 'seed_native_graphs', 'held_native_graphs', 'production_recipe',
            'base', 'out', 'config', 'cases', 'runtime')})
    monkeypatch.setattr(recipe,'resolve_stage_inputs',lambda *args:{'foundation_native_graphs':{'path':str(tmp_path/'exact-foundation')}})
    production.configure_native(args)
    saved = json.loads(args.config.read_text())
    if foundation:
        assert saved['foundation_native_graphs']==str(tmp_path/'exact-foundation')
        assert saved['seed_command'][-2:]==['--foundation-native-graphs','{foundation_native_graphs}']
    assert saved['trainer_parameters']['epochs'] == 3
    assert saved['trainer_parameters']['dtype'] == 'float32'
    assert '--native-state' in saved['actor_command'] and '--heads' not in saved['actor_command']
    assert 'natlang_neuralese.rl.native_stage' in saved['learner_command']
    handler = recipe.HANDLERS['native_state_rl']
    assert handler['required_inputs'].isdisjoint(handler['optional_inputs'])
    assert 'rl_block_bundle' in handler['optional_inputs']


def test_bootstrap_binds_actual_export_and_train_files_then_preserves_hot_configs(tmp_path, monkeypatch):
    import shutil
    from natlang_neuralese.train import native_readiness, recipe
    parent = tmp_path/'joint.pt'; parent.write_bytes(b'fixture-parent')
    def artifact(name, value):
        path = tmp_path/name; path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': sha256_file_hex(path)}
    curriculum = artifact('curriculum.json', {})
    adapter = artifact('adapter.json', {'schema': 'natlang.native-semantic-adapter/1',
        'checkpoint': {'path': str(parent), 'sha256': sha256_file_hex(parent)},
        'models': {name: {'native_identity': {'architecture': architecture}}
                   for name, architecture in [('actual-lfm-id', 'lfm2'), ('actual-mellum-id', 'mellum')]}})
    proof = {'adapter': adapter, 'curriculum': curriculum}
    completion = tmp_path/'shared-stage-completion.json'; completion.write_text(json.dumps(proof))
    monkeypatch.setattr(native_readiness, 'validate_completion', lambda *args: (proof, 'verified-fixture'))
    def production_recipe(path):
        model=Path(path).name.removesuffix('-recipe.json')
        # Pending-obligation inheritance resolves the complete parent stage;
        # retain its real required source inputs in this bootstrap fixture.
        sources={'native_graphs': tmp_path/(model+'-train.jsonl'),
                 'held_native_graphs': tmp_path/(model+'-held.jsonl')}
        return {'input_bindings': {role: {'path': str(source), 'sha256': sha256_file_hex(source)}
                                  for role,source in sources.items()},
            'stages': [{'id': 'native_full_sft', 'kind': 'native_state_sft',
            'inputs': {role: role for role in sources},
            'parameters': {'epochs': 1, 'task_eval': {'packet': 'held'}, 'task_eval_every': 5000}}]}
    monkeypatch.setattr(recipe, 'resolve_recipe_data', production_recipe)
    runtime = tmp_path/'runtime/dist/teacher'; runtime.mkdir(parents=True)
    (runtime/'cli.js').write_text('// fixture build')
    source_dir = tmp_path/'published-rl-task-source'
    source_dir.mkdir()
    published = Path(__file__).resolve().parents[2] / 'runs/decision-rich-train-20261016-v1'
    for name in ('cases.jsonl', 'union-audit.json', 'broad-selection-proof.json', 'task-source-manifest.json'):
        shutil.copyfile(published/name, source_dir/name)
    from natlang_neuralese.rl.task_source import resolve_task_source
    task_source = resolve_task_source(source_dir/'task-source-manifest.json')
    task_source_binding = {'manifest': str(task_source['manifest']),
        'manifest_sha256': task_source['manifest_sha256'], 'cases_sha256': task_source['cases_sha256'],
        'rows': task_source['rows'], 'audit_sha256': task_source['audit_sha256'],
        'selection_proof_sha256': task_source['selection_proof_sha256']}
    models = []
    for model in ('lfm', 'mellum'):
        models.append({'model': model, 'production_recipe': artifact(model+'-recipe.json', {}),
            'production_stage': 'native_full_sft', 'base': str(tmp_path/'base'),
            'seed_native_graphs': artifact(model+'-train.jsonl', {'fixture': 'train'}),
            'held_native_graphs': artifact(model+'-held.jsonl', {'fixture': 'held'}),
            'cases': {'path': str(task_source['cases']), 'sha256': task_source['cases_sha256']},
            'task_source': task_source_binding,
            'runtime': str(runtime.parent.parent), 'port': 8097 if model == 'lfm' else 8098})
    plan = tmp_path/'bootstrap-plan.json'
    plan.write_text(json.dumps({'schema': 'natlang.native-rl-bootstrap/1', 'models': models}))
    configs = native_rounds.bootstrap_configs(plan, completion, tmp_path/'rl')
    assert len(configs) == 2
    first = json.loads(configs[0].read_text())
    assert first['native_participant'] == 'actual-lfm-id'
    assert first['checkpoint'] == str(parent)
    assert first['bootstrap_binding']['adapter_sha256'] == adapter['sha256']
    assert first['task_source'] == task_source_binding
    assert all(json.loads(path.read_text())['cases'] == models[0]['cases']['path'] for path in configs)
    first['workers'] = 7; configs[0].write_text(json.dumps(first))
    native_rounds.bootstrap_configs(plan, completion, tmp_path/'rl')
    assert json.loads(configs[0].read_text())['workers'] == 7
    Path(models[0]['cases']['path']).write_text('{}')
    # Even an unchanged plan rechecks admitted artifacts before reuse.
    with pytest.raises(ValueError, match='changed from its manifest binding'):
        native_rounds.bootstrap_configs(plan, completion, tmp_path/'rl')


def test_joint_stop_waits_for_both_learners_without_refresh_or_publication(tmp_path, monkeypatch):
    import threading
    from natlang_neuralese.rl import native_refresh
    from natlang_neuralese.train.loop import StopSignal
    parent = tmp_path/'parent'; parent.write_bytes(b'joint')
    adapter = tmp_path/'adapter'; adapter.write_text('{}')
    paths = []
    for name in ('lfm', 'mellum'):
        path = tmp_path/(name+'.json')
        path.write_text(json.dumps({'native_state': True, 'native_participant': name,
            'out': str(tmp_path/name), 'checkpoint': str(parent), 'semantic_adapter': str(adapter),
            'semantic_refresh_config': str(tmp_path/'semantic')}))
        paths.append(path)
    barrier = threading.Barrier(2); stop = StopSignal(); drained = []
    def learner(path, **kwargs):
        assert kwargs['stop'] is stop
        barrier.wait(timeout=10)
        stop.request()
        drained.append(path)
        return {'phase': 'learner_pending', 'partial_checkpoint': 'actual-owned-worker-output'}
    monkeypatch.setattr(native_rounds, 'run', learner)
    monkeypatch.setattr(native_refresh, 'refresh', lambda **kw: pytest.fail('stop cannot publish new adapter'))
    result = native_rounds.run_rounds(paths, tmp_path/'joint', rounds=1, stop=stop)
    assert result['phase'] == 'requested_stop' and len(drained) == 2
    assert len(result['states']) == 2
