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


def test_native_parallel_round_merges_both_updates_before_refresh(tmp_path, monkeypatch):
    import threading
    from natlang_neuralese.rl import native_refresh
    parent = tmp_path/'parent.pt'; parent.write_bytes(b'joint-parent')
    adapter = tmp_path/'adapter.json'; adapter.write_text('{}')
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
        assert kwargs == {'rounds': 1, 'defer_native_refresh': True}
        barrier.wait(timeout=10)  # proves independent learner jobs overlap
        directory = Path(config['out']); directory.mkdir()
        state = {'checkpoint': str(parent), 'round': 0, 'trained_rounds': 0,
            'pending_native_checkpoint': updates[config['native_participant']]}
        (directory/'state.json').write_text(json.dumps(state))
        return state
    monkeypatch.setattr(native_rounds, 'run', learner)
    calls = []
    def refresh(**kwargs):
        assert kwargs['updates'] == updates
        calls.append(kwargs)
        return {'checkpoint': {'path': 'refreshed-shared', 'sha256': 'new'},
            'semantic_adapter': {'path': 'refreshed-adapter', 'sha256': 'adapter'},
            'updated_participants': {name: {'sha256': sha256_file_hex(path)} for name, path in updates.items()}}
    monkeypatch.setattr(native_refresh, 'refresh', refresh)
    result = native_rounds.run_rounds(configs, tmp_path/'joint', rounds=1, device='cpu')
    assert len(calls) == 1 and result['deployment']['checkpoint']['path'] == 'refreshed-shared'
    for name in updates:
        state = json.loads((tmp_path/name/'state.json').read_text())
        assert state['round'] == 1 and state['checkpoint'] == 'refreshed-shared'


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


def test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path):
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
    torch.save({'schema': SHARED_CHECKPOINT_SCHEMA,
        'participants': {'lfm': {'model_config': config, 'model_state': model_state(model),
                                  'optimizer': optimizer.state_dict(), 'fp8': None}},
        'training': {'step': 1, 'rng': capture_training_rng_state('cpu'),
                     'config_history': [{'config': {'weight_decay': 0, 'latent_ns': 'quintic'}}]}}, parent)
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
        'turns': [{'messages': [], 'target': {}, 'sampling': capture, 'replay': {'raw_calls_retained': True}}]}
    source = tmp_path/'rollouts.jsonl'; source.write_text(json.dumps(rollout)+'\n')
    held = tmp_path/'held.jsonl'
    held.write_text(json.dumps({'schema': SCHEMA, 'id': 'held-fixture', 'split': 'test',
        'source_groups': ['held'], 'tokenizer_fingerprint': config['tokenizer_fingerprint'],
        'nodes': [{'name': 'held', 'token_ids': [1, 6, 7], 'target_weights': [0, 0, 1]}]})+'\n')
    saved_tokenizer = tmp_path/'tokenizer'; tokenizer.save_pretrained(saved_tokenizer)
    argv = ['--rollouts', str(source), '--rl-tokenizer', str(saved_tokenizer), '--rl-mode', 'rft',
        '--continue-from', str(parent), '--native-participant', 'lfm', '--held-native-graphs', str(held),
        '--out', str(tmp_path/'learner'), '--device', 'cpu', '--epochs', '1', '--rank', '4',
        '--learning-rate', '1e-3', '--weight-decay', '0', '--eval-every', '1']
    assert main(argv) == 0
    slot = tmp_path/'learner/checkpoint.pt'
    state = torch.load(slot, map_location='cpu', weights_only=False)
    assert state['training']['step'] == 1 and state['training']['global_step'] == 2
    assert state['training']['options']['continuation']['checkpoint_sha256'] == sha256_file_hex(parent)
    assert state['training']['optimizer']
    prior_sha = sha256_file_hex(slot)
    assert main(argv) == 0
    assert sha256_file_hex(slot) == prior_sha  # no repeated update/checkpoint rewrite


def test_native_factory_retains_full_recipe_budget_and_uses_headless_actor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.train import recipe
    parent = tmp_path/'shared.pt'; parent.write_bytes(b'actual-bound-parent')
    adapter = tmp_path/'semantic-adapter.json'
    adapter.write_text(json.dumps({'schema': 'natlang.native-semantic-adapter/1',
        'checkpoint': {'sha256': sha256_file_hex(parent)}, 'models': {'lfm': {}}}))
    monkeypatch.setattr(recipe, 'resolve_recipe_data', lambda path: {'stages': [
        {'id': 'native_full', 'kind': 'native_state_sft', 'parameters': {
            'epochs': 3, 'dtype': 'float32', 'batch_tokens': 8192,
            'task_eval': {'cases': 'registered-held'}, 'task_eval_every': 5000}}]})
    args = SimpleNamespace(model='lfm', native_participant='lfm', checkpoint=parent, semantic_adapter=adapter,
        production_stage='native_full', port=8097, **{key: tmp_path/key for key in (
            'semantic_refresh_config', 'seed_native_graphs', 'held_native_graphs', 'production_recipe',
            'base', 'out', 'config', 'cases', 'runtime')})
    production.configure_native(args)
    saved = json.loads(args.config.read_text())
    assert saved['trainer_parameters']['epochs'] == 3
    assert saved['trainer_parameters']['dtype'] == 'float32'
    assert '--native-state' in saved['actor_command'] and '--heads' not in saved['actor_command']
    assert 'natlang_neuralese.rl.native_stage' in saved['learner_command']
    handler = recipe.HANDLERS['native_state_rl']
    assert handler['required_inputs'].isdisjoint(handler['optional_inputs'])
    assert 'rl_block_bundle' in handler['optional_inputs']
