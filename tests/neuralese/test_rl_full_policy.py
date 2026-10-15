"""Policy-gradient mathematics and exact serving replay for both backbone families."""
import json
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.rl.full_policy import (
    FullPolicyObjective, build_documents, clipped_token_loss, validate_rollouts,
)


def rollout(quality, **changes):
    row = {'task_id': 't', 'group': 0, 'sample': quality, 'split': 'train',
           'policy': {'checkpoint_sha256': 'policy', 'temperature': 1.0},
           'reward': {'quality': quality}, 'turns': [{'messages': [],
           'target': {'role': 'assistant', 'content': 'a'}, 'sampling': {
               'schema': 'natlang.rl-sampling/1', 'guided': False, 'temperature': 1.,
               'prompt': [['tokens', [1]]], 'items': [['tok', 2]],
               'samples': [{'item': 0, 'token': 2, 'logp': -1.}]}, 'replay': {'raw_calls_retained': True}}]}
    return {**row, **changes}


def test_token_objective_has_correct_signed_gradient_and_asymmetric_clipping():
    old = torch.tensor([-2., -2., -2., -2.])
    now = (old + torch.tensor([0., .4, -.4, 0.])).requires_grad_()
    advantages = torch.tensor([1., 1., -1., -1.])
    loss = clipped_token_loss(now, old, advantages).sum()
    loss.backward()
    assert now.grad[0] < 0 and now.grad[3] > 0
    assert now.grad[1] == 0 and now.grad[2] == 0
    assert torch.isfinite(now.grad).all()


def test_signed_gradient_matches_finite_difference_away_from_clip_boundary():
    old = torch.tensor([-2., -3.], dtype=torch.float64)
    point = torch.tensor([-2.05, -2.97], dtype=torch.float64, requires_grad=True)
    advantage = torch.tensor([.7, -.4])
    value = clipped_token_loss(point, old, advantage).sum()
    value.backward()
    for i in range(2):
        plus, minus = point.detach().clone(), point.detach().clone()
        plus[i] += .001
        minus[i] -= .001
        numeric = (clipped_token_loss(plus, old, advantage).sum() -
                   clipped_token_loss(minus, old, advantage).sum()) / .002
        assert torch.allclose(point.grad[i].float(), numeric, atol=1e-4)


@pytest.mark.parametrize('changes', [
    {'split': 'test'}, {'split': None}, {'policy': {'checkpoint_sha256': 'other', 'temperature': 1}},
    {'policy': {'checkpoint_sha256': 'policy', 'temperature': .6}},
    {'policy': {'checkpoint_sha256': 'policy', 'temperature': 1, 'top_p': .9}},
    {'reward': {'quality': float('nan')}}, {'turns': []},
])
def test_bad_sampling_or_split_is_never_silently_trained(changes):
    with pytest.raises(ValueError):
        validate_rollouts([rollout(1, **changes)], 'policy')


def test_groups_ignore_infrastructure_and_preserve_only_real_spread():
    docs = build_documents([rollout(1), rollout(0), rollout(0, reward=None),
                            rollout(1, task_id='easy'), rollout(1, task_id='easy')], policy_id='policy')
    assert len(docs) == 2
    assert docs[0]['advantage'] > 0 > docs[1]['advantage']


def test_blocks_require_graph_bundle_instead_of_lossy_text_flattening():
    r = rollout(1)
    r['turns'][0]['target']['content'] = [{'type': 'neuralese', 'id': 'value'}]
    with pytest.raises(ValueError, match='graph/block'):
        validate_rollouts([r], 'policy')


def test_durable_behavior_is_not_recaptured_from_updated_weights(tmp_path, monkeypatch):
    engine = SimpleNamespace(backbone=None, heads=None)
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'documents': [{'document_sha256': 'doc', 'advantage': 1.,
        'messages': [], 'target': {'role': 'assistant', 'content': 'x'}}]}))
    windows = [{'document': 'doc', 'ids': [1, 2, 3]}]
    first = FullPolicyObjective(engine, spec, windows, tmp_path)
    parameter = torch.tensor(-2., requires_grad=True)
    monkeypatch.setattr(first, '_score', lambda row: parameter.expand(2))
    value = first.forward(windows)[0]
    value.backward()
    assert parameter.grad < 0
    restored = FullPolicyObjective(engine, spec, windows, tmp_path)
    monkeypatch.setattr(restored, '_score', lambda row: torch.tensor([-1., -1.], requires_grad=True))
    restored._behavior()
    assert restored.cache['doc'].tolist() == [-2., -2.]


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_actual_backbone_policy_gradient_update(family, tmp_path):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.grad import GradSession
    from natlang_neuralese.serve.store import TensorStore
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    torch.manual_seed(13)
    if family == 'lfm':
        from transformers import Lfm2Config, Lfm2ForCausalLM
        config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            block_multiple_of=8, block_auto_adjust_ff_dim=False,
            layer_types=['conv', 'full_attention', 'conv'])
        backbone = PortBackbone(Lfm2ForCausalLM(config).eval(), ControlTokens(62, 63), fast=False)
    else:
        from test_mellum_port import tiny_mellum
        from natlang_neuralese.maple.model import load_maple
        tiny_mellum(tmp_path)
        source = tmp_path
        # Same loader/port path as production, reduced architecture only.
        model = load_maple(source, device='cpu', dtype=torch.float32, ternary_attention=False)
        from natlang_neuralese.maple.maple_port import MaplePortBackbone
        model.natlang_precision = 'bf16'
        backbone = MaplePortBackbone(model, ControlTokens(126, 127), fast=False)
    named = configure_backbone_training(backbone, 'latent')
    heads = PortHeads(backbone, cutoff=backbone.num_layers, max_length=8, profile="raw-token-v1").eval()
    engine = SimpleNamespace(backbone=backbone, heads=heads, device='cpu',
        token_choice_projection=None)
    engine.advance_token_readout = lambda history, hidden, last_only=False: (backbone.logits(hidden), None)
    session = GradSession(engine)
    prompt = [('tok', i) for i in [1, 4, 9]]
    target = [('tok', i) for i in [7, 8]]
    with torch.no_grad():
        old = session._score(prompt, target, {}, write_terms=False)['token_logp'].detach()
    optimizer = torch.optim.SGD([p for _, p in named], lr=.002)
    now = session._score(prompt, target, {}, write_terms=False)['token_logp']
    objective = clipped_token_loss(now, old, 1.).mean()
    objective.backward()
    norm = sum(float(p.grad.float().square().sum()) for _, p in named if p.grad is not None)
    assert norm > 0 and math_isfinite(norm)
    optimizer.step()
    with torch.no_grad():
        after = session._score(prompt, target, {}, write_terms=False)['token_logp']
    assert after.sum() > old.sum()


def math_isfinite(value):
    import math
    return math.isfinite(value)


def test_replay_geometry_keeps_whole_context_and_one_signed_sample(tmp_path):
    from natlang_neuralese.rl.full_policy import exact_replay_windows
    capture = rollout(1)['turns'][0]['sampling']
    capture['prompt'] = [['tokens', list(range(20))]]
    capture['samples'] = [{'item': 0, 'token': 2, 'logp': -1.}]
    spec = tmp_path / 's.json'
    spec.write_text(json.dumps({'documents': [
        {'document_sha256': 'positive', 'sampling': capture, 'advantage': 1},
        {'document_sha256': 'negative', 'sampling': capture, 'advantage': -1}]}))
    controls = SimpleNamespace(open_id=62, close_id=63)
    engine = SimpleNamespace(backbone=SimpleNamespace(controls=controls),
                             heads=SimpleNamespace(read_markers=False), context=64)
    # Simulate the ordinary loader splitting each turn into two windows.
    windows = [{'document': key, 'groups': ['g']} for key in ['positive', 'positive', 'negative', 'negative']]
    train, targets, held = exact_replay_windows(engine, spec, windows, [])
    assert len(train) == len(targets) == 2 and not held
    assert all(len(w['ids']) == 23 for w in train)
    assert [w['document'] for w in train] == ['positive', 'negative']


def test_ordinary_production_held_windows_are_never_deduplicated(tmp_path):
    from natlang_neuralese.rl.full_policy import exact_replay_windows
    spec = tmp_path / 'empty.json'
    spec.write_text(json.dumps({'documents': []}))
    first = {'document': 'ordinary-held', 'offset': 0}
    second = {'document': 'ordinary-held', 'offset': 1024}
    train, targets, held = exact_replay_windows(object(), spec, [], [first, second])
    assert held == [first, second] and held[0] is first and held[1] is second
    assert not train and not targets


def test_rl_never_silently_ignores_family_distillation(tmp_path):
    from natlang_neuralese.train.direct_sft import parse
    with pytest.raises(SystemExit):
        parse(['--heads', 'heads.pt', '--out', str(tmp_path), '--records', 'r.jsonl',
               '--rl-objective', 'rl.json', '--family', '{}'])


def test_capture_only_blocks_require_durable_bundle():
    r = rollout(1)
    r['turns'][0]['sampling']['prompt'] = [['block', 'nz1_' + 'a' * 52]]
    with pytest.raises(ValueError, match='graph/block'):
        validate_rollouts([r], 'policy')


def test_registered_rft_only_rollouts_cannot_become_grpo():
    with pytest.raises(ValueError, match='RFT-only'):
        validate_rollouts([rollout(1, training_admission='rft-only')], 'policy')


def test_replay_memory_counts_transitive_writer_context(tmp_path):
    from natlang_neuralese.rl.full_policy import exact_replay_windows
    from natlang_neuralese.train.direct_sft import _window_length
    block = 'nz1_' + 'a' * 52
    reader = rollout(1)['turns'][0]['sampling']
    reader['prompt'] = [['block', block]]
    writer = {**reader, 'prompt': [['tokens', list(range(30))]], 'items': [['block', block]], 'samples': []}
    manifest = tmp_path / 'blocks.json'
    manifest.write_text(json.dumps({'blocks': {block: {'length': 2}}}))
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'block_bundle': str(manifest), 'documents': [{
        'document_sha256': 'doc', 'sampling': reader, 'producers': [{'sampling': writer}]}]}))
    engine = SimpleNamespace(backbone=SimpleNamespace(controls=SimpleNamespace(open_id=62, close_id=63)),
                             heads=SimpleNamespace(read_markers=False), context=64)
    train, _, _ = exact_replay_windows(engine, spec, [{'document': 'doc', 'groups': []}], [])
    assert train[0]['visible_forward_tokens'] == 3
    assert train[0]['producer_forward_tokens'] == 32
    assert _window_length(train[0]) == 35


def test_value_rft_reuses_exact_replay_but_never_importance_clips_ce(tmp_path, monkeypatch):
    engine = SimpleNamespace(backbone=None, heads=None)
    spec = tmp_path / 'rft.json'
    spec.write_text(json.dumps({'mode': 'rft', 'documents': [{
        'document_sha256': 'doc', 'advantage': -1., 'messages': [], 'target': {'content': 'x'}}]}))
    objective = FullPolicyObjective(engine, spec, [{'document': 'doc'}], tmp_path)
    parameter = torch.tensor(-2., requires_grad=True)
    objective.cache = {'doc': torch.tensor([-20., -20.])}  # would entirely clip positive GRPO
    monkeypatch.setattr(objective, '_score', lambda row: parameter.expand(2))
    value = objective.forward([{'document': 'doc', 'ids': [1, 2, 3]}])[0]
    value.backward()
    assert value == 2. and parameter.grad == -1.
