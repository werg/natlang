"""Whole-graph policy objectives over exact causal-prefix packing equal the unshared episode plan.

Multi-turn episodes re-render each turn's full prompt (system/tool schemas plus history); the learner forwards the
token trie of all turns of all episodes in a batch. Every sampled action keeps its causal context, JOLT+ twin and
decision readouts map through the original-token -> physical-row map, and the parameter gradients equal the
unshared plan's: tightly in FP32, and in BF16 within the unshared BF16 plan's own distance from an FP64 reference.
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.common.sampling import behavior_top_k_declaration, encode_top_k
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.model.native_graph_plan import Invocation, compile_graph
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import packed_graph_loss
from test_shared_state import tiny

K = 8
TEMPERATURE = .8
JOLT = {'schema': 'jolt-plus/1', 'beta_s': 0.7, 'beta_t': 0.3,
        'outcome': {'schema': 'score-centered/1', 'k': K, 'importance': 'tis'}}


def _device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return torch.device(name)


def _model(family, device='cpu'):
    model = SharedStateModel(tiny(family).to(device), marker_ids=(58, 59, 60, 61), rank=4)
    with torch.no_grad():
        model.adapter.up.weight.normal_(std=.1)
    for parameter in model.port.parameters():
        parameter.requires_grad_(True)
    return model


def _identity(model):
    return {'checkpoint_sha256': '1'*64, 'tokenizer_fingerprint': '2'*64,
            'embedding_weight_sha256': '3'*64, 'architecture': model.port.config.model_type}


def multi_turn_episodes(model, identity, *, episodes=2, turns=3, system=40, seed=0):
    """Actual tiny-actor samples; turn k re-renders system prompt + every earlier turn + its new user text."""
    generator = torch.Generator().manual_seed(seed)
    draw = lambda n: [int(t) for t in torch.randint(3, 58, (n,), generator=generator)]
    header = draw(system)
    rows = []
    for sample in range(episodes):
        history, raw = list(header), []
        for turn in range(turns):
            prompt = history + draw(3)
            sampled = draw(2 + (sample + turn) % 3) + [2]
            with torch.no_grad():
                plan = compile_graph([Invocation('t', tuple(prompt + sampled))])
                hidden = model(plan, retain_states=False)['h_final'][0]
                log_probs = model.logits(hidden[len(prompt) - 1:len(prompt) - 1 + len(sampled)]).float().div(
                    TEMPERATURE).log_softmax(-1).cpu()
            capture = {'schema': 'natlang.native-rl-sampling/1', 'policy': dict(identity),
                'prompt': [{'kind': 'tokens', 'token_ids': prompt, 'producing': [False]*len(prompt)}],
                'samples': [{'index': i, 'token_id': token, 'logp': float(log_probs[i, token]),
                             'input_role': 'ordinary', 'terminal': token == 2,
                             'top_k': encode_top_k(log_probs[i], K)} for i, token in enumerate(sampled)],
                'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
                'sampling_law': {'temperature': TEMPERATURE, 'seed': 3, 'guidance': False,
                                 'probabilities': 'post-processing-normalized'},
                'behavior_top_k': behavior_top_k_declaration(K)}
            raw.append({'messages': [], 'target': {}, 'invocation_id': 'root', 'sampling': capture,
                        'replay': {'raw_calls_retained': True}})
            history = prompt + sampled + draw(4)  # tool result rendered into the next turn's prompt
        rows.append({'task_id': 'prefix-task', 'sample': sample, 'split': 'train', 'group': 'g',
                     'reward': {'quality': float(sample % 2)}, 'policy': {'checkpoint_sha256': '1'*64},
                     'turns': raw})
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(rows)
    return rows


def _objective(tmp_path, spec, name):
    from natlang_neuralese.rl.native_replay import NativePolicyObjective
    path = tmp_path / f'{name}.json'
    path.write_text(json.dumps(spec))
    return NativePolicyObjective(path)


def _prepared(model, tmp_path, mode, policy_loss=None, **options):
    from natlang_neuralese.rl.native_replay import prepare_objective
    identity = _identity(model)
    roles = SimpleNamespace(produce_open=58, produce_close=59, consume_open=60, consume_close=61)
    graphs, spec = prepare_objective(multi_turn_episodes(model, identity, **options), identity=identity,
        roles=roles, fetch_block=lambda key: None, mode=mode, policy_loss=policy_loss)
    examples = [parse_graph(row, tokenizer_fingerprint=identity['tokenizer_fingerprint']) for row in graphs]
    return examples, _objective(tmp_path, spec, f'{mode}-{model.port.embedding_weight.dtype}')


def loss_and_gradients(model, examples, objective, *, share):
    model.zero_grad(set_to_none=True)
    loss, details = packed_graph_loss(model, examples, objective=objective, chunk_tokens=3,
                                      share_causal_prefixes=share, collect_accuracy=False)
    loss.backward()
    return loss.detach().double().cpu(), {name: p.grad.detach().double().cpu() for name, p in model.named_parameters()
                                    if p.grad is not None}, details


def _relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('family', ['lfm3', 'lfm'])
@pytest.mark.parametrize('mode,policy_loss', [('rft', None), ('grpo', JOLT)])
def test_fp32_shared_policy_plan_equals_unshared_episode_rows(tmp_path, family, mode, policy_loss, device):
    model = _model(family, _device(device))
    examples, objective = _prepared(model, tmp_path, mode, policy_loss)
    assert all(len(example.plan.invocations) == 3 for example in examples)
    reference = copy.deepcopy(model)
    loss, gradients, details = loss_and_gradients(model, examples, objective, share=True)
    expected, expected_gradients, unshared = loss_and_gradients(reference, examples, objective, share=False)
    # The trie forwards each shared token once across turns and episodes.
    assert details['original_physical_tokens'] == unshared['packed_physical_tokens']
    assert details['packed_physical_tokens'] < details['original_physical_tokens'] // 2
    assert details['sampled_tokens'] == unshared['sampled_tokens'] > 0
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    assert gradients.keys() == expected_gradients.keys() and any(name.startswith('port.hf.') for name in gradients)
    for name, gradient in gradients.items():
        torch.testing.assert_close(gradient, expected_gradients[name], rtol=2e-4, atol=3e-6, msg=name)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('family', ['lfm3'])
@pytest.mark.parametrize('mode,policy_loss', [('rft', None), ('grpo', JOLT)])
def test_bf16_shared_gradients_are_within_the_unshared_bf16_plan_error(tmp_path, family, mode, policy_loss, device):
    """BF16 fan-in: the shared plan's distance from an FP64 reference stays within the unshared plan's own."""
    source = _model(family, _device(device))
    examples, objective = _prepared(source, tmp_path, mode, policy_loss, episodes=3, turns=4, system=60)
    exact = copy.deepcopy(source).to(torch.bfloat16).double().cpu()  # the BF16 weights, evaluated exactly
    reference, reference_gradients, _ = loss_and_gradients(exact, examples, objective, share=False)
    measured = {}
    for share in (False, True):
        model = copy.deepcopy(source).to(torch.bfloat16)
        loss, gradients, details = loss_and_gradients(model, examples, objective, share=share)
        errors = {name: _relative(gradient, reference_gradients[name]) for name, gradient in gradients.items()}
        measured[share] = (abs(float(loss - reference)), errors, details['packed_physical_tokens'])
    assert measured[True][2] < measured[False][2] // 2
    assert measured[True][0] <= 2 * measured[False][0] + 1e-3
    worst = {share: max(errors.values()) for share, (_, errors, _) in measured.items()}
    total = {share: sum(errors.values()) / len(errors) for share, (_, errors, _) in measured.items()}
    print(json.dumps({'family': family, 'mode': mode, 'device': device, 'loss_error': {str(k): v[0] for k, v in measured.items()},
                      'worst_relative': {str(k): v for k, v in worst.items()},
                      'mean_relative': {str(k): v for k, v in total.items()}}))
    # Normal BF16 tolerance: no parameter's shared-plan error exceeds 1.5x the unshared plan's worst error.
    assert worst[True] <= 1.5 * worst[False]
    assert total[True] <= 1.5 * total[False]


def _assert_shared_equals_unshared(model, examples, objective, *, minimum_saving=1):
    reference = copy.deepcopy(model)
    loss, gradients, details = loss_and_gradients(model, examples, objective, share=True)
    expected, expected_gradients, unshared = loss_and_gradients(reference, examples, objective, share=False)
    assert details['packed_physical_tokens'] + minimum_saving <= unshared['packed_physical_tokens']
    for key in ('sampled_tokens', 'sampled_decisions'):
        assert details[key] == unshared[key]
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-7)
    assert gradients.keys() == expected_gradients.keys()
    for name, gradient in gradients.items():
        torch.testing.assert_close(gradient, expected_gradients[name], rtol=2e-4, atol=3e-6, msg=name)
    return details


@pytest.mark.parametrize('role', ['student', 'teacher'])
def test_jolt_twins_and_guidance_imports_share_only_raw_prefixes(tmp_path, role):
    """Privileged twins (student OPSD / teacher KL) and writer->consumer imports: twin queries map exactly."""
    from test_rl_jolt import episode_with_top_k, load, objective_for, prepare, rollouts
    model, roles, identity = load('lfm')
    episode = episode_with_top_k(model, roles, identity, 'write-only' if role == 'student' else 'import')
    graphs, spec = prepare(rollouts(episode.turns, role=role), episode, roles, identity)
    examples = [parse_graph(graph, tokenizer_fingerprint=identity['tokenizer_fingerprint']) for graph in graphs]
    assert any(node.imports for example in examples for node in example.plan.invocations)
    assert all(spec['rows'][graph['id']]['jolt']['privileged'] for graph in graphs)
    details = _assert_shared_equals_unshared(model, examples, objective_for(tmp_path, spec))
    assert details['jolt']['privileged_episodes'] == 2


def test_jolt_decision_twin_options_map_through_shared_rows(tmp_path):
    from test_rl_jolt import episode_with_top_k, load, objective_for, prepare, rollouts
    model, roles, identity = load('lfm')
    episode = episode_with_top_k(model, roles, identity, 'write-only')
    root = next(turn for turn in episode.turns if turn['graph_turn'] == 'root#turn1')
    old = root['sampling']
    root['sampling'] = {'schema': 'natlang.native-decision-capture/1', 'policy': old['policy'],
        'prompt': old['prompt'], 'privileged': old['privileged'], 'continuations': [[11, 2], [13, 2]],
        'decision_sample': {'schema': 'natlang.decision-sample/1', 'log_probs': [-1., -2.],
                            'chosen': 0, 'temperature': 1., 'seed': 3}}
    graphs, spec = prepare(rollouts(episode.turns), episode, roles, identity)
    assert all(len(spec['rows'][graph['id']]['decision_actions'][0]['twin_options']) == 2 for graph in graphs)
    examples = [parse_graph(graph, tokenizer_fingerprint=identity['tokenizer_fingerprint']) for graph in graphs]
    details = _assert_shared_equals_unshared(model, examples, objective_for(tmp_path, spec))
    assert details['sampled_decisions'] == 2


def test_unsupported_policy_graphs_keep_their_unshared_plan(tmp_path):
    """Archived external states (and per-invocation actors / semantic codecs) retain the original layout."""
    from test_native_rl_replay import actor_rollout
    from natlang_neuralese.rl.native_replay import prepare_objective
    model, roles, identity, _, block, turns = actor_rollout('lfm')
    raw = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'], 'sampling': turn['sampling'],
            'replay': {'raw_calls_retained': True}} for turn in turns]
    episodes = [{'task_id': 't', 'sample': 0, 'split': 'train', 'group': 'g', 'reward': {'quality': 1},
                 'policy': {'checkpoint_sha256': identity['checkpoint_sha256']}, 'turns': raw}]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    graphs, spec = prepare_objective(episodes, identity=identity, roles=roles, fetch_block=lambda key: block,
                                     mode='rft')
    examples = [parse_graph(graph, tokenizer_fingerprint=identity['tokenizer_fingerprint']) for graph in graphs]
    objective = _objective(tmp_path, spec, 'internal')
    assert objective.shares_causal_prefixes(examples)  # in-episode producer/consumer edges are supported
    objective.semantic_channel = object()
    assert not objective.shares_causal_prefixes(examples)
    objective.semantic_channel = None
    external = SimpleNamespace(id=examples[0].id, external_states={'x': {'id': 'block'}})
    assert not objective.shares_causal_prefixes([external])
