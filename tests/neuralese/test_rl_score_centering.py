"""Score-centered policy gradients (arXiv 2609.20807): formula, capture and whole-graph trainer path."""
import copy
import json

import pytest
import torch

from natlang_neuralese.common.sampling import decode_top_k, encode_top_k
from natlang_neuralese.rl.policy_math import (importance_weight, policy_loss_declaration,
                                              score_centered_surrogate, score_centered_token_loss)


def f(ratio, mode):
    return importance_weight(ratio, mode)


def literal_surrogate(log_probs, labels, behavior_logp, top_ids, top_logq, mode, eps):
    """Direct transcription of -R(sg[w_y] log p_y - sum_H sg[q_v w_v - alpha p_v] log p_v), float64."""
    log_probs = log_probs.double()
    p = log_probs.detach().exp()
    values = []
    for row in range(log_probs.shape[0]):
        valid = (top_ids[row] >= 0) & (top_logq[row] > float('-inf'))
        ids = top_ids[row][valid]
        q = top_logq[row][valid].double().exp()
        p_h = p[row, ids]
        rho = (1 - q.sum()).clamp(min=eps) / (1 - p_h.sum()).clamp(min=eps)
        alpha = rho * f(1 / rho, mode)
        w = f(p_h / q, mode)
        y = labels[row]
        w_y = f((log_probs[row, y].detach() - behavior_logp[row].double()).exp(), mode)
        values.append(w_y * log_probs[row, y] - ((q * w - alpha * p_h).detach() * log_probs[row, ids]).sum())
    return torch.stack(values)


def full_vocabulary_surrogate(log_probs, labels, behavior_logp, logq, mode):
    """Eq. 13 (with optional IS weights): sg[w_y] log p_y - sum_V sg[q_v w_v] log p_v."""
    log_probs = log_probs.double()
    q = logq.double().exp()
    w = f(log_probs.detach().exp() / q, mode)
    y = labels[:, None]
    w_y = f((log_probs.detach().gather(1, y).squeeze(1) - behavior_logp.double()).exp(), mode)
    return w_y * log_probs.gather(1, y).squeeze(1) - ((q * w).detach() * log_probs).sum(1)


def gradient(function, logits):
    logits = logits.detach().clone().requires_grad_(True)
    function(torch.log_softmax(logits, -1)).sum().backward()
    return logits.grad.double()


def setup(n=6, width=40, k=8, seed=0, spread=1.):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, width, generator=generator) * 2
    sampler = logits + spread * torch.randn(n, width, generator=generator)
    logq = torch.log_softmax(sampler, -1)
    labels = torch.multinomial(logq.exp(), 1, generator=generator).squeeze(1)
    values, ids = torch.topk(logq, k)
    return logits, logq, labels, ids, values


@pytest.mark.parametrize('mode', ['none', 'tis', 'mis'])
def test_whole_vocabulary_top_k_matches_brute_force_score_centering(mode):
    logits, logq, labels, ids, values = setup(k=40)
    behavior = logq.gather(1, labels[:, None]).squeeze(1)
    actual = gradient(lambda lp: score_centered_surrogate(lp, labels, behavior, ids, values, importance=mode)[0], logits)
    expected = gradient(lambda lp: full_vocabulary_surrogate(lp, labels, behavior, logq, mode), logits)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('mode', ['none', 'tis', 'mis'])
@pytest.mark.parametrize('k', [1, 4, 16])
def test_top_k_gradient_equals_literal_formula_inside_and_outside_h(mode, k):
    logits, logq, labels, ids, values = setup(n=32, k=k, seed=k)
    behavior = logq.gather(1, labels[:, None]).squeeze(1)
    inside = (ids == labels[:, None]).any(1)
    assert inside.any() and (~inside).any(), 'fixture covers y inside and outside H'
    actual = gradient(lambda lp: score_centered_surrogate(lp, labels, behavior, ids, values, importance=mode)[0], logits)
    expected = gradient(lambda lp: literal_surrogate(lp, labels, behavior, ids, values, mode, 1e-6), logits)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('mode', ['none', 'tis', 'mis'])
def test_tail_rescaling_is_exact_when_sampler_tail_is_proportional(mode):
    # q = p on H after a shift, and q_tail = rho p_tail: the top-k estimator then equals the full one exactly.
    logits, _, labels, _, _ = setup(n=8, width=30, seed=3)
    p = torch.softmax(logits.double(), -1)
    k = 6
    _, ids = torch.topk(p, k)
    q = p.clone()
    in_h = torch.zeros_like(p, dtype=torch.bool).scatter(1, ids, True)
    head = q.masked_fill(~in_h, 0) * 1.1
    head = head / head.sum(1, keepdim=True) * (p.masked_fill(~in_h, 0).sum(1, keepdim=True) * .9)
    tail_q = 1 - head.sum(1, keepdim=True)
    tail = p.masked_fill(in_h, 0)
    q = head + tail / tail.sum(1, keepdim=True) * tail_q
    logq = q.log().float()
    values = logq.gather(1, ids)
    order = values.argsort(1, descending=True)
    ids, values = ids.gather(1, order), values.gather(1, order)
    behavior = logq.gather(1, labels[:, None]).squeeze(1)
    actual = gradient(lambda lp: score_centered_surrogate(lp, labels, behavior, ids, values, importance=mode)[0], logits)
    expected = gradient(lambda lp: full_vocabulary_surrogate(lp, labels, behavior, logq, mode), logits)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)


def test_importance_weights_and_masked_entries_and_loss_sign():
    r = torch.tensor([.1, .5, 1., 2., 3., 5., 6.], dtype=torch.float64)
    assert f(r, 'none').tolist() == [1.] * 7
    assert f(r, 'tis').tolist() == [.1, .5, 1., 2., 2., 2., 2.]
    assert f(r, 'mis').tolist() == [0., .5, 1., 2., 3., 5., 0.]
    logits, logq, labels, ids, values = setup(k=8)
    ids = ids.clone(); values = values.clone()
    ids[:, -2:] = -1; values[:, -2:] = float('-inf')  # padding / sampler-masked entries carry no weight
    behavior = logq.gather(1, labels[:, None]).squeeze(1)
    loss, details = score_centered_token_loss(torch.log_softmax(logits, -1), labels, behavior, ids, values, 2.)
    surrogate, _ = score_centered_surrogate(torch.log_softmax(logits, -1), labels, behavior, ids, values)
    assert torch.isfinite(loss).all() and torch.allclose(loss, -2 * surrogate)
    trimmed = gradient(lambda lp: score_centered_surrogate(lp, labels, behavior, ids[:, :6], values[:, :6])[0], logits)
    padded = gradient(lambda lp: score_centered_surrogate(lp, labels, behavior, ids, values)[0], logits)
    torch.testing.assert_close(padded, trimmed)
    # A banned token that the sampler also masked: no NaN from 0 * log 0.
    banned = torch.log_softmax(logits.index_fill(1, torch.tensor([ids[0, 0].item()]), float('-inf')), -1)
    sampler = torch.log_softmax(logq.index_fill(1, torch.tensor([ids[0, 0].item()]), float('-inf')), -1)
    values2, ids2 = torch.topk(sampler, 40)
    surrogate, _ = score_centered_surrogate(banned, labels.where(labels != ids[0, 0], ids[0, 1]),
                                            behavior, ids2, values2)
    assert torch.isfinite(surrogate).all()


def test_policy_loss_declaration_is_canonical_and_strict():
    assert policy_loss_declaration(None) is None
    assert policy_loss_declaration({'schema': 'score-centered/1'}) == {
        'schema': 'score-centered/1', 'k': 128, 'importance': 'none', 'eps': 1e-6}
    assert policy_loss_declaration('{"schema":"score-centered/1","k":32,"importance":"tis"}')['k'] == 32
    for bad in ({'schema': 'grpo'}, {'schema': 'score-centered/1', 'k': 0},
                {'schema': 'score-centered/1', 'importance': 'is'}, {'schema': 'score-centered/1', 'clip': 1}):
        with pytest.raises(ValueError):
            policy_loss_declaration(bad)
    from natlang_neuralese.train.recipe import validate_stage_parameters
    with pytest.raises(ValueError):
        validate_stage_parameters('native_state_rl', {'rl_policy_loss': {'schema': 'score-centered/1', 'k': -1}})
    validate_stage_parameters('native_state_rl', {'rl_policy_loss': {'schema': 'score-centered/1', 'k': 64}})


def test_top_k_receipt_roundtrip_is_compact_and_validated():
    log_probs = torch.log_softmax(torch.randn(100), -1)
    receipt = encode_top_k(log_probs, 32)
    ids, logps = decode_top_k(receipt)
    expected, order = torch.topk(log_probs, 32)
    assert ids.tolist() == order.tolist()
    torch.testing.assert_close(logps, expected.half().float())
    assert len(receipt['ids']) + len(receipt['logps']) == 4 * ((32 * 4 + 2) // 3) + 4 * ((32 * 2 + 2) // 3)
    with pytest.raises(ValueError):
        decode_top_k({'ids': receipt['ids'], 'logps': encode_top_k(log_probs, 16)['logps']})


def test_production_policy_loss_binding_requires_its_learner_argument():
    from natlang_neuralese.rl.production import bind_policy_loss
    loss = policy_loss_declaration({'schema': 'score-centered/1', 'k': 32})
    values = {}
    assert bind_policy_loss({'learner_command': ['x']}, values) is None and values == {}
    with pytest.raises(ValueError, match='--rl-policy-loss'):
        bind_policy_loss({'policy_loss': loss, 'learner_command': ['x']}, {})
    with pytest.raises(ValueError, match='canonical'):
        bind_policy_loss({'policy_loss': {'schema': 'score-centered/1'},
                          'learner_command': ['--rl-policy-loss', '{rl_policy_loss}']}, {})
    assert bind_policy_loss({'policy_loss': loss, 'learner_command': ['--rl-policy-loss', '{rl_policy_loss}']},
                            values) == loss
    assert json.loads(values['rl_policy_loss']) == loss


# ---- end to end: real engine capture under a mismatched sampler, real trainer objective ----

def quantized_sampler(engine, monkeypatch, *, seed=5):
    """The sampler draws from deliberately perturbed logits (coarse rounding plus noise), modelling a quantized
    actor; the trainer later scores the exact unperturbed model."""
    real = engine._native_token
    generator = torch.Generator().manual_seed(seed)
    def token(seq):
        noise = torch.randn(seq.logits.shape, generator=generator).to(seq.logits.device, seq.logits.dtype)
        seq.logits = (seq.logits * 4).round() / 4 + .3 * noise
        return real(seq)
    monkeypatch.setattr(engine, '_native_token', token)


def collect(engine, monkeypatch, *, top_k):
    from natlang_neuralese.serve.engine import GenerationRequest
    quantized_sampler(engine, monkeypatch)
    messages = [{'role': 'user', 'content': 't4 t5 t6'}]
    episodes = []
    engine.start()
    try:
        for sample in range(2):
            reply = engine.generate(GenerationRequest(messages=messages, temperature=.9, seed=61 + sample,
                                                      max_tokens=6, rl_capture=True, rl_top_k=top_k))
            capture = reply['x_natlang_rl_capture']
            episodes.append({'task_id': 'score-centered-train', 'split': 'train', 'sample': sample, 'group': 'g',
                'reward': {'quality': 1 - sample}, 'policy': {'checkpoint_sha256': engine.checkpoint_sha256},
                'turns': [{'messages': messages, 'target': reply['choices'][0]['message'], 'invocation_id': 'actor',
                           'sampling': capture, 'replay': {'raw_calls_retained': True}}]})
    finally:
        engine.stop()
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    return episodes


def admit(engine, tmp_path, episodes, policy_loss, name):
    from natlang_neuralese.rl.native_stage import prepare
    from natlang_neuralese.rl.native_replay import NativePolicyObjective
    from natlang_neuralese.data.native_graph import parse_graph
    engine.tokenizer.save_pretrained(tmp_path / 'rl-tokenizer')
    rows, spec = prepare(episodes, checkpoint=engine.native_checkpoint, tokenizer=tmp_path / 'rl-tokenizer',
                         policy_loss=policy_loss)
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(spec))
    objective = NativePolicyObjective(path)
    objective.bind_actor(engine)
    examples = [parse_graph(row, tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint']) for row in rows]
    return spec, objective, examples


def parameter_gradients(model, loss):
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().double()
                      for g, p in zip(grads, parameters)])


@pytest.mark.parametrize('mode', ['none', 'tis'])
def test_engine_capture_to_trainer_matches_reference_under_sampler_mismatch(tmp_path, monkeypatch, mode):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    engine = engine_fixture(tmp_path, 'lfm')
    k = 8
    episodes = collect(engine, monkeypatch, top_k=k)
    capture = episodes[0]['turns'][0]['sampling']
    assert capture['behavior_top_k'] == {'schema': 'natlang.behavior-top-k/1', 'k': k,
        'distribution': 'post-processing-normalized', 'ids': 'int32-le-base64', 'logps': 'float16-le-base64'}
    sampled = [sample for sample in capture['samples'] if not sample.get('forced')]
    assert sampled and all('top_k' in sample for sample in sampled)
    for sample in sampled:
        ids, logps = decode_top_k(sample['top_k'])
        assert len(ids) == k
        if sample['token_id'] in ids.tolist():
            assert abs(float(logps[ids.tolist().index(sample['token_id'])]) - sample['logp']) < 1e-2
    loss_declaration = {'schema': 'score-centered/1', 'k': k, 'importance': mode}
    spec, objective, examples = admit(engine, tmp_path, episodes, loss_declaration, 'centered')
    assert spec['policy_loss'] == policy_loss_declaration(loss_declaration)
    assert objective.binding()['policy_loss'] == spec['policy_loss']
    model = engine.shared_model
    for parameter in model.parameters():  # the serving engine holds frozen weights; the learner trains them
        parameter.requires_grad_(True)
    total, mismatch = 0, []
    for example in examples:
        loss, details = packed_graph_loss(model, [example], objective=objective)
        assert details['policy_loss'] == 'score-centered/1' and torch.isfinite(loss)
        centering = details['score_centering']
        assert centering['positions'] == details['sampled_tokens']
        actual = parameter_gradients(model, loss)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0

        class Reference(type(objective)):
            def __call__(self, model, result, context):
                row = self.rows[context['examples'][0].id]
                hidden = result['h_final'][0]
                actions = row['actions']
                queries = torch.tensor([a['query'] for a in actions])
                temperature = actions[0]['temperature']
                log_probs = torch.log_softmax(model.logits(hidden[queries]).double() / temperature, -1)
                labels = torch.tensor([a['token_id'] for a in actions])
                behavior = torch.tensor([a['behavior_logp'] for a in actions])
                from natlang_neuralese.rl.native_replay import behavior_top_k_tensors
                ids, logq = behavior_top_k_tensors(actions, k, 'cpu')
                mismatch.append(float((log_probs.detach().gather(1, labels[:, None]).squeeze(1) - behavior).abs().max()))
                value = literal_surrogate(log_probs, labels, behavior, ids, logq, mode, 1e-6)
                return (-row['advantage'] * value).sum() / len(actions), {}
        reference = Reference(objective.path)
        reference.bind_actor(engine)
        expected_loss, _ = packed_graph_loss(model, [example], objective=reference)
        expected = parameter_gradients(model, expected_loss)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-6)
        total += details['sampled_tokens']
    assert total == sum(len(row['actions']) for row in spec['rows'].values())
    assert max(mismatch) > 1e-2, 'the fixture really trains against a mismatched sampler'


def test_score_centered_requires_top_k_and_grpo_without_it_stays_unchanged(tmp_path, monkeypatch):
    from test_native_serving_adapters import engine_fixture
    engine = engine_fixture(tmp_path, 'lfm')
    episodes = collect(engine, monkeypatch, top_k=None)
    capture = episodes[0]['turns'][0]['sampling']
    assert 'behavior_top_k' not in capture and all('top_k' not in sample for sample in capture['samples'])
    with pytest.raises(ValueError, match='x_natlang_rl_top_k'):
        admit(engine, tmp_path, copy.deepcopy(episodes), {'schema': 'score-centered/1', 'k': 8}, 'missing')
    spec, objective, examples = admit(engine, tmp_path, episodes, None, 'grpo')
    assert 'policy_loss' not in spec and 'policy_loss' not in objective.binding()
    assert all('behavior_top_k' not in action for row in spec['rows'].values() for action in row['actions'])
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    loss, details = packed_graph_loss(engine.shared_model, examples, objective=objective)
    assert 'score_centering' not in details and torch.isfinite(loss)


def test_loss_k_larger_than_captured_k_is_refused(tmp_path, monkeypatch):
    from test_native_serving_adapters import engine_fixture
    engine = engine_fixture(tmp_path, 'lfm')
    episodes = collect(engine, monkeypatch, top_k=4)
    with pytest.raises(ValueError, match='k >='):
        admit(engine, tmp_path, episodes, {'schema': 'score-centered/1', 'k': 8}, 'short')
    tampered = copy.deepcopy(episodes)
    sample = next(s for s in tampered[0]['turns'][0]['sampling']['samples'] if not s.get('forced'))
    del sample['top_k']
    with pytest.raises(ValueError, match='every sampled position'):
        admit(engine, tmp_path, tampered, {'schema': 'score-centered/1', 'k': 4}, 'tampered')


def test_whole_producer_consumer_graph_scores_every_sampled_turn_with_top_k(tmp_path):
    """A producer/consumer episode: every sampled action of both turns carries top-k and enters the loss,
    with gradients through the real producer -> consumer KV dependency."""
    from types import SimpleNamespace
    from test_native_rl_replay import actor_rollout
    from test_shared_state import tiny
    from natlang_neuralese.common.sampling import behavior_top_k_declaration
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.rl.native_replay import prepare_objective, NativePolicyObjective, compile_replay
    from natlang_neuralese.data.native_graph import parse_graph
    torch.manual_seed(901)
    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)
    roles = SimpleNamespace(produce_open=58, produce_close=59, consume_open=60, consume_close=61)
    identity = {'checkpoint_sha256': '1'*64, 'tokenizer_fingerprint': '2'*64,
                'embedding_weight_sha256': '3'*64, 'architecture': model.port.config.model_type}
    seen = []
    original = model.logits
    def recording(hidden):
        value = original(hidden)
        seen.append(value.detach().float().div(.8).log_softmax(-1)[0])
        return value
    model.logits = recording  # the actor fixture scores each sample exactly once, in sampling order
    _, _, _, _, block, turns = actor_rollout('lfm', loaded=(model, roles, identity))
    del model.logits
    k = 8
    by_turn = {turn['graph_turn']: turn for turn in turns}
    ordered = [sample for name in ('producer', 'consumer') for sample in by_turn[name]['sampling']['samples']]
    assert len(seen) == len(ordered) == 7
    for sample, log_probs in zip(ordered, seen):
        sample['top_k'] = encode_top_k(log_probs, k)
    for turn in turns:
        turn['sampling']['behavior_top_k'] = behavior_top_k_declaration(k)
    replay = compile_replay(turns, identity=identity, roles=roles, fetch_block=lambda key: block)
    assert {action['turn'] for action in replay.actions} == {'producer', 'consumer'}
    assert all(action['behavior_top_k']['k'] == k for action in replay.actions)
    raw_turns = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'],
                  'sampling': turn['sampling'], 'replay': {'raw_calls_retained': True}} for turn in turns]
    episodes = [{'task_id': 'train-task', 'sample': sample, 'split': 'train', 'group': 'g',
                 'reward': {'quality': reward}, 'policy': {'checkpoint_sha256': identity['checkpoint_sha256']},
                 'turns': copy.deepcopy(raw_turns)} for sample, reward in enumerate((1, 0))]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    declared = {'schema': 'score-centered/1', 'k': k, 'importance': 'mis'}
    graphs, spec = prepare_objective(episodes, identity=identity, roles=roles, fetch_block=lambda key: block,
                                     policy_loss=declared)
    path = tmp_path / 'objective.json'
    path.write_text(json.dumps(spec))
    objective = NativePolicyObjective(path)
    example = parse_graph(graphs[0], tokenizer_fingerprint=identity['tokenizer_fingerprint'])
    context = {'examples': [example], 'row_offsets': {example.id: {'physical_start': 0, 'external_names': {}}}}
    result = model(example.plan, retain_states=False)
    loss, detail = objective(model, result, context)
    assert detail['sampled_tokens'] == 7 and detail['score_centering']['positions'] == 7
    assert abs(detail['score_centering']['w_y_mean'] - 1) < 1e-3  # exact sampler: p = q
    turns_scored = {action['turn'].split('#')[0] for action in objective.rows[example.id]['actions']}
    assert turns_scored == {'producer', 'consumer'}  # one graph forward scores both turns' sampled actions
    loss.backward()
    grad = model.adapter.up.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    path.write_text(json.dumps(dict(spec, mode='rft')))
    with pytest.raises(ValueError, match='inapplicable policy loss'):
        NativePolicyObjective(path)
