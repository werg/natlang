"""natlang ``decide`` readouts as trainable own outputs: graph scores/gradients equal explicit cached execution."""
import copy

import pytest
import torch

from test_shared_state import tiny
from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
from natlang_neuralese.model.shared_state import SharedStateModel, native_step, splice_block, \
    capture_invocation_span, compile_graph, Invocation
from natlang_neuralese.train.decision_supervision import decision_terms
from natlang_neuralese.train.native_state_sft import packed_graph_loss

FINGERPRINT = '1' * 64
PRODUCER = [2, 3, 4, 5, 6]
PROMPT = [7, 8, 9, 10]
OPTIONS = [[11, 12], [13], [14, 15, 16]]


def model_for(family):
    port = tiny(family)
    vocab = port.embedding_weight.shape[0]
    model = SharedStateModel(port, tuple(range(vocab - 4, vocab)), rank=4)
    torch.nn.init.normal_(model.adapter.up.weight, std=0.03)
    return model


def decision_row(*, label=2, stop_gradient=False, extra_nodes=(), producer=True):
    edge = {'at': 2, 'source': 'producer', 'start': 2, 'stop': 4, **({'stop_gradient': True} if stop_gradient else {})}
    nodes = [{'name': 'producer', 'token_ids': PRODUCER, 'producing': [[2, 4]], 'target_weights': [0.] * 5}] \
        if producer else []
    for index, option in enumerate(OPTIONS):
        nodes.append({'name': f'd:option:{index}', 'token_ids': PROMPT + option,
                      'target_weights': [0.] * (len(PROMPT) + len(option)), **({'imports': [edge]} if producer else {})})
    return {'schema': SCHEMA, 'id': 'decision-row', 'split': 'train', 'source_groups': ['decision-fixture'],
            'tokenizer_fingerprint': FINGERPRINT, 'nodes': nodes + list(extra_nodes),
            'decisions': [{'name': 'd', 'options': [f'd:option:{i}' for i in range(len(OPTIONS))],
                           'prompt_tokens': len(PROMPT), 'label': label, 'label_source': 'exact-oracle'}]}


def cached_scores(model, *, stop_gradient=False):
    """Explicit cached execution: run the producer, splice its state into the consumer cache, score each option
    token by token from the shared prompt cache (the serving `_score_from_cache` law)."""
    from types import SimpleNamespace
    mask = torch.tensor([[False, False, True, True, False]])
    producer = native_step(model, torch.tensor([PRODUCER]), producing=mask)
    structure = model(compile_graph([Invocation('producer', tuple(PRODUCER), producing=((2, 4),))]))
    captured = dict(capture_invocation_span(structure, 'producer', 2, 4))
    captured['layers'] = tuple({key: value[2:4] if key != 'kind' else value for key, value in layer.items()}
                               for layer in producer['layers'])
    captured['embeddings'] = producer['embeddings'][2:4]
    identity = dict(checkpoint_sha256='a' * 64, tokenizer_fingerprint='b' * 64, embedding_weight_sha256='c' * 64)
    block = SimpleNamespace(**captured, **identity, architecture=str(model.port.config.model_type),
                            validate=lambda: None)
    prefix = native_step(model, torch.tensor([PROMPT[:2]]))
    cache = splice_block(model, prefix['cache'], block, stop_gradient=stop_gradient, **identity)
    rest = native_step(model, torch.tensor([PROMPT[2:]]), cache)
    scores = []
    for option in OPTIONS:
        state, logits, score = rest['cache'], model.logits(rest['h_final'][:, -1]), 0.
        for token in option:
            score = score + torch.log_softmax(logits.float(), -1)[0, token]
            step = native_step(model, torch.tensor([[token]]), state)
            state, logits = step['cache'], model.logits(step['h_final'][:, -1])
        scores.append(score)
    return torch.stack(scores)


def graph_loss(model, row):
    example = parse_graph(row, tokenizer_fingerprint=FINGERPRINT)
    hidden = model(example.plan, retain_states=False)['h_final'][0]
    return decision_terms(model, hidden, example)


@pytest.mark.parametrize('family', ['lfm', 'mellum', 'lfm3'])
def test_decision_scores_and_gradients_equal_explicit_cached_execution(family):
    model = model_for(family)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    label = 2
    cached = cached_scores(model)
    expected_loss = -torch.log_softmax(cached, 0)[label]
    from natlang_neuralese.train.decision_supervision import option_scores, physical_options
    example = parse_graph(decision_row(label=label), tokenizer_fingerprint=FINGERPRINT)
    hidden = model(example.plan, retain_states=False)['h_final'][0]
    scores = option_scores(model, hidden, physical_options(example, example.decisions[0]))
    torch.testing.assert_close(scores, cached, atol=3e-5, rtol=3e-5)
    loss, count, correct = decision_terms(model, hidden, example)
    assert count == 1 and float(correct) == float(int(cached.argmax()) == label)
    torch.testing.assert_close(loss, expected_loss, atol=3e-5, rtol=3e-5)
    graph = torch.autograd.grad(loss, parameters, allow_unused=True)
    reference = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    for actual, wanted in zip(graph, reference):
        assert (actual is None) == (wanted is None)
        if actual is not None:
            torch.testing.assert_close(actual, wanted, atol=3e-5, rtol=3e-4)
    # The producer's exported state receives the decision's gradient: stopping it at the import changes the update
    # exactly as stopping it in the cached execution does.
    stopped, _, _ = graph_loss(model, decision_row(label=label, stop_gradient=True))
    stopped_reference = -torch.log_softmax(cached_scores(model, stop_gradient=True), 0)[label]
    stopped_graph = torch.autograd.grad(stopped, parameters, allow_unused=True)
    stopped_cached = torch.autograd.grad(stopped_reference, parameters, allow_unused=True)
    difference = 0.
    for full, cut, cut_reference in zip(graph, stopped_graph, stopped_cached):
        if cut is not None:
            torch.testing.assert_close(cut, cut_reference, atol=3e-5, rtol=3e-4)
        if full is not None:
            difference += float((full - (cut if cut is not None else 0)).norm())
    assert difference > 0
    # The trainer's packed loss is that same decision CE (one invocation of mass 1).
    total, details = packed_graph_loss(model, [example], checkpoint_layers=False)
    torch.testing.assert_close(total, expected_loss, atol=3e-5, rtol=3e-5)
    assert details[0]['decisions']['count'] == 1


def test_decision_counts_as_one_invocation_beside_token_targets_and_under_prefix_sharing():
    model = model_for('lfm')
    answer = {'name': 'answer', 'token_ids': [4, 5, 6, 7, 8], 'target_weights': [0., 0., 1., 1., 1.]}
    mixed = parse_graph(decision_row(extra_nodes=[answer]), tokenizer_fingerprint=FINGERPRINT)
    tokens_only = parse_graph({**decision_row(extra_nodes=[answer]), 'nodes': [answer], 'decisions': None,
                               'id': 'tokens-only'}, tokenizer_fingerprint=FINGERPRINT)
    decision_only = parse_graph(decision_row(), tokenizer_fingerprint=FINGERPRINT)
    with torch.no_grad():
        token_loss, _ = packed_graph_loss(model, [tokens_only], checkpoint_layers=False)
        decision_loss, _ = packed_graph_loss(model, [decision_only], checkpoint_layers=False)
        combined, details = packed_graph_loss(model, [mixed], checkpoint_layers=False)
        torch.testing.assert_close(combined, (token_loss + decision_loss) / 2, atol=2e-6, rtol=2e-6)
        assert details[0]['decisions']['count'] == 1 and details[0]['supervised_tokens'] == 3
        batch, _ = packed_graph_loss(model, [mixed, decision_only], checkpoint_layers=False)
        torch.testing.assert_close(batch, (combined + decision_loss) / 2, atol=2e-6, rtol=2e-6)
        # Shared causal prefixes fold the identical option prompts and reuse readouts; the loss law is unchanged.
        shared, _ = packed_graph_loss(model, [mixed, decision_only], checkpoint_layers=False,
                                      share_causal_prefixes=True, collect_accuracy=False)
        torch.testing.assert_close(shared, batch, atol=2e-6, rtol=2e-6)
        alone, _ = packed_graph_loss(model, [decision_only], checkpoint_layers=False,
                                     share_causal_prefixes=True, collect_accuracy=False)
        torch.testing.assert_close(alone, decision_loss, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('change, message', [
    (lambda row: row['decisions'][0].update(label=3), 'label must index'),
    (lambda row: row['decisions'][0].update(options=['d:option:0']), 'at least two'),
    (lambda row: row['nodes'][2]['target_weights'].__setitem__(4, 1.), 'not token CE targets'),
    (lambda row: row['nodes'][2]['token_ids'].__setitem__(0, 9), 'share one exact prompt'),
    (lambda row: row['nodes'][2].pop('imports'), 'share one exact prompt'),
    (lambda row: row['decisions'][0].update(label_source='teacher-guess'), 'label_source'),
    (lambda row: row['decisions'][0].update(probabilities=[1, 0, 0]), 'exactly one of label or probabilities'),
])
def test_invalid_decisions_are_rejected(change, message):
    row = copy.deepcopy(decision_row())
    change(row)
    with pytest.raises(ValueError, match=message):
        parse_graph(row, tokenizer_fingerprint=FINGERPRINT)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_converted_rows_and_served_decisions_share_rendering_scores_and_imported_state(tmp_path, family):
    """CPU conversion renders the served prompt/continuations; a served decision with an imported native state
    scores exactly as its graph row with the archived state as an external input."""
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.data.decision_graph import decision_graph_row, decision_capture_row, tokenizer_renderer
    from natlang_neuralese.serve.native_learning import decide_native, _decision_render
    engine = engine_fixture(tmp_path, family, service_template=True)
    try:
        fingerprint = engine.model_config['tokenizer_fingerprint']
        renderer = tokenizer_renderer(engine.tokenizer, model_type=engine.backbone.config.model_type,
                                      history_reasoning_policy=engine.model_config.get('history_reasoning_policy'))
        messages, options = [{'role': 'user', 'content': 't4 t5'}], ['t7', 't8 t9']
        row = decision_graph_row(renderer, {'id': 'converted', 'split': 'train', 'source_groups': ['g'],
            'messages': messages, 'options': options, 'label': 1, 'label_source': 'exact-oracle'},
            tokenizer_fingerprint=fingerprint)
        plan, continuations = _decision_render(engine, messages, None, options)
        assert [node['token_ids'][row['decisions'][0]['prompt_tokens']:] for node in row['nodes']] == continuations
        assert row['nodes'][0]['token_ids'][:row['decisions'][0]['prompt_tokens']] == plan[0][1]
        state = engine.native_write_text('t5 t6')['id']
        native_messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 't4 '},
                            {'type': 'native_state', 'id': state}, {'type': 'text', 'text': ' t7'}]}]
        served = decide_native(engine, {'messages': native_messages, 'options': options, 'native_recording': True})
        capture = served['native_decision_capture']
        graph = decision_capture_row(capture, id='served', split='train', source_groups=['g'], label=0,
                                     label_source='verified-trajectory', tokenizer_fingerprint=fingerprint,
                                     fetch_block=lambda ident: engine.store.get_native_state(ident).block)
        example = parse_graph(graph, tokenizer_fingerprint=fingerprint)
        blocks = {name: engine.store.get_native_state(binding['id']).block
                  for name, binding in graph['external_states'].items()}
        identity = {name: {key: getattr(block, key) for key in ('checkpoint_sha256', 'tokenizer_fingerprint',
                    'embedding_weight_sha256', 'architecture')} for name, block in blocks.items()}
        from natlang_neuralese.train.decision_supervision import option_scores, physical_options
        with torch.no_grad():
            result = engine.shared_model(example.plan, external_blocks=blocks, external_identity=identity,
                                         retain_states=False)
            scores = option_scores(engine.shared_model, result['h_final'][0],
                                   physical_options(example, example.decisions[0]))
        torch.testing.assert_close(scores.float().cpu(), torch.tensor(served['log_probs']), atol=3e-4, rtol=3e-4)
    finally:
        engine.store.close()


@pytest.mark.parametrize('family', ['lfm', 'mellum', 'lfm3'])
@pytest.mark.parametrize('mode', ['score-centered', 'clipped', 'rft'])
def test_sampled_decision_policy_gradient_through_imported_producer_equals_cached_execution(family, mode):
    """RL over a sampled decision: the option-set policy loss through the one graph forward equals the explicit
    cached execution's, including the gradient into the producer whose state the prompt imports."""
    from natlang_neuralese.rl.decision_policy import behavior_log_probs, decision_policy_terms
    from natlang_neuralese.train.decision_supervision import physical_options
    from natlang_neuralese.rl.policy_math import clipped_token_loss
    model = model_for(family)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    example = parse_graph(decision_row(), tokenizer_fingerprint=FINGERPRINT)
    served = [-1.3, -0.4, -2.2]  # a mismatched (e.g. quantized) actor's scores: behaviour differs from the learner
    temperature, chosen, advantage = 1.5, 0, 0.7
    action = {'kind': 'decision', 'chosen': chosen, 'temperature': temperature,
              'behavior_logps': behavior_log_probs(served, temperature),
              'options': physical_options(example, example.decisions[0])}
    policy_loss = {'schema': 'score-centered/1', 'k': 8, 'importance': 'none', 'eps': 1e-6} \
        if mode == 'score-centered' else None
    hidden = model(example.plan, retain_states=False)['h_final'][0]
    terms, _ = decision_policy_terms(model, hidden, [action], advantage=advantage,
                                     mode='rft' if mode == 'rft' else 'grpo', policy_loss=policy_loss)
    log_pi = torch.log_softmax(cached_scores(model) / temperature, 0)
    behavior = torch.tensor(action['behavior_logps'])
    if mode == 'rft':
        reference = -log_pi[chosen]
    elif mode == 'clipped':
        reference = clipped_token_loss(log_pi[chosen][None], behavior[chosen][None], advantage)[0]
    else:  # full option support: U = log pi_y - sum_v q_v log pi_v (no tail)
        reference = -advantage * (log_pi[chosen] - (behavior.exp() * log_pi).sum())
    torch.testing.assert_close(terms[0], reference.float(), atol=3e-5, rtol=3e-4)
    actual = torch.autograd.grad(terms.sum(), parameters, allow_unused=True)
    wanted = torch.autograd.grad(reference, parameters, allow_unused=True)
    for a, b in zip(actual, wanted):
        if a is None or b is None:
            assert (a is None or not a.abs().sum()) and (b is None or not b.abs().sum())
            continue
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    assert sum(float(g.abs().sum()) for g in actual if g is not None) > 0


def test_sampled_decision_turns_enter_the_whole_episode_policy_objective(tmp_path, monkeypatch):
    """Engine decision capture + sampling receipt -> compile_replay option nodes -> admitted objective -> one packed
    graph loss whose decision term equals its reference; an argmax readout stays context."""
    import json
    from test_native_serving_adapters import engine_fixture
    from test_rl_score_centering import admit, parameter_gradients
    from test_rl_episode_graph import attach_execution_graph
    from natlang_neuralese.serve.native_learning import decide_native
    from natlang_neuralese.rl.decision_policy import sample_receipt
    from natlang_neuralese.serve.chat import RequestError
    from natlang_neuralese.serve.engine import GenerationRequest
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    decision_messages, options = [{'role': 'user', 'content': 't4 t7'}], ['t7', 't8 t9']
    served = decide_native(engine, {'messages': decision_messages, 'options': options, 'native_recording': True})
    messages, episodes, seed = [{'role': 'user', 'content': 't4 t5 t6'}], [], 61
    engine.start()
    try:
        while len(episodes) < 2:  # actual sampled generation turns; seeds ending inside a produce span are skipped
            seed += 1
            try:
                reply = engine.generate(GenerationRequest(messages=messages, temperature=.9, seed=seed,
                                                          max_tokens=8, rl_capture=True, rl_top_k=8))
            except RequestError:
                continue
            sample = len(episodes)
            decision = {**served['native_decision_capture'],
                        'decision_sample': sample_receipt(served['log_probs'], chosen=sample, temperature=1.)}
            episodes.append({'task_id': 'decision-train', 'split': 'train', 'sample': sample, 'group': 'g',
                'reward': {'quality': 1 - sample}, 'policy': {'checkpoint_sha256': engine.checkpoint_sha256},
                'turns': [{'messages': messages, 'target': reply['choices'][0]['message'], 'invocation_id': 'actor',
                           'sampling': reply['x_natlang_rl_capture'], 'replay': {'raw_calls_retained': True}},
                          {'messages': decision_messages, 'target': {'role': 'assistant', 'content': options[sample]},
                           'invocation_id': 'decider', 'sampling': decision,
                           'replay': {'raw_calls_retained': True}}]})
    finally:
        engine.stop()
    attach_execution_graph(episodes)
    argmax = json.loads(json.dumps(episodes))
    for episode in argmax:
        del episode['turns'][1]['sampling']['decision_sample']
    spec, objective, examples = admit(engine, tmp_path, episodes, {'schema': 'score-centered/1', 'k': 8}, 'decide')
    rows = list(spec['rows'].values())
    assert all(len(row['decision_actions']) == 1 and 'decider#turn1:option:1' in row['graph_nodes'] for row in rows)
    model = engine.shared_model
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for example in examples:
        loss, details = packed_graph_loss(model, [example], objective=objective)
        assert details['sampled_decisions'] == 1 and details['sampled_tokens'] > 0
        row = objective.rows[example.id]

        class WithoutDecision(type(objective)):
            def __call__(self, model, result, context):
                from natlang_neuralese.rl.decision_policy import decision_policy_terms
                saved = self.rows[example.id].pop('decision_actions')
                try:
                    token_loss, token_details = super().__call__(model, result, context)
                finally:
                    self.rows[example.id]['decision_actions'] = saved
                decided, _ = decision_policy_terms(model, result['h_final'][0], saved, advantage=row['advantage'],
                    mode='grpo', policy_loss=self.policy_loss,
                    offset=context['row_offsets'][example.id]['physical_start'])
                count = token_details['sampled_tokens']
                return (token_loss * count + decided.sum()) / (count + 1), {}
        reference = WithoutDecision(objective.path)
        reference.bind_actor(engine)
        expected, _ = packed_graph_loss(model, [example], objective=reference)
        torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(parameter_gradients(model, loss), parameter_gradients(model, expected),
                                   atol=1e-7, rtol=1e-4)
    spec, _, _ = admit(engine, tmp_path, argmax, {'schema': 'score-centered/1', 'k': 8}, 'argmax')
    assert all('decision_actions' not in row for row in spec['rows'].values())
    engine.store.close()
