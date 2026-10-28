"""JOLT+ in the whole-graph RL objective: in-run guidance twins, per-term gradients, custody fallback.

Guidance, custody and teacher scoring are the shared mechanism (rl/privileged_context.py, rl/privileged_scoring.py,
tests in test_jolt_privileged_context.py); these tests bind it to the objective. Student episodes carry the
policy's write-only guidance (its OPSD twin imports it); teacher attempts import it, and its writer is trained
through that import by the teacher outcome term.
"""
import copy
import json

import pytest
import torch

from natlang_neuralese.common.sampling import behavior_top_k_declaration, encode_top_k
from natlang_neuralese.rl.jolt import DEFAULT_JOLT, jolt_declaration, privileged_source, root_calls
from natlang_neuralese.rl.policy_math import default_policy_loss, policy_loss_declaration
from natlang_neuralese.common.custody import ProtectedWorlds
from test_jolt_privileged_context import (CONSUMERS, TEMPERATURE, WRITERS, custody, insertion_blocks, load,
                                          protected_case, rendering, run_episode, train_case)

K = 8
DECLARATION = {'schema': 'jolt-plus/1', 'beta_s': 0.7, 'beta_t': 0.3,
               'outcome': {'schema': 'score-centered/1', 'k': K, 'importance': 'tis'}}


def episode_with_top_k(model, roles, identity, mode='write-only', **options):
    """The shared fixture episode with the sampler's top-k recorded at every sampled position."""
    seen, original = [], model.logits
    def recording(hidden):
        value = original(hidden)
        seen.append(value.detach().float().div(TEMPERATURE).log_softmax(-1)[0])
        return value
    model.logits = recording
    try:
        episode = run_episode(model, roles, identity, mode, **options)
    finally:
        del model.logits
    by_name = {turn['graph_turn']: turn for turn in episode.turns}
    ordered = [sample for name in episode.order for sample in by_name[name]['sampling']['samples']]
    assert len(seen) == len(ordered)
    for sample, log_probs in zip(ordered, seen):
        sample['top_k'] = encode_top_k(log_probs, K)
    for turn in episode.turns:
        turn['sampling']['behavior_top_k'] = behavior_top_k_declaration(K)
    return episode


def rollouts(turns, *, role='student', rewards=(1, 0)):
    raw = [{'messages': turn['messages'], 'target': {}, 'invocation_id': turn['graph_turn'].rsplit('#turn', 1)[0],
            'sampling': turn['sampling'], 'replay': {'raw_calls_retained': True}} for turn in turns]
    rows = [{'task_id': train_case()['id'], 'sample': sample, 'split': 'train', 'group': 'g', 'jolt_role': role,
             'reward': {'quality': reward}, 'policy': {'checkpoint_sha256': '1'*64},
             'turns': copy.deepcopy(raw)} for sample, reward in enumerate(rewards)]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(rows)
    return rows


def source_for(rows, episode, protected=None):
    return privileged_source(rows, cases={train_case()['id']: train_case()}, rendering=rendering(),
                             custody=protected or custody(), fetch_text=lambda key: episode.states[key].text)


def prepare(rows, episode, roles, identity, *, declaration=DECLARATION, protected=None):
    from natlang_neuralese.rl.native_replay import prepare_objective
    return prepare_objective(rows, identity=identity, roles=roles, fetch_block=episode.states.__getitem__,
                             policy_loss=declaration, privileged=source_for(rows, episode, protected))


def objective_for(tmp_path, spec, name='objective'):
    from natlang_neuralese.rl.native_replay import NativePolicyObjective
    path = tmp_path / (name + '.json')
    path.write_text(json.dumps(spec))
    return NativePolicyObjective(path)


def forward(model, graph, identity):
    from natlang_neuralese.data.native_graph import parse_graph
    example = parse_graph(graph, tokenizer_fingerprint=identity['tokenizer_fingerprint'])
    context = {'examples': [example], 'row_offsets': {example.id: {'physical_start': 0, 'external_names': {}}}}
    return example, context, model(example.plan, retain_states=False)


def test_declarations_default_to_jolt_plus_and_keep_explicit_opt_outs():
    assert default_policy_loss() == jolt_declaration(DEFAULT_JOLT)
    assert default_policy_loss()['outcome']['schema'] == 'score-centered/1'
    assert policy_loss_declaration({'schema': 'clipped-ratio/1'}) == {'schema': 'clipped-ratio/1'}
    assert policy_loss_declaration({'schema': 'jolt-plus/1', 'beta_s': 2})['beta_s'] == 2.
    for bad in ({'schema': 'jolt-plus/1', 'beta_t': -1}, {'schema': 'jolt-plus/1', 'gamma': 1},
                {'schema': 'jolt-plus/1', 'outcome': {'schema': 'clipped-ratio/1'}},
                {'schema': 'clipped-ratio/1', 'clip_low': .1}):
        with pytest.raises(ValueError):
            policy_loss_declaration(bad)
    from natlang_neuralese.rl.production import bind_policy_loss
    with pytest.raises(ValueError, match='jolt-cases'):
        bind_policy_loss({'policy_loss': default_policy_loss(), 'learner_command': ['{rl_policy_loss}']}, {})



@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_packed_twin_rows_equal_shared_teacher_scores(family):
    """The twin appended to the packed graph scores exactly what privileged_scoring.teacher_scores scores."""
    from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph
    from natlang_neuralese.rl.jolt import add_privileged_twin
    from natlang_neuralese.rl.privileged_scoring import processed_logits, teacher_scores
    model, roles, identity = load(family)
    episode = episode_with_top_k(model, roles, identity)
    rows = rollouts(episode.turns)
    weights = [0.] * episode.replay.plan.token_count  # as prepare_objective: 1 at each sampled action's target
    for action in episode.replay.actions:
        weights[action['query'] + 1] = 1.
    graph = {'nodes': [{'name': node.name, 'target_weights': weights[offset:offset + len(node.token_ids)]}
                       for node, offset in zip(episode.replay.plan.invocations, episode.replay.plan.offsets)]}
    actions, record = add_privileged_twin(episode.replay, rows[0], graph, source_for(rows, episode),
                                          turns=episode.turns)
    assert record['privileged'] and set(record['twin_turns']) == {'root#turn1#privileged', 'child#turn1#privileged'}
    assert set(record['policy_turns']) == set(WRITERS.values()) and record['guidance']['writer_blocks'] == 2
    assert {action['turn'] for action in actions} == set(CONSUMERS)  # the writer carries no student loss
    weights = {node['name']: node['target_weights'] for node in graph['nodes'][:len(episode.replay.plan.invocations)]}
    assert all(not any(weights[name]) for name in WRITERS.values())
    twins = [Invocation(n['name'], tuple(n['token_ids']), tuple(ImportSpan(**edge) for edge in n['imports']),
                        tuple(map(tuple, n['producing'])), tuple(map(tuple, n.get('logical_producing', ()))))
             for n in graph['nodes'][len(episode.replay.plan.invocations):]]
    plan = compile_graph((*episode.replay.plan.invocations, *twins))
    with torch.no_grad():
        hidden = model(plan, retain_states=False)['h_final'][0]
        packed = torch.stack([processed_logits(model.logits(hidden[a['twin_query']]).float(), a).log_softmax(-1)
                              for a in actions])
        shared = teacher_scores(model, episode.replay, insertion_blocks(episode))
        rows_of = {(a['turn'], a['sample']): row for row, a in enumerate(episode.replay.actions)}
        expected = torch.stack([processed_logits(shared.logits[rows_of[(a['turn'], a['sample'])]], a).log_softmax(-1)
                                for a in actions])
    torch.testing.assert_close(packed, expected, rtol=1e-4, atol=1e-4)


def literal_reference(model, hidden, row, declaration):
    """Direct float64 transcription of loss_t = -A U_t + beta KL(p || sg p~) (no twin: KL 0)."""
    from natlang_neuralese.rl.native_replay import behavior_top_k_tensors
    from test_rl_score_centering import literal_surrogate
    actions = row['actions']
    role = row['jolt']['role']
    beta = declaration['beta_s'] if role == 'student' else declaration['beta_t']
    queries = torch.tensor([a['query'] for a in actions])
    twin_queries = torch.tensor([a.get('twin_query', a['query']) for a in actions])
    log_p = torch.log_softmax(model.logits(hidden[queries]).double() / TEMPERATURE, -1)
    log_q = torch.log_softmax(model.logits(hidden[twin_queries]).double() / TEMPERATURE, -1).detach()
    kl = (log_p.exp() * (log_p - log_q)).sum(1)
    labels = torch.tensor([a['token_id'] for a in actions])
    behavior = torch.tensor([a['behavior_logp'] for a in actions])
    ids, logq = behavior_top_k_tensors(actions, K, 'cpu')
    outcome = -row['advantage'] * literal_surrogate(log_p, labels, behavior, ids, logq,
                                                   declaration['outcome']['importance'], declaration['outcome']['eps'])
    return outcome.sum() / len(actions), beta * kl.sum() / len(actions)


def gradient_vector(model, loss):
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().double()
                      for g, p in zip(grads, parameters)])


@pytest.mark.parametrize('role', ['student', 'teacher'])
def test_each_jolt_term_matches_its_reference_gradient(tmp_path, role):
    model, roles, identity = load('lfm')
    episode = episode_with_top_k(model, roles, identity, 'write-only' if role == 'student' else 'import')
    rows = rollouts(episode.turns, role=role)
    graphs, spec = prepare(rows, episode, roles, identity)
    canonical = policy_loss_declaration(DECLARATION)
    assert spec['policy_loss'] == canonical and len(graphs) == 2
    objective = objective_for(tmp_path, spec)
    for graph in graphs:
        row = spec['rows'][graph['id']]
        assert row['jolt']['role'] == role and row['jolt']['privileged'] and abs(row['advantage']) == .5
        assert {n['name'] for n in graph['nodes']} == set(row['episode_turns']) | set(row['jolt']['twin_turns'])
        example, context, result = forward(model, graph, identity)
        loss, details = objective(model, result, context)
        # Students: the consumers' 8 tokens; teachers also train the two writers' 6 tokens each.
        assert details['policy_loss'] == 'jolt-plus/1' and details['sampled_tokens'] == (8 if role == 'student' else 20)
        parts = details['jolt']
        assert parts['privileged_episodes'] == 1 and parts['unprivileged_episodes'] == 0
        assert parts['guidance']['writer_blocks'] == 2 and parts['guidance']['text_fallbacks'] == 0
        assert parts['guidance']['verbatim_reference'] == 1  # the root hint copies a reference line
        outcome_ref, kl_ref = literal_reference(model, result['h_final'][0], row, canonical)
        term = 'student_opsd' if role == 'student' else 'teacher_kl'
        assert parts[role + '_outcome'] != 0  # the surrogate value is estimator-specific; gradients compared below
        assert abs(parts[term] - float(kl_ref)) < 1e-5 and parts[term] > 0
        torch.testing.assert_close(gradient_vector(model, loss), gradient_vector(model, outcome_ref + kl_ref),
                                   rtol=2e-3, atol=1e-7)
        hidden_grad = torch.autograd.grad(loss, result['h_final'], retain_graph=True)[0][0]
        twinned = [a for a in row['actions'] if 'twin_query' in a]
        assert hidden_grad[torch.tensor([a['twin_query'] for a in twinned])].abs().max() == 0
        assert hidden_grad[torch.tensor([a['query'] for a in row['actions']])].abs().max() > 0
        if role == 'teacher':
            # The teacher outcome reaches the writer's payload through the consumers' import.
            writer = [a for a in row['actions'] if a['turn'] in WRITERS.values()]
            assert len(writer) == 12 and all('twin_query' not in a for a in writer)


def test_teacher_twin_is_the_student_graph_and_zero_spread_groups_still_teach(tmp_path):
    model, roles, identity = load('lfm')
    student = episode_with_top_k(model, roles, identity, 'write-only')
    teacher = episode_with_top_k(model, roles, identity, 'import')
    student_graphs, student_spec = prepare(rollouts(student.turns, rewards=(1, 1)), student, roles, identity)
    teacher_graphs, _ = prepare(rollouts(teacher.turns, role='teacher'), teacher, roles, identity)
    strip = lambda name: name.rsplit('#', 1)[0] if name.endswith(('#privileged', '#unprivileged')) else name
    actual = lambda graph: {n['name']: n for n in graph['nodes'] if n['name'] == strip(n['name'])}
    twin = lambda graph: {strip(n['name']): n for n in graph['nodes'] if n['name'] != strip(n['name'])}
    for name in CONSUMERS:
        assert actual(student_graphs[0])[name]['token_ids'] == twin(teacher_graphs[0])[name]['token_ids']
        assert twin(student_graphs[0])[name]['token_ids'] == actual(teacher_graphs[0])[name]['token_ids']
        edges = lambda node: [(e['source'].split('#privileged')[0].split('#unprivileged')[0], e['at'], e['start'],
                               e['stop']) for e in node['imports']]
        assert edges(twin(student_graphs[0])[name]) == edges(actual(teacher_graphs[0])[name])
    assert set(twin(student_graphs[0])) == set(CONSUMERS)  # writers are imported by name, never twinned
    row = student_spec['rows'][student_graphs[0]['id']]
    assert row['advantage'] == 0
    objective = objective_for(tmp_path, student_spec)
    _, context, result = forward(model, student_graphs[0], identity)
    loss, details = objective(model, result, context)
    assert details['jolt']['student_outcome'] == 0 and details['jolt']['student_opsd'] > 0
    loss.backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0


def test_coefficient_actor_sources_cover_decision_options_and_twins():
    from natlang_neuralese.rl.jolt import actor_sources, PRIVILEGED_SUFFIX
    suffix = PRIVILEGED_SUFFIX['student']
    row = {'invocation_actors': {'invocations': {'decision': {}, 'writer': {}}},
           'decision_actions': [{'turn': 'decision', 'option_nodes': ['option0', 'option1']}],
           'jolt': {'role': 'student', 'twin_turns': ['option0' + suffix, 'option1' + suffix, 'writer' + suffix]}}
    assert actor_sources(row) == {'option0': 'decision', 'option1': 'decision', 'writer': 'writer',
        'option0' + suffix: 'decision', 'option1' + suffix: 'decision', 'writer' + suffix: 'writer'}


def test_captured_guidance_decision_twins_train_through_option_nodes(tmp_path):
    model, roles, identity = load('lfm')
    episode = episode_with_top_k(model, roles, identity, 'write-only')
    root = next(turn for turn in episode.turns if turn['graph_turn'] == 'root#turn1')
    old = root['sampling']
    root['sampling'] = {'schema': 'natlang.native-decision-capture/1', 'policy': old['policy'],
        'prompt': old['prompt'], 'privileged': old['privileged'], 'continuations': [[11, 2], [13, 2]],
        'decision_sample': {'schema': 'natlang.decision-sample/1', 'log_probs': [-1., -2.],
                            'chosen': 0, 'temperature': 1., 'seed': 3}}
    graphs, spec = prepare(rollouts(episode.turns), episode, roles, identity)
    row = spec['rows'][graphs[0]['id']]
    assert len(row['decision_actions'][0]['twin_options']) == 2
    assert row['jolt']['privileged'] is not None
    objective = objective_for(tmp_path, spec)
    _, context, result = forward(model, graphs[0], identity)
    loss, details = objective(model, result, context)
    assert details['sampled_decisions'] == 1 and torch.isfinite(loss)
    loss.backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0


def test_custody_refusal_keeps_the_episode_on_its_outcome_term(tmp_path):
    """A world related to a protected packet gets no privileged teacher, never drops out of RL."""
    model, roles, identity = load('lfm')
    episode = episode_with_top_k(model, roles, identity)
    rows = rollouts(episode.turns)
    related = protected_case()
    related['source_groups'] = list(train_case()['source_groups'])  # shares the training world's source group
    protected = ProtectedWorlds.from_cases({'dev': ('0'*64, [related])})
    graphs, spec = prepare(rows, episode, roles, identity, protected=protected)
    assert len(graphs) == 2
    for graph in graphs:
        row = spec['rows'][graph['id']]
        assert row['jolt']['privileged'] is None and 'protected world' in row['jolt']['refusal']
        assert {n['name'] for n in graph['nodes']} == set(row['episode_turns'])  # no twin
        assert all('twin_query' not in action for action in row['actions'])
        assert {action['turn'] for action in row['actions']} == set(CONSUMERS)
    objective = objective_for(tmp_path, spec)
    _, context, result = forward(model, graphs[0], identity)
    loss, details = objective(model, result, context)
    assert details['jolt']['unprivileged_episodes'] == 1 and details['jolt']['student_opsd'] == 0
    assert details['jolt']['student_outcome'] != 0 and torch.isfinite(loss)
    # A teacher attempt without verified guidance is refused outright (no unverified teacher data).
    teacher = episode_with_top_k(model, roles, identity, 'import')
    with pytest.raises(ValueError, match='teacher attempt'):
        prepare(rollouts(teacher.turns, role='teacher'), teacher, roles, identity, protected=protected)


def test_opt_out_keeps_historical_objective_and_student_guidance_capture_is_refused():
    from natlang_neuralese.rl.native_replay import prepare_objective
    model, roles, identity = load('lfm')
    episode = episode_with_top_k(model, roles, identity, 'none')
    kwargs = dict(identity=identity, roles=roles, fetch_block=episode.states.__getitem__)
    _, clipped = prepare_objective(rollouts(episode.turns), policy_loss={'schema': 'clipped-ratio/1'}, **kwargs)
    _, historical = prepare_objective(rollouts(episode.turns), **kwargs)
    assert clipped == historical and 'policy_loss' not in clipped
    imported = episode_with_top_k(model, roles, identity, 'import')
    with pytest.raises(ValueError, match='teacher attempt'):
        prepare(rollouts(imported.turns), imported, roles, identity)


def test_engine_renders_each_guidance_mode_at_the_declared_place(tmp_path):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.rl.privileged_context import REQUEST_SCHEMA, captured_block
    from natlang_neuralese.serve.engine import GenerationRequest
    engine = engine_fixture(tmp_path, 'lfm')
    facts = [{'id': 'expected_result:0', 'kind': 'expected_result', 'text': 'The verified result is t6.'},
             {'id': 'child_value:0', 'kind': 'child_value', 'text': 'The answer is t9.', 'match': ['t5']}]
    custody_receipt = {'schema': 'natlang.jolt-privileged-custody/1', 'case_id': 'c', 'split': 'train',
                       'world_sha256': 'd' * 64, 'protected_sha256': 'c' * 64}
    messages = [{'role': 'system', 'content': 't7'}, {'role': 'user', 'content': 't4 t5'}]
    writer = {'function': 'writeGuidance', 'sha256': 'a' * 64}
    def request(role, mode, state=None):
        value = {'schema': REQUEST_SCHEMA, 'invocation': {'name': 'run-1', 'role': role}, 'mode': mode,
                 'custody': custody_receipt, 'facts': [facts[0] if role == 'root' else facts[1]],
                 'selection': {'implementation': 'nl', 'sha256': 'b' * 64}}
        if mode == 'text':
            value['fallback'] = 'writer failed'
        else:
            value.update(writer=writer, state={'id': state})
        return value
    def generate(**options):
        return engine.generate(GenerationRequest(messages=messages, temperature=.9, seed=5, max_tokens=4,
                                                 rl_capture=True, **options))
    engine.start()
    try:
        state = engine.native_write_text('t8 t9')['id']
        replies = {mode: generate(privileged_context=request('root', mode, state))
                   for mode in ('import', 'write-only', 'text')}
        policy = generate(guidance_policy=writer)
        plain = generate()
        from natlang_neuralese.serve.native_learning import decide_native
        decision_plain = decide_native(engine, {'messages': messages, 'options': ['t8', 't9'], 'native_recording': True})
        decision_guided = decide_native(engine, {'messages': messages, 'options': ['t8', 't9'], 'native_recording': True,
            'x_natlang_privileged_context': request('root', 'import', state)})
        with pytest.raises(Exception, match='not visible'):
            generate(privileged_context={**request('root', 'import', state), 'facts': [facts[1]]})
    finally:
        engine.stop()
    assert policy['x_natlang_rl_capture']['guidance_policy'] == writer
    own = lambda segments: [t for s in segments if s['kind'] == 'tokens' for t in s['token_ids']]
    expected = own(plain['x_natlang_rl_capture']['prompt'])
    spec = engine._privileged_rendering()
    decision = decision_guided['native_decision_capture']
    assert decision['privileged']['mode'] == 'import'
    assert decision['continuations'] == decision_plain['native_decision_capture']['continuations']
    assert any(s.get('id') == state for s in decision['prompt'] if s['kind'] == 'native_state')
    for mode, reply in replies.items():
        capture = reply['x_natlang_rl_capture']
        block = captured_block({'graph_turn': 'run-1#turn1', 'sampling': capture})
        assert block.receipt['mode'] == mode and capture['privileged'] == block.receipt
        if mode == 'write-only':
            assert own(capture['prompt']) == expected and block.state_id == state
            continue
        tokens = spec.state_turn()[0] if mode == 'import' else spec.turn([facts[0]['text']])
        assert list(block.token_ids) == tokens and block.insert_at == block.receipt['insert_at']
        assert own(capture['prompt']) == expected[:block.insert_at] + tokens + expected[block.insert_at:]
        imports = [s for s in capture['prompt'] if s['kind'] == 'native_state']
        assert [s['id'] for s in imports] == ([state] if mode == 'import' else [])



@pytest.mark.parametrize('scale', [1.0, 1e-4])
def test_reverse_kl_from_logits_matches_float64_reference_including_near_agreement(scale):
    from natlang_neuralese.rl.jolt import reverse_kl
    generator = torch.Generator().manual_seed(4)
    # FP32 inputs (what the trainer has); the float64 reference uses exactly those values.
    logits = (torch.randn(6, 50, generator=generator) * 4).double()
    twin = (logits.float() + scale * torch.randn(6, 50, generator=generator)).double()
    logits[0, 3] = twin[0, 3] = float('-inf')  # a shared mask
    def reference(z):
        lp, lq = torch.log_softmax(z, -1), torch.log_softmax(twin, -1)
        return (lp.exp() * torch.where(torch.isfinite(lp), lp - lq, torch.zeros_like(lp))).sum(-1)
    z64 = logits.clone().requires_grad_(True)
    expected = reference(z64)
    (expected * torch.arange(1., 7., dtype=torch.float64)).sum().backward()
    z32 = logits.float().clone().requires_grad_(True)
    actual = reverse_kl(z32, twin.float())
    (actual * torch.arange(1., 7.)).sum().backward()
    # Near agreement (KL ~ 1e-9) the FP32 result stays within 2%; subtracting rounded log-probabilities is off by
    # several hundred times the value.
    tolerance, relative = (1e-12, 2e-2) if scale < 1 else (1e-6, 1e-3)
    torch.testing.assert_close(actual.double(), expected.detach(), rtol=relative, atol=tolerance)
    grad = torch.where(torch.isfinite(logits), z64.grad, torch.zeros_like(z64.grad))
    torch.testing.assert_close(z32.grad.double(), grad, rtol=relative, atol=tolerance * 10)
    with pytest.raises(ValueError, match='mask'):
        reverse_kl(z32.detach(), torch.zeros_like(z32))


def test_root_calls_come_from_the_execution_graph_not_turn_order():
    # The child's turn is captured first; the root is the invocation without a caller edge.
    turns = [{'graph_turn': 'child-1#turn1'}, {'graph_turn': 'run-1#turn1'}]
    graph = [{'kind': 'invocation', 'node': 'call:child-1', 'inputs': [{'node': 'call:run-1', 'port': 'caller'}]},
             {'kind': 'invocation', 'node': 'call:run-1', 'inputs': []},
             {'kind': 'model_turn', 'node': 'child-1#turn1',
              'inputs': [{'node': 'call:child-1', 'port': 'invocation'}]}]
    assert root_calls({'execution_graph': graph}, turns) == {'run-1'}
    assert root_calls({}, turns) == {'child-1'}  # no invocation events: first captured turn's call
    cyclic = [{'kind': 'invocation', 'node': 'call:a', 'inputs': [{'node': 'call:b', 'port': 'caller'}]},
              {'kind': 'invocation', 'node': 'call:b', 'inputs': [{'node': 'call:a', 'port': 'caller'}]}]
    with pytest.raises(ValueError, match='no root invocation'):
        root_calls({'execution_graph': cyclic}, turns)



def test_teacher_and_student_entries_carry_the_worlds_guidance_controls(tmp_path):
    from natlang_neuralese.rl.jolt import GUIDANCE_RUN_SCHEMA, group_feedback, student_guidance, teacher_entries
    from natlang_neuralese.rl.privileged_context import case_facts
    from natlang_neuralese.rl.task_rollouts import build_queue
    case = train_case()
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps(case) + '\n' + json.dumps(protected_case()) + '\n')
    config = tmp_path / 'sampling-config.json'
    config.write_text(json.dumps({'guidance': False, 'x_natlang_rl_capture': True}))
    entries = build_queue(tmp_path / 'packet.jsonl', 2, 2, tmp_path, temperature=1., case_seconds=60,
                          max_model_requests=8, max_turns=8, seed_base=700000)
    students = [{'task_id': case['id'], 'group': 0, 'sample': 0, 'reward': {'quality': 0., 'feedback': 'wrong count'}},
                {'task_id': case['id'], 'group': 0, 'sample': 1, 'reward': {'quality': 0.}},
                {'task_id': protected_case()['id'], 'group': 1, 'sample': 0, 'reward': {'quality': 1.}}]
    result = teacher_entries(entries, students, cases=cases, config=config, out=tmp_path, attempts=1,
                             seed_base=700000, custody=custody())
    assert result['refusal'] is None and len(result['entries']) == 1  # the protected world gets no teacher
    entry = result['entries'][0]
    assert (entry['jolt_role'], entry['group'], entry['sample'], entry['key']) == ('teacher', 0, 2, 's7-task-000-t00')
    assert entry['seed'] not in {e['seed'] for e in entries}
    controls = json.loads(open(entry['chat_request_config']).read())
    assert controls['x_natlang_rl_capture'] is True and 'x_natlang_privileged_context' not in controls
    run = controls['x_natlang_guidance']
    receipt, facts = case_facts(case, custody=custody(), attempt_feedback=group_feedback(case, students[:2], 2))
    assert run['schema'] == GUIDANCE_RUN_SCHEMA and run['role'] == 'teacher' and run['custody'] == receipt
    assert [f['id'] for f in run['facts']] == [f.id for f in facts]
    assert any(f['kind'] == 'attempt_feedback' for f in run['facts'])
    # Students of the training world write guidance for their OPSD twin; the protected world's do not.
    ids = [case['id'], protected_case()['id']]
    assert student_guidance(entries, ids=ids, cases=cases, config=config, out=tmp_path, custody=custody()) is None
    for item in entries:
        if ids[int(item['index'])] == case['id']:
            student = json.loads(open(item['chat_request_config']).read())['x_natlang_guidance']
            assert student['role'] == 'student' and not any(f['kind'] == 'attempt_feedback' for f in student['facts'])
        else:
            assert 'chat_request_config' not in item
