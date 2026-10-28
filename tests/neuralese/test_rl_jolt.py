"""JOLT+ self-teaching: privileged twins of the whole graph, per-term gradients and the leakage guard."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.common.sampling import behavior_top_k_declaration, encode_top_k
from natlang_neuralese.model.shared_state import SharedStateModel, native_step, splice_block, concatenate_captures
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.rl.jolt import (DEFAULT_JOLT, jolt_declaration, privileged_context, privileged_source,
                                       protected_identifiers, require_training_world, twin_invocations)
from natlang_neuralese.rl.policy_math import default_policy_loss, policy_loss_declaration
from natlang_neuralese.serve.store import native_state_id
from test_shared_state import tiny

PRIVILEGED = [7, 8, 9, 10]
ROLES = SimpleNamespace(produce_open=58, produce_close=59, consume_open=60, consume_close=61)
TEMPERATURE = .8


def model_and_identity(family='lfm', seed=901):
    torch.manual_seed(seed)
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    identity = {'checkpoint_sha256': '1'*64, 'tokenizer_fingerprint': '2'*64,
                'embedding_weight_sha256': '3'*64, 'architecture': model.port.config.model_type}
    return model, identity


def explicit_episode(model, identity, *, prefix=(), privileged_flag=False, top_k=8):
    """Explicit cached execution of a producer -> consumer episode; every invocation starts with ``prefix``.

    Returns (turns capture list, block, per-sample log distributions in sampling order)."""
    device = next(model.parameters()).device
    prefix = list(prefix)
    distributions = []
    def capture(prompt):
        segments = ([{'kind': 'tokens', 'token_ids': prefix, 'producing': [False]*len(prefix),
                      **({'privileged': True} if privileged_flag else {})}] if prefix else []) + prompt
        return {'schema': 'natlang.native-rl-sampling/1', 'policy': dict(identity), 'prompt': segments,
                'samples': [], 'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
                'behavior_top_k': behavior_top_k_declaration(top_k),
                'sampling_law': {'temperature': TEMPERATURE, 'seed': 3, 'guidance': False,
                                 'probabilities': 'post-processing-normalized'}}
    def sample(out, record, tokens):
        chunks = []
        for index, (token, producing) in enumerate(tokens):
            log_probs = model.logits(out['h_final'][:, -1]).float().div(TEMPERATURE).log_softmax(-1)[0]
            distributions.append(log_probs)
            record['samples'].append({'index': index, 'token_id': token, 'logp': float(log_probs[token]),
                'input_role': 'produce' if producing else 'ordinary', 'terminal': token == 2,
                'top_k': encode_top_k(log_probs, top_k)})
            if token != 2:
                out = native_step(model, torch.tensor([[token]], device=device), out['cache'], producing=producing)
                if producing:
                    chunks.append(out)
        return out, chunks
    with torch.no_grad():
        producer = capture([{'kind': 'tokens', 'token_ids': [1, 4], 'producing': [False, False]}])
        _, chunks = sample(native_step(model, torch.tensor([prefix + [1, 4]], device=device)), producer,
            [(ROLES.produce_open, False), (5, True), (6, True), (ROLES.produce_close, False), (2, False)])
        block = NativeStateBlock(text='payload', **identity, **concatenate_captures(chunks)).validate()
        block_id = native_state_id(block)
        producer['writes'] = [{'state_id': block_id, 'begin_sample': 0, 'end_sample': 4,
                               'payload_begin': 1, 'payload_end': 3, 'native_identity': identity}]
        consumer = capture([{'kind': 'tokens', 'token_ids': [3, ROLES.consume_open], 'producing': [False, False]},
            {'kind': 'native_state', 'id': block_id, 'native_identity': identity},
            {'kind': 'tokens', 'token_ids': [ROLES.consume_close], 'producing': [False]}])
        out = native_step(model, torch.tensor([prefix + [3, ROLES.consume_open]], device=device))
        cache = splice_block(model, out['cache'], block,
                             **{key: value for key, value in identity.items() if key != 'architecture'})
        out = native_step(model, torch.tensor([[ROLES.consume_close]], device=device), cache)
        sample(out, consumer, [(11, False), (2, False)])
    turns = [{'graph_turn': 'consumer', 'sampling': consumer, 'read_bindings': {block_id: 'write-0'}},
             {'graph_turn': 'producer', 'sampling': producer, 'written_nodes': {block_id: 'write-0'}}]
    return turns, block, distributions


def episodes_from(turns, *, role='student', rewards=(1, 0), extra=None):
    raw = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'], 'sampling': turn['sampling'],
            'replay': {'raw_calls_retained': True}} for turn in turns]
    rows = [{'task_id': 'train-task', 'sample': sample, 'split': 'train', 'group': 'g', 'jolt_role': role,
             'reward': {'quality': reward}, 'policy': {'checkpoint_sha256': '1'*64},
             'turns': copy.deepcopy(raw), **(extra or {})} for sample, reward in enumerate(rewards)]
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(rows)
    return rows


def stub_privileged(rollout, members):
    return {'token_ids': PRIVILEGED, 'record': {'schema': 'natlang.jolt-privileged-context/1',
            'task_id': rollout['task_id'], 'sha256': '9'*64, 'sources': ['authored_reference']}}


def prepare(rollouts, block, identity, declaration=None):
    from natlang_neuralese.rl.native_replay import prepare_objective
    return prepare_objective(rollouts, identity=identity, roles=ROLES, fetch_block=lambda key: block,
        policy_loss=declaration or {'schema': 'jolt-plus/1', 'outcome': {'schema': 'score-centered/1', 'k': 8}},
        privileged=stub_privileged)


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


def test_leakage_guard_and_privileged_context(tmp_path):
    case = {'id': 'world-a', 'split': 'train', 'source_groups': ['grp-a'],
            'curriculum': {'reference': {'root': 'answer()'}, 'decisive': [{'note': 'check x'}], 'split_group': 'sg-a'},
            'semantics': {'expected': {'value': 3}}}
    held = {'id': 'world-h', 'split': 'test', 'source_groups': ['grp-h'], 'curriculum': {'split_group': 'sg-h'}}
    packet = tmp_path / 'cases.jsonl'
    packet.write_text(json.dumps(held) + '\n')
    protected = protected_identifiers([packet])
    context = privileged_context(case, protected=protected,
        earlier_attempts=[{'sample': 0, 'reward': {'quality': 0, 'diagnosis': 'binding error'}}])
    assert 'answer()' in context['text'] and 'binding error' in context['text']
    assert context['sources'] == ['authored_reference', 'reference_values', 'oracle_checks', 'attempt_feedback']
    with pytest.raises(ValueError, match='train-split'):
        require_training_world(held, protected)
    for leak in ({'source_groups': ['grp-h']}, {'id': 'world-h'}, {'curriculum': {**case['curriculum'], 'split_group': 'sg-h'}}):
        with pytest.raises(ValueError, match='protected evaluation packet'):
            privileged_context({**case, **leak}, protected=protected)
    with pytest.raises(ValueError, match='protected'):
        require_training_world(case, frozenset())
    # The learner-side source: a dev/held episode, an unknown world and a forged teacher context are refused.
    tokenize = lambda text: [ord(c) % 50 for c in text[:6]]
    student = {'task_id': 'world-a', 'split': 'train', 'group': 'g', 'sample': 1, 'reward': {'quality': 0}}
    build = privileged_source([student], cases={'world-a': case}, protected=protected, tokenize=tokenize)
    assert build(student, [student])['record']['task_id'] == 'world-a'
    with pytest.raises(ValueError, match='train-split'):
        build({**student, 'split': 'dev'}, [])
    with pytest.raises(ValueError, match='no bound training world'):
        build({**student, 'task_id': 'world-h'}, [])
    teacher = {**student, 'jolt_role': 'teacher', 'privileged_context': {**context, 'text': context['text'] + 'x'}}
    with pytest.raises(ValueError):
        build(teacher, [])
    good = privileged_context(case, protected=protected)
    assert build({**student, 'jolt_role': 'teacher', 'privileged_context': good}, [])['record'] == good
    # The registered protected packets are found from the repository registry.
    from pathlib import Path
    from natlang_neuralese.rl.jolt import registered_protected_packets
    repo = Path(__file__).resolve().parents[2]
    try:
        paths = registered_protected_packets(repo)
    except ValueError:
        pytest.skip('protected packet files are not present in this checkout')
    assert all(path.name == 'cases.jsonl' for path in paths)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_student_twin_matches_explicit_cached_execution_with_privileged_prefix(family):
    """The twin in the packed whole-graph forward equals an explicit cached run with the block in every
    invocation, including the consumer's import of the privileged producer's exported KV."""
    from natlang_neuralese.rl.native_replay import compile_replay
    from natlang_neuralese.rl.jolt import twin_queries
    from natlang_neuralese.model.native_graph_plan import compile_graph
    model, identity = model_and_identity(family)
    turns, block, _ = explicit_episode(model, identity)
    replay = compile_replay(turns, identity=identity, roles=ROLES, fetch_block=lambda key: block)
    twins, shift = twin_invocations(replay.plan.invocations, role='student', prefix_ids=PRIVILEGED)
    combined, queries = twin_queries(replay.actions, replay.plan, twins, shift, {}, role='student')
    with torch.no_grad():
        hidden = model(combined, retain_states=False)['h_final'][0]
        twin = model.logits(hidden[torch.tensor(queries)]).float().div(TEMPERATURE).log_softmax(-1)
        actual = model.logits(hidden[torch.tensor([a['query'] for a in replay.actions])]).float().div(TEMPERATURE).log_softmax(-1)
    _, _, explicit = explicit_episode(model, identity, prefix=PRIVILEGED)
    _, _, original = explicit_episode(model, identity)
    order = [(a['turn'].split('#')[0], a['sample']) for a in replay.actions]
    index = {key: i for i, key in enumerate([('producer', s) for s in range(5)] + [('consumer', s) for s in range(2)])}
    torch.testing.assert_close(twin, torch.stack([explicit[index[key]] for key in order]), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual, torch.stack([original[index[key]] for key in order]), rtol=1e-4, atol=1e-4)
    assert (twin - actual).abs().max() > 1e-3, 'the privileged block really changes the teacher distribution'


def literal_reference(model, hidden, row, declaration, top_k):
    """Direct float64 transcription of loss_t = -A U_t + beta KL(p || sg p~)."""
    from natlang_neuralese.rl.native_replay import behavior_top_k_tensors
    from test_rl_score_centering import literal_surrogate
    actions = row['actions']
    role = row['jolt']['role']
    beta = declaration['beta_s'] if role == 'student' else declaration['beta_t']
    queries = torch.tensor([a['query'] for a in actions])
    twin_queries = torch.tensor([a['twin_query'] for a in actions])
    log_p = torch.log_softmax(model.logits(hidden[queries]).double() / TEMPERATURE, -1)
    log_q = torch.log_softmax(model.logits(hidden[twin_queries]).double() / TEMPERATURE, -1).detach()
    kl = (log_p.exp() * (log_p - log_q)).sum(1)
    labels = torch.tensor([a['token_id'] for a in actions])
    behavior = torch.tensor([a['behavior_logp'] for a in actions])
    ids, logq = behavior_top_k_tensors(actions, top_k, 'cpu')
    outcome = -row['advantage'] * literal_surrogate(log_p, labels, behavior, ids, logq,
                                                   declaration['outcome']['importance'], declaration['outcome']['eps'])
    return outcome.sum() / len(actions), beta * kl.sum() / len(actions)


def gradient_vector(model, loss):
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().double() for g, p in zip(grads, parameters)])


@pytest.mark.parametrize('role', ['student', 'teacher'])
def test_each_jolt_term_matches_its_reference_gradient(tmp_path, role):
    model, identity = model_and_identity('lfm')
    turns, block, _ = explicit_episode(model, identity, prefix=PRIVILEGED if role == 'teacher' else (),
                                       privileged_flag=role == 'teacher')
    rows = episodes_from(turns, role=role)
    declaration = {'schema': 'jolt-plus/1', 'beta_s': 0.7, 'beta_t': 0.3,
                   'outcome': {'schema': 'score-centered/1', 'k': 8, 'importance': 'tis'}}
    graphs, spec = prepare(rows, block, identity, declaration)
    canonical = policy_loss_declaration(declaration)
    assert spec['policy_loss'] == canonical and len(graphs) == 2
    objective = objective_for(tmp_path, spec)
    for graph in graphs:
        row = spec['rows'][graph['id']]
        assert row['jolt']['role'] == role and abs(row['advantage']) == .5
        assert len(graph['nodes']) == 4 and {n['name'] for n in graph['nodes']} == set(row['episode_turns']) | set(row['jolt']['twin_turns'])
        example, context, result = forward(model, graph, identity)
        loss, details = objective(model, result, context)
        assert details['policy_loss'] == 'jolt-plus/1' and details['sampled_tokens'] == 7
        parts = details['jolt']
        outcome_ref, kl_ref = literal_reference(model, result['h_final'][0], row, canonical, 8)
        term = 'student_opsd' if role == 'student' else 'teacher_kl'
        # The outcome surrogate's value is estimator-specific (tail form); its gradient is compared below.
        assert parts[role + '_outcome'] != 0
        assert abs(parts[term] - float(kl_ref)) < 1e-4 and parts[term] > 0
        torch.testing.assert_close(gradient_vector(model, loss), gradient_vector(model, outcome_ref + kl_ref),
                                   rtol=2e-3, atol=1e-7)
        # Gradients flow only through the sampled view, never into the stop-gradient twin positions.
        hidden_grad = torch.autograd.grad(loss, result['h_final'], retain_graph=True)[0][0]
        twin_rows = torch.tensor([a['twin_query'] for a in row['actions']])
        assert hidden_grad[twin_rows].abs().max() == 0
        assert hidden_grad[torch.tensor([a['query'] for a in row['actions']])].abs().max() > 0


def test_teacher_twin_is_the_student_graph_and_zero_spread_groups_still_teach(tmp_path):
    """Removing the captured privileged block from a teacher attempt reproduces the student's actual graph, and
    an all-equal group (no outcome signal) still yields the dense self-teaching term."""
    model, identity = model_and_identity('lfm')
    student_turns, block, _ = explicit_episode(model, identity)
    teacher_turns, teacher_block, _ = explicit_episode(model, identity, prefix=PRIVILEGED, privileged_flag=True)
    student_graphs, student_spec = prepare(episodes_from(student_turns, rewards=(1, 1)), block, identity)
    teacher_graphs, teacher_spec = prepare(episodes_from(teacher_turns, role='teacher'), teacher_block, identity)
    student_row = student_spec['rows'][student_graphs[0]['id']]
    teacher_row = teacher_spec['rows'][teacher_graphs[0]['id']]
    assert student_row['advantage'] == 0  # no spread: outcome term zero, OPSD term remains
    by_name = lambda graph: {n['name'].split('#')[0] + ('*' if n['name'].endswith(('#privileged', '#unprivileged')) else ''): n
                             for n in graph['nodes']}
    s_nodes, t_nodes = by_name(student_graphs[0]), by_name(teacher_graphs[0])
    for name in ('producer', 'consumer'):
        assert s_nodes[name]['token_ids'] == t_nodes[name + '*']['token_ids']
        assert s_nodes[name + '*']['token_ids'] == t_nodes[name]['token_ids']
        assert [(e['at'], e['start'], e['stop']) for e in s_nodes[name]['imports']] == \
               [(e['at'], e['start'], e['stop']) for e in t_nodes[name + '*']['imports']]
    objective = objective_for(tmp_path, student_spec)
    example, context, result = forward(model, student_graphs[0], identity)
    loss, details = objective(model, result, context)
    assert details['jolt']['student_outcome'] == 0 and details['jolt']['student_opsd'] > 0
    loss.backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0


def test_student_episode_with_privileged_capture_and_missing_source_are_refused(tmp_path):
    from natlang_neuralese.rl.native_replay import prepare_objective
    model, identity = model_and_identity('lfm')
    teacher_turns, block, _ = explicit_episode(model, identity, prefix=PRIVILEGED, privileged_flag=True)
    with pytest.raises(ValueError, match='teacher attempt'):
        prepare(episodes_from(teacher_turns, role='student'), block, identity)
    other_turns, other_block, _ = explicit_episode(model, identity, prefix=[7, 8], privileged_flag=True)
    with pytest.raises(ValueError, match='exact privileged block'):
        prepare(episodes_from(other_turns, role='teacher'), other_block, identity)
    turns, block, _ = explicit_episode(model, identity)
    with pytest.raises(ValueError, match='privileged training-world source'):
        prepare_objective(episodes_from(turns), identity=identity, roles=ROLES, fetch_block=lambda key: block,
                          policy_loss=default_policy_loss())
    # The explicit opt-out keeps the historical clipped objective bytes (no policy_loss, no twins).
    from natlang_neuralese.rl.native_replay import prepare_objective as plain
    _, clipped = plain(episodes_from(turns), identity=identity, roles=ROLES, fetch_block=lambda key: block,
                       policy_loss={'schema': 'clipped-ratio/1'})
    _, historical = plain(episodes_from(turns), identity=identity, roles=ROLES, fetch_block=lambda key: block)
    assert clipped == historical and 'policy_loss' not in clipped


def test_engine_teacher_attempt_captures_its_privileged_block(tmp_path, monkeypatch):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.serve.engine import GenerationRequest
    engine = engine_fixture(tmp_path, 'lfm')
    text = '<training-reference>\nReference solution: t5\n</training-reference>\n'
    engine.start()
    try:
        reply = engine.generate(GenerationRequest(messages=[{'role': 'user', 'content': 't4 t5'}], temperature=.9,
            seed=5, max_tokens=4, rl_capture=True, rl_top_k=8, privileged_context=text))
        plain = engine.generate(GenerationRequest(messages=[{'role': 'user', 'content': 't4 t5'}], temperature=.9,
            seed=5, max_tokens=4, rl_capture=True))
    finally:
        engine.stop()
    prompt = reply['x_natlang_rl_capture']['prompt']
    assert prompt[0]['privileged'] is True
    assert prompt[0]['token_ids'] == engine.tokenizer(text, add_special_tokens=False)['input_ids']
    assert not any(segment.get('privileged') for segment in plain['x_natlang_rl_capture']['prompt'])
    assert prompt[1:] == plain['x_natlang_rl_capture']['prompt']
