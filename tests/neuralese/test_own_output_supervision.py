"""Own-output supervision objective (owner 2026-10-28): weights, policy identity and trainer consumption."""
import copy
import json

import pytest
import torch
from torch.nn import functional as F

from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
from natlang_neuralese.train import own_output_supervision as own
from natlang_neuralese.train.text_supervision import ROLE_CODES

FP = 'f' * 64
# Small in-vocabulary IDs so the tiny backbones can execute these graphs.
START, SYSTEM, USER, ASSISTANT, TOOL, THINK, UNTHINK, END = 10, 11, 12, 13, 14, 15, 16, 17
ROLE_TOKENS = {'tokenizer_fingerprint': FP, 'start': START,
               'roles': {'system': SYSTEM, 'user': USER, 'assistant': ASSISTANT, 'tool': TOOL},
               'think_open': THINK, 'think_close': UNTHINK, 'tool_response_prefixes': [[30, 31]]}


def objective(share=.1, **extra):
    return own.resolve({'name': own.NAME, 'reasoning_share': share, 'role_tokens': ROLE_TOKENS, **extra})


def turn(role, body, *, reasoning=None):
    ids = [START, role]
    if reasoning is not None:
        ids += [THINK, *reasoning, UNTHINK]
    return ids + list(body) + [END]


def categories_of(weights_categories, code):
    return [i for i, c in enumerate(weights_categories[1]) if c == code]


R = ROLE_CODES.index('assistant_reasoning')
A = ROLE_CODES.index('assistant_reply')
U = ROLE_CODES.index('user')
T = ROLE_CODES.index('tool')


def test_context_is_zero_and_reasoning_gets_its_separate_share():
    ids = turn(USER, [40] * 30) + turn(ASSISTANT, [41] * 3, reasoning=[42] * 50) + turn(TOOL, [43] * 20)
    labels = own.label_roles(ids, ROLE_TOKENS)
    weights, categories = own.sequence_weights(labels, ids, [True] * len(ids), reasoning_share=.1,
                                               think_ids=(THINK, UNTHINK))
    reply = [i for i, c in enumerate(categories) if c == own.REPLY]
    reasoning = [i for i, c in enumerate(categories) if c == own.REASONING]
    assert sum(weights) == pytest.approx(1.)
    assert sum(weights[i] for i in reasoning) == pytest.approx(.1)
    assert sum(weights[i] for i in reply) == pytest.approx(.9)
    assert len(reasoning) == 50                      # the long reasoning body cannot dominate
    assert ids[reply[0]] == ASSISTANT and THINK in [ids[i] for i in reply] and UNTHINK in [ids[i] for i in reply]
    for i, label in enumerate(labels):
        if label not in (R, A):
            assert weights[i] == 0 and categories[i] == own.CONTEXT
    # Short reasoning is not ignored either: one reasoning token still gets the whole share.
    short = turn(ASSISTANT, [41] * 40, reasoning=[42])
    w, c = own.sequence_weights(own.label_roles(short, ROLE_TOKENS), short, [True] * len(short),
                                reasoning_share=.1, think_ids=(THINK, UNTHINK))
    assert sum(w[i] for i, k in enumerate(c) if k == own.REASONING) == pytest.approx(.1)
    # Markers may instead be declared reasoning.
    w, c = own.sequence_weights(own.label_roles(short, ROLE_TOKENS), short, [True] * len(short),
                                reasoning_share=.1, reasoning_markers='reasoning', think_ids=(THINK, UNTHINK))
    assert [short[i] for i, k in enumerate(c) if k == own.REASONING] == [THINK, 42, UNTHINK]


def test_decisions_are_balanced_and_failed_or_reasoning_only_turns_have_no_positive_weight():
    long_turn = turn(ASSISTANT, [41] * 200)
    failed = turn(ASSISTANT, [44] * 5)
    short_turn = turn(ASSISTANT, [45] * 2)
    ids = turn(USER, [40] * 5) + long_turn + failed + turn(TOOL, [43] * 4) + short_turn + turn(USER, [40])
    labels = list(own.label_roles(ids, ROLE_TOKENS))
    # Reasoning with no eligible action (e.g. a truncated turn whose header is excluded) is not a target.
    ids += [42, 42, 42]
    labels += [R, R, R]
    runs = own.assistant_runs(labels)
    assert len(runs) == 4
    eligible = [True] * len(ids)
    eligible[runs[1][0]:runs[1][1]] = [False] * (runs[1][1] - runs[1][0])   # the failed historical action
    weights, categories = own.sequence_weights(labels, ids, eligible, reasoning_share=.1, think_ids=(THINK, UNTHINK))
    masses = [sum(weights[begin:end]) for begin, end in runs]
    assert masses == pytest.approx([.5, 0., .5, 0.])
    assert all(categories[i] == own.ASSISTANT_CONTEXT for i in range(*runs[1]))
    assert all(weights[i] == 0 for i in range(*runs[3]))


def test_windows_follow_suffix_spans_and_windows_without_targets_are_zero_not_errors():
    ids = turn(USER, [40] * 4) + turn(ASSISTANT, [41] * 3) + turn(TOOL, [43] * 3) + turn(ASSISTANT, [46] * 2)
    roles = torch.tensor([list(own.label_roles(ids, ROLE_TOKENS))] * 3)
    tokens = torch.tensor([ids] * 3)
    second = own.assistant_runs(roles[0].tolist())[1]
    no_assistant = torch.tensor([[U] * len(ids)])
    weights, _ = own.window_weights(no_assistant, reasoning_share=.1)
    assert weights.shape == (1, len(ids)) and not weights.any()
    weights, categories = own.window_weights(roles, ids=tokens, reasoning_share=.1, think_ids=(THINK, UNTHINK),
        suffix_starts=[second[0], None, None], spans=[[[0, len(ids)]], [list(second)], [[0, len(ids)]]],
        exclude=torch.tensor([[False] * len(ids)] * 2 + [[True] * len(ids)]))
    assert weights[0].sum() == pytest.approx(1.) and weights[0, :second[0]].sum() == 0
    torch.testing.assert_close(weights[0], weights[1])
    assert not weights[2].any()
    # The legacy window functions are unchanged and still refuse a zero context coefficient.
    from natlang_neuralese.train.text_supervision import balanced_position_weights
    with pytest.raises(ValueError, match='context weight'):
        balanced_position_weights(torch.ones((1, 4)), [2], context_weight=0.)


def whole_row(*, opening=0, fail_second=False, rid='whole', groups=('family:a',)):
    nodes = turn(SYSTEM, [40] * 6) + turn(ASSISTANT, [47] * 3) + turn(USER, [40] * 8)
    nodes += turn(ASSISTANT, [41] * 4, reasoning=[42] * 9) + turn(TOOL, [43] * 12) + turn(ASSISTANT, [46] * 2)
    weights = [0.] + [.05] * (len(nodes) - 1)
    labels = own.label_roles(nodes, ROLE_TOKENS)
    runs = own.assistant_runs(labels)
    if fail_second:
        weights[runs[1][0]:runs[1][1]] = [0.] * (runs[1][1] - runs[1][0])
    return {'schema': SCHEMA, 'id': rid, 'split': 'train', 'source_groups': list(groups),
            'tokenizer_fingerprint': FP, 'nodes': [{'name': 'text', 'token_ids': nodes, 'target_weights': weights}],
            'conversion': {'role_interpretation': 'whole-trajectory-chat-roles',
                           'runtime_opening_assistant_turns': opening,
                           'supervision_policy': {'name': 'whole-trajectory-context-feedback-failed-turns-v2'}}}, runs


def policy(**extra):
    from natlang_neuralese.train.native_supervision import SCHEMA as POLICY
    return {'schema': POLICY, 'sources': [], 'objective': objective(**extra)}


def test_graph_application_keeps_failed_and_opening_turns_out_and_is_exact_on_positive_positions():
    from natlang_neuralese.train.native_supervision import apply_policy
    row, runs = whole_row(opening=1, fail_second=True)
    example = parse_graph(row, tokenizer_fingerprint=FP)
    weighted = apply_policy(example, row, source_sha256='a' * 64, policy=policy())
    assert isinstance(weighted, own.OwnOutputGraphExample)
    weights = weighted.target_weights
    assert sum(weights[i] for i in range(*runs[0])) == 0          # runtime opening
    assert sum(weights[i] for i in range(*runs[1])) == 0          # failed (raw zero)
    assert sum(weights[i] for i in range(*runs[2])) == pytest.approx(1.)
    assert sum(weights) == pytest.approx(1.)
    assert all(w == 0 or raw > 0 for w, raw in zip(weights, example.target_weights))
    assert weighted.plan is example.plan and weighted.id == example.id
    # Legacy policies leave the declared row weights untouched.
    from natlang_neuralese.train.native_supervision import SCHEMA as POLICY
    assert apply_policy(example, row, source_sha256='a' * 64, policy={
        'schema': POLICY, 'sources': [{'sha256': 'b' * 64, 'encoding': 'native-row-objectives-v1',
                                       'context_weight': .05}]}) is example
    # Role tokens are bound to the exact tokenizer.
    other = copy.deepcopy(row)
    other['tokenizer_fingerprint'] = 'e' * 64
    with pytest.raises(ValueError, match='different tokenizer'):
        own.graph_weights(other, objective())


def test_ordinary_rows_target_only_their_suffix_and_invocations_are_balanced():
    from natlang_neuralese.train.native_supervision import apply_policy
    history = turn(USER, [40] * 6) + turn(ASSISTANT, [41] * 5) + turn(USER, [40] * 3) + [START, ASSISTANT]
    target = [46, 46, END]
    ids = history + target
    row = {'schema': SCHEMA, 'id': 'ordinary', 'split': 'train', 'source_groups': ['g'], 'tokenizer_fingerprint': FP,
           'nodes': [{'name': 'text', 'token_ids': ids,
                      'target_weights': [0.] + [.05] * (len(history) - 1) + [1.] * len(target)}],
           'conversion': {'role_interpretation': 'ordinary-text', 'source_suffix_start': len(history),
                          'context_weight': .05}}
    weighted = apply_policy(parse_graph(row, tokenizer_fingerprint=FP), row, source_sha256='a' * 64,
                            policy={**policy(), 'sources': [{'sha256': 'a' * 64,
                                    'encoding': 'native-row-objectives-v1', 'context_weight': .05}]})
    assert weighted.target_weights[:len(history)] == (0.,) * len(history)
    assert weighted.target_weights[len(history):] == pytest.approx((1 / 3,) * 3)
    # Two invocations in one graph: a 3-decision transcript and a 1-decision call carry equal mass.
    many = turn(USER, [40]) + turn(ASSISTANT, [41] * 30) + turn(TOOL, [43]) + turn(ASSISTANT, [41] * 30) \
        + turn(TOOL, [43]) + turn(ASSISTANT, [41] * 30)
    one = turn(USER, [40]) + turn(ASSISTANT, [46])
    graph = {'schema': SCHEMA, 'id': 'two', 'split': 'train', 'source_groups': ['g'], 'tokenizer_fingerprint': FP,
             'nodes': [{'name': 'parent', 'token_ids': many, 'target_weights': [0.] + [1.] * (len(many) - 1)},
                       {'name': 'child', 'token_ids': one, 'target_weights': [0.] + [1.] * (len(one) - 1)}]}
    weights, _ = own.graph_weights(graph, objective())
    assert sum(weights[:len(many)]) == pytest.approx(1.) and sum(weights[len(many):]) == pytest.approx(1.)


def test_family_scales_equalize_families_with_a_bound_and_unit_mean():
    entries = [{'id': f'a{i}', 'groups': ['fam:a:x']} for i in range(8)] + \
              [{'id': 'b0', 'groups': ['fam:b:y']}, {'id': 'c0', 'groups': ['other']}]
    plain = objective(family_balance={'key': 'source-group', 'pattern': r'^fam:([a-z]+)', 'max_scale': 100.})
    scales, census = own.family_scales(plain, entries)
    assert sum(scales.values()) / len(scales) == pytest.approx(1.)
    assert scales['a0'] * 8 == pytest.approx(scales['b0']) and scales['b0'] == pytest.approx(scales['c0'])
    assert set(census['families']) == {'a', 'b', '<unmatched>'}
    capped = objective(family_balance={'key': 'source-group', 'pattern': r'^fam:([a-z]+)', 'max_scale': 2.})
    scales, _ = own.family_scales(capped, entries)
    assert sum(scales.values()) / len(scales) == pytest.approx(1.)
    assert scales['b0'] / scales['a0'] == pytest.approx(2. / (10 / 24))


def test_objective_identity_is_explicit_and_continuations_cannot_silently_change_it(tmp_path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.native_supervision import load_policy
    new = policy()
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(new))
    loaded = load_policy({'path': str(path), 'sha256': sha256_file_hex(path)})
    assert own.policy_objective(loaded) == objective()
    legacy = own.identity(own.resolve(None))
    current, transition = own.continuation_identity({}, loaded)
    assert current['name'] == own.NAME and transition is None
    with pytest.raises(ValueError, match='objective_transition'):
        own.continuation_identity({'stage_id': 'old'}, loaded)
    with pytest.raises(ValueError, match='objective_transition'):
        own.continuation_identity({'supervision_objective': own.identity(objective(.2))}, loaded)
    moved, transition = own.continuation_identity({'stage_id': 'old'},
                                                  {**loaded, 'objective_transition': {'from': legacy['sha256']}})
    assert transition == {'from': legacy, 'to': {k: moved[k] for k in ('name', 'sha256')}}
    assert own.continuation_identity({'supervision_objective': current}, loaded) == (current, None)
    # Legacy policies keep resolving to the historical objective unchanged.
    assert own.continuation_identity({'stage_id': 'old'}, None) == (legacy, None)
    for bad in [{'name': own.NAME, 'reasoning_share': 1., 'role_tokens': ROLE_TOKENS},
                {'name': own.NAME, 'reasoning_share': .1, 'role_tokens': ROLE_TOKENS, 'context_weight': .05},
                {'name': own.NAME, 'role_tokens': ROLE_TOKENS},
                {'name': 'whole-trajectory'}]:
        with pytest.raises(ValueError):
            own.resolve(bad)


def test_held_rows_get_objective_and_category_measurements_only():
    from natlang_neuralese.train.native_supervision import apply_policy
    row, runs = whole_row()
    row['split'] = 'test'
    example = parse_graph(row, tokenizer_fingerprint=FP)
    held = apply_policy(example, row, source_sha256='a' * 64, policy=policy())
    assert held.target_weights == example.target_weights
    sidecars = held.evaluation_weights
    assert set(sidecars) == {'own_outputs', 'own_reply', 'own_reasoning', 'context_monitor',
                             'assistant_context_monitor'}
    assert sum(sidecars['own_outputs']) == pytest.approx(1.)
    assert sum(sidecars['own_reasoning']) == 9
    assert held.evaluation_binding['objective']['name'] == own.NAME


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_trainer_loss_matches_objective_reference_with_category_telemetry_and_skips_empty_graphs(family):
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    from natlang_neuralese.train.native_supervision import apply_policy
    from test_shared_state import tiny
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    row, _ = whole_row()
    empty = {'schema': SCHEMA, 'id': 'empty', 'split': 'train', 'source_groups': ['g'], 'tokenizer_fingerprint': FP,
             'nodes': [{'name': 'text', 'token_ids': turn(USER, [40] * 5) + turn(TOOL, [43] * 5),
                        'target_weights': [0.] + [.25] * 15}]}
    examples = [apply_policy(parse_graph(item, tokenizer_fingerprint=FP), item, source_sha256='a' * 64,
                             policy=policy()) for item in (row, empty)]
    assert not any(examples[1].target_weights)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    weighted = examples[0]
    result = model(weighted.plan, retain_states=False)
    queries = [i - 1 for i, w in enumerate(weighted.target_weights) if w > 0]
    labels = torch.tensor([weighted.plan.invocations[0].token_ids[i + 1] for i in queries])
    weights = torch.tensor([weighted.target_weights[i + 1] for i in queries])
    ce = F.cross_entropy(model.logits(result['h_final'][0, queries]).float(), labels, reduction='none')
    expected = (ce * weights).sum() / weights.sum()
    expected_gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
    loss, details = packed_graph_loss(model, examples, checkpoint_layers=True, chunk_tokens=3)
    torch.testing.assert_close(loss, expected)
    for observed, reference in zip(torch.autograd.grad(loss, parameters, allow_unused=True), expected_gradients):
        assert (observed is None) == (reference is None)
        if reference is not None:
            torch.testing.assert_close(observed, reference)
    assert len(details) == 1 and details[0]['own_output'] is not None
    reasoning = torch.tensor([weighted.objective_categories[i + 1] == own.REASONING for i in queries])
    telemetry = details[0]['own_output']
    assert telemetry['reasoning']['weight'] == pytest.approx(.1 / 3)   # three decisions, one with reasoning
    assert telemetry['reasoning']['loss_sum'] == pytest.approx(float((ce * weights)[reasoning].sum()), rel=1e-5)
    assert telemetry['reply']['loss_sum'] == pytest.approx(float((ce * weights)[~reasoning].sum()), rel=1e-5)
    report = own.progress_telemetry(details, graphs=2)
    assert report['targeted_graphs'] == 1 and report['reply']['loss'] > 0
    # Family scale multiplies the normalized graph loss; an all-empty batch is a zero, not an error.
    scaled = copy.copy(weighted)
    scaled.loss_scale = 2.
    torch.testing.assert_close(packed_graph_loss(model, [scaled], chunk_tokens=3)[0], 2 * expected.detach(),
                               rtol=1e-5, atol=1e-6)
    zero, none = packed_graph_loss(model, [examples[1]], chunk_tokens=3)
    assert float(zero) == 0 and none == []
    # Legacy rows without the objective still refuse loss-free graphs.
    with pytest.raises(ValueError, match='no ordinary CE targets'):
        packed_graph_loss(model, [parse_graph({**empty, 'nodes': [{**empty['nodes'][0],
            'target_weights': [0.] * 16}]}, tokenizer_fingerprint=FP)], chunk_tokens=3)


def test_conversion_time_objective_matches_load_time_and_is_verified():
    from test_native_graph_conversion import ChatRoleAsciiTokenizer
    from natlang_neuralese.data.native_graph_conversion import whole_trajectory_text_row_graph, text_row_graph
    from natlang_neuralese.train.native_supervision import apply_policy
    tok = ChatRoleAsciiTokenizer()
    tokens = own.role_tokens_from_tokenizer(tok, tokenizer_fingerprint='b' * 64)
    declared = own.resolve({'name': own.NAME, 'reasoning_share': .1, 'role_tokens': tokens})
    text = ('<|im_start|>assistant\nruntime input<|im_end|>'
            '<|im_start|>assistant\nbad call<|im_end|>'
            '<|im_start|>tool\nrejected action: bad argument<|im_end|>'
            '<|im_start|>assistant\n<think>why</think>good call<|im_end|>')
    source = {'id': 'w', 'text': text, 'token_ids': tok(text)['input_ids'], 'tokenizer_sha256': 'a' * 64,
              'split': 'train', 'source_groups': ['g'], 'supervision': 'whole-trajectory',
              'source_record_ids': ['r'], 'runtime_opening_assistant_turns': 1}
    graph = whole_trajectory_text_row_graph(source, source_tokenizer_fingerprint='a' * 64, tokenizer=tok,
                                            expanded_tokenizer_fingerprint='b' * 64, objective=declared)
    weights = graph['nodes'][0]['target_weights']
    runs = own.assistant_runs(own.label_roles(source['token_ids'], tokens))
    assert [sum(weights[b:e]) for b, e in runs] == pytest.approx([0., 0., 1.])
    assert graph['conversion']['supervision_policy']['name'] == own.NAME
    assert graph['conversion']['supervision_policy']['failed_assistant_tokens_masked'] > 0
    loaded = apply_policy(parse_graph(graph, tokenizer_fingerprint='b' * 64), graph, source_sha256='a' * 64,
        policy={'schema': 'natlang.native-ce-supervision-policy/1', 'sources': [], 'objective': declared})
    assert loaded.target_weights == pytest.approx(tuple(weights))
    with pytest.raises(ValueError, match='different own-output objective'):
        apply_policy(parse_graph(graph, tokenizer_fingerprint='b' * 64), graph, source_sha256='a' * 64,
            policy={'schema': 'natlang.native-ce-supervision-policy/1', 'sources': [],
                    'objective': own.resolve({**declared, 'reasoning_share': .2})})
    prefix = '<|im_start|>assistant\nearlier<|im_end|><|im_start|>tool\nok<|im_end|><|im_start|>assistant\n'
    ordinary_text = prefix + 'answer<|im_end|>'
    ordinary = {'id': 'o', 'split': 'train', 'source_groups': ['g'], 'tokenizer_sha256': 'a' * 64,
                'text': ordinary_text, 'token_ids': tok(ordinary_text)['input_ids'],
                'supervised_suffix_start': len(tok(prefix)['input_ids'])}
    converted = text_row_graph(ordinary, source_tokenizer_fingerprint='a' * 64, tokenizer=tok,
                               expanded_tokenizer_fingerprint='b' * 64, objective=declared)
    w = converted['nodes'][0]['target_weights']
    boundary = ordinary['supervised_suffix_start']
    assert sum(w[:boundary]) == 0 and sum(w[boundary:]) == pytest.approx(1.)
