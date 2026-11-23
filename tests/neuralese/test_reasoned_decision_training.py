"""Reasoned decision readouts in training (DECISIONS 2026-11-23 "reasoned decisions", training rule): the collector's
reasoning generations never become turns; a scored sample trains only when its answer is the one the program used;
its options are scored after its own reasoning (prefix) and the reasoning trains as token CE (``exp0_sft
--no-decision-reasoning`` opts out); the capture's scored prompt is the authority for the rendering."""
import collections
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese')]
import exp0_sft  # noqa: E402
import sft_sources  # noqa: E402
from natlang_neuralese.common.decision_objective import decision_loss  # noqa: E402
from natlang_neuralese.data.decision_graph import (DECISION_SYSTEM_PROMPT, decision_messages,  # noqa: E402
                                                   reasoned_decision_plan, token_ids_sha256, tokenizer_renderer)

LFM3 = Path.home() / ('.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                      '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
REASONING = [' The dentist blocks the morning, so I decline. ', ' Thursday looks free enough; I accept. ',
             ' The calendar clashes with the meeting. Decline. ']


@pytest.fixture(scope='module')
def renderer():
    if not (LFM3 / 'tokenizer.json').is_file():
        pytest.skip('LFM3-1.7B snapshot not cached')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True, trust_remote_code=False)
    out = tokenizer_renderer(tokenizer, model_type='lfm3')
    out.render = sft_sources.TemplateRenderer.render.__get__(out)
    return out


def _messages():
    return decision_messages([{'role': 'system', 'content': DECISION_SYSTEM_PROMPT},
                              {'role': 'user', 'content': 'Function: answerInvite\n\nInstructions: Decide.\n\n'
                                                          'Arguments:\ninvite: "Thu 10:00"'}],
                             ['accept', 'decline'])


# ---- collector trajectory -> rollout turns ---------------------------------------------------------------------------
def _trajectory(answers, action, reasonings, scored_prompts, sampled):
    """A reasoned readout as the collector writes it: per sample its reasoning generation, then its scored entry."""
    messages, replies = _messages()
    steps = []
    for j, (answer, reasoning) in enumerate(zip(answers, reasonings)):
        steps.append({'phase': 'decision_reasoning', 'invocation_id': 'c1', 'context': messages,
                      'decision_reasoning': {'index': j, 'samples': len(answers)},
                      'assistant': {'content': 'free reply', 'reasoning': reasoning, 'calls': []},
                      'model_response': {'calls': [], 'raw_calls': [], 'rl_sampling': {
                          'prompt': [{'kind': 'tokens', 'token_ids': [1, 2, 3], 'producing': [False] * 3}],
                          'samples': [{'token_id': t, 'logp': -1.} for t in sampled[j]]}}})
        capture = {'schema': 'natlang.native-decision-capture/1', 'reasoning': reasoning,
                   'prompt': [{'kind': 'tokens', 'token_ids': scored_prompts[j], 'producing': [False] * len(scored_prompts[j])}],
                   **({'decision_sample': {'chosen': replies.index(answer)}} if j == action else {})}
        steps.append({'phase': 'action', 'invocation_id': 'c1', 'context': messages, 'tools_offered': [],
                      'decision_reasoning': {'index': j, 'samples': len(answers), 'action': j == action},
                      'assistant': {'content': answer, 'reasoning': reasoning, 'calls': []},
                      'model_response': {'calls': [], 'raw_calls': [], 'text': answer, 'rl_sampling': capture}})
    return {'trajectory': steps}


def test_rollout_turns_drop_reasoning_generations_and_pair_their_sampled_ids():
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    result = _trajectory(['"decline"', '"accept"'], 0, REASONING[:2], [[1, 2, 3, 9], [1, 2, 3, 8]], [[5, 6], [7]])
    turns = rollout_turns(result)
    assert len(turns) == 2 and all(t['target']['content'] != 'free reply' for t in turns)
    assert [t['target']['reasoning_content'] for t in turns] == REASONING[:2]
    first, second = (t['decision_reasoning'] for t in turns)
    assert first['action'] is True and second['action'] is False and first['samples'] == 2
    assert first['reasoning_sample_ids'] == [5, 6] and second['reasoning_sample_ids'] == [7]
    assert first['scored_prompt_sha256'] == token_ids_sha256([1, 2, 3, 9])
    assert first['reasoning_prompt_sha256'] == token_ids_sha256([1, 2, 3])
    # an instant readout's entry carries no reasoning marker
    plain = {'trajectory': [{**result['trajectory'][1], 'decision_reasoning': None}]}
    assert 'decision_reasoning' not in rollout_turns(plain)[0]


def _turn(answer, j, samples, action, reasoning, invocation='c1', messages=None):
    messages = messages or _messages()[0]
    return {'messages': messages, 'tools': None, 'invocation_id': invocation,
            'target': {'role': 'assistant', 'content': answer, 'reasoning_content': reasoning},
            'decision_reasoning': {'index': j, 'samples': samples, 'action': action}}


def test_the_plan_trains_only_samples_that_agree_with_the_action():
    plain = {'messages': [{'role': 'user', 'content': 'x'}], 'tools': None, 'invocation_id': 'c0',
             'target': {'role': 'assistant', 'content': 'y'}}
    turns = [plain, _turn('"decline"', 0, 3, False, 'a'), _turn('"accept"', 1, 3, False, 'b'),
             _turn('"decline"', 2, 3, True, 'c'),
             # the same readout again in the same invocation: a new group, here without its action entry
             _turn('"decline"', 0, 2, False, 'd'), _turn('"decline"', 1, 2, False, 'e')]
    plan = reasoned_decision_plan(turns)
    assert plan[0] is None
    assert [p['why'] for p in plan[1:]] == ['agrees', 'answer-differs', 'agrees', 'no-action', 'no-action']
    assert [p['train'] for p in plan[1:]] == [True, False, True, False, False]
    assert plan[1]['group'] == plan[3]['group'] != plan[4]['group']
    # a field-mode readout: sample j's later step extends its earlier step's reply (one group; its last reply counts)
    steps = [_turn('{"a":"x","', 0, 2, False, 'r0'), _turn('{"a":"y","', 1, 2, True, 'r1'),
             _turn('{"a":"x","b":1}', 0, 2, False, 'r0'), _turn('{"a":"y","b":1}', 1, 2, True, 'r1')]
    field = reasoned_decision_plan(steps)
    assert len({p['group'] for p in field}) == 1 and [p['why'] for p in field] == ['answer-differs', 'agrees'] * 2


# ---- rendering against the scorer and the capture --------------------------------------------------------------------
def _reasoned(renderer, answer, reasoning, action=True, sampled=None, scored_sha=None):
    from natlang_neuralese.serve.native_learning import decision_items
    messages, replies = _messages()
    before, _ = decision_items(renderer, messages, None, replies, reasoning=reasoning)
    turn = _turn(answer, 0, 1, action, reasoning, messages=messages)
    turn['decision_reasoning']['scored_prompt_sha256'] = scored_sha or token_ids_sha256([v for _, v in before])
    if sampled is not None:
        turn['decision_reasoning']['reasoning_sample_ids'] = sampled
    return turn, [int(v) for _, v in before], replies


def test_a_reasoned_row_scores_after_its_reasoning_and_spans_it(renderer):
    turn, scored, replies = _reasoned(renderer, '"decline"', REASONING[0])
    prompt, target = renderer.render(turn['messages'], None, turn['target'])
    tok = renderer.tokenizer
    assert tok.decode(prompt).endswith('<|im_start|>assistant\n<think>')
    close = tok.convert_tokens_to_ids('</think>')
    sampled = target[:target.index(close) + 1] + tok.encode('free reply<|im_end|>', add_special_tokens=False)
    turn['decision_reasoning']['reasoning_sample_ids'] = sampled
    decision = sft_sources.render_decision(renderer, turn, np.asarray(prompt, np.int32))
    assert decision['label'] == 1 and prompt + decision['prefix'].tolist() == scored
    start, stop = decision['reasoning_span']
    assert start == len(prompt) and tok.decode(scored[start:stop]) == REASONING[0] + '</think>'
    assert decision['reasoning_exact'] is True
    # the plain row is the reasoning and the reply: R </think> option <|im_end|>
    assert tok.decode(target) == REASONING[0] + '</think>"decline"<|im_end|>'
    # sampled ids that tokenize the reasoning otherwise: the rendered ids train, the mismatch is reported
    turn['decision_reasoning']['reasoning_sample_ids'] = [sampled[0] + 1] + sampled[1:]
    assert sft_sources.render_decision(renderer, turn, np.asarray(prompt, np.int32))['reasoning_exact'] is False
    # a teacher's reply: conditioned the same way, its reasoning never a target
    other = sft_sources.render_decision(renderer, turn, np.asarray(prompt, np.int32), own=False)
    assert 'reasoning_span' not in other and other['prefix'].tolist() == decision['prefix'].tolist()
    # the capture's scored prompt is the authority
    turn['decision_reasoning']['scored_prompt_sha256'] = token_ids_sha256(scored[:-1])
    assert sft_sources.render_decision(renderer, turn, np.asarray(prompt, np.int32)) == {
        'error': 'prompt-differs-from-capture'}


def test_render_call_carries_the_reasoned_decision_and_instant_readouts_stay(renderer):
    turn, scored, _ = _reasoned(renderer, '"accept"', REASONING[1])
    prompt, target, context, decision = sft_sources.render_call(renderer, 'rl-own', turn, decisions=True)
    assert context == 0 and 'reasoning_span' in decision
    row, _ = sft_sources.rendered_row((prompt, target, context, decision), 'rl-own', 'k', 0, turn)
    assert row['decision']['reasoning_span'] == decision['reasoning_span']
    instant = {k: v for k, v in turn.items() if k != 'decision_reasoning'}
    instant['target'] = {'role': 'assistant', 'content': '"accept"'}
    plain = sft_sources.render_call(renderer, 'rl-own', instant, decisions=True)[3]
    assert 'reasoning_span' not in plain and renderer.tokenizer.decode(plain['prefix']) == '</think>'


# ---- the learner -----------------------------------------------------------------------------------------------------
class Tiny(torch.nn.Module):
    def __init__(self, vocab=13, width=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(vocab, width, generator=g))
        self.mix = torch.nn.Parameter(torch.randn(width, width, generator=g) / width)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, width, generator=g))


def tiny_forward(model, ids, lengths=None):
    parts = ids.split(list(lengths)) if lengths is not None else [ids]
    out = []
    for part in parts:
        x = model.emb[part]
        out.append(x + torch.tanh(torch.cumsum(x, 0) @ model.mix))
    return torch.cat(out)


def _logp(model, ids):
    return torch.log_softmax(tiny_forward(model, torch.tensor(ids)) @ model.lm_head.weight.T, -1)


def reasoned_chain(row_prompt, reasoning, shared, options, label):
    """A reasoned decision chain as sft_sources builds it: the row prompt ends at the generation prompt, the prefix
    is the reasoning (its close included), then the tokens every option shares."""
    decision = {'options': [np.asarray(o, np.int32) for o in options], 'label': label,
                'label_source': 'verified-trajectory', 'prefix': np.asarray(reasoning + shared, np.int32),
                'reasoning_span': (len(row_prompt), len(row_prompt) + len(reasoning)), 'reasoning_exact': True}
    sample = np.asarray(reasoning + shared + options[label], np.int32)
    row = {'id': 'r', 'invocation': 'c', 'index': 0, 'prompt': np.asarray(row_prompt, np.int32), 'sample': sample,
           'decision': decision}
    return exp0_sft.chains_of([{'task_id': 't', 'round': 'rl:x', 'rows': [row]}])[0]


@pytest.mark.parametrize('fast', [True, False])
def test_the_reasoning_trains_as_token_ce_beside_the_choice(fast):
    model = Tiny()
    row_prompt, reasoning, shared, options = [1, 2, 3], [4, 5, 6], [7], [[8, 9], [10], [11, 12]]
    chain = reasoned_chain(row_prompt, reasoning, shared, options, 2)
    other = reasoned_chain([2, 3], [5], [], [[8], [9, 10]], 0)
    chains = [chain, other]
    assert exp0_sft.mark_decision_objective(chains) == 2
    assert exp0_sft.chain_mass(chain) == exp0_sft.DECISION_MASS + 3 and exp0_sft.chain_mass(other) == 2.
    prompt = row_prompt + reasoning + shared
    scores = torch.stack([sum(_logp(model, prompt + o)[len(prompt) - 1 + j, t] for j, t in enumerate(o)) for o in options])
    lp = _logp(model, prompt)
    reasoning_nll = -sum(lp[len(row_prompt) - 1 + j, t] for j, t in enumerate(reasoning))
    scores2 = torch.stack([sum(_logp(model, [2, 3, 5] + o)[2 + j, t] for j, t in enumerate(o)) for o in [[8], [9, 10]]])
    nll2 = -_logp(model, [2, 3, 5])[1, 5]
    expected = (decision_loss(scores, rule='logLoss', probabilities=[0., 0., 1.]) + reasoning_nll
                + decision_loss(scores2, rule='logLoss', probabilities=[1., 0.]) + nll2) / 9.
    wanted = torch.autograd.grad(expected, list(model.parameters()))
    total, computed, report = exp0_sft.decision_backward(model, tiny_forward, chains, 9., 64 if fast else 0, 0.,
                                                         fast=fast)
    assert abs(total / 9. - float(expected.detach())) < 1e-5
    assert report['decisions'] == 2 and report['reasoning_tokens'] == 4
    assert abs(report['reasoning_nll'] - float((reasoning_nll + nll2).detach())) < 1e-4
    assert computed == sum(len(prompt) + len(o) for o in options) + 2 * 3 + 1 + 2
    for parameter, grad in zip(model.parameters(), wanted):
        torch.testing.assert_close(parameter.grad, grad, atol=1e-6, rtol=1e-5)


def test_the_opt_outs():
    chain = reasoned_chain([1, 2], [4, 5], [], [[8], [9]], 1)
    assert exp0_sft.mark_decision_objective([chain], reasoning=False) == 1
    assert exp0_sft.chain_mass(chain) == exp0_sft.DECISION_MASS and exp0_sft.decision_reasoning_span(chain) is None
    model = Tiny()
    _, _, report = exp0_sft.decision_backward(model, tiny_forward, [chain], 1., 0, 0., fast=False)
    assert report['reasoning_tokens'] == 0
    # without the decision objective the plain row trains the reasoning and the reply once, as token CE
    plain = reasoned_chain([1, 2], [4, 5], [], [[8], [9]], 1)
    exp0_sft.mark_decision_objective([plain], on=False)
    assert not plain['decision_objective'] and exp0_sft.chain_mass(plain) == 3.
    assert plain['segs'][0][2]['sample'].tolist() == [4, 5, 9]


def test_the_build_report_counts_reasoned_chains():
    chain = reasoned_chain([1, 2], [4, 5, 6], [], [[8], [9]], 1)
    chain['part'] = 'rl-own'
    report = sft_sources.decision_report([chain])
    assert report['parts']['rl-own']['reasoned'] == 1 and report['parts']['rl-own']['reasoning_target_tokens'] == 3


# ---- graph rows, lowering, program tune ------------------------------------------------------------------------------
def test_a_reasoned_graph_record_lowers_with_its_reasoning_in_the_prompt(renderer):
    import lower_sft_sources
    from natlang_neuralese.data.decision_graph import decision_graph_row
    from natlang_neuralese.serve.native_learning import decision_items
    messages, replies = _messages()
    row = decision_graph_row(renderer, {'id': 'g', 'split': 'train', 'source_groups': ['s'], 'messages': messages,
                                        'options': replies, 'label': 1, 'label_source': 'verified-trajectory',
                                        'reasoning': REASONING[2]}, tokenizer_fingerprint='f')
    before, continuations = decision_items(renderer, messages, None, replies, reasoning=REASONING[2])
    stats = collections.Counter()
    (ids, weights, _, decision), = lower_sft_sources._decision_sequences(row, stats)
    chain = lower_sft_sources.to_chain({'id': 'g', 'family': 'f', 'ids': ids, 'w': weights, 'decision': decision})
    prompt, options, target = exp0_sft.decision_parts(chain)
    assert prompt.tolist() == [int(v) for _, v in before] and [o.tolist() for o in options] == continuations
    assert renderer.tokenizer.decode(prompt).endswith(REASONING[2] + '</think>') and target == [0., 1.]


def test_program_tune_rows_skip_samples_that_were_not_used():
    import program_tune
    turns = [_turn('"decline"', 0, 2, True, 'r0'), _turn('"accept"', 1, 2, False, 'r1')]
    rows, dispositions = program_tune.episode_turns('t', 'rl:x:g:0', turns)
    assert dispositions == ['admissible', 'reasoned-sample-not-used'] and len(rows) == 1
    assert rows[0]['decision_reasoning'] == {'index': 0, 'samples': 2, 'action': True}
    assert rows[0]['reply'] == {'text': '"decline"', 'calls': [], 'reasoning': 'r0'}


def test_tune_rows_never_train_a_reasoning_generation():
    from natlang_neuralese.tune import rows
    generation = {'schema': rows.TURN_SCHEMA, 'decision_reasoning': {'index': 0, 'samples': 4}}
    action = {'schema': rows.TURN_SCHEMA, 'decision_reasoning': {'index': 0, 'samples': 4, 'action': True},
              'decision': {'options': ['"a"', '"b"'], 'chosen': 0}, 'target': {'reasoning_content': 'R'}}
    assert rows.reasoning_generation(generation) and not rows.reasoning_generation(action)
    assert rows.decision_reasoning_text(action) == 'R'
    assert rows.decision_reasoning_text({**action, 'decision': {**action['decision'], 'reasoning': 'S'}}) == 'S'
    assert rows.decision_reasoning_text({'decision': {'options': [], 'chosen': 0}}) is None
    with pytest.raises(rows.TuneMathError):
        rows.turn_graph(None, generation, prompt_weight=0.)
    assert rows.lower_rows(None, [generation]) == []


def test_observed_live_reasoning_generations_are_not_training_turns():
    import live_teacher_sources
    exchange = {'kind': 'generation', 'request': {'messages': [], 'decision_reasoning': {'index': 0, 'samples': 2}}}
    record = {'call_id': 'c', 'parent_call_id': None, 'definition': {'name': 'f', 'id': 'd'}}
    decision = live_teacher_sources.observed_decision('b', record, {'c': ['c']}, 0, exchange,
                                                      assistant={'content': 'x', 'reasoning': 'r', 'raw_calls': []})
    assert decision['phase'] == 'decision_reasoning'


# ---- graph coverage and the SFT replay -------------------------------------------------------------------------------
def test_graph_coverage_counts_a_reasoned_readouts_scored_entries_only():
    """The collector numbers a readout's scored decision turns 1..n per call and gives its reasoning generations no
    model-turn node: the graph's model turns are exactly the rollout's phase-'action' turns."""
    from natlang_neuralese.rl.episode_graph import rollout_turn_names, validate_rollout_graph
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    turns = rollout_turns(_trajectory(['"decline"', '"accept"', '"decline"'], 2, REASONING,
                                      [[1, 2, 3, 9], [1, 2, 3, 8], [1, 2, 3, 7]], [[5], [6], [7]]))
    assert rollout_turn_names(turns) == ['c1#turn1', 'c1#turn2', 'c1#turn3']
    graph = [{'node': f'c1#turn{n}', 'kind': 'model_turn', 'call_id': 'c1', 'inputs': []} for n in (3, 1, 2)]
    names, nodes = validate_rollout_graph({'turns': turns, 'execution_graph': graph})
    assert set(nodes) == set(names)
    with pytest.raises(ValueError):   # a reasoning generation's node would be an uncaptured model turn
        validate_rollout_graph({'turns': turns, 'execution_graph': graph + [
            {'node': 'c1#turn4', 'kind': 'model_turn', 'call_id': 'c1', 'inputs': []}]})


def test_the_replay_answers_a_reasoned_readout_from_its_action_entry_as_one_call():
    messages = _messages()[0]
    opening = {'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'go'}], 'tools': None,
               'invocation_id': 'c0', 'target': {'role': 'assistant', 'content': 'done'}}
    group = [_turn('"accept"', 0, 3, False, 'r0'), _turn('"decline"', 1, 3, True, 'r1'),
             _turn('"decline"', 2, 3, False, 'r2')]
    replay = sft_sources.ReplayEpisode([opening, *group])
    assert replay.calls == 2
    assert replay.answer({'messages': opening['messages']})['choices'][0]['message']['content'] == 'done'
    # three reasoning requests, the declined scoring, then the generated answer: all from the action entry
    for _ in range(3):
        message = replay.answer({'messages': messages})['choices'][0]['message']
        assert message == {'role': 'assistant', 'content': '"decline"', 'reasoning_content': 'r1'}
    assert replay.answer({'items': [{}]})['results'][0]['error'].startswith('decision-unsupported')
    assert replay.answer({'messages': messages})['choices'][0]['message']['content'] == '"decline"'
    assert replay.answer({'messages': messages}) is None   # the group is spent: a further request is unrecorded
    kinds = collections.Counter(entry['kind'] for entry in replay.log)
    assert kinds['exact'] == 2 and kinds['group-repeat'] == 3 and replay.served.count(False) == 0
