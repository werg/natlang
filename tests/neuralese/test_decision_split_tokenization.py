"""Split-tokenization decision scoring (semantic-FP B10, OPEN_ITEMS 8b34839ab): an option is scored as every token
sequence that writes it, its canonical tokens and its split form (common prefix, own middle, common suffix, each
tokenized alone), summed as a logsumexp, on the Python engines, the decision captures, the SFT rows and the learners.
Real LFM3 tokenizer (tokenizer only)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese')]
import exp0_sft  # noqa: E402
import grpo  # noqa: E402
import sft_sources  # noqa: E402
from natlang_neuralese.common.decision_objective import (decision_loss, option_sequence_scores,  # noqa: E402
                                                         sequence_owner)
from natlang_neuralese.data.decision_graph import (DECISION_SYSTEM_PROMPT, decision_messages,  # noqa: E402
                                                   tokenizer_renderer)
from natlang_neuralese.serve import native_learning  # noqa: E402
from natlang_neuralese.serve.native_learning import (SCORE_ITEM_FIELDS, decision_items,  # noqa: E402
                                                     decision_sequences, option_log_scores, split_pieces)

LFM3 = Path.home() / ('.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                      '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
OPTIONS = ['"spam"', '"billing"']


@pytest.fixture(scope='module')
def tokenizer():
    if not (LFM3 / 'tokenizer.json').is_file():
        pytest.skip('LFM3-1.7B snapshot not cached')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True, trust_remote_code=False)


@pytest.fixture(scope='module')
def renderer(tokenizer):
    out = tokenizer_renderer(tokenizer, model_type='lfm3')
    out.render = sft_sources.TemplateRenderer.render.__get__(out)
    return out


def _messages(values=('spam', 'billing')):
    opening = [{'role': 'system', 'content': DECISION_SYSTEM_PROMPT},
               {'role': 'user', 'content': 'Route the ticket.\n\nticket: "I was charged twice"'}]
    return decision_messages(opening, list(values))


def test_split_pieces_are_common_prefix_own_middle_common_suffix():
    assert split_pieces(['"spam"', '"billing"']) == [['"', 'spam', '"'], ['"', 'billing', '"']]
    assert split_pieces(['{"label":"spam","', '{"label":"ham","']) == [['{"label":"', 'spam', '","'],
                                                                        ['{"label":"', 'ham', '","']]
    # pieces end on word boundaries: a shared 'am"' does not cut spam / ham, a shared '"a' does not cut "a" / "ab"
    assert split_pieces(['"spam"', '"ham"']) == [['"', 'spam', '"'], ['"', 'ham', '"']]
    assert split_pieces(['"a"', '"ab"']) == [['"', 'a', '"'], ['"', 'ab', '"']]
    assert split_pieces(['{"n":1,"', '{"n":12,"']) == [['{"n":', '1', ',"'], ['{"n":', '12', ',"']]
    assert split_pieces(['"zé"', '"té"']) == [['"', 'zé', '"'], ['"', 'té', '"']]   # >= 0x80 is a word char
    assert split_pieces(['"x"', '"x y"']) == [['"x', '"'], ['"x', ' y', '"']]   # empty middle dropped
    assert split_pieces(['true']) is None
    assert SCORE_ITEM_FIELDS['split_tokenization'].startswith('served')


@pytest.mark.parametrize('open_reply', [False, True])
def test_the_split_form_differs_for_spam_and_equals_for_billing(tokenizer, renderer, open_reply):
    """"spam" is canonically '"sp','am','"' and splits as '"','spam','"' (one alternative); "billing" splits as its
    canonical tokens (no alternative, never counted twice). Same tokens before (</think>, moved to the prompt) and after
    (<|im_end|> closed, nothing open) the content; without the flag the canonical rendering is unchanged."""
    messages, replies = _messages()
    prompt, continuations, alternatives = decision_sequences(renderer, messages, None, replies, open_reply, None, True)
    plain_prompt, plain = decision_items(renderer, messages, None, replies, open_reply)
    assert replies == OPTIONS and continuations == plain and prompt == plain_prompt
    decode = lambda ids: [tokenizer.decode([i]) for i in ids]
    end = [] if open_reply else ['<|im_end|>']
    assert decode([i for _, i in prompt][-1:]) == ['</think>']
    assert decode(continuations[0]) == ['"sp', 'am', '"', *end]
    assert alternatives == [[tokenizer('"', add_special_tokens=False)['input_ids']
                             + tokenizer('spam', add_special_tokens=False)['input_ids']
                             + tokenizer('"', add_special_tokens=False)['input_ids']
                             + ([tokenizer.convert_tokens_to_ids('<|im_end|>')] if not open_reply else [])], []]
    assert decode(alternatives[0][0]) == ['"', 'spam', '"', *end]


def test_the_shared_prefix_is_common_to_every_sequence(renderer):
    """Field steps: '{"' and 'label' and '":"' are shared by every canonical and split sequence, so they are prompt;
    each sequence keeps its own tokens."""
    messages, _ = _messages()
    options = ['{"label":"spam","', '{"label":"ham","']
    prompt, continuations, alternatives = decision_sequences(renderer, messages, None, options, True, None, True)
    whole = [i for _, i in prompt]
    for ids, alts in zip(continuations, alternatives):
        for seq in [ids, *alts]:
            assert seq and renderer.tokenizer.decode(whole[-3:] + seq).startswith('{"label":"')


def test_option_scores_are_the_logsumexp_of_their_sequences():
    assert option_log_scores([-1., -2.], [[1], [2]], [[], []]) == [-1., -2.]
    scores = option_log_scores([-1., -2., -3.], [[1], [2]], [[[9]], []])
    assert abs(scores[0] - math.log(math.exp(-1.) + math.exp(-3.))) < 1e-12 and scores[1] == -2.
    s = torch.tensor([-1., -2., -3.], requires_grad=True)
    combined = option_sequence_scores(s, sequence_owner(2, [[[9]], []]), 2)
    torch.testing.assert_close(combined, torch.stack([torch.logaddexp(s[0], s[2]), s[1]]))
    combined.sum().backward()
    torch.testing.assert_close(s.grad, torch.tensor([math.exp(-1.) / (math.exp(-1.) + math.exp(-3.)), 1.,
                                                     math.exp(-3.) / (math.exp(-1.) + math.exp(-3.))]))
    plain = torch.tensor([-1., -2.])
    assert option_sequence_scores(plain, [0, 1], 2) is plain


def _engine(renderer, seen):
    def sglang_score(item, plan, continuations, bindings, alternatives=None):
        seen.append((continuations, alternatives))
        sequence = SimpleNamespace(native_imports=[], native_rl_policy={}, native_rl_adapter_binding=None,
                                   native_guidance_receipt=None, native_rl_prompt=plan)
        flat = native_learning.decision_sequence_list(continuations, alternatives)
        return native_learning.decision_response(item, sequence, continuations,
                                                 [(-float(len(seq)), len(seq)) for seq in flat], alternatives)
    engine = SimpleNamespace(tokenizer=renderer.tokenizer, prompt_rendering=renderer.prompt_rendering,
                             tool_call_format=renderer.tool_call_format, sglang_score=sglang_score)
    for name in ('_template', 'render_request', '_tokens', '_template_tokens', 'specials'):
        setattr(engine, name, getattr(renderer, name))
    return engine


def test_a_split_item_echoes_its_tokenizations_and_the_capture_keeps_the_alternatives(renderer):
    seen = []
    engine = _engine(renderer, seen)
    messages, replies = _messages()
    result = native_learning.decide_native(engine, {'messages': messages, 'options': replies,
                                                    'split_tokenization': True, 'native_recording': True})
    continuations, alternatives = seen[-1]
    assert result['split_tokenization'] is True and result['tokenizations'] == [2, 1]
    spam = math.log(math.exp(-len(continuations[0])) + math.exp(-len(alternatives[0][0])))
    assert abs(result['log_probs'][0] - spam) < 1e-9 and result['log_probs'][1] == -len(continuations[1])
    assert result['tokens'] == [len(ids) for ids in continuations]
    capture = result['native_decision_capture']
    assert capture['split_tokenization'] is True and capture['continuations'] == continuations
    assert capture['continuation_alternatives'] == alternatives
    plain = native_learning.decide_native(engine, {'messages': messages, 'options': replies})
    assert seen[-1][1] is None and 'split_tokenization' not in plain and 'tokenizations' not in plain
    bad = native_learning.decide_many_native(engine, {'items': [{'messages': messages, 'continuations': replies,
                                                                 'split_tokenization': 1}]})['results'][0]
    assert 'split_tokenization' in bad['error']


def test_the_rl_learner_reads_the_capture_alternatives_and_scores_their_logsumexp(renderer):
    """validate_decision_sample returns the alternatives; grpo lays out canonical sequences first, then every
    alternative, and maps each to its option."""
    from natlang_neuralese.rl.decision_policy import validate_decision_sample
    capture = {'continuations': [[1, 2], [3]], 'continuation_alternatives': [[[4, 5, 6]], []],
               'decision_sample': {'schema': 'natlang.decision-sample/1', 'log_probs': [-1., -2.], 'chosen': 0,
                                   'temperature': 1.}}
    assert validate_decision_sample(capture)['alternatives'] == [[[4, 5, 6]], []]
    with pytest.raises(ValueError):
        validate_decision_sample({**capture, 'continuation_alternatives': [[[]], []]})
    row = {'prompt': np.asarray([7, 8], np.int32), 'options': [np.asarray([1, 2]), np.asarray([3])],
           'alternatives': [[np.asarray([4, 5, 6])], []]}
    batch = [[{'row': row, 'advantage': 1.}]]
    ids, lengths, positions, targets, owner, counts, _, _ = grpo.decision_tensors(batch, 'cpu')
    assert counts == [2] and lengths == [4, 3, 5] and owner.tolist() == [0, 0, 1, 2, 2, 2]
    assert grpo.decision_sequence_options(batch) == [0, 1, 0]
    assert grpo.decision_tokens(row) == 4 + 3 + 5


def test_sft_rows_carry_the_alternatives_and_the_objective_sums_them(renderer):
    messages, replies = _messages()
    turn = {'messages': messages, 'tools': None, 'target': {'role': 'assistant', 'content': replies[0]},
            'invocation_id': 'c'}
    prompt, target = renderer.render(messages, None, turn['target'])
    prompt = np.asarray(prompt, np.int32)
    decision = sft_sources.render_decision(renderer, turn, prompt)
    _, continuations, alternatives = decision_sequences(renderer, messages, None, replies, False, None, True)
    assert [a.tolist() for a in decision['alternatives'][0]] == alternatives[0] and decision['alternatives'][1] == []
    row, _ = sft_sources.rendered_row((prompt, np.asarray(target, np.int32), 0, decision), 'rl-own', 'k', 0, turn)
    assert len(row['decision']['alternatives'][0]) == 1


def test_decision_backward_scores_an_option_as_the_logsumexp_of_its_sequences():
    from test_sft_decision_objective import Tiny, decision_chain, explicit_scores, tiny_forward
    model = Tiny()
    chain = decision_chain([1, 2, 3], [[4, 5], [6]], label=0)
    chain['decision']['alternatives'] = [[np.asarray([7, 8, 9], np.int32)], []]
    exp0_sft.mark_decision_objective([chain])
    s = explicit_scores(model, [1, 2, 3], [[4, 5], [6], [7, 8, 9]])
    expected = decision_loss(torch.stack([torch.logaddexp(s[0], s[2]), s[1]]), rule='logLoss',
                             probabilities=[1., 0.])
    wanted = torch.autograd.grad(expected, list(model.parameters()))
    total, computed, _ = exp0_sft.decision_backward(model, tiny_forward, [chain], 1., 64, 0.)
    assert abs(total - float(expected)) < 1e-5 and computed == 5 + 4 + 6 == exp0_sft.decision_tokens(chain)
    for parameter, grad in zip(model.parameters(), wanted):
        torch.testing.assert_close(parameter.grad, grad, atol=1e-6, rtol=1e-5)


def test_reasoning_ids_replace_the_re_encoded_reasoning_up_to_the_think_close(tokenizer, renderer):
    """The engines re-encode the reasoning text ('Hello' -> one token); the sampled ids ('Hel', 'lo') replace it, kept
    up to the first </think> id (the sample runs past it), and they are prompt (every option shares them)."""
    messages, replies = _messages()
    hel, lo = (tokenizer(t, add_special_tokens=False)['input_ids'] for t in ('Hel', 'lo'))
    close = tokenizer.convert_tokens_to_ids('</think>')
    assert tokenizer('Hello', add_special_tokens=False)['input_ids'] != hel + lo
    sampled = hel + lo + [close] + tokenizer('"spam"', add_special_tokens=False)['input_ids']
    info = {}
    prompt, continuations, _ = decision_sequences(renderer, messages, None, replies, False, 'Hello', True, sampled, info)
    plain_prompt, plain = decision_items(renderer, messages, None, replies, reasoning='Hello')
    ids, base = [i for _, i in prompt], [i for _, i in plain_prompt]
    encoded = tokenizer('Hello', add_special_tokens=False)['input_ids']
    head = len(base) - len(encoded) - 1          # the plain prompt: ... <think> Hello </think>
    assert base[head:] == encoded + [close] and ids == base[:head] + hel + lo + [close]
    assert continuations == plain and info['reasoning_ids_kept'] == len(hel + lo)
    # learner re-render: the SFT row's decision prompt takes the same ids
    seen = []
    engine = _engine(renderer, seen)
    result = native_learning.decide_native(engine, {'messages': messages, 'options': replies, 'reasoning': 'Hello',
                                                    'reasoning_ids': sampled, 'native_recording': True})
    assert result['reasoning_ids'] is True and result['native_decision_capture']['reasoning_ids_kept'] == len(hel + lo)
    with pytest.raises(Exception):
        decision_sequences(renderer, messages, None, replies, False, 'Hello', False, [10 ** 9])
    for bad in ({'reasoning_ids': sampled}, {'reasoning': '', 'reasoning_ids': sampled}):
        out = native_learning.decide_many_native(engine, {'items': [{'messages': messages, 'continuations': replies,
                                                                     **bad}]})['results'][0]
        assert 'reasoning' in out['error']


def _split_episode():
    """A reasoned split-tokenized readout as tune rollouts record it (test_tune_rl_decision_reasoning shape): the action
    entry's capture carries continuation_alternatives."""
    import test_tune_rl_decision_reasoning as reasoned
    episode = reasoned._episode(action=1)
    for turn in episode['turns']:
        turn['sampling']['continuations'] = [[11, 12, 2], [13, 2]]
        turn['sampling']['split_tokenization'] = True
        turn['sampling']['continuation_alternatives'] = [[[11, 14, 15, 2]], []]
    return episode, reasoned


def test_graph_replay_scores_a_split_decision_over_its_alternative_branches():
    """Whole-graph replay (tune RL): each alternative is its own branch node beside its option; the decision action
    carries their queries, the reasoned readout's action reasoning still joins the turn, and the row validates."""
    from natlang_neuralese.rl.native_replay import validate_policy_decision_branches
    episode, reasoned = _split_episode()
    turns, replay = reasoned._compile(episode, reasoned.CLOSE)
    (action,) = replay.decision_actions
    assert action['option_nodes'] == ['c1#turn2:option:0', 'c1#turn2:option:1']
    assert action['alternative_nodes'] == [['c1#turn2:option:0:alternative:0'], []]
    assert [[q['token_id'] for q in alt] for alt in action['alternatives'][0]] == [[11, 14, 15, 2]]
    nodes = {node.name: (node, offset) for node, offset in zip(replay.plan.invocations, replay.plan.offsets)}
    node, offset = nodes['c1#turn2:option:0:alternative:0']
    prompt = action['prompt_tokens']
    assert list(node.token_ids[prompt:]) == [11, 14, 15, 2] and action['alternatives'][0][0][0]['query'] == offset + prompt - 1
    assert 'c1#turn2:option:0:alternative:0' in replay.turn_nodes['c1#turn2']
    assert {n for ns in replay.turn_nodes.values() for n in ns} == set(nodes)
    row = {'decision_actions': [action], 'episode_turns': list(replay.turn_nodes)}
    validate_policy_decision_branches(replay.plan, row)
    with pytest.raises(ValueError):
        validate_policy_decision_branches(replay.plan, {**row, 'decision_actions': [
            {k: v for k, v in action.items() if k not in ('alternatives', 'alternative_nodes')}]})


def test_decision_log_policy_sums_an_options_sequences(monkeypatch):
    """pi over options from logsumexp(canonical, alternatives), on the flat and on remapped (physical) queries."""
    from natlang_neuralese.rl import decision_policy
    from natlang_neuralese.train import decision_supervision
    monkeypatch.setattr(decision_supervision, 'option_scores',
                        lambda model, hidden, options, chunk_tokens=128: torch.tensor(
                            [-float(sum(q['query'] for q in option)) for option in options]))
    q = lambda *rows: [{'query': r, 'token_id': 0} for r in rows]
    action = {'options': [q(1), q(2)], 'alternatives': [[q(3)], []], 'temperature': 1.}
    expected = torch.log_softmax(torch.stack([torch.logaddexp(torch.tensor(-1.), torch.tensor(-3.)),
                                              torch.tensor(-2.)]), 0)
    torch.testing.assert_close(decision_policy.decision_log_policy(None, None, action), expected)
    mapped = decision_policy.remap_decision(action, {1: 10, 2: 20, 3: 30})
    assert mapped['alternatives'] == [[q(30)], []]
    torch.testing.assert_close(decision_policy.decision_log_policy(None, None, mapped), torch.log_softmax(
        torch.stack([torch.logaddexp(torch.tensor(-10.), torch.tensor(-30.)), torch.tensor(-20.)]), 0))
    plain = {'options': [q(1), q(2)], 'temperature': 1.}
    torch.testing.assert_close(decision_policy.decision_log_policy(None, None, plain),
                               torch.log_softmax(torch.tensor([-1., -2.]), 0))


def test_the_behavior_probe_maps_a_split_decisions_alternatives_to_physical_rows(monkeypatch):
    from natlang_neuralese.rl import decision_policy
    from natlang_neuralese.rl.native_replay import behavior_logprob_probe
    seen = []

    def policy(model, hidden, action, *, queries, alternative_queries=None, chunk_tokens=128):
        seen.append(([q['query'] for option in queries for q in option],
                     [[q['query'] for q in alt] for alts in alternative_queries or () for alt in alts]))
        return torch.tensor(action['behavior_logps'])
    monkeypatch.setattr(decision_policy, 'decision_log_policy', policy)
    action = {'turn': 'd', 'chosen': 0, 'temperature': 1., 'option_nodes': ['d:option:0', 'd:option:1'],
              'behavior_logps': [-.5, -1.], 'options': [[{'query': 0, 'token_id': 3}], [{'query': 1, 'token_id': 4}]],
              'alternative_nodes': [['d:option:0:alternative:0'], []],
              'alternatives': [[[{'query': 2, 'token_id': 5}, {'query': 3, 'token_id': 6}]], []]}
    behavior_logprob_probe(None, torch.zeros(4, 4), [], graph_id='g', policy={}, limit=None, atol=1e-4,
                           decision_actions=[action], decision_physical_rows=[3, 2, 1, 0])
    assert seen == [([3, 2], [[1, 0]])]
