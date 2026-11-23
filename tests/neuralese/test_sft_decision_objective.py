"""The decision objective in expert-iteration SFT (plans/semantic-fp/05 O3): decision readout rows carry their option
set from ``sft_sources build`` (rendered by the scorer's own rendering) and ``exp0_sft`` trains the readout's
option-normalized log score; ``--no-decision-objective`` keeps the plain token CE of the chosen reply."""
import json
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
from natlang_neuralese.data.decision_graph import (decision_label, decision_messages, decision_replies,  # noqa: E402
                                                   tokenizer_renderer)


def test_decision_replies_read_back_every_request_the_runtime_writes():
    values = [True, False, 'red "x", y', None, 3, 'é☃', {'a': [1, 2]}]
    replies = [json.dumps(v, ensure_ascii=False, separators=(',', ':')) for v in values]
    messages, written = decision_messages([{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'args'}],
                                          values)
    assert written == replies and decision_replies(messages) == replies
    judge = [{'role': 'user', 'content': 'Property: the value is short\n\nDoes the value satisfy the property? Reply with '
                                         'exactly one of these JSON values and nothing else: true, false.'}]
    assert decision_replies(judge) == ['true', 'false']
    assert decision_replies([{'role': 'user', 'content': 'hello.'}]) is None
    assert decision_replies([*messages[:-1], {'role': 'assistant', 'content': messages[-1]['content']}]) is None
    assert decision_replies(decision_messages([{'role': 'system', 'content': 's'}], ['only'])[0]) is None
    assert decision_label(replies, replies[2]) == 2
    assert decision_label(['"yes"', '"no"'], '"no"') == 1 and decision_label(['"yes"', '"no"'], 'maybe') is None
    assert decision_label(['1', 'true'], 'true') == 1 and decision_label(['1', 'true'], '1') == 0
    assert decision_label(['{"a":1}', '{"a":2}'], '{"a": 2}') == 1


WORDS = ['t%d' % i for i in range(58)] + ['"yes"', '"no"', '"maybe"', 'yes', 'no', 'maybe']


def _source(path):
    """The tiny LFM source of the serving fixtures with a vocabulary that spells JSON option replies."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast
    path.mkdir()
    config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, layer_types=['conv', 'full_attention'],
                        conv_L_cache=4, max_position_embeddings=256)
    torch.manual_seed(83)
    Lfm2ForCausalLM(config).save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {word: i for i, word in enumerate(WORDS)}, unk_token='t0')), unk_token='t0', bos_token='t1',
        eos_token='t2', pad_token='t3')
    tokenizer.backend_tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] }}: "
                               "{% if message['role'] == 'assistant' and message.get('reasoning_content') and loop.last %}"
                               "{{ message['reasoning_content'] }} {% endif %}{{ message['content'] }} {% endfor %}"
                               "{% if add_generation_prompt %}assistant: {% endif %}")
    tokenizer.save_pretrained(path)


@pytest.fixture
def engine(tmp_path):
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
    from natlang_neuralese.serve import load_native_engine
    _source(tmp_path / 'base')
    model, _, _, config = load_native_model(tmp_path / 'base', device='cpu', dtype=torch.float32, rank=4)
    checkpoint = tmp_path / 'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, checkpoint)
    engine = load_native_engine(str(checkpoint), device='cpu', dtype=torch.float32)
    yield engine
    engine.store.close()


def _decision_turn(label):
    opening = [{'role': 'system', 'content': 't4 t5'}, {'role': 'user', 'content': 't6 t7 t8 t9'}]
    messages, replies = decision_messages(opening, ['yes', 'no', 'maybe'])
    return {'messages': messages, 'tools': None, 'target': {'role': 'assistant', 'content': replies[label]},
            'invocation_id': 'call-1'}, replies


def _render(renderer, turn):
    from natlang_neuralese.serve.targets import target_items
    before, rest = target_items(renderer, turn['messages'], turn['tools'], turn['target'])
    prompt = np.asarray([i for _, i in before], np.int32)
    return prompt, np.asarray([i for _, i in rest], np.int32)


def test_a_decision_row_renders_the_scorers_tokens_and_trains_its_softmax_cross_entropy(engine):
    """The row's option continuations and prompt are the served decision capture's, and the learner's loss on it is
    the served scores' softmax cross-entropy: what ``decide`` reads."""
    from natlang_neuralese.serve.native_learning import decide_native
    renderer = tokenizer_renderer(engine.tokenizer, model_type=engine.backbone.config.model_type,
                                  history_reasoning_policy=engine.model_config.get('history_reasoning_policy'))
    label = 2
    turn, replies = _decision_turn(label)
    served = decide_native(engine, {'messages': turn['messages'], 'options': replies, 'native_recording': True})
    capture = served['native_decision_capture']
    prompt, target = _render(renderer, turn)
    decision = sft_sources.render_decision(renderer, turn, prompt)
    assert decision['label'] == label and decision['values'] == replies
    assert [o.tolist() for o in decision['options']] == capture['continuations']
    assert prompt.tolist() == [i for part in capture['prompt'] for i in part['token_ids']]
    assert target.tolist() == capture['continuations'][label]
    row, _ = sft_sources.rendered_row((prompt, target, 0, decision), 'rl-own', 'k', 0, turn)
    chains = exp0_sft.chains_of([{'task_id': 't', 'round': 'rl:x:g:0', 'rows': [row]}])
    assert len(chains) == 1 and chains[0]['decision']['label'] == label
    assert exp0_sft.mark_decision_objective(chains) == 1 and exp0_sft.chain_mass(chains[0]) == exp0_sft.DECISION_MASS
    hf = engine.shared_model.port.hf
    with torch.no_grad():
        total, computed, report = exp0_sft.decision_backward(hf, exp0_sft.hidden, chains, 1., 0, 0., fast=False,
                                                             backward=False)
    expected = float(-torch.log_softmax(torch.tensor(served['log_probs'], dtype=torch.float64), 0)[label])
    assert report['decisions'] == 1 and abs(report['ce'] - expected) < 1e-4 and abs(total - expected) < 1e-4
    assert report['correct'] == int(int(np.argmax(served['log_probs'])) == label)
    assert computed == sum(len(prompt) + len(o) for o in decision['options'])


def test_calls_that_are_not_decisions_or_answer_no_option_stay_plain_rows(engine):
    renderer = tokenizer_renderer(engine.tokenizer, model_type=engine.backbone.config.model_type)
    turn, replies = _decision_turn(0)
    prompt, _ = _render(renderer, turn)
    assert sft_sources.render_decision(renderer, {**turn, 'messages': turn['messages'][:-1]}, prompt) is None
    assert sft_sources.render_decision(renderer, {**turn, 'tools': [{'function': {'name': 'eval'}}]}, prompt) is None
    off = {**turn, 'target': {'role': 'assistant', 'content': '"perhaps"'}}
    assert sft_sources.render_decision(renderer, off, prompt) == {'error': 'reply-not-an-option'}
    assert sft_sources.render_decision(renderer, turn, prompt[1:]) == {'error': 'prompt-differs-from-scorer'}
    row, _ = sft_sources.rendered_row((prompt, prompt[:2], 0, {'error': 'reply-not-an-option'}), 'rl-own', 'k', 0, off)
    assert 'decision' not in row and row['decision_error'] == 'reply-not-an-option'


def test_build_rendering_carries_decision_options_into_chains_and_the_mix_report(engine, tmp_path):
    """``render_candidate`` (the build's rendering) attaches the option set; the plain row is kept for the opt-out."""
    renderer = tokenizer_renderer(engine.tokenizer, model_type=engine.backbone.config.model_type)
    renderer.render = sft_sources.TemplateRenderer.render.__get__(renderer)
    turn, replies = _decision_turn(1)
    plain = {'messages': [{'role': 'system', 'content': 't4'}, {'role': 'user', 'content': 't10 t11'}], 'tools': None,
             'target': {'role': 'assistant', 'content': 't12 t13'}, 'invocation_id': 'call-0'}
    replay = tmp_path / 'work' / 'replay' / 'e1'
    replay.mkdir(parents=True)
    (replay / 'rollouts.jsonl').write_text(json.dumps({'turns': [plain, turn]}) + '\n')
    c = {'key': 'rl:x:g:0', 'source': 'rl-own', 'name': 'e1'}
    turns, calls = sft_sources.render_candidate(renderer, tmp_path / 'work', c, 'context', 'context', 'context')
    assert all(len(call) == 4 for call in calls) and calls[0][3] is None and calls[1][3]['label'] == 1
    rows = [sft_sources.rendered_row(call, 'rl-own', c['key'], i, t)[0] for i, (t, call) in enumerate(zip(turns, calls))]
    assert 'decision' not in rows[0] and len(rows[1]['decision']['options']) == 3
    chains = exp0_sft.chains_of([{'task_id': 't', 'round': c['key'], 'rows': rows}])
    for chain in chains:
        chain['part'] = 'rl-own'
    report = sft_sources.decision_report(chains)
    assert report['chains'] == 1 and report['parts']['rl-own']['options'] == 3
    # three-tuples (renders without decisions: call_row, grpo's teacher replay, filled reasoning) keep their rows
    assert 'decision' not in sft_sources.call_row(renderer, 'rl-own', 'k', 1, turn)[0]


class Tiny(torch.nn.Module):
    def __init__(self, vocab=11, width=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(vocab, width, generator=g))
        self.mix = torch.nn.Parameter(torch.randn(width, width, generator=g) / width)
        self.lm_head = torch.nn.Linear(width, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, width, generator=g))


def tiny_forward(model, ids, lengths=None):
    """Causal per document (packed documents never see each other)."""
    parts = ids.split(list(lengths)) if lengths is not None else [ids]
    out = []
    for part in parts:
        x = model.emb[part]
        out.append(x + torch.tanh(torch.cumsum(x, 0) @ model.mix))
    return torch.cat(out)


def decision_chain(prompt, options, label=None, probabilities=None, round='rl:x'):
    decision = {'options': [np.asarray(o, np.int32) for o in options], 'label_source': 'verified-trajectory',
                **({'label': label} if label is not None else {'probabilities': probabilities})}
    chosen = label if label is not None else int(np.argmax(probabilities))
    ids = np.asarray(prompt + options[chosen], np.int32)
    row = {'sample': ids[len(prompt):], 'decision': decision}
    return {'ids': ids, 'segs': [(len(prompt), len(options[chosen]), row)], 'round': round, 'task': 't',
            'decision': decision}


def explicit_scores(model, prompt, options):
    """The served law: each option's continuation log-probability, token by token after the shared prompt."""
    scores = []
    for option in options:
        ids = torch.tensor(prompt + option)
        logp = torch.log_softmax(tiny_forward(model, ids) @ model.lm_head.weight.T, -1)
        scores.append(sum(logp[len(prompt) - 1 + j, token] for j, token in enumerate(option)))
    return torch.stack(scores)


@pytest.mark.parametrize('fast', [True, False])
def test_decision_backward_is_the_option_softmax_cross_entropy_with_its_gradient(fast):
    model = Tiny()
    chains = [decision_chain([1, 2, 3], [[4, 5], [6], [7, 8, 9]], label=2),
              decision_chain([2, 3], [[4], [5, 6]], probabilities=[0.25, 0.75])]
    exp0_sft.mark_decision_objective(chains)
    expected = (decision_loss(explicit_scores(model, [1, 2, 3], [[4, 5], [6], [7, 8, 9]]), rule='logLoss',
                              probabilities=[0., 0., 1.])
                + decision_loss(explicit_scores(model, [2, 3], [[4], [5, 6]]), rule='logLoss',
                                probabilities=[0.25, 0.75])) / 7.
    wanted = torch.autograd.grad(expected, list(model.parameters()))
    total, computed, report = exp0_sft.decision_backward(model, tiny_forward, chains, 7., 64 if fast else 0, 0.,
                                                         fast=fast)
    assert abs(total / 7. - float(expected.detach())) < 1e-5 and report['decisions'] == 2
    assert computed == 3 * 3 + 2 + 1 + 3 + 2 * 2 + 1 + 2
    for parameter, grad in zip(model.parameters(), wanted):
        torch.testing.assert_close(parameter.grad, grad, atol=1e-6, rtol=1e-5)


def test_a_wrongly_preferred_option_is_lowered_and_the_token_ce_alone_does_not():
    """The objective's gradient lowers a competing option; the plain token CE of the chosen reply is the opt-out."""
    model = Tiny(seed=3)
    prompt, options = [1, 2], [[4], [5]]
    chain = decision_chain(prompt, options, label=0)
    exp0_sft.mark_decision_objective([chain])
    exp0_sft.decision_backward(model, tiny_forward, [chain], 1., 0, 0., fast=False)
    step = {name: -0.5 * p.grad for name, p in model.named_parameters()}
    with torch.no_grad():
        before = explicit_scores(model, prompt, options)
        for name, p in model.named_parameters():
            p.add_(step[name])
        after = explicit_scores(model, prompt, options)
    assert after[0] - after[1] > before[0] - before[1]
    plain = decision_chain(prompt, options, label=0)
    assert exp0_sft.mark_decision_objective([plain], on=False) == 1
    assert exp0_sft.chain_mass(plain) == 1. and not plain['decision_objective']   # its one target token


def test_lowered_graph_sources_keep_decisions_as_decision_chains():
    import collections
    import lower_sft_sources
    row = {'id': 'r', 'nodes': [{'name': f'decision:option:{i}', 'token_ids': [5, 6, 7] + cont,
                                 'target_weights': [0.] * (3 + len(cont))} for i, cont in enumerate([[8], [9, 10]])],
           'decisions': [{'name': 'decision', 'options': ['decision:option:0', 'decision:option:1'], 'prompt_tokens': 3,
                          'label': 1, 'label_source': 'exact-oracle', 'values': ['"a"', '"b"']}]}
    stats = collections.Counter()
    sequences = lower_sft_sources._decision_sequences(row, stats)
    assert stats['decisions'] == 1 and len(sequences) == 1
    ids, weights, names, decision = sequences[0]
    chain = lower_sft_sources.to_chain({'id': 'r', 'family': 'f', 'ids': ids, 'w': weights, 'decision': decision})
    assert chain['ids'].tolist() == [5, 6, 7, 9, 10] and chain['segs'][0][0] == 3
    assert [o.tolist() for o in chain['decision']['options']] == [[8], [9, 10]] and chain['decision']['label'] == 1
    prompt, options, target = exp0_sft.decision_parts(chain)
    assert prompt.tolist() == [5, 6, 7] and target == [0., 1.]
    imported = {**row, 'nodes': [{**node, 'imports': [{'at': 0, 'source': 'x', 'start': 0, 'stop': 1}]}
                                 for node in row['nodes']]}
    assert lower_sft_sources._decision_sequences(imported, stats) == [] and stats['skip_decisions_neuralese'] == 1
