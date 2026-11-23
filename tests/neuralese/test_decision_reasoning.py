"""Reasoned decision readouts (DECISIONS 2026-11-23 "reasoned decisions"): a score item's ``reasoning`` is the reply's
sampled reasoning, and every option is scored as the content after it. On the real LFM3 tokenizer and chat template
(tokenizer only, no model): the options continue ``<think>R</think>``, which is prompt; the item fields echo."""
from pathlib import Path

import pytest

from natlang_neuralese.data.decision_graph import DECISION_SYSTEM_PROMPT, tokenizer_renderer
from natlang_neuralese.serve import native_learning
from natlang_neuralese.serve.chat import RequestError
from natlang_neuralese.serve.native_learning import SCORE_ITEM_FIELDS, decision_items, decision_response

LFM3 = Path.home() / ('.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                      '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
OPTIONS = ['"accept"', '"decline"', '"tentative"']
MESSAGES = [{'role': 'system', 'content': DECISION_SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Function: answerInvite\n\nInstructions: Decide how to answer the invitation.'
                                        '\n\nArguments:\ninvite: "Thu 10:00-12:00"\ncalendar: "Thu 09:00-13:00 dentist"'},
            {'role': 'user', 'content': 'Answer this call now, without tools. Reply with exactly one of these JSON values '
                                        'and nothing else: "accept", "decline", "tentative".'}]
REASONING = 'The calendar is blocked by the dentist from 9 to 13, which covers the meeting, so I decline.'


@pytest.fixture(scope='module')
def renderer():
    if not (LFM3 / 'tokenizer.json').is_file():
        pytest.skip('LFM3-1.7B snapshot not cached')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True, trust_remote_code=False)
    return tokenizer_renderer(tokenizer, model_type='lfm3')


def _text(renderer, items):
    return renderer.tokenizer.decode([int(value) for _, value in items])


def test_reasoning_is_a_declared_score_item_field():
    assert SCORE_ITEM_FIELDS['reasoning'].startswith('served')


def test_reasoning_and_the_think_close_are_prompt(renderer):
    prompt, continuations = decision_items(renderer, MESSAGES, None, OPTIONS, reasoning=REASONING)
    text = _text(renderer, prompt)
    assert text.endswith('<|im_start|>assistant\n<think>' + REASONING + '</think>')
    assert [renderer.tokenizer.decode(ids) for ids in continuations] == ['"accept"<|im_end|>', '"decline"<|im_end|>',
                                                                          '"tentative"<|im_end|>']
    # the reasoned prompt is the generation prompt (ending in <think>) followed by the reasoning and its close
    plain, _ = decision_items(renderer, MESSAGES, None, OPTIONS)
    plain_ids = [int(v) for _, v in plain]
    think = renderer.tokenizer.convert_tokens_to_ids('<think>')
    opening = plain_ids[:plain_ids.index(think) + 1]
    assert [int(v) for _, v in prompt][:len(opening)] == opening


def test_open_reply_steps_keep_the_reasoning(renderer):
    prompt, continuations = decision_items(renderer, MESSAGES, None, ['{"a":"x","', '{"a":"y","'], open_reply=True,
                                           reasoning=REASONING)
    assert _text(renderer, prompt).endswith('<think>' + REASONING + '</think>{"a":"')
    assert [renderer.tokenizer.decode(ids) for ids in continuations] == ['x","', 'y","']


def test_reasoning_must_be_text_and_results_echo_it():
    with pytest.raises(RequestError):
        native_learning.reasoning_text({'reasoning': 3})
    assert native_learning.reasoning_text({}) is None

    class Sequence:
        native_imports, native_rl_policy, native_rl_adapter_binding = [], {}, None
        native_guidance_receipt, native_rl_prompt = None, []
    item = {'reasoning': REASONING, 'native_recording': True}
    response = decision_response(item, Sequence(), [[1], [2]], [(-1.0, 1), (-2.0, 1)])
    assert response['reasoning'] is True and response['native_decision_capture']['reasoning'] == REASONING
    assert 'reasoning' not in decision_response({}, Sequence(), [[1], [2]], [(-1.0, 1), (-2.0, 1)])
