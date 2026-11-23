"""Field-mode decision readouts (semantic-FP B10, plans/semantic-fp/02 §4 R3): ``open_reply`` score items and the FORM
request, on the real LFM3 tokenizer and chat template (tokenizer only, no model).

An open item scores each continuation as the start of the reply, without the end of message: the template's rendering
of the final assistant message cut at the end of its content (the fork's ``continue_final_message: "content"``).
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from natlang_neuralese.data.decision_graph import (DECISION_SYSTEM_PROMPT, decision_forms, decision_replies,
                                                   field_candidates, field_path, tokenizer_renderer)
from natlang_neuralese.serve import native_learning
from natlang_neuralese.serve.native_learning import SCORE_ITEM_FIELDS, decision_items

LFM3 = Path.home() / ('.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                      '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')

FORM = ('{"label":<"spam" | "ham" | "other">,"urgency":<"low" | "mid" | "high">,'
        '"route":<"billing" | "tech" | "sales" | "abuse">}')
FIELD_REQUEST = ('Answer this call now, without tools. Reply with exactly one JSON value of this form and nothing else: '
                 + FORM + '.')
UNION_REQUEST = ('Answer this call now, without tools. Reply with exactly one JSON value of one of these forms and '
                 'nothing else: {"kind":"refund","amount":<10 | 50 | 100>,"approved":<true | false>}, '
                 '{"kind":"escalate","team":<"billing" | "legal">}, "none".')


def _messages(request):
    return [{'role': 'system', 'content': DECISION_SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Classify the support message.\n\nmessage: "I was charged twice, fix it now!"'},
            {'role': 'user', 'content': request}]


@pytest.fixture(scope='module')
def tokenizer():
    if not (LFM3 / 'tokenizer.json').is_file():
        pytest.skip('LFM3-1.7B snapshot not cached')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True, trust_remote_code=False)


@pytest.fixture(scope='module')
def renderer(tokenizer):
    return tokenizer_renderer(tokenizer, model_type='lfm3')


def test_open_reply_is_a_declared_score_item_field():
    assert SCORE_ITEM_FIELDS['open_reply'].startswith('served')


def test_forms_parse_the_field_request_and_steps_follow_the_contract():
    forms = decision_forms(_messages(FIELD_REQUEST))
    assert forms == [('record', [('"label"', ['"spam"', '"ham"', '"other"']),
                                 ('"urgency"', ['"low"', '"mid"', '"high"']),
                                 ('"route"', ['"billing"', '"tech"', '"sales"', '"abuse"'])])]
    assert decision_replies(_messages(FIELD_REQUEST)) is None
    assert [t for t, _ in field_candidates(forms, [])] == ['{"label":"spam","', '{"label":"ham","', '{"label":"other","']
    assert [t for t, _ in field_candidates(forms, ['{"label":"spam","'])][0] == '{"label":"spam","urgency":"low","'
    last = field_candidates(forms, ['{"label":"spam","', '{"label":"spam","urgency":"low","'])
    assert last[0] == ('{"label":"spam","urgency":"low","route":"billing"}', True) and len(last) == 4
    steps = field_path(forms, '{"urgency": "high", "label": "ham", "route": "tech"}')   # matched by value
    assert [(len(c), c[label]) for c, label in steps] == [
        (3, '{"label":"ham","'), (3, '{"label":"ham","urgency":"high","'),
        (4, '{"label":"ham","urgency":"high","route":"tech"}')]
    assert field_path(forms, '{"label":"ham","urgency":"high"}') is None
    assert field_path(forms, '{"label":"spam","urgency":"low","route":"nowhere"}') is None
    union = decision_forms(_messages(UNION_REQUEST))
    assert [f[0] for f in union] == ['record', 'record', 'value'] and union[2] == ('value', '"none"')
    assert [t for t, _ in field_candidates(union, [])] == ['{"kind":"refund","', '{"kind":"escalate","', '"none"']
    assert field_path(union, '"none"') == [(['{"kind":"refund","', '{"kind":"escalate","', '"none"'], 2)]
    steps = field_path(union, '{"kind":"refund","amount":50,"approved":false}')
    assert [c[label] for c, label in steps] == ['{"kind":"refund","', '{"kind":"refund","amount":50,"',
                                                '{"kind":"refund","amount":50,"approved":false}']
    # a field with one value is written bare; a record with no fields is a whole value (native/decision.ts armForm)
    bare = decision_forms(_messages(UNION_REQUEST.replace('"none"', '{}').replace('{"kind":"escalate",',
                                                                                    '{"kind":"escalate","tier":2,')))
    assert bare[1] == ('record', [('"kind"', ['"escalate"']), ('"tier"', ['2']), ('"team"', ['"billing"', '"legal"'])])
    assert bare[2] == ('value', '{}')
    # a one-form marker with several forms (or the reverse) is not the runtime's request
    assert decision_forms(_messages(FIELD_REQUEST.replace(FORM, FORM + ', "none"'))) is None


def test_open_continuations_are_the_templates_continue_final_message_rendering(tokenizer, renderer):
    """Open = closed minus its final <|im_end|>, and prompt + open equals the template's own continue_final_message
    rendering (the empty think block included)."""
    messages = _messages(FIELD_REQUEST)
    options = ['{"label":"spam","', '{"label":"ham","', '{"label":"other","']
    prompt, closed = decision_items(renderer, messages, None, options)
    again, opened = decision_items(renderer, messages, None, options, open_reply=True)
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert again == prompt and all(c[-1] == end and o == c[:-1] for c, o in zip(closed, opened))
    ids = [i for _, i in prompt]
    for option, own in zip(options, opened):
        text = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': option}], tokenize=False,
                                             continue_final_message=True)
        assert text.endswith('<think></think>' + option)
        assert tokenizer(text, add_special_tokens=False)['input_ids'] == ids + own


@pytest.mark.parametrize('request_text,reply', [
    (FIELD_REQUEST, '{"label":"spam","urgency":"low","route":"billing"}'),
    (FIELD_REQUEST, '{"label":"other","urgency":"high","route":"abuse"}'),
    (UNION_REQUEST, '{"kind":"refund","amount":100,"approved":true}'),
    (UNION_REQUEST, '{"kind":"escalate","team":"legal"}'),
])
def test_each_steps_open_tokens_are_a_prefix_of_the_whole_replys_tokens(renderer, request_text, reply):
    """The cut before the next key name lands on a token boundary: every step's open continuation of the reply's path
    is a token prefix of the reply's closed continuation (punctuation runs such as '","' merge into one token)."""
    messages = _messages(request_text)
    _, (whole,) = decision_items(renderer, messages, None, [reply])
    for candidates, label in field_path(decision_forms(messages), reply):
        _, opened = decision_items(renderer, messages, None, candidates, open_reply=True)
        assert whole[:len(opened[label])] == opened[label], candidates[label]


def test_an_open_item_scores_open_continuations_and_its_result_and_capture_echo_the_flag(renderer):
    """``_score_item`` (and ``decide_native``, which rebuilds the item) hands the engine open continuations; the result
    and the native decision capture say open_reply: true; a closed item says nothing; a non-boolean is refused."""
    seen = []

    def sglang_score(item, plan, continuations, bindings):
        seen.append(continuations)
        sequence = SimpleNamespace(native_imports=[], native_rl_policy={}, native_rl_adapter_binding=None,
                                   native_guidance_receipt=None, native_rl_prompt=plan)
        return native_learning.decision_response(item, sequence, continuations,
                                                 [(-1.0, len(c)) for c in continuations])
    engine = SimpleNamespace(**{k: getattr(renderer, k) for k in ('tokenizer', 'prompt_rendering', 'tool_call_format')},
                             sglang_score=sglang_score)
    for name in ('_template', 'render_request', '_tokens', '_template_tokens', 'specials'):
        setattr(engine, name, getattr(renderer, name))
    messages = _messages(FIELD_REQUEST)
    options = ['{"label":"spam","', '{"label":"ham","']
    _, opened = decision_items(renderer, messages, None, options, open_reply=True)
    _, closed = decision_items(renderer, messages, None, options)
    result = native_learning.decide_native(engine, {'messages': messages, 'options': options, 'open_reply': True,
                                                    'native_recording': True})
    assert seen[-1] == opened and result['open_reply'] is True
    assert result['native_decision_capture']['open_reply'] is True
    assert result['native_decision_capture']['continuations'] == opened
    many = native_learning.decide_many_native(engine, {'items': [
        {'messages': messages, 'continuations': options, 'open_reply': True},
        {'messages': messages, 'continuations': options}]})['results']
    assert many[0]['open_reply'] is True and 'open_reply' not in many[1]
    assert sorted(map(json.dumps, seen[-2:])) == sorted(map(json.dumps, [opened, closed]))
    bad = native_learning.decide_many_native(engine, {'items': [{'messages': messages, 'continuations': options,
                                                                 'open_reply': 'yes'}]})['results'][0]
    assert bad['error'].startswith('neuralese-decision') and 'open_reply' in bad['error']
