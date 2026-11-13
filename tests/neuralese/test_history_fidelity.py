"""History fidelity: the next call's prompt extends the previous prompt plus the reply as it was sampled.

A conversation goes through the serving path as an episode does: render the request (`Engine.render_request`), parse the
sampled reply (`build_message`), send the parsed turn back with its tool results as the harness does (ts-host
`native/agent.ts`: the reply's text, reasoning and every tool call as reported), render again. Text level always; id
level through `TokenHistory` (the actor's append-only splice) with the pinned LFM3 tokenizer and chat template.
"""
import json
from pathlib import Path

import pytest

from natlang_neuralese.serve.append_only import TokenHistory
from natlang_neuralese.serve.chat import bind_history_reasoning, build_message

SOURCES = [Path('/home/werg/data/lfm3-rl-dgx-20261029/ops/grpo-v1/run2/actor/sglang-hf'),
           Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd']
ORDER = {'type': 'object', 'properties': {'action': {'type': 'string'}, 'customer': {'type': 'string'},
                                           'pence': {'type': 'integer'}}, 'required': ['action', 'customer', 'pence']}
TOOLS = [{'type': 'function', 'function': {'name': 'eval', 'parameters': {'type': 'object', 'properties': {
    'code': {'type': 'string'}, 'finish': {'type': 'boolean'}}, 'required': ['code']}}},
    {'type': 'function', 'function': {'name': 'return_result', 'parameters': {'type': 'object', 'properties': {
        'status': {'type': 'string'}, 'value': ORDER}}}}]


def call(name, **parameters):
    return ('<tool_call>\n<function=' + name + '>\n' + ''.join(
        f'<parameter={key}>\n{value}\n</parameter>\n' for key, value in parameters.items()) + '</function>\n</tool_call>')


LITERAL = '{ action: "order", customer: "Wen", pence: 280 }'
UNRUN = 'Not run: an earlier call in this reply failed; call it again if you still need it.'
# (the reply as sampled after the generation prompt, the harness's answer to each of its calls, the parsed arguments)
EPISODE = [
    # A name that is not an offered tool, its string argument bare, as models write it.
    ('The notes may say which language.</think>' + call('transcript.search', text='French', limit='5') + '<|im_end|>',
     ['transcript.search: bad-action, transcript.search is not a tool'], [{'text': 'French', 'limit': 5}]),
    # Prose between the thought and the call; a parameter the tool does not declare; a boolean written as a word.
    ('Use code instead.</think>I will search in code.\n\n'
     + call('eval', code='transcript.search("French")', mode='fast', finish='no') + '<|im_end|>',
     ['eval: finish: bad-action, expected a boolean'], [{'code': 'transcript.search("French")', 'mode': 'fast', 'finish': 'no'}]),
    # Two calls in one reply: the first fails, the second is not run and stays.
    ('Two steps.</think>' + call('eval', code='throw new Error("first fails")') + '\n\n'
     + call('eval', code='const total = 280;\ntotal', finish='true') + '<|im_end|>',
     ['Error: first fails', UNRUN], [{'code': 'throw new Error("first fails")'}, {'code': 'const total = 280;\ntotal', 'finish': True}]),
    # A record result written as a JS literal (bare keys): a real return_result call whose value is the sampled text.
    ('Return the order.</think>' + call('return_result', status='success', value=LITERAL) + '<|im_end|>',
     ['rejected\nreturn.pence: type-mismatch'], [{'status': 'success', 'value': LITERAL}]),
    # Strings whose bare text would read as another value come back as JSON text, and still extend.
    ('Quote it.</think>' + call('transcript.entry', id='"7"') + '<|im_end|>',
     ['transcript.entry: bad-action, transcript.entry is not a tool'], [{'id': '7'}]),
]


@pytest.fixture(scope='module')
def engine():
    source = next((path for path in SOURCES if (path / 'tokenizer.json').is_file() and (path / 'chat_template.jinja').is_file()), None)
    if source is None:
        pytest.skip('no pinned LFM3 tokenizer and chat template on this machine')
    from transformers import AutoTokenizer
    from natlang_neuralese.serve.engine import Engine
    engine = object.__new__(Engine)
    engine.tokenizer = bind_history_reasoning(AutoTokenizer.from_pretrained(str(source), local_files_only=True,
                                                                            trust_remote_code=False), model_type='lfm3')
    engine.prompt_rendering = 'full'
    engine.tool_call_format = 'xml'
    return engine


def test_next_prompt_extends_prompt_plus_sampled_reply(engine):
    tokenizer = engine.tokenizer
    history = TokenHistory(tokenizer)
    assert history.enabled
    messages = [{'role': 'system', 'content': 'You run natlang programs.'}, {'role': 'user', 'content': 'Read the order.'}]
    before = sent = None
    for index, (reply, answers, expected) in enumerate(EPISODE + [(None, [], [])]):
        rendered = engine.render_request(messages, TOOLS)
        assert rendered.blocks == []
        prompt = ''.join(rendered.segments)
        ids = history.splice(engine._tokens(prompt))
        if before is not None:
            assert prompt.startswith(before + '\n'), (index, prompt[len(before) - 200:len(before) + 200])
            assert ids[:len(sent)] == sent, index
        if reply is None:
            break
        # The reply's own tokenization is one the plain re-tokenization of the next prompt need not reproduce.
        sampled = tokenizer(reply, add_special_tokens=False)['input_ids']
        history.store(ids, sampled)
        before, sent = prompt + reply, ids + sampled
        message = build_message(reply, [], call_prefix=f'call_{index}', tools=TOOLS)
        calls = message.get('tool_calls') or []
        assert [json.loads(item['function']['arguments']) for item in calls] == expected, index
        assert len(calls) == len(answers), 'every sampled call is a call: none is left as markup'
        # The harness: the reply's text, reasoning and every call as reported, then one answer per call.
        messages = messages + [{'role': 'assistant', 'content': message['content'] or '', 'tool_calls': calls,
                                'reasoning_content': message['reasoning_content']}] + [
            {'role': 'tool', 'tool_call_id': item['id'], 'content': answer} for item, answer in zip(calls, answers)]


def test_a_reply_that_is_no_complete_call_stays_as_sampled(engine):
    """Markup that is no call is the reply's text (the host answers it with what was wrong), and the next prompt
    still extends it."""
    reply = 'Run it.</think>Calling eval.\n<tool_call>\n<function=eval>\n<parameter=code>\n1 + 1\n<|im_end|>'
    messages = [{'role': 'user', 'content': 'Add.'}]
    prompt = ''.join(engine.render_request(messages, TOOLS).segments)
    message = build_message(reply, [], tools=TOOLS)
    assert 'tool_calls' not in message and message['content'].startswith('Calling eval.\n<tool_call>')
    later = ''.join(engine.render_request(messages + [
        {'role': 'assistant', 'content': message['content'], 'reasoning_content': message['reasoning_content']},
        {'role': 'user', 'content': 'The tool call in your last reply left a parameter open or gave one twice, so nothing ran; '
                                    'send the call again, complete.'}], TOOLS).segments)
    # The template closes the turn right after the content; the reply's trailing newline before the end is the one
    # difference a server-side strip makes.
    assert later.startswith(prompt + reply.replace('\n<|im_end|>', '<|im_end|>') + '\n')
