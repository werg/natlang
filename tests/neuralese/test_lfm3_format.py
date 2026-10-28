"""LFM3-1.7B chat format on the shared rendering/parsing paths (plans/neuralese/LFM3_PORT_20261028.md, part B).

Real round trips through the pinned LFM3 tokenizer and chat template: XML tool calls render -> tokenize -> decode ->
parse back; Neuralese write-site boundaries on unquoted XML values; producer spans keep the complete response; the
declared history-reasoning policy is prefix-stable; the format is chosen by model_type, never by vocabulary.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.serve.chat import (
    BACKBONE_HISTORY_REASONING, RequestError, XML_PARAMETER_CLOSE, bind_history_reasoning, build_message,
    call_reply, check_xml_call, parse_xml_calls, render_messages, scan_xml_calls, tool_call_format, write_reply,
    write_value_text, xml_value, xml_value_representable, xml_value_text,
)

LFM3 = Path.home() / ('.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                      '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
TOKENIZER_SHA256 = '7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701'
TEMPLATE_SHA256 = 'f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818'
needs_lfm3 = pytest.mark.skipif(not (LFM3 / 'tokenizer.json').is_file(), reason='LFM3-1.7B snapshot not cached')

EVAL_TOOLS = [{'type': 'function', 'function': {'name': 'eval', 'parameters': {'type': 'object', 'properties': {
    'code': {'type': 'string'}, 'finish': {'type': 'boolean'}}, 'required': ['code']}}},
    {'type': 'function', 'function': {'name': 'return_result', 'parameters': {'type': 'object', 'properties': {
        'status': {'type': 'string'}, 'value': {}}}}}]
CODES = [
    'const x = 1;\nreturn x;',
    "const s = 'single' + \"double\" + `tpl ${x}`;\nreturn s.split('\\n');",
    'const html = "<parameter=code>\\n</parameter>";  // a tag-like string, not followed by the delimiter',
    'return "</parameter>" + "</function>" + "</tool_call>";',
    'const t = "<tool_call>\\n<function=eval>\\n";\nreturn t;',
    'emoji = "α β 🙂"; tab = "\t"; cr = "\r";',
    '',
    '\n\nleading and trailing newlines\n',
    'a tag line inside\n</parameter>x\n',
    '<|tool_call_start|> and <|pad|> are content here',  # (generated <|im_end|> ends a turn; not content)
    '42',
    'true',
    '[1, 2]',
]


@pytest.fixture(scope='module')
def tokenizer():
    if not (LFM3 / 'tokenizer.json').is_file():
        pytest.skip('LFM3-1.7B snapshot not cached')
    from transformers import AutoTokenizer
    assert hashlib.sha256((LFM3 / 'tokenizer.json').read_bytes()).hexdigest() == TOKENIZER_SHA256
    assert hashlib.sha256((LFM3 / 'chat_template.jinja').read_bytes()).hexdigest() == TEMPLATE_SHA256
    tok = AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True, trust_remote_code=False)
    return bind_history_reasoning(tok, require_declared=True)


def _apply(tok, tools=None):
    return lambda messages, generation: tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=generation)


def _model_output(tok, message, tools=None, opening=None):
    """What the model generates for `message` after the serving prompt, as token IDs decoded back to text."""
    opening = opening or [{'role': 'user', 'content': 'task'}]
    prompt = tok.apply_chat_template(opening, tools=tools, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(opening + [message], tools=tools, tokenize=False)
    assert full.startswith(prompt)
    ids = tok(full[len(prompt):], add_special_tokens=False).input_ids
    return tok.decode(ids, skip_special_tokens=False)


def test_declarations():
    assert tool_call_format('lfm3') == 'xml' and tool_call_format('qwen3_5') == 'xml'
    assert tool_call_format('lfm2') == 'pythonic' and tool_call_format('mellum') == 'json'
    assert tool_call_format('unknown') is None and tool_call_format(None) is None
    assert BACKBONE_HISTORY_REASONING['lfm3'] == 'keep'


@needs_lfm3
def test_format_is_declared_not_sniffed(tokenizer):
    from types import SimpleNamespace
    from natlang_neuralese.data.render import Renderer
    assert '<|tool_call_start|>' in tokenizer.get_vocab()  # present but unused by LFM3
    renderer = Renderer(tokenizer, SimpleNamespace(open_id=1, close_id=2))
    sp = renderer.special
    assert sp.call_format == 'xml'
    assert sp.bos == tokenizer.convert_tokens_to_ids('<|startoftext|>')
    assert sp.tool_call_start == tokenizer.convert_tokens_to_ids('<tool_call>')
    messages = [{'role': 'system', 'content': 'Be brief.'}, {'role': 'user', 'content': 'Add 2 and 3.'}]
    expected = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '</think>'
    assert renderer.chat(messages, generation_prompt=True) == tokenizer(expected, add_special_tokens=False).input_ids
    history = messages + [{'role': 'assistant', 'content': '5'}, {'role': 'user', 'content': 'Now 4 and 4.'}]
    assert renderer.chat(history) == tokenizer(tokenizer.apply_chat_template(history, tokenize=False),
                                               add_special_tokens=False).input_ids
    with pytest.raises(ValueError, match='declares no tool-call format'):
        Renderer(tokenizer, SimpleNamespace(open_id=1, close_id=2), model_type='not-a-backbone')


@needs_lfm3
@pytest.mark.parametrize('code', CODES)
def test_xml_call_round_trips_through_real_tokens(tokenizer, code):
    arguments = {'code': code, 'finish': True}
    message = {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'eval', 'arguments': arguments}}]}
    check_xml_call('eval', arguments, EVAL_TOOLS)
    out = _model_output(tokenizer, message, EVAL_TOOLS)
    assert out.startswith('</think><tool_call>\n<function=eval>\n<parameter=code>\n')
    parsed = build_message(out, [], tools=EVAL_TOOLS)
    assert parsed['content'] is None and 'reasoning_content' not in parsed
    (call,) = parsed['tool_calls']
    assert call['function']['name'] == 'eval'
    assert json.loads(call['function']['arguments']) == arguments


@needs_lfm3
def test_typed_values_reasoning_and_several_calls(tokenizer):
    value = {'a': 'é', 'b': [1, None, 2.5], 'c': {'d': False}}
    message = {'role': 'assistant', 'reasoning_content': 'think about it', 'content': 'Calling.', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'return_result', 'arguments': {'status': 'success', 'value': value}}},
        {'type': 'function', 'function': {'name': 'eval', 'arguments': {'code': 'return 1;'}}}]}
    out = _model_output(tokenizer, message, EVAL_TOOLS)
    parsed = build_message(out, [], tools=EVAL_TOOLS)
    assert parsed['reasoning_content'] == 'think about it' and parsed['content'] == 'Calling.'
    assert [json.loads(c['function']['arguments']) for c in parsed['tool_calls']] == [
        {'status': 'success', 'value': value}, {'code': 'return 1;'}]
    # Undeclared types: only the template's canonical JSON rendering reads as JSON.
    assert xml_value('{"a": 1}') == {'a': 1} and xml_value('{"a":1}') == '{"a":1}'
    assert xml_value('"quoted"') == '"quoted"' and xml_value('42') == 42 and xml_value('42', {'type': 'string'}) == '42'
    with pytest.raises(ValueError):
        xml_value('not json', {'type': 'integer'})


@needs_lfm3
def test_unrepresentable_values_fail_loudly(tokenizer):
    bad = 'line\n</parameter>\nmore'
    assert not xml_value_representable(bad) and xml_value_representable('x</parameter>y')
    assert not xml_value_representable('ends with\n</parameter>')
    with pytest.raises(RequestError, match='delimiter'):
        check_xml_call('eval', {'code': bad})
    with pytest.raises(RequestError, match='keep its type'):
        check_xml_call('eval', {'code': '42'})  # an undeclared string that renders like a number
    check_xml_call('eval', {'code': '42'}, EVAL_TOOLS)  # declared string: exact
    check_xml_call('eval', {'code': '42'}, types=False)  # prompts check structure only
    with pytest.raises(RequestError):
        check_xml_call('bad>name', {})
    messages = [{'role': 'user', 'content': 'x'}, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'eval', 'arguments': json.dumps({'code': bad})}}]},
                {'role': 'tool', 'content': 'ok'}]
    with pytest.raises(RequestError, match='delimiter'):
        render_messages(messages, None, lambda m, t: tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True), call_format='xml')
    from natlang_neuralese.data.text_corpus import native_gold_packet
    with pytest.raises(RequestError, match='delimiter'):
        native_gold_packet(tokenizer, messages[:1], messages[1], EVAL_TOOLS)


@needs_lfm3
def test_write_site_boundaries_on_unquoted_values(tokenizer):
    apply = _apply(tokenizer, EVAL_TOOLS)
    prefix, suffix = call_reply(apply, 'eval', {}, 'code', call_format='xml')
    assert prefix == '</think><tool_call>\n<function=eval>\n<parameter=code>\n'
    assert suffix == '\n</parameter>\n</function>\n</tool_call><|im_end|>\n'
    assert call_reply(apply, 'eval', {}, 'code', quoted=False, call_format='xml') == (prefix, '')
    with pytest.raises(RequestError, match='writes XML'):
        call_reply(apply, 'eval', {}, 'code', call_format='pythonic')
    # String and unknown write sites share the parameter's own lines; siblings keep their order.
    p, s = write_reply(apply, 'return_result', {'status': 'success'}, 'value', 'string', call_format='xml')
    assert p.endswith('<parameter=status>\nsuccess\n</parameter>\n<parameter=value>\n') and s.startswith(XML_PARAMETER_CLOSE)
    assert write_reply(apply, 'return_result', {'status': 'success'}, 'value', 'unknown', call_format='xml') == (p, s)
    # Gold value spans are the template's own rendering.
    assert write_value_text(apply, 'eval', {}, 'code', 'a = "b";\nc', call_format='xml') == 'a = "b";\nc'
    obj = {'k': [1, True], 'n': None}
    assert write_value_text(apply, 'return_result', {'status': 'success'}, 'value', obj, 'unknown',
                            call_format='xml') == '{"k": [1, true], "n": null}'
    with pytest.raises(RequestError, match='delimiter'):
        write_value_text(apply, 'eval', {}, 'code', 'x\n</parameter>\ny', call_format='xml')
    with pytest.raises(RequestError, match='keep its type'):
        write_value_text(apply, 'return_result', {'status': 'success'}, 'value', '42', 'unknown', call_format='xml')
    # A value nested inside a JSON-valued parameter stays a quoted JSON string.
    nested_p, nested_s = write_reply(apply, 'return_result', {'status': 'success', 'value': {'a': 'x', 'b': 1}},
                                     'value', 'string', argument_path=('value', 'a'), call_format='xml')
    assert nested_p.endswith('<parameter=value>\n{"a": "') and nested_s.startswith('", "b": 1}\n</parameter>\n')


@needs_lfm3
def test_written_block_parses_as_native_state(tokenizer):
    from natlang_neuralese.serve.chat import placeholder
    apply = _apply(tokenizer, EVAL_TOOLS)
    prefix, suffix = write_reply(apply, 'return_result', {'status': 'success'}, 'value', 'unknown', call_format='xml')
    block = 'nzs1_' + 'a' * 52
    message = build_message(prefix + placeholder(0) + suffix, [block], tools=EVAL_TOOLS)
    (call,) = message['tool_calls']
    assert json.loads(call['function']['arguments']) == {
        'status': 'success', 'value': [{'type': 'native_state', 'id': block}]}
    code_prefix, code_suffix = write_reply(apply, 'eval', {}, 'code', 'string', call_format='xml')
    message = build_message(code_prefix + 'const v = ' + placeholder(0) + ';' + code_suffix, [block], tools=EVAL_TOOLS)
    assert json.loads(message['tool_calls'][0]['function']['arguments'])['code'] == [
        {'type': 'text', 'text': 'const v = '}, {'type': 'native_state', 'id': block}, {'type': 'text', 'text': ';'}]


@needs_lfm3
def test_native_state_imports_render_at_exact_positions(tokenizer):
    block = 'nzs1_' + 'b' * 52
    messages = [{'role': 'user', 'content': 'go'}, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'eval', 'arguments': {'code': [
            {'type': 'text', 'text': 'const v = '}, {'type': 'native_state', 'id': block},
            {'type': 'text', 'text': ';'}]}}}]}, {'role': 'tool', 'content': 'done'}]
    rendered = render_messages(messages, EVAL_TOOLS, lambda m, t: tokenizer.apply_chat_template(
        m, tools=t, tokenize=False, add_generation_prompt=True), call_format='xml')
    assert rendered.blocks == [block]
    at = rendered.segments.index(0)
    assert rendered.segments[at - 1].endswith('<parameter=code>\nconst v = ')
    assert rendered.segments[at + 1].startswith(';\n</parameter>\n</function>\n</tool_call>')


def _span_packet(tokenizer, target, span):
    from natlang_neuralese.data.text_corpus import native_gold_packet
    messages = [{'role': 'user', 'content': 'run the program'}]
    baseline = native_gold_packet(tokenizer, messages, copy.deepcopy(target), EVAL_TOOLS)[0]
    marked = native_gold_packet(tokenizer, messages, copy.deepcopy(target), EVAL_TOOLS, producer_span=span,
                                return_producer_span=True)
    return baseline, marked


@needs_lfm3
@pytest.mark.parametrize('value_type,body,value', [
    ('string', 'a compact summary\nwith "quotes"', 'a compact summary\nwith "quotes"'),
    ('unknown', '{"ids":["T2","T3"],"n":2}', {'ids': ['T2', 'T3'], 'n': 2}),
    ('unknown', '"plain text value"', 'plain text value'),
])
def test_producer_span_keeps_the_complete_response(tokenizer, value_type, body, value):
    from natlang_neuralese.model.shared_state import ROLE_MARKERS
    target = {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {
        'name': 'return_result', 'arguments': json.dumps({'status': 'success', 'value': value})}}]}
    span = {'call_index': 0, 'argument_path': ['value'], 'body': body, 'value_type': value_type,
            'payload_sha256': hashlib.sha256(body.encode()).hexdigest()}
    baseline, (text, ids, boundary, _) = _span_packet(tokenizer, target, span)
    opening = '<|im_start|>assistant\n<think>'
    reply = text[text.rindex(opening) + len(opening):]
    rendered_value = xml_value_text(value)
    assert ROLE_MARKERS[0] + rendered_value + ROLE_MARKERS[1] in reply
    # Removing only the role markers gives the template's own rendering of the complete response.
    assert text.replace(ROLE_MARKERS[0], '').replace(ROLE_MARKERS[1], '') == baseline
    clean = reply.replace(ROLE_MARKERS[0], '').replace(ROLE_MARKERS[1], '')
    (call,) = build_message(clean, [], tools=EVAL_TOOLS)['tool_calls']
    assert json.loads(call['function']['arguments']) == {'status': 'success', 'value': value}


@needs_lfm3
def test_inline_producer_span_inside_eval_code(tokenizer):
    from natlang_neuralese.model.shared_state import ROLE_MARKERS
    body = 'judge each ticket\'s churn "signal"\nseparately'
    prefix, suffix = 'const verdict = await nl<string>`', '`(ticket);\nreturn verdict;'
    code = prefix + body + suffix
    target = {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'id': 'c', 'function': {
        'name': 'eval', 'arguments': json.dumps({'code': code})}}]}
    span = {'call_index': 0, 'argument_path': ['code'], 'body': body, 'value_type': 'inline-string',
            'payload_sha256': hashlib.sha256(body.encode()).hexdigest(), 'source_start': len(prefix),
            'source_end': len(prefix) + len(body), 'sidecar_code_sha256': hashlib.sha256(code.encode()).hexdigest()}
    baseline, (text, _, _, binding) = _span_packet(tokenizer, target, span)
    assert binding['rendered_body'] == body  # XML writes code verbatim: no escaping inside the span
    assert ROLE_MARKERS[0] + body + ROLE_MARKERS[1] in text
    assert text.replace(ROLE_MARKERS[0], '').replace(ROLE_MARKERS[1], '') == baseline


@needs_lfm3
def test_history_reasoning_policy_is_prefix_stable(tokenizer):
    assert tokenizer.natlang_history_reasoning == {'policy': 'keep', 'backbone': 'lfm3',
                                                   'template_kwargs': {'preserve_thinking': True}, 'probe': 'keep'}
    call = {'type': 'function', 'function': {'name': 'eval', 'arguments': {'code': 'return 1;'}}}
    conversation = [
        {'role': 'system', 'content': 'You run natlang programs.'},
        {'role': 'user', 'content': 'first task'},
        {'role': 'assistant', 'reasoning_content': 'R1', 'content': '', 'tool_calls': [call]},
        {'role': 'tool', 'content': '1'},
        {'role': 'assistant', 'content': 'one'},  # no reasoning: renders <think></think>
        {'role': 'user', 'content': 'second task'},
        {'role': 'assistant', 'reasoning_content': 'R2', 'content': '', 'tool_calls': [call]},
        {'role': 'tool', 'content': '1'},
        {'role': 'assistant', 'reasoning_content': 'R3', 'content': 'two'},
    ]

    def ids(text):
        return tokenizer(text, add_special_tokens=False).input_ids

    full = ids(tokenizer.apply_chat_template(conversation, tools=EVAL_TOOLS, tokenize=False))
    for end in range(1, len(conversation) + 1):
        before = tokenizer.apply_chat_template(conversation[:end], tools=EVAL_TOOLS, tokenize=False)
        assert full[:len(ids(before))] == ids(before), end
        if end < len(conversation) and conversation[end]['role'] == 'assistant':
            prompt = ids(tokenizer.apply_chat_template(conversation[:end], tools=EVAL_TOOLS, tokenize=False,
                                                       add_generation_prompt=True))
            assert full[:len(prompt)] == prompt, end
    text = tokenizer.decode(full)
    assert text.count('<think>R1</think>') == 1 and '<think></think>one' in text
    # The template's own default (no preserve_thinking) rewrites the first exchange once a second user message
    # arrives: that is why LFM3 declares `keep`.
    original = tokenizer._natlang_history_reasoning_original_apply_chat_template
    first = original(conversation[:5], tools=EVAL_TOOLS, tokenize=False)
    later = original(conversation, tools=EVAL_TOOLS, tokenize=False)
    assert '<think>R1</think>' in first and not later.startswith(first)
    with pytest.raises(ValueError, match='pins chat-template'):
        tokenizer.apply_chat_template(conversation, tokenize=False, preserve_thinking=False)


@needs_lfm3
def test_context_only_history_spans_on_lfm3(tokenizer):
    from natlang_neuralese.data.text_corpus import native_gold_packet_context_only_messages
    call = {'type': 'function', 'function': {'name': 'eval', 'arguments': json.dumps({'code': 'return 1;'})}}
    messages = [{'role': 'user', 'content': 'task'},
                {'role': 'assistant', 'reasoning_content': 'first', 'content': '', 'tool_calls': [call]},
                {'role': 'tool', 'content': 'error'},
                {'role': 'assistant', 'reasoning_content': 'second', 'content': '', 'tool_calls': [call]},
                {'role': 'tool', 'content': '1'}]
    target = {'role': 'assistant', 'reasoning_content': 'done', 'content': 'one'}
    text, ids, start, spans = native_gold_packet_context_only_messages(tokenizer, messages, target, EVAL_TOOLS, [1, 3])
    assert len(spans) == 2 and all(b < e <= start for b, e in spans)
    assert '<think>first</think>' in tokenizer.decode(ids[spans[0][0]:spans[0][1]])


@needs_lfm3
def test_guidance_reads_xml_calls(tokenizer):
    pytest.importorskip('tree_sitter_typescript')
    from natlang_neuralese.serve.guidance import Guide, Settings, code_strings
    reply = _model_output(tokenizer, {'role': 'assistant', 'reasoning_content': 'plan', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': 'eval', 'arguments': {
            'code': 'const s = "</parameter>";\nconst x = [1, 2].filter((v) => v > 1);\nreturn x;'}}}]}, EVAL_TOOLS)
    guide = Guide(Settings(tools=['eval'], call_format='xml', require_call=True))
    assert guide.forced_prefix() == ''
    assert all(guide.check(reply[:end]) is None for end in range(1, len(reply) + 1))
    (code,) = code_strings(reply, 'xml')
    assert code.closed and code.text.startswith('const s = "</parameter>";') and reply[code.raw_lines[1]:].startswith('const x')
    bad = reply.replace('v > 1);', 'v > 1;')
    verdict = next(v for v in (Guide(Settings(call_format='xml')).check(bad[:end]) for end in range(1, len(bad) + 1)) if v)
    assert verdict[0] == 'syntax' and bad[verdict[1]:].startswith('const x')
    assert Guide(Settings(tools=['eval'], call_format='xml')).check(
        'plan</think><tool_call>\n<function=evaluate>\n') == ('unknown-tool', len('plan</think><tool_call>\n<function='))
    assert Guide(Settings(call_format='xml', require_call=True)).check('plan</think>Sure') == ('no-call', len('plan</think>'))
    assert Guide(Settings(call_format='xml', require_call=True)).check('plan</think><tool') is None


def test_generation_checks_lfm3_convention():
    from natlang_neuralese.eval.generation_checks import check
    probe = {'output_convention': 'lfm3', 'enable_thinking': True, 'kind': 'tool', 'tools': EVAL_TOOLS,
             'expected_tool': 'eval', 'check_argument_types': True}
    good = 'r</think><tool_call>\n<function=eval>\n<parameter=code>\nreturn 1;\n</parameter>\n</function>\n</tool_call>'
    result = check(probe, good, [1, 2], 2)
    assert result['think'] and result['tool']
    assert not check(probe, good.replace('<function=eval>', '<function=evil>'), [1, 2], 2)['tool']
    assert not check({**probe, 'enable_thinking': False}, good, [1, 2], 2)['think']


def test_scan_partial_and_malformed_calls():
    text = 'x<tool_call>\n<function=eval>\n<parameter=code>\nopen value'
    (call,) = scan_xml_calls(text)
    assert call.name == 'eval' and call.end is None and call.parameters[0].value_end is None
    with pytest.raises(ValueError):
        parse_xml_calls(text)
    message = build_message('<tool_call>\n<function=eval>\n<parameter=code>\nx\n</parameter>\n</tool_call>', [])
    assert 'tool_calls' not in message and '<tool_call>' in message['content']  # malformed: left as content


def test_records_flag_leaked_xml_calls():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
    from records import residual_markup, strip_markup
    leaked = 'ok <tool_call>\n<function=eval>\n<parameter=code>\nx\n</parameter>\n</function>\n</tool_call>'
    assert residual_markup(leaked)
    assert not residual_markup('a doc that mentions <tool_call> and </parameter> tags')
    assert strip_markup(leaked) == leaked
