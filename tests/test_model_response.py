import json
from pathlib import Path
import pytest
from scripts.model_response import parse_response

TOOLS=[{'function':{'name':'eval','parameters':{'properties':{'code':{'type':'string'},'finish':{'type':'boolean'}}}}}]
LING_TOOLS=[
    {'function': {'name': 'lookup', 'parameters': {
        'type': 'object',
        'properties': {
            'query': {'type': 'string'}, 'enabled': {'type': 'boolean'},
            'count': {'type': 'integer'}, 'options': {'type': 'object'},
            'steps': {'type': 'array'},
        },
        'required': ['query', 'enabled'],
    }}},
    {'function': {'name': 'finish', 'parameters': {
        'type': 'object', 'properties': {'value': {'type': 'string'}},
        'required': ['value'],
    }}},
]


def test_xml_tool_parameters_preserve_exact_code_and_typed_finish():
    text='<think>Use the file handle.</think><function name="eval"><param name="code"><![CDATA[const s = " a < b & c ";\nreturn s;]]></param><param name="finish">true</param></function>'
    result=parse_response(text,TOOLS)
    assert result['content']==''
    assert result['reasoning_content']=='Use the file handle.'
    assert json.loads(result['tool_calls'][0]['function']['arguments'])=={'code':'const s = " a < b & c ";\nreturn s;','finish':True}


def test_thinking_never_executes_embedded_tools_and_unfinished_thinking_is_not_an_answer():
    text='<think>Maybe <function name="eval"><param name="code">bad()</param></function></think>done'
    assert parse_response(text,TOOLS)['tool_calls']==[]
    assert parse_response(text,TOOLS)['content']=='done'
    result=parse_response('<think>Not finished',TOOLS)
    assert result['content']=='' and result['tool_calls']==[]
    result=parse_response('The generation prefix opened thinking.</think>7')
    assert result['content']=='7' and result['reasoning_content']=='The generation prefix opened thinking.'


def test_control_looking_text_inside_cdata_remains_exact_code():
    code='return "</think></function>";'
    result=parse_response('<function name="eval"><param name="code"><![CDATA['+code+']]></param></function>',TOOLS)
    assert json.loads(result['tool_calls'][0]['function']['arguments'])=={'code':code}


def test_invalid_xml_and_unknown_tools_are_not_silently_repaired():
    with pytest.raises(ValueError):parse_response('<function name="unknown"></function>',TOOLS)
    with pytest.raises(ValueError):parse_response('<function name="eval"><param name="finish">yes</param></function>',TOOLS)


def test_literal_model_format_is_decoded_without_evaluating_calls():
    result=parse_response('<|tool_call_start|>[eval(code="return 7;", finish=true)]<|tool_call_end|>')
    assert json.loads(result['tool_calls'][0]['function']['arguments'])=={'code':'return 7;','finish':True}
    with pytest.raises(ValueError):parse_response('<|tool_call_start|>[eval(code=__import__("os"))]<|tool_call_end|>')


def test_live_minicpm_xml_boolean_and_generation_prefix():
    text='Use both required arguments.</think><function name="eval"><param name="code">6 * 7</param><param name="finish">True</param></function><|im_end|>'
    result=parse_response(text,TOOLS)
    assert result['content']==''
    assert result['reasoning_content']=='Use both required arguments.'
    assert json.loads(result['tool_calls'][0]['function']['arguments'])=={'code':'6 * 7','finish':True}


def test_ling_template_tool_call_round_trip_preserves_raw_string_and_json_types():
    raw = '  quote " & <node>\n<think>preserve me</think>\nnext line  '
    nested = {'enabled': False, 'values': [1, {'label': '<x> & "quoted"'}]}
    text = (
        'Need the lookup. </think>'
        '<tool_call>lookup\n'
        '<arg_key>query</arg_key>\n<arg_value>' + raw + '</arg_value>'
        '<arg_key>enabled</arg_key>\n<arg_value>false</arg_value>'
        '<arg_key>count</arg_key>\n<arg_value>3</arg_value>'
        '<arg_key>options</arg_key>\n<arg_value>' + json.dumps(nested, ensure_ascii=False) + '</arg_value>'
        '<arg_key>steps</arg_key>\n<arg_value>' + json.dumps([1, 'two', False]) + '</arg_value>'
        '\n</tool_call><|role_end|>'
    )
    result = parse_response(text, LING_TOOLS)
    assert result['reasoning_content'] == 'Need the lookup.'
    assert result['content'] == ''
    assert len(result['tool_calls']) == 1
    assert json.loads(result['tool_calls'][0]['function']['arguments']) == {
        'query': raw, 'enabled': False, 'count': 3, 'options': nested,
        'steps': [1, 'two', False],
    }


def test_ling_template_multiple_calls_and_open_thinking_prefix():
    text = (
        'thinking before call</think>'
        '<tool_call>lookup<arg_key>query</arg_key>\n<arg_value>Warsaw</arg_value>'
        '<arg_key>enabled</arg_key>\n<arg_value>true</arg_value>\n</tool_call>'
        ' between calls '
        '<tool_call>finish<arg_key>value</arg_key>\n<arg_value>ok</arg_value>\n</tool_call>'
        '<|role_end|>'
    )
    result = parse_response(text, LING_TOOLS)
    assert result['reasoning_content'] == 'thinking before call'
    assert result['content'] == 'between calls'
    assert [call['function']['name'] for call in result['tool_calls']] == ['lookup', 'finish']
    assert [json.loads(call['function']['arguments']) for call in result['tool_calls']] == [
        {'query': 'Warsaw', 'enabled': True}, {'value': 'ok'}
    ]


@pytest.mark.parametrize('text', [
    '<tool_call>missing<arg_key>query</arg_key><arg_value>x</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>query</arg_key><arg_value>x</tool_call>',
    '<tool_call>lookup<arg_key>query</arg_key><arg_value>x</arg_value><arg_key>query</arg_key><arg_value>y</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>extra</arg_key><arg_value>x</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>query</arg_key><arg_value>x</arg_value><arg_key>enabled</arg_key><arg_value>yes</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>query</arg_key><arg_value>x</arg_value><arg_key>enabled</arg_key><arg_value>NaN</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>count</arg_key><arg_value>1e400</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>options</arg_key><arg_value>{"x":1,"x":2}</arg_value></tool_call>',
    '<tool_call>lookup<arg_key>query</arg_key><arg_value>x</arg_value>',
])
def test_ling_template_rejects_unknown_malformed_duplicate_or_mistyped_calls(text):
    with pytest.raises(ValueError):
        parse_response(text, LING_TOOLS)


def test_ling_template_rejects_ambiguous_schema_unions_but_keeps_plain_unknown_string():
    tools = [{'function': {'name': 'union', 'parameters': {
        'properties': {'value': {'type': ['string', 'integer']}}, 'required': ['value'],
    }}}]
    with pytest.raises(ValueError, match='ambiguous'):
        parse_response('<tool_call>union<arg_key>value</arg_key><arg_value>123</arg_value></tool_call>', tools)
    tools[0]['function']['parameters']['properties']['value'] = {}
    plain = parse_response('<tool_call>union<arg_key>value</arg_key><arg_value>plain words</arg_value></tool_call>', tools)
    assert json.loads(plain['tool_calls'][0]['function']['arguments']) == {'value': 'plain words'}
    with pytest.raises(ValueError, match='ambiguous'):
        parse_response('<tool_call>union<arg_key>value</arg_key><arg_value>123</arg_value></tool_call>', tools)


def test_ling_template_accepts_integer_encoding_for_json_number_and_rejects_ambiguous_union():
    number_tool = [{'function': {'name': 'number', 'parameters': {
        'properties': {'value': {'type': 'number'}}, 'required': ['value'],
    }}}]
    parsed = parse_response('<tool_call>number<arg_key>value</arg_key><arg_value>3</arg_value></tool_call>', number_tool)
    assert json.loads(parsed['tool_calls'][0]['function']['arguments']) == {'value': 3}

    union_tool = [{'function': {'name': 'union', 'parameters': {
        'properties': {'value': {'type': ['string', 'number']}}, 'required': ['value'],
    }}}]
    with pytest.raises(ValueError, match='ambiguous'):
        parse_response('<tool_call>union<arg_key>value</arg_key><arg_value>3</arg_value></tool_call>', union_tool)


def test_ling_template_preserves_existing_minicpm_literal_payload_with_marker_text():
    code = 'return "<tool_call>not a top-level Ling call</tool_call>";'
    result = parse_response(
        '<function name="eval"><param name="code"><![CDATA[' + code + ']]></param>'
        '<param name="finish">False</param></function>', TOOLS)
    assert json.loads(result['tool_calls'][0]['function']['arguments']) == {
        'code': code, 'finish': False,
    }


def test_pinned_ling_template_call_round_trips_through_production_renderer():
    transformers = pytest.importorskip('transformers')
    from scripts.render_training_corpus import _call_template

    template_path = (Path(__file__).resolve().parents[1] /
                     'runs/ling-student-compatibility-20261002/tokenizer-only-snapshot/chat_template.jinja')
    if not template_path.exists():
        pytest.skip('pinned Ling chat template is not present in this checkout')
    source = template_path.read_text()
    tokenizer_dir = template_path.parent
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.chat_template = source

    args = {
        'query': '  quote " & <node>\nnext line  ',
        'enabled': False,
        'count': 3,
        'options': {'label': '<detail> & "quoted"', 'nested': {'ok': True}},
        'steps': [1, 'two', False],
    }
    messages = [
        {'role': 'user', 'content': 'Look up the query.'},
        {'role': 'assistant', 'reasoning_content': 'Need a lookup.', 'content': '',
         'tool_calls': [{'id': 'call_1', 'type': 'function', 'function': {
             'name': 'lookup', 'arguments': json.dumps(args, ensure_ascii=False)}}]},
    ]
    rendered = _call_template(tokenizer, messages, LING_TOOLS[:1], False)
    assistant = rendered.rsplit('<role>ASSISTANT</role>', 1)[1]
    result = parse_response(assistant, LING_TOOLS)
    assert result['reasoning_content'] == 'Need a lookup.'
    assert result['tool_calls'][0]['function']['name'] == 'lookup'
    assert json.loads(result['tool_calls'][0]['function']['arguments']) == args
