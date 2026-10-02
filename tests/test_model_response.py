import json
import pytest
from scripts.model_response import parse_response

TOOLS=[{'function':{'name':'eval','parameters':{'properties':{'code':{'type':'string'},'finish':{'type':'boolean'}}}}}]


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
