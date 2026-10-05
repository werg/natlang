from scripts.chunk_scoring import parsed_chunks,_string_map
import pytest


def test_code_lines_preserve_escaped_quotes_unicode_and_offsets():
    value="const café='x';\nreturn café;"
    text="<|tool_call_start|>[eval(code="+repr(value)+")]<|tool_call_end|><|im_end|>"
    chunks=parsed_chunks(text,{'tool_calls':[{'function':{'name':'eval','arguments':{'code':value}}}]})
    assert [c['kind'] for c in chunks]==['tool_name','code_line','code_line']
    assert chunks[1]['value']=="const café='x';\n"
    assert chunks[2]['value_start']==len(chunks[1]['value'])
    assert text[chunks[0]['char_start']:chunks[0]['char_end']]=='eval'
    assert 'return café;' in text[chunks[2]['char_start']:chunks[2]['char_end']]


def test_arguments_are_not_split_at_punctuation_inside_strings():
    text="<|tool_call_start|>[read_code(name='skills.a.b')]<|tool_call_end|><|im_end|>"
    chunks=parsed_chunks(text,{'tool_calls':[{'function':{'name':'read_code','arguments':'{"name":"skills.a.b"}'}}]})
    assert len(chunks)==2 and chunks[1]['value']=='skills.a.b'


def test_prose_sentences_and_unsupported_spelling_fail_closed():
    chunks=parsed_chunks('Hello. Next!<|im_end|>',{'content':'Hello. Next!'})
    assert [c['value'] for c in chunks]==['Hello.',' Next!']
    with pytest.raises(ValueError):_string_map('r"a"','a')


def test_logical_replacement_offsets_are_javascript_utf16_not_python_codepoints():
    value="const emoji='😀';\nreturn emoji;"
    text='<|tool_call_start|>[eval(code='+repr(value)+')]<|tool_call_end|><|im_end|>'
    chunks=parsed_chunks(text,{'tool_calls':[{'function':{'name':'eval','arguments':{'code':value}}}]})
    assert chunks[2]['value_start']==len(value.splitlines(keepends=True)[0])+1
    prose=parsed_chunks('Hi😀. Next!<|im_end|>',{'content':'Hi😀. Next!'})
    assert prose[1]['value_start']==5
