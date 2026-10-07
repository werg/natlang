import json

import pytest

from natlang_neuralese.serve.chat import write_value_text
from natlang_neuralese.train.trajectory_state import validate_continuation


def template(messages, generate):
    prefix = 'USER x ASSISTANT '
    if generate:
        return prefix
    call = messages[-1]['tool_calls'][0]['function']
    args = ', '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in call['arguments'].items())
    return prefix + call['name'] + '(' + args + ') END'


@pytest.mark.parametrize('value', [
    {'id': 'a', 'flag': True}, [{'quote': 'α and evidence'}, {'n': 2}], 3, False,
])
def test_typed_value_uses_native_spacing_and_syntax(value):
    result = write_value_text(template, 'return_result', {'status': 'success'}, 'value', value, 'unknown')
    assert result == json.dumps(value, ensure_ascii=False)
    if isinstance(value, (dict, list)):
        assert result != json.dumps(value, separators=(',', ':'), ensure_ascii=False)


@pytest.mark.parametrize('value', ['a "quote" and a newline\n', "apostrophe's", '\\path', 'α'])
def test_string_span_excludes_quotes_but_retains_native_escaping(value):
    result = write_value_text(template, 'return_result', {'status': 'success'}, 'value', value)
    assert result == json.dumps(value, ensure_ascii=False)[1:-1]


def test_nested_write_boundary_and_gold_value_keep_sibling_fields():
    arguments = {'status': 'success', 'value': {'site': 'JUNIPER-4', 'access': 'old', 'audit': {'kept': True}}}
    path = ('value', 'access')
    prefix, suffix = __import__('natlang_neuralese.serve.chat', fromlist=['write_reply']).write_reply(
        template, 'return_result', arguments, 'value', 'string', argument_path=path)
    assert prefix.endswith('"access": "')
    assert suffix == '", "audit": {"kept": true}}) END'
    value = 'open with buffer'
    result = write_value_text(template, 'return_result', arguments, 'value', value,
                              argument_path=path)
    assert result == json.dumps(value, ensure_ascii=False)[1:-1]


def test_nested_unknown_write_boundary_keeps_structured_json():
    from natlang_neuralese.serve.chat import write_reply
    arguments = {'status': 'success', 'value': {'evidence': []}}
    path = ('value', 'evidence')
    prefix, suffix = write_reply(template, 'return_result', arguments, 'value', 'unknown',
                                 argument_path=path)
    assert prefix.endswith('"evidence": ')
    assert suffix == '}) END'
    value = {'source_id': 'p0', 'quote': 'An observed fact.'}
    assert write_value_text(template, 'return_result', arguments, 'value', value, 'unknown',
                            argument_path=path) == json.dumps(value, ensure_ascii=False)


def test_native_length_requires_explicit_curriculum_change():
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
             'identity': {'options': {}, 'files': {'data': 'pinned'}}}
    identity = {'options': {'writer_length_policy': 'native-value'}, 'files': {'data': 'pinned'}}
    with pytest.raises(ValueError, match='identical inputs'):
        validate_continuation(state, identity)
    validate_continuation(state, identity, allowed_changes=['writer_length_policy'])
