import json

import pytest

from natlang_neuralese.serve.chat import write_reply, write_value_text
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


def test_concrete_json_writer_uses_unknown_serialization_family_without_losing_declared_type():
    from natlang_neuralese.train.trajectories import native_writer_prefix, write_value_type

    declared_type = 'Neuralese<{ count: number, flag: boolean }>'
    body = '{"count":4,"flag":true}'
    marker = {'$write': {'name': 'typed-result:object', 'type': declared_type,
                         'source': body, 'source_encoding': 'json'}}
    target = {'role': 'assistant', 'tool_calls': [{'id': 'r', 'function': {
        'name': 'return_result', 'arguments': json.dumps({'status': 'success', 'value': marker})}}]}
    record = {'messages': [], 'target': target, '_active_write_name': 'typed-result:object'}

    # Training selects the serialization family from the authenticated JSON encoding,
    # while the writer receipt retains the concrete Neuralese<T> contract.
    value_type = write_value_type(record)
    assert value_type == 'unknown'
    assert json.loads(target['tool_calls'][0]['function']['arguments'])['value']['$write']['type'] == declared_type
    training_prefix = native_writer_prefix(record, template)
    serving_prefix, _ = write_reply(template, 'return_result', {'status': 'success'},
                                    argument='value', value_type=value_type)
    assert training_prefix == serving_prefix
    assert training_prefix.endswith('value=')
    assert write_value_text(template, 'return_result', {'status': 'success'}, 'value',
                            json.loads(body), value_type) == json.dumps(json.loads(body), ensure_ascii=False)

    # A JSON-looking Neuralese<string> source stays literal text and keeps the
    # ordinary quoted string boundary.
    literal = {'$write': {'name': 'typed-result:string', 'type': 'Neuralese<string>', 'source': body}}
    literal_target = {'role': 'assistant', 'tool_calls': [{'id': 's', 'function': {
        'name': 'return_result', 'arguments': json.dumps({'status': 'success', 'value': literal})}}]}
    literal_record = {'messages': [], 'target': literal_target, '_active_write_name': 'typed-result:string'}
    assert write_value_type(literal_record) == 'string'
    literal_training_prefix = native_writer_prefix(literal_record, template)
    literal_serving_prefix, _ = write_reply(template, 'return_result', {'status': 'success'},
                                            argument='value', value_type='string')
    assert literal_training_prefix == literal_serving_prefix
    assert literal_training_prefix.endswith('value="')
    assert write_value_text(template, 'return_result', {'status': 'success'}, 'value',
                            body, 'string') == json.dumps(body, ensure_ascii=False)[1:-1]


def test_native_length_requires_explicit_curriculum_change():
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
             'identity': {'options': {}, 'files': {'data': 'pinned'}}}
    identity = {'options': {'writer_length_policy': 'native-value'}, 'files': {'data': 'pinned'}}
    with pytest.raises(ValueError, match='identical inputs'):
        validate_continuation(state, identity)
    validate_continuation(state, identity, allowed_changes=['writer_length_policy'])
