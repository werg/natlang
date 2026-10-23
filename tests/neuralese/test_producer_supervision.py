import json
from natlang_neuralese.data.trajectory_text import producer_text_target


def test_producer_gold_reply_is_not_replaced_by_its_own_generated_payload():
    record = {'messages': [], 'target': {'role': 'assistant', 'tool_calls': [
        {'id': 'return', 'type': 'function', 'function': {'name': 'return_result',
         'arguments': json.dumps({'status': 'success', 'value': {'$write': {
             'name': 'child-result', 'source': '{"answer":42}', 'type': 'Neuralese<unknown>'}}})}}]}}
    target = producer_text_target(record, {}, {'child-result': 'nz1_generated'})
    arguments = json.loads(target['tool_calls'][0]['function']['arguments'])
    assert arguments['value'] == {'answer': 42}
    assert 'nz1_generated' not in json.dumps(target)


def test_crisp_render_preserves_argument_order_and_source_value():
    from natlang_neuralese.data.trajectory_text import render
    call = {"type": "function", "function": {"name": "return_result", "arguments": json.dumps({
        "status": "success", "value": {"$write": {"name": "result", "type": "Neuralese<unknown>", "source": "[42]"}}})}}
    actual = render([{"role": "assistant", "tool_calls": [call]}], lambda name: {}, {})
    args = json.loads(actual[0]['tool_calls'][0]['function']['arguments'])
    assert list(args) == ['status', 'value']
    assert args['value'] == [42]


def test_crisp_render_expands_each_written_argument_without_flat_blocks():
    from natlang_neuralese.data.trajectory_text import render
    call = {"function": {"name": "combine", "arguments": json.dumps({
        "note": {"$write": {"name": "note", "type": "Neuralese<string>", "source": "hello"}},
        "value": {"$write": {"name": "result", "type": "Neuralese<unknown>", "source": "[42]"}}})}}
    actual = render([{"role": "assistant", "tool_calls": [call]}], lambda name: {}, {})
    args = json.loads(actual[0]['tool_calls'][0]['function']['arguments'])
    assert args == {'note': 'hello', 'value': [42]}
