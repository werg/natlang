import json
from natlang_neuralese.train.trajectories import producer_text_target


def test_producer_gold_reply_is_not_replaced_by_its_own_generated_payload():
    record = {'messages': [], 'target': {'role': 'assistant', 'tool_calls': [
        {'id': 'return', 'type': 'function', 'function': {'name': 'return_result',
         'arguments': json.dumps({'status': 'success', 'value': {'$write': {
             'name': 'child-result', 'source': '{"answer":42}', 'type': 'Neuralese<unknown>'}}})}}]}}
    target = producer_text_target(record, {}, {'child-result': 'nz1_generated'})
    arguments = json.loads(target['tool_calls'][0]['function']['arguments'])
    assert arguments['value'] == {'answer': 42}
    assert 'nz1_generated' not in json.dumps(target)


def test_render_preserves_argument_order_and_unknown_type():
    from natlang_neuralese.train.trajectories import render
    call = {"type": "function", "function": {"name": "return_result", "arguments": json.dumps({
        "status": "success", "value": {"$write": {"name": "result", "type": "Neuralese<unknown>", "source": "[42]"}}})}}
    actual = render([{"role": "assistant", "tool_calls": [call]}], lambda name: {}, {}, {'result': 'nz1_test'})
    args = json.loads(actual[0]['tool_calls'][0]['function']['arguments'])
    assert list(args) == ['status', 'value']
    assert args['value'] == [{'type': 'neuralese', 'id': 'nz1_test', 'value_type': 'unknown'}]


def test_render_preserves_each_written_argument_type():
    from natlang_neuralese.train.trajectories import render
    call = {"function": {"name": "combine", "arguments": json.dumps({
        "note": {"$write": {"name": "note", "type": "Neuralese<string>", "source": "hello"}},
        "value": {"$write": {"name": "result", "type": "Neuralese<unknown>", "source": "[42]"}}})}}
    actual = render([{"role": "assistant", "tool_calls": [call]}], lambda name: {}, {},
                    {'note': 'nz1_note', 'result': 'nz1_result'})
    args = json.loads(actual[0]['tool_calls'][0]['function']['arguments'])
    assert args['note'][0]['value_type'] == 'string'
    assert args['value'][0]['value_type'] == 'unknown'
