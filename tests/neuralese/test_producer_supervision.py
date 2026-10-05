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
