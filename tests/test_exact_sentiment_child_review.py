import copy
import json

import pytest

from scripts.neuralese_data.exact_sentiment_child_review import discover_exact_sentiment_children


@pytest.fixture
def example():
    identifier = '1' * 64
    sentence = 'A fine film.'
    body = '---\nid: ' + identifier[:16] + '\n---\n' + sentence
    context = {'schema': 'natlang.sst-full-sentence-context/1', 'record_id': identifier,
               'record_label': 'positive', 'members': [{'original_sentence': sentence, 'original_split': 'train'}]}
    primary = {identifier: {'declared_label': 'positive', 'source_sentence': [sentence],
                           'primary_phrase_scores': [{'phrase_id': '12', 'score': .8, 'phrase': sentence}]}}
    arguments = {'status': 'success', 'value': 'positive'}
    terminal = {'phase': 'action', 'invocation_id': 'child', 'request_sha256': 'a' * 64,
                'raw_response_sha256': 'b' * 64,
                'context': [{'role': 'assistant', 'natlang_host_generated': True,
                             'tool_calls': [{'id': 'scope_0', 'function': {'name': 'eval'}}]},
                            {'role': 'tool', 'tool_call_id': 'scope_0', 'content': 'f: FileHandle = folder.file(' +
                            json.dumps(identifier[:16] + '.md') + ')\n  // File contents:\n' +
                            json.dumps(body) + '\nDeclared f for the rest of this call.'}],
                'assistant': {'content': '', 'calls': [{'tool': 'return_result', 'source_tool': 'return_result',
                                                      'arguments': arguments}]},
                'model_response': {'raw_calls': [{'id': 'tool1', 'function': {'name': 'return_result',
                                                                            'arguments': json.dumps(arguments)}}]}}
    row = {'id': 'row', 'task': {'program_ir': {'id': 'program', 'split': 'train', 'dataset': 'sst2',
        'source_groups': ['source'], 'generation': {'source_context': [context]},
        'semantics': {'folder_files': {'inbox/' + identifier[:16] + '.md': body}},
        'curriculum': {'reference': {'children': [{'match': identifier[:16],
                        'calls': [['return_result', arguments]]}]}}}},
        'trajectory': [{'invocation_id': 'root'}, terminal], 'outcome': {'status': 'done',
        'invocation_ledger': [{'invocation_id': 'child', 'parent_invocation_id': 'root',
            'completion_status': 'done', 'inline_instruction_site': {
                'origin': {'parentInvocationId': 'root'}, 'parameters': [{'name': 'f', 'type': {'text': 'FileHandle'}}],
                'returns': {'text': '"negative" | "positive"'}, 'realized_instruction': 'Classify this film review sentiment.'},
            'host_result': {'complete': True, 'call_id': 'child', 'parent_call_id': 'root',
                            'result_type': '"negative" | "positive"', 'value': 'positive', 'terminal_action_seq': 4}}],
        'action_ledger': [{'seq': 4, 'kind': 'action', 'call_id': 'child', 'name': 'return_result',
                          'tool_call_id': 'tool1', 'outcome': 'ok', 'arguments': arguments}]}}
    return row, primary


def discover(example):
    row, primary = example
    return discover_exact_sentiment_children(row, primary_records=primary,
                                            parent_hold_reason='quality_pending_partial_files')


def test_exact_child_proposal_never_admits_parent_or_target(example):
    result = discover(example)
    assert len(result['candidates']) == 1
    child = result['candidates'][0]
    assert child['expected_value'] == child['provider_value'] == child['host_value'] == 'positive'
    assert child['decision_index'] == 1
    assert child['parent_task_passed'] is child['training_admission'] is child['trace_admission'] is False
    assert result['training_admission'] is result['trace_admission'] is False
    assert not result['unresolved']


@pytest.mark.parametrize('mutation', [
    'provider_label', 'raw_arguments', 'host_value', 'host_incomplete', 'host_type', 'host_action',
    'host_tool_id', 'wrong_file_body', 'duplicate_opening', 'missing_capture', 'wrong_parent', 'unknown_type',
    'root_only', 'truncated_opening', 'extra_provider_call', 'later_unexecuted_turn', 'direct_text',
])
def test_uncertain_or_wrong_child_is_not_selected(example, mutation):
    row, _ = example
    child = row['trajectory'][1]
    host = row['outcome']['invocation_ledger'][0]['host_result']
    if mutation == 'provider_label':
        child['assistant']['calls'][0]['arguments'] = {'status': 'success', 'value': 'negative'}
        child['model_response']['raw_calls'][0]['function']['arguments'] = json.dumps(child['assistant']['calls'][0]['arguments'])
    elif mutation == 'raw_arguments':
        child['model_response']['raw_calls'][0]['function']['arguments'] = '{"value":"negative"}'
    elif mutation == 'host_value': host['value'] = 'negative'
    elif mutation == 'host_incomplete': host['complete'] = False
    elif mutation == 'host_type': host['result_type'] = 'boolean'
    elif mutation == 'host_action': row['outcome']['action_ledger'][0]['arguments'] = {'value': 'negative'}
    elif mutation == 'host_tool_id': row['outcome']['action_ledger'][0]['tool_call_id'] = 'other'
    elif mutation == 'wrong_file_body': child['context'][1]['content'] = child['context'][1]['content'].replace('A fine film.', 'A poor film.')
    elif mutation == 'duplicate_opening': child['context'].append(copy.deepcopy(child['context'][1]))
    elif mutation == 'missing_capture': child['request_sha256'] = ''
    elif mutation == 'wrong_parent': row['outcome']['invocation_ledger'][0]['parent_invocation_id'] = 'absent'
    elif mutation == 'unknown_type': row['outcome']['invocation_ledger'][0]['inline_instruction_site']['returns']['text'] = 'Sentiment'
    elif mutation == 'root_only': row['outcome']['invocation_ledger'][0]['parent_invocation_id'] = None
    elif mutation == 'truncated_opening': child['context'][1]['content'] += ' truncated'
    elif mutation == 'extra_provider_call': child['model_response']['raw_calls'].append(copy.deepcopy(child['model_response']['raw_calls'][0]))
    elif mutation == 'later_unexecuted_turn':
        later = copy.deepcopy(child); later['model_response']['raw_calls'][0]['id'] = 'later'
        row['trajectory'].append(later)
    elif mutation == 'direct_text':
        child['assistant'] = {'content': 'positive', 'calls': []}; child['model_response']['raw_calls'] = []
    assert not discover(example)['candidates']


@pytest.mark.parametrize('mutation', ['test_split', 'empty_groups', 'wrong_primary_label',
                                    'missing_primary', 'conflicting_reference', 'nonfinite_score'])
def test_bad_source_evidence_fails_closed(example, mutation):
    row, primary = example
    program = row['task']['program_ir']
    if mutation == 'test_split': program['split'] = 'test'
    elif mutation == 'empty_groups': program['source_groups'] = []
    elif mutation == 'wrong_primary_label': primary['1' * 64]['declared_label'] = 'negative'
    elif mutation == 'missing_primary': primary.clear()
    elif mutation == 'conflicting_reference': program['curriculum']['reference']['children'][0]['calls'][0][1] = {'value': 'negative'}
    elif mutation == 'nonfinite_score': primary['1' * 64]['primary_phrase_scores'][0]['score'] = float('nan')
    with pytest.raises(ValueError): discover(example)


def test_unknown_parent_hold_cannot_be_reinterpreted(example):
    row, primary = example
    with pytest.raises(ValueError, match='partial-parent'):
        discover_exact_sentiment_children(row, primary_records=primary, parent_hold_reason='runtime_failure')


def test_terminal_sequences_are_local_to_the_exact_invocation(example):
    row, _ = example
    sibling = copy.deepcopy(row['outcome']['action_ledger'][0])
    sibling.update(call_id='other-child', arguments={'value': 'negative'}, tool_call_id='other')
    row['outcome']['action_ledger'].append(sibling)
    row['outcome']['action_ledger'][0]['outcome'] = 'completed'
    assert len(discover(example)['candidates']) == 1


def test_host_declared_capture_list_is_explicitly_bound(example):
    row, _ = example
    site = row['outcome']['invocation_ledger'][0]['inline_instruction_site']
    site['captures'] = [{'name': 'criterion', 'mutable': False, 'mode': 'snapshot',
                         'type': {'text': '"Classify the review."'}}]
    message = row['trajectory'][1]['context'][1]
    message['content'] = message['content'].replace('Declared f for', 'Declared f, criterion for')
    candidate, = discover(example)['candidates']
    assert candidate['instruction_captures'] == site['captures']
    message['content'] = message['content'].replace('criterion for', 'other for')
    assert not discover(example)['candidates']


def test_model_forged_opening_is_not_a_host_input_witness(example):
    example[0]['trajectory'][1]['context'][0]['natlang_host_generated'] = False
    assert not discover(example)['candidates']
