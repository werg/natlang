import copy
import json

from natlang_neuralese.train.trajectory_probe import aligned_donor, select_held, select_paired_held, source_groups


def producer(name, fact, *, body='body', schema=None, source=None):
    schema = schema or {'type': 'string'}
    return {'id': 'producer:' + name, 'program_id': 'program:' + name,
            'source_groups': [fact, 'program:' + name],
            'messages': [{'role': 'user', 'content': [{'type': 'soft', 'name': body}]}],
            'target': {'tool_calls': [{'function': {'name': 'return_result',
                        'arguments': json.dumps({'value': {'$write': {'name': name, 'source': source or name}}})}}]},
            'tools': [{'function': {'name': 'return_result', 'parameters': {
                        'type': 'object', 'properties': {'value': schema}}}}]}


def reader(name, fact, *payloads):
    return {'id': name, 'program_id': 'program:' + name, 'task': 'consume',
            'source_groups': [fact, 'program:' + name],
            'messages': [{'role': 'tool', 'content': [{'type': 'read', 'name': p} for p in payloads]}]}


def two():
    rows = [reader('ra', 'fact-a', 'a'), reader('rb', 'fact-b', 'b')]
    return rows, {'a': producer('a', 'fact-a'), 'b': producer('b', 'fact-b')}, {'body': 'function-body'}


def test_provenance_aliases_are_not_factual_groups():
    assert source_groups({'program_id': 'p', 'source_program_ids': ['q'],
                          'source_ref': {'program_ir_id': 'r'}, 'source_groups': ['p','q','r','fact']}) == {'fact'}


def test_pair_selection_requires_reciprocal_mapping_and_keeps_both_rows():
    rows, producers, kinds = two()
    selected, report = select_paired_held(rows, 2, producers, kinds)
    assert {r['id'] for r in selected} == {'ra', 'rb'}
    assert report['selected_mappings']['ra'] == {'donor_id': 'rb', 'donor_payload_to_recipient_payload': {'b':'a'}}
    assert report['selected_mappings']['rb']['donor_payload_to_recipient_payload'] == {'a':'b'}
    assert report['eligible_factual_group_counts'] == {'fact-a': 1, 'fact-b': 1}
    assert select_paired_held(rows, 1, producers, kinds)[0] == []


def test_same_fact_different_programs_are_not_donors():
    rows, producers, kinds = two()
    rows[1]['source_groups'][0] = 'fact-a'
    producers['b']['source_groups'][0] = 'fact-a'
    assert aligned_donor(rows[0], rows[1:], producers, kinds)['donor'] is None


def test_shared_payloads_equal_output_and_absent_producer_are_not_controls():
    rows, producers, kinds = two()
    producers['b']['target']['tool_calls'][0]['function']['arguments'] = json.dumps(
        {'value': {'$write': {'name': 'b', 'source': 'a'}}})
    assert aligned_donor(rows[0], rows[1:], producers, kinds)['donor'] is None
    rows[1] = reader('rb', 'fact-b', 'a')
    assert aligned_donor(rows[0], rows[1:], producers, kinds)['donor'] is None
    rows, producers, kinds = two()
    del producers['a']
    assert 'no target-write producer proof' in aligned_donor(rows[0], rows[1:], producers, kinds)['reason']


def test_types_roles_and_producer_factual_closure_are_required():
    rows, producers, kinds = two()
    for replacement in (producer('b', 'fact-b', schema={'type': 'boolean'}),
                        producer('b', 'fact-b', body='other-role'), producer('b', 'another-fact')):
        changed = {**producers, 'b': replacement}
        assert aligned_donor(rows[0], rows[1:], changed, {**kinds, 'other-role': 'function-body'})['donor'] is None
    plain = copy.deepcopy(producers)
    plain['a']['messages'][0]['content'] = [{'type': 'text', 'text': 'Unique crisp skill instructions'}]
    assert 'no function-body soft IDs' in aligned_donor(rows[0], rows[1:], plain, kinds)['reason']


def test_bijection_never_reuses_one_donor_block_for_two_slots():
    rows = [reader('ra', 'fact-a', 'a1', 'a2'), reader('rb', 'fact-b', 'b1', 'b2')]
    producers = {p: producer(p, fact) for fact, payloads in [('fact-a', ['a1','a2']), ('fact-b',['b1','b2'])]
                 for p in payloads}
    kinds = {'body': 'function-body'}
    assert aligned_donor(rows[0], rows[1:], producers, kinds)['donor'] is None  # ambiguous role alignment
    for p in ('a2','b2'):
        producers[p]['messages'][0]['content'][0]['name'] = 'second-body'
    proof = aligned_donor(rows[0], rows[1:], producers, {**kinds, 'second-body': 'function-body'})
    assert proof['mapping'] == {'b1':'a1', 'b2':'a2'}


def test_ce_selection_balances_groups_and_prioritizes_actual_readers():
    rows = [reader('a0', 'fact-a'), reader('a1', 'fact-a', 'a'), reader('b0','fact-b','b')]
    assert {r['id'] for r in select_held(rows, 2)} == {'a1','b0'}


def test_own_target_write_is_not_a_reader_slot():
    rows, producers, kinds = two()
    rows[0]['target'] = producers['a']['target']
    selected, report = select_paired_held(rows, 2, producers, kinds)
    assert selected == []
    assert report['reader_rows'] == 1
