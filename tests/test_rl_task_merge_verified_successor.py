from scripts.rl_tasks.merge_verified_successor import _correct_ordered_sql, _treatment_update


def test_ordered_sql_correction_preserves_answer_and_rejects_reversed_alias():
    row = {
        'id': 'ext:spider:q', 'source_revisions': ['spider@fixture'],
        'curriculum': {'rl': {'level': 0}, 'external': {'source': 'spider',
            'gold_sql': 'SELECT name FROM people ORDER BY name',
            'checks': {'alias_answers': [[['Ada'], ['Bea']], [['Bea'], ['Ada']]]}}},
        'semantics': {'expected': [['Ada'], ['Bea']], 'oracle': {
            'level': 'normalized', 'normalization': 'result-set'}}}

    proof = _correct_ordered_sql(row)

    assert proof['reversed_alias_removed'] is True
    assert row['semantics']['expected'] == [['Ada'], ['Bea']]
    assert row['semantics']['oracle']['ordered'] is True
    assert row['curriculum']['external']['checks']['alias_answers'] == [[['Ada'], ['Bea']]]
    assert row['curriculum']['rl']['level'] >= 1


def test_verified_treatment_keeps_original_task_identity_and_groups():
    old = {'id': 'stable-id', 'source': 'external-rl:babi', 'source_ids': ['upstream:q'],
           'source_groups': ['babi:story-1'], 'gold_sources': ['gold'], 'generation': {'method': 'source'},
           'source_revisions': ['babi@old']}
    candidate = {'id': 'stable-id:guided', 'source': 'external-rl:babi-guided',
                 'source_ids': ['babi-guided:upstream:q'], 'source_groups': ['babi:story-1'],
                 'curriculum': {'external': {'source_case_id': 'stable-id'}},
                 'semantics': {'files': {'root.nl': 'guided'}, 'inputs': {'facts': ['1. x']}, 'expected': 'x'}}

    updated = _treatment_update(old, candidate, 'babi-guided')

    assert updated['id'] == 'stable-id'
    assert updated['source'] == old['source']
    assert updated['source_ids'] == old['source_ids']
    assert updated['source_groups'] == old['source_groups']
    assert updated['semantics'] == candidate['semantics']
