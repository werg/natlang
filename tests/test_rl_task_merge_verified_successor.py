import pytest

from scripts.rl_tasks.merge_verified_successor import _correct_ordered_sql, _treatment_update, replay_summary


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


def test_replay_gate_reads_nested_verifier_summary_and_requires_all_rows():
    report = {'summary': {'cases': 710, 'kept': 710, 'wrong_accepted': 0, 'wrong_errors': 0}}
    assert replay_summary(report)['cases'] == 710

    with pytest.raises(ValueError):
        replay_summary({'summary': {'cases': 710, 'kept': 709, 'wrong_accepted': 0, 'wrong_errors': 0}})
    with pytest.raises(ValueError):
        replay_summary({'summary': {'cases': 710, 'kept': 710, 'wrong_accepted': 1, 'wrong_errors': 0}})


def test_successor_keeps_predecessor_and_new_versioned_assets(tmp_path):
    from scripts.rl_tasks.merge_verified_successor import inherit_versioned_assets
    base, addition, out = [tmp_path/name for name in ('base', 'addition', 'out')]
    for root, name, text in ((base, 'old.txt', 'old payload'), (addition, 'new.txt', 'new payload')):
        path=root/'assets-v3'/name;path.parent.mkdir(parents=True);path.write_text(text)
    ref=inherit_versioned_assets(base,[addition],out)
    assert (out/'assets-v3/old.txt').read_text()=='old payload'
    assert (out/'assets-v3/new.txt').read_text()=='new payload'
    assert ref['path']=='assets-v3.manifest.json'
    (addition/'assets-v3/old.txt').write_text('conflicting payload')
    with pytest.raises(ValueError,match='explicit task/source rewrite'):
        inherit_versioned_assets(base,[addition],tmp_path/'other')
