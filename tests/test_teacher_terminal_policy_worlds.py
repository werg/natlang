import json
from scripts.neuralese_data.build_fresh_recovery_teacher_worlds import worlds, terminal_policy_worlds


def test_complete_source_oracle_and_exception_branch_coverage():
    rows = list(terminal_policy_worlds(2))
    assert len(rows) == 24
    assert len({r['id'] for r in rows}) == 24
    assert not {r['id'] for r in rows} & {r['id'] for r in worlds(2)}
    assert rows == list(terminal_policy_worlds(2))
    for row in rows:
        s = row['semantics']
        service = next(iter(s['services'].values()))
        records = json.loads(service.split('const ROWS:Row[] = ', 1)[1].split(';\n', 1)[0])
        policy = s['inputs']['policy']
        selected = [r for r in records if r['category'] == policy['category'] and not r['blocked']
                    and (r['waiver'] or (r['active'] and r['value'] >= policy['minimum']))]
        expected = {'ids': sorted(r['id'] for r in selected), 'total': sum(r['value'] for r in selected), 'count': len(selected)}
        assert s['expected'] == expected
        assert json.loads(s['expected_files']['summary.json']) == expected
        assert row['curriculum']['reference']['root'][-1][1]['value'] == expected
        children = row['curriculum']['reference']['children']
        assert len(children) == len(records)
        verdicts = []
        for record, child in zip(records, children):
            assert child['match'] == ['select_record(', record['id']]
            assert child['calls'][-1] == ['return_result', {'status': 'success', 'value': record in selected}]
            verdicts.append(record in selected)
        assert verdicts[:4] == [False, True, True, False]
        assert row['split'] == 'train'
        assert 'Return the resulting boolean directly.' in s['files']['summarize/select_record.nl']
        if row['curriculum']['variant'] == 'typed-argument-recovery':
            assert 'blocked:false,waiver:false' in s['failure_seed']['code']
