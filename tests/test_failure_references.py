"""Bad calls may appear in history but must never be supervised as correct targets."""
import json

from scripts.generate_failures import generate_group


def test_matched_references_verify_and_keep_pairs_in_one_split_group():
    rows, counts = generate_group((83, 4))
    assert len(counts) == 12 and all(n == 1 for n in counts.values())
    assert len({r['program_id'] for r in rows}) == 1
    errors = [r for r in rows if r['skill'] == 'report_error']
    assert len(errors) == 5
    assert len([r for r in rows if r['skill'] == 'report_blocker']) == 1
    for row in rows:
        for call in row['target'].get('tool_calls', []):
            args = json.loads(call['function']['arguments'])
            assert not (call['function']['name'] == 'call' and args.get('function') == 'size_of' and args['to'] == 'return')
    for family in ('failure_binding_repair', 'failure_binding_error'):
        first = next(r for r in rows if r['family'] == family)
        assert first['recovery']
        assert any(m['role'] == 'tool' and 'type-does-not-fit-slot' in m['content'] for m in first['messages'])
