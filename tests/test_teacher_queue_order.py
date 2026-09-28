import pytest
from scripts.interleave_teacher_queue import interleave_entries


def test_rotation_covers_families_and_sources_without_changing_cases():
    entries = [dict(key=k, family=f, group=g, case_seconds=1200, max_model_requests=672)
               for k, f, g in [('a1', 'logic', 'a'), ('a2', 'logic', 'a'),
                               ('b1', 'logic', 'b'), ('d1', 'folder', 'd'), ('d2', 'folder', 'd')]]
    ordered = interleave_entries(entries, set(), lambda e: (e['family'], e['group']))
    assert [e['key'] for e in ordered] == ['a1', 'd1', 'b1', 'd2', 'a2']
    assert {id(e) for e in ordered} == {id(e) for e in entries}
    assert all(e['case_seconds'] == 1200 and e['max_model_requests'] == 672 for e in ordered)


def test_completed_attempts_remain_first_and_empty_queue_is_valid():
    entries = [dict(key='pending'), dict(key='done')]
    assert interleave_entries(entries, {'done'}, lambda e: ('family', 'source')) == entries[::-1]
    assert interleave_entries([], set(), lambda e: ('family', 'source')) == []


def test_duplicate_attempt_keys_fail_closed():
    with pytest.raises(ValueError, match='duplicate'):
        interleave_entries([dict(key='a'), dict(key='a')], set(), lambda e: ('f', 's'))
