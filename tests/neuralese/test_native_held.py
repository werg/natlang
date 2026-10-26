import copy

import pytest

from natlang_neuralese.train.native_held import (held_selection, commit_held_observations,
    bind_held_quality_exclusions)
from natlang_neuralese.common.hashing import canonical_json_sha256_hex


@pytest.mark.parametrize('final',[False,True])
@pytest.mark.parametrize('completed',[0,1,3])
def test_interrupted_held_counts_only_observed_rows_and_preserves_prior_visits(final,completed):
    entries=rows(4,4)
    _,previous,_=select(entries,graph_budget=2)
    indices,planned,selection=select(entries,previous,final=final,graph_budget=4)
    original=copy.deepcopy(previous)
    ids=[entries[i]['id']for i in indices[:completed]]
    actual,proof=commit_held_observations(entries,previous,planned,selection,ids)
    assert previous==original
    assert actual['graph_visits']==previous['graph_visits']+completed
    assert actual['forward_token_visits']==previous['forward_token_visits']+3*completed
    for kind in ('ordinary','communication'):
        pool=[row for row in entries if row['id'].startswith(kind)]
        assert actual['classes'][kind]['visits']==[
            previous['classes'][kind]['visits'][i]+(row['id']in ids)for i,row in enumerate(pool)]
    assert proof['observed_graphs']==completed and proof['operationally_interrupted']
    # A continued check must revisit every unobserved selected row fairly.
    _,continued,_=select(entries,actual,final=True)
    assert continued['completed_sweeps']>=1


def rows(ordinary=12, communication=12):
    return [{'id': f'{kind}-{i}', 'binding': f'{kind}-binding-{i}',
             'communication': kind == 'communication', 'forward_tokens': 3}
            for kind, count in [('ordinary', ordinary), ('communication', communication)]
            for i in range(count)]


def select(entries, coverage=None, **kwargs):
    return held_selection(entries, coverage, source_sha256='a'*64, **kwargs)


def test_both_classes_share_existing_twenty_graph_budget_and_resume_exactly():
    entries = rows(30, 30)
    indices, coverage, proof = select(entries, graph_budget=20)
    assert len(indices) == 20 and indices == sorted(indices)
    assert len(proof['classes']['ordinary']['selected_ids']) == 10
    assert len(proof['classes']['communication']['selected_ids']) == 10
    resumed = select(entries, copy.deepcopy(coverage), graph_budget=20)
    assert resumed == select(entries, coverage, graph_budget=20)
    assert not set(indices) & set(resumed[0])
    assert resumed[1]['graph_visits'] == 40


def test_legacy_cursor_preserves_every_prior_ordinary_visit_and_reaches_communication():
    entries = rows(3155, 100)
    old = {'cursor': 211, 'graph_visits': 211, 'completed_sweeps': 0,
           'forward_token_visits': 633}
    indices, coverage, proof = select(entries, old, graph_budget=20)
    ordinary = coverage['classes']['ordinary']['visits']
    assert ordinary[:211] == [1]*211
    assert ordinary[211:221] == [1]*10
    assert proof['classes']['communication']['selected_indices'] == list(range(3155, 3165))
    assert coverage['graph_visits'] == 231
    assert coverage['forward_token_visits'] == 693
    assert proof['legacy_coverage_migrated']
    assert len(indices) == 20


def test_append_retains_counts_and_prioritizes_new_rows_before_repeats():
    entries = rows(2, 2)
    _, coverage, _ = select(entries, final=True)
    original = copy.deepcopy(coverage)
    appended = entries + [{'id': 'ordinary-new', 'binding': 'new-o', 'forward_tokens': 3},
                          {'id': 'communication-new', 'binding': 'new-c',
                           'communication': True, 'forward_tokens': 3}]
    indices, after, _ = select(appended, coverage, graph_budget=2)
    assert indices == [4, 5]
    assert coverage == original
    for kind in ['ordinary', 'communication']:
        assert after['classes'][kind]['visits'] == [1, 1, 1]
    assert after['completed_sweeps'] == 1


def test_legacy_append_uses_actual_previous_source_size():
    entries = rows(2, 2)
    appended = entries + [{'id': 'ordinary-new', 'binding': 'new-o', 'forward_tokens': 3},
                          {'id': 'communication-new', 'binding': 'new-c',
                           'communication': True, 'forward_tokens': 3}]
    old = {'cursor': 2, 'graph_visits': 6, 'completed_sweeps': 1}
    indices, state, _ = select(appended, old, previous_result={'source_graphs': 4}, graph_budget=2)
    assert indices == [4, 5]
    assert state['classes']['ordinary']['visits'] == [2, 2, 1]
    assert state['classes']['communication']['visits'] == [1, 1, 1]
    assert state['graph_visits'] == 8


def test_quality_hold_skips_bad_held_rows_and_keeps_full_coverage_incomplete():
    entries = rows(4, 4)
    held = {'ordinary-1', 'communication-2'}
    indices, coverage, proof = select(entries, final=True, excluded_ids=held)
    assert indices == [0, 2, 3, 4, 5, 7]
    assert proof['quality_excluded_ids'] == sorted(held)
    assert coverage['source_graphs'] == 8
    assert coverage['quality_excluded_ids'] == sorted(held)
    assert coverage['classes']['ordinary']['visits'] == [1, 0, 1, 1]
    assert coverage['classes']['communication']['visits'] == [1, 1, 0, 1]
    assert coverage['completed_sweeps'] == 0


def test_quality_hold_transition_resets_only_excluded_visits_and_is_explicit():
    entries = rows(3, 3)
    _, coverage, _ = select(entries, final=True)
    with pytest.raises(ValueError,match='explicit held correction'):
        bind_held_quality_exclusions(entries,coverage,{'ordinary-1'},source_sha256='a'*64)
    rebound = bind_held_quality_exclusions(entries, coverage,
        {'ordinary-1', 'communication-2'}, source_sha256='a'*64, allow_transition=True)
    assert rebound['classes']['ordinary']['visits'] == [1, 0, 1]
    assert rebound['classes']['communication']['visits'] == [1, 1, 0]
    indices, after, proof = select(entries, rebound, graph_budget=4,
                                  excluded_ids={'ordinary-1', 'communication-2'})
    assert not ({1, 5} & set(indices))
    assert proof['quality_excluded_ids'] == ['communication-2', 'ordinary-1']
    with pytest.raises(ValueError, match='exclusion set differs'):
        select(entries, after, graph_budget=4, excluded_ids={'ordinary-1'})


def test_new_quality_exclusion_retires_exact_visits_and_token_total_idempotently():
    entries = rows(3, 2)
    _, coverage, _ = select(entries, final=True)
    assert coverage['forward_token_visits'] == 15
    before = copy.deepcopy(coverage)
    rebound = bind_held_quality_exclusions(entries, coverage, {'ordinary-1'},
        source_sha256='a'*64, allow_transition=True)
    assert coverage == before
    assert rebound['classes']['ordinary']['visits'] == [1, 0, 1]
    assert rebound['classes']['communication']['visits'] == [1, 1]
    assert rebound['graph_visits'] == 4
    assert rebound['forward_token_visits'] == 12
    assert rebound['retired_target_visit_history'] == [{
        'id':'ordinary-1','class':'ordinary',
        'old_binding_sha256':canonical_json_sha256_hex('ordinary-binding-1'),
        'visits':1,'forward_tokens_per_visit':3,'retired_forward_token_visits':3,
        'reason':'held target excluded from loss scoring'}]
    again = bind_held_quality_exclusions(entries, rebound, {'ordinary-1'},
        source_sha256='a'*64, allow_transition=True)
    assert again == rebound


@pytest.mark.parametrize('ordinary,communication', [(0, 5), (5, 0)])
def test_single_class_rotation_eventually_covers_every_row_and_final_is_full(ordinary, communication):
    entries = rows(ordinary, communication)
    coverage, visited = None, set()
    for _ in range(3):
        indices, coverage, _ = select(entries, coverage, graph_budget=2)
        visited.update(indices)
    assert visited == set(range(5))
    indices, coverage, proof = select(entries, coverage, final=True)
    assert indices == list(range(5))
    assert coverage['graph_visits'] == 11
    assert all(c['cursor'] == 0 for c in coverage['classes'].values())
    assert sum(len(c['selected_ids']) for c in proof['classes'].values()) == 5


def test_whole_long_graphs_and_class_minimum_are_explicit_when_budget_too_small():
    entries = rows(2, 2)
    entries[0]['forward_tokens'] = 100
    entries[2]['forward_tokens'] = 200
    indices, state, proof = select(entries, token_budget=10, graph_budget=1)
    assert indices == [0, 2]
    assert state['forward_token_visits'] == 300
    assert proof['whole_graph_class_minimum_over_budget']


def test_rebound_existing_rows_and_unaccounted_legacy_visits_fail_closed():
    entries = rows()
    _, state, _ = select(entries, graph_budget=2)
    rebound = copy.deepcopy(entries)
    rebound[0]['binding'] = 'changed'
    with pytest.raises(ValueError, match='identity/binding'):
        select(rebound, state, graph_budget=2)
    with pytest.raises(ValueError, match='historical rotation'):
        select(entries, {'cursor': 2, 'graph_visits': 99}, graph_budget=2)
