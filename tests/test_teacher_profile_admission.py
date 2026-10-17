"""Source/action pairing across independently declared student rendering views."""
import copy

import pytest

from scripts.coreweave.teacher_tranche_admission import (
    teacher_conversion_profile, verify_teacher_model_actions, verify_paired_teacher_actions,
)
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import teacher_action_graph_id


def view(profile=None, *, duplicate=False):
    raw = {'row': {'trajectory': [{}], 'task': {'program_ir': {'source_groups': ['train-g']}}}}
    digest = 'a' * 64
    identifier = 'text:' + teacher_action_graph_id('row', digest, 0, profile)
    graph = {'id': identifier, 'split': 'train', 'source_groups': ['train-g'],
        'conversion': {'teacher_source': {'source_artifact_sha256': 'b' * 64,
            'source_row_sha256': digest, 'source_row_id': 'row', 'teacher_turn_index': 0,
            'source_groups': ['train-g'],
            'decision_admission': {'materialized_decision_id': 'row:decision:0000'}}}}
    receipt = {'tokenizer': {'history_reasoning': {'policy': profile or 'keep'}},
        'outputs': {'train': {'rows': 0 if duplicate else 1}}, 'turns': 1,
        'deduplication': {'emitted_rows': 0 if duplicate else 1,
            'omitted_rows': 1 if duplicate else 0,
            'omissions': [{'id': identifier}] if duplicate else []}, 'action_omissions': []}
    if profile is not None:
        receipt['explicit_history_reasoning_policy'] = profile
    kwargs = {'raw_rows': raw, 'raw_row_shas': {'row': digest}, 'source_sha256': 'b' * 64,
              'decision_keys': {('row', 0)}, 'hold_ids': set()}
    return ([] if duplicate else [graph]), receipt, kwargs


def verified(profile=None, **options):
    graphs, receipt, kwargs = view(profile, **options)
    return verify_teacher_model_actions(graphs, receipt, **kwargs)


def test_effective_mellum_default_does_not_grant_explicit_profile_identity():
    graphs, receipt, kwargs = view()
    assert teacher_conversion_profile(receipt) is None
    result = verify_teacher_model_actions(graphs, receipt, **kwargs)
    assert result['graph_ids'] == {'text:row:source-row:' + 'a' * 64 + ':action:0000'}


@pytest.mark.parametrize('explicit', ['keep', 'last_turn_only'])
def test_pair_matches_original_actions_across_profile_specific_graph_ids(explicit):
    lf, mellum = verified(explicit), verified()
    proof = verify_paired_teacher_actions({'lfm25': lf, 'mellum21': mellum})
    assert proof == {'paired_model_graph_ids_equal': False,
        'paired_model_source_actions_equal': True, 'eligible_actions': 1, 'omitted_actions': 0}
    assert lf['eligible_actions'] == {('row', 'a' * 64, 0)}


def test_default_pair_compatibility_and_per_model_duplicate_accounting():
    assert verify_paired_teacher_actions({'lfm25': verified(), 'mellum21': verified()})['paired_model_graph_ids_equal']
    lf, mellum = verified('keep'), verified(duplicate=True)
    proof = verify_paired_teacher_actions({'lfm25': lf, 'mellum21': mellum})
    assert proof['paired_model_source_actions_equal']
    assert lf['partition']['emitted_actions'] == 1
    assert mellum['partition']['deduplicated_actions'] == 1


@pytest.mark.parametrize('profile', [None, False, 0, {}, 'unknown'])
def test_malformed_explicit_profile_cannot_become_a_default(profile):
    _, receipt, _ = view()
    receipt['explicit_history_reasoning_policy'] = profile
    with pytest.raises(ValueError, match='profile'):
        teacher_conversion_profile(receipt)


def test_profile_requires_the_same_actual_tokenizer_policy():
    _, receipt, _ = view('keep')
    receipt['tokenizer']['history_reasoning']['policy'] = 'last_turn_only'
    with pytest.raises(ValueError, match='tokenizer authority'):
        teacher_conversion_profile(receipt)


@pytest.mark.parametrize('change', ['legacy-id', 'wrong-sha', 'wrong-turn', 'wrong-groups', 'held-legacy', 'held-profile'])
def test_emitted_profile_cannot_change_source_or_escape_existing_holds(change):
    graphs, receipt, kwargs = view('keep')
    if change == 'legacy-id': graphs[0]['id'] = graphs[0]['id'].removesuffix(':history-reasoning:keep')
    elif change == 'wrong-sha': kwargs['raw_row_shas']['row'] = 'c' * 64
    elif change == 'wrong-turn': graphs[0]['conversion']['teacher_source']['teacher_turn_index'] = True
    elif change == 'wrong-groups': graphs[0]['source_groups'] = ['other']
    elif change == 'held-legacy': kwargs['hold_ids'] = {graphs[0]['id'].removesuffix(':history-reasoning:keep')}
    else: kwargs['hold_ids'] = {graphs[0]['id']}
    with pytest.raises(ValueError):
        verify_teacher_model_actions(graphs, receipt, **kwargs)


@pytest.mark.parametrize('change', ['unknown-id', 'double-duplicate', 'missing-duplicate'])
def test_dedup_profile_ids_must_conserve_exact_original_actions(change):
    graphs, receipt, kwargs = view('keep', duplicate=True)
    if change == 'unknown-id': receipt['deduplication']['omissions'][0]['id'] += ':changed'
    elif change == 'double-duplicate': receipt['deduplication']['omissions'] *= 2
    else: receipt['deduplication']['omissions'] = []
    with pytest.raises(ValueError):
        verify_teacher_model_actions(graphs, receipt, **kwargs)


def test_cross_model_join_cannot_drop_or_rebind_an_original_row():
    lf, mellum = verified('keep'), verified()
    for field, replacement in [('eligible_actions', {('row', 'c' * 64, 0)}),
                               ('omitted_actions', {('row', 'a' * 64, 0)})]:
        changed = copy.deepcopy(mellum); changed[field] = replacement
        with pytest.raises(ValueError, match='original source/action'):
            verify_paired_teacher_actions({'lfm25': lf, 'mellum21': changed})
