"""The learned operators' soft prompt states as tune soft states (scripts/neuralese_data/operator_soft_states.py)."""
import pytest

import operator_soft_states
from operator_soft_states import binding, manifest_for, operator_turn
from natlang_neuralese.tune.rows import evidence_weight, verify_action_evidence
from natlang_neuralese.tune.soft import place_states, soft_key

VIEW = 'Read the value and keep what the instructions need.'
SITE = {'operator': 'compact-value', 'tools': [], 'messages': [
    {'role': 'system', 'content': VIEW}, {'role': 'user', 'content': 'instructions:\nRead this search result for: x\n\nvalue:\n...'}]}


def test_a_verified_seed_is_a_full_weight_row_whose_state_stands_where_view_body_stood():
    turn = operator_turn(SITE, 'the answer line', 'gold', case='c1', index=0)
    assert turn['binding'] == binding('compact-value') == 'prompt:view@compact-value'
    assert verify_action_evidence(turn) == 'admissible' and evidence_weight(turn) == 1.
    key = soft_key(turn['binding'], 'instructions', None)
    messages, placed = place_states(turn, {key: 'nzs1_' + 'a' * 52})
    assert placed == [key] and messages[0]['content'] == [{'type': 'native_state', 'id': 'nzs1_' + 'a' * 52}]
    assert messages[1] == SITE['messages'][1]


def test_a_mechanical_seed_is_only_weak_evidence_and_the_manifest_binds_each_operator():
    assert evidence_weight(operator_turn(SITE, 'x', 'mechanical:answer-lines', case='c1', index=0)) == .5
    manifest = manifest_for({binding('compact-value'), binding('compact-params')}, 'h')
    assert [item['id'] for item in manifest['bindings']] == ['prompt:view@compact-params', 'prompt:view@compact-value']
    assert manifest['base'] == {} and [node['id'] for node in manifest['tree']] == ['tree:']
    with pytest.raises(Exception):
        operator_turn({**SITE, 'messages': SITE['messages'][1:]}, 'x', 'gold', case='c', index=0)


def test_the_paired_report_splits_rows_and_pairs_within_them():
    losses = {'text': [2.0, 1.0, 4.0], 'primed': [1.5, 1.0, 5.0], 'trained': [1.0, 0.5, 6.0]}
    short = operator_soft_states.paired_report(losses, [0, 1])
    long = operator_soft_states.paired_report(losses, [2])
    assert short['mean_nll'] == {'text': 1.5, 'primed': 1.25, 'trained': 0.75}
    assert short['trained_vs_text'] == {'mean_change': -0.75, 'se': 0.25, 'lower_rows': 2, 'rows': 2}
    assert long['trained_vs_primed']['mean_change'] == 1.0 and long['trained_vs_primed']['lower_rows'] == 0
