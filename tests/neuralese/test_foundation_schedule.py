"""Plateau-gated projection-first schedule state is deterministic and resumable."""
import copy

import pytest

from natlang_neuralese.train.foundation_schedule import ProjectionFirstSchedule


def test_projection_only_continues_until_both_held_heads_plateau():
    schedule = ProjectionFirstSchedule(min_evals=2, patience=2,
                                      min_relative_improvement=.02)
    outputs = []
    for shallow, full in [(1., 2.), (1., 1.9), (1., 1.9)]:
        outputs.append(schedule.observe({'shallow': shallow, 'full_depth': full}))
    assert all(row['phase'] == 'projection_only' for row in outputs)
    assert outputs[-1]['plateaus']['shallow']['error'] == 1.
    assert outputs[-1]['plateaus']['full_depth'] is None
    final = schedule.observe({'shallow': 1., 'full_depth': 1.9})
    assert final['phase'] == 'whole_transformer_adaptation'
    assert final['plateau_reached'] is True
    assert final['plateaus']['full_depth']['error'] == 1.9
    assert final['plateaus']['full_depth']['slope'] == pytest.approx(0.)
    assert final['backbone_lr_scale'] > 0
    assert final['projection_lr_scale'] == 1.


def test_significant_relative_improvement_resets_patience_and_records_slope():
    schedule = ProjectionFirstSchedule(min_evals=2, patience=2,
                                      min_relative_improvement=.05)
    schedule.observe({'shallow': 1., 'full_depth': 1.})
    schedule.observe({'shallow': 1., 'full_depth': .9})
    third = schedule.observe({'shallow': 1., 'full_depth': .84})
    assert third['phase'] == 'projection_only'
    # The latest improvement is significant versus the best, resetting this head.
    assert third['plateaus']['full_depth'] is None
    fourth = schedule.observe({'shallow': 1., 'full_depth': .84})
    assert fourth['phase'] == 'projection_only'
    fifth = schedule.observe({'shallow': 1., 'full_depth': .84})
    assert fifth['phase'] == 'whole_transformer_adaptation'
    assert fifth['plateaus']['full_depth']['best_error'] == pytest.approx(.84)
    assert fifth['plateaus']['full_depth']['slope'] == pytest.approx(0.)


def test_adaptation_ramps_backbone_lr_and_sequence_pass_count():
    schedule = ProjectionFirstSchedule(min_evals=1, patience=1,
                                      backbone_ramp_evals=3, pass_ramp_evals=2)
    # Both heads plateau on the second observation (first is the baseline).
    schedule.observe({'shallow': 1., 'full_depth': 2.})
    transition = schedule.observe({'shallow': 1., 'full_depth': 2.})
    assert transition['sequence_passes'] == 1
    assert transition['backbone_lr_scale'] == pytest.approx(1 / 3)
    assert transition['projection_lr_scale'] == 1.
    second = schedule.observe({'shallow': 1., 'full_depth': 2.})
    assert second['sequence_passes'] == 1
    assert second['backbone_lr_scale'] == pytest.approx(2 / 3)
    third = schedule.observe({'shallow': 1., 'full_depth': 2.})
    assert third['sequence_passes'] == 2
    assert third['backbone_lr_scale'] == 1.
    for _ in range(2):
        last = schedule.observe({'shallow': 1., 'full_depth': 2.})
    assert last['sequence_passes'] == 3
    assert last['backbone_lr_scale'] == 1.


@pytest.mark.parametrize(('maximum','expected'), [
    (4,[1,1,2,2,3,3,4,4,4]),
    (5,[1,1,2,2,3,3,4,4,5]),
])
def test_extended_projection_first_schedule_ramps_beyond_three(maximum, expected):
    schedule=ProjectionFirstSchedule(min_evals=1,patience=1,pass_ramp_evals=2,
                                     max_sequence_passes=maximum)
    values=[]
    for _ in range(2+len(expected)-1):
        controls=schedule.observe({'shallow':1.,'full_depth':1.})
        if controls['plateau_reached']:
            values.append(controls['sequence_passes'])
    assert values==expected
    assert controls['target_sequence_passes']==maximum


def test_extension_from_three_resumes_at_three_and_interrupt_resume_is_exact():
    options=dict(min_evals=1,patience=1,pass_ramp_evals=2)
    original=ProjectionFirstSchedule(**options,max_sequence_passes=3)
    for _ in range(8):
        original.observe({'shallow':1.,'full_depth':1.})
    assert original.controls()['adaptation_eval']==7
    legacy=original.state_dict()
    legacy['schema']='natlang.projection-first-schedule/1'
    legacy['config'].pop('max_sequence_passes')
    legacy.pop('depth_ramp_origin_eval')

    extended=ProjectionFirstSchedule(**options,max_sequence_passes=5)
    extended.load_state_dict(legacy)
    assert extended.controls()['sequence_passes']==3
    assert extended.controls()['adaptation_eval']==7
    assert extended.observe({'shallow':1.,'full_depth':1.})['sequence_passes']==3
    assert extended.observe({'shallow':1.,'full_depth':1.})['sequence_passes']==4
    interrupted=extended.state_dict()
    resumed=ProjectionFirstSchedule(**options,max_sequence_passes=5)
    resumed.load_state_dict(interrupted)
    assert resumed.observe({'shallow':1.,'full_depth':1.})==extended.observe({'shallow':1.,'full_depth':1.})
    assert resumed.observe({'shallow':1.,'full_depth':1.})['sequence_passes']==5


@pytest.mark.parametrize('maximum',[2,True,3.5])
def test_extended_projection_schedule_validates_maximum_depth(maximum):
    with pytest.raises(ValueError,match='max_sequence_passes'):
        ProjectionFirstSchedule(max_sequence_passes=maximum)


def test_state_dict_resume_reproduces_next_schedule_decision():
    options = dict(min_evals=2, patience=2, backbone_ramp_evals=3, pass_ramp_evals=2)
    original = ProjectionFirstSchedule(**options)
    for values in [(1., 2.), (1., 2.), (1., 2.), (1., 2.)]:
        expected = original.observe({'shallow': values[0], 'full_depth': values[1]})
    state = original.state_dict()
    resumed = ProjectionFirstSchedule(**options)
    resumed.load_state_dict(state)
    next_values = {'shallow': 1., 'full_depth': 2.}
    assert resumed.observe(next_values) == original.observe(next_values)
    # State is detached from caller mutation and round-trips through plain JSON values.
    snapshot = resumed.state_dict()
    snapshot['head_state']['shallow']['history'].append(7.)
    assert len(resumed.state_dict()['head_state']['shallow']['history']) == resumed.eval_count
    clone = ProjectionFirstSchedule(**options)
    clone.load_state_dict(copy.deepcopy(resumed.state_dict()))
    assert clone.state_dict() == resumed.state_dict()


def test_schedule_validates_observations_and_changed_resume_config():
    schedule = ProjectionFirstSchedule()
    with pytest.raises(ValueError, match='exactly'):
        schedule.observe({'shallow': .1})
    with pytest.raises(ValueError, match='invalid held'):
        schedule.observe({'shallow': -0.1, 'full_depth': .2})
    state = schedule.state_dict()
    other = ProjectionFirstSchedule(patience=4)
    with pytest.raises(ValueError, match='configuration changed'):
        other.load_state_dict(state)


def test_plateau_snapshot_is_retained_after_adaptation_starts():
    schedule = ProjectionFirstSchedule(min_evals=1, patience=1)
    schedule.observe({'shallow': 1., 'full_depth': 2.})
    transition = schedule.observe({'shallow': 1., 'full_depth': 2.})
    frozen = transition['plateaus']
    later = schedule.observe({'shallow': .5, 'full_depth': 1.})
    assert later['plateaus'] == frozen
    assert later['phase'] == 'whole_transformer_adaptation'
