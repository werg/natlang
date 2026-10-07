import copy

import pytest

from natlang_neuralese.train.exposure_curriculum import ExposureCurriculum


def alignment_report(pass_count, *, failing=None):
    strata={}
    for name in (f'pass-{pass_count-1}-length-short-start',
                 f'pass-{pass_count-1}-length-long-tail',
                 f'pass-{pass_count-1}-length-long-tail-last256'):
        strata[name]={'tokens':256,'ce_delta':.04,'embedding_mse_delta':.12,
                      'text_argmax_agreement':.94,'text_ce':.08}
    if failing:
        strata[failing]['text_argmax_agreement']=.75
    return {'step':2048,'strata':strata}


def test_exposure_depth_advances_only_after_all_matched_strata_are_ready():
    schedule=ExposureCurriculum()
    assert schedule.training_passes(1)==1
    assert schedule.training_passes(2)==2
    assert schedule.observe_alignment(alignment_report(3),foundation_schedule_passes=2)['advanced'] is False
    assert schedule.stage=='gold_passes_3'

    blocked=alignment_report(3,failing='pass-2-length-long-tail')
    result=schedule.observe_alignment(blocked,foundation_schedule_passes=3)
    assert result['advanced'] is False
    assert result['readiness']['strata']['pass-2-length-long-tail']['ready'] is False
    assert schedule.stage=='gold_passes_3'

    assert schedule.observe_alignment(alignment_report(3),foundation_schedule_passes=3)['stage']=='gold_passes_5'
    assert schedule.training_passes(1)==5
    assert schedule.observe_alignment(alignment_report(5),foundation_schedule_passes=3)['stage']=='gold_passes_8'
    assert schedule.training_passes(3)==8
    assert schedule.observe_alignment(alignment_report(8),foundation_schedule_passes=3)['stage']=='greedy_rollout'
    assert schedule.training_passes(3) is None


def test_exposure_state_roundtrips_and_rejects_threshold_or_history_changes():
    original=ExposureCurriculum(min_tokens_per_stratum=64)
    original.observe_alignment(alignment_report(3),foundation_schedule_passes=3)
    state=original.state_dict()
    resumed=ExposureCurriculum(min_tokens_per_stratum=64)
    resumed.load_state_dict(state)
    assert resumed.state_dict()==state
    with pytest.raises(ValueError,match='thresholds changed'):
        ExposureCurriculum(min_tokens_per_stratum=128).load_state_dict(state)
    corrupted=copy.deepcopy(state)
    corrupted['transitions'][0]['readiness']['ready']=False
    with pytest.raises(ValueError,match='promotion receipt'):
        ExposureCurriculum(min_tokens_per_stratum=64).load_state_dict(corrupted)


def test_greedy_stage_requires_actual_autoregressive_rollout_evidence():
    schedule=ExposureCurriculum(min_tokens_per_stratum=1)
    for pass_count in (3,5,8):
        assert schedule.observe_alignment(alignment_report(pass_count),
            foundation_schedule_passes=3)['advanced']
    assert schedule.observe_greedy({'generated_tokens':4})['completed'] is False
    assert schedule.observe_greedy({'greedy_autoregressive':True,'generated_tokens':0})['completed'] is False
    evidence={'greedy_autoregressive':True,'generated_tokens':4,'examples':2,
              'max_capacity_tokens':128,'real_close_targets':1,'truncated_examples':0}
    result=schedule.observe_greedy(evidence)
    assert result['completed'] is True
    assert 'no task or runtime admission' in result['scope']
    resumed=ExposureCurriculum(min_tokens_per_stratum=1)
    resumed.load_state_dict(schedule.state_dict())
    assert resumed.greedy_completion['generated_tokens']==4
