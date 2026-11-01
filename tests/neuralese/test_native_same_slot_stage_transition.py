from types import SimpleNamespace
from natlang_neuralese.train.native_state_sft import _same_slot_policy_to_sft_parent


def test_same_slot_transition_requires_explicit_policy_parent_and_sft_frame():
    args=SimpleNamespace(continue_from='checkpoint.pt',objective_continuation=True,objective_spec=None,stage_id='sft')
    saved={'identity':{'objective':{'kind':'policy'}},'sft_progress':{'sft':{'exposure_counts':{'a':1}}},'optimizer':{'x':1},'rng':{'x':1}}
    assert _same_slot_policy_to_sft_parent(args,saved)
    for key,value in [('objective_continuation',False),('objective_spec','rl.json'),('stage_id','other'),('continue_from',None)]:
        changed=SimpleNamespace(**vars(args));setattr(changed,key,value);assert not _same_slot_policy_to_sft_parent(changed,saved)
    assert not _same_slot_policy_to_sft_parent(args,{**saved,'identity':{'objective':None}})
    assert not _same_slot_policy_to_sft_parent(args,{**saved,'rng':{}})
