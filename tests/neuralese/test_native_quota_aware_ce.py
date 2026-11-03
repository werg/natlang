import copy
import pytest
from natlang_neuralese.train.native_quota_aware_ce import make_basis,validate_admission


def test_quota_basis_rejects_changed_exposure_and_survives_admitted_resume():
 frame={'source_row_bindings':{'a':'binding-a','b':'binding-b'},'source_row_order':['a','b'],
        'required_visits':{'a':7,'b':1},'exposure_counts':{'a':1,'b':0},'source_receipts':[{'sha256':'parent'}]}
 basis=make_basis(frame,stage_id='sft',source_sha256='parent',candidate_sha256='child',epochs=1.,communication_min_visits=7)
 policy={'quota_aware_basis':basis};binding={'sha256':'policy'}
 args={'stage_id':'sft','candidate_sha256':'child','epochs':1.,'communication_min_visits':7,'policy_binding':binding}
 record=validate_admission(policy,frame,previous_options={},**args)
 changed=copy.deepcopy(frame);changed['exposure_counts']['a']=2
 with pytest.raises(ValueError,match='quotas or exposure changed'):validate_admission(policy,changed,previous_options={},**args)
 with pytest.raises(ValueError,match='settings'):validate_admission(policy,frame,previous_options={},**{**args,'communication_min_visits':6})
 with pytest.raises(ValueError,match='lacks recorded initial admission'):validate_admission(policy,changed,previous_options={'ce_supervision_policy_binding':binding},**args)
 assert validate_admission(policy,changed,previous_options={'ce_supervision_policy_binding':binding,'quota_aware_ce_history':[record]},**args)==record


def test_frame_digest_rejects_incomplete_quota_or_credit_inventory():
 frame={'source_row_bindings':{'a':'x'},'source_row_order':['a'],'required_visits':{},'exposure_counts':{},'source_receipts':[{'sha256':'p'}]}
 with pytest.raises(ValueError,match='inventories'):make_basis(frame,stage_id='s',source_sha256='p',candidate_sha256='c',epochs=1.,communication_min_visits=7)



def test_frame_digest_binds_retired_rows_historical_exposure():
 frame={'source_row_bindings':{'a':'x'},'source_row_order':['a'],'required_visits':{'a':1},'exposure_counts':{},'source_receipts':[{'sha256':'p'}]}
 kw=dict(stage_id='s',source_sha256='p',candidate_sha256='c',epochs=1.,communication_min_visits=7)
 plain=make_basis(frame,**kw)['parent_frame_sha256']
 retired=copy.deepcopy(frame);retired['required_visits']['gone']=1;retired['exposure_counts']['gone']=1
 digest=make_basis(retired,**kw)['parent_frame_sha256'];assert digest!=plain
 changed=copy.deepcopy(retired);changed['exposure_counts']['gone']=0
 assert make_basis(changed,**kw)['parent_frame_sha256']!=digest
 orphan=copy.deepcopy(frame);orphan['exposure_counts']['gone']=1
 with pytest.raises(ValueError,match='inventories'):make_basis(orphan,**kw)

def test_new_repeated_communication_continuation_requires_quota_basis():
 args={'stage_id':'s','candidate_sha256':'child','epochs':1.,'communication_min_visits':7,'policy_binding':{'sha256':'new'}}
 with pytest.raises(ValueError,match='requires a measured quota-aware basis'):
  validate_admission({},None,previous_options={'ce_supervision_policy_binding':{'sha256':'old'}},**args)
 assert validate_admission({},None,previous_options={'ce_supervision_policy_binding':{'sha256':'new'}},**args) is None
 assert validate_admission({},None,previous_options={'ce_supervision_policy_binding':{'sha256':'old'}},**{**args,'communication_min_visits':1}) is None


def test_remaining_mass_balance_obeys_cap_and_mean_one():
 from scripts.neuralese_data.prepare_quota_aware_acquired_continuation import quota_aware_plan
 plan=quota_aware_plan({'communication':700.,'ordinary':100.},{'communication':.4,'ordinary':.6},{'communication':-20.,'ordinary':-10.},cap=8.)
 assert plan['unscaled_remaining_mass']=={'communication':680.,'ordinary':90.}
 assert plan['effective_remaining_shares']==pytest.approx({'communication':.4,'ordinary':.6})
 assert plan['total_scaled_remaining_mass']==pytest.approx(770.)
 assert max(plan['scales'].values())<=8.
