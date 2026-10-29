import pytest
from scripts.coreweave.retention_eligibility import eligible_pair


def test_pair_refuses_active_pending_current_or_unverified():
    c={'role':'old','sha256':'a'*64,'control_role':'ctl','control_sha256':'b'*64,'full_unit':'f-unit','control_unit':'c-unit'}
    baseline={'latest_role':'new','latest_owner_role':'new','pending_roles':[],'protected_sha256s':[],'active_units':[],'verified_roles':['old','ctl','new','fallback'],'successor_role':'new','fallback_role':'fallback'}
    assert eligible_pair(c,**baseline)
    for change in ({'latest_role':'old'},{'pending_roles':['ctl']},{'active_units':['f-unit']},{'protected_sha256s':['b'*64]},{'verified_roles':['old','ctl','new']}):
        with pytest.raises(ValueError):eligible_pair(c,**(baseline|change))
