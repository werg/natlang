import pytest
from scripts.coreweave.retention_live import live_facts


def test_actual_epoch_process_and_pending_state_are_bound():
    accepted={'accepted':True,'owner_id':'id','generation':1,'boot_id':'boot','pid':7,'process_start_ticks':'9'}
    owner={'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','parent_sha256':'z'*64,'parent_step':5,'accepted':accepted}
    e={'role':'old','step':6,'worker':2,'unit':'u'};new={'role':'new','step':8,'worker':2,'unit':'v'}
    intent={'sha256':'a'*64,'bytes':2,'step':6,'worker':2,'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','owner':accepted,'outer_credit':False};ni={**intent,'sha256':'b'*64,'step':8}
    def receipt(role,digest):return {'role':role,'private':True,'identity':{'sha256':digest,'bytes':2},'independent_verification':{'sha256':digest,'bytes':2}}
    args={'candidate':{'role':'old','sha256':'a'*64,'full_unit':'u'},'exports':[e,new],'intents':{'old':intent,'new':ni},'owners':[owner],'current_owner':owner,'owner_state':{'verified_roles':['new'],'pending':{'role':'old'}},'receipts':[receipt('old','a'*64),receipt('new','b'*64),receipt('fallback','c'*64)],'successor_role':'new','fallback_role':'fallback','active_units':[]}
    facts=live_facts(**args);assert facts['eligibility']['latest_role']=='new' and facts['eligibility']['pending_roles']==['old'] and 'z'*64 in facts['eligibility']['protected_sha256s']
    with pytest.raises(ValueError):live_facts(**(args|{'intents':{'old':intent|{'owner':accepted|{'pid':99}},'new':ni}}))
    with pytest.raises(ValueError):live_facts(**(args|{'candidate':args['candidate']|{'control_role':'fake'}}))
    with pytest.raises(ValueError,match='successor'):live_facts(**(args|{'intents':{'old':intent,'new':ni|{'step':4}}}))
