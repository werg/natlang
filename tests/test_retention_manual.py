import pytest
from scripts.coreweave.retention_manual import normalize_manual
from scripts.coreweave.retention_select import candidates


def test_older_manual_is_eligible_only_after_verified_same_owner_successor():
    accepted={'accepted':True,'owner_id':'id','generation':1,'boot_id':'boot','pid':7,'process_start_ticks':'9'}
    owner={'sandbox_id':'s','plan_sha256':'p','runtime_commit':'runtime-123','source_sha256':'source','parent_sha256':'z'*64,'parent_step':5,'outer_credit':False,'accepted':accepted}
    binding={'sandbox_id':'s','plan_sha256':'p','runtime_commit':'runtime','source_sha256':'source','accepted':accepted}
    def entry(role,step):return {'role':role,'step':step,'worker':1,'unit':role+'-unit','owner_binding':binding}
    old,new=entry('old',6),entry('new',8)
    def raw(step,sha):return {'bytes':2,'worker':1,'sandbox_id':'s','source_sha256':'source','step':step,'sha256':sha}
    old_intent=normalize_manual(old,raw(6,'a'*64),[owner]);new_intent=normalize_manual(new,raw(8,'b'*64),[owner])
    def receipt(role,sha):return {'role':role,'private':True,'identity':{'sha256':sha,'bytes':2},'independent_verification':{'sha256':sha,'bytes':2,'readback_location':'owner'}}
    args={'exports':[old,new],'intents':{'old':old_intent,'new':new_intent},'owners':[owner],'current_owner':owner,'owner_state':{'verified_roles':['new']},'receipts':[receipt('old','a'*64),receipt('new','b'*64),receipt('backup','c'*64)],'fallback_role':'backup','active_units':[],'transitions':{}}
    result=candidates(**args);assert [x['role'] for x in result]==['old']
    assert not candidates(**(args|{'receipts':[receipt('old','a'*64),receipt('backup','c'*64)]}))
    assert not candidates(**(args|{'owner_state':{'verified_roles':['new'],'pending':{'role':'old'}}}))
    with pytest.raises(ValueError):normalize_manual(old,raw(6,'a'*64)|{'source_sha256':'other'},[owner])
    with pytest.raises(ValueError):normalize_manual(old|{'owner_binding':binding|{'accepted':accepted|{'pid':9}}},raw(6,'a'*64),[owner])
