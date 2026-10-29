from scripts.coreweave.retention_select import candidates


def test_selects_same_epoch_only_and_preserves_protected_and_pending():
    accepted={'accepted':True,'owner_id':'id','generation':1,'boot_id':'b','pid':7,'process_start_ticks':'9'};owner={'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','parent_sha256':'z'*64,'parent_step':5,'accepted':accepted}
    def entry(role,step):return {'role':role,'step':step,'worker':2,'unit':role+'-unit'}
    exports=[entry('old',6),entry('new',8)];base={'bytes':2,'worker':2,'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','owner':accepted,'outer_credit':False}
    intents={'old':base|{'sha256':'a'*64,'step':6},'new':base|{'sha256':'b'*64,'step':8}}
    def receipt(role,sha):return {'role':role,'private':True,'identity':{'sha256':sha,'bytes':2},'independent_verification':{'sha256':sha,'bytes':2}}
    args={'exports':exports,'intents':intents,'owners':[owner],'current_owner':owner,'owner_state':{'verified_roles':['new']},'receipts':[receipt('old','a'*64),receipt('new','b'*64),receipt('backup','c'*64)],'fallback_role':'backup','active_units':[],'transitions':{}}
    result=candidates(**args);assert len(result)==1 and result[0]['role']=='old' and result[0]['successor_sha256']=='b'*64
    assert not candidates(**(args|{'protected_sha256s':['a'*64]}))
    assert not candidates(**(args|{'owner_state':{'pending':{'role':'old'},'verified_roles':['new']}}))
    assert not candidates(**(args|{'intents':intents|{'new':intents['new']|{'plan_sha256':'other'}}}))
