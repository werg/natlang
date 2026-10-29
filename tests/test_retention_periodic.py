import hashlib,json,types
from pathlib import Path
from scripts.coreweave import retention_periodic
from scripts.coreweave.retention_eligibility import eligible_full


def test_four_role_cycle_selects_actual_bound_candidate_and_keeps_dry(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE',str(tmp_path/'hash-cache.jsonl'))
    root=tmp_path/'state';ops=tmp_path/'ops';root.mkdir();ops.mkdir();(ops/'private-export-cache').mkdir()
    def put(p,v):p.write_text(json.dumps(v));return p
    accepted={'accepted':True,'owner_id':'id','generation':1,'boot_id':'boot','pid':7,'process_start_ticks':'9'};owner={'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','parent_sha256':'z'*64,'parent_step':5,'accepted':accepted}
    owners=[];roles=[]
    for rank in range(1,5):
        op=put(root/f'owner{rank}.json',owner);owners.append(op);roles.append({'rank':rank,'ownership':op.name})
    manifest=put(root/'roles.json',{'roles':roles});owner_state=put(root/'owner-state.json',{'verified_roles':['new']})
    ck=tmp_path/'checkpoint';ck.write_bytes(b'fallback');fallback_sha=hashlib.sha256(b'fallback').hexdigest();qual=put(root/'q.json',{'passed':True,'checkpoint':str(ck),'bytes':8,'sha256':fallback_sha})
    def receipt(role,sha):return {'role':role,'private':True,'run':'r','name':role,'identity':{'sha256':sha,'bytes':2},'independent_verification':{'sha256':sha,'bytes':2,'readback_location':'owner'}}
    fallback=receipt('backup',fallback_sha);put(ops/'coreweave-production-backup-private-export-receipt-20261014.json',fallback)
    base={'bytes':2,'worker':1,'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','owner':accepted,'outer_credit':False}
    exports=[]
    for role,step,sha in [('old',6,'a'*64),('new',8,'b'*64)]:
        intent=put(root/(role+'.json'),base|{'step':step,'sha256':sha});exports.append({'role':role,'step':step,'worker':1,'unit':role+'-unit','intent':str(intent),'phase_path':str(root/(role+'-phase.json'))});put(ops/('coreweave-production-'+role+'-private-export-receipt-20261014.json'),receipt(role,sha))
    journals=[]
    for rank in range(1,5):journals.append(put(root/f'journal{rank}.json',{'exports':exports if rank==1 else []}))
    config={'state_root':str(root),'operations_root':str(ops),'reference_roots':[str(root)],'fallback_role':'backup','qualified_local_parents':[str(qual)],'active_roles':str(manifest),'owner_state':str(owner_state),'roles':[{'rank':i+1,'journal':str(journals[i]),'owner_epochs':[str(owners[i])]} for i in range(4)]}
    monkeypatch.setattr(retention_periodic.subprocess,'run',lambda *a,**k:types.SimpleNamespace(stdout=''))
    monkeypatch.setattr(retention_periodic,'reference_paths',lambda ds,roots:{d:set() for d in ds})
    def guarded(**kw):
        assert not kw['apply'];live=kw['read_live'](kw['candidate'],'prepare');assert eligible_full(kw['candidate'],**live['eligibility']);return {'role':kw['candidate']['role'],'mode':'dry'}
    monkeypatch.setattr(retention_periodic,'guarded_full',guarded)
    assert retention_periodic.run_cycle(config,api=None,get=None,write_durable=lambda *a:None)==[{'role':'old','mode':'dry'}]
