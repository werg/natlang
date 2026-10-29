import json
import pytest
from scripts.coreweave.retention_guard import guarded_pair


def test_pair_guard_dry_then_apply_and_fence_new_reference(tmp_path):
    root=tmp_path/'refs';root.mkdir();(root/'a.json').write_text('{}');journal=tmp_path/'journal';journal.write_text('{}')
    c={'role':'old','sha256':'a'*64,'control_role':'ctl','control_sha256':'b'*64,'full_unit':'u','control_unit':'v','successor_role':'new','successor_sha256':'c'*64,'fallback_role':'backup','fallback_sha256':'d'*64}
    scope={'allowed_paths':[],'journal_path':journal,'role':'old','intent_path':tmp_path/'intent'}
    eligibility={'latest_role':'new','latest_owner_role':'new','pending_roles':[],'protected_sha256s':[],'active_units':[],'verified_roles':['old','ctl','new','backup'],'successor_role':'new','fallback_role':'backup'}
    class Obj:
        def __init__(self,n):self.name=n;self.size=1
    class Api:
        def run(self,r):return self
        def files(self,*,names):return [Obj(n) for n in names]
    digests={'old':'a'*64,'ctl':'b'*64,'new':'c'*64,'backup':'d'*64}
    receipts=[{'role':x,'private':True,'run':'r','name':x,'identity':{'bytes':1,'sha256':digests[x]},'independent_verification':{'bytes':1,'sha256':digests[x],'readback_location':'owner'}} for x in digests]
    objects={'old':'a'*64,'ctl':'b'*64}
    def scan(digests,roots):return {x:set() for x in digests}
    args={'candidate':c,'roots':[root],'scan':scan,'reference_scope':scope,'read_live':lambda *a:{'eligibility':eligibility,'receipts':receipts},'api':Api(),'lock_path':tmp_path/'lock','state_path':tmp_path/'state','remove':lambda r,h:objects.pop(r),'present':lambda r,h:objects.get(r)==h,'write_durable':lambda p,v:p.write_text(json.dumps(v))}
    assert guarded_pair(**args)['deletion_authorized'] is False and len(objects)==2
    assert guarded_pair(**args,apply=True)['phase']=='deleted' and not objects
    def changed(digests,roots):
        (root/'new.json').write_text(digests[0]);return {x:set() for x in digests}
    with pytest.raises(ValueError,match='external dependency'):guarded_pair(**(args|{'scan':changed}))


def test_historical_full_only_retirement_never_accepts_a_control_record(tmp_path):
    from scripts.coreweave.retention_guard import guarded_full
    root=tmp_path/'refs';root.mkdir();journal=tmp_path/'journal';journal.write_text('{}')
    c={'role':'old','sha256':'a'*64,'full_unit':'u'}
    scope={'allowed_paths':[],'journal_path':journal,'role':'old','intent_path':tmp_path/'intent'}
    eligibility={'latest_role':'new','latest_owner_role':'new','pending_roles':[],'protected_sha256s':[],'active_units':[],'verified_roles':['old','new','backup'],'successor_role':'new','fallback_role':'backup'}
    class O:
        def __init__(self,n):self.name=n;self.size=1
    class Api:
        def run(self,r):return self
        def files(self,*,names):return [O(n) for n in names]
    ds={'old':'a'*64,'new':'c'*64,'backup':'d'*64};receipts=[{'role':r,'private':True,'run':'r','name':r,'identity':{'bytes':1,'sha256':h},'independent_verification':{'bytes':1,'sha256':h,'readback_location':'owner'}} for r,h in ds.items()]
    objects={'old':'a'*64};args={'candidate':c,'roots':[root],'scan':lambda d,r:{d[0]:set()},'reference_scope':scope,'read_live':lambda *a:{'eligibility':eligibility,'receipts':receipts},'api':Api(),'lock_path':tmp_path/'lock','state_path':tmp_path/'state','remove':lambda r,h:objects.pop(r),'present':lambda r,h:objects.get(r)==h,'write_durable':lambda p,v:p.write_text(json.dumps(v))}
    assert guarded_full(**args)['deletion_authorized'] is False
    with pytest.raises(ValueError,match='control-bearing'):guarded_full(**(args|{'candidate':c|{'control_required':True}}))
    assert guarded_full(**args,apply=True)['phase']=='deleted' and not objects
