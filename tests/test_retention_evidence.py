import json
import pytest
from scripts.coreweave.retention_evidence import external_references,control_retirement_blocked


def test_exact_own_bookkeeping_does_not_hide_other_entries_or_receipt_fields(tmp_path):
    h='a'*64;c='b'*64;intent=tmp_path/'intent.json';intent.write_text('{}');journal=tmp_path/'journal.json';receipt=tmp_path/'control.json';phase=tmp_path/'phase.json';other=tmp_path/'consumer.json';archive={'path':'/work/'+h+'-control.tar','sha256':c,'bytes':3};control={'role':'r-control','archive':archive};entry={'role':'r','intent':str(intent),'control':control};journal.write_text(json.dumps({'exports':[entry]}));receipt.write_text(json.dumps({'role':'r-control','source':archive['path'],'identity':{'sha256':c,'bytes':3},'name':'r-control/'+c+'/'+h+'-control.tar'}));phase.write_text(json.dumps({'role':'r-control','source':archive['path']}));refs={str(journal),str(receipt),str(phase)}
    def scan():return external_references(h,refs,allowed_paths=[],journal_path=journal,role='r',intent_path=intent,control=control,control_receipt=receipt,control_phase=phase)
    assert scan()==set()
    journal.write_text(json.dumps({'exports':[entry,{'role':'other','consumer':h}]}));assert scan()=={str(journal)}
    journal.write_text(json.dumps({'exports':[entry]}));value=json.loads(receipt.read_text());value['other_consumer']=h;receipt.write_text(json.dumps(value));assert scan()=={str(receipt)}
    value.pop('other_consumer');value['identity']['sha256']='c'*64;receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='exact candidate'):scan()
    other.write_text(h);refs.add(str(other));assert str(other) in refs


def test_pending_and_completed_control_retirements_never_retry():
    for phase in ('intent','control_deleted','deleted'):
        assert control_retirement_blocked({'operations':[{'phase':phase,'control_role':'c'}]},'c')
    assert control_retirement_blocked({'retired_roles':['c']},'c')
    assert not control_retirement_blocked({'operations':[{'phase':'other','control_role':'c'}]},'c')
def test_authenticated_owner_epoch_binds_runtime_and_parent(tmp_path):
    import hashlib,json
    from scripts.coreweave.retention_evidence import authenticated_owner_epochs,owner_for_intent
    p=tmp_path/'owner';o={'accepted':{'accepted':True},'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','parent_sha256':'a'*64,'parent_step':5};p.write_text(json.dumps(o));b={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    owners=authenticated_owner_epochs([b]);i={'sandbox_id':'s','plan_sha256':'p','runtime_commit':'r','step':6,'outer_credit':False}
    assert owner_for_intent(i,owners)==o
    import pytest
    with pytest.raises(ValueError):owner_for_intent({**i,'runtime_commit':'other'},owners)
    with pytest.raises(ValueError):owner_for_intent({**i,'step':5},owners)
    p.write_text('{}')
    with pytest.raises(ValueError,match='digest'):authenticated_owner_epochs([b])
