import json
import pytest
from scripts.coreweave.retention_bookkeeping import own_bookkeeping,recheck_own_bookkeeping


def test_only_exact_self_records_are_exempt(tmp_path):
    d='a'*64;r={'role':'old','run':'r','name':'n','identity':{'sha256':d,'bytes':4}}
    phase=tmp_path/'phase.json';phase.write_text(json.dumps(r|{'upload':{'sha256':d,'bytes':4}}))
    cache=tmp_path/'cache.json';cache.write_text(json.dumps({'schema':'natlang.private-export-cache/1','identity':r['identity'],'receipt':r,'scope':'role old'}))
    review=tmp_path/'review.json';review.write_text(json.dumps({'schema':'natlang.standalone-retention-guard-review/1','candidates':[{'role':'old','sha256':d}]}))
    state=tmp_path/'state.json';state.write_text(json.dumps({'operations':[{'phase':'deleted','newer_role':'old','newer_sha256':d}]}))
    kw={'role':'old','digest':d,'receipt':r,'phase':phase,'cache':cache,'reviews':[review],'retirement_state':state};docs=own_bookkeeping(**kw);recheck_own_bookkeeping(docs,role='old',digest=d)
    value=json.loads(review.read_text());value['consumer']=d;review.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='another reference'):own_bookkeeping(**kw)
    with pytest.raises(ValueError,match='changed'):recheck_own_bookkeeping(docs,role='old',digest=d)


def test_only_exact_exporter_invocation_metadata_is_bookkeeping(tmp_path):
    from scripts.coreweave.retention_bookkeeping import own_exporter_metadata
    h='a'*64;p=tmp_path/'wandb-metadata.json';program=tmp_path/'exporter.py';v={'program':str(program),'args':['control','/work/'+h+'.tar'],'host':'x'};p.write_text(json.dumps(v))
    assert str(p) in own_exporter_metadata([p],role='control',source=v['args'][1],program=program,watched_digests=[h])
    assert not own_exporter_metadata([p],role='other',source=v['args'][1],program=program,watched_digests=[h])
    p.write_text(json.dumps(v|{'consumer':h}))
    with pytest.raises(ValueError,match='another reference'):own_exporter_metadata([p],role='control',source=v['args'][1],program=program,watched_digests=[h])


def test_exact_local_pin_dry_does_not_hide_other_consumers(tmp_path):
    from scripts.coreweave.retention_bookkeeping import own_pin_maintenance
    h='a'*64;p=tmp_path/'local-pin-maintenance-old.json';v={'schema':'natlang.local-pin-maintenance-dry/1','role':'old','no_unlink_performed':True,'candidate':{'sha256':h}};p.write_text(json.dumps(v))
    assert str(p) in own_pin_maintenance([p],role='old',digest=h)
    assert not own_pin_maintenance([p],role='different',digest=h)
    p.write_text(json.dumps(v|{'consumer':h}))
    with pytest.raises(ValueError,match='another reference'):own_pin_maintenance([p],role='old',digest=h)


def test_local_pin_intent_requires_exact_completed_unlink(tmp_path):
    from scripts.coreweave.retention_bookkeeping import own_completed_pin_intents
    h='a'*64;p=tmp_path/'local-pin-maintenance-x-intent.json';q=tmp_path/'local-pin-maintenance-x-result.json';v={'schema':'natlang.local-pin-exclusive-maintenance/1','role':'old','approved_sha256':h,'candidate':{'sha256':h,'path':'/work/pin','inode':4}};p.write_text(json.dumps(v))
    assert not own_completed_pin_intents([p],role='old',digest=h)
    result={'schema':'natlang.local-pin-exclusive-maintenance-result/1','intent':str(p),'status':'unlinked','remote':{'path':'/work/pin','inode':4,'unlinked':True}};q.write_text(json.dumps(result))
    assert set(own_completed_pin_intents([p],role='old',digest=h))=={str(p),str(q)}
    q.write_text(json.dumps(result|{'status':'pending'}));assert not own_completed_pin_intents([p],role='old',digest=h)
