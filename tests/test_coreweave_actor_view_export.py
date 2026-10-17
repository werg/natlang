import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/coreweave/actor_view_export.py'
spec=importlib.util.spec_from_file_location('actor_view_export_tests',P);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
ROLE='mellum';SID='source-host';PARENT='a'*64;PRODUCER='b'*64


def fixture(tmp):
    source=tmp/'source';source.mkdir();pins=tmp/'pins';pins.mkdir();view=source/'view.pt';view.write_bytes(b'exact native tensor state')
    identity={'bytes':view.stat().st_size,'sha256':hashlib.sha256(view.read_bytes()).hexdigest()};full={'step':21,'sha256':PARENT,'bytes':123456}
    proof={'schema':'natlang.native-actor-view/1','step':21,'source':{'path':'/nonexistent/full48GB.pt',**full},'view':{'path':'view.pt',**identity},'mechanism':{'sha256':PRODUCER}}
    pp=source/'proof.json';pp.write_text(json.dumps(proof));ref={'path':str(pp),'bytes':pp.stat().st_size,'sha256':hashlib.sha256(pp.read_bytes()).hexdigest()}
    fullreceipt={'role':ROLE,'sandbox_id':SID,'private':True,'source':'/nonexistent/full48GB.pt','identity':{'sha256':PARENT,'bytes':123456},'independent_verification':{'sha256':PARENT,'bytes':123456}}
    state=tmp/'full-state.json';state.write_text(json.dumps({'last_verified_step':21,'receipt':fullreceipt,'actor_view':ref}));pointer=tmp/'view-state.json';calls=[]
    def observe(reference,parent):return m.verified_actor_view(reference,parent,ROLE,PRODUCER,pins)
    def export(chosen):
        calls.append(chosen)
        return {'role':ROLE,'sandbox_id':SID,'private':True,'source':chosen['source'],'identity':chosen['identity'],'independent_verification':chosen['identity']}
    return state,pointer,observe,export,calls,ref,full,pins,proof


def test_real_view_keeps_physical_and_full_parent_identities_distinct(tmp_path):
    state,pointer,observe,export,calls,*_=fixture(tmp_path)
    result=m.run_once(tmp_path/'control',ROLE,SID,state,pointer,observe=observe,export=export)
    assert result['inference_source']['training_checkpoint_sha256']==PARENT
    assert result['receipt']['identity']['sha256']!=PARENT
    assert result['derivation']['full_private_receipt']==json.loads(state.read_text())['receipt']
    assert Path(calls[0]['source']).read_bytes()==b'exact native tensor state'
    assert m.run_once(tmp_path/'control',ROLE,SID,state,pointer,observe=lambda *_:pytest.fail('duplicate observation'),export=lambda *_:pytest.fail('duplicate upload'))is None


def test_failed_get_recovers_same_pending_view_after_source_advances(tmp_path):
    state,pointer,observe,export,calls,*_=fixture(tmp_path);control=tmp_path/'control'
    def bad(chosen):
        result=export(chosen);result['independent_verification']={'sha256':'c'*64,'bytes':chosen['identity']['bytes']};return result
    with pytest.raises(ValueError,match='full GET'):m.run_once(control,ROLE,SID,state,pointer,observe=observe,export=bad)
    assert not pointer.exists()and(control/'mellum-actor-view-export-pending.json').exists()
    value=json.loads(state.read_text());value['last_verified_step']=22;value['actor_view']={'future':'view'};state.write_text(json.dumps(value))
    result=m.run_once(control,ROLE,SID,state,pointer,observe=lambda *_:pytest.fail('must reuse pending'),export=export)
    assert result['last_verified_step']==21 and calls[0]['identity']==calls[1]['identity']
    assert not(control/'mellum-actor-view-export-pending.json').exists()


@pytest.mark.parametrize('field,value',[('private',False),('role','lfm'),('sandbox_id','foreign')])
def test_unverified_full_backup_never_reaches_source_or_storage(tmp_path,field,value):
    state,pointer,observe,export,*_=fixture(tmp_path);doc=json.loads(state.read_text());doc['receipt'][field]=value;state.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='full training backup'):m.run_once(tmp_path/'control',ROLE,SID,state,pointer,observe=lambda *_:pytest.fail('unqualified source'),export=export)


@pytest.mark.parametrize('change',[{'source':{'sha256':'c'*64,'bytes':123456}},{'mechanism':{'sha256':'c'*64}},{'view':{'path':'../escape','bytes':1,'sha256':'c'*64}}])
def test_proof_rejects_wrong_parent_writer_or_escape(tmp_path,change):
    *_,ref,full,pins,proof=fixture(tmp_path);proof.update(change);pp=Path(ref['path']);pp.write_text(json.dumps(proof));ref.update(bytes=pp.stat().st_size,sha256=hashlib.sha256(pp.read_bytes()).hexdigest())
    with pytest.raises(ValueError):m.verified_actor_view(ref,full,ROLE,PRODUCER,pins)
    assert not list(pins.iterdir())


def test_same_size_corruption_does_not_publish_alias(tmp_path):
    *_,ref,full,pins,proof=fixture(tmp_path);view=Path(ref['path']).parent/'view.pt';view.write_bytes(b'x'*view.stat().st_size)
    with pytest.raises(ValueError,match='content changed'):m.verified_actor_view(ref,full,ROLE,PRODUCER,pins)
    assert not list(pins.iterdir())


def test_alias_symlink_and_conflict_remain_untouched(tmp_path):
    *_,ref,full,pins,proof=fixture(tmp_path);foreign=tmp_path/'foreign';foreign.write_bytes(b'foreign');alias=pins/'mellum-native-inference-step-21.pt';alias.symlink_to(foreign)
    with pytest.raises(ValueError,match='escaped'):m.verified_actor_view(ref,full,ROLE,PRODUCER,pins)
    assert alias.is_symlink()and foreign.read_bytes()==b'foreign'
    alias.unlink();alias.write_bytes(b'conflict')
    with pytest.raises(ValueError,match='conflicts'):m.verified_actor_view(ref,full,ROLE,PRODUCER,pins)
    assert alias.read_bytes()==b'conflict'


def test_old_pending_completion_cannot_roll_back_newer_pointer(tmp_path):
    state,pointer,observe,export,*_=fixture(tmp_path)
    def newer(chosen):
        receipt=export(chosen);pointer.write_text(json.dumps({'last_verified_step':22,'receipt':{'identity':{'sha256':'d'*64,'bytes':2}}}));return receipt
    assert m.run_once(tmp_path/'control',ROLE,SID,state,pointer,observe=observe,export=newer)is None
    assert json.loads(pointer.read_text())['last_verified_step']==22


def fake_source_host(monkeypatch):
    import subprocess,sys,types
    def execute(argv,**_):
        completed=subprocess.run([sys.executable,'-c',argv[-1]],text=True,capture_output=True)
        result=types.SimpleNamespace(returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr)
        return types.SimpleNamespace(result=lambda **_:result)
    sandbox=types.SimpleNamespace(exec=execute)
    monkeypatch.setitem(sys.modules,'cwctl',types.SimpleNamespace(load_env_file=lambda:None,require_key=lambda:None,read_sandbox=lambda *a,**k:sandbox))


def test_real_remote_verifier_rejection_has_safe_actionable_diagnostic(tmp_path,monkeypatch):
    state,pointer,*_=fixture(tmp_path);fake_source_host(monkeypatch);control=tmp_path/'control'
    args=['--role',ROLE,'--sandbox-id',SID,'--full-state',str(state),'--view-state',str(pointer),'--state-dir',str(control),'--producer-sha256','c'*64,'--pin-directory',str(tmp_path/'pins'),'--export-command','["unused"]','--once']
    with pytest.raises(m.ViewPublicationError,match='producer or full-parent'):m.main(args)
    error=json.loads((control/'mellum-actor-view-last-error.json').read_text())
    assert error['verifier_reason']=='actor view producer or full-parent identity differs'
    assert 'cmd'not in error and 'env'not in error and not pointer.exists()


def test_adapter_failure_persists_returncode_without_command_or_environment(tmp_path,monkeypatch):
    import subprocess,sys
    state,pointer,*_=fixture(tmp_path);fake_source_host(monkeypatch);control=tmp_path/'control'
    args=['--role',ROLE,'--sandbox-id',SID,'--full-state',str(state),'--view-state',str(pointer),'--state-dir',str(control),'--producer-sha256',PRODUCER,'--pin-directory',str(tmp_path/'pins'),'--export-command',json.dumps([sys.executable,'-c','raise SystemExit(7)']),'--once']
    with pytest.raises(subprocess.CalledProcessError):m.main(args)
    error=json.loads((control/'mellum-actor-view-last-error.json').read_text())
    assert error['subprocess_returncode']==7 and 'cmd'not in error and 'env'not in error
    assert(control/'mellum-actor-view-export-pending.json').exists()and not pointer.exists()
