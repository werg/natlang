import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.coreweave.diloco_replica_fence import owner_reference, verify_provider_fence


def test_provider_fence_exact_top_state_and_failures(tmp_path):
    primary={'sandbox_id':'old','boot_id':'boot','owner_id':'owner','pid':12,'start_ticks':'42'}
    worker={'sandbox_id':'peer','boot_id':'b','owner_id':'o','pid':13,'start_ticks':'43'}
    path=tmp_path/'owner.json'; path.write_text(json.dumps({'run_id':'run','coordinator':primary,'workers':{'0':worker}}))
    sha=hashlib.sha256(path.read_bytes()).hexdigest(); proof={'journal':{'path':str(path),'sha256':sha},'expected_run_id':'run'}
    def sandbox(state=3, sandbox_id='old'): return SimpleNamespace(sandbox_id=sandbox_id,status=SimpleNamespace(state=state,containers=[SimpleNamespace(state=2)]))
    def probe(identity,nonce): return {'identity':identity,'nonce':nonce,'process_absent':True,'owner_quiesced':True}
    def check(provider, worker_probe=probe): return verify_provider_fence(proof,expected_primary=primary,expected_workers=[0],get_sandbox=provider,probe_worker=worker_probe)
    assert check(lambda _:sandbox())['primary_fenced']
    for provider in (lambda _:sandbox(2),lambda _:sandbox(3,'different'),lambda _: (_ for _ in ()).throw(OSError('unavailable'))):
        with pytest.raises((ValueError,OSError)): check(provider)
    for changes in ({'nonce':'stale'},{'process_absent':False},{'owner_quiesced':False}):
        def bad(i,n): return dict(probe(i,n),**changes)
        with pytest.raises(ValueError): check(lambda _:sandbox(),bad)
    with pytest.raises(ValueError): owner_reference(path,sha,dict(primary,boot_id='old-boot'))
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError): check(lambda _:sandbox())


def test_actual_historical_owner_journal():
    path=Path('/home/werg/data/lfm3-sft-20261028/production-v3-live/four-rejoin-2512-ownership.json')
    if not path.is_file(): pytest.skip('host owner journal unavailable')
    journal=json.loads(path.read_text()); old=journal['coordinator']
    primary={key:old[key] for key in ('sandbox_id','boot_id','owner_id','pid','start_ticks')}
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    assert owner_reference(path,sha,primary)['run_id']==journal['run_id']
    with pytest.raises(ValueError): owner_reference(path,sha,dict(primary,owner_id='wrong'))
