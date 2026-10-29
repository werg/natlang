import json
import pytest
from scripts.coreweave.diloco_replica_controller import run_promotion


def test_controller_never_releases_before_exact_ready(tmp_path):
    intent={'run_id':'r','generation_nonce':'n'*48,'sequence':4,'head':'h'*64,'new_url':'https://new','new_token_sha256':'t'*64}
    fence=lambda i:[{'sandbox_id':'peer'}]
    verify=lambda i:{'run_id':'r','primary_fenced':True,'all_workers_quiesced':True,'proof_sha256':'p'*64}
    called=[]
    def release(i,f):called.append(True);return f
    with pytest.raises(ValueError):
        run_promotion(tmp_path,intent=intent,fence_workers=fence,verify_fence=verify,start_primary=lambda *_:{'ready':False},release_workers=release)
    assert not called and json.loads((tmp_path/'promotion.json').read_text())['phase']=='provider-fenced'
    with pytest.raises(ValueError):
        run_promotion(tmp_path,intent=intent,fence_workers=fence,verify_fence=verify,start_primary=lambda *_:{},release_workers=release)
    good=tmp_path/'good'
    ready={'ready':True,'run_id':'r','sequence':4,'head':'h'*64,'url':'https://new','token_sha256':'t'*64,'clock_rebase':{},'sealed_rebase':{'sequence':2,'head':'q'*64}}
    result=run_promotion(good,intent=intent,fence_workers=fence,verify_fence=verify,start_primary=lambda *_:ready,release_workers=release)
    assert result['phase']=='released' and called==[True]


def test_concrete_controller_binds_sdk_roles_and_ready_head(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.coreweave import cwctl,diloco_replica_fence as fm
    from scripts.coreweave.diloco_replica_controller import run_coreweave_promotion
    from natlang_neuralese.train.diloco import worker
    import hashlib
    token='new-secret-'+'x'*32;(tmp_path/'token').write_text(token)
    intent={'run_id':'r','generation_nonce':'n'*48,'sequence':4,'head':'h'*64,'new_url':'https://new','new_token_sha256':hashlib.sha256(token.encode()).hexdigest()}
    old={'sandbox_id':'old','boot_id':'b','owner_id':'o','pid':1,'start_ticks':'1'}
    entries=[{'sandbox_id':'peer','boot_id':'bb','owner_id':'oo','directory':'/control','runtime':'/runtime'}]
    monkeypatch.setattr(fm,'owner_reference',lambda *args:{'run_id':'r','workers':{'0':dict(old),'1':dict(entries[0])}})
    calls=[]
    monkeypatch.setattr(fm,'sdk_authorize_coordinator',lambda *a,**k:{})
    monkeypatch.setattr(fm,'sdk_set_worker_authority',lambda ws,nonce,action:(calls.append(action) or [{'sandbox_id':'peer'}]))
    monkeypatch.setattr(fm,'provider_fence_verifier',lambda **kw:lambda proof:{'run_id':'r','primary_fenced':True,'all_workers_quiesced':True,'proof_sha256':'p'*64})
    monkeypatch.setattr(cwctl,'load_env_file',lambda:None);monkeypatch.setattr(cwctl,'require_key',lambda:None)
    class Sandbox:
        def exec(self,*args,**kwargs):return SimpleNamespace(result=lambda **kw:SimpleNamespace(returncode=0,stdout='{"accepted":true}'))
    monkeypatch.setattr(cwctl,'read_sandbox',lambda *args,**kwargs:Sandbox())
    class Transport:
        def __init__(self,*args,**kwargs):pass
        def json(self,*args):return {'ready':True,'run_id':'r','sequence':4,'head':'h'*64,'url':'https://new','token_sha256':intent['new_token_sha256'],'clock_rebase':{},'sealed_rebase':{'sequence':2,'head':'q'*64}}
    monkeypatch.setattr(worker,'Transport',Transport)
    config={'intent':intent,'workers':entries,'expected_primary':old,'expected_workers':[0,1],'control_dirs':{},
        'supervisor_proof':{'journal':{'path':'p','sha256':'x'}},'new_token_file':str(tmp_path/'token'),
        'primary_role':{'sandbox_id':'target','directory':'/control','publisher':{'path':'/publisher','sha256':'a'*64},'request':{'action':'start','role':'coordinator'}}}
    assert run_coreweave_promotion(tmp_path/'journal',config)['phase']=='released'
    assert calls==['fence','release']
