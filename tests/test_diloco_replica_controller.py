import json
import pytest
from scripts.coreweave.diloco_replica_controller import run_promotion


def test_controller_never_releases_before_exact_ready(tmp_path):
    intent={'run_id':'r','generation_nonce':'n'*48,'activation_nonce':'x'*48,'sequence':4,'head':'h'*64,'new_url':'https://new','new_token_sha256':'t'*64}
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
    from natlang_neuralese.train.diloco import replica_transport
    import hashlib
    token='new-secret-'+'x'*32;(tmp_path/'token').write_text(token)
    intent={'run_id':'r','generation_nonce':'n'*48,'activation_nonce':'x'*48,'sequence':4,'head':'h'*64,'new_url':'https://new','new_token_sha256':hashlib.sha256(token.encode()).hexdigest()}
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
    monkeypatch.setattr(replica_transport,'witness_head',lambda *a,**k:{'run_id':'r','baseline_sha256':'a'*64,'sequence':4,'head':'h'*64})
    monkeypatch.setattr(replica_transport,'rotate_replica_credential',lambda *a,**k:{})
    (tmp_path/'old-token').write_text('o'*32)
    config={'intent':intent,'workers':entries,'expected_primary':old,'expected_workers':[0,1],'control_dirs':{},
        'supervisor_proof':{'journal':{'path':'p','sha256':'x'}},'new_token_file':str(tmp_path/'token'),
        'outgoing_replica':{'url':'https://replica','old_token_file':str(tmp_path/'old-token'),'baseline_sha256':'a'*64},'primary_role':{'sandbox_id':'target','directory':'/control','publisher':{'path':'/publisher','sha256':'a'*64},'request':{'action':'start','role':'coordinator'}}}
    assert run_coreweave_promotion(tmp_path/'journal',config)['phase']=='released'
    assert calls==['fence','release']


def test_coreweave_seed_stage_uses_direct_owned_roles_and_stops_source(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.coreweave import cwctl,diloco_replica_fence as fm
    from scripts.coreweave.diloco_replica_controller import run_coreweave_seed_stage
    from natlang_neuralese.train.diloco import replica_transport
    witness={'run_id':'r','baseline_sha256':'a'*64,'sequence':2,'head':'b'*64};calls=[];staged=[]
    class Future:
        def __init__(self,value):self.value=value
        def result(self,**kwargs):return self.value
    class Box:
        def __init__(self,name):self.name=name
        def write_file(self,path,data):staged.append((self.name,path,data));return Future(None)
        def exec(self,argv,**kwargs):
            calls.append((self.name,argv))
            if argv[0]=='env':return Future(SimpleNamespace(returncode=0,stdout='{}'))
            if argv[0]=='python3' and 'file_digest' in argv[2]:return Future(SimpleNamespace(returncode=0,stdout='c'*64+'\n'))
            if argv[0]=='python3' and 'runpy.run_path' in argv[2]:
                request=json.loads(argv[-1]);return Future(SimpleNamespace(returncode=0,stdout=json.dumps({'accepted':True,'owner_id':request['owner_id'],'generation':request['expected_generation']+1,'pid':42,'process_start_ticks':'7'})))
            return Future(SimpleNamespace(returncode=0,stdout=''))
    boxes={'source':Box('source'),'target':Box('target')}
    monkeypatch.setattr(cwctl,'load_env_file',lambda:None);monkeypatch.setattr(cwctl,'require_key',lambda:None)
    monkeypatch.setattr(cwctl,'read_sandbox',lambda sid,**k:boxes[sid])
    monkeypatch.setattr(fm,'sdk_authorize_seed_server',lambda role,nonce:calls.append(('authorize',role['request']['argv'][3])))
    monkeypatch.setattr(fm,'provider_fence_verifier',lambda **kw:lambda proof:{'primary_fenced':True,'all_workers_quiesced':True,'run_id':'r','proof_sha256':'d'*64})
    monkeypatch.setattr(replica_transport,'witness_head',lambda *a,**k:witness)
    monkeypatch.setattr(replica_transport,'freeze_replica_files',lambda *a,**k:calls.append(('freeze',a[2])))
    (tmp_path/'token').write_text('s'*32)
    common={'directory':'/control','boot_id':'boot','owner_id':'owner','generation':1,'runtime':'/runtime','python':'/usr/bin/python3','stage_directory':'/stage','publisher':{'path':'/publisher','sha256':'f'*64},'seed_token_file':'/seed-token','seed_directory':'/seed'}
    source={**common,'sandbox_id':'source','store':'/source-store','replica_url':'https://source','seed_host':'0.0.0.0','seed_port':8080,'seed_url':'https://seed','target_peer_token_file':'/peer-token'}
    target={**common,'sandbox_id':'target','store':'/target-store','work_root':'/workroot','replica_url':'https://target','peer_host':'0.0.0.0','peer_port':8081,'peer_token_file':'/peer-token','max_transaction_bytes':1024}
    result=run_coreweave_seed_stage({'expected_primary':{},'expected_workers':[0],'control_dirs':{},'supervisor_proof':{},'witness':witness,'activation_nonce':'n'*48,'source':source,'target':target,'source_token_file':str(tmp_path/'token'),'target_token_file':str(tmp_path/'token')})
    assert result['source_seed_server_stopped'] and result['witness']==witness
    assert ('authorize','seed-serve') in calls and ('authorize','peer') in calls
    assert all(b'sssssssss' not in raw for _,_,raw in staged)


def test_first_outer_source_evidence_binds_real_parent_and_source_bytes(tmp_path):
    import hashlib,torch
    from scripts.coreweave.diloco_replica_controller import verify_bootstrap_inputs
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
    from natlang_neuralese.train.diloco.replica_baseline import capture
    from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator
    root=tmp_path/'root';initialize_coordinator(root,run_id='new',masters={'w':torch.ones(4)},identity={'parent':'earned'},workers=1,min_quorum=1,fragment_bytes=1024,seed_source={'sha256':'a'*64})
    machine=ReplayCoordinator(root,quorum_since={})
    class Capture:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'baseline';receipt=capture(Capture(),baseline)
    source=tmp_path/'source.json';source.write_bytes(b'actual-source');proof=tmp_path/'receipt.json';proof.write_bytes(b'actual-receipt')
    kw=dict(baseline=baseline,baseline_sha256=receipt['sha256'],source_manifest=source,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_receipt=proof,receipt_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),expected_parent_sha256='a'*64)
    assert verify_bootstrap_inputs(**kw)['earned_parent_sha256']=='a'*64
    with pytest.raises(ValueError):verify_bootstrap_inputs(**dict(kw,expected_parent_sha256='b'*64))
