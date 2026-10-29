import hashlib,json,subprocess
from types import SimpleNamespace
import pytest
from scripts.coreweave.diloco_full_restore_controller import stage_worker_checkpoints


def test_distinct_full_worker_transfer_retry_and_tamper(tmp_path):
    stage=tmp_path/'staged';stage.mkdir();components={};hosts={}
    for i in range(4):
        folder=stage/'workers'/str(i);folder.mkdir(parents=True);data=('distinct-full-state-'+str(i)).encode();(folder/'checkpoint.pt').write_bytes(data)
        components['worker-'+str(i)]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'global_step':10+i}
        hosts[i]={'sandbox_id':'host'+str(i),'checkpoint_slot':'/work/restore/worker'+str(i)+'.pt'}
    (stage/'staging.json').write_text(json.dumps({'schema':'natlang.diloco-staged-recovery/1','replica_baseline_staged':'baseline','worker_staged':{str(i):'workers/'+str(i) for i in range(4)},'components':components}))
    (stage/'replica-restored').mkdir();(stage/'diloco-replica-recovery').mkdir()
    closure=stage/'diloco-replica-recovery'/'recovery-closure.json';closure.write_text('{}')
    (stage/'replica-restored'/'verified-restore.json').write_text(json.dumps({'schema':'natlang.diloco-verified-restore/1','receipt_sha256':hashlib.sha256(closure.read_bytes()).hexdigest(),'worker_sha256':{str(i):components['worker-'+str(i)]['sha256'] for i in range(4)}}))
    class Future:
        def __init__(self,result):self.value=result
        def result(self,**kwargs):return self.value
    class Box:
        def __init__(self,name):self.base=tmp_path/name;self.base.mkdir()
        def path(self,value):return str(self.base/value.removeprefix('/work/')) if value.startswith('/work/') else value
        def exec(self,argv,**kwargs):
            argv=[self.path(x) for x in argv];return Future(subprocess.run(argv,capture_output=True,text=True))
        def write_file_streaming(self,path,chunks,**kwargs):
            with open(self.path(path),'wb') as f:
                for chunk in chunks:f.write(chunk)
            return Future(None)
    boxes={host['sandbox_id']:Box(host['sandbox_id']) for host in hosts.values()}
    sdk=SimpleNamespace(read_sandbox=lambda name,**kwargs:boxes[name])
    first=stage_worker_checkpoints(stage,hosts,cwctl=sdk)
    assert first==stage_worker_checkpoints(stage,hosts,cwctl=sdk)
    assert len({v['sha256'] for v in first.values()})==4
    (boxes['host1'].base/'restore'/'worker1.pt').write_bytes(b'bad')
    with pytest.raises(ValueError):stage_worker_checkpoints(stage,hosts,cwctl=sdk)


def test_restored_workers_start_only_after_ready_and_resume(tmp_path,monkeypatch):
    from scripts.coreweave.diloco_full_restore_controller import start_restored_workers
    from scripts.coreweave import diloco_replica_controller
    nonce='n'*48;promotion=tmp_path/'promotion.json';promotion.write_text(json.dumps({'phase':'primary-ready','intent':{'activation_nonce':nonce,'run_id':'run','new_url':'https://coordinator'}}))
    hosts={};pins={};alive={};starts=[]
    for i in range(4):
        argv=['python3','train.py','--out',f'/work/worker{i}','--diloco',json.dumps({'worker_id':i,'workers':4,'run_id':'run','coordinator_url':'https://coordinator'})];sha=hashlib.sha256(json.dumps(argv,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        hosts[i]={'sandbox_id':f'host{i}','directory':'/owner','owner_id':'o','boot_id':'b','argv':argv,'worker_argv_sha256':sha,'publisher':{'path':'/publisher','sha256':'a'*64}}
        pins[str(i)]={'sandbox_id':f'host{i}','path':f'/work/worker{i}/checkpoint.pt','sha256':'b'*64,'bytes':5}
    class Future:
        def __init__(self,value):self.value=value
        def result(self,**kwargs):return self.value
    class Box:
        def __init__(self,ident):self.ident=ident
        def exec(self,argv,**kwargs):
            if argv[0]=='cat':return Future(SimpleNamespace(returncode=0,stdout=json.dumps({'owner_id':'o','boot_id':'b','generation':0,'roles':{}})))
            if 'runpy.run_path' in argv[2]:
                request=json.loads(argv[-1]);alive[self.ident]=request['argv'];starts.append(self.ident)
                return Future(SimpleNamespace(returncode=0,stdout='{"accepted":true}'))
            return Future(SimpleNamespace(returncode=0,stdout=''))
    sdk=SimpleNamespace(read_sandbox=lambda ident,**kwargs:Box(ident))
    def capture(host,nonce):
        if host['sandbox_id'] not in alive:raise ValueError('not started')
        return {'worker_argv_sha256':host['worker_argv_sha256']}
    fences=SimpleNamespace(sdk_capture_live_worker=capture)
    monkeypatch.setattr(diloco_replica_controller,'capture_worker_ownership',lambda *args,**kwargs:{'captured':len(alive)})
    with pytest.raises(ValueError):start_restored_workers(tmp_path,hosts,pins,nonce=nonce,coordinator_role={},cwctl=sdk,fences=fences)
    promotion.write_text(json.dumps({'phase':'workers-released','intent':{'activation_nonce':nonce,'run_id':'run','new_url':'https://coordinator'}}))
    assert start_restored_workers(tmp_path,hosts,pins,nonce=nonce,coordinator_role={},cwctl=sdk,fences=fences)=={'captured':4}
    start_restored_workers(tmp_path,hosts,pins,nonce=nonce,coordinator_role={},cwctl=sdk,fences=fences)
    assert len(starts)==4


def test_full_restore_orchestrates_and_resumes_without_restarting(tmp_path,monkeypatch):
    from scripts.coreweave import diloco_full_restore_controller as ctl,diloco_replica_fence as fences,diloco_replica_controller as promotion,cwctl
    from natlang_neuralese.train.diloco import replica_recovery,replica_seed
    stage=tmp_path/'staged';(stage/'replica-restored').mkdir(parents=True);(stage/'diloco-replica-recovery').mkdir()
    (stage/'staging.json').write_text(json.dumps({'coordinator_run_id':'run'}))
    (stage/'replica-restored'/'verified-restore.json').write_text(json.dumps({'owner_journal_sha256':'o'*64,'receipt_sha256':'c'*64}))
    token=tmp_path/'token';token.write_text('secret');nonce='n'*48
    workers={str(i):{'sandbox_id':f'h{i}','directory':f'/owner{i}','owner_id':f'o{i}','boot_id':f'b{i}','runtime':'/runtime'} for i in range(4)}
    peers=[{**workers[str(i)],'store':f'/work/store{i}','url':f'https://h{i}'} for i in range(2)]
    config={'mode':'continue-existing-outer','staging_manifest_sha256':hashlib.sha256((stage/'staging.json').read_bytes()).hexdigest(),'staging':str(stage),'worker_hosts':workers,'peer_hosts':peers,'old_token_file':str(token),
        'promotion':{'intent':{'activation_nonce':nonce,'run_id':'run','sequence':0,'head':'h'},'supervisor_proof':{'journal':{'sha256':'o'*64}},
        'expected_primary':{},'expected_workers':list(range(4)),'full_restore':{'replacements':{str(i):{} for i in range(4)}},
        'primary_role':{'sandbox_id':'h0','directory':'/owner0'},'activation_config':{'baseline':'/work/store0/baseline','journal':'/work/store0/journal','file_custody_store':'/work/store0','baseline_sha256':'base'},'outgoing_replica':{'baseline_sha256':'base','url':'https://h1'}}}
    calls=[];monkeypatch.setattr(cwctl,'load_env_file',lambda:None);monkeypatch.setattr(cwctl,'require_key',lambda:None)
    monkeypatch.setattr(fences,'verify_full_restore_fence',lambda **kw:(calls.append('verify') or {'proof_sha256':'f'}))
    monkeypatch.setattr(fences,'sdk_set_worker_authority',lambda *a,**kw:calls.append('fence'))
    monkeypatch.setattr(ctl,'stage_worker_dependencies',lambda *a,**kw:(calls.append('dependencies') or {}))
    monkeypatch.setattr(ctl,'stage_worker_checkpoints',lambda *a,**kw:(calls.append('workers') or {str(i):{} for i in range(4)}))
    def seed(**kw):
        calls.append('seed');kw['seed'].mkdir();(kw['seed']/'seed-receipt.json').write_text('{}');return {'receipt_sha256':'s'}
    monkeypatch.setattr(replica_recovery,'prepare_restored_peer_seed',seed)
    seed_record={'run_id':'run','sequence':0,'head':'h','baseline_sha256':'base'}
    monkeypatch.setattr(replica_seed,'verify_seed',lambda *a,**kw:seed_record)
    monkeypatch.setattr(ctl,'stage_replica_seed',lambda *a,**kw:(calls.append('transfer') or {}))
    monkeypatch.setattr(ctl,'start_restored_peer',lambda host,**kw:(calls.append('peer') or {'sandbox_id':host['sandbox_id'],'directory':host['directory'],'publisher':{},'request':{},'preserve_role':{}}))
    def promote(directory,config):calls.append('promote');return {'phase':'workers-released'}
    monkeypatch.setattr(promotion,'run_coreweave_promotion',promote)
    monkeypatch.setattr(ctl,'start_restored_workers',lambda *a,**kw:(calls.append('start') or {'sha256':'owner'}))
    assert ctl.run_full_restore(tmp_path/'journal',config)['phase']=='complete'
    assert calls==['verify','fence','dependencies','workers','seed','transfer','peer','peer','promote','start']
    ctl.run_full_restore(tmp_path/'journal',config)
    assert calls==['verify','fence','dependencies','workers','seed','transfer','peer','peer','promote','start']
    changed=dict(config,mode='new-outer')
    with pytest.raises(ValueError):ctl.run_full_restore(tmp_path/'journal',changed)


def test_dependency_staging_preserves_exact_logical_paths_and_parent(tmp_path):
    from scripts.coreweave.diloco_full_restore_controller import stage_worker_dependencies
    stage=tmp_path/'stage';(stage/'dependencies'/'objects').mkdir(parents=True);data=b'actual-source';sha=hashlib.sha256(data).hexdigest();(stage/'dependencies'/'objects'/sha).write_bytes(data)
    parent=b'full-historical-parent';parent_sha=hashlib.sha256(parent).hexdigest();(stage/'parent.pt').write_bytes(parent)
    index={'schema':'natlang.diloco-recovery-dependencies/1','bindings':[{'path':'/work/source.json','sha256':sha,'bytes':len(data),'object':'objects/'+sha}]}
    index_path=stage/'dependencies'/'dependency-index.json';index_path.write_text(json.dumps(index))
    manifest={'dependency_index_sha256':hashlib.sha256(index_path.read_bytes()).hexdigest(),'parent':{'staged':'parent.pt','bytes':len(parent),'sha256':parent_sha},
        'components':{'worker-'+str(i):{'runtime_code_sha256':'a'*64} for i in range(4)}}
    (stage/'staging.json').write_text(json.dumps(manifest))
    class Future:
        def __init__(self,value):self.value=value
        def result(self,**kwargs):return self.value
    class Box:
        def __init__(self,name):self.base=tmp_path/name;self.base.mkdir()
        def map(self,v):return str(self.base) if v=='/work' else str(self.base/v.removeprefix('/work/')) if v.startswith('/work/') else v
        def exec(self,argv,**kwargs):
            if argv[0]=='env':return Future(SimpleNamespace(returncode=0,stdout=''))
            return Future(subprocess.run([self.map(v) for v in argv],capture_output=True,text=True))
        def write_file_streaming(self,path,chunks,**kwargs):
            with open(self.map(path),'wb') as f:
                for chunk in chunks:f.write(chunk)
            return Future(None)
    hosts={i:{'sandbox_id':f'h{i}','argv':['python3','train.py','--continue-from','/work/parent/checkpoint.pt'],'runtime':'/runtime','python':'python3'} for i in range(4)}
    boxes={v['sandbox_id']:Box(v['sandbox_id']) for v in hosts.values()};sdk=SimpleNamespace(read_sandbox=lambda name,**kw:boxes[name])
    result=stage_worker_dependencies(stage,hosts,cwctl=sdk)
    assert len(result)==4
    assert (boxes['h3'].base/'source.json').read_bytes()==data and (boxes['h3'].base/'parent/checkpoint.pt').read_bytes()==parent
    (boxes['h2'].base/'source.json').write_bytes(b'wrong')
    with pytest.raises(ValueError):stage_worker_dependencies(stage,hosts,cwctl=sdk)
