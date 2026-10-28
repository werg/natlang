import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/coreweave'))


def load(name):
    spec=importlib.util.spec_from_file_location('test_'+name,ROOT/'scripts/coreweave'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_pins_exact_worker_checkpoint_and_stable_real_coordinator(tmp_path):
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
    from natlang_neuralese.train.diloco.protocol import TensorSurface
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    pin=load('diloco_recovery_pin')
    model=torch.nn.Linear(2,2);surface=TensorSurface.from_module(model)
    root=tmp_path/'coordinator';manifest=initialize_coordinator(root,run_id='test',masters=surface.masters,
        identity={'parent':'seed'},workers=4,min_quorum=3,fragment_bytes=1024)
    package=tmp_path/'package';package.mkdir();(package/'__init__.py').write_text('')
    archive=tmp_path/'pins/coordinator.tar';coordinator=pin.coordinator_pin(root,archive,package)
    assert coordinator['fingerprint']==manifest['fingerprint'] and coordinator['versions']==[0]
    with tarfile.open(archive) as tar:
        assert 'diloco-coordinator/manifest.json' in tar.getnames()
        assert 'diloco-coordinator/fragments/00000.safetensors' in tar.getnames()
    checkpoint=tmp_path/'checkpoint.pt';worker=tmp_path/'pins/worker.pt'
    training={'schema':TRAINING_SCHEMA,'global_step':55,'optimizer':{'state':{'p':1}},'rng':{'x':1},
              'source_receipts':[{'sha256':'source'}],'source_hash':'source','exposure_counts':{'a':1},'options':{},
              'diloco':{'worker':2,'fingerprint':manifest['fingerprint'],'versions':[0],'sequences':[1],'pending':{}}}
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'source_architecture':'lfm3'},'training':training},checkpoint)
    value=pin.worker_pin(checkpoint,worker,2,manifest['fingerprint'],package)
    assert value['global_step']==55 and value['versions']==[0] and value['worker']==2
    replacement=tmp_path/'next.pt';torch.save({'new':True},replacement);replacement.replace(checkpoint)
    assert pin._hash(worker)==value['sha256'] and worker.stat().st_size==value['bytes']
    with pytest.raises(ValueError,match='full-state DiLoCo custody'):
        pin.worker_pin(worker,tmp_path/'wrong.pt',1,manifest['fingerprint'],package)


def test_coordinator_pin_rejects_mutating_inventory(tmp_path,monkeypatch):
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
    from natlang_neuralese.train.diloco.protocol import TensorSurface
    pin=load('diloco_recovery_pin');package=tmp_path/'package';package.mkdir();(package/'__init__.py').write_text('')
    root=tmp_path/'coordinator';surface=TensorSurface.from_module(torch.nn.Linear(2,2))
    initialize_coordinator(root,run_id='test',masters=surface.masters,identity={},workers=1,min_quorum=1,fragment_bytes=1024)
    actual=pin._inventory; calls=[]
    def changing(path):
        value=actual(path);calls.append(1)
        if len(calls)%2==0:value['new-file']=(1,2,3,4)
        return value
    monkeypatch.setattr(pin,'_inventory',changing)
    target=tmp_path/'pin/coordinator.tar'
    with pytest.raises(RuntimeError,match='changed during'):
        pin.coordinator_pin(root,target,package,attempts=2)
    assert not target.exists()


def test_periodic_recovery_retries_missing_worker_and_preserves_topology(tmp_path):
    service=load('diloco_periodic_recovery');record={'run':{'id':'run'}}
    counts={'coord':0,'export':0};fail=[True]
    class Backend:
        workers=['s0','s1','s2','s3']
        def capture_fenced(self,cycle):raise service.FenceUnavailable('fence unavailable on recorded runtime')
        def pin_worker(self,i,cycle):
            if i==3 and fail[0]:raise RuntimeError('not yet available')
            return {'worker':i,'sandbox_id':self.workers[i],'path':'/pin','sha256':str(i)*64,'bytes':5,
                    'global_step':50+i,'fingerprint':'f','versions':[1],'sequences':[1],'pending':{},'awaiting_global':[],'source_hash':'s','runtime_code_sha256':'code',
                    'resident_generation':'gen','recipe_binding':'recipe','ce_supervision_policy_binding':'ce'}
        def pin_resident(self,i,cycle):
            return {'sandbox_id':self.workers[i],'path':'/resident','sha256':str(i)*64,'bytes':5,'generation':'gen'}
        def pin_coordinator(self,cycle):
            counts['coord']+=1
            return {'sandbox_id':'s0','path':'/coord','sha256':'a'*64,'bytes':10,'fingerprint':'f',
                    'versions':[2],'last_seq':[{'0':1,'1':1,'2':1,'3':1}],'queued':[{}],'runtime_code_sha256':'code'}
        def export(self,name,intent):counts['export']+=1;return {'verified':True,'name':name}
        def prepare_dependencies(self,cycle):return {'sha256':'d'*64,'path':'/deps','bytes':5}
        def export_dependencies(self,intent):return {'verified':True,'name':'deps'}
        def present(self,receipt):return True
        def progress(self,i):return {'global_step':60+i}
    state_path=tmp_path/'state.json';backend=Backend()
    first=service.run_once(record,state_path,backend)
    assert first['pending']['phase']=='incomplete' and first['latest_verified'] is None
    assert first['pending']['unfenced']['scope']=='opportunistic sequence-checked component pins'
    assert set(first['pending']['receipts'])=={'worker-0','worker-1','worker-2','resident-0','resident-1','resident-2','resident-3','coordinator'}
    fail[0]=False;second=service.run_once(record,state_path,backend)
    assert second['latest_verified'] is None
    cohort=second['cycles'][-1]
    assert cohort['phase']=='verified-components-and-dependencies' and len(cohort['receipts'])==9
    assert 'restore' in cohort['recovery_blocker'] and cohort['dependency_receipt']['verified']
    assert len(cohort['superseded_coordinator_pins'])==1 and counts['coord']==2
    assert cohort['observed_progress']['0']['updates_since_pinned_checkpoint']==10
    assert 'not a consolidated global checkpoint' in cohort['compatibility']['scope']
    changed={**cohort['components'],'worker-2':{**cohort['components']['worker-2'],'versions':[3]}}
    assert service.compatibility(changed)['compatible'] is False
    ahead={**cohort['components'],'coordinator':{**cohort['components']['coordinator'],'last_seq':[{'0':2,'1':1,'2':1,'3':1}]}}
    assert 'uncheckpointed' in service.compatibility(ahead)['reason']
    missing={**cohort['components'],'worker-0':{**cohort['components']['worker-0'],'sequences':[2],'awaiting_global':[0]}}
    assert 'lacks accepted' in service.compatibility(missing)['reason']
    pending={'sequence':1,'delta_sha256':'a'}
    conflict={**cohort['components'],'worker-0':{**cohort['components']['worker-0'],'pending':{'0':pending}},
              'coordinator':{**cohort['components']['coordinator'],'queued':[{'0':{'sequence':1,'delta_sha256':'b'}}]}}
    assert 'payload differs' in service.compatibility(conflict)['reason']


def test_private_full_state_export_binds_box_hash_get_and_local_copy(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace
    service=load('diloco_periodic_recovery');backend=object.__new__(service.Backend);backend.storage=tmp_path
    payload=b'checkpoint';sha=hashlib.sha256(payload).hexdigest()
    intent={'sandbox_id':'actual','path':'/work/pin/checkpoint.pt','sha256':sha,'bytes':len(payload),'global_step':51}
    class Result:
        returncode=0
        def result(self):return self
    monkeypatch.setattr(service.cwctl,'read_sandbox',lambda ident:SimpleNamespace(exec=lambda args:Result()))
    def run(argv,env,check):
        assert env['CW_EXPORT_SANDBOX_ID']=='actual' and env['CW_EXPORT_VERIFY_ON_OWNER']=='1'
        assert env['CW_EXPORT_EXPECTED_SHA256']==sha and env['CW_EXPORT_EXPECTED_BYTES']==str(len(payload))
        dest=Path(env['CW_EXPORT_OWNER_DESTINATION']);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
        result={'private':True,'source':intent['path'],'sandbox_id':'actual','identity':{'sha256':sha,'bytes':len(payload)},
                'independent_verification':{'sha256':sha,'bytes':len(payload),'local_copy':{'committed':True,'sha256':sha}},
                'verified_at':1,'run':'private','name':'object'}
        Path(env['CW_EXPORT_RECEIPT_FILE']).write_text(json.dumps(result))
    monkeypatch.setattr(service.subprocess,'run',run)
    assert backend.export('worker-0',intent)['identity']['sha256']==sha
    def invalid(argv,env,check):
        run(argv,env,check);p=Path(env['CW_EXPORT_RECEIPT_FILE']);value=json.loads(p.read_text())
        value['independent_verification']['sha256']='f'*64;p.write_text(json.dumps(value))
    monkeypatch.setattr(service.subprocess,'run',invalid)
    with pytest.raises(ValueError,match='independent complete-object GET'):
        backend.export('worker-0',intent)



def test_resident_pin_records_actual_committed_control_and_rejects_wrong_plan(tmp_path):
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    pin = load('diloco_recovery_pin')
    directory = tmp_path / 'resident-data'; directory.mkdir()
    plan = {'recipe': {'id': 'test'}}
    (tmp_path / 'recipe-plan.json').write_text(json.dumps(plan))
    (directory / 'initial.json').write_text(json.dumps({'plan_sha256': canonical_json_sha256_hex(plan)}))
    result = pin.resident_pin(directory, tmp_path / 'control.tar')
    assert result['generation'] == canonical_json_sha256_hex(plan)
    with tarfile.open(result['path']) as tar:
        assert 'diloco-resident/recipe-plan.json' in tar.getnames()
        assert 'diloco-resident/resident-data/initial.json' in tar.getnames()
    (tmp_path / 'recipe-plan.json').write_text('{}')
    with pytest.raises(ValueError, match='active plan differ'):
        pin.resident_pin(directory, tmp_path / 'bad.tar')


def test_dependency_seal_requires_exact_bytes_and_rechecks_archive(tmp_path):
    import hashlib
    module=load('diloco_recovery_dependencies')
    source=tmp_path/'input';source.write_bytes(b'actual source')
    code=tmp_path/'code.tar'
    module_file=tmp_path/'module.py';module_file.write_bytes(b'runtime')
    with tarfile.open(code,'w') as tar:tar.add(module_file,arcname='training/neuralese/natlang_neuralese/module.py')
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    runtime=canonical_json_sha256_hex({'module.py':hashlib.sha256(b'runtime').hexdigest()})
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    bundle=tmp_path/'bundle.json';sources=tmp_path/'sources.json'
    bundle.write_text(json.dumps({'code':{'key':'code','sha256':sha(code),'bytes':code.stat().st_size},'files':[]}))
    sources.write_text(json.dumps({'code':str(code)}))
    result=module.seal([{'path':'/logical/input','sha256':sha(source)}],{sha(source):[str(source)]},bundle,sources,tmp_path/'sealed.tar',runtime_code_sha256=runtime,reserve_bytes=0)
    assert result['objects']==2 and result['bindings']==2
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='missing exact dependency'):
        module.seal([{'path':'/logical/input','sha256':sha(code)}],{},bundle,sources,tmp_path/'bad.tar',runtime_code_sha256=runtime,reserve_bytes=0)


def test_http_restore_continues_saved_optimizer_rng_exposure_without_replay(tmp_path):
    import socket, threading, time
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator
    from natlang_neuralese.train.diloco.protocol import TensorSurface
    from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport
    pin=load('diloco_recovery_pin');torch.manual_seed(123)
    model=torch.nn.Linear(2,2);surface=TensorSurface.from_module(model);root=tmp_path/'coord'
    initialize_coordinator(root,run_id='recover',masters=surface.masters,identity={'parent':'exact'},
                           workers=1,min_quorum=1,fragment_bytes=1024,grace_seconds=0,outer_lr=1.0,outer_momentum=0)
    token='local-http-test-token-at-least-32-characters'
    def server(root):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        stop=threading.Event();thread=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,token),kwargs={'stop':stop},daemon=True);thread.start()
        url=f'http://127.0.0.1:{port}';transport=Transport(url,token,5)
        for _ in range(100):
            try:transport.json('GET','/v1/status');break
            except OSError:time.sleep(.02)
        return url,stop,thread
    url,stop,thread=server(root)
    settings=DilocoSettings(run_id='recover',workers=1,worker_id=0,coordinator_url=url,local_steps=2,
                            wire_dtype='fp32',allow_private_http=True,start_grace_seconds=0,heartbeat_seconds=.05)
    learner=DilocoLearner(settings,surface=surface,identity={'parent':'exact'},output=tmp_path/'worker',transport=Transport(url,token,5))
    learner.start();optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
    optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step();learner.after_step(tokens=4)
    saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
           'exposure':{'row':1},'diloco':learner.state_dict()}
    checkpoint=tmp_path/'full.pt';torch.save(saved,checkpoint)
    package=tmp_path/'package';package.mkdir();(package/'__init__.py').write_text('')
    archive=tmp_path/'snapshot.tar';coord=pin.coordinator_pin(root,archive,package)
    assert coord['versions']==saved['diloco']['versions'] and coord['last_seq']==[{}]
    learner.close();stop.set();thread.join(timeout=3)
    restore=tmp_path/'restore';restore.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(restore,filter='data')
    restored_root=restore/'diloco-coordinator';url2,stop2,thread2=server(restored_root)
    state=torch.load(checkpoint,weights_only=True);restored_model=torch.nn.Linear(2,2);restored_model.load_state_dict(state['model'])
    restored_optimizer=torch.optim.AdamW(restored_model.parameters(),lr=.01);restored_optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng']);assert torch.equal(torch.get_rng_state(),saved['rng'])
    settings2=DilocoSettings(run_id='recover',workers=1,worker_id=0,coordinator_url=url2,local_steps=2,
                             wire_dtype='fp32',allow_private_http=True,start_grace_seconds=0,heartbeat_seconds=.05)
    restored=DilocoLearner(settings2,surface=TensorSurface.from_module(restored_model),identity={'parent':'exact'},
                           output=tmp_path/'restored-worker',transport=Transport(url2,token,5))
    try:
        restored.resume(state['diloco'])
        restored_optimizer.zero_grad();restored_model(torch.ones(1,2)).sum().backward();restored_optimizer.step()
        state['exposure']['row']+=1;restored.after_step(tokens=4)
        assert restored.finish() and state['exposure']=={'row':2}
        assert restored_optimizer.state_dict()['state'] and restored.sequences[0]==1
        assert Transport(url2,token,5).json('GET','/v1/status')['versions'][0]==1
    finally:
        restored.close();stop2.set();thread2.join(timeout=3)


def test_four_worker_http_recovery_preserves_lost_workers_full_state(tmp_path):
    import socket, threading, time
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator
    from natlang_neuralese.train.diloco.protocol import TensorSurface
    from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport
    torch.manual_seed(123);base=torch.nn.Linear(2,2);root=tmp_path/'coord'
    initialize_coordinator(root,run_id='four',masters=TensorSurface.from_module(base).masters,identity={'parent':'exact'},
                           workers=4,min_quorum=3,fragment_bytes=1024,grace_seconds=0,outer_lr=1.0,outer_momentum=0)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    token='local-four-worker-token-at-least-32-characters';url=f'http://127.0.0.1:{port}'
    stop=threading.Event();thread=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,token),kwargs={'stop':stop},daemon=True);thread.start()
    transport=Transport(url,token,5)
    for _ in range(100):
        try:transport.json('GET','/v1/status');break
        except OSError:time.sleep(.02)
    learners=[]
    for worker in range(4):
        model=torch.nn.Linear(2,2);model.load_state_dict(base.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        settings=DilocoSettings(run_id='four',workers=4,worker_id=worker,coordinator_url=url,local_steps=2,
                                wire_dtype='fp32',allow_private_http=True,start_grace_seconds=0,heartbeat_seconds=.05)
        learner=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity={'parent':'exact'},
                              output=tmp_path/f'worker-{worker}',transport=Transport(url,token,5))
        learners.append([model,optimizer,learner,settings])
    threads=[threading.Thread(target=item[2].start) for item in learners]
    for item in threads:item.start()
    for item in threads:item.join(timeout=10);assert not item.is_alive()
    try:
        for model,optimizer,learner,_ in learners:
            optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step();learner.after_step(tokens=4)
        model,optimizer,lost,settings=learners[0]
        saved={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state(),
               'exposure':{'worker-0-row':1},'diloco':lost.state_dict()}
        checkpoint=tmp_path/'worker0-full.pt';torch.save(saved,checkpoint);lost.close()
        state=torch.load(checkpoint,weights_only=True);replacement=torch.nn.Linear(2,2);replacement.load_state_dict(state['model'])
        replacement_optimizer=torch.optim.AdamW(replacement.parameters(),lr=.01);replacement_optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['rng']);assert torch.equal(torch.get_rng_state(),saved['rng'])
        resumed=DilocoLearner(settings,surface=TensorSurface.from_module(replacement),identity={'parent':'exact'},
                             output=tmp_path/'worker-0-restored',transport=Transport(url,token,5))
        resumed.resume(state['diloco']);learners[0]=[replacement,replacement_optimizer,resumed,settings]
        for model,optimizer,learner,_ in learners:
            optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step();learner.after_step(tokens=4)
        state['exposure']['worker-0-row']+=1
        assert state['exposure']=={'worker-0-row':2} and replacement_optimizer.state_dict()['state']
        finishers=[threading.Thread(target=item[2].finish) for item in learners]
        for item in finishers:item.start()
        for item in finishers:item.join(timeout=10);assert not item.is_alive()
        assert transport.json('GET','/v1/status')['versions'][0]>=1
        assert resumed.sequences[0]>=1
    finally:
        for _,_,learner,_ in learners:learner.close()
        stop.set();thread.join(timeout=3)


def test_fence_operator_releases_every_worker_after_capture_or_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    service=load('diloco_periodic_recovery');backend=object.__new__(service.Backend)
    backend.workers=['s0','s1','s2','s3'];backend.fingerprint='fingerprint';backend.out='/work/diloco-learner';backend.repo=tmp_path
    events=[]
    class Result:
        def __init__(self,value=None):self.value=value;self.returncode=0
        def result(self):return self.value if self.value is not None else self
    class Box:
        def __init__(self,worker):
            self.worker=worker;self.files={'/work/diloco-learner/diloco/recovery-fence/capabilities.json':json.dumps({'schema':'natlang.diloco-recovery-fence/1','worker':worker,'fingerprint':'fingerprint','owner':{'pid':1,'boot_id':'test','start_ticks':'1'}})}
        def exec(self,args,**kwargs):
            if args[0]=='mv':
                request=json.loads(self.files[args[1]]);self.files[args[2]]=self.files[args[1]]
                ready={**request,'checkpoint':f"/work/diloco-learner/diloco/recovery-fence/pins/{request['id']}/checkpoint.pt",'global_step':100,'local_step':100,
                       'versions':[0],'sequences':[0],'bytes':4}
                self.files[f"/work/diloco-learner/diloco/recovery-fence/{request['id']}.ready.json"]=json.dumps(ready)
            return Result()
        def write_file(self,path,data):self.files[path]=data;events.append((self.worker,path));return Result()
        def read_file(self,path):
            if path not in self.files:raise FileNotFoundError(path)
            return Result(self.files[path])
    boxes={ident:Box(i) for i,ident in enumerate(backend.workers)}
    monkeypatch.setattr(service.cwctl,'read_sandbox',lambda ident:boxes[ident])
    monkeypatch.setattr(backend,'pin_coordinator',lambda cycle:{'path':'coord'})
    monkeypatch.setattr(backend,'pin_resident',lambda worker,cycle:{'path':'resident-'+str(worker)})
    captured=backend.capture_fenced('cycle',timeout=30)
    assert len(captured['ready'])==4 and len(captured['components'])==5
    assert all(any(i==worker and path.endswith('.release.json') for i,path in events) for worker in range(4))
    events.clear()
    monkeypatch.setattr(backend,'pin_coordinator',lambda cycle:(_ for _ in ()).throw(RuntimeError('pin failure')))
    with pytest.raises(RuntimeError,match='pin failure'):backend.capture_fenced('cycle',timeout=30)
    assert all(any(i==worker and path.endswith('.release.json') for i,path in events) for worker in range(4))


def test_stage_verified_topology_keeps_each_worker_and_parent_separate(tmp_path):
    import hashlib,io
    stage=load('diloco_stage_recovery');storage=tmp_path/'storage';storage.mkdir()
    def sha(data):return hashlib.sha256(data).hexdigest()
    def archive(name,members):
        path=tmp_path/name
        with tarfile.open(path,'w') as tar:
            for key,value in members.items():
                info=tarfile.TarInfo(key);info.size=len(value);tar.addfile(info,io.BytesIO(value))
        return path
    def receipt(identity,name):return {'private':True,'identity':identity,'independent_verification':identity,
                                        'verified_at':1,'run':'e/p/r','name':name,'source':'/work/x'}
    components={};receipts={}
    def add(name,path):
        identity={'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size}
        dest=storage/'objects'/identity['sha256']/path.name;dest.parent.mkdir(parents=True);dest.write_bytes(path.read_bytes())
        components[name]={'path':'/work/'+path.name,**identity};receipts[name]=receipt(identity,name)
    for i in range(4):
        worker=tmp_path/f'w{i}.pt';worker.write_bytes(f'worker{i}'.encode());add(f'worker-{i}',worker)
        add(f'resident-{i}',archive(f'r{i}.tar',{'diloco-resident/initial.json':f'resident{i}'.encode()}))
    add('coordinator',archive('coord.tar',{'diloco-coordinator/manifest.json':b'coordinator'}))
    payload=b'private dependency';h=sha(payload)
    index={'schema':'natlang.diloco-recovery-dependencies/1','bindings':[{'path':'/work/input','sha256':h,'bytes':len(payload),'object':'objects/'+h}]}
    dep=archive('deps.tar',{'objects/'+h:payload,'dependency-index.json':json.dumps(index).encode()})
    dep_id={'sha256':sha(dep.read_bytes()),'bytes':dep.stat().st_size}
    cycle={'id':'cycle','phase':'verified-components-and-dependencies','compatibility':{'compatible':True},
           'components':components,'receipts':receipts,'dependency_intent':{'path':str(dep),**dep_id},
           'dependency_receipt':receipt(dep_id,'dependencies')}
    state=tmp_path/'state.json';state.write_text(json.dumps({'run_id':'run','cycles':[cycle]}))
    parent=tmp_path/'parent.pt';parent.write_bytes(b'parent');parent_id={'sha256':sha(parent.read_bytes()),'bytes':parent.stat().st_size}
    parent_pin=tmp_path/'pin.json';parent_pin.write_text(json.dumps({'path':'/work/parent.pt',**parent_id,'global_step':50}))
    parent_receipt=tmp_path/'parent-receipt.json';parent_receipt.write_text(json.dumps(receipt(parent_id,'parent')))
    for i in range(4):
        components[f'worker-{i}'].update(fingerprint='f',source_hash='s',runtime_code_sha256='code',
            recipe_binding='recipe',ce_supervision_policy_binding='ce',resident_generation='g',versions=[0],
            sequences=[0],pending={},awaiting_global=[])
        components[f'resident-{i}']['generation']='g'
    components['coordinator'].update(fingerprint='f',runtime_code_sha256='code',versions=[0],last_seq=[{}],queued=[{}],
        manifest={'run_id':'run','identity':{'parent_checkpoint_sha256':parent_id['sha256']}})
    state.write_text(json.dumps({'run_id':'run','cycles':[cycle]}))
    out=tmp_path/'staged'
    result=stage.stage(state,'cycle',storage,parent_pin,parent_receipt,parent,out,present=lambda _:True,reserve_bytes=0,external_required=False)
    assert result['recovery_restore_qualified'] is False
    assert (out/'parent.pt').read_bytes()==b'parent'
    for i in range(4):assert (out/f'workers/{i}/checkpoint.pt').read_bytes()==f'worker{i}'.encode()
    assert (out/'diloco-coordinator/manifest.json').read_bytes()==b'coordinator'
    assert (out/'dependencies/objects'/h).read_bytes()==payload
    with pytest.raises(ValueError,match='private original parent is absent'):
        stage.stage(state,'cycle',storage,parent_pin,parent_receipt,parent,tmp_path/'bad',present=lambda _:False,reserve_bytes=0,external_required=False)


def test_original_parent_requires_exact_local_private_and_coordinator_binding(tmp_path):
    import hashlib
    service=load('diloco_periodic_recovery');backend=object.__new__(service.Backend)
    payload=b'earned parent';sha=hashlib.sha256(payload).hexdigest();identity={'sha256':sha,'bytes':len(payload)}
    parent=tmp_path/'parent.pt';parent.write_bytes(payload)
    pin=tmp_path/'pin.json';pin.write_text(json.dumps({'path':'/work/parent.pt',**identity,'global_step':750}))
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'private':True,'identity':identity,'verified_at':1,
        'independent_verification':identity,'run':'e/p/r','name':'parent','source':'/work/parent.pt'}))
    transfer={'pin':str(pin),'pin_sha256':hashlib.sha256(pin.read_bytes()).hexdigest(),'receipt':str(receipt),
              'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()}
    backend.record={'run':{'parent_transfer':transfer}};backend.parent='/work/parent.pt'
    backend.manifest={'identity':{'parent_checkpoint_sha256':sha}};backend.parent_local=parent;backend.external=tmp_path
    backend.present=lambda value:True
    assert backend.verify_parent()['global_step']==750
    parent.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='local full-state recovery copy differs'):backend.verify_parent()


def test_fence_failure_or_expiry_never_retriggers_work(tmp_path):
    from natlang_neuralese.train.diloco import recovery_fence
    root=tmp_path/'out';directory=root/'diloco/recovery-fence';directory.mkdir(parents=True)
    ident='a'*32;request={'schema':recovery_fence.SCHEMA,'id':ident,'worker':0,'fingerprint':'f','deadline':20}
    (directory/'request.json').write_text(json.dumps(request))
    assert recovery_fence.request(root,0,'f',now=lambda:10)==request
    (directory/(ident+'.failed.json')).write_text('{}')
    assert recovery_fence.request(root,0,'f',now=lambda:10) is None
    (directory/(ident+'.failed.json')).unlink()
    assert recovery_fence.request(root,0,'f',now=lambda:21) is None


def test_http_restore_replays_already_queued_payload_once(tmp_path):
    import socket,threading,time
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,serve_coordinator
    from natlang_neuralese.train.diloco.protocol import TensorSurface
    from natlang_neuralese.train.diloco.worker import DilocoLearner,DilocoSettings,Transport
    seed=torch.nn.Linear(2,2);root=tmp_path/'coord'
    initialize_coordinator(root,run_id='queued',masters=TensorSurface.from_module(seed).masters,identity={'parent':'exact'},
                           workers=2,min_quorum=2,fragment_bytes=1024,grace_seconds=0,outer_lr=1.0,outer_momentum=0)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}';token='queued-recovery-test-token-at-least-32-characters';stop=threading.Event()
    thread=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,token),kwargs={'stop':stop},daemon=True);thread.start()
    transport=Transport(url,token,5)
    for _ in range(100):
        try:transport.json('GET','/v1/status');break
        except OSError:time.sleep(.02)
    learners=[]
    for worker in range(2):
        model=torch.nn.Linear(2,2);model.load_state_dict(seed.state_dict())
        settings=DilocoSettings(run_id='queued',workers=2,worker_id=worker,coordinator_url=url,local_steps=2,
                                wire_dtype='bf16',allow_private_http=True,start_grace_seconds=0,heartbeat_seconds=.05)
        learner=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity={'parent':'exact'},
                              output=tmp_path/f'worker{worker}',transport=Transport(url,token,5))
        learners.append([model,learner,settings])
    starters=[threading.Thread(target=item[1].start) for item in learners]
    for starter in starters:starter.start()
    for starter in starters:starter.join(timeout=10);assert not starter.is_alive()
    try:
        model,worker,settings=learners[0]
        for _ in range(2):
            with torch.no_grad():model.weight.add_(.1)
            worker.after_step(tokens=1)
        for _ in range(200):
            if transport.json('GET','/v1/status')['pending']['0']==1:break
            time.sleep(.02)
        else:raise AssertionError('first worker update was not queued')
        saved=worker.state_dict();assert saved['pending']['0']['sequence']==1
        restored_model=torch.nn.Linear(2,2);restored_model.load_state_dict(model.state_dict());worker.close()
        resumed=DilocoLearner(settings,surface=TensorSurface.from_module(restored_model),identity={'parent':'exact'},
                              output=tmp_path/'worker0-restored',transport=Transport(url,token,5))
        resumed.resume(saved);learners[0]=[restored_model,resumed,settings]
        model1,worker1,_=learners[1]
        for _ in range(2):
            with torch.no_grad():model1.weight.add_(.2)
            worker1.after_step(tokens=1)
        for _ in range(200):
            if transport.json('GET','/v1/status')['versions'][0]==1:break
            time.sleep(.02)
        else:raise AssertionError('quorum did not merge queued updates')
        assert resumed.sequences[0]==1
        assert transport.json('GET','/v1/status')['versions'][0]==1
    finally:
        for _,learner,_ in learners:learner.close()
        stop.set();thread.join(timeout=3)
