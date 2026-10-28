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
        def pin_worker(self,i,cycle):
            if i==3 and fail[0]:raise RuntimeError('not yet available')
            return {'worker':i,'sandbox_id':self.workers[i],'path':'/pin','sha256':str(i)*64,'bytes':5,
                    'global_step':50+i,'fingerprint':'f','versions':[1],'source_hash':'s','runtime_code_sha256':'code',
                    'resident_generation':'gen','recipe_binding':'recipe','ce_supervision_policy_binding':'ce'}
        def pin_resident(self,i,cycle):
            return {'sandbox_id':self.workers[i],'path':'/resident','sha256':str(i)*64,'bytes':5,'generation':'gen'}
        def pin_coordinator(self,cycle):
            counts['coord']+=1
            return {'sandbox_id':'s0','path':'/coord','sha256':'a'*64,'bytes':10,'fingerprint':'f',
                    'versions':[2],'runtime_code_sha256':'code'}
        def export(self,name,intent):counts['export']+=1;return {'verified':True,'name':name}
        def prepare_dependencies(self,cycle):return {'sha256':'d'*64,'path':'/deps','bytes':5}
        def export_dependencies(self,intent):return {'verified':True,'name':'deps'}
        def present(self,receipt):return True
        def progress(self,i):return {'global_step':60+i}
    state_path=tmp_path/'state.json';backend=Backend()
    first=service.run_once(record,state_path,backend)
    assert first['pending']['phase']=='incomplete' and first['latest_verified'] is None
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
