import hashlib
import json
from pathlib import Path
import tarfile
import pytest
from test_diloco_recovery import load
from natlang_neuralese.common.hashing import canonical_json_sha256_hex


def fixture(tmp_path):
    def bind(name,data):
        path=tmp_path/name
        path.write_text(data if isinstance(data,str) else json.dumps(data,sort_keys=True))
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    workers={str(i):{'sandbox_id':f's{i}','pid':100+i,'boot_id':f'b{i}','start_ticks':str(i+1)} for i in range(4)}
    settings={'run_id':'run','workers':4,'local_steps':128}
    recipe=bind('recipe.json',{'stage':'native_full_sft'});code={'module.py':hashlib.sha256(b'runtime').hexdigest()}
    roles={name:{'path':'/source/'+name,'sha256':sha*64} for name,sha in
           [('native_graphs','a'),('foundation_native_graphs','a'),('held_native_graphs','b'),('ce_supervision_policy','c')]}
    plan={'recipe_sha256':'e'*64,'code':code,'stage_inputs':{'native_full_sft':roles}}
    planref=bind('plan.json',plan)
    recipe_binding={'path':recipe['path'],'sha256':recipe['sha256'],'file_sha256':recipe['sha256'],'lineage_sha256':'e'*64}
    argv=['--out','/work/out','--continue-from','/work/parent','--stage-id','native_full_sft',
          '--recipe-binding',json.dumps(recipe_binding),'--diloco',json.dumps(settings)]
    for role,flag in [('native_graphs','--native-graphs'),('foundation_native_graphs','--foundation-native-graphs'),
                      ('held_native_graphs','--held-native-graphs'),('ce_supervision_policy','--ce-supervision-policy')]:
        argv.extend([flag,roles[role]['path']])
    coord=['init','--root','/work/coord','--parent','/work/parent','--trainer-argv',json.dumps(argv),'--settings',json.dumps(settings)]
    owner={'run_id':'run','coordinator':{'sandbox_id':'s0','pid':90,'boot_id':'b0','start_ticks':'8','owner_id':'owner'},
           'workers':workers,'worker_out':'/work/out','coordinator_root':'/work/coord',
           'runtime':{'package':'/work/runtime/training/neuralese/natlang_neuralese'},'plan':planref}
    bindings={'plan':planref,'recipe':recipe,'worker_argv':bind('worker.json',argv),'coordinator_argv':bind('coord.json',coord),
              'settings':bind('settings.json',settings),'owner_journal':bind('owner.json',owner),
              'launch_code':bind('launch.py','launch')}
    helper=Path(__file__).parents[1]/'scripts/coreweave'
    for key,name in [('recovery_code','diloco_periodic_recovery.py'),('pin_code','diloco_recovery_pin.py'),
                     ('direct_code','diloco_direct_recovery.py'),('dependencies_code','diloco_recovery_dependencies.py'),
                     ('archive_code','cache_bounded_archive.py'),('guard_code','private_export_guard.py'),
                     ('readiness_code','recovery_readiness.py'),('inputs_code','recovery_inputs.py')]:
        path=helper/name;bindings[key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    control={'schema':'natlang.diloco-direct-recovery-control/1','bindings':bindings,
             'runtime_code_sha256':canonical_json_sha256_hex(code),'storage_reserve_bytes':1024,
             'paths':{'worker_out':'/work/out','coordinator_root':'/work/coord','runtime_repo':'/work/runtime'}}
    descriptor=tmp_path/'control.json';descriptor.write_text(json.dumps(control))
    return descriptor,control,recipe_binding,roles


def test_direct_control_validates_actual_cohort_and_all_launch_bindings(tmp_path):
    mod=load('diloco_direct_recovery');descriptor,control,recipe,roles=fixture(tmp_path)
    record=mod.record_from_control(descriptor,'run');actual=mod.load_control(descriptor,record)
    worker={'runtime_code_sha256':actual['runtime_code_sha256'],'resident_generation':None,'recipe_binding':json.dumps(recipe),
            'source_receipt_sha256':[roles['native_graphs']['sha256'],roles['held_native_graphs']['sha256']],
            'ce_supervision_policy_binding':{'sha256':roles['ce_supervision_policy']['sha256']}}
    assert mod.worker_matches(worker,actual)
    assert not mod.worker_matches({**worker,'source_receipt_sha256':['bad','b']},actual)
    Path(control['bindings']['worker_argv']['path']).write_text('[]')
    with pytest.raises(ValueError,match='binding differs'):mod.load_control(descriptor,record)


def test_real_backend_constructs_from_nonresident_control(tmp_path,monkeypatch):
    from types import SimpleNamespace
    service=load('diloco_periodic_recovery');mod=load('diloco_direct_recovery');descriptor,_,_,_=fixture(tmp_path)
    record=mod.record_from_control(descriptor,'run');manifest={'run_id':'run','workers':4,'fingerprint':'actual'}
    class Future:
        def result(self,timeout):return json.dumps(manifest)
    box=SimpleNamespace(read_file=lambda path:Future())
    monkeypatch.setattr(service.cwctl,'read_sandbox',lambda ident:box)
    monkeypatch.setattr(service.paths,'root',lambda name:tmp_path if name=='data_nvme' else tmp_path/'external' if name=='data_hdd' else Path(__file__).parents[1])
    backend=service.Backend(record,tmp_path/'backups',runtime_bundle=tmp_path/'bundle.json',direct_control=descriptor)
    assert backend.parent=='/work/parent' and backend.out=='/work/out' and backend.root=='/work/coord'
    assert backend.runtime_repo==Path('/work/runtime') and backend.storage_reserve_bytes==1024


def test_direct_fenced_five_component_cycle_has_no_fake_residents(tmp_path):
    service=load('diloco_periodic_recovery');mod=load('diloco_direct_recovery');descriptor,control,recipe,roles=fixture(tmp_path)
    record=mod.record_from_control(descriptor,'run');direct=mod.load_control(descriptor,record)
    class Backend:
        workers=record['learners']
        def __init__(self):self.direct=direct
        def capture_fenced(self,cycle):
            return {'id':'fence','deadline':1,'ready':{str(i):{'checkpoint':f'/f/{i}'} for i in range(4)},
                    'components':{'coordinator':{'path':'/coord','fingerprint':'f','versions':[1],
                        'last_seq':[{'0':1,'1':1,'2':1,'3':1}],'queued':[{}],
                        'runtime_code_sha256':direct['runtime_code_sha256']}}}
        def pin_worker(self,i,cycle,source):
            assert source==f'/f/{i}'
            return {'fingerprint':'f','versions':[1],'sequences':[1],'pending':{},'awaiting_global':[],
                'source_hash':'same','runtime_code_sha256':direct['runtime_code_sha256'],'resident_generation':None,
                'recipe_binding':json.dumps(recipe),'source_receipt_sha256':direct['source_receipt_sha256'],
                'ce_supervision_policy_binding':{'sha256':direct['ce_sha256']},'global_step':2,'path':f'/w{i}'}
        def pin_resident(self,*args):raise AssertionError('direct mode has no resident control')
        def export(self,name,intent):return {'name':name}
        def progress(self,i):return {'global_step':3}
        def verify_parent(self):return {'verified':True}
        def direct_storage_preflight(self,cycle):return {'available_bytes':100,'needed_bytes':20}
        def prepare_dependencies(self,cycle):assert len(cycle['components'])==5;return {'path':'deps'}
        def export_dependencies(self,intent):return {'name':'deps'}
        def present(self,receipt):return True
    state=service.run_once(record,tmp_path/'state.json',Backend());cycle=state['cycles'][-1]
    assert cycle['phase']=='verified-components-and-dependencies' and len(cycle['components'])==5
    assert cycle['compatibility']['compatible'] and not any(k.startswith('resident-') for k in cycle['components'])
    changed={**cycle['components'],'worker-0':{**cycle['components']['worker-0'],'source_receipt_sha256':['bad','bad']}}
    assert not service.compatibility(changed,direct=direct)['compatible']
    incompatible={**cycle['components'],'coordinator':{**cycle['components']['coordinator'],'last_seq':[{'0':2}]}}
    assert 'uncheckpointed' in service.compatibility(incompatible,direct=direct)['reason']


def test_direct_requires_fence_and_does_not_export_unfenced_pins(tmp_path):
    service=load('diloco_periodic_recovery');mod=load('diloco_direct_recovery');descriptor,_,_,_=fixture(tmp_path)
    record=mod.record_from_control(descriptor,'run');direct=mod.load_control(descriptor,record)
    class Backend:
        workers=record['learners']
        def __init__(self):self.direct=direct
        def capture_fenced(self,cycle):raise service.FenceUnavailable('missing capability')
        def pin_worker(self,*args):raise AssertionError('must not take unfenced pins')
    state=service.run_once(record,tmp_path/'state.json',Backend())
    assert state['pending']['phase']=='incomplete' and 'missing capability' in state['pending']['errors']['fence']
    assert not state['pending'].get('unfenced')


def test_direct_stages_and_hash_checks_runtime_helper_dependencies(tmp_path,monkeypatch):
    from types import SimpleNamespace
    service=load('diloco_periodic_recovery');mod=load('diloco_direct_recovery');descriptor,_,_,_=fixture(tmp_path)
    direct=mod.load_control(descriptor,mod.record_from_control(descriptor,'run'))
    backend=object.__new__(service.Backend);backend.direct=direct;backend.runtime_repo=Path('/work/runtime')
    backend.helper=(Path(__file__).parents[1]/'scripts/coreweave/diloco_recovery_pin.py').read_text()
    files={};calls=[]
    class Future:
        def __init__(self,result):self.value=result
        def result(self,timeout):return self.value
    class Box:
        def write_file(self,path,contents):files[path]=contents;return Future(None)
        def exec(self,argv,**kwargs):
            calls.append(argv)
            if argv[0]=='python3':
                expected=json.loads(argv[-1]);assert all(hashlib.sha256(files[k]).hexdigest()==v for k,v in expected.items())
            if argv[0]=='nice':
                assert '/work/runtime/training/neuralese' in argv[4]
                return Future(SimpleNamespace(returncode=0,stdout='{"ok":true}',stderr=''))
            return Future(SimpleNamespace(returncode=0,stdout='',stderr=''))
    monkeypatch.setattr(service.cwctl,'read_sandbox',lambda *args,**kwargs:Box())
    assert backend._exec('box',['worker'],timeout=30)=={'ok':True}
    assert len(files)==6 and len(calls)==3


def test_direct_control_bindings_are_recoverable_from_verified_dependency_archive(tmp_path):
    mod=load('diloco_direct_recovery');deps=load('diloco_recovery_dependencies');descriptor,_,_,_=fixture(tmp_path)
    direct=mod.load_control(descriptor,mod.record_from_control(descriptor,'run'))
    codefile=tmp_path/'module.py';codefile.write_bytes(b'runtime');archive=tmp_path/'runtime.tar'
    with tarfile.open(archive,'w') as tar:tar.add(codefile,arcname='training/neuralese/natlang_neuralese/module.py')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'code':{'key':'runtime.tar','sha256':digest(archive),'bytes':archive.stat().st_size},'files':[]}))
    sources=tmp_path/'bundle.sources.json';sources.write_text(json.dumps({'runtime.tar':str(archive)}))
    result=deps.seal(direct['bindings'],{},bundle,sources,tmp_path/'sealed.tar',
                     runtime_code_sha256=direct['runtime_code_sha256'],reserve_bytes=0)
    with tarfile.open(result['path']) as tar:
        index=json.load(tar.extractfile('dependency-index.json'))
        retained={(item['path'],item['sha256']) for item in index['bindings']}
        assert all((item['path'],item['sha256']) in retained for item in direct['bindings'])
    data=b'external-history';sha=hashlib.sha256(data).hexdigest()
    external={sha:{'sha256':sha,'bytes':len(data),'local_path':'/verified/local','receipt':{'path':'/receipt','sha256':'e'*64}}}
    result=deps.seal([{'path':'/work/historical.pt','sha256':sha}],{},bundle,sources,tmp_path/'external.tar',
                     runtime_code_sha256=direct['runtime_code_sha256'],reserve_bytes=0,external=external)
    with tarfile.open(result['path']) as tar:
        index=json.load(tar.extractfile('dependency-index.json'))
        item=next(item for item in index['bindings'] if item['sha256']==sha)
        assert item['external']==external[sha] and 'objects/'+sha not in tar.getnames()


def test_direct_dependency_preparation_includes_checkpoint_sources_and_parent_proofs(tmp_path,monkeypatch):
    service=load('diloco_periodic_recovery');mod=load('diloco_direct_recovery');descriptor,_,_,_=fixture(tmp_path)
    direct=mod.load_control(descriptor,mod.record_from_control(descriptor,'run'))
    backend=object.__new__(service.Backend);backend.direct=direct;backend.workers=['s0','s1','s2','s3'];backend.storage=tmp_path;backend.storage_reserve_bytes=1024
    backend.runtime_bundle=tmp_path/'bundle.json';backend.runtime_bundle.write_text('{}')
    (tmp_path/'bundle.sources.json').write_text('{}')
    required=[{'path':'/base','sha256':'a'*64,'kinds':['base-model-file']},
              {'path':'/train','sha256':'b'*64,'kinds':['source-manifest']}]
    components={f'worker-{i}':{'required_dependencies':required,'runtime_code_sha256':direct['runtime_code_sha256']} for i in range(4)}
    parents={name:{'path':'/proof/'+name,'sha256':'c'*64} for name in ('pin','receipt','qualification','copy_receipt')}
    captured=[];monkeypatch.setattr(service,'seal',lambda bindings,*args,**kwargs:captured.extend(bindings) or {'ok':True})
    assert backend.prepare_dependencies({'id':'cycle','components':components,'parent_reference':parents})=={'ok':True}
    assert all(value in captured for value in parents.values())
    assert all(value in captured for value in direct['bindings'])


def test_external_history_requires_verified_local_and_present_private_copies(tmp_path,monkeypatch):
    service=load('diloco_periodic_recovery');mod=load('diloco_direct_recovery');descriptor,_,_,_=fixture(tmp_path)
    direct=mod.load_control(descriptor,mod.record_from_control(descriptor,'run'))
    path=tmp_path/'history.pt';path.write_bytes(b'earned history');sha=hashlib.sha256(path.read_bytes()).hexdigest()
    identity={'sha256':sha,'bytes':path.stat().st_size}
    receipt={'identity':identity,'private':True,'run':'run','name':'file','source':'/pin','verified_at':1,
             'independent_verification':{**identity,'local_copy':{**identity,'committed':True,'path':str(path)}}}
    rp=tmp_path/'history-receipt.json';rp.write_text(json.dumps(receipt));ref={'path':str(rp),'sha256':hashlib.sha256(rp.read_bytes()).hexdigest()}
    direct['bindings'].append(ref);direct['external_dependencies']=[{**identity,'local_path':str(path),'receipt':ref}]
    backend=object.__new__(service.Backend);backend.direct=direct;backend.workers=['s0','s1','s2','s3'];backend.storage=tmp_path;backend.storage_reserve_bytes=1024
    backend.external=tmp_path/'external';backend.external_hash_cache={};backend.present=lambda receipt:True
    backend.runtime_bundle=tmp_path/'bundle.json';backend.runtime_bundle.write_text('{}');(tmp_path/'bundle.sources.json').write_text('{}')
    required=[{'path':'/base','sha256':'a'*64,'kinds':['base-model-file']},{'path':'/train','sha256':'b'*64,'kinds':['source-manifest']}]
    components={f'worker-{i}':{'required_dependencies':required,'runtime_code_sha256':direct['runtime_code_sha256']} for i in range(4)}
    parents={name:{'path':'/proof/'+name,'sha256':'c'*64} for name in ('pin','receipt','qualification','copy_receipt')}
    captured={};monkeypatch.setattr(service,'seal',lambda *args,**kwargs:captured.update(kwargs) or {'ok':True})
    result=backend.prepare_dependencies({'id':'cycle','components':components,'parent_reference':parents})
    assert captured['external'][sha]['receipt']==ref and result['external_dependencies'][0]['receipt']==receipt
    copy=tmp_path/'copy-receipt.json';copy.write_text(json.dumps({**identity,'local_copy':receipt['independent_verification']['local_copy']}))
    copyref={'path':str(copy),'sha256':hashlib.sha256(copy.read_bytes()).hexdigest()}
    direct['bindings'].append(copyref);direct['external_dependencies'][0]['copy_receipt']=copyref
    assert backend.prepare_dependencies({'id':'cycle','components':components,'parent_reference':parents})['external_dependencies']
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError,match='external dependency'):backend.prepare_dependencies({'id':'cycle','components':components,'parent_reference':parents})


def test_direct_preflight_accounts_for_all_components_and_dependency_archive(tmp_path,monkeypatch):
    from types import SimpleNamespace
    service=load('diloco_periodic_recovery');mod=load('diloco_direct_recovery');descriptor,_,_,_=fixture(tmp_path)
    direct=mod.load_control(descriptor,mod.record_from_control(descriptor,'run'))
    backend=object.__new__(service.Backend);backend.direct=direct;backend.workers=['s0','s1','s2','s3']
    backend.storage=tmp_path;backend.storage_reserve_bytes=1000
    source=tmp_path/'payload';source.write_bytes(b'1234567');runtime=tmp_path/'runtime.tar';runtime.write_bytes(b'code')
    runtime_sha=hashlib.sha256(runtime.read_bytes()).hexdigest()
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'code':{'key':'runtime','sha256':runtime_sha,'bytes':4},'files':[]}))
    plan=json.loads(Path(direct['plan_path']).read_text())
    shas={item['sha256'] for item in service.declared_bindings(plan)}|{'f'*64}
    source_map={sha:str(source) for sha in shas};source_map['runtime']=str(runtime)
    (tmp_path/'bundle.sources.json').write_text(json.dumps(source_map));backend.runtime_bundle=bundle
    required=[{'path':'/missing/base','sha256':'f'*64,'kinds':['base-model-file']}]
    components={f'worker-{i}':{'required_dependencies':required,'path':f'/w{i}','sha256':str(i)*64,'bytes':100} for i in range(4)}
    components['coordinator']={'path':'/coord','sha256':'9'*64,'bytes':200}
    parents={name:{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()} for name in ('pin','receipt','qualification','copy_receipt')}
    cycle={'components':components,'parent_reference':parents}
    monkeypatch.setattr(service.shutil,'disk_usage',lambda path:SimpleNamespace(free=1<<30))
    result=backend.direct_storage_preflight(cycle)
    assert result['component_bytes']==600 and result['dependency_upper_bound_bytes']>0
    monkeypatch.setattr(service.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000))
    with pytest.raises(OSError,match='storage preflight'):backend.direct_storage_preflight(cycle)
