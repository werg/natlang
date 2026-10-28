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
              'launch_code':bind('launch.py','launch'),'private_export_code':bind('export.py','export'),
              'verified_download_code':bind('download.py','download')}
    helper=Path(__file__).parents[1]/'scripts/coreweave'
    for key,name in [('recovery_code','diloco_periodic_recovery.py'),('pin_code','diloco_recovery_pin.py'),
                     ('direct_code','diloco_direct_recovery.py'),('dependencies_code','diloco_recovery_dependencies.py'),
                     ('archive_code','cache_bounded_archive.py'),('guard_code','private_export_guard.py'),
                     ('readiness_code','recovery_readiness.py'),('inputs_code','recovery_inputs.py'),
                     ('stage_code','diloco_stage_recovery.py')]:
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
    first_control_export=[True]
    class Backend:
        workers=record['learners']
        def __init__(self):self.direct=direct
        def capture_fenced(self,cycle):
            return {'id':'fence','deadline':1,'ready':{str(i):{'checkpoint':f'/f/{i}'} for i in range(4)},
                    'components':{'coordinator':{'path':'/coord','sha256':'a'*64,'bytes':1,'fingerprint':'f','versions':[1],
                        'last_seq':[{'0':1,'1':1,'2':1,'3':1}],'queued':[{}],
                        'runtime_code_sha256':direct['runtime_code_sha256']}}}
        def pin_worker(self,i,cycle,source):
            assert source==f'/f/{i}'
            return {'fingerprint':'f','versions':[1],'sequences':[1],'pending':{},'awaiting_global':[],
                'source_hash':'same','runtime_code_sha256':direct['runtime_code_sha256'],'resident_generation':None,
                'recipe_binding':json.dumps(recipe),'source_receipt_sha256':direct['source_receipt_sha256'],
                'ce_supervision_policy_binding':{'sha256':direct['ce_sha256']},'global_step':2,'path':f'/w{i}','sha256':str(i)*64,'bytes':1}
        def pin_resident(self,*args):raise AssertionError('direct mode has no resident control')
        def export(self,name,intent):return {'name':name}
        def progress(self,i):return {'global_step':3}
        def verify_parent(self):return {'verified':True}
        def direct_storage_preflight(self,cycle):return {'available_bytes':100,'needed_bytes':20}
        def component_storage(self,name):return (tmp_path,0)
        def prepare_dependencies(self,cycle):assert len(cycle['components'])==5;return {'path':'deps'}
        def prepare_cohort_control(self,state,cycle):return {'path':'catalog','sha256':'e'*64,'bytes':1}
        def export_dependencies(self,intent):
            if intent['path']=='catalog' and first_control_export[0]:
                first_control_export[0]=False;raise RuntimeError('temporary catalog PUT failure')
            return {'name':'deps'}
        def present(self,receipt):return True
    backend=Backend();first=service.run_once(record,tmp_path/'state.json',backend)
    assert first['pending']['phase']=='incomplete' and 'cohort-control' in first['pending']['errors']
    assert first['pending']['cohort_control_intent']['path']=='catalog'
    state=service.run_once(record,tmp_path/'state.json',backend);cycle=state['cycles'][-1]
    assert cycle['phase']=='verified-components-and-dependencies' and len(cycle['components'])==5
    assert cycle['cohort_control_receipt']['name']=='deps'
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


def test_final_cohort_catalog_is_self_contained_and_retry_stable(tmp_path):
    service=load('diloco_periodic_recovery');backend=object.__new__(service.Backend);backend.storage=tmp_path
    state={'run_id':'run','coordinator_run_id':'run','recovery_contract':{'mode':'direct','control_sha256':'a'*64}}
    cycle={'id':'cycle','phase':'exporting','components':{'worker-0':{'sha256':'b'*64}},
           'receipts':{'worker-0':{'name':'private'}},'dependency_intent':{'sha256':'c'*64},
           'dependency_receipt':{'name':'deps'},'parent_reference':{'pin':{'sha256':'d'*64}}}
    intent=backend.prepare_cohort_control(state,cycle);assert backend.prepare_cohort_control(state,cycle)==intent
    saved=json.loads(Path(intent['path']).read_text());assert saved['cycles'][0]['phase']=='verified-components-and-dependencies'
    assert saved['cycles'][0]['receipts']==cycle['receipts'] and saved['latest_verified'] is None


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
    external=tmp_path/'external';external.mkdir();nvme=tmp_path/'nvme';nvme.mkdir()
    path=nvme/'history.pt';path.write_bytes(b'earned history');logical=external/'history.pt';logical.symlink_to(path)
    original_is_mount=Path.is_mount
    monkeypatch.setattr(Path,'is_mount',lambda p:True if p==external else original_is_mount(p))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    identity={'sha256':sha,'bytes':path.stat().st_size}
    receipt={'identity':identity,'private':True,'run':'run','name':'file','source':'/pin','verified_at':1,
             'independent_verification':{**identity,'local_copy':{**identity,'committed':True,'path':str(logical)}}}
    rp=tmp_path/'history-receipt.json';rp.write_text(json.dumps(receipt));ref={'path':str(rp),'sha256':hashlib.sha256(rp.read_bytes()).hexdigest()}
    direct['bindings'].append(ref);direct['external_dependencies']=[{**identity,'local_path':str(logical),'receipt':ref}]
    backend=object.__new__(service.Backend);backend.direct=direct;backend.workers=['s0','s1','s2','s3'];backend.storage=tmp_path;backend.storage_reserve_bytes=1024
    backend.external=external;backend.nvme=nvme;backend.external_hash_cache={};backend.present=lambda receipt:True
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
    assert result['filesystems'][0]['component_bytes']==600 and result['filesystems'][0]['dependency_upper_bound_bytes']>0
    other=tmp_path/'other-disk';other.mkdir()
    direct['component_placement']={'worker-0':{'root':str(other),'reserve_bytes':2000}}
    monkeypatch.setattr(service,'_filesystem_id',lambda path:2 if Path(path)==other else 1)
    result=backend.direct_storage_preflight(cycle)
    by_root={v['root']:v for v in result['filesystems']}
    assert by_root[str(other)]['component_bytes']==100 and by_root[str(other)]['reserve_bytes']==2000
    assert by_root[str(tmp_path)]['component_bytes']==500 and by_root[str(tmp_path)]['dependency_upper_bound_bytes']>0
    monkeypatch.setattr(service.shutil,'disk_usage',lambda path:SimpleNamespace(free=1000))
    with pytest.raises(OSError,match='storage preflight'):backend.direct_storage_preflight(cycle)


def test_stage_direct_split_components_and_verified_external_history(tmp_path):
    import io
    stage=load('diloco_stage_recovery');mod=load('diloco_direct_recovery');descriptor,control,recipe,roles=fixture(tmp_path)
    # Recovery controls are independent copies: remove them after sealing to
    # ensure staging reads authenticated archived bytes, not original paths.
    for key,item in control['bindings'].items():
        original=Path(item['path'])
        if not original.is_relative_to(tmp_path):
            copied=tmp_path/('bound-'+key);copied.write_bytes(original.read_bytes());item['path']=str(copied)
    def sha(data):return hashlib.sha256(data).hexdigest()
    def receipt(identity,name):return {'private':True,'identity':identity,'independent_verification':identity,'verified_at':1,'run':'run','name':name,'source':'/pin'}
    parent=tmp_path/'parent.pt';parent.write_bytes(b'parent');pid={'sha256':sha(parent.read_bytes()),'bytes':parent.stat().st_size}
    pin=tmp_path/'parent-pin.json';pin.write_text(json.dumps({'path':'/work/parent',**pid,'global_step':1}))
    parent_receipt=tmp_path/'parent-receipt.json';parent_receipt.write_text(json.dumps(receipt(pid,'parent')))
    qualification=tmp_path/'qualification.json';qualification.write_text(json.dumps({'schema':'natlang.lfm3-seed-independent-full-state/1','passed':True,'global_step':1,**pid}))
    copy=tmp_path/'copy.json';copy.write_text(json.dumps({**pid,'local_copy':{'committed':True,**pid}}))
    parent_refs={key:{'path':str(path),'sha256':sha(path.read_bytes())} for key,path in
                 [('pin',pin),('receipt',parent_receipt),('qualification',qualification),('copy_receipt',copy)]}
    control['parent_transfer']={'pin':str(pin),'pin_sha256':parent_refs['pin']['sha256'],
                                'receipt':str(parent_receipt),'receipt_sha256':parent_refs['receipt']['sha256']}
    disk_a=tmp_path/'disk-a';disk_b=tmp_path/'disk-b';disk_a.mkdir();disk_b.mkdir()
    control['component_placement']={name:{'root':str(disk_b if name=='worker-0' else disk_a),'reserve_bytes':10}
                                    for name in ('worker-0','worker-1','worker-2','worker-3','coordinator')}
    descriptor.write_text(json.dumps(control));direct=mod.load_control(descriptor,mod.record_from_control(descriptor,'run'))
    def sha(data):return hashlib.sha256(data).hexdigest()
    def archive(path,members):
        with tarfile.open(path,'w') as tar:
            for name,data in members.items():
                item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
        return path
    def receipt(identity,name):return {'private':True,'identity':identity,'independent_verification':identity,'verified_at':1,'run':'run','name':name,'source':'/pin'}
    components={};receipts={};localpaths={}
    def add(name,path):
        identity={'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size}
        root=Path(direct['component_placement'][name]['root']);dest=root/'objects'/identity['sha256']/path.name
        dest.parent.mkdir(parents=True);dest.write_bytes(path.read_bytes());localpaths[name]=str(dest)
        components[name]={'path':'/work/'+path.name,**identity};receipts[name]=receipt(identity,name)
    for i in range(4):
        p=tmp_path/f'worker{i}.pt';p.write_bytes(f'worker{i}'.encode());add(f'worker-{i}',p)
        components[f'worker-{i}'].update(fingerprint='f',source_hash='s',runtime_code_sha256=direct['runtime_code_sha256'],
            recipe_binding=json.dumps(recipe),ce_supervision_policy_binding={'sha256':direct['ce_sha256']},resident_generation=None,
            source_receipt_sha256=direct['source_receipt_sha256'],versions=[0],sequences=[0],pending={},awaiting_global=[])
    add('coordinator',archive(tmp_path/'coordinator.tar',{'diloco-coordinator/manifest.json':b'coordinator'}))
    components['coordinator'].update(fingerprint='f',runtime_code_sha256=direct['runtime_code_sha256'],versions=[0],last_seq=[{}],queued=[{}],
        manifest={'run_id':'run','identity':{'parent_checkpoint_sha256':pid['sha256']}})
    history=disk_b/'history.pt';history.write_bytes(b'history');history_link=disk_a/'history-link.pt';history_link.symlink_to(history)
    hid={'sha256':sha(history.read_bytes()),'bytes':history.stat().st_size}
    historical_receipt=json.dumps(receipt(hid,'history')).encode();rh=sha(historical_receipt)
    control_bytes=descriptor.read_bytes();ch=sha(control_bytes)
    index={'schema':'natlang.diloco-recovery-dependencies/1','bindings':[
       {'path':str(descriptor),'sha256':ch,'bytes':len(control_bytes),'object':'objects/'+ch},
       {'path':'/receipt','sha256':rh,'bytes':len(historical_receipt),'object':'objects/'+rh},
       {'path':'/work/history','sha256':hid['sha256'],'bytes':hid['bytes'],
        'external':{**hid,'local_path':str(history_link),'receipt':{'path':'/receipt','sha256':rh}}}]}
    objects={'objects/'+ch:control_bytes,'objects/'+rh:historical_receipt}
    for item in [*control['bindings'].values(),*parent_refs.values()]:
        data=Path(item['path']).read_bytes();assert sha(data)==item['sha256']
        objects['objects/'+item['sha256']]=data
        index['bindings'].append({'path':item['path'],'sha256':item['sha256'],'bytes':len(data),'object':'objects/'+item['sha256']})
    objects['dependency-index.json']=json.dumps(index).encode()
    dep=archive(tmp_path/'deps.tar',objects)
    depid={'sha256':sha(dep.read_bytes()),'bytes':dep.stat().st_size}
    cycle={'id':'cycle','phase':'verified-components-and-dependencies','compatibility':{'compatible':True},'components':components,
           'component_local_paths':localpaths,'parent_reference':parent_refs,'receipts':receipts,'dependency_intent':{'path':str(dep),**depid},'dependency_receipt':receipt(depid,'deps')}
    state=tmp_path/'state.json';state.write_text(json.dumps({'run_id':'run','coordinator_run_id':'run','cycles':[cycle],
        'recovery_contract':{'mode':'direct','control_sha256':ch}}))
    for item in [*control['bindings'].values(),*parent_refs.values()]:Path(item['path']).unlink()
    descriptor.unlink()
    fetched=tmp_path/'fetched';fetched.mkdir();overrides={}
    for name,old in localpaths.items():
        target=fetched/(name+'.bin');Path(old).rename(target);overrides[name]=str(target)
    fetched_dep=fetched/'deps.tar';dep.rename(fetched_dep);overrides['dependencies']=str(fetched_dep)
    fetched_history=fetched/'history.pt';history.rename(fetched_history);overrides['external']={hid['sha256']:str(fetched_history)}
    result=stage.stage(state,'cycle',disk_a,pin,parent_receipt,parent,tmp_path/'restore',present=lambda _:True,
                       reserve_bytes=0,external_required=False,inputs=overrides)
    assert result['recovery_restore_qualified'] is False
    manifest=json.loads((tmp_path/'restore/staging.json').read_text())
    assert manifest['external_dependencies']['/work/history']['sha256']==hid['sha256']
    assert (tmp_path/'restore/workers/0/checkpoint.pt').read_bytes()==b'worker0'
    fetched_history.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='local recovery component differs'):
        stage.stage(state,'cycle',disk_a,pin,parent_receipt,parent,tmp_path/'bad-restore',present=lambda _:True,
                    reserve_bytes=0,external_required=False,inputs=overrides)
    fetched_history.write_bytes(b'history')
    objects.pop('objects/'+control['bindings']['plan']['sha256'])
    bad=archive(tmp_path/'missing.tar',objects);badid={'sha256':sha(bad.read_bytes()),'bytes':bad.stat().st_size}
    cycle['dependency_intent']={'path':str(bad),**badid};cycle['dependency_receipt']=receipt(badid,'deps-missing')
    overrides['dependencies']=str(bad)
    state.write_text(json.dumps({'run_id':'run','coordinator_run_id':'run','cycles':[cycle],
        'recovery_contract':{'mode':'direct','control_sha256':ch}}))
    with pytest.raises(ValueError,match='binding differs'):
        stage.stage(state,'cycle',disk_a,pin,parent_receipt,parent,tmp_path/'missing-restore',present=lambda _:True,
                    reserve_bytes=0,external_required=False,inputs=overrides)
