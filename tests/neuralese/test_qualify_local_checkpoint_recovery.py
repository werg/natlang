import io,json,tarfile
from pathlib import Path
import pytest
from scripts.neuralese_data.archive_local_recovery_runtime import seal
from scripts.neuralese_data.qualify_local_checkpoint_recovery import extract_archive,_mapped_path,bind_restored_asset


def test_restore_checkpoint_is_a_verified_copy_and_rejects_aliases(tmp_path):
    import hashlib,os
    from scripts.neuralese_data.qualify_local_checkpoint_recovery import stage_restore_checkpoint
    source=tmp_path/'owner.pt';source.write_bytes(b'full checkpoint bytes')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    store=tmp_path/'independent'
    result=stage_restore_checkpoint(source,store,digest)
    restored=Path(result['path'])
    assert restored.read_bytes()==source.read_bytes() and not os.path.samefile(restored,source)
    assert stage_restore_checkpoint(source,store,digest)==result
    restored.write_bytes(b'corrupt copy')
    with pytest.raises(ValueError,match='differs'):
        stage_restore_checkpoint(source,store,digest)
    restored.unlink();os.link(source,restored)
    with pytest.raises(ValueError,match='independent inode'):
        stage_restore_checkpoint(source,store,digest)


def test_restored_assets_can_live_on_separate_scratch_and_reject_collision(tmp_path):
    tree=tmp_path/'external-tree';tree.mkdir()
    payload=tmp_path/'local-scratch'/'runtime';payload.mkdir(parents=True)
    (payload/'data.txt').write_text('verified asset')
    target=bind_restored_asset(tree,'/original/assets',payload)
    assert (Path(target)/'data.txt').read_text()=='verified asset'
    assert Path(target).is_symlink()
    assert bind_restored_asset(tree,'/original/assets',payload)==target
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='collides'):
        bind_restored_asset(tree,'/original/assets',other)


def test_extracts_and_validates_runtime_archive(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'module.py').write_text('value=4\n')
    archive=tmp_path/'runtime.tar';seal(root,archive,tmp_path/'receipt.json')
    manifest=extract_archive(archive,tmp_path/'restore');assert manifest['schema']=='natlang.local-recovery-runtime-archive/1'
    assert (tmp_path/'restore/runtime/module.py').read_text()=='value=4\n'


def test_reuses_matching_fixture_without_rewriting_and_rejects_extra_module(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'module.py').write_text('value=4\n')
    archive=tmp_path/'runtime.tar';seal(root,archive,tmp_path/'receipt.json')
    dest=tmp_path/'restore';extract_archive(archive,dest)
    module=dest/'runtime/module.py';before=module.stat()
    extract_archive(archive,dest)
    assert module.stat().st_ino==before.st_ino
    assert module.stat().st_mtime_ns==before.st_mtime_ns
    assert module.stat().st_ctime_ns==before.st_ctime_ns
    (dest/'runtime/unbound.py').write_text('extra code')
    with pytest.raises(ValueError,match='unbound member'):
        extract_archive(archive,dest)


def test_reused_fixture_must_match_archive_bytes(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'module.py').write_text('value=4\n')
    archive=tmp_path/'runtime.tar';seal(root,archive,tmp_path/'receipt.json')
    dest=tmp_path/'restore';extract_archive(archive,dest)
    (dest/'runtime/module.py').write_text('value=5\n')
    with pytest.raises(ValueError,match='differs from archive'):
        extract_archive(archive,dest)


def test_restore_rejects_escaping_archive_and_nonabsolute_source(tmp_path):
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as tar:
        info=tarfile.TarInfo('runtime/../../escape');data=b'bad';info.size=len(data);tar.addfile(info,io.BytesIO(data))
    with pytest.raises((ValueError,tarfile.FilterError)):extract_archive(archive,tmp_path/'restore')
    assert not (tmp_path/'escape').exists()
    with pytest.raises(ValueError,match='absolute'):_mapped_path(tmp_path,'relative/source')


def test_closure_copy_is_bounded_ordered_and_propagates_failure(tmp_path, monkeypatch):
    import threading, time
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    lock=threading.Lock(); state={'active':0,'peak':0}
    def copy(source,target,digest):
        with lock:
            state['active']+=1;state['peak']=max(state['peak'],state['active'])
        try:
            time.sleep(.01)
            if source=='bad':raise ValueError('invalid content')
            return [digest]
        finally:
            with lock:state['active']-=1
    monkeypatch.setattr(q,'verified_recovery_blob',copy)
    rows=[{'path':str(i),'sha256':str(i)} for i in range(12)]
    result=q.copy_closure_blobs(rows,tmp_path)
    assert list(result)==[(x['path'],x['sha256']) for x in rows]
    assert 1 < state['peak'] <= 4
    with pytest.raises(ValueError,match='invalid content'):
        q.copy_closure_blobs(rows+[{'path':'bad','sha256':'bad'}],tmp_path)


def test_closure_copy_refills_while_first_copy_is_slow(tmp_path, monkeypatch):
    import threading
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    progressed=threading.Event()
    def copy(source,target,digest):
        if source=='0':
            assert progressed.wait(2), 'later work did not advance while first copy waited'
        if source=='9':progressed.set()
        return [digest]
    monkeypatch.setattr(q,'verified_recovery_blob',copy)
    rows=[{'path':str(i),'sha256':str(i)} for i in range(12)]
    assert len(q.copy_closure_blobs(rows,tmp_path))==12


def test_historical_program_routes_use_archived_bytes_and_fail_on_conflict():
    from scripts.neuralese_data.qualify_local_checkpoint_recovery import add_historical_program_routes
    import pytest
    row={'original_path':'/old/program.py','archive':{'path':'/archive/exact.py','sha256':'a'*64},'receipt':{'path':'/receipt','sha256':'b'*64}}
    paths={'/archive/exact.py':'/restored/exact.py'}
    add_historical_program_routes(paths,[row]);assert paths['/old/program.py']=='/restored/exact.py'
    with pytest.raises(ValueError,match='conflicts'):
        add_historical_program_routes({'/old/program.py':'/other','/archive/exact.py':'/restored/exact.py'},[row])
    with pytest.raises(ValueError,match='absent'):
        add_historical_program_routes({},[row])


def test_sealed_restore_publication_retries_without_restore_and_rejects_changed_bytes(tmp_path,monkeypatch):
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    monkeypatch.setattr(q,'validate_recovery_plan_shape',lambda plan:None)
    monkeypatch.setattr(q,'validate_recovery_plan_dependencies',lambda plan,**kw:None)
    monkeypatch.setattr(q,'recovery_closure_identity',lambda plan:'closure')
    draft=tmp_path/'draft';draft.write_text('exact draft')
    checkpoint=tmp_path/'owner';checkpoint.write_bytes(b'owner')
    independent=tmp_path/'independent';independent.write_bytes(b'owner')
    store=tmp_path/'store';blob=store/'closure'/('a'*64)
    blob.parent.mkdir(parents=True);blob.write_bytes(b'checked blob')
    qualification=tmp_path/'qualification.json';output=tmp_path/'plan.json';evidence_path=tmp_path/'evidence.json'
    closure=[{'role':'runtime','path':'/source','sha256':'a'*64}]
    plan={'destination':str(store),'inputs':{},'controller':{'config':'config'},'restore':{},
        'closure':closure+[{'role':'controls','path':str(qualification),'sha256':'0'*64}],
        'restore_qualification':{'path':str(qualification),'sha256':'0'*64}}
    evidence={'schema':'natlang.local-recovery-restore-evidence/1',
        'checkpoint':{'path':str(checkpoint),'sha256':q.file_sha256(checkpoint)},
        'independent_checkpoint':{'path':str(independent),'identity':q._identity(independent)},
        'sample':{'custody':{'step':7}},'host_contract':q._host_contract(),'used_host_executables':[],
        'sealed_inputs':{'schema':'natlang.local-recovery-checked-inputs/1',
            'draft':q._checked_hash(draft),'checkpoint':q._checked_hash(checkpoint),
            'qualification_path':str(qualification),'base_closure_sha256':q.canonical_json_sha256_hex(closure),
            'closure_blobs':[{'path':str(blob),'sha256':'a'*64,'identity':q._identity(blob)}]}}
    evidence_path.write_text(json.dumps(evidence));sha=q.file_sha256(evidence_path)
    import copy
    original=q.write_json_durable
    def fail_plan(path,value):
        if path==output:raise OSError('reporting failure')
        original(path,value)
    monkeypatch.setattr(q,'write_json_durable',fail_plan)
    with pytest.raises(OSError,match='reporting failure'):
        q.publish_checked_evidence(copy.deepcopy(plan),draft,checkpoint,output,qualification,evidence_path,sha)
    assert qualification.exists() and not output.exists()
    monkeypatch.setattr(q,'write_json_durable',original)
    result=q.publish_checked_evidence(copy.deepcopy(plan),draft,checkpoint,output,qualification,evidence_path,sha)
    assert Path(result['plan']['path']).is_file()
    blob.write_bytes(b'altered bytes')
    with pytest.raises(ValueError,match='closure blob changed'):
        q.publish_checked_evidence(copy.deepcopy(plan),draft,checkpoint,output,qualification,evidence_path,sha)
    with pytest.raises(ValueError,match='changed'):
        q.publish_checked_evidence(copy.deepcopy(plan),draft,checkpoint,output,qualification,evidence_path,'0'*64)


@pytest.mark.parametrize('tamper', ['none', 'changed', 'replacement', 'extra', 'extra-dir', 'missing', 'asset-link'])
def test_resumed_source_tree_rechecks_exact_bytes_links_and_inventory(tmp_path, tamper):
    import hashlib, os
    from scripts.neuralese_data.qualify_local_checkpoint_recovery import validate_restored_source_tree
    root=tmp_path/'root';tree=root/'tree';tree.mkdir(parents=True)
    blob=tmp_path/'store/closure/blob';blob.parent.mkdir(parents=True);blob.write_bytes(b'authenticated bytes')
    sha=hashlib.sha256(blob.read_bytes()).hexdigest()
    target=tree/'source/file';target.parent.mkdir();os.link(blob,target)
    assets=tmp_path/'fixture/runtime';assets.mkdir(parents=True)
    link=tree/'assets';link.symlink_to(assets,target_is_directory=True)
    closure=[{'role':'source','path':'/source/file','sha256':sha}]
    blobs={('/source/file',sha):{'path':str(blob)}}
    if tamper=='changed': target.write_bytes(b'forged bytes')
    if tamper=='replacement': target.unlink();target.write_bytes(b'authenticated bytes')
    if tamper=='extra': (tree/'unbound.py').write_text('untrusted')
    if tamper=='extra-dir': (tree/'unbound').mkdir()
    if tamper=='missing': target.unlink()
    if tamper=='asset-link':
        other=tmp_path/'other';other.mkdir();link.unlink();link.symlink_to(other,target_is_directory=True)
    if tamper=='none':
        before=target.stat(); identities=validate_restored_source_tree(root,closure,blobs,set(),{'/assets':assets})
        assert identities[str(target)] == [before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns]
        assert target.stat().st_ino==before.st_ino and target.stat().st_ctime_ns==before.st_ctime_ns
    else:
        with pytest.raises(ValueError):validate_restored_source_tree(root,closure,blobs,set(),{'/assets':assets})


def test_resumed_source_tree_requires_bound_checkpoint_and_cannot_mix_evidence(tmp_path, monkeypatch):
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    draft=tmp_path/'draft.json';draft.write_text(json.dumps({'closure':[]}))
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'checkpoint')
    monkeypatch.setattr(q,'validate_recovery_plan_shape',lambda plan:None)
    monkeypatch.setattr(q,'validate_recovery_plan_dependencies',lambda plan,**kwargs:None)
    args=(draft,checkpoint,tmp_path,tmp_path/'output',tmp_path/'qualification',tmp_path/'evidence')
    with pytest.raises(ValueError,match='requires an exact checkpoint'):
        q.qualify(*args,resume_source_tree=tmp_path/'tree')
    with pytest.raises(ValueError,match='cannot also resume'):
        q.qualify(*args,resume_source_tree=tmp_path/'tree',resume_checkpoint_sha256='a'*64,resume_evidence_sha256='b'*64)


def test_resumed_tree_rechecks_identity_and_inventory_after_smoke(tmp_path):
    import hashlib, os
    from scripts.neuralese_data.qualify_local_checkpoint_recovery import validate_restored_source_tree
    root=tmp_path/'root';tree=root/'tree';tree.mkdir(parents=True)
    blob=tmp_path/'blob';blob.write_bytes(b'unchanged')
    digest=hashlib.sha256(blob.read_bytes()).hexdigest()
    target=tree/'input';os.link(blob,target)
    closure=[{'role':'sources','path':'/input','sha256':digest}]
    blobs={('/input',digest):{'path':str(blob)}}
    before=validate_restored_source_tree(root,closure,blobs,set(),{})
    assert validate_restored_source_tree(root,closure,blobs,set(),{},prior_identities=before)==before
    (tree/'injected').write_text('unexpected')
    with pytest.raises(ValueError,match='unbound'):
        validate_restored_source_tree(root,closure,blobs,set(),{},prior_identities=before)
    (tree/'injected').unlink();target.write_bytes(b'altered')
    with pytest.raises(ValueError,match='changed during'):
        validate_restored_source_tree(root,closure,blobs,set(),{},prior_identities=before)


def test_resumed_tree_rejects_wrong_checkpoint_before_staging(tmp_path,monkeypatch):
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    store=tmp_path/'store';store.mkdir()
    draft=tmp_path/'draft.json';draft.write_text(json.dumps({'closure':[],'destination':str(store)}))
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'actual checkpoint')
    monkeypatch.setattr(q,'validate_recovery_plan_shape',lambda plan:None)
    monkeypatch.setattr(q,'validate_recovery_plan_dependencies',lambda plan,**kwargs:None)
    def unexpected(*args):raise AssertionError('staging must not run')
    monkeypatch.setattr(q,'stage_restore_checkpoint',unexpected)
    with pytest.raises(ValueError,match='checkpoint differs'):
        q.qualify(draft,checkpoint,tmp_path,tmp_path/'output',tmp_path/'qualification',tmp_path/'evidence',
                  resume_source_tree=tmp_path/'tree',resume_checkpoint_sha256='0'*64)


@pytest.mark.parametrize('tamper', ['none','plan','source-map'])
def test_resume_uses_verified_existing_tree_and_always_reruns_isolated_checks(tmp_path, monkeypatch, tamper):
    import hashlib, os
    from types import SimpleNamespace
    from scripts.neuralese_data import qualify_local_checkpoint_recovery as q
    monkeypatch.setattr(q,'validate_recovery_plan_shape',lambda plan:None)
    monkeypatch.setattr(q,'validate_recovery_plan_dependencies',lambda plan,**kwargs:None)
    store=tmp_path/'store';store.mkdir();scratch=tmp_path/'scratch';scratch.mkdir()
    fixture=tmp_path/'environment';fixture.mkdir();source=tmp_path/'source';source.mkdir()
    checkpoint=tmp_path/'checkpoint';checkpoint.write_bytes(b'actual checkpoint')
    checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    names={'runtime':'runtime.tar','environment':'venv.tar','system-python':'stdlib.tar',
           'interpreter':'python','node':'node','base':'model','sources':'train',
           'held':'held','sample':'sample','asset-inventory':'inventory','asset-guard':'guard'}
    refs={};closure=[]
    for role,name in names.items():
        path=source/name
        path.write_text('{"roots":{}}' if role=='asset-inventory' else role)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        refs[role]=str(path);closure.append({'role':role,'path':str(path),'sha256':digest})
        blob=store/'closure'/digest;blob.parent.mkdir(exist_ok=True);blob.write_bytes(path.read_bytes())
    restore={'runtime_archive':refs['runtime'],'environment_archive':refs['environment'],
        'system_python_archive':refs['system-python'],'asset_archives':[],
        'interpreter_path':refs['interpreter'],'node_path':refs['node'],'base_path':str(source),
        'sample_graph':refs['sample'],'tokenizer_fingerprint':'a'*64,
        'asset_inventory':refs['asset-inventory'],'asset_guard':refs['asset-guard']}
    draft=tmp_path/'draft.json';qualification=tmp_path/'qualification';output=tmp_path/'output';evidence=tmp_path/'evidence'
    plan={'destination':str(store),'closure':closure,'restore':restore,'historical_programs':[],
          'inputs':{'train':{'path':refs['sources']},'held':{'path':refs['held']}}}
    draft.write_text(json.dumps(plan));staged={**plan,'closure':closure+[{'role':'controls','path':str(qualification),'sha256':'0'*64}],
        'restore_qualification':{'path':str(qualification),'sha256':'0'*64}}
    root=scratch/'interrupted';tree=root/'tree';tree.mkdir(parents=True)
    mapping={}
    for row in closure:
        if row['role'] in ('runtime','environment','system-python'):continue
        target=q._mapped_path(tree,row['path']);target.parent.mkdir(parents=True,exist_ok=True)
        os.link(store/'closure'/row['sha256'],target)
        mapping[row['path']]=str(target);mapping[str(Path(row['path']).resolve())]=str(target)
    (root/'restore-plan.json').write_text(json.dumps(staged))
    (root/'.natlang-source-map.json').write_text(json.dumps({'schema':'natlang.staged-source-map/1','paths':mapping}))
    def extract(archive,destination,*,environment=False):
        destination=Path(destination)
        if environment:
            (destination/'venv/bin').mkdir(parents=True,exist_ok=True)
            link=destination/'venv/bin/python'
            if not link.exists() and not link.is_symlink():link.symlink_to('/usr/bin/python3')
        else:
            (destination/'runtime/ts-host/dist/teacher').mkdir(parents=True,exist_ok=True)
            (destination/'runtime/ts-host/dist/teacher/collector.js').write_text('')
        return {'schema':'archive'}
    monkeypatch.setattr(q,'extract_archive',extract)
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if '-c' in command:stdout='RECOVERY_SMOKE='+json.dumps({'mapped_executables':[],'custody':{}})+'\n'
        elif '--input-type=module' in command:stdout='NODE_RECOVERY='+json.dumps('')+'\n'
        else:stdout='assets checked'
        return SimpleNamespace(returncode=0,stdout=stdout,stderr='')
    monkeypatch.setattr(q.subprocess,'run',run)
    monkeypatch.setattr(q,'publish_checked_evidence',lambda *args:{'status':'published'})
    if tamper=='plan':(root/'restore-plan.json').write_text('{}')
    if tamper=='source-map':(root/'.natlang-source-map.json').write_text('{}')
    args=(draft,checkpoint,scratch,output,qualification,evidence)
    kwargs={'environment_fixture':fixture,'resume_source_tree':root,'resume_checkpoint_sha256':checkpoint_sha}
    if tamper!='none':
        with pytest.raises(ValueError,match='differs'):q.qualify(*args,**kwargs)
        assert calls==[]
        return
    result=q.qualify(*args,**kwargs)
    assert result=={'status':'published'} and len(calls)==3
    assert '-c' in calls[0] and '--input-type=module' in calls[1]
    assert root.is_dir() and (root/'restore-plan.json').is_file()
