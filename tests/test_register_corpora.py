"""Registration publishes only its immutable selected paths from canonical main."""
import json
import shutil
import subprocess
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import register_corpora as rc  # noqa: E402
import sync_training_corpora as sync  # noqa: E402

REAL = Path(__file__).resolve().parents[1]


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def setup(tmp_path, monkeypatch):
    origin, repo = tmp_path / 'origin.git', tmp_path / 'repo'
    sh('git', 'init', '-q', '--bare', '-b', 'main', str(origin), cwd=tmp_path)
    sh('git', 'clone', '-q', str(origin), str(repo), cwd=tmp_path)
    for args in (('user.email', 't@t'), ('user.name', 't')):
        sh('git', 'config', *args, cwd=repo)
    (repo / 'scripts').mkdir()
    shutil.copy(REAL / 'scripts/sync_training_corpora.py', repo / 'scripts/sync_training_corpora.py')
    shutil.copy(REAL / 'scripts/git_publish.py', repo / 'scripts/git_publish.py')
    (repo / 'foreign-dirty.txt').write_text('original dirty\n')
    (repo / 'foreign-staged.txt').write_text('original staged\n')
    (repo / 'training/corpus-manifests').mkdir(parents=True)
    (repo / 'training/corpus-manifests/.keep').write_text('')
    (repo / '.gitignore').write_text('data/\n')
    registry = {'schema': 'natlang.neuralese-corpus-registry/1', 'corpora': [{'id': 'old', 'owner': 'dgx', 'path': 'x'}]}
    (repo / rc.REGISTRY).write_text(json.dumps(registry, indent=2) + '\n')
    sh('git', 'add', '-A', cwd=repo)
    sh('git', 'commit', '-q', '-m', 'init', cwd=repo)
    sh('git', 'push', '-q', 'origin', 'main', cwd=repo)
    data = repo / 'data/corpora/new'
    data.mkdir(parents=True)
    (data / 'a.jsonl').write_text('{"x": 1}\n')
    monkeypatch.setattr(rc, 'REPO', repo)
    monkeypatch.setattr(rc, 'LOCK', tmp_path / 'lock')
    # Guard: every git call must reach the temporary origin, never the real one.
    real_git = rc.git

    def guarded(*args, cwd=None, check=True):
        url = sh('git', 'remote', 'get-url', 'origin', cwd=cwd or rc.REPO).strip()
        assert url == str(origin), f'test git call reached {url}'
        return real_git(*args, cwd=cwd, check=check)

    monkeypatch.setattr(rc, 'git', guarded)
    return origin, repo


def test_registers_from_main_preserving_foreign_staged_and_dirty_files(tmp_path, monkeypatch):
    origin, repo = setup(tmp_path, monkeypatch)
    (repo/'foreign-dirty.txt').write_text('foreign unstaged\n')
    (repo/'foreign-staged.txt').write_text('foreign staged\n')
    sh('git','add','foreign-staged.txt',cwd=repo)
    entry = {'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    shown=json.loads(sh('git','show',f'main:{rc.REGISTRY}',cwd=origin))
    assert [item['id'] for item in shown['corpora']]==['old','new']
    manifest=json.loads(sh('git','show','main:training/corpus-manifests/new.json',cwd=origin))
    assert manifest['files'][0]['path']=='a.jsonl'
    assert (repo/'foreign-dirty.txt').read_text()=='foreign unstaged\n'
    assert sh('git','diff','--cached','--name-only',cwd=repo).splitlines()==['foreign-staged.txt']
    assert sh('git','show',':foreign-staged.txt',cwd=repo)=='foreign staged\n'
    assert sh('git','stash','list',cwd=repo)==''
    assert rc.register([entry],'again',check=False)==[]


@pytest.mark.parametrize('staged',[False,True])
def test_dirty_or_staged_registry_is_actionable_and_preserved(tmp_path,monkeypatch,staged):
    origin,repo=setup(tmp_path,monkeypatch)
    registry=repo/rc.REGISTRY;registry.write_text(registry.read_text().replace('"x"','"foreign"'))
    if staged:sh('git','add',rc.REGISTRY,cwd=repo)
    before=registry.read_bytes();head=sh('git','rev-parse','main',cwd=origin)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='foreign or concurrent'):
        rc.register([entry],'Register new',check=False)
    assert registry.read_bytes()==before and sh('git','rev-parse','main',cwd=origin)==head
    assert bool(sh('git','diff','--cached','--name-only',cwd=repo))==staged
    assert sh('git','stash','list',cwd=repo)==''


def test_untracked_selected_manifest_refused_without_overwrite(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch);manifest=repo/rc.MANIFESTS/'new.json'
    manifest.write_text('{"foreign": true}\n');head=sh('git','rev-parse','main',cwd=origin)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='foreign or concurrent'):
        rc.register([entry],'Register new',check=False)
    assert manifest.read_text()=='{"foreign": true}\n' and sh('git','rev-parse','main',cwd=origin)==head


def test_actual_push_race_rebuilds_without_touching_other_work(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','race_git_publish')
    original_module=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original_module(path,name))
    calls=[]
    def race(attempt):
        if calls:return
        calls.append(attempt);other=tmp_path/'other'
        sh('git','clone','-q',str(origin),str(other),cwd=tmp_path)
        (other/'other.txt').write_text('concurrent\n')
        sh('git','add','other.txt',cwd=other)
        sh('git','-c','user.email=o@o','-c','user.name=o','commit','-q','-m','other',cwd=other)
        sh('git','push','-q','origin','HEAD:main',cwd=other)
    monkeypatch.setattr(publisher,'before_push',race)
    (repo/'foreign-dirty.txt').write_text('untouched\n')
    (repo/'foreign-staged.txt').write_text('staged\n');sh('git','add','foreign-staged.txt',cwd=repo)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    assert calls==[0] and sh('git','show','main:other.txt',cwd=origin)=='concurrent\n'
    assert (repo/'foreign-dirty.txt').read_text()=='untouched\n'
    assert sh('git','diff','--cached','--name-only',cwd=repo).splitlines()==['foreign-staged.txt']
    assert [item['id'] for item in json.loads(sh('git','show',f'main:{rc.REGISTRY}',cwd=origin))['corpora']]==['old','new']


def test_foreign_registry_edit_during_preparation_is_never_published(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    sync_module=rc._module(repo/'scripts/sync_training_corpora.py','race_sync')
    original_publish=sync_module.publish;original_module=rc._module
    def publish(*args,**kwargs):
        result=original_publish(*args,**kwargs)
        path=repo/rc.REGISTRY;path.write_text(path.read_text().replace('"x"','"foreign"'))
        return result
    monkeypatch.setattr(sync_module,'publish',publish)
    monkeypatch.setattr(rc,'_module',lambda path,name: sync_module if name=='natlang_registration_corpus_sync' else original_module(path,name))
    head=sh('git','rev-parse','main',cwd=origin)
    with pytest.raises(ValueError,match='foreign or concurrent'):
        rc.register([{'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}],'Register new',check=False)
    assert 'foreign' in (repo/rc.REGISTRY).read_text() and sh('git','rev-parse','main',cwd=origin)==head


def test_changed_entry_under_registered_id_is_refused(tmp_path, monkeypatch):
    setup(tmp_path, monkeypatch)
    try:
        rc.register([{'id': 'old', 'owner': 'dgx', 'path': 'y'}], 'm', check=False)
    except ValueError as error:
        assert 'immutable' in str(error)
    else:
        raise AssertionError('expected refusal')


def test_publish_reuses_manifest_only_while_entry_and_file_stats_match(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    data = repo / 'data/corpora/new'
    manifests = repo / 'training/corpus-manifests'
    data.mkdir(parents=True)
    manifests.mkdir(parents=True)
    source = data / 'a.jsonl'
    source.write_text('{"x": 1}\n')
    entry = {'id': 'new', 'owner': 'dgx', 'path': 'data/corpora/new', 'include': ['*.jsonl']}
    cache = tmp_path / 'manifest-cache' / 'new.json'

    sync.publish(repo, entry, cache)
    digest = sync.digest

    def no_rehash(_path):
        raise AssertionError('unchanged files should reuse the verified manifest')

    monkeypatch.setattr(sync, 'digest', no_rehash)
    sync.publish(repo, entry, cache)
    assert json.loads((manifests / 'new.json').read_text())['files'][0]['sha256']

    # A newly selected file invalidates the cached file inventory even when prior files are unchanged.
    hashed = []

    def count_hash(path):
        hashed.append(str(path))
        return digest(path)

    monkeypatch.setattr(sync, 'digest', count_hash)
    (data / 'b.jsonl').write_text('{"y": 1}\n')
    try:
        sync.publish(repo, entry, cache)
    except ValueError as error:
        assert 'immutable snapshot changed' in str(error)
    else:
        raise AssertionError('a new selected file must rebuild, not reuse, the old manifest')
    assert len(hashed) == 2

    # Same path and size still cannot reuse a stale manifest after a content write.
    hashed.clear()
    source.write_text('{"x": 2}\n')
    try:
        sync.publish(repo, entry, cache)
    except ValueError as error:
        assert 'immutable snapshot changed' in str(error)
    else:
        raise AssertionError('changed content must rebuild and reject the immutable snapshot')
    assert len(hashed) == 2


def test_concurrent_conflicting_registry_push_is_not_overwritten(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','conflict_git_publish');original=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original(path,name))
    calls=[]
    def race(_attempt):
        if calls:return
        calls.append(True);other=tmp_path/'other'
        sh('git','clone','-q',str(origin),str(other),cwd=tmp_path)
        path=other/rc.REGISTRY;registry=json.loads(path.read_text())
        registry['corpora'].append({'id':'new','owner':'pop','path':'foreign'})
        path.write_text(json.dumps(registry,indent=2)+'\n')
        sh('git','add',rc.REGISTRY,cwd=other)
        sh('git','-c','user.email=o@o','-c','user.name=o','commit','-q','-m','foreign registry',cwd=other)
        sh('git','push','-q','origin','HEAD:main',cwd=other)
    monkeypatch.setattr(publisher,'before_push',race)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='not published'):
        rc.register([entry],'Register new',check=False)
    assert json.loads(sh('git','show',f'main:{rc.REGISTRY}',cwd=origin))['corpora'][-1]['path']=='foreign'
    assert not (repo/rc.MANIFESTS/'new.json').exists()
    assert json.loads((repo/rc.REGISTRY).read_text())['corpora']==[{'id':'old','owner':'dgx','path':'x'}]


def test_selected_registry_edited_after_plan_is_not_published_or_overwritten(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','late_edit_git_publish');original=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original(path,name))
    def late_edit(attempt):
        if attempt==0:
            path=repo/rc.REGISTRY;path.write_text(path.read_text().replace('"x"','"late foreign edit"'))
    monkeypatch.setattr(publisher,'before_push',late_edit)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='not published'):
        rc.register([entry],'Register new',check=False)
    assert 'late foreign edit' in (repo/rc.REGISTRY).read_text()
    committed=json.loads(sh('git','show',f'main:{rc.REGISTRY}',cwd=origin))
    assert committed['corpora']==[{'id':'old','owner':'dgx','path':'x'}]


def test_candidate_admission_gate_receives_exact_proposed_registry_and_manifest(tmp_path,monkeypatch):
    from types import SimpleNamespace
    origin,repo=setup(tmp_path,monkeypatch);(repo/'ts-host').mkdir()
    (repo/'data/corpora/new/turns.jsonl').write_text('{"id":"a"}\n')
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl'],
           'candidate_selection_admission':{'decision':'test'}}
    real_run=subprocess.run;observed=[]
    def intercepted(command,*args,**kwargs):
        if command[:3]==['npm','run','--silent']:
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        if isinstance(command,(list,tuple)) and any('render_native_turn_corpus.py' in str(item) for item in command):
            registry=Path(command[command.index('--admission-registry-file')+1])
            manifest=Path(command[command.index('--admission-manifest-file')+1])
            assert json.loads(registry.read_text())['corpora'][-1]==entry
            assert json.loads(manifest.read_text())['id']=='new'
            assert '--verify-admission-only' in command
            observed.append(True)
            return SimpleNamespace(returncode=1,stdout='',stderr='synthetic admission rejection')
        return real_run(command,*args,**kwargs)
    monkeypatch.setattr(rc.subprocess,'run',intercepted)
    head=sh('git','rev-parse','main',cwd=origin)
    with pytest.raises(ValueError,match='synthetic admission rejection'):
        rc.register([entry],'Register new',check=False)
    assert observed==[True] and sh('git','rev-parse','main',cwd=origin)==head
    assert not (repo/rc.MANIFESTS/'new.json').exists()


def test_selected_source_change_before_push_invalidates_prepared_manifest(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','source_race_publisher');original=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original(path,name))
    def change(_attempt):
        (repo/'data/corpora/new/a.jsonl').write_text('{"x": 2}\n')
    monkeypatch.setattr(publisher,'before_push',change)
    head=sh('git','rev-parse','main',cwd=origin)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='not published'):
        rc.register([entry],'Register new',check=False)
    assert sh('git','rev-parse','main',cwd=origin)==head
    assert (repo/'data/corpora/new/a.jsonl').read_text()=='{"x": 2}\n'
    assert not (repo/rc.MANIFESTS/'new.json').exists()


def test_registration_preserves_canonical_data_symlink(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch);external=tmp_path/'external-data';external.mkdir()
    (external/'source.jsonl').write_text('{"link":true}\n')
    entry={'id':'linked','owner':'dgx','path':'data/corpora/linked','include':['*.jsonl']}
    assert rc.register([entry],'Register link',links={'data/corpora/linked':str(external)},check=False)==['linked']
    assert (repo/'data/corpora/linked').is_symlink()
    manifest=json.loads(sh('git','show','main:training/corpus-manifests/linked.json',cwd=origin))
    assert manifest['files'][0]['path']=='source.jsonl'


def test_concurrent_distinct_registry_addition_is_semantically_rebased(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','registry_race_publisher');original=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original(path,name))
    calls=[]
    def race(_attempt):
        if calls:return
        calls.append(True);other=tmp_path/'other'
        sh('git','clone','-q',str(origin),str(other),cwd=tmp_path)
        path=other/rc.REGISTRY;registry=json.loads(path.read_text())
        registry['corpora'].append({'id':'concurrent','owner':'pop','path':'other'})
        path.write_text(json.dumps(registry,indent=2)+'\n')
        sh('git','add',rc.REGISTRY,cwd=other)
        sh('git','-c','user.email=o@o','-c','user.name=o','commit','-q','-m','other registry',cwd=other)
        sh('git','push','-q','origin','HEAD:main',cwd=other)
    monkeypatch.setattr(publisher,'before_push',race)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    registry=json.loads(sh('git','show',f'main:{rc.REGISTRY}',cwd=origin))
    assert [item['id'] for item in registry['corpora']]==['old','concurrent','new']
    assert json.loads((repo/rc.REGISTRY).read_text())==registry


def test_replace_preserves_existing_manifest_bytes_and_refuses_changed_snapshot(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    manifest_name='training/corpus-manifests/new.json'
    before=sh('git','rev-parse',f'main:{manifest_name}',cwd=origin)
    replacement={**entry,'notes':'reviewed metadata only'}
    assert rc.register([replacement],'Update metadata',replace=True,check=False)==['new']
    assert sh('git','rev-parse',f'main:{manifest_name}',cwd=origin)==before
    head=sh('git','rev-parse','main',cwd=origin)
    (repo/'data/corpora/new/extra.jsonl').write_text('{"new":true}\n')
    with pytest.raises(ValueError,match='immutable snapshot changed'):
        rc.register([{**replacement,'notes':'second change'}],'Unsafe update',replace=True,check=False)
    assert sh('git','rev-parse','main',cwd=origin)==head
    assert sh('git','rev-parse',f'main:{manifest_name}',cwd=origin)==before


def test_staged_selected_manifest_is_preserved_and_blocks_publication(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch);manifest=repo/rc.MANIFESTS/'new.json'
    manifest.write_text('{"original":true}\n')
    sh('git','add',str(manifest.relative_to(repo)),cwd=repo)
    sh('git','commit','-q','-m','existing manifest',cwd=repo)
    sh('git','push','-q','origin','main',cwd=repo)
    manifest.write_text('{"foreign":true}\n');sh('git','add',str(manifest.relative_to(repo)),cwd=repo)
    before=sh('git','rev-parse','main',cwd=origin)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='foreign or concurrent'):
        rc.register([entry],'Register new',check=False)
    assert manifest.read_text()=='{"foreign":true}\n'
    assert sh('git','show',':training/corpus-manifests/new.json',cwd=repo)=='{"foreign":true}\n'
    assert sh('git','rev-parse','main',cwd=origin)==before


def test_red_quick_check_preserves_complete_failure_without_publishing(tmp_path,monkeypatch):
    from types import SimpleNamespace
    origin,repo=setup(tmp_path,monkeypatch);real_run=subprocess.run
    def run(command,*args,**kwargs):
        if command==['scripts/check-main.sh','--quick']:
            return SimpleNamespace(returncode=1,stdout='full stdout',stderr='full stderr')
        return real_run(command,*args,**kwargs)
    monkeypatch.setattr(rc.subprocess,'run',run)
    before=sh('git','rev-parse','main',cwd=origin)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='complete latest failure'):
        rc.register([entry],'Register new',check=True)
    failure=json.loads((tmp_path/'register-corpora-last-check-failure.json').read_text())
    assert failure['checkout']==str(repo) and failure['stdout']=='full stdout' and failure['stderr']=='full stderr'
    assert sh('git','rev-parse','main',cwd=origin)==before
    assert not (repo/rc.MANIFESTS/'new.json').exists()


def test_safely_syncs_new_upstream_registry_before_registration(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch);other=tmp_path/'other'
    sh('git','clone','-q',str(origin),str(other),cwd=tmp_path)
    path=other/rc.REGISTRY;registry=json.loads(path.read_text())
    registry['corpora'].append({'id':'already-upstream','owner':'pop','path':'other'})
    path.write_text(json.dumps(registry,indent=2)+'\n')
    sh('git','add',rc.REGISTRY,cwd=other)
    sh('git','-c','user.email=o@o','-c','user.name=o','commit','-q','-m','upstream',cwd=other)
    sh('git','push','-q','origin','HEAD:main',cwd=other)
    (repo/'foreign-staged.txt').write_text('staged\n');sh('git','add','foreign-staged.txt',cwd=repo)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    registry=json.loads(sh('git','show',f'main:{rc.REGISTRY}',cwd=origin))
    assert [row['id'] for row in registry['corpora']]==['old','already-upstream','new']
    assert sh('git','diff','--cached','--name-only',cwd=repo).splitlines()==['foreign-staged.txt']


def test_unpublished_local_registry_commit_is_never_swept_into_registration(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch);path=repo/rc.REGISTRY;registry=json.loads(path.read_text())
    registry['corpora'].append({'id':'local-only','owner':'pop','path':'other'})
    path.write_text(json.dumps(registry,indent=2)+'\n');sh('git','add',rc.REGISTRY,cwd=repo)
    sh('git','commit','-q','-m','local only',cwd=repo)
    before=sh('git','rev-parse','main',cwd=origin)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='unpublished local commits'):
        rc.register([entry],'Register new',check=False)
    assert sh('git','rev-parse','main',cwd=origin)==before
    assert json.loads(path.read_text())['corpora'][-1]['id']=='local-only'
