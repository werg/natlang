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
OLD = {'id': 'old', 'owner': 'dgx', 'path': 'x'}
NEW = {'id': 'new', 'owner': 'dgx', 'path': 'data/corpora/new', 'include': ['*.jsonl']}


def sh(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout


def text(entry):
    return json.dumps(entry, indent=2) + '\n'


def main_ids(origin):
    """Ids whose entry file is at origin's main."""
    listing = sh('git', 'ls-tree', '--name-only', 'main', rc.ENTRIES + '/', cwd=origin).split()
    return sorted(Path(name).name[:-5] for name in listing if name.endswith('.json'))


def main_entry(origin, ident):
    return sh('git', 'show', f'main:{rc.entry_file(ident)}', cwd=origin)


def push_from_other(tmp_path, origin, files, message):
    """Another clone commits FILES {rel: text} and pushes them to main."""
    other = tmp_path / 'other'
    sh('git', 'clone', '-q', str(origin), str(other), cwd=tmp_path)
    for rel, content in files.items():
        (other / rel).parent.mkdir(parents=True, exist_ok=True)
        (other / rel).write_text(content)
        sh('git', 'add', rel, cwd=other)
    sh('git', '-c', 'user.email=o@o', '-c', 'user.name=o', 'commit', '-q', '-m', message, cwd=other)
    sh('git', 'push', '-q', 'origin', 'HEAD:main', cwd=other)


def setup(tmp_path, monkeypatch):
    origin, repo = tmp_path / 'origin.git', tmp_path / 'repo'
    sh('git', 'init', '-q', '--bare', '-b', 'main', str(origin), cwd=tmp_path)
    sh('git', 'clone', '-q', str(origin), str(repo), cwd=tmp_path)
    for args in (('user.email', 't@t'), ('user.name', 't')):
        sh('git', 'config', *args, cwd=repo)
    (repo / 'scripts').mkdir()
    shutil.copy(REAL / 'scripts/sync_training_corpora.py', repo / 'scripts/sync_training_corpora.py')
    shutil.copy(REAL / 'scripts/git_publish.py', repo / 'scripts/git_publish.py')
    package = repo / 'training/neuralese/natlang_neuralese'
    (package / 'common').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'common/__init__.py').write_text('')
    shutil.copy(REAL / 'training/neuralese/natlang_neuralese/common/corpus_registry.py',
                package / 'common/corpus_registry.py')
    (repo / 'foreign-dirty.txt').write_text('original dirty\n')
    (repo / 'foreign-staged.txt').write_text('original staged\n')
    (repo / 'training/corpus-manifests').mkdir(parents=True)
    (repo / 'training/corpus-manifests/.keep').write_text('')
    (repo / '.gitignore').write_text('data/\n')
    (repo / 'training/corpus-registry.json').write_text(json.dumps({'schema': 'natlang.neuralese-corpus-registry/2'}, indent=2) + '\n')
    (repo / rc.ENTRIES).mkdir(parents=True)
    (repo / rc.entry_file('old')).write_text(text(OLD))
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
    assert main_ids(origin)==['new','old']
    assert main_entry(origin,'new')==text(entry) and main_entry(origin,'old')==text(OLD)
    manifest=json.loads(sh('git','show','main:training/corpus-manifests/new.json',cwd=origin))
    assert manifest['files'][0]['path']=='a.jsonl'
    assert (repo/'foreign-dirty.txt').read_text()=='foreign unstaged\n'
    assert sh('git','diff','--cached','--name-only',cwd=repo).splitlines()==['foreign-staged.txt']
    assert sh('git','show',':foreign-staged.txt',cwd=repo)=='foreign staged\n'
    assert sh('git','stash','list',cwd=repo)==''
    assert rc.register([entry],'again',check=False)==[]


@pytest.mark.parametrize('staged',[False,True])
def test_dirty_or_staged_selected_entry_is_actionable_and_preserved(tmp_path,monkeypatch,staged):
    origin,repo=setup(tmp_path,monkeypatch)
    registry=repo/rc.entry_file('new');registry.write_text(text({**NEW,'path':'foreign'}))
    if staged:sh('git','add',rc.entry_file('new'),cwd=repo)
    before=registry.read_bytes();head=sh('git','rev-parse','main',cwd=origin)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='foreign or concurrent'):
        rc.register([entry],'Register new',check=False)
    assert registry.read_bytes()==before and sh('git','rev-parse','main',cwd=origin)==head
    assert bool(sh('git','diff','--cached','--name-only',cwd=repo))==staged
    assert sh('git','stash','list',cwd=repo)==''


@pytest.mark.parametrize('staged',[False,True])
def test_dirty_or_staged_other_entry_does_not_block_and_is_preserved(tmp_path,monkeypatch,staged):
    origin,repo=setup(tmp_path,monkeypatch)
    other=repo/rc.entry_file('old');other.write_text(text({**OLD,'path':'foreign'}))
    if staged:sh('git','add',rc.entry_file('old'),cwd=repo)
    before=other.read_bytes()
    assert rc.register([NEW],'Register new',check=False)==['new']
    assert main_ids(origin)==['new','old'] and main_entry(origin,'old')==text(OLD)
    assert other.read_bytes()==before
    cached=sh('git','diff','--cached','--name-only',cwd=repo).splitlines()
    assert cached==([rc.entry_file('old')] if staged else [])
    if staged:assert sh('git','show',':'+rc.entry_file('old'),cwd=repo)==before.decode()
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
    assert main_ids(origin)==['new','old'] and main_entry(origin,'new')==text(entry)


@pytest.mark.parametrize('target',['new','old'])
def test_foreign_entry_edit_during_preparation_is_never_published(tmp_path,monkeypatch,target):
    origin,repo=setup(tmp_path,monkeypatch)
    sync_module=rc._module(repo/'scripts/sync_training_corpora.py','race_sync')
    original_publish=sync_module.publish;original_module=rc._module
    def publish(*args,**kwargs):
        result=original_publish(*args,**kwargs)
        (repo/rc.entry_file(target)).write_text(text({'id':target,'owner':'pop','path':'foreign'}))
        return result
    monkeypatch.setattr(sync_module,'publish',publish)
    monkeypatch.setattr(rc,'_module',lambda path,name: sync_module if name=='natlang_registration_corpus_sync' else original_module(path,name))
    head=sh('git','rev-parse','main',cwd=origin)
    if target=='new':
        with pytest.raises(ValueError,match='foreign or concurrent'):
            rc.register([NEW],'Register new',check=False)
        assert sh('git','rev-parse','main',cwd=origin)==head and main_ids(origin)==['old']
    else:  # a foreign edit of another entry is neither published nor a conflict
        assert rc.register([NEW],'Register new',check=False)==['new']
        assert main_ids(origin)==['new','old'] and main_entry(origin,'new')==text(NEW)
        assert main_entry(origin,'old')==text(OLD)
    assert 'foreign' in (repo/rc.entry_file(target)).read_text()


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


def test_concurrent_conflicting_entry_push_is_not_overwritten(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','conflict_git_publish');original=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original(path,name))
    calls=[]
    def race(_attempt):
        if calls:return
        calls.append(True)
        push_from_other(tmp_path,origin,{rc.entry_file('new'):text({'id':'new','owner':'pop','path':'foreign'})},'foreign entry')
    monkeypatch.setattr(publisher,'before_push',race)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    with pytest.raises(ValueError,match='not published'):
        rc.register([entry],'Register new',check=False)
    assert json.loads(main_entry(origin,'new'))['path']=='foreign'
    assert sh('git','ls-tree','--name-only','main',rc.MANIFESTS+'/new.json',cwd=origin)==''
    assert not (repo/rc.MANIFESTS/'new.json').exists()
    assert not (repo/rc.entry_file('new')).exists() and (repo/rc.entry_file('old')).read_text()==text(OLD)


@pytest.mark.parametrize('target',['new','old'])
def test_entry_edited_after_plan_is_not_published_or_overwritten(tmp_path,monkeypatch,target):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','late_edit_git_publish');original=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original(path,name))
    def late_edit(attempt):
        if attempt==0:
            (repo/rc.entry_file(target)).write_text(text({'id':target,'owner':'pop','path':'late foreign edit'}))
    monkeypatch.setattr(publisher,'before_push',late_edit)
    if target=='new':  # the selected entry file: blocks
        with pytest.raises(ValueError,match='not published'):
            rc.register([NEW],'Register new',check=False)
        assert main_ids(origin)==['old']
    else:  # another entry: neither published nor a conflict
        assert rc.register([NEW],'Register new',check=False)==['new']
        assert main_ids(origin)==['new','old'] and main_entry(origin,'new')==text(NEW)
    assert main_entry(origin,'old')==text(OLD)
    assert 'late foreign edit' in (repo/rc.entry_file(target)).read_text()


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
            corpora=json.loads(registry.read_text())['corpora']
            assert [row['id'] for row in corpora]==['new','old']
            assert {row['id']:row for row in corpora}=={'new':entry,'old':OLD}
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
    assert main_ids(origin)==['linked','old'] and main_entry(origin,'linked')==text(entry)


def test_concurrent_distinct_entry_addition_is_rebased(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    publisher=rc._module(repo/'scripts/git_publish.py','registry_race_publisher');original=rc._module
    monkeypatch.setattr(rc,'_module',lambda path,name: publisher if name=='natlang_registration_git_publish' else original(path,name))
    calls=[]
    def race(_attempt):
        if calls:return
        calls.append(True)
        push_from_other(tmp_path,origin,{rc.entry_file('concurrent'):text({'id':'concurrent','owner':'pop','path':'other'})},'other entry')
    monkeypatch.setattr(publisher,'before_push',race)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    assert main_ids(origin)==['concurrent','new','old']
    assert main_entry(origin,'new')==text(entry)
    assert json.loads(main_entry(origin,'concurrent'))=={'id':'concurrent','owner':'pop','path':'other'}
    # the checkout was advanced to main: its entry files are main's
    assert sorted(p.name[:-5] for p in (repo/rc.ENTRIES).iterdir())==main_ids(origin)
    for ident in main_ids(origin):
        assert (repo/rc.entry_file(ident)).read_text()==main_entry(origin,ident)


def test_replace_preserves_existing_manifest_bytes_and_refuses_changed_snapshot(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    manifest_name='training/corpus-manifests/new.json'
    before=sh('git','rev-parse',f'main:{manifest_name}',cwd=origin)
    replacement={**entry,'notes':'reviewed metadata only'}
    assert rc.register([replacement],'Update metadata',replace=True,check=False)==['new']
    assert sh('git','rev-parse',f'main:{manifest_name}',cwd=origin)==before
    assert main_entry(origin,'new')==text(replacement)
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


def test_new_upstream_entry_is_kept_by_registration(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch)
    upstream={'id':'already-upstream','owner':'pop','path':'other'}
    push_from_other(tmp_path,origin,{rc.entry_file('already-upstream'):text(upstream)},'upstream')
    (repo/'foreign-staged.txt').write_text('staged\n');sh('git','add','foreign-staged.txt',cwd=repo)
    entry={'id':'new','owner':'dgx','path':'data/corpora/new','include':['*.jsonl']}
    assert rc.register([entry],'Register new',check=False)==['new']
    assert main_ids(origin)==['already-upstream','new','old']
    assert main_entry(origin,'already-upstream')==text(upstream) and main_entry(origin,'new')==text(entry)
    assert sh('git','diff','--cached','--name-only',cwd=repo).splitlines()==['foreign-staged.txt']


def test_unpublished_local_commit_of_selected_entry_blocks_registration(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch);path=repo/rc.entry_file('new')
    path.write_text(text({'id':'new','owner':'pop','path':'local'}));sh('git','add',rc.entry_file('new'),cwd=repo)
    sh('git','commit','-q','-m','local only',cwd=repo)
    before=sh('git','rev-parse','main',cwd=origin)
    with pytest.raises(ValueError,match='unpublished local commits'):
        rc.register([NEW],'Register new',check=False)
    assert sh('git','rev-parse','main',cwd=origin)==before
    assert json.loads(path.read_text())['path']=='local'


def test_unpublished_local_commit_of_other_entry_is_never_swept_into_registration(tmp_path,monkeypatch):
    origin,repo=setup(tmp_path,monkeypatch);path=repo/rc.entry_file('local-only')
    path.write_text(text({'id':'local-only','owner':'pop','path':'other'}));sh('git','add',rc.entry_file('local-only'),cwd=repo)
    sh('git','commit','-q','-m','local only',cwd=repo)
    assert rc.register([NEW],'Register new',check=False)==['new']
    assert main_ids(origin)==['new','old']
    assert json.loads(path.read_text())['id']=='local-only'


def test_explicit_pin_is_refused_before_links_fetch_or_git_publication(tmp_path, monkeypatch):
    pin = tmp_path / 'pin'
    pin.mkdir()
    marker = pin / '.natlang-pin.json'
    marker.write_text(json.dumps({'source': str(tmp_path / 'owner'), 'commit': 'a' * 40}))
    monkeypatch.setattr(rc, 'REPO', pin)
    monkeypatch.setattr(rc, 'git', lambda *args, **kwargs: pytest.fail('pin refusal must precede git'))
    with pytest.raises(ValueError, match='pin'):
        rc.register([NEW], 'must not mutate pin', links={'data/corpora/new': '/outside'}, check=False)
    assert list(pin.iterdir()) == [marker]


def test_pinned_producer_default_publishes_owner_without_changing_read_snapshot(tmp_path, monkeypatch):
    import importlib.util
    from natlang_neuralese.common import data_registry as dr, corpus_registry
    origin, owner = setup(tmp_path, monkeypatch)
    pin = tmp_path / 'code-pin'
    sh('git', 'worktree', 'add', '--detach', str(pin), 'HEAD', cwd=owner)
    (pin / '.natlang-pin.json').write_text(json.dumps({'source': str(owner), 'commit': sh('git', 'rev-parse', 'HEAD', cwd=pin).strip()}))
    monkeypatch.setattr(dr, 'code_repo', lambda: pin)
    monkeypatch.delenv('NATLANG_REGISTRY_REPO', raising=False)
    spec = importlib.util.spec_from_file_location('pinned_default_registration', REAL / 'scripts/neuralese_data/register_corpora.py')
    producer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    assert producer.REPO == owner
    producer.LOCK = tmp_path / 'publisher.lock'
    before = sh('git', 'rev-parse', 'HEAD', cwd=pin)
    assert producer.register([NEW], 'owner publication from pinned producer', check=False) == ['new']
    assert sh('git', 'rev-parse', 'HEAD', cwd=pin) == before
    assert not (pin / producer.entry_file('new')).exists()
    assert {e['id'] for e in corpus_registry.load(pin)['corpora']} == {'old'}
    assert {e['id'] for e in corpus_registry.load(owner)['corpora']} == {'old', 'new'}


def test_dirty_stale_owner_recognizes_published_entry_and_can_publish_another(tmp_path, monkeypatch):
    origin, repo = setup(tmp_path, monkeypatch)
    original_head = sh('git', 'rev-parse', 'HEAD', cwd=repo)
    push_from_other(tmp_path, origin, {'foreign-dirty.txt': 'new published source\n'}, 'advance unrelated source')
    (repo / 'foreign-dirty.txt').write_text('uncommitted source work\n')
    (repo / 'foreign-staged.txt').write_text('staged source work\n')
    sh('git', 'add', 'foreign-staged.txt', cwd=repo)
    assert rc.register([NEW], 'first publication from stale owner', check=False) == ['new']
    assert sh('git', 'rev-parse', 'HEAD', cwd=repo) == original_head
    assert not (repo / rc.entry_file('new')).exists()
    published_head = sh('git', 'rev-parse', 'main', cwd=origin)
    assert rc.register([NEW], 'idempotent recognition needs no checkout sync', check=False) == []
    assert sh('git', 'rev-parse', 'main', cwd=origin) == published_head
    assert rc.register([{**NEW, 'id': 'another'}], 'next publication', check=False) == ['another']
    assert main_ids(origin) == ['another', 'new', 'old']
    assert sh('git', 'rev-parse', 'HEAD', cwd=repo) == original_head
    assert (repo / 'foreign-dirty.txt').read_text() == 'uncommitted source work\n'
    assert sh('git', 'show', ':foreign-staged.txt', cwd=repo) == 'staged source work\n'


def test_replace_uses_published_manifest_bytes_without_overwriting_stale_owner(tmp_path, monkeypatch):
    origin, repo = setup(tmp_path, monkeypatch)
    push_from_other(tmp_path, origin, {'foreign-dirty.txt': 'new source\n'}, 'advance unrelated source')
    (repo / 'foreign-dirty.txt').write_text('local source\n')
    assert rc.register([NEW], 'publish original', check=False) == ['new']
    original_manifest = sh('git', 'show', 'main:training/corpus-manifests/new.json', cwd=origin)
    updated = {**NEW, 'note': 'curated successor'}
    assert rc.register([updated], 'update metadata against published entry', replace=True, check=False) == ['new']
    assert main_entry(origin, 'new') == text(updated)
    assert sh('git', 'show', 'main:training/corpus-manifests/new.json', cwd=origin) == original_manifest
    assert not (repo / rc.entry_file('new')).exists()


def test_promotion_snapshot_cannot_overwrite_newer_published_curation(tmp_path, monkeypatch):
    origin, repo = setup(tmp_path, monkeypatch)
    assert rc.register([NEW], 'publish original', check=False) == ['new']
    selected_ref = sh('git', 'rev-parse', 'main', cwd=origin).strip()
    curated = {**NEW, 'license': 'new published curator decision'}
    assert rc.register([curated], 'curation moved', replace=True, check=False) == ['new']
    with pytest.raises(ValueError, match='curated registry metadata changed'):
        rc.register([{**NEW, 'roles': {'grpo-pool': {'order': 1}}}], 'stale promotion',
                    replace=True, check=False, expected_registry_ref=selected_ref)
    assert main_entry(origin, 'new') == text(curated)
