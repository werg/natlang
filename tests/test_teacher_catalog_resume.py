import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from scripts.coreweave import teacher_catalog_resume as mechanism
from scripts.coreweave.teacher_catalog_resume import changed_files, scan_file
from scripts.coreweave.teacher_generation_guardian import binding


def catalog():
    db = sqlite3.connect(':memory:')
    db.executescript('''CREATE TABLE files(path TEXT PRIMARY KEY,sha TEXT,bytes INTEGER,cursor INTEGER,complete INTEGER,reason TEXT);
        CREATE TABLE programs(sha TEXT PRIMARY KEY,id TEXT,body BLOB,family TEXT);
        CREATE TABLE occurrences(file TEXT,line INTEGER,row_sha TEXT,program_sha TEXT,reason TEXT,groups_json TEXT,PRIMARY KEY(file,line));
        CREATE TABLE omissions(file TEXT,line INTEGER,reason TEXT,PRIMARY KEY(file,line));''')
    return db


def row(group='g', split='train'):
    return dict(split=split, source_groups=[group], task=dict(program_ir=dict(
        id='p-' + group, version='natlang.program/2', split=split, source_groups=[group])))


def policy():
    def reason(r, program, protection, holds):
        if r['split'] != 'train': return 'nontrain'
        if set(r['source_groups']) & protection['held']: return 'held_group'
        return None
    return SimpleNamespace(reason_for=reason, encode=lambda x: json.dumps(x, sort_keys=True, separators=(',', ':')),
                           floor=lambda _: None)


def write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return dict(path=str(path))


def test_registry_append_keeps_completed_source_cursor(tmp_path):
    db = catalog(); p = policy(); protection = dict(held=set())
    first = write(tmp_path/'first.jsonl', [row(), row('h')])
    scan_file(db,p,first,protection,set())
    before = db.execute('SELECT * FROM files').fetchall()
    p.reason_for = lambda *a: pytest.fail('completed source was parsed again')
    scan_file(db,p,first,protection,set())
    assert db.execute('SELECT * FROM files').fetchall() == before
    p = policy(); second = write(tmp_path/'second.jsonl',[row('new')])
    scan_file(db,p,second,protection,set())
    assert db.execute('SELECT sum(cursor),sum(complete) FROM files').fetchone() == (3,2)


def test_source_mutation_rejected_even_for_completed_file(tmp_path):
    db=catalog(); entry=write(tmp_path/'a.jsonl',[row()]); p=policy()
    scan_file(db,p,entry,dict(held=set()),set())
    previous=db.execute('SELECT * FROM files').fetchall()
    write(tmp_path/'a.jsonl',[row('changed')])
    with pytest.raises(ValueError,match='committed source identity'):
        scan_file(db,p,entry,dict(held=set()),set())
    assert db.execute('SELECT * FROM files').fetchall()==previous


def test_policy_audit_preserves_extraction_cursor_and_replays_after_sidecar_failure(tmp_path):
    db=catalog(); p=policy(); entry=write(tmp_path/'a.jsonl',[row(),row('h')])
    scan_file(db,p,entry,dict(held=set()),set())
    previous=db.execute('SELECT * FROM files').fetchall()
    def interrupted(cursor): raise RuntimeError('sidecar unavailable')
    with pytest.raises(RuntimeError,match='sidecar unavailable'):
        scan_file(db,p,entry,dict(held={'h'}),set(),audit_cursor=0,audit_save=interrupted)
    assert db.execute('SELECT * FROM files').fetchall()==previous
    saved=[]
    scan_file(db,p,entry,dict(held={'h'}),set(),audit_cursor=0,audit_save=saved.append)
    assert saved==[2]
    assert db.execute('SELECT reason FROM occurrences ORDER BY line').fetchall()==[(None,),('held_group',)]
    assert db.execute('SELECT * FROM files').fetchall()==previous


def test_group_change_targets_original_occurrence_groups():
    db=catalog()
    for file,group in [('a','imported-only'),('b','unchanged')]:
        db.execute('INSERT INTO files VALUES(?,?,?,1,1,NULL)',(file,'sha',1))
        db.execute('INSERT INTO occurrences VALUES(?,1,?,?,NULL,?)',(file,'row','program',json.dumps([group])))
    before=dict(held=set(),quality=set(),pending={'unchanged'})
    after=dict(held={'imported-only'},quality=set(),pending={'unchanged'})
    assert changed_files(db,before,after)=={'a'}
    assert changed_files(db,after,before)=={'a'}
    assert changed_files(db,before,before)==set()
    assert changed_files(db,before,before,general=True)=={'a','b'}


def test_audit_only_rewrites_committed_prefix(tmp_path):
    db=catalog(); p=policy(); entry=write(tmp_path/'a.jsonl',[row(),row('h')])
    scan_file(db,p,entry,dict(held=set()),set())
    db.execute('DELETE FROM occurrences WHERE line=2')
    db.execute('UPDATE files SET cursor=1,complete=0');db.commit()
    saved=[]
    scan_file(db,p,entry,dict(held={'g'}),set(),audit_cursor=0,audit_save=saved.append)
    assert saved==[1]
    assert db.execute('SELECT count(*) FROM occurrences').fetchone()==(1,)
    assert db.execute('SELECT cursor,complete FROM files').fetchone()==(1,0)
    scan_file(db,p,entry,dict(held={'g'}),set())
    assert db.execute('SELECT cursor,complete FROM files').fetchone()==(2,1)
    assert db.execute('SELECT reason FROM occurrences ORDER BY line').fetchall()==[('held_group',),(None,)]


@pytest.fixture
def live_catalog(tmp_path, monkeypatch):
    root=tmp_path/'catalog';root.mkdir()
    db=catalog();target=sqlite3.connect(root/'catalog.sqlite');db.backup(target);target.close();db.close()
    registry=tmp_path/'registry.json';registry.write_text(json.dumps(dict(corpora=[dict(
        id='c',training_admission=True,path='.',include=['turns/*.jsonl'])])))
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(dict(source_files={})))
    frozen=tmp_path/'policy.py';frozen.write_text('# exact historical extraction policy\n')
    historical=tmp_path/'old-protection.json';historical.write_text(json.dumps(dict(held=['g'],quality=[],pending=[])))
    current=tmp_path/'protection.json';current.write_bytes(historical.read_bytes())
    entries=[]
    p=policy();p.REPO=tmp_path;p.PROTECTION=current
    def protection(path, **kwargs):
        return {k:set(v) for k,v in json.loads(Path(path).read_text()).items()}
    def inputs():
        groups=protection(p.PROTECTION)
        return groups,set(),dict(registry=binding(registry),protection=binding(p.PROTECTION),
            mechanism=binding(frozen),decision_holds=[],source_reviews_sha256='unchanged',
            initial_selection=binding(selection))
    p.inputs=inputs;p.current_protection=protection
    p.discover=lambda:(list(entries),[])
    p.floor=lambda _:None
    parent=inputs()[2];parent['protection']=binding(historical)
    parent_path=root/'parent.json';parent_path.write_text(json.dumps(parent))
    inventory=root/'parent-population.json';inventory.write_text(json.dumps(dict(files=[],input_sha256=digest(parent))))
    monkeypatch.setattr(mechanism,'load_policy',lambda _:p)
    monkeypatch.setattr(mechanism,'_STATIC',())
    def add(name='one'):
        path=tmp_path/'turns'/(name+'.jsonl');path.parent.mkdir(exist_ok=True)
        entry=write(path,[row()]);entry.update(corpora=['c'],priority=1)
        entries.append(entry);return path
    def run(): return mechanism.resume(root,parent_path,inventory,current)
    def reason():
        db=sqlite3.connect(root/'catalog.sqlite')
        try:return db.execute('SELECT reason FROM occurrences ORDER BY file,line').fetchall()
        finally:db.close()
    return SimpleNamespace(root=root,current=current,registry=registry,run=run,add=add,reason=reason)


@pytest.mark.parametrize('withdraw', [False, True])
def test_interrupted_policy_revert_reaudits_debt_even_after_withdrawal(live_catalog, monkeypatch, withdraw):
    c=live_catalog;c.add();c.run();assert c.reason()==[('held_group',)]
    c.current.write_text(json.dumps(dict(held=[],quality=[],pending=[])))
    original=mechanism.store_row
    def interrupted(db,*args):
        original(db,*args);db.commit();raise RuntimeError('interrupted after committed row')
    monkeypatch.setattr(mechanism,'store_row',interrupted)
    with pytest.raises(RuntimeError,match='committed row'):c.run()
    assert c.reason()==[(None,)]
    monkeypatch.setattr(mechanism,'store_row',original)
    c.current.write_text(json.dumps(dict(held=['g'],quality=[],pending=[])))
    if withdraw:
        registry=json.loads(c.registry.read_text());registry['corpora'][0]['training_admission']=False
        c.registry.write_text(json.dumps(registry))
        c.run()
        assert (c.root/'adaptive-policy-inflight.json').exists()
        registry['corpora'][0]['training_admission']=True;c.registry.write_text(json.dumps(registry))
    c.run();assert c.reason()==[('held_group',)]
    db=sqlite3.connect(c.root/'catalog.sqlite')
    assert db.execute('SELECT cursor,complete FROM files').fetchone()==(1,1);db.close()


def test_interrupted_new_source_keeps_policy_debt(live_catalog, monkeypatch):
    c=live_catalog;c.run()
    c.current.write_text(json.dumps(dict(held=[],quality=[],pending=[])))
    c.add()
    original=mechanism.write_json_durable
    def interrupted(path,value):
        if path.name=='adaptive-catalog-state.json':raise RuntimeError('state publication interrupted')
        return original(path,value)
    monkeypatch.setattr(mechanism,'write_json_durable',interrupted)
    with pytest.raises(RuntimeError,match='state publication'):c.run()
    assert c.reason()==[(None,)]
    monkeypatch.setattr(mechanism,'write_json_durable',original)
    c.current.write_text(json.dumps(dict(held=['g'],quality=[],pending=[])))
    c.run();assert c.reason()==[('held_group',)]


def test_publication_checks_source_bytes_and_excludes_withdrawn_history(live_catalog):
    c=live_catalog;path=c.add();c.current.write_text(json.dumps(dict(held=[],quality=[],pending=[])))
    c.run();reference=binding(c.root/'adaptive-catalog-state.json')
    proof=mechanism.export_candidates(c.root,reference)
    assert json.loads(Path(proof['path']).read_text())['cases']==1
    path.write_text('changed source')
    with pytest.raises(ValueError,match='source bytes changed'):
        mechanism.export_candidates(c.root,reference)


def test_new_glob_member_does_not_require_registry_mutation(live_catalog):
    c=live_catalog;c.add('one');first=c.run();c.add('two');second=c.run()
    assert first['authority']!=second['authority']
    db=sqlite3.connect(c.root/'catalog.sqlite')
    assert db.execute('SELECT count(*),sum(cursor) FROM files').fetchone()==(2,2);db.close()


@pytest.mark.parametrize('cached', [False, True])
def test_publication_rejects_partial_intervening_policy_even_if_external_authority_reverted(live_catalog,monkeypatch,cached):
    c=live_catalog;c.add();c.run();reference=binding(c.root/'adaptive-catalog-state.json')
    if cached:mechanism.export_candidates(c.root,reference)
    c.current.write_text(json.dumps(dict(held=[],quality=[],pending=[])))
    original=mechanism.store_row
    def interrupted(db,*args):original(db,*args);db.commit();raise RuntimeError('interrupt')
    monkeypatch.setattr(mechanism,'store_row',interrupted)
    with pytest.raises(RuntimeError):c.run()
    c.current.write_text(json.dumps(dict(held=['g'],quality=[],pending=[])))
    with pytest.raises(ValueError,match='incomplete policy audit'):
        mechanism.export_candidates(c.root,reference)


def test_initial_selection_does_not_override_current_include_removal(tmp_path):
    path=tmp_path/'turns'/'one.jsonl';path.parent.mkdir();write(path,[row()])
    registry=dict(corpora=[dict(id='c',path='.',include=['turns/*.jsonl'],training_admission=True)])
    initial=dict(source_files={str(path):dict(corpus_id='c',sha256=binding(path)['sha256'])})
    files,_=mechanism.discover_sources(tmp_path,registry,initial)
    assert files[0]['priority']==0
    registry['corpora'][0]['include']=['turns/other.jsonl']
    assert mechanism.discover_sources(tmp_path,registry,initial)[0]==[]


def test_policy_audit_reuses_exact_original_program_storage(tmp_path,monkeypatch):
    db=catalog();entry=write(tmp_path/'a.jsonl',[row()]);p=policy()
    scan_file(db,p,entry,dict(held=set()),set())
    before=db.execute('SELECT sha,body FROM programs').fetchall()
    monkeypatch.setattr(mechanism,'encode_body',lambda _:pytest.fail('audit recompressed an unchanged original program'))
    scan_file(db,p,entry,dict(held={'g'}),set(),audit_cursor=0,audit_save=lambda _:None)
    assert db.execute('SELECT sha,body FROM programs').fetchall()==before
    assert db.execute('SELECT reason FROM occurrences').fetchone()==('held_group',)


def test_cached_publication_rechecks_authority_after_source_verification(live_catalog,monkeypatch):
    c=live_catalog;source=c.add();c.current.write_text(json.dumps(dict(held=[],quality=[],pending=[])))
    c.run();reference=binding(c.root/'adaptive-catalog-state.json')
    mechanism.export_candidates(c.root,reference)
    original=mechanism.binding
    def changed(path):
        result=original(path)
        if Path(path)==source:c.current.write_text(json.dumps(dict(held=['g'],quality=[],pending=[])))
        return result
    monkeypatch.setattr(mechanism,'binding',changed)
    with pytest.raises(ValueError,match='cached candidate authority changed'):
        mechanism.export_candidates(c.root,reference)


def test_publication_binds_shared_selection_rule(live_catalog,monkeypatch):
    c=live_catalog;c.add();c.run();reference=binding(c.root/'adaptive-catalog-state.json')
    monkeypatch.setattr(mechanism,'eligible_program_sql',lambda:'1')
    with pytest.raises(ValueError,match='publication authority changed'):
        mechanism.export_candidates(c.root,reference)


def test_repeated_program_is_encoded_once_and_write_reserve_uses_catalog_disk(tmp_path,monkeypatch):
    db=catalog();entry=write(tmp_path/'source.jsonl',[row() for _ in range(1001)]);p=policy()
    encoded=[];original=mechanism.encode_body
    monkeypatch.setattr(mechanism,'encode_body',lambda body:(encoded.append(body),original(body))[1])
    floors=[];p.floor=floors.append
    destination=tmp_path/'catalog'
    scan_file(db,p,entry,dict(held=set()),set(),storage_root=destination)
    assert len(encoded)==1
    assert floors==[destination]
    assert db.execute('SELECT count(*) FROM occurrences').fetchone()==(1001,)
