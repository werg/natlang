import hashlib,json,sqlite3
from pathlib import Path
import pytest
from types import SimpleNamespace
from scripts.coreweave import teacher_reference_append
from scripts.coreweave.teacher_generation_guardian import binding,write
from scripts.coreweave.teacher_reference_append import prepare,qualify


@pytest.fixture(autouse=True)
def reference_storage(monkeypatch):
    # These small protocol fixtures must not depend on unrelated production
    # writes crossing the machine's reserve during the suite. Keep the real
    # guard, but supply its host-capacity dependency locally to this module.
    storage = SimpleNamespace(disk_usage=lambda path: SimpleNamespace(free=64 * 1024**3))
    monkeypatch.setattr(teacher_reference_append, 'shutil', storage)
    return storage


@pytest.mark.parametrize('location', ['output', 'checkout'])
def test_reference_reserve_rejects_either_full_filesystem(tmp_path, reference_storage, location):
    checkout = Path(teacher_reference_append.__file__).resolve()
    low = tmp_path if location == 'output' else checkout
    reference_storage.disk_usage = lambda path: SimpleNamespace(
        free=20 * 1024**3 - 1 if path == low else 64 * 1024**3)
    with pytest.raises(RuntimeError, match='reserve below 20 GiB'):
        teacher_reference_append.floor(tmp_path)


def test_reference_reserve_accepts_its_exact_boundary(tmp_path, reference_storage):
    reference_storage.disk_usage = lambda path: SimpleNamespace(free=20 * 1024**3)
    teacher_reference_append.floor(tmp_path)


def corpus(tmp_path):
    root=tmp_path/'catalog';root.mkdir();source=root/'original.jsonl'
    rows=[{'id':f'p{i}','curriculum':{'family':'one'if i%2 else'two'}}for i in range(6)]
    raw=[json.dumps(r,sort_keys=True,separators=(',',':')).encode()for r in rows]
    sha=[hashlib.sha256(r).hexdigest()for r in raw];source.write_bytes(b'\n'.join(raw)+b'\n')
    initial=root/'initial.json';write(initial,{'cases':6,'cases_binding':binding(source),'catalog_summary':{},'inputs':{}})
    first=root/'first.jsonl';first.write_bytes(b'\n'.join(raw[:2])+b'\n')
    proof=root/'prefix-proof.json';write(proof,{'cases':2,'cases_binding':binding(first)})
    prefix=root/'prefix.json';write(prefix,{'packets':[binding(proof)],'cases':2})
    file=root/'native-turns.jsonl';file.write_text('original source')
    write(root/'source-inventory.json',{'files':[{'path':str(file),'corpora':['original']}]})
    db=sqlite3.connect(root/'catalog.sqlite');db.executescript('''CREATE TABLE programs(sha TEXT PRIMARY KEY,id TEXT,body TEXT,family TEXT);
      CREATE TABLE occurrences(file TEXT,line INTEGER,row_sha TEXT,program_sha TEXT,reason TEXT,groups_json TEXT);
      CREATE TABLE files(path TEXT PRIMARY KEY,sha TEXT,bytes INTEGER,complete INTEGER);''')
    f=binding(file);db.execute('INSERT INTO files VALUES(?,?,?,1)',(str(file),f['sha256'],f['bytes']))
    for i,r in enumerate(rows):
        db.execute('INSERT INTO programs VALUES(?,?,?,?)',(sha[i],r['id'],raw[i].decode(),r['curriculum']['family']))
        db.execute('INSERT INTO occurrences VALUES(?,?,?,?,?,?)',(str(file),i+1,'row'+str(i),sha[i],'held_group'if i==5 else None,'["source"]'))
    db.execute('INSERT INTO programs VALUES(?,?,?,?)',('conflicting-sha','p4','{"id":"p4","different":true}','two'));db.commit();db.close()
    return root,binding(initial),binding(prefix),sha


def test_prepare_exhausts_difference_preserves_prefix_and_pending_provenance(tmp_path):
    root,source,prefix,sha=corpus(tmp_path);out=tmp_path/'out'
    result=prepare(root,source,prefix,out,1);inventory=json.loads(Path(result['path']).read_text())
    assert inventory['packets'][0]==json.loads(Path(prefix['path']).read_text())['packets'][0]
    assert inventory['appended_cases']==2 and len(inventory['packets'])==3
    unready=json.loads(Path(inventory['unready']['path']).read_text());assert set(unready['program_sha256'])=={sha[4],sha[5]}
    seen=set()
    for ref in inventory['packets'][1:]:
        proof=json.loads(Path(ref['path']).read_text());raw=Path(proof['cases_binding']['path']).read_bytes().strip()
        seen.add(hashlib.sha256(raw).hexdigest());origin=json.loads(Path(proof['source_provenance']['path']).read_text())
        assert origin['occurrences'][0]['source_row_ordinal']in(3,4)
    assert seen=={sha[2],sha[3]}
    assert prepare(root,source,prefix,out,1)==result
    with pytest.raises(ValueError,match='immutable'):prepare(root,source,prefix,out,2)


def test_compressed_catalog_exports_identical_original_packets(tmp_path):
    from scripts.coreweave.teacher_catalog_storage import encode_body
    root,source,prefix,sha=corpus(tmp_path)
    db=sqlite3.connect(root/'catalog.sqlite')
    for key,body in db.execute('SELECT sha,body FROM programs').fetchall():
        db.execute('UPDATE programs SET body=? WHERE sha=?',(encode_body(body),key))
    db.commit();db.close()
    result=prepare(root,source,prefix,tmp_path/'out',1)
    inventory=json.loads(Path(result['path']).read_text())
    seen=set()
    for ref in inventory['packets'][1:]:
        proof=json.loads(Path(ref['path']).read_text())
        seen.add(hashlib.sha256(Path(proof['cases_binding']['path']).read_bytes().strip()).hexdigest())
    assert seen=={sha[2],sha[3]}


def test_prepare_rejects_changed_source_before_catalog_access(tmp_path):
    root,source,prefix,_=corpus(tmp_path);Path(json.loads(Path(source['path']).read_text())['cases_binding']['path']).write_text('changed')
    with pytest.raises(ValueError,match='input changed'):prepare(root,source,prefix,tmp_path/'out',2)


def test_cached_actual_protocol_qualification_extends_prefix_without_rewriting_it(tmp_path):
    from test_teacher_source_sequence import setup
    control,state,config=setup(tmp_path)
    declaration=config['source_sequence'];manifest=json.loads(Path(declaration['ready_manifest']).read_text())
    entries=manifest['entries'];inventory=manifest['candidate_inventory']
    cached=json.loads(Path(entries[1]['qualified_packet']['path']).read_text())
    cached['counts']={'qualified':1,'candidates':1}
    write(Path(entries[1]['candidate_packet']['path']).parent/'qualification/prepared.json',cached)
    old=tmp_path/'old-ready.json';prefix={'entries':[entries[0]]};write(old,prefix)
    qualifier=tmp_path/'qualifier.py';qualifier.write_text('raise AssertionError("cached proofs should not execute")')
    out=tmp_path/'qualification-control';out.mkdir()
    ready=qualify(inventory,binding(old),binding(qualifier),out);value=json.loads(Path(ready['path']).read_text())
    assert value['entries'][0]==entries[0]
    assert value['entries'][1]['candidate_packet']==entries[1]['candidate_packet']
    assert value['complete']and value['entries'][1]['qualified_cases']==1
    assert json.loads(old.read_text())==prefix


def test_no_new_work_retains_complete_original_readiness(tmp_path):
    from test_teacher_source_sequence import setup
    control,state,config=setup(tmp_path);decl=config['source_sequence'];manifest=json.loads(Path(decl['ready_manifest']).read_text())
    dummy=tmp_path/'never.py';dummy.write_text('raise AssertionError("no new work")')
    out=tmp_path/'out';out.mkdir()
    ref=qualify(manifest['candidate_inventory'],binding(Path(decl['ready_manifest'])),binding(dummy),out)
    value=json.loads(Path(ref['path']).read_text());assert value['entries']==manifest['entries']and value['complete']


def test_uncommitted_derived_outputs_rebuild_from_original_inputs_and_verdicts(tmp_path):
    from test_teacher_source_sequence import setup
    control,state,config=setup(tmp_path);decl=config['source_sequence'];manifest=json.loads(Path(decl['ready_manifest']).read_text());entries=manifest['entries']
    old=tmp_path/'prefix-ready.json';write(old,{'entries':[entries[0]]})
    cached=json.loads(Path(entries[1]['qualified_packet']['path']).read_text());cached['counts']={'qualified':1,'candidates':1}
    folder=Path(entries[1]['candidate_packet']['path']).parent/'qualification';folder.mkdir()
    (folder/'qualified-cases.jsonl').write_text('partial output')
    (folder/'reference-verdicts.jsonl').write_text('durable original verdicts')
    (folder/'inputs.json').write_text('durable original inputs')
    qualifier=tmp_path/'qualifier.py'
    qualifier.write_text('import sys,json\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index("--out")+1]);(p/"prepared.json").write_text('+repr(json.dumps(cached))+')\n')
    out=tmp_path/'out';out.mkdir()
    ref=qualify(manifest['candidate_inventory'],binding(old),binding(qualifier),out)
    assert json.loads(Path(ref['path']).read_text())['complete']
    assert not(folder/'qualified-cases.jsonl').exists()
    assert(folder/'reference-verdicts.jsonl').read_text()=='durable original verdicts'
    assert(folder/'inputs.json').read_text()=='durable original inputs'
    receipts=list(folder.glob('incomplete-recovery-*.json'));assert len(receipts)==1
    assert json.loads(receipts[0].read_text())['uncommitted_outputs'][0]['sha256']==hashlib.sha256(b'partial output').hexdigest()


@pytest.mark.parametrize('independent_packets',[False,True])
def test_zero_qualified_packet_is_authenticated_and_kept_pending(tmp_path,independent_packets):
    from test_teacher_source_sequence import setup
    control,state,config=setup(tmp_path,packet_count=3 if independent_packets else 2);decl=config['source_sequence'];manifest=json.loads(Path(decl['ready_manifest']).read_text());entries=manifest['entries']
    old=tmp_path/'old.json';write(old,{'entries':[entries[0]]})
    item=json.loads(Path(entries[1]['qualified_packet']['path']).read_text());q=json.loads(Path(item['qualification']['path']).read_text())
    verdict=Path(q['verdicts']['path']);row=json.loads(verdict.read_text());row['passed']=False;verdict.write_text(json.dumps(row)+'\n')
    empty=tmp_path/'empty.jsonl';empty.write_bytes(b'');q.update(qualified_cases=0,pending_cases=1,qualified_cases_binding=binding(empty),verdicts=binding(verdict))
    qp=tmp_path/'zero-qualification.json';write(qp,q)
    selection=json.loads(Path(item['references']['selection']['path']).read_text());selection.update(cases=0,cases_sha256=binding(empty)['sha256'],reference_qualification=binding(qp))
    sp=tmp_path/'zero-selection.json';write(sp,selection)
    join=json.loads(Path(item['references']['global_protection_join']['path']).read_text());join['selected_packet_sha256']=binding(empty)['sha256'];jp=tmp_path/'zero-join.json';write(jp,join)
    item.update(qualification=binding(qp),cases=binding(empty),counts={'qualified':0,'candidates':1,'pending':1},references={'selected_cases':binding(empty),'selection':binding(sp),'global_protection_join':binding(jp)})
    write(Path(entries[1]['candidate_packet']['path']).parent/'qualification/prepared.json',item)
    if independent_packets:
        third=json.loads(Path(entries[2]['qualified_packet']['path']).read_text())
        third['counts']={'qualified':1,'candidates':1}
        write(Path(entries[2]['candidate_packet']['path']).parent/'qualification/prepared.json',third)
    dummy=tmp_path/'never.py';dummy.write_text('raise AssertionError("cached zero proof")');out=tmp_path/'out';out.mkdir()
    authority=tmp_path/'authority.json';write(authority,config['source_sequence']['authority'])
    ref=qualify(manifest['candidate_inventory'],binding(old),binding(dummy),out,independent_packets=independent_packets,
      qualification_authority_reference=binding(authority)if independent_packets else None);ready=json.loads(Path(ref['path']).read_text())
    assert not ready['complete']and ready['entries'][0]==entries[0]and ready['blocked_unready_packets'][0]['pending_cases']==1
    if independent_packets:
        assert len(ready['entries'])==2
        assert ready['entries'][1]['candidate_packet']==entries[2]['candidate_packet']
        assert ready['entries'][1]['ordinal']==1 and ready['entries'][1]['candidate_ordinal']==2
        published=json.loads(Path(ready['candidate_inventory']['path']).read_text())
        assert published['packets']==[entry['candidate_packet']for entry in ready['entries']]+[entries[1]['candidate_packet']]
        assert not published['unprocessed_packets']
        assert published['qualification_source_inventory']==manifest['candidate_inventory']
    else:assert ready['entries']==[entries[0]]


def test_eligibility_and_packet_ranking_do_not_scan_full_program_blob_table(tmp_path,monkeypatch):
    root,source,prefix,_=corpus(tmp_path);original=sqlite3.connect;plans=[]
    class Observed(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            if sql.startswith('SELECT p.sha FROM selected')or sql.startswith('WITH source AS'):
                plan=list(super().execute('EXPLAIN QUERY PLAN '+sql,*args,**kwargs));plans.append(plan)
                assert not any(row[-1]=='SCAN p'for row in plan)
            return super().execute(sql,*args,**kwargs)
    monkeypatch.setattr(sqlite3,'connect',lambda *a,**kw:original(*a,factory=Observed,**kw))
    ref=prepare(root,source,prefix,tmp_path/'out',2)
    assert len(plans)==2 and json.loads(Path(ref['path']).read_text())['appended_cases']==2


def test_compressed_candidate_inventory_preserves_original_program_hashes(tmp_path):
    from natlang_neuralese.common.jsonio import write_jsonl_text
    root,source,prefix,sha=corpus(tmp_path)
    original=json.loads(Path(source['path']).read_text())['cases_binding']
    compressed=root/'candidates.jsonl.zst'
    with write_jsonl_text(compressed) as stream:stream.write(Path(original['path']).read_text())
    assert list(teacher_reference_append.candidate_hashes(binding(compressed)))==sha
    compressed.write_bytes(b'changed')
    with pytest.raises(ValueError,match='input changed'):
        list(teacher_reference_append.candidate_hashes({**original,'path':str(compressed)}))


def test_adaptive_reference_provenance_excludes_withdrawn_positive_sources(tmp_path):
    root,source,prefix,sha=corpus(tmp_path)
    metadata=json.loads((root/'source-inventory.json').read_text())
    authority=root/'authority.json';write(authority,dict(current={}))
    population=root/'adaptive-population.json';write(population,{**metadata,'authority':binding(authority)})
    summary=root/'summary.json';write(summary,dict(source_population=binding(population)))
    body=json.loads(Path(source['path']).read_text());body.update(source_population=binding(population),catalog_summary=binding(summary))
    current=root/'adaptive-source.json';write(current,body)
    old=root/'aaa-withdrawn.jsonl';old.write_text('historical')
    db=sqlite3.connect(root/'catalog.sqlite');oldref=binding(old)
    db.execute('INSERT INTO files VALUES(?,?,?,1)',(str(old),oldref['sha256'],oldref['bytes']))
    db.execute('INSERT INTO occurrences VALUES(?,1,?,?,NULL,?)',(str(old),'old-row',sha[2],'["old-group"]'))
    db.commit();db.close()
    result=prepare(root,binding(current),prefix,tmp_path/'out',1)
    inventory=json.loads(Path(result['path']).read_text());assert inventory['appended_cases']==2
    for packet in inventory['packets'][1:]:
        proof=json.loads(Path(packet['path']).read_text())
        origins=json.loads(Path(proof['source_provenance']['path']).read_text())['occurrences']
        assert origins and all(r['source_file']['path']!=str(old) for r in origins)


def test_withdrawing_source_never_unprotects_its_held_content(tmp_path):
    root,source,prefix,sha=corpus(tmp_path)
    old=root/'withdrawn-held.jsonl';old.write_text('held source')
    db=sqlite3.connect(root/'catalog.sqlite');ref=binding(old)
    db.execute('INSERT INTO files VALUES(?,?,?,1)',(str(old),ref['sha256'],ref['bytes']))
    db.execute('INSERT INTO occurrences VALUES(?,1,?,?,?,?)',(str(old),'held-row',sha[2],'held_group','["held-group"]'))
    db.commit();db.close()
    result=prepare(root,source,prefix,tmp_path/'out',1)
    inventory=json.loads(Path(result['path']).read_text());assert inventory['appended_cases']==1
    unready=json.loads(Path(inventory['unready']['path']).read_text())
    assert sha[2] in unready['program_sha256']


def test_adaptive_append_retains_authored_prefix_and_quarantines_old_variants(tmp_path):
    root,source,original,sha=corpus(tmp_path)
    authored=root/'authored.jsonl';authored.write_text('{"id":"independent-authoring"}\n')
    proof=root/'authored-proof.json';write(proof,{'cases':1,'cases_binding':binding(authored)})
    prefix=root/'ready-prefix.json';write(prefix,{'packets':[binding(proof)],'cases':1})
    original_source=json.loads(Path(source['path']).read_text())
    rows=Path(original_source['cases_binding']['path']).read_bytes().splitlines()
    blocked=root/'blocked.jsonl';blocked.write_bytes(rows[2]+b'\n')
    blocked_proof=root/'blocked-proof.json';write(blocked_proof,{'cases':1,'cases_binding':binding(blocked)})
    excluded=root/'excluded.json';write(excluded,{'packets':[binding(blocked_proof)],'cases':1})
    # A historical raw prefix no longer present in the current catalog remains
    # dedup debt rather than forcing a restart of authored/qualified work.
    current=root/'current.jsonl';current.write_bytes(b'\n'.join(rows[1:])+b'\n')
    updated=root/'current-proof.json';write(updated,{**original_source,'cases':5,'cases_binding':binding(current)})
    out=tmp_path/'adaptive-out'
    result=prepare(root,binding(updated),binding(prefix),out,1,
      dedup_inventory_reference=original,exclusion_inventory_reference=binding(excluded))
    inventory=json.loads(Path(result['path']).read_text())
    assert inventory['packets'][0]==binding(proof)
    assert inventory['appended_cases']==1 and inventory['cases']==2
    packet=json.loads(Path(inventory['packets'][1]['path']).read_text())
    assert list(teacher_reference_append.candidate_hashes(packet['cases_binding']))==[sha[3]]
    accounting=json.loads((out/'adaptive-exclusion-accounting.json').read_text())
    assert sha[0] in accounting['absent_excluded_variants']
    assert not accounting['training_admission']
    # Cached preparation must still authenticate the exact exclusion bodies.
    blocked.write_text('tampered quarantine')
    with pytest.raises(ValueError,match='input changed'):
        prepare(root,binding(updated),binding(prefix),out,1,
          dedup_inventory_reference=original,exclusion_inventory_reference=binding(excluded))


def test_adaptive_exclusions_cannot_replace_original_dedup_authority(tmp_path):
    root,source,prefix,_=corpus(tmp_path)
    with pytest.raises(ValueError,match='original dedup inventory'):
        prepare(root,source,prefix,tmp_path/'out',exclusion_inventory_reference=prefix)


def test_adaptive_append_rejects_duplicate_source_even_when_already_excluded(tmp_path):
    root,source,prefix,_=corpus(tmp_path)
    body=json.loads(Path(source['path']).read_text());cases=Path(body['cases_binding']['path'])
    cases.write_bytes(cases.read_bytes()+cases.read_bytes().splitlines()[0]+b'\n')
    updated=root/'duplicates.json';write(updated,{**body,'cases':7,'cases_binding':binding(cases)})
    with pytest.raises(ValueError,match='repeated a program variant'):
        prepare(root,binding(updated),prefix,tmp_path/'out',dedup_inventory_reference=prefix)
