import hashlib,json,sqlite3
from pathlib import Path
import pytest
from scripts.coreweave.teacher_generation_guardian import binding,write
from scripts.coreweave.teacher_reference_append import prepare,qualify


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


def test_zero_qualified_packet_is_authenticated_and_kept_pending(tmp_path):
    from test_teacher_source_sequence import setup
    control,state,config=setup(tmp_path);decl=config['source_sequence'];manifest=json.loads(Path(decl['ready_manifest']).read_text());entries=manifest['entries']
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
    dummy=tmp_path/'never.py';dummy.write_text('raise AssertionError("cached zero proof")');out=tmp_path/'out';out.mkdir()
    ref=qualify(manifest['candidate_inventory'],binding(old),binding(dummy),out);ready=json.loads(Path(ref['path']).read_text())
    assert not ready['complete']and ready['entries']==[entries[0]]and ready['blocked_unready_packets'][0]['pending_cases']==1


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
