"""Exhaust declared remaining candidate programs through an existing qualifier."""
from __future__ import annotations
import argparse,collections,fcntl,hashlib,json,os,shutil,sqlite3,subprocess,sys
from pathlib import Path
try:
    from .teacher_generation_guardian import binding,write
    from .teacher_source_sequence import _packet
except ImportError:
    from teacher_generation_guardian import binding,write
    from teacher_source_sequence import _packet


def checked(reference):
    path=Path(reference['path'])
    if binding(path)!=reference:raise ValueError('reference work input changed: '+str(path))
    return path


def read(reference):
    raw=Path(reference['path']).read_bytes()
    if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('reference work input changed: '+reference['path'])
    return json.loads(raw)


def immutable(path,value):
    raw=value if isinstance(value,bytes)else(json.dumps(value,indent=2)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('immutable reference work differs: '+str(path))
    else:write(path,raw)
    return binding(path)


def floor(path):
    if any(shutil.disk_usage(root).free<20*1024**3 for root in (path,Path(__file__).resolve())):
        raise RuntimeError('reference work disk reserve below 20 GiB; resume the same work')


def candidate_hashes(reference):
    path=checked(reference)
    total=0;digest=hashlib.sha256()
    with path.open('rb')as stream:
        for raw in stream:
            total+=len(raw);digest.update(raw)
            if raw.strip():yield hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()
    if total!=reference['bytes']or digest.hexdigest()!=reference['sha256']:
        raise ValueError('candidate bytes changed during streaming')


def prepare(catalog,source_reference,prefix_reference,out,packet_rows=64):
    source=read(source_reference);prefix=read(prefix_reference)
    if packet_rows<1:raise ValueError('positive finite packet size required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);floor(out)
    contract={'source':source_reference,'prefix':prefix_reference,'packet_rows':packet_rows}
    immutable(out/'inputs.json',contract)
    target=out/'packet-inventory.json'
    if target.exists():
        result=json.loads(target.read_text())
        for ref in result['packets']:read(ref)
        return binding(target)
    prior=set()
    for reference in prefix['packets']:
        proof=read(reference)
        for key in candidate_hashes(proof['cases_binding']):
            if key in prior:raise ValueError('candidate prefix repeated a program variant')
            prior.add(key)
    db=sqlite3.connect('file:'+str(Path(catalog)/'catalog.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('CREATE TEMP TABLE selected(sha TEXT PRIMARY KEY)')
    selected=set();excluded=0
    for key in candidate_hashes(source['cases_binding']):
        if key in prior:excluded+=1;continue
        if key in selected:raise ValueError('source repeated a program variant')
        selected.add(key);db.execute('INSERT INTO selected VALUES(?)',(key,))
    if len(selected)+excluded!=source['cases']:raise ValueError('source candidate count differs')
    if excluded!=len(prior):raise ValueError('candidate prefix is not an exact subset of source candidates')
    db.commit();db.execute('BEGIN')
    population=Path(catalog)/'source-inventory-v3.json'
    if not population.exists():population=Path(catalog)/'source-inventory.json'
    metadata={f['path']:f for f in json.loads(population.read_text())['files']}
    eligible='''p.id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM programs q WHERE q.id=p.id AND q.sha!=p.sha)
      AND EXISTS(SELECT 1 FROM occurrences o JOIN files f ON f.path=o.file WHERE o.program_sha=p.sha AND o.reason IS NULL AND f.complete=1)
      AND NOT EXISTS(SELECT 1 FROM occurrences o WHERE o.program_sha=p.sha AND o.reason IN('held_group','quality_group','pending_source_review','nontrain','exact_decision_program_hold'))'''
    missing=db.execute('SELECT s.sha FROM selected s LEFT JOIN programs p ON p.sha=s.sha WHERE p.sha IS NULL').fetchall()
    if missing:raise ValueError('original candidate is absent from its source catalog')
    unready=[row[0]for row in db.execute('SELECT p.sha FROM selected s CROSS JOIN programs p ON p.sha=s.sha WHERE NOT('+eligible+') ORDER BY p.rowid')]
    immutable(out/'unready-programs.json',{'program_sha256':unready,'reason':'current catalog conflict or protected/incomplete occurrence; not omitted','training_admission':False})
    query='''WITH source AS(SELECT p.sha,p.family,p.rowid ordinal,
      (SELECT min(o.file) FROM occurrences o JOIN files f ON f.path=o.file WHERE o.program_sha=p.sha AND o.reason IS NULL AND f.complete=1) file
      FROM selected s CROSS JOIN programs p ON p.sha=s.sha WHERE '''+eligible+'''),
      ranked AS(SELECT *,row_number()OVER(PARTITION BY family,file ORDER BY ordinal) round FROM source)
      SELECT sha,family,file FROM ranked ORDER BY round,file,sha'''
    packet=[];references=list(prefix['packets']);total=0
    def flush():
        nonlocal packet,total
        if not packet:return
        floor(out);folder=out/f'packet-{len(references):05d}';folder.mkdir(exist_ok=True)
        if any(hashlib.sha256(body.encode()).hexdigest()!=sha for sha,body,_,_ in packet):
            raise ValueError('catalog program differs from original canonical bytes')
        raw=b''.join(body.encode()+b'\n'for _,body,_,_ in packet)
        cases=immutable(folder/'cases.jsonl',raw);families=collections.Counter();sources=collections.Counter();provenance=[]
        for sha,body,family,file in packet:
            if hashlib.sha256(body.encode()).hexdigest()!=sha:raise ValueError('catalog program differs from original canonical bytes')
            families[family]+=1;sources[file]+=1
            for path,line,rowsha,groups,file_sha,file_bytes in db.execute('''SELECT o.file,o.line,o.row_sha,o.groups_json,f.sha,f.bytes
              FROM occurrences o JOIN files f ON f.path=o.file WHERE o.program_sha=? AND f.complete=1 ORDER BY o.file,o.line''',(sha,)):
                provenance.append({'program_sha256':sha,'source_file':{'path':path,'sha256':file_sha,'bytes':file_bytes},
                  'source_row_ordinal':line,'row_canonical_sha256':rowsha,'source_groups':json.loads(groups),'corpus_ids':metadata[path]['corpora']})
        origin=immutable(folder/'source-provenance.json',{'occurrences':provenance,'catalog_candidates_proof':source_reference})
        proof={'cases':len(packet),'cases_binding':cases,'catalog_summary':source['catalog_summary'],'inputs':source['inputs'],
          'source_provenance':origin,'family_counts':dict(families),'source_file_counts':dict(sources),
          'scope':'unchanged remaining candidate programs; reference qualification pending; no total/family/corpus cap',
          'training_admission':False,'generator_calls':0}
        references.append(immutable(folder/'cases-proof.json',proof));total+=len(packet);packet=[]
        write(out/'progress.json',{'appended_packets':len(references)-len(prefix['packets']),'appended_cases':total,'complete':False})
    for sha,family,file in db.execute(query):
        body=db.execute('SELECT body FROM programs WHERE sha=?',(sha,)).fetchone()[0];packet.append((sha,body,family,file))
        if len(packet)==packet_rows:flush()
    flush();db.close()
    if total+len(unready)!=len(selected):raise ValueError('reference work silently omitted a remaining candidate')
    return immutable(target,{'packets':references,'cases':prefix['cases']+total,'appended_cases':total,'prefix':prefix_reference,
      'source':source_reference,'unready':binding(out/'unready-programs.json'),'complete':True,
      'training_admission':False,'generator_calls':0,'packet_size_is_selection_cap':False})


def qualify(inventory_reference,prefix_ready_reference,qualifier_reference,out):
    inventory=read(inventory_reference);prefix=read(prefix_ready_reference);out=Path(out)
    entries=list(prefix['entries'])
    if [e['candidate_packet']for e in entries]!=inventory['packets'][:len(entries)]:raise ValueError('readiness prefix differs from candidate order')
    first=read(entries[0]['qualified_packet']);qualification=read(first['qualification']);inputs=read(qualification['inputs'])
    authority={'protection':inputs['authority']['protection'],'decision_hold_bindings':inputs['authority']['decision_holds'],
      'reference_runtime_sha256':qualification['runtime_inventory_sha256']}
    pending=[];later=[];results=[];blocked=False
    write(out/'ready-packet-sequence.json',{'schema':'natlang.strong-teacher-ready-packet-sequence/1',
      'candidate_inventory':inventory_reference,'entries':entries,'complete':False,
      'training_admission':False,'teacher_generations':0,'production_adopted':False})
    for index,reference in enumerate(inventory['packets'][len(entries):],len(entries)):
        floor(out);proof=read(reference);cases=checked(proof['cases_binding']);folder=Path(reference['path']).parent/'qualification';prepared=folder/'prepared.json'
        if not prepared.exists():
            partial=[folder/name for name in ('qualified-cases.jsonl','pending-references.jsonl','reference-qualification.json','selection-proof.json','global-protection-join.json')if(folder/name).exists()]
            if partial:
                refs=[binding(path)for path in partial]
                digest=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest()
                immutable(folder/('incomplete-recovery-'+digest+'.json'),{'uncommitted_outputs':refs,'reason':'No prepared envelope; regenerate exact derived outputs from retained original inputs and fsynced verdicts'})
                for path,ref in zip(partial,refs):checked(ref);path.unlink()
                fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
                try:os.fsync(fd)
                finally:os.close(fd)
            subprocess.run([sys.executable,str(checked(qualifier_reference)),'--cases',str(cases),'--out',str(folder)],check=True)
            checked(qualifier_reference)
        item=json.loads(prepared.read_text())
        for ref in (*item['references'].values(),item['qualification'],item['cases']):checked(ref)
        entry={'ordinal':index,'candidate_packet':reference,'qualified_packet':binding(prepared),
          'qualified_cases':item['counts'].get('qualified',0),'pending_cases':item['counts'].get('pending',0),
          'started':False,'closed':False,'production_adopted':False}
        qualification=read(item['qualification'])
        if (qualification['qualified_cases']!=entry['qualified_cases']
            or qualification['candidate_cases']!=proof['cases']
            or qualification.get('pending_cases',0)!=entry['pending_cases']
            or entry['qualified_cases']+entry['pending_cases']!=proof['cases']
            or read(qualification['inputs'])['cases']!=proof['cases_binding']):
            raise ValueError('cached reference counts/source differ from actual candidates')
        _packet(entry,{'authority':authority},allow_empty=True)
        if not entry['qualified_cases']:blocked=True;pending.append(entry)
        elif blocked:later.append(entry)
        else:entries.append(entry)
        results.append(entry)
        write(out/'qualification-progress.json',{'processed_packets':results,'zero_qualified':pending,'later_qualified':later,'complete':False})
        write(out/'ready-packet-sequence.json',{'schema':'natlang.strong-teacher-ready-packet-sequence/1',
          'candidate_inventory':inventory_reference,'entries':entries,'complete':False,'blocked_unready_packets':pending,
          'later_qualified_packets':later,'training_admission':False,'teacher_generations':0,'production_adopted':False})
        print(json.dumps({'ordinal':index,'qualified':entry['qualified_cases'],'pending':entry['pending_cases']}),flush=True)
    report={'processed_packets':results,'zero_qualified':pending,'later_qualified':later,'complete':True,'training_admission':False}
    write(out/'qualification-progress.json',report)
    ready=json.loads((out/'ready-packet-sequence.json').read_text());ready['complete']=not blocked;write(out/'ready-packet-sequence.json',ready)
    return binding(out/'ready-packet-sequence.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);a=p.parse_args();spec=json.loads(a.spec.read_text())
    out=Path(spec['out']);out.mkdir(parents=True,exist_ok=True)
    with(out/'reference-work.lock').open('a')as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if spec.get('driver')and binding(Path(__file__))!=spec['driver']:raise ValueError('reference driver changed')
        immutable(out/'execution-spec.json',spec)
        immutable(out/'reference-append-mechanism.py',Path(__file__).read_bytes())
        inventory=prepare(spec['catalog'],spec['source'],spec['prefix_inventory'],out,spec.get('packet_rows',64))
        print(json.dumps({'inventory':inventory}),flush=True)
        print(json.dumps({'ready':qualify(inventory,spec['prefix_ready'],spec['qualifier'],out)}),flush=True)

if __name__=='__main__':main()
