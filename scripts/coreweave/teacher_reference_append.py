"""Exhaust declared remaining candidate programs through an existing qualifier."""
from __future__ import annotations
import argparse,collections,fcntl,hashlib,json,os,shutil,sqlite3,subprocess,sys
from pathlib import Path
try:
    from .teacher_generation_guardian import binding,write
    from .teacher_source_sequence import _packet
    from .teacher_catalog_storage import decode_body
except ImportError:
    from teacher_generation_guardian import binding,write
    from teacher_source_sequence import _packet
    from teacher_catalog_storage import decode_body


def checked(reference):
    path=Path(reference['path'])
    if binding(path)!=reference:raise ValueError('reference work input changed: '+str(path))
    return path


def read(reference):
    raw=Path(reference['path']).read_bytes()
    if len(raw)!=reference['bytes']or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('reference work input changed: '+reference['path'])
    return json.loads(raw)



def registry_append(before_reference,after_reference):
    """Verify an exact append, without granting new-source admission."""
    before,after=read(before_reference),read(after_reference)
    if not isinstance(before,dict)or not isinstance(after,dict):
        raise ValueError('source registries must be objects')
    original,current=before.get('corpora'),after.get('corpora')
    if not isinstance(original,list)or not isinstance(current,list):
        raise ValueError('source registries need complete corpus declarations')
    for rows in(original,current):
        names=[row.get('id')if isinstance(row,dict)else None for row in rows]
        if any(not isinstance(name,str)or not name for name in names)or len(set(names))!=len(names):
            raise ValueError('source registry corpus identities must be distinct')
    # JSON structure remains typed: Python equality alone conflates 1 and 1.0.
    comparison={**after,'corpora':current[:len(original)]}
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    if canonical(before)!=canonical(comparison)or len(current)<len(original):
        raise ValueError('source registry changed an existing declaration or field')
    return {'schema':'natlang.teacher-source-registry-append/1',
        'before':before_reference,'after':after_reference,'original_corpora':len(original),
        'current_corpora':len(current),'appended_corpus_ids':[row['id']for row in current[len(original):]],
        'training_admission':False,
        'scope':'exact registry append only; runtime/protection/program/source guards remain caller requirements'}



def registry_cache_cleanup(before_reference,after_reference):
    """Authenticate only retained-receipt/deleted-cache bookkeeping."""
    before,after=read(before_reference),read(after_reference)
    canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    if not isinstance(before,dict)or not isinstance(after,dict):
        raise ValueError('source registries must be objects')
    original,current=before.get('corpora'),after.get('corpora')
    if not isinstance(original,list)or not isinstance(current,list)or len(original)!=len(current):
        raise ValueError('cache cleanup cannot add or remove corpus declarations')
    names=[row.get('id')if isinstance(row,dict)else None for row in original]
    if any(not isinstance(name,str)or not name for name in names)or len(set(names))!=len(names):
        raise ValueError('source registry corpus identities must be distinct')
    changes=[]
    normalized=[]
    for old,new in zip(original,current):
        if canonical(old)==canonical(new):
            normalized.append(new);continue
        try:
            prior=old['source_binding']['decision_admission']
            actual=new['source_binding']['decision_admission']
            receipt_reference=binding(Path(actual['receipt_path']))
            if receipt_reference['sha256']!=prior['receipt_sha256']:
                raise ValueError('retained decision receipt changed')
            receipt=read(receipt_reference)
            if receipt.get('schema')!='natlang.verified-teacher-decision-materialization/1'or receipt.get('output_sha256')!=prior['output_sha256']:
                raise ValueError('retained receipt output differs')
            plan_declaration=actual['cache_cleanup_plan']
            if set(plan_declaration)!={'path','sha256','status'}:
                raise ValueError('unexpected decision cache cleanup fields')
            plan_reference=binding(Path(plan_declaration['path']))
            if plan_reference['sha256']!=plan_declaration['sha256']or plan_declaration['status']!='cache-key-removed-after-output-hash-verification':
                raise ValueError('decision cache cleanup plan changed')
            plan=read(plan_reference)
            if plan.get('schema')!='natlang.rebuildable-decision-cache-retention/1':
                raise ValueError('decision cache cleanup schema differs')
            size=actual['output_bytes']
            if type(size)is not int or size<0:
                raise ValueError('decision cache output size must be a nonnegative integer')
            matching=[item for item in plan['items']if item.get('output_path')==prior['output_path']]
            if len(matching)!=1:
                raise ValueError('decision cache cleanup needs one exact output witness')
            item=matching[0]
            if type(item.get('output_bytes'))is not int:
                raise ValueError('decision cache witness output size must be an integer')
            if any(item.get(key)!=value for key,value in {
                'output_bytes':size,'output_sha256':prior['output_sha256'],
                'receipt_sha256':prior['receipt_sha256'],'retained_receipt':actual['receipt_path'],
                'key_dir':str(Path(prior['output_path']).parent)}.items()):
                raise ValueError('decision cache cleanup output or receipt differs')
            expected={key:value for key,value in prior.items()if key!='output_path'}
            expected.update(receipt_path=actual['receipt_path'],output_storage_status='deleted_rebuildable_generated_cache',
                output_original_path=prior['output_path'],output_bytes=size,cache_cleanup_plan=plan_declaration)
            if canonical(actual)!=canonical(expected):
                raise ValueError('decision cache cleanup changed admission metadata')
            restored={**new,'source_binding':{**new['source_binding'],'decision_admission':prior}}
            if canonical(restored)!=canonical(old):
                raise ValueError('decision cache cleanup changed a source declaration')
        except (KeyError,TypeError,AttributeError) as error:
            raise ValueError('invalid decision cache cleanup declaration')from error
        normalized.append(restored)
        changes.append({'corpus_id':old['id'],'retained_receipt':receipt_reference,'cleanup_plan':plan_reference})
    if canonical({**after,'corpora':normalized})!=canonical(before):
        raise ValueError('decision cache cleanup changed registry fields')
    return {'schema':'natlang.teacher-source-registry-cache-cleanup/1','before':before_reference,'after':after_reference,
        'changes':changes,'training_admission':False,
        'scope':'authenticated generated-cache bookkeeping only; all source/runtime/protection/admission guards remain unchanged'}


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
    if path.name.endswith('.jsonl.zst'):
        from natlang_neuralese.common.jsonio import open_jsonl_text
        with open_jsonl_text(path) as stream:
            for line in stream:
                if line.strip():yield hashlib.sha256(line.rstrip('\r\n').encode('utf-8')).hexdigest()
        checked(reference)
        return
    total=0;digest=hashlib.sha256()
    with path.open('rb')as stream:
        for raw in stream:
            total+=len(raw);digest.update(raw)
            if raw.strip():yield hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()
    if total!=reference['bytes']or digest.hexdigest()!=reference['sha256']:
        raise ValueError('candidate bytes changed during streaming')


def eligible_program_sql():
    """One eligibility rule for current-population export and packet selection."""
    return '''p.id IS NOT NULL AND NOT EXISTS(SELECT 1 FROM programs q WHERE q.id=p.id AND q.sha!=p.sha)
      AND EXISTS(SELECT 1 FROM occurrences o JOIN files f ON f.path=o.file JOIN active_sources a ON a.path=o.file
          WHERE o.program_sha=p.sha AND o.reason IS NULL AND f.complete=1)
      AND NOT EXISTS(SELECT 1 FROM occurrences o WHERE o.program_sha=p.sha AND o.reason IN
          ('held_group','quality_group','pending_source_review','nontrain','exact_decision_program_hold'))'''


def _inventory_programs(reference):
    inventory=read(reference);programs=set();count=0
    for packet in inventory['packets']:
        proof=read(packet);hashes=list(candidate_hashes(proof['cases_binding']))
        if len(hashes)!=proof['cases'] or len(set(hashes))!=len(hashes):
            raise ValueError('candidate packet count or exact variant identity differs')
        programs.update(hashes);count+=len(hashes)
    if count!=inventory['cases']:
        raise ValueError('candidate inventory count differs from bound packets')
    return programs


def prepare(catalog,source_reference,prefix_reference,out,packet_rows=64,*,dedup_inventory_reference=None,exclusion_inventory_reference=None):
    source=read(source_reference);prefix=read(prefix_reference)
    if packet_rows<1:raise ValueError('positive finite packet size required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);floor(out)
    contract={'source':source_reference,'prefix':prefix_reference,'packet_rows':packet_rows,
      **({'dedup_inventory':dedup_inventory_reference}if dedup_inventory_reference else {}),
      **({'excluded_inventory':exclusion_inventory_reference}if exclusion_inventory_reference else {})}
    immutable(out/'inputs.json',contract)
    target=out/'packet-inventory.json'
    prior=set()
    for reference in prefix['packets']:
        proof=read(reference)
        for key in candidate_hashes(proof['cases_binding']):
            if key in prior:raise ValueError('candidate prefix repeated a program variant')
            prior.add(key)
    if len(prior)!=prefix['cases']:raise ValueError('candidate prefix count differs')
    # A retained ready prefix can contain authored/derived programs absent from
    # the raw catalog. Preserve it while explicit historical inventories prevent
    # reissuing old or quarantined variants as newly discovered work.
    if exclusion_inventory_reference and not dedup_inventory_reference:
        raise ValueError('adaptive exclusions require the original dedup inventory')
    excluded_programs=set(prior)
    if dedup_inventory_reference:excluded_programs.update(_inventory_programs(dedup_inventory_reference))
    if exclusion_inventory_reference:excluded_programs.update(_inventory_programs(exclusion_inventory_reference))
    if target.exists():
        result=json.loads(target.read_text())
        for ref in result['packets']:read(ref)
        return binding(target)
    db=sqlite3.connect('file:'+str(Path(catalog)/'catalog.sqlite')+'?mode=ro',uri=True)
    db.execute('PRAGMA temp_store=FILE');db.execute('CREATE TEMP TABLE selected(sha TEXT PRIMARY KEY)')
    selected=set();excluded=0;seen_candidates=set()
    for key in candidate_hashes(source['cases_binding']):
        if key in seen_candidates:raise ValueError('source repeated a program variant')
        seen_candidates.add(key)
        if key in excluded_programs:excluded+=1;continue
        selected.add(key);db.execute('INSERT INTO selected VALUES(?)',(key,))
    if len(selected)+excluded!=source['cases']:raise ValueError('source candidate count differs')
    if not dedup_inventory_reference and excluded!=len(prior):raise ValueError('candidate prefix is not an exact subset of source candidates')
    if dedup_inventory_reference:
        immutable(out/'adaptive-exclusion-accounting.json',{
          'retained_prefix':prefix_reference,'raw_dedup_inventory':dedup_inventory_reference,
          'excluded_inventory':exclusion_inventory_reference,'unique_excluded_variants':len(excluded_programs),
          'present_excluded_variants':excluded,'absent_excluded_variants':sorted(excluded_programs-seen_candidates),
          'training_admission':False,'scope':'generation dedup/quarantine only; current catalog eligibility still required'})
    db.commit();db.execute('BEGIN')
    if source.get('source_population'):
        population_data=read(source['source_population'])
        if population_data.get('authority') is None:raise ValueError('adaptive population needs its exact authority')
        authority=read(population_data['authority'])
        summary=read(source['catalog_summary'])
        if authority['current']!=source['inputs'] or summary.get('source_population')!=source['source_population']:
            raise ValueError('adaptive population is not bound to candidate authority')
    else:
        population=Path(catalog)/'source-inventory-v3.json'
        if not population.exists():population=Path(catalog)/'source-inventory.json'
        population_data=json.loads(population.read_text())
    metadata={f['path']:f for f in population_data['files']}
    db.execute('CREATE TEMP TABLE active_sources(path TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO active_sources VALUES(?)',((path,)for path in metadata))
    if source.get('source_population'):
        for path,sha,size in db.execute('SELECT f.path,f.sha,f.bytes FROM files f JOIN active_sources a ON a.path=f.path'):
            actual=binding(Path(path))
            if actual['sha256']!=sha or actual['bytes']!=size:raise ValueError('adaptive source bytes changed: '+path)
    eligible=eligible_program_sql()
    missing=db.execute('SELECT s.sha FROM selected s LEFT JOIN programs p ON p.sha=s.sha WHERE p.sha IS NULL').fetchall()
    if missing:raise ValueError('original candidate is absent from its source catalog')
    unready=[row[0]for row in db.execute('SELECT p.sha FROM selected s CROSS JOIN programs p ON p.sha=s.sha WHERE NOT('+eligible+') ORDER BY p.rowid')]
    immutable(out/'unready-programs.json',{'program_sha256':unready,'reason':'current catalog conflict or protected/incomplete occurrence; not omitted','training_admission':False})
    query='''WITH source AS(SELECT p.sha,p.family,p.rowid ordinal,
      (SELECT min(o.file) FROM occurrences o JOIN files f ON f.path=o.file JOIN active_sources a ON a.path=o.file WHERE o.program_sha=p.sha AND o.reason IS NULL AND f.complete=1) file
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
              FROM occurrences o JOIN files f ON f.path=o.file JOIN active_sources a ON a.path=o.file WHERE o.program_sha=? AND f.complete=1 ORDER BY o.file,o.line''',(sha,)):
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
        body=decode_body(db.execute('SELECT body FROM programs WHERE sha=?',(sha,)).fetchone()[0],sha);packet.append((sha,body,family,file))
        if len(packet)==packet_rows:flush()
    flush();db.close()
    if total+len(unready)!=len(selected):raise ValueError('reference work silently omitted a remaining candidate')
    return immutable(target,{'packets':references,'cases':prefix['cases']+total,'appended_cases':total,'prefix':prefix_reference,
      'source':source_reference,'unready':binding(out/'unready-programs.json'),'complete':True,
      'training_admission':False,'generator_calls':0,'packet_size_is_selection_cap':False})


def qualify(inventory_reference,prefix_ready_reference,qualifier_reference,out,*,independent_packets=False,qualification_authority_reference=None):
    inventory=read(inventory_reference);prefix=read(prefix_ready_reference);out=Path(out)
    entries=list(prefix['entries'])
    if [e['candidate_packet']for e in entries]!=inventory['packets'][:len(entries)]:raise ValueError('readiness prefix differs from candidate order')
    if qualification_authority_reference:
        authority=read(qualification_authority_reference)
    else:
        if independent_packets:raise ValueError('independent packets require explicit current qualification authority')
        first=read(entries[0]['qualified_packet']);qualification=read(first['qualification']);inputs=read(qualification['inputs'])
        authority={'protection':inputs['authority']['protection'],'decision_hold_bindings':inputs['authority']['decision_holds'],
          'reference_runtime_sha256':qualification['runtime_inventory_sha256']}
    pending=[];later=[];results=[];blocked=False
    def ready_inventory():
        if not independent_packets:return inventory_reference
        # The source-sequence contract requires ready entries to be the exact
        # inventory prefix. Keep pending candidates explicitly after that prefix.
        unprocessed=inventory['packets'][len(prefix['entries'])+len(results):]
        packets=[entry['candidate_packet']for entry in entries]+[entry['candidate_packet']for entry in pending]+unprocessed
        cases=sum(read(packet)['cases']for packet in packets)
        if len(packets)!=len(inventory['packets']):raise ValueError('independent packet accounting lost candidates')
        body={**inventory,'packets':packets,'qualification_source_inventory':inventory_reference,
          'cases':cases,'unprocessed_packets':unprocessed,'qualification_authority':qualification_authority_reference,
          'independent_packet_qualification':True,'training_admission':False}
        digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
        return immutable(out/('ready-candidate-inventory-'+digest+'.json'),body)
    write(out/'ready-packet-sequence.json',{'schema':'natlang.strong-teacher-ready-packet-sequence/1',
      'candidate_inventory':ready_inventory(),'entries':entries,'complete':False,
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
        if independent_packets:
            entry['candidate_ordinal']=index
            if entry['qualified_cases']:entry['ordinal']=len(entries)
        if not entry['qualified_cases']:blocked=True;pending.append(entry)
        elif independent_packets:entries.append(entry)
        elif blocked:later.append(entry)
        else:entries.append(entry)
        results.append(entry)
        write(out/'qualification-progress.json',{'processed_packets':results,'zero_qualified':pending,'later_qualified':later,'complete':False})
        write(out/'ready-packet-sequence.json',{'schema':'natlang.strong-teacher-ready-packet-sequence/1',
          'candidate_inventory':ready_inventory(),'entries':entries,'complete':False,'blocked_unready_packets':pending,
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
        inventory=prepare(spec['catalog'],spec['source'],spec['prefix_inventory'],out,spec.get('packet_rows',64),
          dedup_inventory_reference=spec.get('dedup_inventory'),exclusion_inventory_reference=spec.get('excluded_inventory'))
        print(json.dumps({'inventory':inventory}),flush=True)
        print(json.dumps({'ready':qualify(inventory,spec['prefix_ready'],spec['qualifier'],out,
          independent_packets=spec.get('independent_packet_qualification',False),
          qualification_authority_reference=spec.get('qualification_authority'))}),flush=True)

if __name__=='__main__':main()
