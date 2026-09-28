#!/usr/bin/env python3
"""Verify preserved state/scene IR against pinned raw annotations, without loading CLEVR wholesale."""
import argparse, hashlib, json, re, zipfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def rows(path):
    with open(path) as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def array_items(stream, key):
    decoder=json.JSONDecoder(); buf=''; started=False; eof=False
    for _ in range(10000000):
        if not started:
            chunk=stream.read(65536); buf+=chunk
            match=re.search(r'"'+re.escape(key)+r'"\s*:\s*\[',buf)
            if match: buf=buf[match.end():]; started=True
            elif not chunk: raise ValueError('missing JSON array')
            else: continue
        buf=buf.lstrip()
        if buf.startswith(','): buf=buf[1:].lstrip()
        if buf.startswith(']'): return
        try: value,end=decoder.raw_decode(buf)
        except json.JSONDecodeError:
            chunk=stream.read(65536)
            if not chunk: raise ValueError('truncated JSON array')
            buf+=chunk; continue
        yield value;buf=buf[end:]
    raise ValueError('JSON array limit')

def checked_input(path):
    manifest=json.loads(Path(str(path)+'.manifest.json').read_text())
    actual=digest(path)
    if actual!=manifest.get('ir_sha256'): raise ValueError(f'IR checksum mismatch: {path}')
    return actual,manifest

def audit(root,out):
    out.mkdir(parents=True,exist_ok=True); ledger=[]; inputs={}; schemas={}; raw={}
    for name in ('scone-v1.ir.jsonl','scone-dev-v1.ir.jsonl','sgd-v2.ir.jsonl','sgd-dev-v2.ir.jsonl','clevr-v1.ir.jsonl','clevr-val-v1.ir.jsonl'):
        path=root/name; file_sha,manifest=checked_input(path); inputs[str(path.resolve())]=file_sha
        source=manifest['adapter']; source_info=manifest['source']; records=list(rows(path))
        if source=='scone':
            archive=root/'scone.zip'; archive_sha=raw.setdefault(str(archive.resolve()),digest(archive))
            if archive_sha!=source_info['archive_sha256']: raise ValueError('SCONE archive changed')
            with zipfile.ZipFile(archive) as z:
                annotations={}
                split=source_info['splits'][0]
                for domain in source_info['domains']:
                    for line in z.read(f'rlong/{domain}-{split}.tsv').decode().splitlines():
                        fields=line.split('\t'); annotations[(domain,fields[0])]=fields
            for r in records:
                fields=annotations.get((r['semantics']['domain'],r['source_ids'][0])); sem=r['semantics']
                before=fields[1] if fields else None;steps=[]
                if fields:
                    for i in range(2,len(fields),2):
                        steps.append({'before':before,'utterance':fields[i],'after':fields[i+1]});before=fields[i+1]
                ok=bool(fields) and sem['initial_state']==fields[1] and sem['steps']==steps and r['source_revisions']==[archive_sha]
                ledger.append({'id':r['id'],'verified':ok,'reason':None if ok else 'raw_state_join_mismatch'})
        elif source=='sgd':
            split=source_info['splits'][0]; folder=root/'sgd'/split
            schema_path=folder/'schema.json';raw[str(schema_path.resolve())]=digest(schema_path)
            schema={s['service_name']:s for s in json.loads(schema_path.read_text())};schemas[split]=schema
            by_dialogue={}
            for r in records: by_dialogue.setdefault(r['source_ids'][0],[]).append(r)
            found=set()
            for file in sorted(folder.glob('dialogues_*.json')):
                file_sha=digest(file);raw[str(file.resolve())]=file_sha
                for dialogue in json.loads(file.read_text()):
                    for r in by_dialogue.get(dialogue['dialogue_id'],[]):
                        service=r['semantics']['domain'];state={'active_intent':'NONE','requested_slots':[],'slot_values':{}};cursor=0;steps=[]
                        for i,turn in enumerate(dialogue['turns']):
                            if turn['speaker']!='USER':continue
                            frame=next((f for f in turn['frames'] if f['service']==service),None)
                            if not frame or 'state' not in frame:continue
                            utterance='\n'.join(t['speaker']+': '+t['utterance'] for t in dialogue['turns'][cursor:i+1])
                            steps.append({'before':state,'after':frame['state'],'utterance':utterance});state=frame['state'];cursor=i+1
                        s=schema.get(service);context=r['semantics'].get('task_context','')
                        ok=(r['source_revisions']==[file_sha] and steps==r['semantics']['steps'] and s is not None
                            and s['description'] in context and all(v['description'] in context for v in s['slots']+s['intents']))
                        ledger.append({'id':r['id'],'verified':ok,'reason':None if ok else 'raw_dialogue_or_schema_join_mismatch'});found.add(r['id'])
            ledger.extend({'id':r['id'],'verified':False,'reason':'raw_dialogue_missing'} for r in records if r['id'] not in found)
        else:
            archive=root/'CLEVR_v1.0_no_images.zip';archive_sha=raw.setdefault(str(archive.resolve()),digest(archive))
            if archive_sha!=source_info['archive_sha256']:raise ValueError('CLEVR archive changed')
            split=source_info['splits'][0]; by_index={int(r['source_ids'][0]):r for r in records};found=set()
            with zipfile.ZipFile(archive) as z:
                scenes=json.loads(z.read(f'CLEVR_v1.0/scenes/CLEVR_{split}_scenes.json'))['scenes']
                import io
                with io.TextIOWrapper(z.open(f'CLEVR_v1.0/questions/CLEVR_{split}_questions.json')) as stream:
                    for q in array_items(stream,'questions'):
                        r=by_index.get(q['question_index'])
                        if r is None:continue
                        sem=r['semantics'];scene=scenes[q['image_index']]
                        projected={'objects':[{k:o[k] for k in ('color','material','shape','size')} for o in scene['objects']], 'relationships':scene['relationships']}
                        answer=q['answer'];answer=(answer=='yes') if answer in ('yes','no') else int(answer) if answer.isdigit() else answer
                        ok=(sem['question']==q['question'] and sem['nodes']==q['program'] and sem['scene']==projected and sem['answer']==answer
                            and r['source_revisions']==[archive_sha] and r['source_groups']==[f'{split}:image:{q["image_index"]}'])
                        ledger.append({'id':r['id'],'verified':ok,'reason':None if ok else 'raw_scene_join_mismatch'});found.add(r['id'])
            ledger.extend({'id':r['id'],'verified':False,'reason':'raw_question_missing'} for r in records if r['id'] not in found)
    ledger_path=out/'source-joins.jsonl';ledger_path.write_text(''.join(json.dumps(r)+'\n' for r in ledger))
    schema_path=out/'service-schemas.json';schema_path.write_text(json.dumps(schemas))
    report={'version':'natlang.recovered_source_joins/1','inputs':inputs,'raw_sources':raw,'verified':sum(r['verified'] for r in ledger),'held':sum(not r['verified'] for r in ledger),
        'ledger':{'path':ledger_path.name,'sha256':digest(ledger_path)},'schemas':{'path':schema_path.name,'sha256':digest(schema_path)}}
    (out/'source-joins.manifest.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=Path('data/external_pilot'));parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.source,args.out),indent=2))
