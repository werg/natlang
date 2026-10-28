#!/usr/bin/env python3
"""Resume complete source-bound LFM3 graphs one authenticated trajectory at a time.

An existing full-source execution/actor/family audit anchors the immutable source
bytes. Each closed train trajectory is rendered together, then independently
receipted before the source cursor advances. Test trajectories are never emitted.
"""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import iter_jsonl_rows,write_json_durable
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
from natlang_neuralese.data.native_graph_schema import parse_graph,communication_classification
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.native_chain_render import render_chain
from scripts.neuralese_data.native_chain_materialize import materialize
from scripts.neuralese_data.native_chain_assembly import assemble


def state(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def bound(path):
 p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':sha256_file_hex(p),'stat':state(p)}


def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--authority',type=Path,required=True)
 p.add_argument('--source-binding-receipt',type=Path,required=True);p.add_argument('--protection',type=Path,required=True)
 p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--max-new-train-chains',type=int,default=0);a=p.parse_args()
 if a.max_new_train_chains<0:raise ValueError('negative chain limit')
 authority=json.loads(a.authority.read_text());historical=json.loads(a.source_binding_receipt.read_text());source=historical['source'];records=a.source/'records.jsonl'
 if authority['source_records']['path']!=str(records.resolve()) or authority['source_records']['sha256']!=source['files']['records.jsonl']['sha256'] or authority['source_records']['stat']!=state(records):
  raise ValueError('records differ from the fully authenticated source execution audit')
 for name in ('chains.jsonl','edges.jsonl','pieces.jsonl'):
  if sha256_file_hex(a.source/name)!=source['files'][name]['sha256']:raise ValueError('source file differs: '+name)
 current=current_protection(a.protection)
 holds,hold_refs=load_decision_holds(sorted((a.protection.parent/'decision-holds').glob('*.jsonl')))
 if current['proof']!=authority['current_protection'] or current['proof']!=source['protection'] or hold_refs!=authority['decision_holds'] or hold_refs!=source['decision_holds']:
  raise ValueError('current protection or decision holds differ from authenticated source')
 package=Path(__file__).resolve().parents[2]; codepaths=[Path(__file__).resolve(),*(package/name for name in (
  'scripts/neuralese_data/native_chain_sources.py','scripts/neuralese_data/native_chain_render.py','scripts/neuralese_data/native_chain_materialize.py',
  'scripts/neuralese_data/native_chain_assembly.py','scripts/neuralese_data/build_native_semantic_pairs_e2e.py',
  'training/neuralese/natlang_neuralese/data/text_corpus.py','training/neuralese/natlang_neuralese/data/trajectory_text.py',
  'training/neuralese/natlang_neuralese/data/prompt_rendering.py','training/neuralese/natlang_neuralese/serve/chat.py',
  'training/neuralese/natlang_neuralese/data/native_graph_conversion.py','training/neuralese/natlang_neuralese/data/native_graph_schema.py',
  'training/neuralese/natlang_neuralese/train/own_output_supervision.py','training/neuralese/natlang_neuralese/model/native_graph_plan.py',
  'training/neuralese/natlang_neuralese/model/shared_state.py'))]
 code=[bound(x) for x in codepaths];reviews=[bound(package/'training/source-reviews'/n) for n in ('holds.jsonl','family-holds.json','datasets.json')]
 live=[bound(Path('/home/werg/natlang/training/source-reviews')/n) for n in ('holds.jsonl','family-holds.json','datasets.json')]
 if any(x['sha256']!=y['sha256'] for x,y in zip(reviews,live,strict=True)):raise ValueError('source reviews differ from live')
 protection_doc=json.loads(a.protection.read_text());guarded=[bound(a.protection),*(bound(a.protection.parent/x['path']) for x in protection_doc['files']),*(bound(x['path']) for x in hold_refs)]
 tok=AutoTokenizer.from_pretrained(str(a.tokenizer),local_files_only=True,trust_remote_code=False);role_ids=provision(tok);fp=tokenizer_fingerprint(tok)
 if fp!='133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3' or list(role_ids)!=[125033,125034,125035,125036]:raise ValueError('tokenizer identity differs')
 tokfiles=[bound(a.tokenizer/n) for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja') if (a.tokenizer/n).exists()]
 identity={'schema':'natlang.lfm3-chain-stream-identity/1','authority':bound(a.authority),'source_receipt':bound(a.source_binding_receipt),
  'source':source,'tokenizer_fingerprint':fp,'roles':list(role_ids),'tokenizer_files':tokfiles,'code':code,'reviews':reviews,'live_reviews':live,'protection_files':guarded}
 guarded_all=code+reviews+live+guarded+tokfiles+[identity['authority'],identity['source_receipt']]
 def guard():
  if state(records)!=authority['source_records']['stat']:raise ValueError('authenticated records changed')
  if any(state(Path(x['path']))!=x['stat'] for x in guarded_all):raise ValueError('code/tokenizer/protection dependency changed')
  if [str(x.resolve()) for x in sorted((a.protection.parent/'decision-holds').glob('*.jsonl'))]!=[x['path'] for x in hold_refs]:raise ValueError('decision hold inventory changed')
 chains={x['trajectory']:x for _,x in iter_jsonl_rows(a.source/'chains.jsonl')};edges=defaultdict(list)
 for _,e in iter_jsonl_rows(a.source/'edges.jsonl'):edges[e['trajectory']].append(e)
 pieces=[x for _,x in iter_jsonl_rows(a.source/'pieces.jsonl')]
 if set(chains)!=set(authority['trajectories']):raise ValueError('authority chain inventory differs')
 a.out.mkdir(parents=True,exist_ok=True)
 with (a.out/'stream.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);frontier_path=a.out/'frontier.json'
  journal=json.loads(frontier_path.read_text()) if frontier_path.exists() else {'schema':'natlang.lfm3-chain-stream/1','identity':identity,'events':[],'offset':0,'complete':False}
  if journal['identity']!=identity:raise ValueError('resumed chain stream has different exact identity')
  closed=set();last=0
  for event in journal['events']:
   tid=event['trajectory']
   if tid in closed or tid not in chains or event['start']!=last or event['stop']<=last:raise ValueError('chain source frontier is not contiguous and unique')
   if event['status']=='emitted':
    rp=Path(event['receipt']['path']);r=json.loads(rp.read_text())
    if sha256_file_hex(rp)!=event['receipt']['sha256'] or r['identity']!=identity or sha256_file_hex(Path(r['graph']['path']))!=r['graph']['sha256']:
     raise ValueError('closed chain receipt or graph differs')
   elif event['status'] not in ('held','rejected'):raise ValueError('unknown frontier disposition')
   closed.add(tid);last=event['stop']
  if journal['offset']!=last or journal['offset']>state(records)[2]:raise ValueError('source frontier offset differs')
  if journal['complete']:
   if closed!=set(chains) or last!=state(records)[2]:raise ValueError('completed chain frontier is incomplete')
   print(json.dumps({'complete':True,'events':len(closed),'new_graphs':0}));return
  emitted=0
  def finish(tid, rows, start, stop):
   nonlocal emitted
   if tid in closed or tid not in chains:raise ValueError('duplicate or unknown source trajectory')
   chain=chains[tid];evidence=authority['trajectories'][tid]
   if [r['id'] for r in rows]!=chain['records'] or evidence['record_ids']!=chain['records'] or evidence['split']!=chain['split']:
    raise ValueError('source trajectory membership or ordering differs')
   guard(); event={'trajectory':tid,'start':start,'stop':stop}
   if chain['split']=='test':event['status']='held'
   elif not evidence['admitted'] or any(r['id'] in holds for r in rows):event.update(status='rejected',reason=evidence['reasons'] or ['decision-hold'])
   else:
    records_by_id={r['id']:r for r in rows};rendered=render_chain(chain,records_by_id,edges[tid],pieces,tokenizer=tok,tokenizer_fingerprint=fp)
    nodes=materialize(rendered,records_by_id,edges[tid],tokenizer=tok,role_ids=role_ids)
    graph=assemble(chain,records_by_id,edges[tid],nodes,tokenizer=tok,tokenizer_fingerprint=fp,role_ids=role_ids,source_binding=source)
    parsed=parse_graph(graph,tokenizer_fingerprint=fp)
    if len(parsed.plan.invocations)<3 or not communication_classification(parsed)['producer_consumer']:raise ValueError('trajectory lacks complete multi-node communication')
    data=(json.dumps(graph,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode();key=hashlib.sha256(tid.encode()).hexdigest();graphpath=a.out/(key+'.jsonl');rp=a.out/(key+'.receipt.json')
    if graphpath.exists() and graphpath.read_bytes()!=data:raise ValueError('orphan chain graph differs')
    if not graphpath.exists():
     tmp=graphpath.with_suffix('.tmp');tmp.unlink(missing_ok=True)
     with tmp.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
     os.replace(tmp,graphpath)
    guard();record={'schema':'natlang.lfm3-checked-source-chain-stream-row/1','identity':identity,'trajectory':tid,'source_interval':[start,stop],
      'nodes':len(graph['nodes']),'edges':len(edges[tid]),'imports':sum(len(x['imports']) for x in graph['nodes']),
      'graph':{'path':str(graphpath.resolve()),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)},'training_exposure':0,'optimizer_updates':0}
    if rp.exists() and json.loads(rp.read_text())!=record:raise ValueError('existing chain receipt differs')
    if not rp.exists():write_json_durable(rp,record)
    event.update(status='emitted',receipt={'path':str(rp.resolve()),'sha256':sha256_file_hex(rp)});emitted+=1
   journal['events'].append(event);journal['offset']=stop;write_json_durable(frontier_path,journal);closed.add(tid)
  with records.open('rb') as f:
   f.seek(journal['offset']);tid=None;rows=[];start=journal['offset']
   while True:
    begin=f.tell();line=f.readline()
    if not line:
     if rows:finish(tid,rows,start,f.tell())
     journal['complete']=set(chains)==closed and journal['offset']==f.tell();write_json_durable(frontier_path,journal);break
    row=json.loads(line);current_id=(row.get('source_ref') or {}).get('trajectory_id')
    if tid is not None and current_id!=tid:
     finish(tid,rows,start,begin)
     if a.max_new_train_chains and emitted>=a.max_new_train_chains:break
     rows=[];start=begin
    tid=current_id;rows.append(row)
  guard();print(json.dumps({'complete':journal['complete'],'events':len(journal['events']),'new_graphs':emitted,'offset':journal['offset']}),flush=True)

if __name__=='__main__':main()
