#!/usr/bin/env python3
"""Put representative held-out final-return readers first, retaining every producer and training row."""
import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict
from audit_neuralese_recurrence import names

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--eval',type=int,default=28);a=p.parse_args()
 rows=[json.loads(line) for line in a.records.open()];producers={n:r for r in rows for n in names(r.get('target'),'write')}
 def depth(r,visiting=()):
  deps=names(r['messages'],'read')-names(r.get('target'),'write')
  if deps & set(visiting):raise ValueError('cycle')
  return max([1+depth(producers[n],visiting+(n,)) for n in deps] or [0])
 buckets=defaultdict(list)
 for r in rows:
  if r.get('split')!='test' or not names(r['messages'],'read') or names(r.get('target'),'write'):continue
  calls=r.get('target',{}).get('tool_calls',[])
  if not any(c['function']['name']=='return_result' for c in calls):continue
  # Root readers alone qualify the final program result; middle merges are not extra tasks.
  if r.get('source_ref',{}).get('parent_invocation_id'):continue
  ir=r['task']['program_ir'];family=ir['curriculum']['family'];fixture=tuple(ir['source_groups'])
  buckets[(family,fixture)].append((depth(r),r))
 for key in buckets:buckets[key].sort(key=lambda x:(-x[0],hashlib.sha256(x[1]['id'].encode()).hexdigest()))
 selected=[]
 while any(buckets.values()) and len(selected)<a.eval:
  for key in sorted(buckets):
   if buckets[key] and len(selected)<a.eval:selected.append(buckets[key].pop(0)[1])
 if len(selected)!=a.eval:raise ValueError('not enough held-out root readers')
 if a.out.exists():raise ValueError('fresh output required')
 a.out.mkdir(parents=True);ids={r['id'] for r in selected}
 ordered=[r for r in rows if r['split']=='train']+selected+[r for r in rows if r['split']=='test' and r['id'] not in ids]
 (a.out/'records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ordered))
 report={'schema':'natlang.recurrence-evaluation-selection/1','source_sha256':hashlib.sha256(a.records.read_bytes()).hexdigest(),
  'train_records_retained':sum(r['split']=='train' for r in ordered),'heldout_records_retained':sum(r['split']=='test' for r in ordered),'eval_readers':len(selected),
  'selected':[{'id':r['id'],'depth':depth(r),'source_groups':r['task']['program_ir']['source_groups']} for r in selected],
  'role':'held-out final-value diagnostics; no held-out optimization; every producer retained'}
 (a.out/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='selected'}))
if __name__=='__main__':main()
