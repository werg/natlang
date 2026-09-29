#!/usr/bin/env python3
"""Acquire pinned retired WorkflowEvals sources, preserving original test split and labels."""
import argparse, hashlib, json, pathlib
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
DATASETS={
 'invoice-processing':'6beeb2d2acd65c086c835022f5f4d7434114cafc',
 'customer-service':'b1342f5a704587dbc465867c38c2694348ff86e4',
 'security-incidents':'fbe1ea5c69cf494157fd23f2002a0d9d9a418443',
 'agent-trace-observability':'8635540973910a92465fe2bc53e195375aa6e1a8',
}
CODE_REVISION='0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def acquire(out):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 def one(item):
  name,rev=item;directory=out/name;directory.mkdir(exist_ok=True);files={};tables={}
  for filename in ['README.md','dataset.json','data/cases.parquet','data/questions.parquet']:
   p=pathlib.Path(hf_hub_download('typesafe/evalsafe-'+name,filename,repo_type='dataset',revision=rev,local_dir=directory));files[filename]={'sha256':digest(p),'bytes':p.stat().st_size}
   if filename.endswith('.parquet'):
    key=pathlib.Path(filename).stem;j=directory/(key+'.json');count=0
    with j.open('w') as output:
     output.write('[')
     for batch in pq.ParquetFile(p).iter_batches(batch_size=16):
      for row in batch.to_pylist():
       if count:output.write(',')
       json.dump(row,output,ensure_ascii=False);count+=1
     output.write(']\n')
    tables[key]={'path':str(j.relative_to(out)),'sha256':digest(j),'rows':count}
  print(name,{k:v['rows'] for k,v in tables.items()},flush=True)
  return name,{'repository':'typesafe/evalsafe-'+name,'revision':rev,'original_split':'test','license':'Apache-2.0','license_basis':'repository_license' if name=='invoice-processing' else 'user_confirmation_2026-09-29','files':files,'tables':tables}
 sources=dict(one(item) for item in DATASETS.items())
 manifest={'version':'natlang.workflowevals_sources/1','collection':'typesafe/workflowevals-6abaf8dcb1e283d9e3d57b6b','sources':sources,'code_revision':CODE_REVISION,'split_release':{'status':'retired_evaluation_released_for_training','authorization':'user_request_2026-09-29','scope':'these four pinned revisions only','original_split':'test','training_split':'train'}}
 tmp=out/'manifest.json.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n');tmp.replace(out/'manifest.json')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();acquire(a.out)
