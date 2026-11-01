#!/usr/bin/env python3
"""Record and validate a stopped predecessor before reusing completed shards."""
import hashlib,json,sys
from pathlib import Path

def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(root):
 root=Path(root).resolve(); rows={}
 for s in sorted((root/'turns').glob('*/*.summary.json')):
  base=str(s)[:-13]; streams=[Path(base+'.turns.jsonl.zst'),Path(base+'.rows.jsonl.zst')]
  if not all(p.is_file() and p.stat().st_size for p in streams):continue
  rows[str(s.relative_to(root))]={'sha256':hashfile(s),'streams':[{'path':str(p.relative_to(root)),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in streams]}
 return {'schema':'natlang.port-predecessor/1','root':str(root),'runtime_metadata_sha256':hashfile(root/'runtime-page.json'),'completed':rows,'scope':'Existing historical runtime metadata is retained; its old stamp is not a content identity.'}
if __name__=='__main__':
 op,root,receipt=sys.argv[1:]; path=Path(receipt)
 if op=='create':path.write_text(json.dumps(snapshot(root),sort_keys=True,indent=2)+'\n')
 elif op=='verify':
  if json.loads(path.read_text())!=snapshot(root):raise ValueError('predecessor publication changed since receipt')
 else:raise ValueError('expected create or verify')
