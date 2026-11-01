#!/usr/bin/env python3
"""Bind port replay to the frozen Node runtime and script bytes it executes."""
import hashlib,json,sys
from pathlib import Path
from natlang_neuralese.common.runtime_inventory import task_runtime_inventory

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''): h.update(block)
 return h.hexdigest()

def identity(root,node,reservations=None):
 root=Path(root).resolve(); host=root/'ts-host'; receipt=root/'runtime-input-closure.json'
 if not receipt.is_file():raise ValueError('port replay requires a frozen runtime with an assembly receipt')
 inventory=task_runtime_inventory(host)
 scripts={str(p.relative_to(root)):digest(p) for p in sorted((host/'scripts').rglob('*.mjs'))}
 if not scripts:raise ValueError('frozen runtime has no Node script closure')
 payload={'schema':'natlang.port-replay-runtime/1','runtime':str(root),'runtime_receipt_sha256':digest(receipt),'node':str(Path(node).resolve()),'node_sha256':digest(node),'task_runtime_sha256':hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'scripts':scripts,'driver_controls':{name:digest(Path(__file__).resolve().parent/name) for name in ('port_record_calls.sh','port_runtime_identity.py','port_predecessor_receipt.py')},'reservations_sha256':digest(reservations or root/'data/external_pilot/jeff-test-tasks.jsonl')}
 payload['identity_sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return payload
if __name__=='__main__':print(json.dumps(identity(sys.argv[1],sys.argv[2],sys.argv[3] if len(sys.argv)>3 else None),sort_keys=True,separators=(',',':')))
