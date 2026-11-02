#!/usr/bin/env python3
"""Queue one exact transported actor view through the existing native async evaluator."""
import argparse,json
from pathlib import Path
from natlang_neuralese.eval.native_async import file_binding,seal_transport_request
p=argparse.ArgumentParser(description=__doc__)
for name in ('view','view-sha256','transport','transport-sha256','view-receipt','view-receipt-sha256','source-sha256','spec','spec-sha256','base','out','queue'):
    p.add_argument('--'+name,required=True)
p.add_argument('--step',required=True,type=int);p.add_argument('--device',default='cuda')
p.add_argument('--recovery-receipt');p.add_argument('--recovery-receipt-sha256');p.add_argument('--scope',choices=('diagnostic-local-pin','independent-recovery'),required=True)
a=p.parse_args()
def checked(path,expected):
    ref=file_binding(path)
    if ref['sha256']!=expected:raise ValueError('input differs from its independently declared SHA256')
    return ref
checked(a.view,a.view_sha256);checked(a.spec,a.spec_sha256)
if bool(a.recovery_receipt)!=bool(a.recovery_receipt_sha256):raise ValueError('recovery receipt needs its exact SHA256')
actor={'transport':checked(a.transport,a.transport_sha256),'view_receipt':checked(a.view_receipt,a.view_receipt_sha256),
       'recovery_receipt':checked(a.recovery_receipt,a.recovery_receipt_sha256) if a.recovery_receipt else None,
       'source_sha256':a.source_sha256,'scope':a.scope}
result=seal_transport_request(a.view,step=a.step,spec=json.loads(Path(a.spec).read_text()),out=a.out,
    base=a.base,queue=a.queue,transported_actor=actor,device=a.device)
print(json.dumps(result),flush=True)
