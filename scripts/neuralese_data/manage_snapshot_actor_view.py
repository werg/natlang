#!/usr/bin/env python3
"""Retry, bind, or safely release one checkpoint-owner actor-view pin."""
import argparse,json
from natlang_neuralese.model.native_snapshot_publication import retry_pending,prepare_pending_transport,release_pending
from natlang_neuralese.model.native_bundle import _fresh_file_binding
p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
for action in ('retry','transport'):
    q=sub.add_parser(action);q.add_argument('--owner-directory',required=True)
r=sub.add_parser('release');r.add_argument('--owner-directory',required=True)
for name in ('recovery','recovery-sha256','transport','transport-sha256','delivery','delivery-sha256'):
    r.add_argument('--'+name,required=True)
a=p.parse_args()
if a.action=='retry':result=retry_pending(a.owner_directory)
elif a.action=='transport':result=prepare_pending_transport(a.owner_directory)
else:
    def checked(path,sha):
        value=_fresh_file_binding(path)
        if value['sha256']!=sha:raise ValueError('release input differs from its declared SHA256')
        return value
    result=release_pending(a.owner_directory,recovery_ref=checked(a.recovery,a.recovery_sha256),
        transport_ref=checked(a.transport,a.transport_sha256),delivery_ref=checked(a.delivery,a.delivery_sha256))
print(json.dumps(result),flush=True)
