#!/usr/bin/env python3
"""Audit actual per-invocation positions and exposure for large saved-source graphs.

Graphs are not changed. This examines only rows whose total forward/KV geometry
can exceed the declared position bound, under an authenticated saved endpoint.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train.native_packed_capacity import _rows_digest
from scripts.neuralese_data.partition_acquired_context_pending import _checked,_read,bind

def audit(source_ref,publication_ref,output,*,stage,limit):
    if type(limit) is not int or limit<2:raise ValueError('position limit must be positive')
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    pub=_authenticated_refresh_publication(publication_ref)['publication']
    snapshot_ref=(pub.get('rolling_endpoint_custody_proof') or {}).get('successor');snapshot=_read(snapshot_ref)
    if (snapshot.get('schema')!='natlang.rolling-sft-parent-custody-snapshot/1' or snapshot.get('optimizer_present') is not True
        or snapshot.get('rng_present') is not True or (snapshot.get('checkpoint') or {}).get('sha256')!=(pub.get('checkpoint') or {}).get('sha256')):
        raise ValueError('position audit lacks exact full-checkpoint endpoint')
    source_receipt=_read(source_ref);frame=(snapshot.get('sft_frames') or {}).get(stage)
    if not isinstance(frame,dict) or (frame.get('source_receipts') or [{}])[0].get('sha256')!=(source_receipt.get('source') or {}).get('sha256'):
        raise ValueError('position audit source differs from saved SFT frame')
    geometry=_read(source_receipt['geometry']);identity=geometry['identity'];manifest=_checked(source_receipt['source'])
    source=NativeGraphSource(manifest,identity['tokenizer'],required_split='train',role_ids=identity['roles'])
    if (source.file_sha256!=source_receipt['source']['sha256'] or (geometry.get('source') or {}).get('sha256')!=source.file_sha256
        or _rows_digest(source.entries)!=geometry.get('rows_sha256') or len(source.entries)!=len(geometry.get('kv_slots',[]))):
        raise ValueError('position audit source geometry differs')
    indices=[index for index,(entry,slots) in enumerate(zip(source.entries,geometry['kv_slots'],strict=True))
             if entry['forward_tokens']>limit or slots>limit]
    print(json.dumps({'candidates':len(indices),'source':source.file_sha256}),flush=True)
    records=[];counts=Counter()
    for index,example in zip(indices,source.iter_indices(indices),strict=True):
        entry=source.entries[index];nodes=example.plan.invocations
        maximum=max(example.plan.slot_positions,default=0);own=max((len(node.token_ids) for node in nodes),default=0)
        ordinary=(len(nodes)==1 and not nodes[0].imports and not nodes[0].producing and not nodes[0].logical_producing and not example.external_states)
        visits=(frame.get('exposure_counts') or {}).get(example.id,0)
        if (frame.get('source_row_bindings') or {}).get(example.id)!=entry['binding'] or type(visits) is not int or visits<0:
            raise ValueError('position candidate differs from saved binding or exposure')
        row={'id':example.id,'binding':entry['binding'],'forward_tokens':entry['forward_tokens'],'kv_slots':geometry['kv_slots'][index],
            'nodes':len(nodes),'max_node_tokens':own,'max_position':maximum,'ordinary_single_node':ordinary,
            'communication':entry.get('communication',False),'saved_visits':visits,
            'saved_quota':(frame.get('required_visits') or {}).get(example.id)}
        records.append(row);counts['position_over_limit']+=maximum>=limit
        counts['ordinary_over_limit']+=ordinary and maximum>=limit
        counts['exposed_ordinary_over_limit']+=ordinary and maximum>=limit and visits>0
        if len(records)%8==0:print(json.dumps({'audited':len(records),'counts':dict(counts)}),flush=True)
    if len(records)!=len(indices):raise ValueError('position audit omitted indexed candidates')
    for ref in (source_ref,publication_ref,snapshot_ref,source_receipt['source'],source_receipt['geometry']):_checked(ref)
    output=Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():raise ValueError('fresh audit output required')
    result={'schema':'natlang.native-saved-position-audit/1','source_receipt':source_ref,'source':source_receipt['source'],
        'geometry':source_receipt['geometry'],'publication':publication_ref,'snapshot':snapshot_ref,'stage':stage,
        'position_limit':limit,'candidates':len(indices),'counts':dict(counts),'rows':records,'builder':bind(Path(__file__)),
        'scope':'exact saved source and exposure; rows with total slots within the bound cannot contain a larger slot position; no current checkpoint or behavior claim'}
    write_json_durable(output,result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-receipt','publication'):p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--stage',required=True);p.add_argument('--limit',type=int,default=262144);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    ref=lambda n:{'path':str(getattr(a,n).resolve()),'sha256':getattr(a,n+'_sha256')}
    result=audit(ref('source_receipt'),ref('publication'),a.out,stage=a.stage,limit=a.limit)
    print(json.dumps({'out':str(a.out),'counts':result['counts'],'candidates':result['candidates']}))
if __name__=='__main__':main()
