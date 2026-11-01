#!/usr/bin/env python3
"""Derive source-correction publication evidence from two actual full checkpoints."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from natlang_neuralese.rl.native_direct_sft_recovery import _rolling_endpoint_snapshots
from natlang_neuralese.rl.native_supervised import _rolling_parent_custody_snapshot
from natlang_neuralese.train.native_source_correction import source_correction_dependencies,_proof_file
from scripts.neuralese_data.partition_acquired_context_pending import bind,_checked

def verify(parent,successor,proof_ref,out):
    proof_path=_checked(proof_ref);proof=json.loads(proof_path.read_text());source_correction_dependencies(proof_ref)
    if not proof.get('static_native_context_correction'):raise ValueError('expected typed static native correction')
    out=Path(out).absolute()
    if out.exists() or out.is_symlink() or not out.parent.is_dir():raise ValueError('fresh verification output is required')
    out.mkdir();snapshots=[];checkpoints=[]
    for ref in (parent,successor):
        if not isinstance(ref,dict) or type(ref.get('global_step')) is not int:raise ValueError('checkpoint needs exact global step')
        ref={**ref,'path':str(Path(ref['path']).resolve(strict=True))}
        snapshot_ref=_rolling_parent_custody_snapshot(ref['path'],ref['sha256'],out)
        snapshot=json.loads(_checked(snapshot_ref).read_text())
        if snapshot.get('global_step')!=ref['global_step']:raise ValueError('full checkpoint global step differs')
        snapshots.append((snapshot_ref,snapshot));checkpoints.append(ref)
    declaration={'schema':'natlang.rolling-sft-endpoint-snapshots/1','parent':snapshots[0][0],
        'successor':snapshots[1][0],'source_correction':proof_ref}
    endpoint=_rolling_endpoint_snapshots(declaration,checkpoints[0],checkpoints[1])
    stage=proof['stage_id'];before=snapshots[0][1]['sft_frames'][stage];after=snapshots[1][1]['sft_frames'][stage]
    old_bindings=before['source_row_bindings'];new_bindings=after['source_row_bindings']
    old_quotas=before['required_visits'];new_quotas=after['required_visits']
    old_visits=before['exposure_counts'];new_visits=after['exposure_counts']
    with open_jsonl_text(_proof_file(proof_path,proof['row_bindings_path'])) as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    added=sorted(row['id'] for row in rows if row.get('disposition')=='added')
    changed=[{'id':row['id'],'binding_before':row['old_binding'],'binding_after':row['new_binding']}
        for row in rows if row.get('old_binding') and row.get('new_binding') and row['old_binding']!=row['new_binding']]
    verification={'schema':'natlang.earned-sft-source-transition-verification/1','parent_sha256':parent['sha256'],
        'successor_sha256':successor['sha256'],'proof':proof_ref,
        'parent_source_sha256':proof['prior_source']['sha256'],'successor_source_sha256':proof['current_source']['sha256'],
        'bindings_before_sha256':canonical_json_sha256_hex(old_bindings),'bindings_after_sha256':canonical_json_sha256_hex(new_bindings),
        'quotas_before_sha256':canonical_json_sha256_hex(old_quotas),'quotas_after_sha256':canonical_json_sha256_hex(new_quotas),
        'visits_before_sha256':canonical_json_sha256_hex(old_visits),'visits_after_sha256':canonical_json_sha256_hex(new_visits),
        'old_quotas_preserved':True,'old_exposure_preserved':True,'changed_bindings':changed,'added_rows':len(added),
        'added_ids_sha256':canonical_json_sha256_hex(added),'old_rows':len(old_bindings),'new_rows':len(new_bindings),
        'endpoint_snapshots':declaration,'endpoint_custody':endpoint,
        'scope':'actual full-checkpoint metadata derived afresh; no behavioral or stage completion claim'}
    target=out/'source-transition-verification.json';write_json_durable(target,verification)
    return {'verification':bind(target),'endpoint_snapshots':declaration,'endpoint_custody':endpoint}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','successor','proof'):p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--parent-step',type=int,required=True);p.add_argument('--successor-step',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ref=lambda n:{'path':str(getattr(a,n).resolve()),'sha256':getattr(a,n+'_sha256')}
    print(json.dumps(verify({**ref('parent'),'global_step':a.parent_step},{**ref('successor'),'global_step':a.successor_step},ref('proof'),a.output),sort_keys=True))
if __name__=='__main__':main()
