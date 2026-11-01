#!/usr/bin/env python3
"""Stage graphified versions of an immutable unearned pending partition.

This does not discharge pending obligations, add training exposure, or admit
source data. The original complete graphs remain separately retained.
"""
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from natlang_neuralese.data.static_native_context import graphify
from natlang_neuralese.train.own_output_supervision import policy_objective
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.partition_acquired_context_pending import _checked,_read,bind

def prepare(partition_ref,audit_ref,output):
    partition=_read(partition_ref);audit=_read(audit_ref)
    if (partition.get('schema')!='natlang.acquired-position-pending-partition/1'
        or audit.get('schema')!='natlang.acquired-static-native-context-candidate-audit/1'
        or audit.get('source',{}).get('sha256')!=partition.get('input_graph',{}).get('sha256')):
        raise ValueError('pending partition and static audit do not bind the same original source')
    candidate=_read(partition['candidate'])
    if (candidate.get('ce_policy')!=audit.get('ce_policy') or candidate.get('row_signals')!=audit.get('row_signals')):
        raise ValueError('pending static audit differs from candidate loss custody')
    source=_checked(partition['pending_graph']);objective=policy_objective(_read(candidate['ce_policy']))
    pending={item['id']:item for item in partition['pending_rows']};audited={item['parent_id']:item for item in audit['proofs']}
    if not pending or set(pending)!=set(audited) or len(pending)!=len(partition['pending_rows']) or len(audited)!=len(audit['proofs']):
        raise ValueError('pending and audited graph inventories differ')
    output=Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.stat().st_dev!=source.stat().st_dev:
        raise ValueError('fresh output on original data filesystem required')
    if shutil.disk_usage(output.parent).free<40*1024**3+1024**3:raise ValueError('disk floor')
    stage=output.with_name('.'+output.name+'.staging-'+str(os.getpid()));stage.mkdir()
    graph_writer=GraphWriter(stage/'derived-graphs.jsonl.zst');pair_writer=GraphWriter(stage/'graph-pairs.jsonl.zst');proofs=[];seen=set()
    try:
        with open_jsonl_text(source) as stream:
            for line in stream:
                if not line.strip():continue
                parent=json.loads(line);ident=parent.get('id')
                if ident not in pending or ident in seen or canonical_json_sha256_hex(parent)!=pending[ident]['graph_sha256']:
                    raise ValueError('pending source graph differs from its original proof')
                child,proof=graphify(parent,objective,max_tokens=audit['max_tokens'],consumer_context=audit['consumer_context'],state_tokens=audit['state_tokens'])
                if proof!=audited[ident]:raise ValueError('pending static graph differs from the independent audit')
                seen.add(ident);proofs.append(proof);graph_writer.write(child);pair_writer.write({'parent':parent,'child':child})
        if seen!=set(pending):raise ValueError('pending source omitted an audited graph')
        graph_ref=graph_writer.finish();pair_ref=pair_writer.finish()
        for ref in (graph_ref,pair_ref):ref['path']=str(output/Path(ref['path']).name)
        for ref in (partition_ref,audit_ref,partition['candidate'],partition['pending_graph'],candidate['ce_policy']):_checked(ref)
        receipt={'schema':'natlang.pending-static-context-preparation/1','partition':partition_ref,'audit':audit_ref,
            'candidate':partition['candidate'],'original_pending_graphs':partition['pending_graph'],
            'ce_policy':candidate['ce_policy'],'derived_graphs':graph_ref,'graph_pairs':pair_ref,'proofs':proofs,
            'status':'derived evidence only; original pending obligations remain; no exposure, admission or behavioral claim'}
        write_json_durable(stage/'receipt.json',receipt);os.replace(stage,output)
        fd=os.open(output.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return receipt
    except BaseException:
        graph_writer.abort();pair_writer.abort()
        if stage.exists():shutil.rmtree(stage)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('partition','audit'):p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=lambda name:{'path':str(getattr(a,name).resolve()),'sha256':getattr(a,name+'_sha256')}
    print(json.dumps(prepare(ref('partition'),ref('audit'),a.output),sort_keys=True))
if __name__=='__main__':main()
