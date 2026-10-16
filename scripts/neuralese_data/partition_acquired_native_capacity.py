"""Preserve all audited ordinary graphs while exposing a declared geometry envelope."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex, sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_jsonl_text, write_json_durable
from scripts.neuralese_data.build_acquired_full_session_graphs import (
    sessions, acquire_output_lease, _record_uncommitted_cleanup)
from scripts.neuralese_data.prepare_acquired_session_admission import bound


def ordinary_geometry(graph, proof):
    """One ordinary invocation has exactly one physical/KV slot per own token.

    This restricted invariant avoids recompiling every already parsed graph.
    Imports, external state, cache-prefix exports and multi-node graphs require
    the actual general planner and are rejected here rather than guessed.
    """
    nodes=graph.get('nodes') or []
    if (len(nodes)!=1 or graph.get('external_states') or nodes[0].get('imports')
            or nodes[0].get('cache_prefix') or nodes[0].get('producing') or nodes[0].get('logical_producing')):
        raise ValueError('capacity partition requires already audited ordinary one-node graphs')
    width=len(nodes[0]['token_ids'])
    if width!=proof['tokens'] or canonical_json_sha256_hex(graph)!=proof['graph_sha256']:
        raise ValueError('capacity partition graph differs from the exact all-row admission proof')
    return {'forward_tokens':width,'kv_slots':width}


def inside(geometry,limits):
    return all(geometry[key]<=limits[key] for key in ('forward_tokens','kv_slots'))


def partition(audit_path,samples_path,limits,out):
    audit_path,samples_path,out=Path(audit_path).resolve(strict=True),Path(samples_path).resolve(strict=True),Path(out).resolve()
    audit_binding={'path':str(audit_path),'sha256':sha256_file_hex(audit_path)}
    samples_binding={'path':str(samples_path),'sha256':sha256_file_hex(samples_path)}
    audit=json.loads(audit_path.read_text());sample=json.loads(samples_path.read_text())
    sample_audit=sample.get('cohort_audit',{})
    if (audit.get('schema')!='natlang.acquired-session-admission-audit/1' or audit.get('status')!='passed'
            or sample.get('passed') is not True or sample_audit.get('sha256')!=audit_binding['sha256']
            or not sample_audit.get('path') or Path(sample_audit['path']).resolve()!=audit_path):
        raise ValueError('capacity partition needs its complete source audit and bound source render samples')
    if (set(limits)!=set(audit['by_model']) or any(set(values)!={'forward_tokens','kv_slots'}
            or any(type(v) is not int or v<1 for v in values.values()) for values in limits.values())):
        raise ValueError('declare positive physical/KV limits for every audited model; limits do not qualify execution')
    proof_path=bound(audit_path.parent/'row-proof.jsonl',{'sha256':audit['row_proof_sha256']})
    proofs={}
    for proof in sessions(proof_path):
        key=(proof['model'],proof['id'])
        if key in proofs:raise ValueError('cohort row proof repeats a graph identity')
        proofs[key]=proof
    binding={'audit':audit_binding,'source_render_samples':samples_binding,'limits':limits,
        'verifier_sha256':sha256_file_hex(__file__)}
    with acquire_output_lease(out):
        receipt_path=out/'receipt.json'
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text())
            if receipt['binding']!=binding:raise ValueError('completed capacity partition has different source/envelope bindings')
            for item in receipt['files']:bound(out/item['path'],item)
            for item in receipt['source_bindings']:bound(item['path'],item)
            return receipt
        pending=out/'pending';files=out/'files'
        for directory in (pending,files):
            if directory.exists():
                # Reuse the converter's exact owned-uncommitted-data inventory
                # before deleting a partial, unreceipted partition shard.
                _record_uncommitted_cleanup(out/'shards',directory)
                shutil.rmtree(directory)
        pending.mkdir()
        counts=Counter();tokens=Counter();seen=set();remaining=[];source_bindings=[]
        outputs={}
        with ExitStack() as stack:
            writers={}
            for model in limits:
                for split in ('train','test'):
                    for part in ('fit','remaining-long'):
                        path=pending/model/split/(part+'.jsonl.zst');path.parent.mkdir(parents=True,exist_ok=True)
                        writers[model,split,part]=stack.enter_context(write_jsonl_text(path))
                        outputs[model,split,part]=path
            coverage=stack.enter_context((pending/'coverage.jsonl').open('w',encoding='utf-8',newline='\n'))
            for conversion in audit['conversions']:
                model=conversion['model']
                for output in conversion['outputs']:
                    path=bound(output['path'],output);source_bindings.append(output)
                    with open_jsonl_text(path) as stream:
                        for row_index,line in enumerate(stream):
                            if not line.strip():continue
                            if not line.endswith('\n'):raise ValueError('preserve exact source rows only with original LF boundaries')
                            graph=json.loads(line);key=(model,graph['id']);proof=proofs.get(key)
                            if proof is None or key in seen:raise ValueError('partition source has an unknown or duplicated audited graph')
                            if graph['split']!=proof['split'] or output['split']!=proof['split']:
                                raise ValueError('partition source changed its audited held split')
                            geometry=ordinary_geometry(graph,proof)
                            part='fit' if inside(geometry,limits[model]) else 'remaining-long'
                            label=(model,graph['split'],part);partition_index=counts[label]
                            writers[label].write(line);counts[label]+=1;tokens[label]+=proof['tokens'];seen.add(key)
                            coverage.write(json.dumps({'model':model,'id':graph['id'],'split':graph['split'],
                                'source':{'path':str(path),'sha256':output['sha256'],'row_index':row_index},
                                'source_graph_sha256':proof['graph_sha256'],'source_row_bytes_sha256':sha256_hex(line.encode('utf-8')),
                                'partition':part,'partition_row_index':partition_index,'geometry':geometry},sort_keys=True)+'\n')
                            if part=='remaining-long' and graph['split']=='train':
                                remaining.append({'model':model,'id':graph['id'],'graph_sha256':proof['graph_sha256'],
                                    'required_visits':1,'geometry':geometry})
        if seen!=set(proofs):raise ValueError('capacity partition lost audited graphs')
        summaries={};manifests={}
        for model in limits:
            summaries[model]={}
            for split in ('train','test'):
                for part in ('fit','remaining-long'):
                    label=(model,split,part);path=outputs[label]
                    summaries[model][split+':'+part]={'rows':counts[label],'tokens':tokens[label]}
                    if counts[label]:
                        manifest=path.parent/(part+'-sources.json')
                        write_json_durable(manifest,{'schema':'natlang.native-graph-sources/1',
                            'sources':[{'kind':'native-graphs','path':path.name,'sha256':sha256_file_hex(path),'weight':1.,'epochs':1}]})
                        manifests[':'.join(label)]=str(Path('files')/manifest.relative_to(pending))
        write_json_durable(pending/'pending-sources.json',{
            'schema':'natlang.native-pending-source-declaration/1','full_cohort':audit_binding,
            'binding':binding,'rows':remaining,
            'policy':'source obligations only; earned exposure belongs to existing native_continuation progress'})
        for path in pending.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:os.fsync(stream.fileno())
        pending.replace(files)
        directory_fd=os.open(out,os.O_DIRECTORY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        inventory=[{'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':sha256_file_hex(path)}
            for path in sorted(files.rglob('*')) if path.is_file()]
        if sha256_file_hex(audit_path)!=audit_binding['sha256'] or sha256_file_hex(samples_path)!=samples_binding['sha256']:
            raise ValueError('source evidence changed during capacity partition')
        for item in source_bindings:bound(item['path'],item)
        receipt={'schema':'natlang.acquired-native-capacity-partition/1','binding':binding,
            'source_bindings':source_bindings,
            'counts':summaries,'graphs':len(seen),'coverage':'files/coverage.jsonl','files':inventory,
            'manifests':manifests,'pending_source_declaration':'files/pending-sources.json',
            'source_eligibility_unchanged':True,'exact_original_row_bytes':True,
            'execution_qualified':False,'production_append_requires_actual_capacity_certificate':True,
            'full_recipe_completed':False}
        write_json_durable(receipt_path,receipt)
        return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--source-render-samples',type=Path,required=True)
    parser.add_argument('--limits',type=Path,required=True,help='model keyed forward_tokens/kv_slots; parameters, not certification')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    receipt=partition(args.audit,args.source_render_samples,json.loads(args.limits.read_text()),args.out)
    print(json.dumps({'path':str(args.out/'receipt.json'),'sha256':sha256_file_hex(args.out/'receipt.json'),
        'counts':receipt['counts'],'execution_qualified':False,'full_recipe_completed':False},sort_keys=True))


if __name__=='__main__':main()
