#!/usr/bin/env python3
"""Graphify the exact unearned pending partition against a reviewed append source."""
import argparse, hashlib, json, math, os
from collections import Counter
from pathlib import Path
from scripts.neuralese_data import build_acquired_streaming_successor as b
from scripts.neuralese_data.transform_acquired_ordinary_context import _protected_case
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.audit_acquired_learning_signal import row_signal
from natlang_neuralese.common.custody import ProtectedWorlds, world_identities
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable, write_bytes_durable
from natlang_neuralese.data.static_native_context import graphify, verify_graphification
from natlang_neuralese.train.own_output_supervision import policy_objective, CATEGORIES

BUILDER=Path(__file__).read_bytes()

def build(partition_path,parent_path,output, *, max_tokens=65536):
    partition_path=Path(partition_path).resolve(strict=True);parent_path=Path(parent_path).resolve(strict=True)
    partition=b.read(partition_path);parent=b.read(parent_path);partition_ref=b.bind(partition_path);parent_ref=b.bind(parent_path)
    if type(max_tokens) is not int or not 1026<=max_tokens<=partition['limit']:
        raise ValueError('bounded producer limit must fit the architecture position bound')
    if partition.get('schema')!='natlang.acquired-position-pending-partition/1' or parent.get('training_admission') is not False:
        raise ValueError('requires exact unearned partition and reviewed source preparation')
    protection=ProtectedWorlds.load()
    if protection.digest!=partition['protected_worlds_sha256'] or protection.digest!=parent['current_protection_digest']:
        raise ValueError('current protection union differs')
    pending=b.checked(partition['pending_graph'],'pending graph'); signals=b.checked(partition['original_row_signals'],'original signals')
    b.checked(partition['candidate'],'original candidate');b.checked(partition['original_census'],'original census')
    family_map=b.read(b.checked(parent['family_map'],'parent family map'));policy=policy_objective(b.read(b.checked(parent['ce_policy'],'parent CE')))
    if family_map['source_sha256']!=parent['source']['sha256'] or len(family_map['rows'])!=parent['rows']:
        raise ValueError('parent family map does not bind source')
    prior_ids={r['id'] for r in family_map['rows']}; del family_map
    expected={r['id']:r for r in partition['pending_rows']}
    if len(expected)!=partition['pending_graph']['rows'] or len(expected)!=len(partition['pending_rows']) or set(expected)&prior_ids:
        raise ValueError('pending identities are duplicate or present in predecessor')
    old_signals={}
    with open(signals) as f:
        for line in f:
            item=json.loads(line)
            if item['id'] in expected:
                if item['id'] in old_signals:raise ValueError('repeated original signal')
                old_signals[item['id']]=item
    if set(old_signals)!=set(expected):raise ValueError('missing pending original signal')
    output=Path(output).resolve();b.require_output_headroom(output)
    if output.exists():raise ValueError('fresh output required')
    output.mkdir();builder=output/('builder-'+hashlib.sha256(BUILDER).hexdigest()+'.py');write_bytes_durable(builder,BUILDER)
    writer=GraphWriter(output/'graphs.jsonl.zst');proofs=[];seen=set();mass=Counter();counts=Counter()
    try:
        with open_jsonl_text(pending) as f, (output/'row-signals.jsonl').open('x') as signal_out:
            for line in f:
                row=json.loads(line);ident=row['id'];old=old_signals.get(ident);entry=expected.get(ident)
                if not old or not entry or ident in seen or canonical_json_sha256_hex(row)!=entry['graph_sha256'] or old['graph_sha256']!=entry['graph_sha256']:
                    raise ValueError('pending parent differs from original partition and census')
                if row.get('split')!='train' or world_identities(_protected_case(row),evidence_items=False)&protection.identities:
                    raise ValueError('pending row intersects held protection or changed split')
                child,proof=graphify(row,policy,max_tokens=max_tokens,consumer_context=1024,state_tokens=16)
                verify_graphification(row,child,policy)
                if proof['effective_positive_targets']!=entry['effective_positive_tokens'] or any(not math.isclose(proof['category_mass'].get(str(i),0.),entry['category_mass'].get(name,0.),rel_tol=1e-10,abs_tol=1e-6) for i,name in enumerate(CATEGORIES)):
                    raise ValueError('graphification changed exact effective targets or category mass')
                signal=row_signal(child,policy)
                if signal['effective_positive_tokens']!=entry['effective_positive_tokens'] or any(not math.isclose(signal['category_mass'].get(k,0.),v,rel_tol=1e-10,abs_tol=1e-6) for k,v in entry['category_mass'].items()):
                    raise ValueError('canonical child census differs from original')
                writer.write(child);signal_out.write(json.dumps(signal,sort_keys=True)+'\n');proofs.append(proof);seen.add(ident);mass.update(signal['category_mass'])
                counts.update({'forward_tokens':signal['forward_tokens'],'effective_positive_tokens':signal['effective_positive_tokens']})
            signal_out.flush();os.fsync(signal_out.fileno())
        if seen!=set(expected):raise ValueError('pending graph omitted source rows')
        graph=writer.finish();graph['tokenizer_fingerprint']=partition['pending_graph']['tokenizer_fingerprint'];graph['graph_ids_sha256']=canonical_json_sha256_hex(sorted(seen))
        if ProtectedWorlds.load().digest!=protection.digest or b.bind(parent_path)!=parent_ref or b.bind(partition_path)!=partition_ref:
            raise ValueError('custody changed during graphification')
        result={'schema':'natlang.acquired-pending-static-context/1','training_admission':False,'parent':parent_ref,'parent_source':parent['source'],'partition':partition_ref,'source':partition['pending_graph'],'objective':parent['ce_policy'],'protected_worlds_sha256':protection.digest,'graph':graph,'row_signals':b.bind(output/'row-signals.jsonl'),'rows':len(seen),'max_node_tokens':max_tokens,'counts':dict(counts),'category_mass':dict(mass),'proofs':proofs,'builder':b.bind(builder),'scope':'Static training derivation from source evidence; no observed teacher calls, model exposure, oracle success or admission claimed.'}
        write_json_durable(output/'receipt.json',result)
    except BaseException:
        writer.abort();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partition',required=True);p.add_argument('--parent',required=True);p.add_argument('--output',required=True);p.add_argument('--max-node-tokens',type=int,default=65536);a=p.parse_args();build(a.partition,a.parent,a.output,max_tokens=a.max_node_tokens)
