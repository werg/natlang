#!/usr/bin/env python3
"""Prepare exact old/new graph pairs for unexposed earned-source corrections.

Outputs are evidence only: this does not create a replacement source, recipe,
quota, checkpoint, publication or training admission.
"""
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from natlang_neuralese.data.static_native_context import graphify
from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
from natlang_neuralese.data.native_sources import _row_bindings
from natlang_neuralese.train.own_output_supervision import policy_objective
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.transform_acquired_ordinary_context import _checked,_read

def bind(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}

def prepare(candidate_ref,audit_ref,parent_publication,output):
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    publication=_authenticated_refresh_publication(parent_publication)['publication']
    snapshot_ref=(publication.get('rolling_endpoint_custody_proof') or {}).get('successor')
    snapshot=_read(snapshot_ref);frames=snapshot.get('sft_frames') or {}
    if (snapshot.get('schema')!='natlang.rolling-sft-parent-custody-snapshot/1'
        or snapshot.get('optimizer_present') is not True or snapshot.get('rng_present') is not True
        or (snapshot.get('checkpoint') or {}).get('sha256')!=(publication.get('checkpoint') or {}).get('sha256')):
        raise ValueError('graph pairs require exact published full SFT parent')
    candidate=_read(candidate_ref);audit=_read(audit_ref)
    if (candidate.get('schema')!='natlang.acquired-streaming-source-successor/1'
        or audit.get('schema')!='natlang.acquired-static-native-context-candidate-audit/1'
        or audit.get('source')!=candidate.get('appended_graphs') or audit.get('ce_policy')!=candidate.get('ce_policy')
        or audit.get('row_signals')!=candidate.get('row_signals')):raise ValueError('static audit differs from exact source')
    matches=[(name,frame) for name,frame in frames.items() if (frame.get('source_receipts') or [{}])[0].get('sha256')==candidate['source']['sha256']]
    if len(matches)!=1:raise ValueError('saved parent does not uniquely own the candidate source')
    stage,frame=matches[0];proofs=audit.get('proofs') or [];wanted={p['parent_id']:p for p in proofs}
    if not wanted or len(wanted)!=len(proofs):raise ValueError('static audit graph inventory is invalid')
    objective=policy_objective(_read(candidate['ce_policy']));source=_checked(candidate['appended_graphs'])
    output=Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.stat().st_dev!=source.stat().st_dev:
        raise ValueError('fresh output on original source filesystem required')
    if shutil.disk_usage(output.parent).free<40*1024**3+1024**3:raise ValueError('disk floor')
    stage_dir=output.with_name('.'+output.name+'.staging-'+str(os.getpid()));stage_dir.mkdir()
    pairs=GraphWriter(stage_dir/'graph-pairs.jsonl.zst');graphs=GraphWriter(stage_dir/'derived-graphs.jsonl.zst');rows=[]
    try:
        with open_jsonl_text(source) as stream:
            for line in stream:
                if not line.strip():continue
                parent=json.loads(line);ident=parent.get('id')
                if ident not in wanted:continue
                if ident in {x['id'] for x in rows}:raise ValueError('earned source repeats selected graph')
                prior=_row_bindings([indexed_graph_metadata(parent,tokenizer_fingerprint=parent['tokenizer_fingerprint'])])[ident]
                if ((frame.get('source_row_bindings') or {}).get(ident)!=prior
                    or (frame.get('exposure_counts') or {}).get(ident,0)!=0
                    or (frame.get('required_visits') or {}).get(ident,0)<1):
                    raise ValueError('selected source graph is changed, exposed, or lacks earned quota')
                child,proof=graphify(parent,objective,max_tokens=audit['max_tokens'],consumer_context=audit['consumer_context'],state_tokens=audit['state_tokens'])
                if proof!=wanted[ident]:raise ValueError('derived graph differs from exact real-data audit')
                new=_row_bindings([indexed_graph_metadata(child,tokenizer_fingerprint=parent['tokenizer_fingerprint'])])[ident]
                pairs.write({'parent':parent,'child':child});graphs.write(child)
                rows.append({'id':ident,'old_binding':prior,'new_binding':new,'old_source_sha256':candidate['appended_graphs']['sha256'],
                    'old_communication':False,'prior_exposure':0,'prior_required_visits':frame['required_visits'][ident],
                    'parent_row_sha256':canonical_json_sha256_hex(parent),'derived_row_sha256':canonical_json_sha256_hex(child)})
                if len(rows)==len(wanted):break
        if len(rows)!=len(wanted):raise ValueError('source omitted a selected earned graph')
        pair_ref=pairs.finish();graph_ref=graphs.finish()
        for r in (pair_ref,graph_ref):r['path']=str(output/Path(r['path']).name)
        for r in (candidate_ref,audit_ref,candidate['appended_graphs'],candidate['ce_policy']):_checked(r)
        receipt={'schema':'natlang.earned-static-context-graph-pair-preparation/1','candidate':candidate_ref,'audit':audit_ref,
            'parent_publication':parent_publication,'parent_snapshot':snapshot_ref,'stage_id':stage,
            'source':candidate['source'],'source_leaf':candidate['appended_graphs'],'graph_pairs':pair_ref,
            'derived_graphs':graph_ref,'ce_policy':candidate['ce_policy'],'rows':rows,
            'source_correction_declaration':{'schema':'natlang.static-native-context-source-correction/1',
                'graph_pairs':{k:pair_ref[k] for k in ('path','sha256','bytes')},'ce_policy':candidate['ce_policy'],'rows':len(rows)},
            'status':'evidence prepared; source index, recipe, quota migration, current-parent replay and admission remain required'}
        write_json_durable(stage_dir/'receipt.json',receipt);os.replace(stage_dir,output)
        fd=os.open(output.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return receipt
    except BaseException:
        pairs.abort();graphs.abort()
        if stage_dir.exists():shutil.rmtree(stage_dir)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','audit','parent-publication'):p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    ref=lambda name:{'path':str(getattr(a,name).resolve()),'sha256':getattr(a,name+'_sha256')}
    print(json.dumps(prepare(ref('candidate'),ref('audit'),ref('parent_publication'),a.output),sort_keys=True))
if __name__=='__main__':main()
