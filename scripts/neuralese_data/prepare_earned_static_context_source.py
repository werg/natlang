#!/usr/bin/env python3
"""Stage an exact earned-source replacement and correction ledger.

The resulting source is not indexed, recipe-bound, capacity-qualified, or
admitted. The original checkpoint and source remain untouched.
"""
from __future__ import annotations
import argparse,copy,json,os,shutil
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from natlang_neuralese.data.native_sources import NativeGraphSource
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.prepare_acquired_static_graph_pairs import bind
from scripts.neuralese_data.transform_acquired_ordinary_context import _checked,_read

def prepare(pairs_ref, output, *, model_id):
    if not isinstance(model_id,str) or not model_id:raise ValueError('exact model identity is required')
    pairs=_read(pairs_ref)
    if pairs.get('schema')!='natlang.earned-static-context-graph-pair-preparation/1':raise ValueError('invalid earned graph-pair preparation')
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    publication=_authenticated_refresh_publication(pairs['parent_publication'])['publication']
    if (publication.get('rolling_endpoint_custody_proof') or {}).get('successor')!=pairs.get('parent_snapshot'):
        raise ValueError('earned source parent snapshot differs')
    snapshot=_read(pairs['parent_snapshot']);frame=(snapshot.get('sft_frames') or {}).get(pairs['stage_id'])
    if (not isinstance(frame,dict) or snapshot.get('optimizer_present') is not True or snapshot.get('rng_present') is not True
        or (snapshot.get('checkpoint') or {}).get('sha256')!=(publication.get('checkpoint') or {}).get('sha256')
        or (frame.get('source_receipts') or [{}])[0].get('sha256')!=pairs['source']['sha256']):raise ValueError('earned source parent custody differs')
    config=(frame.get('stage_identity') or {}).get('config') or {}
    if model_id!=str(config.get('architecture'))+':'+str(config.get('base_inventory_sha256')):
        raise ValueError('source correction model identity differs from saved stage')
    from natlang_neuralese.train.native_static_context_correction import verify_static_context_correction
    verified,_=verify_static_context_correction(_checked(pairs_ref),pairs['source_correction_declaration'])
    ledger={item['id']:item for item in pairs['rows']}
    if len(ledger)!=len(pairs['rows']) or set(ledger)!=set(verified):raise ValueError('earned graph pair inventory differs')
    for ident,item in ledger.items():
        if (item['old_binding']!=verified[ident]['old_binding'] or item['new_binding']!=verified[ident]['new_binding']
            or (frame.get('source_row_bindings') or {}).get(ident)!=item['old_binding']
            or (frame.get('exposure_counts') or {}).get(ident,0)!=0
            or (frame.get('required_visits') or {}).get(ident)!=item['prior_required_visits']):
            raise ValueError('earned graph pair changed saved binding, exposure or quota')
    old_manifest_path=_checked(pairs['source']);manifest=json.loads(old_manifest_path.read_text())
    if manifest.get('schema')!='natlang.native-graph-sources/1' or not isinstance(manifest.get('sources'),list):raise ValueError('invalid earned source manifest')
    leaf=_checked(pairs['source_leaf']);target=[]
    for i,item in enumerate(manifest['sources']):
        if item.get('sha256')==pairs['source_leaf']['sha256']:
            path=Path(item['path']);path=(path if path.is_absolute() else old_manifest_path.parent/path).resolve(strict=True)
            if path!=leaf:raise ValueError('source leaf path differs from earned manifest')
            target.append(i)
    if target!=[len(manifest['sources'])-1]:raise ValueError('earned source leaf must be the unique final manifest entry')
    candidate=_read(pairs['candidate']);geometry=_read(candidate['geometry']);identity=geometry.get('identity') or {}
    if ((geometry.get('source') or {}).get('sha256')!=pairs['source']['sha256']
        or identity.get('split')!='train' or not isinstance(identity.get('roles'),list)
        or not isinstance(identity.get('tokenizer'),str)):
        raise ValueError('earned source lacks its exact tokenizer and role geometry binding')
    indexed=NativeGraphSource(old_manifest_path,identity['tokenizer'],required_split='train',role_ids=identity['roles'])
    old_order=frame.get('source_row_order') or [];old_bindings=frame.get('source_row_bindings') or {}
    if (len(indexed.entries)!=len(old_order) or len(old_order)!=len(old_bindings)
        or any(entry['id']!=ident or entry['binding']!=old_bindings.get(ident)
               for ident,entry in zip(old_order,indexed.entries,strict=True))):
        raise ValueError('authenticated source index differs from saved row order or bindings')
    leaf_count=candidate.get('new_rows')
    if type(leaf_count) is not int or not 0<leaf_count<=len(indexed.entries):raise ValueError('invalid earned leaf row count')
    leaf_entries=iter(indexed.entries[-leaf_count:])
    output=Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir() or output.parent.stat().st_dev!=leaf.stat().st_dev:
        raise ValueError('fresh output on original source filesystem required')
    if shutil.disk_usage(output.parent).free<40*1024**3+3*leaf.stat().st_size:raise ValueError('disk floor')
    stage=output.with_name('.'+output.name+'.staging-'+str(os.getpid()));stage.mkdir();writer=GraphWriter(stage/'unchanged-leaf.jsonl.zst')
    try:
        retained=[];selected=[];original_leaf_order=[]
        with open_jsonl_text(leaf) as rows:
            for line in rows:
                if not line.strip():continue
                indexed_entry=next(leaf_entries,None)
                if indexed_entry is None or not line.endswith('\n'):
                    raise ValueError('source leaf has an extra or unterminated indexed row')
                ident=indexed_entry['id'];original_leaf_order.append(ident)
                if ident in ledger:
                    graph=json.loads(line)
                    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
                    if graph.get('id')!=ident or canonical_json_sha256_hex(graph)!=ledger[ident]['parent_row_sha256']:
                        raise ValueError('selected source graph differs from its audited original')
                    selected.append(ident)
                else:
                    retained.append(ident)
                    # The exact source SHA and authenticated execution index bind
                    # this row position. Preserve its original JSON bytes without
                    # parsing and re-serializing an unchanged training example.
                    writer.encoder.stdin.write(line.encode('utf-8'));writer.rows+=1
        if len(set(retained+selected))!=len(retained)+len(selected) or set(selected)!=set(ledger):raise ValueError('source leaf selected identities differ')
        if next(leaf_entries,None) is not None:raise ValueError('source leaf omitted indexed rows')
        leaf_order=retained+selected
        if (len(old_order)!=len(frame['source_row_bindings']) or not set(leaf_order)<=set(old_order)
            or old_order[-len(original_leaf_order):]!=original_leaf_order):
            raise ValueError('saved source row order or inventory differs from the exact final leaf')
        remaining=writer.finish();remaining['path']=str(output/'unchanged-leaf.jsonl.zst')
        child=pairs['derived_graphs'];_checked(child)
        result=copy.deepcopy(manifest)
        for item in result['sources']:
            path=Path(item['path']);item['path']=str((path if path.is_absolute() else old_manifest_path.parent/path).resolve(strict=True))
        original_item=result['sources'][-1];result['sources'][-1]={**original_item,**{k:remaining[k] for k in ('path','sha256','bytes')}}
        result['sources'].append({**original_item,**{k:child[k] for k in ('path','sha256','bytes')}})
        write_json_durable(stage/'train-sources.json',result)
        source_ref={'path':str(output/'train-sources.json'),'sha256':sha256_file_hex(stage/'train-sources.json'),'bytes':(stage/'train-sources.json').stat().st_size,
                    'rows':len(old_order)}
        rows_path=stage/'correction-rows.jsonl'
        with rows_path.open('x') as stream:
            for ident in selected:
                item=ledger[ident]
                stream.write(json.dumps({'id':ident,'old_binding':item['old_binding'],'new_binding':item['new_binding'],
                    'old_source_sha256':item['old_source_sha256'],'new_source_sha256':child['sha256'],
                    'disposition':'corrected','old_communication':False,
                    'reason':'bounded static native context derived from the exact zero-exposure ordinary parent'},sort_keys=True)+'\n')
            stream.flush();os.fsync(stream.fileno())
        correction={'schema':'natlang.native-source-correction/1','model_id':model_id,'stage_id':pairs['stage_id'],
            'prior_source':{**pairs['source'],'rows':len(old_order)},'current_source':source_ref,'row_bindings_path':str(output/'correction-rows.jsonl'),
            'row_bindings_sha256':sha256_file_hex(rows_path),'static_native_context_correction':pairs['source_correction_declaration'],
            'evidence':[pairs_ref]}
        write_json_durable(stage/'source-correction.json',correction)
        for ref in (pairs_ref,pairs['source'],pairs['source_leaf'],pairs['derived_graphs']):_checked(ref)
        receipt={'schema':'natlang.earned-static-context-source-preparation/1','source_correction':{'path':str(output/'source-correction.json'),
            'sha256':sha256_file_hex(stage/'source-correction.json'),'bytes':(stage/'source-correction.json').stat().st_size},
            'train_source':source_ref,'previous_source':pairs['source'],'parent_snapshot':pairs['parent_snapshot'],
            'graph_pairs':pairs_ref,'unchanged_leaf':remaining,'derived_leaf':child,'corrected_ids':selected,
            'unchanged_leaf_rows':len(retained),'old_graph_visits_retained':True,
            'status':'source and typed correction staged; index, geometry, CE family map, recipe, current-parent replay and admission remain required'}
        write_json_durable(stage/'receipt.json',receipt);os.replace(stage,output)
        fd=os.open(output.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return receipt
    except BaseException:
        writer.abort()
        if stage.exists():shutil.rmtree(stage)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--graph-pairs-receipt',type=Path,required=True);p.add_argument('--graph-pairs-sha256',required=True)
    p.add_argument('--model-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare({'path':str(a.graph_pairs_receipt.resolve()),'sha256':a.graph_pairs_sha256},a.output,model_id=a.model_id),sort_keys=True))
if __name__=='__main__':main()
