#!/usr/bin/env python3
"""One bounded production-parser pass for acquired leaf indexes and loss census."""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.own_output_supervision import policy_objective
from scripts.neuralese_data.build_acquired_streaming_successor import prepare_parallel_leaf,bind,checked

CODE=Path(__file__).read_bytes()


def prepare(combined_ref,*,output,workers=2,pending_census_ref=None,resume=False):
    combined_path=checked(combined_ref,'combined source receipt');combined=json.loads(combined_path.read_text())
    if combined.get('schema')!='natlang.combined-acquired-screened-delta/1' or combined.get('training_admission') is not False:
        raise ValueError('requires exact unadmitted combined source receipt')
    pred_path=checked(combined['predecessor'],'prepared predecessor');previous=json.loads(pred_path.read_text())
    manifest_path=checked(combined['source'],'combined source manifest');manifest=json.loads(manifest_path.read_text())
    if manifest.get('schema')!='natlang.native-graph-sources/1' or len(manifest.get('sources',[]))!=combined['source_leaves']:
        raise ValueError('combined source leaf manifest differs')
    policy_path=checked(previous['ce_policy'],'predecessor CE policy');objective=policy_objective(json.loads(policy_path.read_text()))
    geometry_path=checked(previous['geometry'],'predecessor geometry');geometry=json.loads(geometry_path.read_text())
    tokenizer=geometry['identity']['tokenizer'];roles=geometry['identity']['roles']
    if tokenizer!=combined['tokenizer_fingerprint']:raise ValueError('combined source tokenizer changed')
    family_path=checked(previous['family_map'],'predecessor family map');family=json.loads(family_path.read_text())
    old_ids={row['id'] for row in family['rows']}
    if (len(old_ids)!=previous['rows'] or family.get('source_sha256')!=previous['source']['sha256']):
        raise ValueError('predecessor source identity inventory differs')
    if pending_census_ref is None:raise ValueError('source history review needs the complete original pending census')
    pending_census_path=checked(pending_census_ref,'original pending census')
    pending_census=json.loads(pending_census_path.read_text())
    if pending_census.get('schema')!='natlang.acquired-render-pending-census/1':raise ValueError('unsupported source history census')
    original_profiles={}
    for item in pending_census['receipts']:
        audited=json.loads(checked(item,'original source audit').read_text())
        converter_ref=audited['converter_receipt'];converter=json.loads(checked(converter_ref,'original converter receipt').read_text())
        if converter.get('source')!=audited['source']:raise ValueError('original converter/source binding changed')
        original_profiles[audited['source']['path']]={'source':audited['source'],
            'profile':converter.get('explicit_history_reasoning_policy'),'converter':converter_ref}
    component_paths=[Path(__file__),Path(__file__).with_name('parallel_native_graph_index.py'),
        Path(__file__).with_name('acquired_content_review.py'),Path(__file__).with_name('audit_acquired_learning_signal.py'),
        Path(__file__).with_name('build_acquired_streaming_successor.py')]
    components=[bind(path) for path in component_paths]
    component_bytes=[path.read_bytes() for path in component_paths]
    if any(hashlib.sha256(data).hexdigest()!=ref['sha256'] for data,ref in zip(component_bytes,components)):
        raise ValueError('index code changed during capture')
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
    from scripts.neuralese_data.decision_holds import load_decision_holds
    protection_path=Path(combined['quality_protection']['path']);quality=current_protection(protection_path)
    hold_paths=sorted((Path(__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    _,hold_refs=load_decision_holds(hold_paths)
    if quality['proof']!=combined['quality_protection'] or hold_refs!=combined['decision_holds']:
        raise ValueError('source quality or decision holds changed')
    protection=ProtectedWorlds.load()
    if protection.digest!=combined['protected_worlds_sha256']:raise ValueError('current protection union changed')
    lookup={}
    for ref in combined['screens']:
        screen=json.loads(checked(ref,'source screen').read_text())
        for item in screen['output_inventory']:
            if item['path'] in lookup:raise ValueError('source leaf repeats')
            lookup[item['path']]=item
    output=Path(output)
    if output.exists() and (not resume or (output/'receipt.json').exists()):
        raise ValueError('output exists or has already completed')
    output.mkdir(parents=True,exist_ok=resume)
    captured=output/'runtime-source';captured.mkdir(exist_ok=resume)
    for path,data in zip(component_paths,component_bytes):
        target=captured/path.name
        if target.exists():
            if target.read_bytes()!=data:raise ValueError('resumed census runtime differs')
        else:target.write_bytes(data)
    captured_components=[bind(captured/path.name) for path in component_paths]
    signal_path=output/'row-signals.jsonl';quality_path=output/'content-review.jsonl'
    counts=Counter();masses=Counter();stores=Counter();roles_total=Counter();unknown=Counter();fingerprints=defaultdict(list);templates=Counter()
    proofs=[];all_seen=set();geometry_by_id={};ids=[];history_profiles=Counter();source_reviews=[]
    with signal_path.open('w') as signal_out, quality_path.open('w') as quality_out:
        def consume(signal,review):
            if signal['id']!=review['id'] or signal['id'] in all_seen:raise ValueError('census and parser row identity differ')
            all_seen.add(signal['id']);ids.append(signal['id']);signal_out.write(json.dumps(signal,sort_keys=True)+'\n');quality_out.write(json.dumps(review,sort_keys=True)+'\n')
            for k in ('forward_tokens','raw_positive_tokens','effective_positive_tokens'):counts[k]+=signal[k]
            counts['zero_effective_rows']+=not signal['eligible'];counts['source_label_diagnosis_rows']+=signal['source_label_diagnosis']
            counts['oracle_verified_rows']+=signal['oracle_verified'];counts['native_execution_verified_rows']+=signal['native_execution_verified']
            masses.update(signal['category_mass']);stores[signal['source_store'] or '<unknown>']+=1
            if not signal['source_store']:unknown['source_store']+=1
            roles_total.update(review['role_counts']);counts['unknown_role_starts']+=review['unknown_role_starts']
            fingerprints[review['exact_token_weight_sha256']].append(signal['id']);templates[review['role_structure_sha256']]+=1
        for index,leaf in enumerate(manifest['sources']):
            item=lookup.get(leaf['path'])
            if not item or any(item.get(k)!=leaf.get(k) for k in ('path','sha256','bytes')):
                raise ValueError('manifest leaf differs from screened input')
            checked(item,'screened graph leaf')
            converter_path=Path(leaf['path']).parent.parent/'receipt.json';converter_ref=bind(converter_path)
            converter=json.loads(converter_path.read_text())
            if item not in converter.get('outputs',[]) or converter.get('training_admission') is not False:
                raise ValueError('source leaf lacks its exact original conversion receipt')
            export_path=checked(converter['export_receipt'],'normalized source export');export=json.loads(export_path.read_text())
            derived=(export.get('contract') or {}).get('source_syntax_repair')
            if derived:
                parent=derived['parent_source'];original=original_profiles.get(parent['path'])
                if (original is None or original['source']!={k:parent[k] for k in ('path','sha256')}
                        or original['profile']!=converter.get('explicit_history_reasoning_policy')):
                    raise ValueError('syntax-repaired source did not retain its actual original history profile')
            else:original=None
            profile=converter.get('explicit_history_reasoning_policy')
            if profile not in (None,'keep'):raise ValueError('unsupported history profile')
            history_profiles[profile or 'default']+=1
            source_reviews.append({'source':{k:leaf[k] for k in ('path','sha256','bytes')},'converter':converter_ref,
                'export':converter['export_receipt'],'history_reasoning_policy':profile,
                'original_converter':original['converter'] if original else None,
                'syntax_repair':bool(derived),'outcome_scope':'source-reported; tool execution not inferred'})
            part=output/('leaf-%03d'%index);part.mkdir(exist_ok=resume)
            complete_path=part/'completed.json'
            if complete_path.exists():
                if not resume:raise ValueError('leaf already completed')
                complete=json.loads(complete_path.read_text())
                if (complete.get('schema')!='natlang.acquired-index-census-leaf/1'
                        or complete.get('source')!={k:item[k] for k in ('path','sha256','bytes')}
                        or complete.get('combined_sha256')!=combined_ref['sha256']
                        or complete.get('objective_sha256')!=canonical_json_sha256_hex(objective)
                        or complete.get('code_sha256')!=[ref['sha256'] for ref in components]):
                    raise ValueError('completed leaf controls changed')
                proof=complete['index_proof'];proof_value=json.loads(checked(proof,'resumed leaf index proof').read_text())
                if (proof_value.get('schema')!='natlang.parallel-native-source-index/1'
                        or proof_value.get('source')!=complete['source'] or proof_value.get('rows')!=item['rows']
                        or proof_value.get('parallel_builder',{}).get('sha256')!=components[1]['sha256']):
                    raise ValueError('resumed production-parser proof differs')
                saved_geometry=json.loads(checked(complete['geometry'],'resumed leaf geometry').read_text())
                observed={(entry['id'],entry['binding']):entry['slots'] for entry in saved_geometry}
                resumed_ids=[]
                with checked(complete['signals'],'resumed leaf signals').open() as si, checked(complete['reviews'],'resumed leaf review').open() as ri:
                    for a,b in zip(si,ri,strict=True):
                        signal,review=json.loads(a),json.loads(b);resumed_ids.append(signal['id']);consume(signal,review)
                if (resumed_ids!=[entry['id'] for entry in saved_geometry]
                        or canonical_json_sha256_hex([entry['slots'] for entry in saved_geometry])!=proof_value['ordered_geometry_sha256']):
                    raise ValueError('resumed leaf signal/geometry order differs')
                if len(observed)!=len(saved_geometry) or len(observed)!=item['rows']:
                    raise ValueError('resumed leaf geometry count differs')
            else:
                attempt=0
                while (part/('attempt-%03d'%attempt)).exists():attempt+=1
                work=part/('attempt-%03d'%attempt);work.mkdir()
                ls=work/'signals.jsonl';lr=work/'reviews.jsonl'
                with ls.open('x') as sf,lr.open('x') as rf:
                    def save_row(signal,review):
                        sf.write(json.dumps(signal,sort_keys=True)+'\n');rf.write(json.dumps(review,sort_keys=True)+'\n')
                        consume(signal,review)
                    observed,proof=prepare_parallel_leaf(Path(leaf['path']),item,tokenizer,roles,old_ids,work,
                        workers=workers,audit=objective,on_review=save_row)
                    for stream in (sf,rf):stream.flush();os.fsync(stream.fileno())
                saved_geometry=[{'id':ident,'binding':binding,'slots':slots} for (ident,binding),slots in observed.items()]
                gp=work/'geometry.json';write_json_durable(gp,saved_geometry)
                complete={'schema':'natlang.acquired-index-census-leaf/1',
                    'source':{k:item[k] for k in ('path','sha256','bytes')},'combined_sha256':combined_ref['sha256'],
                    'objective_sha256':canonical_json_sha256_hex(objective),'code_sha256':[ref['sha256'] for ref in components],
                    'index_proof':proof,'geometry':bind(gp),'signals':bind(ls),'reviews':bind(lr)}
                write_json_durable(complete_path,complete)
            if len(geometry_by_id.keys() & {key[0] for key in observed}):raise ValueError('parallel leaf repeated an identity')
            geometry_by_id.update({ident:slots for (ident,_),slots in observed.items()});proofs.append(proof)
            print(json.dumps({'kind':'acquired_manifest_census_progress','files':index+1,'of':len(manifest['sources']),'rows':len(ids)}),flush=True)
        for f in (signal_out,quality_out):f.flush();os.fsync(f.fileno())
    if len(ids)!=combined['rows'] or len(all_seen)!=len(ids) or len(geometry_by_id)!=len(ids):
        raise ValueError('combined source index did not cover every screened graph')
    duplicates=[{'sha256':key,'rows':len(values),'ids':values} for key,values in fingerprints.items() if len(values)>1]
    review={'schema':'natlang.acquired-content-template-review/1','rows':len(ids),'exact_token_weight_duplicate_groups':duplicates,
        'duplicate_excess_rows':sum(x['rows']-1 for x in duplicates),'role_counts':dict(roles_total),
        'unknown_role_starts':counts['unknown_role_starts'],'unique_role_structures':len(templates),
        'top_role_structures':[{'sha256':key,'rows':count} for key,count in templates.most_common(30)],
        'scope':'Exact token/raw-mask content and visible role topology within this delta; no semantic, syntax execution or predecessor-content claim.'}
    review_path=output/'content-review-summary.json';write_json_durable(review_path,review)
    if (ProtectedWorlds.load().digest!=protection.digest or bind(combined_path)['sha256']!=combined_ref['sha256']
            or bind(manifest_path)['sha256']!=combined['source']['sha256'] or Path(__file__).read_bytes()!=CODE
            or [bind(path) for path in component_paths[:-1]]!=components[:-1] or current_protection(protection_path)['proof']!=quality['proof']
            or load_decision_holds(hold_paths)[1]!=hold_refs):
        raise ValueError('census source or protection changed during indexing')
    census={'schema':'natlang.acquired-effective-learning-census/1','status':'canonical parser and effective-loss census; no admission',
        'screen':combined_ref,'ce_policy':bind(policy_path),'objective_sha256':canonical_json_sha256_hex(objective),
        'row_signals':bind(signal_path),'rows':len(ids),'counts':dict(counts),'category_mass':dict(masses),
        'source_stores':dict(stores),'unknown':dict(unknown),'training_admission':False,'content_review':bind(review_path),
        'leaf_index_proofs':proofs,'source_format_history':source_reviews,'history_profile_counts':dict(history_profiles),
        'pending_census':pending_census_ref,'code_bindings':captured_components,'ordered_graph_ids_sha256':canonical_json_sha256_hex(ids),
        'ordered_kv_slots': [geometry_by_id[i] for i in ids],
        'scope':'Full graph parser and effective CE mask; measured source labels do not verify outcomes.'}
    write_json_durable(output/'receipt.json',census);return census


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--combined',required=True);p.add_argument('--combined-sha256',required=True);p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--pending-census',required=True);p.add_argument('--pending-census-sha256',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();r=prepare({'path':a.combined,'sha256':a.combined_sha256},output=a.out,workers=a.workers,pending_census_ref={'path':a.pending_census,'sha256':a.pending_census_sha256},resume=a.resume)
    print(json.dumps({'rows':r['rows'],'counts':r['counts'],'category_mass':r['category_mass']}))
if __name__=='__main__':main()
