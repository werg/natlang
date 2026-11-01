#!/usr/bin/env python3
"""Bind staged earned static corrections to index, objective, geometry and recipe.

This stages a candidate only. It does not publish a checkpoint, certify GPU
capacity, alter current exposure or qualify model behavior.
"""
from __future__ import annotations
import argparse,copy,json,os
from pathlib import Path
from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.train import native_packed_capacity as capacity
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.own_output_supervision import identity,policy_objective,prepare_policy,continuation_identity
from natlang_neuralese.train.native_source_correction import source_correction_dependencies
from natlang_neuralese.train.recipe import load_recipe,resolve_stage_inputs,_source_correction_continuation
from scripts.neuralese_data.build_acquired_streaming_successor import (bind,checked,read,require_output_headroom,
    predecessor_mass_census,sft_capacity_descriptor,receipt_protection_digest)

def seed_unchanged_index(prep,prior,old_geo,old_rows):
    """Independently prove the filtered leaf bytes before reusing old row metadata."""
    from natlang_neuralese.common import row_index
    from natlang_neuralese.common.hashing import sha256_file_hex_cached,canonical_json_sha256_hex
    from scripts.neuralese_data.partition_acquired_context_pending import _checked as fresh_checked
    identity=old_geo['identity'];pairs=read(checked(prep['graph_pairs'],'static graph-pair receipt'))
    original=fresh_checked(pairs['source_leaf']);remaining=fresh_checked(prep['unchanged_leaf'])
    original_source=NativeGraphSource(checked(prior['source'],'prior source'),identity['tokenizer'],
                                      required_split='train',role_ids=identity['roles'])
    if (len(original_source.entries)!=len(old_rows) or any(e['id']!=r['id'] or e['binding']!=r['binding']
        for e,r in zip(original_source.entries,old_rows,strict=True))):
        raise ValueError('predecessor execution index differs from its exact family map')
    leaf=original_source.children[-1]
    if leaf.file_sha256!=pairs['source_leaf']['sha256']:raise ValueError('prior indexed final leaf differs')
    evidence={row['id']:row for row in pairs['rows']};selected=set(prep['corrected_ids']);retained=[];seen=set()
    if set(evidence)!=selected:raise ValueError('source partition evidence differs from selected rows')
    with open_jsonl_text(original) as source,open_jsonl_text(remaining) as result:
        source=iter(source);result=iter(result)
        for entry in leaf.entries:
            original_line=next((line for line in source if line.strip()),None)
            if original_line is None:raise ValueError('original leaf omitted an indexed row')
            ident=entry['id']
            if ident in selected:
                parent=json.loads(original_line)
                if parent.get('id')!=ident or canonical_json_sha256_hex(parent)!=evidence[ident]['parent_row_sha256']:
                    raise ValueError('source partition selected row differs from its parent proof')
                seen.add(ident)
            else:
                remainder_line=next((line for line in result if line.strip()),None)
                if remainder_line!=original_line:raise ValueError('unchanged source leaf differs from exact original row bytes')
                retained.append(entry)
        if (next((line for line in source if line.strip()),None) is not None
            or next((line for line in result if line.strip()),None) is not None or seen!=selected or not retained):
            raise ValueError('source partition contains extra or missing rows')
    fresh_checked(pairs['source_leaf']);fresh_checked(prep['unchanged_leaf'])
    package=Path(capacity.__file__).resolve().parents[1]
    names=('data/native_sources.py','data/native_graph_schema.py','data/decision_graph.py','model/native_graph_plan.py',
           'data/native_pair_schema.py','common/row_index.py','common/jsonio.py','common/hashing.py')
    parser=canonical_json_sha256_hex({name:sha256_file_hex_cached(package/name) for name in names})
    if parser!=identity.get('parser_policy_sha256'):raise ValueError('source parser changed; cannot seed prior execution index')
    key={'source_sha256':prep['unchanged_leaf']['sha256'],'tokenizer':identity['tokenizer'],'split':'train',
         'policy_sha256':parser,'kind':'native-graphs','pair_model':None,'role_ids':identity['roles']}
    index_path=row_index.entry_path('native-graphs',canonical_json_sha256_hex(key))
    record={'schema':'natlang.row-index/1','key':key,'entries':retained,'metadata_sha256':canonical_json_sha256_hex(retained)}
    existing=row_index.load(index_path)
    if existing is not None and existing!=record:raise ValueError('unchanged source index conflicts with exact partition proof')
    if existing is None:row_index.save(index_path,record)
    if row_index.load(index_path)!=record:raise ValueError('verified unchanged source index was not durably cached')
    return {'schema':'natlang.static-context-unchanged-index-proof/1','source':pairs['source_leaf'],
        'remainder':prep['unchanged_leaf'],'index':bind(index_path),'rows':len(retained),
        'scope':'fresh exact row-byte comparison plus authenticated prior execution index; no model execution'}

def reconcile_geometry(entries,old_rows,old_geo,ledger,corrected_ids,corrected_slots):
    """Reuse only geometry whose ordered original bindings and token counts agree."""
    old_by_id={r['id']:r for r in old_rows};new_by_id={r['id']:r for r in entries};selected=set(corrected_ids)
    if (len(old_by_id)!=len(old_rows) or len(new_by_id)!=len(entries) or set(old_by_id)!=set(new_by_id)
        or not selected or len(selected)!=len(corrected_ids) or set(ledger)!=selected or set(corrected_slots)!=selected
        or len(old_geo.get('kv_slots',[]))!=len(old_rows)):
        raise ValueError('static source, correction or geometry row inventory differs')
    expected=[r['id'] for r in old_rows if r['id'] not in selected]+corrected_ids
    if [e['id'] for e in entries]!=expected:raise ValueError('static source changed retained earned order')
    for entry in entries:
        ident=entry['id'];old=old_by_id[ident]
        if ident in selected:
            row=ledger[ident]
            if (old['binding']!=row.get('old_binding') or entry['binding']!=row.get('new_binding')
                or not entry.get('producer_consumer') or entry.get('external_state_consumption')):
                raise ValueError('static source changed its authenticated derived class or binding')
        elif old['binding']!=entry['binding']:raise ValueError('static source changed an undeclared earned binding')
    old_virtual=[{'id':r['id'],'binding':r['binding'],'forward_tokens':new_by_id[r['id']]['forward_tokens']} for r in old_rows]
    if capacity._rows_digest(old_virtual)!=old_geo.get('rows_sha256'):
        raise ValueError('predecessor geometry rows differ from exact source and binding inventory')
    old_values={row['id']:value for row,value in zip(old_rows,old_geo['kv_slots'],strict=True)}
    return old_by_id,[corrected_slots[e['id']] if e['id'] in selected else old_values[e['id']] for e in entries]


def prune_replaced_recipe_bindings(recipe, replaced):
    """Drop only replaced names that no stage still references."""
    used={name for stage in recipe['stages'] for name in (stage.get('inputs') or {}).values() if isinstance(name,str)}
    for name in replaced:
        if name not in used:recipe['input_bindings'].pop(name,None)

def finalize(preparation, predecessor, output, *, preparation_sha256=None, predecessor_sha256=None):
    prep_path=Path(preparation).resolve(strict=True);prior_path=Path(predecessor).resolve(strict=True)
    prep_binding=bind(prep_path);prior_binding=bind(prior_path)
    if ((preparation_sha256 is not None and prep_binding['sha256']!=preparation_sha256)
        or (predecessor_sha256 is not None and prior_binding['sha256']!=predecessor_sha256)):
        raise ValueError('static source finalization input receipt changed')
    prep=read(prep_path);prior=read(prior_path);output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise ValueError('finalization output must be fresh')
    require_output_headroom(output)
    if (prep.get('schema')!='natlang.earned-static-context-source-preparation/1' or prep.get('previous_source')!=prior.get('source')
        or prior.get('schema')!='natlang.acquired-streaming-source-successor/1'):
        raise ValueError('earned static source does not bind the exact predecessor')
    source_path=checked(prep['train_source'],'prepared corrected source');old_source=checked(prior['source'],'earned predecessor source')
    correction_path=checked(prep['source_correction'],'static source correction');source_correction_dependencies(bind(correction_path))
    proof=read(correction_path)
    if ((proof.get('prior_source') or {}).get('sha256')!=prior['source']['sha256']
        or (proof.get('current_source') or {}).get('sha256')!=prep['train_source']['sha256']
        or ((proof.get('static_native_context_correction') or {}).get('ce_policy') or {}).get('sha256')!=prior['ce_policy']['sha256']):
        raise ValueError('static correction does not bind prior and corrected manifests')
    protection=ProtectedWorlds.load()
    if receipt_protection_digest(prior)!=protection.digest:raise ValueError('current protected-world union changed')
    old_map=read(checked(prior['family_map'],'predecessor family map'))
    old_geo=read(checked(prior['geometry'],'predecessor geometry'))
    if (old_map.get('schema')!='natlang.logical-family-map/1' or old_map.get('source_sha256')!=prior['source']['sha256']
        or len(old_map.get('rows',[]))!=prior['rows'] or old_geo.get('schema')!='natlang.native-packed-geometry/1'
        or (old_geo.get('source') or {}).get('sha256')!=prior['source']['sha256'] or len(old_geo.get('kv_slots',[]))!=prior['rows']):
        raise ValueError('earned family map or geometry lacks exact source custody')
    index_proof=seed_unchanged_index(prep,prior,old_geo,old_map['rows'])
    ident=old_geo['identity'];source=NativeGraphSource(source_path,ident['tokenizer'],required_split='train',role_ids=ident['roles'])
    entries=source.entries;old_rows=old_map['rows']
    if len(entries)!=prior['rows']:raise ValueError('static source changed the earned graph inventory')
    selected=set(prep.get('corrected_ids') or [])
    if not selected or len(selected)!=len(prep['corrected_ids']):raise ValueError('static source correction identities are invalid')
    with open_jsonl_text(checked({'path':proof['row_bindings_path'],'sha256':proof['row_bindings_sha256']},'source correction row ledger')) as stream:
        ledger={item['id']:item for line in stream if line.strip() for item in [json.loads(line)]}
    if set(ledger)!=selected:raise ValueError('static correction ledger does not cover exact rows')
    pairs=read(checked(prep['graph_pairs'],'static graph-pair receipt'));corrected_slots={}
    with open_jsonl_text(checked(pairs['graph_pairs'],'static graph pairs')) as stream:
        for line in stream:
            if line.strip():
                child=json.loads(line)['child'];example=parse_graph(child,tokenizer_fingerprint=ident['tokenizer'])
                if child['id'] in corrected_slots:raise ValueError('static correction geometry repeats a row')
                corrected_slots[child['id']]=len(example.plan.slot_sources)
    old_by_id,values=reconcile_geometry(entries,old_rows,old_geo,ledger,prep['corrected_ids'],corrected_slots)
    package=Path(capacity.__file__).resolve().parents[1]
    names=('data/native_sources.py','data/native_graph_schema.py','data/decision_graph.py','model/native_graph_plan.py',
           'data/native_pair_schema.py','common/row_index.py','common/jsonio.py','common/hashing.py')
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    current_parser=canonical_json_sha256_hex({name:sha256_file_hex_cached(package/name) for name in names})
    if ident.get('parser_policy_sha256')!=current_parser:raise ValueError('source parser changed; migrate geometry before reuse')
    output.mkdir()
    family_path=output/'families.json';write_json_durable(family_path,{'schema':'natlang.logical-family-map/1',
        'source_sha256':prep['train_source']['sha256'],'rows':[{'id':e['id'],'binding':e['binding'],'family':old_by_id[e['id']]['family']} for e in entries]})
    geometry_path=output/'geometry.json';write_json_durable(geometry_path,{'schema':'natlang.native-packed-geometry/1','identity':ident,
        'source':{'path':str(source_path),'sha256':prep['train_source']['sha256']},'rows_sha256':capacity._rows_digest(entries),'kv_slots':values})
    old_policy=read(checked(prior['ce_policy'],'predecessor CE policy'));old_objective=policy_objective(old_policy)
    policy=copy.deepcopy(old_policy);old_leaf=pairs['source_leaf']['sha256'];replacement=[prep['unchanged_leaf'],prep['derived_leaf']]
    source_defs=[x for x in policy['sources'] if x.get('sha256')==old_leaf]
    if len(source_defs)!=1:raise ValueError('predecessor CE source declaration is ambiguous')
    policy['sources']=[x for x in policy['sources'] if x.get('sha256')!=old_leaf]+[{**source_defs[0],'sha256':x['sha256']} for x in replacement]
    policy['sources'].sort(key=lambda x:x['sha256'])
    active=policy['source_contract']['active_sources'];removed=[x for x in active if x.get('sha256')==old_leaf]
    if len(removed)!=1:raise ValueError('predecessor CE source inventory is ambiguous')
    policy['source_contract']['active_sources']=[x for x in active if x.get('sha256')!=old_leaf]+[{k:x[k] for k in ('path','sha256','bytes')} for x in replacement]
    policy['source_contract']['active_source_count']=len(policy['source_contract']['active_sources'])
    policy['source_contract']['inventory']=[bind(source_path)];policy['inventory']=[{k:v for k,v in bind(source_path).items() if k!='bytes'}]
    policy['objective']['family_balance']['map']={k:v for k,v in bind(family_path).items() if k!='bytes'}
    policy['objective_transition']={'from':identity(old_objective)['sha256']}
    policy_path=output/'ce-policy.json';write_json_durable(policy_path,policy)
    prepared_policy=prepare_policy(load_policy(bind(policy_path)),source)
    _,transition=continuation_identity({'supervision_objective':identity(old_objective)},prepared_policy)
    previous_census,prior_custody=predecessor_mass_census(prior,prior_path,old_objective)
    unscaled=copy.deepcopy(policy);unscaled['objective']['family_balance']=None
    unscaled_path=output/'unscaled-ce-policy.json';write_json_durable(unscaled_path,unscaled)
    mass_path=output/'objective-census.json';write_json_durable(mass_path,{**previous_census,
        'manifest':bind(source_path),'policy':bind(unscaled_path),'rows':len(entries),
        'static_correction':bind(correction_path),'predecessor_mass_custody':prior_custody,
        'scope':'prior measured mass retained exactly; every changed graph has independently replayed effective supervision'})
    mass_receipt=output/'objective-census-receipt.json';write_json_durable(mass_receipt,{'schema':'natlang.measured-mass-append-receipt/1',
        'census':bind(mass_path),'map':bind(family_path),'builder':bind(Path(__file__))})
    descriptor=output/'sft-capacity-descriptor.json';write_json_durable(descriptor,sft_capacity_descriptor(bind(source_path),
        {k:v for k,v in bind(geometry_path).items() if k!='bytes'}))
    old_recipe_path=checked(prior['recipe'],'predecessor recipe');recipe=read(old_recipe_path)
    stage=next(s for s in recipe['stages'] if s['id']==proof['stage_id']);inputs=stage['inputs'];bindings=recipe['input_bindings']
    replaced={inputs.get(role) for role in ('native_graphs','foundation_native_graphs','continued_native_graphs',
                                            'source_correction','ce_supervision_policy')}
    def assign(role,name,ref):bindings[name]={k:v for k,v in ref.items() if k!='bytes'};inputs[role]=name
    assign('native_graphs','static_context_train',bind(source_path));assign('foundation_native_graphs','static_context_train',bind(source_path))
    assign('continued_native_graphs','static_context_previous',bind(old_source));assign('source_correction','static_context_correction',bind(correction_path))
    assign('ce_supervision_policy','static_context_ce_policy',bind(policy_path))
    prune_replaced_recipe_bindings(recipe,replaced)
    stage['parameters']['sft_packed_geometry']={k:v for k,v in bind(geometry_path).items() if k!='bytes'}
    stage['parameters']['memory_qualification']={k:v for k,v in bind(descriptor).items() if k!='bytes'}
    recipe_path=output/'recipe-continuation.json';write_json_durable(recipe_path,recipe)
    old_recipe=load_recipe(old_recipe_path);new_recipe=load_recipe(recipe_path)
    find=lambda r:next(s for s in r['stages'] if s['id']==proof['stage_id'])
    old_roles=resolve_stage_inputs(old_recipe,find(old_recipe),{});new_roles=resolve_stage_inputs(new_recipe,find(new_recipe),{})
    if not _source_correction_continuation(old_roles,new_roles):raise ValueError('static corrected recipe lacks its exact authenticated source/CE transition')
    if (ProtectedWorlds.load().digest!=protection.digest or bind(prep_path)!=prep_binding or bind(prior_path)!=prior_binding):
        raise ValueError('protected-world union or input receipt changed during static source finalization')
    result={'schema':'natlang.acquired-streaming-source-successor/1','status':'static source-ce-recipe-geometry prepared; no training admission',
        'training_admission':False,'predecessor':bind(prior_path),'source':bind(source_path),'rows':len(entries),
        'predecessor_rows':len(entries),'new_rows':0,'family_map':bind(family_path),'ce_policy':bind(policy_path),
        'recipe':bind(recipe_path),'geometry':bind(geometry_path),'capacity_descriptor':bind(descriptor),
        'objective_census':bind(mass_path),'objective_census_receipt':bind(mass_receipt),
        'objective_transition':transition,'family_balance':prior['family_balance'],
        'family_census':prepared_policy['_family_census'],'current_protection_digest':protection.digest,
        'source_correction':bind(correction_path),'static_context_preparation':bind(prep_path),
        'unchanged_source_index_proof':index_proof,
        'limitations':['Current full-checkpoint preflight and actual GPU capacity qualification are still required.',
            'Graph edges are static derived training structure, not observed teacher invocations.']}
    write_json_durable(output/'receipt.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--preparation-sha256',required=True)
    p.add_argument('--predecessor',type=Path,required=True);p.add_argument('--predecessor-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(finalize(a.preparation,a.predecessor,a.output,
        preparation_sha256=a.preparation_sha256,predecessor_sha256=a.predecessor_sha256),sort_keys=True))
if __name__=='__main__':main()
