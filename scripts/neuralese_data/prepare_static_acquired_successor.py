#!/usr/bin/env python3
"""Prepare a checked append of graphified pending acquired context obligations."""
import argparse, copy, hashlib, json, math
from pathlib import Path
from types import SimpleNamespace
from scripts.neuralese_data import build_acquired_streaming_successor as b
from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.jsonio import write_json_durable, write_bytes_durable
from natlang_neuralese.train.native_packed_capacity import prepare_source
from natlang_neuralese.train.own_output_supervision import policy_objective, identity, prepare_policy, continuation_identity
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs, _data_continuation

BUILDER=Path(__file__).read_bytes()

def build(parent_path, graph_receipt_path, output):
    parent_path,graph_receipt_path=map(lambda x:Path(x).resolve(strict=True),(parent_path,graph_receipt_path))
    parent,derivation=map(b.read,(parent_path,graph_receipt_path)); parent_ref=b.bind(parent_path);derivation_ref=b.bind(graph_receipt_path)
    if (derivation.get('schema')!='natlang.acquired-pending-static-context/1' or derivation.get('training_admission') is not False
            or derivation['parent']!=parent_ref or derivation['parent_source']!=parent['source'] or derivation['objective']!=parent['ce_policy']):
        raise ValueError('static derivation does not bind this exact parent')
    protection=ProtectedWorlds.load().digest
    if derivation['protected_worlds_sha256']!=protection or parent['current_protection_digest']!=protection:raise ValueError('protection union differs')
    for ref in (parent['source'],parent['family_map'],parent['ce_policy'],parent['recipe'],parent['geometry'],derivation['graph'],derivation['row_signals'],derivation['partition'],derivation['builder']):b.checked(ref,'source custody')
    graph_path=Path(derivation['graph']['path']).resolve(strict=True);old_map=b.read(parent['family_map']['path']);old_policy=b.read(parent['ce_policy']['path']);objective=policy_objective(old_policy)
    prior_census,custody=b.predecessor_mass_census(parent,parent_path,objective)
    if identity(objective)!=parent['objective_transition']['to'] or len(old_map['rows'])!=parent['rows'] or old_map['source_sha256']!=parent['source']['sha256']:
        raise ValueError('parent objective or family inventory differs')
    total=sum(derivation['category_mass'].values());balance=objective['family_balance'];family='general-agent'
    # A tiny pending append must not silently retarget the existing objective.
    # Preserve its measured effective proportions subject to the same cap and
    # mean-one normalization; the added mass is included in its real family.
    prior_mass={name:item['unscaled_mass'] for name,item in prior_census['families'].items()}
    previous={name:prior_mass[name]*balance['scales'][name] for name in prior_mass}
    previous_total=sum(previous.values());targets={name:value/previous_total for name,value in previous.items()}
    mass=dict(prior_mass);mass[family]+=total
    scales,capped=b.capped_scales(targets,mass,balance['max_scale'])
    weighted=sum(mass[name]*scales[name] for name in mass);unscaled=sum(mass.values())
    if not math.isclose(weighted,unscaled,rel_tol=1e-12,abs_tol=1e-6):raise ValueError('family scales do not conserve mean-one mass')
    plan={'scales':scales,'unscaled_mass':mass,'target_shares':targets,'effective_shares':{name:mass[name]*scales[name]/weighted for name in mass},'capped_families':capped,'total_unscaled_mass':unscaled,'total_effective_mass':weighted,'previous_effective_mass':previous,'max_scale':balance['max_scale'],'scope':'preserve actual predecessor effective family proportions subject to cap'}
    out=Path(output).resolve();b.require_output_headroom(out)
    if out.exists():raise ValueError('fresh output required')
    out.mkdir();builder=out/('builder-'+hashlib.sha256(BUILDER).hexdigest()+'.py');write_bytes_durable(builder,BUILDER)
    manifest=out/'train-sources.json';write_json_durable(manifest,b.appended_manifest(parent['source']['path'],graph_path,derivation['graph']))
    geometry=b.read(parent['geometry']['path']);config=geometry['identity']
    source,slots,geometry_ref,scan=prepare_source(manifest,config['tokenizer'],config['roles'],out,prior=parent['geometry'])
    if len(source.entries)!=parent['rows']+derivation['rows'] or [(e['id'],e['binding']) for e in source.entries[:parent['rows']]]!=[(e['id'],e['binding']) for e in old_map['rows']]:
        raise ValueError('source index changed exact parent prefix')
    proofs=derivation['proofs'];additions=[]
    for entry,proof in zip(source.entries[parent['rows']:],proofs,strict=True):
        if entry['id']!=proof['parent_id'] or not entry['communication'] or not entry['producer_consumer'] or entry.get('external_state_consumption'):
            raise ValueError('indexed native graph differs from derivation')
        additions.append({'id':entry['id'],'binding':entry['binding'],'family':family})
    family_path=out/'families.json';write_json_durable(family_path,{'schema':'natlang.logical-family-map/1','source_sha256':b.bind(manifest)['sha256'],'rows':old_map['rows']+additions})
    policy=copy.deepcopy(old_policy);policy['inventory']=[{k:v for k,v in b.bind(manifest).items() if k!='bytes'}];policy['objective']['family_balance']['map']={k:v for k,v in b.bind(family_path).items() if k!='bytes'};policy['objective']['family_balance']['scales']=plan['scales'];policy['objective_transition']={'from':identity(objective)['sha256']}
    b.append_source_policy(policy,graph_path,derivation['graph'],None,manifest)
    ce_path=out/'ce-policy.json';write_json_durable(ce_path,policy);prepared=prepare_policy(load_policy(b.bind(ce_path)),source)
    unscaled=copy.deepcopy(policy);unscaled['objective']['family_balance']=None;unscaled_path=out/'unscaled-ce-policy.json';write_json_durable(unscaled_path,unscaled)
    families=copy.deepcopy(prior_census['families']);families[family]={'rows':families[family]['rows']+len(additions),'unscaled_mass':families[family]['unscaled_mass']+total}
    census_path=out/'objective-census.json';write_json_durable(census_path,{'schema':'natlang.own-output-mass-census/1','manifest':b.bind(manifest),'policy':b.bind(unscaled_path),'rows':len(source.entries),'objective_without_family_scales':True,'families':families,'unscaled_mass_total':plan['total_unscaled_mass'],'predecessor_mass_custody':custody,'delta_census':derivation_ref,'scope':'Measured parent plus individually projected and verified pending rows.'})
    census_receipt=out/'objective-census-receipt.json';write_json_durable(census_receipt,{'schema':'natlang.measured-mass-append-receipt/1','census':b.bind(census_path),'map':b.bind(family_path),'builder':b.bind(builder)})
    descriptor=out/'sft-capacity-descriptor.json';write_json_durable(descriptor,b.sft_capacity_descriptor(b.bind(manifest),geometry_ref))
    recipe=b.read(parent['recipe']['path']);stage=b.bind_successor_recipe_inputs(recipe,parent['source'],b.bind(manifest),b.bind(ce_path));stage['parameters']['sft_packed_geometry']=geometry_ref;stage['parameters']['memory_qualification']={k:v for k,v in b.bind(descriptor).items() if k!='bytes'}
    recipe_path=out/'recipe-continuation.json';write_json_durable(recipe_path,recipe)
    old,new=map(load_recipe,(parent['recipe']['path'],recipe_path));getstage=lambda r:next(x for x in r['stages'] if x['id']=='native_joint_sft_v4')
    old_roles,new_roles=resolve_stage_inputs(old,getstage(old),{}),resolve_stage_inputs(new,getstage(new),{})
    if not _data_continuation({k:v for k,v in old_roles.items() if k!='ce_supervision_policy'},{k:v for k,v in new_roles.items() if k!='ce_supervision_policy'}) or old_roles.get('held_native_graphs')!=new_roles.get('held_native_graphs'):
        raise ValueError('source continuation or held role differs')
    previous=SimpleNamespace(file_sha256=parent['source']['sha256'],entries=source.entries[:parent['rows']]);extension=extension_policy(old_roles,new_roles,prior_train=previous,candidate_train=source)
    _,transition=continuation_identity({'supervision_objective':parent['objective_transition']['to']},extension)
    if ProtectedWorlds.load().digest!=protection or b.bind(parent_path)!=parent_ref or b.bind(graph_receipt_path)!=derivation_ref:raise ValueError('custody changed while preparing successor')
    result={'schema':b.SCHEMA,'status':'source-ce-recipe-geometry-prepared-capacity-pending-not-published','training_admission':False,'predecessor':parent_ref,'predecessor_source':parent['source'],'source':b.bind(manifest),'appended_graphs':derivation['graph'],'static_context_derivation':derivation_ref,'rows':len(source.entries),'predecessor_rows':parent['rows'],'new_rows':len(additions),'new_family':family,'family_map':b.bind(family_path),'ce_policy':b.bind(ce_path),'recipe':b.bind(recipe_path),'geometry':geometry_ref,'geometry_scan':scan,'capacity_descriptor':b.bind(descriptor),'training_capacity_qualified':False,'new_geometry':{'max_forward_tokens':max(e['forward_tokens'] for e in source.entries[parent['rows']:]),'max_kv_slots':max(slots[parent['rows']:])},'objective_census':b.bind(census_path),'objective_census_receipt':b.bind(census_receipt),'objective_transition':transition,'family_balance':plan,'family_census':prepared['_family_census'],'category_mass':derivation['category_mass'],'pending_render_obligations':parent['pending_render_obligations'],'resolved_over_context_obligations':derivation['partition'],'current_protection_digest':protection,'roles':{k:new_roles.get(k) for k in ('native_graphs','foundation_native_graphs','held_native_graphs')},'builder':b.bind(builder),'limitations':['Preparation only; no source adoption, SGD or behavioral qualification.','Static native edges are derived training structure, not observed teacher execution.','D16/D18 repeated source occurrences retain their distinct original identities.']}
    write_json_durable(out/'receipt.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--derivation',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.parent,a.derivation,a.output)
