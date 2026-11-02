#!/usr/bin/env python3
"""Rebind a reviewed two-append preparation directly to its earned predecessor."""
import argparse, copy, hashlib, math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from scripts.neuralese_data import build_acquired_streaming_successor as b
from natlang_neuralese.train.own_output_supervision import policy_objective, identity, continuation_identity, prepare_policy
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs, _data_continuation
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.common.jsonio import write_json_durable, write_bytes_durable
from natlang_neuralese.common.custody import ProtectedWorlds

BUILDER_BYTES = Path(__file__).read_bytes()


def build(parent_path, middle_path, final_path, out):
    parent_path, middle_path, final_path = [Path(p).resolve(strict=True) for p in (parent_path,middle_path,final_path)]
    parent,middle,final = map(b.read,(parent_path,middle_path,final_path))
    inputs = {str(p):b.bind(p) for p in (parent_path,middle_path,final_path)}
    protection = ProtectedWorlds.load().digest
    if any(item.get('current_protection_digest') != protection for item in (parent,middle,final)):
        raise ValueError('current protection union differs from reviewed sources')
    if (middle['predecessor'] != b.bind(parent_path) or final['predecessor'] != b.bind(middle_path)
            or middle['predecessor_rows'] != parent['rows'] or final['predecessor_rows'] != middle['rows']
            or middle['new_family'] != final['new_family']):
        raise ValueError('prepared append chain does not bind exact parent and family')
    for item in (parent,middle,final):
        for key in ('source','family_map','ce_policy','recipe','objective_census','objective_census_receipt','geometry'):
            b.checked(item[key],key)
    parent_policy=b.read(parent['ce_policy']['path']); middle_policy=b.read(middle['ce_policy']['path']); final_policy=b.read(final['ce_policy']['path'])
    objectives=[policy_objective(p) for p in (parent_policy,middle_policy,final_policy)]
    unscaled=[]
    for obj in objectives:
        obj=copy.deepcopy(obj);obj['family_balance']=None;unscaled.append(obj)
    if not unscaled[0]==unscaled[1]==unscaled[2] or identity(objectives[0])!=parent['objective_transition']['to']:
        raise ValueError('prepared censuses changed unscaled CE law or parent identity')
    censuses=[]; custody=[]
    for item,path,obj in zip((parent,middle,final),(parent_path,middle_path,final_path),objectives,strict=True):
        census,proof=b.predecessor_mass_census(item,path,obj);censuses.append(census);custody.append(proof)
    family=middle['new_family']; mass=censuses[2]['families'][family]['unscaled_mass']-censuses[0]['families'][family]['unscaled_mass']
    for key in censuses[0]['families']:
        if key!=family and any(censuses[0]['families'][key][field]!=censuses[2]['families'][key][field] for field in ('rows','unscaled_mass')):raise ValueError('append changed an existing family mass')
    for item,census in zip((parent,middle,final),censuses,strict=True):
        if census['rows']!=item['rows']:raise ValueError('census row count differs')
    balance=objectives[0]['family_balance']; plan=b.scales_for_append(censuses[0],balance['scales'],mass,share=.07,cap=balance['max_scale'],family=family)
    for name,row in censuses[2]['families'].items():
        if not math.isclose(row['unscaled_mass'],plan['unscaled_mass'][name],rel_tol=1e-10,abs_tol=1e-6):raise ValueError('composed mass differs')
    maps=[b.read(x['family_map']['path']) for x in (parent,middle,final)]
    if (maps[0]['rows']!=maps[1]['rows'][:parent['rows']] or maps[1]['rows']!=maps[2]['rows'][:middle['rows']]
        or any(m['source_sha256']!=x['source']['sha256'] or len(m['rows'])!=x['rows'] for m,x in zip(maps,(parent,middle,final),strict=True))):
        raise ValueError('prepared maps changed an existing ordered row binding')
    out=Path(out).resolve();b.require_output_headroom(out)
    if out.exists():raise ValueError('output already exists')
    out.mkdir()
    builder_path=out/('builder-'+hashlib.sha256(BUILDER_BYTES).hexdigest()+'.py');write_bytes_durable(builder_path,BUILDER_BYTES)
    geometry=b.read(final['geometry']['path']); geom_id=geometry['identity']
    source=NativeGraphSource(final['source']['path'],geom_id['tokenizer'],required_split='train',role_ids=geom_id['roles'])
    if len(source.entries)!=final['rows'] or [(e['id'],e['binding']) for e in source.entries]!=[(e['id'],e['binding']) for e in maps[2]['rows']]:
        raise ValueError('source index differs from complete ordered family map')
    policy=copy.deepcopy(final_policy);policy['objective']['family_balance']['scales']=plan['scales'];policy['objective_transition']={'from':identity(objectives[0])['sha256']}
    ce=out/'ce-policy.json';write_json_durable(ce,policy)
    prepared=prepare_policy(load_policy(b.bind(ce)),source)
    unscaled_policy=copy.deepcopy(policy);unscaled_policy['objective']['family_balance']=None
    unscaled_path=out/'unscaled-ce-policy.json';write_json_durable(unscaled_path,unscaled_policy)
    census_path=out/'objective-census.json';write_json_durable(census_path,{'schema':'natlang.own-output-mass-census/1','manifest':final['source'],'policy':b.bind(unscaled_path),'rows':final['rows'],'objective_without_family_scales':True,'families':censuses[2]['families'],'unscaled_mass_total':censuses[2]['unscaled_mass_total'],'predecessor_mass_custody':custody[0],'composed_preparation_custody':custody[1:],'scope':'Measured parent and two reviewed append censuses, with unchanged unscaled objective.'})
    census_receipt=out/'objective-census-receipt.json';write_json_durable(census_receipt,{'schema':'natlang.measured-mass-append-receipt/1','census':b.bind(census_path),'map':final['family_map'],'builder':b.bind(builder_path)})
    descriptor=out/'sft-capacity-descriptor.json';write_json_durable(descriptor,b.sft_capacity_descriptor(final['source'],final['geometry']))
    recipe=b.read(parent['recipe']['path']);stage=b.bind_successor_recipe_inputs(recipe,parent['source'],final['source'],b.bind(ce));stage['parameters']['sft_packed_geometry']=final['geometry'];stage['parameters']['memory_qualification']={k:v for k,v in b.bind(descriptor).items() if k!='bytes'}
    recipe_path=out/'recipe-continuation.json';write_json_durable(recipe_path,recipe)
    old,new=map(load_recipe,(parent['recipe']['path'],recipe_path)); getstage=lambda r:next(s for s in r['stages'] if s['id']=='native_joint_sft_v4')
    old_roles,new_roles=resolve_stage_inputs(old,getstage(old),{}),resolve_stage_inputs(new,getstage(new),{})
    if not _data_continuation({k:v for k,v in old_roles.items() if k!='ce_supervision_policy'},{k:v for k,v in new_roles.items() if k!='ce_supervision_policy'}) or old_roles.get('held_native_graphs')!=new_roles.get('held_native_graphs'):
        raise ValueError('direct recipe continuation or held custody rejected')
    prior=SimpleNamespace(file_sha256=parent['source']['sha256'],entries=source.entries[:parent['rows']])
    extension=extension_policy(old_roles,new_roles,prior_train=prior,candidate_train=source)
    _,transition=continuation_identity({'supervision_objective':parent['objective_transition']['to']},extension)
    result={'schema':b.SCHEMA,'status':'source-ce-recipe-geometry-qualified-not-published','training_admission':False,'predecessor':b.bind(parent_path),'predecessor_source':parent['source'],'source':final['source'],'rows':final['rows'],'predecessor_rows':parent['rows'],'new_rows':final['rows']-parent['rows'],'new_family':family,'prepared_chain':[b.bind(middle_path),b.bind(final_path)],'family_map':final['family_map'],'ce_policy':b.bind(ce),'recipe':b.bind(recipe_path),'objective_census':b.bind(census_path),'objective_census_receipt':b.bind(census_receipt),'geometry':final['geometry'],'capacity_descriptor':b.bind(descriptor),'family_balance':plan,'family_census':prepared['_family_census'],'objective_transition':transition,'pending_render_obligations':final['pending_render_obligations'],'carried_over_context_obligations':final['carried_over_context_obligations'],'roles':{k:new_roles.get(k) for k in ('native_graphs','foundation_native_graphs','held_native_graphs')},'builder':b.bind(builder_path),'limitations':['Prepared only; no source adoption, SGD or behavioral qualification.','Intermediate sources and policies are preparation history, not earned training.']}
    if ProtectedWorlds.load().digest != protection or any(b.bind(path)!=ref for path,ref in inputs.items()):
        raise ValueError('source receipts or protection union changed during preparation')
    result['current_protection_digest']=protection
    write_json_durable(out/'receipt.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--middle',required=True);p.add_argument('--final',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.parent,a.middle,a.final,a.output)
