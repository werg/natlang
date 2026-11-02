#!/usr/bin/env python3
"""Bind a direct acquired CE continuation to actual remaining SFT quotas."""
import argparse,copy,hashlib,math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from scripts.neuralese_data import build_acquired_streaming_successor as b
from natlang_neuralese.common.jsonio import write_json_durable,write_bytes_durable
from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.own_output_supervision import policy_objective,identity,prepare_policy,continuation_identity
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.recipe import load_recipe,resolve_stage_inputs,_data_continuation

MODULE=Path(__file__).read_bytes()

def quota_aware_plan(parent_mass,targets,adjustments,*,cap):
    remaining=dict(parent_mass)
    for family,delta in adjustments.items():remaining[family]+=delta
    if set(targets)!=set(remaining) or any(not math.isfinite(v) or v<=0 for v in remaining.values()):raise ValueError('remaining family mass is invalid')
    scales,capped=b.capped_scales(targets,remaining,cap);weighted=sum(remaining[k]*scales[k] for k in remaining);total=sum(remaining.values())
    if not math.isclose(weighted,total,rel_tol=1e-12,abs_tol=1e-6):raise ValueError('quota-aware scales do not conserve mean-one remaining mass')
    return {'scales':scales,'unscaled_remaining_mass':remaining,'target_shares':targets,'effective_remaining_shares':{k:remaining[k]*scales[k]/weighted for k in remaining},'total_unscaled_remaining_mass':total,'total_scaled_remaining_mass':weighted,'capped_families':capped,'max_scale':cap}

def sig(path):
 s=Path(path).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def build(checkpoint,proof_path,parent_path,prepared_path,config_path,visited_path,output):
 checkpoint=Path(checkpoint).resolve(strict=True);proof_path=Path(proof_path).resolve(strict=True);parent_path=Path(parent_path).resolve(strict=True);prepared_path=Path(prepared_path).resolve(strict=True);config_path=Path(config_path).resolve(strict=True);visited_path=Path(visited_path).resolve(strict=True)
 proof,parent,prepared,config,visited=map(b.read,(proof_path,parent_path,prepared_path,config_path,visited_path));before=sig(checkpoint)
 if (Path(proof['checkpoint']['path']).resolve()!=checkpoint or proof['successor_recipe']['sha256']!=parent['recipe']['sha256'] or prepared['predecessor']!=b.bind(parent_path)
  or visited.get('schema')!='natlang.sft-visited-row-objective-census/1' or visited['checkpoint']!=proof['checkpoint'] or visited['parent']!=b.bind(parent_path) or visited['successor']!=b.bind(prepared_path)):
    raise ValueError('quota-aware parent or measured row custody differs')
 protection=ProtectedWorlds.load().digest
 if any(x.get('current_protection_digest')!=protection for x in (parent,prepared)):raise ValueError('protection union differs')
 params=config['trainer_parameters'];epoch=params['epochs'];visits=params['communication_min_visits']
 if epoch!=1.0 or type(visits) is not int or visits<1:raise ValueError('unexpected quota geometry')
 state=CheckpointPolicy.load(checkpoint);frame=state['training']['sft_progress'][config['production_stage']];credits=frame['exposure_counts'];quotas=frame['required_visits']
 oldmap=b.read(b.checked(parent['family_map'],'parent family map'));newmap=b.read(b.checked(prepared['family_map'],'successor family map'));oldrows=oldmap['rows'];newrows=newmap['rows']
 if oldrows!=newrows[:len(oldrows)] or len(oldrows)!=parent['rows'] or len(newrows)!=prepared['rows'] or frame['source_row_order']!=[r['id'] for r in oldrows] or any(frame['source_row_bindings'].get(r['id'])!=r['binding'] for r in oldrows):raise ValueError('checkpoint rows differ from exact source prefix')
 if oldmap['source_sha256']!=parent['source']['sha256'] or newmap['source_sha256']!=prepared['source']['sha256']:raise ValueError('family map source differs')
 policy=b.read(b.checked(prepared['ce_policy'],'prepared policy'));prior=b.read(b.checked(parent['ce_policy'],'actual parent policy'));objective=policy_objective(prior)
 if identity(objective)!=parent['objective_transition']['to'] or policy['objective_transition']['from']!=identity(objective)['sha256']:raise ValueError('CE transition does not start at actual parent')
 old_unscaled=copy.deepcopy(objective);new_unscaled=copy.deepcopy(policy_objective(policy));old_unscaled['family_balance']=new_unscaled['family_balance']=None
 if old_unscaled!=new_unscaled:raise ValueError('base loss law changed')
 census,_=b.predecessor_mass_census(prepared,prepared_path,policy_objective(policy));row_mass={r['id']:r for r in visited['rows']}
 if len(row_mass)!=visited['selected_rows'] or len(row_mass)!=len(visited['rows']):raise ValueError('visited census has duplicate rows')
 baseline={name:row['unscaled_mass']*(visits if name=='native-communication' else 1) for name,row in census['families'].items()};adjustments=Counter();required=set();earned=Counter()
 for row in oldrows:
  ident=row['id'];family=row['family'];quota=int(quotas[ident]);credit=int(credits.get(ident,0));default=visits if family=='native-communication' else 1
  if credit<0 or quota<0 or credit>quota:raise ValueError('invalid saved quota/credit')
  if credit or quota!=default:
   required.add(ident);item=row_mass.get(ident)
   if not item or item['binding']!=row['binding'] or item['family']!=family or item['earned_visits']!=credit or item['saved_required_visits']!=quota:raise ValueError('nonuniform row lacks exact mass census')
   mass=item['unscaled_effective_token_mass'];adjustments[family]+=(max(quota,default)-default-credit)*mass;earned[family]+=credit*mass
 if required!=set(row_mass):raise ValueError('visited census is not exact required inventory')
 targets=prepared['family_balance']['target_shares'];plan=quota_aware_plan(baseline,targets,adjustments,cap=objective['family_balance']['max_scale'])
 out=Path(output).resolve();b.require_output_headroom(out)
 if out.exists():raise ValueError('fresh output required')
 out.mkdir();builder=out/('builder-'+hashlib.sha256(MODULE).hexdigest()+'.py');write_bytes_durable(builder,MODULE)
 policy['objective']['family_balance']['scales']=plan['scales']
 from natlang_neuralese.train.native_quota_aware_ce import make_basis
 policy['quota_aware_basis']=make_basis(frame,stage_id=config['production_stage'],source_sha256=parent['source']['sha256'],candidate_sha256=prepared['source']['sha256'],epochs=epoch,communication_min_visits=visits)
 ce=out/'ce-policy.json';write_json_durable(ce,policy)
 geometry=b.read(b.checked(prepared['geometry'],'prepared geometry'));g=geometry['identity'];source=NativeGraphSource(prepared['source']['path'],g['tokenizer'],required_split='train',role_ids=g['roles'])
 if len(source.entries)!=len(newrows) or any(e['id']!=r['id'] or e['binding']!=r['binding'] for e,r in zip(source.entries,newrows,strict=True)):raise ValueError('candidate source index differs')
 prepared_policy=prepare_policy(load_policy(b.bind(ce)),source)
 recipe=b.read(b.checked(parent['recipe'],'parent recipe'));stage=b.bind_successor_recipe_inputs(recipe,parent['source'],prepared['source'],b.bind(ce));stage['parameters']['sft_packed_geometry']=prepared['geometry'];stage['parameters']['memory_qualification']={k:v for k,v in prepared['capacity_descriptor'].items() if k!='bytes'}
 recipe_path=out/'recipe-continuation.json';write_json_durable(recipe_path,recipe)
 old,new=map(load_recipe,(parent['recipe']['path'],recipe_path));getstage=lambda r:next(s for s in r['stages'] if s['id']==config['production_stage'])
 old_roles,new_roles=resolve_stage_inputs(old,getstage(old),{}),resolve_stage_inputs(new,getstage(new),{})
 if not _data_continuation({k:v for k,v in old_roles.items() if k!='ce_supervision_policy'},{k:v for k,v in new_roles.items() if k!='ce_supervision_policy'}) or old_roles.get('held_native_graphs')!=new_roles.get('held_native_graphs'):raise ValueError('recipe source or held continuation rejected')
 old_source=SimpleNamespace(file_sha256=parent['source']['sha256'],entries=source.entries[:parent['rows']]);extension=extension_policy(old_roles,new_roles,prior_train=old_source,candidate_train=source)
 _,transition=continuation_identity({'supervision_objective':parent['objective_transition']['to']},extension)
 if ProtectedWorlds.load().digest!=protection or sig(checkpoint)!=before or Path(__file__).read_bytes()!=MODULE:raise ValueError('custody changed during objective preparation')
 result={'schema':b.SCHEMA,'status':'quota-aware-source-ce-recipe-geometry-prepared-not-published','training_admission':False,'predecessor':b.bind(parent_path),'predecessor_source':parent['source'],'source':prepared['source'],'rows':prepared['rows'],'predecessor_rows':parent['rows'],'new_rows':prepared['rows']-parent['rows'],'new_family':'general-agent','family_map':prepared['family_map'],'ce_policy':b.bind(ce),'recipe':b.bind(recipe_path),'objective_census':prepared['objective_census'],'objective_census_receipt':prepared['objective_census_receipt'],'geometry':prepared['geometry'],'capacity_descriptor':prepared['capacity_descriptor'],'family_balance':plan,'family_census':prepared_policy['_family_census'],'objective_transition':transition,'current_protection_digest':protection,'quota_basis':{'checkpoint':proof['checkpoint'],'checkpoint_proof':b.bind(proof_path),'saved_stage':config['production_stage'],'config':b.bind(config_path),'epochs':epoch,'communication_min_visits':visits,'visited_row_census':b.bind(visited_path),'unscaled_prior_earned_token_mass':dict(earned),'scope':'planned remaining token-loss coefficient mass only; saved credits/quotas retained'},'pending_render_obligations':prepared['pending_render_obligations'],'carried_over_context_obligations':prepared['carried_over_context_obligations'],'roles':{k:new_roles.get(k) for k in ('native_graphs','foundation_native_graphs','held_native_graphs')},'builder':b.bind(builder),'limitations':['No exposure, optimizer state or stage completion changed by this preparation.','Decision-readout mass is not included in the recorded aggregate token census; realized gradients, scheduling and optimizer effects are not predicted.']}
 write_json_durable(out/'receipt.json',result);print(plan)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--proof',required=True);p.add_argument('--parent',required=True);p.add_argument('--prepared',required=True);p.add_argument('--config',required=True);p.add_argument('--visited',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.checkpoint,a.proof,a.parent,a.prepared,a.config,a.visited,a.output)
