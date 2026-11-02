#!/usr/bin/env python3
"""Audit earned SFT row quotas and family coefficient exposure, not gradients."""
import argparse, collections, json, math, os
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


def bind(path):
 p=Path(path).resolve(strict=True);return {'path':str(p),'sha256':sha256_file_hex(p),'bytes':p.stat().st_size}

def read_ref(ref):
 actual=bind(ref['path'])
 if actual['sha256']!=ref['sha256'] or 'bytes' in ref and ref['bytes']!=actual['bytes']:raise ValueError('audit input changed')
 return json.loads(Path(actual['path']).read_bytes())

def sig(path):
 s=Path(path).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def family_stats(rows,credits,quotas):
 result={}
 for row in rows:
  family=row['family'];ident=row['id'];credit=int(credits.get(ident,0));quota=int(quotas[ident]);item=result.setdefault(family,{'rows':0,'quota_histogram':collections.Counter(),'credit_histogram':collections.Counter(),'owed_graph_visits':0,'earned_graph_visits':0})
  if credit<0 or quota<0:raise ValueError('negative quota or exposure')
  item['rows']+=1;item['quota_histogram'][quota]+=1;item['credit_histogram'][credit]+=1;item['earned_graph_visits']+=credit;item['owed_graph_visits']+=max(0,quota-credit)
 for item in result.values():
  item['quota_histogram']=dict(sorted(item['quota_histogram'].items()));item['credit_histogram']=dict(sorted(item['credit_histogram'].items()))
 return result

def mass_plan(receipt, stats):
 plan=receipt['family_balance'];unscaled=plan['unscaled_mass'];scales=plan['scales'];single={k:unscaled[k]*scales[k] for k in unscaled}
 q={}
 for family,mass in single.items():
  histogram=stats[family]['quota_histogram']
  # Uniformity is proven from every row, never assumed from its family.
  if len(histogram)!=1:return {'reason':'nonuniform row quotas require exact per-row effective loss census','one_pass_mass':single}
  q[family]=mass*int(next(iter(histogram)))
 return {'one_pass_mass':single,'one_pass_share':{k:v/sum(single.values()) for k,v in single.items()},'full_quota_mass':q,'full_quota_share':{k:v/sum(q.values()) for k,v in q.items()},'mass_scope':'sum of token-loss coefficients at fixed weights; excludes decision readout contribution and does not predict gradients or realized optimizer effects'}

def build(checkpoint,proof_path,parent_path,successor_path,config_path,output):
 checkpoint=Path(checkpoint).resolve(strict=True);proof_path=Path(proof_path).resolve(strict=True);parent_path=Path(parent_path).resolve(strict=True);successor_path=Path(successor_path).resolve(strict=True);config_path=Path(config_path).resolve(strict=True)
 proof=json.loads(proof_path.read_bytes());parent=json.loads(parent_path.read_bytes());successor=json.loads(successor_path.read_bytes());config=json.loads(config_path.read_bytes())
 if Path(proof['checkpoint']['path']).resolve()!=checkpoint or proof['successor_recipe']['sha256']!=parent['recipe']['sha256'] or successor['predecessor']['sha256']!=bind(parent_path)['sha256']:
  raise ValueError('checkpoint proof and prepared source parent differ')
 params=config['trainer_parameters'];epochs=params['epochs'];visits=params['communication_min_visits']
 if epochs!=1.0 or type(visits) is not int:raise ValueError('audit requires exact whole epoch and positive communication minimum')
 before=sig(checkpoint);state=CheckpointPolicy.load(checkpoint);training=state['training'];frame=(training.get('sft_progress') or {})[config['production_stage']]
 if sig(checkpoint)!=before:raise ValueError('checkpoint changed while mapping')
 old=read_ref(parent['family_map']);new=read_ref(successor['family_map']);oldrows=old['rows'];newrows=new['rows']
 if (len(oldrows)!=parent['rows'] or len(newrows)!=successor['rows'] or oldrows!=newrows[:len(oldrows)]
   or old['source_sha256']!=parent['source']['sha256'] or new['source_sha256']!=successor['source']['sha256']):raise ValueError('family maps lack exact append prefix')
 bindings=frame['source_row_bindings'];order=frame['source_row_order'];credits=frame['exposure_counts'];saved=frame['required_visits']
 if len(bindings)!=len(oldrows) or order!=[r['id'] for r in oldrows] or any(bindings[r['id']]!=r['binding'] for r in oldrows):raise ValueError('carried frame source differs from parent family inventory')
 if set(saved)!=set(bindings) or not set(credits)<=set(bindings):raise ValueError('carried frame quota/credit inventory differs')
 old_stats=family_stats(oldrows,credits,saved)
 # Apply actual source continuation rule: preserve each old quota, enforce
 # one epoch for appended rows, and enforce the declared CE communication minimum.
 future={r['id']:max(1,int(saved.get(r['id'],0)),visits if r['family']=='native-communication' else 1) for r in newrows}
 new_stats=family_stats(newrows,credits,future)
 pending_policy=(frame.get('options') or {}).get('communication_quota_policy')
 result={'schema':'natlang.native-sft-quota-objective-audit/1','checkpoint':proof['checkpoint'],'checkpoint_binding_scope':'exact checkpoint reference from independently authenticated zero-SGD publication; unchanged file identity across mmap metadata read','adoption_proof':bind(proof_path),'parent':bind(parent_path),'successor':bind(successor_path),'config':bind(config_path),'stage_id':config['production_stage'],'global_step':training.get('global_step'),'carried_frame':{'rows':len(oldrows),'actual_saved_quota_policy':pending_policy,'families':old_stats,'mass_exposure':mass_plan(parent,old_stats)},'prepared_continuation':{'rows':len(newrows),'epochs':epochs,'communication_min_visits':visits,'families':new_stats,'mass_exposure':mass_plan(successor,new_stats)},'limitations':['Full-quota coefficient mass is not an observed training mixture, gradient magnitude or capability measure.','Per-family sums do not identify per-row own-output mass; exact already-earned or remaining weighted mass is unknown.','Decision readout terms are outside the recorded token-only census; scheduling order, update packing and optimizer nonlinearity affect realized learning.'],'optimizer_or_source_mutation':False}
 if sig(checkpoint)!=before:raise ValueError('checkpoint changed while auditing')
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise ValueError('fresh output required')
 write_json_durable(output,result);print(json.dumps(result,sort_keys=True))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--proof',required=True);p.add_argument('--parent',required=True);p.add_argument('--successor',required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.checkpoint,a.proof,a.parent,a.successor,a.config,a.output)
