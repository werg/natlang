#!/usr/bin/env python3
"""Measure only visited or nonuniform-quota rows against the real source index."""
import argparse,json,math
from pathlib import Path
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train.own_output_supervision import prepare_policy, policy_objective
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.hashing import sha256_file_hex


def bind(p):
 p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':sha256_file_hex(p),'bytes':p.stat().st_size}

def checked(ref):
 r=bind(ref['path'])
 if r['sha256']!=ref['sha256'] or 'bytes' in ref and r['bytes']!=ref['bytes']:raise ValueError('bound census input changed')
 return json.loads(Path(r['path']).read_bytes())

def signature(p):
 s=Path(p).stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def build(checkpoint,adoption,predecessor,successor,output):
 checkpoint=Path(checkpoint).resolve(strict=True);before=signature(checkpoint);adoption=Path(adoption).resolve(strict=True);predecessor=Path(predecessor).resolve(strict=True);successor=Path(successor).resolve(strict=True)
 proof=json.loads(adoption.read_bytes());parent=json.loads(predecessor.read_bytes());new=json.loads(successor.read_bytes())
 if Path(proof['checkpoint']['path']).resolve()!=checkpoint or proof['successor_recipe']['sha256']!=parent['recipe']['sha256'] or new['predecessor']['sha256']!=bind(predecessor)['sha256']:raise ValueError('census parent identity differs')
 frame=CheckpointPolicy.load(checkpoint)['training']['sft_progress']['native_joint_sft_v4'];credits=frame['exposure_counts'];quotas=frame['required_visits'];oldmap=checked(parent['family_map']);newmap=checked(new['family_map'])
 if oldmap['rows']!=newmap['rows'][:parent['rows']] or oldmap['source_sha256']!=parent['source']['sha256'] or newmap['source_sha256']!=new['source']['sha256']:raise ValueError('census source family prefix changed')
 ids={r['id'] for r in oldmap['rows'] if int(credits.get(r['id'],0))>0 or (r['family']!='native-communication' and int(quotas[r['id']])!=1)}
 geometry=checked(new['geometry']);meta=geometry['identity'];source=NativeGraphSource(new['source']['path'],meta['tokenizer'],required_split='train',role_ids=meta['roles'])
 if len(source.entries)!=new['rows'] or any(e['id']!=r['id'] or e['binding']!=r['binding'] for e,r in zip(source.entries,newmap['rows'],strict=True)):raise ValueError('source index differs from family map')
 objective=policy_objective(checked(new['ce_policy']));objective['family_balance']=None
 from natlang_neuralese.train.own_output_supervision import apply_to_example
 # Let source validate each selected parsed graph and its recorded binding;
 # apply the same unscaled graph-weight projection used by the actual trainer.
 selected=[i for i,e in enumerate(source.entries) if e['id'] in ids]
 byid={r['id']:r for r in oldmap['rows'] if r['id'] in ids}
 # The source loader needs an actual policy to retrieve and project raw rows.
 policy=load_policy(new['ce_policy']);policy=prepare_policy(policy,source);source.set_ce_policy(policy)
 rows=[]
 for example in source.iter_indices(selected):
  row=byid[example.id];weights=example.target_weights;mass=math.fsum(weights);skipped=math.fsum(weights[offset] for offset in example.plan.offsets)
  if skipped!=0:raise ValueError('effective first-token mass is not scored by trainer')
  rows.append({'id':example.id,'binding':row['binding'],'family':row['family'],'earned_visits':int(credits.get(example.id,0)),'saved_required_visits':int(quotas[example.id]),'unscaled_effective_token_mass':mass,'decision_count':len(example.decisions)})
 if len(rows)!=len(ids) or signature(checkpoint)!=before:raise ValueError('selected row census or checkpoint changed')
 output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise ValueError('fresh census output required')
 result={'schema':'natlang.sft-visited-row-objective-census/1','checkpoint':proof['checkpoint'],'parent':bind(predecessor),'successor':bind(successor),'ce_policy':new['ce_policy'],'selected_rows':len(rows),'rows':rows,'scope':'actual trainer projection for visited and nonuniform quota rows; no model forward or gradient.'};write_json_durable(output,result);print(json.dumps({'selected_rows':len(rows),'decision_count':sum(r['decision_count'] for r in rows)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--adoption',required=True);p.add_argument('--parent',required=True);p.add_argument('--successor',required=True);p.add_argument('--output',required=True);a=p.parse_args();build(a.checkpoint,a.adoption,a.parent,a.successor,a.output)
