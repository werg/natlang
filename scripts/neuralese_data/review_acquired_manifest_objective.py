#!/usr/bin/env python3
"""Calculate the existing builder's general-agent append objective without admitting it."""
import argparse
import json
import math
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.own_output_supervision import identity,policy_objective
from scripts.neuralese_data.build_acquired_streaming_successor import bind,checked,read,predecessor_mass_census,scales_for_append


def review(predecessor_path,combined_path,census_path,output,*,share=0.07):
    predecessor_path,combined_path,census_path=[Path(path).resolve(strict=True) for path in
        (predecessor_path,combined_path,census_path)]
    pred_ref,combined_ref,census_ref=map(bind,(predecessor_path,combined_path,census_path))
    previous,combined,census=map(read,(predecessor_path,combined_path,census_path))
    if (combined.get('schema')!='natlang.combined-acquired-screened-delta/1' or combined.get('training_admission') is not False
            or combined.get('predecessor',{}).get('sha256')!=pred_ref['sha256']
            or census.get('schema')!='natlang.acquired-effective-learning-census/1' or census.get('training_admission') is not False
            or census.get('screen',{}).get('sha256')!=combined_ref['sha256'] or census.get('rows')!=combined.get('rows')
            or census.get('ce_policy',{}).get('sha256')!=previous['ce_policy']['sha256']
            or census.get('counts',{}).get('zero_effective_rows')!=0):
        raise ValueError('objective review needs the exact full predecessor-bound census')
    policy=read(checked(previous['ce_policy'],'predecessor CE policy'));objective=policy_objective(policy)
    if census.get('objective_sha256')!=canonical_json_sha256_hex(objective):raise ValueError('loss objective changed')
    balance=objective.get('family_balance') or {}
    if (balance.get('key')!='logical-family' or balance.get('normalization')!='declared/1'
            or balance.get('max_scale')!=8 or identity(objective)!=previous['objective_transition']['to']):
        raise ValueError('predecessor does not use the exact declared family-balanced objective')
    previous_mass,mass_custody=predecessor_mass_census(previous,predecessor_path,objective)
    categories=census['category_mass']
    if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in categories.values()):
        raise ValueError('census category mass is invalid')
    new_mass=sum(categories.values());plan=scales_for_append(previous_mass,balance['scales'],new_mass,share=share,cap=8.0,family='general-agent')
    old_effective={name:value['unscaled_mass']*balance['scales'][name] for name,value in previous_mass['families'].items()}
    old_total=sum(old_effective.values());old_shares={name:mass/old_total for name,mass in old_effective.items()}
    marker_mass=census['counts']['tool_call_markup_effective_mass']
    if not 0<=marker_mass<=categories.get('reply',0)+1e-6:raise ValueError('tool markup mass is not a reply subset')
    result={'schema':'natlang.acquired-manifest-objective-review/1','training_admission':False,
        'predecessor':pred_ref,'combined':combined_ref,'census':census_ref,'predecessor_mass_custody':mass_custody,
        'new_rows':census['rows'],'logical_family':'general-agent','requested_target_share':share,
        'new_unscaled_mass':new_mass,'new_category_mass':categories,'new_tool_markup_subset_mass':marker_mass,
        'previous_effective_shares':old_shares,'proposed_family_balance':plan,
        'proposed_communication_share_change':plan['effective_shares'].get('native-communication',0)-old_shares.get('native-communication',0),
        'new_delta_effective_mass':new_mass*plan['scales']['general-agent'],
        'new_tool_markup_effective_mass':marker_mass*plan['scales']['general-agent'],
        'pending_render_obligations':combined['pending_render_obligations'],
        'carried_over_context_obligations':combined['carried_over_context_obligations'],
        'builder':bind(Path(__file__)),
        'scope':'Existing builder token-mass calculation only. Family labels are not semantic taxonomy, tool markup is not verified action, and no data or checkpoint was admitted.'}
    if any(bind(path)!=ref for path,ref in zip((predecessor_path,combined_path,census_path),(pred_ref,combined_ref,census_ref),strict=True)):
        raise ValueError('review controls changed')
    output=Path(output)
    if output.exists():raise ValueError('objective review output exists')
    write_json_durable(output,result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predecessor',required=True)
    p.add_argument('--combined',required=True);p.add_argument('--census',required=True);p.add_argument('--out',required=True)
    p.add_argument('--share',type=float,default=0.07);a=p.parse_args()
    r=review(a.predecessor,a.combined,a.census,a.out,share=a.share)
    print(json.dumps({k:r[k] for k in ('new_rows','new_unscaled_mass','previous_effective_shares',
        'proposed_communication_share_change','proposed_family_balance','training_admission')},sort_keys=True))
if __name__=='__main__':main()
