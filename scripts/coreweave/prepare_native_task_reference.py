#!/usr/bin/env python3
"""Prepare an explicit task-floor-only recipe continuation; no evaluation calls."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/neuralese'))
from natlang_neuralese.train.native_deployment import binding
from natlang_neuralese.train.native_readiness import _bound
from natlang_neuralese.train.recipe import resolve_recipe_data,effective_stage_parameters
from natlang_neuralese.eval.task_cadence import parse_task_eval
from natlang_neuralese.common.jsonio import write_json_durable
from provider_bridge import checked_binding


def reference(role,source,*,owner_reason):
    binding(source['path'],source)
    if not isinstance(owner_reason,str) or not owner_reason.strip():raise ValueError('task floor needs its explicit supervisor judgment')
    if role=='lfm':
        return 0,{'kind':'owner-historical-zero-floor','source':source,'reason':owner_reason,
                  'base_measurement':False,'historical':True,'qualifies_current_weights':False}
    if role!='mellum':raise ValueError('task reference needs its explicit student role')
    value=json.loads(_bound(source).read_bytes());result=value['result'];provenance=result['evaluation_provenance']
    if (value.get('historical')is not True or value.get('qualifies_current_weights')is not False
            or result.get('schema')!='natlang.in-run-task-eval/1' or result.get('eligible')is not True
            or provenance['model_config'].get('architecture')!='mellum'
            or value['checkpoint_sha256']!=provenance['checkpoint']['sha256']
            or result['step']!=value['step'] or type(result['task_successes'])is not int
            or not 0<=result['task_successes']<=result['cases']):
        raise ValueError('Mellum floor must preserve its genuine historical measured observation')
    return result['task_successes'],{'kind':'observed-prior-lineage-floor','source':source,'reason':owner_reason,
        'historical':True,'qualifies_current_weights':False,'base_measurement':False,
        'source_complete':result['complete'],'source_eligible':result['eligible'],
        'step':value['step'],'checkpoint_sha256':value['checkpoint_sha256'],
        'cases':result['case_indices'],'packet_sha256':result['packet_sha256'],
        'evaluation_spec_sha256':provenance['evaluation_spec_sha256'],
        'model_config':{key:provenance['model_config'][key]for key in
            ('architecture','base_inventory_sha256','dtype','native_shared_execution')},
        'observed_successes':result['task_successes'],'observed_cases':result['cases']}


def portable_reference(role,provenance,evidence):
    local=evidence['local'];binding(local['path'],local)
    remote=checked_binding(evidence['remote'])
    if (remote['sha256']!=local['sha256'] or remote['bytes']!=Path(local['path']).stat().st_size):
        raise ValueError('portable task reference evidence changed its actual bytes')
    original=provenance['source']
    summary={key:value for key,value in provenance.items()if key!='source'}
    summary.update(original_source_path=original['path'],original_source_sha256=original['sha256'],
                   original_source_bytes=Path(original['path']).stat().st_size)
    document=json.loads(_bound(local).read_bytes())
    if (document.get('schema')!='natlang.native-task-reference-evidence/1'
            or document.get('roles',{}).get(role)!=summary):
        raise ValueError('portable task reference differs from its authenticated original observation')
    return {**summary,'source':remote}


def prepare(recipe,role,source,out,*,owner_reason,portable_evidence=None):
    binding(recipe['path'],recipe)
    prior=resolve_recipe_data(recipe['path']);future=copy.deepcopy(prior)
    count,provenance=reference(role,source,owner_reason=owner_reason)
    if portable_evidence is not None:provenance=portable_reference(role,provenance,portable_evidence)
    changes={}
    for stage in future['stages']:
        parameters=effective_stage_parameters(stage,[],prior)
        original=parameters.get('task_eval')
        if not original:continue
        policy=json.loads(original)if isinstance(original,str)else copy.deepcopy(original)
        parsed=parse_task_eval(policy)
        if role=='mellum' and (parsed['packet']['sha256']!=provenance['packet_sha256']
                or parsed['cases']!=provenance['cases']):
            raise ValueError('selected prior-lineage observation uses another packet/case selection')
        policy.update(reference_successes=count,reference_provenance=provenance)
        parse_task_eval(policy)
        value=json.dumps(policy,sort_keys=True)if isinstance(original,str)else policy
        stage.setdefault('parameters',{})['task_eval']=value
        changes[stage['id']]={'task_eval':value}
    if not changes:raise ValueError('normal recipe has no actual task policies to continue')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('task policy continuation needs a fresh immutable namespace')
    out.mkdir(parents=True)
    recipe_path=out/'recipe.json';write_json_durable(recipe_path,future)
    delta=out/'stage-parameter-delta.json';write_json_durable(delta,changes)
    receipt={'schema':'natlang.native-task-reference-continuation/1','role':role,
        'source_recipe':recipe,'reference_source':source,'recipe':binding(recipe_path),
        'stage_parameter_delta':binding(delta),'reference_successes':count,'reference_provenance':provenance,
        'only_task_reference_fields_changed':True,'checkpoint_decodes':0,'optimizer_updates':0,
        'new_exposure':0,'qualifies_current_weights':False,'activated':False}
    write_json_durable(out/'prepared.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True,help='JSON artifact binding')
    p.add_argument('--source',type=Path,required=True,help='JSON historical observation/owner-decision binding')
    p.add_argument('--role',choices=['lfm','mellum'],required=True);p.add_argument('--reason',required=True)
    p.add_argument('--portable-evidence',type=Path,help='checked local/remote evidence bindings')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(json.loads(a.recipe.read_bytes()),a.role,json.loads(a.source.read_bytes()),a.out,
        owner_reason=a.reason,portable_evidence=json.loads(a.portable_evidence.read_bytes())if a.portable_evidence else None)))


if __name__=='__main__':main()
