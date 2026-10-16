"""Exact whole-graph CE continuations between joint/RL publications.

The declared recipe selects sources and checkpoint cadence. This module only
builds/verifies the same shared-trainer invocation and durable pending update.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable


def continuation_config(config,stage,*,checkpoint,adapter,out):
    from ..train.recipe import resolve_recipe_data,resolve_stage_inputs,effective_stage_parameters,native_recipe_binding,recipe_identity_sha256
    recipe=resolve_recipe_data(config['production_recipe'])
    sources=resolve_stage_inputs(recipe,stage,{})
    parameters=effective_stage_parameters(stage,[],recipe)
    parameters={key:value for key,value in parameters.items() if value is not None}
    parameters.pop('communication_foundation',None)
    parameters.pop('memory_qualification',None)  # one restored-parent capacity check, not every CE quantum
    parameters['refresh_at_checkpoint']=True
    parameters['stage_id']=stage['id']
    parameters['recipe_binding']=native_recipe_binding(Path(config['production_recipe']),recipe_identity_sha256(config['production_recipe']))
    result={**config,'mandatory_sft_only':True,'out':str(Path(out).resolve()),'checkpoint':checkpoint,
        'semantic_adapter':adapter,'production_stage':stage['id'],'trainer_parameters':parameters,
        'seed_native_graphs':sources['native_graphs']['path'],
        'held_native_graphs':sources['held_native_graphs']['path']}
    return result,sources


def stage_pending(adapter,participant,stage,sources,epochs):
    progress=(adapter['models'][participant].get('mandatory_sft') or {}).get('stages',{}).get(stage)
    return not (progress and progress.get('complete') is True
        and progress.get('train_source_sha256')==sources['native_graphs']['sha256']
        and progress.get('held_source_sha256')==sources['held_native_graphs']['sha256']
        and progress.get('epochs')==epochs)


def execute(config,state,state_path,control):
    """Existing bridge kind=round enters here without starting a rollout actor."""
    from .production import run_command
    from ..train.recipe import stage_parameter_args
    if state.get('pending_native_checkpoint'):
        return state
    directory=Path(config['out'])/'learner'
    report_path=directory/'report.json'
    values={**{k:v for k,v in config.items() if isinstance(v,(str,int,float,list))},
        'checkpoint':state['checkpoint'],'policy_id':sha256_file_hex_cached(state['checkpoint']),
        'learner_out':str(directory),'trainer_args':stage_parameter_args(config['trainer_parameters'])}
    if control is not None:values['_run_control']=control
    if config.get('before_update'):run_command(config['before_update'],values)
    state.update(phase='updating_mandatory_sft',pending_update_kind='mandatory_sft')
    write_json_durable(state_path,state)
    run_command(config['seed_command'],values)
    checkpoint=directory/'checkpoint.pt'
    if not checkpoint.is_file() or not report_path.is_file():
        if control is not None and control.stop.requested:
            state['phase']='requested_stop';write_json_durable(state_path,state);return state
        raise RuntimeError('mandatory supervised continuation produced no durable checkpoint/report')
    report=json.loads(report_path.read_text())
    boundary=report.get('continuation_boundary') or {}
    usable=(report.get('completed') is True or (boundary.get('kind')=='semantic-refresh'
        and boundary.get('reached') is True and boundary.get('new_optimizer_updates',0)>0))
    if not usable or report.get('numerics_ok') is not True:
        state.update(phase='mandatory_sft_pending',last_learner_progress=report)
        write_json_durable(state_path,state)
        if control is not None and control.stop.requested:return state
        raise RuntimeError('mandatory supervised continuation needs its actual qualified refresh boundary')
    state.update(phase='awaiting_semantic_refresh',pending_native_checkpoint=str(checkpoint),
        pending_native_round_dir=str(Path(config['out'])),last_learner_progress=report,
        mandatory_sft=report.get('mandatory_sft'))
    write_json_durable(state_path,state)
    return state


def interleave(config_paths,configs,adapter_path,directory,*,cycle,latest,device,stop):
    """One checkpoint quantum per declared CE stage, then return to RL.

    Stage index is durable across publication/restart; each changed participant
    is merged and adapted before a subsequent actor or CE stage uses it.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .native_rounds import round_job,refresh_job,apply_deployment
    from .production import run_command,RunControl
    from ..train.recipe import resolve_recipe_data
    from ..train.native_readiness import _bound
    plans=[[stage for stage in resolve_recipe_data(config['production_recipe'])['stages']
            if stage['kind']=='native_state_sft'] for config in configs]
    prior=json.loads(latest.read_text()) if latest.exists() else {}
    if prior.get('supervised_cycle')==cycle:return adapter_path
    cursor=prior.get('supervised_cursor') or {}
    start=cursor.get('next_stage',0) if cursor.get('cycle')==cycle else 0
    directory=Path(directory)
    for index in range(start,max(map(len,plans),default=0)):
        if stop.requested:raise InterruptedError('mandatory supervised continuation checkpointed before publication')
        export=json.loads(Path(adapter_path).read_text())
        _bound(export['checkpoint'])
        jobs=[]
        for config,stages in zip(configs,plans):
            if index>=len(stages):continue
            stage=stages[index]
            out=directory/f'stage-{index:03d}'/config['native_participant']
            declaration,sources=continuation_config(config,stage,checkpoint=export['checkpoint']['path'],
                adapter=adapter_path,out=out)
            if not stage_pending(export,config['native_participant'],stage['id'],sources,
                                 declaration['trainer_parameters'].get('epochs',1)):
                continue
            path=out/'config.json';path.parent.mkdir(parents=True,exist_ok=True)
            write_json_durable(path,declaration)
            jobs.append((path,declaration,out/'bridge'))
        updates={}
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                futures=[pool.submit(round_job,path,config,out,stop) for path,config,out in jobs]
                failures=[]
                for (_,config,_),future in zip(jobs,futures):
                    try:
                        state=future.result()
                        if state.get('pending_native_checkpoint'):
                            updates[config['native_participant']]=state['pending_native_checkpoint']
                    except Exception as error:failures.append(error)
            if failures:raise failures[0]
            if stop.requested:raise InterruptedError('mandatory supervised learners drained; pending update retained')
            if not updates:raise RuntimeError('mandatory supervised jobs returned no actual optimizer continuation')
            curricula={str(Path(config['semantic_refresh_config']).resolve()) for config in configs}
            if len(curricula)!=1:raise ValueError('mandatory supervised participants must share the semantic refresh curriculum')
            stage_dir=directory/f'stage-{index:03d}'
            deployment=refresh_job(configs,adapter_path,updates,next(iter(curricula)),stage_dir,device,stop)
            for config in configs:
                if config.get('after_update'):
                    run_command(config['after_update'],{**config,'_run_control':RunControl(stop),
                        'checkpoint':deployment['checkpoint']['path'],'policy_id':deployment['checkpoint']['sha256'],
                        'semantic_adapter':deployment['semantic_adapter']['path'],
                        'semantic_adapter_sha256':deployment['semantic_adapter']['sha256'],
                        'deployment':str(stage_dir/'semantic-refresh/deployment.json')})
            adapter_path=deployment['semantic_adapter']['path']
            for config in configs:apply_deployment(config,deployment)
        else:
            deployment={'schema':'natlang.native-rl-deployment/1','checkpoint':export['checkpoint'],
                'semantic_adapter':{'path':str(adapter_path),'sha256':sha256_file_hex_cached(adapter_path)},
                'models':export['models'],'updated_participants':{}}
        write_json_durable(latest,{'schema':'natlang.native-rl-joint-controller/1','deployment':deployment,
            'next_round':cycle,'supervised_cursor':{'cycle':cycle,'next_stage':index+1}})
    if latest.exists():
        value=json.loads(latest.read_text());value['supervised_cycle']=cycle
        write_json_durable(latest,value)
    return adapter_path

