"""Exact whole-graph CE and offline-preference continuations between joint/RL publications.

The declared recipe selects sources and checkpoint cadence. This module only
builds/verifies the same shared-trainer invocation and durable pending update.
"""
from __future__ import annotations
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable


MANDATORY_STAGE_KINDS=frozenset({'native_state_sft','native_offline_preference'})


def objective_contract(sources,parameters):
    if 'objective_spec' not in sources:return None
    from ..train.native_preference import NativePreferenceObjective
    reference=sources['objective_spec']
    if sha256_file_hex_cached(reference['path'])!=reference['sha256']:
        raise ValueError('mandatory offline objective changed its declared source binding')
    return NativePreferenceObjective(reference['path'],beta=parameters.get('preference_beta'),
        margin=parameters.get('preference_margin')).binding()


SOURCE_ARGUMENTS = {
    'native_graphs': ('--native-graphs', 'seed_native_graphs'),
    'held_native_graphs': ('--held-native-graphs', 'held_native_graphs'),
    **{role: ('--'+role.replace('_','-'), role) for role in
       ('foundation_native_graphs', 'pending_native_sources', 'continued_native_graphs',
        'continued_held_native_graphs', 'withdrawn_native_graphs', 'unavailable_native_sources',
        'ce_supervision_policy', 'source_classification_migration', 'foundation_qualification_request')},
}


def _without_source_argument(command, flag, *, takes_value=True):
    result=[];index=0
    while index<len(command):
        value=command[index]
        if value==flag:
            if not takes_value:
                index+=1;continue
            if index+1==len(command) or command[index+1].startswith('--'):
                raise ValueError('native source argument lacks a value: '+flag)
            index+=2
        elif value.startswith(flag+'='):
            if not value[len(flag)+1:]:raise ValueError('native source argument lacks a value: '+flag)
            index+=1
        else:
            result.append(value);index+=1
    return result


def bind_seed_parameters(config,parameters):
    """One declared stage value wins over inherited native launcher defaults."""
    command=list(config['seed_command'])
    if not any(value in {'natlang_neuralese.train.native_state_sft',
        'natlang_neuralese.train.native_preference'} for value in command):return config
    for key,value in parameters.items():
        flag='--'+key.replace('_','-')
        command=_without_source_argument(command,flag,takes_value=not isinstance(value,bool))
        if isinstance(value,bool):
            command=_without_source_argument(command,'--no-'+key.replace('_','-'),takes_value=False)
    slots=[value for value in command if value in ('{trainer_args}','{seed_trainer_args}')]
    if len(slots)>1:raise ValueError('native seed command must contain one trainer parameter splice')
    if not slots:command.append('{trainer_args}')
    config['seed_command']=command
    return config


def bind_seed_sources(config, sources):
    """Serialize actual resolved stage roles, including adopted predecessor inputs."""
    command=list(config['seed_command'])
    native=any(value in {'natlang_neuralese.train.native_state_sft',
        'natlang_neuralese.train.native_preference'} for value in command)
    for role,(flag,key) in SOURCE_ARGUMENTS.items():
        present=any(value==flag or value.startswith(flag+'=') for value in command)
        command=_without_source_argument(command,flag)
        config.pop(key,None)
        if role in sources:
            config[key]=sources[role]['path']
            if native or present or role not in ('native_graphs','held_native_graphs'):
                command += [flag,'{'+key+'}']
    config['seed_command']=command
    return config


def bind_seed_objective(config,kind,sources):
    """The same seed command invokes the implemented objective, never CE by omission."""
    command=list(config['seed_command'])
    modules={'natlang_neuralese.train.native_state_sft','natlang_neuralese.train.native_preference'}
    module=next((index for index,value in enumerate(command) if value in modules),None)
    # Ordinary custom launchers retain their established contract. Offline
    # execution must explicitly expose the shared trainer seam.
    if kind=='native_offline_preference' and module is None:
        raise ValueError('offline preference seed command must bind the actual shared native trainer module')
    if module is not None:
        command[module]=('natlang_neuralese.train.native_preference' if kind=='native_offline_preference'
                         else 'natlang_neuralese.train.native_state_sft')
    for flag in ('--objective-spec',):
        while flag in command:
            index=command.index(flag);del command[index:index+2]
    if kind=='native_offline_preference':
        while '--foundation-native-graphs' in command:
            index=command.index('--foundation-native-graphs');del command[index:index+2]
        config['objective_spec']=sources['objective_spec']['path']
        config['offline_objective_binding']=objective_contract(sources,config['trainer_parameters'])
        command+=['--objective-spec','{objective_spec}']
    else:
        config.pop('objective_spec',None);config.pop('offline_objective_binding',None)
    config['seed_command']=command
    return config


def recipe_lineage(config):
    """Preserved lineage is separate from the current recipe policy identity."""
    identities=[]
    if config.get('recipe_lineage_sha256'):identities.append(config['recipe_lineage_sha256'])
    for key in ('trainer_parameters','seed_trainer_parameters'):
        reference=(config.get(key) or {}).get('recipe_binding')
        if isinstance(reference,str):reference=json.loads(reference)
        if reference:
            identities.append(reference.get('lineage_sha256',reference['sha256']))
    if identities:
        if any(value!=identities[0] for value in identities):
            raise ValueError('native continuation contains conflicting preserved recipe lineages')
        return identities[0]
    from ..train.recipe import recipe_identity_sha256
    return recipe_identity_sha256(config['production_recipe'])


def bind_recipe_lineage(config,lineage):
    from ..train.recipe import native_recipe_binding
    if config.get('recipe_lineage_sha256') not in (None,lineage):
        raise ValueError('native raw continuation cannot change its authenticated recipe lineage')
    config['recipe_lineage_sha256']=lineage
    for key in ('trainer_parameters','seed_trainer_parameters'):
        config[key]={**config.get(key,{}),'recipe_binding':native_recipe_binding(Path(config['production_recipe']),lineage)}
    return config


def bind_semantic_recipe_lineage(config,checkpoint,participant,declared,*,preserve_policy=False):
    from ..train.recipe import checkpoint_recipe_binding,native_recipe_provenance
    actual=checkpoint_recipe_binding(checkpoint,participant=participant)
    if actual is None or native_recipe_provenance(declared)!=actual:
        raise ValueError('semantic RL requires exact participant recipe provenance in checkpoint and adapter export')
    lineage=actual.get('lineage_sha256',actual['sha256'])
    if preserve_policy and config.get('recipe_lineage_sha256') is not None:
        if recipe_lineage(config)!=lineage:
            raise ValueError('semantic bootstrap config changed its authenticated participant recipe lineage')
        return config  # A sealed/current policy is not rebound by a bootstrap poll.
    return bind_recipe_lineage(config,lineage)


def continuation_config(config,stage,*,checkpoint,adapter,out):
    from ..train.recipe import resolve_recipe_data,resolve_stage_inputs,effective_stage_parameters,native_recipe_binding
    recipe=resolve_recipe_data(config['production_recipe'])
    sources=resolve_stage_inputs(recipe,stage,{})
    parameters=effective_stage_parameters(stage,[],recipe)
    parameters={key:value for key,value in parameters.items() if value is not None}
    if adapter is not None:parameters.pop('communication_foundation',None)
    # Keep the recipe's explicit capacity requests for ordinary SFT as well as
    # preference training. Restoring optimizer progress does not restore a
    # resident capacity certificate for this runtime/hardware/packing.
    if 'native_checkpoint_publication' in config:
        from ..model.native_checkpoint_control import publication_trainer_parameters
        parameters.update(publication_trainer_parameters(config['native_checkpoint_publication']))
    parameters['refresh_at_checkpoint']=True
    parameters['stage_id']=stage['id']
    parameters['recipe_binding']=native_recipe_binding(Path(config['production_recipe']),recipe_lineage(config))
    result={**config,'recipe_lineage_sha256':parameters['recipe_binding']['lineage_sha256'],'mandatory_sft_only':True,'out':str(Path(out).resolve()),'checkpoint':checkpoint,
        'semantic_adapter':adapter,'production_stage':stage['id'],'trainer_parameters':parameters,
        'seed_native_graphs':sources['native_graphs']['path'],
        'held_native_graphs':sources['held_native_graphs']['path']}
    if adapter is None:result.pop('semantic_adapter',None)
    bind_seed_parameters(result,parameters)
    bind_seed_sources(result,sources)
    bind_seed_objective(result,stage['kind'],sources)
    return result,sources


def stage_pending(adapter,participant,stage,sources,epochs,*,objective=None):
    progress=(adapter['models'][participant].get('mandatory_sft') or {}).get('stages',{}).get(stage)
    if objective is not None:
        actual=(progress or {}).get('objective') or {}
        if any(actual.get(key)!=value for key,value in objective.items()):return True
    pending = sources.get('pending_native_sources')
    if pending and (not progress or pending['sha256'] not in (progress.get('pending_source_declarations') or {})):
        return True  # Persist a newly declared obligation even when active CE is already complete.
    return not (progress and (progress.get('complete') is True or
        progress.get('active_remaining_graph_visits') == 0)
        and progress.get('train_source_sha256')==sources['native_graphs']['sha256']
        and progress.get('held_source_sha256')==sources['held_native_graphs']['sha256']
        and progress.get('epochs')==epochs)



def _resume_source_arguments(config,directory,values,*,parameters=None):
    """Do not reapply an obsolete append predecessor to an already adopted source."""
    command=list(config['seed_command'])
    roles=(('continued_native_graphs','seed_native_graphs',0),
           ('continued_held_native_graphs','held_native_graphs',1))
    argument_lists=[command,values['trainer_args'],values['seed_trainer_args']]
    def present(flag):
        return any(any(value==flag or value.startswith(flag+'=') for value in args)
                   for args in argument_lists)
    checkpoint=Path(directory)/'checkpoint.pt'
    if (config.get('native_actor_mode')!='raw-state' or not checkpoint.is_file()
            or not any(present('--'+role.replace('_','-')) for role,_,_ in roles)):
        return command,None
    from ..train.native_resident import binding,bound,current
    from ..train.checkpoint_policy import CheckpointPolicy
    from .native_raw_policy import require_raw_training
    reference=binding(checkpoint)
    saved=CheckpointPolicy.load(checkpoint)
    training=require_raw_training(saved)
    parameters=config['trainer_parameters'] if parameters is None else parameters
    stage=parameters.get('stage_id',config['production_stage'])
    if ((training.get('options') or {}).get('stage_id')!=stage
            or (training.get('identity') or {}).get('objective') is not None):
        return command,None
    receipts=training.get('source_receipts') or []
    if len(receipts)<2:raise ValueError('supervised retry checkpoint lost its actual source receipts')
    resident_directory=parameters.get('resident_data_directory',config.get('resident_data_directory'))
    resident=current(resident_directory) if resident_directory else None
    selected=(resident['payload'] if resident and resident['payload']['stage_id']==stage else None)
    omitted={}
    for role,name,index in roles:
        flag='--'+role.replace('_','-')
        if not present(flag):continue
        target=(selected['train' if index==0 else 'held'] if selected else binding(config[name]))
        bound(target)
        if target['sha256']!=receipts[index].get('sha256'):continue
        for args in argument_lists:
            cursor=0
            while cursor<len(args):
                if args[cursor]==flag:
                    if cursor+1>=len(args):raise ValueError('supervised retry has a missing source argument')
                    del args[cursor:cursor+2]
                elif args[cursor].startswith(flag+'='):del args[cursor]
                else:cursor+=1
        omitted[role]=target
    if binding(checkpoint)!=reference:raise ValueError('supervised retry checkpoint changed during source verification')
    if not omitted:return command,None
    return command,{'checkpoint':reference,'stage_id':stage,'already_current_sources':omitted}


def prepare_seed_command(config,directory,values,state,state_path,*,parameters=None):
    """Use the same exact resume and publication binding for every SFT entry."""
    # A rollout actor can bind history/rendering controls to the RL trainer.
    # Static SFT and preference use their own declared parameters; forwarding
    # the actor's argument splice can send unsupported serving flags to SFT.
    from ..train.recipe import stage_parameter_args
    selected=config.get('trainer_parameters',{}) if parameters is None else parameters
    values['trainer_args']=stage_parameter_args(selected)
    values['seed_trainer_args']=list(values['trainer_args'])
    command,reused=_resume_source_arguments(config,directory,values,parameters=parameters)
    if reused is not None:state['resumed_current_source_inputs']=reused
    if config.get('native_checkpoint_publication_owner_pointer') is not None:
        from .production import expand_command
        from .native_publication_owner import emit_current
        state['checkpoint_publication_owner']=emit_current(config,expand_command(command,values),
            out=config['out'],selected_checkpoint=values['checkpoint'],
            runtime_root=config.get('native_checkpoint_publication_runtime_root',config.get('runtime')))
    write_json_durable(state_path,state)
    return command


def execute(config,state,state_path,control):
    """Existing bridge kind=round enters here without starting a rollout actor."""
    from .production import run_command
    from ..train.recipe import stage_parameter_args
    if state.get('pending_native_checkpoint'):
        retry=(state.get('last_learner_progress') or {})
        if not (config.get('raw_foundation_qualification')
                and (retry.get('communication_foundation') or {}).get('complete')
                and retry.get('foundation_ready') is not True):
            return state
    directory=Path(config['out'])/'learner'
    report_path=directory/'report.json'
    values={**{k:v for k,v in config.items() if isinstance(v,(str,int,float,list))},
        'checkpoint':state['checkpoint'],'policy_id':sha256_file_hex_cached(state['checkpoint']),
        'learner_out':str(directory),'trainer_args':stage_parameter_args(config['trainer_parameters']),
        'seed_trainer_args':stage_parameter_args(config['trainer_parameters'])}
    if control is not None:values['_run_control']=control
    if config.get('before_update'):run_command(config['before_update'],values)
    state.update(phase='updating_mandatory_sft',pending_update_kind='mandatory_sft')
    command=prepare_seed_command(config,directory,values,state,state_path)
    run_command(command,values)
    checkpoint=directory/'checkpoint.pt'
    if not checkpoint.is_file() or not report_path.is_file():
        if control is not None and control.stop.requested:
            state['phase']='requested_stop';write_json_durable(state_path,state);return state
        raise RuntimeError('mandatory supervised continuation produced no durable checkpoint/report')
    report=json.loads(report_path.read_text())
    from ..train.native_operation_report import control_progress
    summary=control_progress(report,report_path)
    boundary=report.get('continuation_boundary') or {}
    admission = report.get('pending_source_admission') or {}
    declared = config.get('pending_native_sources')
    declared_sha = sha256_file_hex_cached(declared) if declared else None
    metadata_admitted = (admission.get('schema') == 'natlang.native-pending-source-admission/1'
        and admission.get('new_optimizer_updates') == 0
        and admission.get('parent_checkpoint_sha256') == sha256_file_hex_cached(state['checkpoint'])
        and admission.get('checkpoint_sha256') == sha256_file_hex_cached(checkpoint)
        and declared_sha is not None and declared_sha in (admission.get('declarations') or [])
        and any(declared_sha in (stage.get('pending_source_declarations') or {})
            for stage in ((report.get('mandatory_sft') or {}).get('stages') or {}).values()))
    from ..train.native_operation_report import active_stage_handoff
    active_complete=active_stage_handoff(report,{'sha256':sha256_file_hex_cached(checkpoint)},stage_id=config.get('production_stage'))
    usable=(metadata_admitted or active_complete or report.get('completed') is True or (boundary.get('kind')=='semantic-refresh'
        and boundary.get('reached') is True and boundary.get('new_optimizer_updates',0)>0))
    from ..train.native_operation_report import supervised_progress_numerics_ok
    numerics_ok = supervised_progress_numerics_ok(report,
        {'path': str(checkpoint), 'sha256': sha256_file_hex_cached(checkpoint)})
    if not usable or not numerics_ok:
        state.update(phase='mandatory_sft_pending',last_learner_progress=summary)
        write_json_durable(state_path,state)
        if control is not None and control.stop.requested:return state
        raise RuntimeError('mandatory supervised continuation needs its actual qualified refresh boundary')
    state.update(phase='awaiting_semantic_refresh',pending_native_checkpoint=str(checkpoint),
        pending_native_round_dir=str(Path(config['out'])),last_learner_progress=summary,
        mandatory_sft=summary.get('mandatory_sft'))
    if config.get('native_actor_mode')=='raw-state' and report.get('resident_adoption'):
        from ..train.native_resident import current,binding,bound
        from ..common.jsonio import write_bytes_durable
        adopted=report['resident_adoption'];source=Path(adopted['directory']);resident=current(source)
        if (not resident or resident['generation']!=adopted['generation'] or
                resident['payload']['recipe']!=(json.loads((directory/'native-training-receipt.json').read_text())).get('recipe')):
            raise ValueError('raw SFT worker changed its actual final adopted source generation')
        snapshot=directory/'resident-source';snapshot.mkdir(parents=True,exist_ok=True)
        write_bytes_durable(snapshot/'current.json',bound(binding(source/'current.json')).read_bytes())
        recipe=resident['payload']['recipe'];recipe_binding={'path':recipe['path'],'sha256':recipe['file_sha256']}
        write_bytes_durable(snapshot/'recipe.json',bound(recipe_binding).read_bytes())
        state['resident_current']=binding(snapshot/'current.json')
        state['resident_recipe']={'source':recipe_binding,'copy':binding(snapshot/'recipe.json')}
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
            if stage['kind'] in MANDATORY_STAGE_KINDS] for config in configs]
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
                                 declaration['trainer_parameters'].get('epochs',1),
                                 objective=declaration.get('offline_objective_binding')):
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

