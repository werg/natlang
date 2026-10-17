"""Per-student whole-recipe/RL continuation before the shared-codec rendezvous.

The recipe/curriculum declares work. Exact plumbing serializes existing workers,
commits current checkpoint ownership and releases only genuine latest readiness.
"""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time
from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable,write_bytes_durable
from ..train.native_readiness import _bound
from .native_raw_handoff import binding,verify_handoff

SCHEMA='natlang.native-raw-rounds/1'


class StudentConfigurationError(Exception):
    """No operation has been launched by this cycle; other students remain useful."""


def _configuration_signature(declaration,directory):
    from ..common.hashing import canonical_json_sha256_hex
    path=Path(directory)/'config.json'
    return canonical_json_sha256_hex({'declaration':declaration,
        'config':binding(path) if path.is_file() else None})


def raw_out(config):
    return Path(config.get('raw_rl_out',str(Path(config.get('rl_out',Path(config['out'])/'rl')).parent/'native_raw_rl'))).resolve()


def _task_cases(declaration):
    """Check admission and selector metadata before any checkpoint allocation."""
    from ..common.jsonio import iter_jsonl_rows
    path=_bound(declaration['cases'])
    cases=[row for _,row in iter_jsonl_rows(path)]
    if not cases or any(row.get('split')!='train' for row in cases):
        raise ValueError('raw bootstrap requires nonempty exact admitted TRAIN tasks')
    from .curriculum import task_descriptors
    task_descriptors(cases)
    if declaration.get('task_source'):
        from .task_source import resolve_task_source
        declared=declaration['task_source'];actual=resolve_task_source(declared['manifest'])
        if (any(actual[key]!=declared[key] for key in ('manifest_sha256','cases_sha256','rows','audit_sha256','selection_proof_sha256'))
                or actual['cases']!=path):
            raise ValueError('raw bootstrap task-source admission binding changed')
    return path


def _inputs(declaration):
    inputs={name:_bound(declaration[name]) for name in ('production_recipe','seed_native_graphs','held_native_graphs')}
    inputs['cases']=_task_cases(declaration)
    return inputs


def _seed_quantum(config,record):
    """Bind the complete supervised settings to the finite seed operation."""
    config['seed_trainer_parameters']={**config.get('trainer_parameters',{}),
        **config.get('seed_trainer_parameters',{}),'refresh_at_checkpoint':True,
        'stage_id':config['production_stage']}
    if record.get('resident'):
        directory=record['resident']['directory']
        config['resident_data_directory']=directory
        config['seed_trainer_parameters']['resident_data_directory']=directory
    if config.get('seed_command'):
        config['seed_command']=[part.replace('{trainer_args}','{seed_trainer_args}')
            for part in config['seed_command']]
    _supervised_quantum(config,config['seed_trainer_parameters'])
    return config


def _supervised_quantum(config,parameters):
    if 'supervised_quantum_updates' in config:
        quantum=config['supervised_quantum_updates']
        # Existing trainer requires checkpoint/eval alignment. At most one
        # held evaluation per short quantum when the old cadence is longer.
        from ..train.native_state_sft import build_parser
        previous=parameters.get('eval_every',build_parser().get_default('eval_every'))
        evaluation=min(previous,quantum)
        if quantum % evaluation:
            raise ValueError('supervised quantum must be a multiple of its shorter held evaluation cadence')
        parameters['checkpoint_every']=quantum
        parameters['eval_every']=evaluation
        parameters['refresh_at_checkpoint']=True


def _scheduling_controls(config,declaration,plan_path,record,*,pending=False):
    # Pending collection retains its sealed packet. Only subsequent supervised
    # stop cadence may change; it is independently bound to the declaring plan.
    if 'rows_per_round' in declaration and not pending:
        config['rows_per_round']=declaration['rows_per_round']
    if 'supervised_quantum_updates' in declaration:
        config['supervised_quantum_updates']=declaration['supervised_quantum_updates']
        config['supervised_quantum_control']={'plan':binding(plan_path),
            'updates':declaration['supervised_quantum_updates']}
    _seed_quantum(config,record)
    return config


def configure_model(plan_path,declaration,handoff,out):
    from .production import configure_native
    from .native_rounds import configure_provider
    for key in ('rows_per_round','supervised_quantum_updates'):
        if key in declaration and (type(declaration[key]) is not int or declaration[key]<=0):
            raise ValueError(key+' must be a positive integer')
    _task_cases(declaration)
    plan_path=Path(plan_path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/'config.json';verification=out/'bootstrap-verification.json'
    from .native_bootstrap_metadata import unchanged,save
    if path.exists():
        prior=json.loads(path.read_text())
        if prior.get('runtime_handoff') and not declaration.get('runtime_handoff'):
            raise ValueError('raw continuation cannot discard its later runtime authority')
        owner={'plan_sha256':sha256_file_hex_cached(plan_path),'handoff':handoff}
        if prior.get('runtime_binding',prior.get('bootstrap_binding'))==owner and unchanged(verification,plan_path,declaration,handoff,prior):
            return path
        if declaration.get('runtime_handoff'):
            from .native_runtime_handoff import configure_runtime_handoff
            return configure_runtime_handoff(plan_path,declaration,handoff,path)
    elif declaration.get('runtime_handoff'):
        raise ValueError('later runtime authority requires its existing bootstrap and cursor')
    record,saved=verify_handoff(handoff,requirement=declaration['handoff_requirement'])
    actual=saved['model_config']['architecture']+':'+saved['model_config']['base_inventory_sha256']
    if actual!=declaration['handoff_requirement']['model_id']:
        raise ValueError('raw bootstrap drained another student/base identity')
    recipe_binding=(saved['training'].get('options') or {}).get('recipe_binding')
    if isinstance(recipe_binding,str):recipe_binding=json.loads(recipe_binding)
    lineage=(recipe_binding.get('lineage_sha256',recipe_binding['sha256']) if recipe_binding else
        declaration['handoff_requirement'].get('recipe_lineage_sha256',declaration['handoff_requirement']['recipe_sha256']))
    installed_plan=None
    if (record.get('resident') or {}).get('initial'):
        from .native_raw_handoff import _drained_plan_authority
        _,installed_plan,_=_drained_plan_authority(record,saved)
    active_declaration=declaration
    if declaration.get('drained_recipe_handoff'):
        from .native_raw_handoff import verify_drained_recipe_handoff
        transition=verify_drained_recipe_handoff(declaration['drained_recipe_handoff'],handoff=handoff,
            verified_handoff=(record,saved))
        plan=json.loads(_bound(transition['new_plan']).read_text())
        lineage=plan['recipe_sha256']
        stage_inputs=plan['stage_inputs'][declaration['production_stage']]
        active_declaration={**declaration,'production_recipe':transition['recipe'],
            'seed_native_graphs':stage_inputs['native_graphs'],'held_native_graphs':stage_inputs['held_native_graphs']}
    elif (record.get('resident') or {}).get('current'):
        from ..train.native_resident import current
        reference=record['resident']['current']
        _release_requirement(declaration,{'resident_current':reference})
        resident=current(_bound(reference).parent)['payload']
        recipe=resident['recipe']
        lineage=resident['plan']['recipe_sha256']
        active_declaration={**declaration,'production_recipe':{'path':recipe['path'],'sha256':recipe['file_sha256']},
            'seed_native_graphs':resident['train'],'held_native_graphs':resident['held']}
    elif installed_plan is not None:
        lineage=installed_plan['recipe_sha256']
        initial=json.loads(_bound(record['resident']['initial']).read_text())
        recipe=initial['recipe'];stage_inputs=installed_plan['stage_inputs'][declaration['production_stage']]
        active_declaration={**declaration,'production_recipe':{'path':recipe['path'],'sha256':recipe.get('file_sha256',recipe['sha256'])},
            'seed_native_graphs':stage_inputs['native_graphs'],'held_native_graphs':stage_inputs['held_native_graphs']}
    cursor_path=out/'cycle.json'
    cursor=json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
    if cursor.get('resident_current'):
        from ..train.native_resident import current
        _release_requirement(declaration,cursor,verified_handoff=(record,saved))
        resident=current(_bound(cursor['resident_current']).parent)['payload']
        recipe=resident['recipe'];lineage=resident['plan']['recipe_sha256']
        active_declaration={**declaration,'production_recipe':{'path':recipe['path'],'sha256':recipe['file_sha256']},
            'seed_native_graphs':resident['train'],'held_native_graphs':resident['held']}
    if recipe_binding and recipe_binding.get('lineage_sha256',recipe_binding['sha256'])!=lineage:
        raise ValueError('raw handoff saved optimizer recipe lineage differs from its installed plan')
    inputs=_inputs(active_declaration)
    runtime=Path(declaration['runtime']).resolve(strict=True)
    if not (runtime/'dist/teacher/cli.js').is_file():raise ValueError('raw bootstrap needs its actual built collector runtime')
    path=out/'config.json';owner={'plan_sha256':sha256_file_hex_cached(plan_path),'handoff':handoff}
    if path.exists():
        prior=json.loads(path.read_text())
        if (prior.get('bootstrap_binding') or {}).get('handoff')!=handoff:
            raise ValueError('raw bootstrap original optimizer handoff changed')
        if prior.get('bootstrap_binding')==owner:
            if prior.get('recipe_lineage_sha256')==lineage:
                from .native_supervised import recipe_lineage
                recipe_lineage(prior)
                _scheduling_controls(prior,declaration,plan_path,record)
                write_json_durable(path,prior)
                save(verification,plan_path,declaration,handoff,prior)
                return path
            cursor=json.loads((out/'cycle.json').read_text()) if (out/'cycle.json').exists() else {}
            if cursor.get('phase')=='sft' or cursor.get('pending_rl_round'):
                _scheduling_controls(prior,declaration,plan_path,record,pending=True)
                write_json_durable(path,prior)
                return path
            from .native_supervised import bind_recipe_lineage
            bind_recipe_lineage(prior,lineage);_scheduling_controls(prior,declaration,plan_path,record);write_json_durable(path,prior)
            return path
        cursor=json.loads((out/'cycle.json').read_text()) if (out/'cycle.json').exists() else {}
        if cursor.get('phase')=='sft' or cursor.get('pending_rl_round'):
            _scheduling_controls(prior,declaration,plan_path,record,pending=True)
            write_json_durable(path,prior)
            return path  # Retain sealed cases/source/bootstrap identity; repair stop cadence only.
    args=SimpleNamespace(**inputs,model=declaration['model'],native_participant=None,semantic_adapter=None,
        semantic_refresh_config=None,checkpoint=_bound(record['checkpoint']),base=Path(declaration['base']),
        production_stage=declaration['production_stage'],runtime=runtime,out=out/'production',config=path,
        port=declaration['port'],coefficient_policy=None,coefficient_optimizer=None,coefficient_task_eval=None,
        verified_native_checkpoint=(record['checkpoint'],saved))
    config=configure_native(args)
    args.verified_native_checkpoint=None
    del saved  # No mapped tensor state survives this invocation.
    from .native_supervised import bind_recipe_lineage
    bind_recipe_lineage(config,lineage)
    _scheduling_controls(config,declaration,plan_path,record)
    config['raw_model_id']=actual
    configure_provider(config,declaration,{},owner,path)
    save(verification,plan_path,declaration,handoff,config)
    return path


def _publish_checkpoint(config,checkpoint):
    """Only the drained worker's actual checkpoint becomes the sole current parent."""
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=True);path=out/'state.json'
    with (out/'controller.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .production import initial_controller_state
        state=json.loads(path.read_text()) if path.exists() else initial_controller_state(config)
        state.update(checkpoint=str(_bound(checkpoint)),phase='collect',actor_policy_pending_load=checkpoint['sha256'])
        write_json_durable(path,state)


def _current_checkpoint(config):
    path=Path(config['out'])/'state.json'
    return binding(json.loads(path.read_text())['checkpoint'] if path.exists() else config['checkpoint'])


def _progress(checkpoint):
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..train.native_continuation import progress_obligations
    path=_bound(checkpoint)
    from ..train.native_operation_report import saved_progress
    summary=saved_progress(checkpoint)
    if summary is not None:return summary
    state=CheckpointPolicy.load(path)
    return progress_obligations(state['training']['sft_progress'])


def _restore_recipe_alias(source,copy):
    """Install verified immutable bytes without replacing another writer's file."""
    if source['sha256']!=copy['sha256']:raise ValueError('raw recipe alias changed immutable source bytes')
    target=Path(source['path'])
    if not target.exists():
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name('.raw-immutable-'+source['sha256'])
        write_bytes_durable(temporary,_bound(copy).read_bytes())
        try:
            # The newly copied payload is immutable before linking; this is
            # never a link to a mutable checkpoint/source authority.
            try:os.link(temporary,target)
            except FileExistsError:pass
            descriptor=os.open(target.parent,os.O_DIRECTORY)
            try:os.fsync(descriptor)
            finally:os.close(descriptor)
        finally:temporary.unlink(missing_ok=True)
    _bound(source)


def student_cycle(plan_path,declaration,handoff,directory,*,stop):
    """Resume exact per-operation outputs; every source remains a carried obligation."""
    from .native_rounds import round_job
    from .native_supervised import continuation_config,stage_pending,MANDATORY_STAGE_KINDS
    from ..train.recipe import resolve_recipe_data
    directory=Path(directory)
    try:
        config_path=configure_model(plan_path,declaration,handoff,directory)
        config=resolve_recipe_data(config_path)
    except Exception as error:
        raise StudentConfigurationError(str(error)) from error
    cursor_path=directory/'cycle.json'
    cursor=json.loads(cursor_path.read_text()) if cursor_path.exists() else {'cycle':0,'phase':'rl','next_stage':0}
    cycle=cursor['cycle'];work=directory/'cycles'/f'{cycle:06d}'
    if cursor['phase']=='rl':
        if stop.requested:return cursor
        production_state=Path(config['out'])/'state.json'
        state=json.loads(production_state.read_text()) if production_state.exists() else {'round':0}
        if 'pending_rl_round' not in cursor:
            cursor['pending_rl_round']=state['round']+1
            write_json_durable(cursor_path,cursor)
        result=(state if state['round']>=cursor['pending_rl_round'] else
            round_job(config_path,config,work/'rl-bridge',stop))
        if stop.requested:return cursor
        if result.get('phase')!='collect':raise RuntimeError('raw RL operation needs its actual drained published boundary')
        cursor.update(phase='sft',next_stage=0,current_checkpoint=_current_checkpoint(config))
        cursor.pop('pending_rl_round',None)
        write_json_durable(cursor_path,cursor)
    recipe=resolve_recipe_data(config['production_recipe'])
    stages=[stage for stage in recipe['stages']
            if stage['kind'] in MANDATORY_STAGE_KINDS and stage.get('admitted') is not False]
    foundation=[stage for stage in stages if stage['id']==config['production_stage']]
    if len(foundation)!=1:raise ValueError('raw continuation lost its actual foundation recipe stage')
    # The actual foundation trainer is last, so its receipt certifies the current
    # post-preference/RL weights rather than a stale earlier parent.
    stages=[stage for stage in stages if stage['id']!=foundation[0]['id']]+foundation
    for index in range(cursor.get('next_stage',0),len(stages)):
        if stop.requested:return cursor
        stage=stages[index];checkpoint=_current_checkpoint(config)
        jobdir=work/'supervised'/f'{index:03d}'
        continuation,sources=continuation_config(config,stage,checkpoint=str(_bound(checkpoint)),adapter=None,out=jobdir)
        parameters=continuation['trainer_parameters']
        _supervised_quantum(config,parameters)
        is_foundation=stage['id']==config['production_stage']
        # Once the cohort is ready, ordinary mandatory deficits still receive
        # useful finite quanta; a final existing foundation qualification below
        # binds the resulting actual weights before rendezvous.
        parameters['communication_foundation']=bool(is_foundation and not cursor.get('foundation_qualified'))
        if is_foundation and config.get('resident_data_directory'):
            parameters['resident_data_directory']=config['resident_data_directory']
        continuation['raw_foundation_qualification']=parameters['communication_foundation']
        export={'models':{'raw':{'mandatory_sft':_progress(checkpoint)}}}
        pending=stage_pending(export,'raw',stage['id'],sources,parameters.get('epochs',1),
            objective=continuation.get('offline_objective_binding'))
        if pending or is_foundation:
            _supervised_job(continuation,jobdir,config,cursor,stop)
            if stop.requested:return cursor
        cursor.update(next_stage=index+1,current_checkpoint=_current_checkpoint(config))
        write_json_durable(cursor_path,cursor)
    if cursor.get('foundation_qualified'):
        # This is the same shared trainer, not a fabricated copy of an old
        # certificate. It may do zero updates when the exact cohort is complete,
        # but rechecks runtime/task behavior against current post-RL weights.
        checkpoint=_current_checkpoint(config);jobdir=work/'foundation-current'
        continuation,_=continuation_config(config,foundation[0],checkpoint=str(_bound(checkpoint)),adapter=None,out=jobdir)
        _supervised_quantum(config,continuation['trainer_parameters'])
        continuation['trainer_parameters']['communication_foundation']=True
        if config.get('resident_data_directory'):
            continuation['trainer_parameters']['resident_data_directory']=config['resident_data_directory']
        continuation['raw_foundation_qualification']=True
        _supervised_job(continuation,jobdir,config,cursor,stop)
        if stop.requested:return cursor
    cursor.update(cycle=cycle+1,phase='rl',next_stage=0,current_checkpoint=_current_checkpoint(config))
    write_json_durable(cursor_path,cursor)
    return cursor


def _supervised_job(continuation,jobdir,config,cursor,stop):
    from .native_rounds import round_job
    path=Path(jobdir)/'config.json';path.parent.mkdir(parents=True,exist_ok=True)
    write_json_durable(path,continuation)
    result=round_job(path,continuation,Path(jobdir)/'bridge',stop)
    if stop.requested:return
    pending=result.get('pending_native_checkpoint')
    if not pending:raise RuntimeError('raw supervised quantum returned no exact drained checkpoint')
    checkpoint=binding(pending);report_path=Path(pending).parent/'native-training-receipt.json'
    _publish_checkpoint(config,checkpoint)
    cursor.pop('readiness',None)
    if report_path.is_file():
        report=json.loads(report_path.read_text())
        if report['checkpoint']['sha256']!=checkpoint['sha256']:
            raise ValueError('raw supervised readiness belongs to another checkpoint')
        if continuation.get('raw_foundation_qualification') and (report.get('foundation_ready') or report.get('completed')):
            cursor.update(readiness=binding(report_path),foundation_qualified=True)
        elif (continuation.get('raw_foundation_qualification')
                and (report.get('communication_foundation') or {}).get('complete')):
            raise RuntimeError('raw current foundation runtime/task qualification needs repair; exact checkpoint retained')
        adopted=report.get('resident_adoption')
        if adopted and continuation.get('resident_data_directory',continuation['trainer_parameters'].get('resident_data_directory')):
            from ..train.native_resident import current
            source=Path(adopted['directory'])
            reference=result.get('resident_current')
            recipe_copy=result.get('resident_recipe') or {}
            if reference is None or not recipe_copy.get('source') or not recipe_copy.get('copy'):
                raise ValueError('raw SFT worker omitted its final immutable resident source snapshots')
            state=current(_bound(reference).parent)
            if not state or state['generation']!=adopted['generation'] or state['payload']['recipe']!=report['recipe']:
                raise ValueError('raw continuation changed its actual committed resident source generation')
            expected={'path':report['recipe']['path'],'sha256':report['recipe']['file_sha256']}
            if recipe_copy['source']!=expected or recipe_copy['copy']['sha256']!=expected['sha256']:
                raise ValueError('raw resident recipe snapshot changed its exact original bytes')
            _restore_recipe_alias(expected,recipe_copy['copy'])
            snapshot=Path(jobdir)/'resident-snapshot';snapshot.mkdir(parents=True,exist_ok=True)
            write_bytes_durable(snapshot/'current.json',_bound(reference).read_bytes())
            cursor['resident_current']=binding(snapshot/'current.json')
            # Follow the already validated shared resident policy. No new rows
            # earn credit here, and pending.json remains with its existing owner.
            config['production_recipe']=state['payload']['recipe']['path']
            from .native_supervised import bind_recipe_lineage
            bind_recipe_lineage(config,state['payload']['plan']['recipe_sha256'])
            from ..train.recipe import resolve_recipe_data
            recipe=resolve_recipe_data(config['production_recipe'])
            stage=next(stage for stage in recipe['stages'] if stage['id']==config['production_stage'])
            from .native_supervised import continuation_config
            refreshed,_=continuation_config(config,stage,checkpoint=checkpoint['path'],adapter=None,out=config['out'])
            for key in ('seed_command','seed_native_graphs','held_native_graphs','pending_native_sources',
                        'continued_native_graphs','continued_held_native_graphs','objective_spec'):
                if key in refreshed:config[key]=refreshed[key]
                else:config.pop(key,None)
            config['seed_trainer_parameters']={**refreshed['trainer_parameters'],
                'resident_data_directory':str(source),'refresh_at_checkpoint':True}
            write_json_durable(Path(config['out']).parent/'config.json',config)


def _release_requirement(declaration,value,*,verified_handoff=None):
    """Recheck the existing resident append contract, preserving fixed foundation."""
    requirement={**declaration['handoff_requirement'],'trainer_exit':None,'trainer_launch':None,'resident_adoption':None}
    transition_ref=declaration.get('drained_recipe_handoff')
    transition_handoff=None;retired_cursor=None;transition=None
    if declaration.get('runtime_handoff'):
        from .native_runtime_handoff import verify_runtime_handoff
        runtime=json.loads(_bound(declaration['runtime_handoff']).read_text())
        bootstrap=declaration.get('handoff_receipt') or value.get('handoff') or runtime['bootstrap']
        proof=verify_runtime_handoff(declaration['runtime_handoff'],bootstrap=bootstrap)
        transition_ref=proof['recipe_handoff'];transition_handoff=proof['handoff']
        retired_cursor=json.loads(_bound(proof['snapshots']['cursor']).read_text()).get('resident_current')
    if transition_ref:
        from .native_raw_handoff import verify_drained_recipe_handoff
        transition=json.loads(_bound(transition_ref).read_text())
        transition=verify_drained_recipe_handoff(transition_ref,handoff=transition['handoff'],
            **({'verified_handoff':verified_handoff} if verified_handoff is not None and transition_handoff is None else {}))
        from ..train.recipe import recipe_identity_sha256
        requirement['recipe_sha256']=recipe_identity_sha256(_bound(transition['recipe']))
        requirement['recipe_lineage_sha256']=json.loads(_bound(transition['new_plan']).read_text())['recipe_sha256']
    reference=value.get('resident_current')
    # A later handoff explicitly retired this exact historical generation.
    # Keep the cursor receipt, but never reinterpret it as the new live plan.
    if transition_handoff is not None and reference==retired_cursor:return requirement
    if reference is None:return requirement
    from ..train.native_resident import current,_contract
    from ..train.recipe import resolve_recipe_data,resolve_stage_inputs,recipe_identity_sha256
    path=_bound(reference);state=current(path.parent)
    if not state:raise ValueError('raw release lost its exact committed resident source')
    payload=state['payload'];stage=payload['stage_id']
    lineage=payload['plan']['recipe_sha256']
    if requirement.get('recipe_lineage_sha256',lineage)!=lineage:
        raise ValueError('raw release resident plan changed its preserved recipe lineage')
    requirement['recipe_lineage_sha256']=lineage
    if stage not in requirement['stage_ids']:raise ValueError('raw release adopted another production stage')
    bindings=[payload['prior_recipe'],*[item['recipe'] for item in state['history']]]
    start=next((index for index,item in enumerate(bindings) if item['sha256']==requirement['recipe_sha256']),None)
    def plan(reference):
        recipe_path=_bound({'path':reference['path'],'sha256':reference.get('file_sha256',reference['sha256'])})
        if reference.get('lineage_sha256',reference['sha256'])!=lineage:
            raise ValueError('raw resident recipe history changed its preserved lineage')
        if recipe_identity_sha256(recipe_path)!=reference['sha256']:raise ValueError('raw resident recipe identity changed')
        recipe=resolve_recipe_data(recipe_path)
        inputs={item['id']:({} if item.get('admitted') is False else resolve_stage_inputs(recipe,item,{})) for item in recipe['stages']}
        return {**payload['plan'],'recipe':recipe,'stage_inputs':inputs}
    if start is None:
        if transition is None:
            raise ValueError('raw release resident history omitted its declared original recipe')
        old_plan=json.loads(_bound(transition['old_plan']).read_text())
        installed=json.loads(_bound(transition['new_plan']).read_text())
        actual_prior=plan(bindings[0])
        previous=installed.get('resident_adoptions',[])
        adoptions=payload['plan'].get('resident_adoptions',[])
        from ..common.hashing import canonical_json_sha256_hex
        suffix=adoptions[len(previous):]
        if (actual_prior['recipe']!=old_plan['recipe'] or actual_prior['stage_inputs']!=old_plan['stage_inputs']
                or installed['recipe_sha256']!=lineage or installed['code']!=payload['plan']['code']
                or installed.get('device')!=payload['plan'].get('device')
                or payload['plan'].get('handoffs',[])!=installed.get('handoffs',[])
                or adoptions[:len(previous)]!=previous or not suffix
                or suffix[0]['predecessor']!=canonical_json_sha256_hex(installed)
                or [item['recipe'] for item in suffix]!=bindings[1:]
                or any(item['stage_id']!=stage for item in suffix)):
            raise ValueError('raw release omitted recipe lacks its exact installed-plan adoption bridge')
        prior=installed
        remaining=bindings[1:]
    else:
        prior=plan(bindings[start]);remaining=bindings[start+1:]
    for reference in remaining:
        proposed=plan(reference);_contract(prior,proposed,stage);prior=proposed
    if payload['recipe']!=bindings[-1] or prior['recipe']!=payload['plan']['recipe'] or prior['stage_inputs']!=payload['plan']['stage_inputs']:
        raise ValueError('raw release resident history differs from its actual current source plan')
    requirement['recipe_sha256']=payload['recipe']['sha256']
    if value.get('readiness'):
        receipt=json.loads(_bound(value['readiness']).read_text())
        adopted=receipt.get('resident_adoption') or {}
        if adopted.get('generation')!=state['generation'] or adopted.get('history')!=state['history']:
            raise ValueError('raw release readiness omitted the actual current resident history')
    return requirement


def validate_release(path,plan_path,*,allow_live=False):
    from ..train.native_resident import owner_alive
    record=json.loads(Path(path).read_text());plan=json.loads(Path(plan_path).read_text())
    if (record.get('schema')!=SCHEMA or record.get('plan_sha256')!=sha256_file_hex_cached(plan_path)
            or record.get('released') is not True):return None
    if not allow_live and owner_alive(record.get('owner') or {}):return None
    models=record.get('models') or {}
    if set(models)!={model['model'] for model in plan['models']}:raise ValueError('raw release omitted an actual student owner')
    result={}
    from ..train.native_readiness import validate_ready
    for declaration in plan['models']:
        value=models[declaration['model']]
        config=json.loads(_bound(value['config']).read_text())
        if (config.get('runtime_binding',config.get('bootstrap_binding')) or {}).get('plan_sha256')!=record['plan_sha256']:
            return None  # a sealed older operation must finish before live-plan adoption
        verify_handoff(value['handoff'],requirement=declaration['handoff_requirement'])
        current=json.loads(_bound(value['production_state']).read_text())
        if binding(current['checkpoint'])!=value['checkpoint']:
            raise ValueError('raw release substituted its actual last published checkpoint')
        requirement=_release_requirement(declaration,value)
        from ..train.checkpoint_policy import CheckpointPolicy
        from .native_raw_policy import require_raw_training
        saved=CheckpointPolicy.load(_bound(value['checkpoint']));require_raw_training(saved)
        if saved['model_config']['architecture']+':'+saved['model_config']['base_inventory_sha256']!=requirement['model_id']:
            raise ValueError('raw release substituted another student/base optimizer')
        del saved
        ready=validate_ready(_bound(value['readiness']),requirement)
        if ready is None or any(ready['checkpoint'][key]!=value['checkpoint'][key] for key in ('path','sha256')):
            raise ValueError('raw release has no genuine current complete communication-foundation readiness')
        result[requirement['model_id']]=ready
    return result


def run(plan_path,out,*,cycles=None,stop=None,poll_seconds=60):
    from ..train.loop import StopSignal
    from ..train.native_resident import process_owner
    from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
    plan_path=Path(plan_path).resolve(strict=True);plan=json.loads(plan_path.read_text())
    if (plan.get('schema')!='natlang.native-raw-rl-bootstrap/1' or not plan.get('models')
            or len({row['model'] for row in plan['models']})!=len(plan['models'])
            or not 0<poll_seconds<=60):raise ValueError('raw bootstrap requires distinct declared students and bounded polling')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    owned=stop is None;stop=StopSignal().install() if owned else stop
    try:
        with (out/'controller.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state_path=out/'state.json'
            if state_path.exists() and validate_release(state_path,plan_path) is not None:return json.loads(state_path.read_text())
            state={'schema':SCHEMA,'plan_sha256':sha256_file_hex_cached(plan_path),'owner':process_owner(),
                'released':False,'models':{},'production_complete':False}
            counts={row['model']:0 for row in plan['models']}
            eligible=set();rendezvous=False;active={};blocked={}
            with ThreadPoolExecutor(max_workers=len(plan['models'])) as pool:
                try:
                    while True:
                        updated=json.loads(plan_path.read_text())
                        if (updated.get('schema')!=plan['schema'] or
                                {row['model']:row['handoff_requirement']['model_id'] for row in updated['models']}!=
                                {row['model']:row['handoff_requirement']['model_id'] for row in plan['models']}
                                or {row['model']:{key:row['handoff_requirement'].get(key) for key in
                                    ('foundation_sources','foundation_epochs','readiness_boundary')} for row in updated['models']}!=
                                   {row['model']:{key:row['handoff_requirement'].get(key) for key in
                                    ('foundation_sources','foundation_epochs','readiness_boundary')} for row in plan['models']}):
                            raise ValueError('live raw plan cannot silently replace a student optimizer owner')
                        plan=updated
                        state['plan_sha256']=sha256_file_hex_cached(plan_path)
                        models={row['model']:row for row in plan['models']}
                        if not stop.requested and not rendezvous:
                            for name,model in models.items():
                                if name in active or (cycles is not None and counts[name]>=cycles):continue
                                if blocked.get(name)==_configuration_signature(model,out/name):continue
                                blocked.pop(name,None)
                                state.get('configuration_errors',{}).pop(name,None)
                                witness=model.get('handoff_receipt')
                                if witness is None:
                                    candidate=out/'handoffs'/(name+'.json')
                                    if candidate.is_file():witness=binding(candidate)
                                if witness is not None:
                                    # Published eligibility is informational while this
                                    # student's optimizer is owned by another cycle.
                                    active[name]=(pool.submit(student_cycle,plan_path,model,witness,out/name,stop=stop),witness,model)
                        write_json_durable(state_path,state)
                        if not active:
                            if rendezvous and eligible!=set(models):
                                rendezvous=False
                            if rendezvous:
                                state['released']=True;write_json_durable(state_path,state)
                                if validate_release(state_path,plan_path,allow_live=True) is not None:return state
                                state['released']=False;rendezvous=False;eligible.clear()
                            if stop.requested or (cycles is not None and all(counts[name]>=cycles or name in blocked for name in models)):
                                write_json_durable(state_path,state);return state
                            time.sleep(poll_seconds);continue
                        finished,_=wait([value[0] for value in active.values()],timeout=poll_seconds,return_when=FIRST_COMPLETED)
                        for name,(future,witness,submitted) in list(active.items()):
                            if future not in finished:continue
                            try:
                                cursor=future.result()
                            except StudentConfigurationError as error:
                                del active[name];eligible.discard(name)
                                blocked[name]=_configuration_signature(submitted,out/name)
                                cause=error.__cause__ or error
                                state.setdefault('configuration_errors',{})[name]={
                                    'input_sha256':blocked[name],'phase':'configuration',
                                    'error_type':type(cause).__name__,'error':str(cause)}
                                write_json_durable(state_path,state)
                                continue
                            del active[name];counts[name]+=1
                            config=json.loads((out/name/'config.json').read_text())
                            state['models'][name]={'handoff':witness,'checkpoint':_current_checkpoint(config),
                                'config':binding(out/name/'config.json'),
                                'production_state':binding(Path(config['out'])/'state.json'),
                                **({'resident_current':cursor['resident_current']} if cursor.get('resident_current') else {}),
                                **({'readiness':cursor['readiness']} if cursor.get('readiness') else {})}
                            if cursor.get('readiness'):eligible.add(name)
                            else:eligible.discard(name)
                        # Do not idle the first eligible student. Only after every
                        # student has reached an actual eligible boundary do we stop
                        # resubmitting and drain their currently owned operations.
                        if eligible==set(models):rendezvous=True
                except BaseException:
                    # The executor drains only after every owned worker has
                    # observed this shared stop, including config/IO failures.
                    stop.requested=True
                    state['released']=False
                    try:write_json_durable(state_path,state)
                    except Exception:pass  # Preserve the original failure, including storage errors.
                    raise
            return state
    finally:
        if owned:stop.restore()
