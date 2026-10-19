"""Per-student whole-recipe/RL continuation before the shared-codec rendezvous.

The recipe/curriculum declares work. Exact plumbing serializes existing workers,
commits current checkpoint ownership and releases only genuine latest readiness.
"""
from __future__ import annotations
import fcntl
import json
import os
import sys
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


def _validate_stage_quanta(value):
    if (not isinstance(value,dict) or any(not isinstance(name,str)or not name
            or type(updates)is not int or updates<=0 for name,updates in value.items())):
        raise ValueError('supervised stage quanta require stage IDs and positive integer update cadences')
    return dict(value)


def _supervised_quantum(config,parameters):
    # Storage checkpoints preserve progress; RAW scheduling yields only at an
    # explicit relative quantum (or completed stage/supervisor interruption).
    parameters['refresh_at_checkpoint']=False
    mapping=_validate_stage_quanta(config.get('supervised_stage_quantum_updates',{}))
    stage=parameters.get('stage_id',config.get('production_stage'))
    quantum=mapping.get(stage,config.get('supervised_quantum_updates'))
    if quantum is not None:
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
        parameters['refresh_after_updates']=quantum


def _scheduling_controls(config,declaration,plan_path,record,*,pending=False):
    # Pending collection retains its sealed packet. Only subsequent supervised
    # stop cadence may change; it is independently bound to the declaring plan.
    if 'rows_per_round' in declaration and not pending:
        config['rows_per_round']=declaration['rows_per_round']
    if 'supervised_quantum_updates' in declaration:
        config['supervised_quantum_updates']=declaration['supervised_quantum_updates']
        config['supervised_quantum_control']={'plan':binding(plan_path),
            'updates':declaration['supervised_quantum_updates']}
    if 'supervised_stage_quantum_updates' in declaration:
        mapping=_validate_stage_quanta(declaration['supervised_stage_quantum_updates'])
        from ..train.recipe import resolve_recipe_data
        ids={stage['id']for stage in resolve_recipe_data(config['production_recipe'])['stages']}
        if set(mapping)-ids:raise ValueError('supervised stage quantum names an undeclared recipe stage')
        config['supervised_stage_quantum_updates']=mapping
        config['supervised_stage_quantum_control']={'plan':binding(plan_path),'updates':mapping}
    _seed_quantum(config,record)
    return config


def configure_model(plan_path,declaration,handoff,out,*,verified_handoff=None):
    from .production import configure_native
    from .native_rounds import configure_provider
    for key in ('rows_per_round','supervised_quantum_updates'):
        if key in declaration and (type(declaration[key]) is not int or declaration[key]<=0):
            raise ValueError(key+' must be a positive integer')
    if 'supervised_stage_quantum_updates' in declaration:_validate_stage_quanta(declaration['supervised_stage_quantum_updates'])
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
            if declaration.get('completed_supervised_adaptation'):
                from .native_completed_supervised import install_resident_plan
                install_resident_plan(declaration['completed_supervised_adaptation'],prior)
            return path
        if declaration.get('completed_supervised_adaptation'):
            from .native_completed_supervised import configure
            return configure(plan_path,declaration,handoff,path)
        if declaration.get('runtime_handoff'):
            from .native_runtime_handoff import configure_runtime_handoff
            return configure_runtime_handoff(plan_path,declaration,handoff,path,verified_handoff=verified_handoff)
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


def _published_checkpoint_state(state,checkpoint):
    """One publication rule for ordinary bridge and owned-operation commits."""
    return {**state,'checkpoint':str(_bound(checkpoint)),'phase':'collect',
            'actor_policy_pending_load':checkpoint['sha256']}


def _supervised_stage_inventory(recipe,foundation_id):
    from .native_supervised import MANDATORY_STAGE_KINDS
    stages=[stage for stage in recipe['stages']
            if stage['kind'] in MANDATORY_STAGE_KINDS and stage.get('admitted') is not False]
    foundation=[stage for stage in stages if stage['id']==foundation_id]
    if len(foundation)!=1 or len({stage['id']for stage in stages})!=len(stages):
        raise ValueError('raw continuation lost its unique actual foundation recipe stage')
    return [stage for stage in stages if stage['id']!=foundation_id]+foundation


_OWNED_RESTORE_OBSERVER_SHA256='7b1f69d53cd7a8690e3d32940337c6c56f040fca8c0ad2b2dfee93d431c63e9e'


def import_current_owner_checkpoint(config_path,cursor_path,selected_checkpoint,saved_state,authority,out,*,recovery_reason=None,owned_controller_fd=None):
    """Publish an already-loaded current-owner checkpoint into the existing RAW recipe.

    This is a checkpoint import, not a trainer handoff or readiness claim.  The
    in-process entry supplies the exact checkpoint/state pair it already owns;
    this function never deserializes a checkpoint.  Only the two current
    checkpoint references and checkpoint-specific readiness markers change.
    """
    import contextlib
    from ..common.hashing import canonical_json_sha256_hex as digest
    from ..train.native_resident import code_inventory,owner_alive
    from .native_raw_handoff import observe,process_argv,same_process

    def publication(state,cursor,selected):
        next_state=_published_checkpoint_state(state,selected)
        next_state.pop('readiness',None)
        next_cursor={**cursor,'current_checkpoint':selected}
        next_cursor.pop('readiness',None);next_cursor.pop('foundation_qualified',None)
        return next_state,next_cursor

    def validate_intent(value,request,paths):
        if (not isinstance(value,dict) or value.get('schema')!='natlang.native-raw-current-owner-import/1'
                or value.get('request')!=request or value.get('whole_recipe_completed') is not False
                or digest(value.get('authority'))!=request.get('authority_sha256')
                or value.get('selected_checkpoint')!=selected or value.get('selected_global_step')!=global_step
                or value.get('recovery_reason')!=recovery_reason
                or set(value.get('before',{}))!=set(paths) or set(value.get('snapshots',{}))!=set(paths)
                or set(value.get('candidates',{}))!=set(paths)):
            raise ValueError('current-owner import durable intent changed its exact request or parent')
        originals={}
        for name,path in paths.items():
            before=value['before'][name];snapshot=value['snapshots'][name]
            if (before.get('path')!=str(path)
                    or Path(snapshot['path']).resolve()!=directory/('original-'+name+'.json')
                    or Path(value['candidates'][name]['path']).resolve()!=directory/('candidate-'+name+'.json')):
                raise ValueError('current-owner import intent substituted its transaction artifact paths')
            snap_path=_bound(snapshot)
            if binding(snap_path)['sha256']!=before.get('sha256'):
                raise ValueError('current-owner import snapshot does not preserve its exact predecessor')
            originals[name]=json.loads(snap_path.read_text())
        if not isinstance(originals['state'],dict) or not isinstance(originals['cursor'],dict):
            raise ValueError('current-owner import predecessor snapshots are malformed')
        expected=publication(originals['state'],originals['cursor'],selected)
        for name,value_expected in zip(('state','cursor'),expected):
            candidate=_bound(value['candidates'][name])
            if json.loads(candidate.read_text())!=value_expected:
                raise ValueError('current-owner import candidate changed fields outside its narrow pointer update')
        current_before=binding(originals['state']['checkpoint'])
        if value.get('current_checkpoint')!=current_before:
            raise ValueError('current-owner import intent lost its exact predecessor checkpoint binding')
        return value

    def verify_controller_fd(fd,path):
        if type(fd)is not int or fd<0:raise ValueError('current-owner import controller descriptor is invalid')
        if not path.is_file() or not os.path.isfile(f'/proc/self/fd/{fd}'):
            raise ValueError('current-owner import descriptor is not the existing canonical controller lock')
        opened=os.fstat(fd);actual=path.stat()
        if opened.st_dev!=actual.st_dev or opened.st_ino!=actual.st_ino:
            raise ValueError('current-owner import descriptor is not the existing canonical controller lock')
        info=Path('/proc/self/fdinfo',str(fd)).read_text().splitlines()
        locks=[line for line in info if line.startswith('lock:') and 'FLOCK' in line]
        if len(locks)!=1 or 'WRITE' not in locks[0]:
            raise ValueError('current-owner import descriptor does not hold the existing exclusive controller lease')

    config_path=Path(config_path).resolve(strict=True);cursor_path=Path(cursor_path).resolve(strict=True)
    out=Path(out).resolve();config=json.loads(config_path.read_text())
    root=config_path.parent;state_path=Path(config['out']).resolve()/'state.json'
    if cursor_path!=root/'cycle.json' or Path(config['out']).resolve().parent!=root:
        raise ValueError('current-owner import lost its exact raw controller namespace')
    required={'schema','owner','wrapper','process_argv','config','package','wrapper_configuration',
        'effective_argv_file','effective_argv_binding','selected_checkpoint','saved_state'}
    if not isinstance(authority,dict) or set(authority)!=required or authority.get('schema')!='natlang.native-raw-current-owner/2':
        raise ValueError('current-owner import needs the complete explicit in-process authority')
    captured_process_argv=authority['process_argv']
    if not isinstance(captured_process_argv,list) or not captured_process_argv or any(type(x)is not str for x in captured_process_argv):
        raise ValueError('current-owner import process argv must be an exact string vector')
    if binding(config_path)!=authority['config']:
        raise ValueError('current-owner import configuration changed after owner capture')
    owner=authority['owner']
    observed=observe(owner.get('pid')) if isinstance(owner,dict) else None
    if not same_process(owner,observed) or observed.get('argv_hex')!=owner.get('argv_hex'):
        raise ValueError('current-owner process identity changed after capture')
    if process_argv(observed)!=captured_process_argv:
        raise ValueError('current-owner process argv differs from the live wrapper')
    script=next((Path(value).resolve() for value in captured_process_argv if value.endswith('.py') and Path(value).is_file()),None)
    if script is None or binding(script)!=authority['wrapper']:
        raise ValueError('current-owner wrapper bytes differ from the live entry')
    gate_spec_path=_bound(authority['wrapper_configuration'])
    if binding(gate_spec_path)!=authority['wrapper_configuration']:
        raise ValueError('current-owner wrapper gate specification changed after capture')
    gate_spec=json.loads(gate_spec_path.read_text())
    if gate_spec.get('effective_trainer_argv')!=authority['effective_argv_binding']:
        raise ValueError('current-owner gate specification does not bind its serialized trainer arguments')
    args_path=_bound(authority['effective_argv_binding'])
    trainer_args=json.loads(args_path.read_text())
    if not isinstance(trainer_args,list) or any(type(x)is not str for x in trainer_args):
        raise ValueError('current-owner serialized trainer arguments are malformed')
    if gate_spec.get('trainer_arguments')!=trainer_args:
        raise ValueError('current-owner gate specification arguments differ from the serialized argument list')
    effective_path=_bound(authority['effective_argv_file'])
    effective_argv=json.loads(effective_path.read_text())
    expected_effective=[sys.executable,'-u','-m','natlang_neuralese.train.native_state_sft',*trainer_args]
    if effective_argv!=expected_effective:
        raise ValueError('current-owner serialized effective trainer command differs from the gate-bound arguments')
    package=Path(authority['package']['path']).resolve(strict=True)
    if authority['package'].keys()!={'path','code'} or package!=Path(__file__).resolve().parents[1]:
        raise ValueError('current-owner package is not the exact package running this import')
    if code_inventory(package)!=authority['package']['code']:
        raise ValueError('current-owner installed package inventory changed')
    runtime_root=Path(gate_spec.get('runtime','')).resolve(strict=True)
    runtime_package=runtime_root/'training/neuralese/natlang_neuralese'
    if (runtime_package.resolve()!=package
            or gate_spec.get('runtime_code_sha256')!=digest(code_inventory(runtime_package))):
        raise ValueError('current-owner gate specification does not bind the running runtime package')
    selected=authority['selected_checkpoint']
    if selected_checkpoint!=selected:
        raise ValueError('current-owner selected checkpoint argument differs from its authority binding')
    selected_path=_bound(selected)
    if selected.get('bytes')!=selected_path.stat().st_size:
        raise ValueError('current-owner full checkpoint byte binding changed')
    summary=authority['saved_state']
    if (not isinstance(summary,dict) or set(summary)!={'checkpoint','global_step','model_id'}
            or summary['checkpoint']!=selected):
        raise ValueError('current-owner saved-state summary is not bound to its selected full checkpoint')
    training=saved_state.get('training') if isinstance(saved_state,dict) else None
    model=saved_state.get('model_config') if isinstance(saved_state,dict) else None
    global_step=(training.get('global_step',training.get('step')) if isinstance(training,dict) else None)
    model_id=(model.get('architecture','')+':'+model.get('base_inventory_sha256','') if isinstance(model,dict) else None)
    if type(global_step)is not int or type(summary['global_step'])is not int or global_step!=summary['global_step'] or model_id!=summary['model_id']:
        raise ValueError('current-owner loaded state differs from its checkpoint identity summary')
    if config.get('raw_model_id')!=model_id:
        raise ValueError('current-owner checkpoint belongs to another RAW student/base')
    from .native_raw_policy import require_raw_training
    require_raw_training(saved_state)
    def parent_matches(reference):
        return (isinstance(reference,dict) and {'path','sha256'}<=set(reference)
            and set(reference)<={'path','sha256','bytes'}
            and reference['path']==selected.get('path') and reference['sha256']==selected.get('sha256')
            and ('bytes' not in reference or reference['bytes']==selected.get('bytes')))
    def verify_execution_bindings():
        live=observe(owner.get('pid'))
        if not same_process(owner,live) or live.get('argv_hex')!=owner.get('argv_hex') or process_argv(live)!=captured_process_argv:
            raise ValueError('current-owner process changed during import')
        if (binding(config_path)!=authority['config'] or binding(script)!=authority['wrapper']
                or binding(gate_spec_path)!=authority['wrapper_configuration']
                or binding(args_path)!=authority['effective_argv_binding']
                or binding(effective_path)!=authority['effective_argv_file']
                or code_inventory(package)!=authority['package']['code']
                or digest(code_inventory(runtime_package))!=gate_spec.get('runtime_code_sha256')):
            raise ValueError('current-owner execution bindings changed during import')
        latest_gate=json.loads(gate_spec_path.read_text())
        latest_args=json.loads(args_path.read_text())
        latest_command=json.loads(effective_path.read_text())
        if (latest_gate.get('effective_trainer_argv')!=authority['effective_argv_binding']
                or latest_gate.get('trainer_arguments')!=latest_args
                or latest_command!=[sys.executable,'-u','-m','natlang_neuralese.train.native_state_sft',*latest_args]
                or not parent_matches(latest_gate.get('parent'))):
            raise ValueError('current-owner trainer command or selected parent changed during import')
    # The configured arguments are already prepared before checkpoint decoding.
    # Bind their continuation target and output to the exact selected checkpoint
    # and RAW namespace instead of trusting a self-consistent command file.
    gate_parent=gate_spec.get('parent')
    if (trainer_args.count('--continue-from')!=1 or trainer_args.count('--out')!=1
            or trainer_args.count('--stage-id')!=1
            or Path(trainer_args[trainer_args.index('--continue-from')+1]).resolve()!=Path(selected_path).resolve()
            or Path(trainer_args[trainer_args.index('--out')+1]).resolve()!=gate_spec_path.parent/'learner'
            or not parent_matches(gate_parent)
            or gate_spec.get('out')!=trainer_args[trainer_args.index('--out')+1]
            or gate_spec.get('stage_id')!=trainer_args[trainer_args.index('--stage-id')+1]):
        raise ValueError('current-owner effective trainer is not the selected RAW continuation')
    request={'config':binding(config_path),'selected_checkpoint':selected,
        'authority_sha256':digest(authority),'recovery_reason':recovery_reason}
    def same_semantics(prior):
        return all(prior.get(key)==request.get(key) for key in ('config','selected_checkpoint','recovery_reason'))
    directory=out.parent/(out.stem+'-current-owner-import');intent_path=directory/'intent.json'
    paths={'state':state_path,'cursor':cursor_path}
    controller_lock=root.parent/'controller.lock';locks={Path(config['out']).resolve()/'controller.lock'}
    if owned_controller_fd is None:locks.add(controller_lock)
    with contextlib.ExitStack() as stack:
        if owned_controller_fd is not None:
            verify_controller_fd(owned_controller_fd,controller_lock)
        for lock_path in sorted(locks):
            lock_path.parent.mkdir(parents=True,exist_ok=True)
            lease=stack.enter_context(lock_path.open('a'))
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():
            committed=json.loads(out.read_text())
            if committed.get('schema')!='natlang.native-raw-current-owner-import/1':
                raise ValueError('current-owner import receipt has an unsupported schema')
            if Path(_bound(committed['intent']))!=intent_path.resolve():
                raise ValueError('current-owner import receipt points outside its durable intent')
            stored_intent=json.loads(_bound(committed['intent']).read_text());stored_request=stored_intent.get('request')
            if (not same_semantics(stored_request or {}) or committed.get('request_sha256')!=digest(stored_request)):
                raise ValueError('current-owner import retry changed its exact transaction semantics')
            prior_intent=validate_intent(stored_intent,stored_request,paths)
            if committed.get('published_checkpoint')!=selected or committed.get('whole_recipe_completed') is not False or committed.get('readiness_granted') is not False:
                raise ValueError('current-owner import receipt changed its durable request')
            resumed_authority=committed.get('resumed_by_authority')
            if (resumed_authority is None) != (committed.get('resumed_by_authority_sha256') is None):
                raise ValueError('current-owner import recovery receipt omitted its resumer authority')
            if resumed_authority is not None and digest(resumed_authority)!=committed['resumed_by_authority_sha256']:
                raise ValueError('current-owner import resumer authority digest changed')
            return binding(out)
        if intent_path.exists():
            intent=json.loads(intent_path.read_text());stored_request=intent.get('request')
            if not same_semantics(stored_request or {}):
                raise ValueError('current-owner import intent belongs to another checkpoint decision')
            resumed=(stored_request!=request)
            if resumed:
                prior_authority=intent.get('authority') or {}
                if owner_alive(prior_authority.get('owner') or {}):
                    raise ValueError('current-owner import original owner is still alive; transaction cannot be taken over')
            intent=validate_intent(intent,stored_request,paths)
        else:
            if binding(config_path)!=request['config']:
                raise ValueError('current-owner import config changed before transaction')
            state=json.loads(state_path.read_text());cursor=json.loads(cursor_path.read_text())
            if state.get('phase')!='collect' or any(state.get(key) is not None for key in ('pending_native_checkpoint','pending_native_round_dir')):
                raise ValueError('current-owner import cannot bypass an active sealed RAW operation')
            if cursor.get('pending_rl_round') is not None:
                raise ValueError('current-owner import cannot bypass a pending sealed RAW round')
            if (cursor.get('current_checkpoint') is None
                    or {key:cursor['current_checkpoint'].get(key) for key in ('path','sha256')}!=binding(state['checkpoint'])):
                raise ValueError('current-owner import requires the same existing state and cursor checkpoint')
            current=binding(state['checkpoint'])
            same_checkpoint={key:selected.get(key) for key in ('path','sha256')}==current
            if not same_checkpoint and (not isinstance(recovery_reason,str) or not recovery_reason.strip()):
                raise ValueError('selecting a different checkpoint requires an explicit supervised recovery reason')
            if same_checkpoint and recovery_reason is not None:
                raise ValueError('recovery reason is reserved for an explicit checkpoint replacement')
            next_state,next_cursor=publication(state,cursor,selected)
            before={name:binding(path) for name,path in paths.items()}
            directory.mkdir(parents=True,exist_ok=True);snapshots={};candidates={}
            for name,path in paths.items():
                snapshot=directory/('original-'+name+'.json');write_bytes_durable(snapshot,_bound(before[name]).read_bytes());snapshots[name]=binding(snapshot)
            for name,value in (('state',next_state),('cursor',next_cursor)):
                candidate=directory/('candidate-'+name+'.json');write_json_durable(candidate,value);candidates[name]=binding(candidate)
            if any(binding(path)!=before[name] for name,path in paths.items()):
                raise ValueError('current-owner controller changed before import intent')
            intent={'schema':'natlang.native-raw-current-owner-import/1','request':request,'authority':authority,'before':before,
                'snapshots':snapshots,'candidates':candidates,'current_checkpoint':current,
                'selected_checkpoint':selected,'selected_global_step':global_step,
                'recovery_reason':recovery_reason,'whole_recipe_completed':False}
            write_json_durable(intent_path,intent)
            intent=validate_intent(intent,request,paths)
        if binding(config_path)!=request['config']:
            raise ValueError('current-owner import config changed during transaction')
        verify_execution_bindings()
        for name,path in paths.items():
            actual=binding(path);candidate=intent['candidates'][name]
            if actual['sha256']!=candidate['sha256'] and actual!=intent['before'][name]:
                raise ValueError('current-owner import found an unknown controller successor')
        for name,path in paths.items():
            candidate=intent['candidates'][name]
            if binding(path)['sha256']!=candidate['sha256']:
                write_bytes_durable(path,_bound(candidate).read_bytes())
        receipt={'schema':intent['schema'],'request_sha256':digest(intent['request']),'intent':binding(intent_path),
            'published_checkpoint':selected,'selected_global_step':global_step,
            'whole_recipe_completed':False,'readiness_granted':False,
            **({'resumed_by_authority':authority,
                'resumed_by_authority_sha256':request['authority_sha256']} if intent['request']!=request else {})}
        write_json_durable(out,receipt)
    return binding(out)


def bridge_deferred_owner_handoff(config_path,cursor_path,deferred_reference,handoff_reference,
                                  retirement_reference,out,*,recovery_reason,verified_handoff=None,
                                  owned_controller_fd=None):
    """Join a deferred startup parent to its completed child without reloading that parent.

    The startup import may have been deferred because the old RAW cursor had a
    sealed round.  After the owner has trained and drained, this exact bridge
    verifies the original dispatch plus the real child checkpoint's
    continuation SHA, revalidates the no-credit stale-round retirement, and
    atomically publishes only the child pointer.  It never reconstructs a
    saved checkpoint from the startup summary and never advances recipe/RAW
    readiness.
    """
    import contextlib
    from ..common.hashing import canonical_json_sha256_hex as digest
    from ..train.native_resident import code_inventory
    from .native_runtime_handoff import _loaded_handoff

    config_path=Path(config_path).resolve(strict=True);cursor_path=Path(cursor_path).resolve(strict=True)
    out=Path(out).resolve();config=json.loads(config_path.read_text());root=config_path.parent
    state_path=Path(config['out']).resolve()/'state.json'
    if cursor_path!=root/'cycle.json' or Path(config['out']).resolve().parent!=root:
        raise ValueError('deferred owner bridge lost its exact RAW controller namespace')
    deferred_path=_bound(deferred_reference);deferred=json.loads(deferred_path.read_text())
    if (not isinstance(deferred,dict)
            or deferred.get('whole_recipe_completed') is not False
            or deferred.get('readiness_granted') is not False
            or deferred.get('reason') not in ('current-owner import cannot bypass an active sealed RAW operation',
                'current-owner import cannot bypass a pending sealed RAW round')
            or not isinstance(recovery_reason,str) or not recovery_reason.strip()):
        raise ValueError('deferred owner bridge needs the exact no-credit startup deferral and recovery reason')
    authority_ref=deferred.get('authority');authority_path=_bound(authority_ref)
    if binding(authority_path)!=authority_ref:
        raise ValueError('deferred owner startup authority changed')
    authority=json.loads(authority_path.read_text())
    if (authority.get('schema')!='natlang.native-raw-current-owner/2'
            or authority.get('selected_checkpoint')!=deferred.get('selected_checkpoint')
            or authority.get('config')!=deferred.get('config')
            or deferred.get('config')!=binding(config_path)
            or deferred.get('cursor',{}).get('path')!=str(cursor_path)):
        raise ValueError('deferred owner bridge changed its original selected parent or controller authority')
    package=Path(authority['package']['path']).resolve(strict=True)
    if code_inventory(package)!=authority['package'].get('code'):
        raise ValueError('deferred owner bridge runtime package changed after startup')
    selected=authority['selected_checkpoint']
    if (not isinstance(selected,dict) or not {'path','sha256','bytes'}<=set(selected)
            or type(selected.get('bytes')) is not int or selected['bytes']<=0
            or not Path(selected['path']).is_absolute()):
        raise ValueError('deferred owner bridge selected startup checkpoint binding is malformed')
    startup_summary=deferred.get('saved_state')
    if (startup_summary!=authority.get('saved_state')
            or startup_summary.get('checkpoint')!=selected
            or type(startup_summary.get('global_step')) is not int
            or startup_summary['global_step']<0
            or startup_summary.get('model_id')!=config.get('raw_model_id')):
        raise ValueError('deferred owner bridge startup summary is not bound to its selected parent')

    # The recorded launch, dispatch and pre-signal tree are the startup
    # attestation. Reuse their exact validators rather than treating the small
    # summary as if it were loaded optimizer state.
    launch_ref=deferred.get('launch');drain_ref=deferred.get('drain_intent')
    launch_path=_bound(launch_ref);drain_path=_bound(drain_ref)
    if binding(launch_path)!=launch_ref or binding(drain_path)!=drain_ref:
        raise ValueError('deferred owner bridge launch or drain intent changed')
    drain=json.loads(drain_path.read_text());launch=json.loads(launch_path.read_text())
    if (drain.get('schema') not in ('natlang.native-raw-optimizer-drain/2','natlang.native-raw-optimizer-drain/1')
            or drain.get('source_launch')!=launch_ref or drain.get('launch',{}).get('sha256')!=launch_ref.get('sha256')
            or authority.get('owner')!=drain.get('owner')
            or authority.get('process_argv')!=launch.get('argv')
            or launch.get('inprocess_dispatch') is None):
        raise ValueError('deferred owner bridge lost the exact original in-process startup dispatch')
    gate_ref=authority['wrapper_configuration'];gate_path=_bound(gate_ref);gate=json.loads(gate_path.read_text())
    args_ref=authority['effective_argv_binding'];args_path=_bound(args_ref);args=json.loads(args_path.read_text())
    from .native_owned_dispatch import verify as verify_dispatch
    continuation,command,dispatch_refs=verify_dispatch(drain,launch)
    runtime_package=Path(gate.get('runtime','')).resolve(strict=True)/'training/neuralese/natlang_neuralese'
    if (continuation.get('parent')!=selected
            or dispatch_refs.get('supervisor')!=authority.get('wrapper')
            or not isinstance(command,list) or command!=json.loads(_bound(authority['effective_argv_file']).read_text())):
        raise ValueError('deferred owner bridge dispatch did not start from its exact selected checkpoint')
    if (binding(gate_path)!=gate_ref or binding(args_path)!=args_ref
            or gate.get('trainer_arguments')!=args or gate.get('effective_trainer_argv')!=args_ref
            or gate.get('parent')!=selected or runtime_package.resolve()!=package
            or gate.get('runtime_code_sha256')!=digest(code_inventory(runtime_package))):
        raise ValueError('deferred owner bridge selected parent differs from its exact trainer specification')

    from .native_raw_handoff import verify_handoff_controls
    handoff_record,saved=_loaded_handoff(handoff_reference,verified_handoff)
    # Controls are rechecked even when the child was already decoded by the
    # owner; the parent itself is never deserialized here.
    controls=verify_handoff_controls(handoff_reference)
    intent=json.loads(_bound(controls['intent']).read_text())
    if (controls!=handoff_record or controls.get('intent')!=drain_ref
            or intent.get('source_launch')!=launch_ref
            or intent.get('owner')!=drain.get('owner')):
        raise ValueError('completed handoff is not the exact deferred startup owner and drain')
    training=saved.get('training') or {};options=training.get('options') or {}
    model=saved.get('model_config') or {}
    model_id=model.get('architecture','')+':'+model.get('base_inventory_sha256','')
    continuation_parent=(options.get('continuation') or {}).get('checkpoint_sha256')
    step=training.get('global_step',training.get('step'))
    if (continuation_parent!=selected['sha256'] or model_id!=startup_summary['model_id']
            or type(step) is not int or step<startup_summary['global_step']
            or config.get('raw_model_id')!=model_id):
        raise ValueError('completed handoff checkpoint does not continue the selected startup optimizer parent')
    child=handoff_record['checkpoint']
    child_path=_bound(child)
    if child.get('bytes') is not None and child_path.stat().st_size!=child['bytes']:
        raise ValueError('completed handoff child checkpoint byte binding changed')

    # Ask the existing retirement verifier to replay its durable no-credit
    # intent. Its completed-retry path is read-only and verifies the original
    # snapshots, candidates, quiescence evidence, debt and request.
    retirement_path=_bound(retirement_reference);retirement=json.loads(retirement_path.read_text())
    retirement_intent=json.loads(_bound(retirement.get('intent')).read_text())
    if retirement_intent.get('schema')!='natlang.native-raw-stale-pending-retirement/1':
        raise ValueError('deferred owner bridge requires the exact completed stale-round retirement')
    verified_retirement=retire_stale_pending_raw_round(config_path,cursor_path,
        retirement_intent['authority'],retirement_path)
    if verified_retirement!=retirement_reference:
        raise ValueError('stale-round retirement receipt changed during bridge verification')

    request={'deferred_startup':deferred_reference,'handoff':handoff_reference,
        'retirement':retirement_reference,'config':binding(config_path),
        'recovery_reason':recovery_reason,'selected_parent':selected,'child_checkpoint':child}
    directory=out.parent/(out.stem+'-deferred-owner-bridge');intent_path=directory/'intent.json'
    paths={'state':state_path,'cursor':cursor_path}
    def candidates(state,cursor):
        if state.get('phase')!='collect' or cursor.get('phase')!='sft' or cursor.get('next_stage')!=0:
            raise ValueError('deferred owner bridge requires the retired SFT-first RAW boundary')
        if (state.get('stale_pending_raw_retirement',{}).get('request_sha256')!=retirement.get('request_sha256')
                or cursor.get('stale_pending_raw_retirement',{}).get('request_sha256')!=retirement.get('request_sha256')
                or state.get('round')!=retirement.get('retired_round')
                or cursor.get('raw_retry_generation')!=retirement.get('collection_attempt')):
            raise ValueError('deferred owner bridge no longer follows this exact no-credit retirement')
        if state.get('pending_native_checkpoint') is not None or state.get('pending_native_round_dir') is not None:
            raise ValueError('deferred owner bridge cannot bypass a newly pending RAW operation')
        next_state=_published_checkpoint_state(state,child);next_state.pop('readiness',None)
        next_cursor={**cursor,'current_checkpoint':child}
        next_cursor.pop('readiness',None);next_cursor.pop('foundation_qualified',None)
        next_cursor['deferred_owner_handoff_bridge']={'request_sha256':digest(request),
            'checkpoint':child,'whole_recipe_completed':False,'raw_completion_credit':0,
            'readiness_granted':False,'runtime_handoff_required':True}
        return next_state,next_cursor
    def validate_intent(value,expected_request):
        if (not isinstance(value,dict) or value.get('schema')!='natlang.native-raw-deferred-owner-bridge/1'
                or value.get('request')!=expected_request or value.get('whole_recipe_completed') is not False
                or value.get('training_credit')!=0 or value.get('raw_completion_credit')!=0
                or set(value.get('before',{}))!=set(paths) or set(value.get('snapshots',{}))!=set(paths)
                or set(value.get('candidates',{}))!=set(paths)):
            raise ValueError('deferred owner bridge intent changed its exact no-credit request')
        originals={}
        for name,path in paths.items():
            snap=value['snapshots'][name];candidate=value['candidates'][name]
            if (snap.get('path')!=str(directory/('original-'+name+'.json'))
                    or candidate.get('path')!=str(directory/('candidate-'+name+'.json'))
                    or binding(_bound(snap))!=snap or snap.get('sha256')!=value['before'][name].get('sha256')):
                raise ValueError('deferred owner bridge transaction artifact changed')
            originals[name]=json.loads(_bound(snap).read_text())
        expected_state,expected_cursor=candidates(originals['state'],originals['cursor'])
        for name,expected in (('state',expected_state),('cursor',expected_cursor)):
            if json.loads(_bound(value['candidates'][name]).read_text())!=expected:
                raise ValueError('deferred owner bridge candidate changed RAW debt or readiness')
        return value

    controller_lock=root.parent/'controller.lock';locks={Path(config['out']).resolve()/'controller.lock'}
    with contextlib.ExitStack() as stack:
        if owned_controller_fd is not None:
            fd=owned_controller_fd;path=controller_lock
            if type(fd)is not int or fd<0 or not path.is_file() or not os.path.isfile(f'/proc/self/fd/{fd}'):
                raise ValueError('deferred owner bridge lacks its existing canonical controller lease')
            opened=os.fstat(fd);actual=path.stat()
            rows=[line for line in Path('/proc/self/fdinfo',str(fd)).read_text().splitlines()
                  if line.startswith('lock:') and 'FLOCK' in line]
            if (opened.st_dev,opened.st_ino)!=(actual.st_dev,actual.st_ino) or len(rows)!=1 or 'WRITE' not in rows[0]:
                raise ValueError('deferred owner bridge descriptor is not the existing exclusive controller lease')
        else:locks.add(controller_lock)
        for lock_path in sorted(locks):
            lock_path.parent.mkdir(parents=True,exist_ok=True);lease=stack.enter_context(lock_path.open('a'))
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if binding(config_path)!=request['config'] or binding(deferred_path)!=deferred_reference or binding(_bound(handoff_reference))!=handoff_reference:
            raise ValueError('deferred owner bridge source authority changed before commit')
        if out.exists():
            receipt=json.loads(out.read_text());stored=json.loads(_bound(receipt['intent']).read_text())
            if (receipt.get('schema')!='natlang.native-raw-deferred-owner-bridge/1'
                    or receipt.get('request_sha256')!=digest(request)
                    or Path(_bound(receipt['intent'])).resolve()!=intent_path.resolve()):
                raise ValueError('deferred owner bridge retry changed its transaction')
            validate_intent(stored,request)
            if (receipt.get('published_checkpoint')!=child or receipt.get('whole_recipe_completed') is not False
                    or receipt.get('training_credit')!=0 or receipt.get('raw_completion_credit')!=0):
                raise ValueError('deferred owner bridge receipt changed its exact no-credit result')
            return binding(out)
        if intent_path.exists():
            intent=validate_intent(json.loads(intent_path.read_text()),request)
        else:
            before={name:binding(path) for name,path in paths.items()}
            # The current files must still be the exact retired candidates.
            for name,path in paths.items():
                if before[name]['sha256']!=retirement_intent['candidates'][name]['sha256']:
                    raise ValueError('deferred owner bridge found a successor after RAW retirement')
            state=json.loads(state_path.read_text());cursor=json.loads(cursor_path.read_text())
            next_state,next_cursor=candidates(state,cursor)
            directory.mkdir(parents=True,exist_ok=True);snapshots={};candidate_refs={}
            for name,path in paths.items():
                snap=directory/('original-'+name+'.json');write_bytes_durable(snap,_bound(before[name]).read_bytes());snapshots[name]=binding(snap)
            for name,value in (('state',next_state),('cursor',next_cursor)):
                candidate=directory/('candidate-'+name+'.json');write_json_durable(candidate,value);candidate_refs[name]=binding(candidate)
            if any(binding(path)!=before[name] for name,path in paths.items()):
                raise ValueError('deferred owner bridge controller changed before durable intent')
            intent={'schema':'natlang.native-raw-deferred-owner-bridge/1','request':request,
                'before':before,'snapshots':snapshots,'candidates':candidate_refs,
                'whole_recipe_completed':False,'training_credit':0,'raw_completion_credit':0}
            write_json_durable(intent_path,intent);intent=validate_intent(intent,request)
        if binding(config_path)!=request['config']:
            raise ValueError('deferred owner bridge config changed during transaction')
        for name,path in paths.items():
            actual=binding(path);candidate=intent['candidates'][name]
            if actual['sha256']!=candidate['sha256'] and actual!=intent['before'][name]:
                raise ValueError('deferred owner bridge found an unknown controller successor')
        for name,path in paths.items():
            candidate=intent['candidates'][name]
            if binding(path)['sha256']!=candidate['sha256']:
                write_bytes_durable(path,_bound(candidate).read_bytes())
        write_json_durable(out,{'schema':intent['schema'],'request_sha256':digest(request),
            'intent':binding(intent_path),'published_checkpoint':child,'selected_startup_parent':selected,
            'whole_recipe_completed':False,'training_credit':0,'raw_completion_credit':0,
            'readiness_granted':False,'runtime_handoff_required':True})
    return binding(out)


def retire_stale_pending_raw_round(config_path,cursor_path,authority,out,*,owned_controller_fd=None,permitted_controller_owner_fd=None):
    """Retire one authenticated dead RAW operation without crediting its work.

    The operation owner chooses whether a stopped pending round should be
    retried. This exact transaction only preserves its files and debt, clears
    stale operation pointers, and sends the existing scheduler back through
    mandatory SFT before a fresh collection attempt.
    """
    import contextlib
    from ..common.hashing import canonical_json_sha256_hex as digest

    config_path=Path(config_path).resolve(strict=True);cursor_path=Path(cursor_path).resolve(strict=True)
    out=Path(out).resolve();config=json.loads(config_path.read_text());root=config_path.parent
    state_path=Path(config['out']).resolve()/'state.json'
    if cursor_path!=root/'cycle.json' or Path(config['out']).resolve().parent!=root:
        raise ValueError('stale RAW retirement lost its exact controller namespace')
    required={'schema','config','state','cursor','selection','historical_owner_identity_available',
        'process_census','permitted_owner','operation_directories','provider_quiescence','reason'}
    if (not isinstance(authority,dict) or set(authority)!=required
            or authority.get('schema')!='natlang.native-raw-stale-pending-authority/2'
            or authority.get('historical_owner_identity_available') is not False):
        raise ValueError('stale RAW retirement requires complete exact owner and predecessor bindings')
    directory=out.parent/(out.stem+'-stale-pending-retirement');intent_path=directory/'intent.json'
    completed_retry=out.is_file()
    paths={'state':state_path,'cursor':cursor_path}
    if binding(config_path)!=authority['config']:
        raise ValueError('stale RAW retirement predecessor config changed')
    if intent_path.is_file():
        prior_intent=json.loads(intent_path.read_text())
        if (prior_intent.get('authority')!=authority
                or set(prior_intent.get('snapshots',{}))!=set(paths)):
            raise ValueError('stale RAW retirement recovery intent changed its exact predecessor')
        original={name:json.loads(_bound(prior_intent['snapshots'][name]).read_text()) for name in paths}
        for name in paths:
            if prior_intent.get('before',{}).get(name)!=authority[name]:
                raise ValueError('stale RAW retirement recovery snapshot lost its predecessor binding')
        state,cursor=original['state'],original['cursor']
    else:
        for name,path in (('state',state_path),('cursor',cursor_path)):
            if binding(path)!=authority[name]:raise ValueError('stale RAW retirement predecessor '+name+' changed')
        state=json.loads(state_path.read_text());cursor=json.loads(cursor_path.read_text())
    if not isinstance(authority['reason'],str) or not authority['reason'].strip():
        raise ValueError('stale RAW retirement needs the owner decision reason')
    quiescence_ref=authority['provider_quiescence'];quiescence_path=_bound(quiescence_ref)
    if binding(quiescence_path)!=quiescence_ref:
        raise ValueError('stale RAW retirement provider quiescence proof changed')
    quiescence=json.loads(quiescence_path.read_text())
    expected_units={'cw-lfm-cpu26-keep-provider-20261018.service',
        'cw-lfm-cpu26-keep-heartbeat-20261018.service',
        'cw-mellum-cpu26-actual-queue-provider-20261018.service',
        'cw-mellum-cpu26-actual-queue-heartbeat-20261018.service'}
    if (quiescence.get('schema')!='natlang.supervisor-legacy-provider-quiescence/1'
            or type(quiescence.get('at')) not in (int,float)
            or not completed_retry and (time.time()-quiescence['at']<0 or time.time()-quiescence['at']>300)
            or not isinstance(quiescence.get('boot_id'),str) or not quiescence['boot_id']
            or quiescence.get('training_credit')!=0 or type(quiescence.get('learner_checkpoint_mutations'))is not int
            or quiescence.get('learner_checkpoint_mutations')!=0
            or not isinstance(quiescence.get('units'),list)
            or len(quiescence['units'])!=len(expected_units)
            or any(not isinstance(row,dict) for row in quiescence['units'])
            or {row.get('Id') for row in quiescence['units']}!=expected_units
            or any(row.get('ActiveState')!='inactive' or row.get('SubState')!='dead' or row.get('MainPID')!='0'
                   for row in quiescence['units'])):
        raise ValueError('stale RAW retirement requires all exact legacy source-host provider units parked')
    if (state.get('phase') not in ('updating','stopped_collecting')
            or cursor.get('phase')!='rl' or type(cursor.get('pending_rl_round'))is not int
            or type(state.get('round'))is not int or cursor['pending_rl_round']!=state['round']+1
            or type(cursor.get('cycle'))is not int or cursor.get('next_stage')!=0
            or state.get('production_complete') is True):
        raise ValueError('stale RAW retirement does not match an incomplete pending RL round')
    if any(key.startswith('pending_') and value is not None
            for key,value in state.items() if key not in ('pending_native_checkpoint','pending_native_round_dir')):
        raise ValueError('stale RAW retirement cannot bypass another pending production operation')
    if any(key.startswith('pending_') and key!='pending_rl_round' and value is not None
            for key,value in cursor.items()):
        raise ValueError('stale RAW retirement found an unknown pending cursor owner')
    if any(state.get(key) is not None for key in ('pending_native_checkpoint','pending_native_round_dir','learner_pending')):
        raise ValueError('stale RAW retirement cannot bypass an active sealed RAW operation')
    if state.get('checkpoint') is None or cursor.get('current_checkpoint') is None:
        raise ValueError('stale RAW retirement requires the same bound current checkpoint')
    checkpoint=binding(state['checkpoint'])
    if {key:cursor['current_checkpoint'].get(key) for key in ('path','sha256')}!=checkpoint:
        raise ValueError('stale RAW retirement found different state and cursor checkpoints')
    selection_path=Path(config['out']).resolve()/f"round-{state['round']:06d}-selection.json"
    if binding(selection_path)!=authority['selection']:
        raise ValueError('stale RAW retirement selection snapshot changed')
    selection=json.loads(selection_path.read_text())
    if (not isinstance(selection,dict) or not isinstance(selection.get('selection'),list)
            or not selection['selection'] or not isinstance(selection.get('cases'),list)
            or len(selection['cases'])!=len(selection['selection'])):
        raise ValueError('stale RAW retirement requires its original nonempty sealed selection')
    operation_directories=authority['operation_directories']
    expected_bridge=root/'cycles'/f"{cursor['cycle']:06d}"/'rl-bridge'
    expected_update=Path(state.get('active_update_round','')).resolve()
    if (not isinstance(operation_directories,list) or not operation_directories
            or any(not isinstance(path,str) for path in operation_directories)
            or str(expected_bridge.resolve()) not in operation_directories
            or str(expected_update) not in operation_directories
            or any(Path(path).resolve()==root or root.resolve() not in Path(path).resolve().parents for path in operation_directories)):
        raise ValueError('stale RAW retirement operation directories do not bind the exact round and updater')
    bound_operation_roots=[Path(path).resolve() for path in operation_directories]
    for key,value in state.items():
        if key.startswith(('active_update_','update_','partial_update_')) and isinstance(value,str) and value.startswith('/'):
            resolved=Path(value).resolve()
            if not any(resolved==base or base in resolved.parents for base in bound_operation_roots):
                raise ValueError('stale RAW retirement omitted a persisted operation path from its process-check scope')
    if type(state.get('collection_attempt',0)) is not int or state.get('collection_attempt',0)<0:
        raise ValueError('stale RAW retirement collection attempt is malformed')
    from .native_raw_handoff import observe,process_argv,same_process
    census_ref=authority['process_census'];census_path=_bound(census_ref)
    if binding(census_path)!=census_ref:
        raise ValueError('stale RAW retirement process census binding changed')
    census=json.loads(census_path.read_text())
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    recovering=intent_path.exists()
    if (not isinstance(census,dict) or not completed_retry and census.get('boot_id')!=boot
            or not isinstance(census.get('processes'),list) or type(census.get('at')) not in (int,float)
            or not (recovering or completed_retry) and (time.time()-census['at']<0 or time.time()-census['at']>120)):
        raise ValueError('stale RAW retirement needs a current same-host process census')
    permitted=authority['permitted_owner']
    if (not isinstance(permitted,dict) or set(permitted)!={'identity','q_root','gate_spec','wrapper','argv'}
            or not isinstance(permitted['identity'],dict)):
        raise ValueError('stale RAW retirement needs the exact distinct new-entry owner')
    new_root=Path(permitted['q_root']).resolve(strict=True)
    gate_ref=permitted['gate_spec'];gate_path=_bound(gate_ref);wrapper_ref=permitted['wrapper'];wrapper_path=_bound(wrapper_ref)
    if binding(gate_path)!=gate_ref or binding(wrapper_path)!=wrapper_ref:
        raise ValueError('stale RAW retirement new-entry authority changed')
    gate=json.loads(gate_path.read_text())
    if (new_root==root or wrapper_path!=new_root/'owned-entry.py'
            or Path(gate.get('out','')).resolve()!=new_root/'learner'
            or root.as_posix() in str(gate.get('out',''))):
        raise ValueError('stale RAW retirement permitted owner is not bound to a distinct RAW successor')
    permitted_identity=permitted['identity'];permitted_observed=None if completed_retry else observe(permitted_identity.get('pid'))
    expected_wrapper_argv=[permitted['argv'][0],'-u',str(wrapper_path)] if isinstance(permitted.get('argv'),list) and permitted['argv'] else None
    if (not completed_retry and (not same_process(permitted_identity,permitted_observed)
            or process_argv(permitted_observed)!=permitted.get('argv'))
            or any(type(arg)is not str for arg in permitted['argv'])
            or permitted['argv']!=expected_wrapper_argv
            or not any(str(new_root) in arg for arg in permitted['argv'])):
        raise ValueError('stale RAW retirement permitted process is not the exact distinct in-process entry')
    census_owner=False
    for entry in census['processes']:
        if (not isinstance(entry,dict) or type(entry.get('pid'))is not int
                or not isinstance(entry.get('birth'),str) or not isinstance(entry.get('argv'),list)
                or entry.get('boot_id')!=census.get('boot_id')
                or any(type(arg)is not str for arg in entry['argv'])):
            raise ValueError('stale RAW retirement census has a malformed process row')
        if not completed_retry:
            observed=observe(entry['pid'])
            expected={'pid':entry['pid'],'boot_id':census['boot_id'],'start_ticks':entry['birth']}
            if same_process(expected,observed):
                if process_argv(observed)!=entry['argv']:
                    raise ValueError('stale RAW retirement census process argv changed after capture')
                if same_process(permitted_identity,observed):census_owner=True
    if not completed_retry and not census_owner:
        raise ValueError('stale RAW retirement census does not include its exact new-entry owner')

    # A fresh local /proc pass closes the census-to-commit window. Only the
    # captured new entry may mention its new Q/gate-spec. Any other current
    # process pointing into the stale request/config/output blocks retirement.
    stale_paths=[str(config_path),str(state_path),str(cursor_path),*(str(Path(path).resolve()) for path in operation_directories)]
    def references_stale(argv):
        return any(old and any(old in arg for arg in argv) for old in stale_paths)
    def assert_no_live_stale_processes():
        # The immutable census detects a changed/reused PID; this current pass
        # catches workers launched after it. The permitted in-process owner is
        # excluded only by its complete PID/birth/boot and exact argv.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            observed=observe(int(proc.name))
            if observed is None:continue
            argv=process_argv(observed)
            if references_stale(argv) and not same_process(permitted_identity,observed):
                raise ValueError('stale RAW retirement is blocked by a live process using the old RAW operation')
    for operation_directory in operation_directories:
        original_path=Path(operation_directory)
        path=original_path.resolve()
        if original_path.is_symlink() or not path.is_dir():
            raise ValueError('stale RAW retirement operation directory is not an intact directory')

    def verify_controller_fd(fd,path):
        if type(fd)is not int or fd<0 or not path.is_file() or not os.path.isfile(f'/proc/self/fd/{fd}'):
            raise ValueError('stale RAW retirement lacks the inherited canonical controller lease')
        opened=os.fstat(fd);actual=path.stat()
        if opened.st_dev!=actual.st_dev or opened.st_ino!=actual.st_ino:
            raise ValueError('stale RAW retirement inherited descriptor is not the canonical controller lease')
        info=Path('/proc/self/fdinfo',str(fd)).read_text().splitlines()
        locks=[line for line in info if line.startswith('lock:') and 'FLOCK' in line]
        if len(locks)!=1 or 'WRITE' not in locks[0]:
            raise ValueError('stale RAW retirement inherited descriptor has no exclusive controller lease')

    def verify_permitted_owner_fd(fd,path,identity):
        """Verify the live in-process trainer's lease without taking it over."""
        if type(fd)is not int or fd<0 or not path.is_file():
            raise ValueError('stale RAW retirement permitted owner descriptor is invalid')
        observed=observe(identity['pid'])
        if not same_process(identity,observed):
            raise ValueError('stale RAW retirement permitted owner descriptor process changed')
        proc_fd=Path('/proc',str(identity['pid']),'fd',str(fd))
        proc_info=Path('/proc',str(identity['pid']),'fdinfo',str(fd))
        if not proc_fd.exists() or not proc_info.is_file():
            raise ValueError('stale RAW retirement permitted owner no longer holds its controller descriptor')
        opened=os.stat(proc_fd);actual=path.stat()
        if opened.st_dev!=actual.st_dev or opened.st_ino!=actual.st_ino:
            raise ValueError('stale RAW retirement permitted owner descriptor is not the canonical controller lease')
        locks=[line for line in proc_info.read_text().splitlines() if line.startswith('lock:') and 'FLOCK' in line]
        if len(locks)!=1 or 'WRITE' not in locks[0]:
            raise ValueError('stale RAW retirement permitted owner descriptor has no exclusive controller lease')

    request={'authority_sha256':digest(authority),'config':authority['config'],'state':authority['state'],
        'cursor':authority['cursor'],'round':state['round'],'pending_round':cursor['pending_rl_round'],
        'checkpoint':checkpoint,'selection':authority['selection'],'reason':authority['reason']}
    def candidates_from(original_state,original_cursor):
        retired={key:value for key,value in original_state.items()
            if key.startswith(('active_update_','update_','partial_update_'))}
        if not any(key.startswith('active_update_') for key in retired):
            raise ValueError('stale RAW retirement has no active update ownership to retire')
        next_state=dict(original_state)
        for key in retired:next_state.pop(key,None)
        next_state.update(phase='collect',collection_attempt=original_state.get('collection_attempt',0)+1)
        record={'request_sha256':digest(request),'round':original_state['round'],
            'pending_round':original_cursor['pending_rl_round'],'selection':authority['selection'],
            'previous_checkpoint':binding(original_state['checkpoint']),
            'operation_directories':operation_directories,'process_census':census_ref,
            'permitted_owner':permitted,
            'retired_state_fields':retired,'reason':authority['reason'],
            'whole_recipe_completed':False,'training_credit':0}
        next_state['stale_pending_raw_retirement']=record
        next_cursor=dict(original_cursor);next_cursor.update(phase='sft',next_stage=0,
            stale_pending_raw_retirement={'request_sha256':digest(request),'round':original_state['round'],
                'pending_round':original_cursor['pending_rl_round'],'receipt_path':str(out),
                'whole_recipe_completed':False})
        next_cursor.pop('pending_rl_round',None);next_cursor.pop('readiness',None);next_cursor.pop('foundation_qualified',None)
        next_cursor['raw_retry_generation']=next_state['collection_attempt']
        return next_state,next_cursor,retired

    def validate_intent(intent):
        if (not isinstance(intent,dict) or intent.get('schema')!='natlang.native-raw-stale-pending-retirement/1'
                or intent.get('request')!=request or intent.get('authority')!=authority
                or intent.get('whole_recipe_completed') is not False or intent.get('training_credit')!=0
                or set(intent.get('before',{}))!=set(paths) or set(intent.get('snapshots',{}))!=set(paths)
                or set(intent.get('candidates',{}))!=set(paths)):
            raise ValueError('stale RAW retirement durable intent changed its exact request')
        originals={}
        for name,path in paths.items():
            if (intent['before'][name]!=authority[name]
                    or Path(intent['snapshots'][name]['path']).resolve()!=directory/('original-'+name+'.json')
                    or Path(intent['candidates'][name]['path']).resolve()!=directory/('candidate-'+name+'.json')):
                raise ValueError('stale RAW retirement intent substituted a transaction artifact')
            snapshot=_bound(intent['snapshots'][name]);snapshot_binding=binding(snapshot)
            if (snapshot_binding!=intent['snapshots'][name]
                    or snapshot_binding.get('sha256')!=authority[name].get('sha256')):
                raise ValueError('stale RAW retirement snapshot binding changed')
            originals[name]=json.loads(snapshot.read_text())
        expected_state,expected_cursor,retired=candidates_from(originals['state'],originals['cursor'])
        if retired!=intent.get('retired_state_fields'):
            raise ValueError('stale RAW retirement intent lost retired operation fields')
        for name,expected in (('state',expected_state),('cursor',expected_cursor)):
            candidate=_bound(intent['candidates'][name])
            if json.loads(candidate.read_text())!=expected:
                raise ValueError('stale RAW retirement candidate changed debt or unrelated fields')
        return intent

    controller_lock=root.parent/'controller.lock';locks={Path(config['out']).resolve()/'controller.lock'}
    with contextlib.ExitStack() as stack:
        if owned_controller_fd is not None and permitted_controller_owner_fd is not None:
            raise ValueError('stale RAW retirement accepts only one canonical controller lease authority')
        if completed_retry:
            pass
        elif permitted_controller_owner_fd is not None:
            verify_permitted_owner_fd(permitted_controller_owner_fd,controller_lock,permitted_identity)
        elif owned_controller_fd is not None:
            verify_controller_fd(owned_controller_fd,controller_lock)
        else:
            raise ValueError('stale RAW retirement requires the existing canonical controller lease')
        for lock_path in sorted(locks):
            lease=stack.enter_context(lock_path.open('a'))
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if binding(config_path)!=authority['config']:
            raise ValueError('stale RAW retirement configuration changed before commit')
        if not completed_retry:
            assert_no_live_stale_processes()
        if (not completed_retry and (Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=boot
                or not same_process(permitted_identity,observe(permitted_identity['pid'])))):
            raise ValueError('stale RAW retirement process authority changed before commit')

        if out.exists():
            receipt=json.loads(out.read_text())
            if (receipt.get('schema')!='natlang.native-raw-stale-pending-retirement/1'
                    or receipt.get('request_sha256')!=digest(request)
                    or Path(_bound(receipt['intent']))!=intent_path.resolve()):
                raise ValueError('stale RAW retirement receipt belongs to another operation')
            stored=validate_intent(json.loads(_bound(receipt['intent']).read_text()))
            if (receipt.get('whole_recipe_completed') is not False or receipt.get('training_credit')!=0
                    or receipt.get('retired_round')!=request['round'] or receipt.get('pending_round')!=request['pending_round']
                    or receipt.get('same_selection')!=authority['selection']
                    or receipt.get('collection_attempt')!=state.get('collection_attempt',0)+1):
                raise ValueError('stale RAW retirement receipt changed its exact no-credit result')
            return binding(out)

        if intent_path.exists():
            intent=json.loads(intent_path.read_text())
            intent=validate_intent(intent)
        else:
            before={name:binding(path) for name,path in paths.items()}
            if before!={'state':authority['state'],'cursor':authority['cursor']}:
                raise ValueError('stale RAW retirement pointers changed before intent')
            next_state,next_cursor,retired_fields=candidates_from(state,cursor)
            directory.mkdir(parents=True,exist_ok=True);snapshots={};candidates={}
            for name,path in paths.items():
                snapshot=directory/('original-'+name+'.json')
                write_bytes_durable(snapshot,_bound(before[name]).read_bytes());snapshots[name]=binding(snapshot)
            for name,value in (('state',next_state),('cursor',next_cursor)):
                candidate=directory/('candidate-'+name+'.json');write_json_durable(candidate,value);candidates[name]=binding(candidate)
            if any(binding(path)!=before[name] for name,path in paths.items()):
                raise ValueError('stale RAW retirement controller changed before durable intent')
            intent={'schema':'natlang.native-raw-stale-pending-retirement/1','request':request,
                'authority':authority,'before':before,'snapshots':snapshots,'candidates':candidates,
                'retired_state_fields':retired_fields,'whole_recipe_completed':False,'training_credit':0}
            write_json_durable(intent_path,intent)
            intent=validate_intent(intent)
        if binding(config_path)!=authority['config']:
            raise ValueError('stale RAW retirement configuration changed during transaction')
        for name,path in paths.items():
            actual=binding(path);candidate=intent['candidates'][name]
            if actual['sha256']!=candidate['sha256'] and actual!=intent['before'][name]:
                raise ValueError('stale RAW retirement found an unknown controller successor')
        for name,path in paths.items():
            candidate=intent['candidates'][name]
            if binding(path)['sha256']!=candidate['sha256']:
                write_bytes_durable(path,_bound(candidate).read_bytes())
        write_json_durable(out,{'schema':intent['schema'],'request_sha256':digest(request),
            'intent':binding(intent_path),'whole_recipe_completed':False,'training_credit':0,
            'retired_round':state['round'],'pending_round':cursor['pending_rl_round'],
            'same_selection':authority['selection'],'next_phase':'sft','next_stage':0,
            'collection_attempt':state.get('collection_attempt',0)+1})
    return binding(out)


def _owned_training_invocation(argv,parent,*,parent_bytes=None):
    """Resolve only the exact reviewed read-only historical restore observer."""
    if '--argv-binding' not in argv:return argv,None
    flags=['--argv-binding','--parent','--runtime-root','--out']
    if (len(argv)!=11 or argv[:2]!=['python','-u'] or argv[3::2]!=flags):
        raise ValueError('owned restore observer changed its exact invocation')
    observer=binding(_bound({'path':argv[2],'sha256':_OWNED_RESTORE_OBSERVER_SHA256}))
    reference=json.loads(argv[4]);original=_bound(reference);restored=json.loads(argv[6])
    if (restored.get('sha256')!=parent['sha256']
            or Path(restored['path']).resolve()!=Path(parent['path']).resolve()
            or (restored.get('bytes') is not None and restored['bytes']!=parent_bytes)):
        raise ValueError('owned restore observer changed its actual preceding parent')
    command=json.loads(original.read_text())
    if (not isinstance(command,list) or any(not isinstance(item,str)for item in command)
            or command[:3]!=['python','-u','-m'] or len(command)<4
            or command[3]not in('natlang_neuralese.train.native_state_sft','natlang_neuralese.train.native_preference')
            or command.count('--continue-from')!=1
            or command.index('--continue-from')+1>=len(command)
            or Path(command[command.index('--continue-from')+1]).resolve()!=Path(parent['path']).resolve()
            or not Path(argv[8]).is_absolute() or not Path(argv[10]).is_absolute()):
        raise ValueError('owned restore observer lost its exact original trainer command')
    return command,{'observer':observer,'original_argv':binding(original),
                    'parent':restored,'runtime':argv[8],'observations':argv[10]}


def _owned_recipe_invocation(argv, child, record, actual_recipe, stage_id, lineage):
    """A resident launch keeps its original argv while genuine adoption advances sources."""
    from .native_raw_handoff import same_process, _recipe_plan
    from ..train import native_resident as resident
    from ..train.recipe import native_recipe_binding
    def argument(flag):
        return argv[argv.index(flag)+1] if argv.count(flag)==1 and argv.index(flag)+1<len(argv) else None
    declared=json.loads(argument('--recipe-binding'))
    evidence=record.get('resident') or {}
    if not evidence.get('current'):
        return declared==actual_recipe
    initial=json.loads(_bound(evidence['initial']).read_text())
    if not same_process(initial.get('owner') or {},child):return False
    state=json.loads(_bound(evidence['current']).read_text());payload=state.get('payload') or {}
    directory=argument('--resident-data-directory')
    history=state.get('history') or []
    if (not directory or Path(directory).resolve()!=Path(evidence['directory']).resolve()
            or state.get('schema')!=resident.SCHEMA or resident.digest(payload)!=state.get('generation')
            or state.get('generation')!=evidence.get('generation') or not history
            or payload.get('stage_id')!=stage_id or payload.get('recipe')!=actual_recipe
            or initial.get('recipe')!=declared or payload.get('prior_recipe')!=declared
            or history[-1].get('generation')!=state['generation']):
        raise ValueError('owned resident invocation lost its initial/current source authority')
    plan=payload['plan'];previous=_recipe_plan(_bound({'path':declared['path'],
        'sha256':declared.get('file_sha256',declared['sha256'])}),plan['device'])
    previous['code']=plan['code'];last_step=-1
    if native_recipe_binding(Path(declared['path']),lineage)!=declared:
        raise ValueError('owned resident initial recipe lineage changed')
    for entry in history:
        reference=entry['recipe'];path=_bound({'path':reference['path'],
            'sha256':reference.get('file_sha256',reference['sha256'])})
        step=entry.get('at_global_step')
        if (type(step)is not int or step<last_step
                or native_recipe_binding(path,lineage)!=reference):
            raise ValueError('owned resident adoption history changed its order/recipe lineage')
        proposed=_recipe_plan(path,plan['device']);proposed['code']=plan['code']
        resident._contract(previous,proposed,stage_id)
        if (entry.get('train')!={k:proposed['stage_inputs'][stage_id]['native_graphs'][k]for k in('path','sha256')}
                or entry.get('held')!={k:proposed['stage_inputs'][stage_id]['held_native_graphs'][k]for k in('path','sha256')}):
            raise ValueError('owned resident history changed its exact admitted source bindings')
        previous=proposed;last_step=step
    if (history[-1]['recipe']!=actual_recipe or previous['recipe']!=plan['recipe']
            or previous['stage_inputs']!=plan['stage_inputs']):
        raise ValueError('owned resident current plan lost its complete initial-to-adopted history')
    return True


def _owned_operation_chain(operations,published,model_id,lineage):
    """Authenticate completed finite work without decoding historical optimizers."""
    from .native_raw_handoff import verify_handoff_controls,verify_saved_update_acknowledgement
    from ..train.recipe import native_recipe_binding,resolve_recipe_data
    prior=published;prior_bytes=published.get('bytes');result=[]
    for index,operation in enumerate(operations):
        if bool(operation.get('handoff'))==bool(operation.get('saved_update_recovery')):
            raise ValueError('owned operation needs exactly one successful handoff or explicit saved-update recovery')
        recovered=operation.get('saved_update_recovery')
        if recovered:
            if index==len(operations)-1:
                raise ValueError('owned recovery is historical only; the final current owner needs a successful full handoff')
            record=verify_saved_update_acknowledgement(recovered,receipt=operation['receipt'])
        else:record=verify_handoff_controls(operation['handoff'])
        receipt=json.loads(_bound(operation['receipt']).read_text())
        intent=json.loads(_bound(record['intent']).read_text())
        recipe_ref=receipt.get('recipe') or {}
        recipe_path=_bound({'path':recipe_ref['path'],'sha256':recipe_ref.get('file_sha256',recipe_ref['sha256'])})
        actual_recipe=native_recipe_binding(recipe_path,lineage)
        stage_id=receipt.get('stage_id');recipe=resolve_recipe_data(recipe_path)
        stages=[stage for stage in recipe['stages'] if stage['id']==stage_id]
        boundary=receipt.get('continuation_boundary') or {}
        updates=boundary.get('new_optimizer_updates')
        checkpoint=record['checkpoint'];slot=record['checkpoint_pin']['original_slot']
        if (receipt.get('schema')!='natlang.native-training-readiness/1'
                or receipt.get('model_id')!=model_id or recipe_ref!=actual_recipe or len(stages)!=1
                or (receipt.get('checkpoint') or {}).get('sha256')!=checkpoint['sha256']
                or str(Path(receipt['checkpoint']['path']).resolve())!=slot['path']
                or (receipt.get('native_identity') or {}).get('checkpoint_sha256')!=checkpoint['sha256']
                or (receipt.get('continuation') or {}).get('checkpoint_sha256')!=prior['sha256']
                or boundary.get('checkpoint_sha256')!=checkpoint['sha256']
                or type(updates)is not int or updates<=0):
            raise ValueError('owned operation lost its real parent/recipe/model/update ancestry')
        matching=[]
        from .native_runtime_handoff import _owned_launch_chain
        launch=json.loads(_bound(intent['launch']).read_text())
        inprocess = bool(launch.get('inprocess_dispatch'))
        candidates = [intent['owner']] if inprocess else intent['children']
        dispatched = _owned_launch_chain(record)[1] if inprocess else None
        for child in candidates:
            from .native_raw_handoff import process_argv
            argv=dispatched if inprocess else process_argv(child)
            argv,observer=_owned_training_invocation(argv,prior,parent_bytes=prior_bytes)
            def argument(flag):
                if argv.count(flag)!=1 or argv.index(flag)+1>=len(argv):return None
                return argv[argv.index(flag)+1]
            out=argument('--out');binding_arg=argument('--recipe-binding')
            if (out and Path(out).resolve()==Path(operation['receipt']['path']).resolve().parent
                    and argument('--stage-id')==stage_id and binding_arg
                    and _owned_recipe_invocation(argv,child,record,actual_recipe,stage_id,lineage)):matching.append({'child':child,'restore_observer':observer})
        if len(matching)!=1:
            raise ValueError('owned operation receipt lacks its exact captured trainer invocation')
        result.append({**({'saved_update_recovery':recovered,'original_failed_result':record['original_failed_result'],
                          'outcome':record['outcome']}if recovered else{'handoff':operation['handoff']}),
            'receipt':operation['receipt'],
            'parent':prior,'checkpoint':checkpoint,'stage_id':stage_id,'stage':stages[0],
            'recipe':actual_recipe,'new_optimizer_updates':updates,
            **({'restore_observer':matching[0]['restore_observer']}if matching[0]['restore_observer']else{})})
        prior=checkpoint
        prior_bytes=record['checkpoint_pin']['bytes']
    if not result:raise ValueError('owned reconciliation needs actual completed operations')
    return result


def _acknowledged_owned_stage_ids(chain,stages,*,acknowledged_prefix=()):
    ids=[stage['id']for stage in stages]
    prefix=list(acknowledged_prefix)
    if prefix!=stages[:len(prefix)]:
        raise ValueError('owned reconciliation changed its previously acknowledged stage prefix')
    earned=[stage['id']for stage in prefix]
    for item in chain:
        if not earned or earned[-1]!=item['stage_id']:earned.append(item['stage_id'])
    if earned[-len(ids):]!=ids:
        raise ValueError('owned reconciliation cannot acknowledge an unearned recipe stage')
    for stage in stages:
        latest=next((item for item in reversed(chain)if item['stage_id']==stage['id']),None)
        if latest is None:
            if stage not in prefix:raise ValueError('owned reconciliation cannot borrow an unearned stage')
        elif latest['stage']!=stage:raise ValueError('owned reconciliation changed an earned stage contract')
    return ids


def _owned_earned_stage_inventory(transition,stages,stage_id):
    """Separate earned old contracts from verified future operational tuning.

    The caller has already run the full existing runtime/recipe handoff verifier.
    No new source or stage can borrow credits through this metadata-only seam.
    """
    if transition.get('bare'):return stages,None
    old_plan=json.loads(_bound(transition['old_plan']).read_text())
    new_plan=json.loads(_bound(transition['new_plan']).read_text())
    old=_supervised_stage_inventory(old_plan['recipe'],stage_id)
    future=_supervised_stage_inventory(new_plan['recipe'],stage_id)
    if future!=stages:raise ValueError('owned reconciliation future recipe differs from verified handoff')
    if ([(x['id'],x['kind'])for x in old]!=[(x['id'],x['kind'])for x in future]
            or old_plan['stage_inputs']!=new_plan['stage_inputs']
            or old_plan['inputs']!=new_plan['inputs']):
        raise ValueError('owned operational tuning cannot credit new sources or stages')
    if old==future:return stages,None
    return old,{'old_plan':transition['old_plan'],'new_plan':transition['new_plan'],
        'earned_stage_contracts':old,'future_stage_contracts':future,
        'future_recipe_credit_granted':False,
        'authority':'existing verified lineage_handoff operational tuning'}


def reconcile_owned_operations(config_path,operations,recipe_reference,out,*,bare_direct_owner=None,recipe_handoff=None,verified_handoff=None,retain_mandatory_schedule=False):
    """Leased, durable acknowledgement of earned work, never stage completion."""
    import contextlib
    from ..common.hashing import canonical_json_sha256_hex as digest
    from ..train.recipe import resolve_recipe_data
    from .native_runtime_handoff import _authority,_loaded_handoff
    config_path=Path(config_path).resolve();out=Path(out).resolve()
    config=json.loads(config_path.read_text());root=config_path.parent
    if Path(config['out']).resolve().parent!=root:raise ValueError('owned reconciliation lost its controller namespace')
    paths={'config':config_path,'cursor':root/'cycle.json','state':Path(config['out'])/'state.json'}
    if bool(bare_direct_owner)==bool(recipe_handoff):
        raise ValueError('owned reconciliation requires exactly one genuine bare or resident authority')
    requested_authority=bare_direct_owner if bare_direct_owner is not None else {'recipe_handoff':recipe_handoff}
    request={'config':str(config_path),'operations':operations,'recipe':recipe_reference,'authority':requested_authority}
    if type(retain_mandatory_schedule)is not bool:raise ValueError('mandatory schedule continuation must be explicit boolean')
    if retain_mandatory_schedule:request['retain_mandatory_schedule']=True
    directory=out.parent/(out.stem+'-control');intent_path=directory/'intent.json'
    with contextlib.ExitStack()as stack:
        for path in sorted({root.parent/'controller.lock',Path(config['out'])/'controller.lock'}):
            lease=stack.enter_context(path.open('a'));fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if out.exists():
            committed=json.loads(out.read_text())
            if (committed.get('schema')!='natlang.native-owned-operation-reconciliation/1'
                    or committed.get('request_sha256')!=digest(request)
                    or committed.get('intent',{}).get('path')!=str(intent_path)):
                raise ValueError('owned reconciliation retry changed its request')
            completed_intent=json.loads(_bound(committed['intent']).read_text())
            if (completed_intent.get('request')!=request
                    or committed.get('published_checkpoint')!=completed_intent['operations'][-1]['checkpoint']
                    or committed.get('global_step')!=completed_intent['global_step']
                    or committed.get('acknowledged_finite_stages')!=completed_intent['new_stage_ids']
                    or committed.get('whole_recipe_completed')is not False):
                raise ValueError('owned reconciliation completed receipt changed its exact result')
            for reference in completed_intent['candidates'].values():_bound(reference)
            return binding(out)  # Historical success never rewinds subsequently earned work.
        if intent_path.exists():
            intent=json.loads(intent_path.read_text())
            if intent['request']!=request:raise ValueError('owned reconciliation intent belongs to another request')
        else:
            before={name:binding(path)for name,path in paths.items()}
            cursor=json.loads(paths['cursor'].read_text());state=json.loads(paths['state'].read_text())
            if cursor.get('phase')not in('rl','sft'):raise ValueError('owned reconciliation needs an actual recipe cycle')
            if state.get('phase')!='collect':
                raise ValueError('owned reconciliation cannot replace an active central operation')
            if cursor.get('pending_rl_round') is not None and state.get('round',0)<cursor['pending_rl_round']:
                raise ValueError('owned reconciliation cannot discard an unearned sealed policy round')
            old_recipe=resolve_recipe_data(config['production_recipe']);recipe=resolve_recipe_data(_bound(recipe_reference))
            old_stages=_supervised_stage_inventory(old_recipe,config['production_stage'])
            stages=_supervised_stage_inventory(recipe,config['production_stage']);ids=[stage['id']for stage in stages]
            if (type(cursor.get('next_stage',0))is not int or not 0<=cursor.get('next_stage',0)<=len(old_stages)
                    or any(not any(stage['id']==old['id']and stage['kind']==old['kind']for stage in stages)for old in old_stages)):
                raise ValueError('owned reconciliation changed or lost its old stage inventory')
            published={**binding(state['checkpoint']),'bytes':Path(state['checkpoint']).stat().st_size}
            chain=_owned_operation_chain(operations,published,config['raw_model_id'],config['recipe_lineage_sha256'])
            record,saved=_loaded_handoff(operations[-1]['handoff'],verified_handoff)
            from ..train.native_resident import code_inventory
            authority={'recipe_handoff':recipe_handoff}
            if bare_direct_owner is not None:
                authority['bare_direct_owner']={**bare_direct_owner,
                    'code':code_inventory(Path(bare_direct_owner['package'])),
                    'model_config_sha256':digest(saved['model_config'])}
            transition=_authority(authority,operations[-1]['handoff'],record,saved,config)
            if transition['recipe']!=recipe_reference:
                raise ValueError('owned reconciliation recipe differs from actual latest owner')
            earned_stages,operational_tuning=_owned_earned_stage_inventory(transition,stages,config['production_stage'])
            # A new standalone owner need not replay an earlier stage whose
            # finite operation the existing cursor already acknowledged. Its
            # exact carried frame/recipe is still verified below; this carries
            # scheduling history, never full-stage completion or extra visits.
            prior_stages=old_stages[:cursor.get('next_stage',0)] if cursor['phase']=='sft' else []
            if retain_mandatory_schedule:
                # A recovered external owner can publish real updates without
                # pretending to have executed the other mandatory stages.
                owned_ids={item['stage_id'] for item in chain}
                earned_stages=[stage for stage in earned_stages if stage['id'] in owned_ids]
                if owned_ids!={stage['id']for stage in earned_stages}:
                    raise ValueError('owned schedule continuation changed its actual stage inventory')
                ids=_acknowledged_owned_stage_ids(chain,earned_stages)
            else:
                _acknowledged_owned_stage_ids(chain,earned_stages,acknowledged_prefix=prior_stages)
            from ..train.native_continuation import collect_sft_progress
            from ..train.recipe import native_recipe_binding
            progress=collect_sft_progress(saved['training'],borrow_frames=True)
            for stage in earned_stages:
                frame=progress.get(stage['id'])or{};rb=(frame.get('options')or{}).get('recipe_binding')
                if isinstance(rb,str):rb=json.loads(rb)
                if not rb:raise ValueError('owned reconciliation lost a carried stage and its obligations')
                stage_recipe=_bound({'path':rb['path'],'sha256':rb.get('file_sha256',rb['sha256'])})
                if (native_recipe_binding(stage_recipe,config['recipe_lineage_sha256'])!=rb
                        or [item for item in resolve_recipe_data(stage_recipe)['stages']if item['id']==stage['id']]!=[stage]):
                    raise ValueError('owned reconciliation changed a carried stage instead of preserving its debt')
            next_state=_published_checkpoint_state(state,record['checkpoint'])
            next_cursor={**cursor,'phase':'sft','next_stage':0 if retain_mandatory_schedule else len(stages),'current_checkpoint':record['checkpoint'],
                'owned_operation_reconciliation':{'receipt':str(out),'request_sha256':digest(request),
                    'recipe_sha256':recipe_reference['sha256'],'runtime_configuration_required':True,
                    'acknowledged_finite_stages':ids,
                    **({'retain_mandatory_schedule':True}if retain_mandatory_schedule else {})}}
            next_cursor.pop('pending_rl_round',None);next_cursor.pop('readiness',None)
            directory.mkdir(parents=True,exist_ok=True)
            snapshots={};candidates={}
            for name,path in paths.items():
                snapshot=directory/('original-'+name+'.json');write_bytes_durable(snapshot,_bound(before[name]).read_bytes());snapshots[name]=binding(snapshot)
            for name,value in (('state',next_state),('cursor',next_cursor)):
                path=directory/('candidate-'+name+'.json');write_json_durable(path,value);candidates[name]=binding(path)
            if any(binding(path)!=before[name]for name,path in paths.items()):raise ValueError('owned control changed before durable intent')
            intent={'schema':'natlang.native-owned-operation-reconciliation/1','request':request,
                'before':before,'snapshots':snapshots,'candidates':candidates,'operations':chain,
                'old_stage_ids':[stage['id']for stage in old_stages],'new_stage_ids':ids,
                'previously_acknowledged_stage_ids':[stage['id']for stage in prior_stages],
                **({'retain_mandatory_schedule':True,'mandatory_stage_ids':[stage['id']for stage in stages]}if retain_mandatory_schedule else {}),
                'verified_operational_tuning':operational_tuning,
                'global_step':saved['training'].get('global_step',saved['training'].get('step')),
                'whole_recipe_completed':False}
            write_json_durable(intent_path,intent)
        if binding(paths['config'])!=intent['before']['config']:raise ValueError('owned controller config changed during commit')
        for name in('state','cursor'):
            actual=binding(paths[name]);candidate=intent['candidates'][name]
            if actual['sha256']!=candidate['sha256'] and actual!=intent['before'][name]:
                raise ValueError('owned reconciliation found an unknown successor')
        for name in('state','cursor'):
            candidate=intent['candidates'][name]
            if binding(paths[name])['sha256']!=candidate['sha256']:
                write_bytes_durable(paths[name],_bound(candidate).read_bytes())
        write_json_durable(out,{'schema':intent['schema'],'request_sha256':digest(request),
            'intent':binding(intent_path),'published_checkpoint':intent['operations'][-1]['checkpoint'],
            'global_step':intent['global_step'],'acknowledged_finite_stages':intent['new_stage_ids'],
            **({'retain_mandatory_schedule':True}if intent.get('retain_mandatory_schedule') else {}),
            'whole_recipe_completed':False})
    return binding(out)


def _publish_checkpoint(config,checkpoint):
    """Only the drained worker's actual checkpoint becomes the sole current parent."""
    out=Path(config['out']);out.mkdir(parents=True,exist_ok=True);path=out/'state.json'
    with (out/'controller.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .production import initial_controller_state
        state=json.loads(path.read_text()) if path.exists() else initial_controller_state(config)
        write_json_durable(path,_published_checkpoint_state(state,checkpoint))


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


def completed_raw_cursor(config,cursor,result):
    """The existing raw-to-supervised boundary, shared with stopped consumption."""
    if result.get('phase')!='collect':raise RuntimeError('raw RL operation needs its actual drained published boundary')
    if (cursor.get('phase')!='rl' or type(cursor.get('pending_rl_round'))is not int
            or result.get('round',-1)<cursor['pending_rl_round']):
        raise ValueError('completed raw cursor lost its actual pending round')
    updated={**cursor,'phase':'sft','next_stage':0,'current_checkpoint':_current_checkpoint(config)}
    updated.pop('pending_rl_round',None)
    return updated


def _raw_bridge_jobdir(work,cursor):
    """The cycle work directory already separates an explicitly retired attempt."""
    return Path(work)/'rl-bridge'


def cycle_work_directory(root,cursor):
    """Return the immutable work namespace for the current RAW attempt."""
    cycle=cursor.get('cycle');generation=cursor.get('raw_retry_generation',0)
    if type(cycle)is not int or cycle<0 or type(generation)is not int or generation<0:
        raise ValueError('RAW cycle work identity is malformed')
    path=Path(root)/'cycles'/f'{cycle:06d}'
    return path if generation==0 else path/f'retired-attempt-{generation:06d}'


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
    bridge_marker=cursor.get('deferred_owner_handoff_bridge')
    if bridge_marker:
        runtime_reference=config.get('runtime_handoff')
        if not runtime_reference:
            raise StudentConfigurationError('deferred owner bridge still requires its verified runtime handoff')
        from .native_runtime_handoff import verify_runtime_handoff
        runtime_proof=verify_runtime_handoff(runtime_reference,bootstrap=handoff)
        bridge_receipt=(json.loads(_bound(runtime_proof['deferred_owner_bridge']).read_text())
            if runtime_proof.get('deferred_owner_bridge') else {})
        if (runtime_proof.get('deferred_owner_bridge') is None
                or bridge_receipt.get('request_sha256')!=bridge_marker.get('request_sha256')
                or runtime_proof['checkpoint']!=bridge_marker.get('checkpoint')):
            raise StudentConfigurationError('deferred owner bridge runtime handoff does not match its boundary')
        if bridge_marker.get('runtime_handoff_required') not in (True,False):
            raise StudentConfigurationError('deferred owner bridge has an invalid runtime handoff state')
        if (bridge_marker.get('runtime_handoff_required') is False
                and bridge_marker.get('runtime_handoff')!=runtime_reference):
            raise StudentConfigurationError('consumed deferred owner bridge names another runtime handoff')
        cursor['deferred_owner_handoff_bridge_consumed']={
            'request_sha256':bridge_marker['request_sha256'],'bridge':runtime_proof['deferred_owner_bridge'],
            'runtime_handoff':runtime_reference,'checkpoint':bridge_marker['checkpoint'],
            'whole_recipe_completed':False,'training_credit':0,'raw_completion_credit':0}
        cursor.pop('deferred_owner_handoff_bridge',None)
        write_json_durable(cursor_path,cursor)
    reconciliation=cursor.get('owned_operation_reconciliation')or{}
    if reconciliation.get('runtime_configuration_required'):
        if binding(config['production_recipe'])['sha256']!=reconciliation['recipe_sha256']:
            raise StudentConfigurationError('owned completed work requires its actual current runtime configuration')
        reconciliation['runtime_configuration_required']=False;write_json_durable(cursor_path,cursor)
    cycle=cursor['cycle'];work=cycle_work_directory(directory,cursor)
    if cursor['phase']=='rl':
        if stop.requested:return cursor
        production_state=Path(config['out'])/'state.json'
        state=json.loads(production_state.read_text()) if production_state.exists() else {'round':0}
        if 'pending_rl_round' not in cursor:
            cursor['pending_rl_round']=state['round']+1
            write_json_durable(cursor_path,cursor)
        bridge_job=_raw_bridge_jobdir(work,cursor)
        result=(state if state['round']>=cursor['pending_rl_round'] else
            round_job(config_path,config,bridge_job,stop))
        if stop.requested:return cursor
        cursor=completed_raw_cursor(config,cursor,result)
        write_json_durable(cursor_path,cursor)
    recipe=resolve_recipe_data(config['production_recipe'])
    stages=_supervised_stage_inventory(recipe,config['production_stage'])
    foundation=[stage for stage in stages if stage['id']==config['production_stage']]
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
        if is_foundation and cursor.get('foundation_readiness_repair'):
            parameters['refresh_foundation_readiness']=True
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
        continuation['trainer_parameters']['refresh_foundation_readiness']=True
        if config.get('resident_data_directory'):
            continuation['trainer_parameters']['resident_data_directory']=config['resident_data_directory']
        continuation['raw_foundation_qualification']=True
        _supervised_job(continuation,jobdir,config,cursor,stop)
        if stop.requested:return cursor
    cursor.pop('raw_retry_generation',None)
    cursor.update(cycle=cycle+1,phase=('rl' if cursor.get('foundation_qualified') else 'sft'),
        next_stage=0,current_checkpoint=_current_checkpoint(config))
    write_json_durable(cursor_path,cursor)
    return cursor


def _supervised_job(continuation,jobdir,config,cursor,stop):
    from .native_rounds import round_job
    if Path(continuation['out']).resolve() != Path(jobdir).resolve():
        raise ValueError('supervised operation configuration/output differs from its owned job directory')
    path=Path(jobdir)/'config.json';path.parent.mkdir(parents=True,exist_ok=True)
    write_json_durable(path,continuation)
    result=round_job(path,continuation,Path(jobdir)/'bridge',stop)
    if stop.requested:return
    _publish_supervised_result(continuation,jobdir,config,cursor,result)


def _publish_supervised_result(continuation,jobdir,config,cursor,result):
    """Publish the same already authenticated result in live and recovered paths."""
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
            cursor.pop('foundation_readiness_repair',None)
        elif continuation.get('raw_foundation_qualification'):
            cursor.pop('foundation_qualified',None)
            cursor['foundation_readiness_repair']={'report':binding(report_path),
                'checkpoint':checkpoint,'readiness_granted':False,
                'reason':'Current foundation runtime/task qualification incomplete; continue supervised repair'}
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
            _resident_configuration(config,state,source,checkpoint)
            write_json_durable(Path(config['out']).parent/'config.json',config)


def _resident_configuration(config,state,source,checkpoint):
    """Project an already authenticated resident generation into controller inputs."""
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
    _supervised_quantum(config,config['seed_trainer_parameters'])
    return config


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
        if proof.get('bare_direct_owner'):
            from ..train.recipe import recipe_identity_sha256
            requirement['recipe_sha256']=recipe_identity_sha256(_bound(proof['bare_direct_owner']['recipe']))
            requirement['recipe_lineage_sha256']=proof['lineage']
        retired_cursor=json.loads(_bound(proof['snapshots']['cursor']).read_text()).get('resident_current')
    if transition_ref:
        from .native_raw_handoff import verify_drained_recipe_handoff
        transition=json.loads(_bound(transition_ref).read_text())
        transition=verify_drained_recipe_handoff(transition_ref,handoff=transition['handoff'],
            **({'verified_handoff':verified_handoff} if verified_handoff is not None and transition_handoff is None else {}))
        from ..train.recipe import recipe_identity_sha256
        requirement['recipe_sha256']=recipe_identity_sha256(_bound(transition['recipe']))
        requirement['recipe_lineage_sha256']=json.loads(_bound(transition['new_plan']).read_text())['recipe_sha256']
    if declaration.get('completed_supervised_adaptation'):
        from .native_completed_supervised import verify_declaration
        from ..train.recipe import recipe_identity_sha256
        adapted=verify_declaration(declaration['completed_supervised_adaptation'],declaration)
        if requirement.get('recipe_lineage_sha256',adapted['lineage'])!=adapted['lineage']:
            raise ValueError('raw release completed adaptation changed its preserved recipe lineage')
        requirement['recipe_sha256']=recipe_identity_sha256(_bound(adapted['recipe']))
        requirement['recipe_lineage_sha256']=adapted['lineage']
    reference=value.get('resident_current')
    # A later handoff explicitly retired this exact historical generation.
    # Keep the cursor receipt, but never reinterpret it as the new live plan.
    if transition_handoff is not None and reference==retired_cursor:return requirement
    return _resident_release_requirement(requirement,value,transition=transition)


def _resident_release_requirement(requirement,value,*,transition=None):
    """Metadata-only committed source history, after caller authenticates its root."""
    reference=value.get('resident_current')
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


def validate_release(path,plan_path,*,allow_live=False,eligibility=False):
    from ..train.native_resident import owner_alive
    record=json.loads(Path(path).read_text());plan=json.loads(Path(plan_path).read_text())
    from .native_role_release import ownership
    owned,exchange=ownership(plan)
    if (record.get('schema')!=SCHEMA or record.get('plan_sha256')!=sha256_file_hex_cached(plan_path)
            or(record.get('released')is not True and not(eligibility and exchange and record.get('released')is False))):return None
    if not allow_live and owner_alive(record.get('owner') or {}):return None
    models=record.get('models') or {}
    if set(models)!=owned:raise ValueError('raw release omitted an actual student owner')
    if exchange is not None and record.get('owned_models')!=sorted(owned):
        raise ValueError('raw release changed its explicit source-host ownership')
    result={}
    from ..train.native_readiness import validate_ready
    for declaration in plan['models']:
        if declaration['model']not in owned:continue
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
    from . import native_role_release as roles
    owned_models,exchange=roles.ownership(plan);role_contract=roles.contract(plan)if exchange else None
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    owned=stop is None;stop=StopSignal().install() if owned else stop
    try:
        with (out/'controller.lock').open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state_path=out/'state.json'
            if state_path.exists() and validate_release(state_path,plan_path) is not None:return json.loads(state_path.read_text())
            state={'schema':SCHEMA,'plan_sha256':sha256_file_hex_cached(plan_path),'owner':process_owner(),
                'released':False,'models':{},'production_complete':False,
                **({'owned_models':sorted(owned_models)}if exchange else {})}
            counts={name:0 for name in owned_models}
            eligible=set();rendezvous=False;active={};blocked={}
            if exchange:
                write_json_durable(state_path,state);roles.withdraw(plan,state_path)
            with ThreadPoolExecutor(max_workers=len(owned_models)) as pool:
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
                        if roles.ownership(updated)!=(owned_models,exchange)or(exchange and roles.contract(updated)!=role_contract):
                            raise ValueError('live raw plan cannot change source-host role ownership or fixed cohorts')
                        plan=updated
                        state['plan_sha256']=sha256_file_hex_cached(plan_path)
                        models={row['model']:row for row in plan['models']if row['model']in owned_models}
                        if exchange and eligible==owned_models and not stop.requested:
                            try:rendezvous=roles.eligible(plan)
                            except(ValueError,OSError,KeyError)as error:
                                rendezvous=False;state['role_exchange_error']={'error_type':type(error).__name__,'error':str(error)}
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
                        if eligible==set(models):
                            if exchange:
                                write_json_durable(state_path,state)
                                try:
                                    roles.publish(plan_path,state_path,kind='eligible')
                                    rendezvous=roles.eligible(plan)
                                    state.pop('role_exchange_error',None)
                                except(ValueError,OSError,KeyError)as error:
                                    rendezvous=False;state['role_exchange_error']={'error_type':type(error).__name__,'error':str(error)}
                            else:rendezvous=True
                        elif exchange:
                            write_json_durable(state_path,state);roles.withdraw(plan,state_path)
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
