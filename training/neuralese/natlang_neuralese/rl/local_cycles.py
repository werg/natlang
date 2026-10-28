"""Single-device RAW RL and declared supervised recipe continuations.

Policy lives in the supplied plan and recipe. This adapter serializes existing
whole-graph production workers and commits their real checkpoints; it implements
no model, reward, graph replay or optimizer. A durable round target prevents an
RL round from being repeated after publication but before a cursor write.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from ..common.jsonio import write_json_durable
from . import production
from .native_raw_handoff import binding
from .native_raw_rounds import _publish_checkpoint, _supervised_quantum
from .native_supervised import continuation_config, MANDATORY_STAGE_KINDS
from ..train.recipe import resolve_recipe_data


def bind_state_semantic_route(config,state):
    """Use the exact returned semantic inference/control route after VAE adoption."""
    from ..common.hashing import sha256_file_hex_cached
    participant=state.get('native_participant')
    if participant is not None:
        if config.get('native_participant') not in (None,participant):
            raise ValueError('local-cycle state and recipe changed the native participant')
        config={**config,'native_participant':participant}
    reference=state.get('semantic_refresh_config')
    if reference is not None:
        if (not isinstance(reference,dict)or not {'path','sha256'}<=set(reference)
                or set(reference)-{'path','sha256','bytes'}):
            raise ValueError('local-cycle returned semantic refresh config has no exact binding')
        path=Path(reference['path']).resolve(strict=True)
        if sha256_file_hex_cached(path)!=reference['sha256']:
            raise ValueError('local-cycle semantic refresh curriculum changed after adoption')
        declared=config.get('semantic_refresh_config')
        if declared is not None and Path(declared).resolve()!=path:
            raise ValueError('local-cycle recipe changed its adopted semantic refresh curriculum')
        config={**config,'semantic_refresh_config':str(path)}
    return config


def round_target(current_round, completed_target):
    if current_round > completed_target:
        raise ValueError('local RL cursor is behind an unknown controller successor')
    return current_round < completed_target


def publish_supervised(config, parent, checkpoint):
    """Retry a publication only against its exact parent or its earned child."""
    state_path=Path(config['out'])/'state.json'
    state=json.loads(state_path.read_text())
    actual=binding(state['checkpoint'])
    if actual not in (parent, checkpoint):
        raise ValueError('supervised publication found an unknown policy successor')
    if actual != checkpoint:
        _publish_checkpoint(config, checkpoint)


def publish_semantic_supervised(config,parent,child,deployment):
    """Publish a real CE/DPO descendant only after the existing shared refresh qualifies it."""
    from ..common.hashing import sha256_file_hex_cached
    from ..train.native_readiness import _bound
    state_path=Path(config['out'])/'state.json'
    state=json.loads(state_path.read_text())
    current=binding(state['checkpoint'])
    checkpoint=deployment.get('checkpoint')or{}
    adapter=deployment.get('semantic_adapter')or{}
    if (deployment.get('schema')!='natlang.native-rl-deployment/1'
            or not checkpoint.get('path')or not checkpoint.get('sha256')
            or not adapter.get('path')or not adapter.get('sha256')):
        raise ValueError('local semantic supervised refresh returned no exact current deployment')
    completion=deployment.get('shared_stage_completion')or{}
    completion_path=_bound(completion)
    report=json.loads(completion_path.read_text())
    if (report.get('completed')is not True or report.get('production_ready')is not True
            or report.get('checkpoint',{}).get('sha256')!=checkpoint['sha256']
            or report.get('adapter',{}).get('sha256')!=adapter['sha256']):
        raise ValueError('local supervised semantic refresh lacks complete held/runtime/task qualification')
    if sha256_file_hex_cached(adapter['path'])!=adapter['sha256']:
        raise ValueError('local supervised semantic adapter bytes changed')
    semantic=json.loads(Path(adapter['path']).read_text())
    participant=config.get('native_participant')
    updated=(deployment.get('updated_participants')or{}).get(participant)
    if (semantic.get('schema')!='natlang.native-semantic-adapter/1'
            or semantic.get('checkpoint',{}).get('sha256')!=checkpoint['sha256']
            or participant not in semantic.get('models',{})
            or not isinstance(updated,dict)
            or updated.get('sha256')!=binding(child)['sha256']):
        raise ValueError('local supervised deployment lost its exact refreshed participant update')
    expected_adapter=state.get('semantic_adapter',config.get('semantic_adapter'))
    same_parent=all(current.get(key)==parent.get(key)for key in('path','sha256'))
    same_child=all(current.get(key)==checkpoint.get(key)for key in('path','sha256'))
    if (not (same_parent or same_child)
            or (expected_adapter and deployment.get('parent_adapter_sha256')!=sha256_file_hex_cached(expected_adapter))):
        raise ValueError('local supervised refresh changed its actual current parent or semantic route')
    if current!=checkpoint:
        deployment_path=Path(completion_path).parent/'deployment.json'
        if not deployment_path.is_file() or json.loads(deployment_path.read_text())!=deployment:
            raise ValueError('local supervised refresh did not durably publish this exact deployment')
        next_state={**state,'checkpoint':checkpoint['path'],'semantic_adapter':adapter['path'],
            'semantic_deployment':binding(deployment_path),
            'semantic_deployment_parent_adapter_sha256':deployment.get('parent_adapter_sha256'),
            'last_supervised_semantic_refresh':{'parent':parent,'child':binding(child),
                'checkpoint':checkpoint,'semantic_adapter':adapter,'completion':completion,
                'training_credit':0}}
        write_json_durable(state_path,next_state)
    return {'checkpoint':checkpoint,'semantic_adapter':adapter,'completion':completion}


def stage_failure_signature(stage, config):
    from ..common.hashing import canonical_json_sha256_hex
    return canonical_json_sha256_hex({'stage':stage,
        'recipe':binding(config['production_recipe']),
        'runtime':config.get('native_actor_python_runtime')})


def record_unstarted_stage_failure(cursor, stage, config, directory, parent, error, *, enabled):
    """Keep useful stages moving only when the failed child has no saved state.

    Any saved child requires exact recovery; never advance from an obsolete
    parent and silently lose an earned continuation. Stage index is a scheduling
    cursor, not exposure credit. Failed obligations remain explicitly outstanding.
    """
    if not enabled or (Path(directory)/'learner/checkpoint.pt').exists():raise error
    state=json.loads((Path(config['out'])/'state.json').read_text())
    if binding(state['checkpoint'])!=parent:
        raise ValueError('failed supervised worker found an unknown policy successor') from error
    failure={'at':time.time(),'stage_id':stage['id'],'operation':str(directory),
        'signature':stage_failure_signature(stage,config),'returncode':error.returncode,
        'reason':'Worker failed before publishing any saved child; preserve outstanding exposure and continue other declared work.',
        'completed':False,'parent':parent}
    cursor.setdefault('failed_stages',{})[stage['id']]=failure
    cursor['stage_index']+=1
    write_json_durable(Path(directory)/'unstarted-worker-failure.json',failure)
    return failure


def reload_schedule(plan_path, previous_plan, previous_config, previous_stages=(), *, cursor=None):
    """Reload future work while retaining the actual optimizer/controller owner.

    An existing operation keeps its frozen declaration. Changing a completed
    prefix or replacing a pending operation's ID needs an explicit drained
    reschedule; it cannot silently drop earned work or reuse an old bridge.
    """
    plan=json.loads(Path(plan_path).read_text())
    if plan.get('local_cycle_role_plan') is not None:
        raise ValueError('local-cycle plan field local_cycle_role_plan served the removed cross-model role exchange')
    config=resolve_recipe_data(Path(plan['rl_config']))
    publication=config.get('native_checkpoint_publication') or {}
    previous_publication=previous_config.get('native_checkpoint_publication') or {}
    if (any(Path(plan[key]).resolve()!=Path(previous_plan[key]).resolve()
            for key in ('out','optimizer_owner_lock'))
            or Path(config['out']).resolve()!=Path(previous_config['out']).resolve()
            or config.get('native_actor_mode')!='raw-state'
            or config.get('model')!=previous_config.get('model')
            or any(config.get(key)!=previous_config.get(key) for key in (
                'native_checkpoint_publication_owner_pointer',
                'native_checkpoint_publication_runtime_root',
                'native_checkpoint_publication_role', 'native_participant'))
            or any(publication.get(key)!=previous_publication.get(key)
                   for key in ('namespace','producer','dependency_producer'))
            or not set(previous_publication.get('readonly_input_roots',[])).issubset(
                publication.get('readonly_input_roots',[]))
            or config.get('native_actor_python_runtime')!=previous_config.get('native_actor_python_runtime')):
        raise ValueError('live schedule changed optimizer/controller/runtime ownership; drain before rescheduling')
    recipe=resolve_recipe_data(config['production_recipe'])
    stages=[s for s in recipe['stages'] if s['kind'] in MANDATORY_STAGE_KINDS and s.get('admitted') is not False]
    if cursor is not None and previous_stages:
        index=cursor['stage_index'];before=[s['id'] for s in previous_stages];after=[s['id'] for s in stages]
        if after[:index]!=before[:index]:
            raise ValueError('live schedule changed its visited stage prefix; drain before rescheduling')
        if index<len(before):
            operation=Path(plan['out'])/f"cycle-{cursor['cycle']:06d}"/before[index]
            if (operation/'config.json').exists() and (index>=len(after) or after[index]!=before[index]):
                raise ValueError('live schedule replaced a pending operation; drain before rescheduling')
    return plan,config,stages


def run(plan_path, *, cycles=None, stop=None):
    plan_path=Path(plan_path)
    initial=json.loads(plan_path.read_text())
    out=Path(initial['out']);out.mkdir(parents=True,exist_ok=True)
    # The same exclusive owner descriptor protects standalone SFT launches too.
    with Path(initial['optimizer_owner_lock']).open('a') as owner, (out/'cycles.lock').open('a') as lease:
        for lock in (owner,lease):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in ('local_cycle_role_plan','semantic_return'):
            if initial.get(key) is not None:
                raise ValueError(f'local-cycle plan field {key} served the removed cross-model role exchange')
        cursor_path=out/'cursor.json';completed=0
        while cycles is None or completed<cycles:
            if stop is not None and stop.requested:return
            plan=json.loads(plan_path.read_text());config_path=Path(plan['rl_config'])
            config=resolve_recipe_data(config_path)
            if config.get('native_actor_mode')!='raw-state':
                raise ValueError('single-device local cycles require the native RAW actor')
            state_path=Path(config['out'])/'state.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else production.initial_controller_state(config)
            if state_path.exists() and production.normalize_legacy_completed_update(state,state_path):
                write_json_durable(state_path,state)
            config=bind_state_semantic_route(config,state)
            cursor=json.loads(cursor_path.read_text()) if cursor_path.exists() else {
                'cycle':0,'phase':'rl','rl_round_target':state['round']+1,'stage_index':0}
            write_json_durable(cursor_path,cursor)
            if cursor['phase']=='rl':
                if round_target(state['round'],cursor['rl_round_target']):
                    result=production.run(config_path,rounds=1,stop=stop)
                    if result['round']!=cursor['rl_round_target']:
                        return result  # Incomplete/awaiting work is not an earned round.
                cursor.update(phase='supervised',stage_index=0)
                cursor['checkpoint']=binding(json.loads(state_path.read_text())['checkpoint'])
                write_json_durable(cursor_path,cursor)
            # RL can take long enough for new sources or cadence improvements
            # to become ready. Consume the actual current declarations before
            # reserving each next supervised operation, including after RL.
            plan,config,stages=reload_schedule(plan_path,plan,config)
            config=bind_state_semantic_route(config,json.loads(state_path.read_text()))
            while True:
                if stop is not None and stop.requested:return
                plan,config,stages=reload_schedule(plan_path,plan,config,stages,cursor=cursor)
                config=bind_state_semantic_route(config,json.loads(state_path.read_text()))
                if cursor['stage_index']>=len(stages):break
                index=cursor['stage_index'];stage=stages[index]
                failure=cursor.get('failed_stages',{}).get(stage['id'])
                if (plan.get('continue_after_unstarted_stage_failure') and failure
                        and failure['signature']==stage_failure_signature(stage,config)):
                    cursor['stage_index']+=1
                    write_json_durable(cursor_path,cursor)
                    continue
                directory=out/f"cycle-{cursor['cycle']:06d}"/stage['id']
                declaration=directory/'config.json'
                if not declaration.exists():
                    parent=binding(json.loads(state_path.read_text())['checkpoint'])
                    current_state=json.loads(state_path.read_text())
                    continuation,_=continuation_config(config,stage,checkpoint=parent['path'],
                        adapter=current_state.get('semantic_adapter'),out=directory)
                    continuation.update(supervised_quantum_updates=plan['supervised_quantum_updates'],
                        supervised_stage_quantum_updates=plan.get('supervised_stage_quantum_updates',{}))
                    _supervised_quantum(continuation,continuation['trainer_parameters'])
                    directory.mkdir(parents=True,exist_ok=True)
                    write_json_durable(directory/'parent.json',parent)
                    write_json_durable(declaration,continuation)
                parent=json.loads((directory/'parent.json').read_text())
                try:
                    result=production.run(declaration,stop=stop)
                except subprocess.CalledProcessError as error:
                    record_unstarted_stage_failure(cursor,stage,config,directory,parent,error,
                        enabled=plan.get('continue_after_unstarted_stage_failure') is True)
                    write_json_durable(cursor_path,cursor)
                    continue
                if not result.get('pending_native_checkpoint'):return result
                raw_child=binding(result['pending_native_checkpoint'])
                current_state=json.loads(state_path.read_text())
                if current_state.get('semantic_adapter'):
                    from .native_rounds import refresh_job
                    refresh_dir=directory/'semantic-refresh'
                    deployment=refresh_job([continuation],current_state['semantic_adapter'],
                        {config['native_participant']:raw_child['path']},
                        continuation['semantic_refresh_config'],refresh_dir,
                        continuation.get('refresh_device','cuda'),stop)
                    refreshed=publish_semantic_supervised(config,parent,raw_child,deployment)
                    child=refreshed['checkpoint']
                else:
                    publish_supervised(config,parent,raw_child)
                    child=raw_child
                cursor.get('failed_stages',{}).pop(stage['id'],None)
                cursor['stage_index']+=1
                cursor['checkpoint']=child
                if current_state.get('semantic_adapter'):
                    cursor['semantic_adapter']=refreshed['semantic_adapter']
                write_json_durable(cursor_path,cursor)
            state=json.loads(state_path.read_text())
            cursor.update(cycle=cursor['cycle']+1,phase='rl',stage_index=0,rl_round_target=state['round']+1)
            write_json_durable(cursor_path,cursor);completed+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--cycles',type=int)
    args=parser.parse_args()
    from ..train.loop import StopSignal
    stop=StopSignal().install()
    try:run(args.plan,cycles=args.cycles,stop=stop)
    finally:stop.restore()


if __name__=='__main__':main()
