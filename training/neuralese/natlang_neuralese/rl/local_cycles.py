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
        cursor_path=out/'cursor.json';completed=0
        while cycles is None or completed<cycles:
            if stop is not None and stop.requested:return
            plan=json.loads(plan_path.read_text());config_path=Path(plan['rl_config'])
            config=resolve_recipe_data(config_path)
            if config.get('native_actor_mode')!='raw-state':
                raise ValueError('single-device local cycles require the native RAW actor')
            state_path=Path(config['out'])/'state.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else production.initial_controller_state(config)
            cursor=json.loads(cursor_path.read_text()) if cursor_path.exists() else {
                'cycle':0,'phase':'rl','rl_round_target':state['round']+1,'stage_index':0}
            write_json_durable(cursor_path,cursor)
            if cursor['phase']=='rl':
                if round_target(state['round'],cursor['rl_round_target']):
                    result=production.run(config_path,rounds=1,stop=stop)
                    if result['round']!=cursor['rl_round_target']:
                        return result  # Incomplete/awaiting work is not an earned round.
                cursor.update(phase='supervised',stage_index=0)
                write_json_durable(cursor_path,cursor)
            # RL can take long enough for new sources or cadence improvements
            # to become ready. Consume the actual current declarations before
            # reserving each next supervised operation, including after RL.
            plan,config,stages=reload_schedule(plan_path,plan,config)
            while True:
                if stop is not None and stop.requested:return
                plan,config,stages=reload_schedule(plan_path,plan,config,stages,cursor=cursor)
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
                    continuation,_=continuation_config(config,stage,checkpoint=parent['path'],adapter=None,out=directory)
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
                child=binding(result['pending_native_checkpoint'])
                publish_supervised(config,parent,child)
                cursor.get('failed_stages',{}).pop(stage['id'],None)
                cursor['stage_index']+=1
                cursor['checkpoint']=child
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
