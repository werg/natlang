"""Whole collected batches become durable obligations of the joint coordinator.

The declared curriculum chooses policy work. Exact routing retains complete
relative-reward groups, source bytes, all call occurrences and original actors;
no captured turn is split off for a separate raw learner.
"""
from __future__ import annotations
import json
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached,canonical_json_sha256_hex
from ..common.jsonio import iter_jsonl_rows,write_json_durable
from ..train.native_readiness import _bound


def has_semantic_inputs(rows):
    return any('semantic_adaptation' in segment or 'source_state_id' in segment
        for row in rows for turn in row.get('turns',[]) for segment in (turn.get('sampling') or {}).get('prompt',[])
        if segment.get('kind')=='native_state')


def bind_policy_source(config,values,rows,mode):
    if mode not in {'grpo','rft'} or not has_semantic_inputs(rows):return None
    if config.get('coefficient_policy'):
        raise ValueError('semantic coefficient actors require their exactly trained shared participant bindings')
    from .episode_graph import validate_rollout_graph
    for row in rows:
        if row.get('reward') is not None:validate_rollout_graph(row,semantic=True)
    def binding(path):
        path=Path(path).resolve(strict=True);return {'path':str(path),'sha256':sha256_file_hex_cached(path)}
    source={'rollouts':binding(values['rollouts']),'block_bundle':binding(values['block_bundle']),
        'adapter':binding(values['semantic_adapter']),'mode':mode,'collector_participant':config['native_participant']}
    adapter=json.loads(_bound(source['adapter']).read_text())
    archive=json.loads(_bound(source['block_bundle']).read_text())
    if (adapter.get('checkpoint',{}).get('sha256')!=values['policy_id']
            or archive.get('policy_id')!=values['policy_id']
            or config['native_participant'] not in adapter.get('models',{})):
        raise ValueError('semantic policy collection changed its exact joint parent, codec or archive owner')
    return source


def policy_curriculum(config_path,sources,out):
    """Bind actual newly collected sources to the existing shared curriculum."""
    original=Path(config_path).resolve(strict=True)
    value=json.loads(original.read_text());schedule=value.get('policy_schedule')
    if not isinstance(schedule,dict):
        raise ValueError('semantic policy routing requires an explicit shared policy_schedule')
    declaration={**schedule,'mode':'mixed','sources':[]}
    for source in [*(value.get('policy_training') or {}).get('sources',[]),*sources]:
        for key in ('rollouts','block_bundle','adapter'):_bound(source[key])
        if source not in declaration['sources']:declaration['sources'].append(source)
    from ..train.native_semantic_policy import validate_policy_training
    validate_policy_training(declaration,[phase['id'] for phase in value['phases']])
    value['policy_training']=declaration
    out=Path(out);write_json_durable(out,value)
    return out


def accept_policy_deployment(config,state,deployment):
    """Clear collection ownership only after exact checkpointed admission."""
    source=state.get('pending_semantic_source')
    if source is None:return False
    progress=deployment.get('policy_training_progress') or {}
    if source not in progress.get('policy_source_history',[]):
        if state.get('checkpoint')==deployment.get('checkpoint',{}).get('path'):
            return False  # Reapplying the previous parent before this round joins.
        raise ValueError('joint deployment omitted a pending whole semantic batch obligation')
    completion=json.loads(_bound(deployment['shared_stage_completion']).read_text())
    if completion.get('policy_training_progress')!=progress:
        raise ValueError('joint deployment changed actual policy checkpoint progress')
    phase=progress['policy_training_phase'];counts=completion['exposure'].get(phase,{})
    trained=set(state.get('trained_task_ids',[]));new_visits=0
    for _,row in iter_jsonl_rows(_bound(source['rollouts'])):
        visits=counts.get(canonical_json_sha256_hex(row),{}).get('episode',0)
        if visits:
            trained.add(row['task_id']);new_visits+=visits
    state.update(round=state['round']+1,phase='collect',collection_attempt=0,
        trained_rounds=state.get('trained_rounds',0)+int(new_visits>0),trained_task_ids=sorted(trained),
        last_learner_progress={'schema':'natlang.native-semantic-policy-admission/1','source':source,
            'actual_episode_visits':new_visits,'remaining_obligations':progress['policy_obligations'],
            'checkpoint':deployment['checkpoint']})
    if 'pending_semantic_next_offset' in state:state['offset']=state.pop('pending_semantic_next_offset')
    state.pop('pending_semantic_source');state.pop('pending_semantic_round_dir',None)
    return True


def publish_pending(config,state,*,stop=None):
    """Resume the same finite shared stage; publish only its qualified result."""
    from .native_refresh import refresh
    source=state['pending_semantic_source'];directory=Path(state['pending_semantic_round_dir'])
    curriculum=policy_curriculum(config['semantic_refresh_config'],[source],directory/'policy-curriculum.json')
    deployment=refresh(adapter=_bound(source['adapter']),updates={},config=curriculum,
        out=directory/'semantic-refresh',device=config.get('refresh_device','cuda'),stop=stop)
    accept_policy_deployment(config,state,deployment)
    state.update(checkpoint=deployment['checkpoint']['path'],semantic_adapter=deployment['semantic_adapter']['path'])
    if config.get('after_update'):
        from .production import run_command,RunControl
        values={**config,'out':str(directory),'checkpoint':state['checkpoint'],
            'policy_id':deployment['checkpoint']['sha256'],'semantic_adapter':state['semantic_adapter'],
            'semantic_adapter_sha256':deployment['semantic_adapter']['sha256'],
            'deployment':str(directory/'semantic-refresh'/'deployment.json')}
        if stop is not None:values['_run_control']=RunControl(stop)
        run_command(config['after_update'],values)
    return deployment
