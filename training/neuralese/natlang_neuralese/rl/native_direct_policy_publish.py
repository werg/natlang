"""Publish a completed direct same-policy group without inventing missing groups."""
from __future__ import annotations
import fcntl,json,math
from pathlib import Path
from ..common.jsonio import read_jsonl,write_json_durable
from ..train.native_readiness import _bound
from ..train.native_resident import binding


def publish(state_path, *, report_ref, receipt_ref, custody_ref, provenance_ref, objective_ref, log_ref, out):
    from .native_raw_rounds import _published_checkpoint_state
    state_path,out=Path(state_path).resolve(),Path(out).resolve()
    def read(ref):return json.loads(_bound(ref).read_text())
    with (state_path.parent/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(state_path.read_text());report,receipt,custody,provenance,objective=map(read,(report_ref,receipt_ref,custody_ref,provenance_ref,objective_ref))
        log=list(read_jsonl(_bound(log_ref)))
        parent,child=custody['parent'],custody['successor'];parent_path,child_path=_bound(parent),_bound(child)
        if state.get('phase')!='collect' or Path(state['checkpoint']).resolve()!=parent_path:
            raise ValueError('direct policy publication lost its parent controller')
        if (custody.get('schema')!='natlang.earned-rl-checkpoint-custody/1'
                or custody.get('sft_source_quota_exposure_unchanged') is not True
                or child['global_step']!=parent['global_step']+1
                or not all(x.get('optimizer_present') and x.get('rng_present') for x in (parent,child))
                or report.get('completed') is not True or report.get('numerics_ok') is not True
                or report.get('checkpoint_sha256')!=child['sha256']
                or receipt.get('checkpoint',{}).get('sha256')!=child['sha256']
                or receipt.get('continuation',{}).get('checkpoint_sha256')!=parent['sha256']):
            raise ValueError('direct policy publication lacks completed full-state custody')
        certificate=report.get('runtime_certificate') or {}
        if certificate.get('passed') is not True or read(certificate).get('passed') is not True:
            raise ValueError('direct policy publication requires its passed runtime certificate')
        rollouts=list(read_jsonl(_bound(provenance['rollouts'])));score=read(provenance['score'])
        rows=objective.get('rows') or {};values=list(rows.values()) if isinstance(rows,dict) else rows
        if (provenance.get('schema')!='natlang.completed-same-policy-group/1'
                or provenance.get('checkpoint_sha256')!=parent['sha256'] or len(rollouts)!=6 or len(values)!=6
                or len({(r.get('group'),r.get('sample')) for r in rollouts})!=6
                or len({r.get('task_id') for r in rollouts})!=1
                or any((r.get('policy') or {}).get('checkpoint_sha256')!=parent['sha256'] for r in rollouts)
                or any(not (v.get('jolt') or {}).get('privileged') for v in values)
                or len(log)!=1 or log[0].get('global_step')!=child['global_step']):
            raise ValueError('direct policy publication lacks its complete same-policy privileged group')
        details=(log[0].get('objective_details') or {}).get('jolt') or {};opsd=details.get('student_opsd')
        if (details.get('privileged_episodes')!=6 or type(opsd) not in (int,float) or not math.isfinite(opsd)
                or opsd<=0 or details.get('student_outcome')!=0):
            raise ValueError('direct policy publication did not perform the declared text-twin update')
        task=rollouts[0]['task_id'];groups=[g for g in score.get('groups',[]) if g.get('task_id')==task]
        if len(groups)!=1 or groups[0].get('scored')!=6:
            raise ValueError('direct policy publication group score is incomplete')
        if out.exists():raise ValueError('direct policy publication intent already exists')
        result={'schema':'natlang.direct-policy-group-publication/1','parent':binding(parent_path),'checkpoint':binding(child_path),
                'report':report_ref,'receipt':receipt_ref,'custody':custody_ref,'group_provenance':provenance_ref,
                'objective':objective_ref,'training_log':log_ref,'task_id':task,'samples':6,'student_opsd':opsd,
                'uncompleted_groups_credited':0,'breadth_offset_before':state['offset'],'breadth_offset_after':state['offset']}
        write_json_durable(out,result)
        next_state=_published_checkpoint_state(state,{'path':str(child_path),'sha256':child['sha256']})
        next_state['round']=state['round']+1;next_state['trained_rounds']=state['trained_rounds']+1
        next_state['collection_attempt']=0;next_state['group_stats']=[]
        next_state.setdefault('task_curriculum_history',{})[task]={**groups[0],'observed_policy_id':parent['sha256'],'observed_round':state['round'],'collection_attempt':2}
        for key in ('collected_task_ids','trained_task_ids'):
            next_state[key]=sorted(set(next_state.get(key,[]))|{task})
        next_state.setdefault('direct_policy_update_history',[]).append(binding(out))
        write_json_durable(state_path,next_state)
        return result
