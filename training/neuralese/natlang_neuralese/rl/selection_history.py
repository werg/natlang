"""Carry selector observations across controllers without replay or training credit."""
import json
import math
import re
from pathlib import Path
from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import iter_jsonl_rows, write_json_durable
from .native_direct_sft_recovery import _verify_historical_file_ref


def snapshot(controller_ref, directory, cases):
    from .native_supervised import snapshot_execution_source_controller
    source=_verify_historical_file_ref(controller_ref)
    state=json.loads(source.read_text())
    if not state.get('task_curriculum_history'):return None
    ref=snapshot_execution_source_controller(controller_ref,directory)
    record={'schema':'natlang.task-selector-history/1','source_controller':ref,
            'cases_sha256':sha256_file_hex_cached(cases,trust_archive_receipt=False),
            'scope':'historical task selection only; no current-policy rollouts, model qualification, trained rounds or optimizer exposure'}
    path=Path(directory)/('task-selector-history-'+ref['sha256']+'-'+record['cases_sha256']+'.json')
    if path.exists():
        if json.loads(path.read_text())!=record:raise ValueError('immutable selector history differs')
    else:write_json_durable(path,record)
    return {'path':str(path.resolve()),'sha256':sha256_file_hex_cached(path,trust_archive_receipt=False)}


def initial(config):
    ref=config.get('task_selection_history')
    if ref is None:return {}
    record=json.loads(_verify_historical_file_ref(ref).read_text())
    if (record.get('schema')!='natlang.task-selector-history/1'
            or record.get('cases_sha256')!=sha256_file_hex_cached(config['cases'],trust_archive_receipt=False)):
        raise ValueError('task selector history differs from the actual training packet')
    state=json.loads(_verify_historical_file_ref(record['source_controller']).read_text())
    history=state.get('task_curriculum_history')
    if not isinstance(history,dict):raise ValueError('task selector history is malformed')
    ids={row['id'] for _,row in iter_jsonl_rows(Path(config['cases']))}
    retained={}
    for task,row in history.items():
        if (isinstance(row,dict) and row.get('task_id')==task and type(row.get('scored')) is int and row['scored']==1
                and 'mean' not in row):
            # Single-scored groups carried no mean before 2026-11-10 (rollout_math.group_advantages): not an
            # observation the selector can use; skipped, not refused (w5 SFT window 0).
            print(json.dumps({'event':'task selector observation without a mean skipped','task_id':task}),flush=True)
            continue
        if (not isinstance(row,dict) or row.get('task_id')!=task
                or not re.fullmatch('[0-9a-f]{64}',str(row.get('observed_policy_id','')))
                or type(row.get('scored')) is not int or row['scored']<=0
                or type(row.get('mean')) not in (int,float) or not math.isfinite(row['mean'])
                or not 0<=row['mean']<=1):
            raise ValueError('task selector observation lacks exact actor and score provenance')
        if task in ids:
            # Round numbers belong to the source controller. Kept as observed_round
            # they would collide with this controller's rounds (or, being larger,
            # stay "newest" forever) in the spread-driven difficulty shift.
            row=dict(row);source_round=row.pop('observed_round',None)
            if source_round is not None:row['source_controller_round']=source_round
            retained[task]={**row,'source_controller_sha256':record['source_controller']['sha256']}
    offset=state.get('offset',0)
    if type(offset) is not int or offset<0:raise ValueError('invalid historical selector offset')
    return {'task_curriculum_history':retained,'offset':offset%len(ids) if ids else 0,
            'task_selection_history_binding':dict(ref),
            'task_selection_history_scope':'historical selection observations only; no policy/replay/training credit',
            'task_selection_history_excluded_ids':sorted(set(history)-ids)}
