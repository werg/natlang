"""Pinned new agent snapshots through the existing trajectory representation."""
from __future__ import annotations

import json
import gzip
from pathlib import Path

from .common import Reject
from .records import group_key

ACQUIRED_FILES = None  # Optional exact receipt-file shard, never an implicit row limit.


def supported_acquired_file(name):
    """One shared pinned-file predicate for conversion and export inventories."""
    return str(name).endswith(('.parquet','.jsonl','.jsonl.gz','.jsonl.zst','.json','.json.gz'))


def iter_acquired_file_rows(path):
    """Yield original integer rows; blank JSONL lines keep their positions."""
    path=Path(path)
    if path.suffix=='.parquet':
        from . import agents
        yield from agents._parquet_groups(path)
    elif str(path).endswith(('.json','.json.gz')):
        opener=gzip.open if str(path).endswith('.gz') else open
        with opener(path,'rt',encoding='utf-8') as stream:
            value=json.load(stream)
        if isinstance(value,list):yield from enumerate(value)
        else:yield 0,value
    elif supported_acquired_file(path):
        from natlang_neuralese.common.jsonio import open_jsonl_text
        opener=gzip.open if str(path).endswith('.gz') else open_jsonl_text
        options={'mode':'rt','encoding':'utf-8'} if str(path).endswith('.gz') else {}
        with opener(path,**options) as stream:
            for number,line in enumerate(stream):
                if line.strip():yield number,json.loads(line)
    else:
        raise Reject('unsupported acquired source file: '+path.name)


def acquired_rows(ref, limit=None):
    from . import agents
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    root = agents.EXPANSION_ROOT / ref
    receipt = json.loads((root / 'acquisition.json').read_text())
    if receipt.get('status') != 'acquired' or not receipt.get('revision'):
        raise Reject('source snapshot is not completely acquired and pinned: '+ref)
    known = {entry['path'] for entry in receipt['files'] if supported_acquired_file(entry['path'])}
    if ACQUIRED_FILES is not None and not ACQUIRED_FILES <= known:
        raise Reject('requested file is not a bound source in this receipt')
    index = 0
    for entry in receipt['files']:
        relative = Path(entry['path'])
        if not supported_acquired_file(relative):
            continue
        if ACQUIRED_FILES is not None and str(relative) not in ACQUIRED_FILES:
            continue
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != entry['bytes']:
            raise Reject('acquired source file differs: '+str(relative))
        if sha256_file_hex_cached(path) != entry['sha256']:
            raise Reject('acquired source digest differs: '+str(relative))
        rows=iter_acquired_file_rows(path)
        for row_index, row in rows:
            if limit is not None and index >= limit:
                return
            index += 1
            yield receipt, f'{relative}:{row_index}', row


def _reported_session(ref, receipt, rid, row, messages, *, task_id, session_id, teacher, tools=None):
    from . import agents
    if not isinstance(messages, list) or not messages:
        raise Reject('missing original session messages')
    if not task_id or not session_id:
        raise Reject('missing original task/session identity')
    meta = row.get('metadata') or {}
    if isinstance(meta, str):
        meta = json.loads(meta)
    trajectory = agents.expansion_trajectory(ref, receipt, rid, {
        'messages': messages, 'tools': tools,
        'metadata': {**meta, 'trajectory_id': str(task_id), 'session_id': str(session_id),
                     'source_row_id': rid, 'original_model': teacher},
        '_expansion_split': 'test' if row.get('split') in ('test', 'validation', 'eval') else 'train',
    })
    trajectory.lineage['teacher'] = agents._teacher(teacher)
    instance = row.get('instance_id')
    if instance:
        trajectory.split_groups.append(group_key('swe-instance', instance))
        if instance in agents._bench_ids():
            trajectory.split = 'test'
    if row.get('repo'):
        trajectory.split_groups.append(group_key('repo', row['repo']))
    reported = row.get('resolved')
    trajectory.outcome = {'label': 'failed' if reported is False or reported == 0 else 'teacher',
        'checked': None, 'details': {'reported_resolved': reported,
                                   'origin': 'upstream_metadata', 'executed_locally': False}}
    return trajectory


def _coding(ref, limit=None):
    from . import agents
    for receipt, rid, row in acquired_rows(ref, limit):
        try:
            meta = row.get('metadata') or {}
            teacher_meta = meta.get('teacher_model') if isinstance(meta, dict) else None
            teacher = teacher_meta.get('name') if isinstance(teacher_meta, dict) else teacher_meta
            tools = [agents._prune(json.loads(value) if isinstance(value, str) else value)
                     for value in row.get('tools') or []]
            trajectory = _reported_session(ref, receipt, rid, row,
                row.get('messages') if ref == 'D22' else row.get('trajectory'),
                task_id=row.get('instance_id'), session_id=row.get('trajectory_id'),
                teacher=teacher, tools=tools)
            yield rid, trajectory
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))


def open_swe_acquired(limit=None):
    yield from _coding('D22', limit)


def swe_rebench_acquired(limit=None):
    yield from _coding('D25', limit)


def nemotron_terminal_acquired(limit=None):
    for receipt, rid, row in acquired_rows('D35', limit):
        try:
            trajectory = _reported_session('D35', receipt, rid, row, row.get('conversations'),
                task_id=row.get('task'), session_id=rid, teacher=row.get('model'))
            trajectory.lineage['notes']['original_terminal'] = {key: row.get(key) for key in
                ('agent', 'model', 'model_provider', 'date', 'task', 'episode', 'run_id',
                 'trial_name', 'enable_thinking', 'source')}
            yield rid, trajectory
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))


def cooperdata_acquired(limit=None):
    for receipt, rid, row in acquired_rows('D17', limit):
        trajectories = row.get('trajectories') or []
        if not trajectories:
            yield rid, Reject('missing original agent trajectories; communication log retained in pinned raw source')
            continue
        for index, agent in enumerate(trajectories):
            child = f'{rid}:agent:{index}'
            try:
                raw = agent.get('messages_json')
                messages = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(messages, dict):
                    messages = messages.get('messages') or messages.get('trajectory')
                trajectory = _reported_session('D17', receipt, child, row, messages,
                    task_id=row.get('pair_id'), session_id=f'{rid}:{agent.get("agent") or index}',
                    teacher=row.get('model'))
                trajectory.lineage['notes']['original_cooperation'] = {key: row.get(key) for key in
                    ('source', 'source_type', 'pair_id', 'repo', 'task_id', 'features', 'fa_fb',
                     'setting', 'bucket', 'split', 'bucket_reason', 'coord_channel', 'both_passed',
                     'score', 'session_success', 'merge_status', 'conversation', 'agents')}
                trajectory.split_groups.append(group_key('cooper-task', f'{row.get("repo")}:{row.get("task_id")}'))
                if row.get('both_passed') is False:
                    trajectory.outcome['label'] = 'failed'
                trajectory.outcome['details']['reported_both_passed'] = row.get('both_passed')
                yield child, trajectory
            except (Reject, ValueError, TypeError, KeyError) as error:
                yield child, Reject(str(error))


def general_agent_acquired(ref, limit=None):
    """Structured upstream sessions; no tool observations or rewards invented."""
    from . import agents
    for receipt, rid, row in acquired_rows(ref, limit):
        try:
            if not isinstance(row, dict):
                raise Reject('source data row is not an object')
            path = rid.rsplit(':', 1)[0]
            if ref == 'D36' and not path.startswith('trajectories/'):
                raise Reject('duplicate tokenized representation; original trajectories are the consumer')
            if row.get('images'):
                raise Reject('image-bearing session requires supported multimodal input; raw retained')
            raw = row.get('messages') or row.get('conversations')
            if isinstance(raw, str):
                raw = json.loads(raw)
            tools = row.get('tools') or row.get('available_tools')
            if isinstance(tools, str):
                tools = json.loads(tools)
            metadata = row.get('metadata') or {}
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            bound_row = dict(row)
            for key in ('instance_id', 'repo', 'resolved', 'reward', 'status'):
                if bound_row.get(key) is None and key in metadata:
                    bound_row[key] = metadata[key]
            task = bound_row.get('instance_id') or row.get('question') or row.get('uuid') or row.get('trajectory_id') or rid
            teacher = row.get('model') or row.get('teacher_model') or metadata.get('model')
            trajectory = _reported_session(ref, receipt, rid, bound_row, raw,
                task_id=task, session_id=row.get('uuid') or row.get('trajectory_id') or rid,
                teacher=teacher, tools=tools)
            trajectory.lineage['notes']['original_source_metadata'] = {key: row.get(key) for key in
                ('subset_name', 'source', 'domain', 'exit_status', 'trajectory_format',
                 'reward', 'status', 'used_in', 'question_quality_assessment', 'response_quality_assessment')}
            if bound_row.get('reward') is not None and float(bound_row['reward']) <= 0:
                trajectory.outcome['label'] = 'failed'
            if row.get('exit_status') in ('incomplete', 'error', 'failed') or bound_row.get('status') in ('error', 'failed'):
                trajectory.outcome['label'] = 'failed'
            trajectory.outcome['details'].update(reported_reward=bound_row.get('reward'),
                reported_exit_status=row.get('exit_status'), reported_status=bound_row.get('status'))
            yield rid, trajectory
        except (Reject, ValueError, TypeError, KeyError) as error:
            yield rid, Reject(str(error))


def toucan_acquired(limit=None):
    yield from general_agent_acquired('D26', limit)


def toolmind_acquired(limit=None):
    yield from general_agent_acquired('D27', limit)


def nemotron_agentic_acquired(limit=None):
    yield from general_agent_acquired('D28', limit)


def ultradata_acquired(limit=None):
    yield from general_agent_acquired('D38', limit)


def swe_zero_acquired(limit=None):
    yield from general_agent_acquired('D23', limit)


def coderforge_acquired(limit=None):
    yield from general_agent_acquired('D36', limit)


def orchard_acquired(limit=None):
    yield from general_agent_acquired('D37', limit)
