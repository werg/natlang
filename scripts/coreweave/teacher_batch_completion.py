"""Exact terminal accounting for continuous self-hosted teacher collection.

Generation policy: keep collecting after recorded model failures, retain their
true outcomes, and stop for an incomplete or broken collection. Worker exit2
alone cannot distinguish those cases. This never grants training admission.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def sample_rewards(entries, rewards, cases):
    """Join terminal per-sample rewards to the exact queue/case identity.

    ``score.json`` groups contain aggregate per-group outcome lists. Those
    lists are not indexed by the queue's sample number, so they must never be
    used to label individual result files. ``rewards.jsonl`` is the
    per-rollout authority and is joined on (group, sample), with the task ID
    checked against the source-case order and the queue key.
    """
    if not isinstance(cases, list) or not cases:
        raise ValueError('teacher source cases must be a non-empty ordered list')
    case_ids = []
    for case in cases:
        program = case.get('program_ir', case) if isinstance(case, dict) else None
        task_id = program.get('id') if isinstance(program, dict) else None
        if not isinstance(task_id, str) or not task_id:
            raise ValueError('teacher source case has no exact task ID')
        case_ids.append(task_id)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('teacher source cases contain duplicate task IDs')

    indexed = {}
    for row in rewards:
        if not isinstance(row, dict):
            raise ValueError('teacher reward row must be an object')
        group, sample, task_id = row.get('group'), row.get('sample'), row.get('task_id')
        if type(group) is not int or type(sample) is not int or not isinstance(task_id, str):
            raise ValueError('teacher reward row lacks typed group/sample/task identity')
        key = (group, sample)
        if key in indexed:
            raise ValueError('duplicate teacher reward sample identity')
        indexed[key] = row

    queue_keys = set()
    result = []
    for entry in entries:
        group, sample = entry.get('group'), entry.get('sample')
        index, count = entry.get('index'), entry.get('count')
        if (type(group) is not int or type(sample) is not int or
                not (0 <= group < len(case_ids)) or type(index) is not int or
                index != group or type(count) is not int or count != 1):
            raise ValueError('teacher queue entry has invalid index/group/sample/count')
        identity = (group, sample)
        if identity in queue_keys:
            raise ValueError('duplicate teacher queue sample identity')
        queue_keys.add(identity)
        row = indexed.pop(identity, None)
        if row is None or row['task_id'] != case_ids[group]:
            raise ValueError('teacher reward does not bind the queued source case')
        expected_key = f's7-task-{group:03d}-{sample:02d}'
        if entry.get('key') != expected_key:
            raise ValueError('teacher queue key differs from its group/sample identity')
        result.append({**entry, 'outcome': row.get('outcome'), 'reward': row.get('reward'),
                       'task_id': row['task_id'], 'finish_status': row.get('finish_status')})
    if indexed:
        raise ValueError('teacher rewards contain samples absent from the queue')
    return result


def batch_completion(entries, journal, returncodes, score, *, resolve_path=Path):
    """Verify collector accounting; model-budget errors retain exact error bytes.

    Completed-result accounting is the existing collector's verifier output.
    Budget errors additionally require their actual file SHA and typed error,
    exact job identity, and the collector's source-bound output manifest. Old
    errors without per-error source digests are reported as legacy provenance;
    they become neither successful outputs nor on-policy training samples.
    """
    keys = [row['key'] for row in entries]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('teacher batch requires distinct actual queue keys')
    finishes = {row['key']: row for row in journal if row.get('event') == 'finish'}
    repairs, budgets, closed = [], [], []
    score_valid = (score.get('schema') == 'natlang.s7-task-rollout-score/1'
        and score.get('rollouts') == len(entries)
        and sum(score.get('outcome_counts', {}).values()) == len(entries))
    codes_valid = bool(returncodes) and all(type(code) is int and code in (0, 2) for code in returncodes)
    manifests, source_hashes = {}, {}
    for entry in entries:
        key = entry['key']; finish = finishes.get(key) or {}
        accounting = finish.get('output_accounting') or {}
        states = accounting.get('job_states') or []
        valid = (accounting.get('version') == 'natlang.supervisor_output_accounting/2'
            and accounting.get('manifest_valid') is True
            and not accounting.get('manifest_errors')
            and accounting.get('expected_jobs') == len(states) > 0)
        if valid and accounting.get('complete') is True:
            valid = all(row.get('state') in ('result', 'generation_held') for row in states)
            if valid:
                closed.append(key); continue
        if not (valid and accounting.get('complete') is False
                and finish.get('resource_limit_reason') == 'model_request_budget'):
            repairs.append({'key': key, 'reason': 'missing or unresolved collector accounting'})
            continue
        try:
            manifest_path = resolve_path(str(entry['output']) + '.manifest.json')
            if str(manifest_path) not in manifests:
                manifests[str(manifest_path)] = json.loads(manifest_path.read_text())
            manifest = manifests[str(manifest_path)]
            source = resolve_path(entry['source'])
            if str(source) not in source_hashes: source_hashes[str(source)] = _hash(source)
            if (manifest.get('version') != 'natlang.teacher_batch.native/1'
                    or resolve_path(manifest['source']).resolve() != source.resolve()
                    or manifest.get('source_sha256') != source_hashes[str(source)]
                    or manifest.get('range') != {'start': entry['index'], 'count': len(states)}
                    or manifest.get('missing') != [row['index'] for row in states if row['state'] != 'result']):
                raise ValueError('budget manifest differs from actual queue/source')
            errors = []
            for state in states:
                if state.get('state') in ('result', 'generation_held'): continue
                error_binding = state.get('terminal_error') or {}
                path = resolve_path(error_binding['path'])
                if not path.resolve().is_relative_to(resolve_path(entry['jobs']).resolve()):
                    raise ValueError('budget error escapes its original job directory')
                if _hash(path) != error_binding['sha256']:
                    raise ValueError('budget error bytes changed after collection')
                error = json.loads(path.read_text())
                if (state.get('state') != 'failed'
                        or state.get('resource_limit_reason') != 'model_request_budget'
                        or error.get('code') != 'NATLANG_MODEL_REQUEST_BUDGET'
                        or error.get('index') != state['index']
                        or error.get('program_id') != state['program_id']
                        or ('program_ir_sha256' in error and error['program_ir_sha256'] != state['digest'])
                        or ('source_sha256' in error and error['source_sha256'] != source_hashes[str(source)])):
                    raise ValueError('error is not the bound model-request budget outcome')
                errors.append({'path': str(path), 'sha256': error_binding['sha256'],
                    'program_id': state['program_id'], 'index': state['index'],
                    'per_error_source_digests_present': all(name in error for name in ('source_sha256','program_ir_sha256'))})
            if not errors: raise ValueError('budget disposition has no actual budget error')
            budgets.append({'key': key, 'errors': errors, 'training_admission': False})
        except (OSError, ValueError, KeyError, TypeError):
            repairs.append({'key': key, 'reason': 'budget error/source/manifest binding failed'})
    if score.get('outcome_counts', {}).get('budget_exhaustion', 0) < len(budgets):
        repairs.append({'key': None, 'reason': 'score lost recorded budget failures'})
    return {'schema': 'natlang.strong-teacher-batch-completion/1',
        'can_advance': codes_valid and score_valid and not repairs,
        'expected_queue_keys': len(keys), 'exact_completed_queue_keys': len(closed),
        'model_budget_failures': budgets, 'repair_required': repairs,
        'worker_returncodes': returncodes, 'score_accounting_valid': score_valid,
        'training_admission': False,
        'scope': 'collection scheduling only; original failures and source/error identities retained'}
