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


def batch_completion(entries, journal, returncodes, score):
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
            manifest_path = Path(str(entry['output']) + '.manifest.json')
            if str(manifest_path) not in manifests:
                manifests[str(manifest_path)] = json.loads(manifest_path.read_text())
            manifest = manifests[str(manifest_path)]
            source = Path(entry['source'])
            if str(source) not in source_hashes: source_hashes[str(source)] = _hash(source)
            if (manifest.get('version') != 'natlang.teacher_batch.native/1'
                    or Path(manifest['source']).resolve() != source.resolve()
                    or manifest.get('source_sha256') != source_hashes[str(source)]
                    or manifest.get('range') != {'start': entry['index'], 'count': len(states)}
                    or manifest.get('missing') != [row['index'] for row in states if row['state'] != 'result']):
                raise ValueError('budget manifest differs from actual queue/source')
            errors = []
            for state in states:
                if state.get('state') in ('result', 'generation_held'): continue
                error_binding = state.get('terminal_error') or {}
                path = Path(error_binding['path'])
                if not path.resolve().is_relative_to(Path(entry['jobs']).resolve()):
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
