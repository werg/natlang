"""Correct derived terminal rewards without changing or recollecting a trajectory."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from ..common.jsonio import read_jsonl
from ..eval import task_packet
from .rollout_math import group_advantages
from .sealed_collection import resume_sealed_collection

SCHEMA = 'natlang.accepted-stop-reward-reclassification/1'


def _read(reference):
    from ..train.native_readiness import _bound
    path = _bound(reference)
    if path.stat().st_size != reference.get('bytes'):
        raise ValueError('reward correction input size changed')
    return path


def verify_unearned_preparation(reference, request):
    """Authenticate an interrupted admission that produced no recoverable update.

    This does not manufacture a zero-step training report or assert that an
    optimizer ever ran. The exact current full parent is retained and all old
    preparatory artifacts remain available as evidence.
    """
    evidence = json.loads(_read(reference).read_text())
    out = Path(request['out']).resolve(strict=True)
    learner = out / 'learner'
    if (evidence.get('schema') != 'natlang.unearned-rl-preparation/1'
            or evidence.get('out') != str(out)
            or evidence.get('parent', {}).get('sha256') != request['policy_id']):
        raise ValueError('unearned preparation names a different collection or parent')
    _read(evidence['parent'])
    if (learner / 'checkpoint.pt').exists() or (learner / 'report.json').exists():
        raise ValueError('unearned preparation cannot discard a recoverable learner update')
    capacity_path = _read(evidence['capacity_request'])
    admission_path = _read(evidence['admission'])
    if capacity_path != learner / 'rl-policy-capacity-request.json' or admission_path != learner / 'rl-data-receipt.json':
        raise ValueError('unearned preparation artifacts escaped their actual learner')
    capacity = json.loads(capacity_path.read_text())
    admission = json.loads(admission_path.read_text())
    source = capacity.get('source') or {}
    if (capacity.get('schema') != 'natlang.native-policy-capacity-request/1'
            or admission.get('schema') != 'natlang.native-rl-admission/1'
            or admission.get('policy', {}).get('checkpoint_sha256') != request['policy_id']
            or admission.get('rollouts') != {str(out / 'rollouts.jsonl'): sha256_file_hex(out / 'rollouts.jsonl')}
            or capacity.get('objective', {}).get('sha256') != admission.get('objective_sha256')
            or source.get('sha256') != admission.get('native_graphs_sha256')
            or Path(source.get('path', '')).resolve() != learner / 'rl-native-graphs.jsonl'
            or sha256_file_hex(learner / 'rl-native-graphs.jsonl') != source.get('sha256')):
        raise ValueError('unearned preparation differs from the actual original admission')
    return evidence


def projected_rows(out, original):
    """Reassess exact raw results; preserve captures and historical cost policy."""
    entries = list(read_jsonl(out / 'queue.jsonl'))
    finishes = {row['key']: row for row in read_jsonl(out / 'journal.jsonl') if row.get('event') == 'finish'}
    old = {(row['group'], row['sample']): row for row in original}
    if len(old) != len(original) or set(old) != {(e['group'], e['sample']) for e in entries}:
        raise ValueError('reward correction sample identities differ')
    result, changes = [], []
    for entry in entries:
        key = (entry['group'], entry['sample'])
        row = old[key]
        raw = list(read_jsonl(Path(entry['output']))) if Path(entry['output']).exists() else []
        if len(raw) > 1:
            raise ValueError('single-sample correction has multiple raw results')
        outcome = task_packet.classify_case(raw[0] if raw else None, finishes.get(entry['key']))
        amended = copy.deepcopy(row)
        if outcome != row['outcome']:
            actual = (raw[0].get('outcome') or {}) if raw else {}
            accepted_stop = (actual.get('status') in {'quiesced', 'failed'}
                             and task_packet._accepted_terminal(actual, finishes.get(entry['key'])))
            accepted_qa = bool(raw and task_packet.accepted_babi_article_answer(raw[0], finishes.get(entry['key'])))
            if outcome != 'task_success' or not (accepted_stop or accepted_qa):
                raise ValueError('classification changed outside the verified accepted-stop repair')
            from .task_rollouts import neuralese_exchanges
            exchanges = neuralese_exchanges(row)
            binding = json.loads((out / 'collection-binding.json').read_text())
            bonus = float(binding.get('neuralese_use_bonus', .1))
            if not 0 <= bonus <= 1:
                raise ValueError('invalid recorded Neuralese bonus')
            amended['outcome'] = outcome
            amended['unscored'] = None
            amended['reward'] = {**(row.get('reward') or {}), 'quality': 1.}
            if exchanges and row.get('jolt_role', 'student') == 'student' and bonus:
                amended['reward']['neuralese_bonus'] = bonus
            changes.append({'group': key[0], 'sample': key[1], 'task_id': row['task_id'],
                            'before': {k: row.get(k) for k in ('outcome', 'reward', 'unscored')},
                            'after': {k: amended.get(k) for k in ('outcome', 'reward', 'unscored')},
                            'raw_result_path': entry['output'], 'finish_key': entry['key']})
        result.append(amended)
    return result, changes


def projected_report(original, rows, path):
    """Task counters change; original collection and actor receipts remain intact."""
    from .task_rollouts import reward_of, neuralese_exchanges, neuralese_metrics
    report = copy.deepcopy(original)
    report['emitted_rollouts'] = str(path)
    report['outcome_counts'] = {key: sum(row['outcome'] == key for row in rows)
                                for key in sorted({row['outcome'] for row in rows})}
    report['successes'] = sum(row['outcome'] == 'task_success' for row in rows)
    report['scored'] = sum(reward_of(row['outcome']) is not None for row in rows)
    groups = []
    metrics = []
    for row in rows:
        metrics.append({'reward': reward_of(row['outcome']), 'jolt_role': row.get('jolt_role', 'student'),
                        'neuralese_exchanges': neuralese_exchanges(row),
                        'neuralese_bonus': (row.get('reward') or {}).get('neuralese_bonus', 0.)})
    for old_group in original['groups']:
        members = sorted((r for r in rows if r['task_id'] == old_group['task_id']
                          and r.get('jolt_role', 'student') == 'student'), key=lambda r: r['sample'])
        _, stats = group_advantages([reward_of(row['outcome']) for row in members])
        mean = stats.get('mean', 0.)
        costs = [(row.get('reward') or {}).get('execution_efficiency', {}).get('cost')
                 for row in members if reward_of(row['outcome']) is not None]
        valid_costs = all(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0 for cost in costs)
        efficiency_spread = bool(members) and all(reward_of(row['outcome']) == 1 for row in members) \
            and valid_costs and len(set(costs)) > 1
        groups.append({'task_id': old_group['task_id'], **stats,
                       'efficiency_spread': efficiency_spread,
                       'outcomes': [r['outcome'] for r in members],
                       'route': 'grpo' if stats['spread'] or efficiency_spread else 'retire' if mean >= 1. else 'rejection-sampling'})
    report['groups'] = groups
    report['groups_with_spread'] = sum(g['spread'] for g in groups)
    report['groups_with_quality_spread'] = sum(g['spread'] for g in groups)
    report['groups_with_efficiency_spread'] = sum(g['efficiency_spread'] for g in groups)
    report['neuralese'] = neuralese_metrics(metrics, original.get('neuralese', {}).get('bonus', .1))
    return report


def resume_reclassified_collection(reference, request, *, validate_actors):
    proof_path = _read(reference)
    proof = json.loads(proof_path.read_text())
    out = Path(request['out']).resolve(strict=True)
    if proof.get('schema') != SCHEMA or proof.get('out') != str(out):
        raise ValueError('reward correction names a different collection')
    original = resume_sealed_collection(proof['original_collection'], request, validate_actors=validate_actors)
    if proof.get('parent_sha256') != request['policy_id']:
        raise ValueError('reward correction is not from the actual current policy')
    classifier = _read(proof['classifier'])
    expected_id = canonical_json_sha256_hex([proof['original_collection']['sha256'], proof['classifier']['sha256']])[:16]
    if proof.get('id') != expected_id:
        raise ValueError('reward correction identity is not bound to its source and classifier')
    if sha256_file_hex(Path(task_packet.__file__)) != proof['classifier']['sha256']:
        raise ValueError('reward correction requires its exact reviewed classifier')
    # The abandoned learner may have changed metadata, but earned no SGD.
    if proof.get('unearned_preparation') is not None:
        if 'no_update_report' in proof or 'no_update_receipt' in proof:
            raise ValueError('reward repair mixes distinct interruption evidence')
        verify_unearned_preparation(proof['unearned_preparation'], request)
    else:
        report = json.loads(_read(proof['no_update_report']).read_text())
        receipt = json.loads(_read(proof['no_update_receipt']).read_text())
        if (report.get('step') != 0 or report.get('completed') is not False
                or report.get('continuation_boundary', {}).get('new_optimizer_updates') != 0
                or report.get('continuation', {}).get('checkpoint_sha256') != request['policy_id']
                or receipt.get('operation_parent', {}).get('sha256') != request['policy_id']
                or any(receipt.get('exposure', {}).get(k) != 0 for k in ('consumed_updates', 'graph_visits'))
                or receipt.get('training_receipt', {}).get('sha256') != proof['no_update_report']['sha256']):
            raise ValueError('reward repair cannot discard earned or uncertain learner updates')
    corrected = _read(proof['corrected_rollouts'])
    score = _read(proof['corrected_score'])
    directory = out / ('reward-reclassification-' + proof['id'])
    if corrected != directory / 'rollouts.jsonl' or score != directory / 'score.json':
        raise ValueError('reward correction outputs escaped their exact version directory')
    expected, changes = projected_rows(out, list(read_jsonl(out / 'rollouts.jsonl')))
    actual = list(read_jsonl(corrected))
    if (not changes or proof.get('changes') != changes
            or canonical_json_sha256_hex(expected) != canonical_json_sha256_hex(actual)
            or projected_report(original, expected, corrected) != json.loads(score.read_text())):
        raise ValueError('reward correction changed trajectories, policy, counters or unreviewed rewards')
    return {**json.loads(score.read_text()), 'reclassified_rollouts': str(corrected),
            'reward_reclassification': reference,
            'reclassified_learner_out': str(out / ('learner-reclassified-' + proof['id']))}


def verify_reclassified_attempt_retry(reference, request, correction):
    """Retain an incorrectly admitted attempt while restarting from its real parent."""
    evidence = json.loads(_read(reference).read_text())
    out = Path(request['out']).resolve(strict=True)
    proof = json.loads(_read(correction).read_text())
    attempt = out / ('learner-reclassified-' + proof['id'])
    if (evidence.get('schema') != 'natlang.reward-reclassification-attempt-retry/1'
            or evidence.get('reason') != 'original-rollouts-admitted-after-reclassification'
            or evidence.get('out') != str(out) or evidence.get('reclassification') != correction
            or evidence.get('parent', {}).get('sha256') != request['policy_id']
            or proof.get('parent_sha256') != request['policy_id']):
        raise ValueError('retry does not bind the reviewed correction and current parent')
    _read(evidence['parent'])
    checkpoint = _read(evidence['discarded_checkpoint'])
    admission_path = _read(evidence['discarded_admission'])
    if checkpoint != (attempt / 'checkpoint.pt').resolve(strict=True) or admission_path != (attempt / 'rl-data-receipt.json').resolve(strict=True):
        raise ValueError('retry evidence does not name the actual discarded attempt')
    admission = json.loads(admission_path.read_text())
    original = out / 'rollouts.jsonl'
    if (admission.get('schema') != 'natlang.native-rl-admission/1'
            or admission.get('policy', {}).get('checkpoint_sha256') != request['policy_id']
            or admission.get('rollouts') != {str(original): sha256_file_hex(original)}
            or evidence['discarded_checkpoint']['sha256'] == request['policy_id']):
        raise ValueError('retry attempt lacks the actual incorrect admission and distinct checkpoint')
    return str(out / ('learner-reclassified-' + proof['id'] + '-attempt-1'))
