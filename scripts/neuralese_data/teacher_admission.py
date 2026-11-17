#!/usr/bin/env python3
"""Verify supplemental review evidence for canonical evaluation-fixture teacher pools.

ProgramRecord, native teacher trajectories and natlang.rollout/1 remain the only
execution/training formats. This module validates admission sidecars; it never
renders tokens, builds chains or substitutes a second teacher execution path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex as file_sha, sha256_hex
from natlang_neuralese.common.jsonio import canonical_json_str

SCHEMA = 'natlang.teacher-admission/1'
SHA = re.compile(r'^[0-9a-f]{64}$')


canonical = partial(canonical_json_str, allow_nan=False)
digest = partial(canonical_json_sha256_hex, allow_nan=False)


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def bound(binding, root):
    if not isinstance(binding, dict) or not isinstance(binding.get('path'), str) or not binding['path']:
        raise ValueError('missing admission file binding')
    path = Path(binding['path'])
    if not path.is_absolute():
        path = Path(root) / path
    if not SHA.fullmatch(binding.get('sha256', '')) or file_sha(path) != binding['sha256']:
        raise ValueError(f'admission file changed: {path}')
    return path.resolve()


def fixture_task(task):
    return task.get('kind') == 'evaluation_fixture' or 'evaluation_fixture' in (task.get('semantics') or {})


def source_input(task):
    semantics = task['semantics']
    fixture = semantics.get('evaluation_fixture') or {}
    return {'inputs': semantics['inputs'], 'fixture_input': (fixture.get('case') or {}).get('input')}


def protected_inputs(programs):
    """Return protected ids/groups/inputs over the complete canonical source inventory."""
    ids, groups, inputs, protected = {}, {}, {}, set()
    for task in programs:
        if task.get('version') != 'natlang.program/2' or task.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('admission inventory needs canonical ProgramRecords with explicit splits')
        if not task.get('id') or task['id'] in ids:
            raise ValueError('admission inventory has missing or duplicate program ids')
        ids[task['id']] = task
        for key in ('source_ids', 'source_groups', 'source_revisions'):
            if not isinstance(task.get(key), list) or not task[key] or any(not isinstance(x, str) or not x for x in task[key]):
                raise ValueError(f'program inventory needs {key}')
        namespace = task.get('source_namespace')
        if not isinstance(namespace, str) or not namespace:
            raise ValueError('fixture source inventory needs a stable source_namespace')
        key = (namespace, digest(source_input(task)))
        inputs.setdefault(key, set()).add(task['split'])
        for group in task['source_groups']:
            groups.setdefault(group, set()).add(task['split'])
    for ident, task in ids.items():
        key = (task['source_namespace'], digest(source_input(task)))
        if task['split'] != 'train' or len(inputs[key]) > 1 or any(len(groups[g]) > 1 for g in task['source_groups']):
            protected.add(ident)
    return ids, protected


def semantic_review(review, *, case, evaluation):
    if not isinstance(review, dict) or review.get('verdict') != 'accepted' or not isinstance(review.get('reason'), str) or not review['reason'].strip():
        raise ValueError('semantic review did not accept the trajectory with evidence')
    reviewer = review.get('reviewer') or {}
    if not reviewer.get('provider') or not reviewer.get('model_id') or not reviewer.get('identity'):
        raise ValueError('semantic review needs actual reviewer identity')
    request, reply = review.get('request'), review.get('reply')
    if not isinstance(request, dict) or not request.get('messages') or not isinstance(reply, dict):
        raise ValueError('semantic review needs its exact request and reply')
    messages = request['messages']
    if (not isinstance(messages, list) or len(messages) != 2 or
            [message.get('role') for message in messages] != ['system', 'user'] or
            not isinstance(messages[0].get('content'), str) or not messages[0]['content'].strip() or
            not isinstance(messages[1].get('content'), str)):
        raise ValueError('semantic review must be an isolated system and user conversation')
    payload = json.loads(messages[1]['content'])
    expected = {key: case[key] for key in ('input', 'expected') if key in case}
    expected.update({key: evaluation[key] for key in ('outcome', 'observation', 'observations') if key in evaluation})
    if canonical(payload) != canonical(expected):
        raise ValueError('semantic review payload differs from the actual source case or fixture result')
    raw = review.get('request_canonical')
    if raw is not None:
        if not isinstance(raw, str) or canonical(json.loads(raw)) != canonical(request):
            raise ValueError('semantic request canonical bytes differ from request')
        request_sha = sha256_hex(raw.encode('utf-8'))
    else:
        request_sha = digest(request)
    if review.get('request_sha256') != request_sha:
        raise ValueError('semantic review request digest changed')
    verdicts = [args for name, args in reply.get('calls') or [] if name == 'verify_result']
    if len(verdicts) != 1 or verdicts[0] != {'verdict': review['verdict'], 'reason': review['reason']}:
        raise ValueError('semantic verdict does not match the actual reviewer reply')


def invocation_metadata(result):
    out = {}
    for item in (result.get('outcome') or {}).get('invocation_ledger') or []:
        ident = item.get('invocation_id')
        if not isinstance(ident, str) or not ident:
            continue
        role = 'unknown' if 'parent_invocation_id' not in item else 'root' if item['parent_invocation_id'] is None else 'child'
        out[ident] = {'function': item.get('name') or 'unknown', 'definition_id': item.get('definition_id'),
                      'parent_invocation': item.get('parent_invocation_id'), 'invocation_role': role}
    return out


def check_record(record, pool):
    """Return checked samples keyed by (program id, sample), or refuse the sidecar."""
    from natlang_neuralese.common.task_roots import student_root_row, authored_root_row
    from natlang_neuralese.rl.task_rollouts import rollout_turns
    cases_path = Path(record['rollouts']).parent / 'cases.jsonl'
    cases = {task['id']: task for task in rows(cases_path)}
    if not any(fixture_task(task) for task in cases.values()):
        return {}
    sidecar_path = bound(record.get('admission'), pool)
    receipt = json.loads(sidecar_path.read_text())
    if receipt.get('schema') != SCHEMA:
        raise ValueError(f'expected {SCHEMA}')
    if bound(receipt.get('cases'), sidecar_path.parent) != cases_path.resolve():
        raise ValueError('admission cases binding does not match teacher pool')
    rollout_path = bound(receipt.get('rollouts'), sidecar_path.parent)
    if rollout_path != Path(record['rollouts']).resolve():
        raise ValueError('admission rollouts binding does not match teacher manifest')
    if receipt.get('teacher') != record.get('teacher'):
        raise ValueError('admission teacher differs from actual teacher manifest')
    inventory_path = bound(receipt.get('inventory'), sidecar_path.parent)
    inventory, protected = protected_inputs(rows(inventory_path))
    rollout_rows = {(row['task_id'], row['sample']): row for row in rows(rollout_path)}
    checked = {}
    transform = student_root_row if record.get('root_mode', 'student') == 'student' else authored_root_row
    for sample in receipt.get('samples') or []:
        key = sample.get('task_id'), sample.get('sample')
        if key in checked or key not in rollout_rows or key[0] not in cases:
            raise ValueError('admission sample is duplicate or absent from canonical rollouts/cases')
        task, rollout = cases[key[0]], rollout_rows[key]
        if key[0] not in inventory or canonical(inventory[key[0]]) != canonical(task):
            raise ValueError('task differs from complete source inventory')
        if key[0] in protected:
            continue
        if rollout.get('outcome') != 'task_success' or (rollout.get('reward') or {}).get('quality') != 1:
            continue
        if not rollout.get('turns'):
            continue  # a mechanism-only success provides no executor targets
        result_path = bound(sample.get('result'), sidecar_path.parent)
        results = rows(result_path)
        if len(results) != 1:
            raise ValueError('admission result binding must contain exactly one native teacher trajectory')
        result = results[0]
        if result.get('version') != 'natlang.teacher_trajectory.native/1':
            raise ValueError('admission requires the canonical native teacher trajectory')
        actual_task = (result.get('task') or {}).get('program_ir')
        if canonical(actual_task) != canonical(transform(task)):
            raise ValueError('native trajectory executed another ProgramRecord')
        provenance = result.get('provenance') or {}
        if not SHA.fullmatch(sample.get('program_ir_sha256', '')) or provenance.get('program_ir_sha256') != sample['program_ir_sha256']:
            raise ValueError('native source program digest differs from admission')
        if provenance.get('model') != record['teacher']['model_id']:
            raise ValueError('native trajectory teacher differs from manifest')
        if provenance.get('provider') and provenance['provider'] != record['teacher'].get('provider'):
            raise ValueError('native trajectory provider differs from manifest')
        def turn_source(turn):
            return {field: turn.get(field) for field in ('messages', 'target', 'invocation_id')} | {'tools': turn.get('tools') or []}
        if canonical([turn_source(t) for t in rollout_turns(result)]) != canonical([turn_source(t) for t in rollout.get('turns') or []]):
            raise ValueError('rollout targets differ from the native teacher trajectory')
        code = sample.get('code') or {}
        if (not isinstance(code.get('git_commit'), str) or not re.fullmatch(r'[0-9a-f]{40}', code['git_commit']) or
                code.get('dirty') is not False or code != rollout.get('code')):
            raise ValueError('admission needs clean collector code provenance')
        snapshot = task.get('source_snapshot')
        if snapshot is not None:
            if not isinstance(snapshot, dict) or snapshot.get('dirty') is not False:
                raise ValueError('fixture source snapshot must attest clean source')
            if snapshot.get('natlang_commit') is not None and snapshot['natlang_commit'] != code['git_commit']:
                raise ValueError('fixture source snapshot natlang commit differs from actual collection code')
        outcome = result.get('outcome') or {}
        if outcome.get('accepted') is not True or outcome.get('quality_pending'):
            continue
        checks = outcome.get('checks')
        if not isinstance(checks, dict) or not checks or any(value is not True for value in checks.values()):
            continue
        evidence = sample.get('outcome') or {}
        evaluation = outcome.get('evaluation') or {}
        if (evidence.get('quality') != evaluation.get('quality') or
                evidence.get('gates') != evaluation.get('gates') or evidence.get('evaluation') != evaluation):
            raise ValueError('admission suite outcome differs from actual fixture evidence')
        gates = evidence.get('gates')
        if evidence.get('quality') != 1 or not isinstance(gates, dict) or not gates or any(value is not True for value in gates.values()):
            continue
        if type(evidence.get('tool_error_count')) is not int or evidence['tool_error_count'] != 0 or outcome.get('scope_failures'):
            continue
        ledger = outcome.get('action_ledger') or []
        if any(event.get('success') is False or event.get('outcome') in ('failed', 'error', 'rejected', 'refused') for event in ledger):
            continue
        replay = sample.get('replay') or {}
        hashes = [(turn.get('replay') or {}).get('request_sha256') for turn in rollout.get('turns') or []]
        if not hashes or any(not isinstance(h, str) or not SHA.fullmatch(h) for h in hashes):
            raise ValueError('canonical rollout is missing actual request digest evidence')
        if replay.get('verified') is not True or replay.get('request_hashes') != hashes:
            raise ValueError('admission replay evidence does not bind canonical requests')
        replay_path = bound(replay.get('record'), sidecar_path.parent)
        replay_record = json.loads(replay_path.read_text())
        if (replay_record.get('exact') is not True or replay_record.get('outcome') != 'task_success' or
                replay_record.get('task_id') != key[0] or replay_record.get('recorded_turns') != len(hashes)):
            raise ValueError('admission requires the shared collector exact replay verdict')
        semantic_review(sample.get('semantic'), case=task['semantics']['evaluation_fixture']['case'],
                        evaluation=evaluation)
        checked[key] = {'path': str(sidecar_path), 'sha256': file_sha(sidecar_path),
                        'sample': sample, 'invocations': invocation_metadata(result)}
    return checked


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_sha(path)}


def replay_pool(args):
    """Stage canonical episodes through sft_sources' ordinary collector replay."""
    from types import SimpleNamespace
    import sft_sources as sft
    from natlang_neuralese.common.code_provenance import code_provenance
    pool = Path(args.pool).resolve()
    manifest_path = pool / 'manifest.jsonl'
    work = Path(args.work).resolve() if args.work else pool / 'admission-replay'
    work.mkdir(parents=True, exist_ok=True)
    manifest_records = rows(manifest_path)
    inputs = {'manifest_source_sha256': digest([{key: value for key, value in record.items() if key != 'admission'}
                                               for record in manifest_records]),
              'code': code_provenance(args.runtime_repo, fresh=True)}
    receipt = work / 'source.json'
    if receipt.exists() and json.loads(receipt.read_text()) != inputs:
        raise ValueError('replay source/runtime changed; choose a fresh --work')
    receipt.write_text(json.dumps(inputs, indent=2) + '\n')
    episodes, drafts, episode_index = [], [], {}
    label = 'teacher:' + '/'.join(pool.parts[-2:])
    for manifest_index, record in enumerate(manifest_records):
        rollout_path = Path(record['rollouts'])
        directory = rollout_path.parent
        cases = {task['id']: task for task in rows(directory / 'cases.jsonl')}
        queue = rows(directory / 'queue.jsonl')
        draft = {'manifest_index': manifest_index, 'rollouts': str(rollout_path), 'receipt': {
            'schema': SCHEMA, 'cases': binding(directory / 'cases.jsonl'), 'rollouts': binding(rollout_path),
            'teacher': record['teacher'], 'samples': []}}
        for rollout in rows(rollout_path):
            if rollout.get('outcome') != 'task_success' or (rollout.get('reward') or {}).get('quality') != 1:
                continue
            if not rollout.get('turns'):
                continue
            task = cases[rollout['task_id']]
            if not fixture_task(task):
                continue
            entries = [entry for entry in queue if entry['sample'] == rollout['sample'] and
                       (entry.get('task_id') or task['id']) == task['id']]
            if len(entries) != 1:
                raise ValueError('native result needs one exact queue output binding')
            result_path = Path(entries[0]['output'])
            native_rows = rows(result_path)
            if len(native_rows) != 1:
                raise ValueError('native result file must contain exactly one canonical trajectory')
            native = native_rows[0]
            evaluation = (native.get('outcome') or {}).get('evaluation') or {}
            outcome = native.get('outcome') or {}
            errors = len(outcome.get('scope_failures') or []) + sum(
                event.get('success') is False or event.get('outcome') in ('failed', 'error', 'rejected', 'refused')
                for event in outcome.get('action_ledger') or [])
            key = f'{label}:{directory.name}:{rollout["sample"]}'
            episodes.append({'key': key, 'source': 'teacher', 'run': label, 'task_id': task['id'],
                             'family': task.get('family'), 'sample': rollout['sample'], 'seed': rollout.get('seed'),
                             'root_mode': record.get('root_mode', 'student'), 'task': task,
                             'turns': [sft.semantic_turn(turn, False) for turn in rollout['turns']]})
            sample = {'task_id': task['id'], 'sample': rollout['sample'],
                      'program_ir_sha256': native['provenance']['program_ir_sha256'], 'result': binding(result_path),
                      'code': rollout.get('code'), 'outcome': {'quality': evaluation.get('quality'),
                          'gates': evaluation.get('gates'), 'evaluation': evaluation, 'tool_error_count': errors}}
            draft['receipt']['samples'].append(sample)
            episode_index[key] = sample, rollout
        drafts.append(draft)
    replay_args = SimpleNamespace(work=str(work), runtime_repo=args.runtime_repo, workers=args.workers,
        limit=0, case_seconds=args.case_seconds, max_model_requests=args.max_model_requests)
    sft.replay_episodes(replay_args, iter(episodes))
    for key, (sample, rollout) in episode_index.items():
        path = work / 'replay' / sft.safe_name(key) / 'replay.json'
        record = json.loads(path.read_text())
        sample['replay'] = {'verified': record.get('exact') is True and record.get('outcome') == 'task_success',
                            'request_hashes': [(turn.get('replay') or {}).get('request_sha256')
                                               for turn in rollout['turns']], 'record': binding(path)}
    result = {'schema': 'natlang.teacher-admission-drafts/1', 'records': drafts}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'drafts': str(Path(args.out).resolve()), 'samples': len(episodes)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', nargs='?', choices=('check', 'replay'), default='check')
    parser.add_argument('--pool', required=True)
    parser.add_argument('--out')
    parser.add_argument('--work')
    parser.add_argument('--runtime-repo', default=str(REPO))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--case-seconds', type=int, default=3600)
    parser.add_argument('--max-model-requests', type=int, default=None)
    args = parser.parse_args()
    if args.command == 'replay':
        if not args.out:
            parser.error('replay requires --out FILE')
        replay_pool(args)
        return
    total = 0
    for record in rows(Path(args.pool) / 'manifest.jsonl'):
        total += len(check_record(record, args.pool))
    print(json.dumps({'schema': SCHEMA, 'pool': str(Path(args.pool).resolve()), 'admitted_samples': total}))


if __name__ == '__main__':
    main()
