#!/usr/bin/env python3
"""Verify registered preparation artifacts; never infer training admission."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _relative_path(root, value, label):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'{label} must be a repository-relative path without parent traversal')
    return root / path

def _jsonl_and_digest(path):
    rows = []
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path} line {number} is invalid JSON: {error}') from error
            if not isinstance(row, dict):
                raise ValueError(f'{path} line {number} is not an object')
            rows.append(row)
    return rows, digest.hexdigest()

def _scan_corpus_jsonl(path, count_cases):
    row_count = 0
    case_count = 0
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path} line {number} is invalid JSON: {error}') from error
            if not isinstance(row, dict):
                raise ValueError(f'{path} line {number} is not an object')
            row_count += 1
            if count_cases:
                case_count += sum(len(row.get(role, {}).get('cases', [])) for role in ('support', 'query', 'transfer'))
    return row_count, case_count, digest.hexdigest()

def _declared_episode_count(manifest):
    audit = manifest.get('audit')
    audit_object = audit if isinstance(audit, dict) else {}
    return manifest.get('episodes', audit_object.get('episodes'))

def _canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def _observe_screened_plan(root, row):
    report = {'id': row.get('id'), 'declared_state': row.get('state'),
        'publication': 'Collection candidates only; admission is not inferred',
        'planned': 0, 'raw_screen_rows': 0, 'handoff_screened': 0, 'selected': 0, 'terminal': 0,
        'positive_candidates': 0, 'waiting_for_screen': 0, 'waiting_for_screen_handoff': 0,
        'unreviewed_handoffs': 0, 'waiting_for_queue': 0,
        'waiting_for_terminal': 0, 'screened_out': 0, 'unavailable': 0,
        'terminal_dispositions': {}, 'admitted_training_rows': None,
        'receipt_integrity': 'not_observed'}
    try:
        plan_path = _relative_path(root, row['plan_manifest'], 'plan_manifest')
        input_path = _relative_path(root, row['input_path'], 'input_path')
        runtime_path = _relative_path(root, row['runtime_path'], 'runtime_path')
        batch_plan_dir = _relative_path(root, row['batch_plan_dir'], 'batch_plan_dir') if row.get('batch_plan_dir') else None
        input_rows, source_hash = _jsonl_and_digest(input_path)
        if source_hash != row['input_sha256']:
            raise ValueError('collection source input SHA differs from registry')
        runtime_manifest = runtime_path / 'frozen-runtime.json'
        if _digest(runtime_manifest) != row['runtime_manifest_sha256']:
            raise ValueError('runtime manifest SHA differs from registry')
        if row.get('runtime_receipt_path') or row.get('runtime_receipt_sha256'):
            receipt_path = _relative_path(root, row['runtime_receipt_path'], 'runtime_receipt_path')
            if _digest(receipt_path) != row.get('runtime_receipt_sha256'):
                raise ValueError('collection runtime receipt SHA differs from registry')
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('schema') != 'natlang.collection-runtime-receipt/1' or receipt.get('input_sha256') != row['input_sha256'] or receipt.get('runtime_manifest_sha256') != row['runtime_manifest_sha256']:
                raise ValueError('collection runtime receipt source/runtime binding differs')
        plan_bytes = plan_path.read_bytes()
        if hashlib.sha256(plan_bytes).hexdigest() != row['plan_manifest_sha256']:
            raise ValueError('batch plan manifest SHA differs from registry')
        plan = json.loads(plan_bytes)
        input_by_id = {item.get('id'): item for item in input_rows if isinstance(item.get('id'), str)}
        input_order = [item['id'] for item in input_rows]
        if len(input_by_id) != len(input_rows):
            raise ValueError('collection source has missing or duplicate episode IDs')
        if plan.get('schema') == 'natlang.headroom-batch-plan/1':
            if not batch_plan_dir or batch_plan_dir.resolve() != plan_path.parent.resolve():
                raise ValueError('batch ID directory must be the plan manifest directory')
            if plan.get('input_sha256') != row['input_sha256'] or plan.get('collection_runtime_sha256') != row['runtime_manifest_sha256']:
                raise ValueError('batch plan source/runtime identity differs from registry')
            plan_batches = plan.get('batches')
        elif plan.get('schema') == 'natlang.contractnli-pilot/1':
            if plan.get('sha256') != row['input_sha256'] or plan.get('episodes') != len(input_rows):
                raise ValueError('ContractNLI pilot manifest source/count differs')
            if row.get('upstream_input_path') or row.get('upstream_input_sha256'):
                if not row.get('upstream_input_path') or not row.get('upstream_input_sha256') or plan.get('source_sha256') != row['upstream_input_sha256']:
                    raise ValueError('ContractNLI upstream source identity differs from registry')
                if _digest(_relative_path(root, row['upstream_input_path'], 'upstream_input_path')) != row['upstream_input_sha256']:
                    raise ValueError('ContractNLI upstream source bytes differ from registry')
            plan_batches = [{'id': row.get('single_batch_id', 'all'), 'episodes': len(input_rows), 'sha256': None, 'ids': None}]
        else:
            raise ValueError('unsupported screened collection plan schema')
        planned_episode_count = sum(item.get('episodes', 0) for item in plan_batches) if isinstance(plan_batches, list) else None
        if planned_episode_count != len(input_rows) or (plan.get('episodes') is not None and plan.get('episodes') != len(input_rows)):
            raise ValueError('batch plan source row count differs')
        batch_paths = {item.get('id'): item for item in row['batches']}
        if not isinstance(plan_batches, list) or len(batch_paths) != len(row['batches']) or set(batch_paths) != {item.get('id') for item in plan_batches}:
            raise ValueError('registered batch paths differ from the fixed batch plan')
        assigned = []
        report['planned'] = len(input_rows)
        batch_reports = []
        observed_screen_files = {}
        for batch in plan_batches:
            batch_id = batch['id']
            paths = batch_paths[batch_id]
            if batch.get('ids') is None:
                batch_ids = input_order
            else:
                if Path(batch['ids']).name != batch['ids']:
                    raise ValueError(f'{batch_id} ID list must be a filename inside the pinned plan directory')
                ids_path = batch_plan_dir / batch['ids']
                ids_bytes = ids_path.read_bytes()
                if hashlib.sha256(ids_bytes).hexdigest() != batch['sha256']:
                    raise ValueError(f'{batch_id} id-list hash differs from fixed plan')
                batch_ids = json.loads(ids_bytes)
            if not isinstance(batch_ids, list) or len(batch_ids) != batch.get('episodes') or len(set(batch_ids)) != len(batch_ids):
                raise ValueError(f'{batch_id} id list is malformed')
            if any(item_id not in input_by_id for item_id in batch_ids):
                raise ValueError(f'{batch_id} contains an ID outside the source input')
            if batch_ids != [item_id for item_id in input_order if item_id in set(batch_ids)]:
                raise ValueError(f'{batch_id} IDs do not preserve source input order')
            assigned.extend(batch_ids)
            kept_path = _relative_path(root, paths['kept_path'], f'{batch_id}.kept_path')
            queue_path = _relative_path(root, paths['queue_path'], f'{batch_id}.queue_path')
            handoff_path = _relative_path(root, paths['handoff_path'], f'{batch_id}.handoff_path')
            item_report = {'id': batch_id, 'planned': len(batch_ids), 'selected': 0,
                'terminal': 0, 'positive_candidates': 0, 'waiting_for_screen': 0,
                'waiting_for_queue': 0, 'waiting_for_terminal': 0, 'screened_out': 0,
                'unavailable': 0, 'terminal_dispositions': {}, 'receipt_integrity': 'not_observed'}
            screen_path = _relative_path(root, paths['screen_path'], f'{batch_id}.screen_path') if paths.get('screen_path') else None
            raw_screens = []
            if screen_path and screen_path.is_file():
                cache_key = str(screen_path.resolve())
                if cache_key not in observed_screen_files:
                    file_rows, raw_screen_hash = _jsonl_and_digest(screen_path)
                    file_by_id = {screen.get('episode'): screen for screen in file_rows}
                    if len(file_by_id) != len(file_rows) or not set(file_by_id) <= set(input_by_id):
                        raise ValueError(f'{batch_id} raw screen file has duplicate or out-of-source episode IDs')
                    observed_screen_files[cache_key] = (file_rows, file_by_id, raw_screen_hash, len(file_rows))
                    report['raw_screen_rows'] += len(file_rows)
                file_rows, file_by_id, raw_screen_hash, raw_file_row_count = observed_screen_files[cache_key]
                raw_screens = [screen for screen in file_rows if screen.get('episode') in set(batch_ids)]
                item_report['raw_screen_rows'] = len(raw_screens)
                item_report['raw_screen_file_rows'] = raw_file_row_count
                item_report['raw_screen_sha256'] = raw_screen_hash
                if paths.get('screen_sha256') and raw_screen_hash != paths['screen_sha256']:
                    raise ValueError(f'{batch_id} raw screen SHA differs from registry')
                for screen in raw_screens:
                    if screen.get('input_sha256') != row['input_sha256']:
                        raise ValueError(f'{batch_id} raw screen input differs')
                    if screen.get('schema') != 'natlang.episode-headroom/2':
                        raise ValueError(f'{batch_id} raw screen schema differs')
                    if row.get('screen_identity') and screen.get('screen') != row['screen_identity']:
                        raise ValueError(f'{batch_id} raw screen identity differs')
                    if row.get('executor') and screen.get('executor') != row['executor']:
                        raise ValueError(f'{batch_id} raw screen executor differs')
                    if screen.get('cases') != len(input_by_id[screen['episode']].get('support', {}).get('cases', [])):
                        raise ValueError(f'{batch_id}/{screen["episode"]} raw screen support case count differs')
                item_report['waiting_for_screen'] = len(batch_ids) - len(raw_screens)
            else:
                item_report['waiting_for_screen'] = len(batch_ids)
            if not handoff_path.is_file():
                if paths.get('handoff_sha256'):
                    raise ValueError(f'{batch_id} registered immutable handoff receipt is missing')
                item_report['waiting_for_screen_handoff'] = len(batch_ids)
                item_report['observed_state'] = 'waiting_for_screen_handoff'
                report['waiting_for_screen_handoff'] += len(batch_ids)
                report['waiting_for_screen'] += item_report['waiting_for_screen']
                batch_reports.append(item_report)
                continue
            handoff_bytes = handoff_path.read_bytes()
            if paths.get('handoff_sha256') and hashlib.sha256(handoff_bytes).hexdigest() != paths['handoff_sha256']:
                raise ValueError(f'{batch_id} handoff receipt SHA differs from registry')
            handoff = json.loads(handoff_bytes)
            if handoff.get('schema') != 'natlang.screened-skill-handoff/1':
                raise ValueError(f'{batch_id} handoff schema differs')
            if handoff.get('input_sha256') != row['input_sha256'] or handoff.get('runtime_manifest_sha256') != row['runtime_manifest_sha256']:
                raise ValueError(f'{batch_id} handoff source/runtime identity differs')
            if row.get('screen_identity') and handoff.get('screen_identity') != row['screen_identity']:
                raise ValueError(f'{batch_id} handoff screen identity differs from registry')
            if handoff.get('screened_batch_ids_sha256') != batch['sha256'] or handoff.get('screened') != len(batch_ids):
                raise ValueError(f'{batch_id} handoff is not bound to the fixed ID batch')
            screen_rows = handoff.get('screen_rows')
            if not isinstance(screen_rows, list) or {item.get('episode') for item in screen_rows} != set(batch_ids) or len(screen_rows) != len(batch_ids):
                raise ValueError(f'{batch_id} handoff does not account for each planned screen row')
            # Match run_screened_skill_queue's canonical JSON serialization (ASCII escapes enabled).
            screen_hash = hashlib.sha256(json.dumps(screen_rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if handoff.get('screen_sha256') != screen_hash:
                raise ValueError(f'{batch_id} screen rows differ from receipt hash')
            if not kept_path.is_file():
                raise ValueError(f'{batch_id} handoff exists but kept packet is unavailable')
            kept_rows, kept_hash = _jsonl_and_digest(kept_path)
            if kept_hash != handoff.get('kept_sha256'):
                raise ValueError(f'{batch_id} kept packet differs from immutable receipt')
            handoff_pin = paths.get('handoff_sha256')
            kept_pin = paths.get('kept_sha256')
            if bool(handoff_pin) != bool(kept_pin):
                raise ValueError(f'{batch_id} registry must pin both handoff and kept bytes, or neither')
            if handoff_pin and hashlib.sha256(handoff_bytes).hexdigest() != handoff_pin:
                raise ValueError(f'{batch_id} handoff receipt SHA differs from registry')
            if kept_pin and kept_hash != kept_pin:
                raise ValueError(f'{batch_id} kept packet SHA differs from registry')
            item_report['receipt_integrity'] = 'registry_pinned' if handoff_pin else 'self_consistent_unpinned'
            band = handoff.get('band')
            if not isinstance(band, list) or len(band) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in band) or not 0 <= band[0] < band[1] <= 1:
                raise ValueError(f'{batch_id} selection band is malformed')
            by_screen = {item['episode']: item for item in screen_rows}
            expected_rows = []
            for episode_id in batch_ids:
                screen = by_screen[episode_id]
                if screen.get('input_sha256') != row['input_sha256']:
                    raise ValueError(f'{batch_id}/{episode_id} screen input differs')
                if screen.get('screen') != handoff.get('screen_identity') or screen.get('schema') != 'natlang.episode-headroom/2':
                    raise ValueError(f'{batch_id}/{episode_id} screen identity/schema differs')
                if row.get('executor') and screen.get('executor') != row['executor']:
                    raise ValueError(f'{batch_id}/{episode_id} screen executor differs from registry')
                quality = screen.get('support_quality')
                if screen.get('error') or screen.get('skipped'):
                    continue
                if isinstance(quality, bool) or not isinstance(quality, (int, float)) or not math.isfinite(quality) or not 0 <= quality <= 1:
                    raise ValueError(f'{batch_id}/{episode_id} has invalid support quality')
                if not band[0] <= quality <= band[1]:
                    continue
                episode = input_by_id[episode_id]
                expected_rows.append({**episode, 'provenance': {**episode.get('provenance', {}), 'headroom': {
                    'executor': screen.get('executor'), 'screen': screen.get('screen'),
                    'support_quality': quality, 'band': band}}})
            if kept_rows != expected_rows or len(kept_rows) != handoff.get('selected'):
                raise ValueError(f"{batch_id} kept rows differ from the receipt's support-only selection")
            if screen_path and not screen_path.is_file():
                raise ValueError(f'{batch_id} handoff exists but registered raw screen file is missing')
            if screen_path:
                if raw_screens != screen_rows:
                    raise ValueError(f'{batch_id} raw screen file differs from handoff screen rows')
            item_report.update(handoff_screened=len(screen_rows), selected=len(kept_rows), screened_out=len(batch_ids)-len(kept_rows),
                handoff_sha256=hashlib.sha256(handoff_bytes).hexdigest(), kept_sha256=kept_hash)
            report['handoff_screened'] += len(screen_rows)
            if item_report['receipt_integrity'] == 'self_consistent_unpinned':
                report['unreviewed_handoffs'] += 1
            report['selected'] += len(kept_rows)
            report['screened_out'] += len(batch_ids)-len(kept_rows)
            queue_json = queue_path / 'queue.json'
            if not queue_json.is_file():
                item_report['waiting_for_queue'] = len(kept_rows)
                item_report['observed_state'] = 'selected_waiting_for_collection_queue'
                report['waiting_for_queue'] += len(kept_rows)
                batch_reports.append(item_report)
                continue
            queue_bytes = queue_json.read_bytes()
            queue_identity = json.loads(queue_bytes)
            kept_ids = [item['id'] for item in kept_rows]
            if queue_identity.get('schema') != 'natlang.skill-authoring-queue/1' or queue_identity.get('input_sha256') != kept_hash:
                raise ValueError(f'{batch_id} queue input does not match receipt-pinned kept packet')
            if queue_identity.get('runtime_manifest_sha256') != row['runtime_manifest_sha256'] or queue_identity.get('episode_ids') != kept_ids:
                raise ValueError(f'{batch_id} queue runtime or selected IDs differ from pinned handoff')
            tasks_root = queue_path / 'tasks'
            if tasks_root.exists():
                expected_dirs = {hashlib.sha256(item_id.encode()).hexdigest()[:20] for item_id in kept_ids}
                actual_dirs = {path.name for path in tasks_root.iterdir() if path.is_dir()}
                if not actual_dirs <= expected_dirs:
                    raise ValueError(f'{batch_id} contains collection task directories outside selected IDs')
            terminal_states = []
            for episode_id, episode in zip(kept_ids, kept_rows):
                task = tasks_root / hashlib.sha256(episode_id.encode()).hexdigest()[:20]
                if task.is_symlink():
                    raise ValueError(f'{batch_id}/{episode_id} task directory is a symlink')
                episode_path = task / 'episode.jsonl'
                state_path = task / 'state.json'
                if episode_path.is_symlink() or state_path.is_symlink():
                    raise ValueError(f'{batch_id}/{episode_id} collection input/state is a symlink')
                if not episode_path.exists():
                    item_report['waiting_for_terminal'] += 1
                    continue
                # run_skill_authoring_queue writes per-task episode files with Python's default
                # ensure_ascii=True serialization; compare exactly that queue input encoding.
                expected_task = json.dumps(episode, separators=(',', ':')) + '\n'
                if episode_path.read_text(encoding='utf-8') != expected_task:
                    raise ValueError(f'{batch_id}/{episode_id} task input differs from selected kept bytes')
                if not state_path.is_file():
                    item_report['waiting_for_terminal'] += 1
                    continue
                state = json.loads(state_path.read_text(encoding='utf-8'))
                if state.get('episode') != episode_id:
                    raise ValueError(f'{batch_id}/{episode_id} state names a different episode')
                if 'terminal' in state and not isinstance(state['terminal'], bool):
                    raise ValueError(f'{batch_id}/{episode_id} terminal flag is not boolean')
                if 'positive' in state and not isinstance(state['positive'], bool):
                    raise ValueError(f'{batch_id}/{episode_id} positive flag is not boolean')
                if state.get('terminal') is True:
                    attempts = state.get('attempts', [])
                    if not isinstance(attempts, list):
                        raise ValueError(f'{batch_id}/{episode_id} attempt list is malformed')
                    attempt_numbers = []
                    for attempt in attempts:
                        if not isinstance(attempt, dict):
                            raise ValueError(f'{batch_id}/{episode_id} attempt receipt is not an object')
                        number = attempt.get('attempt')
                        if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                            raise ValueError(f'{batch_id}/{episode_id} has invalid attempt number')
                        if number in attempt_numbers:
                            raise ValueError(f'{batch_id}/{episode_id} has duplicate attempt receipts')
                        attempt_numbers.append(number)
                        result_path = task / f'attempt-{number:03d}' / hashlib.sha256(episode_id.encode()).hexdigest()[:20] / 'result.json'
                        if result_path.is_symlink() or not result_path.is_file() or _digest(result_path) != attempt.get('sha256'):
                            raise ValueError(f'{batch_id}/{episode_id} terminal attempt result missing or hash-mismatched')
                        result_row = json.loads(result_path.read_text(encoding='utf-8'))
                        if result_row.get('episode') != episode_id or result_row.get('disposition') != attempt.get('disposition') or (result_row.get('positive') is True) != (attempt.get('positive') is True):
                            raise ValueError(f'{batch_id}/{episode_id} attempt receipt differs from result')
                    if attempts:
                        latest = max(attempts, key=lambda attempt: attempt['attempt'])
                        if state.get('disposition') != latest.get('disposition') or (state.get('positive') is True) != (latest.get('positive') is True):
                            raise ValueError(f'{batch_id}/{episode_id} terminal state differs from latest result')
                    elif state.get('positive') is True:
                        raise ValueError(f'{batch_id}/{episode_id} positive candidate has no saved result receipt')
                    terminal_states.append(state)
            item_report['terminal'] = len(terminal_states)
            item_report['positive_candidates'] = sum(state.get('positive') is True for state in terminal_states)
            item_report['terminal_dispositions'] = dict(Counter(state.get('disposition', 'unknown') for state in terminal_states))
            if item_report['waiting_for_terminal']:
                item_report['observed_state'] = 'collection_incomplete_or_waiting'
            else:
                item_report['observed_state'] = 'collection_terminal'
            if item_report['receipt_integrity'] == 'self_consistent_unpinned':
                item_report['review_state'] = 'handoff_observed_unreviewed'
            report['terminal'] += item_report['terminal']
            report['positive_candidates'] += item_report['positive_candidates']
            report['waiting_for_terminal'] += item_report['waiting_for_terminal']
            for disposition, count in item_report['terminal_dispositions'].items():
                report['terminal_dispositions'][disposition] = report['terminal_dispositions'].get(disposition, 0) + count
            batch_reports.append(item_report)
        if len(assigned) != len(input_rows) or len(set(assigned)) != len(input_rows) or set(assigned) != set(input_by_id):
            raise ValueError('fixed batch plan does not partition every source ID exactly once')
        report['batches'] = batch_reports
        receipt_counts = Counter(item.get('receipt_integrity', 'not_observed') for item in batch_reports)
        report['receipt_integrity_counts'] = dict(receipt_counts)
        observed_receipts = {key for key in receipt_counts if key != 'not_observed'}
        if observed_receipts == {'registry_pinned'}:
            report['receipt_integrity'] = 'registry_pinned'
        elif observed_receipts == {'self_consistent_unpinned'}:
            report['receipt_integrity'] = 'handoff_observed_unreviewed'
        elif observed_receipts:
            report['receipt_integrity'] = 'mixed_pinned_and_unreviewed'
        if report['unavailable'] == 0:
            if report['waiting_for_screen']:
                report['observed_state'] = 'waiting_for_screen'
            elif report['waiting_for_screen_handoff']:
                report['observed_state'] = 'waiting_for_screen_handoff'
            elif report['waiting_for_queue']:
                report['observed_state'] = 'selected_waiting_for_queue'
            elif report['waiting_for_terminal']:
                report['observed_state'] = 'collection_incomplete_or_waiting'
            else:
                report['observed_state'] = 'collection_terminal'
        if report['unreviewed_handoffs'] and report['observed_state'] == 'collection_terminal':
            report['observed_state'] = 'handoff_observed_unreviewed'
        report['source_input_sha256'] = source_hash
        report['plan_manifest_sha256'] = row['plan_manifest_sha256']
        report['runtime_manifest_sha256'] = row['runtime_manifest_sha256']
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        report.update(observed_state='unavailable_or_invalid', error=str(error), unavailable=1)
    return report

def _observe_artifact_source(root, row):
    """Keep imported artifact tasks separate from executable SkillEpisode targets."""
    report = {'id': row['id'], 'state': row['state'], 'training_admitted': False}
    try:
        from audit_visual_source_tasks import audit
        paths = {key: _relative_path(root, row[key], key) for key in ('tasks', 'manifest', 'audit', 'acquisition', 'source_registry')}
        for key in paths:
            expected = row['sha256'] if key == 'tasks' else row[key + '_sha256']
            if _digest(paths[key]) != expected:
                raise ValueError(key + ' SHA differs from registry')
        manifest = json.loads(paths['manifest'].read_text())
        receipt = json.loads(paths['acquisition'].read_text())
        sources = json.loads(paths['source_registry'].read_text())['sources']
        source = next(s for s in sources if s['id'] == row['source_id'])
        if manifest['source'] != source or receipt['source'] != source or receipt['status'] != 'complete':
            raise ValueError('source lock or acquisition identity differs')
        if manifest['acquisition_receipt_sha256'] != row['acquisition_sha256']:
            raise ValueError('preparation acquisition binding differs')
        actual_audit = audit(paths['manifest'].parent)
        stored_audit = json.loads(paths['audit'].read_text())
        if actual_audit != stored_audit or not actual_audit['passed']:
            raise ValueError('live preparation audit differs or fails')
        report.update(tasks=manifest['tasks'], groups=manifest['groups'], statuses=manifest['statuses'],
            blockers=manifest['blockers'], unsupported_rows=manifest['held_unsupported_rows'],
            audit='preparation_identity_and_input_separation_only', native_collector_episodes=0)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        report.update(state='held_artifact_unavailable_or_invalid', error=str(error))
    return report


def inventory(root, registry):
    result = {'schema': 'natlang.self-improvement-task-inventory/1', 'corpora': [], 'backlog': registry['backlog'], 'errors': []}
    for row in registry['corpora']:
        report = {'id': row['id'], 'state': row['state']}
        try:
            tasks_path = root / row['tasks']
            manifest_path = root / row['manifest']
            manifest_bytes = manifest_path.read_bytes()
            manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
            if row.get('manifest_sha256') and manifest_digest != row['manifest_sha256']:
                raise ValueError('corpus manifest SHA differs from registry')
            manifest = json.loads(manifest_bytes)
            declared_episodes = _declared_episode_count(manifest)
            row_count, actual_cases, actual = _scan_corpus_jsonl(tasks_path, declared_episodes is not None)
            manifest_hash = manifest.get('sha256', manifest.get('outputs',manifest.get('artifacts',{})).get(Path(row['tasks']).name, {}).get('sha256'))
            if actual != row['sha256'] or actual != manifest_hash:
                raise ValueError('task hash differs from registry/manifest')
            expected_count = declared_episodes if declared_episodes is not None else manifest.get('tasks', manifest.get('task_count', manifest.get('counts', {}).get('candidate_rows', manifest.get('candidate_counts', {}).get('task_count'))))
            if row_count != expected_count:
                raise ValueError('manifest task/episode count differs')
            if declared_episodes is None:
                actual_cases = row_count
            declared_cases = manifest.get('cases', manifest.get('candidate_rows'))
            if declared_cases is not None and actual_cases != declared_cases:
                raise ValueError('manifest case count differs')
            report.update(sha256=actual, episodes=declared_episodes or 0, cases=actual_cases)
            if row.get('audit'):
                audit_path = root / row['audit']
                if audit_path.suffix == '.jsonl' and row.get('selection_ledger_sha256'):
                    ledger_digest = _digest(audit_path)
                    if ledger_digest != row['selection_ledger_sha256']:
                        raise ValueError('selection ledger SHA differs from registry')
                    report['audit'] = 'selection_ledger_hash_verified; semantic admission not inferred'
                    report['selection_ledger_sha256'] = ledger_digest
                    result['corpora'].append(report)
                    continue
                audit = json.loads(audit_path.read_text())
                audit_hash = audit.get('input_sha256')
                if audit_hash is None:
                    packets = audit.get('packets', [])
                    if len(packets) != 1:
                        raise ValueError('audit must bind exactly one packet')
                    audit_hash = packets[0].get('sha256')
                if audit_hash != actual or audit.get('errors') or audit.get('passed') is False:
                    raise ValueError('audit has errors or different input hash')
                report['audit'] = 'passed'
            else:
                report['audit'] = 'executor_pending'
        except (OSError, ValueError, KeyError) as error:
            report['state'] = 'held_artifact_unavailable_or_invalid'
            report['error'] = str(error)
            result['errors'].append({'id': row['id'], 'error': str(error)})
        result['corpora'].append(report)
    active = [report for report in result['corpora'] if not report['state'].startswith(('excluded_', 'held_', 'source_backing_', 'superseded_'))]
    result['totals'] = {'active_prepared_episodes': sum(report.get('episodes', 0) for report in active),
        'audited_problem_instances': sum(report.get('cases', 0) for report in active if report['audit'] == 'passed'),
        'executor_pending_source_tasks': sum(report.get('cases', 0) for report in active if report['audit'] == 'executor_pending'),
        'admitted_training_trajectories': None}
    result['artifact_sources'] = [_observe_artifact_source(root, row) for row in registry.get('artifact_sources', [])]
    result['artifact_source_totals'] = {'prepared_task_packets': sum(r.get('tasks', 0) for r in result['artifact_sources']),
        'source_groups': sum(r.get('groups', 0) for r in result['artifact_sources']),
        'only_executor_pending': sum(r.get('statuses', {}).get('prepared_requires_artifact_executor', 0) for r in result['artifact_sources']),
        'held_source_or_evaluator_review': sum(r.get('statuses', {}).get('held_source_or_evaluator_review', 0) for r in result['artifact_sources']),
        'native_collector_episodes': 0, 'admitted_training_trajectories': 0}
    result['errors'].extend({'id': r['id'], 'error': r['error']} for r in result['artifact_sources'] if r.get('error'))
    result['held'] = [{'id': report['id'], 'state': report['state'], 'episodes': report.get('episodes', 0), 'cases': report.get('cases', 0)} for report in result['corpora'] if report['state'].startswith('held_')]
    result['collections'] = []
    for row in registry.get('collections', []):
        report = {'id': row['id'], 'declared_state': row['state'], 'publication': 'Candidates only; admission not inferred'}
        try:
            directory = root / row['path']
            identity = json.loads((directory / 'queue.json').read_text())
            if identity['input_sha256'] != row['input_sha256'] or identity['runtime_manifest_sha256'] != row['runtime_manifest_sha256']:
                raise ValueError('collection identity differs from registry')
            ids = set(identity['episode_ids'])
            states = [json.loads(path.read_text()) for path in (directory / 'tasks').glob('*/state.json')]
            if len({state['episode'] for state in states}) != len(states) or any(state['episode'] not in ids for state in states):
                raise ValueError('collection state is duplicated or outside pinned input')
            terminal = [state for state in states if state.get('terminal') is True]
            report.update(episodes=len(ids), terminal=len(terminal), unfinished=len(ids)-len(terminal),
                dispositions=dict(Counter(state.get('disposition', 'unknown') for state in terminal)),
                positive_candidates=sum(state.get('positive') is True for state in terminal))
        except (OSError, ValueError, KeyError) as error:
            report['observed_state'] = 'unavailable_or_invalid'
            report['error'] = str(error)
        result['collections'].append(report)
    result['screened_collection_plans'] = [_observe_screened_plan(root, row) for row in registry.get('screened_collection_plans', [])]
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--registry', default='training/self_improvement_tasks.json')
    args = parser.parse_args()
    result = inventory(args.root, json.loads((args.root / args.registry).read_text()))
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['errors']))
