#!/usr/bin/env python3
"""Persist hourly generation audits and flag idle workers for agent review."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import hashlib
from generation_authority import authority_lock


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def live(pid, signature):
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
        return signature.encode() in command
    except (OSError, TypeError, ValueError):
        return False


def additional_teacher_alerts(authority):
    alerts = []
    for name, teacher in authority.get('additional_teachers', {}).items():
        state = teacher.get('state', 'unknown')
        if teacher.get('status_file'):
            try:
                state = read_json(Path(teacher['status_file'])).get('state', 'unknown')
            except (OSError, ValueError):
                alerts.append(f'{name}: worker status unreadable')
        if state.startswith('paused') or state in {'failed', 'stopped'}:
            alerts.append(f'{name}: {state}; investigate before resuming')
        elif state == 'finished':
            alerts.append(f'{name}: assignment finished; review results and replenish qualified work')
        elif teacher.get('sync_status'):
            try:
                sync = read_json(Path(teacher['sync_status']))
                checked = dt.datetime.fromisoformat(sync['checked_at'].replace('Z', '+00:00'))
                if (dt.datetime.now(dt.timezone.utc) - checked).total_seconds() > 180:
                    alerts.append(f'{name}: remote sync stale; remote progress unknown')
                if sync.get('transfer_exit_code') or sync.get('import_exit_code'):
                    alerts.append(f'{name}: remote transfer or import failed')
            except (OSError, ValueError, KeyError, TypeError):
                alerts.append(f'{name}: remote sync status unreadable')
    return alerts


def same_queue_worker_live(command):
    """Find any already-running supervisor bound to this exact queue and journal."""
    expected = [os.fsencode(str(part)) for part in command]
    if len(expected) < 4:
        return None
    try:
        entries = Path('/proc').iterdir()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                actual = [part for part in (entry / 'cmdline').read_bytes().split(b'\0') if part]
            except OSError:
                continue
            if len(actual) >= 4 and actual[1:4] == expected[1:4]:
                return int(entry.name)
    except OSError:
        return None
    return None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


SUCCESSOR_PIN_BASENAMES = {
    'curriculum-policy.js', 'curriculum.js', 'source-review.js', 'source-conversion.js',
    'run_bonsai_queue.py', 'start_reviewed_luna_slots.py',
    'start_reviewed_generation_successor.py', 'generation_authority.py',
}
SUCCESSOR_COMPLETE_STATUSES = {'complete', 'complete_with_skips', 'skipped'}
SUCCESSOR_TERMINAL_LAUNCH_STATUSES = {'running', 'complete', 'completed', 'finished', 'success'}


def process_command(pid):
    try:
        return [part.decode(errors='replace') for part in
                Path(f'/proc/{int(pid)}/cmdline').read_bytes().split(b'\0') if part]
    except (OSError, TypeError, ValueError):
        return None


def reviewed_predecessor_failures(plan):
    """Report unreviewed failed canonical parent finishes, never child/member aliases."""
    reviewed = plan.get('reviewed_failed_finishes', {})
    alerts = []
    for predecessor in plan.get('predecessors', []):
        queue_path = Path(predecessor.get('queue', ''))
        journal_path = Path(predecessor.get('journal', ''))
        try:
            keys = {json.loads(line)['key'] for line in queue_path.read_text().splitlines() if line.strip()}
            finishes = {}
            for line in journal_path.read_text().splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get('event') == 'finish' and not event.get('batch_key'):
                    finishes[event['key']] = event
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            alerts.append(f'predecessor_evidence_unreadable:{journal_path}:{type(error).__name__}:{error}')
            continue
        for key in keys:
            # Mirror the reviewed handoff check: only the latest parent finish per queue key counts.
            event = finishes.get(key)
            if event is None:
                continue
            good = (event.get('status') in SUCCESSOR_COMPLETE_STATUSES
                    and event.get('output_accounting', {}).get('complete') is True)
            if good:
                continue
            canonical_hash = hashlib.sha256(json.dumps(
                event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if reviewed.get(key) == canonical_hash:
                continue
            alerts.append(f'unreviewed_predecessor_finish:{key}:sha256={canonical_hash}')
    return alerts


def reviewed_successor_health(authority):
    """Check small, root-pinned successor controller bindings on each monitor heartbeat."""
    checks, alerts = [], []
    registry = authority.get('reviewed_successor_controllers', [])
    if not isinstance(registry, list):
        return [{'status': 'blocked', 'error': 'reviewed_successor_controllers must be a list'}], [
            'successor_registry_invalid:reviewed_successor_controllers must be a list']
    for descriptor in registry:
        if not isinstance(descriptor, dict):
            checks.append({'status': 'needs_review', 'error': 'successor descriptor must be an object'})
            alerts.append('successor_registry_invalid:descriptor must be an object')
            continue
        check = {'pid': descriptor.get('pid'), 'plan_path': descriptor.get('plan_path'),
                 'plan_sha256': descriptor.get('plan_sha256'), 'launch_record': descriptor.get('launch_record'),
                 'status': 'ok', 'alerts': [], 'checked_artifact_hashes': {}}
        local = check['alerts']
        try:
            plan_path = Path(descriptor['plan_path'])
            launch_path = Path(descriptor['launch_record'])
            expected_plan_sha = descriptor['plan_sha256']
            expected_command = descriptor['command']
            if not isinstance(expected_command, list) or len(expected_command) < 2:
                raise ValueError('reviewed successor command is missing or incomplete')
            if not plan_path.is_file():
                raise ValueError('reviewed successor plan is missing')
            actual_plan_sha = sha256_file(plan_path)
            if actual_plan_sha != expected_plan_sha:
                local.append(f'plan_hash_mismatch:expected={expected_plan_sha}:actual={actual_plan_sha}')
            plan = read_json(plan_path)
            if str(Path(plan.get('launch_record', '')).resolve()) != str(launch_path.resolve()):
                local.append('plan_launch_record_path_mismatch')
            if plan.get('root_approved') is not True:
                local.append('plan_not_root_approved')
            for file, expected_hash in plan.get('artifact_hashes', {}).items():
                if Path(file).name not in SUCCESSOR_PIN_BASENAMES:
                    continue
                try:
                    actual_hash = sha256_file(file)
                except OSError as error:
                    local.append(f'pinned_artifact_missing:{file}:{type(error).__name__}')
                    continue
                check['checked_artifact_hashes'][file] = actual_hash
                if actual_hash != expected_hash:
                    local.append(f'pinned_artifact_hash_mismatch:{file}:expected={expected_hash}:actual={actual_hash}')
            local.extend(reviewed_predecessor_failures(plan))
            controller_live = live(descriptor.get('pid'), Path(expected_command[1]).name)
            check['controller_live'] = controller_live
            if controller_live:
                actual_command = process_command(descriptor.get('pid'))
                if actual_command != expected_command:
                    local.append(f'controller_command_mismatch:actual={actual_command!r}')
            launch = read_json(launch_path) if launch_path.is_file() else None
            if launch is None:
                if not controller_live:
                    local.append('controller_dead_before_launch_record')
                check['launch_status'] = 'missing'
            else:
                check['launch_status'] = launch.get('status')
                if (str(Path(launch.get('plan', '')).resolve()) != str(plan_path.resolve())
                        or launch.get('plan_sha256') != expected_plan_sha):
                    local.append('launch_record_plan_binding_mismatch')
                workers = launch.get('workers', [])
                planned_workers = plan.get('workers', [])
                if (launch.get('status') in SUCCESSOR_TERMINAL_LAUNCH_STATUSES
                        and (not isinstance(workers, list) or not isinstance(planned_workers, list)
                             or len(workers) != len(planned_workers))):
                    local.append(f'terminal_launch_worker_count_mismatch:launched={len(workers) if isinstance(workers, list) else "invalid"}:'
                                 f'planned={len(planned_workers) if isinstance(planned_workers, list) else "invalid"}')
                if not controller_live and launch.get('status') not in SUCCESSOR_TERMINAL_LAUNCH_STATUSES:
                    local.append(f'controller_dead_during_handoff:launch_status={launch.get("status")}')
            if local:
                check['status'] = 'needs_review'
        except (OSError, ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            local.append(f'{type(error).__name__}: {error}')
            check['status'] = 'needs_review'
        alerts.extend(f'successor_controller:{check.get("pid")}:{alert}' for alert in local)
        checks.append(check)
    return checks, alerts


def verify_frozen_runtime(runtime_path, manifest_path):
    runtime_path = Path(runtime_path).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('frozen runtime manifest has no file hash map')
    checked = 0
    for relative, expected in files.items():
        target = (runtime_path / relative).resolve()
        if runtime_path not in target.parents:
            raise ValueError(f'frozen runtime file escapes runtime root: {relative}')
        if sha256_file(target) != expected:
            raise ValueError(f'frozen runtime file hash mismatch: {relative}')
        checked += 1
    return checked


def latest_journal_event(path):
    latest = None
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                latest = json.loads(line)
    return latest


def storage_recovery_target(authority, approval):
    """Return the currently authorized worker binding, or fail closed."""
    kind = approval.get('worker_kind')
    queue = str(Path(approval['queue_path']).resolve())
    journal = str(Path(approval['journal_path']).resolve())
    runtime = str(Path(approval['runtime_path']).resolve())
    if kind == 'luna':
        matches = [w for w in authority.get('luna_workers', [])
                   if str(Path(w.get('queue', '')).resolve()) == queue
                   and str(Path(w.get('journal', '')).resolve()) == journal]
        if len(matches) != 1:
            raise ValueError('approved Luna queue/journal is not bound to exactly one authority worker')
        worker = matches[0]
        if str(Path(worker.get('runtime', authority.get('luna_runtime', ''))).resolve()) != runtime:
            raise ValueError('approved Luna runtime differs from current authority')
        command = worker.get('command')
        if not isinstance(command, list) or command != approval.get('command'):
            raise ValueError('approved command differs from current Luna worker command')
        return kind, worker, command
    if kind == 'bonsai':
        if (str(Path(authority.get('bonsai_queue', '')).resolve()) != queue
                or str(Path(authority.get('bonsai_journal', '')).resolve()) != journal
                or str(Path(authority.get('bonsai_runtime', '')).resolve()) != runtime):
            raise ValueError('approved Bonsai queue/journal/runtime differs from current authority')
        command = approval.get('command')
        if not isinstance(command, list) or len(command) < 4:
            raise ValueError('approved Bonsai command is missing')
        return kind, {'pid': authority.get('bonsai_supervisor'), 'queue': queue,
                      'journal': journal, 'runtime': runtime}, command
    raise ValueError('storage recovery approval has unsupported worker_kind')


def unattended_storage_pauses(authority):
    targets = [('bonsai', authority.get('bonsai_supervisor'), authority.get('bonsai_journal'))]
    targets.extend(('luna', worker.get('pid'), worker.get('journal'))
                   for worker in authority.get('luna_workers', []))
    paused = []
    for kind, pid, journal in targets:
        if not journal or live(pid, 'run_bonsai_queue.py'):
            continue
        try:
            event = latest_journal_event(journal)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            paused.append({'kind': kind, 'pid': pid, 'journal': str(Path(journal).resolve()),
                           'queue': None, 'error': f'{type(error).__name__}: {error}'})
            continue
        if event and event.get('event') == 'storage_pause':
            if kind == 'bonsai':
                queue = authority.get('bonsai_queue')
            else:
                matching = next((w for w in authority.get('luna_workers', [])
                                 if str(Path(w.get('journal', '')).resolve()) == str(Path(journal).resolve())), None)
                queue = matching.get('queue') if matching else None
            if not queue:
                paused.append({'kind': kind, 'pid': pid, 'journal': str(Path(journal).resolve()),
                               'queue': None, 'error': 'authority has no queue bound to paused journal'})
                continue
            paused.append({'kind': kind, 'pid': pid, 'journal': str(Path(journal).resolve()),
                           'queue': str(Path(queue).resolve()), 'event': event})
    return paused


def recovery_approval_entries(config):
    entries = config.get('approvals', [])
    if not entries and config.get('approval_path') and config.get('approval_sha256'):
        entries = [{'approval_path': config['approval_path'], 'approval_sha256': config['approval_sha256']}]
    return entries if isinstance(entries, list) else []


def recover_one_storage_pause(authority_path, paused, approval_entry, config_snapshot):
    try:
        approval_path = Path(approval_entry['approval_path'])
        expected_approval_hash = approval_entry['approval_sha256']
        if not approval_path.is_file() or not isinstance(expected_approval_hash, str):
            raise ValueError('root-pinned recovery approval is missing')
        with authority_lock(authority_path):
            current = read_json(authority_path)
            current_config = current.get('storage_pause_recovery', {})
            if current_config != config_snapshot or current_config.get('enabled') is not True:
                raise ValueError('recovery authority changed or became disabled')
            if approval_entry not in recovery_approval_entries(current_config):
                raise ValueError('recovery approval is no longer configured in authority')
            approval_bytes = approval_path.read_bytes()
            if hashlib.sha256(approval_bytes).hexdigest() != expected_approval_hash:
                raise ValueError('root recovery approval hash mismatch')
            approval = json.loads(approval_bytes)
            if approval.get('schema') != 'generation-storage-recovery/1' or approval.get('root_approved') is not True:
                raise ValueError('recovery approval lacks explicit root approval')
            if approval.get('authority_path') != str(Path(authority_path).resolve()):
                raise ValueError('recovery approval authority path mismatch')
            kind, worker, command = storage_recovery_target(current, approval)
            if kind != paused['kind'] or str(Path(approval['journal_path']).resolve()) != paused['journal']:
                raise ValueError('approval does not bind this unattended storage pause')
            binding = approval.get('authority_binding', {})
            if kind == 'luna' and binding.get('worker_number') != worker.get('number'):
                raise ValueError('approved Luna worker slot differs from current authority')
            if binding.get('queue_path') != str(Path(approval['queue_path']).resolve()) or binding.get('journal_path') != str(Path(approval['journal_path']).resolve()):
                raise ValueError('approved authority queue/journal binding is missing or mismatched')
            if binding.get('runtime_path') != str(Path(approval['runtime_path']).resolve()):
                raise ValueError('approved authority runtime binding is missing or mismatched')
            if approval.get('command') != command:
                raise ValueError('recovery command does not match approved worker command')
            supervisor = Path(approval['supervisor_path']).resolve()
            if not supervisor.is_file() or sha256_file(supervisor) != approval.get('supervisor_sha256'):
                raise ValueError('supervisor code hash mismatch')
            if not command or len(command) < 4 or Path(command[1]).resolve() != supervisor:
                raise ValueError('approved command is not bound to the approved supervisor')
            if (str(Path(command[2]).resolve()) != str(Path(approval['queue_path']).resolve())
                    or str(Path(command[3]).resolve()) != str(Path(approval['journal_path']).resolve())):
                raise ValueError('approved command queue/journal binding mismatch')
            if '--runtime' not in command or str(Path(command[command.index('--runtime') + 1]).resolve()) != str(Path(approval['runtime_path']).resolve()):
                raise ValueError('approved command runtime binding mismatch')
            queue_path, journal_path = Path(approval['queue_path']), Path(approval['journal_path'])
            runtime_path = Path(approval['runtime_path'])
            manifest_path = Path(approval['runtime_manifest_path'])
            if sha256_file(queue_path) != approval.get('queue_sha256'):
                raise ValueError('queue hash mismatch')
            if str(manifest_path.resolve()) != str((runtime_path / 'frozen-runtime.json').resolve()):
                raise ValueError('runtime manifest path is not the frozen runtime manifest')
            if sha256_file(manifest_path) != approval.get('runtime_manifest_sha256'):
                raise ValueError('runtime manifest hash mismatch')
            runtime_files_checked = verify_frozen_runtime(runtime_path, manifest_path)
            scope = approval.get('scope')
            if scope not in {'exact_storage_pause', 'future_storage_pauses_same_queue'}:
                raise ValueError('recovery approval scope is invalid')
            if scope == 'exact_storage_pause' and sha256_file(journal_path) != approval.get('journal_sha256'):
                raise ValueError('journal hash mismatch for exact-event approval')
            last = latest_journal_event(journal_path)
            if not last or last.get('event') != 'storage_pause':
                return {'status': 'no_longer_paused', 'journal_path': str(journal_path.resolve())}
            event_key = f"{last.get('key')}@{last.get('time')}"
            if scope == 'exact_storage_pause' and (last.get('key') != approval.get('paused_key')
                    or last.get('time') != approval.get('paused_event_time')):
                raise ValueError('latest journal storage_pause differs from approved pause event')
            # In both modes, the pause must refer to exactly one uncompleted key on the pinned queue.
            queued_key_count = 0
            with queue_path.open() as stream:
                for line in stream:
                    if line.strip() and json.loads(line).get('key') == last.get('key'):
                        queued_key_count += 1
            if queued_key_count != 1:
                raise ValueError('paused key is not unique in the approved queue')
            finished = False
            with journal_path.open() as stream:
                for line in stream:
                    if line.strip():
                        entry = json.loads(line)
                        if entry.get('event') == 'finish' and entry.get('key') == last.get('key'):
                            finished = True
                            break
            if finished:
                raise ValueError('paused key already has a finish event; refusing replay')
            pid = worker.get('pid')
            if live(pid, supervisor.name):
                return {'status': 'already_live', 'pid': pid, 'event_key': event_key}
            duplicate_pid = same_queue_worker_live(command)
            if duplicate_pid is not None:
                raise ValueError(f'approved command is already running under unbound PID {duplicate_pid}')
            prior_claims = current.get('storage_pause_recovery_history', [])
            if (current.get('storage_pause_recovery_claim', {}).get('event_key') == event_key
                    or any(item.get('event_key') == event_key for item in prior_claims)):
                raise ValueError('this pause already has a recovery claim; refusing duplicate resume')
            original_floor = int(last.get('minimum_free_mib', 0))
            approved_floor = int(approval.get('original_minimum_free_mib', original_floor))
            minimum_free = max(original_floor, 4096)
            if approved_floor != original_floor:
                raise ValueError('approved original storage floor differs from journal')
            command_floor = int(command[command.index('--min-free-mib') + 1]) if '--min-free-mib' in command else 0
            if command_floor != original_floor:
                raise ValueError('original command storage floor differs from pause event')
            free_mib = shutil.disk_usage(journal_path.parent).free // (1024 * 1024)
            if free_mib < minimum_free:
                return {'status': 'waiting_for_space', 'event_key': event_key,
                        'free_mib': free_mib, 'required_free_mib': minimum_free}
            cwd = Path(approval['cwd']).resolve()
            if not cwd.is_dir():
                raise ValueError('approved working directory is invalid')
            log_path = Path(approval['log_path']).resolve()
            if str(log_path) != str(Path(worker.get('log', approval['log_path'])).resolve()):
                raise ValueError('approved log path differs from current worker binding')
            state_path, state = None, None
            if kind == 'luna':
                state_path = Path(current.get('luna_state', '')).resolve()
                if not state_path.is_file() or state_path != Path(approval['state_path']).resolve():
                    raise ValueError('approved Luna state path missing or differs from authority')
                state = read_json(state_path)
                if not any(str(Path(item.get('journal', '')).resolve()) == str(journal_path.resolve())
                           and str(Path(item.get('queue', '')).resolve()) == str(queue_path.resolve())
                           for item in state.get('workers', [])):
                    raise ValueError('Luna state has no matching queue/journal worker')
            claim = {'event_key': event_key, 'approval_sha256': expected_approval_hash,
                     'scope': scope, 'claimed_at': utc(), 'status': 'claimed', 'previous_pid': pid,
                     'pause_event': last,
                     'journal_sha256_at_claim': sha256_file(journal_path), 'queue_sha256': approval['queue_sha256'],
                     'runtime_manifest_sha256': approval['runtime_manifest_sha256'],
                     'runtime_files_checked': runtime_files_checked,
                     'supervisor_sha256': approval['supervisor_sha256']}
            current['storage_pause_recovery_claim'] = claim
            current.setdefault('storage_pause_recovery_history', []).append(claim)
            atomic_json(authority_path, current)
            with log_path.open('ab') as log:
                process = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            resumed_at = utc()
            claim.update(status='running', resumed_at=resumed_at, resumed_pid=process.pid,
                         queue_path=str(queue_path.resolve()), journal_path=str(journal_path.resolve()),
                         runtime_path=str(runtime_path.resolve()))
            current['storage_pause_recovery_claim'] = claim
            if kind == 'luna':
                for item in current['luna_workers']:
                    if str(Path(item.get('journal', '')).resolve()) == str(journal_path.resolve()):
                        item.update(pid=process.pid, status='resumed_after_storage_pause',
                                    resumed_at=resumed_at, resumed_from_pid=pid,
                                    recovery_approval_sha256=expected_approval_hash)
                        break
                atomic_json(authority_path, current)
                for item in state.get('workers', []):
                    if str(Path(item.get('journal', '')).resolve()) == str(journal_path.resolve()):
                        item.update(pid=process.pid, status='resumed_after_storage_pause',
                                    resumed_at=resumed_at, resumed_from_pid=pid,
                                    recovery_approval_sha256=expected_approval_hash)
                state.setdefault('storage_pause_recovery_history', []).append(claim)
                state.update(status='resumed_after_storage_pause', latest_storage_resume=claim)
                atomic_json(state_path, state)
            else:
                current.update(bonsai_supervisor=process.pid, bonsai_status='resumed_after_storage_pause',
                               last_completed_bonsai_pid=pid, bonsai_storage_resume=claim)
                atomic_json(authority_path, current)
            return {'status': 'resumed', 'pid': process.pid, 'previous_pid': pid,
                    'event_key': event_key, 'approval_sha256': expected_approval_hash,
                    'approval_scope': scope, 'runtime_files_checked': runtime_files_checked,
                    'required_free_mib': minimum_free, 'free_mib': free_mib}
    except Exception as error:
        return {'status': 'blocked', 'reason': f'{type(error).__name__}: {error}',
                'worker_kind': paused.get('kind'), 'queue_path': paused.get('queue'), 'journal_path': paused.get('journal')}


def recover_storage_pauses(authority_path, state_dir, authority):
    """Recover every approved unattended pause, at most once per pause event."""
    paused_workers = unattended_storage_pauses(authority)
    if not paused_workers:
        return []
    config = authority.get('storage_pause_recovery')
    if not isinstance(config, dict) or config.get('enabled') is not True:
        return [{'status': 'blocked', 'reason': 'storage recovery is not explicitly enabled',
                 'worker_kind': p.get('kind'), 'queue_path': p.get('queue'), 'journal_path': p.get('journal')}
                for p in paused_workers]
    entries = recovery_approval_entries(config)
    if not entries:
        return [{'status': 'blocked', 'reason': 'no root-pinned recovery approvals are configured',
                 'worker_kind': p.get('kind'), 'queue_path': p.get('queue'), 'journal_path': p.get('journal')}
                for p in paused_workers]
    results = []
    for paused in paused_workers:
        if paused.get('error'):
            results.append({'status': 'blocked', 'reason': 'journal could not be read: ' + paused['error'],
                            'worker_kind': paused['kind'], 'journal_path': paused['journal']})
            continue
        candidate = None
        for entry in entries:
            try:
                approval_path = Path(entry['approval_path'])
                if hashlib.sha256(approval_path.read_bytes()).hexdigest() != entry['approval_sha256']:
                    continue
                approval = json.loads(approval_path.read_bytes())
                if (approval.get('worker_kind') == paused['kind']
                        and str(Path(approval.get('queue_path', '')).resolve()) == paused.get('queue')
                        and str(Path(approval.get('journal_path', '')).resolve()) == paused.get('journal')):
                    candidate = entry
                    break
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if candidate is None:
            results.append({'status': 'blocked', 'reason': 'no matching root-pinned approval for paused queue/journal',
                            'worker_kind': paused['kind'], 'queue_path': paused.get('queue'), 'journal_path': paused['journal']})
        else:
            results.append(recover_one_storage_pause(authority_path, paused, candidate, config))
    return results


def mark_storage_pause_status(authority_path, paused_workers, results):
    """Keep authority/state honest when a dead worker remains paused or blocked."""
    updates = []
    with authority_lock(authority_path):
        current = read_json(authority_path)
        for paused, result in zip(paused_workers, results):
            status = result.get('status')
            if status not in {'blocked', 'waiting_for_space'}:
                continue
            journal = str(Path(paused['journal']).resolve())
            if live(paused.get('pid'), 'run_bonsai_queue.py'):
                continue
            try:
                last = latest_journal_event(journal)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not last or last.get('event') != 'storage_pause':
                continue
            worker_status = ('storage_pause_needs_agent_review' if status == 'blocked'
                             else 'storage_pause_waiting_for_space')
            if paused['kind'] == 'luna':
                for worker in current.get('luna_workers', []):
                    if str(Path(worker.get('journal', '')).resolve()) == journal:
                        worker['status'] = worker_status
                        worker['storage_pause_event_key'] = f"{last.get('key')}@{last.get('time')}"
                        break
                state_path = Path(current.get('luna_state', ''))
                if state_path.is_file():
                    state = read_json(state_path)
                    for worker in state.get('workers', []):
                        if str(Path(worker.get('journal', '')).resolve()) == journal:
                            worker['status'] = worker_status
                            worker['storage_pause_event_key'] = f"{last.get('key')}@{last.get('time')}"
                    active_workers = [w for w in current.get('luna_workers', [])
                                      if live(w.get('pid'), 'run_bonsai_queue.py')]
                    state['status'] = worker_status if not active_workers else 'running_with_storage_pause'
                    atomic_json(state_path, state)
                active_workers = [w for w in current.get('luna_workers', [])
                                  if live(w.get('pid'), 'run_bonsai_queue.py')]
                current['luna_status'] = worker_status if not active_workers else 'running_with_storage_pause'
            else:
                current['bonsai_status'] = worker_status
                current['bonsai_storage_pause_event_key'] = f"{last.get('key')}@{last.get('time')}"
            updates.append({'journal_path': journal, 'status': worker_status})
        if updates:
            atomic_json(authority_path, current)
    return updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True, type=Path)
    parser.add_argument('--health-reader', required=True, type=Path)
    parser.add_argument('--spool-reader', required=True, type=Path)
    parser.add_argument('--state-dir', required=True, type=Path)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                authority = read_json(args.authority)
                paused_workers = unattended_storage_pauses(authority)
                recovery = recover_storage_pauses(args.authority, args.state_dir, authority)
                status_updates = mark_storage_pause_status(args.authority, paused_workers, recovery)
                if any(item.get('status') == 'resumed' for item in recovery) or status_updates:
                    authority = read_json(args.authority)
                for item in recovery:
                    if item.get('status') == 'resumed':
                        event = dict(event='storage_pause_resume', time=utc(), recovery=item)
                        with (args.state_dir / 'journal.jsonl').open('a') as stream:
                            stream.write(json.dumps(event) + '\n')
                        print(json.dumps(event), flush=True)
                bonsai_live = live(authority.get('bonsai_supervisor'), 'run_bonsai_queue.py')
                luna = authority.get('luna_workers', [])
                luna_live = [w.get('pid') for w in luna if live(w.get('pid'), 'run_bonsai_queue.py')]
                needs_review = []
                needs_review.extend(additional_teacher_alerts(authority))
                needs_review.extend('storage_pause_recovery_blocked: ' + item.get('reason', 'approval mismatch')
                                    for item in recovery if item.get('status') == 'blocked')
                successor_checks, successor_alerts = reviewed_successor_health(authority)
                needs_review.extend(successor_alerts)
                if not bonsai_live:
                    needs_review.append('bonsai_idle: review completion and replenish qualified work')
                if len(luna_live) < authority.get('luna_concurrency', 2):
                    needs_review.append('luna_idle: review completion and replenish qualified work')
                heartbeat = dict(checked_at=utc(), pid=os.getpid(), bonsai_live=bonsai_live,
                                 luna_live_pids=luna_live, needs_agent_review=needs_review,
                                 next_check_at=authority['next_check_at'],
                                 storage_pause_recovery=recovery,
                                 storage_pause_status_updates=status_updates,
                                 reviewed_successor_controllers=successor_checks)
                atomic_json(args.state_dir / 'status.json', heartbeat)
                due = dt.datetime.fromisoformat(authority['next_check_at'].replace('Z', '+00:00'))
                if dt.datetime.now(dt.timezone.utc) >= due:
                    health = subprocess.run(['node', str(args.health_reader)], check=True,
                                            capture_output=True, text=True, timeout=900)
                    metadata = json.loads(health.stdout)
                    report_path = Path(metadata['jsonPath']).resolve()
                    report = read_json(report_path)
                    spool = subprocess.run(['node', str(args.spool_reader), str(report_path)],
                                           check=True, capture_output=True, text=True, timeout=300)
                    spool_metadata = json.loads(spool.stdout)
                    with authority_lock(args.authority):
                        current = read_json(args.authority)
                        # Do not overwrite a newer inspection's baseline or worker authority.
                        if current['checked_at'] == authority['checked_at']:
                            checked = dt.datetime.fromisoformat(report['checked_at'].replace('Z', '+00:00'))
                            current.update(checked_at=report['checked_at'],
                                           next_check_at=(checked + dt.timedelta(hours=1)).isoformat(),
                                           last_status_report=str(report_path),
                                           latest_bonsai_spool=spool_metadata['op'])
                            atomic_json(args.authority, current)
                    event = dict(event='hourly_audit', time=utc(), health=metadata,
                                 spool=spool_metadata,
                                 blocking_reader_errors=report['completeness']['blocking_error_count'],
                                 needs_agent_review=needs_review)
                    with (args.state_dir / 'journal.jsonl').open('a') as stream:
                        stream.write(json.dumps(event) + '\n')
                    print(json.dumps(event), flush=True)
            except Exception as error:
                event = dict(event='monitor_error', time=utc(), error=f'{type(error).__name__}: {error}')
                with (args.state_dir / 'journal.jsonl').open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
