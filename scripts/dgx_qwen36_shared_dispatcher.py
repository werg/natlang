#!/usr/bin/env python3
"""Drainable, work-sharing supervisor for reviewed DGX queue entries.

This is a local-only orchestration candidate. It never edits source IR, original
queue files, existing journals, or runtime files. One queue entry is leased to
one existing run_queue.py invocation; entries remain indivisible batches.
"""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256  # noqa: E402
from natlang_neuralese.common.jsonio import utc_now_iso as utc_now  # noqa: E402


draining = False


def read_rows(path):
    rows = []
    with Path(path).open('rb') as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except (ValueError, UnicodeError) as error:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {error}') from error
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_number}: queue row must be an object')
            rows.append((value, raw.rstrip(b'\r\n'), line_number))
    return rows


def journal_finish_state(paths):
    finished = set()
    events = []
    for journal in paths:
        path = Path(journal)
        if not path.exists():
            continue
        with path.open('rb') as source:
            for line_number, raw in enumerate(source, 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except (ValueError, UnicodeError) as error:
                    raise ValueError(f'{path}:{line_number}: malformed journal row: {error}') from error
                if row.get('event') == 'finish' and isinstance(row.get('key'), str):
                    finished.add(row['key'])
                    if not row.get('batch_key'):
                        events.append(row)
    events.sort(key=lambda row: float(row.get('time', 0)))
    streak = 0
    next_allowed_at = 0.0
    for row in events:
        status = row.get('status')
        streak = 0 if status in {'complete', 'complete_with_skips', 'skipped'} else streak + 1
        try:
            next_allowed_at = float(row.get('next_allowed_at') or 0)
        except (TypeError, ValueError):
            next_allowed_at = 0.0
    return finished, streak, next_allowed_at


def verify_pins(path):
    pins = json.loads(Path(path).read_text())
    if pins.get('version') != 'natlang.dgx_dispatch_pins/1':
        raise ValueError('pins file has an unsupported version')
    checked = {}
    for item in pins.get('files', []):
        target = Path(item['path'])
        actual = sha256(target)
        if actual != item['sha256']:
            raise ValueError(f'pin mismatch: {target}: expected {item["sha256"]}, got {actual}')
        checked[str(target.resolve())] = actual
    if not checked:
        raise ValueError('pins file has no file hashes')
    runtime_root = Path(pins['runtime_root']).resolve()
    runtime_manifest_path = Path(pins['runtime_manifest_path']).resolve()
    if checked.get(str(runtime_manifest_path)) != pins.get('runtime_manifest_sha256'):
        raise ValueError('runtime manifest is not pinned to its declared digest')
    manifest = json.loads(runtime_manifest_path.read_text())
    manifest_files = manifest.get('files')
    if not isinstance(manifest_files, dict) or len(manifest_files) != pins.get('runtime_file_count'):
        raise ValueError('runtime freeze manifest has an unexpected file inventory')
    expected_runtime = {str((runtime_root / relative).resolve()): digest
                        for relative, digest in manifest_files.items()}
    if any(checked.get(path) != digest for path, digest in expected_runtime.items()):
        raise ValueError('runtime file hashes differ from frozen runtime receipt')
    actual_runtime = set()
    for root, _dirs, files in os.walk(runtime_root):
        for name in files:
            actual_runtime.add(str((Path(root) / name).resolve()))
    pinned_runtime = {name for name in checked if Path(name).resolve().is_relative_to(runtime_root)}
    if actual_runtime != pinned_runtime:
        raise ValueError('runtime tree differs from exact pinned runtime file set')
    return checked


def verify_drain_receipt(path, checked_pins):
    receipt = json.loads(Path(path).read_text())
    if receipt.get('version') != 'natlang.dgx_dispatch_handoff/1' or receipt.get('status') != 'drained':
        raise ValueError('handoff receipt must explicitly attest drained status')
    if receipt.get('active_supervisor_pids') != [] or receipt.get('active_collector_pids') != []:
        raise ValueError('handoff receipt reports active prior processes')
    if receipt.get('pins') != checked_pins:
        raise ValueError('handoff receipt pins differ from the verified current pins')
    return receipt


def validate_queue_rows(queue_paths):
    entries = []
    keys = set()
    ranges = {}
    queue_hashes = {}
    for queue_path in queue_paths:
        queue_hashes[str(Path(queue_path).resolve())] = sha256(queue_path)
        for entry, raw, line_number in read_rows(queue_path):
            key = entry.get('key')
            source = entry.get('source')
            start = entry.get('index')
            count = entry.get('count', 1)
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError(f'{queue_path}:{line_number}: missing or duplicate queue key {key!r}')
            if not isinstance(source, str) or not source or type(start) is not int or type(count) is not int or count < 1:
                raise ValueError(f'{queue_path}:{line_number}: invalid source range')
            if count > 16:
                raise ValueError(f'{key}: batch count {count} exceeds current collector ceiling 16')
            span = ranges.setdefault(source, [])
            end = start + count
            if any(start < previous_end and previous_start < end for previous_start, previous_end, _ in span):
                raise ValueError(f'{key}: overlapping source range in {source}')
            span.append((start, end, key))
            keys.add(key)
            entries.append({'entry': entry, 'raw': raw, 'origin_queue': str(Path(queue_path).resolve()),
                            'origin_line': line_number})
    return entries, queue_hashes


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, target)


def log_event(path, event):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as output:
        output.write(json.dumps(event, sort_keys=True) + '\n')
        output.flush()
        os.fsync(output.fileno())


def current_failure_gate(journals):
    _, streak, next_allowed_at = journal_finish_state(journals)
    return streak, next_allowed_at


def handle_signal(_signum, _frame):
    global draining
    draining = True


def terminate_child_if_dispatcher_dies():
    # An unplanned parent death must not leave an unjournaled collector running.
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM)
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        # The root handoff/active-PID gate remains the authoritative duplicate guard.
        pass


def dispatch(args):
    global draining
    pins = verify_pins(args.pins)
    handoff = verify_drain_receipt(args.handoff, pins)
    entries, queue_hashes = validate_queue_rows(args.queue)
    for path, digest in queue_hashes.items():
        if pins.get(path) != digest:
            raise ValueError(f'queue is not pinned: {path}')
    required_pins = {str(Path(args.run_queue).resolve()), str(Path(__file__).resolve())}
    required_pins.update(str(Path(item['entry']['source']).resolve()) for item in entries)
    if args.chat_request_config:
        required_pins.add(str(Path(args.chat_request_config).resolve()))
    if not required_pins.issubset(pins):
        raise ValueError(f'missing required execution pins: {sorted(required_pins - set(pins))}')
    if handoff.get('queue_hashes') != queue_hashes:
        raise ValueError('handoff receipt queue hashes differ from current immutable queue files')
    prior_journals = [str(Path(p).resolve()) for p in args.prior_journal]
    journal_hashes = {path: (sha256(path) if Path(path).exists() else None) for path in prior_journals}
    if handoff.get('prior_journal_hashes') != journal_hashes:
        raise ValueError('prior journal bytes differ from the drained handoff receipt')
    slot_journals = [str(Path(args.journal_dir) / f'journal-slot-{slot:02d}.jsonl')
                     for slot in range(args.max_slots)]
    all_journals = prior_journals + slot_journals
    attempted, initial_streak, initial_cooldown = journal_finish_state(all_journals)
    pending = [item for item in entries if item['entry']['key'] not in attempted]
    stages = args.stage
    if stages[0]['after_completed_batches'] != 0:
        raise ValueError('first ramp stage must begin at zero completed batches')
    if stages[-1]['slots'] > args.max_slots:
        raise ValueError('ramp requires more slots than --max-slots')
    for stage in stages:
        if stage['slots'] < 1 or stage['model_concurrency_per_slot'] < 1:
            raise ValueError('stage slots and per-slot concurrency must be positive')
        stage['request_ceiling'] = stage['slots'] * stage['model_concurrency_per_slot']
        stage['active_case_ceiling'] = stage['slots'] * max(
            (item['entry'].get('count', 1) for item in entries), default=1)

    control = Path(args.control_dir)
    control.mkdir(parents=True, exist_ok=True)
    lock_path = control / 'dispatcher.lock'
    lock = lock_path.open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError('another dispatcher holds the exclusive campaign lock') from error

    queue_dir = control / 'leased-entry-queues'
    queue_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for ordinal, item in enumerate(pending):
        suffix = hashlib.sha256(item['entry']['key'].encode()).hexdigest()[:16]
        one_entry = queue_dir / f'{ordinal:06d}-{suffix}.jsonl'
        if one_entry.exists():
            if one_entry.read_bytes() != item['raw'] + b'\n':
                raise ValueError(f'entry queue identity collision: {one_entry}')
        else:
            one_entry.write_bytes(item['raw'] + b'\n')
        staged.append({**item, 'one_entry_queue': str(one_entry)})

    summary = {
        'version': 'natlang.dgx_worksharing_dispatch/1', 'status': 'running',
        'started_at': utc_now(), 'campaign': args.campaign,
        'pins': pins, 'handoff_receipt': str(Path(args.handoff)),
        'queue_hashes': queue_hashes, 'prior_journals': prior_journals,
        'slot_journals': slot_journals, 'pending_before_dispatch': len(pending),
        'attempted_keys_skipped': len(entries) - len(pending),
        'initial_failure_streak': initial_streak, 'initial_retry_cooldown_until': initial_cooldown,
        'ramp': stages, 'maximum_batch_cases': max(
            (item['entry'].get('count', 1) for item in entries), default=1),
        'case_budgets_unchanged_per_entry': True,
        'entry_attempts_preserve_original_keys_and_payloads': True,
    }
    atomic_json(control / 'dispatch-state.json', summary)
    if args.status_file:
        atomic_json(args.status_file, {'state': 'running' if summary['status'] == 'running' else ('finished' if summary['status'] == 'complete' else 'stopped' if summary['status'] == 'drained' else 'paused'), 'dispatcher_pid': os.getpid(), 'updated_at': utc_now(), 'dispatch': summary})
    log_event(args.journal, {'event': 'dispatcher_start', 'time': time.time(), 'at': utc_now(),
        'campaign': args.campaign, 'pending_entries': len(pending), 'initial_failure_streak': initial_streak,
        'handoff_sha256': sha256(args.handoff), 'pins_sha256': sha256(args.pins)})
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    active = {}
    completed_keys = set(attempted)
    completed_batches = 0
    last_stage_index = -1
    cursor = 0
    paused_reason = None
    failed = 0

    while cursor < len(staged) or active:
        if draining:
            paused_reason = 'drain_requested; active batch children are allowed to finish'
        stage_index = max(index for index, stage in enumerate(stages)
                          if completed_batches >= stage['after_completed_batches'])
        if stage_index != last_stage_index:
            stage = stages[stage_index]
            log_event(args.journal, {'event': 'ramp_stage', 'time': time.time(), 'at': utc_now(),
                'stage': stage_index, 'completed_batches': completed_batches,
                'slots': stage['slots'], 'model_concurrency_per_slot': stage['model_concurrency_per_slot'],
                'request_ceiling': stage['request_ceiling'],
                'active_case_ceiling': stage['active_case_ceiling']})
            last_stage_index = stage_index
        else:
            stage = stages[stage_index]

        if not draining and paused_reason is None:
            streak, cooldown = current_failure_gate(all_journals)
            if streak >= args.max_failure_streak:
                paused_reason = 'consecutive_failure_pause_gate'
                log_event(args.journal, {'event': 'dispatcher_pause', 'time': time.time(), 'at': utc_now(),
                    'reason': paused_reason, 'failure_streak': streak})
            elif cooldown > time.time():
                # Cooldown blocks every new lease, including while older batches run.
                time.sleep(min(1, cooldown - time.time()))
            else:
                available_slots = stage['slots'] - len(active)
                while available_slots > 0 and cursor < len(staged) and not draining and paused_reason is None:
                    item = staged[cursor]
                    cursor += 1
                    entry = item['entry']
                    if entry['key'] in completed_keys:
                        continue
                    if args.min_free_mib:
                        free = os.statvfs(control).f_bavail * os.statvfs(control).f_frsize // (1024 * 1024)
                        if free < args.min_free_mib:
                            paused_reason = f'storage_pause:{free}MiB<{args.min_free_mib}MiB'
                            log_event(args.journal, {'event': 'dispatcher_pause', 'time': time.time(), 'at': utc_now(),
                                'reason': paused_reason, 'free_mib': free,
                                'minimum_free_mib': args.min_free_mib})
                            break
                    slot = next(index for index in range(args.max_slots) if index not in active)
                    slot_journal = slot_journals[slot]
                    log_path = control / f'slot-{slot:02d}.supervisor.log'
                    command = [sys.executable, args.run_queue, item['one_entry_queue'], slot_journal,
                        '--runtime', args.runtime, '--case-seconds', str(args.case_seconds),
                        '--model-id', args.model_id, '--server', args.server,
                        '--model-concurrency', str(stage['model_concurrency_per_slot']),
                        '--min-free-mib', str(args.min_free_mib), '--reasoning-effort', args.reasoning_effort]
                    if args.chat_request_config:
                        command += ['--chat-request-config', args.chat_request_config]
                    if args.no_observation_seconds:
                        command += ['--no-observation-seconds', str(args.no_observation_seconds)]
                    if args.execution_plans:
                        command.append('--execution-plans')
                    log = log_path.open('ab')
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True,
                                             preexec_fn=terminate_child_if_dispatcher_dies)
                    active[slot] = {'process': child, 'log': log, 'entry': item,
                                    'started_at': time.time(), 'command': command}
                    log_event(args.journal, {'event': 'lease_start', 'time': time.time(), 'at': utc_now(),
                        'slot': slot, 'key': entry['key'], 'queue_entry_sha256': hashlib.sha256(item['raw']).hexdigest(),
                        'per_slot_request_cap': stage['model_concurrency_per_slot'],
                        'global_request_ceiling': stage['request_ceiling'],
                        'active_case_ceiling': stage['active_case_ceiling']})
                    available_slots -= 1

        finished = []
        for slot, item in list(active.items()):
            code = item['process'].poll()
            if code is not None:
                item['log'].close()
                finished.append((slot, item, code))
        for slot, item, code in finished:
            del active[slot]
            key = item['entry']['entry']['key']
            # One-entry run_queue journal has the terminal exact queue-key event.
            terminal = None
            for journal_path in [slot_journals[slot]]:
                path = Path(journal_path)
                if path.exists():
                    with path.open() as source:
                        for line in source:
                            if line.strip():
                                row = json.loads(line)
                                if row.get('event') == 'finish' and row.get('key') == key:
                                    terminal = row
            if terminal:
                completed_keys.add(key)
                if code == 0 and terminal.get('status') in {'complete', 'complete_with_skips'} and terminal.get('output_accounting', {}).get('complete'):
                    completed_batches += 1
                if terminal.get('status') in {'complete', 'complete_with_skips', 'skipped'}:
                    pass
                else:
                    failed += 1
            else:
                failed += 1
            if code != 0 or terminal is None:
                paused_reason = 'child_exit_without_successful_terminal'
                log_event(args.journal, {'event': 'dispatcher_pause', 'time': time.time(), 'at': utc_now(),
                    'reason': paused_reason, 'key': key, 'child_exit_code': code})
            log_event(args.journal, {'event': 'lease_finish', 'time': time.time(), 'at': utc_now(),
                'slot': slot, 'key': key, 'child_exit_code': code,
                'terminal_status': terminal.get('status') if terminal else None,
                'terminal_accounting_complete': bool(terminal and terminal.get('output_accounting', {}).get('complete')),
                'elapsed_seconds': round(time.time() - item['started_at'], 2)})
        if finished:
            # Reread journals after completions; a failure threshold pauses only new leases.
            streak, _ = current_failure_gate(all_journals)
            if streak >= args.max_failure_streak and not draining:
                paused_reason = 'consecutive_failure_pause_gate'
                log_event(args.journal, {'event': 'dispatcher_pause', 'time': time.time(), 'at': utc_now(),
                    'reason': paused_reason, 'failure_streak': streak})
        if active:
            time.sleep(0.5)
        elif cursor < len(staged) and not draining and paused_reason is None:
            time.sleep(0.2)
        elif draining or paused_reason is not None:
            break

    terminal_status = 'drained' if draining else ('paused' if paused_reason else 'complete')
    end = {'event': 'dispatcher_finish', 'time': time.time(), 'at': utc_now(), 'status': terminal_status,
        'reason': paused_reason, 'completed_batches': completed_batches,
        'failed_batches': failed, 'unleased_pending_entries': max(0, len(staged) - cursor),
        'active_children': len(active), 'active_supervisor_pids': [item['process'].pid for item in active.values()]}
    log_event(args.journal, end)
    summary.update({'status': terminal_status, 'finished_at': utc_now(), 'completed_batches': completed_batches,
                    'failed_batches': failed, 'unleased_pending_entries': max(0, len(staged) - cursor),
                    'drain_reason': paused_reason})
    atomic_json(control / 'dispatch-state.json', summary)
    if args.status_file:
        atomic_json(args.status_file, {'state': 'running' if summary['status'] == 'running' else ('finished' if summary['status'] == 'complete' else 'stopped' if summary['status'] == 'drained' else 'paused'), 'dispatcher_pid': os.getpid(), 'updated_at': utc_now(), 'dispatch': summary})
    print(json.dumps(end, sort_keys=True))


def parse_stage(text):
    try:
        value = json.loads(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f'invalid JSON ramp: {error}') from error
    if not isinstance(value, list) or not value:
        raise argparse.ArgumentTypeError('ramp must be a non-empty JSON array')
    required = {'after_completed_batches', 'slots', 'model_concurrency_per_slot'}
    for stage in value:
        if not isinstance(stage, dict) or set(stage) != required or any(
                type(stage[key]) is not int or stage[key] < 0 for key in required):
            raise argparse.ArgumentTypeError(f'each stage must have exactly integer fields {sorted(required)}')
    if value != sorted(value, key=lambda item: item['after_completed_batches']):
        raise argparse.ArgumentTypeError('ramp thresholds must be ascending')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--queue', action='append', required=True, help='reviewed, immutable queue JSONL; repeat for multiple inputs')
    parser.add_argument('--prior-journal', action='append', default=[], help='read-only prior journal; repeat for all attempt history')
    parser.add_argument('--journal', required=True, help='append-only dispatcher event log')
    parser.add_argument('--journal-dir', required=True, help='new per-slot append-only run_queue journals')
    parser.add_argument('--status-file', help='atomic worker status for the existing result sync service')
    parser.add_argument('--control-dir', required=True, help='new isolated control/lease directory')
    parser.add_argument('--pins', required=True, help='hash-pinned queues, runtime, and supervisor')
    parser.add_argument('--handoff', required=True, help='root/controller drained receipt bound to exact pins')
    parser.add_argument('--run-queue', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--max-slots', type=int, default=16)
    parser.add_argument('--stage', action='append', type=parse_stage, required=True,
                        help='JSON array of ramp stages, each with after_completed_batches, slots, model_concurrency_per_slot')
    parser.add_argument('--case-seconds', type=int, default=19200)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--server', default='http://127.0.0.1:8082')
    parser.add_argument('--chat-request-config')
    parser.add_argument('--reasoning-effort', default='high')
    parser.add_argument('--min-free-mib', type=int, default=4096)
    parser.add_argument('--no-observation-seconds', type=int, default=0)
    parser.add_argument('--max-failure-streak', type=int, default=3)
    parser.add_argument('--execution-plans', action='store_true')
    args = parser.parse_args()
    if args.max_slots < 1 or args.case_seconds < 1 or args.min_free_mib < 0 or args.max_failure_streak < 1:
        parser.error('slot, time, storage, and failure limits must be positive/nonnegative')
    args.stage = [stage for profile in args.stage for stage in profile]
    try:
        dispatch(args)
    except Exception as error:
        # A crashed coordinator must not leave a misleading running status.
        if args.status_file:
            atomic_json(args.status_file, {'state': 'paused', 'reason': 'dispatcher_exception',
                'exception_type': type(error).__name__, 'dispatcher_pid': os.getpid(),
                'updated_at': utc_now()})
        raise


if __name__ == '__main__':
    main()
