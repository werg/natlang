#!/usr/bin/env python3
"""Hand one completed OpenRouter assignment to a separately reviewed successor."""
import argparse
import hashlib
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from generation_authority import authority_lock
from start_reviewed_generation_successor import atomic_json, digest, running
from start_reviewed_luna_slots import verify_finished


def real_file(path, label):
    """Require a pinned artifact path to be absolute, canonical, and present."""
    raw = str(path)
    candidate = Path(raw)
    if not candidate.is_absolute():
        raise ValueError(f'{label} path must be absolute: {raw}')
    resolved = candidate.resolve(strict=True)
    if str(resolved) != raw or not resolved.is_file():
        raise ValueError(f'{label} must name a canonical regular file: {raw}')
    return resolved


def checked_sha(path, expected, label):
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
        raise ValueError(f'{label} has no valid SHA-256 pin')
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'{label} hash mismatch: {path}')
    return actual


def path_identity(value):
    return Path(value).expanduser().resolve(strict=False)


def validate_path_hash_map(values, label, protected_paths=(), protected_dirs=()):
    if not isinstance(values, dict) or not values:
        raise ValueError(f'{label} must be a nonempty file-path-to-SHA-256 map')
    checked = {}
    protected = {str(path_identity(path)) for path in protected_paths}
    protected_directory_ids = tuple(path_identity(path) for path in protected_dirs)
    for raw_path, expected in values.items():
        identity = path_identity(raw_path)
        if (str(identity) in protected or
                any(identity == directory or directory in identity.parents for directory in protected_directory_ids)):
            raise ValueError(f'{label} must not include mutable authority/status/journal/control/launch artifacts')
        path = real_file(raw_path, label)
        checked_sha(path, expected, label)
        checked[str(path)] = expected
    return checked


def validate_worker_plan(successor, protected_paths=(), protected_dirs=()):
    """Validate the actual worker plan and resolve its launcher before any claim."""
    plan_path = real_file(successor.get('plan', ''), 'successor worker plan')
    expected_plan_sha = successor.get('plan_sha256')
    checked_sha(plan_path, expected_plan_sha, 'successor worker plan')
    worker_plan = json.loads(plan_path.read_text())
    if (worker_plan.get('approved') is not True or worker_plan.get('root_approved') is not True or
            worker_plan.get('provider') != 'openrouter' or
            worker_plan.get('model') != 'stealth/space-bunny-alpha' or
            worker_plan.get('model_concurrency') != 1):
        raise ValueError('successor worker plan must be root-approved for the single-request OpenRouter model')
    if not isinstance(worker_plan.get('cases'), int) or worker_plan['cases'] < 1:
        raise ValueError('successor worker plan must declare a positive case count')
    if successor.get('cases') != worker_plan['cases']:
        raise ValueError('successor case count differs from its approved worker plan')

    queue = real_file(worker_plan.get('queue', ''), 'successor queue')
    journal = Path(worker_plan.get('journal', ''))
    if not journal.is_absolute() or str(journal) != str(journal.resolve(strict=False)):
        raise ValueError('successor journal must be a canonical absolute path')
    if str(queue) != str(successor.get('queue')) or str(journal) != str(successor.get('journal')):
        raise ValueError('successor queue/journal binding differs from its approved worker plan')
    queue_count = sum(1 for line in queue.open() if line.strip())
    if queue_count != worker_plan['cases']:
        raise ValueError(f'approved worker case count {worker_plan["cases"]} differs from queue lines {queue_count}')

    # start_openrouter_teacher.py writes status to the worker plan's own parent,
    # regardless of a misleading status_file field in the plan.
    actual_status = plan_path.parent / 'worker-status.json'
    if str(actual_status) != str(actual_status.resolve(strict=False)):
        raise ValueError('worker status parent is not a canonical path')
    if str(successor.get('status_file')) != str(actual_status):
        raise ValueError('successor status_file must match start_openrouter_teacher.py output path beside its plan')
    if worker_plan.get('status_file') != str(actual_status):
        raise ValueError('approved worker plan declares a status_file that the launcher does not actually write')

    # The successor must point at the real OpenRouter worker launcher, never at
    # this rollover helper or another wrapper with a different CLI.
    launcher = real_file(worker_plan.get('launcher', ''), 'successor worker launcher')
    if launcher.name != 'start_openrouter_teacher.py':
        raise ValueError('successor worker launcher must be start_openrouter_teacher.py')
    mutable_worker_paths = list(protected_paths)
    mutable_worker_dirs = list(protected_dirs)
    for field in ('status_file', 'worker_status_file', 'journal', 'worker_log', 'worker_lock', 'control_dir',
                  'pool_status_file', 'hard_abort_file', 'case_events_file'):
        if worker_plan.get(field):
            if field == 'control_dir':
                mutable_worker_dirs.append(worker_plan[field])
            else:
                mutable_worker_paths.append(worker_plan[field])
    for field in ('status_file', 'journal', 'launcher_log'):
        if successor.get(field):
            mutable_worker_paths.append(successor[field])
    mutable_worker_paths.append(actual_status)
    pins = validate_path_hash_map(worker_plan.get('pins'), 'successor worker pin', mutable_worker_paths,
                                  mutable_worker_dirs)
    if pins.get(str(launcher)) != digest(launcher):
        raise ValueError('actual worker launcher is not pinned by its approved plan')
    if str(real_file(worker_plan.get('supervisor', ''), 'successor supervisor')) not in pins:
        raise ValueError('actual successor supervisor is not pinned by its approved plan')
    for required in ('provider_request_config', 'provider_dependency_pins', 'native_review'):
        target = real_file(worker_plan.get(required, ''), f'successor {required}')
        if str(target) not in pins:
            raise ValueError(f'successor {required} is not pinned by its approved plan')
    native_path = Path(worker_plan['native_review'])
    if worker_plan.get('native_review_sha256') != digest(native_path):
        raise ValueError('successor native review SHA differs from approved worker plan')
    native = json.loads(native_path.read_text())
    if native.get('model_calls') != 0 or native.get('runtime_manifest_sha256') != worker_plan.get('runtime_manifest_sha256'):
        raise ValueError('successor native proof must be provider-free and match its worker runtime')
    counts = native.get('counts')
    if (not isinstance(counts, dict) or counts.get('cases') != worker_plan['cases'] or
            counts.get('admitted') != worker_plan['cases'] or counts.get('denied') != 0 or
            native.get('rejected') != []):
        raise ValueError('successor native proof counts do not match the approved case count')

    dep_path = real_file(worker_plan['provider_dependency_pins'], 'provider dependency manifest')
    dep_manifest = json.loads(dep_path.read_text())
    validate_path_hash_map(dep_manifest.get('files'), 'provider dependency pin')

    runtime_path = Path(worker_plan.get('runtime', ''))
    if not runtime_path.is_absolute() or str(runtime_path) != str(runtime_path.resolve(strict=True)) or not runtime_path.is_dir():
        raise ValueError('successor runtime must be a canonical absolute directory')
    runtime_manifest_path = real_file(runtime_path / 'frozen-runtime.json', 'successor runtime manifest')
    checked_sha(runtime_manifest_path, worker_plan.get('runtime_manifest_sha256'), 'successor runtime manifest')
    runtime_manifest = json.loads(runtime_manifest_path.read_text())
    files = runtime_manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('successor runtime manifest has no file inventory')
    expected_files = set(files)
    for relative, expected in files.items():
        raw_target = runtime_path / relative
        target = raw_target.resolve(strict=True)
        if not target.is_relative_to(runtime_path):
            raise ValueError(f'successor runtime file escapes sealed directory: {relative}')
        if str(raw_target) != str(target):
            raise ValueError(f'successor runtime file path is not canonical: {relative}')
        checked_sha(target, expected, 'successor runtime file')
    actual_files, actual_links = set(), {}
    for path in runtime_path.rglob('*'):
        relative = path.relative_to(runtime_path).as_posix()
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(runtime_path):
                raise ValueError(f'successor runtime symlink escapes sealed directory: {relative}')
            actual_links[relative] = path.readlink().as_posix()
        elif path.is_file() and relative != 'frozen-runtime.json':
            actual_files.add(relative)
    expected_links = {item['path']: item['target'] for item in runtime_manifest.get('symlinks', [])}
    if actual_files != expected_files or actual_links != expected_links:
        raise ValueError('successor runtime file/symlink inventory differs from its seal')

    # Any rollover artifact pin must itself be a path, not a semantic label.
    if str(queue) not in pins:
        raise ValueError('successor queue is not pinned by its approved worker plan')
    return worker_plan, plan_path, launcher


def new_authority_binding(worker_plan, successor, launcher_pid, plan_path):
    """Build a fresh binding; never carry predecessor proof/counts forward."""
    queue = Path(worker_plan['queue']).resolve(strict=True)
    journal = Path(worker_plan['journal']).resolve(strict=False)
    runtime = Path(worker_plan['runtime']).resolve(strict=True)
    native_review = Path(worker_plan['native_review']).resolve(strict=True)
    return {
        'model': worker_plan['model'],
        'provider': worker_plan['provider'],
        'queue': str(queue),
        'queue_sha256': digest(queue),
        'journal': str(journal),
        'runtime': str(runtime),
        'runtime_manifest_sha256': worker_plan['runtime_manifest_sha256'],
        'cases': worker_plan['cases'],
        'concurrency': worker_plan['model_concurrency'],
        'launcher_pid': launcher_pid,
        'supervisor_pid': None,
        'status_file': str(Path(successor['status_file']).resolve(strict=False)),
        'launcher_log': str(Path(successor['launcher_log']).resolve(strict=False)),
        'plan': str(plan_path.resolve(strict=True)),
        'plan_sha256': digest(plan_path),
        'native_review': str(native_review),
        'native_review_sha256': worker_plan['native_review_sha256'],
        'state': 'starting',
        'actual_results_seen': 0,
        'completed_cases': 0,
        'updated_at': time.time(),
    }


def preflight_successor_outputs(successor):
    """Prepare output parents before claiming an immutable handoff receipt."""
    for field in ('launcher_log', 'journal', 'status_file'):
        value = successor.get(field)
        if not value:
            continue
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        if field == 'launcher_log':
            # Detect an unwritable log path before the launch is claimed.
            with path.open('a'):
                pass


def verify_predecessor_terminal(predecessor, reviewed_failures=None):
    """Verify terminal queue output while retaining explicitly reviewed failures.

    A failed key is never counted as complete.  The optional map is an exact
    per-key hash of the saved terminal journal event, matching
    ``start_reviewed_luna_slots.verify_finished``.  Extra, missing, or changed
    failures block the handoff.
    """
    if reviewed_failures is not None:
        if not isinstance(reviewed_failures, dict) or not reviewed_failures:
            raise ValueError('reviewed_failed_finishes must be a nonempty key-to-event-SHA map')
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or
               not re.fullmatch(r'[0-9a-f]{64}', value)
               for key, value in reviewed_failures.items()):
            raise ValueError('reviewed_failed_finishes contains an invalid key or SHA-256')

    if digest(predecessor['queue']) != predecessor['queue_sha256']:
        raise ValueError('Predecessor queue changed')
    keys = [json.loads(line)['key'] for line in Path(predecessor['queue']).read_text().splitlines() if line.strip()]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Predecessor queue keys are empty or duplicated')
    finishes = {}
    for journal in predecessor.get('journals') or [predecessor['journal']]:
        journal_path = Path(journal)
        if not journal_path.is_file():
            raise ValueError(f'Predecessor journal is missing: {journal_path}')
        for line in journal_path.read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get('event') == 'finish' and not event.get('batch_key'):
                prior = finishes.get(event.get('key'))
                if prior is None or event.get('time', 0) >= prior.get('time', 0):
                    finishes[event.get('key')] = event

    success = {'complete', 'complete_with_skips', 'skipped'}
    failures = {}
    failed_events = {}
    successful = []
    for key in keys:
        event = finishes.get(key)
        if event is None:
            raise ValueError(f'Predecessor has no terminal finish for {key}')
        if event.get('status') in success and event.get('output_accounting', {}).get('complete') is True:
            successful.append(key)
            continue
        actual = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if reviewed_failures is None or reviewed_failures.get(key) != actual:
            raise ValueError(f'Incomplete predecessor needs agent review: {key}')
        failures[key] = actual
        failed_events[key] = event
    if set(reviewed_failures or {}) != set(failures):
        raise ValueError('reviewed_failed_finishes must match exactly the incomplete predecessor keys')
    verify_finished(predecessor, reviewed_failures)
    return {'expected_keys': len(keys), 'successful_keys': len(successful),
            'reviewed_failed_keys': sorted(failures), 'reviewed_failed_event_sha256': failures,
            'reviewed_failed_events': failed_events,
            'all_keys_accounted': len(successful) + len(failures) == len(keys),
            'disposition': 'all_successfully_exported' if not failures else 'finished_with_reviewed_failures'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.sha256:
        raise ValueError('Reviewed successor plan changed')
    plan = json.loads(args.plan.read_text())
    if plan.get('root_approved') is not True:
        raise ValueError('Successor lacks root review')
    successor = plan['successor']
    authority_path = real_file(plan.get('authority', ''), 'generation authority')
    predecessor = plan.get('predecessor')
    if not isinstance(predecessor, dict):
        raise ValueError('rollover plan has no predecessor binding')
    reviewed_failures = plan.get('reviewed_failed_finishes')
    if reviewed_failures is not None:
        # Validate its shape before the wait loop; contents are rechecked from
        # the immutable predecessor journal when it becomes inactive.
        if not isinstance(reviewed_failures, dict) or not reviewed_failures:
            raise ValueError('reviewed_failed_finishes must be a nonempty key-to-event-SHA map')
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or
               not re.fullmatch(r'[0-9a-f]{64}', value)
               for key, value in reviewed_failures.items()):
            raise ValueError('reviewed_failed_finishes contains an invalid key or SHA-256')
    record = Path(plan.get('launch_record', ''))
    if not record.is_absolute() or str(record) != str(record.resolve(strict=False)):
        raise ValueError('launch_record must be a canonical absolute path')
    if record.exists() or record.is_symlink() or record.with_suffix('.lock').exists():
        raise ValueError('successor launch record or lock already exists')
    if not record.parent.is_dir():
        raise ValueError('launch_record parent directory must already exist')
    cwd = Path(plan.get('cwd', ''))
    if not cwd.is_absolute() or str(cwd) != str(cwd.resolve(strict=True)) or not cwd.is_dir():
        raise ValueError('rollover cwd must be a canonical absolute directory')
    protected_paths = [authority_path, record, record.with_suffix('.lock')]
    protected_dirs = []
    for source in (predecessor, successor):
        for field in ('journal', 'status_file', 'launcher_log', 'control_dir', 'pool_status_file',
                      'hard_abort_file', 'case_events_file', 'worker_log', 'worker_lock'):
            if source.get(field):
                if field == 'control_dir':
                    protected_dirs.append(source[field])
                else:
                    protected_paths.append(source[field])
    for field in ('status_file', 'journal', 'launcher_log', 'control_dir', 'pool_status_file',
                  'hard_abort_file', 'case_events_file', 'worker_log', 'worker_lock'):
        if plan.get(field):
            if field == 'control_dir':
                protected_dirs.append(plan[field])
            else:
                protected_paths.append(plan[field])
    worker_plan, worker_plan_path, worker_launcher = validate_worker_plan(successor, protected_paths, protected_dirs)
    artifact_hashes = validate_path_hash_map(plan.get('artifact_hashes'), 'rollover artifact pin', protected_paths,
                                             protected_dirs)
    script_path = Path(__file__).resolve(strict=True)
    if artifact_hashes.get(str(script_path)) != digest(script_path):
        raise ValueError('rollover plan must pin this exact reviewed successor helper')
    if artifact_hashes.get(str(worker_plan_path)) != digest(worker_plan_path):
        raise ValueError('rollover plan must pin the approved successor worker plan')
    current = json.loads(authority_path.read_text())['additional_teachers']['space_bunny']
    if (current.get('queue'), current.get('journal'), current.get('status_file')) != (
            predecessor.get('queue'), predecessor.get('journal'), predecessor.get('status_file')):
        raise ValueError('predecessor queue/journal/status binding is not current in authority')
    preflight_successor_outputs(successor)
    with record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists():
            raise ValueError('Successor has already been claimed')
        atomic_json(record, dict(status='waiting_for_predecessor', plan=str(args.plan.resolve()),
                                 plan_sha256=args.sha256))
        while True:
            current = json.loads(authority_path.read_text())['additional_teachers']['space_bunny']
            if (current['queue'], current['journal'], current.get('status_file')) != (
                    predecessor['queue'], predecessor['journal'], predecessor['status_file']):
                raise ValueError('Predecessor authority superseded')
            try:
                state = json.loads(Path(current['status_file']).read_text())
            except FileNotFoundError:
                # A reviewed launcher may still be checking pins/authentication
                # before writing its first status. Its live PID owns this wait.
                if running(current.get('launcher_pid')):
                    time.sleep(5)
                    continue
                raise ValueError('Predecessor exited without status; agent review required')
            if state['state'] in {'finished', 'incomplete'} and not running(state.get('launcher_pid')):
                # Incomplete can advance only when every failing terminal row
                # is explicitly reviewed and its exact journal event is pinned.
                predecessor_review = verify_predecessor_terminal(predecessor, reviewed_failures)
                break
            if state['state'].startswith('paused') or state['state'] == 'stopped':
                raise ValueError('Predecessor needs review before rollover')
            time.sleep(30)
        predecessor_review = verify_predecessor_terminal(predecessor, reviewed_failures)
        # Recheck every successor byte immediately before claim. The predecessor
        # may run long enough for any reviewed input or dependency to drift.
        worker_plan, worker_plan_path, worker_launcher = validate_worker_plan(successor, protected_paths, protected_dirs)
        artifact_hashes = validate_path_hash_map(plan.get('artifact_hashes'), 'rollover artifact pin', protected_paths,
                                                 protected_dirs)
        if artifact_hashes.get(str(script_path)) != digest(script_path):
            raise ValueError('rollover helper changed while waiting for predecessor')
        if artifact_hashes.get(str(worker_plan_path)) != digest(worker_plan_path):
            raise ValueError('successor worker plan changed while waiting for predecessor')
        if Path(successor['status_file']).exists() or Path(successor['journal']).exists():
            raise ValueError('Successor already has worker evidence')
        with authority_lock(authority_path):
            authority = json.loads(authority_path.read_text())
            current = authority['additional_teachers']['space_bunny']
            if (current['queue'], current['journal'], current.get('status_file')) != (
                    predecessor['queue'], predecessor['journal'], predecessor['status_file']):
                raise ValueError('Predecessor authority changed during review')
            latest_predecessor_review = verify_predecessor_terminal(predecessor, reviewed_failures)
            if latest_predecessor_review != predecessor_review:
                raise ValueError('Predecessor terminal accounting changed during review')
            atomic_json(record, dict(status='claimed', plan=str(args.plan.resolve()),
                                     plan_sha256=args.sha256,
                                     predecessor_review=predecessor_review))
            with Path(successor['launcher_log']).open('a') as log:
                command = [sys.executable, str(worker_launcher), str(worker_plan_path)]
                child = subprocess.Popen(command, cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            authority.setdefault('completed_additional_assignments', []).append(
                {**current, 'completion_state': predecessor_review['disposition'],
                 'reviewed_failed_keys': predecessor_review['reviewed_failed_keys'],
                 'reviewed_failed_event_sha256': predecessor_review['reviewed_failed_event_sha256'],
                 'reviewed_failed_events': predecessor_review['reviewed_failed_events'],
                 'successful_keys': predecessor_review['successful_keys'],
                 'expected_keys': predecessor_review['expected_keys'],
                 'completed_at': time.time()})
            authority['additional_teachers']['space_bunny'] = new_authority_binding(
                worker_plan, successor, child.pid, worker_plan_path)
            active_journals = authority.setdefault('active_journals', [])
            active_journals[:] = [path for path in active_journals if path != predecessor['journal']]
            new_journal = str(Path(successor['journal']).resolve(strict=False))
            if new_journal not in active_journals:
                active_journals.append(new_journal)
            atomic_json(authority_path, authority)
            atomic_json(record, dict(status='running', plan=str(args.plan.resolve()),
                                     plan_sha256=args.sha256, launcher_pid=child.pid,
                                     predecessor_review=predecessor_review))


if __name__ == '__main__':
    main()
