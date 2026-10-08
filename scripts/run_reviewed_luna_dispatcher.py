#!/usr/bin/env python3
"""Dispatch a root-reviewed immutable Luna case list through the normal queue runner.

The dispatcher only schedules work. Each claimed case is written as an immutable
single-case queue and executed by run_bonsai_queue.py, so provider controls,
output accounting, and persistent per-slot backoff stay on the established path.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import time
from datetime import datetime, timezone

from generation_authority import authority_lock, reconcile_luna_authority
from freeze_training_runtime import tree_identity
from start_reviewed_generation_successor import atomic_json, digest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / 'scripts/run_bonsai_queue.py').resolve()
DISPATCHER = Path(__file__).resolve()
ALLOWED_REASONING = {'low', 'medium', 'high'}
_stop_requested = False


def _stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path):
    """Persist directory-entry changes made below this directory (Linux/Pop)."""
    fd = os.open(Path(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path):
    """Create missing directory components and persist each new entry."""
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)


def _append_event(path, event):
    path = Path(path)
    _mkdir_durable(path.parent)
    created = not path.exists()
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    if created:
        _fsync_directory(path.parent)


def _read_events(path):
    path = Path(path)
    if not path.exists():
        return []
    events = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f'Corrupt claim ledger at line {line_no}; preserve and review it') from error
        if not isinstance(row, dict):
            raise ValueError(f'Invalid claim ledger event at line {line_no}')
        events.append(row)
    return events


def _source_rows(source):
    rows = []
    for line_no, line in enumerate(Path(source).read_text(encoding='utf-8').splitlines(), 0):
        if line.strip():
            rows.append((line_no, json.loads(line)))
    return rows


def validate_plan(plan_path, expected_sha256):
    plan_path = Path(plan_path).resolve(strict=True)
    if digest(plan_path) != expected_sha256:
        raise ValueError('Reviewed dispatcher plan SHA-256 mismatch')
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('schema') != 'natlang.reviewed_luna_dispatch_plan/1':
        raise ValueError('Unsupported dispatcher plan schema')
    if plan.get('root_approved') is not True:
        raise ValueError('Dynamic dispatch requires explicit root approval')
    campaign_id = plan.get('campaign_id')
    if not isinstance(campaign_id, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{2,100}', campaign_id):
        raise ValueError('Invalid campaign_id')
    if plan.get('provider') != 'openai-codex' or plan.get('model_id') != 'gpt-6-luna':
        raise ValueError('This dispatcher is limited to reviewed Luna requests')
    if plan.get('model_concurrency') != 1:
        raise ValueError('Each Luna slot must remain limited to one provider request')
    slots = plan.get('slots')
    if type(slots) is not int or not 1 <= slots <= 5:
        raise ValueError('slots must be an integer from one through five')
    if plan.get('reasoning_effort', 'low') not in ALLOWED_REASONING:
        raise ValueError('Invalid reasoning effort')
    if plan.get('execution_plans') is not True:
        raise ValueError('Reviewed Luna dispatch requires execution plans')

    cwd = Path(plan.get('cwd', '')).expanduser().resolve(strict=True)
    if cwd != ROOT:
        raise ValueError('Dispatcher must use the canonical checkout')
    supervisor = Path(plan.get('supervisor', '')).expanduser().resolve(strict=True)
    if supervisor != RUNNER:
        raise ValueError('Dispatcher must reuse scripts/run_bonsai_queue.py')
    source = Path(plan.get('source', '')).expanduser().resolve(strict=True)
    runtime = Path(plan.get('runtime', '')).expanduser().resolve(strict=True)
    receipt = runtime / 'frozen-runtime.json'
    pins = plan.get('artifact_hashes')
    if not isinstance(pins, dict):
        raise ValueError('artifact_hashes must pin dispatcher, source, queue supervisor, and runtime receipt')
    required_pins = {str(DISPATCHER): plan.get('dispatcher_sha256'),
                     str(source): plan.get('source_sha256'),
                     str(supervisor): plan.get('supervisor_sha256'),
                     str(receipt): plan.get('runtime_receipt_sha256')}
    for path, expected in required_pins.items():
        if not isinstance(expected, str) or pins.get(path) != expected or digest(path) != expected:
            raise ValueError(f'Missing or changed reviewed artifact pin: {path}')
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError(f'Reviewed artifact changed: {path}')
    runtime_manifest = json.loads(receipt.read_text(encoding='utf-8'))
    if tree_identity(runtime) != runtime_manifest.get('files'):
        raise ValueError('Frozen runtime tree changed')

    cases = plan.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('Plan requires a nonempty immutable master case list')
    rows = dict(_source_rows(source))
    seen = set()
    normalized_cases = []
    for case in cases:
        if not isinstance(case, dict) or type(case.get('index')) is not int:
            raise ValueError('Every master case requires an integer source index')
        index = case['index']
        if index in seen or index not in rows:
            raise ValueError(f'Duplicate or out-of-range source index: {index}')
        seen.add(index)
        seed = case.get('seed')
        if type(seed) is not int or seed < 0:
            raise ValueError(f'Case {index} requires a nonnegative integer seed')
        normalized_cases.append({'index': index, 'seed': seed})

    for field in ('campaign_root', 'claim_ledger', 'launch_record', 'authority'):
        if not isinstance(plan.get(field), str) or not plan[field]:
            raise ValueError(f'Plan requires {field}')
    campaign_root = Path(plan['campaign_root']).expanduser().resolve()
    runs_root = (ROOT / 'runs').resolve()
    if runs_root not in campaign_root.parents:
        raise ValueError('campaign_root must be a fresh directory below the canonical runs/ tree')
    for field in ('claim_ledger', 'launch_record'):
        target = Path(plan[field]).expanduser().resolve()
        if campaign_root not in target.parents:
            raise ValueError(f'{field} must be inside campaign_root')
    authority = Path(plan['authority']).expanduser().resolve(strict=True)
    if plan.get('case_seconds', 1800) < 1 or type(plan.get('case_seconds', 1800)) is not int:
        raise ValueError('case_seconds must be a positive integer')
    if plan.get('min_free_mib', 1024) < 0 or type(plan.get('min_free_mib', 1024)) is not int:
        raise ValueError('min_free_mib must be a nonnegative integer')
    if plan.get('context_tokens', 32768) < 1 or type(plan.get('context_tokens', 32768)) is not int:
        raise ValueError('context_tokens must be a positive integer')
    if plan.get('max_turns', 60) < 1 or type(plan.get('max_turns', 60)) is not int:
        raise ValueError('max_turns must be a positive integer')
    if plan.get('max_model_requests', 384) < 1 or type(plan.get('max_model_requests', 384)) is not int:
        raise ValueError('max_model_requests must be a positive integer')
    transport_retries = plan.get('max_transport_retries')
    if type(transport_retries) is not int or transport_retries < 0:
        raise ValueError('max_transport_retries must be an explicit nonnegative integer')
    if type(plan.get('text_neuralese_emulation', False)) is not bool:
        raise ValueError('text_neuralese_emulation must be an explicit boolean')
    if 'provider_request_config' in plan:
        config = Path(plan['provider_request_config']).expanduser().resolve(strict=True)
        if pins.get(str(config)) != plan.get('provider_request_config_sha256'):
            raise ValueError('Provider request config must be pinned in the reviewed plan')
    return plan, {'plan_path': str(plan_path), 'plan_sha256': expected_sha256,
                  'campaign_id': campaign_id, 'source': str(source),
                  'dispatcher_sha256': required_pins[str(DISPATCHER)],
                  'source_sha256': required_pins[str(source)], 'runtime': str(runtime),
                  'runtime_receipt_sha256': required_pins[str(receipt)],
                  'supervisor': str(supervisor), 'supervisor_sha256': required_pins[str(supervisor)],
                  'campaign_root': str(campaign_root), 'ledger': str(Path(plan['claim_ledger']).resolve()),
                  'launch_record': str(Path(plan['launch_record']).resolve()),
                  'authority': str(authority), 'slots': slots,
                  'max_transport_retries': transport_retries,
                  'cases': normalized_cases}


def _verify_ledger_identity(events, identity):
    headers = [row for row in events if row.get('event') == 'campaign_open']
    if len(headers) > 1:
        raise ValueError('Claim ledger has multiple campaign headers')
    if headers and headers[0].get('identity') != identity:
        raise ValueError('Existing claim ledger belongs to another plan/source/runtime')
    if events and not headers:
        raise ValueError('Existing ledger lacks its identity header; review required')


def _verify_ledger_events(events, identity):
    """Fail closed on duplicate, unbound, or altered claim history."""
    expected_hash = identity['identity_sha256']
    approved = {case['index'] for case in identity['cases']}
    claims, claimed_indices, terminals, started = {}, set(), set(), set()
    for row in events:
        event = row.get('event')
        if event == 'campaign_open':
            continue
        if row.get('identity_sha256') != expected_hash:
            raise ValueError('Claim ledger event has a different campaign identity')
        if event == 'claim':
            index = row.get('index')
            claim_id = row.get('claim_id')
            expected_id = f"{identity['campaign_id']}-case-{index:06d}" if type(index) is int else None
            if (index not in approved or index in claimed_indices or claim_id != expected_id
                    or claim_id in claims or row.get('source') != identity['source']
                    or row.get('runtime') != identity['runtime']):
                raise ValueError('Duplicate or unbound source case in claim ledger')
            queue = Path(row.get('queue', '')).resolve()
            if Path(identity['campaign_root']) not in queue.parents:
                raise ValueError('Claim queue escaped campaign_root')
            slot = row.get('slot')
            if type(slot) is not int or not 1 <= slot <= identity['slots']:
                raise ValueError('Claim uses a slot outside the reviewed slot ceiling')
            queue_sha = row.get('queue_sha256')
            if not isinstance(queue_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', queue_sha):
                raise ValueError('Invalid claim queue SHA-256')
            claims[claim_id] = row
            claimed_indices.add(index)
        elif event == 'runner_started':
            claim_id = row.get('claim_id')
            if claim_id not in claims or claim_id in started or type(row.get('runner_pid')) is not int:
                raise ValueError('Unbound or duplicate runner_started event')
            started.add(claim_id)
        elif event in {'terminal', 'abandoned'}:
            claim_id = row.get('claim_id')
            claim = claims.get(claim_id)
            if (claim is None or claim_id in terminals or row.get('index') != claim.get('index')
                    or row.get('queue') != claim.get('queue')
                    or row.get('queue_sha256') != claim.get('queue_sha256')):
                raise ValueError('Unbound or duplicate terminal claim event')
            terminals.add(claim_id)
        elif event in {'storage_paused', 'storage_resumed'}:
            if (type(row.get('free_mib')) is not int or type(row.get('min_free_mib')) is not int
                    or row['free_mib'] < 0 or row['min_free_mib'] < 0):
                raise ValueError('Invalid storage pause/resume event')
        elif event not in {'campaign_stopped', 'campaign_finished', 'dispatcher_error'}:
            raise ValueError(f'Unknown claim ledger event type: {event!r}')
    open_slots = set()
    for claim_id, claim in claims.items():
        if claim_id in terminals:
            continue
        slot = claim['slot']
        if slot in open_slots:
            raise ValueError('More than one unresolved claim is recorded for a slot')
        open_slots.add(slot)


def _proc_argv(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    except (OSError, UnicodeError):
        return []


def _is_our_runner(pid, queue, journal, runtime, process_argv=_proc_argv):
    argv = process_argv(pid)
    def option(name):
        try:
            return argv[argv.index(name) + 1]
        except (ValueError, IndexError):
            return None
    try:
        executable = Path(argv[1]).resolve() if len(argv) >= 2 else None
    except (OSError, RuntimeError):
        executable = None
    return (len(argv) >= 4 and executable == RUNNER and
            str(Path(argv[2]).resolve()) == str(Path(queue).resolve()) and
            str(Path(argv[3]).resolve()) == str(Path(journal).resolve()) and
            option('--runtime') == str(Path(runtime).resolve()) and
            option('--provider') == 'openai-codex' and option('--model-id') == 'gpt-6-luna')


def _find_runner(queue, journal, runtime):
    matches = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit() and _is_our_runner(int(proc.name), queue, journal, runtime):
            matches.append(int(proc.name))
    if len(matches) > 1:
        raise ValueError(f'Multiple runners found for one claim queue: {queue}')
    return matches[0] if matches else None


class _AttachedRunner:
    """Observe a verified orphan from a previous controller without restarting it."""
    def __init__(self, pid, queue, journal, runtime):
        self.pid = pid
        self.queue = queue
        self.journal = journal
        self.runtime = runtime
        self.returncode = None
        self._termination_sent = False

    def poll(self):
        if _is_our_runner(self.pid, self.queue, self.journal, self.runtime):
            return None
        self.returncode = 0
        return self.returncode

    def terminate(self):
        if self._termination_sent or self.poll() is not None:
            return
        self._termination_sent = True
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            self.returncode = 0

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(f'pid {self.pid}', timeout)
            time.sleep(0.25)
        return self.returncode


def _journal_finish(journal, key):
    found = None
    if Path(journal).exists():
        for line in Path(journal).read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get('event') == 'finish' and row.get('key') == key and not row.get('batch_key'):
                    found = row
    return found


def _authority_update(identity, binding=None, *, preflight=False):
    path = Path(identity['authority'])
    with authority_lock(path):
        state = json.loads(path.read_text(encoding='utf-8'))
        workers = list(state.get('luna_workers', []))
        if binding is not None:
            workers = [row for row in workers if not (
                row.get('pid') == binding.get('pid') and
                row.get('queue') == binding.get('queue') and
                row.get('journal') == binding.get('journal'))]
            workers.append(binding)
        state['luna_workers'] = workers
        state['luna_runtime'] = identity['runtime']
        snapshot = reconcile_luna_authority(state)
        if preflight:
            live = state.get('additional_teachers', {}).get('luna', {}).get('workers', [])
            foreign = [worker for worker in live if worker.get('campaign') != identity['campaign_id']]
            if foreign:
                raise ValueError('Another verified Luna campaign is active; wait for reviewed handoff')
            if snapshot['actual_live_workers'] > identity['slots']:
                raise ValueError('Verified campaign workers exceed this plan slot ceiling')
        state['luna_active_campaigns'] = snapshot['active_campaigns']
        state['luna_status'] = 'running' if snapshot['actual_live_workers'] else 'idle'
        atomic_json(path, state)


def _authority_preflight(identity):
    """Do not let a new dynamic plan silently overlap another Luna campaign."""
    _authority_update(identity, preflight=True)


def _free_space_mib(path=ROOT):
    """Return free filesystem space for the dispatcher pre-claim guard."""
    return shutil.disk_usage(path).free // (1024 * 1024)


def _set_launch_status(path, identity, status, **extra):
    path = Path(path)
    record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'identity': identity}
    record['status'] = status
    record.update(extra)
    atomic_json(path, record)


def _claim_queue(plan, identity, slot, case):
    root = Path(identity['campaign_root'])
    index = case['index']
    slot_root = root / f'slot-{slot:02d}'
    claim_id = f"{identity['campaign_id']}-case-{index:06d}"
    queue = slot_root / 'claims' / claim_id / 'queue.jsonl'
    entry = {
        'key': claim_id, 'source': identity['source'], 'jobs': str(slot_root / 'jobs'),
        'output': str(slot_root / 'results' / f'case-{index:06d}.results.jsonl'),
        'index': index, 'count': 1, 'seed': case['seed'],
        'log': str(slot_root / 'collector-logs' / f'case-{index:06d}.log'),
        'max_turns': plan.get('max_turns', 60),
        'max_model_requests': plan.get('max_model_requests', 384),
        'context_tokens': plan.get('context_tokens', 32768),
        'case_seconds': plan.get('case_seconds', 1800),
        'transport_retries': identity['max_transport_retries'],
        'text_neuralese_emulation': bool(plan.get('text_neuralese_emulation', False)),
    }
    if plan.get('surface') is not None:
        entry['surface'] = plan['surface']
    payload = (json.dumps(entry, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8')
    return (entry, str(queue), hashlib.sha256(payload).hexdigest(),
            str(slot_root / 'journal.jsonl'), str(slot_root / 'supervisor.log'), payload)


def _write_claim_queue(queue, payload):
    queue = Path(queue)
    _mkdir_durable(queue.parent)
    with queue.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(queue.parent)


def _terminalize(claim, identity, ledger):
    finish = _journal_finish(claim['journal'], claim['key'])
    if finish is not None:
        event = {'event': 'terminal', 'claim_id': claim['claim_id'],
                 'index': claim['index'], 'slot': claim['slot'], 'terminal_at': _now(),
                 'disposition': 'runner_finished', 'runner_status': finish.get('status'),
                 'exit_code': finish.get('exit_code'),
                 'output_accounting': finish.get('output_accounting'),
                 'journal': claim['journal'], 'queue': claim['queue'],
                 'queue_sha256': claim['queue_sha256']}
    else:
        event = {'event': 'abandoned', 'claim_id': claim['claim_id'],
                 'index': claim['index'], 'slot': claim['slot'], 'terminal_at': _now(),
                 'disposition': 'attempted_or_uncertain; held from automatic retry',
                 'journal': claim['journal'], 'queue': claim['queue'],
                 'queue_sha256': claim['queue_sha256'],
                 'runner_pid': claim.get('runner_pid')}
    _append_event(ledger, {'identity_sha256': identity['identity_sha256'], **event})
    return event


def _reconcile_claims(events, identity, ledger):
    _verify_ledger_events(events, identity)
    terminals = {row['claim_id'] for row in events if row.get('event') in {'terminal', 'abandoned'}}
    claims = {}
    for row in events:
        if row.get('event') == 'claim':
            claims[row['claim_id']] = dict(row)
        elif row.get('event') == 'runner_started' and row.get('claim_id') in claims:
            claims[row['claim_id']]['runner_pid'] = row['runner_pid']
    active = {}
    for claim_id, claim in list(claims.items()):
        if claim_id in terminals:
            continue
        if Path(claim['queue']).exists() and digest(claim['queue']) != claim.get('queue_sha256'):
            raise ValueError(f'Claim queue changed after it was recorded: {claim["queue"]}')
        pid = claim.get('runner_pid')
        if pid is None or not _is_our_runner(pid, claim['queue'], claim['journal'], identity['runtime']):
            pid = _find_runner(claim['queue'], claim['journal'], identity['runtime'])
        if pid is not None and _is_our_runner(pid, claim['queue'], claim['journal'], identity['runtime']):
            claim['runner_pid'] = pid
            _authority_update(identity, {'pid': pid, 'queue': claim['queue'],
                              'journal': claim['journal'], 'runtime': identity['runtime'],
                              'campaign': identity['campaign_id'], 'source': identity['source'],
                              'source_index': claim['index'], 'status': 'running'})
            active[claim['slot']] = (_AttachedRunner(pid, claim['queue'], claim['journal'], identity['runtime']), claim)
            continue
        # A durable claim with no runner terminal is never silently returned to the pool.
        _terminalize(claim, identity, ledger)
        _authority_update(identity)
    return active, _read_events(ledger)


def _runner_command(plan, entry, queue, journal):
    command = ['python3', str(RUNNER), queue, journal, '--runtime', str(Path(plan['runtime']).resolve()),
               '--case-seconds', str(entry['case_seconds']), '--provider', 'openai-codex',
               '--model-id', 'gpt-6-luna', '--model-concurrency', '1', '--execution-plans',
               '--reasoning-effort', plan.get('reasoning_effort', 'low'),
               '--min-free-mib', str(plan.get('min_free_mib', 1024))]
    if plan.get('provider_request_config'):
        command += ['--provider-request-config', str(Path(plan['provider_request_config']).resolve())]
    return command


def run_dispatcher(plan_path, expected_sha256, *, resume=False):
    global _stop_requested
    _stop_requested = False
    plan, identity = validate_plan(plan_path, expected_sha256)
    identity['identity_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    root = Path(identity['campaign_root'])
    root.mkdir(parents=True, exist_ok=True)
    ledger = Path(identity['ledger'])
    launch_record = Path(identity['launch_record'])
    launch_record.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root / 'dispatcher.lock'
    with lock_path.open('a', encoding='utf-8') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        exists = ledger.exists() or launch_record.exists()
        if exists and not resume:
            raise ValueError('Campaign evidence exists; pass --resume after verifying the same plan')
        if resume and not ledger.exists():
            raise ValueError('--resume requires an existing claim ledger')
        events = _read_events(ledger) if ledger.exists() else []
        _verify_ledger_identity(events, identity)
        _verify_ledger_events(events, identity)
        if exists:
            prior_launch = json.loads(launch_record.read_text(encoding='utf-8')) if launch_record.exists() else {}
            if prior_launch.get('status') == 'finished':
                raise ValueError('Campaign is already finished; use a separately reviewed retry plan')
        _authority_preflight(identity)
        if not exists:
            occupied = [str(root / f'slot-{slot:02d}') for slot in range(1, identity['slots'] + 1)
                        if (root / f'slot-{slot:02d}').exists()]
            if occupied:
                raise ValueError(f'Fresh campaign slot directories already exist: {occupied}')
            _append_event(ledger, {'event': 'campaign_open', 'identity': identity, 'opened_at': _now()})
            atomic_json(launch_record, {'status': 'running', 'identity': identity, 'started_at': _now()})
        else:
            atomic_json(launch_record, {'status': 'resuming', 'identity': identity,
                                        'resumed_at': _now(), 'ledger': str(ledger)})
        active, events = _reconcile_claims(events, identity, ledger)
        claimed = {row['index'] for row in events if row.get('event') == 'claim'}
        claimed.update(row['index'] for row in events if row.get('event') in {'terminal', 'abandoned'})
        cases = [case for case in identity['cases'] if case['index'] not in claimed]
        next_case = 0
        all_children = list(active.values())
        termination_sent = set()
        storage_paused = False

        def terminate_once(process):
            pid = process.pid
            if pid not in termination_sent and process.poll() is None:
                termination_sent.add(pid)
                process.terminate()

        try:
            while next_case < len(cases) or active:
                if _stop_requested:
                    for process, claim in active.values():
                        terminate_once(process)
                for slot in range(1, identity['slots'] + 1):
                    if _stop_requested or next_case >= len(cases):
                        break
                    if slot in active:
                        continue
                    free_mib = _free_space_mib(ROOT)
                    minimum = plan.get('min_free_mib', 1024)
                    if free_mib < minimum:
                        if not storage_paused:
                            _append_event(ledger, {'event': 'storage_paused',
                                                   'identity_sha256': identity['identity_sha256'],
                                                   'free_mib': free_mib,
                                                   'min_free_mib': minimum,
                                                   'next_case_index': cases[next_case]['index'],
                                                   'paused_at': _now(),
                                                   'disposition': 'no case claimed; pending case list preserved'})
                            _set_launch_status(launch_record, identity, 'storage_paused',
                                               paused_at=_now(), free_mib=free_mib,
                                               min_free_mib=minimum,
                                               pending_indices=[row['index'] for row in cases[next_case:]])
                            storage_paused = True
                        break
                    if storage_paused:
                        _append_event(ledger, {'event': 'storage_resumed',
                                               'identity_sha256': identity['identity_sha256'],
                                               'free_mib': free_mib,
                                               'min_free_mib': minimum,
                                               'resumed_at': _now(),
                                               'disposition': 'pending case list retained'})
                        _set_launch_status(launch_record, identity, 'running', resumed_at=_now())
                        storage_paused = False
                    case = cases[next_case]
                    next_case += 1
                    entry, queue, queue_sha, journal, supervisor_log, queue_payload = _claim_queue(plan, identity, slot, case)
                    claim_id = entry['key']
                    claim = {'claim_id': claim_id, 'index': case['index'], 'slot': slot,
                             'key': entry['key'], 'queue': queue, 'queue_sha256': queue_sha,
                             'journal': journal, 'source': identity['source'],
                             'runtime': identity['runtime'], 'campaign': identity['campaign_id']}
                    _append_event(ledger, {'event': 'claim', 'identity_sha256': identity['identity_sha256'],
                                           **claim, 'claimed_at': _now()})
                    _write_claim_queue(queue, queue_payload)
                    command = _runner_command(plan, entry, queue, journal)
                    Path(supervisor_log).parent.mkdir(parents=True, exist_ok=True)
                    with Path(supervisor_log).open('ab') as output:
                        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                                   stdout=output, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                    claim['runner_pid'] = process.pid
                    # Own the child before any fallible ledger/authority writes. The
                    # exception path can now always stop and account for this runner.
                    active[slot] = (process, claim)
                    all_children.append((process, claim))
                    _append_event(ledger, {'event': 'runner_started', 'identity_sha256': identity['identity_sha256'],
                                           'claim_id': claim_id, 'runner_pid': process.pid,
                                           'command': command, 'started_at': _now()})
                    binding = {'pid': process.pid, 'queue': queue, 'journal': journal,
                               'runtime': identity['runtime'], 'campaign': identity['campaign_id'],
                               'source': identity['source'], 'source_index': case['index'],
                               'status': 'running'}
                    _authority_update(identity, binding)
                finished_slots = []
                for slot, (process, claim) in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        _terminalize(claim, identity, ledger)
                        _authority_update(identity)
                        finished_slots.append(slot)
                    elif _stop_requested:
                        # SIGTERM makes the normal queue runner preserve partials and stop its collector.
                        continue
                for slot in finished_slots:
                    del active[slot]
                if active:
                    time.sleep(1)
                elif next_case < len(cases) and not _stop_requested:
                    # No child can make progress while storage is low. Poll at a
                    # short interval so SIGINT/SIGTERM and freed space are noticed.
                    time.sleep(2)
                    continue
                elif _stop_requested:
                    break
            if _stop_requested:
                for process, claim in all_children:
                    terminate_once(process)
                for process, claim in all_children:
                    if process.poll() is None:
                        process.wait()
                    # Record claims whose supervisor exited without an ordinary finish.
                    if not any(row.get('claim_id') == claim['claim_id'] and row.get('event') in {'terminal', 'abandoned'}
                               for row in _read_events(ledger)):
                        _terminalize(claim, identity, ledger)
                _append_event(ledger, {'event': 'campaign_stopped', 'identity_sha256': identity['identity_sha256'],
                                       'stopped_at': _now(), 'disposition': 'current claims terminalized; unclaimed cases remain resumable'})
                atomic_json(launch_record, {'status': 'operator_stopped', 'identity': identity,
                                            'stopped_at': _now(), 'ledger': str(ledger)})
                _authority_update(identity)
                return 130
            events = _read_events(ledger)
            terminals = {row['index'] for row in events if row.get('event') in {'terminal', 'abandoned'}}
            pending = [case['index'] for case in identity['cases'] if case['index'] not in terminals]
            status = 'finished' if not pending else 'incomplete'
            _append_event(ledger, {'event': 'campaign_finished', 'identity_sha256': identity['identity_sha256'],
                                   'finished_at': _now(), 'status': status, 'pending_indices': pending})
            atomic_json(launch_record, {'status': status, 'identity': identity,
                                        'finished_at': _now(), 'pending_indices': pending,
                                        'ledger': str(ledger)})
            _authority_update(identity)
            return 0 if status == 'finished' else 1
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                _stop_requested = True
                for process, claim in active.values():
                    terminate_once(process)
                for process, claim in active.values():
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        # Keep durable claim and partial evidence; resume will reconcile it.
                        pass
                _append_event(ledger, {'event': 'campaign_stopped', 'identity_sha256': identity['identity_sha256'],
                                       'stopped_at': _now(), 'disposition': 'operator stop; claims remain durable'})
                atomic_json(launch_record, {'status': 'operator_stopped', 'identity': identity,
                                            'stopped_at': _now(), 'ledger': str(ledger)})
                return 130
            for process, claim in active.values():
                terminate_once(process)
            for process, claim in active.values():
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    # The durable claim remains held; resume reconciles the verified runner PID.
                    pass
            for process, claim in active.values():
                if process.poll() is not None and not any(
                        row.get('claim_id') == claim['claim_id'] and row.get('event') in {'terminal', 'abandoned'}
                        for row in _read_events(ledger)):
                    _terminalize(claim, identity, ledger)
            _authority_update(identity)
            _append_event(ledger, {'event': 'dispatcher_error', 'identity_sha256': identity['identity_sha256'],
                                   'at': _now(), 'error': f'{type(error).__name__}: {error}'})
            atomic_json(launch_record, {'status': 'dispatcher_error', 'identity': identity,
                                        'at': _now(), 'error': f'{type(error).__name__}: {error}',
                                        'ledger': str(ledger)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--resume', action='store_true', help='continue only unclaimed cases under the exact same plan identity')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return run_dispatcher(args.plan, args.sha256, resume=args.resume)


if __name__ == '__main__':
    raise SystemExit(main())
