#!/usr/bin/env python3
"""Run an immutable, reviewed fresh campaign with up to five Luna workers."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from start_reviewed_generation_successor import atomic_json, digest, running
from freeze_training_runtime import tree_identity


def worker_ordinal(worker):
    """Read a stable lane number from the worker record or its queue path."""
    value = worker.get('number')
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Worker number must be a positive integer')
        return value
    match = re.search(r'(?:^|/)worker-(\d+)(?:/|$)', str(worker.get('queue', '')))
    return int(match.group(1)) if match else None


def predecessor_lanes(workers, predecessors):
    """Pair predecessors to successors by explicit worker number or path lane."""
    if not predecessors:
        return {}
    if len(predecessors) != len(workers):
        raise ValueError('Predecessors must map one-to-one to successor worker lanes')
    worker_numbers = [worker_ordinal(worker) for worker in workers]
    predecessor_numbers = [worker_ordinal(worker) for worker in predecessors]
    if all(number is not None for number in worker_numbers + predecessor_numbers):
        if len(set(worker_numbers)) != len(worker_numbers) or len(set(predecessor_numbers)) != len(predecessor_numbers):
            raise ValueError('Ambiguous duplicate worker lane numbers')
        if set(worker_numbers) != set(predecessor_numbers):
            raise ValueError('Predecessor and successor worker lanes do not match')
        return {number: predecessor for number, predecessor in zip(predecessor_numbers, predecessors)}
    raise ValueError('Ambiguous predecessor pairing: each lane needs a worker number or worker-NN queue path')


def validate_predecessor_identity(predecessors):
    pids, queues, journals = set(), set(), set()
    for predecessor in predecessors:
        pid = predecessor.get('pid')
        if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1 or pid in pids:
            raise ValueError('Predecessor PIDs must be distinct positive integers')
        pids.add(pid)
        queue = str(Path(predecessor['queue']).resolve())
        journal = str(Path(predecessor['journal']).resolve())
        if queue == journal or queue in queues or journal in journals or queue in journals or journal in queues:
            raise ValueError('Predecessor queue and journal paths must be distinct one-to-one pairs')
        queues.add(queue)
        journals.add(journal)


def predecessor_completion(predecessor):
    """Check terminal scheduling accounting, independently of training success."""
    queue_path = Path(predecessor['queue'])
    journal_path = Path(predecessor['journal'])
    if digest(queue_path) != predecessor['queue_sha256']:
        raise ValueError(f'Predecessor queue changed: {queue_path}')
    queue = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
    keys = [entry.get('key') for entry in queue]
    if not keys or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        return False, 'queue keys are empty, missing, or duplicated'
    if not journal_path.is_file():
        return False, 'journal is missing'
    finishes = {}
    try:
        journal_lines = journal_path.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        return False, f'journal unreadable: {error}'
    for line in journal_lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            return False, f'journal contains malformed event: {error}'
        if not isinstance(event, dict):
            return False, 'journal event must be an object'
        if event.get('event') == 'finish' and not event.get('batch_key'):
            key = event.get('key')
            if key in finishes:
                return False, f'duplicate terminal journal event for {key}'
            finishes[key] = event
    if set(finishes) != set(keys):
        missing = sorted(set(keys) - set(finishes))
        extra = sorted(set(finishes) - set(keys))
        return False, f'journal terminal keys differ from queue (missing={missing}, extra={extra})'
    terminal_failures = []
    entries = {entry['key']: entry for entry in queue}
    for key in keys:
        event = finishes[key]
        if event.get('status') not in {'complete', 'complete_with_skips', 'skipped'}:
            accounting = event.get('output_accounting', {})
            if (event.get('status') != 'incomplete' or event.get('exit_code') != 2
                    or accounting.get('disposition') not in {'failed', 'transport_failed'}):
                return False, f'{key} terminal status is {event.get("status")!r}'
            # An exhausted request budget or transport error ends that attempt.
            # It must not starve unrelated work in the same lane. Recompute the
            # shared collector accounting from the exact files, rather than
            # accepting a journal assertion or manufacturing a result.
            runtime_path = predecessor.get('runtime')
            if not runtime_path:
                return False, f'{key} failed-attempt runtime is not declared'
            runtime = Path(runtime_path)
            frozen = json.loads((runtime / 'frozen-runtime.json').read_text())['files']
            if tree_identity(runtime) != frozen:
                raise ValueError(f'Failed predecessor runtime changed: {runtime}')
            from run_bonsai_queue import output_accounting
            actual = output_accounting(entries[key], runtime=runtime)
            states = actual.get('job_states', [])
            if (actual != accounting or not actual.get('manifest_valid')
                    or not states or len(states) != actual.get('expected_jobs')
                    or any(state.get('state') not in
                           {'result', 'generation_held', 'failed', 'transport_failed'}
                           for state in states)):
                return False, f'{key} failed-attempt evidence is not exactly terminal-accounted'
            terminal_failures.append({'key': key, 'disposition': actual['disposition'],
                                      'resource_limit_reason': actual.get('resource_limit_reason')})
            continue
        if not event.get('output_accounting', {}).get('complete'):
            return False, f'{key} output accounting is incomplete'
    return True, {'terminal_failed_attempts': terminal_failures} if terminal_failures else None


def verify_execution_pins(plan, runtime, plan_path, expected_plan_sha):
    if digest(plan_path) != expected_plan_sha:
        raise ValueError('Reviewed plan changed before worker launch')
    for path, expected in plan['artifact_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Reviewed artifact changed before worker launch: {path}')
    frozen = json.loads((runtime / 'frozen-runtime.json').read_text())['files']
    if tree_identity(runtime) != frozen:
        raise ValueError('Frozen runtime changed before worker launch')


def worker_command(plan, worker, runtime):
    """Build the exact, plan-pinned Luna queue command."""
    execution_plans = plan.get('execution_plans', True)
    if not isinstance(execution_plans, bool):
        raise ValueError('Reviewed execution_plans setting must be boolean')
    command = ['python3', plan['supervisor'], worker['queue'], worker['journal'],
               '--runtime', str(runtime), '--provider', 'openai-codex',
               '--model-id', 'gpt-6-luna', '--model-concurrency', '1']
    if execution_plans:
        command.append('--execution-plans')
    command.extend(['--reasoning-effort', plan.get('reasoning_effort', 'low'),
                    '--min-free-mib', '1024'])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.sha256:
        raise ValueError('Reviewed plan changed')
    plan = json.loads(args.plan.read_text())
    workers = plan['workers']
    reasoning_effort = plan.get('reasoning_effort', 'low')
    if reasoning_effort not in {'low', 'medium', 'high'}:
        raise ValueError('Invalid reviewed reasoning effort')
    if plan.get('root_approved') is not True or not 1 <= len(workers) <= 5:
        raise ValueError('Requires approval for one to five Luna workers')
    for path, expected in plan['artifact_hashes'].items():
        if digest(path) != expected:
            raise ValueError(f'Reviewed artifact changed: {path}')
    runtime = Path(plan['runtime'])
    if tree_identity(runtime) != json.loads((runtime / 'frozen-runtime.json').read_text())['files']:
        raise ValueError('Frozen runtime changed')
    keys, indices, evidence = set(), set(), set()
    pinned_paths = {str(Path(path).resolve()) for path in plan['artifact_hashes']}
    for worker in workers:
        if worker['queue'] not in plan['artifact_hashes']:
            raise ValueError('Worker queue is not pinned')
        for field in ('journal', 'log'):
            path = Path(worker[field]).resolve()
            if path in evidence or path.exists():
                raise ValueError('Fresh, distinct evidence paths required')
            evidence.add(path)
        for line in Path(worker['queue']).read_text().splitlines():
            entry = json.loads(line)
            source_path = Path(entry['source'])
            if not source_path.is_absolute():
                source_path = Path(plan['cwd']) / source_path
            source_path = str(source_path.resolve())
            identity = (source_path, entry['index'])
            if entry['key'] in keys or identity in indices or entry.get('count', 1) != 1:
                raise ValueError('Workers must have disjoint single-case queues')
            if source_path not in pinned_paths:
                raise ValueError('Queue source is not pinned')
            keys.add(entry['key'])
            indices.add(identity)
    record = Path(plan['launch_record'])
    predecessors = plan.get('predecessors', [])
    validate_predecessor_identity(predecessors)
    lanes = predecessor_lanes(workers, predecessors)
    # Reject queue drift anywhere in the approved predecessor set before any
    # lane is allowed to start, even if that predecessor is still running.
    for predecessor in predecessors:
        if digest(predecessor['queue']) != predecessor['queue_sha256']:
            raise ValueError(f'Predecessor queue changed: {predecessor["queue"]}')
    worker_lanes = {worker_ordinal(worker) or index + 1: worker for index, worker in enumerate(workers)}
    if len(worker_lanes) != len(workers):
        raise ValueError('Successor worker lanes must be uniquely numbered')
    def stop(signum, frame):
        raise KeyboardInterrupt('operator stopped campaign')
    signal.signal(signal.SIGTERM, stop)
    processes, launched = [], []
    lane_states = {
        lane: {'status': 'waiting_for_predecessor', 'predecessor_pid': predecessor['pid'],
               'queue': predecessor['queue'], 'journal': predecessor['journal']}
        for lane, predecessor in lanes.items()
    }
    for lane in worker_lanes:
        lane_states.setdefault(lane, {'status': 'ready'})
    waiting_record = record.with_suffix('.waiting.json')
    with record.open('x') as stream:
        json.dump({'status': 'waiting_for_predecessor_lanes' if lanes else 'starting',
                   'plan': str(args.plan), 'time': time.time(), 'lanes': lane_states}, stream)

    def persist_waiting():
        atomic_json(waiting_record, {'status': 'waiting_for_predecessor_lanes',
                                    'plan': str(args.plan), 'time': time.time(), 'lanes': lane_states,
                                    'launched': launched})
        atomic_json(record, {'status': 'waiting_for_predecessor_lanes', 'plan': str(args.plan),
                             'time': time.time(), 'lanes': lane_states, 'workers': launched})

    def launch_lane(lane, worker):
        # Pins and the exact lane's old queue are checked immediately before
        # every spawn, so a long wait on another lane cannot stale approval.
        try:
            verify_execution_pins(plan, runtime, args.plan, args.sha256)
            for pinned_predecessor in predecessors:
                if digest(pinned_predecessor['queue']) != pinned_predecessor['queue_sha256']:
                    raise ValueError(f'Predecessor queue changed before lane {lane}: '
                                     f'{pinned_predecessor["queue"]}')
        except Exception as error:
            # A late lane whose approval can no longer be verified must stay
            # unlaunched. Its sibling workers already have independent sealed
            # queues and should finish; pin drift is not a reason to kill them.
            lane_states[lane] = {
                'status': 'blocked_invalid_approval_pins',
                'reason': f'{type(error).__name__}: {error}',
            }
            return False
        predecessor = lanes.get(lane)
        if predecessor:
            if running(predecessor['pid']):
                lane_states[lane] = {'status': 'waiting_for_predecessor',
                                     'predecessor_pid': predecessor['pid']}
                return False
            complete, reason = predecessor_completion(predecessor)
            if not complete:
                lane_states[lane] = {'status': 'blocked_incomplete_predecessor',
                                     'predecessor_pid': predecessor['pid'], 'reason': reason,
                                     'queue': predecessor['queue'], 'journal': predecessor['journal']}
                return False
        command = worker_command(plan, worker, runtime)
        Path(worker['log']).parent.mkdir(parents=True, exist_ok=True)
        with Path(worker['log']).open('xb') as log:
            process = subprocess.Popen(command, cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        launched.append({**worker, 'lane': lane, 'pid': process.pid, 'command': command,
                         'runtime': str(runtime),
                         'runtime_manifest_sha256': digest(runtime / 'frozen-runtime.json'),
                         **({'predecessor_accounting': reason} if predecessor and reason else {})})
        lane_states[lane] = {'status': 'running', 'pid': process.pid,
                             'predecessor_pid': predecessor['pid'] if predecessor else None,
                             **({'predecessor_accounting': reason} if predecessor and reason else {})}
        atomic_json(record, {'status': 'running', 'workers': launched, 'lanes': lane_states,
                             'plan': str(args.plan), 'time': time.time()})
        return True

    try:
        pending = set(worker_lanes)
        while pending:
            made_progress = False
            for lane in sorted(pending):
                predecessor = lanes.get(lane)
                if predecessor and running(predecessor['pid']):
                    lane_states[lane] = {'status': 'waiting_for_predecessor',
                                         'predecessor_pid': predecessor['pid'],
                                         'queue': predecessor['queue'], 'journal': predecessor['journal']}
                    continue
                worker = worker_lanes[lane]
                if launch_lane(lane, worker):
                    pending.remove(lane)
                    made_progress = True
                elif lane_states[lane]['status'] in {
                        'blocked_incomplete_predecessor', 'blocked_invalid_approval_pins'}:
                    pending.remove(lane)
                    made_progress = True
            if pending:
                persist_waiting()
            else:
                waiting_record.unlink(missing_ok=True)
            if pending and not made_progress:
                time.sleep(5)

        waiting_record.unlink(missing_ok=True)
        # Wait for every started lane while keeping SIGTERM cleanup active.
        while any(process.poll() is None for process in processes):
            for process, worker in zip(processes, launched):
                code = process.poll()
                if code is not None:
                    lane_states[worker['lane']] = {'status': 'finished', 'pid': process.pid,
                                                   'exit_code': code}
            atomic_json(record, {'status': 'running', 'workers': launched, 'lanes': lane_states,
                                 'plan': str(args.plan), 'time': time.time()})
            time.sleep(1)
        codes = []
        for process, worker in zip(processes, launched):
            code = process.poll()
            if code is None:
                code = process.wait()
            lane_states[worker['lane']] = {'status': 'finished', 'pid': process.pid,
                                           'exit_code': code}
            codes.append(code)
        blocked_pins = [lane for lane, state in lane_states.items()
                        if state['status'] == 'blocked_invalid_approval_pins']
        blocked_predecessors = [lane for lane, state in lane_states.items()
                                if state['status'] == 'blocked_incomplete_predecessor']
        status = ('finished_with_blocked_approval_pins' if blocked_pins else
                  'finished_with_blocked_predecessor_lanes' if blocked_predecessors else
                  'finished' if not any(codes) else 'finished_with_incomplete_work')
        atomic_json(record, {'status': status, 'workers': launched, 'lanes': lane_states,
                             'blocked_approval_pin_lanes': blocked_pins,
                             'blocked_predecessor_lanes': blocked_predecessors,
                             'exit_codes': codes, 'plan': str(args.plan), 'time': time.time()})
        return int(bool(blocked_pins or blocked_predecessors) or any(codes))
    except BaseException as error:
        for process in processes:
            try:
                # Signal the whole isolated group even if the supervisor has
                # already exited; a child may still be cleaning up.
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        remaining_groups = {process.pid for process in processes}
        while remaining_groups and time.monotonic() < deadline:
            for pgid in list(remaining_groups):
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    remaining_groups.remove(pgid)
                except PermissionError:
                    pass
            if remaining_groups:
                time.sleep(0.1)
        for pgid in remaining_groups:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait()
            except ChildProcessError:
                pass
        interrupted = isinstance(error, KeyboardInterrupt)
        waiting_record.unlink(missing_ok=True)
        atomic_json(record, {'status': 'operator_stopped' if interrupted else 'launch_failed',
                             'workers': launched, 'lanes': lane_states,
                             'plan': str(args.plan), 'time': time.time(),
                             'disposition': 'Unfinished attempts remain unscored; preserve partial evidence.'})
        if interrupted:
            return 130
        raise


if __name__ == '__main__':
    raise SystemExit(main())
