#!/usr/bin/env python3
"""Roll each of two reviewed Luna slots forward as its own queue finishes."""
import argparse
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from generation_authority import authority_lock
from start_reviewed_generation_successor import atomic_json, digest, running


def verify_finished(predecessor, reviewed_failures=None):
    if digest(predecessor['queue']) != predecessor['queue_sha256']:
        raise ValueError('Predecessor queue changed')
    keys = {json.loads(line)['key'] for line in Path(predecessor['queue']).read_text().splitlines() if line.strip()}
    finishes = {}
    journals = predecessor.get('journals') or [predecessor['journal']]
    for journal in journals:
        for line in Path(journal).read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                if event.get('event') == 'finish' and not event.get('batch_key'):
                    prior = finishes.get(event['key'])
                    if prior is None or event.get('time', 0) >= prior.get('time', 0):
                        finishes[event['key']] = event
    for key in keys:
        event = finishes.get(key, {})
        if event.get('status') in {'complete', 'complete_with_skips', 'skipped'} and event.get('output_accounting', {}).get('complete'):
            continue
        # A reviewed failed attempt stays excluded; it must never become a positive.
        expected = (reviewed_failures or {}).get(key)
        actual = hashlib.sha256(json.dumps(event, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if event and expected == actual:
            continue
        raise ValueError(f'Incomplete predecessor needs agent review: {key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.sha256:
        raise ValueError('Reviewed plan changed')
    plan = json.loads(args.plan.read_text())
    if (plan.get('root_approved') is not True or plan.get('provider') != 'openai-codex'
            or plan.get('model_id') != 'gpt-6-luna' or plan.get('model_concurrency') != 1
            or len(plan['workers']) != 2 or len(plan['predecessors']) != 2):
        raise ValueError('Requires two root-reviewed one-request Luna slots')
    record = Path(plan['launch_record'])
    authority_path = Path(plan['authority'])
    with record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists() or any(Path(w['journal']).exists() for w in plan['workers']):
            raise ValueError('Existing handoff evidence; refuse duplicate launch')
        launch = dict(status='waiting_for_slots', plan=str(args.plan), plan_sha256=args.sha256,
                      claimed_at=time.time(), workers=[])
        with record.open('x') as stream:
            json.dump(launch, stream)
            stream.flush()
            os.fsync(stream.fileno())
        started = set()
        while len(started) < 2:
            for index, (predecessor, worker) in enumerate(zip(plan['predecessors'], plan['workers'])):
                if index in started:
                    continue
                observed = json.loads(authority_path.read_text())
                bindings = [w for w in observed.get('luna_workers', [])
                            if w['queue'] == predecessor['queue'] and w['journal'] == predecessor['journal']]
                if len(bindings) != 1:
                    raise ValueError('Predecessor queue authority superseded')
                # A storage resume changes PID while preserving the reviewed queue.
                if running(bindings[0]['pid']):
                    continue
                verify_finished(predecessor, plan.get('reviewed_failed_finishes'))
                for file, expected in plan['artifact_hashes'].items():
                    if digest(file) != expected:
                        raise ValueError(f'Reviewed artifact/policy changed: {file}')
                if Path(worker['journal']).exists():
                    raise ValueError('Successor slot already has a journal')
                with authority_lock(authority_path):
                    authority = json.loads(authority_path.read_text())
                    old = authority.get('luna_workers', [])
                    matches = [i for i, w in enumerate(old) if w['queue'] == predecessor['queue'] and w['journal'] == predecessor['journal']]
                    if len(matches) != 1:
                        raise ValueError('Slot authority superseded; refuse extra worker')
                    if running(old[matches[0]]['pid']):
                        continue
                    command = ['python3', plan['supervisor'], worker['queue'], worker['journal'], '--runtime', plan['runtime'],
                               '--case-seconds', '1200', '--provider', 'openai-codex', '--model-id', 'gpt-6-luna',
                               '--model-concurrency', '1', '--execution-plans', '--reasoning-effort', 'low', '--min-free-mib', '1024']
                    Path(worker['log']).parent.mkdir(parents=True, exist_ok=True)
                    with Path(worker['log']).open('ab') as log:
                        process = subprocess.Popen(command, cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active = {**worker, 'pid': process.pid, 'command': command, 'status': 'running'}
                    launch['workers'].append(active)
                    started.add(index)
                    launch.update(status='running' if len(started) == 2 else 'partial_handoff', updated_at=time.time())
                    atomic_json(record, launch)
                    old[matches[0]] = active
                    authority.update(luna_workers=old, luna_runtime=plan['runtime'], luna_status=launch['status'])
                    authority.setdefault('active_journals', []).append(worker['journal'])
                    atomic_json(authority_path, authority)
                    state_path = Path(plan['luna_state'])
                    state = json.loads(state_path.read_text())
                    state.update(workers=old, runtime=plan['runtime'], status=launch['status'], latest_launch=str(record),
                                 active_campaigns=sorted({str(Path(w['queue']).parent) for w in old}))
                    atomic_json(state_path, state)
                    print(json.dumps(dict(event='slot_handoff', slot=index+1, worker=active)), flush=True)
            if len(started) < 2:
                time.sleep(30)


if __name__ == '__main__':
    main()
