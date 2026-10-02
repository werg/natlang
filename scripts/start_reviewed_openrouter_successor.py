#!/usr/bin/env python3
"""Hand one completed OpenRouter assignment to a separately reviewed successor."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from generation_authority import authority_lock
from start_reviewed_generation_successor import atomic_json, digest, running
from start_reviewed_luna_slots import verify_finished


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
    record = Path(plan['launch_record'])
    authority_path = Path(plan['authority'])
    with record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists():
            raise ValueError('Successor has already been claimed')
        atomic_json(record, dict(status='waiting_for_predecessor', plan=str(args.plan.resolve()),
                                 plan_sha256=args.sha256))
        while True:
            current = json.loads(authority_path.read_text())['additional_teachers']['space_bunny']
            predecessor = plan['predecessor']
            if (current['queue'], current['journal']) != (predecessor['queue'], predecessor['journal']):
                raise ValueError('Predecessor authority superseded')
            state = json.loads(Path(current['status_file']).read_text())
            if state['state'] == 'finished' and not running(state.get('launcher_pid')):
                break
            if state['state'].startswith('paused') or state['state'] == 'stopped':
                raise ValueError('Predecessor needs review before rollover')
            time.sleep(30)
        verify_finished(predecessor)
        for file, expected in plan['artifact_hashes'].items():
            if digest(file) != expected:
                raise ValueError('Reviewed artifact changed: ' + file)
        successor = plan['successor']
        if Path(successor['status_file']).exists() or Path(successor['journal']).exists():
            raise ValueError('Successor already has worker evidence')
        with authority_lock(authority_path):
            authority = json.loads(authority_path.read_text())
            current = authority['additional_teachers']['space_bunny']
            if (current['queue'], current['journal']) != (predecessor['queue'], predecessor['journal']):
                raise ValueError('Predecessor authority changed during review')
            atomic_json(record, dict(status='claimed', plan=str(args.plan.resolve()),
                                     plan_sha256=args.sha256))
            with Path(successor['launcher_log']).open('a') as log:
                command = ['python3', plan['launcher'], successor['plan']]
                child = subprocess.Popen(command, cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            authority.setdefault('completed_additional_assignments', []).append(
                dict(current, completion_state='finished'))
            authority['additional_teachers']['space_bunny'] = {
                **current, **successor, 'launcher_pid': child.pid, 'supervisor_pid': None,
                'state': 'starting'}
            authority.setdefault('active_journals', []).append(successor['journal'])
            atomic_json(authority_path, authority)
            atomic_json(record, dict(status='running', plan=str(args.plan.resolve()),
                                     plan_sha256=args.sha256, launcher_pid=child.pid))


if __name__ == '__main__':
    main()
