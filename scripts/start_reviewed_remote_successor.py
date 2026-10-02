#!/usr/bin/env python3
"""Start a pinned remote successor only after complete predecessor exports."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from generation_authority import authority_lock
from start_reviewed_generation_successor import atomic_json, digest
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
    authority_path = Path(plan['authority'])
    record = Path(plan['launch_record'])
    with record.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.exists():
            raise ValueError('Successor already claimed')
        atomic_json(record, dict(status='waiting_for_predecessor', plan=str(args.plan),
                                 plan_sha256=args.sha256))
        while True:
            authority = json.loads(authority_path.read_text())
            current = authority['additional_teachers'][plan['teacher_key']]
            if current['assignment'] != plan['predecessor_assignment']:
                raise ValueError('Predecessor authority superseded')
            state = json.loads(Path(plan['predecessor_status']).read_text())['state']
            if state == 'finished':
                break
            if state.startswith('paused') or state == 'stopped':
                raise ValueError('Predecessor needs review')
            time.sleep(30)
        try:
            for predecessor in plan['predecessors']:
                verify_finished(predecessor, plan.get('reviewed_failed_finishes'))
        except ValueError as error:
            atomic_json(record, dict(status='paused_predecessor_review_required', plan=str(args.plan),
                                     plan_sha256=args.sha256, reason=str(error)))
            raise
        imported = json.loads(Path(plan['predecessor_import_status']).read_text())
        ledger = [json.loads(line) for line in Path(plan['predecessor_ledger']).read_text().splitlines() if line]
        expected_exports = plan.get('predecessor_expected_exports', plan['predecessor_cases'])
        if (imported['total_unique_artifacts'] != expected_exports
                or len({row['program_id'] for row in ledger}) != expected_exports
                or any(row['disposition'] == 'assignment_held' for row in ledger)):
            raise ValueError('Incomplete or unassigned predecessor evidence')
        if expected_exports != plan['predecessor_cases']:
            assigned = {json.loads(line)['id'] for line in Path(plan['predecessor_ir']).read_text().splitlines() if line}
            exported = {row['program_id'] for row in ledger}
            reviewed_missing = set(plan['reviewed_missing_program_ids'])
            if (len(assigned) != plan['predecessor_cases'] or not exported <= assigned
                    or assigned - exported != reviewed_missing
                    or expected_exports + len(reviewed_missing) != plan['predecessor_cases']):
                raise ValueError('Missing predecessor cases lack exact root accounting')
        for file, expected in plan['artifact_hashes'].items():
            if digest(file) != expected:
                raise ValueError('Reviewed artifact changed: ' + file)
        successor = plan['successor']
        campaign = Path(successor['assignment']).parent
        if (campaign / 'sync-status.json').exists():
            raise ValueError('Successor already has sync evidence')
        # The old service must have exited; the model server stays running.
        inactive = subprocess.run(plan['predecessor_inactive_command'], capture_output=True, text=True)
        # systemd discards completed transient units. is-active returns 4 for
        # an absent unit, versus 3 for a retained inactive unit. Complete
        # journals and imports were verified above; neither state has a worker.
        if inactive.returncode not in (3, 4):
            raise ValueError('Predecessor service is not inactive')
        with authority_lock(authority_path):
            authority = json.loads(authority_path.read_text())
            current = authority['additional_teachers'][plan['teacher_key']]
            if current['assignment'] != plan['predecessor_assignment']:
                raise ValueError('Predecessor authority changed during review')
            atomic_json(record, dict(status='claimed', plan=str(args.plan), plan_sha256=args.sha256))
            if plan.get('retire_command'):
                retired = subprocess.run(plan['retire_command'], capture_output=True, text=True)
                if retired.returncode:
                    raise ValueError('Predecessor service retirement failed')
            started = subprocess.run(plan['start_command'], capture_output=True, text=True)
            if started.returncode:
                raise ValueError('Remote service launch failed')
            with (campaign / 'sync.log').open('ab') as log:
                child = subprocess.Popen(plan['sync_command'], cwd=plan['cwd'], stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            authority.setdefault('completed_additional_assignments', []).append(dict(current, completion_state='finished'))
            authority['additional_teachers'][plan['teacher_key']] = dict(successor, state='starting', sync_pid=child.pid)
            atomic_json(authority_path, authority)
            atomic_json(record, dict(status='running', plan=str(args.plan), plan_sha256=args.sha256, sync_pid=child.pid))


if __name__ == '__main__':
    main()
