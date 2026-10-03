#!/usr/bin/env python3
"""Pull a bounded remote teacher assignment without losing raw checkpoints or evidence."""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
from generation_authority import authority_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--teacher-key', default='dgx_horizon',
                        help='Authority entry for this teacher (default preserves Horizon assignments)')
    args = parser.parse_args()
    root = args.campaign.resolve()
    assignment_path = root / 'assignment.json'
    expected_assignment_sha = None
    if args.authority:
        with authority_lock(args.authority):
            authority = json.loads(args.authority.read_text())
            teacher = authority.get('additional_teachers', {}).get(args.teacher_key, {})
            if teacher.get('campaign') != str(root):
                raise ValueError('Authority does not bind this campaign')
            assignment_path = Path(teacher['assignment']).resolve()
            expected_assignment_sha = teacher['assignment_sha256']
    assignment_bytes = assignment_path.read_bytes()
    assignment_sha = hashlib.sha256(assignment_bytes).hexdigest()
    if expected_assignment_sha and assignment_sha != expected_assignment_sha:
        raise ValueError('Authority assignment hash mismatch')
    assignment = json.loads(assignment_bytes)
    staging = root / 'runtime-import-staging'  # excluded by the automatic generated-snapshot walker
    staging.mkdir(exist_ok=True)
    with (root / 'sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if hashlib.sha256(assignment_path.read_bytes()).hexdigest() != assignment_sha:
                raise ValueError('Assignment changed while synchronizing')
            prefix = assignment['host'] + ':' + assignment['remote_directory'] + '/'
            command = ['rsync', '-a', '--checksum', '--timeout=120', '--ignore-missing-args',
                       '--exclude=*.tmp', '--exclude=*.tmp-*', '--include=jobs/***',
                       '--include=exports/***', '--include=journal*.jsonl', '--include=worker-status.json',
                       '--include=bootstrap.log', '--include=supervisor.log', '--include=server-launch.log',
                       '--exclude=*', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15', prefix, str(staging) + '/']
            # Bound stalled transport, not total copy duration. Large raw
            # evidence can take longer than two minutes while making progress.
            pulled = subprocess.run(command, capture_output=True, text=True)
            if pulled.returncode == 0:
                imported = subprocess.run(['node', 'scripts/import_remote_teacher_results.mjs', str(root), str(assignment_path)],
                                          capture_output=True, text=True, timeout=120)
                report = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'ssh_host': assignment['host'], 'transfer_exit_code': 0,
                          'import_exit_code': imported.returncode}
                if imported.returncode == 0:
                    report['import'] = json.loads(imported.stdout)
                else:
                    report['import_error'] = imported.stderr[-2000:]
            else:
                report = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          'ssh_host': assignment['host'], 'transfer_exit_code': pulled.returncode,
                          'transfer_failure_kind': 'source_vanished_during_live_copy' if pulled.returncode == 24 else 'transport_or_copy_failure',
                          'transfer_error': pulled.stderr[-2000:],
                          'disposition': ('live_source_changed_retry_pending; assignment remains remote-owned'
                                          if pulled.returncode == 24 else
                                          'remote_unavailable_or_transfer_failed; assignment remains remote-owned')}
            try:
                state = json.loads((staging / 'worker-status.json').read_text())['state']
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                state = 'unknown'
            transfer_and_import_succeeded = (report.get('transfer_exit_code') == 0
                                             and report.get('import_exit_code') == 0)
            exact_finished_coverage = (report.get('import', {}).get('total_unique_artifacts')
                                       == assignment.get('cases'))
            terminal_import_ready = (transfer_and_import_succeeded and
                                    (state == 'stopped' or (state == 'finished' and exact_finished_coverage)))
            report.update(remote_worker_state=state, terminal_import_ready=terminal_import_ready)
            tmp = root / 'sync-status.json.tmp'
            tmp.write_text(json.dumps(report, indent=2) + '\n')
            tmp.replace(root / 'sync-status.json')
            print(json.dumps(report), flush=True)
            if not args.loop:
                break
            if args.authority and args.authority.exists():
                with authority_lock(args.authority):
                    authority = json.loads(args.authority.read_text())
                    teacher = authority.get('additional_teachers', {}).get(args.teacher_key)
                    if (teacher and teacher.get('assignment') == str(assignment_path)
                            and teacher.get('assignment_sha256') == assignment_sha):
                        teacher['state'] = state
                        teacher['last_sync'] = report
                        for journal in staging.glob('journal*.jsonl'):
                            if str(journal) not in authority.setdefault('active_journals', []):
                                authority['active_journals'].append(str(journal))
                        tmp = args.authority.with_suffix('.sync.tmp')
                        tmp.write_text(json.dumps(authority, indent=2) + '\n')
                        tmp.replace(args.authority)
            # A paused deployment can recover independently. Keep pulling its status
            # and evidence so the local monitor does not retain a stale failure.
            if terminal_import_ready:
                break
            time.sleep(45)

if __name__ == '__main__':
    main()
