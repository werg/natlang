#!/usr/bin/env python3
"""Pull a bounded remote teacher assignment without losing raw checkpoints or evidence."""
import argparse
import datetime
import fcntl
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
    args = parser.parse_args()
    root = args.campaign.resolve()
    assignment = json.loads((root / 'assignment.json').read_text())
    staging = root / 'runtime-import-staging'  # excluded by the automatic generated-snapshot walker
    staging.mkdir(exist_ok=True)
    with (root / 'sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            prefix = assignment['host'] + ':' + assignment['remote_directory'] + '/'
            command = ['rsync', '-a', '--checksum', '--ignore-missing-args', '--include=jobs/***',
                       '--include=exports/***', '--include=journal*.jsonl', '--include=worker-status.json',
                       '--include=bootstrap.log', '--include=supervisor.log', '--include=server-launch.log',
                       '--exclude=*', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15', prefix, str(staging) + '/']
            try:
                pulled = subprocess.run(command, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                pulled = subprocess.CompletedProcess(command, 124, '', '')
            if pulled.returncode == 0:
                imported = subprocess.run(['node', 'scripts/import_remote_teacher_results.mjs', str(root)],
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
                          'disposition': 'remote_unavailable_or_transfer_failed; assignment remains remote-owned'}
            tmp = root / 'sync-status.json.tmp'
            tmp.write_text(json.dumps(report, indent=2) + '\n')
            tmp.replace(root / 'sync-status.json')
            print(json.dumps(report), flush=True)
            if not args.loop:
                break
            try:
                state = json.loads((staging / 'worker-status.json').read_text())['state']
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                state = 'unknown'
            if args.authority and args.authority.exists():
                with authority_lock(args.authority):
                    authority = json.loads(args.authority.read_text())
                    teacher = authority.get('additional_teachers', {}).get('dgx_horizon')
                    if teacher and teacher.get('assignment') == str(root / 'assignment.json'):
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
            if state in {'finished', 'stopped'}:
                break
            time.sleep(45)

if __name__ == '__main__':
    main()
