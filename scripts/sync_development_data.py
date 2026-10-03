#!/usr/bin/env python3
"""Replicate pipeline artifacts with explicit ownership, without deleting data.

Home owns the corpus mirror. DGX development generation has its own namespace
which is pulled first and is excluded from the home-to-DGX transfer. Production
teacher campaigns still use sync_remote_teacher.py for verified imports.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    plan_path = args.plan.resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    if plan.get('version') != 'natlang.development_data_sync/1':
        raise ValueError('unsupported synchronization plan')
    root = Path(plan['local_root']).resolve(strict=True)
    if not (root / '.git').is_dir():
        raise ValueError('local root must be the development checkout')
    owned = set(plan['pull_roots'])
    for relative in [*plan['push_roots'], *owned]:
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError(f'unsafe synchronization root: {relative}')
    report_root = plan_path.parent
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    common = ['rsync', '-aH', '--relative', '--keep-dirlinks', '--delay-updates',
              '--partial', '--partial-dir=.natlang-sync-partial', '--stats',
              f"--bwlimit={int(plan['bandwidth_kib_per_second'])}",
              '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15']
    if args.dry_run:
        common.append('--dry-run')
    for pattern in plan['exclude_patterns']:
        common += ['--exclude', pattern]
    # Safety check on every pass; never silently move data onto the internal disk
    # if the external drive is absent or the checkout links have been replaced.
    guard = """import json,os,sys,shutil
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['remote_root']);s=Path(p['remote_storage'])
assert os.path.ismount('/mnt/external'), 'external storage is not mounted'
assert shutil.disk_usage(s).free >= p.get('minimum_free_bytes', 20*1024**3), 'external storage reserve reached'
for n in ('data','runs'):
 assert (r/n).is_symlink() and (r/n).resolve()==(s/n).resolve(), n
assert r.joinpath('.git').is_dir(), 'remote checkout missing'
for n in p['pull_roots']: r.joinpath(n).mkdir(parents=True,exist_ok=True)
"""
    with (report_root / 'development-sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            started = timestamp()
            status = {'version': plan['version'], 'started_at': started,
                      'status': 'running', 'dry_run': args.dry_run,
                      'ownership': plan['ownership'], 'transfers': []}
            output = report_root / 'sync-status.json'

            def save():
                temporary = output.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(status, indent=2) + '\n')
                os.replace(temporary, output)

            save()
            try:
                import shlex
                subprocess.run([*ssh, plan['host'], 'python3 -c ' + shlex.quote(guard)],
                               input=json.dumps(plan), text=True, check=True, timeout=30)
                for relative in sorted(owned):
                    (root / relative).mkdir(parents=True, exist_ok=True)
                    command = [*common, f"{plan['host']}:{plan['remote_root']}/./{relative}/", str(root) + '/']
                    with (report_root / 'pull.log').open('a') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    status['transfers'].append({'direction': 'dgx_to_home', 'root': relative, 'completed_at': timestamp()})
                    save()
                # Preserve destination revisions before replacing mirrored files.
                # The backup is outside the checkout and cannot enter a corpus.
                backup = plan['remote_storage'] + '/.sync-history/' + started.replace(':', '-')
                command = [*common, '--backup', '--backup-dir=' + backup]
                for relative in sorted(owned):
                    command += ['--exclude', '/' + relative + '/***']
                command += [str(root) + '/./' + relative + '/' for relative in plan['push_roots']]
                command += [f"{plan['host']}:{plan['remote_root']}/"]
                with (report_root / 'push.log').open('a') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                status.update(status='completed', completed_at=timestamp())
                status['transfers'].append({'direction': 'home_to_dgx', 'roots': plan['push_roots'], 'completed_at': timestamp()})
            except Exception as error:
                status.update(status='failed_retry_pending' if args.loop else 'failed',
                              error=str(error), completed_at=timestamp())
                if not args.loop:
                    save()
                    raise
            save()
            print(json.dumps(status), flush=True)
            if not args.loop:
                break
            time.sleep(int(plan['interval_seconds']))


if __name__ == '__main__':
    main()
