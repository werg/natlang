#!/usr/bin/env python3
"""Finish and verify initial pipeline replication, including after a reboot."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from sync_development_data import valid_initial_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root = Path(plan['local_root'])
    manifest = Path(plan['priority_manifest'])
    receipt = Path(plan['initial_verification_receipt'])
    if valid_initial_receipt(plan, receipt):
        print('Initial recipe input verification is already complete')
        return
    # A transfer already started by this session remains its sole writer.
    previous = plan.get('priority_transfer_service')
    while previous:
        state = subprocess.run(['systemctl', '--user', 'show', previous,
                                '--property=ActiveState', '--property=Result',
                                '--property=ExecMainStatus'],
                               capture_output=True, text=True, check=False)
        fields = dict(line.split('=', 1) for line in state.stdout.splitlines() if '=' in line)
        if fields.get('ActiveState') not in {'active', 'activating', 'deactivating'}:
            if fields.get('Result') not in {None, 'success'} or fields.get('ExecMainStatus') not in {None, '0'}:
                print('Previous transfer failed; retrying its staged files:', fields, flush=True)
            break
        time.sleep(15)
    guard = """import os,sys,json,shutil
from pathlib import Path
p=json.load(sys.stdin);s=Path(p['remote_storage']);external=Path('/mnt/external')
assert os.path.ismount(external)
assert s.resolve().is_relative_to(external.resolve())
assert s.stat().st_dev==external.stat().st_dev
assert shutil.disk_usage(s).free>=p.get('minimum_free_bytes',20*1024**3)
"""
    subprocess.run(['ssh', '-o', 'BatchMode=yes', plan['host'],
                    'python3 -c ' + shlex.quote(guard)],
                   input=json.dumps(plan), text=True, check=True)
    subprocess.run(['ssh', '-o', 'BatchMode=yes', plan['host'],
                    shlex.join(['python3', plan['remote_root'] + '/scripts/prepare_development_sync_partials.py',
                                '--root', plan['remote_storage']])], check=True)
    command = ['rsync', '-aH', '--relative', '--delay-updates', '--partial',
               '--partial-dir=.natlang-priority-partial', '--stats',
               f"--bwlimit={int(plan['bandwidth_kib_per_second'])}",
               '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=15',
               '--files-from=' + plan['priority_file_list'], str(root) + '/',
               plan['host'] + ':' + plan['remote_storage'] + '/']
    subprocess.run(command, check=True)
    subprocess.run(['/usr/bin/python3', str(root / 'scripts/verify_development_pipeline_sync.py'),
                    '--manifest', str(manifest), '--root', str(root),
                    '--remote-root', plan['remote_root'], '--remote-storage', plan['remote_storage'],
                    '--expected-manifest-sha256', plan['priority_manifest_sha256'],
                    '--host', plan['host'], '--output', str(receipt)], check=True)


if __name__ == '__main__':
    main()
