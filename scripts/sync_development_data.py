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
import re
import subprocess
import time


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


RSYNC_VANISHED_SUMMARY = re.compile(r'^rsync warning: some files vanished before they could be transferred \(code 24\)(?: at .*)?$')
RSYNC_VANISHED_PATH = re.compile(r'^file has vanished: "(.+)"$')


def classify_rsync_output(returncode, output):
    """Tolerate only code 24 whose every named vanished path is a live worker partial."""
    lines = output.splitlines()
    if returncode == 0:
        if any(RSYNC_VANISHED_PATH.match(line) or RSYNC_VANISHED_SUMMARY.match(line) for line in lines):
            raise ValueError('rsync reported vanished paths but returned success')
        return None
    if returncode != 24:
        raise subprocess.CalledProcessError(returncode, ['rsync'])
    paths = [match.group(1) for line in lines if (match := RSYNC_VANISHED_PATH.match(line))]
    summaries = [line for line in lines if RSYNC_VANISHED_SUMMARY.match(line)]
    other_warnings = [line for line in lines if 'warning:' in line.lower() and not RSYNC_VANISHED_SUMMARY.match(line)]
    other_errors = [line for line in lines if any(token in line.lower() for token in ('error', 'failed', 'permission denied', 'operation not permitted'))]
    if not paths or not summaries or other_warnings or other_errors:
        raise subprocess.CalledProcessError(returncode, ['rsync'])
    for path in paths:
        parts = Path(path).parts
        jobs = [index for index, part in enumerate(parts) if part == 'jobs']
        if not path.endswith('.partial.json') or not any(index > 0 and parts[index - 1].startswith('worker') for index in jobs):
            raise subprocess.CalledProcessError(returncode, ['rsync'])
    return {'warning': 'active worker partial files vanished while jobs completed',
            'vanished_paths_count': len(paths), 'vanished_paths_sample': paths[:50],
            'omitted_paths': max(0, len(paths) - 50)}


class RsyncFailure(RuntimeError):
    def __init__(self, returncode, log_path, start_byte, end_byte):
        super().__init__(f'rsync exited with code {returncode}; inspect {log_path} bytes {start_byte}:{end_byte}')
        self.returncode = returncode
        self.log_path = str(log_path)
        self.log_start_byte = start_byte
        self.log_end_byte = end_byte


def run_rsync(command, log_path, direction, root):
    """Append output to the durable sync log and return a narrow code-24 warning, if any."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a+b') as log:
        log.seek(0, os.SEEK_END)
        start_byte = log.tell()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.flush()
        end_byte = log.tell()
        log.seek(start_byte)
        output = log.read().decode('utf-8', errors='replace')
    if completed.returncode == 0:
        classify_rsync_output(0, output)
        return None
    try:
        warning = classify_rsync_output(completed.returncode, output)
    except (ValueError, subprocess.CalledProcessError):
        raise RsyncFailure(completed.returncode, log_path, start_byte, end_byte) from None
    return {**warning, 'direction': direction, 'root': root, 'exit_code': completed.returncode,
            'rsync_log': str(log_path), 'log_start_byte': start_byte, 'log_end_byte': end_byte}


def valid_initial_receipt(plan, path):
    import hashlib
    manifest_path = Path(plan['priority_manifest'])
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != plan['priority_manifest_sha256']:
        raise ValueError('priority manifest changed; refresh the reviewed sync plan')
    manifest = json.loads(manifest_path.read_text())
    expected = {item['path']: item['sha256'] for item in manifest['inputs']}
    if len(expected) != manifest['input_count'] or len(expected) != plan['priority_input_count']:
        raise ValueError('priority input count or unique identity mismatch')
    if not path.exists():
        return False
    saved = json.loads(path.read_text())
    actual = {item['path']: item['sha256'] for item in saved.get('checked', [])}
    return (saved.get('status') == 'passed' and not saved.get('errors')
            and saved.get('manifest_sha256') == digest
            and saved.get('host') == plan['host']
            and saved.get('remote_root') == plan['remote_root']
            and saved.get('remote_storage') == plan['remote_storage']
            and saved.get('input_count') == len(expected)
            and saved.get('source_hashes_verified') == len(expected)
            and len(saved.get('checked', [])) == len(expected)
            and actual == expected)


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
    if set(plan['push_roots']) != {'data', 'runs', 'vendor/datasets',
                                  'vendor/directory-sources',
                                  'vendor/directory-sources-expansion-20260929'}:
        raise ValueError('push roots differ from the reviewed pipeline scope')
    if owned != {'runs/dgx-development-generated'}:
        raise ValueError('pull roots differ from the DGX development namespace')
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
    # Public source fixtures and dependency lockfiles are pipeline inputs.
    # Permit only reviewed exceptions before the credential/coordination filters.
    reviewed_includes = {'/data/**/yarn.lock',
                         '/data/direct-code-2026-09-23/deno-std/dotenv/testdata/.env'}
    includes = plan.get('include_patterns', [])
    if not isinstance(includes, list) or not set(includes).issubset(reviewed_includes):
        raise ValueError('unreviewed synchronization filter exception')
    for pattern in includes:
        common += ['--include', pattern]
    for pattern in plan['exclude_patterns']:
        common += ['--exclude', pattern]
    # Safety check on every pass; never silently move data onto the internal disk
    # if the external drive is absent or the checkout links have been replaced.
    guard = """import json,os,sys,shutil
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['remote_root']);s=Path(p['remote_storage'])
assert os.path.ismount('/mnt/external'), 'external storage is not mounted'
assert s.resolve().is_relative_to(Path('/mnt/external').resolve()), 'storage path outside external drive'
assert s.stat().st_dev==Path('/mnt/external').stat().st_dev, 'storage on unexpected device'
assert shutil.disk_usage(s).free >= p.get('minimum_free_bytes', 20*1024**3), 'external storage reserve reached'
for n in ('data','runs'):
 assert (r/n).is_dir() and not (r/n).is_symlink(), 'Git parents must be real directories: '+n
assert r.joinpath('.git').is_dir(), 'remote checkout missing'
for n in p['pull_roots']: s.joinpath(n).mkdir(parents=True,exist_ok=True)
"""
    with (report_root / 'development-sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            initial_receipt = plan.get('initial_verification_receipt')
            if initial_receipt:
                receipt = Path(initial_receipt)
                if not valid_initial_receipt(plan, receipt):
                    waiting = {'status': 'waiting_for_priority_input_verification',
                               'receipt': str(receipt), 'time': timestamp(),
                               'scope': 'Initial copy is incomplete; broad mirror has not started.'}
                    temporary = report_root / 'sync-status.json.tmp'
                    temporary.write_text(json.dumps(waiting, indent=2) + '\n')
                    os.replace(temporary, report_root / 'sync-status.json')
                    print(json.dumps(waiting), flush=True)
                    if not args.loop:
                        raise ValueError('priority pipeline inputs have not been verified')
                    time.sleep(30)
                    continue
            started = timestamp()
            status = {'version': plan['version'], 'started_at': started,
                      'status': 'running', 'dry_run': args.dry_run,
                      'snapshot_status': 'in_progress',
                      'namespace_history_note': 'An initial bootstrap copy through the checkout runs path omitted an optimization directory before the external projection existed; a direct external-storage copy restored it. This recurring sync validates the external mount and projection before transferring.',
                      'ownership': plan['ownership'], 'transfers': [], 'warnings': []}
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
                from prepare_development_sync_partials import prepare
                for relative in sorted(owned):
                    pull_root = root / relative
                    for component in [pull_root, *pull_root.parents]:
                        if component == root:
                            break
                        if component.is_symlink():
                            raise ValueError(f'home pull path contains a symlink: {component}')
                    pull_root.mkdir(parents=True, exist_ok=True)
                    if not pull_root.resolve(strict=True).is_relative_to(root):
                        raise ValueError('home pull root escapes checkout')
                    prepare(pull_root)
                staging_command = shlex.join(['python3', plan['remote_root'] + '/scripts/prepare_development_sync_partials.py',
                                              '--root', plan['remote_storage']])
                subprocess.run([*ssh, plan['host'], staging_command], check=True)
                projection = 'python3 ' + shlex.quote(plan['remote_root'] + '/scripts/link_development_data.py')
                projection += ' --repo ' + shlex.quote(plan['remote_root']) + ' --storage ' + shlex.quote(plan['remote_storage'])
                subprocess.run([*ssh, plan['host'], projection], check=True, timeout=60)
                return_guard = """import json,sys
from pathlib import Path
p=json.load(sys.stdin)
for relative in p['pull_roots']:
 assert (Path(p['remote_root'])/relative).resolve()==(Path(p['remote_storage'])/relative).resolve(), 'DGX return namespace is not projected into the mirror: '+relative
"""
                subprocess.run([*ssh, plan['host'], 'python3 -c ' + shlex.quote(return_guard)],
                               input=json.dumps(plan), text=True, check=True, timeout=30)
                for relative in sorted(owned):
                    (root / relative).mkdir(parents=True, exist_ok=True)
                    backup = report_root / 'pull-revisions' / started.replace(':', '-')
                    command = [*common, '--backup', '--backup-dir=' + str(backup),
                               f"{plan['host']}:{plan['remote_storage']}/./{relative}/", str(root) + '/']
                    warning = run_rsync(command, report_root / 'pull.log', 'dgx_to_home', relative)
                    if warning: status['warnings'].append(warning)
                    status['transfers'].append({'direction': 'dgx_to_home', 'root': relative, 'completed_at': timestamp(),
                                                'status': 'completed_with_warnings' if warning else 'completed',
                                                'vanished_paths_count': warning['vanished_paths_count'] if warning else 0})
                    save()
                # Preserve destination revisions before replacing mirrored files.
                # The backup is outside the checkout and cannot enter a corpus.
                backup = plan['remote_storage'] + '/.sync-history/' + started.replace(':', '-')
                command = [*common, '--backup', '--backup-dir=' + backup]
                for relative in sorted(owned):
                    command += ['--exclude', '/' + relative + '/***']
                command += [str(root) + '/./' + relative + '/' for relative in plan['push_roots']]
                command += [f"{plan['host']}:{plan['remote_storage']}/"]
                warning = run_rsync(command, report_root / 'push.log', 'home_to_dgx', plan['push_roots'])
                if warning: status['warnings'].append(warning)
                subprocess.run([*ssh, plan['host'], projection], check=True, timeout=60)
                status.update(status='completed_with_warnings' if status['warnings'] else 'completed',
                              snapshot_status='completed_with_vanished_worker_partials' if status['warnings'] else 'rsync_pass_completed_live_tree_not_point_in_time',
                              completed_at=timestamp())
                status['transfers'].append({'direction': 'home_to_dgx', 'roots': plan['push_roots'], 'completed_at': timestamp(),
                                            'status': 'completed_with_warnings' if warning else 'completed',
                                            'vanished_paths_count': warning['vanished_paths_count'] if warning else 0})
            except Exception as error:
                status.update(status='failed_retry_pending' if args.loop else 'failed',
                              snapshot_status='failed', error=str(error), completed_at=timestamp())
                if isinstance(error, RsyncFailure):
                    status['rsync_failure'] = {'exit_code': error.returncode, 'log': error.log_path,
                                               'log_start_byte': error.log_start_byte, 'log_end_byte': error.log_end_byte}
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
