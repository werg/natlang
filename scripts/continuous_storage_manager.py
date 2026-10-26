#!/usr/bin/env python3
"""Poll checkpoint headroom and durable recovery bindings; default behavior is read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'training' / 'continuous-storage-policy.json'


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.pending-{os.getpid()}')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(path):
    return json.loads(Path(path).read_text())


def _stable_file_sha256(path):
    path = Path(path)
    before = path.stat()
    body = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'file changed during metadata binding: {path}')
    return hashlib.sha256(body).hexdigest()


def _append_journal(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(event, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _mount_uuid(path):
    result = subprocess.run(['findmnt', '-n', '-o', 'UUID', '--target', str(path)],
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        return None
    return result.stdout.strip() or None


def _owner_status(owner_path, command_fragment):
    owner = _json(owner_path)
    pid = owner.get('pid')
    if type(pid) is not int or pid <= 1:
        raise ValueError('owner record has no valid process id')
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace').strip()
    except OSError as error:
        return {'live': False, 'pid': pid, 'unit': owner.get('unit'), 'reason': str(error)}
    if command_fragment not in command:
        raise ValueError('owner pid exists but its command no longer matches the declared learner')
    return {'live': True, 'pid': pid, 'unit': owner.get('unit'), 'runtime': owner.get('runtime'),
            'scope': owner.get('scope'), 'command_sha256': hashlib.sha256(command.encode()).hexdigest()}


def _checkpoint_status(cursor_path, internal_path):
    cursor_file = Path(cursor_path)
    cursor_before = cursor_file.stat()
    cursor = _json(cursor_file)
    checkpoint = cursor.get('checkpoint') or {}
    raw_path = checkpoint.get('path')
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError('cycle cursor has no selected checkpoint path')
    path, internal = Path(raw_path), Path(internal_path).resolve()
    if not path.is_absolute() or not path.resolve().is_relative_to(internal):
        raise ValueError('selected checkpoint escapes the configured internal training root')
    stat_result = path.stat()
    if not path.is_file():
        raise ValueError('selected checkpoint is not a regular file')
    after_path = path.stat()
    cursor_after = cursor_file.stat()
    if ((stat_result.st_dev, stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns)
            != (after_path.st_dev, after_path.st_ino, after_path.st_size, after_path.st_mtime_ns)
            or (cursor_before.st_dev, cursor_before.st_ino, cursor_before.st_size, cursor_before.st_mtime_ns)
            != (cursor_after.st_dev, cursor_after.st_ino, cursor_after.st_size, cursor_after.st_mtime_ns)):
        raise ValueError('owner cursor or selected checkpoint changed during storage observation')
    return {'path': str(path), 'bytes': stat_result.st_size, 'device': stat_result.st_dev,
            'inode': stat_result.st_ino, 'mtime_ns': stat_result.st_mtime_ns,
            'cursor_checkpoint_sha256': checkpoint.get('sha256'), 'cycle': cursor.get('cycle'),
            'phase': cursor.get('phase'), 'stage_index': cursor.get('stage_index')}


def _archive_protected_paths(policy, checkpoint):
    root = Path(policy['archive_root']).resolve()
    protected = {str(Path(checkpoint['path']).parent)}
    active_cycle = next((parent for parent in Path(checkpoint['path']).parents
                         if parent.parent == root and parent.name.startswith('cycle-')), None)
    if active_cycle is not None:
        protected.add(str(active_cycle))
        try:
            previous = int(active_cycle.name.removeprefix('cycle-')) - 1
            if previous >= 0:
                protected.add(str(root / f'cycle-{previous:06d}'))
        except ValueError:
            pass
    try:
        config = _json(policy['config_path'])
        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    yield from walk(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    yield from walk(item)
            elif isinstance(value, str) and value.startswith(str(root) + '/'):
                candidate = Path(value)
                if candidate.exists():
                    yield str(candidate if candidate.is_dir() else candidate.parent)
        protected.update(walk(config))
    except (OSError, ValueError, TypeError):
        pass
    return sorted(protected)


def _mellum_status(policy):
    result = {'protected_paths': list(policy.get('mellum_protected_paths') or []),
              'automatic_pruning': False}
    for label, pointer_path in (('local', policy['mellum_local_pointer']),
                                ('private', policy['mellum_private_pointer'])):
        try:
            pointer = _json(pointer_path)
            checkpoint = pointer.get('checkpoint') or {}
            path, digest, size = checkpoint.get('path'), checkpoint.get('sha256'), checkpoint.get('bytes')
            entry = {'global_step': pointer.get('global_step'), 'path': path, 'sha256': digest,
                     'bytes': size, 'local_copy_pending': bool(pointer.get('local_copy_pending', False))}
            if label == 'local' and isinstance(path, str):
                local = Path(path)
                if local.exists():
                    st = local.stat()
                    entry.update(local_present=local.is_file() and st.st_size == size,
                                 local_inode=st.st_ino, local_device=st.st_dev)
                    result['protected_paths'].append(str(local))
                else:
                    entry['local_present'] = False
            elif label == 'private':
                receipt_path = pointer.get('receipt') or pointer.get('private_receipt')
                entry['private_receipt'] = receipt_path
                if isinstance(digest, str) and len(digest) == 64:
                    pin = Path('/mnt/external/natlang-active-lineage/mellum-resume-20261025') / f'mellum-{digest}.pt'
                    if pin.exists():
                        result['protected_paths'].append(str(pin))
                    entry['matching_local_pin_present'] = pin.is_file() and pin.stat().st_size == size
                if isinstance(receipt_path, str) and Path(receipt_path).is_file():
                    sys.path.insert(0, str(Path(__file__).resolve().parent / 'coreweave'))
                    from private_export_guard import validate_receipt
                    receipt = _json(receipt_path)
                    validate_receipt(receipt, {'sha256': digest, 'bytes': size})
                    entry['private_receipt_valid'] = True
                    result['protected_paths'].append(receipt_path)
                else:
                    entry['private_receipt_valid'] = False
            result[label] = entry
        except (OSError, ValueError, KeyError, TypeError) as error:
            result[label] = {'available': False, 'error': f'{type(error).__name__}: {str(error)[:500]}'}
    result['protected_paths'] = sorted(set(result['protected_paths']))
    inventory = []
    cache_root_value = policy.get('mellum_cache_root')
    cache_root = Path(cache_root_value) if isinstance(cache_root_value, str) and cache_root_value else None
    if cache_root is not None and cache_root.is_dir():
        for path in sorted(cache_root.glob('mellum-*.pt')):
            match = re.fullmatch(r'mellum-([0-9a-f]{64})\.pt', path.name)
            if not match or path.is_symlink() or not path.is_file():
                continue
            try:
                st = path.stat()
            except OSError:
                continue
            digest = match.group(1)
            receipt = Path(policy.get('mellum_receipt_root', '')) / f'{digest}.private-receipt.json'
            inventory.append({'path': str(path), 'sha256_in_name': digest, 'bytes': st.st_size,
                'device': st.st_dev, 'inode': st.st_ino, 'mtime_ns': st.st_mtime_ns,
                'protected_by_policy': str(path) in result['protected_paths'],
                'private_receipt_path': str(receipt), 'private_receipt_present': receipt.is_file(),
                'deletion_eligible': False,
                'eligibility_reason': 'local pin content must be independently reverified against exact receipt before pruning'})
    result['cache_inventory'] = inventory
    return result


def _archive_command(policy, budget_bytes, protected_paths=(), *, apply=False):
    batch_cap = int(policy.get('archive_batch_max_bytes', 60 * GIB))
    budget_gb = max(0.1, min(float(min(budget_bytes, batch_cap)) / GIB, 60.0))
    reserve = int(policy['external_floor_bytes']) + int(policy.get('external_archive_safety_bytes', 0))
    pending_path = Path(policy.get('mellum_private_pointer', ''))
    if pending_path.is_file():
        pending = _json(pending_path)
        if pending.get('local_copy_pending') is True:
            pending_checkpoint = pending.get('checkpoint') or {}
            pending_bytes = pending_checkpoint.get('bytes')
            if type(pending_bytes) is not int or pending_bytes <= 0:
                raise ValueError('pending Mellum local copy has no exact byte size')
            reserve += pending_bytes
    command = [policy.get('project_python', sys.executable), policy['archive_script'], '--root', policy['archive_root'],
               '--budget-gb', str(budget_gb), '--min-age-hours', '0', '--idle-run-min-age-hours', '0', '--hdd-floor-gb',
               str(reserve / GIB)]
    if policy.get('archive_destination'):
        command.extend(['--archive', policy['archive_destination']])
    for path in sorted(set(map(str, protected_paths))):
        command.extend(['--exclude', path])
    if apply:
        command.append('--apply')
    return command


def _archive_dry_run(policy, budget_bytes, protected_paths=()):
    command = _archive_command(policy, budget_bytes, protected_paths)
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    return {'command': command, 'returncode': result.returncode,
            'stdout': result.stdout[-20000:], 'stderr': result.stderr[-5000:]}


def _archive_apply(policy, budget_bytes, protected_paths, expected):
    """Apply only a dry-run batch after rechecking owner, controls, cursor and mount."""
    owner = _owner_status(policy['owner_path'], policy['owner_command_contains'])
    config_sha = _stable_file_sha256(policy['config_path'])
    checkpoint = _checkpoint_status(policy['cursor_path'], policy['internal_path'])
    observed = {'owner': {k: owner.get(k) for k in ('pid', 'unit', 'runtime', 'scope', 'command_sha256')},
                'config_sha256': config_sha,
                'checkpoint': {k: checkpoint.get(k) for k in (
                    'path', 'device', 'inode', 'bytes', 'mtime_ns', 'cursor_checkpoint_sha256',
                    'cycle', 'phase', 'stage_index')}}
    if not owner['live'] or observed != expected:
        return {'applied': False, 'skipped': 'owner, configuration, or selected checkpoint changed after dry-run',
                'observed': observed}
    mount = Path(policy['external_mount'])
    if _mount_uuid(mount) != policy['external_uuid']:
        return {'applied': False, 'skipped': 'external mount identity changed after dry-run'}
    usage = shutil.disk_usage(mount)
    floor = int(policy['external_floor_bytes'])
    if usage.free < floor:
        return {'applied': False, 'skipped': 'external free-space floor no longer passes',
                'external_free_bytes': usage.free, 'external_floor_bytes': floor}
    command = _archive_command(policy, budget_bytes, protected_paths, apply=True)
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    summary = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and 'moved_files' in candidate:
            summary = candidate
            break
    return {'executed': True, 'returncode': result.returncode,
            'moved_files': (summary or {}).get('moved_files', 0),
            'moved_bytes': int(float((summary or {}).get('moved_gb', 0)) * GIB),
            'summary': summary,
            'command': command, 'stdout': result.stdout[-20000:], 'stderr': result.stderr[-5000:]}


def _lfm_retention_reference(policy, selected_sha256, owner):
    """Bind an already prepared exact plan for review; never rescan or apply here."""
    destination = Path(policy['retention_plan_path'])
    try:
        before = destination.stat()
        body = destination.read_bytes()
        value = json.loads(body)
        after = destination.stat()
        signature = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if signature(before) != signature(after):
            raise ValueError('retention plan changed during read')
        recovery = value.get('recovery') or {}
        recovery_sha = ((recovery.get('checkpoint') or {}).get('sha256'))
        plan_owner = value.get('owner_identity') or {}
        return {'path': str(destination), 'plan_sha256': hashlib.sha256(body).hexdigest(),
                'candidate_count': len(value.get('candidates') or []),
                'candidate_bytes': sum(int(row.get('bytes', 0)) for row in value.get('candidates') or []),
                'plan_recovery_sha256': recovery_sha, 'current_cursor_sha256': selected_sha256,
                'plan_owner': plan_owner, 'current_owner': {k: owner.get(k) for k in ('pid', 'unit', 'runtime')},
                'current_binding_matches': (recovery_sha == selected_sha256
                    and plan_owner.get('pid') == owner.get('pid')
                    and plan_owner.get('unit') == owner.get('unit')),
                'review_required': True, 'applied': False}
    except (OSError, ValueError, TypeError, KeyError) as error:
        return {'path': str(destination), 'available': False,
                'error': f'{type(error).__name__}: {str(error)[:500]}', 'applied': False}


def inspect(policy):
    status = {'schema': 'natlang.continuous-storage-status/1', 'at': time.time(),
              'policy': str(policy.get('_path', '')), 'actions_applied': []}
    try:
        internal = Path(policy['internal_path'])
        usage = shutil.disk_usage(internal)
        status['internal'] = {'path': str(internal), 'total_bytes': usage.total,
                              'used_bytes': usage.used, 'free_bytes': usage.free,
                              'device': internal.stat().st_dev}
        status['owner'] = _owner_status(policy['owner_path'], policy['owner_command_contains'])
        if not status['owner']['live']:
            raise ValueError('declared training owner process is no longer live')
        status['config_sha256'] = _stable_file_sha256(policy['config_path'])
        status['checkpoint'] = _checkpoint_status(policy['cursor_path'], internal)
        needed = int(status['checkpoint']['bytes'] * float(policy['checkpoint_peak_factor'])) \
            + int(policy['internal_min_free_bytes'])
        status['internal'].update(next_write_required_headroom_bytes=needed,
                                  headroom_ok=usage.free >= needed,
                                  target_free_bytes=int(policy.get('internal_target_free_bytes', needed)),
                                  critical_free_bytes=int(policy.get('internal_critical_free_bytes', 0)),
                                  target_headroom_ok=usage.free >= int(policy.get('internal_target_free_bytes', needed)),
                                  critical=usage.free < int(policy.get('internal_critical_free_bytes', 0)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        status['internal_state_error'] = f'{type(error).__name__}: {str(error)[:500]}'

    mount = Path(policy['external_mount'])
    actual_uuid = _mount_uuid(mount)
    external_ok = mount.is_dir() and actual_uuid == policy['external_uuid']
    external = {'mount': str(mount), 'expected_uuid': policy['external_uuid'],
                'actual_uuid': actual_uuid, 'identity_ok': external_ok,
                'floor_bytes': int(policy['external_floor_bytes'])}
    if external_ok:
        external_usage = shutil.disk_usage(mount)
        external.update(free_bytes=external_usage.free, floor_ok=external_usage.free >= external['floor_bytes'])
    else:
        external.update(free_bytes=None, floor_ok=False)
    status['external'] = external
    status['mellum'] = _mellum_status(policy)

    internal = status.get('internal') or {}
    pressure = (not status.get('internal_state_error')
                and not internal.get('target_headroom_ok', internal.get('headroom_ok', False)))
    if pressure and external_ok and external.get('floor_ok'):
        free = internal.get('free_bytes')
        target = internal.get('target_free_bytes', internal.get('next_write_required_headroom_bytes'))
        protected = _archive_protected_paths(policy, status['checkpoint'])
        status['archive_protected_paths'] = protected
        plan = _archive_dry_run(policy, max(0, int(target or 0) - int(free or 0)), protected)
        status['archive_dry_run'] = plan
        status['action_required'] = ('Review bounded closed-file archive candidates; no file was moved.'
                                     if plan['returncode'] == 0 else
                                     'Archive dry-run failed; preserve current checkpoint and review the error.')
        if plan['returncode'] == 0 and policy.get('archive_apply_enabled') is True:
            expected = {
                'owner': {k: status['owner'].get(k) for k in
                          ('pid', 'unit', 'runtime', 'scope', 'command_sha256')},
                'config_sha256': status.get('config_sha256'),
                'checkpoint': {k: status['checkpoint'].get(k) for k in (
                    'path', 'device', 'inode', 'bytes', 'mtime_ns', 'cursor_checkpoint_sha256',
                    'cycle', 'phase', 'stage_index')},
            }
            applied = _archive_apply(policy, max(0, int(target or 0) - int(free or 0)), protected, expected)
            status['archive_apply'] = applied
            if applied.get('moved_files', 0):
                status['actions_applied'].append('verified-closed-file-archive')
            if applied.get('returncode') == 0:
                status['action_required'] = ('Archive batch completed; recheck headroom and recovery status.'
                                             if applied.get('moved_files', 0) else
                                             'Archive batch completed without moving candidates; inspect exact plan.')
            elif applied.get('executed'):
                status['action_required'] = 'Archive batch reported failures; inspect exact per-file results.'
            elif applied.get('skipped'):
                status['action_required'] = f"Archive apply skipped safely: {applied['skipped']}"
            else:
                status['action_required'] = 'Archive apply failed; inspect exact per-file results and preserve recovery.'
    elif pressure:
        status['action_required'] = 'Internal free space is below the64 GiB target and external archive is unavailable.'
    elif status.get('internal_state_error'):
        status['action_required'] = 'Owner or selected checkpoint could not be verified.'
    else:
        status['action_required'] = None
    if not external_ok:
        status['action_required'] = 'External mount identity is not verified; retain local recovery and investigate mount.'
    elif not external.get('floor_ok'):
        status['action_required'] = 'External free space is below the configured 80 GiB floor; do not archive or prune.'
    status['lfm_retention_plan'] = _lfm_retention_reference(
        policy, (status.get('checkpoint') or {}).get('cursor_checkpoint_sha256'), status.get('owner') or {})
    status['state'] = 'healthy' if status['action_required'] is None else 'attention'
    status['next_write_admission'] = 'atomic writer estimates the current full state and reserves transfer headroom'
    return status


def run_once(policy):
    status = inspect(policy)
    durable_json(policy['status_path'], status)
    _append_journal(policy['journal_path'], {key: status.get(key) for key in
        ('at', 'state', 'action_required')} | {'internal_free_bytes': (status.get('internal') or {}).get('free_bytes'),
        'external_identity_ok': (status.get('external') or {}).get('identity_ok'),
        'external_floor_ok': (status.get('external') or {}).get('floor_ok')})
    return status


def _self_test():
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        checkpoint = root / 'internal' / 'cycle' / 'checkpoint.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint')
        owner, cursor = root/'owner.json', root/'cursor.json'
        config = root/'config.json'
        config.write_text('{}')
        owner.write_text(json.dumps({'pid': os.getpid(), 'unit': 'test', 'runtime': '/runtime'}))
        cursor.write_text(json.dumps({'checkpoint': {'path': str(checkpoint), 'sha256': 'fixture'},
                                      'cycle': 1, 'phase': 'supervised', 'stage_index': 0}))
        external = root/'external'
        external.mkdir()
        policy = {'_path': str(root/'policy.json'), 'status_path': str(root/'status.json'),
            'journal_path': str(root/'events.jsonl'), 'internal_path': str(root/'internal'),
            'owner_path': str(owner), 'cursor_path': str(cursor),
            'config_path': str(config),
            'owner_command_contains': Path(sys.executable).name, 'internal_min_free_bytes': 1,
            'checkpoint_peak_factor': 2, 'external_mount': str(external), 'external_uuid': 'fixture',
            'external_floor_bytes': 1, 'archive_script': str(Path(__file__)),
            'archive_root': str(checkpoint.parent), 'mellum_local_pointer': str(root/'missing-local.json'),
            'mellum_private_pointer': str(root/'missing-private.json'), 'mellum_protected_paths': []}
        usage = type('Usage', (), {'total': 10**9, 'used': 1, 'free': 10**9})()
        with mock.patch(__name__+'._mount_uuid', return_value='fixture'), \
                mock.patch(__name__+'.shutil.disk_usage', return_value=usage):
            assert inspect(policy)['state'] == 'healthy'
            status = run_once(policy)
            assert json.loads(Path(policy['status_path']).read_text())['state'] == status['state']
        with mock.patch(__name__+'._mount_uuid', return_value='other'), \
                mock.patch(__name__+'.shutil.disk_usage', return_value=usage):
            assert inspect(policy)['external']['identity_ok'] is False
    print('self-test passed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=DEFAULT_POLICY)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true')
    mode.add_argument('--watch', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return
    policy = _json(args.policy)
    policy['_path'] = str(args.policy.resolve())
    if args.once:
        print(json.dumps(run_once(policy), sort_keys=True))
        return
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    while not stop:
        try:
            status = run_once(policy)
            print(json.dumps({'at': status['at'], 'state': status['state'],
                              'action_required': status['action_required']}, sort_keys=True), flush=True)
        except Exception as error:
            failure = {'schema': 'natlang.continuous-storage-status/1', 'at': time.time(),
                       'state': 'attention', 'action_required': f'{type(error).__name__}: {str(error)[:500]}'}
            durable_json(policy['status_path'], failure)
            _append_journal(policy['journal_path'], failure)
        deadline = time.monotonic() + int(policy.get('poll_seconds', 300))
        while not stop and time.monotonic() < deadline:
            time.sleep(min(1, deadline-time.monotonic()))


if __name__ == '__main__':
    main()
