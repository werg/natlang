#!/usr/bin/env python3
"""Poll checkpoint headroom and durable recovery bindings; default behavior is read-only."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

GIB = 1024 ** 3
DEFAULT_POLICY = Path(__file__).resolve().parents[1] / 'training' / 'continuous-storage-policy.json'
_MELLUM_METADATA_READERS = {}


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
                    pin = Path(policy['mellum_cache_root']) / f'mellum-{digest}.pt'
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


def _stat_binding(path):
    st = Path(path).stat(follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f'cache candidate is not a regular file: {path}')
    return {'device': st.st_dev, 'inode': st.st_ino, 'bytes': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns,
            'links': st.st_nlink, 'allocated_bytes': st.st_blocks * 512}


def _hash_mellum_pin(path, expected_stat, *, block_bytes=8 * 1024 * 1024):
    """Stream an external full-state pin with bounded memory and drop consumed cache pages."""
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        binding = {'device': before.st_dev, 'inode': before.st_ino, 'bytes': before.st_size,
                   'mtime_ns': before.st_mtime_ns, 'ctime_ns': before.st_ctime_ns,
                   'links': before.st_nlink, 'allocated_bytes': before.st_blocks * 512}
        if binding != expected_stat:
            raise ValueError(f'cache candidate changed before hashing: {path}')
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.read(fd, block_bytes)
            if not block:
                break
            digest.update(block)
            offset += len(block)
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, offset - len(block), len(block), os.POSIX_FADV_DONTNEED)
        after = os.fstat(fd)
        final = {'device': after.st_dev, 'inode': after.st_ino, 'bytes': after.st_size,
                 'mtime_ns': after.st_mtime_ns, 'ctime_ns': after.st_ctime_ns,
                 'links': after.st_nlink, 'allocated_bytes': after.st_blocks * 512}
        if final != expected_stat or offset != expected_stat['bytes']:
            raise ValueError(f'cache candidate changed while hashing: {path}')
        return digest.hexdigest()
    finally:
        os.close(fd)


def _mellum_receipt(path, sha256, size):
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'coreweave'))
    from private_export_guard import validate_receipt
    receipt_path = Path(path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError('private receipt is not a regular file')
    receipt = _json(receipt_path)
    validate_receipt(receipt, {'sha256': sha256, 'bytes': size})
    expected_source = f'/work/resume/independent-exports/mellum-{sha256}.pt'
    expected_name = f'mellum/{sha256}/mellum-{sha256}.pt'
    if (receipt.get('role') != 'mellum' or receipt.get('source') != expected_source
            or receipt.get('name') != expected_name):
        raise ValueError('private receipt is not the exact Mellum hash-addressed full export')
    return receipt


def _authenticated_private_presence(policy, receipt):
    """Confirm the receipt's exact object still exists in its private run, without downloading it."""
    sdk_python = policy.get('mellum_sdk_python')
    if not isinstance(sdk_python, str) or not Path(sdk_python).is_file() or not os.access(sdk_python, os.X_OK):
        raise ValueError('explicit executable CoreWeave SDK Python is not configured')
    query = r'''import json,sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import cwctl
cwctl.load_env_file()
import wandb
run, name, size = sys.argv[2], sys.argv[3], int(sys.argv[4])
matches = list(wandb.Api(timeout=45).run(run).files(names=[name]))
present = (len(matches) == 1 and getattr(matches[0], 'name', None) == name
           and getattr(matches[0], 'size', None) == size)
print('NATLANG_PRIVATE_OBJECT_PRESENT=' + ('1' if present else '0'))
'''
    result = subprocess.run([sdk_python, '-c', query,
        str(Path(__file__).resolve().parent / 'coreweave'), receipt['run'], receipt['name'],
        str(receipt['identity']['bytes'])], capture_output=True, text=True, timeout=90)
    if result.returncode != 0:
        raise RuntimeError('authenticated private-object query failed in the declared SDK interpreter '
                           f'(exit {result.returncode}; stderr tail: {result.stderr[-400:]})')
    markers = [line for line in result.stdout.splitlines() if line.startswith('NATLANG_PRIVATE_OBJECT_PRESENT=')]
    if len(markers) != 1 or markers[0] not in ('NATLANG_PRIVATE_OBJECT_PRESENT=0',
                                               'NATLANG_PRIVATE_OBJECT_PRESENT=1'):
        raise RuntimeError('SDK private-object query returned an invalid result')
    return markers[0].endswith('=1')


def _mellum_pointer_binding(policy, pointer_key, cache_root):
    pointer_declared = Path(policy[pointer_key])
    if pointer_declared.is_symlink() or not pointer_declared.is_file():
        raise ValueError(f'{pointer_key} is not a regular pointer file')
    pointer_path = pointer_declared.resolve(strict=True)
    pointer = _json(pointer_path)
    checkpoint = pointer.get('checkpoint') or {}
    sha, size, path = checkpoint.get('sha256'), checkpoint.get('bytes'), checkpoint.get('path')
    step = pointer.get('global_step')
    if ((pointer_key == 'mellum_local_pointer' and pointer.get('passed') is not True)
            or not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha)
            or type(size) is not int or size <= 0 or type(step) is not int or step < 0):
        raise ValueError(f'{pointer_key} is not a passed exact full-state checkpoint pointer')
    if pointer_key == 'mellum_local_pointer':
        local = Path(path)
        if (local.parent.resolve() != cache_root or local.name != f'mellum-{sha}.pt'
                or local.is_symlink() or not local.is_file() or local.stat().st_size != size):
            raise ValueError('current Mellum local recovery does not name an exact cache pin')
        receipt_path = pointer.get('private_receipt') or pointer.get('receipt')
        receipt_file = Path(receipt_path) if isinstance(receipt_path, str) else None
        if (receipt_file is None or receipt_file.is_symlink() or not receipt_file.is_file()
                or receipt_file.resolve(strict=True).parent != Path(policy['mellum_receipt_root']).resolve(strict=True)):
            raise ValueError('current Mellum local recovery lacks a receipt in the exact receipt root')
    else:
        if path != f'/work/resume/independent-exports/mellum-{sha}.pt':
            raise ValueError('current Mellum private recovery path is not the exact hash-addressed export')
        receipt_path = pointer.get('receipt') or pointer.get('private_receipt')
        receipt_file = Path(receipt_path) if isinstance(receipt_path, str) else None
        if receipt_file is None or receipt_file.is_symlink() or not receipt_file.is_file():
            raise ValueError('current Mellum private recovery lacks its exact receipt path')
        if receipt_file.resolve(strict=True).parent != Path(policy['mellum_receipt_root']).resolve(strict=True):
            raise ValueError('current Mellum private receipt escaped the exact receipt root')
    return {'pointer_path': str(pointer_path), 'pointer_sha256': _stable_file_sha256(pointer_path),
            'sha256': sha, 'bytes': size, 'global_step': step, 'path': path,
            'receipt_path': str(Path(receipt_path).resolve(strict=True))}


def _verify_mellum_local_pin(path, sha, size, receipt_path, proof_cache):
    binding = _stat_binding(path)
    if binding['bytes'] != size:
        raise ValueError('local cache pin size differs from its private receipt')
    receipt = _mellum_receipt(receipt_path, sha, size)
    receipt_sha = _stable_file_sha256(receipt_path)
    key = str(Path(path))
    proof = proof_cache['entries'].get(key)
    if not (isinstance(proof, dict) and proof.get('sha256') == sha
            and proof.get('stat') == binding and proof.get('receipt_sha256') == receipt_sha):
        actual = _hash_mellum_pin(path, binding)
        if actual != sha:
            raise ValueError('local bytes do not match the hash-addressed pin name')
        proof = {'path': key, 'sha256': sha, 'stat': binding, 'receipt_sha256': receipt_sha,
                 'verified_at': time.time(), 'method': 'streaming-sha256-8mib-stat-bound-v1'}
        proof_cache['entries'][key] = proof
    return binding, receipt_sha, receipt


def _mellum_checkpoint_lineage(policy, path, sha):
    """Read only exact torch archive metadata and bind the checkpoint to its model ancestry."""
    reader_value = policy.get('mellum_checkpoint_metadata_reader_path')
    reader_sha = policy.get('mellum_checkpoint_metadata_reader_sha256')
    source_value = policy.get('mellum_checkpoint_metadata_source_path')
    source_sha = policy.get('mellum_checkpoint_metadata_source_sha256')
    if (not isinstance(reader_value, str) or not isinstance(source_value, str)
            or not re.fullmatch(r'[0-9a-f]{64}', str(reader_sha))
            or not re.fullmatch(r'[0-9a-f]{64}', str(source_sha))):
        raise ValueError('exact metadata-only checkpoint reader custody is not configured')
    reader_declared = Path(reader_value)
    source_declared = Path(source_value)
    if reader_declared.is_symlink() or source_declared.is_symlink():
        raise ValueError('metadata-only reader dependency cannot be a symlink')
    reader_path = reader_declared.resolve(strict=True)
    source_path = source_declared.resolve(strict=True)
    actual_reader_sha = _stable_file_sha256(reader_path)
    actual_source_sha = _stable_file_sha256(source_path)
    if actual_reader_sha != reader_sha or actual_source_sha != source_sha:
        raise ValueError('metadata-only checkpoint reader source changed from its reviewed binding')
    cache_key = (str(reader_path), reader_sha, str(source_path), source_sha)
    read_metadata = _MELLUM_METADATA_READERS.get(cache_key)
    if read_metadata is None:
        spec = importlib.util.spec_from_file_location(
            'natlang_bound_mellum_checkpoint_metadata_reader', reader_path)
        if spec is None or spec.loader is None:
            raise ValueError('cannot load the exact metadata-only checkpoint reader')
        module = importlib.util.module_from_spec(spec)
        old_hash_cache = os.environ.get('NATLANG_HASH_CACHE')
        had_hash_cache = 'NATLANG_HASH_CACHE' in os.environ
        try:
            spec.loader.exec_module(module)
        finally:
            if had_hash_cache:
                os.environ['NATLANG_HASH_CACHE'] = old_hash_cache
            else:
                os.environ.pop('NATLANG_HASH_CACHE', None)
        read_metadata = getattr(module, 'checkpoint_metadata', None)
        if not callable(read_metadata):
            raise ValueError('bound metadata-only checkpoint reader has no checkpoint_metadata function')
        _MELLUM_METADATA_READERS[cache_key] = read_metadata

    before = _stat_binding(path)
    metadata = read_metadata(path)
    after = _stat_binding(path)
    if after != before:
        raise ValueError('checkpoint changed during metadata-only lineage inspection')
    if not isinstance(metadata, dict):
        raise ValueError('checkpoint metadata root is not an object')
    training = metadata.get('training')
    config = metadata.get('model_config')
    options = training.get('options') if isinstance(training, dict) else None
    step = metadata.get('step')
    if type(step) is not int and isinstance(training, dict):
        step = training.get('global_step', training.get('step'))
    architecture = config.get('architecture') if isinstance(config, dict) else None
    base_sha = config.get('base_inventory_sha256') if isinstance(config, dict) else None
    ancestors = options.get('native_snapshot_ancestors') if isinstance(options, dict) else None
    if (type(step) is not int or step < 0 or architecture != 'mellum'
            or not isinstance(base_sha, str) or not re.fullmatch(r'[0-9a-f]{64}', base_sha)
            or not isinstance(ancestors, list)
            or any(not isinstance(item, str) or not re.fullmatch(r'[0-9a-f]{64}', item)
                   for item in ancestors)
            or len(set(ancestors)) != len(ancestors)):
        raise ValueError('checkpoint metadata lacks exact Mellum model identity, step, or ancestor bindings')
    if training.get('global_step') != step:
        raise ValueError('checkpoint root step differs from its training metadata')
    return {'sha256': sha, 'global_step': step,
            'model_identity': f'{architecture}:{base_sha}',
            'native_snapshot_ancestors': ancestors,
            'native_snapshot_ancestors_sha256': hashlib.sha256(
                json.dumps(ancestors, separators=(',', ':')).encode()).hexdigest(),
            'metadata_reader': {'path': str(reader_path), 'sha256': reader_sha,
                                'metadata_source_path': str(source_path),
                                'metadata_source_sha256': source_sha},
            'checkpoint_stat': before}


def _live_file_inodes():
    found = set()
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for descriptor in (proc / 'fd').iterdir():
                try:
                    st = descriptor.stat()
                    found.add((st.st_dev, st.st_ino))
                except OSError:
                    pass
            maps = proc / 'maps'
            if maps.is_file():
                for line in maps.read_text(errors='ignore').splitlines():
                    fields = line.split(None, 5)
                    if len(fields) != 6 or not fields[5].startswith('/'):
                        continue
                    try:
                        major, minor = fields[3].split(':', 1)
                        inode = int(fields[4])
                        device = os.makedev(int(major, 16), int(minor, 16))
                        found.add((device, inode))
                    except (ValueError, OSError):
                        pass
        except OSError:
            pass
    return found


def _mellum_cache_maintenance(policy, *, apply=False, max_candidates=None):
    lock_value = policy.get('mellum_prune_lock_path')
    if not isinstance(lock_value, str) or not lock_value:
        return {'state': 'skipped', 'reason': 'Mellum prune lock path is not configured'}
    lock_path = Path(lock_value)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.parent.is_symlink():
        return {'state': 'skipped', 'reason': 'Mellum prune lock directory cannot be a symlink'}
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    except OSError as error:
        return {'state': 'skipped', 'reason': f'cannot open exact Mellum prune lock: {error}'}
    with os.fdopen(lock_fd, 'r+b') as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            return {'state': 'skipped', 'reason': 'Mellum prune lock is not a regular file'}
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'state': 'skipped', 'reason': 'another Mellum cache verification or prune pass holds the lock'}
        try:
            return _mellum_cache_maintenance_locked(policy, apply=apply, max_candidates=max_candidates)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _mellum_cache_maintenance_locked(policy, *, apply=False, max_candidates=None):
    """Verify cached pins against private receipts and persist a plan before any deletion."""
    cache_value = policy.get('mellum_cache_root')
    receipt_value = policy.get('mellum_receipt_root')
    if not isinstance(cache_value, str) or not isinstance(receipt_value, str):
        return {'state': 'skipped', 'reason': 'Mellum cache or receipt root is not configured'}
    cache_declared, receipt_declared = Path(cache_value), Path(receipt_value)
    if cache_declared.is_symlink() or receipt_declared.is_symlink():
        return {'state': 'skipped', 'reason': 'Mellum cache or receipt root cannot be a symlink'}
    try:
        cache_root = cache_declared.resolve(strict=True)
        receipt_root = receipt_declared.resolve(strict=True)
    except OSError as error:
        return {'state': 'skipped', 'reason': f'Mellum cache or receipt root is unavailable: {error}'}
    if not cache_root.is_dir() or not receipt_root.is_dir():
        return {'state': 'skipped', 'reason': 'Mellum cache or receipt root is not a verified directory'}
    mount_value, expected_uuid = policy.get('external_mount'), policy.get('external_uuid')
    if not isinstance(mount_value, str) or not isinstance(expected_uuid, str) or not expected_uuid:
        return {'state': 'skipped', 'reason': 'external cache mount identity is not configured'}
    mount_declared = Path(mount_value)
    try:
        mount_root = mount_declared.resolve(strict=True)
        mount_uuid = _mount_uuid(mount_root)
        if (not mount_root.is_dir() or mount_uuid != expected_uuid
                or not cache_root.is_relative_to(mount_root)
                or cache_root.stat().st_dev != mount_root.stat().st_dev):
            return {'state': 'skipped', 'reason': 'Mellum cache is not on the declared external mount'}
    except OSError as error:
        return {'state': 'skipped', 'reason': f'external Mellum cache mount is unavailable: {error}'}

    try:
        local_pointer = _mellum_pointer_binding(policy, 'mellum_local_pointer', cache_root)
        private_pointer = _mellum_pointer_binding(policy, 'mellum_private_pointer', cache_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'state': 'skipped', 'reason': f'current recovery pointer is invalid: {error}'}
    protected_paths = set(map(str, policy.get('mellum_protected_paths') or []))
    protected_paths.add(str(local_pointer['path']))
    protected_shas = {local_pointer['sha256'], private_pointer['sha256']}

    try:
        latest_private_receipt = _mellum_receipt(
            private_pointer['receipt_path'], private_pointer['sha256'], private_pointer['bytes'])
        latest_private_present = _authenticated_private_presence(policy, latest_private_receipt)
        if not latest_private_present:
            return {'state': 'skipped', 'reason': 'latest private full-state object is not currently present'}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        return {'state': 'skipped', 'reason': f'latest private full-state presence could not be verified: {error}'}

    state_value = policy.get('mellum_retention_state_path')
    if not isinstance(state_value, str) or not state_value:
        return {'state': 'skipped', 'reason': 'Mellum retention-state path is not configured'}
    state_path = Path(state_value)
    old_state = None
    if state_path.exists():
        try:
            old_state = _json(state_path)
            if old_state.get('schema') != 'natlang.mellum-local-retention-state/1':
                raise ValueError('unsupported retention-state schema')
            for name in ('current_local', 'previous_local'):
                row = old_state.get(name)
                if row is not None and (not isinstance(row, dict)
                    or not re.fullmatch(r'[0-9a-f]{64}', str(row.get('sha256', '')))
                    or type(row.get('bytes')) is not int or row['bytes'] <= 0
                    or not isinstance(row.get('path'), str)):
                    raise ValueError('malformed retained Mellum local parent')
        except (OSError, ValueError, AttributeError) as error:
            return {'state': 'skipped', 'reason': f'Mellum retention state is invalid: {error}'}
    if old_state and old_state.get('current_local', {}).get('sha256') != local_pointer['sha256']:
        previous_local = old_state.get('current_local')
        current_local = {key: local_pointer[key] for key in ('sha256', 'bytes', 'global_step', 'path')}
    elif old_state:
        previous_local = old_state.get('previous_local')
        current_local = old_state.get('current_local')
    else:
        current_local = {key: local_pointer[key] for key in ('sha256', 'bytes', 'global_step', 'path')}
        previous_shas = [Path(path).stem.removeprefix('mellum-')
                         for path in policy.get('mellum_protected_paths', [])]
        previous_shas = [sha for sha in previous_shas if re.fullmatch(r'[0-9a-f]{64}', sha)
                         and sha not in protected_shas]
        previous_local = None
        if previous_shas:
            sha = previous_shas[0]
            candidate = Path(cache_root / f'mellum-{sha}.pt')
            if candidate.is_file() and not candidate.is_symlink():
                previous_local = {'sha256': sha, 'bytes': candidate.stat().st_size,
                                  'global_step': None, 'path': str(candidate)}
    for parent in (current_local, previous_local):
        if parent:
            protected_shas.add(parent['sha256'])
            protected_paths.add(parent['path'])

    proof_value = policy.get('mellum_local_proof_cache')
    if not isinstance(proof_value, str) or not proof_value:
        return {'state': 'skipped', 'reason': 'Mellum local proof-cache path is not configured'}
    proof_path = Path(proof_value)
    proof_cache = {'schema': 'natlang.mellum-local-cache-proofs/1', 'entries': {}}
    if proof_path.exists():
        try:
            prior = _json(proof_path)
            if prior.get('schema') != proof_cache['schema'] or not isinstance(prior.get('entries'), dict):
                raise ValueError('local pin proof cache has an unsupported schema')
            proof_cache = prior
        except (OSError, ValueError, AttributeError) as error:
            return {'state': 'skipped', 'reason': f'local pin proof cache is invalid: {error}'}

    protected_inodes = _live_file_inodes()
    candidates = []
    inventory = []
    verified_by_sha = {}
    current_lineage = None
    required_parent_shas = {local_pointer['sha256']}
    if previous_local:
        required_parent_shas.add(previous_local['sha256'])
    paths = sorted(cache_root.glob('mellum-*.pt'), key=lambda path: (
        0 if path.name == f"mellum-{local_pointer['sha256']}.pt" else
        1 if previous_local and path.name == f"mellum-{previous_local['sha256']}.pt" else 2,
        path.name))
    visited_paths = set()
    for path in paths:
        match = re.fullmatch(r'mellum-([0-9a-f]{64})\.pt', path.name)
        if not match or path.is_symlink() or not path.is_file() or path.resolve().parent != cache_root:
            continue
        visited_paths.add(str(path))
        sha = match.group(1)
        if sha in protected_shas and sha not in required_parent_shas:
            inventory.append({'path': str(path), 'sha256': sha, 'state': 'protected',
                              'reason': 'pending or explicitly protected Mellum pin; no prune attempted'})
            continue
        try:
            receipt_path = receipt_root / f'{sha}.private-receipt.json'
            binding, receipt_sha, receipt = _verify_mellum_local_pin(
                path, sha, path.stat().st_size, receipt_path, proof_cache)
            # Persist each completed file proof so a stopped or interrupted pass
            # resumes at the next exact file instead of rereading verified bytes.
            durable_json(proof_path, proof_cache)
            present = _authenticated_private_presence(policy, receipt)
            protected = (str(path) in protected_paths or sha in protected_shas
                         or (binding['device'], binding['inode']) in protected_inodes)
            row = {'path': str(path), 'sha256': sha, 'bytes': binding['bytes'],
                   'allocated_bytes': binding['allocated_bytes'], 'stat': binding,
                   'receipt_path': str(receipt_path), 'receipt_sha256': receipt_sha,
                   'receipt_run': receipt.get('run'),
                   'local_digest_verified': True, 'private_object_present': bool(present),
                   'protected': protected}
            if sha != private_pointer['sha256'] and receipt.get('run') != latest_private_receipt.get('run'):
                raise ValueError('cached pin private receipt is not from the current Mellum training run')
            if sha == local_pointer['sha256']:
                lineage = _mellum_checkpoint_lineage(policy, path, sha)
                if lineage['global_step'] != local_pointer['global_step']:
                    raise ValueError('current local pointer step differs from checkpoint metadata')
                ancestors_sha = hashlib.sha256(json.dumps(
                    lineage['native_snapshot_ancestors'], separators=(',', ':')).encode()).hexdigest()
                if lineage.get('native_snapshot_ancestors_sha256') != ancestors_sha:
                    raise ValueError('current parent ancestry digest is inconsistent')
                lineage['receipt_run'] = receipt.get('run')
                current_lineage = lineage
                row['lineage'] = {'model_identity': lineage['model_identity'],
                    'global_step': lineage['global_step'], 'current_parent': True,
                    'metadata_reader': lineage['metadata_reader']}
            else:
                if current_lineage is None:
                    raise ValueError('current local model ancestry has not been authenticated')
                ancestor = sha in current_lineage['native_snapshot_ancestors']
                same_model_run = receipt.get('run') == current_lineage['receipt_run']
                row['lineage'] = {'model_identity': current_lineage['model_identity'],
                    'current_global_step': current_lineage['global_step'],
                    'ancestor_of_current': ancestor, 'same_model_run': same_model_run,
                    'current_checkpoint_sha256': current_lineage['sha256'],
                    'metadata_reader': current_lineage['metadata_reader']}
                if not (ancestor and same_model_run):
                    raise ValueError('candidate is not an exact older same-model ancestor of the current local parent')
            inventory.append(row)
            verified_by_sha[sha] = row
            if not protected and present:
                candidates.append(row)
                if (max_candidates is not None and type(max_candidates) is int
                        and max_candidates > 0 and len(candidates) >= max_candidates):
                    break
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            inventory.append({'path': str(path), 'sha256': sha, 'state': 'ineligible',
                              'reason': f'{type(error).__name__}: {str(error)[:400]}'})
    if max_candidates is not None and type(max_candidates) is int and max_candidates > 0 \
            and len(candidates) >= max_candidates:
        for path in paths:
            match = re.fullmatch(r'mellum-([0-9a-f]{64})\.pt', path.name)
            if match and str(path) not in visited_paths and path.is_file() and not path.is_symlink():
                inventory.append({'path': str(path), 'sha256': match.group(1), 'state': 'not_examined',
                                  'reason': 'per-pass candidate limit reached; no receipt or content claim made'})

    # Current recovery must still have two locally verified parents (the live
    # pointer plus the newest prior protected cache pin) and a fresh private
    # object before any old cache is eligible for pruning.
    current_row = verified_by_sha.get(local_pointer['sha256'])
    private_error = None
    prior_parent_rows = [verified_by_sha.get(previous_local['sha256'])] if previous_local else []
    prior_parent_rows = [row for row in prior_parent_rows if row and row.get('local_digest_verified')
                         and row.get('private_object_present')]
    local_recovery_ready = bool(current_row and current_row.get('local_digest_verified')
                                and current_row.get('lineage', {}).get('current_parent')
                                and latest_private_present and prior_parent_rows
                                and all(row.get('lineage', {}).get('ancestor_of_current')
                                        and row.get('lineage', {}).get('same_model_run')
                                        for row in prior_parent_rows)
                                and current_row.get('receipt_run') == latest_private_receipt.get('run'))
    if not local_recovery_ready:
        candidates = []
    else:
        state = {'schema': 'natlang.mellum-local-retention-state/1',
                 'current_local': current_local, 'previous_local': previous_local,
                 'updated_at': time.time()}
        durable_json(state_path, state)

    plan = {'schema': 'natlang.mellum-local-cache-prune-plan/1', 'at': time.time(),
            'operator': policy.get('_operator_binding'),
            'cache_root': str(cache_root), 'receipt_root': str(receipt_root),
            'current_local_pointer': local_pointer, 'current_private_pointer': private_pointer,
            'retention_state': {'current_local': current_local, 'previous_local': previous_local},
            'protected_sha256': sorted(protected_shas), 'protected_paths': sorted(protected_paths),
            'verified_current_and_previous_bindings': [row for row in inventory if row.get('protected')],
            'current_local_lineage': ({key: current_lineage[key] for key in (
                'sha256', 'global_step', 'model_identity', 'native_snapshot_ancestors',
                'native_snapshot_ancestors_sha256', 'metadata_reader', 'checkpoint_stat')}
                if current_lineage else None),
            'inventory': inventory,
            'candidates': candidates,
            'candidate_bytes': sum(row['bytes'] for row in candidates),
            'candidate_allocated_bytes': sum(row['allocated_bytes'] for row in candidates),
            'latest_private_object_present': latest_private_present,
            'latest_private_error': private_error,
            'local_recovery_ready': local_recovery_ready,
            'candidate_limit': max_candidates,
            'scan_complete': not any(row.get('state') == 'not_examined' for row in inventory),
            'apply_enabled': bool(apply and local_recovery_ready), 'deletions': [],
            'scope': 'hash-addressed Mellum local cache only; no remote object or source is changed'}
    plan_value = policy.get('mellum_prune_plan_path')
    if not isinstance(plan_value, str) or not plan_value:
        return {'state': 'skipped', 'reason': 'Mellum prune-plan path is not configured', 'inventory': inventory}
    plan_path = Path(plan_value)
    durable_json(plan_path, plan)
    # Rebind both pointers after the expensive hash/API pass. A learner export
    # can advance while the external cache is being checked.
    try:
        local_after = _mellum_pointer_binding(policy, 'mellum_local_pointer', cache_root)
        private_after = _mellum_pointer_binding(policy, 'mellum_private_pointer', cache_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {'state': 'skipped', 'path': str(plan_path), 'reason': f'pointer changed during verification: {error}',
                'inventory': inventory, 'deletions': []}
    pointers_stable = local_after == local_pointer and private_after == private_pointer
    if not pointers_stable or _live_file_inodes() != protected_inodes:
        # Inode tables are naturally volatile; only discard stale candidates
        # whose identity is now live, then proceed with the exact old plan.
        live_now = _live_file_inodes()
        candidates = [row for row in candidates
                      if (row['stat']['device'], row['stat']['inode']) not in live_now]
        plan['candidates'] = candidates
        plan['candidate_bytes'] = sum(row['bytes'] for row in candidates)
        plan['candidate_allocated_bytes'] = sum(row['allocated_bytes'] for row in candidates)
        plan['apply_enabled'] = bool(apply and local_recovery_ready and pointers_stable)
        if not pointers_stable:
            plan['skip_reason'] = 'current local or private recovery pointer changed during verification'
        durable_json(plan_path, plan)
    if not pointers_stable:
        return {'state': 'skipped', 'path': str(plan_path), 'reason': plan['skip_reason'],
                'candidate_count': len(candidates), 'candidate_bytes': plan['candidate_bytes'],
                'deletions': [], 'inventory': inventory}
    if not apply or not plan['apply_enabled']:
        return {'state': 'planned', 'path': str(plan_path), 'sha256': _stable_file_sha256(plan_path),
                'candidate_count': len(candidates), 'candidate_bytes': plan['candidate_bytes'],
                'candidate_allocated_bytes': plan['candidate_allocated_bytes'],
                'ineligible_count': sum(row.get('state') == 'ineligible' for row in inventory),
                'deletions': [], 'inventory': inventory}

    # The plan is durable before unlink. Recheck pointers, path, receipt, private
    # object and open/mapped readers immediately before removal; never add rows.
    deleted = []
    for row in candidates:
        path = Path(row['path'])
        try:
            local_now = _mellum_pointer_binding(policy, 'mellum_local_pointer', cache_root)
            private_now = _mellum_pointer_binding(policy, 'mellum_private_pointer', cache_root)
            if local_now != local_pointer or private_now != private_pointer:
                break
            if path.resolve(strict=True).parent != cache_root or path.is_symlink():
                raise ValueError('candidate escaped the exact cache root')
            if (str(path) in protected_paths or row['sha256'] in protected_shas
                    or (path.stat().st_dev, path.stat().st_ino) in _live_file_inodes()):
                continue
            if _stat_binding(path) != row['stat']:
                continue
            receipt_now = _mellum_receipt(row['receipt_path'], row['sha256'], row['bytes'])
            if _stable_file_sha256(row['receipt_path']) != row['receipt_sha256']:
                continue
            if not _authenticated_private_presence(policy, receipt_now):
                continue
            st = path.stat()
            path.unlink()
            fd = os.open(cache_root, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if path.exists():
                raise RuntimeError('cache path remains after unlink')
            inode_still_live = (st.st_dev, st.st_ino) in _live_file_inodes()
            reclaimed = row['allocated_bytes'] if st.st_nlink == 1 and not inode_still_live else 0
            deleted.append({'path': str(path), 'sha256': row['sha256'], 'bytes': row['bytes'],
                            'links_before': st.st_nlink, 'confirmed_absent': True,
                            'reclaimed_bytes': reclaimed})
            plan['deletions'] = deleted
            plan['deletions_updated_at'] = time.time()
            durable_json(plan_path, plan)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            row['apply_error'] = f'{type(error).__name__}: {str(error)[:400]}'
    plan['deletions'] = deleted
    plan['applied_at'] = time.time()
    durable_json(plan_path, plan)
    return {'state': 'applied' if deleted else 'planned-no-deletions', 'path': str(plan_path),
            'sha256': _stable_file_sha256(plan_path), 'candidate_count': len(candidates),
            'deleted': deleted, 'reclaimed_bytes': sum(row['reclaimed_bytes'] for row in deleted),
            'inventory': inventory}


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
    # The archiver prints one success event only after verified_copy completed,
    # the source was atomically replaced by its destination symlink, and the
    # corresponding manifest record was appended. Preserve those successes if
    # a later candidate fails and the process exits before its final summary.
    archived_records = _verified_archive_events(result.stdout, policy)
    moved_files = max(int((summary or {}).get('moved_files', 0)), len(archived_records))
    if archived_records:
        moved_bytes = sum(record['bytes'] for record in archived_records)
    else:
        moved_bytes = int(float((summary or {}).get('moved_gb', 0)) * GIB)
    return {'executed': True, 'returncode': result.returncode,
            'moved_files': moved_files,
            'moved_bytes': moved_bytes,
            'verified_archived_records': archived_records,
            'summary': summary,
            'command': command, 'stdout': result.stdout[-20000:], 'stderr': result.stderr[-5000:]}


def _verified_archive_events(stdout, policy):
    """Return exact manifest-backed archive events that remain linked in place."""
    archive_value = policy.get('archive_destination')
    if not isinstance(archive_value, str) or not archive_value:
        return []
    manifest = Path(archive_value) / 'manifest.jsonl'
    try:
        entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
    by_source = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get('source'), str):
            by_source.setdefault(entry['source'], []).append(entry)
    verified = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        source_value = event.get('archived') if isinstance(event, dict) else None
        if not isinstance(source_value, str) or source_value in verified:
            continue
        source = Path(source_value)
        try:
            if not source.is_symlink():
                continue
            target = source.resolve(strict=True)
            stat_result = target.stat()
            if not target.is_file():
                continue
        except OSError:
            continue
        for entry in by_source.get(str(source), []):
            try:
                recorded_target = Path(entry['dest']).resolve(strict=True)
                size = int(entry['bytes'])
                digest = entry['sha256']
            except (KeyError, TypeError, ValueError, OSError):
                continue
            if (recorded_target == target and size == stat_result.st_size
                    and isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest)):
                verified[source_value] = {'source': source_value, 'dest': str(target),
                                          'bytes': size, 'sha256': digest}
                break
    return list(verified.values())


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


def inspect(policy, *, mellum_prune_plan=False):
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
    should_maintain_mellum = mellum_prune_plan or policy.get('mellum_prune_enabled') is True
    if should_maintain_mellum:
        apply_mellum = policy.get('mellum_prune_enabled') is True and not mellum_prune_plan
        if apply_mellum and not (external.get('identity_ok') and external.get('floor_ok')):
            status['mellum_cache_prune'] = {'state': 'skipped',
                'reason': 'external mount identity or configured free-space floor is not verified'}
        else:
            max_candidates = (policy.get('mellum_prune_max_candidates_per_pass', 1) if apply_mellum else
                              policy.get('mellum_plan_max_candidates', 1))
            result = _mellum_cache_maintenance(policy, apply=apply_mellum,
                                               max_candidates=max_candidates)
            status['mellum_cache_prune'] = result
            if result.get('deleted'):
                status['actions_applied'].append('verified-old-mellum-cache-prune')

    internal = status.get('internal') or {}
    pressure = (not status.get('internal_state_error')
                and not internal.get('target_headroom_ok', internal.get('headroom_ok', False)))
    if pressure and mellum_prune_plan:
        status['action_required'] = 'Mellum cache plan mode skipped unrelated LFM archive planning or apply.'
    elif pressure and external_ok and external.get('floor_ok'):
        free = internal.get('free_bytes')
        target = internal.get('target_free_bytes', internal.get('next_write_required_headroom_bytes'))
        protected = _archive_protected_paths(policy, status['checkpoint'])
        status['archive_protected_paths'] = protected
        plan = _archive_dry_run(policy, max(0, int(target or 0) - int(free or 0)), protected)
        status['archive_dry_run'] = plan
        status['action_required'] = ('Review bounded closed-file archive candidates; no file was moved.'
                                     if plan['returncode'] == 0 else
                                     'Archive dry-run failed; preserve current checkpoint and review the error.')
        if (not mellum_prune_plan and plan['returncode'] == 0
                and policy.get('archive_apply_enabled') is True):
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
    # The active frozen learner has not been rebound to CheckpointWriteAdmission.
    # This service can estimate pressure and archive eligible closed files, but it
    # is not in the learner's first-write path and must not claim to reserve space.
    status['next_write_admission'] = {
        'mode': 'monitor-and-recover',
        'live_writer_guard_installed': False,
        'estimated_required_headroom_bytes': internal.get('next_write_required_headroom_bytes'),
        'reason': ('the active frozen runtime does not call the shared checkpoint admission guard; '
                   'this estimate is advisory until an explicitly bound runtime cutover'),
    }
    return status


def run_once(policy, *, mellum_prune_plan=False):
    status = inspect(policy, mellum_prune_plan=mellum_prune_plan)
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
    mode.add_argument('--mellum-prune-plan', action='store_true',
                      help='stream-verify local Mellum cache pins and write a plan; never unlink')
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return
    policy = _json(args.policy)
    policy_path = args.policy.resolve(strict=True)
    if args.policy.is_symlink() or not policy_path.is_file():
        raise ValueError('storage policy must be a regular non-symlink file')
    policy['_path'] = str(policy_path)
    policy['_operator_binding'] = {
        'script_path': str(Path(__file__).resolve(strict=True)),
        'script_sha256': _stable_file_sha256(Path(__file__).resolve(strict=True)),
        'policy_path': str(policy_path),
        'policy_sha256': _stable_file_sha256(policy_path),
    }
    if args.mellum_prune_plan:
        print(json.dumps(run_once(policy, mellum_prune_plan=True), sort_keys=True))
        return
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
