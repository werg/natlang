#!/usr/bin/env python3
"""Review-only-by-default recovery for the two user-authorized DGX data moves.

No container is started and no tree is checksummed unless --execute is supplied.
The destructive operation mounts only one fixed, reviewed backup directory in a
root container. `.cache`, models, containers, checkpoints and arbitrary paths are
outside this script's path allowlist.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

EXTERNAL = Path('/mnt/external')
DEST_ROOT = EXTERNAL / 'natlang-storage-20261002'
STATE = Path('/home/werg/natlang-remote/storage-relocation-20261002')
DOCKER_IMAGE = 'alpine@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b'
TREES = (
    ('sdkb-runs', Path('/home/werg/sdkb-runs'), Path('/home/werg/sdkb-runs.before-external-move-20261002')),
    ('bgkit-data-nvme', Path('/home/werg/bgkit-data-nvme'), Path('/home/werg/bgkit-data-nvme.before-external-move-20261002')),
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def append_event(name: str, state: str, **fields) -> None:
    path = STATE / f'{name}.recovery-events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at': now(), 'state': state, **fields}, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def run(args: list[str], *, capture=False):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def writers_under(prefixes: tuple[Path, ...]) -> list[dict]:
    roots = tuple(str(p) + '/' for p in prefixes)
    writers = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.endswith(' (deleted)'):
                        target = target[:-10]
                    if not target.startswith(roots):
                        continue
                    info = (proc / 'fdinfo' / fd.name).read_text(encoding='ascii')
                    flags = next(line.split()[1] for line in info.splitlines() if line.startswith('flags:'))
                    if int(flags, 8) & os.O_ACCMODE:
                        writers.append({'pid': int(proc.name), 'path': target})
                except (OSError, StopIteration, ValueError):
                    continue
        except OSError:
            continue
    return writers


def canonical(path: Path) -> Path:
    return path.resolve(strict=True)


def state_path(name: str) -> Path:
    return STATE / f'{name}.release-state.json'


def receipt_path(name: str) -> Path:
    return STATE / f'{name}.release-receipt.json'


def target_for(name: str) -> Path:
    return DEST_ROOT / name


def nested_mountpoints(root: Path, mountinfo: Path = Path('/proc/self/mountinfo')) -> list[str]:
    """Find mountpoints strictly below root, decoding kernel octal escapes first."""
    root_abs = os.path.normpath(os.path.abspath(root))
    found = []
    for line in mountinfo.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if len(fields) < 6:
            raise RuntimeError(f'malformed mountinfo row; refuse mount traversal: {line[:300]}')
        raw = fields[4]
        decoded = re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), raw)
        point = os.path.normpath(decoded)
        try:
            inside = os.path.commonpath((root_abs, point)) == root_abs
        except ValueError:
            inside = False
        if inside and point != root_abs:
            found.append(point)
    return sorted(set(found))


def reject_nested_mountpoints(root: Path, *, view: str) -> None:
    nested = nested_mountpoints(root)
    if nested:
        raise RuntimeError(f'nested mountpoint(s) under exact {view} tree; refuse recursive deletion: {nested[:20]}')


def validate_removed_receipt(name: str, source: Path, backup: Path, target: Path) -> dict:
    receipt_file = receipt_path(name)
    if not receipt_file.is_file():
        raise RuntimeError(f'backup absent without release receipt: {backup}')
    receipt = json.loads(receipt_file.read_text())
    expected = {'schema': 'natlang.dgx-backup-release-receipt/1', 'name': name,
                'source': str(source), 'destination': str(target), 'backup': str(backup),
                'exact_bind_mount': str(backup), 'cache_touched': False}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'backup release receipt identity mismatch: {name}')
    if receipt.get('phase') not in {'backup_removed', 'backup_removed_recovered'}:
        raise RuntimeError(f'backup release receipt phase is not complete: {name}')
    return receipt


def lightweight_plan(name: str, source: Path, backup: Path, target: Path) -> dict:
    source_kind = 'missing'
    link = None
    if source.is_symlink():
        source_kind = 'symlink'
        link = os.readlink(source)
    elif source.is_dir():
        source_kind = 'directory'
    elif source.exists():
        source_kind = 'other'
    backup_kind = 'directory' if backup.is_dir() and not backup.is_symlink() else ('symlink' if backup.is_symlink() else ('other' if backup.exists() else 'missing'))
    target_kind = 'directory' if target.is_dir() and not target.is_symlink() else ('symlink' if target.is_symlink() else ('other' if target.exists() else 'missing'))
    return {'name': name, 'source': str(source), 'source_kind': source_kind, 'source_link': link,
            'backup': str(backup), 'backup_kind': backup_kind, 'target': str(target), 'target_kind': target_kind,
            'release_receipt_exists': receipt_path(name).is_file(), 'state_exists': state_path(name).is_file()}


def verify_mounts() -> None:
    if not EXTERNAL.is_mount():
        raise RuntimeError(f'expected external mount not mounted: {EXTERNAL}')
    if not DEST_ROOT.is_dir() or DEST_ROOT.is_symlink():
        raise RuntimeError(f'destination root missing or unsafe: {DEST_ROOT}')


def verify_live_alias(source: Path, target: Path) -> None:
    if not source.is_symlink():
        raise RuntimeError(f'original path is not a symlink; refusing backup deletion: {source}')
    if canonical(source) != canonical(target):
        raise RuntimeError(f'original symlink does not resolve to exact destination: {source} -> {os.readlink(source)}')
    if not target.is_dir() or target.is_symlink():
        raise RuntimeError(f'external destination is missing or unsafe: {target}')


def verify_backup_subset(name: str, backup: Path, target: Path) -> dict:
    if not backup.is_dir() or backup.is_symlink():
        raise RuntimeError(f'backup is missing or not a real directory: {backup}')
    reject_nested_mountpoints(backup, view='host backup')
    open_writers = writers_under((backup, target))
    if open_writers:
        raise RuntimeError(f'open writable descriptors under backup/destination: {open_writers[:20]}')
    # No --delete: every backup entry must match the destination; extra live destination entries are allowed.
    args = ['rsync', '-aHnci', '--itemize-changes', '--out-format=%i %n%L', str(backup) + '/', str(target) + '/']
    result = run(args, capture=True)
    output = result.stdout + result.stderr
    if output.strip():
        raise RuntimeError(f'backup is not a byte-identical subset of destination; first differences: {output[:4000]}')
    if writers_under((backup, target)):
        raise RuntimeError('writable descriptors appeared during subset verification')
    return {'method': 'rsync archive+hardlink+checksum dry-run, no --delete (subset semantics)',
            'command': args, 'matched': True, 'difference_output_sha256': sha256(output.encode()),
            'difference_output_bytes': len(output.encode()), 'verified_at': now()}


def docker_empty_exact_backup(backup: Path) -> None:
    # This bind-mount is intentionally the only host path made visible to container root.
    reject_nested_mountpoints(backup, view='host backup before Docker bind')
    # Escaped subcomponent bytes cannot hide this ASCII prefix in mountinfo.
    # Host inspection above decodes the full paths; this checks Docker's namespace.
    awk_guard = r'''{ if ($5 != "/backup" && index($5, "/backup/") == 1) { print "nested mountpoint: " $5 > "/dev/stderr"; bad=1 } } END { exit bad ? 1 : 0 }'''
    script = ("set -eu; awk '" + awk_guard + "' /proc/self/mountinfo; "
              "test -d /backup; find /backup -xdev -mindepth 1 -depth -exec rm -rf -- '{}' \\;; "
              "test -z \"$(find /backup -mindepth 1 -print -quit)\"")
    args = [
        'docker', 'run', '--rm', '--pull=never', '--network=none', '--read-only',
        '--user=0:0', '--cap-drop=ALL', '--cap-add=DAC_OVERRIDE',
        '--security-opt=no-new-privileges',
        '--mount', f'type=bind,src={backup},dst=/backup',
        DOCKER_IMAGE, '/bin/sh', '-ec', script,
    ]
    run(args)
    # Container emptied its one bind mount. This unlink uses the user's writable /home parent.
    backup.rmdir()


def release_existing(name: str, source: Path, backup: Path, target: Path, *, execute: bool) -> dict:
    verify_live_alias(source, target)
    record = state_path(name)
    prior = json.loads(record.read_text()) if record.is_file() else {}
    if not backup.exists():
        existing_receipt = validate_removed_receipt(name, source, backup, target) if receipt_path(name).is_file() else None
        if existing_receipt:
            return existing_receipt
        if (prior.get('phase') == 'deleting_backup'
                and prior.get('name') == name
                and prior.get('source') == str(source)
                and prior.get('destination') == str(target)
                and prior.get('backup') == str(backup)
                and prior.get('subset_proof', {}).get('matched') is True):
            receipt = {'schema': 'natlang.dgx-backup-release-receipt/1', 'name': name,
                       'source': str(source), 'destination': str(target), 'backup': str(backup),
                       'symlink_target': os.readlink(source), 'phase': 'backup_removed_recovered',
                       'subset_proof': prior['subset_proof'], 'container_image': DOCKER_IMAGE,
                       'exact_bind_mount': str(backup), 'cache_touched': False,
                       'recovery_script_sha256': prior.get('recovery_script_sha256'), 'recovered_at': now()}
            atomic_json(receipt_path(name), receipt)
            atomic_json(record, {**prior, 'phase': 'backup_removed_recovered', 'updated_at': now()})
            append_event(name, 'backup_removed_recovered', receipt=str(receipt_path(name)))
            return receipt
        raise RuntimeError(f'backup absent without a matching durable release receipt: {backup}')
    proof = verify_backup_subset(name, backup, target)
    script_hash = sha256(Path(__file__).read_bytes())
    state = {'schema': 'natlang.dgx-backup-release-state/1', 'name': name,
             'source': str(source), 'destination': str(target), 'backup': str(backup),
             'symlink_target': os.readlink(source), 'phase': 'subset_verified',
             'subset_proof': proof, 'recovery_script_sha256': script_hash, 'updated_at': now()}
    atomic_json(record, state)
    append_event(name, 'subset_verified', destination=str(target), backup=str(backup), proof=proof)
    if not execute:
        return state
    if writers_under((backup, target)):
        raise RuntimeError('writable descriptors appeared after subset verification')
    state.update({'phase': 'deleting_backup', 'updated_at': now()})
    atomic_json(record, state)
    append_event(name, 'deleting_backup', container_image=DOCKER_IMAGE, mount=str(backup))
    docker_empty_exact_backup(backup)
    receipt = {'schema': 'natlang.dgx-backup-release-receipt/1', 'name': name,
               'source': str(source), 'destination': str(target), 'backup': str(backup),
               'symlink_target': os.readlink(source), 'phase': 'backup_removed',
               'subset_proof': proof, 'container_image': DOCKER_IMAGE,
               'exact_bind_mount': str(backup), 'cache_touched': False,
               'recovery_script_sha256': script_hash, 'removed_at': now()}
    atomic_json(receipt_path(name), receipt)
    state.update({'phase': 'backup_removed', 'updated_at': now()})
    atomic_json(record, state)
    append_event(name, 'backup_removed', receipt=str(receipt_path(name)))
    return receipt


def copy_and_switch(name: str, source: Path, backup: Path, target: Path, *, execute: bool) -> dict:
    if source.is_symlink():
        verify_live_alias(source, target)
        if backup.exists():
            return release_existing(name, source, backup, target, execute=execute)
        raise RuntimeError(f'already-switched source has no release receipt: {source}')
    if not source.exists() and backup.is_dir() and not backup.is_symlink():
        state_file = state_path(name)
        state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        expected = {'name': name, 'source': str(source), 'destination': str(target), 'backup': str(backup)}
        if any(state.get(key) != value for key, value in expected.items()) or state.get('phase') not in {'cutover_started', 'copied_verified'}:
            raise RuntimeError(f'missing source with backup but no valid pre-cutover receipt: {name}')
        proof = verify_backup_subset(name, backup, target)
        if not execute:
            return {'phase': 'cutover_recovery_plan', **expected, 'subset_proof': proof}
        source.symlink_to(target, target_is_directory=True)
        verify_live_alias(source, target)
        state.update({'phase': 'cutover_complete_recovered', 'symlink_target': os.readlink(source),
                      'recovery_subset_proof': proof, 'updated_at': now()})
        atomic_json(state_file, state)
        append_event(name, 'cutover_complete_recovered', backup=str(backup), destination=str(target))
        return release_existing(name, source, backup, target, execute=True)
    if not source.is_dir() or source.is_symlink() or backup.exists():
        raise RuntimeError(f'unsafe or ambiguous second-tree state: {source}, {backup}')
    state_file = state_path(name)
    resume = json.loads(state_file.read_text()) if state_file.is_file() else {}
    identity = {'schema': 'natlang.dgx-backup-release-state/1', 'name': name,
                'source': str(source), 'destination': str(target), 'backup': str(backup)}
    if resume and any(resume.get(key) != value for key, value in identity.items()):
        raise RuntimeError(f'recovery state identity mismatch: {state_file}')
    resumable = resume.get('phase') in {'copying', 'paused_copy_mismatch', 'copied_verified', 'cutover_started'}
    if (target.exists() and not resumable) or (resume and not resumable):
        raise RuntimeError(f'destination/state requires review; refusing to reuse: {target} / {state_file}')
    if not execute:
        return {'phase': 'plan_only', 'source': str(source), 'destination': str(target), 'backup': str(backup), 'copy_or_cutover_started': False}
    writers = writers_under((source, target))
    if writers:
        raise RuntimeError(f'open writable descriptors before copy: {writers[:20]}')
    state = {**identity, 'phase': 'copying', 'updated_at': now()}
    atomic_json(state_path(name), state)
    append_event(name, 'copying', source=str(source), destination=str(target), cache_touched=False)
    target.mkdir(parents=True, exist_ok=True)
    run(['rsync', '-aH', '--stats', str(source) + '/', str(target) + '/'])
    verify = run(['rsync', '-aHnci', '--delete', '--itemize-changes', '--out-format=%i %n%L', str(source) + '/', str(target) + '/'], capture=True)
    differences = verify.stdout + verify.stderr
    if differences.strip():
        atomic_json(state_path(name), {**state, 'phase': 'paused_copy_mismatch', 'difference_output_sha256': sha256(differences.encode()), 'difference_output_prefix': differences[:4000], 'updated_at': now()})
        raise RuntimeError('second-tree checksum verification differed; source kept live')
    if writers_under((source, target)):
        raise RuntimeError('writable descriptors appeared during copy verification; source kept live')
    state.update({'phase': 'copied_verified', 'copy_verification': 'rsync archive+hardlink+checksum dry-run; exact tree; no differences', 'updated_at': now()})
    atomic_json(state_path(name), state)
    append_event(name, 'copied_verified', destination=str(target), difference_output_sha256=sha256(differences.encode()))
    state.update({'phase': 'cutover_started', 'updated_at': now()})
    atomic_json(state_path(name), state)
    append_event(name, 'cutover_started', source=str(source), backup=str(backup), destination=str(target))
    source.rename(backup)
    try:
        source.symlink_to(target, target_is_directory=True)
    except BaseException:
        backup.rename(source)
        raise
    verify_live_alias(source, target)
    state.update({'phase': 'cutover_complete', 'symlink_target': os.readlink(source), 'updated_at': now()})
    atomic_json(state_path(name), state)
    append_event(name, 'cutover_complete', backup=str(backup), destination=str(target))
    return release_existing(name, source, backup, target, execute=execute)


def main() -> int:
    execute = '--execute' in sys.argv[1:]
    if set(sys.argv[1:]) - {'--execute'}:
        raise SystemExit('usage: recover_reviewed_dgx_data_relocation.py [--execute]')
    if not EXTERNAL.is_mount():
        raise RuntimeError(f'external mount unavailable: {EXTERNAL}')
    if not DEST_ROOT.is_dir() or DEST_ROOT.is_symlink():
        raise RuntimeError(f'external destination root unavailable: {DEST_ROOT}')
    cache = Path('/home/werg/.cache')
    if not cache.is_dir() or cache.is_symlink():
        raise RuntimeError('the held .cache path is not an in-place real directory; refuse all storage operations')
    if not execute:
        print(json.dumps({'mode': 'plan-only-no-checksum-no-mutation',
                          'external_mount': str(EXTERNAL),
                          'trees': [lightweight_plan(name, source, backup, target_for(name)) for name, source, backup in TREES],
                          'cache_touched': False, 'container_started': False, 'heavy_checksum_io': False}, indent=2))
        return 0
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, source, backup in TREES:
            target = target_for(name)
            if source.is_symlink():
                verify_live_alias(source, target)
                if backup.exists():
                    release_existing(name, source, backup, target, execute=True)
                else:
                    release_existing(name, source, backup, target, execute=True)
                    continue
            else:
                copy_and_switch(name, source, backup, target, execute=True)
        final = {'schema': 'natlang.dgx-relocation-recovery/1', 'phase': 'complete',
                 'updated_at': now(), 'trees': [str(receipt_path(name)) for name, _, _ in TREES],
                 'cache_touched': False}
        atomic_json(STATE / 'recovery-complete.json', final)
        print(json.dumps(final, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # This error receipt is only written in --execute mode after explicit operator run.
        # Plan-only remains read-only in the state directory.
        if '--execute' in sys.argv[1:]:
            STATE.mkdir(parents=True, exist_ok=True)
            atomic_json(STATE / 'recovery-error.json', {'at': now(), 'error_type': type(exc).__name__, 'message': str(exc), 'cache_touched': False})
        print(f'recovery refused: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise
