#!/usr/bin/env python3
"""Publish immutable corpus manifests and synchronize owner snapshots over SSH.

Code moves through Git. Data moves only by exact file manifests, without deletes
or silent replacements. Sync is not training admission.
"""
import argparse
import datetime
import hashlib
import fnmatch
import os
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import socket
import subprocess
import tempfile


def digest(path):
    h, lines, last = hashlib.sha256(), 0, b''
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
            lines += block.count(b'\n')
            last = block[-1:]
    return h.hexdigest(), lines + int(bool(last) and last != b'\n')


def relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'unsafe relative path: {value}')
    return str(path)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def offload_unlink_preflight(paths):
    """Report effective unlink permission on each resolved parent directory."""
    results = {}
    for path in paths:
        parent = path.parent.resolve(strict=True)
        mode = parent.stat()
        permitted = os.access(parent, os.W_OK | os.X_OK)
        results[str(path)] = {
            'parent': str(parent), 'parent_mode': oct(mode.st_mode & 0o7777),
            'parent_uid': mode.st_uid, 'parent_gid': mode.st_gid,
            'write_and_search_access': permitted,
        }
    return results


def save_offload_receipt(repo, path, receipt):
    """Persist the latest receipt and append every state to immutable attempt history."""
    history = repo / '.coordination' / 'artifact-evictions' / 'history' / (receipt['verification_id'] + '.jsonl')
    history.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(receipt, sort_keys=True) + '\n').encode()
    descriptor = os.open(history, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, 'ab', closefd=False) as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    save(path, receipt)


def manifest_path(repo, identity):
    if '/' in identity or identity in {'.', '..'}:
        raise ValueError('corpus id must be one path component')
    return repo / 'training/corpus-manifests' / (identity + '.json')


def require_committed_offload_snapshot(repo, identity):
    """Require the selected registry entry and immutable manifest at HEAD."""
    registry_path = repo / 'training/neuralese_corpora.json'
    selected_manifest_path = manifest_path(repo, identity)
    committed = {}
    for path in [registry_path, selected_manifest_path]:
        relative_path = path.relative_to(repo).as_posix()
        result = subprocess.run(['git', 'show', f'HEAD:{relative_path}'], cwd=repo,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode or not path.is_file():
            raise ValueError(f'offload requires a committed registry entry and manifest: {relative_path}')
        committed[relative_path] = result.stdout
    try:
        current_registry = json.loads(registry_path.read_text())
        committed_registry = json.loads(committed[registry_path.relative_to(repo).as_posix()])
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'offload requires a readable registry: {registry_path}') from error
    def find_entry(registry):
        matches = [entry for entry in registry.get('corpora', []) if entry.get('id') == identity]
        return matches[0] if len(matches) == 1 else None
    if find_entry(current_registry) is None or find_entry(current_registry) != find_entry(committed_registry):
        raise ValueError(f'offload requires the selected registry entry unchanged at HEAD: {identity}')
    manifest_rel = selected_manifest_path.relative_to(repo).as_posix()
    if selected_manifest_path.read_bytes() != committed[manifest_rel]:
        raise ValueError(f'offload requires the immutable manifest unchanged at HEAD: {manifest_rel}')


def publish(repo, entry):
    root = repo / relative(entry['path'])
    files = []
    patterns = entry.get('include', ['**/*'])
    def selected(name):
        return any(fnmatch.fnmatch(name, pattern) or
                   (pattern.startswith('**/') and fnmatch.fnmatch(name, pattern[3:])) for pattern in patterns)
    for directory, children, names in os.walk(root):
        children[:] = sorted(name for name in children if
                             (Path(directory) != root or not entry.get('include_root_prefixes') or
                              any(name.startswith(prefix) for prefix in entry['include_root_prefixes']))
                             and name not in entry.get('exclude_dirs', [])
                             and not any(name.startswith(prefix) for prefix in entry.get('exclude_dir_prefixes', [])))
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_file() and selected(str(path.relative_to(root))):
                files.append(path)
    files.sort()
    if not files:
        raise ValueError(f"empty corpus: {entry['id']} at {root}")
    rows = []
    for p in files:
        before = p.stat()
        sha, count = digest(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f'file changed during publication: {p}')
        item = {'path': str(p.relative_to(root)), 'bytes': after.st_size, 'sha256': sha}
        if p.suffix == '.jsonl':
            item['lines'] = count
        rows.append(item)
    value = {'schema': 'natlang.corpus-snapshot/1', 'id': entry['id'], 'owner': entry['owner'],
             'path': entry['path'], 'files': rows, 'bytes': sum(x['bytes'] for x in rows)}
    target = manifest_path(repo, entry['id'])
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError(f"immutable snapshot changed: {entry['id']}; publish a new id")
    save(target, value)
    print(json.dumps({'published': entry['id'], 'files': len(rows), 'bytes': value['bytes']}), flush=True)


def current_receipt(repo, manifest):
    path = repo / '.coordination/corpus-receipts' / (manifest['id'] + '.json')
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if receipt.get('status') != 'verified' or receipt.get('manifest_sha256') != identity:
        return False
    root = repo / relative(manifest['path'])
    for item in manifest['files']:
        p = root / relative(item['path'])
        if not p.is_file() or receipt.get('file_stats', {}).get(item['path']) != file_stat(p):
            return False
    return True


def file_stat(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def machine_boot_id():
    """Return this kernel's boot ID where Linux exposes it."""
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() or None
    except OSError:
        return None


def verify_files(root, files, reject_symlinks=False):
    errors, checked, stats = [], [], {}
    for item in files:
        p = root / relative(item['path'])
        if reject_symlinks and p.is_symlink():
            errors.append({'path': item['path'], 'reason': 'symlink'})
        elif not p.is_file() or p.stat().st_size != item['bytes']:
            errors.append({'path': item['path'], 'reason': 'missing_or_size'})
        else:
            before = file_stat(p)
            sha, _ = digest(p)
            if before != file_stat(p):
                errors.append({'path': item['path'], 'reason': 'changed_during_verification'})
            elif sha != item['sha256']:
                errors.append({'path': item['path'], 'reason': 'hash'})
            else:
                checked.append(item['path'])
                stats[item['path']] = before
    return errors, checked, stats


def verify(repo, manifest, receipt_group='corpus-receipts', verification_scope='local', receipt_key=None):
    root = repo / relative(manifest['path'])
    errors, checked, stats = verify_files(root, manifest['files'])
    receipt = {'id': manifest['id'], 'status': 'failed' if errors else 'verified', 'checked': len(checked),
               'verification_scope': verification_scope, 'verified_hostname': socket.gethostname(),
               'machine_boot_id': machine_boot_id(),
               'verified_repo': str(repo.resolve()),
               'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'errors': errors, 'file_stats': stats}
    receipt['verification_id'] = receipt_key or manifest['id']
    save(repo / '.coordination' / receipt_group / (receipt['verification_id'] + '.json'), receipt)
    if errors:
        raise ValueError(json.dumps(receipt))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'file_stats'}), flush=True)


def verify_remote(manifest, args, receipt_group='corpus-receipts', receipt_key=None):
    """Hash the actual SSH destination; --machine names this CLI's local role."""
    remote_script = "import json,sys; from pathlib import Path; from sync_training_corpora import verify; p=json.load(sys.stdin); verify(Path(sys.argv[1]),p['manifest'],receipt_group=p['receipt_group'],verification_scope='ssh-remote',receipt_key=p['receipt_key'])"
    command = f'cd {shlex.quote(args.remote_repo + "/scripts")} && python3 -c {shlex.quote(remote_script)} {shlex.quote(args.remote_repo)}'
    payload = {'manifest': manifest, 'receipt_group': receipt_group, 'receipt_key': receipt_key}
    try:
        result = subprocess.run(['ssh', args.host, command], input=json.dumps(payload), text=True, check=True,
                                capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or str(error)).strip()
        raise RuntimeError(f'remote verification failed on {args.host}: {detail}') from error
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError('remote verification returned no receipt')
    receipt = json.loads(lines[-1])
    if (receipt.get('status') != 'verified' or receipt.get('verification_scope') != 'ssh-remote' or
            receipt.get('manifest_sha256') != hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() or
            receipt.get('verification_id') != (receipt_key or manifest['id']) or
            receipt.get('checked') != len(manifest['files'])):
        raise ValueError('remote verification receipt does not bind the selected manifest')
    return receipt


def select_manifest_files(manifest, requested):
    paths = [relative(path) for path in requested]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('offload requires distinct exact manifest file paths')
    known = {item['path'] for item in manifest['files']}
    missing = sorted(set(paths) - known)
    if missing:
        raise ValueError('paths absent from immutable manifest: ' + str(missing))
    requested_set = set(paths)
    files = [item for item in manifest['files'] if item['path'] in requested_set]
    return {**manifest, 'files': files, 'bytes': sum(item['bytes'] for item in files)}


def local_references(paths, proc_root=Path('/proc')):
    """Return open-FD and live-process references visible from this host."""
    targets = {str(path.resolve()) for path in paths}
    open_fds, jobs, inaccessible = [], [], 0
    try:
        processes = list(proc_root.iterdir())
    except OSError as error:
        raise ValueError(f'cannot inspect local process references: {error}') from error
    for process in processes:
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        pid = process.name
        try:
            fd_dir = process / 'fd'
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                target = target.removesuffix(' (deleted)')
                if target in targets:
                    open_fds.append({'pid': int(pid), 'fd': fd.name, 'path': target})
        except OSError:
            inaccessible += 1
        # Only command-line/environment/cwd references from same-user processes are
        # considered live-job references. System processes are not job owners.
        try:
            if (process.stat().st_uid != os.getuid()):
                continue
            command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            environment = (process / 'environ').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = os.readlink(process / 'cwd')
        except OSError:
            continue
        for resolved in targets:
            parent = str(Path(resolved).parent)
            if resolved in command or resolved in environment or cwd == resolved:
                jobs.append({'pid': int(pid), 'path': resolved, 'source': 'process-arguments-or-environment'})
            elif parent in command or parent in environment or cwd == parent:
                jobs.append({'pid': int(pid), 'path': resolved, 'source': 'artifact-directory-reference'})
    return {'open_fds': open_fds, 'live_job_references': jobs,
            'inaccessible_fd_directories': inaccessible}


def offload(repo, entry, manifest, args):
    if entry['owner'] != args.machine:
        raise ValueError(f'offload requires the local {args.machine} machine to own the registered snapshot')
    selected = select_manifest_files(manifest, args.files or [])
    root = repo / relative(manifest['path'])
    paths = [root / relative(item['path']) for item in selected['files']]
    errors, checked, stats = verify_files(root, selected['files'], reject_symlinks=True)
    if errors:
        raise ValueError('local files do not match the immutable manifest: ' + json.dumps(errors))
    unlink_permissions = offload_unlink_preflight(paths)
    references = local_references(paths)
    if references['open_fds'] or references['live_job_references']:
        raise ValueError('selected files are still referenced locally: ' + json.dumps(references))
    denied_unlinks = {path: details for path, details in unlink_permissions.items()
                      if not details['write_and_search_access']}
    if args.execute and denied_unlinks:
        raise ValueError('selected file parent directories are not writable/searchable for unlink: ' +
                         json.dumps(denied_unlinks, sort_keys=True))
    local_boot_id = machine_boot_id()

    selection_hash = hashlib.sha256(json.dumps(selected['files'], sort_keys=True).encode()).hexdigest()
    verification_id = f"{manifest['id']}--{selection_hash[:16]}"
    restore = f"python3 scripts/sync_training_corpora.py restore --machine {shlex.quote(args.machine)} --host {shlex.quote(args.host)} " + \
        f"--remote-repo {shlex.quote(args.remote_repo)} --id {shlex.quote(manifest['id'])} " + \
        ' '.join(f'--file {shlex.quote(item["path"])}' for item in selected['files'])
    if not args.execute:
        print(json.dumps({'status': 'preflight_only', 'id': manifest['id'], 'machine': args.machine, 'files': selected['files'],
                          'bytes': selected['bytes'], 'local_sha256_verified': True,
                          'local_hostname': socket.gethostname(), 'local_boot_id': local_boot_id,
                          'local_references': references, 'unlink_permissions': unlink_permissions,
                          'remote': f'{args.host}:{args.remote_repo}/{relative(manifest["path"])}',
                          'will_transfer_and_verify_before_unlink': True, 'restore': restore,
                          'execute_command': f"python3 scripts/sync_training_corpora.py offload --machine {shlex.quote(args.machine)} --host {shlex.quote(args.host)} --remote-repo {shlex.quote(args.remote_repo)} --id {shlex.quote(manifest['id'])} " +
                                             ' '.join(f'--file {shlex.quote(item["path"])}' for item in selected['files']) + ' --execute'}), flush=True)
        return

    transfer_args = argparse.Namespace(**vars(args))
    transfer_args.action = 'sync'
    transfer_args.files = [item['path'] for item in selected['files']]
    transfer_args.remote_receipt_group = 'artifact-offload-verifications'
    transfer_args.remote_receipt_key = verification_id
    remote_receipt = sync(repo, entry, manifest, transfer_args)
    if (not remote_receipt or remote_receipt.get('status') != 'verified' or
            remote_receipt.get('manifest_sha256') != hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()):
        raise ValueError('selected remote files were not transferred and verified against their exact manifest')
    remote_boot_id = remote_receipt.get('machine_boot_id')
    if local_boot_id and remote_boot_id and local_boot_id == remote_boot_id:
        raise ValueError('remote verification ran on the same kernel boot as this local host; refusing to unlink a local copy')

    # Re-hash and re-check references after the remote operation, immediately
    # before the optional unlink, so a changed/opened file is never removed.
    errors, checked, final_stats = verify_files(root, selected['files'], reject_symlinks=True)
    final_references = local_references(paths)
    if errors or final_stats != stats or final_references['open_fds'] or final_references['live_job_references']:
        raise ValueError('local files or references changed after remote verification: ' + json.dumps({
            'errors': errors, 'stats_unchanged': final_stats == stats, 'references': final_references}))

    attempt_id = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='microseconds') + f'-pid{os.getpid()}'
    receipt = {'schema': 'natlang.artifact-offload/1', 'id': manifest['id'], 'verification_id': verification_id,
               'attempt_id': attempt_id,
               'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               'selected_files': selected['files'], 'bytes': selected['bytes'], 'local_file_stats': final_stats,
               'local_hostname': socket.gethostname(), 'local_boot_id': local_boot_id,
               'local_references': final_references, 'unlink_permissions': unlink_permissions,
               'remote_verification': remote_receipt,
               'restore': restore, 'status': 'verified_not_unlinked',
               'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    receipt_path = repo / '.coordination' / 'artifact-evictions' / (verification_id + '.json')
    save_offload_receipt(repo, receipt_path, receipt)
    removed = []
    try:
        for item, path in zip(selected['files'], paths):
            if path.is_symlink() or not path.is_file() or file_stat(path) != final_stats[item['path']]:
                raise ValueError(f'local file changed before unlink: {item["path"]}')
            # Check references once more per file to narrow the check/unlink window.
            current_refs = local_references([path])
            if current_refs['open_fds'] or current_refs['live_job_references']:
                raise ValueError(f'local file gained a live reference before unlink: {item["path"]}')
            path.unlink()
            removed.append(item['path'])
    except Exception as error:
        receipt.update({'status': 'partial_unlink', 'removed_files': removed,
                        'unlink_error': str(error), 'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        save_offload_receipt(repo, receipt_path, receipt)
        raise
    receipt.update({'status': 'unlinked', 'removed_files': removed,
                    'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    save_offload_receipt(repo, receipt_path, receipt)
    print(json.dumps({'status': receipt['status'], 'receipt': str(receipt_path),
                      'files': removed, 'restore': restore}), flush=True)


def sync(repo, entry, manifest, args):
    root = repo / relative(entry['path'])
    remote = args.remote_repo + '/' + relative(entry['path'])
    pull = args.action == 'restore' or entry['owner'] != args.machine
    selected = getattr(args, 'files', None)
    if selected:
        requested = {relative(path) for path in selected}
        known = {item['path'] for item in manifest['files']}
        if not requested <= known:
            raise ValueError('restore paths absent from immutable manifest: ' + str(requested - known))
        manifest = {**manifest, 'files': [item for item in manifest['files'] if item['path'] in requested]}
        manifest['bytes'] = sum(item['bytes'] for item in manifest['files'])
    if pull and not selected and current_receipt(repo, manifest):
        print(json.dumps({'id': entry['id'], 'status': 'verified_unchanged'}), flush=True)
        return
    if pull:
        # A local owner can restore an evicted mirror too. Verify that mirror's
        # exact bytes before transfer, without claiming subset = full snapshot.
        source_guard = """import json,sys,hashlib
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['root'])
for i in p['files']:
 f=root/i['path']; assert f.is_file() and f.stat().st_size==i['bytes'], 'mirror missing: '+str(f)
 h=hashlib.sha256()
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
 assert h.hexdigest()==i['sha256'], 'mirror hash mismatch: '+str(f)
"""
        subprocess.run(['ssh', args.host, 'python3 -c ' + shlex.quote(source_guard)],
                       input=json.dumps({'root': remote, 'files': manifest['files']}), text=True, check=True)
    # Destination existing bytes are immutable; matching files can be reused.
    # Mismatches fail rather than replacing another agent's artifact.
    guard = """import json,sys,hashlib,shutil
from pathlib import Path
p=json.load(sys.stdin); root=Path(p['root']);root.mkdir(parents=True,exist_ok=True); need=0
for i in p['files']:
 f=root/i['path']
 if f.exists():
  h=hashlib.sha256()
  with f.open('rb') as s:
   for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
  assert f.stat().st_size==i['bytes'] and h.hexdigest()==i['sha256'], 'destination collision: '+str(f)
 else:need+=i['bytes']
assert shutil.disk_usage(root).free >= need+p['reserve'], 'insufficient destination space'
"""
    payload = {'root': str(root) if pull else remote, 'files': manifest['files'], 'reserve': int(args.reserve_gib * 2**30)}
    command = ['python3', '-c', guard] if pull else ['ssh', args.host, 'python3 -c ' + shlex.quote(guard)]
    subprocess.run(command, input=json.dumps(payload), text=True, check=True)
    with tempfile.NamedTemporaryFile('w') as file_list:
        for item in manifest['files']:
            file_list.write(relative(item['path']) + '\n')
        file_list.flush()
        source, destination = (f'{args.host}:{remote}/', str(root) + '/') if pull else (str(root) + '/', f'{args.host}:{remote}/')
        # Manifests govern content, not machine/container ownership or mtimes.
        # The preflight hashed every existing destination; never rewrite it.
        # Preserve finalized best/current checkpoint and corpus hard links so
        # transferring aliases does not duplicate gigabytes of immutable bytes.
        subprocess.run(['rsync', '-rzH', '--ignore-existing', '--protect-args', '--partial', '--partial-dir=.sync-partial',
                        '--files-from=' + file_list.name, '--stats', source, destination], check=True)
    if pull:
        verify(repo, manifest, receipt_group='corpus-restores' if selected else 'corpus-receipts')
    else:
        return verify_remote(manifest, args,
                             receipt_group=getattr(args, 'remote_receipt_group', 'corpus-receipts'),
                             receipt_key=getattr(args, 'remote_receipt_key', None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['publish', 'verify', 'verify-remote', 'sync', 'restore', 'status', 'offload'],
                        help='offload transfers and verifies exact manifest files before optional local unlink')
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--machine', choices=['pop', 'dgx'], required=True,
                        help='identity of the machine running this command, not a verification destination')
    parser.add_argument('--id', action='append', dest='ids')
    parser.add_argument('--file', action='append', dest='files', help='select exact manifest paths for restore or offload')
    parser.add_argument('--execute', action='store_true', help='for offload, transfer/verify then unlink selected local files')
    parser.add_argument('--host', default='dgx')
    parser.add_argument('--remote-repo', default='/home/werg/natlang')
    parser.add_argument('--reserve-gib', type=float, default=8)
    args = parser.parse_args()
    if args.files and (args.action not in {'restore', 'offload'} or not args.ids or len(args.ids) != 1):
        parser.error('--file requires restore/offload and exactly one --id')
    if args.action == 'offload' and (not args.files or not args.ids or len(args.ids) != 1):
        parser.error('offload requires exactly one --id and one or more --file paths')
    if args.execute and args.action != 'offload':
        parser.error('--execute is only valid with offload')
    registry = json.loads((args.repo / 'training/neuralese_corpora.json').read_text())
    known = {entry['id'] for entry in registry['corpora']}
    if args.ids and not set(args.ids).issubset(known):
        raise ValueError('unknown corpus ids: ' + str(set(args.ids) - known))
    for entry in registry['corpora']:
        if args.ids and entry['id'] not in args.ids:
            continue
        if entry.get('retired') and not args.ids and args.action != 'status':
            continue
        if args.action == 'publish':
            if entry['owner'] == args.machine:
                publish(args.repo, entry)
            continue
        path = manifest_path(args.repo, entry['id'])
        if not path.exists():
            if args.action == 'status':
                print(json.dumps({'id': entry['id'], 'status': 'unpublished', 'admission': entry['admission']}))
                continue
            raise ValueError(f'missing manifest: {path}; owner must publish and push it first')
        manifest = json.loads(path.read_text())
        if any(manifest.get(k) != entry[k] for k in ['id', 'path', 'owner']):
            raise ValueError('registry and snapshot identity differ')
        if args.action == 'verify':
            verify(args.repo, manifest)
        elif args.action == 'verify-remote':
            print(json.dumps(verify_remote(manifest, args)), flush=True)
        elif args.action in {'sync', 'restore'}:
            receipt = sync(args.repo, entry, manifest, args)
            if receipt:
                print(json.dumps(receipt), flush=True)
        elif args.action == 'offload':
            require_committed_offload_snapshot(args.repo, entry['id'])
            offload(args.repo, entry, manifest, args)
        else:
            missing = [x['path'] for x in manifest['files'] if not (args.repo / entry['path'] / x['path']).is_file()]
            print(json.dumps({'id': entry['id'], 'status': 'missing' if missing else ('verified' if current_receipt(args.repo, manifest) else 'present_unverified'),
                              'missing': len(missing), 'bytes': manifest['bytes'], 'admission': entry['admission']}))


if __name__ == '__main__':
    main()
