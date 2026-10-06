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


def manifest_path(repo, identity):
    if '/' in identity or identity in {'.', '..'}:
        raise ValueError('corpus id must be one path component')
    return repo / 'training/corpus-manifests' / (identity + '.json')


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


def verify(repo, manifest, receipt_group='corpus-receipts'):
    root = repo / relative(manifest['path'])
    errors, checked, stats = [], [], {}
    for item in manifest['files']:
        p = root / relative(item['path'])
        if not p.is_file() or p.stat().st_size != item['bytes']:
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
    receipt = {'id': manifest['id'], 'status': 'failed' if errors else 'verified', 'checked': len(checked),
               'manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
               'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'errors': errors, 'file_stats': stats}
    save(repo / '.coordination' / receipt_group / (manifest['id'] + '.json'), receipt)
    if errors:
        raise ValueError(json.dumps(receipt))
    print(json.dumps({k: v for k, v in receipt.items() if k != 'file_stats'}), flush=True)


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
        subprocess.run(['rsync', '-rz', '--ignore-existing', '--protect-args', '--partial', '--partial-dir=.sync-partial',
                        '--files-from=' + file_list.name, '--stats', source, destination], check=True)
    if pull:
        verify(repo, manifest, receipt_group='corpus-restores' if selected else 'corpus-receipts')
    else:
        remote_script = "import json,sys; from pathlib import Path; from sync_training_corpora import verify; verify(Path(sys.argv[1]),json.load(sys.stdin))"
        command = f'cd {shlex.quote(args.remote_repo + "/scripts")} && python3 -c {shlex.quote(remote_script)} {shlex.quote(args.remote_repo)}'
        subprocess.run(['ssh', args.host, command], input=json.dumps(manifest), text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['publish', 'verify', 'sync', 'restore', 'status'])
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--machine', choices=['pop', 'dgx'], required=True)
    parser.add_argument('--id', action='append', dest='ids')
    parser.add_argument('--file', action='append', dest='files', help='restore only named manifest paths; does not grant a full-snapshot receipt')
    parser.add_argument('--host', default='dgx')
    parser.add_argument('--remote-repo', default='/home/werg/natlang')
    parser.add_argument('--reserve-gib', type=float, default=8)
    args = parser.parse_args()
    if args.files and (args.action != 'restore' or not args.ids or len(args.ids) != 1):
        parser.error('--file requires restore and exactly one --id')
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
        elif args.action in {'sync', 'restore'}:
            sync(args.repo, entry, manifest, args)
        else:
            missing = [x['path'] for x in manifest['files'] if not (args.repo / entry['path'] / x['path']).is_file()]
            print(json.dumps({'id': entry['id'], 'status': 'missing' if missing else ('verified' if current_receipt(args.repo, manifest) else 'present_unverified'),
                              'missing': len(missing), 'bytes': manifest['bytes'], 'admission': entry['admission']}))


if __name__ == '__main__':
    main()
