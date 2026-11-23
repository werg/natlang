"""Lossless streaming archives of opaque stopped-instance state, never training inputs."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile

ROOT_NAME = re.compile(r'retained-state-[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
SCHEMA = 'vibestudio.retained-state-pack/1'


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root, hashes=True):
    """Do not follow symlinks, including links to state outside this capture."""
    root = Path(root)
    rows = []
    inodes = {}
    def visit(path):
        info = path.lstat()
        row = {'path': path.relative_to(root.parent).as_posix(), 'mode': stat.S_IMODE(info.st_mode),
               'uid': info.st_uid, 'gid': info.st_gid, 'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns}
        if stat.S_ISDIR(info.st_mode):
            row['kind'] = 'directory'
        elif stat.S_ISLNK(info.st_mode):
            row.update(kind='symlink', link=os.readlink(path))
        elif stat.S_ISREG(info.st_mode):
            row.update(kind='file', bytes=info.st_size)
            if hashes:
                row['sha256'] = digest_file(path)
            inode = (info.st_dev, info.st_ino)
            if info.st_nlink > 1 and inode in inodes:
                row['hardlink_to'] = inodes[inode]
            inodes.setdefault(inode, row['path'])
        else:
            raise ValueError(f'opaque state contains unsupported special member: {path}')
        rows.append(row)
        if row['kind'] == 'directory':
            for child in sorted(path.iterdir()):
                visit(child)
    visit(root)
    return rows


def stable_rows(rows):
    return [{key: value for key, value in row.items() if key != 'sha256'} for row in rows]


def source_differences(expected, actual):
    """Report exact source drift; this never substitutes for the member proof."""
    before = {row['path']: row for row in stable_rows(expected)}
    after = {row['path']: row for row in stable_rows(actual)}
    changes = []
    for name in sorted(before.keys() | after.keys()):
        left, right = before.get(name), after.get(name)
        if left == right:
            continue
        fields = ({key: {'expected': left.get(key), 'actual': right.get(key)}
                   for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)}
                  if left is not None and right is not None else
                  {'member': {'expected': left, 'actual': right}})
        changes.append({'path': name, 'fields': fields})
    return changes


def validate_pack(manifest, directory, root=None):
    if manifest.get('schema') != SCHEMA or not ROOT_NAME.fullmatch(manifest.get('root', '')):
        raise ValueError('invalid opaque state pack manifest')
    archive_name = manifest.get('archive', '')
    if Path(archive_name).name != archive_name or not archive_name.startswith(manifest['root'] + '.') or not archive_name.endswith('.tar.gz'):
        raise ValueError('invalid opaque state archive name')
    archive = Path(directory) / archive_name
    if digest_file(archive) != manifest.get('sha256'):
        raise ValueError('opaque state archive SHA mismatch')
    expected = manifest.get('members')
    if not isinstance(expected, list) or not expected:
        raise ValueError('opaque state pack has no member proof')
    expected = {row['path']: row for row in expected}
    if len(expected) != len(manifest['members']):
        raise ValueError('opaque state member proof repeats a path')
    for name, row in expected.items():
        parts = Path(name).parts
        if not parts or parts[0] != manifest['root'] or name.startswith('/') or '..' in parts:
            raise ValueError('opaque state member escaped root')
        for index in range(1, len(parts)):
            if expected.get('/'.join(parts[:index]), {}).get('kind') != 'directory':
                raise ValueError('opaque state member has a non-directory ancestor')
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            row = expected.get(member.name)
            if not row or member.name in seen or member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError('unexpected opaque state archive member')
            if member.name != manifest['root'] and not member.name.startswith(manifest['root'] + '/'):
                raise ValueError('opaque state archive member escaped root')
            seen.add(member.name)
            if member.mode != row['mode'] or member.uid != row['uid'] or member.gid != row['gid']:
                raise ValueError('opaque state member metadata mismatch')
            if member.pax_headers.get('retained.mtime_ns') != str(row['mtime_ns']):
                raise ValueError('opaque state member timestamp mismatch')
            if row['kind'] == 'directory':
                valid = member.isdir()
            elif row['kind'] == 'symlink':
                valid = member.issym() and member.linkname == row['link']
            else:
                valid = (member.islnk() and member.linkname == row.get('hardlink_to')) if row.get('hardlink_to') else member.isfile()
                if valid and member.islnk():
                    target = expected.get(member.linkname, {})
                    valid = member.linkname in seen and target.get('kind') == 'file' and not target.get('hardlink_to') and target.get('sha256') == row['sha256'] and target.get('bytes') == row['bytes']
                elif valid:
                    value = tar.extractfile(member)
                    h, size = hashlib.sha256(), 0
                    for chunk in iter(lambda: value.read(1024 * 1024), b''):
                        h.update(chunk); size += len(chunk)
                    valid = size == row['bytes'] and h.hexdigest() == row['sha256']
            if not valid:
                raise ValueError('opaque state member content/type mismatch')
    if seen != set(expected):
        raise ValueError('opaque state archive omitted a member')
    if root is not None:
        actual = inventory(root, hashes=False)
        changes = source_differences(manifest['members'], actual)
        if changes and all(set(change['fields']) == {'ctime_ns'} for change in changes):
            # ctime is a race detector, not portable archive data: adding an
            # external hardlink to an immutable CAS blob changes it without
            # changing a captured byte or metadata we restore. Prove identity
            # again instead of treating that event as lost recovery state.
            by_name = {row['path']: row for row in manifest['members']}
            for change in changes:
                row = by_name[change['path']]
                if row['kind'] != 'file':
                    continue
                path = Path(root).parent / row['path']
                before = path.lstat()
                actual_hash = digest_file(path)
                after = path.lstat()
                identity = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_uid,
                                         info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                if identity(before) != identity(after) or actual_hash != row['sha256']:
                    raise ValueError('opaque state source changed during content revalidation: ' + row['path'])
            confirmed = inventory(root, hashes=False)
            if stable_rows(confirmed) == stable_rows(actual):
                changes = []
        if changes:
            raise ValueError('opaque state source changed after packing: ' +
                             json.dumps({'changed_members': len(changes), 'sample': changes[:8]}, sort_keys=True))
    return archive


def pack_root(root, directory):
    root, directory = Path(root), Path(directory)
    if not ROOT_NAME.fullmatch(root.name) or root.is_symlink() or not root.is_dir():
        raise ValueError('only actual opaque retained-state roots may be packed')
    directory.mkdir(exist_ok=True)
    for pattern in (root.name + '.building-*', root.name + '.manifest-*'):
        for abandoned in directory.glob(pattern):
            abandoned.unlink()
    manifest_path = directory / (root.name + '.json')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_pack(manifest, directory, root)
        return manifest
    members = inventory(root)
    descriptor, temporary = tempfile.mkstemp(prefix=root.name + '.building-', dir=directory)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as output:
            with gzip.GzipFile(fileobj=output, mode='wb', compresslevel=1, mtime=0, filename='') as zipped:
                with tarfile.open(fileobj=zipped, mode='w|', dereference=False, format=tarfile.PAX_FORMAT) as tar:
                    for row in members:
                        path = root.parent / row['path']
                        info = tar.gettarinfo(str(path), arcname=row['path'])
                        info.pax_headers['retained.mtime_ns'] = str(row['mtime_ns'])
                        if info.isfile():
                            with path.open('rb') as stream:
                                tar.addfile(info, stream)
                        else:
                            tar.addfile(info)
            output.flush(); os.fsync(output.fileno())
        digest = digest_file(temporary)
        name = root.name + '.' + digest + '.tar.gz'
        archive = directory / name
        temporary.replace(archive)
        manifest = {'schema': SCHEMA, 'root': root.name, 'archive': name, 'sha256': digest, 'members': members}
        validate_pack(manifest, directory, root)
        descriptor, staged = tempfile.mkstemp(prefix=root.name + '.manifest-', dir=directory)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(manifest, stream, sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            Path(staged).replace(manifest_path)
            directory_fd = os.open(directory, os.O_DIRECTORY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            Path(staged).unlink(missing_ok=True)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def prepare_retained_state_packs(capture):
    """Return exact anchored rsync exclusions only after archive/member/source proof."""
    if os.environ.get('VIBESTUDIO_PACK_RETAINED_STATE', '1') == '0':
        return []
    capture = Path(capture)
    roots = [path for path in sorted(capture.iterdir()) if ROOT_NAME.fullmatch(path.name) and not path.is_symlink() and path.is_dir()]
    if not roots:
        return []
    directory = capture / '.retained-state-packs'
    directory.mkdir(exist_ok=True)
    with (directory / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for root in roots:
            pack_root(root, directory)
    return ['/' + root.name + '/' for root in roots]


def restore_pack(manifest, directory, destination):
    """Restore a validated archive into a new root, preserving nanosecond mtimes."""
    archive = validate_pack(manifest, directory)
    destination = Path(destination)
    root = destination / manifest['root']
    if root.exists() or root.is_symlink():
        raise ValueError('opaque state restore destination already exists')
    destination.mkdir(parents=True, exist_ok=True)
    # Validation forbids escaping members and any descendant of a symlink/file;
    # retained symlink targets themselves are opaque and must remain exact.
    with tarfile.open(archive, 'r:gz') as tar:
        tar.extractall(destination, filter='fully_trusted')
    for row in reversed(manifest['members']):
        path = destination / row['path']
        os.utime(path, ns=(row['mtime_ns'], row['mtime_ns']), follow_symlinks=False)
    return root
