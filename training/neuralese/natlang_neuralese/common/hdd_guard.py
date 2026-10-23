"""Keep training-step I/O off the slow external HDD (DGX).

Data inputs (``STAGED_ROLES``: the corpora a stage reads, again at every resume) that resolve (realpath) under the
HDD prefix are copied once to an NVMe cache keyed by SHA-256 and the staged path is passed to the stage; the declared
hash is checked on the copy. Load-once state (heads, ``continue_from`` and its sidecars, student/warm-up checkpoints:
read once at startup, so a copy buys no step speed, only disk churn; the 44 GB v3 warm-up state was copied back from
the archive) is read directly from the HDD and recorded as ``read_from_hdd``. The cache is bounded
(NATLANG_STAGED_INPUTS_MAX_GB, default 64): least recently used entries are evicted before a new copy, and the
archiver or storage retention may delete any entry (a stage re-stages what it needs). The out dir (checkpoints, logs)
must not resolve under the HDD. A no-op where the prefix is absent or its device is known to be
nonrotating (Pop's `/srv/storage` is an NVMe partition). The prefix comes from
NATLANG_HDD_PREFIX (default: the data_hdd root of common.paths); the cache from NATLANG_STAGED_INPUTS (default
~/data/staged-inputs).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

from .paths import resolve

RECORDS: list[dict] = []
# Native graph/rollout corpora are streamed during a run and on resume, so stage
# them once to local storage when their source lives on the external HDD. Other
# roles are small configuration, proofs, or load-once state and stay in place.
STAGED_ROLES = frozenset({'native_graphs', 'held_native_graphs', 'continued_native_graphs',
                          'continued_held_native_graphs', 'foundation_native_graphs', 'rollouts',
                          'rl_block_bundle', 'pending_native_sources'})
_STAGE_LOCKS: dict[str, threading.Lock] = {}
_STAGE_LOCKS_GUARD = threading.Lock()


def hdd_prefix() -> Path | None:
    explicit = os.environ.get('NATLANG_HDD_PREFIX')
    prefix = Path(explicit or resolve('data_hdd'))
    if not prefix.exists():
        return None
    prefix = prefix.resolve()
    # A machine may use its logical data_hdd root for a second NVMe partition.
    # Keep an explicit prefix authoritative, but avoid staging from that NVMe
    # back onto itself when the logical root is only a storage location name.
    if not explicit and _rotational(prefix) is False:
        return None
    return prefix


def _rotational(path: Path) -> bool | None:
    """Read the mounted block device's rotation flag; unknown keeps old behavior."""
    try:
        device = path.stat().st_dev
        sys_device = Path(f'/sys/dev/block/{os.major(device)}:{os.minor(device)}').resolve()
        for candidate in (sys_device, *sys_device.parents):
            flag = candidate / 'queue' / 'rotational'
            if flag.is_file():
                return flag.read_text().strip() == '1'
    except OSError:
        pass
    return None


def on_hdd(path) -> bool:
    prefix = hdd_prefix()
    if prefix is None:
        return False
    real = Path(os.path.realpath(path))
    return real == prefix or prefix in real.parents


def require_nvme(path, what='output directory') -> None:
    if on_hdd(path):
        raise ValueError(f'the {what} {path} resolves to the external HDD; use a directory on the NVMe')


def cache_dir() -> Path:
    return Path(os.environ.get('NATLANG_STAGED_INPUTS', Path.home() / 'data' / 'staged-inputs'))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def input_path(role: str, path, sha256: str | None = None, dependencies=None) -> str:
    """The path a stage reads for input ``role``: data roles staged (``stage_input``), load-once state in place."""
    if role in STAGED_ROLES:
        return stage_input(path, sha256, dependencies)
    if on_hdd(path):
        RECORDS.append({'role': role, 'source': os.path.realpath(path), 'read_from_hdd': True,
                        'reason': 'load-once state: read once at startup, staging buys no step speed'})
    return str(path)


def _evict(cache: Path, incoming: int, *, exclude: Path | None = None) -> None:
    """Least recently used cache entries out until ``incoming`` more bytes fit the bound."""
    bound = int(float(os.environ.get('NATLANG_STAGED_INPUTS_MAX_GB', '64')) * 2 ** 30)
    if not cache.is_dir():
        return
    entries = []
    for file in cache.rglob('*'):
        if not file.is_file() or '.locks' in file.parts or exclude is not None and file.is_relative_to(exclude):
            continue
        try:
            state = file.stat()
        except FileNotFoundError:
            continue
        entries.append((state.st_mtime, state.st_size, file))
    used = sum(size for _, size, _ in entries)
    for _, size, path in sorted(entries):
        if used + incoming <= bound:
            break
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != cache:
            try: parent.rmdir()
            except OSError: break
            parent = parent.parent
        used -= size


def _tree_files(root: Path) -> list[Path]:
    """Return an exact directory bundle's regular files, rejecting unsafe symlinks and special files."""
    files = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError(f'directory input symlink escapes its exact bundle: {path}')
            if not resolved.is_file():
                raise ValueError(f'directory input contains a non-file symlink: {path}')
            files.append(path)
        elif path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ValueError(f'directory input contains a special file: {path}')
    if not files:
        raise ValueError(f'directory input is empty: {root}')
    return files


def _directory_sha256(root: Path, files: list[Path] | None = None) -> str:
    """Stable SHA-256 over a directory's relative names, byte counts, and actual file hashes."""
    files = files if files is not None else _tree_files(root)
    records = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        records.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': _sha(path)})
    raw = json.dumps({'schema': 'natlang.hdd-staged-directory/1', 'files': records},
                     sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def _manifest_refs(path: Path) -> list[dict]:
    """Read only executable relative dependencies from the current native source and RL bundle manifests."""
    if path.suffix != '.json' and path.name != 'manifest.json':
        return []
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    schema = value.get('schema') if isinstance(value, dict) else None
    refs = []
    if schema == 'natlang.native-graph-sources/1':
        for source in value.get('sources') or []:
            if isinstance(source, dict) and isinstance(source.get('path'), str):
                refs.append({'path': source['path'], 'sha256': source.get('sha256')})
    elif schema == 'natlang.native-pending-sources/1':
        for source in value.get('sources') or []:
            if not isinstance(source, dict): continue
            for key in ('full_cohort', 'pending_native_graphs'):
                binding = source.get(key)
                if isinstance(binding, dict): refs.append(binding)
            refs.extend(item for item in source.get('evidence') or [] if isinstance(item, dict))
    elif schema == 'natlang.rl-block-bundle/1':
        for entry in (value.get('blocks') or {}).values():
            if isinstance(entry, dict) and isinstance(entry.get('path'), str):
                refs.append({'path': entry['path'], 'sha256': entry.get('sha256'), 'bytes': entry.get('bytes')})
    return refs


def _input_closure(path: Path, sha256: str | None, dependencies=None) -> tuple[list[dict], Path, str]:
    """Bind the primary file plus all manifest leaves, keeping source-relative paths unchanged."""
    primary = Path(os.path.abspath(path))
    if primary.is_dir():
        root = primary.parent
        files = _tree_files(primary)
        declared_tree_sha = _directory_sha256(primary, files)
        if sha256 is not None and sha256 != declared_tree_sha:
            raise ValueError(f'directory input {primary} does not match its declared SHA-256')
        members = {Path(os.path.abspath(item)): {'sha256': _sha(item), 'bytes': item.stat().st_size}
                   for item in files}
        root_digest = declared_tree_sha
        primary_is_dir = True
    else:
        if not primary.is_file():
            raise ValueError(f'staged input is neither a regular file nor a directory: {primary}')
        root = primary.parent
        actual = _sha(primary)
        if sha256 is not None and actual != sha256:
            raise ValueError(f'input {primary} does not match its declared SHA-256')
        root_digest = sha256 or actual
        members = {primary: {'sha256': actual, 'bytes': primary.stat().st_size}}
        primary_is_dir = False

    pending = []
    seen_manifests = set()
    if not primary_is_dir:
        pending.append((primary, _manifest_refs(primary)))
    else:
        for item in members:
            pending.append((item, _manifest_refs(item)))
    pending.append((primary, dependencies or []))
    while pending:
        manifest_path, refs = pending.pop()
        logical = Path(os.path.abspath(manifest_path))
        if logical in seen_manifests:
            continue
        seen_manifests.add(logical)
        for ref in refs:
            if not isinstance(ref, dict) or not isinstance(ref.get('path'), str):
                raise ValueError(f'{logical} contains a malformed relative source binding')
            child = Path(ref['path'])
            if not child.is_absolute(): child = logical.parent / child
            child = Path(os.path.abspath(child))
            if child.is_dir():
                leaves = _tree_files(child)
                directory_digest = _directory_sha256(child, leaves)
                if not isinstance(ref.get('sha256'), str) or directory_digest != ref['sha256']:
                    raise ValueError(f'manifest directory dependency {child} does not match its declared SHA-256')
                if 'bytes' in ref and ref['bytes'] != sum(leaf.stat().st_size for leaf in leaves):
                    raise ValueError(f'manifest directory dependency {child} does not match its declared byte count')
                for leaf in leaves:
                    logical_leaf = Path(os.path.abspath(leaf))
                    digest = _sha(leaf)
                    members[logical_leaf] = {'sha256': digest, 'bytes': leaf.stat().st_size}
                    pending.append((logical_leaf, _manifest_refs(logical_leaf)))
                continue
            if not child.is_file():
                raise ValueError(f'manifest dependency is missing or not a file: {child}')
            digest = _sha(child)
            if not isinstance(ref.get('sha256'), str) or digest != ref['sha256']:
                raise ValueError(f'manifest dependency {child} does not match its declared SHA-256')
            size = child.stat().st_size
            if 'bytes' in ref and ref['bytes'] != size:
                raise ValueError(f'manifest dependency {child} does not match its declared byte count')
            previous = members.get(child)
            if previous and previous['sha256'] != digest:
                raise ValueError(f'conflicting immutable input bindings for {child}')
            members[child] = {'sha256': digest, 'bytes': size}
            pending.append((child, _manifest_refs(child)))

    common = Path(os.path.commonpath([str(path) for path in members]))
    if common.is_file(): common = common.parent
    if common == primary: common = primary.parent
    if primary_is_dir and common == primary: common = primary.parent
    return [{'path': item, **members[item]} for item in sorted(members)], common, root_digest


def _stage_files(primary: Path, members: list[dict], common: Path, digest: str, *, directory_input: bool) -> Path:
    cache = cache_dir()
    # The file digest already binds declared child names/hashes. Include the expanded
    # closure too, so conflicting runtime dependency inventories cannot share a cache key.
    closure = [{'path': Path(item['path']).relative_to(common).as_posix(), 'sha256': item['sha256'],
                'bytes': item['bytes']} for item in members]
    key_bytes = json.dumps({'root_sha256': digest, 'closure': closure}, sort_keys=True,
                           separators=(',', ':')).encode()
    key = digest if len(members) == 1 and not directory_input else hashlib.sha256(key_bytes).hexdigest()
    entry = cache / key
    tree = entry / 'tree'
    if directory_input:
        staged_input = tree / primary.relative_to(common)
    elif len(members) == 1:
        staged_input = entry / primary.name
    else:
        staged_input = tree / primary.relative_to(common)
    lock_dir = cache / '.locks'
    lock_dir.mkdir(parents=True, exist_ok=True)
    with _STAGE_LOCKS_GUARD:
        lock = _STAGE_LOCKS.setdefault(key, threading.Lock())
    lock_path = lock_dir / f'{key}.lock'
    with lock, open(lock_path, 'a+b') as process_lock:
        fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX)
        if all((entry / ('tree' / Path(item['path']).relative_to(common) if len(members) > 1 or directory_input
                         else Path(item['path']).name)).is_file() and
               _sha(entry / ('tree' / Path(item['path']).relative_to(common) if len(members) > 1 or directory_input
                             else Path(item['path']).name)) == item['sha256'] for item in members):
            for item in members:
                cached = entry / ('tree' / Path(item['path']).relative_to(common) if len(members) > 1 or directory_input
                                  else Path(item['path']).name)
                os.utime(cached)
            return staged_input
        incoming = sum(item['bytes'] for item in members)
        _evict(cache, incoming, exclude=entry)
        entry.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f'.{key}.part-', dir=cache))
        try:
            for item in members:
                source = Path(item['path'])
                relative = source.relative_to(common) if len(members) > 1 or directory_input else Path(source.name)
                target = (temp / 'tree' / relative) if len(members) > 1 or directory_input else temp / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if target.stat().st_size != item['bytes'] or _sha(target) != item['sha256']:
                    raise ValueError(f'staged copy of {source} does not match its exact SHA-256 and size')
            if entry.exists(): shutil.rmtree(entry)
            os.replace(temp, entry)
        except BaseException:
            shutil.rmtree(temp, ignore_errors=True)
            raise
    return staged_input


def stage_input(path, sha256: str | None = None, dependencies=None) -> str:
    """Stage a file, a directory bundle, or a manifest and its transitive relative-file closure."""
    logical = Path(os.path.abspath(path))
    refs = [] if logical.is_dir() else _manifest_refs(logical)
    if not logical.is_dir() and not on_hdd(logical) and not dependencies and not refs:
        return str(path)
    if logical.is_symlink() and not dependencies and not refs:
        logical = Path(os.path.realpath(logical))
    members, common, digest = _input_closure(logical, sha256, dependencies)
    if not any(on_hdd(item['path']) for item in members):
        return str(path)
    staged = _stage_files(logical, members, common, digest, directory_input=logical.is_dir())
    entry = staged.parents[1] if logical.is_dir() or len(members) > 1 else staged.parent
    def cached_member(item):
        source = Path(item['path'])
        if logical.is_dir() or len(members) > 1:
            return str(entry / 'tree' / source.relative_to(common))
        return str(entry / source.name)
    RECORDS.append({'source': str(logical), 'staged': str(staged), 'sha256': digest,
                    'declared_hash': sha256 is not None,
                    'dependencies': [{'source': str(item['path']), 'staged': cached_member(item),
                                      'sha256': item['sha256']} for item in members
                                     if logical.is_dir() or Path(item['path']) != logical]})
    return str(staged)
