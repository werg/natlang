"""The training-data registry: every dataset we have, what produced it, what it derives from, and which runs read it.

One registry, two layers, one id namespace (plans/neuralese/TRAINING_DATA_DISCOVERABILITY_20261120.md):

- **Git layer**: ``training/corpora`` with immutable ``training/corpus-manifests/<id>.json`` snapshots
  (``scripts/sync_training_corpora.py`` publishes and syncs them, ``register_corpora.py`` commits them unattended). It
  is pinned with the code, so what a pinned run *selects* comes from here: ``roles`` (``grpo-pool``,
  ``grpo-memetic-tasks``, ``grpo-sd-teacher``, ``held-eval``) replace the hand-edited lists in ``grpo.py``.
- **Journal layer**: an append-only ``journal.jsonl`` under ``<data_nvme>/data-registry`` (``NATLANG_DATA_REGISTRY``),
  machine-wide, independent of the checkout (pins write to it too). Producers append ``dataset`` records (a snapshot:
  manifest sha, files, rows, format, producer commit, inputs, license, protected, supersedes) and ``files`` records
  (append-only directories such as rollout archives); consumers append ``consume`` records (run, dataset, sha of what
  was read, how, amount). Manifests are content addressed under ``manifests/<sha256>.json``.

``load()`` merges both layers. ``consume()`` refuses a path that no dataset of either layer contains
(``UnregisteredData``; ``NATLANG_ALLOW_UNREGISTERED_DATA=1`` opts out, and the opt-out is itself recorded), so data
cannot reach a learner without an entry. ``promote()`` (``scripts/data_registry.py promote``) folds closed journal
snapshots into the git layer.

Paths are portable: ``<root>:<relative>`` for the logical machine roots of ``common.paths`` (``data_nvme:...``,
``data_hdd:...``) and plain relative paths for the canonical checkout (``runs/...``, as the git layer always used).
Stdlib only.
"""
from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from .jsonio import json_line, utf8
from .paths import root

RECORD_SCHEMA = 'natlang.data-registry-record/1'
SNAPSHOT_SCHEMA = 'natlang.corpus-snapshot/1'
from . import corpus_registry  # noqa: E402  (the git layer: one file per entry)
GIT_MANIFESTS = 'training/corpus-manifests'
PORTABLE_ROOTS = ('data_nvme', 'data_hdd', 'archive', 'models')
ALLOW_ENV = 'NATLANG_ALLOW_UNREGISTERED_DATA'
# Kinds a dataset record may carry. Producers name one; the catalog groups by it.
KINDS = ('rl-tasks', 'held', 'held-eval', 'teacher-demos', 'trajectories', 'rollouts', 'step-chains', 'sft-rows',
         'chains', 'index', 'corpus', 'edits', 'probe', 'archive')
# An `archive` (checkpoints, moved or retired trees) is registered as an inventory: per-directory file counts and bytes
# (`stat-tree`), no content digests. Data inside one is discoverable but not admitted: a consumer refuses it until it is
# recorded as a dataset of its own.
INVENTORY = 'inventory-only'
SKIP_DIRS = {'node_modules', '.git', '__pycache__'}


class UnregisteredData(RuntimeError):
    """A consumer was about to read data no registry entry contains."""


def machine() -> str:
    """The registry's owner name of this machine (``owner`` in the git layer): ``pop`` or ``dgx``."""
    return os.environ.get('NATLANG_MACHINE') or ('pop' if 'pop' in socket.gethostname() else 'dgx')


def registry_dir() -> Path:
    override = os.environ.get('NATLANG_DATA_REGISTRY')
    return Path(override) if override else root('data_nvme') / 'data-registry'


def assert_registry_writable(path) -> Path:
    """Refuse registry mutations anywhere inside a marked immutable code pin.

    Resolve links and inspect every ancestor before mkdir, Git fetch/sync, entry,
    manifest or symlink changes. An invalid/dangling marker is still a pin.
    """
    resolved = Path(path).expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        marker = parent / '.natlang-pin.json'
        if marker.exists() or marker.is_symlink():
            raise ValueError(f'registry writes cannot mutate immutable code pin {parent}; '
                             'select the canonical registry owner')
    return resolved


def canonical_checkout_repo(repo=None) -> Path:
    """The mutable primary checkout for a code repository, never another pin.

    Git's common worktree registration remains authoritative after an earlier
    pin in a marker.source chain has been retired. Marker-only directory views
    follow and validate every source hop rather than stopping after one pin.
    """
    current = Path(repo if repo is not None else code_repo()).expanduser().resolve()
    visited = set()
    while True:
        if current in visited:
            raise ValueError('cyclic immutable pin source chain')
        visited.add(current)
        if not current.is_dir():
            raise ValueError(f'immutable pin source checkout is unavailable: {current}')
        git_marker = current / '.git'
        if git_marker.exists() or git_marker.is_symlink():
            env = {key: value for key, value in os.environ.items()
                   if key not in {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE'}}
            result = subprocess.run(['git', '-C', str(current), 'worktree', 'list', '--porcelain', '-z'],
                                    capture_output=True, text=True, env=env, check=True)
            records = [dict(field.split(' ', 1) if ' ' in field else (field, '')
                            for field in record.split('\0') if field)
                       for record in result.stdout.split('\0\0') if record]
            primary = next((row['worktree'] for row in records if 'worktree' in row and 'bare' not in row), None)
            if primary is None:
                raise ValueError('Git repository has no mutable primary worktree')
            return assert_registry_writable(primary)
        marker = current / '.natlang-pin.json'
        if not marker.exists() and not marker.is_symlink():
            return assert_registry_writable(current)
        metadata = json.loads(marker.read_text())
        source = metadata.get('source')
        if not isinstance(source, str) or not source or not Path(source).is_absolute():
            raise ValueError(f'immutable pin lacks an absolute source checkout: {current}')
        current = Path(source).expanduser().resolve()


def registry_owner_repo() -> Path:
    """Default Git registry WRITE owner; code identity and pinned reads stay separate."""
    explicit = os.environ.get('NATLANG_REGISTRY_REPO')
    return assert_registry_writable(explicit) if explicit else canonical_checkout_repo(code_repo())


def _registry_ref_commit(repo, ref: str) -> str:
    result = subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', '--end-of-options', f'{ref}^{{commit}}'],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'registry ref {ref!r} is unavailable in {repo}; fetch the registry owner first')
    return result.stdout.strip()


def registry_owner_ref(repo=None) -> str | None:
    """Resolve one authoritative LIVE registry snapshot, never silently stale HEAD.

    A published owner follows origin/main. HEAD is allowed only for an actual
    local Git registry without an origin. The returned OID freezes one scan or
    build; the next scan resolves the publication ref again.
    """
    owner = assert_registry_writable(repo) if repo is not None else registry_owner_repo()
    configured = os.environ.get('NATLANG_REGISTRY_REF')
    if configured is not None:
        if not configured: raise ValueError('NATLANG_REGISTRY_REF must name a published registry ref')
        return _registry_ref_commit(owner, configured)
    git_marker = owner / '.git'
    if not git_marker.exists() and not git_marker.is_symlink():
        return None  # explicit local filesystem registry / dry-run; no publication is claimed
    has_origin = subprocess.run(['git', '-C', str(owner), 'config', '--get', 'remote.origin.url'],
                                capture_output=True, text=True).returncode == 0
    if has_origin:
        return _registry_ref_commit(owner, 'origin/main')
    return _registry_ref_commit(owner, 'HEAD')


def canonical_repo() -> Path:
    """Mutable data anchor independent of an immutable code pin or registry override."""
    return canonical_checkout_repo(root('repo'))


def code_repo() -> Path:
    """The checkout this code runs from (a pin or main): its git layer is the one pinned with the code."""
    return Path(__file__).resolve().parents[4]


# ---------- paths ----------

def portable(path) -> str:
    """``root:rel`` for a path under a logical data root, a plain relative path under the canonical checkout."""
    p = Path(os.path.abspath(path))
    repo = canonical_repo()
    try:
        return str(p.relative_to(repo))
    except ValueError:
        pass
    best = None
    for name in PORTABLE_ROOTS:
        base = root(name)
        try:
            rel = p.relative_to(base)
        except ValueError:
            continue
        if best is None or len(base.parts) > len(best[1].parts):
            best = (name, base, rel)
    if best is None:
        if os.environ.get(ALLOW_ENV) == '1':     # the strictness opt-out (tests, scratch): an absolute, local name
            return str(p)
        raise ValueError(f'{path} is under no logical data root ({", ".join(PORTABLE_ROOTS)}) or the checkout')
    return f'{best[0]}:{best[2]}' if str(best[2]) != '.' else f'{best[0]}:'


def run_name(prefix: str, path) -> str:
    """A run's name in consumption records: PREFIX plus the run directory's portable path (absolute outside the roots)."""
    try:
        return f'{prefix}:{portable(path)}'
    except ValueError:
        return f'{prefix}:{os.path.abspath(path)}'


def dataset_id(path, kind: str) -> str:
    """The id a producer gives the dataset it writes at PATH: the directory name for a top-level directory of a data
    root or of the checkout's ``runs/`` (as the git layer always named them), else ``path_id``."""
    p = Path(os.path.abspath(path))
    if p.parent in {root(n) for n in PORTABLE_ROOTS} | {canonical_repo() / 'runs'}:
        return p.name
    return path_id(kind, p)


def path_id(kind: str, path) -> str:
    """The id of a dataset named by where it lives (a run's rollout archive, a held eval): KIND plus its portable path
    as one path component."""
    import re
    try:
        where = portable(path).split(':', 1)[-1]
    except ValueError:
        where = os.path.abspath(path)
    return kind + '-' + re.sub(r'[^A-Za-z0-9._-]+', '-', where).strip('-')


def absolute(value: str, repo=None) -> Path:
    """The local path of a portable registry path."""
    head, sep, rel = value.partition(':')
    if sep and head in PORTABLE_ROOTS + ('repo',):
        return root(head) / rel if rel else root(head)
    if os.path.isabs(value) and os.environ.get(ALLOW_ENV) == '1':
        return Path(value)
    if os.path.isabs(value) or '..' in Path(value).parts:
        raise ValueError(f'registry paths are portable (root:rel or checkout-relative), not {value!r}')
    return Path(repo or canonical_repo()) / value


# ---------- file digests (cached by stat) ----------

_DIGESTS = None    # sqlite connection of the stat-keyed digest cache (one lookup per file, nothing held in memory)


def _digest_cache_path() -> Path:
    return registry_dir() / 'file-digests.sqlite'


def _digest_db():
    global _DIGESTS
    if _DIGESTS is None or _DIGESTS[0] != str(_digest_cache_path()):
        import sqlite3
        path = _digest_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(path), timeout=60, check_same_thread=False)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS digests (k TEXT PRIMARY KEY, v TEXT NOT NULL)')
        _DIGESTS = (str(path), db)
    return _DIGESTS[1]


def _stat_key(path: Path) -> str:
    real = os.path.realpath(path)
    s = os.stat(real)
    return f'{real}|{s.st_dev}|{s.st_ino}|{s.st_size}|{s.st_mtime_ns}'


def file_digest(path) -> dict:
    """``{bytes, sha256[, lines]}`` (lines for ``.jsonl``), the corpus-snapshot file fields; cached by stat."""
    path = Path(path)
    key = _stat_key(path)
    try:
        hit = _digest_db().execute('SELECT v FROM digests WHERE k = ?', (key,)).fetchone()
    except Exception:
        hit = None
    if hit:
        return json.loads(hit[0])
    h, lines, last, size = hashlib.sha256(), 0, b'', 0
    with open(path, 'rb') as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
            lines += block.count(b'\n')
            last = block[-1:]
            size += len(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or _stat_key(path) != key:
        raise RuntimeError(f'file changed while hashing: {path}')
    value = {'bytes': size, 'sha256': h.hexdigest()}
    if path.name.endswith('.jsonl'):
        value['lines'] = lines + int(bool(last) and last != b'\n')
    try:
        db = _digest_db()
        with db:
            db.execute('INSERT OR REPLACE INTO digests VALUES (?, ?)', (key, json.dumps(value)))
    except Exception:
        pass     # the cache only saves work
    return dict(value)


# ---------- journal ----------

def _append_line(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = utf8(json_line(value, sort_keys=True))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, data)
        os.fsync(fd)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def journal_path() -> Path:
    return registry_dir() / 'journal.jsonl'


def _producer(producer=None) -> dict:
    from .code_provenance import code_provenance
    value = {'script': os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None,
             'argv': list(sys.argv[1:])[:200], 'host': socket.gethostname(), 'pid': os.getpid(),
             'code': code_provenance()}
    value.update(producer or {})
    return value


def _append(record: dict) -> dict:
    record = {'schema': RECORD_SCHEMA, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'machine': machine(), **record}
    _append_line(journal_path(), record)
    global _LOADED
    _LOADED = None
    return record


def _store_manifest(manifest: dict) -> str:
    data = utf8(json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(',', ':')))
    sha = hashlib.sha256(data).hexdigest()
    path = registry_dir() / 'manifests' / f'{sha}.json'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_bytes(data)
        tmp.replace(path)
    return sha


def stored_manifest(sha: str) -> dict | None:
    try:
        return json.loads((registry_dir() / 'manifests' / f'{sha}.json').read_text())
    except (OSError, ValueError):
        return None


def _selected(rel: str, include, exclude) -> bool:
    if include and not any(fnmatch.fnmatch(rel, pat) for pat in include):
        return False
    return not any(fnmatch.fnmatch(rel, pat) for pat in exclude or ())


def _walk(base: Path, include=None, exclude=()):
    if base.is_file():
        return [base]
    files = []
    for directory, children, names in os.walk(base):   # like sync_training_corpora: no dir links
        children.sort()
        for name in sorted(names):
            p = Path(directory) / name
            rel = str(p.relative_to(base))
            if p.is_file() and not name.endswith('.tmp') and _selected(rel, include, exclude):
                files.append(p)
    return files


def snapshot(identity: str, path, include=None, exclude=()) -> dict:
    """A ``natlang.corpus-snapshot/1`` manifest of PATH (a directory or a file) as it is now."""
    base = absolute(path) if isinstance(path, str) and not os.path.isabs(path) else Path(path)
    if base.is_file():
        base, include = base.parent, [base.name]
    files = []
    for p in _walk(base, include, exclude):
        try:
            digest = file_digest(p)
        except FileNotFoundError:
            continue   # removed after the walk listed it (a live directory's transient file): it is not part of the snapshot
        files.append({'path': str(p.relative_to(base)), **digest})
    return {'schema': SNAPSHOT_SCHEMA, 'id': identity, 'owner': machine(), 'path': portable(base),
            'files': files, 'bytes': sum(f['bytes'] for f in files)}


def stat_tree(base: Path) -> list[dict]:
    """Per-directory inventory of BASE (recursive file count and bytes of each directory, deepest first two levels
    listed): what an archive holds without reading it."""
    totals: dict[str, list] = {}
    for directory, children, names in os.walk(base):
        children[:] = sorted(c for c in children if c not in SKIP_DIRS and not c.startswith('pin-'))
        rel = os.path.relpath(directory, base)
        parts = [] if rel == '.' else rel.split(os.sep)
        size = 0
        for name in names:
            try:
                size += os.lstat(os.path.join(directory, name)).st_size
            except OSError:
                pass
        for depth in range(0, min(len(parts), 2) + 1):
            key = '/'.join(parts[:depth]) or '.'
            t = totals.setdefault(key, [0, 0])
            t[0] += len(names)
            t[1] += size
    return [{'path': k, 'files': v[0], 'bytes': v[1]} for k, v in sorted(totals.items())]


def record_inventory(identity: str, path, *, note=None, kind='archive', extra=None) -> dict:
    """Register a tree as an inventory (``archive``): discoverable with its sizes, not admitted for training."""
    base = Path(path)
    tree = stat_tree(base)
    manifest = {'schema': 'natlang.data-registry-inventory/1', 'id': identity, 'owner': machine(),
                'path': portable(base), 'tree': tree}
    sha = _store_manifest(manifest)
    top = next((t for t in tree if t['path'] == '.'), {'files': 0, 'bytes': 0})
    record = {'op': 'dataset', 'id': identity, 'path': manifest['path'], 'kind': kind, 'format': 'inventory (stat-tree)',
              'digest': 'stat-tree', 'manifest_sha256': sha, 'files': top['files'], 'bytes': top['bytes'],
              'admission': INVENTORY, 'closed': False, 'note': note, 'producer': _producer()}
    record.update(extra or {})
    return _append({k: v for k, v in record.items() if v is not None})


def _input_refs(inputs, *, repo=None, ref=None) -> list:
    """Inputs as ``{id, manifest_sha256 | sha256, path?}``: ids stay ids (their latest snapshot), paths resolve to the
    dataset that holds them plus the file's own digest."""
    refs = []
    reg = load(repo=repo, ref=ref) if ref is not None else (load() if repo is None else load(repo=repo))
    for item in inputs or ():
        if isinstance(item, dict):
            refs.append(item)
            continue
        item = str(item)
        if item in reg.datasets:
            refs.append({'id': item, 'manifest_sha256': reg.datasets[item].get('manifest_sha256')})
            continue
        p = Path(item)
        if not p.exists():
            refs.append({'path': item, 'missing': True})
            continue
        ds = reg.find(p)
        ref = {'id': ds['id'] if ds else None, 'path': portable(p)}
        if p.is_file():
            ref['sha256'] = file_digest(p)['sha256']
        elif ds:
            ref['manifest_sha256'] = ds.get('manifest_sha256')
        refs.append(ref)
    return refs


def record_dataset(identity: str, path, *, kind: str, format: str, include=None, exclude=(), rows=None,
                   inputs=(), license=None, protected=False, supersedes=None, roles=None, note=None,
                   producer=None, admission=None, closed=True, extra=None, registry_repo=None, registry_ref=None) -> dict:
    """Append a dataset snapshot record (what a producer wrote). PATH is a directory (``include`` globs select files)
    or a single file. ``rows`` defaults to the line count of the selected ``.jsonl`` files. ``closed`` False marks a
    dataset that keeps growing (a teacher campaign, a refreshed source): every later snapshot is another record."""
    if '/' in identity or not identity or identity in {'.', '..'}:
        raise ValueError('dataset id must be one nonempty path component')
    if kind not in KINDS:
        raise ValueError(f'unknown dataset kind {kind!r}; known: {KINDS}')
    base = absolute(path) if isinstance(path, str) and not os.path.isabs(path) else Path(path)
    if base.is_file():
        base, include = base.parent, [base.name]
    manifest = snapshot(identity, base, include, exclude)
    if not manifest['files']:
        raise ValueError(f'empty dataset {identity}: no files at {path}')
    sha = _store_manifest(manifest)
    if rows is None:
        counts = [f['lines'] for f in manifest['files'] if 'lines' in f]
        rows = sum(counts) if counts else None
    record = {'op': 'dataset', 'id': identity, 'path': manifest['path'], 'kind': kind, 'format': format,
              'include': list(include) if include else None, 'exclude': list(exclude) if exclude else None,
              'manifest_sha256': sha, 'files': len(manifest['files']), 'bytes': manifest['bytes'], 'rows': rows,
              'inputs': _input_refs(inputs, repo=registry_repo, ref=registry_ref), 'license': license, 'protected': bool(protected),
              'supersedes': supersedes, 'roles': roles, 'note': note, 'admission': admission, 'closed': bool(closed),
              'producer': _producer(producer)}
    if extra:
        record.update(extra)
    return _append({k: v for k, v in record.items() if v is not None})


def declare_dataset(identity: str, path, *, kind: str, format: str, **fields) -> dict:
    """Announce a dataset that starts empty and grows file by file (a rollout archive, step chains): a dataset record
    without a manifest; ``record_files`` adds its files."""
    if kind not in KINDS:
        raise ValueError(f'unknown dataset kind {kind!r}')
    Path(path).mkdir(parents=True, exist_ok=True) if not Path(path).suffix else None
    fields = {k: v for k, v in fields.items() if v is not None}
    if 'inputs' in fields:
        fields['inputs'] = _input_refs(fields['inputs'])
    return _append({'op': 'dataset', 'id': identity, 'path': portable(path), 'kind': kind, 'format': format,
                    'closed': False, 'growing': 'files', **fields, 'producer': _producer(fields.pop('producer', None))})


def record_files(identity: str, paths, base) -> dict | None:
    """Append-only directories: record new files of dataset IDENTITY rooted at BASE (declared by ``declare_dataset``).
    Reads no registry, so it is cheap enough for a per-group hot path."""
    base = Path(base).resolve()
    files = [{'path': str(Path(p).resolve().relative_to(base)), **file_digest(p)} for p in paths]
    if not files:
        return None
    return _append({'op': 'files', 'id': identity, 'files': files})


def consume(items, *, run: str, how: str, amount=None, allow_unregistered=None, record=True, repo=None, ref=None) -> list[dict]:
    """Record that RUN read ITEMS (paths or dataset ids) and return the resolved references. A path no dataset contains
    raises ``UnregisteredData`` unless ``allow_unregistered`` (default: ``NATLANG_ALLOW_UNREGISTERED_DATA=1``)."""
    if allow_unregistered is None:
        allow_unregistered = os.environ.get(ALLOW_ENV) == '1'
    reg = load(repo=repo, ref=ref) if ref is not None else (load() if repo is None else load(repo=repo))
    refs, missing = [], []
    for item in items:
        if item is None:
            continue
        item = str(item)
        if item in reg.datasets:
            ds = reg.datasets[item]
            if ds.get('admission') == INVENTORY:
                missing.append(item)
                refs.append({'id': item, 'unregistered': True, 'inventory_only': True})
                continue
            refs.append({'id': item, 'manifest_sha256': ds.get('manifest_sha256')})
            continue
        p = Path(item)
        ds = reg.find(p)
        if ds is None or ds.get('admission') == INVENTORY:
            missing.append(item)
            refs.append({'id': ds and ds['id'], 'path': item, 'unregistered': True,
                         **({'inventory_only': True} if ds else {})})
            continue
        ref = {'id': ds['id'], 'path': portable(p) if p.exists() else item}
        if p.is_file():
            ref['sha256'] = file_digest(p)['sha256']
        elif ds.get('manifest_sha256'):
            ref['manifest_sha256'] = ds['manifest_sha256']
        refs.append(ref)
    if missing and not allow_unregistered:
        raise UnregisteredData(
            'not in the data registry: ' + ', '.join(missing) + '. Register it (producers do this automatically; by '
            'hand: scripts/data_registry.py record --id ID --path PATH --kind KIND --format ...) or set '
            f'{ALLOW_ENV}=1 to read it unregistered (recorded).')
    if record:
        _append({'op': 'consume', 'run': run, 'how': how, 'amount': amount, 'items': refs,
                 'allowed_unregistered': bool(missing), 'producer': _producer()})
    return refs


def require_registered(paths, *, repo=None, ref=None) -> None:
    """Producer-side check: refuse inputs no dataset contains (same opt-out as ``consume``)."""
    if os.environ.get(ALLOW_ENV) == '1':
        return
    reg = load(repo=repo, ref=ref) if ref is not None else (load() if repo is None else load(repo=repo))
    missing = []
    for path in paths:
        if path is None:
            continue
        ds = reg.datasets.get(str(path))
        if ds is None:
            ds = reg.find(Path(path))
        if ds is None or ds.get('admission') == INVENTORY:
            missing.append(str(path))
    if missing:
        raise UnregisteredData('not in the data registry: ' + ', '.join(missing))


# ---------- reading the merged registry ----------

class Registry:
    """Merged view: git entries (pinned) plus journal records (machine-wide)."""

    def __init__(self, git_entries: list, journal: list, *, repo=None, ref=None):
        self.repo = Path(repo) if repo is not None else code_repo()
        self.git_ref = ref
        self.git = {e['id']: e for e in git_entries}
        self.git_order = [e['id'] for e in git_entries]
        self.journal = journal
        self.datasets: dict[str, dict] = {}
        for e in git_entries:
            self.datasets[e['id']] = {**e, 'layer': 'git'}
        self.snapshots: dict[str, list] = {}
        self.extra_files: dict[str, list] = {}
        self.consumption: list = []
        for r in journal:
            op = r.get('op')
            if op == 'dataset':
                self.snapshots.setdefault(r['id'], []).append(r)
                base = self.datasets.get(r['id'], {})
                merged = {**base, **{k: v for k, v in r.items() if k not in ('schema', 'op')}}
                merged['layer'] = 'git+journal' if base.get('layer', '').startswith('git') else 'journal'
                if base.get('layer', '').startswith('git'):
                    merged['path'] = base['path']        # the pinned path wins; the journal adds facts
                    if base.get('roles'):
                        merged['roles'] = base['roles']
                self.datasets[r['id']] = merged
            elif op == 'files':
                self.extra_files.setdefault(r['id'], []).extend(r.get('files') or [])
            elif op == 'consume':
                self.consumption.append(r)
        self._index = None
        self._keymap = None

    # paths
    def _paths(self):
        if self._index is None:
            index = []
            self._roots = {}
            listings = {}

            def listdir(d):
                if d not in listings:
                    try:
                        listings[d] = os.listdir(d)
                    except OSError:
                        listings[d] = []
                return listings[d]
            for ident, ds in self._by_admission():
                try:
                    p = absolute(ds['path'])
                except (ValueError, KeyError):
                    continue
                include = ds.get('include') or []
                if ds.get('include_manifest'):      # its manifest lists the files; prefixes name where they live
                    include = [f'{d}/*' if d != '.' else '*' for d in ds.get('include_prefixes') or ['.']]
                # A dataset holds what its includes name, not its whole directory: an exact name is that file, a glob
                # the directory before its first wildcard (two datasets may share a directory, such as runs/)
                targets = []
                for pattern in include or ['']:
                    parts = Path(pattern).parts
                    fixed = []
                    for part in parts:
                        if any(c in part for c in '*?['):
                            break
                        fixed.append(part)
                    if len(fixed) < len(parts) and parts[len(fixed)] != '**':
                        # the first wildcard component names entries of a shared directory (runs/x-v*/...): claim the
                        # entries it matches, not the directory
                        base = p.joinpath(*fixed)
                        hits = [n for n in listdir(base) if fnmatch.fnmatch(n, parts[len(fixed)])]
                        targets += [base / n for n in hits] if len(fixed) + 1 < len(parts) or hits else []
                        continue
                    targets.append(p.joinpath(*fixed) if fixed else p)
                exact = bool(include) and p not in targets
                keys = set()
                real_root = os.path.realpath(p)     # once per dataset: resolving every named file is slow on the HDD
                for t in targets:
                    keys.add(os.path.normpath(str(t)))
                    keys.add(os.path.normpath(os.path.join(real_root, os.path.relpath(t, p))))
                for k in keys:
                    index.append((k, ident))
                if exact:     # the dataset's own directory names it too, but holds nothing else by prefix
                    for k in {os.path.normpath(str(p)), os.path.realpath(p)}:
                        self._roots.setdefault(k, ident)
            index.sort(key=lambda kv: -len(kv[0]))
            self._index = index
        return self._index

    def _by_admission(self):
        """Datasets with inventory-only entries last: a producer's record of a directory wins over the backfill's
        inventory placeholder for the same path (the 2026-11-21 04:20 backfill registered the then-empty ei-sources
        WORK as an inventory, which then refused sft-v7's build input although sft_sources had recorded it)."""
        return sorted(self.datasets.items(), key=lambda item: item[1].get('admission') == INVENTORY)

    def _lexical(self):
        """Declared literal/recursive prefixes, first-wildcard rules, and exact dataset roots. No directory I/O."""
        if getattr(self, '_lexical_index', None) is None:
            keys, patterns, roots = {}, [], {}
            self._lexical_order = {}
            for order, (ident, ds) in enumerate(self._by_admission()):
                self._lexical_order[ident] = order
                try:
                    p = absolute(ds['path'])
                except (ValueError, KeyError):
                    continue
                include = ds.get('include') or []
                if ds.get('include_manifest'):
                    include = [f'{d}/*' if d != '.' else '*' for d in ds.get('include_prefixes') or ['.']]
                own_root = os.path.normpath(str(p))
                for pattern in include or ['']:
                    parts = Path(pattern).parts
                    fixed = []
                    for part in parts:
                        if any(c in part for c in '*?['):
                            break
                        fixed.append(part)
                    base = os.path.normpath(str(p.joinpath(*fixed)))
                    if len(fixed) < len(parts) and parts[len(fixed)] != '**':
                        patterns.append((base, parts[len(fixed)], ident))
                    else:
                        keys.setdefault(base, ident)
                if include:
                    roots.setdefault(own_root, ident)
            self._lexical_index = keys, patterns, roots
        return self._lexical_index

    def find(self, path) -> dict | None:
        """Find the deepest declared owner without enumerating unrelated glob datasets.

        Includes have the same prefix semantics as ``_paths``: the first wildcard names owned directory members;
        ``**`` owns its fixed prefix. Only the requested member needs an existence check, not a directory listing.
        Alias-only paths still use the resolving full-index fallback when the declared spelling has no owner.
        """
        if os.environ.get('NATLANG_REGISTRY_SCOPED_GLOBS') == '0':
            return self._find_full(path)
        given = os.path.normpath(os.path.abspath(str(path)))
        keys, patterns, roots = self._lexical()
        matches = []
        probe = given
        while True:
            if probe in keys:
                matches.append((probe, keys[probe]))
                break
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        for base, wildcard, ident in patterns:
            if given == base or not _under(given, base):
                continue
            member = os.path.relpath(given, base).split(os.sep, 1)[0]
            target = os.path.join(base, member)
            if fnmatch.fnmatch(member, wildcard) and os.path.lexists(target):
                matches.append((target, ident))
        best = min(matches, key=lambda item: (-len(item[0]), self._lexical_order[item[1]])) if matches else None
        if given in roots and (best is None or best[0] != given or
                               self.datasets[best[1]].get('admission') == INVENTORY):
            return self.datasets[roots[given]]
        if best is not None:
            return self.datasets[best[1]]
        return self._find_full(path)

    def _temp_view(self, temp):
        if getattr(self, '_temp', None) is None:
            view = object.__new__(Registry)
            view.datasets = {}
            for ident, ds in self.datasets.items():
                try:
                    if any(_under(os.path.normpath(str(absolute(ds['path']))), t) for t in temp):
                        view.datasets[ident] = ds
                except (ValueError, KeyError):
                    continue
            view._index = view._keymap = None
            self._temp = view
        return self._temp

    def _find_full(self, path) -> dict | None:
        candidates = {os.path.normpath(os.path.abspath(str(path)))}
        try:
            candidates.add(os.path.realpath(path))
        except OSError:
            pass
        temp = _temp_roots()
        # A temporary path (test fixtures, scratch) can only lie in a dataset recorded in a temporary directory: no
        # portable (registered) path is temporary. The index over every registered directory lists and resolves them on
        # the HDD (minutes under load: test_extract_takes_teacher_successes_in_student_form hung three times).
        src = self._temp_view(temp) if all(any(_under(c, t) for t in temp) for c in candidates) else self
        src._paths()
        if src._keymap is None:
            src._keymap = {}
            for k, ident in src._index:
                src._keymap.setdefault(k, ident)
        best = None
        for c in candidates:      # the deepest ancestor (or the path itself) that a dataset names
            if c in src._roots and (c not in src._keymap or
                                    self.datasets[src._keymap[c]].get('admission') == INVENTORY):
                return self.datasets[src._roots[c]]
            probe = c
            while True:
                ident = src._keymap.get(probe)
                if ident is not None:
                    if best is None or len(probe) > len(best[0]):
                        best = (probe, ident)
                    break
                parent = os.path.dirname(probe)
                if parent == probe:
                    break
                probe = parent
        return self.datasets[best[1]] if best else None

    def role(self, name: str) -> list[tuple[dict, Path]]:
        """Datasets of the git layer (pinned) with role NAME, in role order: ``[(entry, path of its role file)]``."""
        out = []
        for ident in self.git_order:
            e = self.git[ident]
            spec = (e.get('roles') or {}).get(name)
            if spec is None or e.get('retired'):
                continue
            spec = spec if isinstance(spec, dict) else {}
            base = absolute(e['path'])
            target = base / spec['file'] if spec.get('file') else base
            # the data's real location: a pin has no runs/ links, and runs record the same paths as before the registry
            out.append((spec.get('order', 0), ident, e, Path(os.path.realpath(target))))
        out.sort(key=lambda x: (x[0], self.git_order.index(x[1])))
        return [(e, p) for _, _, e, p in out]

    def consumers(self, identity: str) -> list[dict]:
        """Who read IDENTITY: the runs that recorded a read, then the datasets recorded as built from it."""
        runs = [{'run': r['run'], 'how': r['how'], 'amount': r.get('amount'), 'time': r.get('time'),
                 'code': ((r.get('producer') or {}).get('code') or {}).get('git_commit'),
                 'sha256': next((i.get('sha256') or i.get('manifest_sha256') for i in r.get('items') or []
                                 if i.get('id') == identity), None)}
                for r in self.consumption if any(i.get('id') == identity for i in r.get('items') or [])]
        derived = []
        for ident, snaps in self.snapshots.items():
            for snap in snaps:
                if any(isinstance(i, dict) and i.get('id') == identity for i in snap.get('inputs') or []):
                    derived.append({'run': f'dataset:{ident}', 'how': f"input of {snap.get('kind')} {ident}",
                                    'time': snap.get('time'),
                                    'code': ((snap.get('producer') or {}).get('code') or {}).get('git_commit')})
                    break
        return runs + derived

    def protected_ids(self) -> list[str]:
        return [i for i, d in self.datasets.items() if d.get('protected') or
                str(d.get('admission') or '').startswith('evaluation-only')]

    def files(self, identity: str) -> list[dict]:
        ds = self.datasets[identity]
        manifest = stored_manifest(ds['manifest_sha256']) if ds.get('manifest_sha256') else None
        if manifest is None and ds.get('layer', '').startswith('git'):
            try:
                if self.git_ref is None:
                    text = (self.repo / GIT_MANIFESTS / f'{identity}.json').read_text()
                else:
                    text = subprocess.run(['git', '-C', str(self.repo), 'show', f'{self.git_ref}:{GIT_MANIFESTS}/{identity}.json'],
                                          capture_output=True, text=True, check=True).stdout
                manifest = json.loads(text)
            except (OSError, ValueError, subprocess.CalledProcessError):
                manifest = None
        return list((manifest or {}).get('files') or []) + self.extra_files.get(identity, [])


_LOADED = None
_GIT_CACHE: dict = {}


def _temp_roots() -> set:
    import tempfile
    base = tempfile.gettempdir()
    return {os.path.normpath(base), os.path.realpath(base)}


def _under(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip(os.sep) + os.sep)


def _read_journal() -> list:
    rows = []
    try:
        with open(journal_path()) as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def load(*, repo=None, ref=None, fresh=False) -> Registry:
    """The merged registry. The git layer is read from REPO (default: the checkout this code runs from)."""
    global _LOADED
    repo = Path(repo) if repo is not None else code_repo()
    oid = _registry_ref_commit(repo, ref) if ref is not None else None
    key = (str(repo), oid if oid is not None else corpus_registry.signature(repo), _mtime(journal_path()))
    if not fresh and _LOADED is not None and _LOADED[0] == key:
        return _LOADED[1]
    if _GIT_CACHE.get('key') != key[:2]:
        try:
            document = corpus_registry.load_ref(repo, oid) if oid is not None else corpus_registry.load(repo)
            git_entries = document['corpora']
        except (OSError, ValueError, KeyError):
            if oid is not None: raise  # an explicit published snapshot must not become an empty registry
            git_entries = []
        _GIT_CACHE.update(key=key[:2], entries=git_entries)
    else:
        git_entries = _GIT_CACHE['entries']
    # Keep this call's selected entries local across journal I/O / other readers.
    reg = Registry(git_entries, _read_journal(), repo=repo, ref=oid)
    _LOADED = (key, reg)
    return reg


def _mtime(path: Path):
    try:
        s = path.stat()
        return (s.st_size, s.st_mtime_ns)
    except OSError:
        return None


def role_paths(name: str, *, repo=None, ref=None) -> list[str]:
    """The role's files as local paths (``grpo.py`` POOLS and friends)."""
    reg = load(repo=repo, ref=ref) if ref is not None else load(repo=repo)
    return [str(p) for _, p in reg.role(name)]


def role_prefix_paths(prefix: str, *, repo=None, ref=None) -> list[tuple[str, str]]:
    """``[(role, local path)]`` of every role whose name starts with PREFIX, sorted by role then role order (the held
    sets: every ``held-eval*`` role is excluded from training, ``held_eval.training_exclusion_ids``)."""
    reg = load(repo=repo, ref=ref) if ref is not None else (load() if repo is None else load(repo=repo))
    names = sorted({name for ident in reg.git_order for name in (reg.git[ident].get('roles') or {})
                    if name.startswith(prefix)})
    return [(name, str(p)) for name in names for _, p in reg.role(name)]


def role_path(name: str, *, repo=None, ref=None) -> str:
    paths = role_paths(name, repo=repo, ref=ref) if ref is not None else role_paths(name, repo=repo)
    if len(paths) != 1:
        raise ValueError(f'registry role {name!r} names {len(paths)} datasets; exactly one expected')
    return paths[0]


# ---------- promotion into the git layer ----------

GIT_FIELDS = ('kind', 'format', 'rows', 'license', 'license_provenance', 'protected', 'supersedes', 'roles', 'note',
              'admission', 'inputs', 'producer', 'manifest_sha256')


def git_entry(record: dict) -> dict:
    """The git-layer entry of a journal snapshot (the manifest itself is republished by ``sync_training_corpora``)."""
    entry = {'id': record['id'], 'owner': record.get('machine') or machine(), 'path': record['path']}
    if record.get('include'):
        entry['include'] = record['include']
    if record.get('exclude'):
        entry['exclude_globs'] = record['exclude']
    for k in GIT_FIELDS:
        if record.get(k) not in (None, [], {}):
            entry[k] = record[k]
    if 'producer' in entry:
        p = entry['producer']
        entry['producer'] = {'script': p.get('script'), 'argv': p.get('argv'), 'git_commit': (p.get('code') or {}).get('git_commit'),
                             'dirty': (p.get('code') or {}).get('dirty'), 'host': p.get('host')}
    entry['training_admission'] = entry.get('training_admission',
                                            'evaluation-only' if record.get('protected') else 'registry-recorded')
    return entry
