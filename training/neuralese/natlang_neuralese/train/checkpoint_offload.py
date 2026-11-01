"""Asynchronous checkpoint offload to the external disk (owner 2026-10-13: "move active run checkpoints to external on
an ongoing basis, async from write"). plans/STORAGE_POLICY.md section 2a.

Writes are unchanged: every full state goes to the NVMe (pending file, fsync, atomic rename; trajectory_state). After a
successful write the file is handed to one background worker per process, which copies it to the mirror

    <mirror root>/<path relative to ~/natlang/runs>      (default mirror root: <data_hdd>/natlang-active-lineage)

in a child process at idle I/O class and nice 19: copy to ``<mirror>.offload.part`` in 8 MiB chunks with a streamed
SHA-256, fsync, check the source did not change, re-read the copy (page cache dropped) and compare size and hash, then
atomically rename it into place, fsync the directory and write ``<mirror>.offload.json`` (source identity, bytes,
SHA-256). Two roles:

- ``latest`` (the rolling resumable slot, ``checkpoint.pt``): mirrored as a disaster copy and kept on the NVMe; resume
  reads the NVMe copy. When the NVMe slot is missing or truncated, ``recover_latest`` points it at the verified mirror.
- ``archive`` (``best-checkpoint.pt``, final exports, superseded/periodic checkpoints): once the mirror is verified,
  the NVMe file is atomically replaced by a symlink to it (link written beside it, then ``os.replace``), so every
  consumer path keeps working and only the resumable slot (plus the trainer's disk reserve) stays on the NVMe.

Crash consistency: the NVMe path is always the old regular file or a symlink to a mirror that was renamed into place
and made durable first, never a dangling link. Stale ``.offload.part``/``.offload-link`` files of a crash are removed
when a directory is next swept. Foreign partial copies (``<mirror>.part``) are never touched; a file with one is
skipped. An offload failure is an event (stdout and ``<out>/checkpoint-offload.jsonl``), never a training failure; the
failed file is retried with the next write. The worker coalesces repeated writes of one path to the newest.

The mirror is used only when its root lies on a mounted filesystem other than the source's (never a directory on the
same NVMe, e.g. inside a container without the mount). Settings (recorded in ``checkpoint-policy.json``):
``NATLANG_CHECKPOINT_OFFLOAD=0`` disables, ``NATLANG_CHECKPOINT_MIRROR`` sets the root,
``NATLANG_CHECKPOINT_OFFLOAD_MIN_GB`` (default 1) the smallest file offloaded, ``NATLANG_CHECKPOINT_OFFLOAD_LATEST=0``
stops mirroring the resumable slot, ``NATLANG_CHECKPOINT_OFFLOAD_HDD_FLOOR_GB`` (default 50) the free space kept on the
external disk. Stdlib only: the copy child runs this file as a script.
"""
from __future__ import annotations

import ctypes
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import re
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

GIB = 2**30
CHUNK = 8 << 20
SYNC_EVERY = 512 << 20
PART_SUFFIX = '.offload.part'
LINK_SUFFIX = '.offload-link'
RECORD_SUFFIX = '.offload.json'
LATEST_NAMES = ('checkpoint.pt',)
ARCHIVE_GLOBS = ('best-*.pt', 'best-*.safetensors', 'final-*.pt', 'final-*.safetensors', 'checkpoint-*.pt',
                 'step-*.pt')
KEEP_MARKER = '.keep-on-nvme'
# Full states whose non-latest copies no consumer continues from (prune='auto': sweeps and the one-shot). The direct
# lineage continues from the predecessor's checkpoint.pt (recipe DIRECT_PHASE_CONTINUATION), never from a best; the
# text warm-up continues from its verified best, causal bootstrap and trajectories pin or resume theirs: not listed.
PRUNABLE_SCHEMAS = ('natlang.neuralese-direct/1',)


def _paths():
    try:
        from ..common import paths
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from natlang_neuralese.common import paths
    return paths


@dataclass
class OffloadSettings:
    enabled: bool = True
    mirror_root: str = ''
    min_bytes: int = GIB
    mirror_latest: bool = True
    link_archived: bool = True
    prune_archived: bool = True
    hdd_floor_bytes: int = 50 * GIB
    stale_part_seconds: float = 1800.0
    archive_globs: tuple = ARCHIVE_GLOBS
    latest_names: tuple = LATEST_NAMES

    @classmethod
    def from_env(cls) -> 'OffloadSettings':
        env = os.environ.get
        off = lambda key: env(key, '1').strip().lower() in ('0', 'off', 'false', 'no')
        root = env('NATLANG_CHECKPOINT_MIRROR') or str(_paths().resolve('data_hdd', 'natlang-active-lineage'))
        return cls(enabled=not off('NATLANG_CHECKPOINT_OFFLOAD'), mirror_root=root,
                   min_bytes=int(float(env('NATLANG_CHECKPOINT_OFFLOAD_MIN_GB', '1')) * GIB),
                   mirror_latest=not off('NATLANG_CHECKPOINT_OFFLOAD_LATEST'),
                   prune_archived=not off('NATLANG_CHECKPOINT_PRUNE_OPTIMIZER'),
                   hdd_floor_bytes=int(float(env('NATLANG_CHECKPOINT_OFFLOAD_HDD_FLOOR_GB', '50')) * GIB))

    def record(self) -> dict:
        out = asdict(self)
        out['archive_globs'], out['latest_names'] = list(self.archive_globs), list(self.latest_names)
        return out


# Mirror placement ------------------------------------------------------------------------------------------------
def _absolute(path) -> Path:
    path = Path(path)
    return Path(os.path.realpath(path.parent)) / path.name


def mirror_path(path, settings: OffloadSettings) -> Path | None:
    """Mirror a checkpoint at a stable path preserving its source location."""
    path = _absolute(path)
    repo = Path(os.path.realpath(_paths().root('repo')))
    root = Path(settings.mirror_root)
    for base, prefix in ((repo / 'runs', ()), (repo, ('repo',)), (Path(os.path.realpath(Path.home())), ('home',))):
        if base in path.parents:
            return root.joinpath(*prefix, path.relative_to(base))
    # An additional active-artifact root can live outside the checkout/home
    # (Pop uses /srv/storage). Declare it on that machine; keep unknown paths
    # outside the offload policy.
    extra = os.environ.get('NATLANG_CHECKPOINT_SOURCE_ROOT')
    if extra:
        base = Path(os.path.realpath(extra))
        if base in path.parents:
            return root / 'source-root' / path.relative_to(base)
    return None


def _mount_of(path: Path) -> Path | None:
    """The mirror root's mount; / is valid when the source is another filesystem."""
    for candidate in (path, *path.parents):
        if candidate.exists() and os.path.ismount(candidate):
            return candidate
    return None


def _same_filesystem(a: Path, b: Path) -> bool:
    return a.stat().st_dev == b.stat().st_dev


def mirror_unavailable(source, settings: OffloadSettings) -> str | None:
    """Why ``source`` cannot be mirrored now, or None."""
    if not settings.enabled:
        return 'offload disabled'
    if mirror_path(source, settings) is None:
        return 'outside the mirrored trees'
    try:
        mount = _mount_of(Path(settings.mirror_root))
    except OSError as error:
        # A disconnected optional mirror must remain an unavailable backup,
        # not prevent the primary local checkpoint policy from starting.
        return 'mirror filesystem inaccessible: ' + str(error)
    if mount is None:
        return f'mirror root {settings.mirror_root} is on no mounted filesystem of its own'
    try:
        if _same_filesystem(mount, _absolute(source).parent):
            return 'mirror root is on the source filesystem'
    except OSError as error:
        return str(error)
    return None


# Copy, verify, link ----------------------------------------------------------------------------------------------
def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop(*handles) -> None:
    for handle in handles:
        try:
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass


def _identity(st) -> list:
    return [st.st_ino, st.st_size, st.st_mtime_ns]


def file_sha256(path, *, drop_cache=True) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        if drop_cache:
            _drop(stream)
        read = 0
        while block := stream.read(CHUNK):
            digest.update(block)
            read += len(block)
            if drop_cache and read % SYNC_EVERY == 0:
                _drop(stream)  # unified memory (GB10): cached pages come out of the same RAM as CUDA allocations
    return digest.hexdigest()


def verified_copy(path, dest, *, partial_suffix=PART_SUFFIX) -> dict:
    """Durably copy the exact opened source, checking its identity, size and SHA-256.

    Atomic pathname replacement by training does not invalidate that intact
    snapshot. Its receipt retains the opened file's identity; mirror_current
    and the archive link swap still refuse to treat it as a newer checkpoint.
    An in-place source change fails and removes the partial copy.
    """
    path, dest = Path(path), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + partial_suffix)
    source_hash = hashlib.sha256()
    try:
        with open(path, 'rb') as src, open(partial, 'wb') as out:
            before = os.fstat(src.fileno())
            copied = 0
            while block := src.read(CHUNK):
                source_hash.update(block)
                out.write(block)
                copied += len(block)
                if copied % SYNC_EVERY == 0:
                    out.flush()
                    os.fdatasync(out.fileno())
                    _drop(src, out)
            out.flush()
            os.fsync(out.fileno())
            _drop(src, out)
            if _identity(os.fstat(src.fileno())) != _identity(before):
                raise RuntimeError(f'{path} changed during the copy')
        os.utime(partial, ns=(before.st_atime_ns, before.st_mtime_ns))
        digest = source_hash.hexdigest()
        if partial.stat().st_size != before.st_size or file_sha256(partial) != digest:
            raise RuntimeError(f'verification failed for {path}')
        os.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    _fsync_dir(dest.parent)
    return {'source': str(path), 'dest': str(dest), 'bytes': before.st_size, 'sha256': digest,
            'source_identity': _identity(before), 'mtime': before.st_mtime, 'time': time.time()}


def is_optimizer_key(key) -> bool:
    """Top-level state entries that only a resume needs: the optimizer's slots and their parameter naming."""
    return isinstance(key, str) and 'optimizer' in key.lower() and key != 'optimizer_pruned'


def load_state(path):
    """A full training state, memory-mapped (the trainers' own files)."""
    import torch
    from torch.torch_version import TorchVersion
    try:
        with torch.serialization.safe_globals([TorchVersion]):
            return torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def _same_values(a, b, where='state') -> None:
    import torch
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
            raise RuntimeError(f'pruned copy differs at {where}')
        x, y = a.detach().contiguous().reshape(-1), b.detach().contiguous().reshape(-1)
        try:
            x, y = x.view(torch.uint8), y.view(torch.uint8)  # bitwise (NaN-safe)
        except RuntimeError:
            pass
        if not torch.equal(x, y):
            raise RuntimeError(f'pruned copy differs at {where}')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or list(a) != list(b):
            raise RuntimeError(f'pruned copy differs at {where}')
        for key in a:
            _same_values(a[key], b[key], f'{where}.{key}')
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            raise RuntimeError(f'pruned copy differs at {where}')
        for index, (x, y) in enumerate(zip(a, b)):
            _same_values(x, y, f'{where}[{index}]')
    elif a != b and not (a != a and b != b):  # NaN equals NaN here
        raise RuntimeError(f'pruned copy differs at {where}')


def pruned_copy(path, dest, *, partial_suffix=PART_SUFFIX, schemas=None) -> dict | None:
    """A weights-only copy of a full training state (owner 2026-10-13: only the latest resume checkpoint keeps the
    optimizer): every top-level optimizer entry dropped, ``optimizer_pruned: True`` and the dropped keys recorded in
    the state. Written through the partial name, fsynced, re-read and compared value by value (bitwise) with the
    source's kept entries, hashed, then renamed into place. None when the file is no dict state with an optimizer
    (the caller copies it byte for byte)."""
    import torch
    path, dest = Path(path), Path(dest)
    before = path.lstat()
    try:
        state = load_state(path)
    except Exception:
        return None
    dropped = sorted(k for k in state if is_optimizer_key(k)) if isinstance(state, dict) else []
    native_training = (state.get('training') if isinstance(state, dict) and
        state.get('schema') == 'natlang.neuralese-native-checkpoint/1' else None)
    nested = sorted(k for k in native_training if is_optimizer_key(k)) if isinstance(native_training, dict) else []
    dropped += ['training.' + key for key in nested]
    if schemas is not None and isinstance(state, dict) and state.get('schema') not in schemas:
        dropped = []  # 'auto': a writer whose non-latest copies are continuation parents or pinned
    if not dropped:
        return None
    kept = {k: v for k, v in state.items() if not is_optimizer_key(k)}
    if nested:
        kept['training'] = {key: value for key, value in native_training.items() if key not in nested}
        try:
            from ..common.hashing import sha256_file_hex_cached
        except ImportError:  # The existing offload worker runs this file directly.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from natlang_neuralese.common.hashing import sha256_file_hex_cached
        kept['inference_source'] = {'schema': 'natlang.native-inference-source/1',
            'training_checkpoint_sha256': sha256_file_hex_cached(path),
            'training_checkpoint_bytes': before.st_size, 'step': state.get('step')}
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + partial_suffix)
    try:
        with open(partial, 'wb') as out:
            torch.save({**kept, 'optimizer_pruned': True, 'optimizer_pruned_keys': dropped}, out)
            out.flush()
            os.fsync(out.fileno())
            _drop(out)
        os.utime(partial, ns=(before.st_atime_ns, before.st_mtime_ns))
        if _identity(path.lstat()) != _identity(before):
            raise RuntimeError(f'{path} changed during the copy')
        written = load_state(partial)
        if written.get('optimizer_pruned') is not True:
            raise RuntimeError(f'verification failed for {path}')
        _same_values(kept, {k: written[k] for k in kept if k in written})
        del written
        digest = file_sha256(partial)
        os.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        state = kept = None
    _fsync_dir(dest.parent)
    return {'source': str(path), 'dest': str(dest), 'bytes': dest.stat().st_size, 'sha256': digest,
            'source_identity': _identity(before), 'source_bytes': before.st_size, 'optimizer_pruned': True,
            'optimizer_pruned_keys': dropped, 'mtime': before.st_mtime, 'time': time.time()}


def local_copy(path, dest, prune=False) -> dict:
    """The offload copy in this process: weights-only when ``prune`` (True, or 'auto': only a PRUNABLE_SCHEMAS state)
    and the file carries an optimizer, else bytes."""
    pruned = (pruned_copy(path, dest, schemas=PRUNABLE_SCHEMAS if prune == 'auto' else None) if prune else None)
    return pruned or verified_copy(path, dest)


def require_optimizer(state, path) -> None:
    """A continuation that restores optimizer state refuses a weights-only (pruned) non-latest checkpoint clearly."""
    if isinstance(state, dict) and state.get('optimizer_pruned'):
        raise ValueError(f'{path} is a weights-only checkpoint (optimizer pruned by the checkpoint offload, '
                         f'plans/STORAGE_POLICY.md 2a): continue from the stage\'s latest checkpoint.pt, which keeps '
                         'its optimizer')


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def path_lock(path) -> threading.RLock:
    """One in-process lock per NVMe path: a best publication and the offload's link swap never interleave."""
    key = str(_absolute(path))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def replace_with_link(path, dest, *, expect: list | None = None) -> None:
    """Atomically replace ``path`` by a symlink to the (existing, durable) ``dest``. With ``expect`` (a source
    identity), refuse when ``path`` is no longer that file. Leaves no link file behind on failure."""
    path, dest = Path(path), Path(dest)
    if not dest.is_file():
        raise FileNotFoundError(f'link target {dest} is missing: refusing a dangling link')
    link = path.with_name(path.name + LINK_SUFFIX)
    link.unlink(missing_ok=True)
    try:
        os.symlink(dest, link)
        with path_lock(path):
            if expect is not None and _identity(path.lstat()) != list(expect):
                raise RuntimeError(f'{path} changed before its link swap')
            os.replace(link, path)  # readers see the old file or the link, never nothing
    except BaseException:
        link.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def read_record(mirror: Path) -> dict | None:
    try:
        return json.loads(mirror.with_name(mirror.name + RECORD_SUFFIX).read_text())
    except (OSError, ValueError):
        return None


def write_record(mirror: Path, record: dict) -> None:
    target = mirror.with_name(mirror.name + RECORD_SUFFIX)
    temp = target.with_name(target.name + '.pending')
    temp.write_text(json.dumps(record, indent=1) + '\n')
    os.replace(temp, target)


def mirror_current(path: Path, mirror: Path) -> dict | None:
    """The mirror's record when the mirror is a verified copy of ``path`` as it is now."""
    record = read_record(mirror)
    try:
        if (record and record.get('source_identity') == _identity(path.lstat())
                and mirror.is_file() and mirror.stat().st_size == record.get('bytes')):
            return record
    except OSError:
        pass
    return None


def foreign_partial(mirror: Path) -> Path | None:
    """Another copier's partial file of this mirror (e.g. ``cp`` to ``<mirror>.part``): leave the file alone."""
    other = mirror.with_name(mirror.name + '.part')
    return other if other.exists() else None


def clean_stale(directory: Path, *, suffixes=(PART_SUFFIX, LINK_SUFFIX), older_than: float = 1800.0) -> list[str]:
    """Remove our own crash leftovers (never another copier's ``.part``)."""
    removed = []
    if not directory.is_dir():
        return removed
    now = time.time()
    for child in directory.iterdir():
        if any(child.name.endswith(suffix) for suffix in suffixes):
            try:
                if now - child.lstat().st_mtime > older_than:
                    child.unlink()
                    removed.append(str(child))
            except OSError:
                pass
    return removed


def role_of(path: Path, settings: OffloadSettings) -> str | None:
    if path.name in settings.latest_names:
        return 'latest'
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in settings.archive_globs):
        return 'archive'
    return None


def _hdd_link_shortcut(path: Path, mirror: Path) -> dict | None:
    """A best published as a hard link of the slot whose mirror is current: link the mirror on the external disk too
    (the same retention the NVMe hard link gives), no second copy."""
    slot_mirror = mirror.with_name('checkpoint.pt')
    slot = path.with_name('checkpoint.pt')
    try:
        if path.lstat().st_ino != slot.lstat().st_ino or slot.is_symlink():
            return None
    except OSError:
        return None
    record = mirror_current(slot, slot_mirror)
    if record is None:
        return None
    pending = mirror.with_name(mirror.name + '.pending-link')
    pending.unlink(missing_ok=True)
    os.link(slot_mirror, pending)
    os.replace(pending, mirror)
    _fsync_dir(mirror.parent)
    record = dict(record, source=str(path), dest=str(mirror), linked_from=str(slot_mirror), time=time.time())
    write_record(mirror, record)
    return record


def _rebind(node, name: str, record: dict) -> bool:
    changed = False
    if isinstance(node, dict):
        for key, value in node.items():
            if key == name and isinstance(value, dict) and 'sha256' in value:
                if value.get('sha256') != record['sha256']:
                    value.update(sha256=record['sha256'], bytes=record['bytes'], optimizer_pruned=True)
                    changed = True
            else:
                changed = _rebind(value, name, record) or changed
    elif isinstance(node, list):
        for value in node:
            changed = _rebind(value, name, record) or changed
    return changed


def rebind_receipts(path: Path, record: dict) -> list[str]:
    """Receipts beside ``path`` that pin its bytes (``{"<name>": {"sha256": ...}}``, e.g. best-checkpoint.json's
    ``files``) are rebound to the weights-only file before the swap, marked ``optimizer_pruned``."""
    rebound = []
    for receipt in sorted(path.parent.glob('*.json')):
        try:
            if receipt.is_symlink() or receipt.stat().st_size > 8 << 20:
                continue
            data = json.loads(receipt.read_text())
        except (OSError, ValueError):
            continue
        if _rebind(data, path.name, record):
            temp = receipt.with_name(receipt.name + '.pending')
            with open(temp, 'w') as stream:
                json.dump(data, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, receipt)
            rebound.append(str(receipt))
    return rebound


_MANIFEST_FILES: dict[str, str] | None = None


def pinned_by_manifest(path) -> str | None:
    """The corpus manifest (training/corpus-manifests) that lists ``path``: published held/training data is verified
    byte for byte and refuses symlinks, so such a file is never linked or rewritten."""
    global _MANIFEST_FILES
    if _MANIFEST_FILES is None:
        repo = Path(os.path.realpath(_paths().root('repo')))
        files = {}
        for manifest in sorted((repo / 'training' / 'corpus-manifests').glob('*.json')):
            try:
                data = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            root, listed = data.get('path'), data.get('files')
            if not isinstance(root, str) or not isinstance(listed, list):
                continue
            base = os.path.realpath(root if root.startswith('/') else repo / root)
            for item in listed:
                if isinstance(item, dict) and isinstance(item.get('path'), str):
                    files[os.path.join(base, item['path'])] = manifest.name
        _MANIFEST_FILES = files
    return _MANIFEST_FILES.get(str(_absolute(path)))


def offload_one(path, role: str, settings: OffloadSettings, *, copier=None, prune=False) -> dict:
    """Mirror ``path`` (verify, record) and, for the archive role, replace it by a link to the mirror. Returns an
    event; raises on failure (the NVMe file is then untouched)."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return {'event': 'checkpoint_offload_skipped', 'path': str(path), 'reason': 'not a regular file'}
    reason = mirror_unavailable(path, settings)
    if reason:
        return {'event': 'checkpoint_offload_skipped', 'path': str(path), 'reason': reason}
    mirror = mirror_path(path, settings)
    if foreign_partial(mirror):
        return {'event': 'checkpoint_offload_skipped', 'path': str(path),
                'reason': f'another copy in progress: {foreign_partial(mirror)}'}
    clean_stale(mirror.parent, older_than=settings.stale_part_seconds)
    record = mirror_current(path, mirror)
    copied = False
    prune = prune if role == 'archive' and settings.prune_archived else False
    pinned = pinned_by_manifest(path) if role == 'archive' else None
    if pinned:
        return {'event': 'checkpoint_offload_skipped', 'path': str(path), 'reason': f'pinned by corpus manifest {pinned}'}
    if record is None and role == 'archive' and not prune:
        record = _hdd_link_shortcut(path, mirror)
    if record is None:
        size = path.stat().st_size
        free = shutil.disk_usage(_mount_of(Path(settings.mirror_root)) or settings.mirror_root).free
        if free - size < settings.hdd_floor_bytes:
            raise OSError(f'external disk floor: {free} bytes free, {size} to copy, floor {settings.hdd_floor_bytes}')
        record = (copier or local_copy)(path, mirror, prune)
        write_record(mirror, record)
        copied = True
    event = {'event': 'checkpoint_offloaded', 'role': role, 'path': str(path), 'mirror': str(mirror),
             'bytes': record['bytes'], 'sha256': record['sha256'], 'copied': copied,
             'optimizer_pruned': bool(record.get('optimizer_pruned')), 'source_bytes': record.get('source_bytes')}
    if role == 'archive' and settings.link_archived:
        # Publication uses this same lock. Check the identity before changing
        # any receipt, and hold it through both receipt updates and the swap.
        with path_lock(path):
            if _identity(path.lstat()) != list(record['source_identity']):
                raise RuntimeError(f'{path} changed before its link swap')
            if record.get('optimizer_pruned'):
                event['receipts'] = rebind_receipts(path, record)
            replace_with_link(path, mirror, expect=record['source_identity'])
        event['linked'] = True
    return event


# Resume ----------------------------------------------------------------------------------------------------------
def truncated(path: Path) -> bool:
    """A torch zip checkpoint whose central directory is unreadable (a cut-off write or copy), or an empty file."""
    try:
        if path.stat().st_size == 0:
            return True
        with open(path, 'rb') as stream:
            if stream.read(2) != b'PK':
                return False  # not a zip checkpoint: not ours to judge
        zipfile.ZipFile(path).close()
        return False
    except (zipfile.BadZipFile, OSError):
        return True


def recover_latest(slot, settings: OffloadSettings | None = None) -> dict | None:
    """When the NVMe resumable slot is missing or truncated but this run recorded mirroring and the mirror holds a
    verified copy of it, point the slot at the mirror (an atomic symlink); the next write replaces the link by a
    regular NVMe file. Returns the event, or None when nothing was done."""
    slot = Path(slot)
    settings = settings or OffloadSettings.from_env()
    if slot.is_file() and not truncated(slot):
        return None
    try:
        meta = json.loads((slot.parent / 'checkpoint-policy.json').read_text())
    except (OSError, ValueError):
        return None
    if not (meta.get('offload') or {}).get('mirror'):
        return None  # this run never mirrored: a missing slot is a fresh start
    mirror = mirror_path(slot, settings)
    record = read_record(mirror) if mirror is not None else None
    if record is None or not mirror.is_file() or mirror.stat().st_size != record.get('bytes') or truncated(mirror):
        return None
    state = 'missing' if not slot.exists() and not slot.is_symlink() else 'truncated'
    replace_with_link(slot, mirror)
    event = {'event': 'checkpoint_resume_from_mirror', 'slot': str(slot), 'mirror': str(mirror), 'local': state,
             'bytes': record['bytes'], 'sha256': record['sha256']}
    print(json.dumps(event), flush=True)
    return event


def update_policy_record(out, **fields) -> dict:
    """Merge ``fields`` into ``<out>/checkpoint-policy.json`` (atomic)."""
    target = Path(out) / 'checkpoint-policy.json'
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        data = {}
    data.update(fields)
    temp = target.with_name(target.name + '.pending')
    temp.write_text(json.dumps(data) + '\n')
    os.replace(temp, target)
    return data


# The background worker -------------------------------------------------------------------------------------------
def _lower_priority(pid: int) -> None:
    """nice 19 and the idle I/O class (best effort)."""
    try:
        os.setpriority(os.PRIO_PROCESS, pid, 19)
    except OSError:
        pass
    number = {'x86_64': 251, 'aarch64': 30}.get(platform.machine())
    if number is not None:
        try:
            ctypes.CDLL(None, use_errno=True).syscall(number, 1, pid, 3 << 13)  # IOPRIO_WHO_PROCESS, CLASS_IDLE
        except (OSError, AttributeError):
            pass


def child_copy(path, dest, prune=False) -> dict:
    """``local_copy`` in a child process at idle priority: the trainer keeps its GIL and its I/O share."""
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), {True: 'prune', 'auto': 'prune-auto'}.get(prune, 'copy'),
                                str(path), str(dest)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _lower_priority(process.pid)
    out, err = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f'offload copy failed ({process.returncode}): {err.strip()[-500:]}')
    return json.loads(out.strip().splitlines()[-1])


class CheckpointOffloader:
    """One background worker per process. ``submit`` never blocks on I/O and never raises."""

    def __init__(self, settings: OffloadSettings | None = None, *, copier=child_copy):
        self.settings = settings or OffloadSettings.from_env()
        self.copier = copier
        self.pending: dict[str, str] = {}
        self.failed: dict[str, str] = {}
        self.events: list[dict] = []
        self.swept: set[str] = set()
        self.cond = threading.Condition()
        self.busy = False
        self.thread = threading.Thread(target=self._run, name='checkpoint-offload', daemon=True)
        self.thread.start()

    def submit(self, path, role: str | None = None, prune=False) -> None:
        try:
            path = _absolute(path)
            role = role or role_of(path, self.settings)
            if role is None or (role == 'latest' and not self.settings.mirror_latest):
                return
            try:
                if path.is_symlink() or path.stat().st_size < self.settings.min_bytes:
                    return
            except OSError:
                return
            with self.cond:
                for retry, job in list(self.failed.items()):
                    self.pending.setdefault(retry, job)
                self.failed.clear()
                self.pending.pop(str(path), None)  # coalesce: the newest submission goes last
                self.pending[str(path)] = (role, prune)
                if str(path.parent) not in self.swept:  # leftovers of earlier processes: this and sibling stages
                    self.swept.add(str(path.parent))
                    siblings = [d for d in sorted(path.parent.parent.iterdir()) if d.is_dir() and d != path.parent
                                ] if _within_a_run(path) else []
                    for directory, own in ((path.parent, True), *((d, False) for d in siblings)):
                        for item, item_role in sweep_targets(directory, self.settings, own=own):
                            self.pending.setdefault(str(item), (item_role, 'auto'))
                self.cond.notify()
        except Exception as error:  # never fail a write
            self._event({'event': 'checkpoint_offload_failed', 'path': str(path), 'error': repr(error)})

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.cond:
            while self.pending or self.busy:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self.cond.wait(left)
        return True

    def _event(self, event: dict) -> None:
        event = dict(event, time=time.time())
        self.events.append(event)
        print(json.dumps(event), flush=True)
        path = event.get('path')
        if path:
            try:
                with open(Path(path).parent / 'checkpoint-offload.jsonl', 'a') as log:
                    log.write(json.dumps(event) + '\n')
            except OSError:
                pass

    def _run(self) -> None:
        while True:
            with self.cond:
                while not self.pending:
                    self.cond.wait()
                path, (role, prune) = next(iter(self.pending.items()))
                del self.pending[path]
                self.busy = True
            try:
                self._event(offload_one(path, role, self.settings, copier=self.copier, prune=prune))
            except Exception as error:
                with self.cond:
                    self.failed[path] = (role, prune)
                self._event({'event': 'checkpoint_offload_failed', 'path': path, 'role': role,
                             'error': repr(error)[:500], 'retry': 'at the next checkpoint write'})
            finally:
                with self.cond:
                    self.busy = False
                    self.cond.notify_all()


def _within_a_run(path: Path) -> bool:
    """``path`` lies at least two levels inside a run directory (runs/<lineage>/<run>/<stage>/file): its stage's
    siblings belong to the same run, never to other runs."""
    runs = Path(os.path.realpath(_paths().root('repo'))) / 'runs'
    return runs in path.parents and len(path.relative_to(runs).parts) >= 4


def sweep_targets(directory: Path, settings: OffloadSettings, *, own: bool) -> list[tuple[Path, str]]:
    """Regular files of one directory to offload: archive-role names always, the latest slot only in the writer's
    own directory (``own``). Skipped under a ``.keep-on-nvme`` marker."""
    if not directory.is_dir() or any((parent / KEEP_MARKER).exists() for parent in (directory, *directory.parents)):
        return []
    clean_stale(directory, suffixes=(LINK_SUFFIX,), older_than=60.0)
    found = []
    for child in sorted(directory.iterdir()):
        try:
            if child.is_symlink() or not child.is_file() or child.stat().st_size < settings.min_bytes:
                continue
        except OSError:
            continue
        role = role_of(child, settings)
        if role == 'archive' or (role == 'latest' and own and settings.mirror_latest):
            found.append((child, role))
    return found


_OFFLOADER: CheckpointOffloader | None = None
_OFFLOADER_GUARD = threading.Lock()


def offloader() -> CheckpointOffloader:
    global _OFFLOADER
    with _OFFLOADER_GUARD:
        if _OFFLOADER is None:
            _OFFLOADER = CheckpointOffloader()
        return _OFFLOADER


_RECOVERY_QUEUES: dict[str, 'RecoveryExportQueue'] = {}
_RECOVERY_QUEUES_GUARD = threading.Lock()


def after_write(path, role: str | None = None, prune=False) -> None:
    """The hook every checkpoint writer calls after a successful write or best publication. Never raises."""
    try:
        with _RECOVERY_QUEUES_GUARD:
            if str(Path(path).resolve()) in _RECOVERY_QUEUES:
                return
        settings = _OFFLOADER.settings if _OFFLOADER is not None else OffloadSettings.from_env()
        if mirror_unavailable(path, settings) is None:
            offloader().submit(path, role, prune)
    except Exception as error:
        print(json.dumps({'event': 'checkpoint_offload_failed', 'path': str(path), 'error': repr(error)}),
              flush=True)


def policy_record(out) -> dict:
    """The offload settings for ``checkpoint-policy.json``: the mirror directory, or why there is none."""
    settings = _OFFLOADER.settings if _OFFLOADER is not None else OffloadSettings.from_env()
    probe = Path(out) / 'checkpoint.pt'
    reason = mirror_unavailable(probe, settings)
    return dict(settings.record(), mirror=None if reason else str(mirror_path(probe, settings).parent),
                unavailable=reason)


if __name__ == '__main__' and len(sys.argv) == 4 and sys.argv[1] in ('copy', 'prune', 'prune-auto'):
    mode = {'copy': False, 'prune': True, 'prune-auto': 'auto'}[sys.argv[1]]
    print(json.dumps(local_copy(sys.argv[2], sys.argv[3], prune=mode)), flush=True)


def recovery_training_custody(state):
    """Compact the exact saved exposure/source metadata for independent readback."""
    from ..common.hashing import canonical_json_sha256_hex
    training=state.get('training') if isinstance(state,dict) else None
    if not isinstance(training,dict) or not training.get('optimizer') or not training.get('rng'):
        raise ValueError('recovery export requires full optimizer and RNG state')
    step=training.get('global_step')
    if type(step) is not int:
        raise ValueError('recovery export requires an earned global step')
    from .native_continuation import collect_sft_progress
    fields=('source_row_bindings','source_row_order','held_row_bindings','exposure_counts','required_visits',
            'cursor','epoch','step','global_step','source_hash')
    options=('withdrawal_history','pending_focus_history','fresh_source_frontier_history',
             '_required_visits','_objective_required_visits')
    frames=collect_sft_progress(training,borrow_frames=True)
    frame_custody={}
    for name,frame in frames.items():
        selected={key:frame.get(key) for key in fields}
        selected['options']={key:(frame.get('options') or {}).get(key) for key in options}
        selected['ce_binding']=(frame.get('options') or {}).get('ce_supervision_policy_binding')
        selected['source_roots']=[{key:ref.get(key) for key in ('path','sha256')} for ref in (frame.get('source_receipts') or [])]
        frame_custody[name]=canonical_json_sha256_hex(selected)
    return {'global_step':step,'optimizer_present':True,'rng_present':True,
            'custody_sha256':{key:canonical_json_sha256_hex(training.get(key)) for key in fields},
            'sft_frames_sha256':frame_custody,
            'source_roots':[{key:ref.get(key) for key in ('path','sha256')} for ref in (training.get('source_receipts') or [])[:2]],
            'ce':(training.get('options') or {}).get('ce_supervision_policy_binding')}


def validate_full_recovery_checkpoint(path, expected):
    """Decode the independently copied checkpoint and compare its actual custody."""
    from .checkpoint_policy import CheckpointPolicy
    state=CheckpointPolicy.load(path)
    if recovery_training_custody(state)!=expected:
        raise ValueError('independent checkpoint full-state custody differs')


def validate_recovery_plan_shape(plan):
    if not isinstance(plan,dict) or plan.get('schema')!='natlang.local-full-recovery-plan/1':
        raise ValueError('invalid full recovery plan schema')
    session=plan.get('session')
    if (not isinstance(session,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}',session)
            or session in ('.','..')):
        raise ValueError('recovery plan needs exact session identity')
    for key in ('destination','journal_directory'):
        if not isinstance(plan.get(key),str) or not Path(plan[key]).is_absolute():
            raise ValueError('recovery plan paths must be absolute')
    closure=plan.get('closure')
    if not isinstance(closure,list) or not closure:
        raise ValueError('recovery plan needs the sealed recovery closure')
    for item in closure:
        if (not isinstance(item,dict) or set(item)!={'role','path','sha256'}
                or not isinstance(item['role'],str) or not item['role']
                or not isinstance(item['path'],str) or not Path(item['path']).is_absolute()
                or not isinstance(item['sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',item['sha256'])):
            raise ValueError('invalid recovery closure binding')
    if not {'runtime','environment','system-python','system-libraries','node','interpreter','base','tokenizer','task-assets','controls','sources'} <= {item['role'] for item in closure}:
        raise ValueError('recovery closure lacks a required runtime, environment, system, base, tokenizer, task asset, control or source role')
    inputs=plan.get('inputs')
    if not isinstance(inputs,dict) or set(inputs)!={'train','foundation','held','ce'}:
        raise ValueError('recovery plan needs exact train, foundation, held and CE bindings')
    for role,ref in inputs.items():
        if ref is None and role in ('foundation','ce'):continue
        if (not isinstance(ref,dict) or set(ref)!={'path','sha256'} or not isinstance(ref['path'],str)
                or not Path(ref['path']).is_absolute() or not isinstance(ref['sha256'],str)
                or not re.fullmatch(r'[0-9a-f]{64}',ref['sha256'])):
            raise ValueError('invalid recovery input binding')
    restore=plan.get('restore')
    restore_fields={'runtime_archive','environment_archive','system_python_archive','base_path','node_path','interpreter_path','sample_graph','tokenizer_fingerprint','asset_archives','asset_guard','asset_inventory'}
    if (not isinstance(restore,dict) or set(restore)!=restore_fields
            or any(not isinstance(restore[key],str) or not Path(restore[key]).is_absolute() for key in restore_fields-{'tokenizer_fingerprint','asset_archives'})
            or not isinstance(restore['tokenizer_fingerprint'],str) or not re.fullmatch(r'[0-9a-f]{64}',restore['tokenizer_fingerprint'])):
        raise ValueError('recovery plan lacks exact restore inputs')
    for key,role in (('runtime_archive','runtime'),('environment_archive','environment'),('system_python_archive','system-python'),('node_path','node'),('interpreter_path','interpreter'),('sample_graph','sources')):
        if not any(item['role']==role and item['path']==restore[key] for item in closure):
            raise ValueError('recovery closure omits restore input '+key)
    if not any(item['role']=='base' and Path(item['path']).is_relative_to(restore['base_path']) for item in closure):
        raise ValueError('recovery closure omits the pinned model base')
    assets=restore['asset_archives']
    if (not isinstance(assets,list) or len(assets)!=2
            or {item.get('env') for item in assets if isinstance(item,dict)}!={'NATLANG_RL_TASK_ASSETS','NATLANG_RL_TASK_ASSETS_V3'}):
        raise ValueError('recovery plan needs both exact task asset roots')
    for item in assets:
        if (not isinstance(item,dict) or set(item)!={'env','original_root','archive'}
                or not isinstance(item['original_root'],str) or not Path(item['original_root']).is_absolute()
                or not any(ref['role']=='task-assets' and ref['path']==item['archive'] for ref in closure)):
            raise ValueError('recovery task asset archive binding differs')
    for key in ('asset_guard','asset_inventory'):
        if not any(item['role']=='controls' and item['path']==restore[key] for item in closure):
            raise ValueError('recovery closure omits task asset guard or inventory')
    qualification=plan.get('restore_qualification')
    if (not isinstance(qualification,dict) or set(qualification)!={'path','sha256'}
            or not isinstance(qualification['path'],str) or not Path(qualification['path']).is_absolute()
            or not isinstance(qualification['sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',qualification['sha256'])
            or not any(item['role']=='controls' and item['path']==qualification['path']
                       and item['sha256']==qualification['sha256'] for item in closure)):
        raise ValueError('recovery plan needs an exact restore qualification binding')
    additional=plan.get('additional_inputs',[])
    if not isinstance(additional,list):raise ValueError('invalid additional recovery input inventory')
    for item in additional:
        if (not isinstance(item,dict) or set(item)!={'kind','path','sha256'} or item['kind'] not in ('source','ce')
                or not isinstance(item['path'],str) or not Path(item['path']).is_absolute()
                or not isinstance(item['sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',item['sha256'])):
            raise ValueError('invalid additional recovery input binding')
    controller=plan.get('controller')
    if (not isinstance(controller,dict) or set(controller)!={'state_path','config'}
            or not isinstance(controller['state_path'],str) or not Path(controller['state_path']).is_absolute()
            or not isinstance(controller['config'],dict) or set(controller['config'])!={'path','sha256'}):
        raise ValueError('recovery plan needs exact controller state and config bindings')
    if not any(item['role']=='controls' and item['path']==controller['config']['path']
               and item['sha256']==controller['config']['sha256'] for item in closure):
        raise ValueError('recovery closure omits the actual controller config')


def recovery_closure_identity(plan):
    from ..common.hashing import canonical_json_sha256_hex
    qualification=plan['restore_qualification']
    values=[item for item in plan['closure'] if not (item['role']=='controls' and item['path']==qualification['path'] and item['sha256']==qualification['sha256'])]
    return canonical_json_sha256_hex(sorted(values,key=lambda x:(x['role'],x['path'],x['sha256'])))


def validate_restore_qualification(plan):
    reference=plan['restore_qualification'];path=Path(reference['path']).resolve(strict=True)
    if file_sha256(path)!=reference['sha256']:
        raise ValueError('recovery restore qualification binding changed')
    value=json.loads(path.read_text());checks=value.get('checks') or {}
    required=('all_payloads_reread','source_closure_resolved','environment_imports','tokenizer_loaded',
              'checkpoint_full_state_loaded','node_collector_imported','task_assets_read')
    evidence=value.get('evidence') or {}
    available={(item['path'],item['sha256']) for item in plan['closure'] if item['role']=='controls'}
    if (value.get('schema')!='natlang.local-recovery-restore-qualification/1'
            or value.get('status')!='passed' or value.get('inputs')!=plan['inputs']
            or value.get('restore')!=plan.get('restore') or value.get('additional_inputs',[])!=plan.get('additional_inputs',[])
            or value.get('controller_config')!=plan['controller']['config']
            or value.get('closure_identity')!=recovery_closure_identity(plan)
            or set(evidence)!=set(required) or any(checks.get(key) is not True for key in required)):
        raise ValueError('recovery restore qualification does not bind the complete closure')
    for ref in evidence.values():
        if (not isinstance(ref,dict) or set(ref)!={'path','sha256'}
                or (ref['path'],ref['sha256']) not in available
                or file_sha256(Path(ref['path']).resolve(strict=True))!=ref['sha256']):
            raise ValueError('recovery restore qualification evidence changed or is not retained')


def validate_recovery_plan_dependencies(plan, *, require_restore=True):
    """Check current recipe's transitive graph and CE dependencies once per plan."""
    from .recipe import native_source_dependencies
    from .native_supervision import ce_policy_dependencies
    from ..common.hashing import sha256_file_hex_cached
    inputs=plan.get('inputs') or {}
    available={(str(Path(item['path']).resolve()),item['sha256']) for item in plan.get('closure',[])}
    required=set()
    for role in ('train','foundation','held'):
        ref=inputs[role]
        if ref is None:continue
        path=Path(ref['path']).resolve(strict=True)
        if sha256_file_hex_cached(path)!=ref['sha256']:
            raise ValueError('recovery input source digest changed')
        required.add((str(path),ref['sha256']))
        required.update((item['path'],item['sha256']) for item in native_source_dependencies(path))
    if inputs['ce'] is not None:
        required.update((item['path'],item['sha256']) for item in ce_policy_dependencies(inputs['ce']))
    for ref in plan.get('additional_inputs',[]):
        path=Path(ref['path']).resolve(strict=True)
        if sha256_file_hex_cached(path)!=ref['sha256']:raise ValueError('additional recovery input digest changed')
        required.add((str(path),ref['sha256']))
        dependencies=native_source_dependencies(path) if ref['kind']=='source' else ce_policy_dependencies({'path':str(path),'sha256':ref['sha256']})
        required.update((item['path'],item['sha256']) for item in dependencies)
    missing=required-available
    if missing:
        raise ValueError('recovery plan omits transitive source or CE dependencies: '+str(sorted(missing)[:3]))
    if require_restore:
        validate_restore_qualification(plan)


def verified_recovery_blob(source, target, digest, *, copier=child_copy, cached_identity=None):
    """Copy or reuse one independently checked content-addressed closure blob."""
    import fcntl
    source=Path(source).resolve(strict=True);target=Path(target)
    if not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest):
        raise ValueError('invalid recovery closure digest')
    lock_dir=target.parent/'.locks';lock_dir.mkdir(parents=True,exist_ok=True)
    fd=os.open(lock_dir/(digest+'.lock'),os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX)
        if target.is_symlink():raise ValueError('independent recovery blob cannot be a symlink')
        if target.exists():
            st=target.stat();identity=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
            if cached_identity!=identity and file_sha256(target)!=digest:
                raise ValueError('existing independent recovery closure differs')
        else:
            copied=copier(source,target)
            if copied['sha256']!=digest:
                raise ValueError('recovery closure source differs from declared identity')
            st=target.stat();identity=[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]
        return identity
    finally:
        os.close(fd)


class RecoveryExportQueue:
    """Journaled, single-pin full-state exporter for an explicitly bound recovery plan.

    The plan names already-qualified immutable closure archives.  One slot inode
    is pinned while copying; newer commits are coalesced as an inode identity and
    pinned only after the current export is verified.  This object deliberately
    never prunes independent exports or an unverified pin.
    """
    def __init__(self, slot, plan, *, copier=verified_copy, start_worker=True, plan_validator=validate_recovery_plan_dependencies, checkpoint_validator=validate_full_recovery_checkpoint):
        import fcntl
        from ..common.jsonio import write_json_durable
        self.slot = Path(slot).resolve()
        self.plan = dict(plan)
        validate_recovery_plan_shape(self.plan)
        self.inputs=self.plan['inputs']
        self.plan_validator=plan_validator
        self.plan_validator(self.plan)
        self.destination = Path(self.plan['destination']).resolve()
        self.directory = Path(self.plan['journal_directory']).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.destination.mkdir(parents=True, exist_ok=True)
        if self.directory.stat().st_dev != self.slot.parent.stat().st_dev:
            raise ValueError('recovery journal and slot must share a filesystem for inode pinning')
        if self.destination.stat().st_dev == self.slot.parent.stat().st_dev:
            raise ValueError('independent recovery destination must use a separate filesystem')
        self.plan_sha256 = hashlib.sha256(json.dumps(self.plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.state_path = self.directory/'recovery-export-journal.json'
        self.pin = self.directory/'inflight-checkpoint.pt'
        self._owner = (self.directory/'recovery-export-owner.lock').open('a+b')
        fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._write_json = write_json_durable
        self.copier = copier
        self.checkpoint_validator = checkpoint_validator
        self.cond = threading.Condition()
        self.closed = False
        self.thread = None
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if (self.state.get('schema') != 'natlang.local-full-recovery-journal/1'
                    or self.state.get('session') != self.plan['session']
                    or self.state.get('slot') != str(self.slot)):
                raise ValueError('recovery journal belongs to another session or slot')
        else:
            if self.pin.exists():
                raise ValueError('unowned recovery pin requires reconciliation')
            self.state = {'schema':'natlang.local-full-recovery-journal/1', 'session':self.plan['session'],
                          'slot':str(self.slot), 'plan_sha256':self.plan_sha256, 'inflight':None, 'candidate':None, 'latest_verified':None, 'closure_cache':{}}
            self._persist()
        for record in (self.state.get('inflight'),self.state.get('candidate')):
            if record and record.get('plan_sha256') != self._plan_digest(record.get('plan')):
                raise ValueError('recovery journal plan snapshot changed')
        if self.state['inflight']:
            if self.state['inflight'].get('verified_receipt') and not self.pin.exists():
                self._verify_verified_receipt(self.state['inflight'])
                self.state['inflight']=None
                self._persist()
            else:
                self._validate_pin(self.state['inflight'])
        elif self.pin.exists():
            candidate=self.state.get('candidate')
            if candidate is None:
                raise ValueError('recovery pin has no journal custody')
            self._validate_pin(candidate)
            self.state['inflight']=candidate
            self.state['candidate']=None
            self._persist()
        self.state['latest_verified_status']=self._latest_status(self.state.get('latest_verified'))
        self._persist()
        if start_worker:
            self.thread = threading.Thread(target=self._run, name='checkpoint-recovery-export', daemon=True)
            self.thread.start()

    @staticmethod
    def _plan_digest(plan):
        return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def update_plan(self, plan):
        """Install the controller-qualified next source closure without changing an in-flight copy."""
        plan=json.loads(json.dumps(plan))
        validate_recovery_plan_shape(plan)
        if (plan.get('schema')!='natlang.local-full-recovery-plan/1'
                or any(plan.get(k)!=self.plan.get(k) for k in ('session','destination','journal_directory'))
                or plan['controller']['state_path']!=self.plan['controller']['state_path']
                or not isinstance(plan.get('inputs'),dict) or set(plan['inputs'])!={'train','foundation','held','ce'}
                or not {'runtime','environment','system-python','system-libraries','node','interpreter','base','tokenizer','task-assets','controls','sources'} <= {x.get('role') for x in plan.get('closure',[]) if isinstance(x,dict)}):
            raise ValueError('invalid recovery plan transition')
        self.plan_validator(plan)
        with self.cond:
            self.plan=plan
            self.inputs=plan['inputs']
            self.plan_sha256=self._plan_digest(plan)
            self.state['plan_sha256']=self.plan_sha256
            self._persist()

    def _persist(self):
        self._write_json(self.state_path, self.state)

    @staticmethod
    def _stat5(path):
        st=Path(path).stat()
        return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]

    @staticmethod
    def _stat(path):
        st = Path(path).stat()
        return {'device':st.st_dev, 'inode':st.st_ino, 'size':st.st_size, 'mtime_ns':st.st_mtime_ns}

    def _latest_status(self, reference):
        if not reference:return 'none'
        try:
            path=Path(reference['path']).resolve(strict=True)
            if file_sha256(path)!=reference['sha256']:return 'unavailable'
            receipt=json.loads(path.read_text())
            if (receipt.get('schema')!='natlang.local-full-recovery-export/1'
                    or receipt.get('full_state') is not True or receipt.get('independent_readback') is not True
                    or self._plan_digest(receipt.get('plan'))!=receipt.get('plan_sha256')
                    or receipt.get('checkpoint',{}).get('sha256')!=reference['checkpoint_sha256']):
                return 'unavailable'
            expected={(x['role'],x['path'],x['sha256']) for x in receipt['plan']['closure']}
            controller=receipt['controller_state']
            expected.add(('controller-state',controller['path'],controller['sha256']))
            if {(x.get('role'),x.get('declared_path'),x.get('sha256')) for x in receipt['closure']}!=expected:
                return 'unavailable'
            items=[receipt['checkpoint'],*receipt['closure']]
            for item in items:
                target=Path(item['path']).resolve(strict=True)
                before=self._stat5(target)
                if before!=item.get('file_identity'):
                    if file_sha256(target)!=item['sha256'] or self._stat5(target)!=before:
                        return 'unavailable'
            return 'verified-current-identity'
        except (OSError,ValueError,KeyError,TypeError):
            return 'unavailable'

    def _verify_verified_receipt(self, record):
        reference=record['verified_receipt']
        receipt_path=Path(reference['path']).resolve(strict=True)
        if file_sha256(receipt_path)!=reference['sha256']:
            raise ValueError('verified recovery receipt changed')
        receipt=json.loads(receipt_path.read_text())
        checkpoint=receipt.get('checkpoint') or {}
        if (receipt.get('schema')!='natlang.local-full-recovery-export/1'
                or receipt.get('session')!=self.plan['session']
                or receipt.get('global_step')!=record['global_step']
                or receipt.get('plan_sha256')!=record['plan_sha256']
                or self._plan_digest(receipt.get('plan'))!=record['plan_sha256']
                or receipt.get('full_state') is not True or receipt.get('independent_readback') is not True
                or receipt.get('training_custody')!=record.get('custody')
                or receipt.get('controller_state')!=record.get('controller_state')
                or receipt.get('source_identity')!=record['identity']
                or checkpoint.get('sha256')!=reference['checkpoint_sha256']
                or checkpoint.get('sha256')!=record['checkpoint_sha256']
                or file_sha256(Path(checkpoint.get('path','')).resolve(strict=True))!=checkpoint.get('sha256')):
            raise ValueError('verified recovery receipt does not bind this export')
        observed=receipt.get('closure')
        expected={(x['role'],x['path'],x['sha256']) for x in record['plan']['closure']}
        controller=record['controller_state']
        expected.add(('controller-state',controller['path'],controller['sha256']))
        if (not isinstance(observed,list) or {(x.get('role'),x.get('declared_path'),x.get('sha256')) for x in observed}!=expected
                or any(file_sha256(Path(x['path']).resolve(strict=True))!=x['sha256'] for x in observed)):
            raise ValueError('verified recovery closure is missing or changed')

    def _validate_pin(self, record):
        if self.pin.is_symlink() or not self.pin.is_file() or self._stat(self.pin) != record['identity']:
            raise ValueError('recovery pin inode or bytes identity changed')

    def _pin_candidate(self):
        candidate = self.state.get('candidate')
        if not candidate:
            return
        if not self.slot.is_file() or self._stat(self.slot) != candidate['identity']:
            self.state['candidate'] = None
            self._persist()
            return
        os.link(self.slot, self.pin)
        _fsync_dir(self.directory)
        try:
            self._validate_pin(candidate)
            if self._stat(self.slot) != candidate['identity']:
                raise ValueError('rolling checkpoint changed while creating recovery pin')
            self.state['inflight'] = candidate
            self.state['candidate'] = None
            self._persist()
        except BaseException:
            self.pin.unlink(missing_ok=True)
            _fsync_dir(self.directory)
            raise

    def _capture_controller(self, plan=None):
        from ..common.jsonio import write_bytes_durable
        path=Path((plan or self.plan)['controller']['state_path']).resolve(strict=True)
        before=self._stat5(path);data=path.read_bytes()
        if self._stat5(path)!=before:
            raise ValueError('controller state changed during recovery capture')
        value=json.loads(data)
        if not isinstance(value,dict) or any(key not in value for key in ('phase','checkpoint','round','offset','trained_rounds')):
            raise ValueError('controller recovery state lacks operational custody')
        digest=hashlib.sha256(data).hexdigest()
        target=self.directory/'controller-states'/f'{digest}.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if file_sha256(target)!=digest:raise ValueError('cached controller snapshot differs')
        else:write_bytes_durable(target,data)
        return {'path':str(target),'sha256':digest,'observed_path':str(path),'phase':value['phase'],
                'round':value['round'],'offset':value['offset'],'trained_rounds':value['trained_rounds']}

    def submit(self, path, state, event):
        if Path(path).resolve() != self.slot or not event.get('previous_preserved_until_replace'):
            raise ValueError('recovery submission is not this committed rolling checkpoint')
        with self.cond:
            active_plan=json.loads(json.dumps(self.plan))
            active_plan_sha=self.plan_sha256
        inputs=active_plan['inputs']
        custody=recovery_training_custody(state)
        training = state.get('training') or {}
        if not training.get('optimizer') or not training.get('rng'):
            raise ValueError('recovery export requires full optimizer and RNG state')
        receipts=training.get('source_receipts')
        options=training.get('options') or {}
        if not isinstance(receipts,list) or len(receipts)<2:
            raise ValueError('recovery checkpoint lacks actual source receipts')
        for index,role in enumerate(('train','held')):
            expected=inputs[role]
            actual=receipts[index]
            if any(actual.get(key)!=expected[key] for key in ('path','sha256')):
                raise ValueError('recovery checkpoint source binding differs from plan')
        if options.get('ce_supervision_policy_binding') != inputs['ce']:
            raise ValueError('recovery checkpoint CE binding differs from plan')
        for key,role in (('native_graphs','train'),('foundation_native_graphs','foundation'),('held_native_graphs','held')):
            expected=inputs[role]['path'] if inputs[role] is not None else None
            if options.get(key)!=expected:
                raise ValueError('recovery checkpoint source role differs from plan')
        available={(item.get('path'),item.get('sha256')) for item in active_plan['closure']}
        required={(ref['path'],ref['sha256']) for ref in inputs.values() if ref is not None}
        def add_receipt(receipt):
            if not isinstance(receipt,dict) or not receipt.get('path') or not receipt.get('sha256'):
                raise ValueError('recovery checkpoint source receipt is incomplete')
            required.add((receipt['path'],receipt['sha256']))
            children=receipt.get('sources')
            if children is not None:
                if not isinstance(children,list) or not children:raise ValueError('invalid recovery source tree')
                for child in children:add_receipt(child)
        for receipt in receipts[:2]:add_receipt(receipt)
        from .native_continuation import collect_sft_progress
        declared={(item['kind'],item['path'],item['sha256']) for item in active_plan.get('additional_inputs',[])}
        for role in ('train','foundation','held','ce'):
            ref=inputs[role]
            if ref is not None:declared.add(('ce' if role=='ce' else 'source',ref['path'],ref['sha256']))
        for frame in collect_sft_progress(training,borrow_frames=True).values():
            roots=frame.get('source_receipts') or []
            for root in roots:
                add_receipt(root)
                if ('source',root['path'],root['sha256']) not in declared:
                    raise ValueError('recovery plan lacks a carried SFT source root')
            saved_options=frame.get('options') or {};ce=saved_options.get('ce_supervision_policy_binding')
            if ce:
                required.add((ce['path'],ce['sha256']))
                if ('ce',ce['path'],ce['sha256']) not in declared:
                    raise ValueError('recovery plan lacks a carried SFT CE root')
            foundation=saved_options.get('foundation_native_graphs')
            if foundation and not any(kind=='source' and path==foundation for kind,path,_ in declared):
                raise ValueError('recovery plan lacks a carried SFT foundation root')
        if not required <= available:
            raise ValueError('recovery closure omits actual source or CE dependencies')
        step = training.get('global_step')
        if type(step) is not int:
            raise ValueError('recovery export requires an earned global step')
        controller_state=self._capture_controller(active_plan)
        from ..common.checkpoint_digest import verified_checkpoint_digest
        source_before=self._stat5(self.slot)
        checkpoint_sha256=verified_checkpoint_digest(self.slot)
        if self._stat5(self.slot)!=source_before:
            raise ValueError('committed checkpoint changed during recovery binding')
        with self.cond:
            if self.plan_sha256!=active_plan_sha:
                raise ValueError('recovery plan changed while binding checkpoint')
            prior=[x['global_step'] for x in (self.state.get('candidate'),self.state.get('inflight'),self.state.get('latest_verified')) if x]
            if prior and step < max(prior):
                raise ValueError('recovery checkpoint steps must advance')
            self.state['candidate'] = {'identity':self._stat(self.slot), 'global_step':step,
                                       'plan':active_plan, 'plan_sha256':active_plan_sha, 'custody':custody,
                                       'checkpoint_sha256':checkpoint_sha256,'controller_state':controller_state}
            self._persist()
            if self.state['inflight'] is None:
                self._pin_candidate()
            self.cond.notify_all()

    def _copy_closure(self, plan, controller_state=None):
        result=[]
        closure=list(plan['closure'])
        if controller_state:
            closure.append({'role':'controller-state','path':controller_state['path'],'sha256':controller_state['sha256']})
        for item in closure:
            if set(item) != {'role','path','sha256'} or not isinstance(item['role'], str):
                raise ValueError('invalid recovery closure binding')
            source=Path(item['path']).resolve(strict=True)
            digest=item['sha256']
            if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):
                raise ValueError('invalid recovery closure digest')
            target=self.destination/'closure'/digest
            cache=self.state.setdefault('closure_cache',{})
            identity=verified_recovery_blob(source,target,digest,copier=self.copier,cached_identity=cache.get(digest))
            with self.cond:
                cache[digest]=identity
                self._persist()
            result.append({'role':item['role'],'source':str(source),'declared_path':item['path'],'sha256':digest,'path':str(target),'file_identity':identity})
        return result

    def _leased(self, record):
        from scripts.coreweave.retention_leases import leased_roles
        leases=self.directory/'standalone-retention-consumer-leases.json'
        role=f"local-recovery-pin:{self.plan['session']}:{record['global_step']}:{record['identity']['device']}:{record['identity']['inode']}"
        return role in leased_roles(leases)

    def _export_one(self):
        import fcntl
        with self.cond:
            if self.state['inflight'] is None:
                self._pin_candidate()
            record=self.state['inflight']
        if record is None:
            return False
        self._validate_pin(record)
        if record.get('plan_sha256') != self._plan_digest(record.get('plan')):
            raise ValueError('recovery export plan snapshot changed')
        closure=self._copy_closure(record['plan'],record['controller_state'])
        stage=self.destination/'checkpoints'/f"{self.plan['session']}-{record['global_step']}-{record['identity']['device']}-{record['identity']['inode']}"
        stage.mkdir(parents=True,exist_ok=True)
        target=stage/'checkpoint.pt'
        if target.exists():
            actual=file_sha256(target)
            if actual != file_sha256(self.pin):
                raise ValueError('existing recovery checkpoint differs')
            copied={'sha256':actual,'bytes':target.stat().st_size}
        else:
            copied=self.copier(self.pin,target)
        self._validate_pin(record)
        if copied['sha256']!=record['checkpoint_sha256']:
            raise ValueError('independent recovery bytes differ from committed checkpoint digest')
        self.checkpoint_validator(target, record['custody'])
        target_identity=self._stat5(target)
        if file_sha256(target)!=copied['sha256'] or self._stat5(target)!=target_identity:
            raise ValueError('independent checkpoint changed before retirement admission')
        receipt={'schema':'natlang.local-full-recovery-export/1','session':self.plan['session'],
                 'global_step':record['global_step'],'source_identity':record['identity'],'training_custody':record['custody'],
                 'controller_state':record['controller_state'],
                 'checkpoint':{'path':str(target),'sha256':copied['sha256'],'bytes':copied['bytes'],'file_identity':target_identity},
                 'closure':closure,'plan':record['plan'],'plan_sha256':record['plan_sha256'],'inputs':record['plan']['inputs'],'full_state':True,'independent_readback':True}
        receipt_path=stage/'receipt.json'
        self._write_json(receipt_path,receipt)
        checked={str(target):target_identity,str(receipt_path):self._stat5(receipt_path)}
        checked.update({item['path']:item['file_identity'] for item in closure})
        # Registration and retirement share the same lock used by managed consumers.
        with (self.directory/'standalone-cloud-retention.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if any(self._stat5(path)!=identity for path,identity in checked.items()):
                raise RuntimeError('recovery dependency changed during retirement admission')
            if self._leased(record):
                raise RuntimeError('recovery pin has an active consumer lease')
            retirement_path=self.directory/'local-recovery-retirement.json'
            retirement=(json.loads(retirement_path.read_text()) if retirement_path.exists()
                        else {'retired_roles':[],'operations':[]})
            role=f"local-recovery-pin:{self.plan['session']}:{record['global_step']}:{record['identity']['device']}:{record['identity']['inode']}"
            if role not in retirement['retired_roles']:
                retirement['retired_roles'].append(role)
                self._write_json(retirement_path,retirement)
            with self.cond:
                self._validate_pin(record)
                self.state['previous_verified']=self.state.get('latest_verified')
                self.state['latest_verified']={'path':str(receipt_path),'sha256':file_sha256(receipt_path),
                                                'checkpoint_sha256':copied['sha256'],'global_step':record['global_step']}
                self.state['latest_verified_status']='verified-current-identity'
                self.state['inflight']['verified_receipt']=self.state['latest_verified']
                self._persist()
                self.pin.unlink()
                _fsync_dir(self.directory)
                self.state['inflight']=None
                self._persist()
                self._pin_candidate()
        return True

    def _run(self):
        while True:
            with self.cond:
                while not self.closed and not self.state['inflight'] and not self.state['candidate']:
                    self.cond.wait()
                if self.closed:
                    return
            try:
                self._export_one()
            except Exception as error:
                with self.cond:
                    self.state['last_error']={'type':type(error).__name__,'message':str(error)[:500]}
                    self._persist()
                    self.cond.wait(timeout=30)

    def close(self):
        with self.cond:
            self.closed=True
            self.cond.notify_all()
        if self.thread is not None:
            self.thread.join()
        self._owner.close()


def _read_bound_recovery_bytes(reference):
    if not isinstance(reference,dict) or set(reference)!={'path','sha256'}:
        raise ValueError('invalid recovery plan reference')
    path=Path(reference['path']).resolve(strict=True)
    before=path.stat();data=path.read_bytes();after=path.stat()
    identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if identity(before)!=identity(after) or hashlib.sha256(data).hexdigest()!=reference['sha256']:
        raise ValueError('recovery plan binding changed')
    return data


def _read_bound_recovery_json(reference):
    return json.loads(_read_bound_recovery_bytes(reference))


def _plan_matches_checkpoint(plan, state):
    training=state.get('training') if isinstance(state,dict) else None
    if not isinstance(training,dict):return False
    receipts=training.get('source_receipts') or [];options=training.get('options') or {};inputs=plan.get('inputs') or {}
    if len(receipts)<2:return False
    for index,role in enumerate(('train','held')):
        expected=inputs.get(role)
        if not isinstance(expected,dict) or any(receipts[index].get(k)!=expected.get(k) for k in ('path','sha256')):
            return False
    if options.get('ce_supervision_policy_binding')!=inputs.get('ce'):return False
    for key,role in (('native_graphs','train'),('foundation_native_graphs','foundation'),('held_native_graphs','held')):
        ref=inputs.get(role);expected=ref.get('path') if isinstance(ref,dict) else None
        if options.get(key)!=expected:return False
    from .native_continuation import collect_sft_progress
    declared={(x['kind'],x['path'],x['sha256']) for x in plan.get('additional_inputs',[])}
    for role in ('train','foundation','held','ce'):
        ref=inputs.get(role)
        if isinstance(ref,dict):declared.add(('ce' if role=='ce' else 'source',ref['path'],ref['sha256']))
    for frame in collect_sft_progress(training,borrow_frames=True).values():
        for ref in frame.get('source_receipts') or []:
            if ('source',ref.get('path'),ref.get('sha256')) not in declared:return False
        saved=frame.get('options') or {};ce=saved.get('ce_supervision_policy_binding')
        if ce and ('ce',ce.get('path'),ce.get('sha256')) not in declared:return False
        foundation=saved.get('foundation_native_graphs')
        if foundation and not any(kind=='source' and path==foundation for kind,path,_ in declared):return False
    return True


def _read_recovery_catalog(plan_path):
    path=Path(plan_path).resolve(strict=True);before=path.stat();data=path.read_bytes();after=path.stat()
    identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if identity(before)!=identity(after):raise ValueError('recovery plan catalog changed while reading')
    document=json.loads(data)
    if document.get('schema')!='natlang.local-full-recovery-catalog/1':
        if os.environ.get('NATLANG_REQUIRE_FULL_RECOVERY')=='1':
            raise ValueError('required full recovery needs an immutable plan catalog')
        return [document],0
    references=document.get('plans');initial=document.get('initial_plan');control=document.get('controller_config')
    if (not isinstance(references,list) or not references or initial not in references
            or not isinstance(control,dict) or set(control)!={'path','sha256'}
            or os.environ.get('NATLANG_RECOVERY_CONTROL_SHA')!=control['sha256']):
        raise ValueError('recovery catalog lacks its exact active controller binding')
    _read_bound_recovery_bytes(control)
    plans=[_read_bound_recovery_json(ref) for ref in references]
    for plan in plans:
        validate_recovery_plan_shape(plan)
        if plan['controller']['config']!=control:raise ValueError('recovery catalog mixes controller identities')
    return plans,references.index(initial)


def _read_recovery_selection(plan_path, state=None):
    plans,initial=_read_recovery_catalog(plan_path)
    if state is None:return plans[initial]
    matches=[plan for plan in plans if _plan_matches_checkpoint(plan,state)]
    if len(matches)!=1:raise ValueError('saved checkpoint has no unique restore-qualified recovery plan')
    return matches[0]


def require_recovery_candidate(plan_path, inputs):
    """Require a closed staged plan without selecting it for an older checkpoint."""
    plans,_=_read_recovery_catalog(plan_path)
    matches=[plan for plan in plans if plan.get('inputs')==inputs]
    if len(matches)!=1:raise ValueError('resident source has no unique staged recovery plan')
    validate_recovery_plan_dependencies(matches[0])
    return matches[0]['restore_qualification']


def recovery_queue_for(slot, plan_path, state=None):
    """Select by actual saved inputs; staged catalog entries are never promoted early."""
    plan=_read_recovery_selection(plan_path,state)
    key=str(Path(slot).resolve())
    with _RECOVERY_QUEUES_GUARD:
        queue=_RECOVERY_QUEUES.get(key)
        if queue is None:
            queue=RecoveryExportQueue(slot,plan,copier=child_copy)
            _RECOVERY_QUEUES[key]=queue
        elif queue.plan_sha256!=queue._plan_digest(plan):
            queue.update_plan(plan)
        return queue
