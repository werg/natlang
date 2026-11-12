"""Asynchronous checkpoint offload to the external disk (owner 2026-10-13: "move active run checkpoints to external on
an ongoing basis, async from write"). plans/STORAGE_POLICY.md section 2a.

Writes are unchanged: every full state goes to the NVMe (pending file, fsync, atomic rename; trajectory_state). After a
successful write it is pinned and queued to an independently supervised low-priority memory-ledger owner

    <mirror root>/<path relative to ~/natlang/runs>      (default mirror root: <data_hdd>/natlang-active-lineage)

in that service at best-effort I/O priority 2 and CPU nice 19: copy to ``<mirror>.offload.part`` in 8 MiB chunks with a streamed
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
failed file remains queued for retry. New pending writes of one path supersede older pending generations. An active
copy is a protected backup and retains its inode pin until completion; its newest queued successor follows it.

The mirror is used only when its root lies on a mounted filesystem other than the source's (never a directory on the
same NVMe, e.g. inside a container without the mount). Settings (recorded in ``checkpoint-policy.json``):
``NATLANG_CHECKPOINT_OFFLOAD=0`` disables, ``NATLANG_CHECKPOINT_MIRROR`` sets the root,
``NATLANG_CHECKPOINT_OFFLOAD_MIN_GB`` (default 1) the smallest file offloaded, ``NATLANG_CHECKPOINT_OFFLOAD_LATEST=0``
stops mirroring the resumable slot, ``NATLANG_CHECKPOINT_OFFLOAD_HDD_FLOOR_GB`` (default 50) the free space kept on the
external disk. The checkpoint policy only detaches from the independent owner; explicit final shutdown may call
``drain_offloader`` to wait for all durable queue entries.
"""
from __future__ import annotations

import ctypes
import fcntl
import fnmatch
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import asdict, dataclass
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
OWNER_UNIT = 'natlang-checkpoint-offload-owner'
OWNER_IDLE_SECONDS = 2.0


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


def file_sha256(path, *, drop_cache=True, cancel=None) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        if drop_cache:
            _drop(stream)
        read = 0
        while block := stream.read(CHUNK):
            if cancel is not None and cancel():
                raise CheckpointCopySuperseded(f'{path} has a higher-priority checkpoint waiting')
            digest.update(block)
            read += len(block)
            if drop_cache and read % SYNC_EVERY == 0:
                _drop(stream)  # unified memory (GB10): cached pages come out of the same RAM as CUDA allocations
    return digest.hexdigest()


class CheckpointCopySuperseded(RuntimeError):
    pass


def verified_copy(path, dest, *, partial_suffix=PART_SUFFIX, cancel=None) -> dict:
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
                if cancel is not None and cancel():
                    raise CheckpointCopySuperseded(f'{path} has a newer pending checkpoint generation')
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
            if cancel is not None and cancel():
                raise CheckpointCopySuperseded(f'{path} has a newer pending checkpoint generation')
        os.utime(partial, ns=(before.st_atime_ns, before.st_mtime_ns))
        digest = source_hash.hexdigest()
        if partial.stat().st_size != before.st_size or file_sha256(partial, cancel=cancel) != digest:
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
    with open(temp, 'w') as stream:
        stream.write(json.dumps(record, indent=1) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)
    _fsync_dir(target.parent)


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


# The independently supervised checkpoint offload owner ----------------------------------------------------------
def _lower_priority(pid: int) -> None:
    """Keep CPU work gentle, but prioritize earned recovery above bulk conversion."""
    try:
        os.setpriority(os.PRIO_PROCESS, pid, 19)
    except OSError:
        pass
    number = {'x86_64': 251, 'aarch64': 30}.get(platform.machine())
    if number is not None:
        try:
            ctypes.CDLL(None, use_errno=True).syscall(number, 1, pid, (2 << 13) | 2)  # IOPRIO_WHO_PROCESS, CLASS_BE; training BE0 remains ahead
        except (OSError, AttributeError):
            pass


def owned_copy(path, dest, prune=False) -> dict:
    """Copy in a caller-owned recovery-export worker thread at low CPU/I/O priority."""
    _lower_priority(threading.get_native_id())
    return local_copy(path, dest, prune=prune)


def queue_root() -> Path:
    return Path(os.environ.get('NATLANG_CHECKPOINT_OFFLOAD_QUEUE') or
                _paths().resolve('data_nvme', 'natlang-checkpoint-offload-queue'))


def _durable_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.pending')
    with open(temp, 'w') as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


def _queue_files(root: Path) -> list[Path]:
    pending = root / 'pending'
    return sorted(pending.glob('*.json')) if pending.is_dir() else []


def _queue_key(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()


def _pending_id(descriptor: Path) -> str | None:
    try:
        return json.loads(descriptor.read_text()).get('id')
    except (OSError, ValueError, AttributeError):
        return None


def _owner_unit(root: Path) -> str:
    return OWNER_UNIT + '-' + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:10]


def _enqueue(root: Path, path: Path, role: str, prune, settings: OffloadSettings) -> dict:
    """Hard-link the exact completed checkpoint inode and atomically replace its pending generation."""
    identity = _identity(path.stat())
    key = _queue_key(path)
    job_id = f'{time.time_ns()}-{os.getpid()}-{threading.get_native_id()}'
    pin = path.with_name(f'.{path.name}.offload-pin-{job_id}')
    os.link(path, pin, follow_symlinks=False)
    pinned = os.stat(pin, follow_symlinks=False)
    if _identity(pinned) != identity:
        pin.unlink(missing_ok=True)
        raise RuntimeError('checkpoint changed while creating immutable offload pin')
    job = {'schema': 'natlang.checkpoint-offload-job/1', 'id': job_id, 'key': key,
           'path': str(path), 'pin': str(pin), 'role': role, 'prune': prune,
           'source_identity': identity, 'source_bytes': pinned.st_size,
           'mirror_root': str(settings.mirror_root), 'settings': settings.record(),
           'queue_root': str(root), 'created': time.time()}
    descriptor = root / 'pending' / f'{key}.json'
    try:
        old = json.loads(descriptor.read_text())
    except (OSError, ValueError):
        old = None
    try:
        active = json.loads((root / 'active' / f'{key}.json').read_text())
    except (OSError, ValueError):
        active = None
    _durable_json(descriptor, job)
    if old and old.get('id') != job_id and not (active and active.get('id') == old.get('id')):
        old_pin = Path(str(old.get('pin', '')))
        if old_pin != pin and old_pin.name.startswith('.') and 'offload-pin-' in old_pin.name:
            old_pin.unlink(missing_ok=True)
    return job


def _enqueue_initial_sweep(root: Path, path: Path, settings: OffloadSettings) -> None:
    """Queue checkpoint artifacts already present in this run, once per stage directory."""
    key = hashlib.sha256(str(path.parent).encode()).hexdigest()
    marker = root / 'swept' / f'{key}.json'
    if marker.exists():
        return
    siblings = [d for d in sorted(path.parent.parent.iterdir()) if d.is_dir() and d != path.parent] \
        if _within_a_run(path) else []
    for directory, own in ((path.parent, True), *((d, False) for d in siblings)):
        for item, item_role in sweep_targets(directory, settings, own=own):
            if item == path:
                continue
            _enqueue(root, item, item_role, 'auto', settings)
    _durable_json(marker, {'schema': 'natlang.checkpoint-offload-sweep/1',
                           'directory': str(path.parent), 'time': time.time()})


def _launch_owner(root: Path) -> None:
    """Admit the copy service under memory_ledger; no mirror copy runs in the learner cgroup."""
    repo = Path(_paths().root('repo'))
    ledger = repo / 'scripts' / 'memory_ledger.py'
    py_path = str(repo / 'training' / 'neuralese') + os.pathsep + str(repo)
    root.mkdir(parents=True, exist_ok=True)
    unit = _owner_unit(root)
    owner_service = unit + '.service'
    waiter_service = unit + '-waiter.service'
    for _ in range(100):
        active = any(subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
                     for unit in (owner_service, waiter_service))
        idle_marker = root / 'owner-idle.json'
        if active and not idle_marker.exists():
            return
        if active:
            time.sleep(.1)
            continue
        idle_marker.unlink(missing_ok=True)
        command = [sys.executable, str(ledger), 'run', '--unit', unit, '--budget-gb', '2',
               '--reserve-gb', '0.5', '--class', 'service', '--priority', 'low',
               '--resumable', '--wait', '86400', '--detach', '--workdir', str(repo),
               '--env', 'PYTHONPATH=' + py_path,
               '--env', 'NATLANG_CHECKPOINT_OFFLOAD_QUEUE=' + str(root), '--',
               sys.executable, '-m', 'natlang_neuralese.train.checkpoint_offload', '--owner-loop']
        result = subprocess.run(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                timeout=30)
        if result.returncode == 0:
            return
        time.sleep(.1)
    raise RuntimeError('could not start checkpoint offload ledger owner: ' +
                       (result.stderr or result.stdout)[-500:])


def _offload_pinned(job: dict, settings: OffloadSettings, copier=None) -> dict:
    path, pin = Path(job['path']), Path(job['pin'])
    if pin.parent != path.parent or not pin.name.startswith(f'.{path.name}.offload-pin-'):
        raise ValueError('offload job pin is not an adjacent checkpoint snapshot')
    if not pin.is_file() or _identity(pin.stat()) != list(job['source_identity']):
        raise RuntimeError(f"immutable checkpoint pin is missing or changed: {pin}")
    settings = OffloadSettings(**job['settings'])
    role = job['role']
    mirror = mirror_path(path, settings)
    if mirror is None:
        raise RuntimeError('source path is outside the mirrored trees')
    if mirror_unavailable(path, settings):
        raise OSError('mirror filesystem unavailable')
    manifest = pinned_by_manifest(path) if role == 'archive' else None
    if manifest:
        return {'event': 'checkpoint_offload_skipped', 'path': str(path),
                'reason': f'pinned by corpus manifest {manifest}'}
    if foreign_partial(mirror):
        raise OSError(f'another copy in progress: {foreign_partial(mirror)}')
    clean_stale(mirror.parent, older_than=settings.stale_part_seconds)
    record = read_record(mirror)
    copied = False
    record_current = bool(record and record.get('source_identity') == list(job['source_identity']) and
                          mirror.is_file() and mirror.stat().st_size == record.get('bytes') and
                          (record.get('optimizer_pruned') or record.get('bytes') == job['source_bytes']))
    if not record_current:
        record = None
        prune = job.get('prune') if role == 'archive' and settings.prune_archived else False
        if role == 'archive' and not prune:
            record = _hdd_link_shortcut(path, mirror)
        if record is None:
            free = shutil.disk_usage(_mount_of(Path(settings.mirror_root)) or settings.mirror_root).free
            if free - int(job['source_bytes']) < settings.hdd_floor_bytes:
                raise OSError(f'external disk floor: {free} bytes free, {job["source_bytes"]} to copy, floor {settings.hdd_floor_bytes}')
            if copier is not None:
                record = copier(pin, mirror, prune)
            elif prune:
                record = local_copy(pin, mirror, prune)
            else:
                # Active copies are protected backups. New generations replace
                # only the pending descriptor and follow this pinned copy.
                record = verified_copy(pin, mirror)
            record = dict(record, source=str(path), source_identity=list(job['source_identity']),
                          source_pin=str(pin), time=time.time())
            write_record(mirror, record)
            copied = True
    event = {'event': 'checkpoint_offloaded', 'role': role, 'path': str(path), 'mirror': str(mirror),
             'bytes': record['bytes'], 'sha256': record['sha256'], 'copied': copied,
             'optimizer_pruned': bool(record.get('optimizer_pruned')), 'source_bytes': record.get('source_bytes'),
             'queue_job_id': job['id'], 'source_identity': list(job['source_identity'])}
    if role == 'archive' and settings.link_archived and not path.is_symlink():
        with path_lock(path):
            try:
                unchanged = _identity(path.lstat()) == list(job['source_identity'])
            except OSError:
                unchanged = False
            if unchanged:
                if record.get('optimizer_pruned'):
                    event['receipts'] = rebind_receipts(path, record)
                replace_with_link(path, mirror, expect=job['source_identity'])
                event['linked'] = True
    return event


def _process_queue_once(root: Path, *, settings=None, copier=None) -> dict | None:
    settings = settings or OffloadSettings.from_env()
    decoded = []
    for descriptor in _queue_files(root):
        try:
            job = json.loads(descriptor.read_text())
            if job.get('schema') != 'natlang.checkpoint-offload-job/1' or descriptor.stem != job.get('key'):
                continue
            decoded.append((0 if job.get('role') == 'latest' else 1, -float(job.get('created', 0)), descriptor, job))
        except (OSError, ValueError):
            continue
    if not decoded:
        return None
    _, _, descriptor, job = min(decoded)
    event_path = Path(job['path']).parent / 'checkpoint-offload.jsonl'
    active_path = root / 'active' / f"{job['key']}.json"
    _durable_json(active_path, {'schema': 'natlang.checkpoint-offload-active/1',
                                'id': job['id'], 'pin': job['pin'], 'started': time.time()})
    try:
        event = _offload_pinned(job, settings, copier=copier)
        with open(event_path, 'a') as stream:
            stream.write(json.dumps(dict(event, time=time.time())) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        try:
            current = json.loads(descriptor.read_text())
        except (OSError, ValueError):
            current = None
        if current and current.get('id') == job.get('id'):
            descriptor.unlink(missing_ok=True)
            Path(job['pin']).unlink(missing_ok=True)
            _fsync_dir(descriptor.parent)
        elif current:
            Path(job['pin']).unlink(missing_ok=True)
        return event
    except CheckpointCopySuperseded as error:
        superseded = {'event': 'checkpoint_offload_superseded', 'path': job.get('path'),
                      'queue_job_id': job.get('id'), 'error': str(error)[:300]}
        try:
            with open(event_path, 'a') as stream:
                stream.write(json.dumps(dict(superseded, time=time.time())) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            pass
        return superseded
    except Exception as error:
        failure = {'event': 'checkpoint_offload_failed', 'path': job.get('path'), 'role': job.get('role'),
                   'queue_job_id': job.get('id'), 'error': repr(error)[:500], 'retry': 'owner will retry'}
        try:
            with open(event_path, 'a') as stream:
                stream.write(json.dumps(dict(failure, time=time.time())) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            pass
        print(json.dumps(failure), flush=True)
        return failure
    finally:
        try:
            active = json.loads(active_path.read_text())
        except (OSError, ValueError):
            active = None
        if active and active.get('id') == job.get('id'):
            active_path.unlink(missing_ok=True)
            _fsync_dir(active_path.parent)
        try:
            current = json.loads(descriptor.read_text())
        except (OSError, ValueError):
            current = None
        if not current or current.get('id') != job.get('id'):
            Path(job['pin']).unlink(missing_ok=True)


def _clean_stale_active(root: Path) -> None:
    """Remove pins left by a crashed owner only when they are no longer the pending retry."""
    active_dir = root / 'active'
    if not active_dir.is_dir():
        return
    for active_path in active_dir.glob('*.json'):
        try:
            active = json.loads(active_path.read_text())
            pending = json.loads((root / 'pending' / active_path.name).read_text())
        except (OSError, ValueError):
            pending = None
            active = {}
        if pending and pending.get('id') == active.get('id'):
            continue  # interrupted in-flight copy; the exact pin is its retry input
        pin = Path(str(active.get('pin', '')))
        if pin.name.startswith('.') and 'offload-pin-' in pin.name:
            pin.unlink(missing_ok=True)
        active_path.unlink(missing_ok=True)


def run_offload_owner(*, root=None, idle_seconds=OWNER_IDLE_SECONDS, retry_seconds=60.0, copier=None) -> None:
    """Ledger service entrypoint. Queue pins/descriptors are its durable recovery state."""
    root = Path(root or queue_root())
    (root / 'owner-idle.json').unlink(missing_ok=True)
    _clean_stale_active(root)
    _lower_priority(os.getpid())
    last_work = time.monotonic()
    while True:
        event = _process_queue_once(root, copier=copier)
        if event and event.get('event') in ('checkpoint_offloaded', 'checkpoint_offload_superseded'):
            last_work = time.monotonic()
            continue
        if _queue_files(root):
            # Keep the ledger owner alive for a bounded retry window; failures
            # such as a disconnected HDD remain durable across service restarts.
            time.sleep(retry_seconds if event else min(1.0, retry_seconds))
            continue
        # Idle exit is serialized with producer writes to avoid a missed wakeup.
        root.mkdir(parents=True, exist_ok=True)
        with open(root / 'launch.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if _queue_files(root):
                continue
            if time.monotonic() - last_work >= idle_seconds:
                _durable_json(root / 'owner-idle.json', {'pid': os.getpid(), 'at': time.time()})
                return
        time.sleep(min(1.0, max(.05, idle_seconds - (time.monotonic() - last_work))))
        if _queue_files(root):
            time.sleep(retry_seconds if (root / 'last-failure').exists() else .05)


class CheckpointOffloader:
    """Queue producer. ``close`` never waits for the independent copy owner."""

    def __init__(self, settings: OffloadSettings | None = None, *, copier=None, queue=None, launch=None):
        self.settings = settings or OffloadSettings.from_env()
        self.copier = copier
        self.root = Path(queue or queue_root())
        self.launch = (copier is None) if launch is None else launch
        self.events: list[dict] = []
        self.closing = False

    def submit(self, path, role: str | None = None, prune=False) -> None:
        try:
            path = _absolute(path)
            role = role or role_of(path, self.settings)
            if role is None or (role == 'latest' and not self.settings.mirror_latest):
                return
            if path.is_symlink() or path.stat().st_size < self.settings.min_bytes:
                return
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.root / 'launch.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                _enqueue(self.root, path, role, prune, self.settings)
                try:
                    _enqueue_initial_sweep(self.root, path, self.settings)
                except OSError as error:
                    self._event({'event': 'checkpoint_offload_sweep_failed', 'path': str(path),
                                 'error': repr(error)})
                if self.launch:
                    _launch_owner(self.root)
        except Exception as error:  # checkpoint publication remains successful
            self._event({'event': 'checkpoint_offload_failed', 'path': str(path), 'error': repr(error)})

    def wait(self, timeout: float | None = None) -> bool:
        """Explicit synchronous drain for tests/operators; training handoffs never call it."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while _queue_files(self.root):
            if deadline is not None and time.monotonic() >= deadline:
                return False
            event = _process_queue_once(self.root, settings=self.settings, copier=self.copier)
            if event is None:
                return False
            self.events.append(event)
            if event.get('event') == 'checkpoint_offload_failed':
                if deadline is None:
                    return False
                time.sleep(.05)
        return True

    def close(self, timeout: float | None = None) -> bool:
        self.closing = True
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


def _owner_cli() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-loop', action='store_true')
    parser.add_argument('--drain', action='store_true')
    args = parser.parse_args()
    if args.owner_loop:
        run_offload_owner()
        return 0
    if args.drain:
        return 0 if drain_offloader(None) else 1
    parser.error('choose --owner-loop or --drain')


if __name__ == '__main__':
    raise SystemExit(_owner_cli())


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


def close_offloader(timeout: float | None = None) -> bool:
    """Close the process-local queue producer without waiting for mirrored checkpoint I/O."""
    global _OFFLOADER
    with _OFFLOADER_GUARD:
        worker = _OFFLOADER
        _OFFLOADER = None
    if worker is None:
        return True
    return worker.close(timeout)


def drain_offloader(timeout: float | None = None) -> bool:
    """Explicit final-shutdown/operator drain; resumable training handoffs must not call this."""
    root = queue_root()
    deadline = None if timeout is None else time.monotonic() + timeout
    while _queue_files(root):
        try:
            root.mkdir(parents=True, exist_ok=True)
            with open(root / 'launch.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                _launch_owner(root)
        except Exception:
            if deadline is not None and time.monotonic() >= deadline:
                return False
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(.2)
    return True


def after_write(path, role: str | None = None, prune=False) -> None:
    """The hook every checkpoint writer calls after a successful write or best publication. Never raises."""
    try:
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
