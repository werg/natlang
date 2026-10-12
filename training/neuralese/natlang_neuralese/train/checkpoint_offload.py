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
    """Where ``path`` is mirrored: relative to ~/natlang/runs under the root (``repo/``, ``home/`` for other files of
    the checkout or the home directory); None outside them."""
    path = _absolute(path)
    repo = Path(os.path.realpath(_paths().root('repo')))
    root = Path(settings.mirror_root)
    for base, prefix in ((repo / 'runs', ()), (repo, ('repo',)), (Path(os.path.realpath(Path.home())), ('home',))):
        if base in path.parents:
            return root.joinpath(*prefix, path.relative_to(base))
    return None


def _mount_of(path: Path) -> Path | None:
    """The mount point (other than ``/``) the mirror root lives on, or None."""
    for candidate in (path, *path.parents):
        if candidate == Path('/'):
            return None
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
    mount = _mount_of(Path(settings.mirror_root))
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
    """Copy ``path`` to ``dest`` through ``dest + partial_suffix`` (mtime kept): fsync, check the source did not
    change, re-read the copy and compare size and SHA-256, then rename into place and fsync the directory. Raises,
    leaving nothing at ``dest``'s partial name, on any doubt."""
    path, dest = Path(path), Path(dest)
    before = path.lstat()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + partial_suffix)
    source_hash = hashlib.sha256()
    try:
        with open(path, 'rb') as src, open(partial, 'wb') as out:
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
        os.utime(partial, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.lstat()
        if _identity(after) != _identity(before):
            raise RuntimeError(f'{path} changed during the copy')
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


_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def path_lock(path) -> threading.Lock:
    """One in-process lock per NVMe path: a best publication and the offload's link swap never interleave."""
    key = str(_absolute(path))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


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


def offload_one(path, role: str, settings: OffloadSettings, *, copier=None) -> dict:
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
    if record is None and role == 'archive' and path.name != 'checkpoint.pt':
        record = _hdd_link_shortcut(path, mirror)
    if record is None:
        size = path.stat().st_size
        free = shutil.disk_usage(_mount_of(Path(settings.mirror_root)) or settings.mirror_root).free
        if free - size < settings.hdd_floor_bytes:
            raise OSError(f'external disk floor: {free} bytes free, {size} to copy, floor {settings.hdd_floor_bytes}')
        record = (copier or verified_copy)(path, mirror)
        write_record(mirror, record)
        copied = True
    event = {'event': 'checkpoint_offloaded', 'role': role, 'path': str(path), 'mirror': str(mirror),
             'bytes': record['bytes'], 'sha256': record['sha256'], 'copied': copied}
    if role == 'archive' and settings.link_archived:
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


def child_copy(path, dest) -> dict:
    """``verified_copy`` in a child process at idle priority: the trainer keeps its GIL and its I/O share."""
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'copy', str(path), str(dest)],
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

    def submit(self, path, role: str | None = None) -> None:
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
                for retry, retry_role in list(self.failed.items()):
                    self.pending.setdefault(retry, retry_role)
                self.failed.clear()
                self.pending.pop(str(path), None)  # coalesce: the newest submission goes last
                self.pending[str(path)] = role
                if str(path.parent) not in self.swept:  # leftovers of earlier processes: this and sibling stages
                    self.swept.add(str(path.parent))
                    siblings = [d for d in sorted(path.parent.parent.iterdir()) if d.is_dir() and d != path.parent
                                ] if _within_a_run(path) else []
                    for directory, own in ((path.parent, True), *((d, False) for d in siblings)):
                        for item, item_role in sweep_targets(directory, self.settings, own=own):
                            self.pending.setdefault(str(item), item_role)
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
                path, role = next(iter(self.pending.items()))
                del self.pending[path]
                self.busy = True
            try:
                self._event(offload_one(path, role, self.settings, copier=self.copier))
            except Exception as error:
                with self.cond:
                    self.failed[path] = role
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


def after_write(path, role: str | None = None) -> None:
    """The hook every checkpoint writer calls after a successful write or best publication. Never raises."""
    try:
        settings = _OFFLOADER.settings if _OFFLOADER is not None else OffloadSettings.from_env()
        if mirror_unavailable(path, settings) is None:
            offloader().submit(path, role)
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


if __name__ == '__main__' and len(sys.argv) == 4 and sys.argv[1] == 'copy':
    print(json.dumps(verified_copy(sys.argv[2], sys.argv[3])), flush=True)
