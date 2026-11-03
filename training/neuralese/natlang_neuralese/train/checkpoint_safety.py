"""Disk-space guard and recovery metadata for complete warm-up checkpoints."""
from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import stat
import time
from contextlib import contextmanager
from pathlib import Path


class CheckpointReserveError(RuntimeError):
    """There is not enough disk space to guarantee an atomic full checkpoint."""


class CheckpointSpaceUnavailable(CheckpointReserveError):
    """An authenticated atomic write is temporarily short of reserved disk space."""


# Large learners reserve a little extra in the same pending inode before an
# optimizer update. Controller and snapshot metadata can grow between that
# reservation and serialization, even when tensor geometry is unchanged.
_PERSISTENT_HEADROOM_THRESHOLD = 64 * 1024 * 1024
_PERSISTENT_METADATA_HEADROOM = 16 * 1024 * 1024
_PERSISTENT_ALLOCATION_ALIGNMENT = 4096


def checkpoint_size_upper_bound(state, *, margin=1.2, minimum_overhead=16 * 1024 * 1024):
    """Bound the next serialization without materializing another checkpoint copy.

    Dense and sparse tensor storage is counted once by storage identity; aliased
    views therefore do not inflate the estimate, while non-tensor metadata
    receives a conservative fixed allowance. Unknown tensor layouts fail closed.
    """
    import torch

    if not math.isfinite(margin) or margin < 1.0:
        raise ValueError('checkpoint size margin must be finite and at least one')
    seen = set()
    tensor_bytes = 0
    metadata_bytes = 0

    def visit_tensor_storage(value):
        nonlocal tensor_bytes
        storage = value.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes(), value.device.type, value.device.index)
        if key not in seen:
            seen.add(key)
            tensor_bytes += int(storage.nbytes())

    def visit(value):
        nonlocal tensor_bytes, metadata_bytes
        if isinstance(value, torch.Tensor):
            if value.layout == torch.strided:
                if value.is_conj() or value.is_neg():
                    raise CheckpointReserveError('cannot bound conjugate/negative tensor views for checkpoint admission')
                visit_tensor_storage(value)
                return
            if value.layout == torch.sparse_coo:
                visit(value._indices())
                visit(value._values())
                return
            compressed = {
                torch.sparse_csr: ('crow_indices', 'col_indices', 'values'),
                torch.sparse_csc: ('ccol_indices', 'row_indices', 'values'),
                torch.sparse_bsr: ('crow_indices', 'col_indices', 'values'),
                torch.sparse_bsc: ('ccol_indices', 'row_indices', 'values'),
            }
            members = compressed.get(value.layout)
            if members is None:
                raise CheckpointReserveError('cannot bound unsupported tensor storage for checkpoint admission')
            for member in members:
                visit(getattr(value, member)())
            return
        if isinstance(value, dict):
            metadata_bytes += len(value) * 128
            for key, item in value.items():
                if isinstance(key, (str, bytes)):
                    metadata_bytes += len(key)
                visit(item)
            return
        if isinstance(value, (tuple, list, set, frozenset)):
            metadata_bytes += len(value) * 64
            for item in value:
                visit(item)
            return
        if isinstance(value, (str, bytes, bytearray)):
            metadata_bytes += len(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            metadata_bytes += 16
        elif hasattr(value, 'state_dict'):
            visit(value.state_dict())
        else:
            # Pickle metadata is usually small. Unknown objects must not silently
            # be treated as zero-sized.
            metadata_bytes += 4096

    visit(state)
    return int(math.ceil((tensor_bytes + max(minimum_overhead, metadata_bytes)) * margin))


def _checkpoint_admission_lock_path(device):
    configured = os.environ.get('NATLANG_CHECKPOINT_ADMISSION_LOCK_ROOT')
    root = (Path(configured).expanduser() if configured else
            Path.home() / '.local' / 'state' / 'natlang' / 'checkpoint-admission')
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise CheckpointReserveError('checkpoint admission lock root must be a real shared directory')
    return root / f'checkpoint-admission-{os.getuid()}-{int(device)}.lock'


def checkpoint_minimum_free_bytes(target):
    """Apply the GB10 hot-volume floor only to configured NVMe data/repo paths."""
    explicit=os.environ.get('NATLANG_CHECKPOINT_MIN_FREE_BYTES')
    if explicit is not None:return max(0,int(explicit))
    floor=os.environ.get('NATLANG_CHECKPOINT_DGX_NVME_MIN_FREE_BYTES')
    if floor is None:return 512*1024**2
    from ..common.paths import resolve
    target=Path(target).resolve(strict=False);data=resolve('data_nvme').resolve(strict=False);runs=(resolve('repo')/'runs').resolve(strict=False)
    for root in (data,runs):
        if (target==root or root in target.parents) and root.exists():
            parent=target.parent
            while not parent.exists() and parent!=parent.parent:parent=parent.parent
            if os.stat(parent).st_dev==os.stat(root).st_dev:return max(0,int(floor))
    return 512*1024**2


class CheckpointWriteAdmission:
    """Serialize checkpoint writers per filesystem and hold concurrent-transfer space.

    The reserved blocks remain allocated while the atomic writer is active, so
    a same-filesystem checkpoint copy cannot consume the space needed for the
    checkpoint being committed. Writers recheck available bytes only after
    taking the shared per-device lock. Host processes use a stable user-state
    lock root; writers in isolated container mount namespaces must set
    ``NATLANG_CHECKPOINT_ADMISSION_LOCK_ROOT`` to the same shared directory.
    The lock coordinates processes that can see that same lock file.
    """

    _MAGIC = b'NATLANG-CHECKPOINT-TRANSFER-RESERVE/1\n'

    def __init__(self, target, write_bound_bytes, *, transfer_bytes=None,
                 minimum_free_bytes=None, lock_timeout_seconds=None):
        self.target = Path(target)
        self.write_bound_bytes = int(write_bound_bytes)
        if self.write_bound_bytes < 1:
            raise ValueError('checkpoint write bound must be positive')
        if transfer_bytes is None:
            factor = float(os.environ.get('NATLANG_CHECKPOINT_TRANSFER_HEADROOM_FACTOR', '1.0'))
            if not math.isfinite(factor) or factor < 0:
                raise ValueError('checkpoint transfer headroom factor must be finite and nonnegative')
            transfer_bytes = math.ceil(self.write_bound_bytes * factor)
        self.transfer_bytes = max(0, int(transfer_bytes))
        if minimum_free_bytes is None:
            minimum_free_bytes = checkpoint_minimum_free_bytes(self.target)
        self.minimum_free_bytes = max(0, int(minimum_free_bytes))
        self.lock_timeout_seconds = (float(os.environ.get('NATLANG_CHECKPOINT_LOCK_TIMEOUT_SECONDS', '1800'))
                                     if lock_timeout_seconds is None else float(lock_timeout_seconds))
        if not math.isfinite(self.lock_timeout_seconds) or self.lock_timeout_seconds <= 0:
            raise ValueError('checkpoint lock timeout must be finite and positive')
        self._lock_fd = None
        self._reserve = None

    def __enter__(self):
        self.target.parent.mkdir(parents=True, exist_ok=True)
        device = os.stat(self.target.parent).st_dev
        # /tmp may be private to systemd/container namespaces. A stable user
        # state path is shared by host writers; container writers must set the
        # same explicit NATLANG_CHECKPOINT_ADMISSION_LOCK_ROOT on their shared mount.
        lock_path = _checkpoint_admission_lock_path(device)
        self._lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        if not stat.S_ISREG(os.fstat(self._lock_fd).st_mode):
            os.close(self._lock_fd)
            self._lock_fd = None
            raise CheckpointReserveError('checkpoint admission lock is not a regular file')
        deadline = time.monotonic() + self.lock_timeout_seconds
        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(self._lock_fd)
                    self._lock_fd = None
                    raise CheckpointReserveError('timed out waiting for another full-state checkpoint write')
                time.sleep(0.25)
        try:
            reserve_fd = None
            additional_transfer = self.transfer_bytes
            if self.transfer_bytes:
                reserve_path = self.target.parent / f'.natlang-transfer-reserve-{device}'
                reserve_fd = os.open(reserve_path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                try:
                    reserve_stat = os.fstat(reserve_fd)
                    if not stat.S_ISREG(reserve_stat.st_mode) or reserve_stat.st_dev != device:
                        raise CheckpointReserveError(f'checkpoint transfer reserve is not a regular file on target filesystem: {reserve_path}')
                    size = reserve_stat.st_size
                    if size and os.pread(reserve_fd, len(self._MAGIC), 0) != self._MAGIC:
                        raise CheckpointReserveError(f'unowned checkpoint transfer reserve path: {reserve_path}')
                    additional_transfer = max(0, self.transfer_bytes - size)
                    if size < len(self._MAGIC):
                        os.pwrite(reserve_fd, self._MAGIC, 0)
                    free = shutil.disk_usage(self.target.parent).free
                    required = self.write_bound_bytes + additional_transfer + self.minimum_free_bytes
                    if free < required:
                        raise CheckpointSpaceUnavailable(
                            f'atomic checkpoint admission needs {required} free bytes '
                            f'(write={self.write_bound_bytes}, additional_transfer={additional_transfer}, '
                            f'safety={self.minimum_free_bytes}); only {free} are free')
                    if additional_transfer:
                        os.posix_fallocate(reserve_fd, 0, self.transfer_bytes)
                        os.pwrite(reserve_fd, self._MAGIC, 0)
                        os.fsync(reserve_fd)
                    # Keep allocated blocks for the duration of the write.
                    self._reserve = (reserve_fd, reserve_path)
                    reserve_fd = None
                except BaseException:
                    if reserve_fd is not None:
                        os.close(reserve_fd)
                    raise
            else:
                free = shutil.disk_usage(self.target.parent).free
                required = self.write_bound_bytes + self.minimum_free_bytes
                if free < required:
                    raise CheckpointSpaceUnavailable(
                        f'atomic checkpoint admission needs {required} free bytes '
                        f'(write={self.write_bound_bytes}, safety={self.minimum_free_bytes}); only {free} are free')
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        if self._reserve is not None:
            fd, reserve_path = self._reserve
            self._reserve = None
            try:
                st = os.fstat(fd)
                try:
                    path_st = reserve_path.stat(follow_symlinks=False)
                except FileNotFoundError:
                    path_st = None
                if path_st and (path_st.st_dev, path_st.st_ino) == (st.st_dev, st.st_ino):
                    os.unlink(reserve_path)
            finally:
                os.close(fd)
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None
        return False


@contextmanager
def checkpoint_target_lock(target, *, reservation=None):
    """Exclude other cooperating writers while a target has a live owner."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + '.reservation.lock')
    if reservation is not None:
        # The persistent writer owns this lock across updates, including its
        # post-commit hooks. Reuse its open file description without unlocking
        # it; opening the same pathname again would conflict with our own lock.
        fd = reservation._lock_fd
        if reservation.target.resolve() != target.resolve() or fd is None:
            raise CheckpointReserveError('checkpoint lock owner must be live and target the same slot')
        held = os.fstat(fd)
        current = lock.stat(follow_symlinks=False)
        if not stat.S_ISREG(held.st_mode) or (held.st_dev, held.st_ino) != (current.st_dev, current.st_ino):
            raise CheckpointReserveError('checkpoint lock owner inode changed')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
        return
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise CheckpointReserveError('checkpoint reservation lock is not a regular file')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class PersistentCheckpointReservation:
    """Own preallocated blocks in the very inode later published as the checkpoint.

    The target lock remains held across training updates. A crash leaves the
    inode and its identity receipt for the next owner to validate and reuse.
    A receipt never authorizes modifying the previously committed target.
    """
    def __init__(self, target, *, transfer_bytes=None, minimum_free_bytes=None, slot_suffix='', shared_lock=None):
        self.target = Path(target)
        if slot_suffix not in ('', '.alternate'):
            raise ValueError('invalid checkpoint reservation slot')
        self.pending = self.target.with_name(self.target.name + slot_suffix + '.reserved-pending')
        self.receipt = self.target.with_name(self.target.name + slot_suffix + '.reservation.json')
        self.lock = self.target.with_name(self.target.name + '.reservation.lock')
        self.transfer_bytes = transfer_bytes
        self.minimum_free_bytes = minimum_free_bytes
        if shared_lock is not None and (shared_lock.target != self.target or shared_lock.lock != self.lock
                                        or shared_lock._lock_fd is None):
            raise CheckpointReserveError('shared checkpoint lock must be live and target the same slot')
        self._lock_fd = os.dup(shared_lock._lock_fd) if shared_lock is not None else None
        self._fd = None
        self.bound = 0
        self._allocated_bound = 0

    def _check_pending_identity(self):
        st = os.fstat(self._fd)
        path_st = self.pending.stat(follow_symlinks=False)
        if (not stat.S_ISREG(st.st_mode) or st.st_nlink != 1
                or (st.st_dev, st.st_ino) != (path_st.st_dev, path_st.st_ino)):
            raise CheckpointReserveError('checkpoint reservation path or link count changed')
        try:
            target_st = self.target.stat(follow_symlinks=False)
        except FileNotFoundError:
            target_st = None
        if target_st is not None and (st.st_dev, st.st_ino) == (target_st.st_dev, target_st.st_ino):
            raise CheckpointReserveError('checkpoint reservation aliases the committed checkpoint')
        return st

    def acquire(self, bound):
        bound = int(bound)
        if bound < 1:
            raise ValueError('checkpoint reservation bound must be positive')
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self._lock_fd is None:
            self._lock_fd = os.open(self.lock, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
            try:
                if not stat.S_ISREG(os.fstat(self._lock_fd).st_mode):
                    raise CheckpointReserveError('checkpoint reservation lock is not a regular file')
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(self._lock_fd)
                self._lock_fd = None
                raise
        if self._fd is None:
            # A crash after rename can leave the old receipt but no pending
            # name. Remove it only if it names the committed target inode.
            if self.receipt.exists() and not self.pending.exists() and self.target.exists():
                try:
                    old = json.loads(self.receipt.read_text())
                    target_st = self.target.stat(follow_symlinks=False)
                except (OSError, ValueError) as error:
                    raise CheckpointReserveError('invalid post-commit reservation receipt') from error
                if old != {'schema':'natlang.checkpoint-reservation/1',
                           'device':target_st.st_dev, 'inode':target_st.st_ino}:
                    raise CheckpointReserveError('orphaned checkpoint reservation receipt')
                self.receipt.unlink()
                self._sync_directory()
            if self.pending.exists() or self.receipt.exists():
                try:
                    receipt = json.loads(self.receipt.read_text())
                    fd = os.open(self.pending, os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0))
                except (OSError, ValueError) as error:
                    raise CheckpointReserveError('incomplete or invalid checkpoint reservation') from error
                st = os.fstat(fd)
                if (not stat.S_ISREG(st.st_mode) or receipt != {'schema':'natlang.checkpoint-reservation/1',
                        'device':st.st_dev, 'inode':st.st_ino}):
                    os.close(fd)
                    raise CheckpointReserveError('checkpoint reservation inode does not match receipt')
                self._fd = fd
            else:
                fd = os.open(self.pending, os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
                self._fd = fd
                st = os.fstat(fd)
                temp = self.receipt.with_name(self.receipt.name + '.tmp')
                try:
                    with temp.open('x') as stream:
                        json.dump({'schema':'natlang.checkpoint-reservation/1', 'device':st.st_dev,
                                   'inode':st.st_ino}, stream)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temp, self.receipt)
                    self._sync_directory()
                except BaseException:
                    temp.unlink(missing_ok=True)
                    os.close(fd)
                    self._fd = None
                    self.pending.unlink(missing_ok=True)
                    raise
        st = self._check_pending_identity()
        if st.st_size < self._allocated_bound:
            # A failed final truncate or an external truncation invalidates
            # our prior allocation proof for the whole logical range.
            self._allocated_bound = 0
        # Only allocations completed by this owner count. An existing inode's
        # size (or aggregate st_blocks) cannot prove its logical range is
        # allocated after a crash: admit the whole bound conservatively.
        capacity = bound
        if bound > self._allocated_bound and bound >= _PERSISTENT_HEADROOM_THRESHOLD:
            capacity = ((bound + _PERSISTENT_METADATA_HEADROOM + _PERSISTENT_ALLOCATION_ALIGNMENT - 1)
                        // _PERSISTENT_ALLOCATION_ALIGNMENT * _PERSISTENT_ALLOCATION_ALIGNMENT)
        additional = max(0, capacity - self._allocated_bound)
        if additional:
            transfer = self.transfer_bytes
            if transfer is None:
                factor = float(os.environ.get('NATLANG_CHECKPOINT_TRANSFER_HEADROOM_FACTOR', '1.0'))
                if not math.isfinite(factor) or factor < 0:
                    raise ValueError('checkpoint transfer headroom factor must be finite and nonnegative')
                transfer = math.ceil(capacity * factor)
            def allocate():
                with CheckpointWriteAdmission(self.target, additional, transfer_bytes=transfer,
                                              minimum_free_bytes=self.minimum_free_bytes):
                    os.posix_fallocate(self._fd, 0, capacity)
                    os.fsync(self._fd)
            try:
                allocate()
            except CheckpointSpaceUnavailable:
                # A prior owner may have left a fully allocated unused spare.
                # Its size/blocks cannot prove its logical allocation, but
                # admitting that whole range again can double-count it. Only
                # after checking every byte and its unchanged inode under the
                # owner lock may we release a zero spare and re-admit fresh
                # blocks. Preserve any partial serialization and known live
                # allocations; never relax the disk floor or transfer budget.
                if self._allocated_bound or not st.st_size or not self._is_empty():
                    raise
                os.ftruncate(self._fd,0)
                os.fsync(self._fd)
                self.bound=0
                allocate()
        else:
            os.posix_fallocate(self._fd, 0, capacity)
            os.fsync(self._fd)
        self._allocated_bound = max(self._allocated_bound, capacity)
        self.bound = bound
        return self

    def stream(self):
        if self._fd is None or not self.bound:
            raise CheckpointReserveError('checkpoint reservation is not allocated')
        st = self._check_pending_identity()
        if st.st_size < self.bound or self._allocated_bound < self.bound:
            raise CheckpointReserveError('checkpoint reservation changed before write')
        stream = os.fdopen(os.dup(self._fd), 'r+b')
        stream.seek(0)
        return stream

    def committed(self):
        """Called after the pending inode has been atomically renamed to target."""
        os.close(self._fd)
        self._fd = None
        self.bound = 0
        self._allocated_bound = 0
        self.receipt.unlink(missing_ok=True)
        self._sync_directory()

    def _sync_directory(self):
        fd = os.open(self.target.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def discard_uncommitted(self):
        """Release an unused spare inode while retaining the durable checkpoint."""
        if self._fd is not None:
            st=self._check_pending_identity()
            receipt=json.loads(self.receipt.read_text())
            if receipt!={'schema':'natlang.checkpoint-reservation/1','device':st.st_dev,'inode':st.st_ino}:
                raise CheckpointReserveError('cannot discard an unbound spare reservation')
            self.pending.unlink(); self.receipt.unlink(); self._sync_directory()
        self.close()

    def _is_empty(self):
        """Check every byte and the exact exclusively owned inode identity."""
        if self._fd is None:
            return False
        before = self._check_pending_identity()
        signature = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        offset = 0
        zeros = bytes(4 * 1024 * 1024)
        while offset < before.st_size:
            size = min(len(zeros), before.st_size - offset)
            block = os.pread(self._fd, size, offset)
            if len(block) != size:
                raise CheckpointReserveError('spare changed during empty reservation verification')
            if block != zeros[:size]:
                return False
            offset += size
        if signature(before) != signature(self._check_pending_identity()):
            raise CheckpointReserveError('spare changed during empty reservation verification')
        return True

    def discard_empty(self):
        """Retire only an entirely zero spare; preserve partial serialization."""
        if not self._is_empty():return False
        self.discard_uncommitted()
        return True

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self.bound = 0
        self._allocated_bound = 0
        if self._lock_fd is not None:
            # Duplicated descriptors share one flock ownership. Explicitly
            # unlocking here would also unlock a still-live sibling slot.
            os.close(self._lock_fd)
            self._lock_fd = None
        self.bound = 0
        self._allocated_bound = 0


class CheckpointDiskReserve:
    """Hold real filesystem blocks for one future atomic checkpoint write."""

    _MAGIC = b'NATLANG-CHECKPOINT-SPACE-RESERVE/1\n'

    def __init__(self, path, required_bytes):
        self.path = Path(path)
        required_bytes = int(required_bytes)
        if required_bytes < 1:
            raise ValueError('checkpoint reserve must be positive')
        self.required_bytes = max(len(self._MAGIC), required_bytes)
        self._fd = None
        self._created = False

    @property
    def active(self):
        return self._fd is not None

    def acquire(self):
        if self.active:
            return self.ensure()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            size = os.fstat(fd).st_size
            if size and os.pread(fd, len(self._MAGIC), 0) != self._MAGIC:
                raise CheckpointReserveError(
                    f'reserve path exists but is not owned by this policy: {self.path}')
            self._created = size == 0
            self._fd = fd
            return self.ensure()
        except Exception:
            if self._fd is not None:
                self._fd = None
                if self._created:
                    self.path.unlink(missing_ok=True)
                self._created = False
            os.close(fd)
            raise

    def grow_to(self, required_bytes):
        """Raise the reservation (never lower it) as the state grows, e.g. when optimizer slots appear."""
        self.required_bytes = max(self.required_bytes, int(required_bytes))
        return self.ensure() if self.active else self.required_bytes

    def ensure(self):
        if not self.active:
            raise RuntimeError('checkpoint reserve is not acquired')
        current = os.fstat(self._fd).st_size
        if current >= self.required_bytes:
            return current
        additional = self.required_bytes - current
        free = shutil.disk_usage(self.path.parent).free
        if free < additional:
            raise CheckpointReserveError(
                f'checkpoint safety preflight needs {additional} additional bytes '
                f'but only {free} are free')
        try:
            os.posix_fallocate(self._fd, 0, self.required_bytes)
        except (AttributeError, OSError) as error:
            raise CheckpointReserveError(
                f'cannot reserve {self.required_bytes} bytes for an atomic checkpoint: {error}') from error
        os.pwrite(self._fd, self._MAGIC, 0)
        os.fsync(self._fd)
        return self.required_bytes

    def release_space(self):
        """Free reserved blocks while retaining our lock for a checkpoint write.

        The lock coordinates users of this reserve file only. Another process
        can consume the released filesystem blocks before the atomic save.
        """
        if not self.active:
            return
        os.ftruncate(self._fd, len(self._MAGIC))
        os.pwrite(self._fd, self._MAGIC, 0)
        os.fsync(self._fd)

    def cleanup(self):
        """Remove only this marked reserve, then release its lock."""
        if not self.active:
            return
        fd, self._fd = self._fd, None
        try:
            try:
                path_stat = self.path.stat(follow_symlinks=False)
            except FileNotFoundError:
                return
            fd_stat = os.fstat(fd)
            owned_path = (path_stat.st_dev == fd_stat.st_dev and path_stat.st_ino == fd_stat.st_ino)
            marked = self._created or os.pread(fd, len(self._MAGIC), 0) == self._MAGIC
            if owned_path and marked:
                os.unlink(self.path)
        finally:
            self._created = False
            os.close(fd)


def warmup_checkpoint_size_upper_bound(named_parameters, heads, optimizer, *, lazy=True, margin=1.15):
    """Conservative serialized checkpoint bound, including future lazy optimizer slots (``lazy``); without them, the
    bound of the state as it is now (the next write's size), for sizing the disk reserve."""
    model_bytes = sum(parameter.numel() * parameter.element_size()
                      for _, parameter in named_parameters)
    head_bytes = sum(value.numel() * value.element_size()
                     for value in heads.state_dict().values()
                     if hasattr(value, 'numel') and hasattr(value, 'element_size'))
    optimizer_parameters = {id(parameter): parameter
                            for group in optimizer.param_groups for parameter in group['params']}
    optimizer_state_bytes = optimizer_state_size_upper_bound(optimizer, lazy=lazy)
    tensor_payload = model_bytes + head_bytes + optimizer_state_bytes
    # Pickle/zip metadata, RNG/scheduler state, and small non-tensor payloads.
    parameter_count = len(optimizer_parameters)
    overhead = max(16 * 1024 * 1024, parameter_count * 1024)
    return int(math.ceil((tensor_payload + overhead) * margin))


def native_async_snapshot_headroom(state, optimizer, *, margin=1.15, available_margin=1.25):
    """Conservatively admit a detached native-state snapshot using host/cgroup headroom.

    Count each tensor storage occurrence (aliases can overcount, never undercount),
    add the remaining lazy optimizer-state bound, then apply the same 1.15 state
    margin and 1.25 available-memory margin used by text warmup. Unknown memory
    or unsupported tensor storage fails closed to synchronous persistence.
    """
    from .trajectory_state import available_system_memory_bytes, page_cache_reclaimable_for_host_allocations
    import torch

    if (not math.isfinite(margin) or margin < 1.0
            or not math.isfinite(available_margin) or available_margin < 1.0):
        raise ValueError('async checkpoint memory margins must be finite and at least one')

    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_conj() or value.is_neg():
                raise CheckpointReserveError('async checkpoint cannot bound unsupported tensor storage')
            return int(value.untyped_storage().nbytes())
        if isinstance(value, dict):
            return sum(tensor_bytes(item) for item in value.values())
        if isinstance(value, (tuple, list, set, frozenset)):
            return sum(tensor_bytes(item) for item in value)
        return 0

    payload = tensor_bytes(state)
    current_slots = optimizer_state_size_upper_bound(optimizer, lazy=False)
    lazy_slots = optimizer_state_size_upper_bound(optimizer, lazy=True)
    parameter_count = sum(len(group['params']) for group in optimizer.param_groups)
    overhead = max(16 * 1024 * 1024, parameter_count * 1024)
    size_bound = int(math.ceil((payload + max(0, lazy_slots-current_slots) + overhead) * margin))
    available = available_system_memory_bytes(reclaimable_page_cache=page_cache_reclaimable_for_host_allocations())
    # Leave room for the measured device high-water mark to recur after this
    # boundary, plus 25% margin. CUDA allocations can share physical memory
    # with host snapshots; a boundary-only host check misses that growth.
    device_peak = device_reserved = device_growth = 0
    if torch.cuda.is_initialized():
        device_peak = int(torch.cuda.max_memory_reserved())
        try:
            options=((state.get('training') or {}).get('options') or {})
            lifetime_peak=options.get('_cuda_lifetime_peak_reserved_bytes',0)
            if type(lifetime_peak) is int and lifetime_peak>=0:
                device_peak=max(device_peak,lifetime_peak)
        except AttributeError:
            pass
        device_reserved = int(torch.cuda.memory_reserved())
        if device_peak <= 0:
            return {'eligible': False, 'reason': 'device_peak_unmeasured',
                    'available_host_bytes': available, 'snapshot_size_bound_bytes': size_bound,
                    'required_headroom_bytes': None}
        device_growth = max(0, math.ceil(device_peak * 1.25) - device_reserved)
    required = int(math.ceil(size_bound * available_margin)) + device_growth
    return {'eligible': available is not None and available >= required,
            'available_host_bytes': available, 'snapshot_size_bound_bytes': size_bound,
            'required_headroom_bytes': required, 'state_tensor_bytes': payload,
            'lazy_optimizer_bytes': max(0, lazy_slots-current_slots),
            'available_margin': available_margin, 'size_margin': margin,
            'device_peak_reserved_bytes': device_peak, 'device_reserved_bytes': device_reserved,
            'future_device_growth_allowance_bytes': device_growth,
            'future_device_peak_margin': 1.25 if device_peak else None}


def _state_tensor_bytes(value):
    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_state_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_state_tensor_bytes(item) for item in value)
    return 0


def _optimizer_family(optimizer):
    import torch

    if type(optimizer).__name__ in ('LionSR', 'MuonSR'):
        return 'lion'  # one BF16 momentum per latent
    if type(optimizer).__name__ == 'AdamMiniSR':
        return 'adamini'
    muon_type = getattr(torch.optim, 'Muon', None)
    if (muon_type is not None and isinstance(optimizer, muon_type)) or type(optimizer).__name__ == 'Muon':
        return 'muon'
    if isinstance(optimizer, torch.optim.AdamW) or type(optimizer).__name__ == 'AdamWSR':
        return 'adamw'
    raise CheckpointReserveError(
        f'checkpoint reserve does not know optimizer state layout for {type(optimizer).__name__}')


def optimizer_state_size_upper_bound(optimizer, lazy=True):
    """Exact initialized slots plus (``lazy``) layout-specific lazy state for warm-up optimizers."""
    children = ((getattr(optimizer, 'muon', None), getattr(optimizer, 'auxiliary', None),
                 getattr(optimizer, 'latent', None))
                if hasattr(optimizer, 'muon') else (optimizer,))
    seen = set()
    total = 0
    for child in children:
        if child is None:
            continue
        family = _optimizer_family(child)
        for group in child.param_groups:
            for parameter in group['params']:
                if id(parameter) in seen:
                    raise CheckpointReserveError('optimizer parameter appears in multiple state partitions')
                seen.add(id(parameter))
                state = child.state.get(parameter, {})
                total += _state_tensor_bytes(state)
                if not lazy:
                    continue
                if family == 'lion':
                    if 'momentum' not in state:
                        total += parameter.numel() * 2  # one BF16 momentum
                elif family == 'adamini':
                    if 'momentum' not in state:  # BF16 first moment, FP32 per-row second moment and step counts
                        rows = parameter.numel() // parameter.shape[-1]
                        total += parameter.numel() * 2 + rows * 4 + (rows // parameter.shape[-2]) * 4
                elif family == 'muon':
                    if 'momentum_buffer' not in state:
                        total += parameter.numel() * parameter.element_size()
                else:
                    # torch.optim.AdamW stores moments in parameter dtype and
                    # one FP32 scalar step. AMSGrad adds one maximum moment.
                    for name in ('exp_avg', 'exp_avg_sq'):
                        if name not in state:
                            total += parameter.numel() * parameter.element_size()
                    if 'step' not in state:
                        total += torch_float32_scalar_bytes()
                    if group.get('amsgrad', False) and 'max_exp_avg_sq' not in state:
                        total += parameter.numel() * parameter.element_size()
    return int(total)


def torch_float32_scalar_bytes():
    # AdamW's non-capturable and capturable step counter is a scalar FP32 tensor.
    return 4


def postcommit_recovery_metadata(step, error):
    return {
        'schema': 'natlang.text-warmup-emergency-recovery/1',
        'safe_to_resume': True,
        'failure_stage': 'after_optimizer_commit',
        'failed_attempt_step': int(step),
        'last_committed_step': int(step),
        'optimizer_step_committed': True,
        'current_rng_saved_for_resume': True,
        'schedule_state_is_current': True,
        'persistence_failure': True,
        'error_type': type(error).__name__,
        'error': str(error)[:1000],
    }


def persist_postcommit_recovery(save, reserve, *, step, error, current_rng):
    """Save committed state with current RNG, freeing its reserved atomic-write space first."""
    reserve.release_space()
    recovery = postcommit_recovery_metadata(step, error)
    save(rng_state=current_rng, emergency_recovery=recovery, write_export=False)
    return recovery


def save_with_space_retry(save, *, retry_seconds=15, sleep=time.sleep):
    """Keep the exact committed in-memory state until an atomic save can be admitted."""
    import errno
    if not math.isfinite(retry_seconds) or retry_seconds <= 0:
        raise ValueError('checkpoint retry interval must be positive and finite')
    attempt = 0
    while True:
        try:
            return save()
        except (CheckpointSpaceUnavailable, OSError) as error:
            if isinstance(error, OSError) and error.errno != errno.ENOSPC:
                raise
            attempt += 1
            print(f'Checkpoint save is waiting for disk space (attempt {attempt}): {error}', flush=True)
            sleep(retry_seconds)


def reserve_future_with_space_retry(reserve, *, stop_requested, retry_seconds=15, sleep=time.sleep):
    """Retry only a future reservation; stop at a caller-proven durable boundary."""
    import errno
    if not math.isfinite(retry_seconds) or retry_seconds <= 0:
        raise ValueError('checkpoint retry interval must be positive and finite')
    attempt=0
    while True:
        if stop_requested():return False
        try:
            reserve()
            return True
        except (CheckpointSpaceUnavailable,OSError) as error:
            if isinstance(error,OSError) and error.errno!=errno.ENOSPC:raise
            attempt+=1
            print(f'Future checkpoint reservation is waiting for disk space (attempt {attempt}): {error}',flush=True)
            remaining=retry_seconds
            while remaining>0:
                if stop_requested():return False
                interval=min(1.,remaining)
                sleep(interval);remaining-=interval
