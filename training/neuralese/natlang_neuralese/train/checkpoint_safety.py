"""Disk-space guard and recovery metadata for complete warm-up checkpoints."""
from __future__ import annotations

import fcntl
import math
import os
import shutil
import stat
import time
from pathlib import Path


class CheckpointReserveError(RuntimeError):
    """There is not enough disk space to guarantee an atomic full checkpoint."""


def checkpoint_size_upper_bound(state, *, margin=1.2, minimum_overhead=16 * 1024 * 1024):
    """Bound the next serialization without materializing another checkpoint copy.

    Tensor storage is counted once by storage identity; aliased views therefore do
    not inflate the estimate, while non-tensor metadata receives a conservative
    fixed allowance. Unsupported tensor layouts fail closed.
    """
    import torch

    if not math.isfinite(margin) or margin < 1.0:
        raise ValueError('checkpoint size margin must be finite and at least one')
    seen = set()
    tensor_bytes = 0
    metadata_bytes = 0

    def visit(value):
        nonlocal tensor_bytes, metadata_bytes
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_conj() or value.is_neg():
                raise CheckpointReserveError('cannot bound unsupported tensor storage for checkpoint admission')
            storage = value.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes(), value.device.type, value.device.index)
            if key not in seen:
                seen.add(key)
                tensor_bytes += int(storage.nbytes())
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


class CheckpointWriteAdmission:
    """Serialize checkpoint writers per filesystem and hold concurrent-transfer space.

    The reserved blocks remain allocated while the atomic writer is active, so
    a same-filesystem checkpoint copy cannot consume the space needed for the
    checkpoint being committed. Writers recheck available bytes only after
    taking the shared per-device lock.
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
            minimum_free_bytes = int(os.environ.get('NATLANG_CHECKPOINT_MIN_FREE_BYTES', str(512 * 1024**2)))
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
        lock_path = Path('/tmp') / f'natlang-checkpoint-admission-{os.getuid()}-{device}.lock'
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
                        raise CheckpointReserveError(
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
                    raise CheckpointReserveError(
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
    from .trajectory_state import available_system_memory_bytes
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
    available = available_system_memory_bytes()
    required = int(math.ceil(size_bound * available_margin))
    return {'eligible': available is not None and available >= required,
            'available_host_bytes': available, 'snapshot_size_bound_bytes': size_bound,
            'required_headroom_bytes': required, 'state_tensor_bytes': payload,
            'lazy_optimizer_bytes': max(0, lazy_slots-current_slots),
            'available_margin': available_margin, 'size_margin': margin}


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
    if isinstance(optimizer, torch.optim.AdamW):
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
