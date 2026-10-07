"""Disk-space guard and recovery metadata for complete warm-up checkpoints."""
from __future__ import annotations

import fcntl
import math
import os
import shutil
from pathlib import Path


class CheckpointReserveError(RuntimeError):
    """There is not enough disk space to guarantee an atomic full checkpoint."""


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
        """Free reserved blocks while retaining the lock for a checkpoint write."""
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


def warmup_checkpoint_size_upper_bound(named_parameters, heads, optimizer):
    """Conservative serialized checkpoint bound, including future lazy optimizer slots."""
    model_bytes = sum(parameter.numel() * parameter.element_size()
                      for _, parameter in named_parameters)
    head_bytes = sum(value.numel() * value.element_size()
                     for value in heads.state_dict().values()
                     if hasattr(value, 'numel') and hasattr(value, 'element_size'))
    optimizer_parameters = {}
    for group in optimizer.param_groups:
        for parameter in group['params']:
            optimizer_parameters[id(parameter)] = parameter
    # Three FP32-equivalent slots per trainable parameter conservatively cover
    # Muon momentum plus AdamW's two moments; use actual initialized state when
    # it is larger. The named and head tensors are separately present in state.
    slot_bound = sum(3 * parameter.numel() * max(4, parameter.element_size())
                     for parameter in optimizer_parameters.values())
    live_state = 0
    children = ((getattr(optimizer, 'muon', None), getattr(optimizer, 'auxiliary', None))
                if hasattr(optimizer, 'muon') else (optimizer,))
    for child in children:
        if child is None:
            continue
        for state in child.state.values():
            live_state += sum(value.numel() * value.element_size()
                              for value in state.values()
                              if hasattr(value, 'numel') and hasattr(value, 'element_size'))
    tensor_payload = model_bytes + head_bytes + max(slot_bound, live_state)
    # Pickle/zip metadata, RNG/scheduler state, and small non-tensor payloads.
    parameter_count = len(optimizer_parameters)
    overhead = max(16 * 1024 * 1024, parameter_count * 1024)
    return int(math.ceil((tensor_payload + overhead) * 1.15))


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
