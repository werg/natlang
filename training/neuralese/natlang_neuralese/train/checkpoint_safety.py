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


def warmup_checkpoint_size_upper_bound(named_parameters, heads, optimizer):
    """Conservative serialized checkpoint bound, including future lazy optimizer slots."""
    model_bytes = sum(parameter.numel() * parameter.element_size()
                      for _, parameter in named_parameters)
    head_bytes = sum(value.numel() * value.element_size()
                     for value in heads.state_dict().values()
                     if hasattr(value, 'numel') and hasattr(value, 'element_size'))
    optimizer_parameters = {id(parameter): parameter
                            for group in optimizer.param_groups for parameter in group['params']}
    optimizer_state_bytes = optimizer_state_size_upper_bound(optimizer)
    tensor_payload = model_bytes + head_bytes + optimizer_state_bytes
    # Pickle/zip metadata, RNG/scheduler state, and small non-tensor payloads.
    parameter_count = len(optimizer_parameters)
    overhead = max(16 * 1024 * 1024, parameter_count * 1024)
    return int(math.ceil((tensor_payload + overhead) * 1.15))


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

    muon_type = getattr(torch.optim, 'Muon', None)
    if (muon_type is not None and isinstance(optimizer, muon_type)) or type(optimizer).__name__ == 'Muon':
        return 'muon'
    if isinstance(optimizer, torch.optim.AdamW):
        return 'adamw'
    raise CheckpointReserveError(
        f'checkpoint reserve does not know optimizer state layout for {type(optimizer).__name__}')


def optimizer_state_size_upper_bound(optimizer):
    """Exact initialized slots plus layout-specific lazy state for warm-up optimizers."""
    children = ((getattr(optimizer, 'muon', None), getattr(optimizer, 'auxiliary', None))
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
                if family == 'muon':
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
    save(rng_state=current_rng, emergency_recovery=recovery, write_export=True)
    return recovery
