"""Private file mappings for exact saved tensors, with graph-owned lifetimes."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import weakref

import torch

SCHEMA = 'natlang.checkpoint-input-store/1'
PREFIX = 'natlang-checkpoint-input-'


def _remove_store(path, lock):
    try:
        shutil.rmtree(path)
    finally:
        lock.close()


def _remove_tensor(path, store, size):
    try:
        path.unlink(missing_ok=True)
    finally:
        with store.lock:
            store.counters['live_bytes'] -= size
            store.counters['live_files'] -= 1


def _release_cache(fd):
    # Linux training hosts: a flushed, unmapped file need not retain page cache.
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


class SavedFileTensor:
    def __init__(self, store, path, tensor, size, span):
        self.store, self.path = store, path
        self.shape, self.dtype, self.device = tuple(tensor.shape), tensor.dtype, tensor.device
        self.stride = tuple(tensor.stride())
        self.numel, self.size = span, size
        self.cleanup = weakref.finalize(self, _remove_tensor, path, store, size)

    def unpack(self):
        # A new device tensor owns its values independently of the scratch file.
        # Keep the holder/file for retain_graph and repeated higher derivatives.
        mapped = torch.from_file(str(self.path), shared=False, size=self.numel,
                                 dtype=self.dtype)
        restored = self.store.restore(mapped, self.device)
        del mapped
        with self.path.open('rb') as stream:
            _release_cache(stream.fileno())
        with self.store.lock:
            self.store.counters['read_bytes'] += self.size
            self.store.counters['unpacks'] += 1
        return restored.as_strided(self.shape, self.stride)


class CheckpointFileStore:
    def __init__(self, directory, *, pin_memory=True, transfer_chunk_bytes=64 << 20):
        if type(pin_memory) is not bool:
            raise ValueError('file checkpoint pinning must be an explicit boolean')
        if (type(transfer_chunk_bytes) is not int or transfer_chunk_bytes < 16
                or transfer_chunk_bytes % 16):
            raise ValueError('file checkpoint transfer chunk must be positive and aligned to 16 bytes')
        parent = Path(directory).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = str(parent)
        self.lock = threading.Lock()
        self.transfer_lock = threading.Lock()
        self.pin_memory = pin_memory
        self.transfer_chunk_bytes = transfer_chunk_bytes
        self.transfer_buffer = None
        self.counters = dict(written_bytes=0, read_bytes=0, live_bytes=0,
                             peak_live_bytes=0, live_files=0, packs=0, unpacks=0,
                             pinned_transfer_bytes=0, pinned_pack_chunks=0,
                             pinned_unpack_chunks=0)
        # Creation and abandoned-store cleanup use one parent lock. The store's
        # separate lock remains held for the lifetime of every saved holder.
        with (parent / '.natlang-checkpoint-input.lock').open('a+b') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            for stale in parent.glob(PREFIX + '*'):
                if not stale.is_dir() or stale.is_symlink():
                    continue
                try:
                    marker = json.loads((stale / 'store.json').read_text())
                    if marker.get('schema') != SCHEMA:
                        continue
                    with (stale / '.lock').open('r+b') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        shutil.rmtree(stale)
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
            self.path = Path(tempfile.mkdtemp(prefix=PREFIX, dir=parent))
            lease = (self.path / '.lock').open('a+b')
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (self.path / 'store.json').write_text(json.dumps({'schema': SCHEMA}))
            self.cleanup = weakref.finalize(self, _remove_store, self.path, lease)

    def _staging(self, dtype):
        # One byte buffer is reused across dtypes. A file holder must never
        # retain it: the next checkpoint tensor is allowed to overwrite it.
        if self.transfer_buffer is None:
            self.transfer_buffer = torch.empty(self.transfer_chunk_bytes,
                                               dtype=torch.uint8, pin_memory=True)
            with self.lock:
                self.counters['pinned_transfer_bytes'] = self.transfer_buffer.numel()
        return self.transfer_buffer.view(dtype)

    def copy_to_file(self, mapped, source):
        if not self.pin_memory or source.device.type != 'cuda':
            mapped.copy_(source, non_blocking=False)
            return
        with self.transfer_lock:
            staging = self._staging(source.dtype)
            chunks = 0
            for first in range(0, source.numel(), staging.numel()):
                last = min(first + staging.numel(), source.numel())
                block = staging[:last-first]
                # Synchronous completion protects the shared staging buffer
                # and makes the following CPU/file copy observe actual values.
                block.copy_(source[first:last], non_blocking=False)
                mapped[first:last].copy_(block)
                chunks += 1
            with self.lock:
                self.counters['pinned_pack_chunks'] += chunks

    def restore(self, mapped, device):
        if not self.pin_memory or device.type != 'cuda':
            return mapped.to(device=device, copy=True, non_blocking=False)
        restored = torch.empty(mapped.numel(), dtype=mapped.dtype, device=device)
        with self.transfer_lock:
            staging = self._staging(mapped.dtype)
            chunks = 0
            for first in range(0, mapped.numel(), staging.numel()):
                last = min(first + staging.numel(), mapped.numel())
                block = staging[:last-first]
                block.copy_(mapped[first:last])
                restored[first:last].copy_(block, non_blocking=False)
                chunks += 1
            with self.lock:
                self.counters['pinned_unpack_chunks'] += chunks
        return restored

    def pack(self, tensor):
        if not tensor.numel():
            return tensor.detach()  # The checkpoint's empty sentinel owns no bytes.
        if tensor.layout != torch.strided:
            raise ValueError('file checkpoint inputs require dense strided tensors')
        # Preserve physical strides, including overlap/expansion, because native
        # attention dispatch binds actual layouts. Rebase the offset as an
        # ordinary device copy does; never copy the unused storage prefix.
        span = 1 + sum((length - 1) * stride for length, stride in zip(tensor.shape, tensor.stride()))
        size = span * tensor.element_size()
        fd, name = tempfile.mkstemp(prefix='tensor-', suffix='.bin', dir=self.path)
        path = Path(name)
        try:
            os.ftruncate(fd, size)
            mapped = torch.from_file(name, shared=True, size=span, dtype=tensor.dtype)
            # Copy logical values directly into file-backed host memory: no
            # accumulated anonymous CPU tensor or serialization storage alias.
            source = tensor.detach().as_strided((span,), (1,), storage_offset=tensor.storage_offset())
            self.copy_to_file(mapped, source)
            del mapped
            del source
            os.fdatasync(fd)
            _release_cache(fd)
            with self.lock:
                self.counters['written_bytes'] += size
                self.counters['live_bytes'] += size
                self.counters['peak_live_bytes'] = max(self.counters['peak_live_bytes'],
                                                      self.counters['live_bytes'])
                self.counters['live_files'] += 1
                self.counters['packs'] += 1
            return SavedFileTensor(self, path, tensor, size, span)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)

    @staticmethod
    def unpack(saved):
        return saved.unpack() if isinstance(saved, SavedFileTensor) else saved

    def snapshot(self):
        with self.lock:
            return {'storage': 'file', 'directory': self.directory,
                    'pin_memory': self.pin_memory,
                    'transfer_chunk_bytes': self.transfer_chunk_bytes, **self.counters}
