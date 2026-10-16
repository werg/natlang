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
    def __init__(self, store, path, tensor, size):
        self.store, self.path = store, path
        self.shape, self.dtype, self.device = tuple(tensor.shape), tensor.dtype, tensor.device
        self.numel, self.size = tensor.numel(), size
        self.cleanup = weakref.finalize(self, _remove_tensor, path, store, size)

    def unpack(self):
        # A new device tensor owns its values independently of the scratch file.
        # Keep the holder/file for retain_graph and repeated higher derivatives.
        mapped = torch.from_file(str(self.path), shared=False, size=self.numel,
                                 dtype=self.dtype).reshape(self.shape)
        restored = mapped.to(device=self.device, copy=True, non_blocking=False)
        del mapped
        with self.path.open('rb') as stream:
            _release_cache(stream.fileno())
        with self.store.lock:
            self.store.counters['read_bytes'] += self.size
            self.store.counters['unpacks'] += 1
        return restored


class CheckpointFileStore:
    def __init__(self, directory):
        parent = Path(directory).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = str(parent)
        self.lock = threading.Lock()
        self.counters = dict(written_bytes=0, read_bytes=0, live_bytes=0,
                             peak_live_bytes=0, live_files=0, packs=0, unpacks=0)
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

    def pack(self, tensor):
        if not tensor.numel():
            return tensor.detach()  # The checkpoint's empty sentinel owns no bytes.
        if tensor.layout != torch.strided:
            raise ValueError('file checkpoint inputs require dense strided tensors')
        size = tensor.numel() * tensor.element_size()
        fd, name = tempfile.mkstemp(prefix='tensor-', suffix='.bin', dir=self.path)
        path = Path(name)
        try:
            os.ftruncate(fd, size)
            mapped = torch.from_file(name, shared=True, size=tensor.numel(),
                                     dtype=tensor.dtype).reshape(tensor.shape)
            # Copy logical values directly into file-backed host memory: no
            # accumulated anonymous CPU tensor or serialization storage alias.
            mapped.copy_(tensor.detach(), non_blocking=False)
            del mapped
            os.fdatasync(fd)
            _release_cache(fd)
            with self.lock:
                self.counters['written_bytes'] += size
                self.counters['live_bytes'] += size
                self.counters['peak_live_bytes'] = max(self.counters['peak_live_bytes'],
                                                      self.counters['live_bytes'])
                self.counters['live_files'] += 1
                self.counters['packs'] += 1
            return SavedFileTensor(self, path, tensor, size)
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
            return {'storage': 'file', 'directory': self.directory, **self.counters}
