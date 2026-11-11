"""Advisory sequential access for the exact checkpoint's surviving mmap.

Torch closes its file descriptor after constructing the storage map. Advice on
a newly opened descriptor cannot configure that map; Linux madvise can. This
never faults pages eagerly and never changes checkpoint contents.
"""
from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path

SETTING = 'NATLANG_CHECKPOINT_MMAP_READAHEAD'


def enabled():
    value = os.environ.get(SETTING, 'sequential')
    if value not in ('off', 'sequential'):
        raise ValueError(f'{SETTING} must be off or sequential')
    return value == 'sequential'


def mapped_regions(path):
    """Return only readable nonexecuting mappings of this device/inode."""
    identity = Path(path).stat()
    device = (os.major(identity.st_dev), os.minor(identity.st_dev))
    regions = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) < 5:
            continue
        addresses, permissions, _, mapped_device, inode = fields[:5]
        if ('r' not in permissions or 'x' in permissions or int(inode) != identity.st_ino
                or tuple(int(part, 16) for part in mapped_device.split(':')) != device):
            continue
        start, stop = (int(part, 16) for part in addresses.split('-'))
        if stop > start:
            regions.append((start, stop-start))
    return regions


def advise_checkpoint(path):
    """Best-effort kernel hint; unsupported/advisory failures are no-ops.

    Return actual successfully advised bytes, not bytes read or prefetched.
    """
    if not hasattr(mmap, 'MADV_SEQUENTIAL'):
        return 0
    try:
        regions = mapped_regions(path)
        advise = ctypes.CDLL(None, use_errno=True).madvise
        advise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        advise.restype = ctypes.c_int
        return sum(length for address, length in regions
                   if advise(address, length, mmap.MADV_SEQUENTIAL) == 0)
    except (OSError, AttributeError, ValueError):
        return 0


PREFETCH_SETTING = 'NATLANG_CHECKPOINT_PREFETCH'
PREFETCH_MIN_BYTES = 1 << 30
PREFETCH_CHUNK_BYTES = 16 << 20


def prefetch_enabled():
    value = os.environ.get(PREFETCH_SETTING, 'sequential')
    if value not in ('off', 'sequential'):
        raise ValueError(f'{PREFETCH_SETTING} must be off or sequential')
    return value == 'sequential'


def prefetch_checkpoint(path):
    """Warm the page cache of a checkpoint the learner is about to restore, with one sequential read.

    The restore maps the file and copies each tensor to the device in turn. On a cold cache that is a stream of
    small page faults: 0.1 GB/s measured on the GB10 (18 GB, three minutes), against 1.2 GB/s for a sequential read
    and 19 GB/s warm. A daemon thread reads the file front to back in large chunks while the base model loads, so
    the restore finds its pages resident. It never writes, never changes the file, and a failure only loses the
    speedup. Returns the thread, or None when disabled or the file is small.
    """
    import threading
    try:
        if not prefetch_enabled():
            return None
        path = Path(path).resolve(strict=True)
        if path.stat().st_size < PREFETCH_MIN_BYTES:
            return None
    except (OSError, ValueError):
        return None

    def read():
        try:
            buffer = bytearray(PREFETCH_CHUNK_BYTES)
            with open(path, 'rb', buffering=0) as stream:
                while stream.readinto(buffer):
                    pass
        except OSError:
            pass

    thread = threading.Thread(target=read, name='checkpoint-prefetch', daemon=True)
    thread.start()
    return thread
