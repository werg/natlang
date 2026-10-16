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
