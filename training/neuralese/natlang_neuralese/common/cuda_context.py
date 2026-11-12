"""CUDA context creation on the GB10 under host-memory fragmentation (2026-11-12).

A new CUDA context's graphics buffers come from physically contiguous host pages. When the kernel's Normal zone has
almost no free order-8 (1 MB) blocks, context creation fails with `CUDA_ERROR_OUT_OF_MEMORY` from
`cuDevicePrimaryCtxRetain` (kernel log: `NVRM ... NV_ERR_NO_MEMORY ... kgrctxAllocMainCtxBuffer`) while tens of GB are
free. `compact()` makes the kernel compact memory without root: it faults a THP-advised anonymous region (direct
compaction) and releases it. The persistent `natlang-gb10-compactor` user service does the same continuously
(scripts/gb10_compactor.py); the root fix is `vm.compaction_proactiveness` / `/proc/sys/vm/compact_memory`.
"""
from __future__ import annotations

import json
import mmap
import os
import sys
import time

SIGNATURES = ('cuDevicePrimaryCtxRetain', 'NV_ERR_NO_MEMORY', 'CUDA_ERROR_OUT_OF_MEMORY')
RETRIES = 3
WAIT_SECONDS = 20.0
ATTEMPT_ENV = 'NATLANG_CUDA_CONTEXT_ATTEMPT'


def high_order_free_mb() -> int | None:
    """Free memory in blocks of order >= 8 (1 MB with 4 KB pages) in the Normal zone, in MB; None off Linux."""
    try:
        for line in open('/proc/buddyinfo'):
            fields = line.split()
            if fields[3] == 'Normal':
                page_kb = os.sysconf('SC_PAGE_SIZE') // 1024
                return sum(int(count) * (page_kb << order) for order, count in enumerate(fields[4:]) if order >= 8) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def compact(gb: float = 2.0) -> dict:
    """Fault `gb` of THP-advised anonymous memory and release it, so the kernel compacts free memory into large blocks."""
    before, start = high_order_free_mb(), time.time()
    try:
        size = int(gb * 2**30)
        region = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        try:
            region.madvise(mmap.MADV_HUGEPAGE)
            for offset in range(0, size, 2 << 20):
                region[offset] = 1
        finally:
            region.close()
    except (OSError, ValueError, AttributeError) as error:
        return {'error': str(error)}
    return {'high_order_free_mb_before': before, 'after': high_order_free_mb(), 'seconds': round(time.time() - start, 1)}


def context_failure(text: str) -> bool:
    """Whether an error text is a CUDA context-creation failure (not an ordinary allocation that ran out)."""
    return any(signature in text for signature in SIGNATURES)


def ensure_context(device: str = 'cuda') -> None:
    """Create this process's CUDA context before anything else uses the GPU. A context-creation OOM re-executes the
    process (a failed context cannot be retried in place) after `compact()` and `WAIT_SECONDS`, up to `RETRIES` times."""
    if not str(device).startswith('cuda'):
        return
    import torch
    if not torch.cuda.is_available():
        return
    try:
        torch.zeros(1, device=device)
        return
    except Exception as error:   # torch.AcceleratorError / RuntimeError
        text = str(error)
        if 'out of memory' not in text.lower() and not context_failure(text):
            raise
        attempt = int(os.environ.get(ATTEMPT_ENV, '0')) + 1
        if attempt > RETRIES:
            raise
        report = compact()
        print(json.dumps({'event': 'cuda_context_retry', 'attempt': attempt, 'of': RETRIES, 'error': text[:200],
                          'compact': report, 'wait_seconds': WAIT_SECONDS}), flush=True)
        time.sleep(WAIT_SECONDS)
        os.environ[ATTEMPT_ENV] = str(attempt)
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, *sys.orig_argv[1:]])
