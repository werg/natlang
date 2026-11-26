"""Bound a many-threaded process's glibc heap (one mechanism for every thread-per-slot or thread-per-request process).

glibc gives each thread that allocates its own malloc arena (up to 8 x cores). Each arena grows to the peak of what
passed through it and keeps that memory after it is freed, so a process whose threads handle large transient values
(request bodies, rollouts parsed for scoring, episode evidence) grows with uptime while its live allocations stay flat:
the SGLang front on Pop (RSS 0.82 -> 1.83 GB an hour, ~220 MB live), and the collectors (2026-11-25: the Pop collection
client's main process, 64 slot threads, reached 7.0 GB at 1 h 55 min and 8.1 GB at 2 h 9 min and was OOM-killed at each
cap; the DGX collection's main process held ~470 64 MB arena heaps, 2 GB, beside its 0.9 GB main heap). Few arenas
reuse one another's free memory, and a periodic malloc_trim hands free pages back.

Consumers: serve_native_checkpoint (the front), expert_iteration (collect, the driver), grpo (learner + collector
slots), held_eval (measure slots), filled_reasoning (generate workers), teacher_gen (campaign slots).
A launcher's MALLOC_ARENA_MAX takes precedence; NATLANG_BOUND_HOST_HEAP=0 opts out.
"""
from __future__ import annotations

import os
import threading

_STARTED: list = []


def bound_host_heap(arenas=2, trim_seconds=60.0):
    """Cap the malloc arenas threads attach to from now on at ``arenas`` (unless MALLOC_ARENA_MAX is set) and start a
    daemon thread that calls malloc_trim every ``trim_seconds``. Returns the trim thread (None without glibc, with the
    opt-out, or when this process already started one)."""
    import ctypes
    if os.environ.get('NATLANG_BOUND_HOST_HEAP', '1') == '0':
        return None
    try:
        libc = ctypes.CDLL('libc.so.6')
        mallopt, malloc_trim = libc.mallopt, libc.malloc_trim
    except (OSError, AttributeError):
        return None
    if 'MALLOC_ARENA_MAX' not in os.environ:
        mallopt(-8, int(arenas))   # M_ARENA_MAX
    if _STARTED and _STARTED[0].is_alive():
        return None

    tick = threading.Event()    # never set: a wait that a test's patched time.sleep does not see

    def trim():
        while True:
            tick.wait(trim_seconds)
            malloc_trim(0)
    thread = threading.Thread(target=trim, name='host-heap-trim', daemon=True)
    thread.start()
    _STARTED[:] = [thread]
    return thread
