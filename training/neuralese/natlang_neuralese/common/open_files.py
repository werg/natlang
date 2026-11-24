"""The open-file limit of processes that hold one socket per stream.

A user unit starts with the manager's soft RLIMIT_NOFILE (1024 on the DGX, hard 500000). The SGLang front holds two
sockets per stream (its client's and its SGLang request's), so 128 collection streams use ~260 descriptors, and any
pile-up (retried requests whose first attempt still streams, decision fan-outs of one connection per option) reaches
1024: `URLError: [Errno 24] Too many open files` preceded the 03:04 OOM kill of aei1-collect-v0001 (2026-11-24).
Node raises its own soft limit to the hard one at start; Python does not, so the Python servers do it here."""
from __future__ import annotations

import os

CEILING = 1 << 20


def raise_open_files_limit(ceiling: int = CEILING):
    """Raise the soft RLIMIT_NOFILE to the hard limit (at most ``ceiling``); returns (before, after) soft limits, or
    None where there is no such limit. ``NATLANG_RAISE_NOFILE=0`` opts out."""
    if os.environ.get('NATLANG_RAISE_NOFILE', '1') == '0':
        return None
    try:
        import resource
    except ImportError:      # not POSIX
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = ceiling if hard == resource.RLIM_INFINITY else min(hard, ceiling)
    if soft != resource.RLIM_INFINITY and soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            return soft, soft
        return soft, target
    return soft, soft
