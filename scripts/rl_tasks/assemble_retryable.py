#!/usr/bin/env python3
"""Retry-safe wrapper for immutable broad-task assemblies.

Each execution writes to a fresh output directory. A ledger restart can rerun
the bounded assembly from immutable source builds without colliding with a
partially written directory from an interrupted attempt. The ready pointer is
published only after assemble.py exits successfully.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit('usage: assemble_retryable.py OUT_PARENT/PREFIX [assemble.py arguments excluding --out]')
    prefix = Path(argv.pop(0))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(f'{prefix}-{stamp}-{__import__("os").getpid()}')
    cmd = [sys.executable, '-I', str(Path(__file__).with_name('assemble.py')), *argv, '--out', str(out)]
    subprocess.run(cmd, check=True)
    ready = prefix.with_suffix(prefix.suffix + '.ready.json')
    ready.write_text(json.dumps({'schema': 'natlang.retryable-assembly-ready/1', 'output': str(out),
                                 'qualification_sha256': __import__('hashlib').sha256(
                                     (out / 'qualification.json').read_bytes()).hexdigest()}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
