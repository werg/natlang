"""Batch exact digest-reference scans for guarded artifact retention."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile


def reference_paths(digests, roots, *, rg='rg', chunk_bytes=1 << 20):
    """Scan JSON once for all candidate identities; callers serialize retention.

    No cached negative result survives this call. A missing/changing matched
    file or search failure aborts the scan, so callers must retain artifacts.
    """
    values = sorted(set(digests))
    if any(not isinstance(x, str) or re.fullmatch('[0-9a-f]{64}', x) is None for x in values):
        raise ValueError('reference scan needs exact SHA-256 identities')
    if chunk_bytes < 64:
        raise ValueError('reference scan window is too small')
    result = {value: set() for value in values}
    if not values:
        return result
    identities = {}
    def identity(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    with tempfile.NamedTemporaryFile(mode='w') as patterns:
        patterns.write('\n'.join(values) + '\n')
        patterns.flush()
        # Fixed-string matching stays in ripgrep's multi-pattern engine. Avoid
        # rescanning large matching JSON records with a Python alternation.
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen([rg, '--only-matching', '--with-filename', '--null',
                '--no-line-number', '--no-column', '--no-heading', '--color=never',
                '-F', '-f', patterns.name, '--glob', '*.json', *map(str, roots)],
                stdout=subprocess.PIPE, stderr=errors)
            try:
                pending = b''
                while block := process.stdout.read(chunk_bytes):
                    pending += block
                    offset = 0
                    while True:
                        end = pending.find(b'\0', offset)
                        if end < 0 or len(pending) < end + 66:
                            break
                        name = os.fsdecode(pending[offset:end])
                        digest = pending[end + 1:end + 65].decode('ascii')
                        if pending[end + 65:end + 66] != b'\n' or digest not in result:
                            raise RuntimeError('unexpected dependency reference search record')
                        result[digest].add(name)
                        if name not in identities:
                            identities[name] = identity(Path(name).stat())
                        offset = end + 66
                    pending = pending[offset:]
                    if len(pending) > 1 << 20:
                        raise RuntimeError('unbounded dependency reference search record')
                status = process.wait()
                if pending or status not in (0, 1):
                    errors.seek(0)
                    raise RuntimeError('dependency reference search failed: ' + os.fsdecode(errors.read(500)))
            except BaseException:
                process.terminate()
                process.wait()
                raise
            finally:
                process.stdout.close()
    for name, before in identities.items():
        if identity(Path(name).stat()) != before:
            raise RuntimeError('dependency reference path changed while reading')
    return result
