from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from natlang_neuralese.data import provider_teacher_source as source


def test_bound_source_cache_reuses_only_verified_five_field_path_identity(tmp_path, monkeypatch):
    first = tmp_path / 'first.jsonl'
    second = tmp_path / 'second.jsonl'
    first.write_bytes(b'{"id":"one"}\n')
    second.write_bytes(b'{"id":"one"}\n')
    link = tmp_path / 'current.jsonl'
    link.symlink_to(first)
    expected = first.read_bytes()
    digest = hashlib.sha256(expected).hexdigest()

    original_read = Path.read_bytes
    reads = []

    def count_reads(path):
        if path.resolve() in {first.resolve(), second.resolve()}:
            reads.append(path.resolve())
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', count_reads)
    assert source._read_bound(str(link), digest) == expected
    assert source._read_bound(str(link), digest) == expected
    assert reads == [first.resolve()]

    # A same-size edit with restored mtime cannot pass the cached identity.
    before = first.stat()
    first.write_bytes(b'{"id":"two"}\n')
    os.utime(first, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='digest mismatch'):
        source._read_bound(str(link), digest)
    assert reads[-1] == first.resolve()

    # A symlink retarget to identical bytes is still a new source path and is read.
    first.write_bytes(b'{"id":"one"}\n')
    link.unlink()
    link.symlink_to(first)
    assert source._read_bound(str(link), digest) == expected
    link.unlink()
    link.symlink_to(second)
    assert source._read_bound(str(link), digest) == expected
    assert reads[-1] == second.resolve()
