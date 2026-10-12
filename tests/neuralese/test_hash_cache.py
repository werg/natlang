"""sha256_file_hex_cached (common/hashing.py): an unchanged file is never re-read; an archived predecessor's digest
comes from its archive receipt or the archive manifest (v4 resume 2026-10-12: a 47 GB re-read from the HDD)."""
import hashlib
import json
import os

import pytest

from natlang_neuralese.common import hashing


@pytest.fixture
def no_reads(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError('file re-read')
    return lambda: monkeypatch.setattr(hashing, 'sha256_file_hex', refuse)


def test_unchanged_files_hash_once_and_changes_rehash(tmp_path, monkeypatch, no_reads):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'cache.jsonl'))
    data = tmp_path / 'state.pt'
    data.write_bytes(b'abc')
    assert hashing.sha256_file_hex_cached(data) == hashlib.sha256(b'abc').hexdigest()
    real = hashing.sha256_file_hex
    no_reads()
    assert hashing.sha256_file_hex_cached(data) == hashlib.sha256(b'abc').hexdigest()
    monkeypatch.setattr(hashing, 'sha256_file_hex', real)
    data.write_bytes(b'abcd')
    os.utime(data, ns=(1, 2))
    assert hashing.sha256_file_hex_cached(data) == hashlib.sha256(b'abcd').hexdigest()


def test_archived_state_digest_comes_from_its_receipt_or_the_archive_manifest(tmp_path, monkeypatch, no_reads):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'cache.jsonl'))
    monkeypatch.setenv('NATLANG_DATA_HDD', str(tmp_path / 'hdd'))
    archive = tmp_path / 'hdd' / 'natlang-development-data' / 'archive'
    dest = archive / 'runs' / 'lineage' / 'checkpoint.pt'
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'state')
    slot = tmp_path / 'checkpoint.pt'
    slot.symlink_to(dest)
    record = {'dest': str(dest), 'bytes': 5, 'sha256': 'f' * 64}
    (tmp_path / 'checkpoint.pt.archived.json').write_text(json.dumps(record))
    no_reads()
    assert hashing.sha256_file_hex_cached(slot) == 'f' * 64
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'other-cache.jsonl'))
    (archive / 'manifest.jsonl').write_text(json.dumps({**record, 'sha256': 'e' * 64}) + '\n')
    assert hashing.sha256_file_hex_cached(dest) == 'e' * 64  # given resolved: the manifest
