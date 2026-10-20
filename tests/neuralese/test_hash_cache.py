"""sha256_file_hex_cached (common/hashing.py): an unchanged file is never re-read; an archived predecessor's digest
comes from its archive receipt or the archive manifest (v4 resume 2026-10-12: a 47 GB re-read from the HDD)."""
import hashlib
import json
import os
from pathlib import Path

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


def test_same_size_rewrite_with_restored_mtime_invalidates_content_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'cache.jsonl'))
    data = tmp_path / 'source.zst'
    data.write_bytes(b'original')
    original = hashing.sha256_file_hex_cached(data, trust_archive_receipt=False)
    status = data.stat()
    data.write_bytes(b'modified')
    os.utime(data, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert data.stat().st_ctime_ns != status.st_ctime_ns
    assert hashing.sha256_file_hex_cached(data, trust_archive_receipt=False) == hashlib.sha256(b'modified').hexdigest()
    assert hashing.sha256_file_hex_cached(data, trust_archive_receipt=False) != original


def test_replacing_inode_with_identical_bytes_does_not_reuse_old_cache_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'cache.jsonl'))
    data = tmp_path / 'source'
    data.write_bytes(b'same')
    hashing.sha256_file_hex_cached(data, trust_archive_receipt=False)
    original_inode = data.stat().st_ino
    replacement = tmp_path / 'replacement'
    replacement.write_bytes(b'same')
    replacement.replace(data)
    assert data.stat().st_ino != original_inode
    # The exact new object is hashed, even though the resulting digest matches.
    cache=tmp_path/'cache.jsonl'
    before_keys={json.loads(line)['key'] for line in cache.read_text().splitlines()}
    assert hashing.sha256_file_hex_cached(data, trust_archive_receipt=False) == hashlib.sha256(b'same').hexdigest()
    after_keys={json.loads(line)['key'] for line in cache.read_text().splitlines()}
    assert len(after_keys-before_keys)==1


def test_cached_content_mode_does_not_accept_archive_sidecar_as_content_hash(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'cache.jsonl'))
    data = tmp_path / 'source'
    data.write_bytes(b'actual')
    (tmp_path / 'source.archived.json').write_text(json.dumps({'dest':str(data),'bytes':6,'sha256':'f'*64}))
    assert hashing.sha256_file_hex_cached(data, trust_archive_receipt=False) == hashlib.sha256(b'actual').hexdigest()


def test_symlink_retarget_uses_new_content_object(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'cache.jsonl'))
    first=tmp_path/'first';second=tmp_path/'second';link=tmp_path/'source'
    first.write_bytes(b'one');second.write_bytes(b'two')
    link.symlink_to(first)
    assert hashing.sha256_file_hex_cached(link,trust_archive_receipt=False)==hashlib.sha256(b'one').hexdigest()
    link.unlink();link.symlink_to(second)
    assert hashing.sha256_file_hex_cached(link,trust_archive_receipt=False)==hashlib.sha256(b'two').hexdigest()


def test_content_hash_rejects_mutation_during_read(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'cache.jsonl'))
    data=tmp_path/'large';data.write_bytes(b'abcdefgh')
    original_open=open
    class RacingReader:
        def __init__(self,stream):self.stream=stream;self.changed=False
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def fileno(self):return self.stream.fileno()
        def read(self,count):
            chunk=self.stream.read(count)
            if not self.changed:
                self.changed=True
                data.write_bytes(b'ABCDEFGH')
            return chunk
    def racing_open(path,*args,**kwargs):
        stream=original_open(path,*args,**kwargs)
        return RacingReader(stream) if Path(path)==data else stream
    monkeypatch.setattr(hashing,'open',racing_open,raising=False)
    monkeypatch.setattr(hashing,'FILE_CHUNK_BYTES',2)
    with pytest.raises(RuntimeError,match='changed while hashing'):
        hashing.sha256_file_hex_cached(data,trust_archive_receipt=False)


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


def test_incremental_log_lookup_reads_only_appends_and_keeps_first_entry(tmp_path):
    cache=tmp_path/'cache.jsonl'
    cache.write_text(json.dumps({'key':'first','sha256':'a'})+'\n')
    assert hashing._cached_file_digest(cache,'first')=='a'
    original=hashing._CACHE_INDEX['entries']
    position=hashing._CACHE_INDEX['position']
    assert hashing._cached_file_digest(cache,'missing') is None
    assert hashing._CACHE_INDEX['position']==position
    with cache.open('a') as stream:
        stream.write(json.dumps({'key':'first','sha256':'b'})+'\n')
        stream.write(json.dumps({'key':'second','sha256':'c'})+'\n')
    assert hashing._cached_file_digest(cache,'second')=='c'
    assert hashing._cached_file_digest(cache,'first')=='a'
    assert hashing._CACHE_INDEX['entries'] is original
    assert hashing._CACHE_INDEX['position']==cache.stat().st_size


def test_partial_append_is_retried_and_malformed_complete_lines_do_not_block(tmp_path):
    cache=tmp_path/'cache.jsonl'
    cache.write_bytes(b'not-json\n{"key":"later","sha256":')
    assert hashing._cached_file_digest(cache,'later') is None
    assert hashing._CACHE_INDEX['position']==len(b'not-json\n')
    with cache.open('ab') as stream:stream.write(b'"d"}\n')
    assert hashing._cached_file_digest(cache,'later')=='d'


@pytest.mark.parametrize('change',['replace','truncate','same-size'])
def test_cache_rotation_truncation_and_rewrite_invalidate_old_entries(tmp_path,change):
    cache=tmp_path/'cache.jsonl'
    first=json.dumps({'key':'first','sha256':'a'})+'\n'
    second=json.dumps({'key':'other','sha256':'b'})+'\n'
    cache.write_text(first+(' ' * 50+'\n' if change=='truncate' else ''))
    assert hashing._cached_file_digest(cache,'first')=='a'
    if change=='replace':
        replacement=tmp_path/'replacement';replacement.write_text(second);replacement.replace(cache)
    else:
        cache.write_text(second)
    assert hashing._cached_file_digest(cache,'first') is None
    assert hashing._cached_file_digest(cache,'other')=='b'


def test_verified_hardlink_aliases_keep_all_bridge_paths_with_one_payload_read(tmp_path, monkeypatch):
    from natlang_neuralese.rl import native_bridge
    cache = tmp_path / 'hashes.jsonl'
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(cache))
    out = tmp_path / 'out'; out.mkdir()
    checkpoint = out / 'checkpoint.pt'; checkpoint.write_bytes(b'exact checkpoint')
    best = out / 'best-checkpoint.pt'; os.link(checkpoint, best)
    original_open = open; reads = []
    def count_open(path, *args, **kwargs):
        if Path(path) in (checkpoint, best) and args and args[0] == 'rb':
            reads.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(hashing, 'open', count_open, raising=False)
    first = native_bridge.binding(checkpoint)
    records = native_bridge.inventory(out)
    assert reads == [str(checkpoint)]
    assert {x['path'] for x in records} == {str(checkpoint), str(best)}
    assert all(x['sha256'] == first['sha256'] and x['bytes'] == first['bytes'] for x in records)
    # Same-size shared-inode mutation with restored mtime must invalidate both.
    status = checkpoint.stat(); best.write_bytes(b'other checkpoint')
    os.utime(best, ns=(status.st_atime_ns, status.st_mtime_ns))
    changed = native_bridge.inventory(out)
    assert len(reads) == 2
    assert all(x['sha256'] == hashlib.sha256(b'other checkpoint').hexdigest() for x in changed)
    assert all(x['sha256'] != first['sha256'] for x in changed)


def test_old_verified_path_entry_promotes_to_physical_alias_identity(tmp_path, monkeypatch):
    from natlang_neuralese.common.checkpoint_digest import verified_checkpoint_digest
    data = tmp_path / 'checkpoint.pt'; data.write_bytes(b'exact')
    alias = tmp_path / 'best.pt'; os.link(data, alias)
    cache = tmp_path / 'hashes.jsonl'; monkeypatch.setenv('NATLANG_HASH_CACHE', str(cache))
    s = data.stat(); digest = hashlib.sha256(b'exact').hexdigest()
    key = f'content|{data}|{s.st_dev}|{s.st_ino}|{s.st_size}|{s.st_mtime_ns}|{s.st_ctime_ns}'
    cache.write_text(json.dumps(dict(key=key, sha256=digest))+'\n')
    original_open = open
    def no_payload(path, *args, **kwargs):
        assert Path(path) not in (data, alias), 'verified alias was needlessly reread'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(hashing, 'open', no_payload, raising=False)
    assert verified_checkpoint_digest(data) == digest
    assert verified_checkpoint_digest(alias) == digest


def test_legacy_path_cache_never_authenticates_an_alias(tmp_path, monkeypatch):
    data = tmp_path / 'checkpoint.pt'; data.write_bytes(b'exact')
    alias = tmp_path / 'best.pt'; os.link(data, alias)
    cache = tmp_path / 'hashes.jsonl'; monkeypatch.setenv('NATLANG_HASH_CACHE', str(cache))
    s = data.stat(); old_key = f'{data}|{s.st_size}|{s.st_mtime_ns}'
    cache.write_text(json.dumps(dict(key=old_key, sha256='f'*64))+'\n')
    from natlang_neuralese.rl import native_bridge
    assert native_bridge.binding(data)['sha256'] == hashlib.sha256(b'exact').hexdigest()
    assert native_bridge.binding(alias)['sha256'] == hashlib.sha256(b'exact').hexdigest()
