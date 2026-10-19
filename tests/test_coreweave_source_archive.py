import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest


spec = importlib.util.spec_from_file_location(
    'source_archive', Path(__file__).resolve().parents[1] / 'scripts/coreweave/source_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
pytestmark = pytest.mark.skipif(shutil.which('zstd') is None, reason='zstd CLI is required')


def test_member_hashes_preserve_binary_crlf_unicode_and_regular_hardlinks(tmp_path):
    source, spool = tmp_path / 'source', tmp_path / 'spool'
    source.mkdir()
    (source / 'text.jsonl').write_bytes('{"value":"café"}\r\n'.encode())
    (source / 'binary').write_bytes(bytes(range(256)))
    os.link(source / 'binary', source / 'binary-again')
    receipt = archive.pack(source, spool, 'closed-native', reserve_bytes=0)
    proof = archive.verify(spool, 'closed-native')
    assert proof['files'] == 3
    assert proof['source_bytes'] == sum(path.stat().st_size for path in source.iterdir())
    assert receipt['manifest']['files'][1]['sha256'] == receipt['manifest']['files'][0]['sha256']
    assert proof['training_admission'] is False
    assert all(path.exists() for path in source.iterdir())
    with pytest.raises(FileExistsError):
        archive.pack(source, spool, 'closed-native', reserve_bytes=0)


def test_verify_rejects_a_changed_original_byte_binding(tmp_path):
    source, spool = tmp_path / 'source', tmp_path / 'spool'
    source.mkdir()
    (source / 'record').write_bytes(b'original\r\n')
    archive.pack(source, spool, 'native', reserve_bytes=0)
    path = spool / 'native.receipt.json'
    receipt = json.loads(path.read_text())
    receipt['manifest']['files'][0]['sha256'] = '0' * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='original bytes|embedded manifest'):
        archive.verify(spool, 'native')


def test_pack_refuses_symlinks_and_spools_inside_source(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'record').write_bytes(b'keep')
    (source / 'alias').symlink_to(source / 'record')
    with pytest.raises(ValueError, match='symlinks'):
        archive.pack(source, tmp_path / 'spool', 'native', reserve_bytes=0)
    with pytest.raises(ValueError, match='outside'):
        archive.pack(source, source / 'spool', 'native', reserve_bytes=0)


def test_atomic_restore_preserves_exact_bytes_and_rejects_escaped_target(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'nested').mkdir()
    (source/'nested/value').write_bytes(b'unchanged\r\n\x00')
    spool=tmp_path/'spool';archive.pack(source,spool,'native',reserve_bytes=0)
    target=tmp_path/'target'
    assert archive.restore(spool,'native',target)['restored_files']==1
    assert (target/'nested/value').read_bytes()==b'unchanged\r\n\x00'
    (target/'nested/value').unlink();(target/'nested').rmdir()
    outside=tmp_path/'outside';outside.mkdir();(target/'nested').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='escaped through a link'):
        archive.restore(spool,'native',target)
    assert not (outside/'value').exists()


def test_explicit_file_symlink_mode_seals_regular_bytes_and_target_provenance(tmp_path):
    source, spool = tmp_path / 'source', tmp_path / 'spool'
    source.mkdir()
    target = tmp_path / 'repaired-shard'
    target.write_bytes(b'immutable repaired records\r\n\x00')
    (source / 'shard.records.jsonl').symlink_to(target)
    receipt = archive.pack(source, spool, 'native', reserve_bytes=0, dereference_file_symlinks=True)
    binding = receipt['manifest']['files'][0]
    assert binding['resolved_source'] == str(target)
    assert binding['source_symlink'] == str(target)
    assert archive.verify(spool, 'native')['source_bytes'] == target.stat().st_size
    (source / 'directory').symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinks'):
        archive.pack(source, spool, 'directory-link', reserve_bytes=0, dereference_file_symlinks=True)


def test_retargeted_file_symlink_aborts_without_sealing_archive(tmp_path, monkeypatch):
    source, spool = tmp_path / 'source', tmp_path / 'spool'
    source.mkdir()
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.write_bytes(b'same bytes'); second.write_bytes(b'same bytes')
    link = source / 'shard'
    link.symlink_to(first)
    original = archive.tarfile.TarFile.addfile
    def swap_after_read(self, info, fileobj=None):
        result = original(self, info, fileobj)
        if info.name.endswith('/shard'):
            link.unlink(); link.symlink_to(second)
        return result
    monkeypatch.setattr(archive.tarfile.TarFile, 'addfile', swap_after_read)
    with pytest.raises(ValueError, match='changed during archive'):
        archive.pack(source, spool, 'native', reserve_bytes=0, dereference_file_symlinks=True)
    assert not (spool / 'native.tar.zst').exists()
    assert not (spool / 'native.tar.pending').exists()


def recovery_authority(packed):
    return {'private_receipt':'private-full-get.json',
        'archive_identity':{'sha256':packed['archive_sha256'],'bytes':packed['archive_bytes']},
        'keys':['key-0'],'source_context':{'cases':'exact source'},'training_admission':False}


def test_repeated_actual_restore_preserves_sealed_receipt_bytes_and_inode(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'row').write_bytes(b'original\r\n')
    spool=tmp_path/'spool';packed=archive.pack(source,spool,'native',reserve_bytes=0)
    target=tmp_path/'restored';receipt=spool/'recovery-receipt.json';authority=recovery_authority(packed)
    first=archive.restore_with_stable_receipt(spool,'native',target,receipt_path=receipt,authority=authority)
    raw=receipt.read_bytes();inode=receipt.stat().st_ino
    second=archive.restore_with_stable_receipt(spool,'native',target,receipt_path=receipt,authority=authority)
    assert first==second and receipt.read_bytes()==raw and receipt.stat().st_ino==inode
    assert (target/'row').read_bytes()==b'original\r\n'
    observations=[json.loads(line)for line in receipt.with_name(receipt.name+'.observations.jsonl').read_text().splitlines()]
    assert len(observations)==2 and [o['reused_receipt']for o in observations]==[False,True]
    assert observations[0]['receipt']==observations[1]['receipt']


@pytest.mark.parametrize('change',['private_receipt','keys','source_context','archive_identity'])
def test_changed_recovery_authority_rejects_before_restore(tmp_path,monkeypatch,change):
    source=tmp_path/'source';source.mkdir();(source/'row').write_text('original')
    spool=tmp_path/'spool';packed=archive.pack(source,spool,'native',reserve_bytes=0)
    target=tmp_path/'restored';receipt=spool/'recovery.json';authority=recovery_authority(packed)
    archive.restore_with_stable_receipt(spool,'native',target,receipt_path=receipt,authority=authority)
    raw=receipt.read_bytes();bad={**authority,change:'changed'}
    monkeypatch.setattr(archive,'restore',lambda *a,**kw:pytest.fail('changed authority reached restore'))
    with pytest.raises(ValueError,match='authority changed'):
        archive.restore_with_stable_receipt(spool,'native',target,receipt_path=receipt,authority=bad)
    assert receipt.read_bytes()==raw


def test_recovery_rejects_wrong_compressed_size_and_changed_archive(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'row').write_text('original')
    spool=tmp_path/'spool';packed=archive.pack(source,spool,'native',reserve_bytes=0)
    target=tmp_path/'restored';receipt=spool/'recovery.json';authority=recovery_authority(packed)
    wrong={**authority,'archive_identity':{**authority['archive_identity'],'bytes':0}}
    with pytest.raises(ValueError,match='sealed bytes'):
        archive.restore_with_stable_receipt(spool,'native',target,receipt_path=receipt,authority=wrong)
    assert not receipt.exists() and not target.exists()
    archive.restore_with_stable_receipt(spool,'native',target,receipt_path=receipt,authority=authority)
    raw=receipt.read_bytes();Path(packed['archive']).write_bytes(b'changed')
    with pytest.raises(ValueError):
        archive.restore_with_stable_receipt(spool,'native',target,receipt_path=receipt,authority=authority)
    assert receipt.read_bytes()==raw
