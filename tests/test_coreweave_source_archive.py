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
