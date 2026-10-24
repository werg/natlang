import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'coreweave'))
import recovery_inputs as recovery


def identity(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def test_verified_download_is_atomic_reusable_and_refuses_corruption(tmp_path):
    target = tmp_path / 'object'
    data = b'exact source bytes'
    with pytest.raises(ValueError, match='SHA-256'):
        recovery.write_verified([b'wrong'], target, identity(data), reserve_bytes=0)
    assert not target.exists() and not list(tmp_path.glob('*.pending'))
    assert recovery.write_verified([data[:5], data[5:]], target, identity(data), reserve_bytes=0) is False
    assert recovery.write_verified([], target, identity(data), reserve_bytes=0) is True
    with pytest.raises(ValueError, match='refusing replacement'):
        recovery.write_verified([], target, identity(b'other'), reserve_bytes=0)
    assert target.read_bytes() == data


def test_reserve_failure_preserves_other_pending_and_removes_only_owned_partial(tmp_path, monkeypatch):
    target = tmp_path / 'object'
    pending = tmp_path / 'object.pending'
    pending.write_bytes(b'other invocation')
    with pytest.raises(FileExistsError):
        recovery.write_verified([b'abc'], target, identity(b'abc'), reserve_bytes=0)
    assert pending.read_bytes() == b'other invocation'
    pending.unlink()
    monkeypatch.setattr(recovery.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match='reserve'):
        recovery.write_verified([b'abc'], target, identity(b'abc'), reserve_bytes=2)
    assert not pending.exists() and not target.exists()


def test_interrupted_owned_transfer_resumes_with_full_final_digest(tmp_path):
    target = tmp_path / 'object'
    data = b'abcdef'

    def interrupted():
        yield b'abc'
        raise OSError('connection interrupted')

    with pytest.raises(OSError):
        recovery.write_verified(interrupted(), target, identity(data), reserve_bytes=0, keep_partial=True)
    assert target.with_name('object.pending').read_bytes() == b'abc'
    recovery.write_verified([b'def'], target, identity(data), reserve_bytes=0, resume=True, keep_partial=True)
    assert target.read_bytes() == data
    assert not target.with_name('object.pending').exists()


def test_corrupted_complete_partial_cannot_be_published_on_resume(tmp_path):
    target = tmp_path / 'object'
    target.with_name('object.pending').write_bytes(b'bad')
    with pytest.raises(ValueError, match='SHA-256'):
        recovery.write_verified([], target, identity(b'abc'), reserve_bytes=0, resume=True, keep_partial=True)
    assert not target.exists() and not target.with_name('object.pending').exists()


def test_catalog_preserves_all_historical_sources_and_rejects_changed_custody(tmp_path):
    receipt = {'private': True, 'identity': identity(b'abc'), 'independent_verification': identity(b'abc'),
               'run': 'entity/project/run', 'name': 'bulk/object', 'source': '/old/source'}
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(receipt))
    item = {'role': 'bulk', 'receipt': str(path), 'identity': identity(b'abc'),
            'run': receipt['run'], 'name': receipt['name']}
    jobs = recovery.catalog_jobs({'objects': [item]})
    assert jobs[0]['sources'] == ['/old/source']
    with pytest.raises(ValueError, match='differs'):
        recovery.catalog_jobs({'objects': [{**item, 'name': 'substituted'}]})


def test_explicit_learner_source_archives_keep_custody_without_selecting_checkpoints(tmp_path):
    ident = identity(b'original source')
    receipt = {'private': True, 'identity': ident, 'independent_verification': ident,
               'run': 'entity/project/learner', 'name': 'lfm/source.tar.zst',
               'source': '/original/source.tar.zst'}
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(receipt))
    item = {'role': 'lfm', 'artifact_kind': 'source-archive', 'receipt': str(path),
            'identity': ident, 'run': receipt['run'], 'name': receipt['name']}
    # Implicit learner exports and explicitly typed checkpoint exports remain
    # outside input recovery, including checkpoints in a bulk-owned run.
    skipped = [{**item, 'artifact_kind': 'checkpoint'},
               {**item, 'role': 'bulk', 'artifact_kind': 'checkpoint'},
               {key: value for key, value in item.items() if key != 'artifact_kind'}]
    jobs = recovery.catalog_jobs({'objects': [*skipped, item]})
    assert len(jobs) == 1
    assert (jobs[0]['run'], jobs[0]['name'], jobs[0]['sources']) == (
        receipt['run'], receipt['name'], [receipt['source']])
    assert jobs[0]['custody_receipts'] == [{
        'path': str(path), 'sha256': recovery.digest(path),
        'role': 'lfm', 'artifact_kind': 'source-archive'}]
    assert json.loads(path.read_text()) == receipt
    with pytest.raises(ValueError, match='differs'):
        recovery.catalog_jobs({'objects': [{**item, 'run': 'entity/project/another'}]})
    path.write_text(json.dumps({**receipt, 'independent_verification': {'local_sha256': ident['sha256']}}))
    with pytest.raises(ValueError, match='independent verification'):
        recovery.catalog_jobs({'objects': [item]})


def make_archive(path, names):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as tar:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 3
            tar.addfile(member, io.BytesIO(b'abc'))
    data = zstandard.ZstdCompressor().compress(raw.getvalue())
    path.write_bytes(data)
    return identity(data)


def test_inventory_hashes_exact_members_without_extracting(tmp_path):
    path = tmp_path / 'source.tar.zst'
    ident = make_archive(path, ['bundle/cases.jsonl'])
    result = recovery.inventory_archive(path, ident)
    assert result['members'] == [{'path': 'bundle/cases.jsonl', **identity(b'abc')}]
    assert result['training_admission'] is False and result['extracted'] is False
    assert not (tmp_path / 'bundle').exists()


def test_private_receipt_files_and_gzip_runtime_are_not_treated_as_zstd(tmp_path):
    path = tmp_path / 'receipt.object'
    data = b'{"source":"original"}'
    path.write_bytes(data)
    assert recovery.inventory_archive(path, identity(data))['format'] == 'opaque-file'
    path = tmp_path / 'runtime.object'
    with tarfile.open(path, 'w:gz') as tar:
        member = tarfile.TarInfo('runtime/entry.py')
        member.size = 3
        tar.addfile(member, io.BytesIO(b'abc'))
    result = recovery.inventory_archive(path, identity(path.read_bytes()))
    assert result['format'] == 'tar+gzip'
    assert result['members'][0]['sha256'] == identity(b'abc')['sha256']


@pytest.mark.parametrize('names', [['../escape'], ['/absolute'], ['same', 'same']])
def test_inventory_rejects_unsafe_or_duplicate_members(tmp_path, names):
    path = tmp_path / 'source.tar.zst'
    ident = make_archive(path, names)
    with pytest.raises(ValueError, match='unsafe or duplicate'):
        recovery.inventory_archive(path, ident)
