import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/neuralese_data'))
import archive_reuse
import vibestudio_campaign
from natlang_neuralese.common import hashing


@pytest.fixture
def archived(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'hashes.jsonl'))
    source = tmp_path / 'episode'
    source.mkdir()
    (source / 'call.json').write_text('original')
    (source / 'link').symlink_to('call.json')
    destination = tmp_path / 'remote/seed-0000'
    destination.mkdir(parents=True)
    shutil.copytree(source, destination / 'capture', symlinks=True)
    records = {'receipt.json': {'goal': {'status': 'verified', 'manifest': 'manifest.json'}}}
    (destination / 'receipt.json').write_text(json.dumps(records['receipt.json']))
    return source, destination, records


def test_reuse_does_not_reread_unchanged_content(archived, monkeypatch):
    source, destination, records = archived
    signature = archive_reuse.capture_signature(source)
    assert archive_reuse.archive_matches(destination, records, signature)
    original_open = open
    protected = {source / 'call.json', destination / 'capture/call.json'}
    def no_content_read(path, *args, **kwargs):
        if Path(path) in protected:
            pytest.fail('unchanged archive content was reread')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(hashing, 'open', no_content_read, raising=False)
    assert archive_reuse.capture_signature(source) == signature
    assert archive_reuse.archive_matches(destination, records, signature)


def test_same_size_corruption_with_restored_mtime_is_not_reused(archived):
    source, destination, records = archived
    signature = archive_reuse.capture_signature(source)
    assert archive_reuse.archive_matches(destination, records, signature)
    path = destination / 'capture/call.json'
    before = path.stat()
    path.write_text('tampered')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not archive_reuse.archive_matches(destination, records, signature)


def test_receipt_mismatch_is_not_silently_overwritten(archived):
    source, destination, records = archived
    path = destination / 'receipt.json'
    path.write_text('{"different":true}')
    with pytest.raises(ValueError, match='immutable archive record differs'):
        archive_reuse.archive_matches(destination, records, archive_reuse.capture_signature(source))
    assert json.loads(path.read_text()) == {'different': True}


def test_missing_files_extra_files_and_symlink_drift_prevent_reuse(archived):
    source, destination, records = archived
    signature = archive_reuse.capture_signature(source)
    extra = destination / 'capture/extra.json'
    extra.write_text('{}')
    assert not archive_reuse.archive_matches(destination, records, signature)
    extra.unlink()
    link = destination / 'capture/link'
    link.unlink()
    link.symlink_to('somewhere-else')
    assert not archive_reuse.archive_matches(destination, records, signature)
    (destination / 'capture/call.json').unlink()
    assert not archive_reuse.archive_matches(destination, records, signature)


def test_capture_change_during_fingerprint_is_rejected(archived, monkeypatch):
    source, _, _ = archived
    original = archive_reuse.sha256_file_hex_cached
    def changing(path, **kwargs):
        result = original(path, **kwargs)
        Path(path).write_text('modified after hash')
        return result
    monkeypatch.setattr(archive_reuse, 'sha256_file_hex_cached', changing)
    with pytest.raises(ValueError, match='capture changed'):
        archive_reuse.capture_signature(source)


def test_real_transfer_reuse_corruption_repair_and_opt_out(tmp_path, monkeypatch):
    if not shutil.which('rsync'):
        pytest.skip('rsync is required for the archive transport integration test')
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'hashes.jsonl'))
    root = tmp_path / 'campaign'
    root.mkdir()
    (root / 'waves').mkdir()
    source = tmp_path / 'episode'
    source.mkdir()
    original = source / 'call.json'
    original.write_text('original')
    destination = tmp_path / 'remote/seed-0000'
    args = SimpleNamespace(root=root, archive_host='dgx', defer_export=True,
                           archive_repo=Path(__file__).resolve().parents[1])
    receipt = {'goal': {'status': 'verified'}}
    commands = []
    def local_transport(command, **kwargs):
        commands.append(command)
        if command[0] == 'ssh':
            argv = shlex.split(command[-1])
        else:
            argv = [arg.removeprefix('dgx:') for arg in command if not arg.startswith('--rsh=')]
        return subprocess.run(argv, check=True, text=True, **kwargs)
    monkeypatch.setattr(vibestudio_campaign, 'archive_transport', local_transport)
    vibestudio_campaign.remote_archive(args, destination, receipt, source)
    assert (destination / 'capture/call.json').read_text() == 'original'
    commands.clear()
    vibestudio_campaign.remote_archive(args, destination, receipt, source)
    assert len(commands) == 1 and commands[0][0] == 'ssh'
    target = destination / 'capture/call.json'
    before = target.stat()
    target.write_text('tampered')
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    commands.clear()
    vibestudio_campaign.remote_archive(args, destination, receipt, source)
    assert target.read_text() == 'original'
    assert any(command[0] == 'rsync' and '--checksum' in command and '--dry-run' not in command for command in commands)
    monkeypatch.setenv('NATLANG_REUSE_SEED_ARCHIVES', '0')
    commands.clear()
    vibestudio_campaign.remote_archive(args, destination, receipt, source)
    assert any(command[0] == 'rsync' for command in commands)
    assert original.read_text() == 'original'
    monkeypatch.delenv('NATLANG_REUSE_SEED_ARCHIVES')
    (destination / 'receipt.json').write_text('{"different":true}')
    commands.clear()
    with pytest.raises(subprocess.CalledProcessError):
        vibestudio_campaign.remote_archive(args, destination, receipt, source)
    assert not any(command[0] == 'rsync' for command in commands)
    assert json.loads((destination / 'receipt.json').read_text()) == {'different': True}


def test_reused_seed_still_reaches_export(archived, monkeypatch, tmp_path):
    source, destination, records = archived
    root = tmp_path / 'campaign'
    (root / 'waves').mkdir(parents=True)
    args = SimpleNamespace(root=root, archive_host='dgx', defer_export=False,
                           archive_repo=Path('/reader'), archive=tmp_path, archive_held=Path('/held'))
    calls = []
    monkeypatch.setattr(vibestudio_campaign, 'archive_transport',
                        lambda *a, **kw: SimpleNamespace(stdout='true'))
    monkeypatch.setattr(vibestudio_campaign, 'transfer', lambda *_: pytest.fail('unchanged seed retransferred'))
    def run(command, **kwargs):
        calls.append(command)
        # First export marker is absent; the launched export then finishes.
        probe = command[-1].startswith('test -f ')
        return SimpleNamespace(returncode=1 if probe and len(calls) == 1 else 0)
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', run)
    vibestudio_campaign.remote_archive(args, destination, records['receipt.json'], source)
    assert any('memory_ledger.py' in command[-1] for command in calls)


def test_mutable_index_uses_checksum_repair_with_explicit_opt_out(monkeypatch):
    calls = []
    monkeypatch.setattr(vibestudio_campaign, 'archive_transport', lambda command: calls.append(command))
    args = SimpleNamespace(archive_host='dgx', archive=Path('/archive'))
    vibestudio_campaign.sync_index(args, [Path('/status.json')])
    assert '--checksum' in calls[-1]
    monkeypatch.setenv('NATLANG_ARCHIVE_CHECKSUM_TRANSFER', '0')
    vibestudio_campaign.sync_index(args, [Path('/status.json')])
    assert '--checksum' not in calls[-1]
