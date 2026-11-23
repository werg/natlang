import copy
import json
import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
from retained_state_archive import prepare_retained_state_packs, validate_pack, restore_pack, inventory
import vibestudio_campaign


def fixture(tmp_path):
    capture = tmp_path / 'capture'; capture.mkdir()
    root = capture / 'retained-state-owned'; root.mkdir()
    (root / 'empty').mkdir()
    (root / 'nested').mkdir()
    (root / 'nested' / 'sqlite').write_bytes(bytes(range(256)) * 200)
    (root / 'nested' / 'sqlite').chmod(0o640)
    os.link(root / 'nested' / 'sqlite', root / 'same-bytes')
    (root / 'relative-link').symlink_to('nested/sqlite')
    (root / 'outside-link').symlink_to('/not/followed')
    os.utime(root / 'nested' / 'sqlite', ns=(1234567890123456789, 1234567890123456789))
    (capture / 'manifest.json').write_text('ordinary structured capture')
    return capture, root


def test_exact_streamed_file_symlink_hardlink_member_roundtrip_and_reuse(tmp_path):
    capture, root = fixture(tmp_path)
    assert prepare_retained_state_packs(capture) == ['/retained-state-owned/']
    directory = capture / '.retained-state-packs'
    manifest = json.loads((directory / (root.name + '.json')).read_text())
    assert prepare_retained_state_packs(capture) == ['/retained-state-owned/']
    restored = restore_pack(manifest, directory, tmp_path / 'restored')
    def portable(rows):
        return [{k: v for k, v in row.items() if k != 'ctime_ns'} for row in rows]
    assert portable(inventory(root)) == portable(inventory(restored))
    assert (restored / 'same-bytes').stat().st_ino == (restored / 'nested/sqlite').stat().st_ino
    assert (capture / 'manifest.json').read_text() == 'ordinary structured capture'
    assert root.is_dir()


def test_interrupt_rebuild_and_tamper_rejection_without_hiding_source(tmp_path):
    capture, root = fixture(tmp_path)
    directory = capture / '.retained-state-packs'; directory.mkdir()
    (directory / (root.name + '.building-interrupted')).write_bytes(b'partial')
    assert prepare_retained_state_packs(capture)
    assert not (directory / (root.name + '.building-interrupted')).exists()
    manifest = json.loads((directory / (root.name + '.json')).read_text())
    bad = copy.deepcopy(manifest); bad['members'][1]['mode'] ^= 1
    with pytest.raises(ValueError, match='metadata'):
        validate_pack(bad, directory)
    archive = directory / manifest['archive']; archive.write_bytes(archive.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='SHA'):
        prepare_retained_state_packs(capture)
    assert root.exists()


def test_source_mutation_rejected_optout_and_nonopaque_roots_not_hidden(tmp_path, monkeypatch):
    capture, root = fixture(tmp_path)
    (capture / 'retained-state-symlink').symlink_to(root, target_is_directory=True)
    prepare_retained_state_packs(capture)
    (root / 'nested/sqlite').write_bytes(b'changed')
    with pytest.raises(ValueError, match='source changed') as caught:
        prepare_retained_state_packs(capture)
    detail = json.loads(str(caught.value).split('packing: ', 1)[1])
    assert any(row['path'].endswith('/nested/sqlite') and 'bytes' in row['fields'] for row in detail['sample'])
    monkeypatch.setenv('VIBESTUDIO_PACK_RETAINED_STATE', '0')
    assert prepare_retained_state_packs(capture) == []


def test_transfer_and_checksum_use_only_identical_validated_anchored_exclusions(tmp_path, monkeypatch):
    capture, root = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', lambda command, **kw: calls.append(command))
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'check_output', lambda command, **kw: calls.append(command) or '')
    vibestudio_campaign.transfer(capture, 'archive', Path('/durable/capture'))
    sent, checked = calls[1:]
    assert [arg for arg in sent if arg.startswith('--exclude=')] == ['--exclude=/retained-state-owned/']
    assert [arg for arg in checked if arg.startswith('--exclude=')] == ['--exclude=/retained-state-owned/']
    assert '--delete' not in sent and '--delete' in checked
    assert root.exists()


def test_metadata_only_drift_reports_exact_field_and_keeps_archive_and_source(tmp_path):
    capture, root = fixture(tmp_path)
    prepare_retained_state_packs(capture)
    directory = capture / '.retained-state-packs'
    manifest = json.loads((directory / (root.name + '.json')).read_text())
    (root / 'nested/sqlite').chmod(0o600)
    with pytest.raises(ValueError, match='source changed') as caught:
        validate_pack(manifest, directory, root)
    detail = json.loads(str(caught.value).split('packing: ', 1)[1])
    changed = next(row for row in detail['sample'] if row['path'].endswith('/nested/sqlite'))
    assert changed['fields']['mode'] == {'expected': 0o640, 'actual': 0o600}
    assert (directory / manifest['archive']).is_file() and root.is_dir()


def test_external_hardlink_ctime_drift_rehashes_same_bytes_but_rejects_changed_payload(tmp_path):
    capture, root = fixture(tmp_path)
    prepare_retained_state_packs(capture)
    directory = capture / '.retained-state-packs'
    manifest = json.loads((directory / (root.name + '.json')).read_text())
    payload = root / 'nested/sqlite'
    os.link(payload, tmp_path / 'external-cas-borrower')
    assert prepare_retained_state_packs(capture) == ['/retained-state-owned/']
    original = payload.stat()
    with payload.open('r+b') as stream:
        stream.write(b'changed')
    os.utime(payload, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match='source changed during content revalidation'):
        validate_pack(manifest, directory, root)
    assert payload.exists() and (directory / manifest['archive']).exists()
