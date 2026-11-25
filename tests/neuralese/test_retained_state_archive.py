import copy
import json
import os
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
from retained_state_archive import prepare_retained_state_packs, validate_pack, restore_pack, inventory, digest_file
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
    from subprocess import CompletedProcess
    capture, root = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run',
                        lambda command, **kw: calls.append(command) or CompletedProcess(command, 0, stdout=''))
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


def test_source_rehash_checks_open_descriptor_identity_not_just_path_and_bytes(tmp_path, monkeypatch):
    capture, root = fixture(tmp_path)
    prepare_retained_state_packs(capture)
    payload = root / 'nested/sqlite'
    other = tmp_path / 'substituted-file'
    other.write_bytes(payload.read_bytes())
    os.link(payload, tmp_path / 'external-borrower')
    original_open = open

    def substituted_open(path, *args, **kwargs):
        # Deterministically model opening another inode while the pathname's
        # lstat remains unchanged. Matching bytes alone do not prove identity.
        return original_open(other if path == payload else path, *args, **kwargs)

    monkeypatch.setattr('retained_state_archive.open', substituted_open, raising=False)
    with pytest.raises(ValueError, match='source changed during content revalidation'):
        prepare_retained_state_packs(capture)
    assert payload.exists()


def test_stable_source_hash_refuses_symlinks(tmp_path):
    payload = tmp_path / 'payload'
    payload.write_bytes(b'original')
    link = tmp_path / 'link'
    link.symlink_to(payload)
    with pytest.raises(OSError):
        digest_file(link, stable=True)


def stopped_fixture(tmp_path):
    capture = tmp_path / 'capture'; capture.mkdir(parents=True)
    root = capture / 'retained-state-owned'; root.mkdir()
    (root / 'retention.json').write_text(json.dumps({'instance': 'owned', 'stopped': True}))
    generation = root / ('a' * 32)
    state = generation / 'workspaces/example/state'
    preserved = ['.databases/native.sqlite', '.databases/native.sqlite-wal', 'blobs/value',
                 'model-journal/exchanges.json', 'build-sources/source.ts', 'scratch/user/builds/file',
                 'extensions/storage/user/native-runtime/file']
    for name in [*preserved, 'builds/key/bundle.js', 'native-runtime/id/typescript/tsc']:
        path = state / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(name.encode())
    for name in ['source/model-authored.ts', 'capture-evidence/calls.json', 'cas/source', 'effects.json']:
        path = generation / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(name.encode())
    return capture, root, state, preserved


def test_slim_stopped_diagnostic_pack_omits_only_owned_installed_and_build_caches(tmp_path):
    capture, root, state, preserved = stopped_fixture(tmp_path)
    prepare_retained_state_packs(capture)
    directory = capture / '.retained-state-packs'
    manifest = json.loads((directory / (root.name + '.json')).read_text())
    assert manifest['stoppedInstance'] == 'owned'
    assert len(manifest['omittedGeneratedDirectories']) == 2
    restored = restore_pack(manifest, directory, tmp_path / 'restored')
    restored_state = restored / state.relative_to(root)
    assert not (restored_state / 'builds').exists()
    assert not (restored_state / 'native-runtime').exists()
    for name in preserved:
        assert (restored_state / name).read_bytes() == (state / name).read_bytes()
    assert (restored / ('a' * 32) / 'source/model-authored.ts').read_bytes() == b'source/model-authored.ts'
    assert 'must be rebuilt' in manifest['restoreScope']
    assert (state / 'builds/key/bundle.js').exists()  # packing never mutates working state
    validate_pack(manifest, directory, root)
    bad = copy.deepcopy(manifest)
    bad['omittedGeneratedDirectories'] = [root.name + '/../../source']
    with pytest.raises(ValueError, match='omission path'):
        validate_pack(bad, directory)


@pytest.mark.parametrize('mode', ['unjoined', 'full-optout'])
def test_unjoined_or_explicit_full_native_archives_preserve_generated_files(tmp_path, monkeypatch, mode):
    capture, root, state, _ = stopped_fixture(tmp_path)
    if mode == 'unjoined':
        (root / 'retention.json').write_text(json.dumps({'instance': 'owned', 'stopped': False}))
    else:
        monkeypatch.setenv('VIBESTUDIO_ARCHIVE_GENERATED_NATIVE_STATE', '1')
    prepare_retained_state_packs(capture)
    manifest = json.loads((capture / '.retained-state-packs' / (root.name + '.json')).read_text())
    assert manifest['omittedGeneratedDirectories'] == []
    assert any(row['path'].endswith('/builds/key/bundle.js') for row in manifest['members'])


def ingestion_fixture(tmp_path, monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from natlang_neuralese.common import data_registry
    archive = tmp_path / 'wave'; archive.mkdir()
    capture, root, state, _ = stopped_fixture(archive)
    prepare_retained_state_packs(capture)
    raw = capture / 'original-trajectory.json.gz'; raw.write_bytes(b'original paid calls, prompts and native observations')
    (archive / 'receipt.json').write_text(json.dumps({'partial': {'status': 'verified'}}))
    (archive / 'exported.json').write_text(json.dumps({'dataset': 'ds', 'verified': 1}))
    work = tmp_path / 'work'; work.mkdir()
    (work / 'episodes.jsonl.zst').write_bytes(b'independently admitted native IR')
    record = {'dataset': 'ds', 'verified': 1, 'extraction': 'episodes.jsonl.zst',
              'artifact_paths': [str(raw)], 'episodes': [{'task_completed': False, 'decisions': 3}]}
    (work / 'ingest.jsonl').write_text(json.dumps(record) + '\n')
    snapshot = {'files': [{'path': 'capture/original-trajectory.json.gz'}]}
    sha = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    registry = SimpleNamespace(repo=tmp_path, datasets={'ds': {'id': 'ds', 'path': str(archive),
                'manifest_sha256': sha, 'native_state_policy': 'recovery-infrastructure-excluded/1'}},
                snapshots={}, extra_files={}, consumption=[])
    monkeypatch.setattr(data_registry, 'load', lambda **kw: registry)
    monkeypatch.setattr(data_registry, 'absolute', lambda value, **kw: Path(value))
    monkeypatch.setattr(data_registry, 'stored_manifest', lambda key: snapshot)
    return archive, work, record, registry, snapshot, raw, root


def test_ingested_partial_goal_retires_only_unregistered_pack_infrastructure(tmp_path, monkeypatch):
    from retained_state_archive import retire_ingested_native_state
    archive, work, record, _, _, raw, root = ingestion_fixture(tmp_path, monkeypatch)
    original = raw.read_bytes()
    retired = retire_ingested_native_state(archive, record, work=work)
    assert len(retired) == 1
    assert raw.read_bytes() == original
    assert root.is_dir()  # No working or manually restored state is deleted.
    assert not list((archive / 'capture/.retained-state-packs').glob('*.tar.gz'))
    proof = json.loads((archive / 'native-state-retirement.json').read_text())
    assert proof['retired'][0]['archive_sha256']
    assert retire_ingested_native_state(archive, record, work=work) == []


@pytest.mark.parametrize('blocker', ['legacy', 'old-snapshot', 'data-file-ref', 'artifact-ref',
                                    'unrecovered', 'not-ingested', 'missing-ir', 'optout', 'hardlink', 'consumer'])
def test_native_retirement_preserves_real_recovery_and_immutable_dependencies(tmp_path, monkeypatch, blocker):
    import hashlib
    from retained_state_archive import retire_ingested_native_state
    archive, work, record, registry, snapshot, _, _ = ingestion_fixture(tmp_path, monkeypatch)
    pack = next((archive / 'capture/.retained-state-packs').glob('*.tar.gz'))
    if blocker == 'legacy':
        registry.datasets['ds'].pop('native_state_policy')
    elif blocker == 'old-snapshot':
        old = {'files': [{'path': str(pack.relative_to(archive))}]}
        sha = hashlib.sha256(json.dumps(old, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        registry.snapshots['ds'] = [{'id': 'ds', 'path': str(archive), 'manifest_sha256': sha}]
        monkeypatch.setattr('natlang_neuralese.common.data_registry.stored_manifest',
                            lambda key: old if key == sha else snapshot)
    elif blocker == 'data-file-ref':
        registry.extra_files['ds'] = [{'path': str(pack.relative_to(archive))}]
    elif blocker == 'artifact-ref':
        record['artifact_paths'].append(str(pack))
        (work / 'ingest.jsonl').write_text(json.dumps(record) + '\n')
    elif blocker == 'unrecovered':
        (archive / 'receipt.json').write_text(json.dumps({'partial': {'status': 'test-failed'}}))
    elif blocker == 'not-ingested':
        (work / 'ingest.jsonl').write_text('')
    elif blocker == 'missing-ir':
        (work / 'episodes.jsonl.zst').unlink()
    elif blocker == 'optout':
        monkeypatch.setenv('VIBESTUDIO_RETIRE_INGESTED_NATIVE_STATE', '0')
    elif blocker == 'hardlink':
        os.link(pack, tmp_path / 'actual-donor-alias')
    elif blocker == 'consumer':
        registry.consumption.append({'items': [{'path': str(pack), 'unregistered': True}]})
    assert retire_ingested_native_state(archive, record, work=work) == []
    assert pack.exists()


def test_native_producer_excludes_only_recovery_working_state_from_training_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    seen = []
    monkeypatch.setattr(vibestudio_campaign, 'ZstdWriter', lambda path: SimpleNamespace(
        rows=1, write=lambda value: None, close=lambda **kw: None))
    monkeypatch.setattr(vibestudio_campaign, 'verified_episode', lambda *args: {
        'trajectory_ir': {'outcome': {'status': 'partial', 'task_completed': False}}, 'turns': [1]})
    monkeypatch.setattr(vibestudio_campaign, 'write_graph_telemetry', lambda *args: {})
    monkeypatch.setattr(vibestudio_campaign.data_registry, 'record_dataset',
                        lambda *args, **kw: seen.append(kw) or {'id': 'ds'})
    vibestudio_campaign._export_trajectories(tmp_path, {'a': {'status': 'verified', 'manifest': 'a.json'}}, None, None)
    assert seen[0]['exclude'] == ['capture/retained-state-*', 'capture/.retained-state-packs/**']
    assert seen[0]['extra']['native_state_policy'] == 'recovery-infrastructure-excluded/1'
    assert 'capture/**' in seen[0]['include']  # all original trajectory artifacts stay registered


def test_producer_keeps_actually_bound_native_state_in_registry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from live_teacher_sources import observe_artifact
    seen = []
    referenced = tmp_path / 'capture/.retained-state-packs/needed.json'
    referenced.parent.mkdir(parents=True); referenced.write_text('{}')
    def verified(*args):
        observe_artifact(referenced)
        return {'trajectory_ir': {'outcome': {'status': 'partial'}}, 'turns': [1]}
    monkeypatch.setattr(vibestudio_campaign, 'ZstdWriter', lambda path: SimpleNamespace(
        rows=1, write=lambda value: None, close=lambda **kw: None))
    monkeypatch.setattr(vibestudio_campaign, 'verified_episode', verified)
    monkeypatch.setattr(vibestudio_campaign, 'write_graph_telemetry', lambda *args: {})
    monkeypatch.setattr(vibestudio_campaign.data_registry, 'record_dataset',
                        lambda *args, **kw: seen.append(kw) or {'id': 'ds'})
    vibestudio_campaign._export_trajectories(tmp_path, {'a': {'status': 'verified', 'manifest': 'a.json'}}, None, None)
    assert seen[0]['exclude'] == []
    assert seen[0]['extra']['native_state_policy'] == 'recovery-referenced/1'
