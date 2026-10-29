import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
from natlang_neuralese.train.diloco.replica_seed import verify_seed
from natlang_neuralese.train.diloco import replica_baseline
from scripts.coreweave import diloco_replica_export as ex


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _store(tmp_path):
    base = tmp_path / 'baseline'; (base / 'fragments').mkdir(parents=True)
    contents = {'manifest.json': json.dumps({'run_id': 'run', 'fingerprint': 'f', 'fragments': [{'index': 0}]}).encode(),
                'seed.safetensors': b's' * 3000, 'progress.json': b'{}', 'fragments/00000.safetensors': b'x' * 2000}
    for name, data in contents.items(): (base / name).write_bytes(data)
    from natlang_neuralese.train.native_resident import code_inventory, digest
    runtime = digest(code_inventory(Path(replica_baseline.__file__).resolve().parents[2]))
    value = {'schema': 'natlang.diloco-replica-baseline/1', 'run_id': 'run', 'fingerprint': 'f', 'quorum_since': {},
             'runtime_code_sha256': runtime,
             'files': [{'path': k, 'bytes': len(v), 'sha256': _sha(v)} for k, v in sorted(contents.items())]}
    (base / 'replica-baseline.json').write_text(json.dumps(value))
    baseline_sha = _sha((base / 'replica-baseline.json').read_bytes())
    journal = ReplicaJournal(tmp_path / 'journal', run_identity='run', baseline_sha256=baseline_sha)
    for index in range(5):  # two sealed operations plus one unsealed
        payload = tmp_path / f'p{index}'; payload.write_bytes(bytes([index]) * (1500 if index % 2 == 0 else 0))
        journal.append({'kind': 'submit' if index % 2 == 0 else 'seal'}, payload,
                       expected_sequence=journal.sequence + 1, expected_previous=journal.head)
    store = tmp_path / 'store'; store.mkdir()
    (store / 'primary-active.json').write_text(json.dumps({'id': 'initial', 'baseline': str(base), 'journal': str(journal.root),
        'baseline_sha256': baseline_sha, 'head': ex.identity(journal.identity), 'work_root': str(tmp_path / 'work')}))
    journal.close()
    return store


def test_pack_verify_restore_seed_and_dedupe(tmp_path):
    store = _store(tmp_path)
    first = ex.pack({'store': str(store), 'archive': str(tmp_path / 'a' / 'one.tar'), 'export_id': 'one', 'dedupe_bytes': 1000})
    assert first['export']['sequence'] == 4 and first['export']['references'] == {}
    assert first['sha256'] == _sha(Path(first['path']).read_bytes())
    one = ex.verify_archive(first['path'], expected_export_id='one')
    assert one['sequence'] == 4 and one['stored'] == one['files']
    exclude = {d: {'export_id': 'one', 'member': m, 'object': {}} for d, m in first['blobs'].items()}
    second = ex.pack({'store': str(store), 'archive': str(tmp_path / 'a' / 'two.tar'), 'export_id': 'two',
                      'dedupe_bytes': 1000, 'exclude': exclude})
    two = ex.verify_archive(second['path'])
    assert two['referenced'] == len(first['blobs']) > 0 and second['bytes'] < first['bytes']
    journal = tmp_path / 'journal.json'
    journal.write_text(json.dumps({'exports': [
        {'export_id': 'one', 'phase': 'verified', 'local_copy': first['path'], 'verification': one},
        {'export_id': 'two', 'phase': 'verified', 'local_copy': second['path'], 'verification': two}]}))
    restored = ex.restore(journal, 'two', tmp_path / 'seed')
    receipt_sha = _sha((tmp_path / 'seed' / 'seed-receipt.json').read_bytes())
    record = verify_seed(tmp_path / 'seed', expected_receipt_sha256=receipt_sha)
    assert record['sequence'] == restored['sequence'] == 4


def test_verify_rejects_tampered_member(tmp_path):
    store = _store(tmp_path)
    packed = ex.pack({'store': str(store), 'archive': str(tmp_path / 'one.tar'), 'export_id': 'one'})
    data = bytearray(Path(packed['path']).read_bytes()); index = data.index(b'x' * 2000); data[index] = ord('y')
    Path(packed['path']).write_bytes(bytes(data))
    with pytest.raises(ValueError, match='member differs'):
        ex.verify_archive(packed['path'])


def test_pack_waits_for_generation_registration(tmp_path):
    store = _store(tmp_path)
    (store / ('generation-' + 'a' * 32)).mkdir()
    with pytest.raises(ex.RotationInProgress):
        ex.pack({'store': str(store), 'archive': str(tmp_path / 'one.tar'), 'export_id': 'one'})
    (store / 'retention.json').write_text(json.dumps({'generations': [{'id': 'a' * 32}]}))
    assert ex.pack({'store': str(store), 'archive': str(tmp_path / 'one.tar'), 'export_id': 'one'})['export']['sequence'] == 4


def test_prune_keeps_referenced_blobs_for_restore(tmp_path):
    store = _store(tmp_path)
    first = ex.pack({'store': str(store), 'archive': str(tmp_path / 'one' / 'coordinator.tar'), 'export_id': 'one', 'dedupe_bytes': 1000})
    one = ex.verify_archive(first['path'])
    state = {'exports': [{'export_id': 'one', 'phase': 'verified', 'local_copy': first['path'], 'pack': first,
                          'object': {'run': 'r', 'name': 'n1'}, 'verification': one}]}
    second = ex.pack({'store': str(store), 'archive': str(tmp_path / 'two' / 'coordinator.tar'), 'export_id': 'two',
                      'dedupe_bytes': 1000, 'exclude': ex._exclusions(state)})
    state['exports'].append({'export_id': 'two', 'phase': 'verified', 'local_copy': second['path'], 'pack': second,
                             'object': {'run': 'r', 'name': 'n2'}, 'verification': ex.verify_archive(second['path'])})
    journal = tmp_path / 'journal.json'; log = tmp_path / 'deletions.txt'
    removed = ex.prune(state, journal, {'local_root': str(tmp_path / 'local')}, deletion_log=log)
    assert removed == [first['path']] and not Path(first['path']).exists() and 'n1' in log.read_text()
    assert set(state['blobs']) == {r['sha256'] for r in second['export']['references'].values()}
    assert set(ex._exclusions(state)) >= set(state['blobs'])
    journal.write_text(json.dumps(state))
    restored = ex.restore(journal, 'two', tmp_path / 'seed')
    assert verify_seed(tmp_path / 'seed', expected_receipt_sha256=_sha((tmp_path / 'seed' / 'seed-receipt.json').read_bytes()))['sequence'] == restored['sequence']
