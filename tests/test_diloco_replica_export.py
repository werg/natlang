import base64
import hashlib
import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal
from natlang_neuralese.train.diloco.replica_seed import verify_seed
from natlang_neuralese.train.diloco import replica_baseline
from scripts.coreweave import diloco_replica_export as ex


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch):
    monkeypatch.setattr(ex, 'SMALL', 1000)


def _store(tmp_path, *, wall=1.0):
    base = tmp_path / 'baseline'; (base / 'fragments').mkdir(parents=True)
    contents = {'manifest.json': json.dumps({'run_id': 'run', 'fingerprint': 'f', 'fragments': [{'index': 0}]}).encode(),
                'seed.safetensors': b's' * 3000, 'progress.json': b'{}', 'fragments/00000.safetensors': b'x' * 2000}
    for name, data in contents.items(): (base / name).write_bytes(data)
    from natlang_neuralese.train.native_resident import code_inventory, digest
    runtime = digest(code_inventory(Path(replica_baseline.__file__).resolve().parents[2]))
    value = {'schema': 'natlang.diloco-replica-baseline/1', 'run_id': 'run', 'fingerprint': 'f', 'quorum_since': {},
             'runtime_code_sha256': runtime, 'files': [{'path': k, 'bytes': len(v), 'sha256': _sha(v)} for k, v in sorted(contents.items())]}
    (base / 'replica-baseline.json').write_text(json.dumps(value))
    baseline_sha = _sha((base / 'replica-baseline.json').read_bytes())
    journal = ReplicaJournal(tmp_path / 'journal', run_identity='run', baseline_sha256=baseline_sha)
    for index in range(5):  # two sealed operations plus one unsealed
        payload = tmp_path / f'p{index}'; payload.write_bytes(bytes([index]) * (1500 if index % 2 == 0 else 0))
        event = {'kind': 'submit', 'args': {}, 'wall': wall, 'monotonic': 1.0} if index % 2 == 0 else {'kind': 'seal'}
        journal.append(event, payload, expected_sequence=journal.sequence + 1, expected_previous=journal.head)
    work = tmp_path / 'work'; work.mkdir(); (work / 'progress.json').write_text(json.dumps({'rounds': [{'at': 5.0, 'version': 1}]}))
    store = tmp_path / 'store'; store.mkdir()
    (store / 'primary-active.json').write_text(json.dumps({'id': 'initial', 'baseline': str(base), 'journal': str(journal.root),
        'baseline_sha256': baseline_sha, 'head': ex.identity(journal.identity), 'work_root': str(work)}))
    journal.close()
    return store


def _select(tmp_path, store):
    out = tmp_path / 'selection.json'
    ex.select({'store': str(store), 'output': str(out)})
    return json.loads(out.read_text())


def test_select_build_seed_restore_verifies_with_replica_seed(tmp_path):
    store = _store(tmp_path); selection = _select(tmp_path, store)
    assert selection['sequence'] == 4 and selection['last_round_at'] == 5.0
    large = {x['path'] for x in selection['large']}
    assert large == {'baseline/seed.safetensors', 'baseline/fragments/00000.safetensors', 'journal/00000000000000000001/payload',
                     'journal/00000000000000000003/payload'}
    receipt_raw, small = ex.build_seed(selection)
    blobs = tmp_path / 'blobs'; blobs.mkdir(); state_blobs = {}
    for item in selection['large']:
        shutil.copyfile(item['source'], blobs / item['sha256']); state_blobs[item['sha256']] = {'local': str(blobs / item['sha256']), 'bytes': item['bytes']}
    manifest = tmp_path / 'manifest.tar'
    ex.write_manifest(manifest, receipt_raw, small, {'schema': ex.EXPORT_SCHEMA})
    journal = tmp_path / 'journal.json'
    journal.write_text(json.dumps({'exports': [{'export_id': 'one', 'phase': 'verified', 'manifest_local': str(manifest)}], 'blobs': state_blobs}))
    restored = ex.restore(journal, 'one', tmp_path / 'seed')
    record = verify_seed(tmp_path / 'seed', expected_receipt_sha256=_sha((tmp_path / 'seed' / 'seed-receipt.json').read_bytes()))
    assert record['sequence'] == restored['sequence'] == 4


def test_build_seed_rejects_tampered_small_file_and_broken_chain(tmp_path):
    store = _store(tmp_path); selection = _select(tmp_path, store)
    bad = json.loads(json.dumps(selection)); bad['small']['baseline/progress.json'] = base64.b64encode(b'{"x":1}').decode()
    with pytest.raises(ValueError, match='small baseline file differs'): ex.build_seed(bad)
    bad = json.loads(json.dumps(selection)); bad['head'] = '0' * 64
    with pytest.raises(ValueError, match='sealed head differs'): ex.build_seed(bad)


def test_gate_closes_for_recent_merge_activity_and_rotation(tmp_path):
    store = _store(tmp_path, wall=time.time())
    state = ex.gate(store, quiet_seconds=60, rotation_bytes=0.75)
    assert not state['open'] and state['reasons'] == ['merge activity']
    assert ex.gate(store, quiet_seconds=0, rotation_bytes=0.75)['open']
    (store / ('generation-' + 'a' * 32)).mkdir()
    assert 'rotation in progress' in ex.gate(store, quiet_seconds=0, rotation_bytes=0.75)['reasons']
    with pytest.raises(ex.RotationInProgress): _select(tmp_path, store)
    (store / 'retention.json').write_text(json.dumps({'generations': [{'id': 'a' * 32}]}))
    assert ex.gate(store, quiet_seconds=0, rotation_bytes=1e-9)['reasons'] == ['rotation imminent']


class _Sink(BaseHTTPRequestHandler):
    received = {}
    def log_message(self, *_): pass
    def do_PUT(self):
        size = int(self.headers['Content-Length']); body = self.rfile.read(size)
        _Sink.received[self.path] = body; self.send_response(200); self.end_headers()


def _server():
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Sink); threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _capability(tmp_path, selection, port):
    items = [{**x, 'url': f'http://127.0.0.1:{port}/{x["sha256"]}', 'headers': {}} for x in selection['large']]
    path = tmp_path / 'cap.json'; path.write_text(json.dumps({'items': items})); return path, items


def test_upload_streams_exact_bytes_at_capped_rate_from_open_inodes(tmp_path, capsys):
    store = _store(tmp_path); selection = _select(tmp_path, store); server = _server(); _Sink.received.clear()
    try:
        cap, items = _capability(tmp_path, selection, server.server_address[1])
        total = sum(x['bytes'] for x in items); start = time.monotonic()
        code = ex.upload({'capability': str(cap), 'store': str(store), 'quiet_seconds': 0, 'rotation_bytes': 0.75,
                          'rate_bytes': total / 1.0, 'max_pause_seconds': 0})
        assert code == 0 and time.monotonic() - start >= 0.9  # the rate cap held
        for item in items: assert _sha(_Sink.received['/' + item['sha256']]) == item['sha256']
    finally: server.shutdown()


def test_upload_pauses_then_aborts_while_merges_run_and_rejects_changed_inode(tmp_path):
    store = _store(tmp_path, wall=time.time()); selection = _select(tmp_path, store); server = _server()
    try:
        cap, items = _capability(tmp_path, selection, server.server_address[1])
        assert ex.upload({'capability': str(cap), 'store': str(store), 'quiet_seconds': 3600, 'rotation_bytes': 0.75,
                          'rate_bytes': 1e9, 'max_pause_seconds': 0}) == 3
        source = Path(items[0]['source']); data = source.read_bytes(); source.unlink(); source.write_bytes(data)  # replaced inode
        with pytest.raises(ValueError, match='selected inode changed'):
            ex.upload({'capability': str(cap), 'store': str(store), 'quiet_seconds': 0, 'rotation_bytes': 0.75, 'rate_bytes': 1e9, 'max_pause_seconds': 0})
    finally: server.shutdown()


def test_prune_keeps_blobs_of_newest_export_and_logs(tmp_path):
    keep = tmp_path / 'keep'; drop = tmp_path / 'drop'; keep.write_bytes(b'k'); drop.write_bytes(b'd')
    state = {'exports': [{'export_id': 'new', 'blob_shas': ['k']}],
             'blobs': {'k': {'local': str(keep), 'bytes': 1, 'object': {'name': 'blobs/k'}},
                       'd': {'local': str(drop), 'bytes': 1, 'object': {'name': 'blobs/d'}}}}
    log = tmp_path / 'deletions.txt'
    assert ex.prune_blobs(state, tmp_path / 'j.json', deletion_log=log, keep_export_id='new') == ['d']
    assert keep.exists() and not drop.exists() and 'blobs/d' in log.read_text() and state['blobs']['d']['local'] is None


def test_run_refuses_unless_enabled(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'state_root': str(tmp_path), 'journal_name': 'j.json', 'enabled': False}))
    with pytest.raises(SystemExit, match='not enabled'): ex.run(config)


def test_remote_runner_executes_gate_and_select_at_idle_priority(tmp_path):
    import subprocess, sys
    store = _store(tmp_path); repo = Path(ex.__file__).resolve().parents[2]
    code = tmp_path / 'code'
    for relative in ex.CODE: (code / relative).parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(repo / relative, code / relative)
    plan = tmp_path / 'plan.json'; plan.write_text(json.dumps({'store': str(store), 'output': str(tmp_path / 'out.json')}))
    argv = ex.remote_argv(str(code), str(repo / 'training/neuralese'), 'select', str(plan), python=sys.executable)
    assert argv[:9] == ['nice', '-n', '19', 'chrt', '-i', '0', 'ionice', '-c', '3']
    result = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1])['sequence'] == 4
    assert json.loads((tmp_path / 'out.json').read_text())['sequence'] == 4
