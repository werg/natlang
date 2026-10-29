import json
import os
import threading
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train.diloco import replica_baseline
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
from natlang_neuralese.train.diloco.protocol import save_tensors
from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator


@pytest.fixture(autouse=True)
def isolated_digest_index():
    replica_baseline._KNOWN.clear(); replica_baseline._UNCHANGED.clear()
    yield
    replica_baseline._KNOWN.clear(); replica_baseline._UNCHANGED.clear()


class Immediate:
    def __init__(self, machine): self.machine = machine
    def read(self, fn): return fn()


def _hash_log(monkeypatch, log):
    real = replica_baseline._hash
    def logged(path):
        st = Path(path).stat()
        with open(log, 'a') as out:
            out.write(json.dumps({'pid': os.getpid(), 'key': list(replica_baseline.inode_key(st)), 'path': str(path), 'bytes': st.st_size}) + '\n')
        return real(path)
    monkeypatch.setattr(replica_baseline, '_hash', logged)


def _entries(log):
    # The tiny receipt itself is always hashed directly; count only the pinned state files.
    entries = [json.loads(line) for line in Path(log).read_text().splitlines()] if Path(log).exists() else []
    return [e for e in entries if not e['path'].endswith('replica-baseline.json')]


def test_capture_then_verify_hashes_each_inode_state_once_and_still_detects_tamper(tmp_path, monkeypatch):
    log = tmp_path / 'hashes.jsonl'; _hash_log(monkeypatch, log)
    root = tmp_path / 'root'
    initialize_coordinator(root, run_id='r', masters={'weight': torch.arange(2048, dtype=torch.float32)}, identity={'parent': 'p'},
                           workers=1, min_quorum=1, fragment_bytes=2048)
    machine = ReplayCoordinator(root, quorum_since={})
    receipt = replica_baseline.capture(Immediate(machine), tmp_path / 'baseline')
    pinned = len(_entries(log)); assert pinned == len(receipt['value']['files'])
    replica_baseline.verify(tmp_path / 'baseline', receipt['sha256'])
    assert len(_entries(log)) == pinned  # retention-style verify reuses the pin's exact digests
    second = replica_baseline.capture(Immediate(machine), tmp_path / 'baseline2')
    assert second['value']['files'] == receipt['value']['files']
    assert len(_entries(log)) == pinned  # unchanged inodes: the next rotation hashes nothing
    # A cold process (empty index) performs the full, exact verification.
    replica_baseline._KNOWN.clear(); replica_baseline._UNCHANGED.clear(); replica_baseline.verify(tmp_path / 'baseline', receipt['sha256'])
    assert len(_entries(log)) == 2 * pinned
    # In-place tamper with preserved size and mtime changes ctime and is detected.
    fragment = next((tmp_path / 'baseline' / 'fragments').iterdir()); before = fragment.stat()
    data = bytearray(fragment.read_bytes()); data[-1] ^= 1; fragment.write_bytes(bytes(data))
    os.utime(fragment, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='baseline file differs'):
        replica_baseline.verify(tmp_path / 'baseline', receipt['sha256'])


def test_replay_digests_cover_changed_fragments_at_pin(tmp_path, monkeypatch):
    log = tmp_path / 'hashes.jsonl'; _hash_log(monkeypatch, log)
    root = tmp_path / 'root'
    initialize_coordinator(root, run_id='r', masters={'weight': torch.ones(1024)}, identity={'parent': 'p'}, workers=1, min_quorum=1,
                           fragment_bytes=1024, grace_seconds=0)
    machine = ReplayCoordinator(root, quorum_since={}); session = '00000000-0000-0000-0000-000000000001'
    replica_baseline.capture(Immediate(machine), tmp_path / 'b0'); start = len(_entries(log))
    delta = tmp_path / 'delta'; save_tensors(delta, {'delta': torch.ones(256)})
    machine.apply({'kind': 'lease', 'args': {'worker': 0, 'session': session}, 'wall': 1.0, 'monotonic': 1.0}, None)
    state = machine.apply({'kind': 'submit', 'args': {'index': 0, 'worker': 0, 'session': session, 'sequence': 1, 'base_version': 0,
                                                       'steps': 1, 'tokens': 3}, 'wall': 2.0, 'monotonic': 2.0}, delta)
    assert state['changed_fragments']  # the merge hashed the new fragment inode during sealing
    replica_baseline.capture(Immediate(machine), tmp_path / 'b1')
    rehashed = [e['path'] for e in _entries(log)[start:]]
    assert not any('fragments/' in p for p in rehashed), rehashed


def test_rotation_under_load_hashes_no_inode_state_twice_and_peer_recovers_exactly(tmp_path, monkeypatch):
    import multiprocessing as mp, socket, time
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal, _identity
    from natlang_neuralese.train.diloco.replica_service import serve_peer, serve_new_primary
    from natlang_neuralese.train.diloco.replica_generation import load_active
    from natlang_neuralese.train.diloco.worker import Transport
    log = tmp_path / 'hashes.jsonl'; _hash_log(monkeypatch, log)
    base = tmp_path / 'base'
    initialize_coordinator(base, run_id='r', masters={'weight': torch.arange(1024, dtype=torch.float32)}, identity={'parent': 'p'},
                           workers=2, min_quorum=2, fragment_bytes=1024, grace_seconds=0)
    baseline = tmp_path / 'baseline'; receipt = replica_baseline.capture(Immediate(ReplayCoordinator(base, quorum_since={})), baseline)
    # The peer is another host in production: give it its own inodes, or its links change the leader's ctimes.
    import shutil; peer_baseline = tmp_path / 'peer-baseline'; shutil.copytree(baseline, peer_baseline)
    limits = {'max_payload_bytes': 1 << 30, 'max_transactions': 24}
    store = tmp_path / 'peerstore'; store.mkdir()
    journal = ReplicaJournal(store / 'journal0', run_identity='r', baseline_sha256=receipt['sha256'], **limits)
    write_json_durable(store / 'replica-active.json', {'schema': 'natlang.diloco-replica-active-generation/1', 'id': 'initial', 'previous': None,
        'baseline': str(peer_baseline.resolve()), 'baseline_sha256': receipt['sha256'], 'journal': str(journal.root.resolve()),
        'head': _identity(journal.identity), **limits}); journal.close()
    def freeport():
        with socket.socket() as sock: sock.bind(('127.0.0.1', 0)); return sock.getsockname()[1]
    peerport = freeport(); port = freeport(); ctx = mp.get_context('fork'); peerstop = ctx.Event(); leaderstop = ctx.Event(); token = 's' * 32
    identity = {'boot_id': 'boot', 'pid': 1, 'start_ticks': '1'}
    evidence = {'primary_fenced': True, 'all_workers_quiesced': True, 'primary': identity, 'quiesced_workers': [0, 1], 'run_id': 'r', 'proof_sha256': 'f' * 64}
    def peer():
        serve_peer(baseline=peer_baseline, baseline_sha256=receipt['sha256'], store=store, work_root=tmp_path / 'peerwork', host='127.0.0.1',
                   port=peerport, token=token, max_transaction_bytes=1 << 20, stop=peerstop)
    def leader():
        j = ReplicaJournal(tmp_path / 'leaderjournal', run_identity='r', baseline_sha256=receipt['sha256'], **limits)
        serve_new_primary(baseline=baseline, baseline_sha256=receipt['sha256'], store=tmp_path / 'leaderstore', root=tmp_path / 'leaderroot',
            host='127.0.0.1', port=port, token=token, peer_url=f'http://127.0.0.1:{peerport}', supervisor_proof={},
            verify_supervisor_fence=lambda _: evidence, expected_primary=identity, expected_workers=[0, 1], stop=leaderstop, journal=j,
            sequence=0, head=j.head, session_transition={'previous_url': 'http://old', 'new_url': 'http://new',
            'previous_token_sha256': __import__('hashlib').sha256(b'old').hexdigest(), 'new_token_sha256': __import__('hashlib').sha256(token.encode()).hexdigest(),
            'generation_nonce': 'n' * 48}, previous_token='old', file_custody_store=store, file_freeze_nonce='f' * 48, allow_http=True, max_generations=3)
    write_json_durable(store / 'replica-files-frozen.json', {'schema': 'natlang.diloco-replica-file-freeze/1', 'nonce': 'f' * 48})
    peerproc = ctx.Process(target=peer); peerproc.start(); time.sleep(.3)
    leaderproc = ctx.Process(target=leader); leaderproc.start()
    try:
        client = Transport(f'http://127.0.0.1:{port}', token, 30, allow_http=True)
        for _ in range(300):
            try: client.json('GET', '/v1/manifest'); break
            except OSError: time.sleep(.02)
        from natlang_neuralese.train.diloco.protocol import Fragment
        sizes = [Fragment.from_dict(f).length for f in client.json('GET', '/v1/manifest')['fragments']]; fragments = len(sizes)
        sessions = ['00000000-0000-0000-0000-00000000000%d' % (i + 1) for i in range(2)]
        for worker in range(2): client.json('POST', '/v1/lease', {'worker': worker, 'session': sessions[worker]})
        errors = []; rounds = 3
        def load(worker):
            c = Transport(f'http://127.0.0.1:{port}', token, 60, allow_http=True)
            delta = tmp_path / f'delta{worker}'
            try:
                for r in range(1, rounds + 1):
                    for index in range(fragments):
                        size = sizes[index]
                        save_tensors(delta, {'delta': torch.full((size,), 0.01 * (worker + 1))})
                        status, answer, _ = c.request('POST', f'/v1/updates/{index}', file=delta, headers={'X-Worker': str(worker),
                            'X-Session': sessions[worker], 'X-Sequence': str(r), 'X-Base-Version': str(r - 1), 'X-Steps': '1', 'X-Tokens': '3'})
                        if status != 200: errors.append((worker, r, index, status, answer))
                        c.json('POST', '/v1/progress', {'worker': worker, 'session': sessions[worker], 'steps': 1, 'tokens': 3, 'metrics': None})
                    for _ in range(500):
                        if min(c.json('GET', '/v1/status')['versions']) >= r: break
                        time.sleep(.02)
            except Exception as error: errors.append((worker, repr(error)))
        threads = [threading.Thread(target=load, args=(w,)) for w in range(2)]
        for t in threads: t.start()
        for t in threads: t.join(300)
        assert not errors, errors
        status = client.json('GET', '/v1/status'); assert status['versions'] == [rounds] * fragments
        generations = sorted(p.name for p in (tmp_path / 'leaderstore').glob('generation-*'))
        retention = json.loads((tmp_path / 'leaderstore' / 'retention.json').read_text())
        assert len(retention['generations']) + len(retention['retired']) >= 3, generations  # several real rotations under load
        progress = json.loads((tmp_path / 'leaderroot' / 'progress.json').read_text())
    finally:
        leaderstop.set(); leaderproc.join(30); peerstop.set(); peerproc.join(30)
    assert leaderproc.exitcode == 0 and peerproc.exitcode == 0
    seen = {}
    for entry in _entries(log):
        key = (entry['pid'], tuple(entry['key']))
        assert key not in seen, ('inode state hashed twice in one process', entry['path'], seen[key])
        seen[key] = entry['path']
    for pid in {e['pid'] for e in _entries(log)}:
        seeds = [e for e in _entries(log) if e['pid'] == pid and e['path'].endswith('seed.safetensors') and 'generation-' in e['path']]
        assert len(seeds) <= 1, [(e['key'], e['path'][-90:]) for e in seeds]  # the constant seed is hashed at most once per process across rotations
    replica_baseline._KNOWN.clear(); replica_baseline._UNCHANGED.clear()
    recovered, machine, _ = load_active(store, run_identity='r', destination=tmp_path / 'recovered')
    try: assert machine.coordinator.progress['rounds'] == progress['rounds']
    finally: recovered.close()
