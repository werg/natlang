"""Elastic DiLoCo join: namespaced role admission, the local relay and join.py's exact role computation."""
import hashlib
import json
import socket
import threading
import time

import pytest
import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.diloco import join
from natlang_neuralese.train.diloco.coordinator import (Coordinator, initialize_coordinator, map_namespace,
                                                         serve_coordinator)
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.relay import serve as serve_relay
from natlang_neuralese.train.diloco.role_contract import select
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, RemoteError, Transport

TOKEN = 'x' * 32


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def lineage(tmp_path, name, shared, seed_bytes):
    """One worker's relocated window: its own window dir (with the window recipe) and its own parent copy."""
    root = tmp_path / name
    (root / 'w' / 'learner').mkdir(parents=True)
    (root / 'w' / 'recipe-runtime.json').write_text('{"recipe": 1}')
    (root / 'parent.pt').write_bytes(seed_bytes)
    recipe = root / 'w' / 'recipe-runtime.json'
    binding = {'path': str(recipe), 'file_sha256': hashlib.sha256(recipe.read_bytes()).hexdigest(),
               'lineage_sha256': 'b' * 64}
    block = {'run_id': 'elastic', 'workers': 4, 'coordinator_url': 'http://127.0.0.1:1', 'local_steps': 2,
             'wire_dtype': 'fp32'}
    argv = ['--continue-from', str(root / 'parent.pt'), '--out', str(root / 'w' / 'learner'),
            '--recipe-binding', json.dumps(binding), '--stage-id', 'sft', '--native-graphs', str(shared['train']),
            '--held-native-graphs', str(shared['held']), '--diloco', json.dumps(block)]
    return root, argv


def test_namespaced_admission_relay_and_learner_start(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    seed = tmp_path / 'seed.pt'
    torch.save({'model': model.state_dict()}, seed)
    shared = {'train': tmp_path / 'train.json', 'held': tmp_path / 'held.json'}
    for path in shared.values():
        path.write_text(path.name)
    root0, argv0 = lineage(tmp_path, 'L0', shared, seed.read_bytes())
    root1, argv1 = lineage(tmp_path, 'L1', shared, seed.read_bytes())
    item1, contract, ns1, settings = join.compute_role(argv1, 4)
    item0, _, ns0, _ = join.compute_role(argv0, 4)
    assert ns1 == {'out': str(root1 / 'w'), 'parent': str(root1 / 'parent.pt')}
    assert item0 != item1 and map_namespace(item0, ns0, ns1) == item1
    declared = {**contract, 'namespace_template': {'worker': 1, 'namespace': ns1,
                                                  'role_sha256': canonical_json_sha256_hex(item1)}}
    coordinator_root = tmp_path / 'coordinator'
    initialize_coordinator(coordinator_root, run_id='elastic', masters=TensorSurface.from_module(model).masters,
                           identity=declared, workers=4, min_quorum=1, grace_seconds=0, fragment_bytes=1024,
                           seed_source={'sha256': item1['identity']['parent_checkpoint_sha256']})
    coordinator_port, relay_port = port(), port()
    secrets = tmp_path / 'secrets'
    secrets.mkdir()
    (secrets / 'diloco-url').write_text(f'http://127.0.0.1:{coordinator_port}')
    (secrets / 'diloco-gateway').write_text('')
    stop = threading.Event()
    threading.Thread(target=serve_coordinator, args=(coordinator_root, '127.0.0.1', coordinator_port, TOKEN),
                     kwargs={'stop': stop}, daemon=True).start()
    threading.Thread(target=serve_relay, args=(relay_port, str(secrets)), kwargs={'stop': stop}, daemon=True).start()
    transport = Transport(f'http://127.0.0.1:{relay_port}', TOKEN, 10, gateway='stale-share-token')
    for _ in range(200):
        try:
            transport.json('GET', '/v1/manifest')
            break
        except (OSError, RemoteError):
            time.sleep(0.02)
    try:
        # The template worker's own role is declared; another worker's relocated role is admitted once.
        assert transport.json('POST', '/v1/admit-role', {'worker': 0, 'contract': item0, 'namespace': ns0})['status'] == 'admitted'
        assert transport.json('POST', '/v1/admit-role', {'worker': 0, 'contract': item0, 'namespace': ns0})['status'] == 'already_admitted'
        tampered = json.loads(json.dumps(item0))
        tampered['identity']['trainer_argv'][tampered['identity']['trainer_argv'].index('--stage-id') + 1] = 'other'
        with pytest.raises(RemoteError, match='session template'):
            transport.json('POST', '/v1/admit-role', {'worker': 2, 'contract': tampered, 'namespace': ns0})
        with pytest.raises(RemoteError, match='session template'):
            transport.json('POST', '/v1/admit-role', {'worker': 2, 'contract': item0, 'namespace': ns1})
        authority = transport.json('GET', '/v1/role-contracts?worker=0')
        assert select(authority, 0, item0['identity']) == canonical_json_sha256_hex(item0)
        # A learner on worker 0 joins through the relay with its exact relocated identity and installs the seed.
        worker0 = DilocoSettings(**{**settings.__dict__, 'worker_id': 0,
                                    'coordinator_url': f'http://127.0.0.1:{relay_port}'})
        learner = DilocoLearner(worker0, surface=TensorSurface.from_module(model), identity=item0['identity'],
                                output=tmp_path / 'learner0', transport=transport)
        learner.start()
        assert learner.contract_sha256 == canonical_json_sha256_hex(item0)
        learner.close()
        assert Coordinator(coordinator_root).progress['admitted_roles']['0'][0]['namespace'] == ns0
        # Large bodies stream through the relay in both directions.
        blob = tmp_path / 'blob'
        blob.write_bytes(bytes(range(256)) * 40000)
        transport.upload_file(blob, 'blob', chunk_bytes=1 << 20)
        transport.download_file('blob', tmp_path / 'blob.back')
        assert (tmp_path / 'blob.back').read_bytes() == blob.read_bytes()
        # A failover is a rewrite of the relay's target files: an unreachable target is a transient 502.
        (secrets / 'diloco-url').write_text(f'http://127.0.0.1:{port()}')
        with pytest.raises(RemoteError) as error:
            transport.json('GET', '/v1/manifest')
        assert error.value.status == 502
    finally:
        stop.set()
