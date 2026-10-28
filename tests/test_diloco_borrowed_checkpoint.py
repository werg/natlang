import io
import socket
import threading
import time

import torch

from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport


def test_synchronous_checkpoint_borrows_stable_baselines_across_real_merge(tmp_path):
    model = torch.nn.Linear(2, 2)
    surface = TensorSurface.from_module(model)
    root = tmp_path / 'coordinator'
    initialize_coordinator(root, run_id='borrow', masters=surface.masters, identity={'seed': 'same'},
                           workers=1, min_quorum=1, fragment_bytes=1024, grace_seconds=0)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    token = 'borrowed-checkpoint-test-' + 'x' * 32
    stop = threading.Event()
    server = threading.Thread(target=serve_coordinator, args=(root, '127.0.0.1', port, token),
                              kwargs={'stop': stop}, daemon=True)
    server.start()
    url = f'http://127.0.0.1:{port}'
    transport = Transport(url, token, 10)
    for _ in range(100):
        try:
            transport.json('GET', '/v1/status'); break
        except OSError:
            time.sleep(.02)
    learner = DilocoLearner(DilocoSettings(run_id='borrow', workers=1, worker_id=0,
        coordinator_url=url,
        local_steps=1, heartbeat_seconds=.05, allow_private_http=True),
        surface=surface, identity={'seed': 'same'}, output=tmp_path / 'worker', transport=transport)
    try:
        learner.start()
        borrowed = learner.state_dict(clone_baselines=False)
        independent = learner.state_dict()
        for live, shared, copied in zip(learner.baselines, borrowed['baselines'], independent['baselines']):
            assert live.data_ptr() == shared.data_ptr()
            assert copied.data_ptr() != live.data_ptr() and torch.equal(shared, copied)
        # This is the metadata-only path used after a synchronous recovery save.
        assert learner.cursors() == {key: borrowed[key] for key in ('versions', 'sequences')}
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        optimizer.zero_grad(); model(torch.ones(1, 2)).sum().backward(); optimizer.step()
        learner.after_step(tokens=4)
        assert learner.finish()
        for shared, copied in zip(borrowed['baselines'], independent['baselines']):
            assert torch.equal(shared, copied)
        payload = io.BytesIO(); torch.save(borrowed, payload); payload.seek(0)
        restored = torch.load(payload, weights_only=True)
        assert restored['versions'] == borrowed['versions']
        assert all(torch.equal(a, b) for a, b in zip(restored['baselines'], independent['baselines']))
    finally:
        learner.close(); stop.set(); server.join(timeout=3)
