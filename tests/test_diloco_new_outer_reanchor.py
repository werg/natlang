import socket
import threading
import time

import torch
import pytest

from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport


def test_fresh_outer_retains_peer_weights_until_actual_update(tmp_path):
    seed = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        seed.weight.fill_(1)
    root = tmp_path / 'coordinator'
    initialize_coordinator(root, run_id='new', masters=TensorSurface.from_module(seed).masters,
                           identity={'seed': 'authenticated'}, workers=3, min_quorum=3,
                           fragment_bytes=1024, grace_seconds=0, outer_lr=1, outer_momentum=0)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    stop = threading.Event()
    server = threading.Thread(target=serve_coordinator, args=(root, '127.0.0.1', port, 'x'*32),
                              kwargs={'stop': stop}, daemon=True)
    server.start()
    url = f'http://127.0.0.1:{port}'
    transport = Transport(url, 'x'*32, 10)
    for _ in range(100):
        try:
            transport.json('GET', '/v1/status')
            break
        except OSError:
            time.sleep(.02)
    learners = []
    try:
        for worker in range(3):
            model = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(worker + 1)
            before = model.weight.detach().clone()
            settings = DilocoSettings(run_id='new', workers=3, worker_id=worker,
                                      coordinator_url=url, local_steps=100,
                                      wire_dtype='fp32', heartbeat_seconds=.1)
            learner = DilocoLearner(settings, surface=TensorSurface.from_module(model),
                                    identity={'seed': 'authenticated'}, output=tmp_path / str(worker),
                                    transport=transport)
            deviations = learner.reanchor_fresh()
            learners.append((model, learner))
            assert torch.equal(before, model.weight)
            assert deviations[0]['max_abs'] == worker
            assert learner.local_steps == 0 and not learner._flushable(0)
            assert torch.equal(learner.baselines[0], seed.weight.flatten())
        for model, learner in learners:
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            learner.after_step(tokens=1)
            learner._queue(0)
        for _ in range(200):
            if transport.json('GET', '/v1/status')['versions'] == [1]:
                break
            time.sleep(.02)
        assert transport.json('GET', '/v1/status')['versions'] == [1]
        update = learners[0][1]._fetch(0, -1)
        assert torch.allclose(update[1], torch.full((2,), 1.9), atol=1e-6)
        with pytest.raises(ValueError, match='unused coordinator'):
            learners[0][1].reanchor_fresh()
    finally:
        for _, learner in learners:
            learner.close()
        stop.set()
        server.join(timeout=3)
