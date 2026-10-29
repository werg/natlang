"""A coordinator outage is waited out: startup/lease retry, training continues offline, catch-up memory is bounded."""

import threading
import time

import pytest
import torch

from natlang_neuralese.train.diloco import worker as worker_module
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator
from natlang_neuralese.train.diloco.protocol import TensorSurface, save_tensors
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, RemoteError, Transport, patient, transient

TOKEN = "t" * 32


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(32, 64), torch.nn.Linear(64, 8))


def _serve(root, port):
    stop = threading.Event()
    thread = threading.Thread(target=serve_coordinator, args=(root, "127.0.0.1", port, TOKEN), kwargs={"stop": stop}, daemon=True)
    thread.start()
    transport = Transport(f"http://127.0.0.1:{port}", TOKEN, 5)
    for _ in range(200):
        try:
            transport.json("GET", "/v1/status")
            return stop, thread
        except OSError:
            time.sleep(0.02)
    raise AssertionError("coordinator did not start")


def _port():
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_transient_classification_and_patience():
    assert transient(RemoteError(503, "upstream connect error")) and transient(RemoteError(502, "")) and transient(TimeoutError())
    assert transient(ConnectionRefusedError()) and transient(OSError("Incomplete HTTP response"))
    assert not transient(RemoteError(403, "")) and not transient(RemoteError(409, "{}")) and not transient(ValueError())
    assert not transient(FileNotFoundError("x"))
    calls, sleeps = [], []
    def flaky():
        calls.append(1)
        if len(calls) < 5:
            raise RemoteError(503, "upstream connect error")
        return "ok"
    assert patient(flaky, what="unit", sleep=sleeps.append) == "ok" and sleeps == [1.0, 2.0, 4.0, 8.0]
    with pytest.raises(RemoteError, match="403"):
        patient(lambda: (_ for _ in ()).throw(RemoteError(403, "denied")), what="unit", sleep=sleeps.append)
    with pytest.raises(RemoteError, match="503"):
        patient(lambda: (_ for _ in ()).throw(RemoteError(503, "down")), what="unit", wait_seconds=0, sleep=sleeps.append)


def test_learner_startup_waits_out_gateway_503(tmp_path, monkeypatch):
    root, port = tmp_path / "coord", _port()
    initialize_coordinator(root, run_id="unit", masters=TensorSurface.from_module(_model()).masters,
                           identity={"objective": "toy"}, workers=1, min_quorum=1, fragment_bytes=4096, grace_seconds=0.0)
    stop, thread = _serve(root, port)

    class Flaky(Transport):
        failures = 0
        def request(self, method, path, **kwargs):
            # Fail every distinct startup call (manifest, lease, fragment fetches) twice at the gateway first.
            if type(self).failures < 6:
                type(self).failures += 1
                raise RemoteError(503, "upstream connect error or disconnect/reset before headers")
            return super().request(method, path, **kwargs)

    real = worker_module.patient
    monkeypatch.setattr(worker_module, "patient", lambda call, **kw: real(call, **{**kw, "sleep": lambda _: None}))
    settings = DilocoSettings(run_id="unit", workers=1, worker_id=0, coordinator_url=f"http://127.0.0.1:{port}",
                              local_steps=4, wire_dtype="fp32", heartbeat_seconds=100)
    learner = DilocoLearner(settings, surface=TensorSurface.from_module(_model()), identity={"objective": "toy"},
                            output=tmp_path / "w", transport=Flaky(settings.coordinator_url, TOKEN, 5))
    try:
        learner.start()
        assert Flaky.failures == 6 and all(v == 0 for v in learner.versions)
    finally:
        learner.close()
        stop.set()
        thread.join(timeout=5)


def test_outage_counts_offline_and_catch_up_downloads_stay_bounded(tmp_path):
    root, port = tmp_path / "coord", _port()
    initialize_coordinator(root, run_id="unit", masters=TensorSurface.from_module(_model()).masters,
                           identity={"objective": "toy"}, workers=2, min_quorum=1, fragment_bytes=1024,
                           grace_seconds=0.0, outer_lr=1.0, outer_momentum=0.0)
    stop, thread = _serve(root, port)
    url = f"http://127.0.0.1:{port}"
    settings = DilocoSettings(run_id="unit", workers=2, worker_id=0, coordinator_url=url, local_steps=10_000,
                              wire_dtype="fp32", heartbeat_seconds=0.05, io_workers=2, max_offline_steps=64, timeout_seconds=2)
    learner = DilocoLearner(settings, surface=TensorSurface.from_module(_model()), identity={"objective": "toy"},
                            output=tmp_path / "w0", transport=Transport(url, TOKEN, 2))
    try:
        learner.start()
        count = len(learner.fragments)
        assert count > 4 * settings.io_workers
        # Outage: the learner keeps taking updates and counts them offline instead of dying.
        stop.set()
        thread.join(timeout=10)
        time.sleep(0.3)
        for _ in range(5):
            learner.after_step(tokens=1)
            time.sleep(0.05)
        assert learner.offline_steps >= 1
        stop, thread = _serve(root, port)
        # A peer merged every fragment during the outage: the learner is behind everywhere at once.
        peer, session = Transport(url, TOKEN, 5), "00000000-0000-0000-0000-000000000001"
        peer.json("POST", "/v1/lease", {"worker": 1, "session": session})
        for fragment in learner.fragments:
            update = tmp_path / f"{fragment.index}.safetensors"
            save_tensors(update, {"delta": torch.full((fragment.length,), 0.01)})
            peer.request("POST", f"/v1/updates/{fragment.index}", file=update, headers={
                "X-Worker": "1", "X-Session": session, "X-Sequence": "1", "X-Base-Version": "0", "X-Steps": "1",
                "X-Tokens": "1", "Content-Type": "application/octet-stream"})
        deadline = time.time() + 30
        while time.time() < deadline and not all(v >= 1 for v in learner.versions):
            learner.after_step(tokens=1)
            assert len(learner.downloads) <= settings.io_workers
            time.sleep(0.02)
        assert all(v >= 1 for v in learner.versions)
        assert learner.peak_downloads <= settings.io_workers and learner.offline_steps == 0
    finally:
        learner.close()
        stop.set()
        thread.join(timeout=5)
