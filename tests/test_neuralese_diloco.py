"""Shared DiLoCo: protocol numerics, durable coordinator and HTTP learners on CPU."""

import socket
import threading
import time
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train.diloco.coordinator import Conflict, Coordinator, global_masters, initialize_coordinator, serve_coordinator
from natlang_neuralese.train.diloco.protocol import (
    TensorSurface,
    merge_pseudogradients,
    outer_nesterov,
    pack_fragment,
    plan_fragments,
    save_tensors,
)
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport

TOKEN = "t" * 32


def _model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def test_fragments_cover_every_parameter_exactly_once():
    masters = {"a": torch.randn(1000), "b": torch.randn(7, 300), "c": torch.randn(3)}
    fragments = plan_fragments(masters, target_bytes=4096)
    covered = {name: torch.zeros(p.numel(), dtype=torch.int32) for name, p in masters.items()}
    for fragment in fragments:
        assert fragment.length <= 1024
        for part in fragment.slices:
            covered[part.name][part.offset : part.offset + part.length] += 1
    assert all(bool((c == 1).all()) for c in covered.values())


def test_merge_weights_tokens_and_staleness_and_nesterov():
    merged = merge_pseudogradients([torch.ones(4), 3 * torch.ones(4)], [10, 10], [1, 1], [0, 1])
    # weights 100 and 50: (1*100 + 3*50) / 150
    assert torch.allclose(merged, torch.full((4,), 250 / 150))
    theta, velocity = outer_nesterov(torch.zeros(2), torch.zeros(2), torch.ones(2), lr=0.5, momentum=0.9)
    assert torch.allclose(velocity, torch.ones(2)) and torch.allclose(theta, torch.full((2,), -0.95))


def _init(root, model, *, workers=2, quorum=2, **kwargs):
    surface = TensorSurface.from_module(model)
    return initialize_coordinator(
        root,
        run_id="unit",
        masters=surface.masters,
        identity={"objective": "toy"},
        workers=workers,
        min_quorum=quorum,
        fragment_bytes=1024,
        grace_seconds=0.0,
        **kwargs,
    )


def test_coordinator_dedups_replayed_sequence_and_rejects_stale(tmp_path):
    model = _model()
    _init(tmp_path / "c", model, workers=1, quorum=1, max_staleness=0)
    coordinator = Coordinator(tmp_path / "c")
    session = "0" * 8 + "-0000-0000-0000-" + "0" * 12
    coordinator.lease(0, session)
    length = coordinator.fragments[0].length
    update = tmp_path / "u.safetensors"
    save_tensors(update, {"delta": torch.full((length,), 0.01)})
    assert coordinator.submit(0, 0, session, 1, 0, 4, 100, update)["version"] == 1
    # The same sequence again (a restored learner resending) is not applied twice.
    assert coordinator.submit(0, 0, session, 1, 0, 4, 100, update)["status"] == "already_merged"
    with pytest.raises(Conflict):  # base version 0 is now older than max_staleness allows
        coordinator.submit(0, 0, session, 2, 0, 4, 100, update)
    # A restart reloads the committed ledger.
    assert Coordinator(tmp_path / "c")._sequences(0)[0] == 1


def test_heartbeat_reports_a_merge_at_once(tmp_path):
    # The learner whose upload did not complete the quorum learns of the merge from heartbeat versions; reporting
    # them only on the 1 s reconcile tick kept it training against its upload point, unable to send its next round.
    model = _model()
    _init(tmp_path / "c", model)
    coordinator = Coordinator(tmp_path / "c")
    sessions = ["0" * 8 + f"-0000-0000-0000-{worker:012d}" for worker in range(2)]
    length = coordinator.fragments[0].length
    update = tmp_path / "u.safetensors"
    save_tensors(update, {"delta": torch.full((length,), 0.01)})
    for worker, session in enumerate(sessions):
        coordinator.lease(worker, session)
        coordinator.submit(0, worker, session, 1, 0, 4, 100, update)
    assert coordinator.heartbeat(0, sessions[0], 4, 100)["versions"][0] == 1


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_two_http_learners_train_and_merge(tmp_path):
    torch.manual_seed(1)
    inputs = torch.randn(256, 8)
    targets = inputs @ torch.linspace(-1, 1, 8).unsqueeze(1)
    seed = _model()
    # Plain averaging (lr 1, no momentum) isolates the protocol from outer-optimizer tuning.
    _init(tmp_path / "coord", seed, outer_lr=1.0, outer_momentum=0.0)
    port = _free_port()
    stop = threading.Event()
    server = threading.Thread(target=serve_coordinator, args=(tmp_path / "coord", "127.0.0.1", port, TOKEN), kwargs={"stop": stop}, daemon=True)
    server.start()
    deadline = time.time() + 10
    while True:
        try:
            Transport(f"http://127.0.0.1:{port}", TOKEN, 5).json("GET", "/v1/status")
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.05)

    def loss_of(model):
        with torch.no_grad():
            return float(torch.nn.functional.mse_loss(model(inputs), targets))

    initial = loss_of(seed)
    learners = []
    for worker in range(2):
        model = _model()  # every learner loads the same parent
        settings = DilocoSettings(coordinator_url=f"http://127.0.0.1:{port}", run_id="unit", workers=2, worker_id=worker, local_steps=8, wire_dtype="fp32", heartbeat_seconds=0.05)
        surface = TensorSurface.from_module(model)
        learner = DilocoLearner(settings, surface=surface, identity={"objective": "toy"}, output=tmp_path / f"w{worker}", transport=Transport(settings.coordinator_url, TOKEN, 5))
        learners.append((model, torch.optim.AdamW(model.parameters(), lr=1e-2), learner))
    # Start barrier: learner 0 boots first and waits until its peer has registered.
    learners[0][2].start()  # Returns before quorum so qualification can run.
    starter = threading.Thread(target=learners[0][2].await_training_start)
    starter.start()
    time.sleep(0.5)
    assert starter.is_alive()
    learners[1][2].start()
    starter.join(timeout=30)
    assert not starter.is_alive() and learners[0][2].start_wait["all"] and learners[0][2].start_wait["seconds"] >= 0.4
    for model, _, _ in learners:
        for name, param in model.named_parameters():
            assert torch.equal(param, dict(seed.named_parameters())[name])
    shard = [i for i in range(len(inputs))]
    for step in range(1, 81):
        for worker, (model, optimizer, learner) in enumerate(learners):
            rows = shard[worker::2]
            batch = torch.tensor(rows[(step * 16) % len(rows) :][:16] or rows[:16])
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(inputs[batch]), targets[batch]).backward()
            optimizer.step()
            learner.after_step(tokens=len(batch))
        time.sleep(0.02)
    # Learners finish concurrently, as separate sandboxes do: each final merge needs its peer.
    finished = []
    threads = [threading.Thread(target=lambda item=item: finished.append(item[2].finish())) for item in learners]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert finished == [True, True]
    status = Transport(f"http://127.0.0.1:{port}", TOKEN, 5).json("GET", "/v1/status")
    assert set(status["registered"]) == {"0", "1"} and status["recent_rounds"]
    assert all({"steps", "tokens", "staleness"} <= set(w) for r in status["recent_rounds"] for w in r["windows"].values())
    state = learners[0][2].state_dict()
    for _, _, learner in learners:
        learner.close()
    stop.set()
    server.join(timeout=10)
    masters, info = global_masters(tmp_path / "coord")
    assert min(info["versions"]) >= 3
    final = _model()
    final.load_state_dict({name: value for name, value in masters.items()})
    # Reference: one plain learner taking the same number of updates over all rows.
    reference = _model()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2)
    for step in range(1, 81):
        batch = torch.tensor(shard[(step * 16) % len(shard) :][:16] or shard[:16])
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(reference(inputs[batch]), targets[batch]).backward()
        optimizer.step()
    # Merge timing is asynchronous, so bounds leave room for run-to-run variation.
    assert loss_of(final) < 0.05 * initial
    assert loss_of(final) < 3 * loss_of(reference)
    assert state["worker"] == 0 and not state["pending"]


def test_learner_rejects_a_different_objective(tmp_path):
    model = _model()
    _init(tmp_path / "coord", model)
    port = _free_port()
    stop = threading.Event()
    threading.Thread(target=serve_coordinator, args=(tmp_path / "coord", "127.0.0.1", port, TOKEN), kwargs={"stop": stop}, daemon=True).start()
    time.sleep(0.5)
    settings = DilocoSettings(coordinator_url=f"http://127.0.0.1:{port}", run_id="unit", workers=2, worker_id=0)
    with pytest.raises(ValueError, match="identity"):
        DilocoLearner(settings, surface=TensorSurface.from_module(model), identity={"objective": "other"}, output=tmp_path / "w", transport=Transport(settings.coordinator_url, TOKEN, 5))
    stop.set()


def test_learner_shards_partition_the_exposure_schedule():
    from natlang_neuralese.train.native_state_sft import pending_exposure_batches

    entries = [{"id": f"row-{i}", "forward_tokens": 10} for i in range(40)]
    seen = []
    for worker in range(3):
        quotas, batches = pending_exposure_batches(entries, {}, epochs=1, batch_graphs=4, shard=(worker, 3))
        rows = [entries[i]["id"] for _, indices, _ in batches for i in indices]
        assert sorted(rows) == sorted(quotas)
        seen += rows
    assert sorted(seen) == sorted(entry["id"] for entry in entries)


def test_export_merges_disjoint_learner_exposure_onto_the_parent():
    from natlang_neuralese.train.native_diloco import merged_exposure, trainer_argv

    parent = {"a": 1, "b": 1, "c": 0}
    learners = [{"a": 2, "b": 1, "c": 0}, {"a": 1, "b": 1, "c": 1, "d": 1}]
    assert merged_exposure(parent, learners) == {"a": 2, "b": 1, "c": 1, "d": 1}
    with pytest.raises(ValueError):
        merged_exposure(parent, [{"a": 0}])
    argv = ["--out", "/work/out", "--diloco", '{"coordinator_url":"https://x"}', "--diloco={}", "--epochs", "1"]
    assert trainer_argv(argv) == ["--out", "/work/out", "--epochs", "1"]


def test_outer_schedule_switches_at_its_round_with_a_fresh_momentum_buffer(tmp_path):
    import json
    from natlang_neuralese.train.diloco.coordinator import _metadata, validate_outer_schedule

    with pytest.raises(ValueError):
        validate_outer_schedule([{"from_round": 1, "lr": 1.0, "momentum": 0.0}])
    schedule = [{"from_round": 0, "lr": 1.0, "momentum": 0.0}, {"from_round": 2, "lr": 0.5, "momentum": 0.5}]
    manifest = _init(tmp_path / "c", _model(), workers=1, quorum=1, max_staleness=4, outer_schedule=schedule)
    assert manifest["outer_schedule"] == schedule
    session = "0" * 8 + "-0000-0000-0000-" + "0" * 12
    coordinator = Coordinator(tmp_path / "c")
    coordinator.lease(0, session)
    fragment = coordinator.fragments[0]
    start = pack_fragment(global_masters(tmp_path / "c")[0], fragment)
    update = tmp_path / "u.safetensors"
    save_tensors(update, {"delta": torch.full((fragment.length,), 0.01)})
    # lr 1, momentum 0 twice; then a fresh buffer: v = d, step 0.75 d; v = 1.5 d, step 0.875 d.
    expected = [0.01, 0.02, 0.0275, 0.03625]
    for sequence in range(1, 5):
        if sequence == 3:  # a restart between rounds keeps the schedule (it lives in the manifest)
            coordinator = Coordinator(tmp_path / "c")
        coordinator.submit(0, 0, session, sequence, sequence - 1, 4, 100, update)
        assert json.loads(_metadata(coordinator._path(0))["outer"])["entry"] == (0 if sequence <= 2 else 1)
        now = pack_fragment(global_masters(tmp_path / "c")[0], fragment)
        assert torch.allclose(start - now, torch.full((fragment.length,), expected[sequence - 1]), atol=1e-6)
    assert schedule[1] in coordinator.status()["outer_next"]


def test_export_global_writes_fp32_masters_at_parameter_dtype(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import natlang_neuralese.train.checkpoint_policy as checkpoint_policy
    import natlang_neuralese.train.native_diloco as native_diloco
    import natlang_neuralese.train.weight_ema as weight_ema
    from natlang_neuralese.common.hashing import sha256_file_hex_cached

    surface = TensorSurface.from_module(_model().to(torch.bfloat16))
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    initialize_coordinator(tmp_path / "c", run_id="unit", masters=surface.masters,
                           identity={"parent_checkpoint_sha256": sha256_file_hex_cached(parent)}, workers=1,
                           min_quorum=1, fragment_bytes=1024, ema={"decay": 0.5, "warmup": 0})
    masters = global_masters(tmp_path / "c")[0]
    assert all(value.dtype == torch.float32 for value in masters.values())  # FP32 globals from a BF16 parent
    target = _model(seed=1).to(torch.bfloat16)
    monkeypatch.setattr(native_diloco, "_trainable", lambda path, base: (target, list(target.named_parameters())))
    monkeypatch.setattr(checkpoint_policy.CheckpointPolicy, "load", staticmethod(
        lambda path: {"schema": "s", "model_config": {}, "training": {"global_step": 7}}))
    written = {}

    def write(model, config, schema, step, export, out):
        written[out.name] = ({n: p.detach().clone() for n, p in model.named_parameters()}, step, export)

    monkeypatch.setattr(weight_ema, "write_weights_checkpoint", write)
    native_diloco.export_global(SimpleNamespace(root=tmp_path / "c", parent=parent, base=None,
                                                out=tmp_path / "g", ema_out=tmp_path / "e"))
    weights, step, export = written["g"]
    assert step == 7 and set(export["fragment_versions"]) == {0}
    assert all(torch.equal(weights[n], masters[n].to(torch.bfloat16)) for n in weights)
    assert written["e"][2]["source"] == "diloco-coordinator-ema"


def test_export_global_from_a_running_coordinator_equals_the_volume_read(tmp_path, monkeypatch):
    """Remote mode (an evaluation sandbox): theta and the EMA over the API equal the coordinator-volume read."""
    from types import SimpleNamespace

    import natlang_neuralese.train.checkpoint_policy as checkpoint_policy
    import natlang_neuralese.train.native_diloco as native_diloco
    import natlang_neuralese.train.weight_ema as weight_ema
    from natlang_neuralese.common.hashing import sha256_file_hex_cached

    surface = TensorSurface.from_module(_model().to(torch.bfloat16))
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    initialize_coordinator(tmp_path / "c", run_id="unit", masters=surface.masters,
                           identity={"parent_checkpoint_sha256": sha256_file_hex_cached(parent)}, workers=1,
                           min_quorum=1, fragment_bytes=1024, ema={"decay": 0.5, "warmup": 0})
    port = _free_port()
    stop = threading.Event()
    threading.Thread(target=serve_coordinator, args=(tmp_path / "c", "127.0.0.1", port, TOKEN),
                     kwargs={"stop": stop}, daemon=True).start()
    deadline = time.time() + 10
    while True:
        try:
            Transport(f"http://127.0.0.1:{port}", TOKEN, 5).json("GET", "/v1/status")
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.05)
    target = _model(seed=1).to(torch.bfloat16)
    monkeypatch.setattr(native_diloco, "_trainable", lambda path, base: (target, list(target.named_parameters())))
    monkeypatch.setattr(checkpoint_policy.CheckpointPolicy, "load", staticmethod(
        lambda path: {"schema": "s", "model_config": {}, "training": {"global_step": 7}}))
    written = {}

    def write(model, config, schema, step, export, out):
        written[out.name] = ({n: p.detach().clone() for n, p in model.named_parameters()}, export)

    monkeypatch.setattr(weight_ema, "write_weights_checkpoint", write)
    monkeypatch.setenv("NATLANG_DILOCO_TOKEN", TOKEN)
    try:
        native_diloco.export_global(SimpleNamespace(root=tmp_path / "c", parent=parent, base=None,
                                                    out=tmp_path / "local", ema_out=tmp_path / "local-ema"))
        native_diloco.export_global(SimpleNamespace(root=None, coordinator_url=f"http://127.0.0.1:{port}",
                                                    parent=parent, base=None, timeout=5, workdir=tmp_path / "dl",
                                                    out=tmp_path / "remote", ema_out=tmp_path / "remote-ema"))
    finally:
        stop.set()
    for local, remote in (("local", "remote"), ("local-ema", "remote-ema")):
        (a, ea), (b, eb) = written[local], written[remote]
        assert all(torch.equal(a[n], b[n]) for n in a)
        assert ea["fragment_versions"] == eb["fragment_versions"] and ea["source"] == eb["source"]
    assert not list((tmp_path / "dl").iterdir())  # downloaded fragments are not kept


def test_file_store_streams_resumes_verifies_and_serves_files(tmp_path):
    import hashlib
    import os
    import tracemalloc

    from natlang_neuralese.train.diloco.worker import RemoteError
    from natlang_neuralese.train.native_diloco import uploaded

    _init(tmp_path / "coord", _model())
    port = _free_port()
    stop = threading.Event()
    threading.Thread(target=serve_coordinator, args=(tmp_path / "coord", "127.0.0.1", port, TOKEN),
                     kwargs={"stop": stop}, daemon=True).start()
    transport = Transport(f"http://127.0.0.1:{port}", TOKEN, 60)
    deadline = time.time() + 10
    while True:
        try:
            transport.json("GET", "/v1/status")
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(0.05)
    try:
        big = tmp_path / "learner-0.pt"
        with big.open("wb") as handle:  # 256 MiB, written in 4 MiB blocks
            for index in range(64):
                handle.write(hashlib.sha256(str(index).encode()).digest() * ((4 << 20) // 32))
        tracemalloc.start()
        # 100 MiB chunks: the upload continues the stored partial file twice before committing.
        result = transport.upload_file(big, "learner-0.pt", chunk_bytes=100 << 20)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert result["status"] == "committed" and result["bytes"] == big.stat().st_size
        assert peak < 64 << 20, peak  # streamed both ways: no buffer proportional to the 256 MiB file
        assert transport.upload_file(big, "learner-0.pt")["status"] == "present"
        stored = tmp_path / "coord" / "uploads" / "learner-0.pt"
        expected = hashlib.sha256(big.read_bytes()).hexdigest()
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == expected
        back = transport.download_file("learner-0.pt", tmp_path / "back" / "learner-0.pt")
        assert back["sha256"] == expected and back["bytes"] == big.stat().st_size
        # A wrong declared sha256 is refused and nothing is committed under that name.
        small = tmp_path / "small.bin"
        small.write_bytes(os.urandom(1000))
        with pytest.raises(RemoteError) as refused:
            with small.open("rb") as handle:
                transport.request("PUT", "/v1/files/other.bin", file=handle, headers={
                    "Content-Length": "1000", "X-Sha256": "0" * 64, "X-Offset": "0", "X-Total-Bytes": "1000"})
        assert refused.value.status == 400 and not (tmp_path / "coord" / "uploads" / "other.bin").exists()
        with pytest.raises(ValueError):  # a committed name is immutable
            transport.upload_file(small, "learner-0.pt")
        with pytest.raises(RemoteError):
            transport.download_file("missing.pt", tmp_path / "missing.pt")
        assert uploaded(tmp_path / "coord", "learner-0.pt") == stored
    finally:
        stop.set()


def test_launcher_creates_learners_concurrently_and_reports_failures(monkeypatch):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "coreweave"))
    import diloco as launcher

    active, peak, lock = [0], [0], threading.Lock()

    def fake_launch(job, dry_run, wait, price, bundle_file, secrets=None):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.3)
        with lock:
            active[0] -= 1
        if job["id"].endswith("-2") and fake_launch.fail:
            raise launcher.cwctl.ControllerError("capacity")
        return {"sandbox_id": job["id"], "status": "running", "push": {}}

    monkeypatch.setattr(launcher, "learner_job", lambda run, worker, url: {"id": f"learner-{worker}"})
    monkeypatch.setattr(launcher.cwctl, "launch", fake_launch)
    run = {"settings": {"workers": 4}, "learners": {"bundle_file": "b.json"}}
    fake_launch.fail = False
    started = []
    assert launcher.launch_learners(run, "https://c", "t" * 32, b"g", started) == [f"learner-{w}" for w in range(4)]
    assert peak[0] >= 2 and sorted(started) == [f"learner-{w}" for w in range(4)]
    fake_launch.fail = True
    started = []
    with pytest.raises(launcher.cwctl.ControllerError):
        launcher.launch_learners(run, "https://c", "t" * 32, b"g", started)
    assert len(started) == 3  # the caller stops every sandbox that did start


def test_export_calibration_custody_requires_new_merged_weight_window():
    from natlang_neuralese.train.native_diloco import _retire_learner_calibration
    first = {'schema': 'calibration', 'amax': {'x': 1}, 'window': {'batches': 2}}
    second = {'schema': 'calibration', 'amax': {'x': 3}}
    learners = [{'diloco': {'worker': 0}, 'quantization': first},
                {'diloco': {'worker': 1}, 'quantization': second}]
    training = {'quantization': first, 'options': {'quantization': {'sha256': 'binding'}}}
    custody = _retire_learner_calibration(training, learners)
    assert 'quantization' not in training
    assert training['options']['quantization']['sha256'] == 'binding'
    assert custody['action']=='recalibrate-merged-weights' and custody['learners']=={'0':first,'1':second}
    assert custody['learner_declarations']=={'0':{'declaration':None,'history':[]},'1':{'declaration':None,'history':[]}}


def test_export_recalibrates_merged_weights_and_preserves_worker_custody(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.train import native_diloco, native_continuation, checkpoint_policy
    from natlang_neuralese.model import native_model

    parent = {'global_step': 4, 'step': 4, 'exposure_counts': {'a': 1}}
    states = {
        'parent': {'training': parent},
        'w0': {'schema': 's', 'training': {'global_step': 5, 'step': 5, 'exposure_counts': {'a': 2},
                'diloco': {'worker': 0, 'fingerprint': 'f'}, 'quantization': {'amax': {'x': 1}},
                'options': {'quantization': {'sha256': 'q'}}}},
        'w1': {'training': {'global_step': 5, 'step': 5, 'exposure_counts': {'a': 1, 'b': 1},
                'diloco': {'worker': 1, 'fingerprint': 'f'}, 'quantization': {'window': {'batches': 1}}}},
    }
    import copy
    monkeypatch.setattr(checkpoint_policy.CheckpointPolicy, 'load', staticmethod(lambda p: copy.deepcopy(states[str(p)])))
    saved = {}
    monkeypatch.setattr(checkpoint_policy.CheckpointPolicy, 'save', lambda self, state: saved.update(state))
    monkeypatch.setattr(native_diloco, 'read_manifest', lambda root: {'identity': {'parent_checkpoint_sha256': 'p'},
        'workers': 2, 'fingerprint': 'f', 'run_id': 'r'})
    monkeypatch.setattr(native_diloco, 'sha256_file_hex_cached', lambda p: 'p')
    target = _model().to(torch.bfloat16)
    named = list(target.named_parameters())
    monkeypatch.setattr(native_diloco, '_trainable', lambda *args: (target, named))
    monkeypatch.setattr(native_diloco, 'global_masters', lambda root: ({n: torch.zeros_like(p) for n, p in named}, {'versions': []}))
    monkeypatch.setattr(native_continuation, 'collect_sft_progress', lambda training: {})
    monkeypatch.setattr(native_model, 'model_state', lambda model: {})
    args = SimpleNamespace(root=tmp_path, parent='parent', learner_checkpoint=['w0', 'w1'], learner_upload=[],
                           base=None, out=tmp_path / 'out', ema_out='off')
    native_diloco.export(args)
    training = saved['training']
    assert training['global_step'] == 6 and training['exposure_counts'] == {'a': 2, 'b': 1}
    assert 'quantization' not in training
    assert training['options']['quantization']['sha256'] == 'q'
    custody = training['options']['diloco_export']['quantization_calibration']
    assert custody['learners'] == {'0': {'amax': {'x': 1}}, '1': {'window': {'batches': 1}}}
    assert custody['action'] == 'recalibrate-merged-weights'


def test_four_unequal_epochs_drain_every_partial_before_global_export(tmp_path):
    import json,time
    from natlang_neuralese.train.diloco.coordinator import global_ema,_metadata
    seed=_model();root=tmp_path/'coord';surface=TensorSurface.from_module(seed)
    manifest=initialize_coordinator(root,run_id='full-epoch',masters=surface.masters,identity={'parent':'exact'},
        workers=4,min_quorum=3,fragment_bytes=1024,grace_seconds=0,outer_lr=1.0,outer_momentum=0,
        ema={'decay':.88,'warmup':0})
    port=_free_port();token=TOKEN;url=f'http://127.0.0.1:{port}';stop=threading.Event()
    server=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,token),kwargs={'stop':stop},daemon=True);server.start()
    transport=Transport(url,token,5)
    for _ in range(100):
        try:transport.json('GET','/v1/status');break
        except OSError:time.sleep(.02)
    learners=[]
    for worker in range(4):
        model=_model();model.load_state_dict(seed.state_dict());optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        settings=DilocoSettings(run_id='full-epoch',workers=4,worker_id=worker,coordinator_url=url,local_steps=128,
            wire_dtype='fp32',allow_private_http=True,start_grace_seconds=0,heartbeat_seconds=.05,timeout_seconds=10)
        learner=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity={'parent':'exact'},
            output=tmp_path/f'w{worker}',transport=Transport(url,token,5))
        learners.append((model,optimizer,learner))
    starters=[threading.Thread(target=item[2].start) for item in learners]
    for t in starters:t.start()
    for t in starters:t.join(10);assert not t.is_alive()
    failures=[];completed={};sequences={}
    def train(i,steps):
        model,optimizer,learner=learners[i]
        try:
            # Earlier workers finish while peers are still earning updates.
            if i>=2:time.sleep(.4)
            for _ in range(steps):
                optimizer.zero_grad();model(torch.ones(1,8)).sum().backward();optimizer.step();learner.after_step(tokens=1)
            learner.declare_completion(drained=False)
            assert learner.finish()
            learner.declare_completion(drained=True)
            completed[i]=steps;sequences[i]=list(learner.sequences)
        except BaseException as error:failures.append(error)
    threads=[threading.Thread(target=train,args=(i,n)) for i,n in enumerate([128,129,130,131])]
    try:
        for t in threads:t.start()
        for t in threads:t.join(20);assert not t.is_alive()
        assert not failures and completed=={0:128,1:129,2:130,3:131}
        status=transport.json('GET','/v1/status')
        assert len(status['completion'])==4 and all(v['drained'] for v in status['completion'].values())
        assert all(count==0 for count in status['pending'].values())
        for fragment in manifest['fragments']:
            index=fragment['index'];ledger=json.loads(_metadata(root/'fragments'/f'{index:05d}.safetensors')['last_seq'])
            assert all(int(ledger[str(i)])==sequences[i][index] for i in range(4))
        masters,global_info=global_masters(root);average,ema_info=global_ema(root)
        assert global_info['versions']==ema_info['versions']==status['versions']
        assert any(not torch.equal(masters[name],surface.masters[name]) for name in masters)
        assert set(average)==set(masters) and ema_info['ema']=={'decay':.88,'warmup':0}
    finally:
        for _,_,learner in learners:learner.close()
        stop.set();server.join(timeout=3)
