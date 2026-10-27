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
        learner.start()
        for name, param in model.named_parameters():
            assert torch.equal(param, dict(seed.named_parameters())[name])
        learners.append((model, torch.optim.AdamW(model.parameters(), lr=1e-2), learner))
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
