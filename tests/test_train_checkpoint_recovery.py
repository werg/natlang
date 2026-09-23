import json
import random
import signal

import pytest
import torch

from scripts.train_lora import (capture_rng_state, install_stop_handlers,
                                order_training_pairs, recover_checkpoint_directory,
                                restore_rng_state, write_checkpoint_directory)


def test_checkpoint_recovery_restores_old_complete_directory(tmp_path):
    old = tmp_path / 'checkpoint.old'
    old.mkdir()
    (old / 'state.json').write_text('{"step": 7}')
    temporary = tmp_path / 'checkpoint.tmp'
    temporary.mkdir()
    (temporary / 'partial').write_text('incomplete')

    recover_checkpoint_directory(tmp_path)

    assert (tmp_path / 'checkpoint' / 'state.json').read_text() == '{"step": 7}'
    assert not old.exists() and not temporary.exists()


def test_checkpoint_recovery_keeps_current_and_removes_stale_swap_dirs(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'state.json').write_text('{"step": 8}')
    for name in ('checkpoint.old', 'checkpoint.tmp'):
        (tmp_path / name).mkdir()

    recover_checkpoint_directory(tmp_path)

    assert (checkpoint / 'state.json').read_text() == '{"step": 8}'
    assert not (tmp_path / 'checkpoint.old').exists()
    assert not (tmp_path / 'checkpoint.tmp').exists()


def test_failed_checkpoint_write_preserves_previous_complete_boundary(tmp_path):
    prior = tmp_path / "checkpoint"
    prior.mkdir()
    (prior / "state.json").write_text('{"step": 4}')

    def fail_during_weights(_weights):
        raise RuntimeError("simulated interrupted serialization")

    with pytest.raises(RuntimeError, match="simulated interrupted"):
        write_checkpoint_directory(tmp_path, fail_during_weights, {}, {}, {}, {"step": 5})

    assert json.loads((prior / "state.json").read_text()) == {"step": 4}
    assert not (tmp_path / "checkpoint.old").exists()
    assert not (tmp_path / "checkpoint.tmp").exists()


def test_deferred_stop_handler_only_sets_flag(monkeypatch):
    callbacks = {}
    monkeypatch.setattr(signal, 'signal', lambda sig, callback: callbacks.__setitem__(sig, callback))
    stop = install_stop_handlers()

    callbacks[signal.SIGTERM](signal.SIGTERM, None)

    assert stop['now'] is True


def test_rng_state_restores_cpu_torch_and_python(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(23)
    random.seed(29)
    state = capture_rng_state()
    expected_torch = torch.rand(3)
    expected_python = random.random()

    torch.rand(8)
    random.random()
    restore_rng_state(state)

    assert torch.equal(torch.rand(3), expected_torch)
    assert random.random() == expected_python


def test_source_order_preserves_input_offsets():
    rows = [{'offset': 20}, {'offset': 4}, {'offset': 13}]
    ordered = order_training_pairs(rows, 'source')
    assert [row['offset'] for row in ordered] == [4, 13, 20]


def test_repeated_stop_signals_remain_deferred(monkeypatch):
    callbacks = {}
    monkeypatch.setattr(signal, 'signal', lambda sig, callback: callbacks.__setitem__(sig, callback))
    stop = install_stop_handlers()
    assert stop == {'now': False}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTERM, signal.SIGINT):
        callbacks[sig](sig, None)
        assert stop == {'now': True}


def test_legacy_tensor_rng_checkpoint_restores_cpu_sequence():
    original = torch.get_rng_state()
    try:
        torch.manual_seed(41)
        saved = torch.get_rng_state()
        expected = torch.rand(4)
        torch.rand(9)
        restore_rng_state(saved)
        assert torch.equal(torch.rand(4), expected)
    finally:
        torch.set_rng_state(original)


def test_cuda_rng_states_round_trip_without_real_gpu(monkeypatch):
    original_torch, original_python = torch.get_rng_state(), random.getstate()
    gpu_states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    restored = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: gpu_states)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', restored.append)
    try:
        saved = capture_rng_state()
        restore_rng_state(saved)
        assert len(restored) == 1
        assert all(torch.equal(actual, expected) for actual, expected in zip(restored[0], gpu_states))
        assert len(restored[0]) == len(gpu_states)
        # A GPU-bearing checkpoint remains loadable for CPU-side inspection.
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        restore_rng_state(saved)
        assert len(restored) == 1
    finally:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)


def _toy_components():
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(0.2), torch.nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    return model, optimizer, scheduler


def _toy_update(model, optimizer, scheduler, x, y):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()


def _write_toy_checkpoint(path, model, optimizer, scheduler, step):
    def save_model(weights):
        weights.mkdir()
        torch.save(model.state_dict(), weights / "model.pt")
    write_checkpoint_directory(path, save_model, optimizer.state_dict(), scheduler.state_dict(),
                               capture_rng_state(), {"step": step})


def test_real_adamw_checkpoint_reload_matches_uninterrupted_next_update(tmp_path):
    """Exercise production checkpoint directory and optimizer/scheduler/RNG serialization."""
    original_torch, original_python = torch.get_rng_state(), random.getstate()
    try:
        torch.manual_seed(173)
        random.seed(181)
        x, y = torch.randn(4, 3), torch.randn(4, 2)
        # Independent deterministic initialization.
        torch.manual_seed(191)
        baseline, baseline_opt, baseline_sched = _toy_components()

        # Produce the checkpoint after two actual updates.
        _toy_update(baseline, baseline_opt, baseline_sched, x, y)
        _toy_update(baseline, baseline_opt, baseline_sched, x, y)
        saved_dir = tmp_path / "out"
        _write_toy_checkpoint(saved_dir, baseline, baseline_opt, baseline_sched, 2)
        # Continue once to establish the expected subsequent update.
        _toy_update(baseline, baseline_opt, baseline_sched, x, y)
        expected_model = {k: v.clone() for k, v in baseline.state_dict().items()}
        expected_opt = baseline_opt.state_dict()
        expected_sched = baseline_sched.state_dict()

        resumed, resumed_opt, resumed_sched = _toy_components()
        resumed.load_state_dict(torch.load(saved_dir / "checkpoint/weights/model.pt", weights_only=True))
        resumed_opt.load_state_dict(torch.load(saved_dir / "checkpoint/optimizer.pt", weights_only=False))
        resumed_sched.load_state_dict(torch.load(saved_dir / "checkpoint/scheduler.pt", weights_only=False))
        restore_rng_state(torch.load(saved_dir / "checkpoint/rng.pt", weights_only=False))
        _toy_update(resumed, resumed_opt, resumed_sched, x, y)

        assert all(torch.equal(resumed.state_dict()[k], value) for k, value in expected_model.items())
        actual_opt = resumed_opt.state_dict()
        assert actual_opt["param_groups"] == expected_opt["param_groups"]
        assert actual_opt["state"].keys() == expected_opt["state"].keys()
        for param_id, expected_values in expected_opt["state"].items():
            for name, value in expected_values.items():
                actual_value = actual_opt["state"][param_id][name]
                assert torch.equal(actual_value, value) if isinstance(value, torch.Tensor) else actual_value == value
        assert resumed_sched.state_dict() == expected_sched
        assert json.loads((saved_dir / "checkpoint/state.json").read_text()) == {"step": 2}
    finally:
        torch.set_rng_state(original_torch)
        random.setstate(original_python)
