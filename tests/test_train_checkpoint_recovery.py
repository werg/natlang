import random
import signal

import torch

from scripts.train_lora import (capture_rng_state, install_stop_handlers,
                                order_training_pairs, recover_checkpoint_directory,
                                restore_rng_state)


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
