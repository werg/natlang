import copy
import json
import random

import pytest
import torch

from natlang_neuralese.common import profiling
from natlang_neuralese.train import native_state_sft as native
from natlang_neuralese.train.optim import PortMuonAdamW
from scripts.neuralese_data.profile_native_training import profile_native_updates


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.fixture
def task(monkeypatch):
    # Genuine shared complete-update/optimizer path, with a small CPU task in
    # place of the large model's packed graph loss. Random inputs test RNG too.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    def loss(model, batch, **kwargs):
        x = torch.rand(3, 4)
        target = torch.rand(3, 2)
        return (model(x) - target).square().mean(), []
    monkeypatch.setattr(native, 'packed_graph_loss', loss)
    def run(context=None):
        torch.manual_seed(73)
        random.seed(19)
        model = torch.nn.Linear(4, 2)
        named = list(model.named_parameters())
        optimizer = PortMuonAdamW(named, lr=.001, vocab_size=100)
        losses = []
        def updates():
            for _ in range(6):
                value, _ = native.native_training_update(model, optimizer, [])
                losses.append(value.detach().clone())
        if context is None:
            updates()
        else:
            with context as state:
                updates()
            assert state['restored_original_functions']
        return {'parameters': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()),
                'rng': torch.get_rng_state().clone(), 'python_rng': random.getstate(),
                'losses': losses}
    return run


def test_real_cpu_profile_preserves_all_updates_optimizer_and_rng(task, tmp_path):
    expected = task()
    originals = (native.native_training_update, native.packed_graph_loss,
                 torch.autograd.backward, PortMuonAdamW.step)
    actual = task(profile_native_updates(tmp_path, first=2, last=3))
    assert_tree_equal(actual, expected)
    assert originals == (native.native_training_update, native.packed_graph_loss,
                         torch.autograd.backward, PortMuonAdamW.step)
    assert not profiling.ENABLED
    report = json.loads((tmp_path / 'profile-2-3.json').read_text())
    assert set(report['updates']) == {'2', '3'} and report['requested'] == [2, 3]
    assert (tmp_path / 'profile-2-3.trace.json.gz').is_file()
    observer = json.loads((tmp_path / 'native-profile-observer.json').read_text())
    assert observer['completed_updates'] == 3 and observer['diagnostic_errors'] == []


def test_diagnostic_failure_disables_observation_and_keeps_training_exact(task, tmp_path):
    class Broken:
        done = False
        def __init__(self, *args, **kwargs):
            pass
        def begin(self, update):
            raise RuntimeError('fixture profiler unavailable')
        def close(self, update):
            pass
    expected = task()
    actual = task(profile_native_updates(tmp_path, profiler_factory=Broken))
    assert_tree_equal(actual, expected)
    assert not profiling.ENABLED
    assert 'RuntimeError' in json.loads((tmp_path / 'native-profile-observer.json').read_text())['diagnostic_errors']


def test_training_error_propagates_and_restores_every_observer(task, tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise FloatingPointError('real failed update')
    monkeypatch.setattr(native, 'native_training_update', failed)
    before = (native.packed_graph_loss, torch.autograd.backward, PortMuonAdamW.step)
    with pytest.raises(FloatingPointError, match='real failed update'):
        task(profile_native_updates(tmp_path, first=1, last=2))
    assert native.native_training_update is failed
    assert before == (native.packed_graph_loss, torch.autograd.backward, PortMuonAdamW.step)
    observer = json.loads((tmp_path / 'native-profile-observer.json').read_text())
    assert observer['completed_updates'] == 0 and observer['attempted_updates'] == 1
    assert not profiling.ENABLED


def test_cli_forwards_exact_trainer_arguments_to_original_shared_entry(tmp_path, monkeypatch):
    from scripts.neuralese_data.profile_native_training import main
    captured = []
    monkeypatch.setattr(native, 'main', lambda argv: captured.append(argv) or 17)
    arguments = ['--out', str(tmp_path / 'training'), '--steps', '1000', '--task-eval=']
    assert main(['--profile-out', str(tmp_path / 'profile'), '--', *arguments]) == 17
    assert captured == [arguments]


def test_wrong_bound_observer_cannot_start_training(tmp_path, monkeypatch):
    from scripts.neuralese_data.profile_native_training import main
    def forbidden(*args):
        raise AssertionError('wrong installed observer must be rejected before training')
    monkeypatch.setattr(native, 'main', forbidden)
    with pytest.raises(SystemExit):
        main(['--profile-out', str(tmp_path / 'profile'), '--expected-observer-sha256',
              '0' * 64, '--', '--out', str(tmp_path / 'training')])
