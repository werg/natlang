import copy
import hashlib
import json
import random

import pytest
import torch

from natlang_neuralese.train import native_state_sft as native
from scripts.neuralese_data.native_update_observer import observe_backbone_updates, _coordinates, _sample
from test_native_training_profile import assert_tree_equal


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_actual_both_ports_preserve_parameters_optimizer_rng_and_report_updates(tmp_path, family):
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from test_shared_state import tiny
    from test_native_state_sft import graph

    def run(observe):
        torch.manual_seed(73)
        random.seed(19)
        model = SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4)
        if family == 'mellum':
            # The tiny source loads dense BF16-policy experts without the
            # production loader's attestation; use the existing port-fixture law.
            model.port.hf.natlang_precision = 'bf16'
        model.port.checkpoint_layers = False
        optimizer, _ = native.make_native_optimizer(model, lr=.001, in_backward=True)
        example = parse_graph(graph('actual-update'), tokenizer_fingerprint='tokenizer-sha')
        losses = []
        def updates():
            for _ in range(4):
                loss, _ = native.native_training_update(model, optimizer, [example], checkpoint_layers=False)
                losses.append(loss.detach().clone())
        original = native.native_training_update
        if observe:
            with observe_backbone_updates(tmp_path, every=3, samples=7) as state:
                updates()
            assert state['restored_original_function'] and state['diagnostic_errors'] == []
        else:
            updates()
        assert native.native_training_update is original
        return {'parameters': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict()), 'losses': losses,
                'rng': torch.get_rng_state().clone(), 'python_rng': random.getstate()}

    assert_tree_equal(run(False), run(True))
    rows = [json.loads(line) for line in (tmp_path/'native-backbone-updates.jsonl').read_text().splitlines()]
    assert [row['relative_update'] for row in rows] == [1, 3]
    resources = [json.loads(line) for line in (tmp_path/'native-update-resources.jsonl').read_text().splitlines()]
    assert [row['relative_update'] for row in resources] == [1, 2, 3, 4]
    assert all(row['update_completed'] and row['host_wall_seconds'] >= 0 for row in resources)
    assert all(row['sampled_process_peak_bytes']['VmRSS_bytes'] >= row['before']['process']['VmRSS_bytes'] for row in resources)
    assert all('cuda' not in row['before'] for row in resources)
    for row in rows:
        assert row['parameters'] and all(p['name'].startswith('backbone.') for p in row['parameters'])
        assert any(p['sampled_update_l2'] > 0 and p['changed_sampled_coordinates'] > 0
                   for p in row['parameters'])
        assert all(p['sampled_coordinates'] <= 7 for p in row['parameters'])
        # The actual in-backward optimizer clears some gradients. Their true
        # displacement remains measurable and is never mislabeled a lost step.
        assert any(not p['gradient_present'] and p['sampled_update_l2'] > 0 for p in row['parameters'])


def test_noncontiguous_parameter_is_sampled_in_logical_order_without_flat_copy():
    value = torch.arange(24.).reshape(4, 6).T
    assert not value.is_contiguous()
    coordinates = _coordinates(tuple(value.shape), 7)
    actual = _sample(value, coordinates)
    assert actual.tolist() == [float(value[point]) for point in coordinates]
    assert len(coordinates) == 7


def test_diagnostic_write_failure_keeps_actual_updates_and_rng(tmp_path, monkeypatch):
    from natlang_neuralese.train.optim import PortMuonAdamW
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 2)
        def forward(self, x):
            return self.backbone(x)
    monkeypatch.setattr(native, 'packed_graph_loss', lambda m, b, **kw:
        ((m(torch.rand(3, 4)) - torch.rand(3, 2)).square().mean(), []))
    bad = tmp_path/'occupied'
    bad.write_text('not a directory')
    def run(observe):
        torch.manual_seed(73)
        model = Model()
        named = list(model.named_parameters())
        optimizer = PortMuonAdamW(named, lr=.001, vocab_size=100)
        optimizer._native_named_parameters = named
        def updates():
            for _ in range(4):
                native.native_training_update(model, optimizer, [])
        if observe:
            with observe_backbone_updates(bad, every=2) as state:
                updates()
            assert state['completed_updates'] == 4 and state['diagnostic_errors']
        else:
            updates()
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), torch.get_rng_state()
    assert_tree_equal(run(False), run(True))


def test_training_failure_propagates_and_restores_wrapper(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise FloatingPointError('actual update failed')
    monkeypatch.setattr(native, 'native_training_update', failed)
    with pytest.raises(FloatingPointError, match='actual update failed'):
        with observe_backbone_updates(tmp_path) as state:
            native.native_training_update(None, object(), [])
    assert native.native_training_update is failed and state['completed_updates'] == 0


def test_bound_helper_cli_retains_observer_after_finite_profile(tmp_path, monkeypatch):
    import scripts.neuralese_data.native_update_observer as helper
    from scripts.neuralese_data.profile_native_training import main
    source = __import__('pathlib').Path(helper.__file__)
    raw = source.read_bytes()
    binding = {'path': str(source), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    actual_calls = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(native, 'packed_graph_loss', lambda m, b, **kw:
        ((m(torch.rand(3, 4)) - torch.rand(3, 2)).square().mean(), []))
    def trainer(argv):
        from natlang_neuralese.train.optim import PortMuonAdamW
        actual_calls.append(argv)
        model = torch.nn.Sequential()
        model.add_module('backbone', torch.nn.Linear(4, 2))
        named = list(model.named_parameters())
        optimizer = PortMuonAdamW(named, lr=.001, vocab_size=100)
        optimizer._native_named_parameters = named
        for _ in range(4):
            native.native_training_update(model, optimizer, [])
        return 17
    monkeypatch.setattr(native, 'main', trainer)
    arguments = ['--out', str(tmp_path/'training'), '--task-eval', '']
    assert main(['--profile-out', str(tmp_path/'observer'), '--last-update', '2',
                 '--backbone-update-every', '3',
                 '--backbone-update-observer', json.dumps(binding), '--', *arguments]) == 17
    assert actual_calls == [arguments]
    rows = [json.loads(line) for line in (tmp_path/'observer/native-backbone-updates.jsonl').read_text().splitlines()]
    assert [row['relative_update'] for row in rows] == [1, 3]
    assert all(any(parameter['sampled_update_l2'] > 0 for parameter in row['parameters']) for row in rows)
    binding['sha256'] = '0' * 64
    with pytest.raises(SystemExit):
        main(['--profile-out', str(tmp_path/'wrong'), '--backbone-update-every', '250',
              '--backbone-update-observer', json.dumps(binding), '--', *arguments])


def test_actual_update_io_bracket_excludes_the_observer_output(tmp_path, monkeypatch):
    import os
    from types import SimpleNamespace
    payload = b'x' * (2 * 1024 * 1024)
    optimizer = SimpleNamespace(_native_named_parameters=[('backbone.weight', torch.nn.Parameter(torch.ones(1)))])
    token = object()
    def update(*args, **kwargs):
        with (tmp_path/'actual-update-output').open('wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return token
    monkeypatch.setattr(native, 'native_training_update', update)
    # CPU observations must never initialize a CUDA context just because a GPU exists.
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *args: (_ for _ in ()).throw(AssertionError('CPU observer touched CUDA')))
    with observe_backbone_updates(tmp_path/'observations', samples=1):
        assert native.native_training_update(None, optimizer, []) is token
    report = json.loads((tmp_path/'observations/native-update-resources.jsonl').read_text())
    assert report['update_completed'] and report['io_delta']['wchar'] == len(payload)
    assert report['io_delta']['write_bytes'] >= len(payload)
    assert report['sampling_error'] is None


def test_resource_sampling_error_keeps_actual_update_result(tmp_path, monkeypatch):
    import scripts.neuralese_data.native_update_observer as helper
    from types import SimpleNamespace
    optimizer = SimpleNamespace(_native_named_parameters=[('backbone.weight', torch.nn.Parameter(torch.ones(1)))])
    token = object()
    monkeypatch.setattr(native, 'native_training_update', lambda *args, **kwargs: token)
    monkeypatch.setattr(helper, '_resource_point', lambda *args: (_ for _ in ()).throw(OSError('proc unavailable')))
    with observe_backbone_updates(tmp_path, samples=1) as state:
        assert native.native_training_update(None, optimizer, []) is token
    assert state['completed_updates'] == 1 and state['diagnostic_errors'] == ['OSError']
