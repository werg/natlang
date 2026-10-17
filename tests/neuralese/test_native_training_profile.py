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


def test_startup_phase_hooks_preserve_real_updates_and_profile_window(task, tmp_path):
    from contextlib import contextmanager
    from scripts.neuralese_data.profile_native_training import observe_native_phases
    @contextmanager
    def observed():
        with observe_native_phases(tmp_path) as phases, profile_native_updates(
                tmp_path, first_update_observer=phases['finish_startup']) as state:
            yield state
    baseline = task()
    measured = task(observed())
    assert_tree_equal(baseline, measured)
    phases = [json.loads(line) for line in (tmp_path/'native-startup-phases.jsonl').read_text().splitlines()]
    assert sum(row['phase']=='startup' and row['status']=='end' for row in phases)==1
    receipt = json.loads((tmp_path/'native-profile-observer.json').read_text())
    assert receipt['completed_updates']==5
    assert (tmp_path/'profile-2-5.json').exists()


def test_checkpoint_phase_reads_once_returns_same_object_and_restores_descriptor(monkeypatch, tmp_path):
    from scripts.neuralese_data.profile_native_training import observe_native_phases
    payload = {'optimizer': object(), 'training': object()}
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        if path=='bad':
            raise ValueError('actual checkpoint failure')
        return payload
    monkeypatch.setattr(native.CheckpointPolicy, 'load', staticmethod(load))
    descriptor = vars(native.CheckpointPolicy)['load']
    with observe_native_phases(tmp_path):
        assert native.CheckpointPolicy.load('parent', expand_shared=False) is payload
        with pytest.raises(ValueError, match='actual checkpoint failure'):
            native.CheckpointPolicy.load('bad')
    assert calls==[('parent', {'expand_shared': False}), ('bad', {})]
    assert vars(native.CheckpointPolicy)['load'] is descriptor
    events = [json.loads(line) for line in (tmp_path/'native-startup-phases.jsonl').read_text().splitlines()]
    assert any(e['phase']=='restore/checkpoint-decode' and e['status']=='error' for e in events)


def test_phase_output_failure_does_not_change_checkpoint_call(monkeypatch, tmp_path):
    from scripts.neuralese_data.profile_native_training import observe_native_phases
    out = tmp_path/'a-file'
    out.write_text('occupied')
    payload = object()
    monkeypatch.setattr(native.CheckpointPolicy, 'load', staticmethod(lambda *a, **k: payload))
    with observe_native_phases(out) as state:
        assert native.CheckpointPolicy.load('parent') is payload
    assert state['diagnostic_errors']


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('preference_entry', [False, True])
def test_real_native_factory_tuple_and_preference_alias_load_phase(monkeypatch, tmp_path, family, preference_entry):
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.data.native_graph import parse_graph, SCHEMA
    from natlang_neuralese.train import native_preference
    from scripts.neuralese_data.profile_native_training import observe_native_phases
    port = tiny(family)
    if family == 'mellum':
        port.to(dtype=torch.bfloat16)
        port.hf.natlang_precision = 'bf16'
    model = SharedStateModel(port, (58, 59, 60, 61), rank=4)
    example = parse_graph({'schema': SCHEMA, 'id': 'phase-update', 'split': 'train',
        'tokenizer_fingerprint': 'fixture', 'source_groups': ['phase-update'],
        'nodes': [{'name': 'answer', 'token_ids': [1, 58, 54, 59, 60, 55, 61, 2],
                   'target_weights': [0, 1, 1, 1, 1, 1, 1, 1]}]}, tokenizer_fingerprint='fixture')
    factory = native.make_native_optimizer
    returned = []
    def record_factory(*args, **kwargs):
        result = factory(*args, **kwargs)
        returned.append(result)
        return result
    monkeypatch.setattr(native, 'make_native_optimizer', record_factory)
    with observe_native_phases(tmp_path) as phases:
        def trainer(arguments):
            result = native.make_native_optimizer(model, lr=1e-3, in_backward=True)
            assert result is returned[-1]
            optimizer, named = result
            assert isinstance(optimizer, PortMuonAdamW) and named
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            assert 'load_state_dict' in vars(optimizer)
            with profile_native_updates(tmp_path, first_update_observer=phases['finish_startup']):
                native.native_training_update(model, optimizer, [example], chunk_tokens=3)
            assert 'load_state_dict' not in vars(optimizer)
            return 0
        if preference_entry:
            monkeypatch.setattr(native, 'main', trainer)
            assert native_preference.main(['--help']) == 0
        else:
            assert trainer([]) == 0
    events = [json.loads(line) for line in (tmp_path/'native-startup-phases.jsonl').read_text().splitlines()]
    assert any(e['phase']=='restore/optimizer-state' and e['status']=='end' for e in events)
    assert not phases['diagnostic_errors']
