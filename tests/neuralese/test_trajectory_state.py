import pytest
import torch
import random
import threading

from natlang_neuralese.train.trajectory_state import (
    AsyncAtomicCheckpointWriter, atomic_checkpoint, immutable_cpu_snapshot,
    trajectory_optimizer, validate_resume,
)


def test_immutable_cpu_snapshot_detaches_nested_tensor_state():
    from collections import OrderedDict

    source = torch.tensor([1., 2.], requires_grad=True)
    state = {'model': OrderedDict(weight=source, tail=source[1:]), 'rng': (1, 2, 3)}
    state['model']._metadata = {'version': 1}
    snapshot = immutable_cpu_snapshot(state)
    source.data.fill_(9.)
    assert snapshot['model']['weight'].device.type == 'cpu'
    assert snapshot['model']['weight'].tolist() == [1., 2.]
    assert snapshot['model']['weight'].untyped_storage().data_ptr() == snapshot['model']['tail'].untyped_storage().data_ptr()
    snapshot['model']['tail'][0] = 5.
    assert snapshot['model']['weight'][1].item() == 5.
    assert snapshot['model']['weight'].requires_grad is False
    assert snapshot['model']._metadata == {'version': 1}
    assert snapshot['rng'] == (1, 2, 3)


def test_async_atomic_checkpoint_writes_snapshot_and_runs_callback(tmp_path):
    destination = tmp_path / 'state.pt'
    source = torch.tensor([3., 4.])
    callbacks = []
    writer = AsyncAtomicCheckpointWriter()
    writer.submit(destination, {'step': 7, 'tensor': source, 'tail': source[1:]},
                  after_write=lambda snapshot: callbacks.append(snapshot['step']))
    source.fill_(8.)
    writer.drain()
    saved = torch.load(destination, map_location='cpu', weights_only=False)
    assert saved['step'] == 7
    assert torch.equal(saved['tensor'], torch.tensor([3., 4.]))
    assert saved['tensor'].untyped_storage().data_ptr() == saved['tail'].untyped_storage().data_ptr()
    assert callbacks == [7]
    assert not destination.with_suffix('.pending').exists()


def test_async_atomic_checkpoint_failure_propagates_and_writer_recovers(tmp_path, monkeypatch):
    import natlang_neuralese.train.trajectory_state as state_module

    writer = AsyncAtomicCheckpointWriter()
    destination = tmp_path / 'state.pt'
    real_atomic_checkpoint = state_module.atomic_checkpoint

    def fail_once(path, state):
        if state['step'] == 1:
            raise OSError('disk full')
        return real_atomic_checkpoint(path, state)

    monkeypatch.setattr(state_module, 'atomic_checkpoint', fail_once)
    writer.submit(destination, {'step': 1})
    with pytest.raises(OSError, match='disk full'):
        writer.drain()
    writer.submit(destination, {'step': 2}, wait=True)
    assert torch.load(destination, map_location='cpu', weights_only=False)['step'] == 2


def test_async_atomic_checkpoint_propagates_post_write_failure(tmp_path):
    writer = AsyncAtomicCheckpointWriter()
    destination = tmp_path / 'state.pt'

    def fail_after_write(_snapshot):
        raise OSError('best receipt failed')

    writer.submit(destination, {'step': 3}, after_write=fail_after_write)
    with pytest.raises(OSError, match='best receipt failed'):
        writer.drain()
    assert torch.load(destination, map_location='cpu', weights_only=False)['step'] == 3
    writer.submit(destination, {'step': 4}, wait=True)
    assert torch.load(destination, map_location='cpu', weights_only=False)['step'] == 4


def test_async_snapshot_preserves_committed_model_optimizer_and_rng(tmp_path):
    destination = tmp_path / 'full-state.pt'
    original_python_rng = random.getstate()
    original_torch_rng = torch.get_rng_state()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    state = {
        'step': 19,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'python_rng': random.getstate(),
        'torch_rng': torch.get_rng_state(),
    }
    expected = immutable_cpu_snapshot(state)
    callback_started = threading.Event()
    release_callback = threading.Event()
    writer = AsyncAtomicCheckpointWriter()

    def blocked_callback(_snapshot):
        callback_started.set()
        if not release_callback.wait(5):
            raise TimeoutError('test did not release checkpoint callback')

    writer.submit(destination, state, after_write=blocked_callback)
    try:
        assert callback_started.wait(5)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(3)
            for values in optimizer.state.values():
                for value in values.values():
                    if isinstance(value, torch.Tensor):
                        value.add_(2)
        random.random()
        torch.manual_seed(9981)
    finally:
        release_callback.set()
        random.setstate(original_python_rng)
        torch.set_rng_state(original_torch_rng)
    writer.drain()
    saved = torch.load(destination, map_location='cpu', weights_only=False)
    assert saved['step'] == expected['step']
    assert torch.equal(saved['model']['weight'], expected['model']['weight'])
    assert torch.equal(saved['model']['bias'], expected['model']['bias'])
    assert saved['optimizer']['state'].keys() == expected['optimizer']['state'].keys()
    for key, expected_slots in expected['optimizer']['state'].items():
        for name, expected_value in expected_slots.items():
            actual = saved['optimizer']['state'][key][name]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual, expected_value)
            else:
                assert actual == expected_value
    assert saved['python_rng'] == expected['python_rng']
    assert torch.equal(saved['torch_rng'], expected['torch_rng'])


def test_synchronous_emergency_write_drains_prior_snapshot(tmp_path):
    destination = tmp_path / 'state.pt'
    callback_started = threading.Event()
    release_callback = threading.Event()
    writer = AsyncAtomicCheckpointWriter()

    def blocked_callback(_snapshot):
        callback_started.set()
        if not release_callback.wait(5):
            raise TimeoutError('test did not release pending write')

    writer.submit(destination, {'step': 1}, after_write=blocked_callback)
    assert callback_started.wait(5)
    timer = threading.Timer(0.05, release_callback.set)
    timer.start()
    writer.write_synchronously(destination, {'step': 2, 'emergency': True})
    timer.join()
    assert torch.load(destination, map_location='cpu', weights_only=False) == {
        'step': 2, 'emergency': True,
    }
    assert not writer.pending


def test_immutable_cpu_snapshot_rejects_unsupported_leaves():
    with pytest.raises(TypeError, match='unsupported checkpoint snapshot leaf'):
        immutable_cpu_snapshot({'bad': object()})
    from collections import defaultdict
    with pytest.raises(TypeError, match='defaultdict factories'):
        immutable_cpu_snapshot(defaultdict(list, value=1))


def test_sketch_horizon_change_requires_declared_continuation():
    from natlang_neuralese.train.trajectory_state import validate_continuation
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
             'identity': {'options': {'sketch_gradient': 'one_step'}, 'files': {'source': 'same-sha'}}}
    identity = {'options': {'sketch_gradient': 'local_stage'}, 'files': {'source': 'same-sha'}}
    with pytest.raises(ValueError, match='training controls'):
        validate_continuation(state, identity)
    validate_continuation(state, identity, allowed_changes=['sketch_gradient'])
    with pytest.raises(ValueError, match='changed'):
        validate_resume(state, identity)


def test_resume_uses_saved_initial_rows_without_changing_identity():
    from natlang_neuralese.train.trajectory_state import resumed_initial_rows
    original = torch.tensor([[1., 2., 3.]])
    state = {'init': {'prompt': original}, 'params': {'prompt': original + 7}}
    assert resumed_initial_rows(None, ['prompt'], 3) == {}
    assert resumed_initial_rows(state, ['prompt'], 3)['prompt'] is original
    with pytest.raises(ValueError, match='names changed'):
        resumed_initial_rows(state, ['other'], 3)
    with pytest.raises(ValueError, match='shape'):
        resumed_initial_rows(state, ['prompt'], 4)
    state['init']['prompt'] = torch.full_like(original, float('nan'))
    with pytest.raises(ValueError, match='values'):
        resumed_initial_rows(state, ['prompt'], 3)


def test_incomplete_iteration_restores_all_rngs_and_baseline():
    import random
    from natlang_neuralese.train.trajectory_state import iteration_rng_state, restore_iteration_rng
    write = random.Random(7)
    stop = torch.Generator().manual_seed(9)
    baseline = {'value': .5}
    state = iteration_rng_state(write, stop, baseline)
    def draw():
        return random.random(), write.random(), torch.rand(3, generator=stop), torch.rand(3)
    expected = draw()
    baseline['value'] = 100
    draw()
    restore_iteration_rng(state, write, stop, baseline)
    actual = draw()
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    torch.testing.assert_close(actual[3], expected[3], rtol=0, atol=0)
    assert baseline == {'value': .5}


@pytest.mark.parametrize('policy', ['adamw', 'muon'])
def test_optimizer_state_roundtrip_and_parameter_group_rates(tmp_path, policy):
    torch.manual_seed(0)
    params = {'prompt': torch.nn.Parameter(torch.randn(3, 8))}
    lora = [torch.nn.Parameter(torch.randn(2, 8))]
    heads = [torch.nn.Parameter(torch.randn(13, 8)), torch.nn.Parameter(torch.randn(8, 8))]
    options = dict(vocab_size=13, lr=.01, lora_lr=.002, heads_lr=.003)
    optimizer = trajectory_optimizer(policy, params, lora, heads, **options)
    for g in optimizer.param_groups:
        expected = .01 if any(q is params['prompt'] for q in g['params']) else (.002 if any(q is lora[0] for q in g['params']) else .003)
        assert g['lr'] == expected
    values = list(params.values()) + lora + heads
    sum(q.square().sum() for q in values).backward()
    optimizer.step()
    path = tmp_path / 'checkpoint.pt'
    identity = {'input': 'fixed', 'optimizer': policy}
    atomic_checkpoint(path, {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': identity,
                             'optimizer': optimizer.state_dict(), 'weights': [q.detach().clone() for q in values]})
    saved = torch.load(path, weights_only=False)
    validate_resume(saved, identity)
    with pytest.raises(ValueError, match='changed'):
        validate_resume(saved, {**identity, 'input': 'other'})
    other = trajectory_optimizer(policy, params, lora, heads, **options)
    other.load_state_dict(saved['optimizer'])
    assert len(other.param_groups) == len(optimizer.param_groups)
    assert not path.with_suffix('.pending').exists()


def test_evaluation_restores_training_rng_and_baseline_even_on_error():
    import random
    from natlang_neuralese.train.trajectory_state import evaluation_state
    rng = random.Random(12)
    stop = torch.Generator().manual_seed(13)
    baseline = {'value': 1.25}
    state = rng.getstate(), stop.get_state(), torch.get_rng_state(), random.getstate()
    with pytest.raises(RuntimeError):
        with evaluation_state(rng, stop, baseline):
            rng.random()
            torch.rand(3, generator=stop)
            torch.rand(3)
            random.random()
            baseline['value'] = 999
            raise RuntimeError('probe failure')
    assert rng.getstate() == state[0]
    assert torch.equal(stop.get_state(), state[1])
    assert torch.equal(torch.get_rng_state(), state[2])
    assert random.getstate() == state[3]
    assert baseline == {'value': 1.25}


def test_soft_warm_start_requires_matching_text_and_valid_shape(tmp_path):
    from natlang_neuralese.train.trajectory_state import soft_initialization
    path=tmp_path/'soft.pt'
    rows=torch.ones(3,8)
    torch.save({'params':{'shared':rows,'old':rows},'texts':{'shared':'unchanged','old':'old'}},path)
    result=soft_initialization(path,{'shared':'unchanged','new':'new'},8)
    assert list(result)==['shared']
    assert torch.equal(result['shared'],rows)
    with pytest.raises(ValueError,match='text changed'):
        soft_initialization(path,{'shared':'different'},8)
    with pytest.raises(ValueError,match='tensor'):
        soft_initialization(path,{'shared':'unchanged'},9)


def test_legacy_soft_warm_start_checks_pinned_piece_file(tmp_path):
    import hashlib,json
    from natlang_neuralese.train.trajectory_state import soft_initialization
    pieces=tmp_path/'pieces.jsonl';pieces.write_text(json.dumps({'name':'p','text':'original'})+'\n')
    state={'params':{'p':torch.ones(2,8)},'identity':{'options':{'pieces':str(pieces)},'files':{str(pieces.resolve()):hashlib.sha256(pieces.read_bytes()).hexdigest()}}}
    path=tmp_path/'state.pt';torch.save(state,path)
    assert 'p' in soft_initialization(path,{'p':'original'},8)
    pieces.write_text('changed')
    with pytest.raises(ValueError,match='texts changed'):
        soft_initialization(path,{'p':'original'},8)


def test_explicit_continuation_preserves_fixed_inputs_but_allows_new_code_stage():
    from natlang_neuralese.train.trajectory_state import validate_continuation, validate_resume
    identity = {'options': {'batch': 1}, 'files': {'data': 'fixed'}, 'code': {'module': 'old'}}
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': identity}
    new = dict(identity, code={'module': 'fixed'}, continuation={'checkpoint_sha256': 'pinned'})
    validate_continuation(state, new)
    with pytest.raises(ValueError):
        validate_resume(state, new)
    with pytest.raises(ValueError):
        validate_continuation(state, dict(new, files={'data': 'changed'}))


def test_curriculum_change_is_named_and_cannot_change_data_or_optimizer():
    from natlang_neuralese.train.trajectory_state import validate_continuation
    old = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
           'identity': {'options': {'tokens_per_vector': 2, 'lr': .01}, 'files': {'data': 'same'}}}
    current = {'options': {'tokens_per_vector': 1, 'lr': .01, 'writer_text_weight': 1}, 'files': {'data': 'same'}}
    with pytest.raises(ValueError):
        validate_continuation(old, current)
    validate_continuation(old, current, allowed_changes=['tokens_per_vector', 'writer_text_weight'])


def test_legacy_continuation_can_enable_projection_anchor_as_one_named_change():
    from natlang_neuralese.train.trajectory_state import validate_continuation
    old = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
           'identity': {'options': {'lr': .01}, 'files': {'data': 'same'}}}
    current = {'options': {'lr': .01, 'projection_anchor_weight': 1.0,
                           'projection_anchor_backbone_scale': .05},
               'files': {'data': 'same'}}
    with pytest.raises(ValueError, match='training controls'):
        validate_continuation(old, current)
    validate_continuation(old, current, allowed_changes=['projection_anchor_weight'])
    with pytest.raises(ValueError):
        validate_continuation(old, current, allowed_changes=['lr'])
    with pytest.raises(ValueError):
        validate_continuation(old, dict(current, files={'data': 'different'}),
                              allowed_changes=['tokens_per_vector', 'writer_text_weight'])


def test_paired_probe_coverage_counts_readers_not_all_held_turns():
    from natlang_neuralese.train.trajectory_state import paired_probe_complete
    assert paired_probe_complete({'expected_n': 12, 'n': 12})
    assert not paired_probe_complete({'expected_n': 12, 'n': 11})
    assert not paired_probe_complete({'expected_n': 12, 'n': 12, 'reader_errors': ['failed']})
    assert not paired_probe_complete({'n': 12})


def test_gradient_clip_preserves_large_finite_adjoints():
    from natlang_neuralese.train.trajectory_state import clip_finite_gradients
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3e30, 4e30])
    norm = clip_finite_gradients([parameter])
    assert torch.isfinite(norm)
    torch.testing.assert_close(parameter.grad, torch.tensor([.6, .8]))


def test_gradient_clip_rejects_nan_before_mutation():
    from natlang_neuralese.train.trajectory_state import clip_finite_gradients
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([float('nan')])
    with pytest.raises(RuntimeError, match='nonfinite'):
        clip_finite_gradients([parameter])


def test_best_probe_candidate_is_not_inherited_across_unknown_or_changed_regimes():
    from natlang_neuralese.train.trajectory_state import compatible_best_evaluation
    signature = {'content_transport': 'raw-identity', 'writer_length_policy': 'native-value'}
    assert compatible_best_evaluation({'written': .01}, signature) is None
    previous = {'written': .01, 'selection_signature': {**signature, 'writer_length_policy': 'source-text'}}
    assert compatible_best_evaluation(previous, signature) is None
    current = {'written': .5, 'selection_signature': signature}
    assert compatible_best_evaluation(current, signature) is current
    assert compatible_best_evaluation(None, signature) is None


def test_resume_follows_new_code_but_not_new_inputs():
    identity = {'options': {'batch': 1}, 'files': {'data': 'fixed'}, 'code': {'a.py': 'old', 'b.py': 'same'}}
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': identity}
    assert validate_resume(state, dict(identity, code={'a.py': 'new', 'b.py': 'same'})) == ['a.py']
    with pytest.raises(ValueError, match='changed'):
        validate_resume(state, dict(identity, options={'batch': 2}))


def test_missing_saved_option_counts_as_its_default():
    state = {'schema': 'natlang.neuralese_recurrence_checkpoint/1',
             'identity': {'options': {'batch': 1}, 'files': {'data': 'fixed'}}}
    current = {'options': {'batch': 1, 'qat_latent_lr': 0.0}, 'files': {'data': 'fixed'}}
    with pytest.raises(ValueError):
        validate_resume(state, current)
    assert validate_resume(state, current, defaults={'qat_latent_lr': 0.0}) == []
    with pytest.raises(ValueError):
        validate_resume(state, {**current, 'options': {'batch': 1, 'qat_latent_lr': 1e-3}}, defaults={'qat_latent_lr': 0.0})


def test_qat_latents_get_their_own_adamw_groups():
    if not hasattr(torch.optim, 'Muon'):
        pytest.skip('no Muon')
    params = {'prompt': torch.nn.Parameter(torch.randn(2, 8))}
    dense = torch.nn.Parameter(torch.zeros(8, 8))
    other = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = trajectory_optimizer('muon', params, [dense, other], [], vocab_size=13, lr=.01, lora_lr=3e-5,
                                     heads_lr=.003, lora_names=['a.dense', 'b.weight'], latent_lrs={'a.dense': 5e-5})
    groups = [g for g in optimizer.param_groups if any(q is dense for q in g['params'])]
    assert len(groups) == 1 and groups[0]['lr'] == 5e-5 and len(groups[0]['params']) == 1
    assert not any(q is dense for g in optimizer.muon.param_groups for q in g['params'])
    assert any(q is other for g in optimizer.muon.param_groups for q in g['params'])


def test_named_optimizer_restore_keeps_matching_moments_and_starts_only_new_parameters():
    import torch
    from natlang_neuralese.train.optim import named_optimizer_state, parameter_names, restore_optimizer_named

    old = torch.nn.ModuleDict({"a": torch.nn.Linear(3, 3), "b": torch.nn.Linear(3, 2)})
    opt = torch.optim.AdamW(old.parameters(), lr=1e-2)
    old["a"](torch.randn(4, 3)).sum().backward(); old["b"](torch.randn(4, 3)).sum().backward(); opt.step()
    saved = named_optimizer_state(opt, parameter_names(m=old))
    new = torch.nn.ModuleDict({"a": torch.nn.Linear(3, 3), "c": torch.nn.Linear(3, 1), "b": torch.nn.Linear(3, 2)})
    new.load_state_dict(old.state_dict(), strict=False)
    regrouped = torch.optim.AdamW([{"params": list(new["c"].parameters())},
                                   {"params": list(new["a"].parameters()) + list(new["b"].parameters())}], lr=1e-2)
    report = restore_optimizer_named(regrouped, saved, parameter_names(m=new))
    assert report["mode"] == "named" and report["restored"] == 4
    assert sorted(report["added"]) == ["m.c.bias", "m.c.weight"] and report["dropped"] == []
    assert torch.equal(regrouped.state[new["a"].weight]["exp_avg"], opt.state[old["a"].weight]["exp_avg"])
    assert new["c"].weight not in regrouped.state
    legacy = restore_optimizer_named(regrouped, opt.state_dict(), parameter_names(m=new))
    assert legacy["mode"] == "fresh-legacy"
