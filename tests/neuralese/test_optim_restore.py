"""Named optimizer-state restore (train/optim_restore.py): tiny CPU modules, no model, no GPU."""
import pytest
import torch

from natlang_neuralese.train.optim import LionSR, PortMuonAdamW
from natlang_neuralese.train.optim_restore import (OptimizerRestoreError, declared_added_names,
                                                   optimizer_param_names, restore_optimizer_state)
from natlang_neuralese.train.trajectory_state import atomic_checkpoint

needs_muon = pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="PyTorch Muon unavailable")
VOCAB = 1000


def tiny(extra=(), drop=(), seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = {"backbone.w1": (6, 5), "backbone.w2": (5, 6), "backbone.bias": (6,),
              "heads.w": (4, 7), "heads.b": (7,), "heads.norm": (7,)}
    shapes.update({k: v for k, v in extra})
    named = {k: torch.nn.Parameter(torch.randn(*v, generator=g)) for k, v in shapes.items() if k not in drop}
    return named


def build_port(named, vocab=VOCAB):
    """Build a split-rate optimizer over shared and task-specific parameter groups."""
    opt = PortMuonAdamW(list(named.items()), lr=1e-2, vocab_size=vocab)
    for child in (opt.muon, opt.auxiliary):
        if child is None:
            continue
        original = dict(child.param_groups[0])
        buckets = {}
        by_id = {id(p): n for n, p in named.items()}
        for p in original["params"]:
            buckets.setdefault(3e-2 if by_id[id(p)].startswith("heads.") else 1e-2, []).append(p)
        first, *rest = buckets.items()
        child.param_groups[0].update(params=first[1], lr=first[0])
        for rate, ps in rest:
            child.add_param_group({**original, "params": ps, "lr": rate})
    opt.param_groups = opt._groups()
    return opt


def train(opt, named, steps=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        for p in named.values():
            p.grad = torch.randn(p.shape, generator=g)
        opt.step()


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def _legacy_muonsr_restore(optimizer, saved):
    """Original loader as an independent precision/ownership reference."""
    protected = {index: {key: value for key, value in entry.items()
                         if key in ('momentum', 'grad_norm') and torch.is_tensor(value)}
                 for index, entry in saved['state'].items()}
    parameters = {index: parameter
                  for previous, current in zip(saved['param_groups'], optimizer.param_groups)
                  for index, parameter in zip(previous['params'], current['params'])}
    LionSR.load_state_dict(optimizer, saved)
    for index, values in protected.items():
        for key, value in values.items():
            optimizer.state[parameters[index]][key] = value.to(parameters[index].device, copy=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('hooks', [False, True])
def test_muonsr_restore_matches_legacy_state_hooks_and_next_stochastic_updates(tmp_path, dtype, hooks):
    import copy
    from natlang_neuralese.train.optim import MuonSR
    generator = torch.Generator().manual_seed(4523)
    value = torch.randn(3, 6, 8, generator=generator).to(dtype)
    source = torch.nn.Parameter(value.clone())
    original = MuonSR([source], lr=0.002, idle_fraction=0.1)
    # Deliberately retain precision not representable by a BF16 parameter.
    original.state[source] = {
        'momentum': torch.randn(source.shape, generator=generator),
        'grad_norm': torch.tensor([0.1234567, 0.2345678, 0.3456789]),
        'updates': torch.tensor(11, dtype=torch.int64),
    }
    original.param_groups[0]['row_scale'] = torch.tensor([1.125, 0.8125, 1.0]).reshape(3, 1, 1)
    path = tmp_path/'optimizer.pt'
    torch.save(original.state_dict(), path)
    saved = torch.load(path, mmap=True, weights_only=False)
    untouched = copy.deepcopy(saved)
    reference_parameter = torch.nn.Parameter(value.clone())
    actual_parameter = torch.nn.Parameter(value.clone())
    reference = MuonSR([reference_parameter], lr=0.9)
    actual = MuonSR([actual_parameter], lr=0.9)
    observed = [[], []]
    if hooks:
        for optimizer, events in zip((reference, actual), observed):
            def before(opt, state, events=events):
                events.append(('before', copy.deepcopy(state)))
            def after(opt, events=events):
                events.append(('after', copy.deepcopy(opt.state_dict())))
            optimizer.register_load_state_dict_pre_hook(before)
            optimizer.register_load_state_dict_post_hook(after)
    rng = torch.get_rng_state().clone()
    _legacy_muonsr_restore(reference, saved)
    actual.load_state_dict(saved)
    assert torch.equal(rng, torch.get_rng_state())
    assert equal(reference.state_dict(), actual.state_dict())
    assert equal(untouched, saved)
    assert equal(observed[0], observed[1])
    for key in ('momentum', 'grad_norm'):
        tensor = actual.state[actual_parameter][key]
        assert tensor.dtype == torch.float32
        assert tensor.data_ptr() != saved['state'][0][key].data_ptr()
    assert actual.param_groups[0]['row_scale'].data_ptr() != saved['param_groups'][0]['row_scale'].data_ptr()
    for step in range(3):
        grad = torch.randn(source.shape, generator=generator).to(dtype)
        # Include an idle expert to exercise its saved running-norm history.
        grad[1].zero_()
        reference_parameter.grad = grad.clone()
        actual_parameter.grad = grad.clone()
        torch.manual_seed(1200+step)
        reference.step()
        after_rng = torch.get_rng_state().clone()
        torch.manual_seed(1200+step)
        actual.step()
        assert torch.equal(after_rng, torch.get_rng_state())
        assert torch.equal(reference_parameter, actual_parameter)
        assert equal(reference.state_dict(), actual.state_dict())


def test_muonsr_restore_eliminates_discarded_precision_cast_copies():
    from natlang_neuralese.train.optim import MuonSR
    parameter = torch.nn.Parameter(torch.zeros(2, 6, 8, dtype=torch.bfloat16))
    original = MuonSR([parameter], lr=0.002)
    original.state[parameter] = {
        'momentum': torch.ones(parameter.shape, dtype=torch.float32),
        'grad_norm': torch.tensor([0.1234567, 0.2345678]),
    }
    saved = original.state_dict()
    def copies(legacy):
        target = MuonSR([torch.nn.Parameter(torch.zeros_like(parameter))], lr=0.002)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            if legacy:
                _legacy_muonsr_restore(target, saved)
            else:
                target.load_state_dict(saved)
        return sum(event.count for event in profile.key_averages() if event.key=='aten::_to_copy')
    # Both saved FP32 tensors used to be copied/cast and immediately discarded.
    # The CPU profiler measures real Torch allocations; no speedup is inferred.
    assert copies(True)-copies(False) == 2


def state_by_name(opt, named):
    names = optimizer_param_names(opt, named)
    sd = opt.state_dict()
    children = [("muon", sd["muon"]), ("adamw", sd["adamw"])] if "muon" in sd else [("plain", sd)]
    out, offset = {}, 0
    for _, c in children:
        order = [i for grp in c["param_groups"] for i in grp["params"]]
        for slot, i in enumerate(order):
            out[names[offset + slot]] = c["state"].get(i)
        offset += len(order)
    return out


@needs_muon
def test_exact_four_group_restore_is_bitwise_load_state_dict():
    named = tiny()
    opt = build_port(named)
    assert [len(c.param_groups) for c in (opt.muon, opt.auxiliary)] == [2, 2]
    train(opt, named)
    saved = opt.state_dict()
    names = optimizer_param_names(opt, named)

    named_a, named_b = tiny(seed=5), tiny(seed=5)
    a, b = build_port(named_a), build_port(named_b)
    a.load_state_dict(saved)
    report = restore_optimizer_state(b, saved, named_b, saved_names=names)
    assert report["mode"] == "exact" and report["initialized"] == []
    assert equal(a.state_dict(), b.state_dict())


@needs_muon
def test_declared_read_adapter_keeps_others_bitwise_and_initializes_new_only():
    named = tiny()
    opt = build_port(named)
    train(opt, named)
    saved = opt.state_dict()
    saved_names = optimizer_param_names(opt, named)
    before = state_by_name(opt, named)

    added = (("heads.read_adapter.proj.weight", (4, 4)), ("heads.read_adapter.proj.bias", (4,)))
    named2 = tiny(extra=added, seed=9)
    opt2 = build_port(named2)
    declared = declared_added_names(named2, ["heads.read_adapter."])
    report = restore_optimizer_state(opt2, saved, named2, declared, saved_names=saved_names)
    assert report["mode"] == "by-name"
    assert report["initialized"] == sorted(declared) and report["kept"] == len(saved_names)
    after = state_by_name(opt2, named2)
    for name in saved_names:
        assert equal(before[name], after[name]), name
        assert before[name]  # real moments, not empty
    for name in declared:
        assert not after[name]
    # new groups use the current optimizer's hyperparameters and training continues
    train(opt2, named2, steps=1)
    assert all(after_name for after_name in state_by_name(opt2, named2).values())


@needs_muon
def test_undeclared_new_parameter_is_refused_by_name():
    named = tiny()
    opt = build_port(named)
    train(opt, named)
    saved, names = opt.state_dict(), optimizer_param_names(opt, named)
    named2 = tiny(extra=(("heads.read_adapter.proj.bias", (4,)),))
    with pytest.raises(OptimizerRestoreError, match="heads.read_adapter.proj.bias"):
        restore_optimizer_state(build_port(named2), saved, named2, set(), saved_names=names)


@needs_muon
def test_removed_parameter_is_refused_by_name():
    named = tiny()
    opt = build_port(named)
    train(opt, named)
    saved, names = opt.state_dict(), optimizer_param_names(opt, named)
    named2 = tiny(drop=("heads.b",))
    with pytest.raises(OptimizerRestoreError, match="heads.b"):
        restore_optimizer_state(build_port(named2), saved, named2, {"heads.b"}, saved_names=names)


@needs_muon
def test_muon_adamw_routing_change_is_refused():
    named = tiny()
    opt = build_port(named)
    train(opt, named)
    saved, names = opt.state_dict(), optimizer_param_names(opt, named)
    named2 = tiny(seed=3)
    # a vocabulary of 6 makes backbone.w1 (6, 5) and backbone.w2 (5, 6) look like readouts: Muon -> AdamW
    with pytest.raises(OptimizerRestoreError, match="backbone.w1"):
        restore_optimizer_state(build_port(named2, vocab=6), saved, named2, saved_names=names)


@needs_muon
def test_old_positional_checkpoint_still_loads_and_mismatch_is_refused_not_reset():
    named = tiny()
    opt = build_port(named)
    train(opt, named)
    saved = opt.state_dict()
    named2 = tiny(seed=4)
    opt2 = build_port(named2)
    report = restore_optimizer_state(opt2, saved, named2, saved_names=None)
    assert report["mode"] == "positional"
    reference = build_port(tiny(seed=4))
    reference.load_state_dict(saved)
    assert equal(reference.state_dict(), opt2.state_dict())
    named3 = tiny(extra=(("heads.read_adapter.proj.bias", (4,)),))
    with pytest.raises(OptimizerRestoreError, match="optimizer-state fresh"):
        restore_optimizer_state(build_port(named3), saved, named3, {"heads.read_adapter.proj.bias"}, saved_names=None)


@needs_muon
def test_explicit_fresh_leaves_state_empty_and_reports_it():
    named = tiny()
    opt = build_port(named)
    report = restore_optimizer_state(opt, {}, named, fresh=True)
    assert report["mode"] == "fresh" and report["initialized_count"] == len(named)


def test_plain_adamw_and_lion_named_restore():
    for make in (lambda ps: torch.optim.AdamW([{"params": ps[:3], "lr": 1e-2}, {"params": ps[3:], "lr": 3e-2}]),
                 lambda ps: LionSR([{"params": ps[:3], "lr": 1e-2}, {"params": ps[3:], "lr": 3e-2}], lr=1e-2)):
        named = tiny()
        opt = make(list(named.values()))
        train(opt, named)
        saved, names = opt.state_dict(), optimizer_param_names(opt, named)
        before = state_by_name(opt, named)
        named2 = tiny(extra=(("heads.read_adapter.bias", (4,)),), seed=2)
        opt2 = make(list(named2.values()))
        with pytest.raises(OptimizerRestoreError, match="heads.read_adapter.bias"):
            restore_optimizer_state(opt2, saved, named2, set(), saved_names=names)
        report = restore_optimizer_state(opt2, saved, named2, {"heads.read_adapter.bias"}, saved_names=names)
        assert report["initialized"] == ["heads.read_adapter.bias"]
        after = state_by_name(opt2, named2)
        # (torch casts floating state to the parameter dtype on load: LionSR's bf16 momentum becomes fp32 here,
        # exactly as with a plain load_state_dict, so compare values)
        assert all(equal({k: v.float() if torch.is_tensor(v) else v for k, v in before[n].items()},
                         {k: v.float() if torch.is_tensor(v) else v for k, v in after[n].items()}) for n in names)
        # exact path for AdamW
        named3 = tiny(seed=6)
        opt3 = make(list(named3.values()))
        assert restore_optimizer_state(opt3, saved, named3, saved_names=names)["mode"] == "exact"


@needs_muon
def test_round_trip_through_checkpoint_save_and_load(tmp_path):
    named = tiny()
    opt = build_port(named)
    train(opt, named)
    path = tmp_path / "checkpoint.pt"
    atomic_checkpoint(path, {"optimizer": opt.state_dict(), "optimizer_param_names": optimizer_param_names(opt, named)})
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["optimizer_param_names"] == optimizer_param_names(opt, named)
    named2 = tiny(extra=(("heads.read_adapter.proj.bias", (4,)),), seed=8)
    opt2 = build_port(named2)
    report = restore_optimizer_state(opt2, loaded["optimizer"], named2, {"heads.read_adapter.proj.bias"},
                                     saved_names=loaded["optimizer_param_names"])
    assert report["initialized"] == ["heads.read_adapter.proj.bias"]
    before, after = state_by_name(opt, named), state_by_name(opt2, named2)
    assert all(equal(before[n], after[n]) for n in before)


def test_restore_from_an_mmap_checkpoint_keeps_no_view_of_the_file(tmp_path):
    from natlang_neuralese.train.optim_restore import restore_optimizer_state

    def model():
        torch.manual_seed(0)
        return {'w': torch.nn.Parameter(torch.randn(8, 4)), 'b': torch.nn.Parameter(torch.randn(4))}

    named = model()
    optimizer = torch.optim.AdamW(list(named.values()), lr=1e-3)
    sum(p.square().sum() for p in named.values()).backward()
    optimizer.step()
    torch.save({'optimizer': optimizer.state_dict()}, tmp_path / 'state.pt')
    saved = torch.load(tmp_path / 'state.pt', mmap=True, weights_only=False)['optimizer']
    mapped = set()
    for state in saved['state'].values():
        mapped |= {v.untyped_storage().data_ptr() for v in state.values() if torch.is_tensor(v)}
    fresh_named = model()
    fresh = torch.optim.AdamW(list(fresh_named.values()), lr=1e-3)
    restore_optimizer_state(fresh, saved, fresh_named, saved_names=['w', 'b'])
    kept = {v.untyped_storage().data_ptr() for state in fresh.state.values() for v in state.values() if torch.is_tensor(v)}
    assert kept and not (kept & mapped)  # AdamW's CPU 'step' and moments are owned copies: the mapping can close


@needs_muon
def test_a_removed_component_drops_its_slots_and_keeps_every_other_state_exact():
    """The direct stage's NeuraleseProjection left the stage (owner 2026-10-13): a checkpoint whose Muon child held
    only the projection's matrices restores into an optimizer without it (no Muon child), the AdamW and latent
    states unchanged (drop_port_optimizer_slots)."""
    from natlang_neuralese.train.optim_restore import drop_port_optimizer_slots
    g = torch.Generator().manual_seed(0)
    make = lambda *shape: torch.nn.Parameter(torch.randn(*shape, generator=g))
    latent = [('backbone.attn', make(6, 6)), ('backbone.expert', make(6, 6))]
    other = [('backbone.norm', make(6))]
    head = [('neuralese_projection.up', make(8, 6)), ('neuralese_projection.bias', make(8))]

    def build(with_head):
        return PortMuonAdamW((head if with_head else []) + other, lr=1e-2, vocab_size=VOCAB,
                             latent=[(n, p, 1e-2, 1) for n, p in latent], latent_optimizer='muonsr')
    before = build(True)
    for _, p in head + other + latent:
        p.grad = torch.randn(p.shape, generator=g)
    before.step()
    names = {id(p): n for n, p in head + other + latent}
    groups = [[names[id(p)] for p in group['params']] for group in before.param_groups]
    saved = before.state_dict()
    state, kept, dropped = drop_port_optimizer_slots(saved, groups, lambda n: n.startswith('neuralese_projection.'))
    assert sorted(dropped) == ['neuralese_projection.bias', 'neuralese_projection.up'] and state['muon'] is None
    after = build(False)
    assert after.muon is None and state['schema'] == after.schema
    assert kept == [[names[id(p)] for p in group['params']] for group in after.param_groups]
    after.load_state_dict(state)
    reference = before.state_dict()
    for kind in ('adamw', 'muonsr'):
        current = after.state_dict()[kind]['state']
        old = {names[id(p)]: v for p, v in zip([p for grp in getattr(before, 'auxiliary' if kind == 'adamw' else
                                                                       'latent').param_groups for p in grp['params']],
                                                [reference[kind]['state'].get(i) for i in range(10)])}
        new_params = [p for grp in getattr(after, 'auxiliary' if kind == 'adamw' else 'latent').param_groups
                      for p in grp['params']]
        for index, p in enumerate(new_params):
            for key, value in current[index].items():
                expected = old[names[id(p)]][key]
                assert torch.equal(value, expected) if torch.is_tensor(value) else value == expected


@needs_muon
@pytest.mark.parametrize('corrupt', [None, 'grad_norm', 'momentum'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_projection_addition_preserves_stacked_muonsr_norms_and_rejects_wrong_shapes(corrupt, dtype):
    import copy
    torch.manual_seed(17)
    expert = torch.nn.Parameter(torch.randn(3, 6, 6, dtype=dtype))
    norm = torch.nn.Parameter(torch.randn(6))
    before_named = {'backbone.norm': norm, 'backbone.expert': expert}
    before = PortMuonAdamW([('backbone.norm', norm)], lr=1e-2, vocab_size=VOCAB,
                           latent=[('backbone.expert', expert, 1e-2, 1)], latent_optimizer='muonsr')
    for _ in range(2):
        for parameter in before_named.values():
            parameter.grad = torch.randn_like(parameter)
        before.step()
    saved = copy.deepcopy(before.state_dict())
    latent_state = next(iter(saved['muonsr']['state'].values()))
    assert latent_state['grad_norm'].shape == (3,)
    assert latent_state['momentum'].dtype == torch.bfloat16 and latent_state['grad_norm'].dtype == torch.float32
    if corrupt is not None:
        latent_state[corrupt] = torch.zeros(4 if corrupt == 'grad_norm' else 3)
    projection = torch.nn.Parameter(torch.randn(8, 6))
    after_named = dict(before_named, **{'token_choice_projection.weight': projection})
    after = PortMuonAdamW([('backbone.norm', norm), ('token_choice_projection.weight', projection)],
                          lr=1e-2, vocab_size=VOCAB, latent=[('backbone.expert', expert, 1e-2, 1)],
                          latent_optimizer='muonsr')
    kwargs = dict(saved_names=optimizer_param_names(before, before_named),
                  declared_added=['token_choice_projection.weight'])
    if corrupt is not None:
        with pytest.raises(OptimizerRestoreError, match='saved state shape differs'):
            restore_optimizer_state(after, saved, after_named, **kwargs)
    else:
        report = restore_optimizer_state(after, saved, after_named, **kwargs)
        assert report['mode'] == 'by-name' and report['initialized'] == ['token_choice_projection.weight']
        assert equal(before.latent.state_dict(), after.latent.state_dict())
