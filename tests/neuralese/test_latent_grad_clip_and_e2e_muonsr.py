"""The spike-only gradient clip, exact and in backward (train/optim.py), the layer staging of every layer loop, the shared latent split used by the direct and the end-to-end
stage, the E2E stage's MuonSR optimizer and its resume refusal (train/trajectory_state.py), and the shared drift
monitor's guard (train/drift_monitor.py), on small CPU tensors."""
import argparse
import math

import pytest
import torch
from torch import nn

from natlang_neuralese.train.optim import (MuonSR, PortMuonAdamW, SpikeClip, latent_entries, latent_options,
                                           latent_split)


def _muonsr(weights, spike):
    optimizer = MuonSR([{'params': [w], 'lr': 1e-2, 'name': f'w{i}'} for i, w in enumerate(weights)], lr=1e-2)
    optimizer.step_in_backward(gated=True, spike=spike)
    return optimizer


def _update(optimizer, weights, grads, seen=None):
    if seen is not None:
        original = optimizer._update

        def spy(p, group):
            seen[group['name']] = p.grad.detach().clone()
            return original(p, group)
        optimizer._update = spy
    with optimizer.in_backward() as armed:
        sum((w * g).sum() for w, g in zip(weights, grads)).backward()
    return armed


def test_spike_clip_threshold_is_a_multiple_of_the_running_median_after_warmup():
    spike = SpikeClip(5.0)
    for norm in [10.0] * (SpikeClip.WARMUP - 1):
        spike.observe(norm)
    assert spike.threshold() is None and spike.coefficient(1e6) == 1.0
    spike.observe(1000.0)  # a spike enters the history; the median ignores it
    assert spike.threshold() == pytest.approx(50.0)
    assert spike.coefficient(40.0) == 1.0 and spike.coefficient(100.0) == pytest.approx(0.5)
    for _ in range(2 * SpikeClip.WINDOW):
        spike.observe(2.0)
    assert len(spike.history) == SpikeClip.WINDOW and spike.threshold() == pytest.approx(10.0)
    again = SpikeClip(5.0)
    again.load_state_dict(spike.state_dict())
    assert again.threshold() == spike.threshold()
    with pytest.raises(ValueError):
        SpikeClip(1.0)


def test_exact_clip_clips_only_a_spike():
    from natlang_neuralese.train.trajectory_state import clip_finite_gradients
    spike = SpikeClip(5.0)
    for _ in range(SpikeClip.WARMUP):
        spike.observe(1.0)
    q = nn.Parameter(torch.zeros(4))
    q.grad = torch.full((4,), 1.0)  # norm 2: no spike
    assert float(clip_finite_gradients([q], spike=spike)) == pytest.approx(2.0) and torch.equal(q.grad, torch.ones(4))
    q.grad = torch.full((4,), 10.0)  # norm 20 > 5: clipped to 5
    clip_finite_gradients([q], spike=spike)
    assert float(q.grad.norm()) == pytest.approx(5.0, rel=1e-5)


def test_in_backward_spike_clip_leaves_normal_updates_and_clips_latents_after_the_bound_crosses():
    torch.manual_seed(0)
    weights = [nn.Parameter(torch.randn(8, 8)) for _ in range(3)]
    spike = SpikeClip(5.0)
    optimizer = _muonsr(weights, spike)
    armed = _update(optimizer, weights, [torch.randn(8, 8) for _ in weights])
    assert armed.threshold is None  # warm-up: never clipped
    for _ in range(SpikeClip.WARMUP):
        spike.observe(10.0)
    seen = {}
    normal = [torch.randn(8, 8) for _ in weights]  # norm ~14 < 50
    armed = _update(optimizer, weights, normal, seen)
    record = armed.clip_record(armed.norm())
    assert record['exact_coefficient'] == 1.0 and record['latent_applied_norm'] == pytest.approx(record['latent_norm'])
    for i, g in enumerate(normal):
        assert torch.allclose(seen[f'w{i}'], g)
    seen.clear()
    big = [torch.randn(8, 8) * 20 for _ in weights]  # each ~160: the bound crosses 50 at the first latent
    armed = _update(optimizer, weights, big, seen)
    record = armed.clip_record(armed.norm())
    assert record['exact_coefficient'] < 1.0
    # Every latent clipped by threshold / the norm so far (>= the exact factor): applied lies between exact and raw.
    assert record['latent_exact_norm'] <= record['latent_applied_norm'] < record['latent_norm']
    assert all(float(seen[k].norm()) < float(g.norm()) for k, g in zip(sorted(seen), big))
    assert math.isclose(float(armed.norm()), float(torch.stack([g.norm() for g in big]).norm()), rel_tol=1e-5)


def test_isolated_sequence_and_batched_step_join_the_layer_staging_so_each_latent_steps_once():
    """Formerly 'a latent received a second gradient': these passes ran their layers outside the tape."""
    from natlang_neuralese.model.layer_staging import LayerStaging
    from natlang_neuralese.model.lfm2_port import PortCache
    from natlang_neuralese.serve.batch import step_rows
    from test_isolated_sequence import tiny_lfm
    import copy

    def losses(backbone):
        torch.manual_seed(3)
        prefix, fixed, replacements = (torch.randn(1, n, 32) for n in (3, 4, 4))
        cache = PortCache.empty(backbone.num_layers)
        _, cache = backbone.run_layers(prefix, range(backbone.num_layers), cache)
        out = backbone.isolated_sequence(fixed, replacements, cache, cutoff=2)
        stepped, _ = step_rows(backbone, torch.randn(2, 1, 32), [PortCache.empty(backbone.num_layers)] * 2,
                               range(backbone.num_layers))
        return out['final'].square().mean() + out['history_shallow'].tanh().sum() * .1 + stepped.square().mean()

    backbone = tiny_lfm()
    reference = copy.deepcopy(backbone)
    named = [(n, q) for n, q in backbone.hf.named_parameters() if n.startswith('model.layers.') and q.ndim >= 2]
    for _, q in named:
        q.requires_grad_(True)
    for n, q in reference.hf.named_parameters():
        q.requires_grad_(n.startswith('model.layers.') and q.ndim >= 2)
    losses(reference).backward()
    expected = {n: q.grad.clone() for n, q in reference.hf.named_parameters() if q.grad is not None}
    optimizer = MuonSR([{'params': [q], 'lr': 0.0, 'name': n} for n, q in named if min(q.shape[-2:]) > 1], lr=0.0)
    optimizer.step_in_backward(gated=True)
    seen = {}
    original = optimizer._update
    optimizer._update = lambda p, group: (seen.__setitem__(group['name'], p.grad.detach().clone()), original(p, group))
    with LayerStaging() as tape:
        total = losses(backbone)
        with optimizer.in_backward():
            total.backward()
            tape.backward()
    assert seen and set(seen) <= set(expected)
    for name, grad in seen.items():
        torch.testing.assert_close(grad, expected[name], rtol=1e-4, atol=1e-6, msg=name)


def test_latent_split_keeps_routers_vectors_and_narrow_matrices_out_of_muonsr():
    named = [('backbone.layers.0.self_attn.q_proj.weight', torch.zeros(32, 32)),
             ('backbone.layers.0.mlp.gate.weight', torch.zeros(8, 32)),
             ('backbone.layers.0.conv.conv.weight', torch.zeros(32, 1, 3)),
             ('backbone.layers.0.mlp.experts.gate_up', torch.zeros(4, 64, 32)),
             ('backbone.layers.0.input_layernorm.weight', torch.zeros(32))]
    latent, other = latent_split(named, 'muonsr')
    assert [n for n, _ in latent] == ['backbone.layers.0.self_attn.q_proj.weight', 'backbone.layers.0.mlp.experts.gate_up']
    assert len(other) == 3
    lion_latent, _ = latent_split(named, 'lionsr')
    assert 'backbone.layers.0.conv.conv.weight' in [n for n, _ in lion_latent]
    entries = latent_entries(latent, 'muonsr', 1e-5)
    assert [e[3] for e in entries] == [1, 2] and all(e[2] == 1e-5 for e in entries)
    assert latent_options('muonsr', weight_decay=0.1, ns='cubic5') == {'weight_decay': 0.1, 'ns': 'cubic5'}
    assert latent_options('lionsr') == {'weight_decay': 0.0, 'skip_idle': True, 'cautious': False}


def test_trajectory_optimizer_steps_e2e_latents_with_muonsr_and_its_options():
    from natlang_neuralese.train.trajectory_state import trajectory_optimizer
    q = nn.Parameter(torch.randn(32, 32).bfloat16())
    router = nn.Parameter(torch.randn(8, 32))
    soft = {'a': nn.Parameter(torch.randn(2, 32))}
    head = nn.Parameter(torch.randn(32, 32))
    latent, _ = latent_split([('backbone.q', q), ('backbone.mlp.gate.weight', router)], 'muonsr')
    optimizer = trajectory_optimizer('muon', soft, [q, router], [head], vocab_size=1000, lr=3e-4, lora_lr=7.5e-6,
                                     heads_lr=3e-4, lora_names=['q', 'mlp.gate.weight'],
                                     lion=latent_entries(latent, 'muonsr', 1e-5), latent_optimizer='muonsr',
                                     latent_options=latent_options('muonsr', weight_decay=0.1, ns='cubic5'))
    assert isinstance(optimizer.latent, MuonSR) and optimizer.latent_key == 'muonsr'
    assert optimizer.latent.ns == 'cubic5' and optimizer.latent.param_groups[0]['weight_decay'] == 0.1
    assert optimizer.latent.param_groups[0]['lr'] == 1e-5
    rates = {id(p): g['lr'] for g in optimizer.param_groups for p in g['params']}
    assert rates[id(router)] == 7.5e-6  # the router stays with the backbone group
    assert optimizer.state_dict()['muonsr'] is not None and 'lion' not in optimizer.state_dict()


def _state(latent_optimizer):
    return {'identity': {'options': {'latent_optimizer': latent_optimizer}}}


def test_a_lionsr_full_state_refuses_to_resume_under_muonsr_in_one_sentence():
    from natlang_neuralese.train.trajectory_state import check_latent_optimizer_change
    check_latent_optimizer_change(_state('muonsr'), 'muonsr', continuation=False)
    with pytest.raises(ValueError) as error:
        check_latent_optimizer_change(_state('lionsr'), 'muonsr', continuation=False)
    message = str(error.value)
    assert 'LionSR' in message and 'MuonSR' in message and message.count('. ') == 0
    with pytest.raises(ValueError, match='--optimizer-state fresh'):
        check_latent_optimizer_change(_state('lionsr'), 'muonsr', continuation=True,
                                      curriculum_change=['latent_optimizer'])
    # A declared switch that starts the optimizer fresh is the explicit way.
    check_latent_optimizer_change(_state('lionsr'), 'muonsr', continuation=True,
                                  curriculum_change=['latent_optimizer'], optimizer_state='fresh')
    # Checkpoints written before the option existed ran the default (Muon) latent policy.
    with pytest.raises(ValueError, match='Muon latents'):
        check_latent_optimizer_change({'identity': {'options': {}}}, 'muonsr', continuation=False)


def test_the_named_restore_also_refuses_a_lionsr_latent_state_under_muonsr():
    from natlang_neuralese.train.optim_restore import OptimizerRestoreError, restore_optimizer_state, optimizer_param_names
    from natlang_neuralese.train.optim import latent_partition
    q = nn.Parameter(torch.randn(32, 32).bfloat16())
    head = nn.Parameter(torch.randn(32, 32))
    named = {'heads.h': head, 'backbone.q': q}
    lion = PortMuonAdamW([('heads.h', head)], lr=1e-3, vocab_size=1000, latent=latent_partition([('backbone.q', q)], 1e-6))
    muon = PortMuonAdamW([('heads.h', head)], lr=1e-3, vocab_size=1000, latent=[('backbone.q', q, 1e-5, 1)],
                         latent_optimizer='muonsr')
    with pytest.raises(OptimizerRestoreError):
        restore_optimizer_state(muon, lion.state_dict(), named, saved_names=optimizer_param_names(lion, named))
    assert restore_optimizer_state(muon, lion.state_dict(), named, saved_names=optimizer_param_names(lion, named),
                                   fresh=True)['mode'] == 'fresh'


def _options(**changes):
    parser = argparse.ArgumentParser()
    from natlang_neuralese.train.drift_monitor import add_drift_arguments
    add_drift_arguments(parser)
    return parser.parse_args(['--drift-warn', '0.5', '--drift-stop', '1.5', *changes.get('argv', [])])


class _Behaviour:
    def __init__(self, gate_pass):
        self.config = {'reference': {'gate_pass': 0.6}, 'max_gate_drop': 0.1}
        self.gate_pass = gate_pass


# Owner decision 2026-10-12: the behaviour gate is report-only. A gate below its floor at a warn is reported
# (gate_below_floor) and the guard stays at warn; only response KL past the hard stop stops.
@pytest.mark.parametrize('kl,gate_pass,guard', [(0.3, 0.5, 'ok'), (0.8, 0.55, 'warn'), (0.8, 0.45, 'warn'),
                                                (1.6, 0.6, 'stop')])
def test_drift_guard_warns_at_half_runs_the_gate_and_stops_at_one_and_a_half(monkeypatch, kl, gate_pass, guard):
    from natlang_neuralese.maple import generation_gate
    from natlang_neuralese.train.drift_monitor import DriftMonitor
    monkeypatch.setattr(generation_gate, 'held_kl', lambda *a, **k: {'held_response_kl': kl,
                                                                    'held_response_agreement': 0.99})
    monitor = DriftMonitor(None, None, {'rows': 1}, _Behaviour(gate_pass), _options())
    monitor.gate = lambda: {'gate_pass': gate_pass, 'gate_checks': 1, 'gate_kinds': {}, 'gate_results': [],
                            'passed': gate_pass >= 0.5}
    drift = monitor.check(10, 100)
    assert drift['guard'] == guard
    assert ('gate_at_warn' in drift) == (0.5 < kl <= 1.5)
    assert drift.get('gate_below_floor', False) == (0.5 < kl <= 1.5 and gate_pass < 0.5)
    assert monitor.state() == ({'last_gate': 10, 'last_gate_kl': kl} if 0.5 < kl <= 1.5
                               else {'last_gate': None, 'last_gate_kl': None})
