"""The in-backward latent gradient clip (train/optim.py), the shared latent split used by the direct and the end-to-end
stage, the E2E stage's MuonSR optimizer and its resume refusal (train/trajectory_state.py), and the shared drift
monitor's guard (train/drift_monitor.py), on small CPU tensors."""
import argparse
import math

import pytest
import torch
from torch import nn

from natlang_neuralese.train.optim import (MuonSR, PortMuonAdamW, latent_entries, latent_options, latent_split)


def _muonsr(weight, clip):
    optimizer = MuonSR([{'params': [weight], 'lr': 1e-2}], lr=1e-2)
    optimizer.step_in_backward(gated=True, clip=clip)
    return optimizer


def _update(optimizer, weight, grad):
    with optimizer.in_backward() as armed:
        (weight * grad).sum().backward()
    return armed


def test_the_first_update_is_unclipped_and_later_ones_use_the_previous_global_norm():
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(8, 8))
    optimizer = _muonsr(weight, clip=1.0)
    grad = torch.randn(8, 8) * 3
    armed = _update(optimizer, weight, grad)
    assert armed.coefficient == 1.0 and armed.estimate is None
    # The accumulated norm is the unclipped one.
    assert math.isclose(float(armed.norm()), float(grad.norm()), rel_tol=1e-5)
    record = armed.clip_record(float(armed.norm()) * 2)  # with other trainables, the global norm is larger
    assert record['latent_coefficient'] == 1.0
    assert math.isclose(record['exact_coefficient'], 1.0 / (2 * float(grad.norm())), rel_tol=1e-4)
    assert optimizer.global_norm_estimate == pytest.approx(2 * float(grad.norm()))
    # The next update's latents are scaled by min(1, clip / previous global norm) before they step.
    seen = {}
    original = optimizer._update

    def spy(p, group):
        seen['grad'] = p.grad.detach().clone()
        return original(p, group)
    optimizer._update = spy
    second = torch.randn(8, 8)
    armed = _update(optimizer, weight, second)
    factor = 1.0 / (2 * float(grad.norm()) + 1e-6)
    assert armed.coefficient == pytest.approx(factor)
    assert torch.allclose(seen['grad'], second * factor, atol=1e-6)
    assert math.isclose(float(armed.norm()), float(second.norm()), rel_tol=1e-5)  # still the unclipped norm


def test_no_clip_leaves_gradients_and_records_factor_one():
    weight = nn.Parameter(torch.randn(8, 8))
    optimizer = _muonsr(weight, clip=0.0)
    armed = _update(optimizer, weight, torch.randn(8, 8) * 100)
    record = armed.clip_record(armed.norm())
    assert record['latent_coefficient'] == 1.0 and record['exact_coefficient'] == 1.0 and record['clip'] == 0.0
    armed = _update(optimizer, weight, torch.randn(8, 8) * 100)
    assert armed.coefficient == 1.0


def test_a_small_previous_norm_does_not_scale_up():
    weight = nn.Parameter(torch.randn(8, 8))
    optimizer = _muonsr(weight, clip=1.0)
    _update(optimizer, weight, torch.randn(8, 8)).clip_record(0.25)
    assert _update(optimizer, weight, torch.randn(8, 8)).coefficient == 1.0


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


@pytest.mark.parametrize('kl,gate_pass,guard', [(0.3, 0.5, 'ok'), (0.8, 0.55, 'warn'), (0.8, 0.45, 'stop'),
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
    assert monitor.state() == ({'last_gate': 10, 'last_gate_kl': kl} if 0.5 < kl <= 1.5
                               else {'last_gate': None, 'last_gate_kl': None})
