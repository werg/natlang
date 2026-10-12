"""LionSR's idle-expert skipping and cautious mask, and AdamMiniSR (train/optim.py), on small CPU tensors."""
import pytest
import torch
from torch import nn

from natlang_neuralese.train.optim import (AdamMiniSR, LionSR, PortMuonAdamW, beta_for_half_life, latent_block_rows,
                                           stack_activity)


def _stack(seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    return nn.Parameter(torch.randn(3, 4, 6).to(dtype))


def _grad_with_idle(expert=1, seed=1):
    torch.manual_seed(seed)
    grad = torch.randn(3, 4, 6)
    grad[expert] = 0
    return grad


def _lion(weight, path, **options):
    group = {'params': [weight], 'lr': 1e-2, 'weight_decay': 0.1}
    if path == 'rows':
        group['row_scale'] = torch.rand(*weight.shape[:-1], 1) + .5
    return LionSR([group], lr=1e-2, chunk=48 if path == 'chunked' else 1 << 26, **options)


def test_stack_activity_marks_matrices_with_any_nonzero_entry():
    grad = torch.zeros(2, 3, 4, 5)
    grad[0, 1, 2, 3] = -1e-30
    grad[1, 2, 0, 0] = 1.
    assert stack_activity(grad).tolist() == [[False, True, False], [False, False, True]]


@pytest.mark.parametrize('path', ['flat', 'chunked', 'rows', 'in_backward'])
def test_lionsr_leaves_an_idle_expert_untouched_by_default(path):
    """An expert routed no tokens (exactly zero gradient slice) gets no update, no weight decay, and keeps its momentum,
    on every LionSR path; the routed experts move."""
    weight = _stack()
    optimizer = _lion(weight, 'rows' if path == 'rows' else path)
    optimizer.state[weight]['momentum'] = torch.randn(3, 4, 6).bfloat16()  # a history from earlier updates
    before, momentum = weight.detach().clone(), optimizer.state[weight]['momentum'].clone()
    grad = _grad_with_idle()
    if path == 'in_backward':
        optimizer.step_in_backward(gated=True)
        with optimizer.in_backward():
            (weight.float() * grad).sum().backward()
    else:
        weight.grad = grad.bfloat16()
        optimizer.step()
    assert torch.equal(weight.detach()[1], before[1])
    assert torch.equal(optimizer.state[weight]['momentum'][1], momentum[1])
    assert not torch.equal(weight.detach()[0], before[0]) and not torch.equal(weight.detach()[2], before[2])
    assert not torch.equal(optimizer.state[weight]['momentum'][0], momentum[0])
    assert optimizer.idle_updates[id(weight)].tolist() == [0, 1, 0]


def test_lionsr_without_idle_skipping_moves_an_idle_expert_at_full_rate():
    """The defect the default fixes: with a momentum history, Lion's sign moves a zero-gradient expert by lr."""
    weight = _stack(dtype=torch.float32)
    optimizer = _lion(weight, 'flat', skip_idle=False)
    optimizer.param_groups[0]['weight_decay'] = 0.0
    optimizer.state[weight]['momentum'] = torch.randn(3, 4, 6).bfloat16()
    before = weight.detach().clone()
    weight.grad = _grad_with_idle()
    optimizer.step()
    assert torch.allclose((weight.detach()[1] - before[1]).abs(), torch.full((4, 6), 1e-2))
    assert optimizer.idle_updates[id(weight)].tolist() == [0, 1, 0]  # still counted


def test_lionsr_dense_matrices_follow_the_unchanged_lion_rule():
    """2-D latents (no stack): sign(b1 m + (1 - b1) g) with decoupled decay, as before the change."""
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(5, 7))
    momentum = torch.randn(5, 7).bfloat16()
    grad = torch.randn(5, 7)
    optimizer = LionSR([weight], lr=1e-2, weight_decay=0.1)
    optimizer.state[weight]['momentum'] = momentum.clone()
    expected = weight.detach() * (1 - 1e-3) - 1e-2 * torch.sign(.9 * momentum.float() + .1 * grad)
    weight.grad = grad
    optimizer.step()
    assert torch.allclose(weight.detach(), expected, atol=1e-6)
    assert torch.equal(optimizer.state[weight]['momentum'], (.99 * momentum.float() + .01 * grad).bfloat16())


@pytest.mark.parametrize('path', ['flat', 'rows'])
def test_lionsr_cautious_moves_only_coordinates_whose_update_agrees_with_the_gradient(path):
    """Cautious mask without renormalization: a coordinate whose sign(update) disagrees with sign(g) (or g == 0) stays;
    the others move by exactly lr (x row scale)."""
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(4, 6))
    group = {'params': [weight], 'lr': 1e-2, 'weight_decay': 0.0}
    scale = torch.ones(4, 1)
    if path == 'rows':
        scale = torch.rand(4, 1) + .5
        group['row_scale'] = scale
    optimizer = LionSR([group], lr=1e-2, cautious=True)
    momentum = torch.randn(4, 6).bfloat16()
    optimizer.state[weight]['momentum'] = momentum.clone()
    grad = torch.randn(4, 6)
    grad[0, 0] = 0
    update = torch.sign(.9 * momentum.float() + .1 * grad)
    agree = update * grad > 0
    before = weight.detach().clone()
    weight.grad = grad
    optimizer.step()
    moved = before - weight.detach()
    assert torch.equal(moved[~agree], torch.zeros_like(moved[~agree]))
    assert torch.allclose(moved[agree], (1e-2 * scale.expand(4, 6) * update)[agree], atol=1e-6)
    assert bool((~agree).any()) and bool(agree.any()) and not bool(agree[0, 0])


def test_lionsr_options_survive_a_state_reload_and_the_state_format_is_unchanged():
    weight = _stack()
    optimizer = _lion(weight, 'flat', cautious=True)
    weight.grad = _grad_with_idle().bfloat16()
    optimizer.step()
    saved = optimizer.state_dict()
    assert set(saved['state'][0]) == {'momentum'}
    assert all('cautious' not in group and 'skip_idle' not in group for group in saved['param_groups'])
    fresh = _lion(nn.Parameter(weight.detach().clone()), 'flat', cautious=True)
    fresh.load_state_dict(saved)
    assert fresh.cautious and fresh.skip_idle


def _adam_reference(weight, grad, lr, block_rows, eps=1e-8):
    """Adam-mini's first step: m_hat = g, v_hat = the block's mean of g^2."""
    lead, rows, cols = grad.shape[:-2], grad.shape[-2], grad.shape[-1]
    blocks = grad.reshape(*lead, rows // block_rows, block_rows * cols)
    v = blocks.square().mean(-1, keepdim=True)
    return weight - lr * (blocks / (v.sqrt() + eps)).reshape(grad.shape)


def test_adaminisr_first_step_is_the_gradient_over_its_blocks_rms():
    torch.manual_seed(0)
    weight = nn.Parameter(torch.randn(8, 5))
    grad = torch.randn(8, 5) * torch.arange(1, 9).float().view(8, 1)
    optimizer = AdamMiniSR([{'params': [weight], 'block_rows': 4}], lr=1e-3)
    expected = _adam_reference(weight.detach(), grad, 1e-3, 4)
    weight.grad = grad
    optimizer.step()
    assert torch.allclose(weight.detach(), expected, atol=1e-6)
    assert optimizer.state[weight]['v'].shape == (2, 1) and optimizer.state[weight]['v'].dtype == torch.float32
    assert optimizer.state[weight]['momentum'].dtype == torch.bfloat16


def test_adaminisr_scales_the_step_with_the_gradient_across_steps():
    """Unlike Lion's sign, a block whose gradient shrinks moves less (second step of a decaying gradient)."""
    torch.manual_seed(0)
    weight = nn.Parameter(torch.zeros(2, 4))
    optimizer = AdamMiniSR([weight], lr=1e-2, betas=(0.0, 0.5))
    weight.grad = torch.ones(2, 4)
    optimizer.step()
    first = weight.detach().clone()
    weight.grad = torch.tensor([[1.] * 4, [.01] * 4])
    optimizer.step()
    second = weight.detach() - first
    assert second[1].abs().max() < .05 * second[0].abs().max()


@pytest.mark.parametrize('in_backward', [False, True])
def test_adaminisr_leaves_an_idle_expert_and_its_moments_and_count_untouched(in_backward):
    weight = _stack()
    optimizer = AdamMiniSR([weight], lr=1e-3)
    weight.grad = torch.randn(3, 4, 6).bfloat16()
    optimizer.step()
    weight.grad = None
    state = {k: v.clone() for k, v in optimizer.state[weight].items()}
    before = weight.detach().clone()
    grad = _grad_with_idle()
    if in_backward:
        optimizer.step_in_backward(gated=True)
        with optimizer.in_backward():
            (weight.float() * grad).sum().backward()
    else:
        weight.grad = grad.bfloat16()
        optimizer.step()
    after = optimizer.state[weight]
    assert torch.equal(weight.detach()[1], before[1])
    for key in ('momentum', 'v', 'step'):
        assert torch.equal(after[key][1], state[key][1])
        assert not torch.equal(after[key][0], state[key][0])
    assert after['step'].view(-1).tolist() == [2., 1., 2.]
    assert not torch.equal(weight.detach()[0], before[0])


def test_adaminisr_resume_keeps_fp32_second_moments_and_continues_identically():
    weights = [_stack(), _stack()]
    optimizers = [AdamMiniSR([w], lr=1e-3, cautious=True) for w in weights]
    torch.manual_seed(3)
    grads = [torch.randn(3, 4, 6).bfloat16() for _ in range(3)]
    for w, o in zip(weights, optimizers):
        torch.manual_seed(5)
        w.grad = grads[0]
        o.step()
    saved = optimizers[0].state_dict()
    resumed = AdamMiniSR([weights[0]], lr=1e-3, cautious=True)
    resumed.load_state_dict(saved)
    assert resumed.state[weights[0]]['v'].dtype == torch.float32
    assert torch.equal(resumed.state[weights[0]]['v'], optimizers[1].state[weights[1]]['v'])
    for grad in grads[1:]:
        torch.manual_seed(7)
        weights[0].grad = grad
        resumed.step()
        torch.manual_seed(7)
        weights[1].grad = grad
        optimizers[1].step()
    assert torch.equal(weights[0].detach(), weights[1].detach())


def test_adamini_partition_and_half_life():
    q = torch.zeros(4 * 16, 32)
    assert latent_block_rows('backbone.model.layers.0.self_attn.q_proj.weight', q, 16) == 16
    assert latent_block_rows('backbone.model.layers.0.self_attn.k_proj.weight', q, 16) == 16
    assert latent_block_rows('backbone.model.layers.0.self_attn.v_proj.weight', q, 16) == 1
    assert latent_block_rows('backbone.model.layers.0.mlp.experts.gate_up', torch.zeros(8, 6, 4), 16) == 1
    beta = beta_for_half_life(8192, 4e6)
    assert abs(beta ** (4e6 / 8192) - .5) < 1e-9


def test_port_optimizer_routes_latents_to_adaminisr_with_its_own_state_key():
    torch.manual_seed(0)
    hidden = nn.Parameter(torch.randn(6, 6))
    q = nn.Parameter(torch.randn(8, 6).bfloat16())
    experts = nn.Parameter(torch.randn(3, 4, 6).bfloat16())
    optimizer = PortMuonAdamW([('heads.hidden', hidden)], lr=1e-3, vocab_size=100,
                              latent=[('backbone.layers.0.self_attn.q_proj.weight', q, 1e-4, 4),
                                      ('backbone.layers.0.mlp.experts.down', experts, 1e-4, 1)],
                              latent_optimizer='adaminisr', latent_options={'cautious': True, 'betas': (.9, .99)})
    assert isinstance(optimizer.latent, AdamMiniSR) and optimizer.latent.cautious and optimizer.latent.skip_idle
    assert [g['block_rows'] for g in optimizer.latent.param_groups] == [4, 1]
    optimizer.latent.step_in_backward(gated=True)
    with optimizer.latent.in_backward():
        ((q.float() ** 2).sum() + (experts.float() ** 2).sum() + (hidden ** 2).sum()).backward()
    optimizer.step()
    state = optimizer.state_dict()
    assert 'adamini' in state and state['schema'][-1]['optimizer'] == 'adamini'
    again = PortMuonAdamW([('heads.hidden', hidden)], lr=1e-3, vocab_size=100,
                          latent=[('backbone.layers.0.self_attn.q_proj.weight', q, 1e-4, 4),
                                  ('backbone.layers.0.mlp.experts.down', experts, 1e-4, 1)],
                          latent_optimizer='adaminisr')
    again.load_state_dict(state)
    assert again.latent.state[q]['v'].dtype == torch.float32


def test_checkpoint_reserve_sizes_adaminisr_and_muonsr_latents():
    from natlang_neuralese.train.checkpoint_safety import optimizer_state_size_upper_bound
    from natlang_neuralese.train.optim import MuonSR
    experts = nn.Parameter(torch.randn(3, 4, 6).bfloat16())
    assert optimizer_state_size_upper_bound(AdamMiniSR([experts], lr=1e-3)) == 72 * 2 + 12 * 4 + 3 * 4
    assert optimizer_state_size_upper_bound(MuonSR([experts], lr=1e-3)) == 72 * 2


def test_cubic5_newton_schulz_orthogonalizes_each_expert_like_the_quintic():
    from natlang_neuralese.train.optim import newton_schulz
    torch.manual_seed(0)
    stacked = torch.randn(3, 12, 20)
    cubic = newton_schulz(stacked, variant='cubic5')
    singular = torch.linalg.svdvals(cubic.float())
    assert singular.min() > 0.3 and singular.max() < 1.5
    for e in range(3):
        assert torch.allclose(cubic[e].float(), newton_schulz(stacked[e], variant='cubic5').float(), atol=1e-2)
    tall = newton_schulz(stacked.transpose(-2, -1), variant='cubic5')
    assert torch.allclose(tall.float(), cubic.transpose(-2, -1).float(), atol=1e-2)
    with pytest.raises(ValueError):
        newton_schulz(stacked, variant='septic')


def test_muonsr_gives_an_idle_expert_no_weight_decay_and_keeps_ns_across_reload():
    from natlang_neuralese.train.optim import MuonSR
    weight = _stack(dtype=torch.float32)
    optimizer = MuonSR([{'params': [weight], 'weight_decay': 0.1}], lr=1e-2, ns='cubic5')
    before = weight.detach().clone()
    weight.grad = _grad_with_idle()
    optimizer.step()
    assert torch.equal(weight.detach()[1], before[1])
    assert not torch.equal(weight.detach()[0], before[0])
    again = MuonSR([{'params': [weight], 'weight_decay': 0.1}], lr=1e-2, ns='cubic5')
    again.load_state_dict(optimizer.state_dict())
    assert again.ns == 'cubic5'


def test_port_optimizer_passes_latent_weight_decay_and_ns_to_muonsr():
    from natlang_neuralese.train.optim import MuonSR
    hidden = nn.Parameter(torch.randn(6, 6))
    experts = nn.Parameter(torch.randn(3, 4, 6).bfloat16())
    optimizer = PortMuonAdamW([('heads.hidden', hidden)], lr=1e-3, vocab_size=100,
                              latent=[('backbone.layers.0.mlp.experts.down', experts, 1e-4, None)],
                              latent_optimizer='muonsr', latent_options={'weight_decay': 0.1, 'ns': 'cubic5'})
    assert isinstance(optimizer.latent, MuonSR) and optimizer.latent.ns == 'cubic5'
    assert optimizer.latent.param_groups[0]['weight_decay'] == 0.1


def test_muonsr_split_rows_orthogonalizes_each_row_block_separately():
    """A fused [gate; up] expert matrix with split_rows 2 steps exactly like the two matrices stepped separately."""
    from natlang_neuralese.train.optim import MuonSR
    torch.manual_seed(0)
    fused = nn.Parameter(torch.randn(2, 8, 12))
    gate, up = (nn.Parameter(fused.detach()[:, :4].clone()), nn.Parameter(fused.detach()[:, 4:].clone()))
    grad = torch.randn(2, 8, 12)
    optimizer = MuonSR([{'params': [fused], 'split_rows': 2}], lr=1e-2)
    fused.grad = grad.clone()
    optimizer.step()
    separate = MuonSR([gate, up], lr=1e-2)
    gate.grad, up.grad = grad[:, :4].clone(), grad[:, 4:].clone()
    separate.step()
    assert torch.allclose(fused.detach(), torch.cat([gate.detach(), up.detach()], 1), atol=1e-6)
    with pytest.raises(ValueError):
        MuonSR([{'params': [nn.Parameter(torch.randn(7, 4))], 'split_rows': 2}], lr=1e-2)


def test_folded_cubic5_newton_schulz_is_at_least_as_close_to_the_fp64_iteration(monkeypatch):
    """The folded cubic step (M = aI + b X X^T by one baddbmm, then M X) against the same iteration in FP64: no less
    accurate than the unfolded BF16 form it replaces (wide and tall stacks, split rows as MuonSR passes them)."""
    from natlang_neuralese.train.optim import CUBIC5_COEFFICIENTS, newton_schulz

    def fp64(update):
        x = update.double()
        transposed = x.shape[-2] > x.shape[-1]
        x = x.transpose(-2, -1) if transposed else x
        x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + 1e-7)
        for a, b in CUBIC5_COEFFICIENTS:
            x = a * x + b * ((x @ x.transpose(-2, -1)) @ x)
        return x.transpose(-2, -1) if transposed else x

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for shape in ((4, 2, 32, 80), (4, 1, 80, 32)):
        update = torch.randn(*shape, device=device)
        reference = fp64(update)
        monkeypatch.setenv('NATLANG_MUON_NS_FOLDED', '1')
        folded = newton_schulz(update, variant='cubic5').double()
        monkeypatch.setenv('NATLANG_MUON_NS_FOLDED', '0')
        plain = newton_schulz(update, variant='cubic5').double()
        error = lambda x: float((x - reference).norm() / reference.norm())
        assert folded.shape == plain.shape == update.shape
        assert error(folded) <= 1.1 * error(plain) and error(folded) < 0.03, (shape, error(folded), error(plain))
