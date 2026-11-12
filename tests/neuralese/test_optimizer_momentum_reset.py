"""A policy (RL) stage entered from SFT-fed momentum starts its first moments at zero (w5 RL rounds 2-3, 2026-11-11)."""
import torch

from natlang_neuralese.train.optim import PortMuonAdamW, momentum_reset_needed


def _optimizer():
    torch.manual_seed(3)
    head = torch.nn.Parameter(torch.randn(8) * .1)
    q = torch.nn.Parameter((torch.randn(32, 64) * .02).to(torch.bfloat16))
    optimizer = PortMuonAdamW([('heads.h', head)], lr=1e-3, vocab_size=1000, latent=[('backbone.q', q, 1e-4, 1)],
                              latent_optimizer='muonsr', latent_options={'rounding': 'kahan'})
    return optimizer, head, q


def _step(optimizer, head, q, scale, seed):
    generator = torch.Generator().manual_seed(seed)
    head.grad = torch.randn(head.shape, generator=generator) * scale
    q.grad = (torch.randn(q.shape, generator=generator) * scale).to(torch.bfloat16)
    optimizer.step()


def test_reset_clears_first_moments_only():
    optimizer, head, q = _optimizer()
    for seed in range(4):
        _step(optimizer, head, q, 10., seed)
    aux, latent = optimizer.auxiliary.state[head], optimizer.latent.state[q]
    assert aux['exp_avg'].abs().sum() > 0 and latent['momentum'].abs().sum() > 0
    second, step, compensation = aux['exp_avg_sq'].clone(), aux['step'].clone(), latent['compensation'].clone()
    cleared = optimizer.reset_momentum()
    assert cleared['buffers'] == 2 and cleared['elements'] == head.numel() + q.numel()
    assert aux['exp_avg'].abs().sum() == 0 and latent['momentum'].abs().sum() == 0
    assert torch.equal(aux['exp_avg_sq'], second) and torch.equal(aux['step'], step)
    assert torch.equal(latent['compensation'], compensation)


def test_after_reset_a_small_policy_gradient_steers_muon_instead_of_the_stale_momentum():
    """Muon orthogonalizes g + beta*m: the step has full size whatever g is, so an SFT-sized buffer owns its direction."""
    def run(reset):
        optimizer, head, q = _optimizer()
        for seed in range(6):                               # SFT-sized gradients leave a large momentum
            _step(optimizer, head, q, 10., seed)
        stale = optimizer.latent.state[q]['momentum'].float().clone()
        if reset:
            optimizer.reset_momentum()
        before = (q.float() + optimizer.latent.state[q].get('compensation', 0)).clone()
        generator = torch.Generator().manual_seed(99)
        gradient = (torch.randn(q.shape, generator=generator) * 1e-3).to(torch.bfloat16)   # policy-sized gradient
        for _ in range(3):
            q.grad = gradient.clone()
            head.grad = torch.zeros_like(head)
            optimizer.step()
        moved = (q.float() + optimizer.latent.state[q]['compensation']) - before
        cosine = lambda a, b: float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))
        return cosine(moved, -stale), cosine(moved, -gradient.float())
    stale_cos, gradient_cos = run(reset=False)
    assert abs(stale_cos) > 5 * abs(gradient_cos) and stale_cos > .1, (stale_cos, gradient_cos)
    stale_cos, gradient_cos = run(reset=True)
    assert gradient_cos > .2 and abs(stale_cos) < gradient_cos, (stale_cos, gradient_cos)


def test_only_entering_a_policy_stage_from_other_momentum_resets():
    assert momentum_reset_needed(None, 'policy') and momentum_reset_needed('other', 'policy')
    assert not momentum_reset_needed('policy', 'policy')           # a round continues the previous round's momentum
    assert not momentum_reset_needed('policy', 'other') and not momentum_reset_needed('other', 'other')
    assert not momentum_reset_needed('other', 'policy', keep=True)
