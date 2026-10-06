import pytest
import torch
from torch import nn

from natlang_neuralese.maple.ternary import (
    QATTernaryLoRA, add_qat_lora, export_ternary, freeze_ternary, qat_adapters, ternarize, ternary_codes,
    tq2_0_exact, tq2_0_roundtrip,
)


def test_rule_by_hand():
    w = torch.tensor([[1.0, -0.5, 0.1, 0.0], [0.0, 0.0, 0.0, 0.0]])
    codes, alpha = ternary_codes(w)
    # mean|w| = 0.4, threshold 0.28: keeps 1.0 and -0.5, alpha = 0.75
    assert codes.tolist() == [[1, -1, 0, 0], [0, 0, 0, 0]]
    assert alpha.float().flatten().tolist() == [0.75, 0.0]
    assert ternarize(w).tolist() == [[0.75, -0.75, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_threshold_is_strict_and_uses_bf16_rounding():
    w = torch.tensor([[0.7, 1.0, 1.3, 0.0]])  # mean 0.75, threshold 0.525
    assert ternary_codes(w)[0].tolist() == [[1, 1, 1, 0]]
    w = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # threshold 0.7: all kept
    assert ternary_codes(w)[0].tolist() == [[1, 1, 1, 1]]
    w = torch.tensor([[0.7, 0.7, 0.7, 0.7]])  # |w| equals mean; 0.7*mean < |w|: kept
    assert ternary_codes(w)[0].tolist() == [[1, 1, 1, 1]]
    # a value that differs from its BF16 rounding is judged after rounding
    x = torch.tensor([[1.0 + 2**-10, 1.0, 1.0, 1.0]])
    assert ternary_codes(x)[1].float().item() == 1.0


def test_qat_lora_starts_at_deployed_model_and_trains_through_quantizer():
    torch.manual_seed(0)
    layer = nn.Linear(512, 8, bias=False)
    base = layer.weight.detach().clone()
    adapter = add_qat_lora(layer, rank=4, alpha=8)
    assert not layer.parametrizations.weight.original.requires_grad
    assert torch.equal(layer.weight, ternarize(base))
    x = torch.randn(3, 512)
    layer(x).pow(2).sum().backward()
    assert adapter.lora_B.grad is not None and adapter.lora_B.grad.abs().sum() > 0
    with torch.no_grad():
        adapter.lora_B.normal_(0, 0.05)
    expected = ternarize(base + adapter.delta())
    assert torch.equal(layer.weight, expected)
    assert not torch.equal(layer.weight, ternarize(base) + adapter.delta())
    assert set(qat_adapters(nn.Sequential(layer))) == {"0"}
    assert torch.equal(export_ternary(base, adapter), expected)


def test_straight_through_gradient_is_identity():
    adapter = QATTernaryLoRA(4, 256, rank=2)
    base = torch.randn(4, 256, requires_grad=True)
    out = adapter(base)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    assert torch.allclose(base.grad, upstream)


def test_frozen_ternary():
    layer = nn.Linear(256, 4, bias=False)
    base = layer.weight.detach().clone()
    freeze_ternary(layer)
    assert torch.equal(layer.weight, ternarize(base))


def test_tq2_0_exact_on_maple_form_only():
    torch.manual_seed(1)
    w = torch.randn(16, 1024) * 0.02
    assert not tq2_0_exact(w)
    t = export_ternary(w)
    assert tq2_0_exact(t)
    # raw weights through TQ2_0 lose most of the small ones: a different model
    assert (tq2_0_roundtrip(w) != 0).float().mean() < (t != 0).float().mean()


def test_tq2_0_rounds_halves_away_from_zero():
    w = torch.zeros(1, 256)
    w[0, 0], w[0, 1], w[0, 2] = 1.0, 0.5, -0.5
    assert tq2_0_roundtrip(w)[0, :3].tolist() == [1.0, 1.0, -1.0]


def test_flip_fraction():
    from natlang_neuralese.maple.ternary import flip_fraction

    layer = nn.Linear(256, 4, bias=False)
    adapter = add_qat_lora(layer, rank=2)
    assert flip_fraction(layer) == 0.0
    with torch.no_grad():
        adapter.lora_B.fill_(1.0)
    assert flip_fraction(layer) > 0.0


def test_learned_block_scales_start_exact_train_and_export_to_tq2_0():
    from natlang_neuralese.maple.ternary import adapters_disabled

    torch.manual_seed(3)
    layer = nn.Linear(512, 6, bias=False)
    base = ternarize(layer.weight.detach().clone())
    with torch.no_grad():
        layer.weight.copy_(base)
    adapter = add_qat_lora(layer, rank=2)
    scale = adapter.learn_scales(base)
    assert scale.shape == (6, 2)
    assert torch.allclose(layer.weight, base, atol=1e-3)  # FP16 rounding of the row scale only
    layer(torch.randn(3, 512)).pow(2).sum().backward()
    assert scale.grad is not None and scale.grad.abs().sum() > 0
    with torch.no_grad():
        scale.mul_(torch.rand_like(scale) + 0.5)
    exported = export_ternary(base, adapter)
    assert tq2_0_exact(exported)
    assert torch.allclose(layer.weight.float(), exported, atol=1e-6)
    with adapters_disabled():
        assert torch.equal(layer.weight, ternarize(base))


def test_private_deltas_apply_only_to_their_size():
    from natlang_neuralese.maple.ternary import STATE

    layer = nn.Linear(256, 4, bias=False)
    adapter = add_qat_lora(layer, rank=2)
    private = adapter.add_private(32, rank=2, alpha=4.0)
    with torch.no_grad():
        private.lora_B.fill_(0.5)
    shared = layer.weight.clone()
    STATE["size"] = 32
    try:
        member = layer.weight.clone()
    finally:
        STATE["size"] = None
    assert not torch.equal(shared, member)
    assert torch.equal(layer.weight, shared)
    assert torch.equal(export_ternary(layer.parametrizations.weight.original, adapter, size=32).to(member.dtype),
                       member)


@pytest.mark.parametrize("learned", [False, True])
def test_qat_weight_cache_is_exact_and_follows_updates(learned):
    """The cached quantized weight equals a fresh computation, gives the same gradients, and is recomputed after an
    in-place parameter update (an optimizer step)."""
    from natlang_neuralese.maple.ternary import QATTernaryLoRA

    torch.manual_seed(0)
    base = torch.randn(16, 512)
    adapter = QATTernaryLoRA(16, 512, rank=4)
    with torch.no_grad():
        adapter.lora_B.normal_(0, 0.05)
    if learned:
        adapter.learn_scales(base)

    def fresh():
        adapter.__dict__.pop("_cache", None)
        return adapter(base)

    def grads(out):
        params = [adapter.lora_A, adapter.lora_B] + ([adapter.learned_scale] if learned else [])
        return torch.autograd.grad((out.float() * torch.linspace(-1, 1, out.numel()).view_as(out)).sum(), params)

    reference = fresh()
    cached = adapter(base)  # served from the cache built by `fresh`
    assert torch.equal(cached, reference)
    for a, b in zip(grads(cached), grads(fresh())):
        torch.testing.assert_close(a, b)
    with torch.no_grad():
        assert torch.equal(adapter(base), reference)
        adapter.lora_B.add_(0.5)  # optimizer-style in-place update
    after = adapter(base)
    assert not torch.equal(after, reference)
    assert torch.equal(after, fresh())
