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
