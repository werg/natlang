import torch

from natlang_neuralese.maple.ternary import ternarize, ternary_codes


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
