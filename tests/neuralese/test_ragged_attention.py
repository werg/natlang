from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from natlang_neuralese.model.lfm2_port import PortBackbone, _left_pad_mask


@pytest.mark.parametrize('steps,total,pads', [(9, 9, [0, 3]), (3, 9, [0, 3]), (3, 9, [8, 9])])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_unpadded_attention_matches_valid_outputs_and_gradients(steps, total, pads, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(0)
    # Double precision checks mathematical gradient parity independently of
    # different fused kernels' float32/TF32 rounding on the NVIDIA image.
    q = torch.randn(2, 4, steps, 8, device=device, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, 2, total, 8, device=device, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, total, 8, device=device, dtype=torch.float64, requires_grad=True)
    pad = torch.tensor(pads, device=device)
    attn = SimpleNamespace(scaling=8**-.5)
    actual = PortBackbone._attend_ragged(attn, q, k, v, left_pad=pad)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=_left_pad_mask(pad, steps, total, q.device),
                                             scale=attn.scaling, enable_gqa=True)
    valid = (torch.arange(total-steps, total, device=device)[None] >= pad[:, None])[:, None, :, None]
    torch.testing.assert_close(actual * valid, expected * valid)
    ga = torch.autograd.grad((actual * valid).square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad((expected * valid).square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, atol=2e-6, rtol=2e-5)


def test_unpadded_right_padding_matches_real_positions():
    torch.manual_seed(1)
    q = torch.randn(2, 4, 9, 8)
    k = torch.randn(2, 2, 9, 8)
    v = torch.randn(2, 2, 9, 8)
    padding = torch.tensor([[1]*9, [1]*6+[0]*3])
    actual = PortBackbone._attend_ragged(SimpleNamespace(scaling=8**-.5), q, k, v, right_padding=padding)
    for b, n in enumerate([9, 6]):
        expected = F.scaled_dot_product_attention(q[b:b+1, :, :n], k[b:b+1, :, :n], v[b:b+1, :, :n],
                                                 is_causal=True, scale=8**-.5, enable_gqa=True)
        torch.testing.assert_close(actual[b:b+1, :, :n], expected)
