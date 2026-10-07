"""The FlexAttention branch path of isolated_sequence against its tiled masked SDPA: outputs and all input gradients,
with a cache prefix, a sliding window, left padding, fp32 (LFM) and bf16 (Maple)."""
import pytest
import torch

from natlang_neuralese.model.isolated_sequence import branch_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention branch path is the CUDA path")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("start,steps,window,padded", [(0, 37, None, False), (5, 700, None, True),
                                                       (3, 1100, 64, False), (0, 600, 512, True)])
def test_flex_branch_matches_tiled(dtype, start, steps, window, padded):
    torch.manual_seed(steps)
    batch, heads, groups, dim = 2, 4, 2, 32
    def leaf(*shape):
        return torch.randn(*shape, device="cuda", dtype=dtype, requires_grad=True)
    qb = leaf(batch, heads, steps, dim)
    kh, vh = leaf(batch, groups, start + steps, dim), leaf(batch, groups, start + steps, dim)
    kb, vb = leaf(batch, groups, steps, dim), leaf(batch, groups, steps, dim)
    pad = torch.tensor([3, 0], device="cuda") if padded else None
    leaves = (qb, kh, vh, kb, vb)
    weight = torch.randn(batch, heads, steps, dim, device="cuda")
    results = []
    for flex in (False, True):
        # The NGC image defaults to TF32, so the tiled math-kernel reference
        # itself loses precision. Measure it in IEEE fp32; the production Flex
        # path must match even under the caller's original matmul setting.
        precision = torch.get_float32_matmul_precision()
        try:
            if not flex:
                torch.set_float32_matmul_precision("highest")
            out = branch_attention(qb, kh, vh, kb, vb, start, window, pad, dim ** -.5, flex=flex)
            results.append((out, torch.autograd.grad((out.float() * weight).sum(), leaves)))
        finally:
            torch.set_float32_matmul_precision(precision)
    tol = dict(atol=2e-4, rtol=2e-4) if dtype == torch.float32 else dict(atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(results[1][0].float(), results[0][0].float(), **tol)
    for flexed, tiled in zip(results[1][1], results[0][1]):
        scale = tiled.float().abs().max()
        torch.testing.assert_close(flexed.float() / scale, tiled.float() / scale, **tol)
