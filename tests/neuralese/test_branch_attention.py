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


@pytest.mark.parametrize('start,steps,window,padded', [
    (0, 1, None, False), (5, 37, None, True), (129, 257, None, True),
    (3, 281, 64, False), (128, 270, 512, True), (9, 131, 1, True)])
def test_branch_block_geometry_matches_exact_token_mask(start, steps, window, padded):
    from natlang_neuralese.model.isolated_sequence import branch_block_mask
    length = start + steps
    pad = torch.tensor([3, 141]) if padded else None
    mask = branch_block_mask(steps, length, start, window, pad, 'cpu')
    q = torch.arange(steps)[None, None, :, None]
    kv = torch.arange(length + steps)[None, None, None, :]
    exact = mask.mask_mod(torch.arange(2 if padded else 1)[:, None, None, None], 0, q, kv)
    padded_exact = torch.nn.functional.pad(exact, (0, (-exact.shape[-1]) % 128, 0, (-steps) % 128))
    expected = padded_exact.reshape(exact.shape[0], 1, (steps + 127)//128, 128,
                                    (length + steps + 127)//128, 128).any(-1).any(-2)
    torch.testing.assert_close(mask.to_dense().bool(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA allocation regression')
def test_long_branch_mask_has_no_quadratic_token_allocation():
    from natlang_neuralese.model.isolated_sequence import branch_block_mask
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    mask = branch_block_mask(16384, 16416, 32, None, None, 'cuda')
    torch.cuda.synchronize()
    assert mask.seq_lengths == (16384, 32800)
    assert torch.cuda.max_memory_allocated() - baseline < 32 * 1024 * 1024


@pytest.mark.skipif(not torch.cuda.is_available(), reason="dispatch of the CUDA branch path")
def test_short_query_span_uses_tiles_not_flex_decode():
    # Under 128 queries Flex would dispatch its decode kernel (incompatible with the 128-token mask blocks).
    from natlang_neuralese.model import isolated_sequence
    calls = []
    original = isolated_sequence.branch_block_mask
    isolated_sequence.branch_block_mask = lambda *a, **k: calls.append(a) or original(*a, **k)
    try:
        q = torch.randn(1, 4, 16, 32, device="cuda", dtype=torch.bfloat16)
        kh = torch.randn(1, 2, 4016, 32, device="cuda", dtype=torch.bfloat16)
        kb = torch.randn(1, 2, 16, 32, device="cuda", dtype=torch.bfloat16)
        out = isolated_sequence.branch_attention(q, kh, kh, kb, kb, 4000, None, None, 32 ** -.5)
    finally:
        isolated_sequence.branch_block_mask = original
    assert out.shape == q.shape and not calls
