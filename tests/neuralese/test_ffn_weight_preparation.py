"""Exact reuse and actual retained FP8 storage through chunked recomputation."""
import gc
import weakref

import pytest
import torch

from natlang_neuralese.model import native_gradient_partitions as partitions


def test_preparation_scope_invalidation_and_independent_calls():
    weight = torch.ones(3, 4)
    calls = []
    def prepare():
        calls.append(weight.clone())
        return calls[-1]
    def invoke(_):
        first = partitions.reuse_weight_preparation(weight, 'fixture', prepare)
        assert partitions.reuse_weight_preparation(weight, 'fixture', prepare) is first
        weight.add_(1)
        second = partitions.reuse_weight_preparation(weight, 'fixture', prepare)
        assert second is not first and torch.equal(second, weight)
        return _
    partitions.token_local_chunks(torch.ones(1, 1, 1), 0, invoke)
    assert len(calls) == 2
    partitions.reuse_weight_preparation(weight, 'fixture', prepare)
    assert len(calls) == 3  # no reuse outside the scope
    partitions.token_local_chunks(torch.ones(1, 1, 1), 0, invoke)
    assert len(calls) == 5


def test_nested_scope_and_exception_release_do_not_keep_prepared_tensors():
    weight = torch.ones(3, 4)
    refs = []
    def prepare():
        value = weight.clone()
        refs.append(weakref.ref(value))
        return value
    def nested(h):
        partitions.reuse_weight_preparation(weight, 'fixture', prepare)
        raise ValueError('fixture')
    def outer(h):
        a = partitions.reuse_weight_preparation(weight, 'fixture', prepare)
        with pytest.raises(ValueError, match='fixture'):
            partitions.token_local_chunks(h, 0, nested)
        assert partitions.reuse_weight_preparation(weight, 'fixture', prepare) is a
        return h
    partitions.token_local_chunks(torch.ones(1, 1, 1), 0, outer)
    gc.collect()
    assert len(refs) == 2 and all(ref() is None for ref in refs)


@pytest.mark.parametrize('checkpointed', [False, True])
def test_actual_expert_chunks_share_preparation_with_identical_vjp(monkeypatch, checkpointed):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from torch.utils.checkpoint import checkpoint
    from natlang_neuralese.maple import expert_gemm as expert
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope

    torch.manual_seed(483)
    experts, width, hidden, tokens = 3, 16, 16, 37
    weight = torch.nn.Parameter(torch.randn(experts, width, hidden, device='cuda', dtype=torch.bfloat16) * .1)
    inputs = torch.randn(tokens, hidden, device='cuda', dtype=torch.bfloat16) * .1
    cotangent = torch.randn(tokens, width, device='cuda', dtype=torch.bfloat16)
    original_prepare = partitions.reuse_weight_preparation
    original_quant = expert.rowwise_quant

    def run(reuse):
        x = inputs.clone().requires_grad_()
        quantizations, live = [], []
        peak = 0
        def quantize(value):
            nonlocal peak
            result = original_quant(value)
            if value.data_ptr() == weight.data_ptr():
                quantizations.append(result[0].data_ptr())
                live.append(weakref.ref(result[0]))
                peak = max(peak, sum(ref() is not None for ref in live))
            return result
        monkeypatch.setattr(expert, 'rowwise_quant', quantize)
        monkeypatch.setattr(partitions, 'reuse_weight_preparation',
            original_prepare if reuse else lambda weight, format_key, prepare: prepare())
        with ParameterCotangentScope([('experts.weight', weight)], mode='vjp',
                                     accumulator_backend='device_fp32') as scope:
            def layer(h):
                def project(part):
                    rows = part[0]
                    offsets = torch.tensor([0, 0, rows.shape[0]], device='cuda', dtype=torch.int32)
                    table = expert.BlockTable(offsets, rows.shape[0])
                    return expert.Fp8ExpertProjection.apply(rows, None, None, 1, weight,
                        table, offsets, None, *scope.anchors_for_weight(weight))[None]
                with partitions.gradient_partitions(partitions.GradientPartitions((tokens,))):
                    return partitions.token_local_chunks(h, 7, project)
            output = checkpoint(layer, x[None], use_reentrant=False) if checkpointed else layer(x[None])
            dx, dw = torch.autograd.grad((output[0] * cotangent).sum(), (x, weight))
            scope.finish()
        return output.detach(), dx.detach(), dw.detach(), len(quantizations), peak

    baseline = run(False)
    reused = run(True)
    for before, after in zip(baseline[:3], reused[:3]):
        assert torch.equal(before, after)
    assert baseline[3] == (12 if checkpointed else 6)
    assert reused[3] == (2 if checkpointed else 1)
    assert reused[4] < baseline[4], (baseline[4], reused[4])
