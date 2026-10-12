"""CPU mechanism checks for the explicitly experimental TE FP8 direct-SFT adapter."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from natlang_neuralese.train.fp8_backend import (
    ROW_MULTIPLE, SavedTensorAudit, _eligible_paths, make_te_linear_preserving_rng,
)
from natlang_neuralese.train.direct_sft import direct_sft_calibration_kind


class FakeTELinear(nn.Linear):
    def __init__(self, in_features, out_features, *, bias=True, params_dtype=torch.bfloat16,
                 device=None, save_original_input=False, return_bias=False):
        super().__init__(in_features, out_features, bias=bias, dtype=params_dtype, device=device)
        self.save_original_input = save_original_input
        self.return_bias = return_bias
        self.row_shapes = []

    def forward(self, value):
        self.row_shapes.append(value.shape[0])
        return nn.functional.linear(value, self.weight, self.bias)


class LFMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Module()
        for name in ('w1', 'w2', 'w3'):
            setattr(self.feed_forward, name, nn.Linear(16, 16, bias=False, dtype=torch.bfloat16))
        self.self_attn = nn.Module()
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            setattr(self.self_attn, name, nn.Linear(16, 16, bias=False, dtype=torch.bfloat16))
        self.conv = nn.Module()
        for name in ('in_proj', 'out_proj'):
            setattr(self.conv, name, nn.Linear(16, 16, bias=False, dtype=torch.bfloat16))
        self.norm = nn.LayerNorm(16)


class LFMFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='lfm2')
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([LFMBlock(), LFMBlock()])
        self.model.embed_tokens = nn.Embedding(16, 16)
        self.lm_head = nn.Linear(16, 16, bias=False)


def test_exact_lfm_allowlist_includes_dense_sites_and_excludes_other_ops():
    paths = _eligible_paths(LFMFixture())
    assert len(paths) == 18
    assert 'model.layers.0.feed_forward.w1' in paths
    assert 'model.layers.1.self_attn.out_proj' in paths
    assert 'model.layers.0.conv.in_proj' in paths
    assert not any('norm' in path or 'embed' in path or path.startswith('lm_head') for path in paths)


def test_te_replacement_keeps_master_parameter_objects_and_pads_only_row_axis():
    torch.manual_seed(13)
    source = nn.Linear(16, 32, bias=True, dtype=torch.bfloat16)
    replacement = make_te_linear_preserving_rng(source, FakeTELinear)
    assert replacement.weight is source.weight
    assert replacement.bias is source.bias
    assert replacement.save_original_input is False
    value = torch.randn(2, 3, 16, dtype=torch.bfloat16, requires_grad=True)
    expected = nn.functional.linear(value, source.weight, source.bias)
    actual = replacement(value)
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
    assert replacement.row_shapes == [ROW_MULTIPLE]
    actual.float().square().mean().backward()
    assert value.grad is not None and value.grad.shape == value.shape
    assert source.weight.grad is not None and source.weight.grad.shape == source.weight.shape


def test_fp8_layer_initialization_does_not_advance_rng():
    source = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    before = torch.random.get_rng_state().clone()
    make_te_linear_preserving_rng(source, FakeTELinear)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_saved_tensor_audit_counts_storage_and_preserves_autograd_result():
    value = torch.arange(8.0, requires_grad=True)
    audit = SavedTensorAudit()
    with audit:
        result = value.square().sum()
        result.backward()
    inventory = audit.report()
    assert inventory['unique_storage_count'] >= 1
    assert inventory['unique_storage_bytes_by_dtype']['torch.float32'] >= value.numel() * value.element_size()
    assert value.grad is not None


def test_non_lfm_model_is_rejected_by_allowlist():
    model = LFMFixture()
    model.config.model_type = 'qwen3'
    with pytest.raises(ValueError, match='supports LFM2'):
        _eligible_paths(model)


def test_activation_calibration_namespaces_do_not_mix_fp8_backends():
    assert direct_sft_calibration_kind('off', 'delayed', 'sequential', False) == \
        'direct-sft-sequential-checkpointed-0'
    assert direct_sft_calibration_kind('te-fp8', 'delayed', 'sequential', False) == \
        'direct-sft-fp8-delayed-sequential-checkpointed-0'
    assert direct_sft_calibration_kind('triton-experts', 'delayed', 'sequential', False) == \
        'direct-sft-fp8-experts-sequential-checkpointed-0'
