"""Exact per-position replacement replay using shared full-sequence history."""
import pytest
import torch

from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone, PortCache


def tiny_lfm():
    try:
        from transformers import Lfm2Config, Lfm2ForCausalLM
    except ImportError:
        pytest.skip("installed transformers does not provide the tiny LFM2 model")
    torch.manual_seed(1701)
    config = Lfm2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        block_multiple_of=8, block_auto_adjust_ff_dim=False,
        layer_types=['conv', 'full_attention', 'conv', 'full_attention'],
    )
    model = Lfm2ForCausalLM(config).eval()
    return PortBackbone(model, ControlTokens(62, 63), fast=False)


def _cache_for(backbone, prefix, left_pad=None):
    cache = PortCache.empty(backbone.num_layers)
    _, cache = backbone.run_layers(prefix, range(backbone.num_layers), cache, left_pad=left_pad)
    return cache


def _literal_position_branches(backbone, fixed, replacements, cache, cutoff):
    """Reference: replay fixed prefix/history plus one changed current input per row."""
    history_shallow, _ = backbone.run_layers(fixed, range(cutoff), cache)
    shallow_rows, final_rows = [], []
    for position in range(fixed.shape[1]):
        branch_inputs = torch.cat((fixed[:, :position], replacements[:, position:position + 1]), dim=1)
        shallow, branch_cache = backbone.run_layers(branch_inputs, range(0, cutoff), cache)
        final, _ = backbone.run_layers(shallow, range(cutoff, backbone.num_layers), branch_cache)
        shallow_rows.append(shallow[:, -1])
        final_rows.append(final[:, -1])
    return {
        'history_shallow': history_shallow,
        'shallow': torch.stack(shallow_rows, dim=1),
        'final': torch.stack(final_rows, dim=1),
    }


def _isolated(backbone, fixed, replacements, cache, cutoff):
    method = getattr(backbone, 'isolated_sequence', None)
    if method is None:
        pytest.skip("PortBackbone.isolated_sequence is not installed yet")
    return method(fixed, replacements, cache, cutoff=cutoff)


def _compare_caches(actual, expected):
    assert actual.lengths == expected.lengths
    assert actual.pad_offsets == expected.pad_offsets
    assert len(actual.states) == len(expected.states)
    from natlang_neuralese.model.lfm2_port import AttentionState, ConvState
    for left, right in zip(actual.states, expected.states):
        if isinstance(left, AttentionState):
            assert isinstance(right, AttentionState)
            torch.testing.assert_close(left.k, right.k, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(left.v, right.v, atol=3e-5, rtol=3e-5)
        elif isinstance(left, ConvState):
            assert isinstance(right, ConvState)
            torch.testing.assert_close(left.window, right.window, atol=3e-5, rtol=3e-5)
        else:
            assert left is right is None


def _weighted_loss(out):
    loss = out['history_shallow'].square().mean() * .13
    loss = loss + out['shallow'].square().mean() * .27
    loss = loss + out['final'].square().mean() * .61
    return loss


def _compare_outputs(actual, expected):
    # The isolated attention path uses one sequence query layout while the
    # reference appends one query at a time; allow only their small FP32 kernel
    # summation-order difference.
    for name in ('history_shallow', 'shallow', 'final'):
        torch.testing.assert_close(actual[name], expected[name], atol=3e-5, rtol=3e-5, msg=name)


def test_isolated_sequence_matches_literal_replacement_branches_and_adjoints():
    torch.set_num_threads(1)
    backbone = tiny_lfm()
    cutoff = 2
    prefix = torch.randn(2, 3, 32, requires_grad=True)
    fixed = torch.randn(2, 4, 32, requires_grad=True)
    replacements = torch.randn(2, 4, 32, requires_grad=True)
    cache = _cache_for(backbone, prefix)
    actual = _isolated(backbone, fixed, replacements, cache, cutoff)
    expected = _literal_position_branches(backbone, fixed, replacements, cache, cutoff)
    _compare_outputs(actual, expected)

    parameters = list(backbone.parameters())
    watched = [replacements, fixed, prefix, *parameters]
    actual_grads = torch.autograd.grad(_weighted_loss(actual), watched, allow_unused=True, retain_graph=True)
    expected_grads = torch.autograd.grad(_weighted_loss(expected), watched, allow_unused=True, retain_graph=True)
    for name, got, wanted in zip(['replacements', 'fixed', 'prefix', 'weights'],
                                 actual_grads[:3], expected_grads[:3]):
        assert got is not None and wanted is not None, name
        torch.testing.assert_close(got, wanted, atol=8e-5, rtol=8e-5, msg=name)
    for index, (got, wanted) in enumerate(zip(actual_grads[3:], expected_grads[3:])):
        if got is None or wanted is None:
            assert got is None and wanted is None, f'parameter {index}'
        else:
            torch.testing.assert_close(got, wanted, atol=1e-4, rtol=1e-4, msg=f'parameter {index}')

    # Each output position is completed with only its own replacement tensor.
    for position in range(fixed.shape[1]):
        grad, = torch.autograd.grad(actual['final'][:, position].square().sum(), replacements,
                                    retain_graph=True, allow_unused=True)
        assert grad is not None
        for other in range(fixed.shape[1]):
            if other == position:
                assert grad[:, other].abs().sum() > 0
            else:
                assert not grad[:, other].any(), (position, other)


def test_checkpointed_and_plain_isolated_sequence_match_reference_and_gradients():
    torch.set_num_threads(1)
    backbone = tiny_lfm()
    cutoff = 2
    prefix = torch.randn(1, 2, 32, requires_grad=True)
    fixed = torch.randn(1, 3, 32, requires_grad=True)
    replacements = torch.randn(1, 3, 32, requires_grad=True)
    cache = _cache_for(backbone, prefix)
    outputs, gradients = [], []
    params = [replacements, fixed, prefix, *backbone.parameters()]
    for checkpointed in (False, True):
        backbone.checkpoint_layers = checkpointed
        actual = _isolated(backbone, fixed, replacements, cache, cutoff)
        reference = _literal_position_branches(backbone, fixed, replacements, cache, cutoff)
        _compare_outputs(actual, reference)
        grads = torch.autograd.grad(_weighted_loss(actual), params, allow_unused=True, retain_graph=True)
        ref_grads = torch.autograd.grad(_weighted_loss(reference), params, allow_unused=True, retain_graph=True)
        for got, wanted in zip(grads, ref_grads):
            if got is None or wanted is None:
                assert got is None and wanted is None
            else:
                torch.testing.assert_close(got, wanted, atol=1e-4, rtol=1e-4)
        outputs.append(actual)
        gradients.append(grads)
    _compare_outputs(outputs[0], outputs[1])
    for left, right in zip(gradients[0], gradients[1]):
        if left is None or right is None:
            assert left is None and right is None
        else:
            torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4)


def test_left_padding_cache_is_preserved_and_replacements_do_not_change_history():
    torch.set_num_threads(1)
    backbone = tiny_lfm()
    cutoff = 2
    prefix = torch.randn(2, 3, 32, requires_grad=True)
    left_pad = torch.tensor([1, 0])
    # Position 0 of row 0 is padding; hide it from all cached attention layers.
    prefix = torch.cat((torch.cat((torch.zeros_like(prefix[:1, :1]), prefix[:1, 1:]), dim=1),
                        prefix[1:]), dim=0).detach().requires_grad_(True)
    cache = _cache_for(backbone, prefix, left_pad=left_pad)
    cache_lengths = cache.lengths
    fixed = torch.randn(2, 4, 32)
    replacements_a = torch.randn(2, 4, 32)
    replacements_b = torch.randn(2, 4, 32)
    cache_snapshot = tuple(None if state is None else
                          (state.k.detach().clone(), state.v.detach().clone()) if hasattr(state, 'k')
                          else state.window.detach().clone() for state in cache.states)
    out_a = _isolated(backbone, fixed, replacements_a, cache, cutoff)
    out_b = _isolated(backbone, fixed, replacements_b, cache, cutoff)
    torch.testing.assert_close(out_a['history_shallow'], out_b['history_shallow'], atol=0, rtol=0)
    assert not torch.allclose(out_a['final'], out_b['final'])
    assert cache.lengths == cache_lengths
    assert cache.pad_offsets == (1, 0)
    for state, snapshot in zip(cache.states, cache_snapshot):
        if snapshot is None:
            assert state is None
        elif hasattr(state, 'k'):
            torch.testing.assert_close(state.k, snapshot[0], atol=0, rtol=0)
            torch.testing.assert_close(state.v, snapshot[1], atol=0, rtol=0)
        else:
            torch.testing.assert_close(state.window, snapshot, atol=0, rtol=0)
    _, ordinary_cache = backbone.run_layers(fixed, range(backbone.num_layers), cache)
    _compare_caches(out_a['cache'], ordinary_cache)
    _compare_caches(out_b['cache'], ordinary_cache)
    _compare_outputs(out_a, _literal_position_branches(backbone, fixed, replacements_a, cache, cutoff))


def test_frozen_no_grad_values_match_gradient_enabled_primal():
    torch.set_num_threads(1)
    backbone = tiny_lfm()
    cutoff = 2
    prefix = torch.randn(1, 3, 32)
    fixed = torch.randn(1, 3, 32)
    replacements = torch.randn(1, 3, 32)
    cache = _cache_for(backbone, prefix)
    actual = _isolated(backbone, fixed, replacements, cache, cutoff)
    with torch.no_grad():
        frozen = _isolated(backbone, fixed, replacements, cache, cutoff)
    for name in ('history_shallow', 'shallow', 'final'):
        torch.testing.assert_close(frozen[name], actual[name].detach(), atol=3e-5, rtol=3e-5)
