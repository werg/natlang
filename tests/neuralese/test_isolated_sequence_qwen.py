"""Qwen/Maple ``isolated_sequence`` against literal per-position replacement branches (the LFM checks of
test_isolated_sequence.py on the Qwen-family port): primal, adjoints, isolation, checkpointing, cache and padding."""
import pytest
import torch

from natlang_neuralese.maple.maple_port import MaplePortBackbone
from natlang_neuralese.maple.model import load_maple
from natlang_neuralese.model.hf_port import QwenPortBackbone
from natlang_neuralese.model.lfm2_port import ControlTokens, PortCache

from test_isolated_sequence import _compare_caches, _compare_outputs, _literal_position_branches, _weighted_loss
from test_maple_model import _reference, _save


def _qwen(sliding):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    kwargs = {}
    if sliding:
        kwargs = dict(use_sliding_window=True, sliding_window=sliding, max_window_layers=0,
                      layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"])
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
                         max_position_embeddings=128, attn_implementation="eager", **kwargs)
    torch.manual_seed(0)
    return QwenPortBackbone(Qwen3ForCausalLM(config).eval(), ControlTokens(open_id=62, close_id=63), fast=False)


def _maple(tmp_path):
    reference, config = _reference()
    model = load_maple(_save(reference, config, tmp_path / "ckpt"), dtype=torch.float32)
    return MaplePortBackbone(model, ControlTokens(open_id=94, close_id=95))


@pytest.fixture(params=["qwen", "qwen-sliding", "maple"])
def backbone(request, tmp_path):
    torch.set_num_threads(1)
    if request.param == "maple":
        port = _maple(tmp_path)
    else:
        port = _qwen(3 if request.param == "qwen-sliding" else None)
    for parameter in port.hf.parameters():
        parameter.requires_grad_(True)
    return port


def _cache(backbone, prefix, left_pad=None):
    _, cache = backbone.run_layers(prefix, range(backbone.num_layers), PortCache.empty(backbone.num_layers),
                                   left_pad=left_pad)
    return cache


def test_matches_literal_branches_adjoints_and_isolation(backbone):
    dim = backbone.embedding_weight.shape[1]
    torch.manual_seed(3)
    prefix = torch.randn(2, 3, dim, requires_grad=True)
    fixed = torch.randn(2, 5, dim, requires_grad=True)  # 8 positions: past the sliding windows (3 and 6)
    replacements = torch.randn(2, 5, dim, requires_grad=True)
    cache = _cache(backbone, prefix)
    actual = backbone.isolated_sequence(fixed, replacements, cache, cutoff=2)
    expected = _literal_position_branches(backbone, fixed, replacements, cache, 2)
    _compare_outputs(actual, expected)
    _, ordinary = backbone.run_layers(fixed, range(backbone.num_layers), cache)
    _compare_caches(actual["cache"], ordinary)
    watched = [replacements, fixed, prefix, *[p for p in backbone.hf.parameters() if p.requires_grad]]
    got = torch.autograd.grad(_weighted_loss(actual), watched, allow_unused=True, retain_graph=True)
    want = torch.autograd.grad(_weighted_loss(expected), watched, allow_unused=True, retain_graph=True)
    for index, (left, right) in enumerate(zip(got, want)):
        if left is None or right is None:
            assert left is None and right is None, index
        else:
            torch.testing.assert_close(left, right, atol=1e-4, rtol=1e-4, msg=str(index))
    for position in range(fixed.shape[1]):
        grad, = torch.autograd.grad(actual["final"][:, position].square().sum(), replacements, retain_graph=True)
        for other in range(fixed.shape[1]):
            assert (grad[:, other].abs().sum() > 0) if other == position else not grad[:, other].any()


def test_checkpointed_matches_plain(backbone):
    dim = backbone.embedding_weight.shape[1]
    torch.manual_seed(4)
    prefix, fixed, replacements = (torch.randn(1, n, dim, requires_grad=True) for n in (2, 5, 5))
    cache = _cache(backbone, prefix)
    params = [replacements, fixed, prefix, *[p for p in backbone.hf.parameters() if p.requires_grad]]
    results = []
    for checkpointed in (False, True):
        backbone.checkpoint_layers = checkpointed
        out = backbone.isolated_sequence(fixed, replacements, cache, cutoff=2)
        results.append((out, torch.autograd.grad(_weighted_loss(out), params, allow_unused=True, retain_graph=True)))
    backbone.checkpoint_layers = False
    _compare_outputs(results[0][0], results[1][0])
    for left, right in zip(results[0][1], results[1][1]):
        if left is None or right is None:
            assert left is None and right is None
        else:
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)


def test_left_padding_and_history_independent_of_replacements(backbone):
    dim = backbone.embedding_weight.shape[1]
    torch.manual_seed(5)
    prefix = torch.randn(2, 3, dim)
    prefix[0, 0] = 0
    cache = _cache(backbone, prefix, left_pad=torch.tensor([1, 0]))
    fixed = torch.randn(2, 5, dim)
    with torch.no_grad():
        a = backbone.isolated_sequence(fixed, torch.randn(2, 5, dim), cache, cutoff=2)
        b = backbone.isolated_sequence(fixed, torch.randn(2, 5, dim), cache, cutoff=2)
        torch.testing.assert_close(a["history_shallow"], b["history_shallow"], atol=0, rtol=0)
        assert not torch.allclose(a["final"], b["final"])
        _, ordinary = backbone.run_layers(fixed, range(backbone.num_layers), cache)
        _compare_caches(a["cache"], ordinary)
        replacements = torch.randn(2, 5, dim)
        _compare_outputs(backbone.isolated_sequence(fixed, replacements, cache, cutoff=2),
                         _literal_position_branches(backbone, fixed, replacements, cache, 2))
