import weakref

import torch

from natlang_neuralese.model.lfm2_port import PortCache


def test_recurrent_attention_fields_share_prefix_and_preserve_gradients():
    from natlang_neuralese.model.lfm2_port import AttentionState, append_kv
    torch.manual_seed(13)
    prefix = torch.randn(2, 1, 128, 4, requires_grad=True)
    tokens = [torch.randn(2, 1, 1, 4, requires_grad=True) for _ in range(8)]
    state = AttentionState(prefix, prefix * 2)
    original = state
    actual_loss, expected_loss = 0, 0
    for index, token in enumerate(tokens):
        state = append_kv(state, token, token * 2, static=False)
        expected = torch.cat([prefix, *tokens[:index + 1]], 2)
        torch.testing.assert_close(state.k, expected, rtol=0, atol=0)
        assert state.prefix_k is prefix
        assert state.tail_k.shape[2] == index + 1
        assert state.length == 128 + index + 1
        restored = AttentionState.from_fields(state.fields())
        torch.testing.assert_close(restored.k, expected, rtol=0, atol=0)
        torch.testing.assert_close(restored.select([1]).v, (expected * 2)[1:2], rtol=0, atol=0)
        actual_loss = actual_loss + state.k.square().mean() + state.v.square().mean()
        expected_loss = expected_loss + expected.square().mean() + (expected * 2).square().mean()
    torch.testing.assert_close(original.k, prefix, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual_loss, [prefix, *tokens], retain_graph=True)
    expected_grad = torch.autograd.grad(expected_loss, [prefix, *tokens])
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected)


def test_layer_checkpointing_preserves_cached_recurrence_gradient(loaded):
    _, tokenizer, backbone = loaded
    ids = tokenizer('A nested call returns useful information.', return_tensors='pt').input_ids
    h = backbone.embed(ids).detach().requires_grad_(True)
    weight = backbone.hf.model.layers[0].feed_forward.w1.weight
    original_requires_grad = weight.requires_grad
    weight.requires_grad_(True)
    cache_refs = []
    def loss(checkpointed, attention_only=False):
        backbone.checkpoint_layers = checkpointed
        backbone.checkpoint_attention_only = attention_only
        out, cache = backbone.run_layers(h, range(backbone.num_layers), PortCache.empty(backbone.num_layers))
        if checkpointed:
            cache_refs.append(weakref.ref(cache))
        # A second causal chunk reads the cache built by the first.
        suffix, _ = backbone.run_layers(h[:, :2], range(backbone.num_layers), cache)
        return out.square().mean() + suffix.square().mean()
    try:
        plain = loss(False)
        plain_grad = torch.autograd.grad(plain, (h, weight))
        actual = loss(True)
        # The loss graph may retain cache tensors, but never their opaque
        # Python container (which defeats autograd's release and offload).
        assert cache_refs[0]() is None
        actual_grad = torch.autograd.grad(actual, (h, weight))
        torch.testing.assert_close(actual, plain)
        for got, expected in zip(actual_grad, plain_grad):
            torch.testing.assert_close(got, expected)
        selective = loss(True, attention_only=True)
        selective_grad = torch.autograd.grad(selective, (h, weight))
        torch.testing.assert_close(selective, plain)
        for got, expected in zip(selective_grad, plain_grad):
            torch.testing.assert_close(got, expected)
    finally:
        backbone.checkpoint_layers = False
        backbone.checkpoint_attention_only = False
        weight.requires_grad_(original_requires_grad)
