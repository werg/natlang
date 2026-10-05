import torch

from natlang_neuralese.model.lfm2_port import PortCache


def test_layer_checkpointing_preserves_cached_recurrence_gradient(loaded):
    _, tokenizer, backbone = loaded
    ids = tokenizer('A nested call returns useful information.', return_tensors='pt').input_ids
    h = backbone.embed(ids).detach().requires_grad_(True)
    weight = backbone.hf.model.layers[0].feed_forward.w1.weight
    original_requires_grad = weight.requires_grad
    weight.requires_grad_(True)
    def loss(checkpointed):
        backbone.checkpoint_layers = checkpointed
        out, cache = backbone.run_layers(h, range(backbone.num_layers), PortCache.empty(backbone.num_layers))
        # A second causal chunk reads the cache built by the first.
        suffix, _ = backbone.run_layers(h[:, :2], range(backbone.num_layers), cache)
        return out.square().mean() + suffix.square().mean()
    try:
        plain = loss(False)
        plain_grad = torch.autograd.grad(plain, (h, weight))
        actual = loss(True)
        actual_grad = torch.autograd.grad(actual, (h, weight))
        torch.testing.assert_close(actual, plain)
        for got, expected in zip(actual_grad, plain_grad):
            torch.testing.assert_close(got, expected)
    finally:
        backbone.checkpoint_layers = False
        weight.requires_grad_(original_requires_grad)
