"""Gaussian import channel (model/import_channel.py): the packed graph forward with the channel equals explicit cached
execution that splices the perturbed state, its KL equals the closed form on the clean imported state, and gradients
of both agree."""
import math
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.import_channel import ImportChannel, import_channel, measure_scales
from natlang_neuralese.model.shared_state import (ImportSpan, Invocation, SharedStateModel, capture_invocation_span,
                                                  compile_graph, forward_reference, native_step, splice_block)
from test_shared_state import tiny

IDENTITY = dict(checkpoint_sha256='a'*64, tokenizer_fingerprint='b'*64, embedding_weight_sha256='c'*64)
SIGMA, PRIOR = .7, 1.3


def setup(family):
    port = tiny(family)
    vocab = port.embedding_weight.shape[0]
    model = SharedStateModel(port, tuple(range(vocab-4, vocab)), rank=4)
    torch.nn.init.normal_(model.adapter.up.weight, std=0.03)
    plan = compile_graph([
        Invocation('source', (2, 3, 4, 5, 6), producing=((2, 4),)),
        Invocation('target', (7, 8, 9, 10, 11), imports=(ImportSpan(2, 'source', 2, 4),)),
    ])
    scales = {key: value * (1 + .1 * number) for number, (key, value) in enumerate(sorted(measure_scales(model, plan).items()))}
    return model, plan, ImportChannel(('source',), SIGMA, PRIOR, scales, seed=5)


def explicit(model, plan, channel):
    """Producer, then the consumer splicing the producer's span with the channel's noise added; and the closed-form KL."""
    port = model.port
    with torch.no_grad():
        captured = capture_invocation_span(model(plan), 'source', 2, 4)
    producer = native_step(model, torch.tensor([[2, 3, 4, 5, 6]]),
                           producing=torch.tensor([[False, False, True, True, False]]))
    layers, kl = [], 0.
    for index, layer in enumerate(producer['layers']):
        noisy = {'kind': layer['kind']}
        for key, name in (('raw_k', 'k'), ('v', 'v'), ('gated_inputs', 'gated')):
            if key not in layer:
                continue
            value = layer[key][2:4]
            c = channel.scale(index, name)
            eps = channel.noise('source', index, name, 5, tuple(value.shape[1:]), value.device)[2:4]
            noisy[key] = value + SIGMA * c * eps
            kl = kl + ((value.double().pow(2).sum() / c**2 + value.numel() * SIGMA**2) / (2 * PRIOR**2)
                       + value.numel() * (math.log(PRIOR / SIGMA) - .5))
        layers.append(noisy)
    block = SimpleNamespace(token_ids=captured['token_ids'], embeddings=producer['embeddings'][2:4],
                            positions=captured['positions'], layers=tuple(layers), **IDENTITY,
                            architecture=port.config.model_type, validate=lambda: None)
    prefix = native_step(model, torch.tensor([[7, 8]]))
    suffix = native_step(model, torch.tensor([[9, 10, 11]]), splice_block(model, prefix['cache'], block, **IDENTITY))
    return torch.cat([producer['h_final'], prefix['h_final'], suffix['h_final']], 1), kl


@pytest.mark.parametrize('family', ['lfm', 'lfm3'])
def test_channel_forward_kl_and_gradients_match_explicit_perturbed_splice(family):
    model, plan, channel = setup(family)
    clean = model(plan)
    with import_channel(channel):
        noisy = model(plan)
    report = noisy['import_channel']
    assert report['sources'] == ('source',) and report['imported_rows'] == 2
    actual, kl = explicit(model, plan, channel)
    torch.testing.assert_close(actual, noisy['h_final'], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(noisy['import_channel']['kl'][0].double(), kl, atol=1e-6, rtol=1e-5)
    # The producer's own rows are never perturbed; the consumer's rows after the import are.
    torch.testing.assert_close(noisy['h_final'][:, :5], clean['h_final'][:, :5])
    assert (noisy['h_final'][:, -3:] - clean['h_final'][:, -3:]).abs().max() > 1e-4
    weight = torch.randn_like(actual)
    params = [p for p in model.parameters() if p.requires_grad]
    a = torch.autograd.grad((actual*weight).sum() + kl.float(), params, allow_unused=True)
    b = torch.autograd.grad((noisy['h_final']*weight).sum() + noisy['import_channel']['kl'].sum(), params,
                            allow_unused=True)
    for ga, gb in zip(a, b):
        if ga is None:
            assert gb is None or gb.abs().max() == 0
            continue
        torch.testing.assert_close(ga, gb, atol=3e-5, rtol=3e-4)


def test_channel_is_deterministic_through_checkpoint_recompute_and_refuses_the_cached_oracle():
    model, plan, channel = setup('lfm')
    with import_channel(channel):
        plain = model(plan)
        weight = torch.randn_like(plain['h_final'])
        expected = torch.autograd.grad((plain['h_final']*weight).sum() + plain['import_channel']['kl'].sum(),
                                       model.adapter.up.weight)[0]
        model.port.checkpoint_layers = True
        try:
            again = model(plan)
            actual = torch.autograd.grad((again['h_final']*weight).sum() + again['import_channel']['kl'].sum(),
                                         model.adapter.up.weight)[0]
        finally:
            model.port.checkpoint_layers = False
        torch.testing.assert_close(again['h_final'], plain['h_final'])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        with pytest.raises(ValueError, match='cached oracle'):
            forward_reference(model.port, plan, model.adapter)
    with pytest.raises(ValueError, match='names no invocation'):
        with import_channel(ImportChannel(('nobody',), SIGMA, PRIOR, channel.scales)):
            model(plan)
    with pytest.raises(ValueError, match='positive'):
        ImportChannel(('source',), 0., PRIOR, channel.scales)
