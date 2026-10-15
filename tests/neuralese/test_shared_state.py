import pytest
import torch

from natlang_neuralese.model.shared_state import (
    Invocation, ImportSpan, LowRankTokenEmbedding, compile_graph,
    forward_graph, forward_reference, capture_span,
)
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
from natlang_neuralese.model.hf_port import QwenPortBackbone


def tiny(family):
    torch.manual_seed(18)
    if family == 'lfm':
        from transformers import Lfm2Config, Lfm2ForCausalLM
        config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            layer_types=['conv', 'full_attention', 'conv', 'full_attention'],
            conv_L_cache=4, max_position_embeddings=128)
        model = Lfm2ForCausalLM(config).eval()
        port = PortBackbone(model, ControlTokens(62, 63), fast=False)
    elif family == 'mellum':
        import tempfile
        from pathlib import Path
        from test_mellum_port import tiny_mellum
        from natlang_neuralese.maple.model import load_maple
        from natlang_neuralese.maple.maple_port import MaplePortBackbone
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            tiny_mellum(path)
            model = load_maple(path, device='cpu', dtype=torch.float32, ternary_attention=False)
        port = MaplePortBackbone(model, ControlTokens(126, 127), fast=False)
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=128, sliding_window=3, use_sliding_window=True,
            layer_types=['sliding_attention', 'full_attention'] * 2)
        port = QwenPortBackbone(Qwen3ForCausalLM(config).eval(), ControlTokens(62, 63), fast=False)
    return port


def graph():
    return compile_graph([
        Invocation('a', (2, 3, 4, 5, 6), producing=((2, 5),)),
        Invocation('b', (7, 8, 9), imports=(ImportSpan(1, 'a', 2, 5),), producing=((1, 3),)),
        Invocation('c', (10, 11, 12, 13), imports=(ImportSpan(0, 'a', 2, 4),
                   ImportSpan(2, 'b', 1, 3), ImportSpan(3, 'a', 3, 5))),
    ])


@pytest.mark.parametrize('family', ['lfm', 'qwen', 'mellum'])
def test_single_sequence_matches_existing_port(family):
    port = tiny(family)
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]])
    plan = compile_graph([Invocation('a', tuple(ids[0].tolist()))])
    actual = forward_graph(port, plan)['h_final']
    expected = port.forward_ids(ids)['h_final']
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize('family', ['lfm', 'qwen', 'mellum'])
def test_dag_forward_and_gradients_match_invocation_reference(family):
    port = tiny(family)
    plan = graph()
    adapter = LowRankTokenEmbedding(port.embedding_weight.shape[1], 4)
    torch.nn.init.normal_(adapter.up.weight, std=0.02)
    x = torch.randn(1, plan.token_count, port.embedding_weight.shape[1], requires_grad=True)
    weight = torch.randn_like(x)
    variables = (x, adapter.down.weight, adapter.up.weight, *(p for p in port.layers[0].parameters() if p.requires_grad))
    flat = forward_graph(port, plan, adapter, embeddings=x)
    ref = forward_reference(port, plan, adapter, embeddings=x)
    torch.testing.assert_close(flat['h_final'], ref['h_final'], atol=3e-6, rtol=2e-5)
    a = torch.autograd.grad((flat['h_final'] * weight).sum(), variables, allow_unused=True)
    b = torch.autograd.grad((ref['h_final'] * weight).sum(), variables, allow_unused=True)
    for ga, gb in zip(a, b):
        assert (ga is None) == (gb is None)
        if ga is not None:
            torch.testing.assert_close(ga, gb, atol=2e-5, rtol=2e-4)
    block = capture_span(flat, 2, 5)
    assert len(block['layers']) == port.num_layers
    assert tuple(block['positions']) == (2, 3, 4)


def test_alias_masks_exclude_source_private_prefix_and_future():
    plan = graph()
    mask = plan.attention_mask('cpu')[0, 0]
    for row in range(5, 8):
        visible = {plan.slot_sources[s] for s in mask[row].nonzero().flatten().tolist()}
        assert not ({0, 1} & visible)
        assert row in visible
        assert not any(i > row for i in visible)
    assert plan.token_count == 12
    assert len(plan.slot_sources) > plan.token_count


@pytest.mark.parametrize('bad', [
    [Invocation('a', (1,), imports=(ImportSpan(0, 'a', 0, 1),))],
    [Invocation('a', (1,)), Invocation('b', (2,), imports=(ImportSpan(0, 'a', 0, 1),))],
    [Invocation('a', (1,), producing=((0, 2),))],
])
def test_invalid_graph_rejected(bad):
    with pytest.raises(ValueError):
        compile_graph(bad)


def test_role_rows_and_logical_capture_are_native_embeddings():
    from natlang_neuralese.model.shared_state import SharedStateModel, capture_invocation_span
    port = tiny('lfm')
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    ids = torch.tensor([[60, 61, 62, 63]])
    assert torch.equal(model.embed(ids)[0], model.input_rows)
    h = torch.randn(1, 4, 32)
    expected = port.final_norm(h) @ model.input_rows.T
    torch.testing.assert_close(model.logits(h)[..., [60, 61, 62, 63]], expected)
    plan = compile_graph([
        Invocation('a', (2, 3, 4), producing=((1, 3),)),
        Invocation('b', (5, 6, 7), imports=(ImportSpan(2, 'a', 1, 3),), producing=((1, 3),)),
    ])
    result = model(plan)
    captured = capture_invocation_span(result, 'b', 1, 3)
    assert captured['token_ids'].tolist() == [6, 3, 4, 7]
    assert captured['positions'].tolist() == [1, 2, 3, 4]
    assert captured['embeddings'].shape == (4, 32)
    assert captured['layers'][0]['gated_inputs'].shape[0] == 4
    assert captured['layers'][1]['raw_k'].shape[0] == 4
    model.logits(result['h_final']).square().mean().backward()
    assert model.adapter.up.weight.grad.abs().sum() > 0
