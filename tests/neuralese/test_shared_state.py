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


@pytest.mark.parametrize('family', ['lfm', 'qwen', 'mellum'])
def test_incremental_capture_splice_matches_one_pass_and_gradients(family):
    from types import SimpleNamespace
    from natlang_neuralese.model.shared_state import SharedStateModel, native_step, splice_block, capture_invocation_span
    port = tiny(family)
    vocab = port.embedding_weight.shape[0]
    model = SharedStateModel(port, tuple(range(vocab-4, vocab)), rank=4)
    torch.nn.init.normal_(model.adapter.up.weight, std=0.03)
    plan = compile_graph([
        Invocation('source', (2, 3, 4, 5, 6), producing=((2, 4),)),
        Invocation('target', (7, 8, 9, 10), imports=(ImportSpan(2, 'source', 2, 4),)),
    ])
    flat = model(plan)
    producer = native_step(model, torch.tensor([[2, 3, 4, 5, 6]]),
                           producing=torch.tensor([[False, False, True, True, False]]))
    captured = {**capture_invocation_span(flat, 'source', 2, 4)}
    # Capture from the independently executed producer, not the flat result.
    captured['layers'] = tuple({key: value[2:4] if key != 'kind' else value
                               for key, value in layer.items()} for layer in producer['layers'])
    captured['embeddings'] = producer['embeddings'][2:4]
    identity = dict(checkpoint_sha256='a'*64, tokenizer_fingerprint='b'*64, embedding_weight_sha256='c'*64)
    block = SimpleNamespace(**captured, **identity, architecture=port.config.model_type, validate=lambda: None)
    prefix = native_step(model, torch.tensor([[7, 8]]))
    cache = splice_block(model, prefix['cache'], block, **identity)
    suffix = native_step(model, torch.tensor([[9, 10]]), cache)
    actual = torch.cat([producer['h_final'], prefix['h_final'], suffix['h_final']], 1)
    torch.testing.assert_close(actual, flat['h_final'], atol=3e-6, rtol=3e-5)
    weight = torch.randn_like(actual)
    params = (model.input_rows, model.adapter.up.weight)
    a = torch.autograd.grad((actual*weight).sum(), params)
    b = torch.autograd.grad((flat['h_final']*weight).sum(), params)
    for ga, gb in zip(a, b):
        torch.testing.assert_close(ga, gb, atol=3e-5, rtol=3e-4)
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        splice_block(model, prefix['cache'], block, **{**identity, 'checkpoint_sha256': 'd'*64})


def test_marker_tokenizer_expansion_preserves_all_original_rows():
    from natlang_neuralese.model.shared_state import provision_role_tokens, ROLE_MARKERS, SharedStateModel
    class Tokenizer:
        unk_token_id = 0
        def __init__(self):
            self.vocab = {f'word{i}': i for i in range(64)}
        def add_special_tokens(self, mapping, **kwargs):
            for token in mapping['additional_special_tokens']:
                self.vocab.setdefault(token, len(self.vocab))
        def convert_tokens_to_ids(self, token):
            return self.vocab.get(token, 0)
        def get_vocab(self):
            return self.vocab
    port = tiny('lfm')
    original = port.embedding_weight.detach().clone()
    tokenizer = Tokenizer()
    tokens = provision_role_tokens(port, tokenizer)
    assert tokens.ids == (64, 65, 66, 67)
    assert port.embedding_weight.shape[0] == 68
    assert port.output_weight is port.embedding_weight
    assert torch.equal(port.embedding_weight[:64], original)
    assert [tokenizer.vocab[x] for x in ROLE_MARKERS] == list(tokens.ids)
    model = SharedStateModel(port, tokens.ids)
    assert model.logits(torch.randn(1, 1, 32)).shape[-1] == 68


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_nested_export_keeps_intervening_import_aliases(family):
    from types import SimpleNamespace
    from natlang_neuralese.model.shared_state import SharedStateModel, native_step, splice_block, capture_invocation_span
    port = tiny(family)
    vocab = port.embedding_weight.shape[0]
    model = SharedStateModel(port, tuple(range(vocab-4, vocab)), rank=4)
    plan = compile_graph([
        Invocation('a', (2, 3, 4, 5), producing=((2, 4),)),
        Invocation('b', (6, 7, 8, 9), imports=(ImportSpan(2, 'a', 2, 4),), producing=((1, 3),)),
        Invocation('c', (10, 11, 12), imports=(ImportSpan(1, 'b', 1, 3),)),
    ])
    # b exports [7, 4, 5, 8], never its private [6] or suffix [9].
    assert plan.sequences[2] == (8, 5, 2, 3, 6, 9, 10)
    flat = model(plan)
    reference = model(plan, reference=True)
    torch.testing.assert_close(flat['h_final'], reference['h_final'], atol=4e-6, rtol=3e-5)
    fields = capture_invocation_span(flat, 'b', 1, 3)
    assert fields['token_ids'].tolist() == [7, 4, 5, 8]
    identity = dict(checkpoint_sha256='a'*64, tokenizer_fingerprint='b'*64, embedding_weight_sha256='c'*64)
    block = SimpleNamespace(**fields, **identity, architecture=port.config.model_type, validate=lambda: None)
    prefix = native_step(model, torch.tensor([[10]]))
    cache = splice_block(model, prefix['cache'], block, **identity)
    suffix = native_step(model, torch.tensor([[11, 12]]), cache)
    torch.testing.assert_close(suffix['h_final'], flat['h_final'][:, -2:], atol=4e-6, rtol=3e-5)
    weight = torch.randn_like(suffix['h_final'])
    a = torch.autograd.grad((suffix['h_final']*weight).sum(), model.adapter.up.weight, retain_graph=True)[0]
    b = torch.autograd.grad((flat['h_final'][:, -2:]*weight).sum(), model.adapter.up.weight)[0]
    torch.testing.assert_close(a, b, atol=3e-5, rtol=4e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_sparse_cuda_attention_matches_dense_forward_and_gradients():
    from natlang_neuralese.model.shared_state import graph_attention
    plan = compile_graph([
        Invocation('a', tuple(range(270)), producing=((120, 270),)),
        Invocation('b', tuple(range(140)), imports=(ImportSpan(2, 'a', 130, 270),), producing=((1, 139),)),
        Invocation('c', tuple(range(155)), imports=(ImportSpan(1, 'b', 1, 139), ImportSpan(130, 'a', 120, 270))),
    ])
    torch.manual_seed(5)
    q = torch.randn(1, 4, plan.token_count, 32, device='cuda', requires_grad=True)
    k = torch.randn(1, 2, len(plan.slot_sources), 32, device='cuda', requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    for window in (None, 61):
        actual = graph_attention(q, k, v, plan, window, 32**-0.5, flex=True)
        expected = graph_attention(q, k, v, plan, window, 32**-0.5, flex=False)
        torch.testing.assert_close(actual, expected, atol=5e-5, rtol=2e-4)
        weight = torch.randn_like(actual)
        a = torch.autograd.grad((actual*weight).sum(), (q, k, v))
        b = torch.autograd.grad((expected*weight).sum(), (q, k, v))
        for ga, gb in zip(a, b):
            torch.testing.assert_close(ga, gb, atol=8e-5, rtol=3e-4)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_external_native_state_has_exact_forward_and_joint_gradients(family):
    from types import SimpleNamespace
    from natlang_neuralese.model.shared_state import SharedStateModel, capture_invocation_span
    port = tiny(family)
    vocab = port.embedding_weight.shape[0]
    model = SharedStateModel(port, tuple(range(vocab-4, vocab)), rank=4)
    torch.nn.init.normal_(model.adapter.up.weight, std=0.03)
    source = Invocation('source', (2, 3, 4, 5), producing=((2, 4),))
    target = Invocation('target', (6, 7, 8), imports=(ImportSpan(1, 'source', 2, 4),))
    complete = model(compile_graph([source, target]))
    produced = model(compile_graph([source]))
    fields = capture_invocation_span(produced, 'source', 2, 4)
    from natlang_neuralese.model.state_block import NativeStateBlock
    identity = dict(checkpoint_sha256='a'*64, tokenizer_fingerprint='b'*64,
                    embedding_weight_sha256='c'*64, architecture=port.config.model_type)
    block = NativeStateBlock(text='shared', **fields, **identity)
    plan = compile_graph([Invocation('target', target.token_ids,
                         imports=(ImportSpan(1, 'wire', 0, 2),))], external_sizes={'wire': 2})
    external = model(plan, external_blocks={'wire': block}, external_identity=identity)
    reference = model(plan, external_blocks={'wire': block}, external_identity=identity, reference=True)
    torch.testing.assert_close(external['h_final'], complete['h_final'][:, -3:], atol=4e-6, rtol=3e-5)
    torch.testing.assert_close(external['h_final'], reference['h_final'], atol=4e-6, rtol=3e-5)
    weight = torch.randn_like(external['h_final'])
    a = torch.autograd.grad((external['h_final']*weight).sum(), model.adapter.up.weight, retain_graph=True)[0]
    b = torch.autograd.grad((complete['h_final'][:, -3:]*weight).sum(), model.adapter.up.weight)[0]
    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)
    # Simulate a trainable semantic decoder. Its native state tensors retain
    # gradient paths into both decoder parameters and source-produced state.
    delta = torch.nn.Parameter(torch.tensor(0.01))
    decoded_layers = tuple({key: value + delta if key != 'kind' else value
                            for key, value in layer.items()} for layer in fields['layers'])
    decoded = NativeStateBlock(text='shared', **{**fields, 'layers': decoded_layers}, **identity)
    adapted = model(plan, external_blocks={'wire': decoded}, external_identity=identity)
    grad_delta, grad_source = torch.autograd.grad(adapted['h_final'].square().sum(),
                                                (delta, model.adapter.up.weight))
    assert grad_delta.abs() > 0
    assert grad_source.abs().sum() > 0


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_checkpointed_graph_preserves_outputs_and_gradients(family):
    from natlang_neuralese.model.shared_state import SharedStateModel
    port = tiny(family)
    vocab = port.embedding_weight.shape[0]
    model = SharedStateModel(port, tuple(range(vocab-4, vocab)), rank=4)
    plan = graph()
    torch.nn.init.normal_(model.adapter.up.weight, std=0.03)
    plain = model(plan, retain_states=False)['h_final']
    expected = torch.autograd.grad(plain.square().sum(), model.adapter.up.weight)[0]
    port.checkpoint_layers = True
    checkpointed = model(plan, retain_states=False)
    assert checkpointed['states'] == ()
    torch.testing.assert_close(checkpointed['h_final'], plain)
    actual = torch.autograd.grad(checkpointed['h_final'].square().sum(), model.adapter.up.weight)[0]
    torch.testing.assert_close(actual, expected)


def test_channel_checkpoint_is_complete_and_rejects_legacy_or_wrong_contract():
    from natlang_neuralese.model.shared_state import SharedStateModel, channel_checkpoint, restore_channel_checkpoint
    model = SharedStateModel(tiny('qwen'), (60, 61, 62, 63), rank=4)
    torch.nn.init.normal_(model.adapter.up.weight, std=0.03)
    state = channel_checkpoint(model)
    other = SharedStateModel(tiny('qwen'), (60, 61, 62, 63), rank=4)
    restore_channel_checkpoint(other, state)
    for a, b in zip(model.adapter.parameters(), other.adapter.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(model.input_rows, other.input_rows)
    assert torch.equal(model.output_rows, other.output_rows)
    with pytest.raises(ValueError, match='format'):
        restore_channel_checkpoint(other, {**state, 'format': 'neuralese_direct/1'})
    with pytest.raises(ValueError, match='marker_ids'):
        restore_channel_checkpoint(other, {**state, 'marker_ids': (59, 60, 61, 62)})


def test_external_states_require_explicit_target_identity():
    from natlang_neuralese.model.shared_state import SharedStateModel, capture_block
    port = tiny('lfm')
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4)
    source = model(compile_graph([Invocation('source', (2, 3), producing=((0, 2),))]))
    identity = dict(checkpoint_sha256='a'*64, tokenizer_fingerprint='b'*64,
                    embedding_weight_sha256='c'*64, architecture=port.config.model_type)
    block = capture_block(source, 'source', 0, 2, text='source', **identity)
    plan = compile_graph([Invocation('target', (4, 5), imports=(ImportSpan(1, 'external', 0, 2),))],
                         external_sizes={'external': 2})
    with pytest.raises(ValueError, match='explicit target'):
        model(plan, external_blocks={'external': block})
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        model(plan, external_blocks={'external': block}, external_identity={**identity, 'checkpoint_sha256': 'd'*64})


def test_marker_provisioning_with_installed_transformers_tokenizer_api():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from natlang_neuralese.model.shared_state import provision_role_tokens, ROLE_MARKERS
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        vocab={f'word{i}': i for i in range(64)}, unk_token='word0')), unk_token='word0')
    port = tiny('lfm')
    controls = provision_role_tokens(port, tokenizer)
    assert controls.ids == (64, 65, 66, 67)
    assert tokenizer(list(ROLE_MARKERS), add_special_tokens=False).input_ids == [[i] for i in controls.ids]
    assert all(token in tokenizer.all_special_tokens for token in ROLE_MARKERS)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_token_boundaries_include_start_import_and_exclude_stop_import(family):
    from natlang_neuralese.model.shared_state import SharedStateModel, capture_invocation_span
    port = tiny(family)
    vocab = port.embedding_weight.shape[0]
    model = SharedStateModel(port, tuple(range(vocab-4, vocab)), rank=4)
    plan = compile_graph([
        Invocation('a', (2, 3), producing=((0, 2),)),
        Invocation('b', (4, 5, 6), imports=(ImportSpan(1, 'a', 0, 2), ImportSpan(2, 'a', 0, 1)), producing=((1, 2),)),
        Invocation('c', (7, 8), imports=(ImportSpan(1, 'b', 1, 2),)),
    ])
    result = model(plan)
    fields = capture_invocation_span(result, 'b', 1, 2)
    assert fields['token_ids'].tolist() == [2, 3, 5]
    assert fields['positions'].tolist() == [1, 2, 3]
    assert plan.sequences[2] == (5, 0, 1, 3, 6)
    reference = model(plan, reference=True)
    torch.testing.assert_close(result['h_final'], reference['h_final'], atol=4e-6, rtol=3e-5)


def test_imported_only_forwarding_preserves_state_without_executing_tokens():
    from natlang_neuralese.model.shared_state import SharedStateModel, capture_block, capture_logical_span
    from natlang_neuralese.model.state_block import NativeStateBlock
    model = SharedStateModel(tiny('lfm'), (60, 61, 62, 63), rank=4)
    source = model(compile_graph([Invocation('source', (2, 3), producing=((0, 2),))]))
    identity = dict(checkpoint_sha256='a'*64, tokenizer_fingerprint='b'*64,
                    embedding_weight_sha256='c'*64, architecture=model.port.config.model_type)
    block = capture_block(source, 'source', 0, 2, text='source', **identity)
    plan = compile_graph([
        Invocation('forwarded', (), imports=(ImportSpan(0, 'external', 0, 2),), logical_producing=((0, 2),)),
        Invocation('consumer', (4, 5), imports=(ImportSpan(1, 'forwarded', 0, 2, logical=True),)),
    ], external_sizes={'external': 2})
    result = model(plan, external_blocks={'external': block}, external_identity=identity)
    forwarded = capture_logical_span(result, 'forwarded', 0, 2)
    assert torch.equal(forwarded['token_ids'], block.token_ids)
    for original, copied in zip(block.layers, forwarded['layers']):
        for key in original:
            if key != 'kind':
                assert torch.equal(copied[key], original[key])
    pure = compile_graph([Invocation('forward', (), imports=(ImportSpan(0, 'external', 0, 2),),
                                             logical_producing=((0, 2),))], external_sizes={'external': 2})
    with torch.no_grad():
        pure_result = model(pure, external_blocks={'external': block}, external_identity=identity)
    assert pure_result['h_final'].shape == (1, 0, 32)
    assert pure.token_count == 0
    assert capture_logical_span(pure_result, 'forward', 0, 2)['token_ids'].tolist() == [2, 3]
