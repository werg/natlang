"""LFM3 native port (``model/lfm3_port.Lfm3PortBackbone``) against the vendored pinned reference, on a tiny twin.

The twin is built from the vendored ``Lfm3Config`` with every LFM3 layer form (width-7 short convolutions, full
attention with zero-tail partial RoPE, sliding attention with its own theta and a window shorter than the test
sequences, per-head output gates, sandwich norms). Real-weight qualification is in ``test_lfm3_real_parity.py``.
"""
import copy

import pytest
import torch

from natlang_neuralese.model.lfm2_port import ControlTokens, ConvState, PortCache


def tiny_lfm3_hf(seed=31, *, window=4, layer_types=None, sandwich=True):
    """A random tiny LFM3 (vendored modeling code) whose norms and gates are perturbed so every form matters."""
    from natlang_neuralese.third_party import lfm3
    config_class, model_class = lfm3.config_class(), lfm3.causal_lm_class()
    layer_types = layer_types or ['conv', 'full_attention', 'conv', 'sliding_attention', 'conv', 'full_attention']
    config = config_class(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=len(layer_types),
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, layer_types=layer_types,
                          sliding_window=window, max_position_embeddings=256, pad_token_id=None, bos_token_id=1,
                          eos_token_id=2, use_sandwich_norm=sandwich)
    config._attn_implementation = 'eager'
    torch.manual_seed(seed)
    model = model_class(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('norm.weight'):
                parameter.copy_(1 + 0.2 * torch.randn_like(parameter))
            elif 'gate_proj' in name or 'conv.conv' in name:
                parameter.copy_(0.3 * torch.randn_like(parameter))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


# LFM3 declares history_reasoning=keep: the twin's template renders every assistant turn's reasoning.
LFM3_TINY_TEMPLATE = ("{% for message in messages %}{% if message['role'] == 'assistant' and "
                      "message.get('reasoning_content') %}{{ message['reasoning_content'] }}{% endif %}"
                      "{{ message['content'] }}\\n{% endfor %}")


def tiny_lfm3(seed=31, *, fast=False, markers=True, window=4, **kwargs):
    from natlang_neuralese.model.lfm3_port import Lfm3PortBackbone
    return Lfm3PortBackbone(tiny_lfm3_hf(seed, window=window, **kwargs), ControlTokens(62, 63), fast=fast,
                            markers=markers)


def test_vendored_sources_are_pinned_and_registered():
    from transformers import AutoConfig
    from natlang_neuralese.third_party import lfm3
    assert lfm3.verify_sources()['modeling_lfm3.py'] == (
        'e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab')
    lfm3.register()
    config = AutoConfig.for_model('lfm3')
    assert type(config).__module__ == 'natlang_neuralese.third_party.lfm3.configuration_lfm3'


def test_explicit_role_markers_are_required():
    from natlang_neuralese.model.lfm3_port import Lfm3PortBackbone
    with pytest.raises(ValueError, match='role tokens'):
        Lfm3PortBackbone(tiny_lfm3_hf())


@pytest.mark.parametrize('fast', [False, True])
@pytest.mark.parametrize('sandwich', [True, False])
def test_port_forward_matches_reference_beyond_window(fast, sandwich):
    port = tiny_lfm3(fast=fast, markers=False, sandwich=sandwich)
    ids = torch.randint(3, 60, (2, 19), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = port.hf(ids).logits
        actual = port.forward_ids(ids)['logits']
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_layer_forms_are_declared():
    port = tiny_lfm3()
    assert port.attention_gate == 'per_head_input' and port.sandwich_norm and port.conv_window == 6
    assert [port.window(i) for i in range(port.num_layers)] == [None, None, None, 4, None, None]
    cos, sin = port._rope(port.embedding_weight[None, :3], torch.arange(3)[None])
    assert set(cos) == {'full_attention', 'sliding_attention'}
    # Zero-tail partial rotary on full attention: the second quarter and last quarter of each head stay unrotated.
    full = cos['full_attention'][0, 2]
    assert torch.equal(full[2:4], torch.ones(2)) and torch.equal(full[6:], torch.ones(2))
    assert not torch.equal(cos['sliding_attention'][0, 2, 2:4], torch.ones(2))


@pytest.mark.parametrize('fast', [False, True])
def test_incremental_cache_equals_full_forward(fast):
    port = tiny_lfm3(fast=fast, markers=False)
    ids = torch.randint(3, 60, (1, 17), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = port.forward_ids(ids)
        cache, pieces = PortCache.empty(port.num_layers), []
        for start, stop in ((0, 5), (5, 6), (6, 9), *((i, i + 1) for i in range(9, 17))):
            result = port.forward_ids(ids[:, start:stop], cache=cache)
            cache = result['cache']
            pieces.append(result['logits'])
    torch.testing.assert_close(torch.cat(pieces, 1), full['logits'], atol=2e-5, rtol=2e-5)
    for index, state in enumerate(cache.states):
        if port.layer_types[index] == 'conv':
            assert isinstance(state, ConvState) and state.window.shape[-1] == 6
            torch.testing.assert_close(state.window, full['cache'].states[index].window, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(state.k, full['cache'].states[index].k, atol=1e-5, rtol=1e-5)


def test_left_padded_rows_match_unpadded_rows():
    port = tiny_lfm3(markers=False)
    generator = torch.Generator().manual_seed(2)
    rows = [torch.randint(3, 60, (n,), generator=generator) for n in (11, 7)]
    pad = torch.tensor([0, 4])
    batch = torch.stack([rows[0], torch.cat([torch.zeros(4, dtype=torch.long), rows[1]])])
    with torch.no_grad():
        padded = port.forward_ids(batch, left_pad=pad)['logits']
        for row, ids in enumerate(rows):
            alone = port.forward_ids(ids[None])['logits'][0]
            torch.testing.assert_close(padded[row, int(pad[row]):], alone, atol=2e-5, rtol=2e-5)


def test_gradients_match_reference():
    port = tiny_lfm3(markers=False)
    names = ['model.embed_tokens.weight', 'model.layers.0.conv.conv.weight', 'model.layers.0.conv.in_proj.weight',
             'model.layers.1.self_attn.gate_proj.weight', 'model.layers.1.self_attn.q_norm.weight',
             'model.layers.3.self_attn.gate_proj.weight', 'model.layers.3.self_attn.k_proj.weight',
             'model.layers.3.post_operator_norm.weight', 'model.layers.4.post_ffn_norm.weight',
             'model.layers.5.self_attn.o_proj.weight', 'model.embedding_norm.weight']
    parameters = dict(port.hf.named_parameters())
    watched = [parameters[name] for name in names]
    for parameter in watched:
        parameter.requires_grad_(True)
    ids = torch.randint(3, 60, (2, 15), generator=torch.Generator().manual_seed(3))
    weight = torch.randn(2, 15, 64, generator=torch.Generator().manual_seed(4))
    expected = torch.autograd.grad((port.hf(ids).logits * weight).sum(), watched)
    actual = torch.autograd.grad((port.forward_ids(ids)['logits'] * weight).sum(), watched)
    for name, left, right in zip(names, actual, expected):
        assert right.abs().sum() > 0, name
        # FP32 roundoff relative to each gradient's scale (the tied embedding's reaches ~200).
        scale = float(right.abs().max())
        assert float((left - right).abs().max()) <= 2e-6 * scale + 1e-7, name


def test_checkpointed_layers_match_plain_outputs_and_gradients():
    port = tiny_lfm3(markers=False)
    for parameter in port.hf.parameters():
        parameter.requires_grad_(True)
    ids = torch.randint(3, 60, (1, 13), generator=torch.Generator().manual_seed(5))
    prefix = port.forward_ids(ids[:, :4])['cache']
    results = []
    for checkpointed in (False, True):
        port.checkpoint_layers = checkpointed
        out = port.forward_ids(ids[:, 4:], cache=prefix)
        loss = out['logits'].square().mean()
        grads = torch.autograd.grad(loss, list(port.hf.parameters()), allow_unused=True, retain_graph=True)
        results.append((out, grads))
    port.checkpoint_layers = False
    torch.testing.assert_close(results[0][0]['logits'], results[1][0]['logits'])
    for left, right in zip(results[0][1], results[1][1]):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    for left, right in zip(results[0][0]['cache'].states, results[1][0]['cache'].states):
        for a, b in zip(left.fields(), right.fields()):
            torch.testing.assert_close(a, b)


def test_serving_batch_step_matches_independent_rows():
    from natlang_neuralese.serve.batch import BATCHED_STACKS, step_rows
    assert 'lfm3' in BATCHED_STACKS
    port = tiny_lfm3(markers=False)
    generator = torch.Generator().manual_seed(6)
    caches = []
    with torch.no_grad():
        for length in (3, 9, 6):  # rows cross the sliding window (4) at different steps
            caches.append(port.forward_ids(torch.randint(3, 60, (1, length), generator=generator))['cache'])
        reference = [copy.copy(cache) for cache in caches]
        for _ in range(3):
            h = port.embed(torch.randint(3, 60, (3, 1), generator=generator))
            out, caches = step_rows(port, h, caches, range(port.num_layers))
            for row in range(3):
                single, reference[row] = port.run_layers(h[row:row + 1], range(port.num_layers), reference[row])
                torch.testing.assert_close(out[row:row + 1], single, atol=2e-6, rtol=2e-5)


def test_training_plumbing_declares_lfm3_forms():
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.adapters import lora_target_names
    from natlang_neuralese.train.fp8_backend import _eligible_paths
    from natlang_neuralese.train.memory_estimator import backbone_memory_layout
    from natlang_neuralese.train.native_state_sft import native_trainable_parameters
    from natlang_neuralese.train.optim import latent_split, muon_eligible
    port = tiny_lfm3(markers=False)
    names = lora_target_names(port, [0, 1])
    assert 'model.layers.1.self_attn.o_proj' in names and 'model.layers.1.self_attn.gate_proj' in names
    assert 'model.layers.0.conv.out_proj' in names and not any('out_proj' in n and 'self_attn' in n for n in names)
    paths = _eligible_paths(port.hf)
    assert 'model.layers.1.self_attn.o_proj' in paths and 'model.layers.1.self_attn.gate_proj' not in paths
    half = copy.deepcopy(port)
    half.hf.to(torch.bfloat16)
    layout = backbone_memory_layout(half, checkpointed=False)
    # operator/ffn/post-operator/post-ffn norms on each layer, plus q/k norms over all heads on attention layers.
    assert layout['fp32_norm_saved_bytes_per_token'] == 4 * (6 * 4 * 32 + 3 * (32 + 16))
    model = SharedStateModel(port, (58, 59, 60, 61), rank=4)
    latent, ordinary = latent_split(native_trainable_parameters(model), 'muonsr')
    latent_names = {name for name, _ in latent}
    ordinary_names = {name for name, _ in ordinary}
    # The real per-class routing (gate_proj [16, 2048] on Muon) is checked on the LFM3-1.7B inventory in
    # test_lfm3_real_parity.py; this twin's 4-head gate is narrower than MUON_MIN_SIDE.
    assert 'backbone.model.layers.1.self_attn.o_proj.weight' in latent_names
    assert 'backbone.model.layers.0.conv.conv.weight' in ordinary_names
    assert {'interface.input_rows', 'interface.adapter.up.weight'} <= ordinary_names
    gate = dict(port.hf.named_parameters())['model.layers.1.self_attn.gate_proj.weight']
    assert muon_eligible('backbone.model.layers.1.self_attn.gate_proj.weight', gate, 64, set())


def test_native_precision_profiles_cover_every_lfm3_norm():
    from natlang_neuralese.model.native_parameter_precision import PROFILE, configure_norm_precision, norm_parameters
    from natlang_neuralese.model.shared_execution import configure_shared_execution, validate_shared_execution
    from natlang_neuralese.model.shared_state import SharedStateModel
    port = tiny_lfm3(markers=False)
    port.hf.to(torch.bfloat16)
    model = SharedStateModel(port, (58, 59, 60, 61), rank=4)
    found = norm_parameters(model)
    suffixes = {name.split('.', 3)[-1] for name in found}
    assert {'operator_norm.weight', 'ffn_norm.weight', 'post_operator_norm.weight', 'post_ffn_norm.weight',
            'self_attn.q_norm.weight', 'self_attn.k_norm.weight'} <= suffixes
    assert len(found) == 6 * 4 + 3 * 2
    configure_norm_precision(model, PROFILE)
    assert all(parameter.dtype == torch.float32 for _, parameter in found.values())
    configure_shared_execution(model, 'stable-v1')
    validate_shared_execution(model, {'native_shared_execution': 'stable-v1'})


def test_native_loader_dispatches_lfm3_and_restores_saved_backbone(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from natlang_neuralese.model.lfm3_port import Lfm3PortBackbone
    from natlang_neuralese.model.native_backbone_restore import restore_backbone
    from natlang_neuralese.model.native_model import load_native_model, model_state
    from natlang_neuralese.third_party import lfm3

    hf = tiny_lfm3_hf()
    hf.save_pretrained(tmp_path, safe_serialization=True)
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'sample': 1, '<think>': 2, '</think>': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    hf_tokenizer.chat_template = LFM3_TINY_TEMPLATE
    hf_tokenizer.save_pretrained(tmp_path)

    # FP32 is a full-precision diagnostic load (no profile); LFM3 has no FP32-master profile (owner 2026-10-28).
    with pytest.raises(ValueError, match='no FP32-master'):
        load_native_model(tmp_path, rank=4, dtype=torch.float32, native_ordinary_precision='fp32-norms-v1')
    model, tokenizer, roles, config = load_native_model(tmp_path, rank=4, dtype=torch.float32)
    assert 'native_ordinary_precision' not in config
    assert isinstance(model.port, Lfm3PortBackbone)
    assert config['architecture'] == 'lfm3' and config['vendored_modeling']['sources'] == lfm3.SOURCES
    assert config['vendored_modeling']['revision'] == lfm3.REVISION
    assert model.port.embedding_weight.shape[0] == 64  # the four role tokens fit the existing rows: no resize
    assert roles.ids == (4, 5, 6, 7) and model.token_override_ids == (2, 3)
    ids = torch.tensor([[1, 4, 9, 10, 11, 5, 12, 13, 14, 15]])
    with torch.no_grad():
        expected = hf(ids).logits
        actual = model.logits(model.port.forward_ids(ids)['h_final'])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    state = model_state(model)['backbone_state']
    restored, _ = restore_backbone(tmp_path, 'lfm3', state, device='cpu', dtype=torch.float32)
    assert type(restored).__module__ == 'natlang_neuralese.third_party.lfm3.modeling_lfm3'
    assert all(torch.equal(state[name], restored.state_dict()[name]) for name in state)
    # A BF16 load selects the all-BF16 profile: backbone, norm gains, learned rows and adapter, no FP32 masters.
    half, _, _, half_config = load_native_model(tmp_path, rank=4, dtype=torch.bfloat16)
    assert half_config['native_ordinary_precision'] == 'bf16-stochastic-rounding-v1'
    assert half_config['sparse_row_dtype'] == 'torch.bfloat16'
    assert half_config['token_override_dtype'] == 'torch.bfloat16'
    assert all(parameter.dtype == torch.bfloat16 for parameter in half.parameters())
    with pytest.raises(ValueError, match='BF16'):
        load_native_model(tmp_path, rank=4, dtype=torch.bfloat16, sparse_row_dtype=torch.float32)


def test_lfm3_checkpoint_weights_only_reads_cast_across_dtypes(tmp_path):
    """An LFM3 checkpoint is one dtype throughout: GGUF export reads an all-BF16 checkpoint at FP32 (and an FP32
    diagnostic at BF16) as a cast of every tensor; only a weights-only read may change the dtype."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from natlang_neuralese.model.native_model import (CHECKPOINT_SCHEMA, load_native_checkpoint,
                                                      load_native_model, model_state)
    base = tmp_path / 'base'
    tiny_lfm3_hf().save_pretrained(base, safe_serialization=True)
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'sample': 1, '<think>': 2, '</think>': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    hf_tokenizer.chat_template = LFM3_TINY_TEMPLATE
    hf_tokenizer.save_pretrained(base)
    ids = torch.tensor([[1, 4, 9, 10, 11, 5, 12, 13, 2, 3]])
    for saved_dtype, read_dtype in ((torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)):
        model, _, _, config = load_native_model(base, rank=4, dtype=saved_dtype)
        with torch.no_grad():
            model.adapter.up.weight.normal_(0, 0.1)
            model.input_rows.add_(0.05)
            model.extra_input_rows.add_(0.05)
        path = tmp_path / f'{saved_dtype}.pt'
        torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, path)
        with pytest.raises(ValueError, match='only at its saved dtype'):
            load_native_checkpoint(path, dtype=read_dtype)
        read, _, _, read_config, _ = load_native_checkpoint(path, dtype=read_dtype, allow_weights_only=True)
        assert {parameter.dtype for parameter in read.parameters()} == {read_dtype}
        assert read_config.get('native_ordinary_precision') == (
            'bf16-stochastic-rounding-v1' if read_dtype == torch.bfloat16 else None)
        for name in ('input_rows', 'extra_input_rows'):
            torch.testing.assert_close(getattr(read, name), getattr(model, name).to(read_dtype))
        torch.testing.assert_close(read.adapter.up.weight, model.adapter.up.weight.to(read_dtype))
        with torch.no_grad():
            expected = model.logits(model.port.forward_ids(ids)['h_final']).float()
            actual = read.logits(read.port.forward_ids(ids)['h_final']).float()
        torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.05)


def test_native_optimizer_updates_every_lfm3_layer_form_from_an_import_graph():
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, packed_graph_loss

    torch.manual_seed(42)
    port = tiny_lfm3(markers=False)
    model = SharedStateModel(port, (58, 59, 60, 61), rank=4)
    optimizer, _ = make_native_optimizer(model, lr=1e-3, weight_decay=0., in_backward=False)
    raw = {'schema': SCHEMA, 'id': 'lfm3-optimizer', 'split': 'train', 'source_groups': ['lfm3-optimizer-contract'],
           'tokenizer_fingerprint': '1' * 64,
           'nodes': [{'name': 'producer', 'token_ids': [4, 5, 6, 7, 8, 9], 'producing': [[1, 5]],
                      'target_weights': [0.] * 6},
                     {'name': 'consumer', 'token_ids': [10, 11, 12, 13, 14, 15, 16, 17],
                      'target_weights': [0., 1., 0., 1., 1., 1., 1., 1.],
                      'imports': [{'at': 2, 'source': 'producer', 'start': 1, 'stop': 5}]}]}
    example = parse_graph(raw, tokenizer_fingerprint='1' * 64)
    parameters = dict(port.hf.named_parameters())
    targets = {name: parameters[name] for name in (
        'model.layers.0.conv.conv.weight', 'model.layers.1.self_attn.gate_proj.weight',
        'model.layers.3.self_attn.o_proj.weight', 'model.layers.3.post_operator_norm.weight',
        'model.layers.5.post_ffn_norm.weight')}
    before = {name: value.detach().clone() for name, value in targets.items()}
    loss, _ = packed_graph_loss(model, [example], checkpoint_layers=False)
    loss.backward()
    for name, value in targets.items():
        assert value.grad is not None and value.grad.abs().sum() > 0, name
    optimizer.step()
    for name, value in targets.items():
        assert not torch.equal(before[name], value), name


def test_adamwsr_matches_adamw_in_fp32_and_keeps_small_bf16_updates_in_expectation():
    from natlang_neuralese.train.optim import AdamWSR
    torch.manual_seed(0)
    gradients = [torch.randn(64, 32) for _ in range(5)]
    exact = torch.nn.Parameter(torch.randn(64, 32))
    mine = torch.nn.Parameter(exact.detach().clone())
    reference = torch.optim.AdamW([exact], lr=1e-2, weight_decay=0.1, foreach=False)
    candidate = AdamWSR([mine], lr=1e-2, weight_decay=0.1)
    for gradient in gradients:
        exact.grad, mine.grad = gradient.clone(), gradient.clone()
        reference.step()
        candidate.step()
    torch.testing.assert_close(mine, exact, atol=1e-6, rtol=1e-6)
    # A constant gradient makes every AdamW step ~lr = 1e-4, far below BF16's 2**-8 spacing just under 1.0.
    start = torch.ones(256, 64, dtype=torch.bfloat16)
    nearest = torch.nn.Parameter(start.clone())
    stochastic = torch.nn.Parameter(start.clone())
    fp32 = torch.nn.Parameter(start.float())
    optimizers = [torch.optim.AdamW([nearest], lr=1e-4, weight_decay=0.),
                  AdamWSR([stochastic], lr=1e-4, weight_decay=0.),
                  torch.optim.AdamW([fp32], lr=1e-4, weight_decay=0.)]
    for _ in range(100):
        for parameter, optimizer in zip((nearest, stochastic, fp32), optimizers):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
    assert torch.equal(nearest, start)  # round-to-nearest loses every update
    expected = float((fp32 - 1).mean())
    actual = float((stochastic.float() - 1).mean())
    assert expected == pytest.approx(-0.01, rel=1e-3)
    assert actual == pytest.approx(expected, rel=0.05)
    assert stochastic.dtype == torch.bfloat16 and optimizers[1].state[stochastic]['exp_avg'].dtype == torch.bfloat16


def test_all_bf16_native_optimizer_moves_small_sparse_row_updates_with_stochastic_rounding(tmp_path):
    from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.native_state_sft import (make_native_optimizer, packed_graph_loss,
                                                          sparse_row_updates)
    from natlang_neuralese.train.optim import AdamWSR
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    tiny_lfm3_hf().save_pretrained(tmp_path, safe_serialization=True)
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'sample': 1, '<think>': 2, '</think>': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]')
    hf_tokenizer.chat_template = LFM3_TINY_TEMPLATE
    hf_tokenizer.save_pretrained(tmp_path)
    raw = {'schema': SCHEMA, 'id': 'lfm3-bf16-rows', 'split': 'train', 'source_groups': ['lfm3-bf16-rows'],
           'tokenizer_fingerprint': '1' * 64,
           'nodes': [{'name': 'producer', 'token_ids': [4, 9, 10, 11, 5, 12], 'producing': [[0, 5]],
                      'target_weights': [0., 1., 1., 1., 1., 1.]},
                     {'name': 'consumer', 'token_ids': [6, 13, 14, 7, 15, 16, 17, 18],
                      'target_weights': [0., 0., 1., 1., 1., 1., 1., 1.],
                      'imports': [{'at': 1, 'source': 'producer', 'start': 1, 'stop': 4}]}]}
    example = parse_graph(raw, tokenizer_fingerprint='1' * 64)
    changes = {}
    for rounding in ('stochastic', 'nearest'):
        torch.manual_seed(5)
        model, _, _, _ = load_native_model(tmp_path, rank=4, dtype=torch.bfloat16)
        # Per-step updates (~lr) well below the role rows' BF16 spacing.
        optimizer, _ = make_native_optimizer(model, lr=2e-5, weight_decay=0., in_backward=False,
                                             learned_row_lr=2e-5)
        assert isinstance(optimizer.auxiliary, AdamWSR) and optimizer.auxiliary_rounding == 'stochastic'
        assert {entry['dtype'] for entry in optimizer.schema} == {'torch.bfloat16'}
        assert optimizer.state_dict()['adamw_rounding'] == 'stochastic'
        if rounding == 'nearest':  # the same routing with round-to-nearest writes, for contrast
            optimizer.auxiliary = torch.optim.AdamW(optimizer.auxiliary.param_groups, lr=2e-5, weight_decay=0.)
        with torch.no_grad():  # rows of magnitude 0.5-1 (BF16 spacing >= 2**-9): every step is far below it
            sign = torch.randint(0, 2, model.input_rows.shape, generator=torch.Generator().manual_seed(1)) * 2 - 1
            model.input_rows.copy_(sign * (0.5 + 0.4 * torch.rand(model.input_rows.shape)))
        start = model.input_rows.detach().clone()
        for _ in range(40):
            before = {'input_rows': model.input_rows.detach().clone()}
            loss, _ = packed_graph_loss(model, [example], checkpoint_layers=False)
            loss.backward()
            optimizer.auxiliary.step()
            telemetry = sparse_row_updates(model, before)
            optimizer.zero_grad(set_to_none=True)
        changes[rounding] = float((model.input_rows.detach().float() - start.float()).abs().sum())
        rows = telemetry['input_rows']
        assert rows['dtype'] == 'torch.bfloat16' and all(row['gradient_present'] for row in rows['rows'])
    assert changes['nearest'] == 0
    assert changes['stochastic'] > 0
