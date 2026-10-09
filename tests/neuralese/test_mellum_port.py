"""Mellum 2.x through the Maple-family implementation and port: exact parity with transformers' Mellum on a tiny
random checkpoint (per-layer-type RoPE with YaRN on global layers, sliding window, unclamped SwiGLU, BF16 experts)."""
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "MellumConfig"):
    pytest.skip("installed transformers has no Mellum", allow_module_level=True)

from natlang_neuralese.maple.model import DenseExperts, MapleConfig, load_maple

LAYERS, EXPERTS, FF = 4, 4, 32


def tiny_mellum(tmp_path):
    from safetensors.torch import save_file
    from transformers import MellumConfig, MellumForCausalLM

    torch.manual_seed(0)
    config = MellumConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, moe_intermediate_size=FF, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=EXPERTS, num_experts_per_tok=2,
        norm_topk_prob=True, sliding_window=8, use_sliding_window=True, max_position_embeddings=64,
        layer_types=["sliding_attention"] * 3 + ["full_attention"], mlp_layer_types=["sparse"] * LAYERS,
        rope_parameters={
            "full_attention": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                               "original_max_position_embeddings": 16, "beta_fast": 32.0, "beta_slow": 1.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
        tie_word_embeddings=False, dtype="float32")
    reference = MellumForCausalLM(config).float().eval()
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(0, 0.05)
        for module in reference.modules():
            if type(module).__name__.endswith("RMSNorm"):
                module.weight.uniform_(0.8, 1.2)
    tensors = {}
    for name, value in reference.state_dict().items():
        if ".mlp.experts." in name:
            prefix = name.split(".mlp.experts.")[0] + ".mlp.experts"
            for expert in range(EXPERTS):
                if name.endswith("gate_up_proj"):
                    tensors[f"{prefix}.{expert}.gate_proj.weight"] = value[expert, :FF].contiguous()
                    tensors[f"{prefix}.{expert}.up_proj.weight"] = value[expert, FF:].contiguous()
                else:
                    tensors[f"{prefix}.{expert}.down_proj.weight"] = value[expert].contiguous()
        else:
            tensors[name.replace("mlp.gate.weight", "mlp.gate.weight")] = value.contiguous()
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}))
    data = config.to_dict()
    data["model_type"] = "mellum"
    (tmp_path / "config.json").write_text(json.dumps(data))
    return reference


def test_mellum_config_selects_full_rotary_per_layer_type_and_no_clamp(tmp_path):
    tiny_mellum(tmp_path)
    config = MapleConfig.from_dir(tmp_path)
    assert config.model_type == "mellum" and config.partial_rotary_factor == 1.0 and config.swiglu_clamp is None
    assert set(config.rope_parameters) == {"full_attention", "sliding_attention"}


def test_mellum_logits_match_transformers_reference_across_window_and_yarn_context(tmp_path):
    reference = tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False)
    assert isinstance(model.model.layers[0].mlp.experts, DenseExperts)
    ids = torch.randint(0, 128, (2, 24))  # longer than the window (8) and YaRN's original context (16)
    with torch.no_grad():
        expected = reference(input_ids=ids).logits
        actual = model(ids).logits
    assert torch.allclose(actual, expected, atol=2e-4, rtol=1e-4), (actual - expected).abs().max()


def test_mellum_truncated_load_keeps_per_type_rotary(tmp_path):
    tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False, layers=2)
    assert len(model.model.layers) == 2
    with torch.no_grad():
        assert model(torch.randint(0, 128, (1, 12))).logits.shape == (1, 12, 128)


def test_mellum_markers_are_its_spare_added_tokens_and_maple_keeps_unused_rows(tmp_path):
    from types import SimpleNamespace

    from natlang_neuralese.model.hf_port import QWEN_OPEN_ID, family_controls

    class Vocab:
        def __init__(self, vocab):
            self.vocab = vocab

        def get_vocab(self):
            return self.vocab

    mellum = SimpleNamespace(config=SimpleNamespace(model_type="mellum"))
    controls = family_controls(mellum, Vocab({"<|extra_token_7|>": 33, "<|extra_token_8|>": 34}))
    assert (controls.open_id, controls.close_id) == (33, 34)
    with pytest.raises(ValueError):
        family_controls(mellum, Vocab({}))
    maple = SimpleNamespace(config=SimpleNamespace(model_type="maple"))
    assert family_controls(maple, Vocab({"a": 0})).open_id == QWEN_OPEN_ID


def test_latent_ternary_experts_forward_their_codes_and_train_the_latents(tmp_path):
    from natlang_neuralese.maple.ternary import ternarize

    tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False)
    reference = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False)
    for layer, ref in zip(model.model.layers, reference.model.layers):
        layer.mlp.experts.make_latent()
        with torch.no_grad():
            for e in range(EXPERTS):
                ref.mlp.experts.gate_up[e] = ternarize(ref.mlp.experts.gate_up[e])
                ref.mlp.experts.down[e] = ternarize(ref.mlp.experts.down[e])
    ids = torch.randint(0, 128, (1, 12))
    logits = model(ids).logits
    with torch.no_grad():
        assert torch.allclose(logits, reference(ids).logits, atol=1e-5)
    logits.float().square().mean().backward()
    latent = model.model.layers[0].mlp.experts.gate_up
    assert isinstance(latent, torch.nn.Parameter) and latent.grad is not None and latent.grad.abs().sum() > 0


def test_stochastic_rounding_is_unbiased_and_lion_moves_by_lr():
    from natlang_neuralese.train.optim import LionSR, stochastic_round_

    torch.manual_seed(0)
    target = torch.empty(200_000, dtype=torch.bfloat16)
    stochastic_round_(target, torch.full((200_000,), 1.0 + 1e-4))
    assert abs(float(target.float().mean()) - (1.0 + 1e-4)) < 2e-5  # BF16 spacing at 1.0 is 7.8e-3
    p = torch.nn.Parameter(torch.zeros(1000, dtype=torch.float32))
    p.grad = torch.ones(1000)
    LionSR([p], lr=0.01).step()
    assert torch.allclose(p.detach(), torch.full((1000,), -0.01))


def test_full_latent_qat_ternarizes_attention_and_experts_and_trains_latents(tmp_path):
    from natlang_neuralese.maple.qat_convert import install_full_latent_qat, ternary_scale, topk_kl
    from natlang_neuralese.maple.ternary import ternarize

    tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False)
    reference = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False)
    latents = install_full_latent_qat(model)
    assert len(latents) == LAYERS * 6 and all(ternary_scale(q) > 0 for _, q in latents)
    with torch.no_grad():
        for layer in reference.model.layers:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(layer.self_attn, proj)
                module.weight.copy_(ternarize(module.weight))
            for e in range(EXPERTS):
                layer.mlp.experts.gate_up[e] = ternarize(layer.mlp.experts.gate_up[e])
                layer.mlp.experts.down[e] = ternarize(layer.mlp.experts.down[e])
    ids = torch.randint(0, 128, (1, 12))
    logits = model(ids).logits
    with torch.no_grad():
        assert torch.allclose(logits, reference(ids).logits, atol=1e-5)
    logits.float().square().mean().backward()
    attention = dict(latents)["model.layers.0.self_attn.q_proj.weight"]
    assert attention.grad is not None and attention.grad.abs().sum() > 0
    assert not any(p.requires_grad for n, p in model.named_parameters() if "embed" in n or "norm" in n)
    full = torch.log_softmax(torch.randn(5, 128), -1)
    values, index = full.topk(128, -1)
    assert float(topk_kl(full, index, values)) < 1e-6


def test_ternarization_ramp_interpolates_from_bf16_to_deployed(tmp_path):
    from natlang_neuralese.maple.qat_convert import install_full_latent_qat
    from natlang_neuralese.maple.ternary import QUANT_MIX

    tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False)
    original = load_maple(tmp_path, device="cpu", dtype=torch.float32, ternary_attention=False)
    install_full_latent_qat(model)
    ids = torch.randint(0, 128, (1, 12))
    try:
        with torch.no_grad():
            QUANT_MIX["value"] = 0.0
            assert torch.allclose(model(ids).logits, original(ids).logits, atol=1e-5)
            QUANT_MIX["value"] = 1.0
            deployed = model(ids).logits
            QUANT_MIX["value"] = 0.5
            half = model(ids).logits
        assert not torch.allclose(half, deployed, atol=1e-4) and not torch.allclose(half, original(ids).logits, atol=1e-4)
        QUANT_MIX["value"] = 0.5
        model(ids).logits.square().mean().backward()
        assert model.model.layers[0].self_attn.q_proj.parametrizations.weight.original.grad is not None
    finally:
        QUANT_MIX["value"] = 1.0
