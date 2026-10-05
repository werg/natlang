import json
import sys
from pathlib import Path

import pytest
import torch

from natlang_neuralese.maple.maple_port import MaplePortBackbone
from natlang_neuralese.maple.model import load_maple
from natlang_neuralese.maple.ternary import ternarize
from natlang_neuralese.model.lfm2_port import ControlTokens

sys.path.insert(0, str(Path(__file__).parent))
from maple_reference import fa3 as reference_attention  # noqa: E402
from maple_reference.configuration_maple import MapleConfig as ReferenceConfig  # noqa: E402
from maple_reference.modeling_maple import MapleForCausalLM as ReferenceMaple  # noqa: E402

WINDOW = 6


def _reference(experts=8, seed=0):
    config = ReferenceConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=experts, num_experts_per_tok=3,
        sliding_window=WINDOW, layer_types=["sliding_attention", "sliding_attention", "sliding_attention",
                                            "full_attention"],
        partial_rotary_factor=0.5, rope_theta=10000, rms_norm_eps=1e-6, max_position_embeddings=256,
        tie_word_embeddings=False, nope_on_global_attention=True, norm_topk_prob=True)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = ReferenceMaple(config).float().eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and not name.endswith("lm_head"):
                module.weight.copy_(ternarize(module.weight))
        for norm in [m for n, m in model.named_modules() if n.endswith("norm")]:
            norm.weight.normal_(1.0, 0.1)
    return model, config


def _save(model, config, path: Path):
    from safetensors.torch import save_file

    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    for name in list(state):
        if name.endswith("mlp.gate.weight"):
            continue
        state[name] = state[name].to(torch.bfloat16).float()  # the checkpoint is BF16
    save_file(state, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in state}}))
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    return path


@pytest.fixture
def pair(tmp_path):
    reference, config = _reference()
    ours = load_maple(_save(reference, config, tmp_path / "ckpt"), dtype=torch.float32)
    reference.load_state_dict({k: v.to(torch.bfloat16).float() if not k.endswith("mlp.gate.weight") else v
                               for k, v in reference.state_dict().items()})
    return reference, ours


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_matches_reference_within_window(pair):
    reference, ours = pair
    ids = torch.randint(0, 96, (2, WINDOW))
    assert torch.allclose(_logits(ours, ids), _logits(reference, ids), atol=2e-4)


def test_window_rule_is_llamacpp(pair, monkeypatch):
    reference, ours = pair
    ids = torch.randint(0, 96, (1, 3 * WINDOW))
    monkeypatch.setattr(reference_attention, "INCLUSIVE", False)
    assert torch.allclose(_logits(ours, ids), _logits(reference, ids), atol=2e-4)
    monkeypatch.setattr(reference_attention, "INCLUSIVE", True)
    assert not torch.allclose(_logits(ours, ids), _logits(reference, ids), atol=2e-4)


def test_nested_experts_equal_a_truncated_model(pair, tmp_path):
    reference, ours = pair
    small_ref, config = _reference(experts=5)
    state = reference.state_dict()
    small_state = {}
    for k, v in state.items():
        if ".mlp.experts." in k and int(k.split(".mlp.experts.")[1].split(".")[0]) >= 5:
            continue
        small_state[k] = v[:5] if k.endswith("mlp.gate.weight") else v
    small_ref.load_state_dict(small_state)
    ids = torch.randint(0, 96, (2, WINDOW))
    ours.set_active_experts(5)
    assert torch.allclose(_logits(ours, ids), _logits(small_ref, ids), atol=2e-4)
    ours.set_active_experts(None)


def test_expert_order_is_invisible_and_statistics_add_up(pair):
    _, ours = pair
    ids = torch.randint(0, 96, (2, WINDOW))
    before = _logits(ours, ids)
    stats = ours.collect_routing()
    _logits(ours, ids)
    assert stats[0]["tokens"] == 2 * WINDOW
    assert torch.allclose(stats[0]["mass"].sum(), torch.tensor(2.0 * WINDOW), atol=1e-4)
    assert stats[0]["count"].sum() == 2 * WINDOW * 3
    orders = [torch.argsort(s["mass"], descending=True) for s in stats]
    ours.collect_routing(False)
    ours.order_experts(orders)
    assert torch.allclose(_logits(ours, ids), before, atol=1e-5)


def test_port_backbone_runs_maple_layer_ranges(pair):
    _, ours = pair
    port = MaplePortBackbone(ours, ControlTokens(open_id=94, close_id=95))
    assert not port.tied
    ids = torch.randint(0, 90, (1, 2 * WINDOW))
    with torch.no_grad():
        full = port.forward_ids(ids)
        split = port.forward_ids(ids, cutoff=2)
        prefix = port.forward_ids(ids[:, :WINDOW])
        step = port.forward_ids(ids[:, WINDOW:], cache=prefix["cache"])
    assert torch.allclose(full["logits"][..., :94], _logits(ours, ids)[..., :94], atol=2e-4)
    assert torch.allclose(split["h_final"], full["h_final"], atol=1e-5)
    assert torch.allclose(step["h_final"], full["h_final"][:, WINDOW:], atol=2e-4)


def test_nested_step_trains_shared_lora_and_size_bias(pair):
    from natlang_neuralese.maple.nested_train import nested_losses, prepare

    _, ours = pair
    params = prepare(ours, [(5, 1.0)], rank=2, alpha=4.0)
    adapters = [p for p in params if p.dim() == 2]
    with torch.no_grad():
        for p in adapters:
            if p.shape[1] == 2:  # lora_B
                p.normal_(0, 0.01)
    ids = torch.randint(0, 96, (1, WINDOW))
    labels = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1)
    loss, parts = nested_losses(ours, ids, labels, 5, kl_weight=1.0, small_weight=1.0, normaliser=WINDOW - 1,
                                chunk=4)
    loss.backward()
    bias = ours.model.layers[0].mlp.size_bias["5"]
    assert bias.grad is not None and bias.grad.abs().sum() > 0
    assert all(p.grad is not None for p in adapters)
    assert parts["kl"] > 0 and ours.model.layers[0].mlp.active_experts is None


def test_converted_cache_roundtrip(tmp_path):
    reference, config = _reference()
    ckpt = _save(reference, config, tmp_path / "ckpt")
    cache = tmp_path / "converted.pt"
    first = load_maple(ckpt, dtype=torch.bfloat16, cache=cache)
    assert cache.exists()
    second = load_maple(ckpt, dtype=torch.bfloat16, cache=cache)
    assert second.lm_head.weight.dtype == torch.bfloat16 and second.model.layers[0].mlp.gate.weight.dtype == torch.float32
    ids = torch.randint(0, 96, (1, WINDOW))
    assert torch.equal(_logits(first, ids), _logits(second, ids))


def test_rotary_frequencies_stay_fp32(tmp_path):
    reference, config = _reference()
    model = load_maple(_save(reference, config, tmp_path / "ckpt"), dtype=torch.bfloat16)
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32


def test_expert_scale_only_qat(pair):
    from natlang_neuralese.maple.ternary import adapters_disabled

    _, ours = pair
    ids = torch.randint(0, 96, (1, WINDOW))
    before = _logits(ours, ids)
    params = [p for layer in ours.model.layers for p in layer.mlp.experts.learn_scales(block=16)]
    after = ours(input_ids=ids).logits
    assert torch.allclose(after, before, atol=1e-2)  # FP16 rounding of the scales only
    after.float().pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    with torch.no_grad():
        for p in params:
            p.mul_(1.5)
        with adapters_disabled():
            assert torch.allclose(_logits(ours, ids), before, atol=1e-5)
        assert not torch.allclose(_logits(ours, ids), before, atol=1e-2)
