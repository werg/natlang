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


def test_family_step_bootstrap_and_joint(pair):
    from natlang_neuralese.maple.nested_train import Member, member_step, setup

    _, ours = pair
    members = [Member.parse("5", 4), Member.parse("2x3", 4)]
    assert members[1].layers == 2 and members[1].key == "2x3" and members[0].key == "4x5"
    adapters, scales, private = setup(ours, members, rank=2, private_rank=2, learn_scales=True, expert_scales=True)
    ids = torch.randint(0, 96, (1, WINDOW))
    labels = torch.cat([ids[:, 1:], torch.full((1, 1), -100)], 1)
    common = dict(normaliser=WINDOW - 1, ce_member=0.1, kl_weight=1.0, hidden_weight=1.0, anchor_weight=1.0,
                  chunk=4, total_layers=4)
    for p in adapters + scales:
        p.requires_grad_(False)
    sums = member_step(ours, members, ids, labels, phase="bootstrap", **common)
    assert sums["4x5/kl"] >= 0 and "2x3/hidden" in sums and "full_ce" not in sums
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in private)
    router = ours.model.layers[0].mlp.private_gate["2x3"]
    assert router.shape == (3, 64) and router.grad is not None and "2x3" not in ours.model.layers[3].mlp.private_gate
    for p in adapters + scales:
        p.requires_grad_(True)
    sums = member_step(ours, members, ids, labels, phase="joint", **common)
    assert "full_ce" in sums and "anchor_kl" in sums
    assert all(p.grad is not None for p in adapters)
    assert ours.model.layers[0].mlp.active_experts is None and ours.model.active_layers is None


def test_converted_cache_roundtrip(tmp_path):
    reference, config = _reference()
    ckpt = _save(reference, config, tmp_path / "ckpt")
    cache = tmp_path / "converted"
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


def test_member_export_matches_the_trained_forward(pair):
    from natlang_neuralese.maple.export_member import member_attention, member_experts
    from natlang_neuralese.maple.nested_train import Member, setup

    _, ours = pair
    members = [Member.parse("5", 4)]
    adapters, scales, private = setup(ours, members, rank=2, private_rank=2, learn_scales=True, expert_scales=True)
    torch.manual_seed(4)
    with torch.no_grad():
        for p in adapters + private:
            if p.dim() == 2:
                p.add_(torch.randn_like(p) * 0.02)
        for p in scales:
            p.mul_(1 + 0.1 * torch.rand_like(p))
    state = {k.removeprefix("model."): v.detach() for k, v in ours.state_dict(keep_vars=True).items()}
    module = ours.model.layers[1].self_attn.k_proj
    prefix = "layers.1.self_attn.k_proj.parametrizations.weight.0"
    base = module.parametrizations.weight.original
    for key in (None, "4x5"):
        ours.set_member(key, experts=5 if key else None)
        trained = module.weight.detach().float()
        ours.set_member(None)
        assert torch.allclose(member_attention(base, state, prefix, key, 2.0, 2.0), trained, atol=1e-6)
    experts = ours.model.layers[0].mlp.experts
    gate_up, _ = experts.weights(slice(0, 5), torch.float32)
    original = experts.gate_up_codes[:5].float() * experts.gate_up_scale[:5].float()
    blocks = state["layers.0.mlp.experts.gate_up_blocks"][:5]
    assert torch.allclose(member_experts(original, blocks.repeat(1, 1, 1)), gate_up, atol=1e-6)


@pytest.mark.parametrize("block", [4, 6, 16])
def test_ternary_matmul_matches_dense_gradients(block):
    from natlang_neuralese.maple.model import _TernaryMatmul, _dense

    torch.manual_seed(0)
    codes = torch.randint(-1, 2, (10, 16), dtype=torch.int8)
    scale = torch.rand(10, -(-16 // block), dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 5, 16, dtype=torch.float64, requires_grad=True)
    y = _TernaryMatmul.apply(x, codes, scale, block)
    g = torch.randn_like(y)
    gx, gs = torch.autograd.grad(y, (x, scale), g)
    x2, s2 = x.detach().requires_grad_(), scale.detach().requires_grad_()
    y2 = x2 @ _dense(codes, s2, block, torch.float64).T
    ex, es = torch.autograd.grad(y2, (x2, s2), g)
    assert torch.allclose(y, y2) and torch.allclose(gx, ex) and torch.allclose(gs, es)
