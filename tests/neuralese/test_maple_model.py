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


def test_phase_f_ternary_adapters_and_student_teacher(pair, tmp_path):
    """Phase F on Maple: the student's own QAT adapters are released, a snapshot is the teacher ("deltas off"),
    and checkpoint state covers only released adapters."""
    from natlang_neuralese.maple.nested_train import Member, setup
    from natlang_neuralese.train.adapters import adapter_layers, deltas_off, inject_lora, lora_state

    _, ours = pair
    setup(ours, [Member.parse("2x3", 4)], rank=2, private_rank=2, learn_scales=True, expert_scales=True)
    with torch.no_grad():
        for name, p in ours.named_parameters():
            if name.endswith("lora_B"):
                p.normal_(0, 0.05)  # a trained student
    for p in ours.parameters():
        p.requires_grad_(False)
    port = MaplePortBackbone(ours, ControlTokens(open_id=94, close_id=95))
    ids = torch.randint(0, 90, (1, WINDOW))
    with torch.no_grad():
        student = port.forward_ids(ids)["logits"]
    assert adapter_layers(port) == [] and lora_state(port) == {}
    grouped = inject_lora(port, [3, 2], rank=2, alpha=4)
    assert sorted(grouped) == [2, 3] and all(p.requires_grad for ps in grouped.values() for p in ps)
    assert len(grouped[3]) == 12  # q, k, v, o: lora_A, lora_B, learned scale
    with torch.no_grad():
        for p in grouped[3]:
            p.add_(0.1)
        moved = port.forward_ids(ids)["logits"]
        with deltas_off(port):
            teacher = port.forward_ids(ids)["logits"]
    assert not torch.allclose(moved, student)
    assert torch.allclose(teacher, student, atol=1e-5)
    assert adapter_layers(port) == [2, 3]
    names = dict(ours.named_parameters())
    state = lora_state(port)
    assert len(state) == 24 and all(k in names for k in state)
    assert not any(p.requires_grad for n, p in ours.named_parameters() if ".layers.0." in n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused MoE kernel is Triton/CUDA")
@pytest.mark.parametrize("blocks", [False, True])
@pytest.mark.parametrize("tokens", [1, 7, 300])
def test_fused_moe_matches_reference_loop(tokens, blocks, monkeypatch):
    """Fused grouped ternary experts (maple/fused_moe.py) against SparseMoE's reference loop: outputs and input
    gradients, for decode-sized and prefill-sized inputs."""
    from natlang_neuralese.maple.model import MapleConfig, SparseMoE

    torch.manual_seed(0)
    config = MapleConfig(hidden_size=128, num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64)
    moe = SparseMoE(config).cuda().to(torch.bfloat16)
    for e in range(16):
        moe.experts.set_expert(e, torch.randn(64, 128), torch.randn(64, 128), torch.randn(128, 64))
    moe.experts.gate_up_scale.mul_(0.05)
    moe.experts.down_scale.mul_(0.05)
    if blocks:  # the Maple student's frozen learned block scales (one per 256 columns), perturbed per block
        for p in moe.experts.learn_scales(block=32):
            p.data.mul_(torch.rand_like(p) + 0.5)
            p.requires_grad_(False)
    x = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)

    def run(fused):
        monkeypatch.setenv("NATLANG_MAPLE_FUSED_MOE", "1" if fused else "0")
        inp = x.clone().requires_grad_(True)
        out = moe(inp)
        (out.float() * torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)).sum().backward()
        return out.detach().float(), inp.grad.float()

    ref_out, ref_grad = run(False)
    out, grad = run(True)
    scale = ref_out.abs().max()
    assert (out - ref_out).abs().max() <= 2e-2 * scale
    assert (grad - ref_grad).abs().max() <= 2e-2 * ref_grad.abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused MoE kernel is Triton/CUDA")
@pytest.mark.parametrize("tokens", [3, 300])
def test_fused_moe_trains_block_scales(tokens, monkeypatch):
    """QAT of the experts' learned block scales through the fused kernels: block-scale and input gradients match the
    reference loop."""
    from natlang_neuralese.maple.model import MapleConfig, SparseMoE

    torch.manual_seed(0)
    config = MapleConfig(hidden_size=128, num_experts=16, num_experts_per_tok=4, moe_intermediate_size=64)
    moe = SparseMoE(config).cuda().to(torch.bfloat16)
    for e in range(16):
        moe.experts.set_expert(e, torch.randn(64, 128), torch.randn(64, 128), torch.randn(128, 64))
    moe.experts.gate_up_scale.mul_(0.05)
    moe.experts.down_scale.mul_(0.05)
    blocks = moe.experts.learn_scales(block=32)
    for p in blocks:
        p.data.mul_(torch.rand_like(p) + 0.5)
    x = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)

    def run(fused):
        monkeypatch.setenv("NATLANG_MAPLE_FUSED_MOE", "1" if fused else "0")
        for p in blocks:
            p.grad = None
        inp = x.clone().requires_grad_(True)
        out = moe(inp)
        (out.float() * torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out)).sum().backward()
        return inp.grad.float(), [p.grad.float().clone() for p in blocks]

    ref_input, ref_scales = run(False)
    grad_input, grad_scales = run(True)
    assert (grad_input - ref_input).abs().max() <= 2e-2 * ref_input.abs().max()
    for got, want in zip(grad_scales, ref_scales):
        assert got.shape == want.shape
        assert (got - want).abs().max() <= 2e-2 * want.abs().max()


def test_full_qat_policy(pair):
    """Maple's full QAT policy: unchanged initial forward; dense attention latents (Muon), learned attention and
    expert block scales, routers and layer norm gains (AdamW) train; LoRA, embedding, head and final norm stay
    frozen; gradients reach every kind; a dense latent flips codes and the tracker sees flips and oscillation."""
    from natlang_neuralese.maple.nested_train import Member, setup
    from natlang_neuralese.maple.ternary import CodeTracker
    from natlang_neuralese.train.adapters import install_maple_qat, maple_qat_parameters
    from natlang_neuralese.train.optim import PortMuonAdamW

    _, ours = pair
    setup(ours, [Member.parse("2x3", 4)], rank=2, private_rank=2, learn_scales=True, expert_scales=True)
    for p in ours.parameters():
        p.requires_grad_(False)
    port = MaplePortBackbone(ours, ControlTokens(open_id=94, close_id=95))
    ids = torch.randint(0, 90, (1, WINDOW))
    with torch.no_grad():
        before = port.forward_ids(ids)["logits"]
    named = maple_qat_parameters(port)
    install_maple_qat(port)  # idempotent (serving installs before restoring values)
    assert len(named) == len(maple_qat_parameters(port))
    names = [n for n, _ in named]
    layers = port.num_layers
    assert sum(n.endswith(".dense") for n in names) == 4 * layers
    assert sum(n.endswith(".learned_scale") for n in names) == 4 * layers
    assert sum(n.endswith("_blocks") for n in names) == 2 * layers
    assert sum(n.endswith(".mlp.gate.weight") for n in names) == layers
    assert sum(n.endswith("norm.weight") for n in names) == 4 * layers
    assert not any("lora_" in n or "embed" in n or "lm_head" in n or n == "model.norm.weight" for n in names)
    with torch.no_grad():
        assert torch.allclose(port.forward_ids(ids)["logits"], before, atol=1e-5)
    port.forward_ids(ids)["logits"].float().pow(2).mean().backward()
    for kind in (".dense", ".learned_scale", "_blocks", ".mlp.gate.weight", "norm.weight"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in named if n.endswith(kind)), kind
    schema = {row["name"]: row["optimizer"] for row in PortMuonAdamW(named, lr=1e-3, vocab_size=96).schema}
    assert all((v == "muon") == k.endswith(".dense") for k, v in schema.items())
    tracker = CodeTracker(ours)
    assert tracker.update()["flipped_from_base"] == 0
    dense = next(p for n, p in named if n.endswith("q_proj.parametrizations.weight.0.dense"))
    base = dict(ours.named_parameters())[next(n for n in names if n.endswith("q_proj.parametrizations.weight.0.dense"))
                                         .replace("0.dense", "original")]
    with torch.no_grad():
        dense[0, 0] = -3 * base[0, 0].sign() * base.abs().max() - 1.0
    first = tracker.update()
    assert first["flipped_from_base"] > 0 and first["changed_since_last"] > 0
    with torch.no_grad():
        dense.zero_()
    second = tracker.update()
    assert second["flipped_from_base"] == 0 and second["changed_twice_or_more"] > 0


def test_family_term_and_member_evaluation_for_neuralese_stages(pair):
    from types import SimpleNamespace

    from natlang_neuralese.maple.family import (evaluate_members, family_members, member_backward, private_parameters,
                                                window_labels)
    from natlang_neuralese.maple.nested_train import Member, setup

    _, ours = pair
    members = [Member.parse("5", 4), Member.parse("2x3", 4)]
    _, _, private = setup(ours, members, rank=2, private_rank=2, learn_scales=True, expert_scales=True)
    ours.members = members
    backbone = SimpleNamespace(hf=ours)
    assert [m.key for m in family_members(backbone)] == ["4x5", "2x3"]
    assert {id(p) for _, p in private_parameters(backbone)} == {id(p) for p in private}
    ids = torch.randint(0, 90, (1, 2 * WINDOW))
    labels = window_labels(ids, WINDOW)
    assert (labels[:, :WINDOW - 1] == -100).all() and torch.equal(labels[0, WINDOW - 1:-1], ids[0, WINDOW:])
    for p in ours.parameters():
        p.requires_grad_(False)
    for p in private:
        p.requires_grad_(True)
    parts = member_backward(backbone, members[1], ids, labels, weight=0.5, chunk=4)
    assert parts.tokens == WINDOW and parts.kl >= 0
    assert ours.model.layers[0].mlp.private_gate["2x3"].grad is not None
    assert ours.model.active_layers is None
    report = evaluate_members(backbone, [(ids, labels)], chunk=4)
    assert set(report) == {"full", "4x5", "2x3"} and report["full"]["kl"] < 1e-5
    assert all(row["tokens"] == WINDOW and row["ce"] > 0 for row in report.values())


def test_qat_latent_scales_name_every_dense_latent_with_its_ternary_scale(pair):
    from natlang_neuralese.train.adapters import maple_qat_parameters, qat_latent_scales

    _, ours = pair
    port = MaplePortBackbone(ours, ControlTokens(open_id=94, close_id=95))
    names = {n for n, _ in maple_qat_parameters(port) if n.endswith(".dense")}
    scales = qat_latent_scales(port)
    assert names and set(scales) == names and all(v > 0 for v in scales.values())


def test_member_backward_full_anchor_reaches_shared_weights(pair):
    from types import SimpleNamespace

    from natlang_neuralese.maple.family import member_backward, window_labels
    from natlang_neuralese.maple.nested_train import Member, setup

    _, ours = pair
    members = [Member.parse("2x3", 4)]
    adapters, _, private = setup(ours, members, rank=2, private_rank=2, learn_scales=True, expert_scales=True)
    for p in ours.parameters():
        p.requires_grad_(False)
    for p in private + adapters:
        p.requires_grad_(True)
    ids = torch.randint(0, 90, (1, 2 * WINDOW))
    member_backward(SimpleNamespace(hf=ours), members[0], ids, window_labels(ids, WINDOW), weight=0.5, chunk=4,
                    full_weight=1.0)
    layer3 = [p for n, p in ours.named_parameters() if "layers.3." in n and "lora_" in n and ".private." not in n]
    assert layer3 and any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer3)  # only the full model reaches layer 3
