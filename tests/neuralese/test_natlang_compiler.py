"""Targeted unit & parity tests for the Natlang Compiler program (`plans/NATLANG_COMPILER.md`)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "training/neuralese", ROOT / "scripts/neuralese_data"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from natlang_neuralese.compiler import hkb_dependency

try:
    hkb_dependency.require()
except ModuleNotFoundError as error:  # the declared dependency is absent on this machine
    pytest.skip(str(error), allow_module_level=True)

from hindsight_kb.config import ModelConfig
from hindsight_kb.integrations.native_lfm import NativeLFMDecoder
from hindsight_kb.models.adapters import AdapterLinear
from natlang_neuralese.third_party.lfm3.configuration_lfm3 import Lfm3Config
from natlang_neuralese.third_party.lfm3.modeling_lfm3 import Lfm3ForCausalLM

from natlang_neuralese.compiler import (
    HeaderEncoder,
    HeaderEncoderConfig,
    UnifiedCompilerHead,
    collapse_mode_p_memory,
    collapse_to_lora_factors,
    compose_cores,
    default_lfm3_tap_layers,
    provenance_key,
)
from natlang_neuralese.compiler.train_compiler import (
    CompilerTrainConfig,
    build_dual_lr_optimizer,
    context_distillation_loss,
    evaluate_reader_and_shuffle_ablations,
    fo_maml_inner_adapt_z,
    gate_floor_for_step,
    header_dropout_prob,
)
from natlang_neuralese.train.step_verifier import (
    BranchCandidate,
    StepVerifierHead,
    VerifierTrainConfig,
    compute_auroc,
    rank_branches,
    train_step_verifier,
)
from natlang_neuralese.tune.decision_compile import (
    compute_ece,
    decision_margin_loss,
    fit_temperature,
    should_settle_fast,
)
from natlang_neuralese.tune.latent_ttt import LatentTTTConfig, refine_site_latents, split_support_and_query
import call_level_engine


def _tiny_lfm3_decoder(rank: int = 4):
    lfm3_cfg = Lfm3Config(
        vocab_size=96,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        conv_L_cache=7,
        conv_bias=True,
        sliding_window=4,
        use_sandwich_norm=True,
        layer_types=["conv", "sliding_attention", "conv", "full_attention"],
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": 1.0},
        },
        norm_eps=1e-5,
    )
    hkb_cfg = ModelConfig(
        vocab_size=lfm3_cfg.vocab_size,
        hidden_dim=lfm3_cfg.hidden_size,
        encoder_layers=1,
        layer_types=("conv", "attention", "conv", "attention"),
        num_heads=lfm3_cfg.num_attention_heads,
        kv_heads=lfm3_cfg.num_key_value_heads,
        ff_dim=lfm3_cfg.intermediate_size,
        kv_head_dim=lfm3_cfg.head_dim,
        key_dim=16,
        value_dim=32,
        slots=2,
        rank=rank,
        max_positions=lfm3_cfg.max_position_embeddings,
        conv_width=7,
        kernel_backend="torch",
        memory_bank_size=4,
        adapter_heads=4,
        adapter_inner_rank=2,
    )
    hf = Lfm3ForCausalLM(lfm3_cfg).eval()
    dec = NativeLFMDecoder(hf, hkb_cfg).eval()
    return lfm3_cfg, hkb_cfg, dec


def test_header_encoder_unified_head_and_collapse_exact_parity():
    torch.manual_seed(42)
    lfm3_cfg, hkb_cfg, dec = _tiny_lfm3_decoder(rank=4)
    taps = default_lfm3_tap_layers(lfm3_cfg.num_hidden_layers)
    assert len(taps) == 3

    enc = HeaderEncoder(
        HeaderEncoderConfig(
            hidden_dim=lfm3_cfg.hidden_size,
            num_tap_layers=len(taps),
            num_slots=4,
            value_dim=hkb_cfg.value_dim,
            key_dim=hkb_cfg.key_dim,
            num_heads=4,
        )
    )
    head = UnifiedCompilerHead(hkb_cfg, dec.adapter_count, module_families=dec.module_families)
    with torch.no_grad():
        head.memory_bank.layer_gate.fill_(0.5)

    tokens = torch.randint(0, lfm3_cfg.vocab_size, (1, 8))
    blocks = torch.zeros_like(tokens)
    out0 = dec(tokens, blocks, tap_layers=taps)
    extra_guidance = torch.randn(1, 2, hkb_cfg.value_dim)
    z_site, k_site, w_site = enc(out0.tapped_hidden, out0.hidden, extra_context_slots=extra_guidance)
    assert z_site.shape == (1, 4, hkb_cfg.value_dim)
    assert k_site.shape == (1, 4, hkb_cfg.key_dim)

    head.eval()
    compiled = head(z_site, k_site, w_site)
    # Test linear weight-space composition
    c_twice = compose_cores(compiled.c_cores, compiled.c_cores)
    assert torch.allclose(c_twice, compiled.c_cores * 2.0)

    # Collapse B_eff = B @ C and verify exact matrix equality against B @ C @ A for every wrapped layer
    factors = collapse_to_lora_factors(dec, compiled.c_cores)
    assert len(factors) == dec.adapter_count
    by_id = {m.module_id: m for m in dec.modules() if isinstance(m, AdapterLinear)}
    for idx, name in enumerate(dec.adapter_names):
        a_exp, b_exp = factors[name]
        m = by_id[idx]
        c_mat = compiled.c_cores[0, idx].detach().float()
        delta_expected = m.B.detach().float() @ c_mat @ m.A.detach().float()
        delta_collapsed = b_exp @ a_exp
        assert torch.allclose(delta_collapsed, delta_expected, atol=1e-5, rtol=1e-5)

    mem_bundle = collapse_mode_p_memory(compiled.mem_kv, compiled.mem_valid, compiled.mem_gate)
    assert mem_bundle.memory_mode == "positioned"
    assert mem_bundle.slot_count == 4
    assert mem_bundle.pre_rope_k.shape == (4, hkb_cfg.attention_layers, hkb_cfg.kv_heads, hkb_cfg.head_dim)
    k1 = provenance_key(site_id="s", header_sha="h1", compiler_sha="c1", policy_sha="p1")
    k2 = provenance_key(site_id="s", header_sha="h2", compiler_sha="c1", policy_sha="p1")
    assert k1 != k2


def test_compiler_train_objectives_maml_and_ablations():
    torch.manual_seed(5)
    lfm3_cfg, hkb_cfg, dec = _tiny_lfm3_decoder(rank=4)
    head = UnifiedCompilerHead(hkb_cfg, dec.adapter_count, module_families=dec.module_families)
    cfg = CompilerTrainConfig(total_steps=100, gate_floor_steps=20)
    assert gate_floor_for_step(0, cfg) == pytest.approx(0.1)
    assert gate_floor_for_step(20, cfg) == 0.0
    assert header_dropout_prob(10, 100) == 0.0
    assert header_dropout_prob(100, 100) == pytest.approx(1.0)

    opt = build_dual_lr_optimizer(fresh_modules=[head], warm_modules=[], cfg=cfg)
    assert opt.param_groups[0]["lr"] == cfg.fresh_lr

    tokens = torch.randint(0, lfm3_cfg.vocab_size, (2, 8))
    blocks = torch.zeros_like(tokens)
    labels = tokens.clone()
    labels[:, :4] = -100
    z = torch.randn(2, 4, hkb_cfg.value_dim)
    k = torch.nn.functional.normalize(torch.randn(2, 4, hkb_cfg.key_dim), dim=-1)
    w = torch.ones(2, 4)

    with torch.no_grad():
        teacher_logits = dec(tokens, blocks).logits

    def _loss_fn(z_in, k_in, w_in):
        c = head(z_in, k_in, w_in, gate_floor=0.1)
        out = dec(
            tokens, blocks, controls=c.c_cores, memory_kv=c.mem_kv, memory_valid=c.mem_valid,
            memory_gate=c.mem_gate, memory_mode="positioned",
        )
        return context_distillation_loss(out.logits, teacher_logits, labels, topk=16)

    z_adapted = fo_maml_inner_adapt_z(z, k, w, _loss_fn, steps=2, inner_lr=0.1)
    assert z_adapted.shape == z.shape
    assert not torch.allclose(z_adapted, z)

    abl = evaluate_reader_and_shuffle_ablations(
        dec, head, tokens=tokens, blocks=blocks, labels=labels, z_slots=z, k_slots=k, weights=w,
    )
    assert set(abl.keys()) >= {"base_nll", "joint_nll", "shuffled_nll", "kv_only_nll", "adapter_only_nll"}


def test_tier15_latent_ttt_accepts_gain_and_rejects_regression():
    examples = [{"id": i, "group": f"g{i % 4}"} for i in range(12)]
    sup, qry = split_support_and_query(examples, query_fraction=0.25)
    assert len(sup) > 0 and len(qry) > 0
    assert not ({x["group"] for x in sup} & {x["group"] for x in qry})

    target = torch.ones(1, 4, 16) * 2.0
    z0 = torch.zeros(1, 4, 16)
    k0 = torch.ones(1, 4, 8)
    w0 = torch.ones(1, 4)

    res_good = refine_site_latents(
        z0, k0, w0,
        support_loss_fn=lambda z, _k, _w: (z - target).pow(2).mean(),
        query_loss_fn=lambda z, _k, _w: (z - target).pow(2).mean(),
        cfg=LatentTTTConfig(steps=20, lr=0.2, prox_lambda=0.01, eval_every=5),
    )
    assert res_good.accepted is True
    assert res_good.best_query_loss < res_good.tier1_query_loss

    # Overfitting adversarial case: query loss moves opposite to support loss -> must reject & return z0
    res_bad = refine_site_latents(
        z0, k0, w0,
        support_loss_fn=lambda z, _k, _w: (z - target).pow(2).mean(),
        query_loss_fn=lambda z, _k, _w: (z + target).pow(2).mean(),
        cfg=LatentTTTConfig(steps=12, lr=0.2, prox_lambda=0.01, eval_every=4),
    )
    assert res_bad.accepted is False
    assert torch.allclose(res_bad.z_best, z0)


def test_step_verifier_and_branch_ranking():
    torch.manual_seed(3)
    head = StepVerifierHead(hidden_dim=16, mlp_dim=32)
    # Directional signal (not a uniform DC shift, which LayerNorm cancels)
    direction = torch.cat([torch.ones(8), -torch.ones(8)])
    pos = torch.randn(80, 16) + 1.5 * direction
    neg = torch.randn(80, 16) - 1.5 * direction
    x = torch.cat([pos, neg], dim=0)
    y = torch.cat([torch.ones(80), torch.zeros(80)], dim=0)
    metrics = train_step_verifier(
        head, train_hidden=x, train_success=y, val_hidden=x, val_success=y,
        cfg=VerifierTrainConfig(epochs=5, batch_size=32, lr=5e-3),
    )
    assert metrics["val_auroc"] > 0.95
    assert compute_auroc(torch.tensor([0.1, 0.4, 0.9]), torch.tensor([0.0, 0.0, 1.0])) == pytest.approx(1.0)

    ranked = rank_branches([
        BranchCandidate("b-fail-high-v", ensures_ok=False, type_ok=True, tool_exit_ok=True, verifier_score=0.99),
        BranchCandidate("b-pass-low-v", ensures_ok=True, type_ok=True, tool_exit_ok=True, verifier_score=0.40),
        BranchCandidate("b-pass-high-v", ensures_ok=True, type_ok=True, tool_exit_ok=True, verifier_score=0.85),
    ])
    assert [b.branch_id for b in ranked] == ["b-pass-high-v", "b-pass-low-v", "b-fail-high-v"]


def test_decision_compile_calibration_and_margin_loss():
    torch.manual_seed(9)
    # Deliberately overconfident logits (scale 5x) with 25% errors -> T > 1 should strictly reduce NLL & ECE
    logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],
    ])
    targets = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2])
    cal = fit_temperature(logits, targets)
    assert cal.temperature > 1.2
    assert cal.nll_after < cal.nll_before
    assert cal.ece_after < cal.ece_before

    logits_grad = logits.clone().requires_grad_(True)
    loss = decision_margin_loss(logits_grad, targets)
    loss.backward()
    assert logits_grad.grad is not None

    # Asymmetric risk check: top-1 is "allow" (0.72), but "block" has prob 0.28 >= risk 0.25 -> must NOT settle
    settled, label, _ = should_settle_fast(
        [0.72, 0.28], ["allow", "block"], fast_threshold=0.70, risk_thresholds={"block": 0.25}
    )
    assert settled is False and label == "allow"


def test_call_level_engine_extraction_and_held_exclusion(tmp_path: Path):
    episodes = [
        {
            "task_id": "held-1",
            "family": "triage",
            "calls": [{"site_id": "route.nl", "header": "Pick label", "inputs": {"t": "a"}, "target_output": "spam"}],
        },
        {
            "task_id": "train-1",
            "family": "triage",
            "calls": [
                {
                    "site_id": "route.nl",
                    "readout": "decision",
                    "header": "Pick label",
                    "inputs": {"t": "invoice error"},
                    "teacher_samples": ["billing", "billing", "technical"],
                    "teacher_reasoning": "Let me think...",
                    "decision_labels": ["billing", "technical", "spam"],
                },
                # Duplicate of same (site_id, header, input) must be dropped
                {
                    "site_id": "route.nl",
                    "readout": "decision",
                    "header": "Pick label",
                    "inputs": {"t": "invoice error"},
                    "teacher_samples": ["billing", "billing", "billing"],
                },
                # Low agreement must be dropped
                {
                    "site_id": "judge:holds",
                    "combinator_role": "holds",
                    "header": "Check summary",
                    "inputs": {"s": "x"},
                    "teacher_samples": [True, False, None],
                },
            ],
        },
    ]
    rows, stats = call_level_engine.extract_call_sites(episodes, excluded_task_ids={"held-1"}, min_agreement=0.66)
    assert stats["held_excluded"] == 1
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_low_agreement"] == 1
    assert stats["admitted"] == 1
    assert stats["admitted_judges"] == 1
    assert rows[0].site_kind == "decision"
    assert rows[0].target_output == "billing"

    out_file = tmp_path / "call_sites.jsonl"
    n = call_level_engine.write_call_sites_jsonl(rows, out_file)
    assert n == 1
    parsed = json.loads(out_file.read_text(encoding="utf-8").strip())
    assert parsed["teacher_reasoning_weight"] == 0.0


def test_warmstart_extractor_and_prep_bundle(tmp_path: Path):
    import run_compiler_prep
    sys.path.insert(0, str(hkb_dependency.checkout() / "scripts"))
    import extract_lfm3_warmstart

    # Synthetic 000000002 warm-start extraction check
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({
        "model": {
            "writer.encoder.weight": torch.randn(4, 4),
            "writer.value_head.weight": torch.randn(8, 4),
            "query.encoder.weight": torch.randn(4, 4),
            "decoder.layers.0.weight": torch.randn(4, 4),
            "kv_reader.net.0.weight": torch.randn(4, 4),
        }
    }, ckpt_path)
    manifest = extract_lfm3_warmstart.extract_hq_warmstart(ckpt_path, tmp_path / "warm")
    assert manifest["writer_tensors"] == 2
    assert manifest["query_tensors"] == 1
    assert "value_head.weight" in manifest["refit_fresh_lr_keys"]
    assert "decoder" in manifest["dropped_decoder_prefixes"]

    # Recipe prep bundle generation check
    prep = run_compiler_prep.build_prep_bundle(tmp_path / "prep")
    assert len(prep["recipes_written"]) == 4
    assert (tmp_path / "prep/t2-unified-compiler.json").exists()

