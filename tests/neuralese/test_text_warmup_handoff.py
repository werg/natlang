"""Continuation semantics for the teacherless text foundation."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
from natlang_neuralese.train.text_warmup import (
    _apply_requested_sketch_cutoff,
    same_alignment_data,
    same_foundation_context,
    same_resume_identity,
    text_supervision_policy,
    warmup_display_labels,
)


def _identity():
    options = {
        "cutoff": 2, "tokens": 256, "prefix_tokens": 32, "group_size": 8,
        "embedding_weight": 1.0, "sketch_weight": 1.0, "text_weight": 0.25,
        "projection_patience": 3, "projection_min_evals": 2,
        "projection_min_improvement": 0.01, "backbone_ramp_evals": 4,
        "pass_ramp_evals": 2,
        "mask_system_prompt": True, "held_documents": 16,
        "rollout_passes": 0, "rollout_start_passes": 4,
        "records": "records.jsonl", "pieces": "pieces.jsonl", "text_data": "text.jsonl",
    }
    return {
        "options": options,
        "inputs": {"records.jsonl": "r", "pieces.jsonl": "p", "text.jsonl": "t"},
        "target": "E(gold next token), fixed raw input table; no teacher; full-stack next-token CE",
        "text_history": "gold seed; repeated shared shallow sequence passes with aligned predictions",
        "sketch_gradient": "local_stage",
        "sketch_target_backbone_scale": 0.05,
        "supervision_policy": {
            "all_positions_fraction": 0.5,
            "observed_suffix_fraction": 0.5,
            "objectives": ["full_projection", "sketch_projection", "next_token_ce"],
        },
    }


def _tiny_engine(cutoff=2):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Lfm2Config"):
        pytest.skip("installed transformers does not provide a tiny LFM2 model")
    config = transformers.Lfm2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        block_multiple_of=8, block_auto_adjust_ff_dim=False,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    )
    model = transformers.Lfm2ForCausalLM(config).eval()
    backbone = PortBackbone(model, ControlTokens(62, 63), fast=False)
    heads = PortHeads(backbone, cutoff=cutoff, max_length=8,
                      profile="latent-sketch-v2").eval()
    engine = SimpleNamespace(
        backbone=backbone, heads=heads, max_block=heads.max_length,
        dialect=heads.dialect, foundation={"qualified": True, "runtime_qualified": True},
    )
    return engine


def test_unchanged_foundation_context_preserves_continuation_schedule():
    before = _identity()
    after = _identity()
    assert same_foundation_context(before, after)


def test_cosmetic_map_report_labels_preserve_saved_resume_semantics():
    before = _identity()
    before.update(
        text_history="gold seed; detached causal token-to-Neuralese input map; one parallel consumer pass",
        sketch_gradient="detached_consumer",
        sketch_target_backbone_scale=0.0,
        supervision_policy={
            "all_positions_fraction": 0.5,
            "observed_suffix_fraction": 0.5,
            "unannotated_or_no_suffix_window": "uniform-all-positions",
            "objectives": ["full_projection", "sketch_projection", "next_token_ce"],
            "qualification": "unweighted full-history complete-window and last256 strata",
        },
    )
    after = {**before, "display": warmup_display_labels("map"),
             "supervision_policy": text_supervision_policy("map")}
    assert same_resume_identity(before, after)
    assert same_foundation_context(before, after)
    assert same_alignment_data(before, after)
    assert after["display"]["secondary_head"] == "heads.input_map"
    changed = {**after, "sketch_gradient": "changed-gradient-policy"}
    assert not same_resume_identity(before, changed)


def test_newer_text_corpus_keeps_plateau_and_ramp():
    # Owner: adopt new data at once; the new held set still has to pass the gates consecutively.
    before = _identity()
    after = _identity()
    after["inputs"].update({"text.jsonl": "different corpus"})
    assert same_foundation_context(before, after)


def test_mask_change_remeasures_text_baseline_but_preserves_full_state_schedule():
    before = _identity()
    after = _identity()
    assert same_foundation_context(before, after)
    assert same_alignment_data(before, after)
    after["options"]["mask_system_prompt"] = False
    assert same_foundation_context(before, after)
    assert not same_alignment_data(before, after)


@pytest.mark.parametrize("change", [
    lambda identity: identity["options"].update(cutoff=3),
    lambda identity: identity["supervision_policy"].update(all_positions_fraction=0.75),
    lambda identity: identity.update(text_history="different recurrence history"),
])
def test_changed_depth_or_foundation_semantics_restarts_plateau(change):
    before = _identity()
    after = _identity()
    change(after)
    assert not same_foundation_context(before, after)


def test_requested_cli_cutoff_rebuilds_loaded_sketch_and_preserves_same_shaped_weights():
    torch.manual_seed(27)
    engine = _tiny_engine(cutoff=2)
    previous = engine.heads
    with torch.no_grad():
        previous.feedback.correction.weight.fill_(0.125)
        previous.content.proj.weight.fill_(-0.25)
    old_state = {name: value.detach().clone() for name, value in previous.state_dict().items()}

    result = _apply_requested_sketch_cutoff(engine, 3, "cpu")

    assert result is engine.heads
    assert result is not previous
    assert result.cutoff == 3
    assert engine.max_block == result.max_length
    assert engine.dialect == result.dialect
    for name, expected in old_state.items():
        torch.testing.assert_close(result.state_dict()[name], expected, atol=0, rtol=0)
    assert engine.foundation["qualified"] is False
    assert engine.foundation["runtime_qualified"] is False
    assert engine.foundation["requires_requalification"] is True


def test_requested_cutoff_is_not_rebuilt_when_already_correct():
    engine = _tiny_engine(cutoff=2)
    previous = engine.heads
    assert _apply_requested_sketch_cutoff(engine, 2, "cpu") is previous
