"""Teacherless text warm-up keeps gold-token prediction causally aligned."""
import pytest
import torch

from natlang_neuralese.train.text_warmup import (
    qualification,
    relative_mse,
    scheduled_completion,
)
from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone


def tiny_student():
    try:
        from transformers import Lfm2Config, Lfm2ForCausalLM
    except ImportError:
        pytest.skip("installed transformers does not provide the tiny LFM2 model")

    torch.manual_seed(21)
    config = Lfm2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        block_multiple_of=8, block_auto_adjust_ff_dim=False,
        layer_types=['conv', 'full_attention', 'conv', 'full_attention'],
    )
    model = Lfm2ForCausalLM(config).eval()
    backbone = PortBackbone(model, ControlTokens(62, 63), fast=False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    heads = PortHeads(backbone, cutoff=2, max_length=8, profile='latent-sketch-v2').eval()
    with torch.no_grad():
        heads.feedback.correction.weight.normal_(0, .02)
    return backbone, heads


def test_fraction_zero_matches_ordinary_causal_gold_forward():
    backbone, heads = tiny_student()
    prefix = torch.tensor([[1, 4, 7]])
    span = torch.tensor([[9, 3, 5, 8]])
    result = scheduled_completion(backbone, heads, prefix, span, fraction=0., group_size=2)
    ids = torch.cat([prefix, span[:, :-1]], dim=1)
    expected = backbone.forward_ids(ids, logits=False)['h_final'][:, prefix.shape[1] - 1:]
    assert result['top'].shape == expected.shape
    torch.testing.assert_close(result['top'], expected, atol=2e-5, rtol=2e-5)
    assert result['sketches'].shape == span.shape + (backbone.embedding_weight.shape[1],)
    assert result['replay_delta'] < 2e-5


def test_fraction_one_trains_full_stack_and_feedback_without_embedding_targets_grad():
    backbone, heads = tiny_student()
    layer_weight = next(p for n, p in backbone.hf.named_parameters()
                        if 'layers.0.' in n and p.ndim == 2)
    layer_weight.requires_grad_(True)
    prefix = torch.tensor([[1, 4, 7]])
    span = torch.tensor([[9, 3, 5]])
    result = scheduled_completion(backbone, heads, prefix, span, fraction=1., group_size=2)
    # CE/top-state supervision reaches the real student stack and feedback projection.
    loss = result['top'].square().mean() + relative_mse(result['sketches'], backbone.embed(span))
    loss.backward()
    assert layer_weight.grad is not None and layer_weight.grad.abs().sum() > 0
    assert heads.feedback.correction.weight.grad is not None
    assert heads.feedback.correction.weight.grad.abs().sum() > 0
    # Gold token embeddings are fixed coordinates, never trainable target parameters.
    assert not backbone.embedding_weight.requires_grad


def test_suffix_changes_do_not_change_earlier_completion():
    backbone, heads = tiny_student()
    prefix = torch.tensor([[1, 4, 7]])
    first = torch.tensor([[9, 3, 5, 8]])
    changed_suffix = torch.tensor([[9, 3, 12, 13]])
    out_a = scheduled_completion(backbone, heads, prefix, first, fraction=1., group_size=2)
    out_b = scheduled_completion(backbone, heads, prefix, changed_suffix, fraction=1., group_size=2)
    torch.testing.assert_close(out_a['top'][:, :2], out_b['top'][:, :2], atol=0, rtol=0)
    torch.testing.assert_close(out_a['sketches'][:, :2], out_b['sketches'][:, :2], atol=0, rtol=0)


def test_each_completion_has_only_its_own_replayed_sketch_credit():
    backbone, heads = tiny_student()
    prefix = torch.tensor([[1, 4, 7]])
    span = torch.tensor([[9, 3, 5, 8]])
    sketches = []
    hook = heads.feedback.register_forward_hook(
        lambda _module, _inputs, output: sketches.append(output) if torch.is_grad_enabled() else None
    )
    try:
        # Re-run with hook active so the replay sketch nodes are captured.
        result = scheduled_completion(backbone, heads, prefix, span, fraction=1., group_size=2)
    finally:
        hook.remove()
    # The output sketches form one tensor, and each completion is locally replayed.
    assert result['top'].shape[1] == span.shape[1]
    assert result['sketches'].shape[1] == span.shape[1]
    assert len(sketches) >= 1
    grads = torch.autograd.grad(result['top'][:, 1].square().sum(), sketches,
                                allow_unused=True, retain_graph=True)
    # Replay calls that do not correspond to position 1 carry no path to its completion.
    active = [i for i, grad in enumerate(grads) if grad is not None and grad.abs().sum() > 0]
    assert active == [0]
    # Position 1 is the only replacement in this first two-position replay group.
    torch.testing.assert_close(grads[0][:, 0], torch.zeros_like(grads[0][:, 0]))
    assert grads[0][:, 1].abs().sum() > 0


def test_relative_mse_is_per_vector_and_targets_are_detached():
    pred = torch.tensor([[[2., 0.], [0., 3.]]], requires_grad=True)
    target = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    value = relative_mse(pred, target)
    assert value.item() == pytest.approx(2.5)
    grad, = torch.autograd.grad(value, [target], allow_unused=True)
    assert grad is None or not grad.any()


def test_qualification_requires_every_nonempty_stratum_to_pass():
    passing = {'strata': {
        'short': {'tokens': 12, 'ce_delta': .01, 'embedding_mse_delta': .1, 'text_argmax_agreement': .95},
        'long': {'tokens': 4, 'ce_delta': .05, 'embedding_mse_delta': .2, 'text_argmax_agreement': .91},
    }}
    assert qualification(passing)
    failing = {'strata': {**passing['strata'],
        'held-source': {'tokens': 2, 'ce_delta': .01, 'embedding_mse_delta': .1, 'text_argmax_agreement': .2}}}
    assert not qualification(failing)
    assert not qualification({'strata': {}})
    assert not qualification({'strata': {'empty': {'tokens': 0, 'ce_delta': 0.,
        'embedding_mse_delta': 0., 'text_argmax_agreement': 1.}}})


def _write_admission_fixture(tmp_path, *, warmup=True, qualified=True,
                             bad_report_hash=False, bad_weights=False):
    import json
    from natlang_neuralese.train.output_embedding_projection import sha
    from natlang_neuralese.train.warmup_admission import weights_digest

    artifact = tmp_path / 'heads.pt'
    report_path = tmp_path / 'report.json'
    backbone = {'layers.0.weight': torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    heads_state = {'feedback.weight': torch.ones(2, 2)}
    report = {
        'qualified': qualified,
        'weights_digest': 'wrong' if bad_weights else weights_digest(backbone, heads_state),
        'updates': {'backbone': True, 'sketch': True},
    }
    report_path.write_text(json.dumps(report))
    state = {'backbone_trainables': backbone, 'heads': heads_state}
    if warmup:
        state['warmup'] = {
            'alignment_qualified': True,
            'report_sha256': 'wrong' if bad_report_hash else sha(report_path),
        }
    torch.save(state, artifact)
    return artifact, report_path


def test_admission_rejects_missing_or_false_warmup(tmp_path):
    from natlang_neuralese.train.warmup_admission import require_text_warmup

    absent, _ = _write_admission_fixture(tmp_path / 'absent', warmup=False)
    with pytest.raises(ValueError, match='has not qualified'):
        require_text_warmup(absent)
    false, _ = _write_admission_fixture(tmp_path / 'false', qualified=False)
    with pytest.raises(ValueError, match='report does not qualify'):
        require_text_warmup(false)


def test_admission_binds_report_and_exact_adapted_weights(tmp_path):
    from natlang_neuralese.train.output_embedding_projection import sha
    from natlang_neuralese.train.warmup_admission import require_text_warmup

    report_hash, _ = _write_admission_fixture(tmp_path / 'report-hash', bad_report_hash=True)
    with pytest.raises(ValueError, match='report missing or changed'):
        require_text_warmup(report_hash)
    wrong_weights, _ = _write_admission_fixture(tmp_path / 'weights-hash', bad_weights=True)
    with pytest.raises(ValueError, match='does not qualify these exact'):
        require_text_warmup(wrong_weights)

    valid, _ = _write_admission_fixture(tmp_path / 'valid')
    report = require_text_warmup(valid)
    assert report['qualified'] is True
    runtime = valid.parent / 'runtime.json'
    runtime.write_text(json.dumps({
        'parent_sha256': sha(valid),
        'runtime_qualified': True,
        'output_reference_qualified': True,
    }))
    assert require_text_warmup(valid, runtime)['qualified'] is True
    runtime.write_text(json.dumps({
        'parent_sha256': 'different-weights',
        'runtime_qualified': True,
        'output_reference_qualified': True,
    }))
    with pytest.raises(ValueError, match='lack exact output/transport'):
        require_text_warmup(valid, runtime)


def test_admission_detects_tensor_mutation_after_report(tmp_path):
    from natlang_neuralese.train.warmup_admission import require_text_warmup

    artifact, _ = _write_admission_fixture(tmp_path)
    state = torch.load(artifact, map_location='cpu', weights_only=False)
    state['backbone_trainables']['layers.0.weight'][0, 0] += 1
    torch.save(state, artifact)
    with pytest.raises(ValueError, match='does not qualify these exact'):
        require_text_warmup(artifact)


def test_raw_recurrence_recipe_requires_text_warmup_and_runtime(tmp_path):
    import json
    from pathlib import Path
    from natlang_neuralese.train.recipe import load_recipe

    recipe_path = Path(__file__).resolve().parents[2] / 'training' / 'neuralese' / 'recipes' / 'raw-recurrence-v1.json'
    recipe = json.loads(recipe_path.read_text())
    stages = {stage['id']: stage for stage in recipe['stages']}
    assert 'core_text_warmup' in stages['adapted_runtime']['requires']
    assert {'core_text_warmup', 'adapted_runtime'} <= set(stages['recurrence_warmup']['requires'])
    assert load_recipe(recipe_path)['id'] == 'raw-recurrence-v1'

    recipe['stages'] = [stage for stage in recipe['stages'] if stage['id'] != 'core_text_warmup']
    modified = tmp_path / 'without-text-warmup.json'
    modified.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='dependencies must precede'):
        load_recipe(modified)
