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
    # Sketch 0 is the only replacement in this first two-position replay group.
    torch.testing.assert_close(grads[0][:, 1], torch.zeros_like(grads[0][:, 1]))
    assert grads[0][:, 0].abs().sum() > 0


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


@pytest.mark.parametrize('length', [1, 2, 5, 6, 7, 11, 12, 13, 100])
def test_document_boundaries_are_real_and_every_target_is_supervised_once(length):
    from natlang_neuralese.train.text_warmup import document_windows
    text=list(range(10,10+length))
    windows=document_windows(text,open_id=1000,close_id=1001,tokens=9,prefix_tokens=3)
    assert windows[0]['ids'][0]==1000
    assert windows[0]['prefix']==1
    assert windows[-1]['ids'][-1]==1001
    assert sum(w['ids'].count(1001) for w in windows)==1
    targets=[token for w in windows for token in w['ids'][w['prefix']:]]
    assert targets==text+[1001]
    assert all(len(w['ids'])<=9 for w in windows)


def test_single_close_target_tail_is_supported():
    backbone,heads=tiny_student()
    prefix=torch.tensor([[1,4,7]])
    span=torch.tensor([[backbone.controls.close_id]])
    result=scheduled_completion(backbone,heads,prefix,span,fraction=1.)
    assert result['top'].shape[:2]==(1,1)
    loss=relative_mse(result['sketches'],backbone.embed(span))
    loss.backward()
    assert heads.feedback.correction.weight.grad is not None


def test_branch_checkpoint_preserves_values_and_parameter_gradients():
    backbone,heads=tiny_student()
    prefix=torch.tensor([[1,4,7],[2,3,8]])
    span=torch.tensor([[9,3,5,8],[4,11,12,7]])
    parameter=next(p for n,p in backbone.hf.named_parameters() if 'layers.0.' in n and p.ndim==2)
    parameter.requires_grad_(True)
    results=[]
    for checkpointed in [False,True]:
        backbone.checkpoint_layers=checkpointed
        backbone.hf.zero_grad();heads.zero_grad()
        out=scheduled_completion(backbone,heads,prefix,span,fraction=1.,group_size=2)
        loss=out['top'].square().mean()+relative_mse(out['sketches'],backbone.embed(span))
        loss.backward()
        results.append((out['top'].detach(),parameter.grad.clone(),heads.feedback.correction.weight.grad.clone()))
    for plain,checkpointed in zip(*results):
        torch.testing.assert_close(plain,checkpointed,atol=2e-5,rtol=2e-5)
