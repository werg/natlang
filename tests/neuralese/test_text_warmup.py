"""Teacherless text warm-up keeps gold-token prediction causally aligned."""
import pytest
import torch

from natlang_neuralese.train.text_warmup import (
    chunked_readout,
    balanced_position_weights,
    qualification,
    relative_mse,
    sequence_completions,
    projection_losses,
    alignment_region_metrics,
)
from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone


class TinyReadout(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight=torch.nn.Parameter(weight.clone())
        self.bias=torch.nn.Parameter(bias.clone())

    def logits(self, states):
        return states @ self.weight.t() + self.bias


def test_final_position_alignment_is_not_hidden_by_easy_prompt_tokens():
    gold=torch.tensor([[1,2]])
    target=torch.ones(1,2,3)
    metrics=alignment_region_metrics(torch.tensor([[1.,3.]]),torch.tensor([[1,0]]),
        torch.tensor([[.2,.2]]),gold,target*2,target*3,target,target,gold)
    assert metrics['tokens']==2
    assert metrics['ce_delta']==pytest.approx(1.8)
    assert metrics['relative_mse']==pytest.approx(1.)
    assert metrics['sketch_mse']==pytest.approx(4.)
    assert metrics['text_argmax_agreement']==pytest.approx(.5)
    easy={**metrics,'ce_delta':0.,'relative_mse':0.,'embedding_mse_delta':0.,'text_argmax_agreement':1.}
    assert qualification({'strata':{'full-document':easy}})
    assert not qualification({'strata':{'full-document':easy,'last256':metrics}})


def test_chunked_readout_matches_full_ce_metrics_and_gradients_uneven_chunks():
    torch.manual_seed(72)
    batch,time,width,vocab=2,7,5,11
    initial=torch.randn(vocab,width)
    bias=torch.randn(vocab)
    states=torch.randn(batch,time,width)
    targets=torch.randint(vocab,(batch,time))
    chunked_model=TinyReadout(initial,bias)
    full_model=TinyReadout(initial,bias)
    chunk_states=states.clone().requires_grad_(True)
    full_states=states.clone().requires_grad_(True)

    actual_ce,training_ce,actual_pred,actual_close,actual_positions=chunked_readout(
        chunked_model,chunk_states,targets,close_id=4,chunk_size=3)
    full_logits=full_model.logits(full_states).float()
    expected_ce=torch.nn.functional.cross_entropy(
        full_logits.reshape(-1,vocab),targets.reshape(-1))
    expected_pred=full_logits.argmax(-1)
    expected_close=(full_logits[...,4]-torch.logsumexp(full_logits,-1)).exp()

    # Chunk reduction changes FP32 summation order by under 1e-6 here.
    torch.testing.assert_close(actual_ce,expected_ce,atol=2e-6,rtol=2e-7)
    torch.testing.assert_close(actual_pred,expected_pred,atol=0,rtol=0)
    torch.testing.assert_close(actual_close,expected_close,atol=1e-7,rtol=1e-7)
    expected_positions=torch.nn.functional.cross_entropy(
        full_logits.reshape(-1,vocab),targets.reshape(-1),reduction='none').reshape_as(targets)
    torch.testing.assert_close(actual_positions,expected_positions.detach(),atol=1e-6,rtol=1e-6)
    assert not actual_positions.requires_grad
    torch.testing.assert_close(training_ce,actual_ce)
    actual_ce.backward()
    expected_ce.backward()
    torch.testing.assert_close(chunk_states.grad,full_states.grad,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(chunked_model.weight.grad,full_model.weight.grad,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(chunked_model.bias.grad,full_model.bias.grad,atol=2e-7,rtol=2e-6)


def scheduled_completion(backbone,heads,prefix,span,*,fraction=1.,group_size=16):
    # Exercise the final sequence pass with the historical test geometries.
    return list(sequence_completions(backbone,heads,prefix,span,passes=2,
                                     fraction=fraction,group_size=group_size))[-1]


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
    assert active == [1]
    # Sketch 0 is the only replacement in this first two-position replay group.
    torch.testing.assert_close(grads[1][:, 1:], torch.zeros_like(grads[1][:, 1:]))
    assert grads[1][:, 0].abs().sum() > 0


def test_relative_mse_is_per_vector_and_targets_are_detached():
    pred = torch.tensor([[[2., 0.], [0., 3.]]], requires_grad=True)
    target = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    value = relative_mse(pred, target)
    assert value.item() == pytest.approx(2.5)
    grad, = torch.autograd.grad(value, [target], allow_unused=True)
    assert grad is None or not grad.any()


@pytest.mark.parametrize('passes',[1,2,3])
def test_sequence_passes_match_causal_shifted_primal(passes):
    backbone,heads=tiny_student()
    prefix=torch.tensor([[1,4,7],[2,3,8]])
    span=torch.tensor([[9,3,5,8],[4,11,12,7]])
    with torch.no_grad():
        actual=list(sequence_completions(backbone,heads,prefix,span,passes=passes,group_size=2))
        body=backbone.embed(span[:,:-1])
        for depth in range(passes):
            ordinary=backbone.forward_embeds(torch.cat([backbone.embed(prefix),body],1),
                                            cutoff=heads.cutoff,logits=False)
            start=prefix.shape[1]-1
            expected=ordinary['h_final'][:,start:]
            guesses=heads.feedback(ordinary['h_cut'][:,start:])
            torch.testing.assert_close(actual[depth]['top'],expected,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(actual[depth]['sketches'],guesses,atol=2e-5,rtol=2e-5)
            assert actual[depth]['pass_index']==depth
            # The prediction for body token j goes at INPUT j on the next pass.
            body=guesses[:,:-1]


def test_sequence_depth_and_position_sketch_credit_are_one_consumer_only():
    backbone,heads=tiny_student()
    prefix=torch.tensor([[1,4,7]])
    span=torch.tensor([[9,3,5,8]])
    calls=[]
    hook=heads.feedback.register_forward_hook(lambda _m,_i,out:calls.append(out))
    try:
        outputs=list(sequence_completions(backbone,heads,prefix,span,passes=3,group_size=2))
    finally:
        hook.remove()
    gradients=torch.autograd.grad(outputs[2]['top'][:,2].square().sum(),calls,
                                  allow_unused=True,retain_graph=True)
    # gold target F; pass1 producer F/aux F; pass2 producer F/aux F.
    assert len(calls)==5
    active=[i for i,g in enumerate(gradients) if g is not None and g.abs().sum()>0]
    assert active==[3]
    assert gradients[3][:,1].abs().sum()>0
    torch.testing.assert_close(gradients[3][:,0],torch.zeros_like(gradients[3][:,0]))
    torch.testing.assert_close(gradients[3][:,2:],torch.zeros_like(gradients[3][:,2:]))


def test_sequence_passes_can_backpropagate_and_release_each_stage():
    backbone,heads=tiny_student()
    layer=next(p for n,p in backbone.hf.named_parameters() if 'layers.0.' in n and p.ndim==2)
    layer.requires_grad_(True)
    prefix=torch.tensor([[1,4,7]])
    span=torch.tensor([[9,3,5,8]])
    for output in sequence_completions(backbone,heads,prefix,span,passes=3,group_size=2):
        (output['top'].square().mean()+relative_mse(output['sketches'],backbone.embed(span))).backward()
    assert layer.grad is not None and layer.grad.abs().sum()>0
    assert heads.feedback.correction.weight.grad.abs().sum()>0


def test_projection_bootstrap_trains_both_maps_with_frozen_transformer():
    backbone,heads=tiny_student()
    prefix=torch.tensor([[1,4,7]])
    span=torch.tensor([[9,3,5,8]])
    first=next(sequence_completions(backbone,heads,prefix,span,passes=1))
    full,shallow=projection_losses(heads,first['top'],first['sketches'],backbone.embed(span))
    (full+shallow).backward()
    assert heads.feedback.correction.weight.grad.abs().sum()>0
    assert heads.content.proj.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in backbone.hf.parameters())


def test_main_saves_both_projection_updates_then_resumes_sequence_schedule(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup
    def load(*_args):
        backbone,heads=tiny_student()
        return SimpleNamespace(backbone=backbone,heads=heads,tokenizer=None,
                               _tokens=lambda _text:[9,3,5,8]),None
    monkeypatch.setattr(text_warmup,'load_initial',load)
    heads_path=tmp_path/'heads.pt';torch.save({},heads_path)
    records=tmp_path/'records.jsonl';records.write_text('')
    text=tmp_path/'text.jsonl'
    text.write_text('\n'.join(json.dumps({'text':s,'split':split,'source_groups':[s]})
                              for s,split in [('train','train'),('held','test')])+'\n')
    args=['--heads',str(heads_path),'--records',str(records),'--text-data',str(text),
          '--out',str(tmp_path/'run'),'--device','cpu','--steps','4','--tokens','8',
          '--prefix-tokens','2','--batch','1','--eval-batch','1','--held-documents','1',
          '--eval-every','1','--checkpoint-every','1','--optimizer','adamw',
          '--backbone-training','full','--projection-patience','1','--projection-min-evals','2',
          '--projection-min-improvement','1','--backbone-ramp-evals','1','--pass-ramp-evals','1']
    text_warmup.main(args)
    rows=list(map(json.loads,(tmp_path/'run'/'train.jsonl').read_text().splitlines()))
    assert rows[0]['phase']=='projection_only'
    assert rows[0]['backbone_gradient_norm']==0
    assert rows[0]['updates']['sketch'] and rows[0]['updates']['full_projection']
    assert [r['schedule']['sequence_passes'] for r in rows]==[1,1,2,3]
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert saved['schedule']['adaptation_started_eval']==2
    # Exact resume does not repeat evaluations, reset phase, or lose optimizer state.
    text_warmup.main(args)
    resumed=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert resumed['step']==saved['step']==4
    assert resumed['schedule']==saved['schedule']
    handoff=list(args)
    handoff[handoff.index('--out')+1]=str(tmp_path/'handoff')
    handoff[handoff.index('--backbone-training')+1]='auto'
    handoff+=['--continue-from',str(tmp_path/'run'/'checkpoint.pt')]
    text_warmup.main(handoff)
    continued=torch.load(tmp_path/'handoff'/'checkpoint.pt',weights_only=False)
    assert continued['schedule']==saved['schedule']
    assert continued['step']==saved['step']
    incompatible=list(handoff)
    incompatible[incompatible.index('--out')+1]=str(tmp_path/'incompatible')
    incompatible[incompatible.index('--backbone-training')+1]='adapters'
    with pytest.raises(ValueError,match='resolved backbone parameter policy differs'):
        text_warmup.main(incompatible)


def test_main_scores_final_positions_with_full_history_without_training_regions(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup
    def load(*_args):
        backbone,heads=tiny_student()
        return SimpleNamespace(backbone=backbone,heads=heads,tokenizer=None,
                               _tokens=lambda _text:[9,3,5,8]*75),None
    monkeypatch.setattr(text_warmup,'load_initial',load)
    heads_path=tmp_path/'heads.pt';torch.save({},heads_path)
    records=tmp_path/'records.jsonl';records.write_text('')
    text=tmp_path/'text.jsonl'
    text.write_text('\n'.join(json.dumps({'text':s,'split':split,'source_groups':[s]})
        for s,split in [('train','train'),('held','test')])+'\n')
    out=tmp_path/'run'
    text_warmup.main(['--heads',str(heads_path),'--records',str(records),'--text-data',str(text),
        '--out',str(out),'--device','cpu','--steps','1','--tokens','512','--prefix-tokens','2',
        '--batch','1','--eval-batch','1','--held-documents','1','--eval-every','1',
        '--checkpoint-every','1','--optimizer','adamw','--backbone-training','full'])
    baseline=json.loads((out/'baseline.json').read_text())
    for index in range(3):
        key=f'pass-{index}-length-long-start'
        assert baseline['strata'][key]['tokens']==301  # all text plus its real close marker
        assert baseline['strata'][key+'-last256']['tokens']==256
    assert baseline['boundary_supervision']['close_targets']==1
    assert baseline['projection_held_errors']['shallow']==pytest.approx(
        baseline['strata']['pass-0-length-long-start']['sketch_mse'])
    trained=json.loads((out/'train.jsonl').read_text().splitlines()[0])
    assert all('regions' not in row for row in trained['pass_metrics'])


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


def test_evaluation_batches_preserve_strata_and_every_window():
    from natlang_neuralese.train.text_warmup import evaluation_batches
    windows=[{'ids':list(range(length)), 'prefix':prefix, 'offset':offset, 'index':index}
             for index,(length,prefix,offset) in enumerate([(9,1,0),(9,1,0),(9,1,0),(9,3,6),(9,3,12),(4,3,18)])]
    batches=list(evaluation_batches(windows,2))
    assert sorted(w['index'] for batch in batches for w in batch)==list(range(6))
    assert max(map(len,batches))==2
    for batch in batches:
        assert len({(len(w['ids']),w['prefix'],w['offset']==0) for w in batch})==1


def test_full_sketch_fraction_still_uses_gold_text_history():
    backbone,heads=tiny_student()
    prefix=torch.tensor([[backbone.controls.open_id]])
    first=torch.tensor([[9,3,5,8]])
    changed=torch.tensor([[12,11,5,8]])
    a=scheduled_completion(backbone,heads,prefix,first,fraction=1.,group_size=2)
    b=scheduled_completion(backbone,heads,prefix,changed,fraction=1.,group_size=2)
    # First decision shares the opening; later decisions must see the actual
    # distinct document. Unconditional rollout ignored both documents here.
    torch.testing.assert_close(a['top'][:,:2],b['top'][:,:2],atol=0,rtol=0)
    assert not torch.equal(a['top'][:,2:],b['top'][:,2:])


def test_teacher_forced_history_is_identical_in_training_and_evaluation():
    backbone,heads=tiny_student()
    prefix=torch.tensor([[1,4,7]])
    span=torch.tensor([[9,3,5,8]])
    trained=scheduled_completion(backbone,heads,prefix,span,fraction=1.,group_size=2)
    with torch.no_grad():
        evaluated=scheduled_completion(backbone,heads,prefix,span,fraction=1.,group_size=2)
    for key in ['top','sketches']:
        torch.testing.assert_close(trained[key],evaluated[key],atol=2e-5,rtol=2e-5)


def test_balanced_suffix_weights_and_exact_weighted_readout_gradients():
    torch.manual_seed(84)
    targets=torch.randint(9,(3,7))
    weights=balanced_position_weights(targets,[5,None,10])
    torch.testing.assert_close(weights.mean(1),torch.ones(3))
    torch.testing.assert_close(weights[0,:5],torch.full((5,),.5))
    torch.testing.assert_close(weights[1:],torch.ones(2,7))
    model=TinyReadout(torch.randn(9,4),torch.randn(9))
    other=TinyReadout(model.weight.detach(),model.bias.detach())
    states=torch.randn(3,7,4,requires_grad=True)
    full=states.detach().clone().requires_grad_(True)
    ce,weighted,*_=chunked_readout(model,states,targets,2,chunk_size=3,position_weights=weights)
    losses=torch.nn.functional.cross_entropy(other.logits(full).reshape(-1,9),targets.reshape(-1),reduction='none').reshape_as(targets)
    torch.testing.assert_close(ce,losses.mean())
    torch.testing.assert_close(weighted,(losses*weights).mean())
    weighted.backward();(losses*weights).mean().backward()
    torch.testing.assert_close(states.grad,full.grad)
    torch.testing.assert_close(model.weight.grad,other.weight.grad)
    torch.testing.assert_close(model.bias.grad,other.bias.grad)

def test_suffix_coordinates_survive_real_document_windowing():
    from natlang_neuralese.train.text_warmup import document_windows
    windows=document_windows(range(18),open_id=40,close_id=41,tokens=10,prefix_tokens=3,supervised_suffix_start=12)
    marked=[]
    for window in windows:
        span=window['ids'][window['prefix']:]
        marked.extend(span[window['supervised_suffix_start']:])
    assert marked==list(range(12,18))+[41]


def test_best_alignment_selection_cannot_hide_tail_drift_behind_prompt_volume():
    from natlang_neuralese.train.text_warmup import alignment_selection_score
    def region(gap,tokens):
        return {'tokens':tokens,'ce_delta':gap,'embedding_mse_delta':0.,'text_argmax_agreement':1.}
    bad={'strata':{'prompt':region(.001,100000),'tail':region(2.,10)}}
    better={'strata':{'prompt':region(.04,100000),'tail':region(.2,10)}}
    assert alignment_selection_score(better)<alignment_selection_score(bad)
    qualified={'qualified':True,'strata':{'tail':region(.08,10)}}
    assert alignment_selection_score(qualified)<alignment_selection_score(better)
    assert alignment_selection_score({'strata':{}})[1]==float('inf')
    assert alignment_selection_score({'strata':{'exact':region(0.,10)}},max_ce_delta=0,min_agreement=1)==(1,0.)

def test_best_checkpoint_hard_links_keep_full_state_after_latest_replacement(tmp_path):
    import hashlib,json
    from natlang_neuralese.train.text_warmup import retain_best_checkpoint
    from natlang_neuralese.train.trajectory_state import atomic_checkpoint
    state={'step':7,'student_parameters':{'weight':torch.ones(3)},'heads':{'projection':torch.ones(2)},
           'optimizer':{'momentum':torch.full((3,),2.)},'python_rng':(1,2,3),'schedule':{'phase':'adaptation'}}
    atomic_checkpoint(tmp_path/'checkpoint.pt',state)
    atomic_checkpoint(tmp_path/'heads.pt',{'heads':state['heads']})
    retain_best_checkpoint(tmp_path,{'step':7,'qualified':False})
    assert (tmp_path/'checkpoint.pt').stat().st_ino==(tmp_path/'best-checkpoint.pt').stat().st_ino
    atomic_checkpoint(tmp_path/'checkpoint.pt',{'step':8})
    best=torch.load(tmp_path/'best-checkpoint.pt',weights_only=False)
    assert best['step']==7 and best['schedule']==state['schedule']
    torch.testing.assert_close(best['optimizer']['momentum'],state['optimizer']['momentum'])
    receipt=json.loads((tmp_path/'best-checkpoint.json').read_text())
    assert all(hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==info['sha256'] for name,info in receipt['files'].items())
