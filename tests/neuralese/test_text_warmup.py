"""Teacherless text warm-up keeps gold-token prediction causally aligned."""
import pytest
import torch

from natlang_neuralese.train.text_warmup import (
    chunked_readout,
    balanced_position_weights,
    materialize_objective_metrics,
    LinearModuleCallCounter,
    qualification,
    relative_mse,
    sequence_completions,
    projection_losses,
    alignment_region_metrics,
)
from natlang_neuralese.train.checkpoint_safety import (
    CheckpointDiskReserve,
    CheckpointReserveError,
    optimizer_state_size_upper_bound,
    warmup_checkpoint_size_upper_bound,
)
from natlang_neuralese.model.heads import PortHeads
from natlang_neuralese.model.lfm2_port import ControlTokens, PortBackbone
from natlang_neuralese.train.optim import PortMuonAdamW


class TinyReadout(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight=torch.nn.Parameter(weight.clone())
        self.bias=torch.nn.Parameter(bias.clone())

    def logits(self, states):
        return states @ self.weight.t() + self.bias


def test_linear_module_call_counter_counts_shapes_grad_mode_and_cleans_up_on_error():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj=torch.nn.Linear(3,2)

        def forward(self, value):
            return self.proj(value)

    backbone=Toy()
    heads=Toy()
    counter=LinearModuleCallCounter(backbone=backbone,heads=heads)
    with pytest.raises(RuntimeError,match='intentional'):
        with counter:
            backbone(torch.ones(4,3))
            with torch.no_grad():
                heads(torch.ones(1,3))
            raise RuntimeError('intentional')

    assert counter.closed and not counter.handles
    assert not backbone.proj._forward_pre_hooks
    assert not heads.proj._forward_pre_hooks
    assert counter.rows()==[
        {'module':'backbone.proj','module_class':'Linear','input_shape':[4,3],
         'grad_enabled':True,'calls':1},
        {'module':'heads.proj','module_class':'Linear','input_shape':[1,3],
         'grad_enabled':False,'calls':1},
    ]
    assert all(not isinstance(value,torch.Tensor)
               for row in counter.rows() for value in row.values())


def test_training_metric_batching_preserves_values_empty_close_and_loss_mean():
    def packet(tokens, close_targets, *, close_probability, close_top1, premature):
        values={key:torch.tensor(float(index + 1),requires_grad=True)
                for index,key in enumerate((
                    'ce','text_ce','ce_delta','relative_mse','sketch_mse',
                    'text_embedding_mse','embedding_mse_delta','text_argmax_agreement',
                    'gold_accuracy','close_targets','close_probability','close_top1',
                    'premature_close_top1','supervised_ce','supervised_embedding_mse',
                    'supervised_sketch_mse'))}
        values.update(close_targets=torch.tensor(float(close_targets)),
                      close_probability=torch.tensor(close_probability),
                      close_top1=torch.tensor(close_top1),
                      premature_close_top1=torch.tensor(premature),
                      tokens=tokens,positions=tokens,pass_index=0)
        return values

    metrics=[packet(4,0,close_probability=float('nan'),close_top1=float('nan'),premature=.25),
             packet(3,3,close_probability=0.,close_top1=0.,premature=float('nan'))]
    losses=[torch.tensor(1.25,requires_grad=True),torch.tensor(2.5,requires_grad=True)]
    result,total=materialize_objective_metrics(metrics,losses,2)

    assert result[0]['close_targets']==0
    assert result[0]['close_probability'] is None
    assert result[0]['close_top1'] is None
    assert result[0]['premature_close_top1']==.25
    assert result[1]['close_targets']==3
    assert result[1]['close_probability']==0.
    assert result[1]['close_top1']==0.
    assert result[1]['premature_close_top1']==0.
    assert result[0]['ce']==1.
    assert result[1]['supervised_sketch_mse']==16.
    expected=0.
    for loss in losses:
        expected+=float(loss.detach())/2
    assert total==expected
    assert all(not isinstance(result[0][key],torch.Tensor)
               for key in ('ce','close_targets','close_probability'))
    for index in (0,1):
        for key in ('ce','text_ce','ce_delta','relative_mse','sketch_mse',
                    'text_embedding_mse','embedding_mse_delta','text_argmax_agreement',
                    'gold_accuracy','supervised_ce','supervised_embedding_mse',
                    'supervised_sketch_mse'):
            assert result[index][key]==float(metrics[index][key].detach())

    evaluated,_=materialize_objective_metrics([metrics[0]])
    assert evaluated[0]['ce']==result[0]['ce']
    assert evaluated[0]['close_targets']==result[0]['close_targets']
    assert evaluated[0]['close_probability'] is None
    assert evaluated[0]['premature_close_top1']==result[0]['premature_close_top1']


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


def test_adaptive_readout_chunks_preserve_ce_metrics_and_gradients():
    generator=torch.Generator().manual_seed(731)
    base_states=torch.randn(2,1025,16,generator=generator)
    targets=torch.randint(0,80,(2,1025),generator=generator)
    weights=torch.rand(2,1025,generator=generator)+.25
    initial_weight=torch.randn(80,16,generator=generator)
    initial_bias=torch.randn(80,generator=generator)
    reference=None
    for chunk_size in (128,256,512):
        backbone=TinyReadout(initial_weight,initial_bias)
        states=base_states.clone().requires_grad_()
        result=chunked_readout(backbone,states,targets,79,
            chunk_size=chunk_size,position_weights=weights)
        result[1].backward()
        observed={
            'ce':result[0].detach(),
            'weighted_ce':result[1].detach(),
            'prediction':result[2],
            'close_probability':result[3],
            'token_losses':result[4],
            'input_gradient':states.grad.detach(),
            'weight_gradient':backbone.weight.grad.detach(),
            'bias_gradient':backbone.bias.grad.detach(),
        }
        if reference is None:
            reference=observed
        else:
            for key in ('ce','weighted_ce','close_probability','token_losses',
                        'input_gradient','weight_gradient','bias_gradient'):
                torch.testing.assert_close(observed[key],reference[key],rtol=2e-6,atol=2e-6)
            assert torch.equal(observed['prediction'],reference['prediction'])


def test_held_probe_selection_is_hash_ordered_fair_and_keeps_bounded_windows():
    import hashlib
    import random
    from natlang_neuralese.train.text_warmup import select_held_document_windows, evaluation_batches

    windows=[]
    # Several documents share each factual tuple; a lexical-prefix selector
    # would spend the whole limit on the first tuple.
    for group_index in range(24):
        group=f'{group_index:02d}-factual-group'
        for document_index in range(3):
            document=hashlib.sha256(f'{group}/{document_index}'.encode()).hexdigest()
            for offset in (0, 10, 20):
                windows.append({'document':document,'groups':[group],'offset':offset,
                                'prefix':3,'ids':[group_index,document_index,offset]})

    selected, metadata=select_held_document_windows(windows,limit=12)
    assert metadata['policy']=='sha256-ordered-complete-factual-group-tuples-round-robin-v1'
    assert len(metadata['selected_documents'])==12
    assert len({tuple(row['source_groups']) for row in metadata['selected_documents']})==12
    assert len({row['selected_group_tuple'][0] for row in metadata['selected_documents']})==12
    # Hash order, rather than the lexical prefix, decides the first group.
    assert metadata['selected_documents'][0]['selected_group_tuple'] != ['00-factual-group']
    assert all(row['window_count']==3 and row['selected_window_offsets']==[0,20]
               for row in metadata['selected_documents'])

    shuffled=windows.copy()
    random.Random(77).shuffle(shuffled)
    shuffled_selected, shuffled_metadata=select_held_document_windows(shuffled,limit=12)
    assert shuffled_metadata==metadata
    assert shuffled_selected==selected

    # Evaluation batching keeps every selected bounded window exactly once.
    batched=[window for batch in evaluation_batches(selected,limit=5) for window in batch]
    assert sorted((w['document'],w['offset']) for w in batched)==sorted((w['document'],w['offset']) for w in selected)
    assert len(batched)==24


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
    # A code change (a fix) resumes in place as a logged handoff; a recipe change still refuses.
    # A code change, an option newer code added and an operational option change are one logged handoff.
    options=dict(resumed['identity']['options']);del options['checkpoint_minutes']
    edited=dict(resumed);edited['identity']={**resumed['identity'],'options':options,
        'code':{**resumed['identity']['code'],'train/text_warmup.py':'0'*64,'train/retired.py':'1'*64}}
    torch.save(edited,tmp_path/'run'/'checkpoint.pt')
    text_warmup.main(args+['--eval-every','2'])
    logged=[json.loads(l) for l in (tmp_path/'run'/'code-handoffs.jsonl').read_text().splitlines()]
    assert logged==[{'event':'code_handoff','step':4,'changed':['train/text_warmup.py'],'added':[],
                     'removed':['train/retired.py'],'options_added':{'checkpoint_minutes':10.0},
                     'options_changed':{'eval_every':[1,2]}}]
    recipe=list(args);recipe+=['--lr','0.5']
    with pytest.raises(ValueError,match='resume identity changed'):
        text_warmup.main(recipe)
    torch.save(resumed,tmp_path/'run'/'checkpoint.pt')
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


def test_evaluation_batches_cap_tokens_per_batch():
    from natlang_neuralese.train.text_warmup import evaluation_batches
    windows=[{'ids':list(range(length)), 'prefix':1, 'offset':0, 'index':index}
             for index,length in enumerate([9]*5+[4]*5+[30])]
    batches=list(evaluation_batches(windows,4,max_tokens=20))
    assert sorted(w['index'] for batch in batches for w in batch)==list(range(11))
    assert {len(b[0]['ids']):max(len(x) for x in batches if len(x[0]['ids'])==len(b[0]['ids'])) for b in batches}=={9:2,4:4,30:1}

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


def _tiny_warmup_run_inputs(tmp_path, monkeypatch, *, steps=3):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup
    engines=[]
    def load(*_args):
        backbone,heads=tiny_student()
        engine=SimpleNamespace(backbone=backbone,heads=heads,tokenizer=None,
                               _tokens=lambda _text:[9,3,5,8])
        engines.append(engine)
        return engine,None
    monkeypatch.setattr(text_warmup,'load_initial',load)
    heads_path=tmp_path/'heads.pt';torch.save({},heads_path)
    records=tmp_path/'records.jsonl';records.write_text('')
    text=tmp_path/'text.jsonl'
    text.write_text('\n'.join(json.dumps({'text':name,'split':split,'source_groups':[name]})
                              for name,split in [('train','train'),('held','test')])+'\n')
    args=['--heads',str(heads_path),'--records',str(records),'--text-data',str(text),
          '--out',str(tmp_path/'run'),'--device','cpu','--steps',str(steps),'--tokens','8',
          '--prefix-tokens','2','--batch','1','--eval-batch','1','--held-documents','1',
          '--eval-every','99','--checkpoint-every','2','--optimizer','adamw',
          '--backbone-training','full']
    return text_warmup,args,engines


def test_held_evaluation_records_matched_projection_and_crisp_history_without_changing_gates(tmp_path,monkeypatch):
    import json
    from natlang_neuralese.train import text_warmup
    module,args,_engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=1)
    module.main(args)
    report=json.loads((tmp_path/'run'/'report.json').read_text())
    diagnostic=report['matched_projected_history']
    assert diagnostic['schema']=='natlang.text-warmup-matched-history/2'
    assert diagnostic['weights_digest']==report['weights_digest']
    assert diagnostic['consumers']==['full_projection','sketch_projection','live_greedy']
    assert diagnostic['held_probe_selection']==report['held_probe_selection']
    assert diagnostic['batch_policy']['diagnostic_forward_passes_per_batch']==3
    assert diagnostic['future_gold_inputs'] is False
    assert diagnostic['changes_qualification_gates'] is False
    assert len(diagnostic['windows'])==1
    window=diagnostic['windows'][0]
    projected=window['scores']['full_projection']
    crisp=window['scores']['live_greedy']
    sketch=window['scores']['sketch_projection']
    assert sketch['whole']['tokens']==window['target_tokens']
    assert 0<=sketch['whole']['argmax_agreement_with_live_greedy']<=1
    assert set(report['matched_projected_history']['weighted_summary'])=={'full_projection','sketch_projection','live_greedy'}
    assert projected['whole']['tokens']==window['target_tokens']
    assert projected['last256']['tokens']==window['target_tokens']
    assert 0<=projected['whole']['argmax_agreement_with_live_greedy']<=1
    assert projected['whole']['read_history_mse_vs_live_greedy']>=0
    assert crisp['whole']['read_history_mse_vs_live_greedy']==0
    assert 'alignment_gate_passed' in report


def test_periodic_full_checkpoints_skip_heads_export_until_final(tmp_path,monkeypatch):
    import json
    from natlang_neuralese.train import text_warmup
    module,args,_engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=3)
    atomic=module.atomic_checkpoint
    checkpoint_steps=[];export_steps=[]
    def record(path,state):
        name=pathlib.Path(path).name
        if name=='checkpoint.pt':checkpoint_steps.append(state['step'])
        if name=='heads.pt':export_steps.append(state['warmup']['step'])
        return atomic(path,state)
    import pathlib
    monkeypatch.setattr(module,'atomic_checkpoint',record)
    module.main(args)
    assert checkpoint_steps==[0,2,3]
    assert export_steps==[0,3]
    plan=json.loads((tmp_path/'run'/'plan.json').read_text())
    policy=plan['receipt']['serving_heads_export_policy']
    assert 'non-best periodic checkpoints do not' in policy['policy']
    assert 'may represent an earlier step' in policy['lag']


def test_resume_releases_parent_mmap_checkpoint_aliases(tmp_path,monkeypatch):
    import gc,weakref
    from natlang_neuralese.train import text_warmup
    module,args,_engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=2)
    module.main(args)
    original_load=torch.load;loaded=[]
    class WeakableCheckpoint(dict):
        pass
    def track_mmap(*load_args,**load_kwargs):
        value=original_load(*load_args,**load_kwargs)
        if load_kwargs.get('mmap') and pathlib.Path(load_args[0]).name=='checkpoint.pt':
            value=WeakableCheckpoint(value)
            loaded.append(weakref.ref(value))
        return value
    import pathlib
    monkeypatch.setattr(torch,'load',track_mmap)
    module.main(args)
    gc.collect()
    assert loaded
    assert all(reference() is None for reference in loaded)


def _assert_nested_state_equal(actual, expected):
    if isinstance(expected,torch.Tensor):
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:_assert_nested_state_equal(actual[key],expected[key])
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,e in zip(actual,expected):_assert_nested_state_equal(a,e)
    else:
        assert actual==expected


def test_training_rng_snapshot_replays_python_and_torch_random_streams():
    import random
    from natlang_neuralese.train.text_warmup import capture_training_rng_state, restore_training_rng_state
    random.seed(101);torch.manual_seed(202)
    state=capture_training_rng_state('cpu')
    expected=(random.randrange(100000),torch.rand(4))
    random.randrange(100000);torch.rand(9)
    restore_training_rng_state(state,'cpu')
    actual=(random.randrange(100000),torch.rand(4))
    assert actual[0]==expected[0]
    torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
    assert state['cuda_rng']==[]


def test_pre_optimizer_failure_checkpoints_last_commit_and_replays_attempt_rng(tmp_path,monkeypatch):
    import copy,hashlib,random
    from natlang_neuralese.train import text_warmup
    module,args,engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=3)
    original_clip=module.clip_finite_gradients;clip_count=0
    def fail_third_clip(parameters):
        nonlocal clip_count
        clip_count+=1
        if clip_count==3:
            assert any(p.grad is not None for p in parameters)
            raise RuntimeError('injected failure after backward, before optimizer step')
        return original_clip(parameters)
    monkeypatch.setattr(module,'clip_finite_gradients',fail_third_clip)
    rng_snapshots=[];original_capture=module.capture_training_rng_state
    def capture(device):
        state=original_capture(device);rng_snapshots.append(copy.deepcopy(state));return state
    monkeypatch.setattr(module,'capture_training_rng_state',capture)
    sampled=[];original_randrange=random.randrange
    def capture_randrange(*values):
        value=original_randrange(*values);sampled.append(value);return value
    monkeypatch.setattr(random,'randrange',capture_randrange)

    atomic=module.atomic_checkpoint;committed_state={}
    def capture_atomic(path,state):
        if pathlib.Path(path).name=='checkpoint.pt' and state.get('step')==2 and 'step2' not in committed_state:
            committed_state['step2']=copy.deepcopy(state)
        return atomic(path,state)
    import pathlib
    monkeypatch.setattr(module,'atomic_checkpoint',capture_atomic)
    with pytest.raises(RuntimeError,match='injected failure after backward'):
        module.main(args)
    assert 'step2' in committed_state
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    expected=committed_state['step2']
    assert saved['step']==2
    _assert_nested_state_equal(saved['student_parameters'],expected['student_parameters'])
    _assert_nested_state_equal(saved['heads'],expected['heads'])
    _assert_nested_state_equal(saved['optimizer'],expected['optimizer'])
    _assert_nested_state_equal(saved['schedule'],expected['schedule'])
    assert saved['python_rng']==rng_snapshots[-1]['python_rng']
    torch.testing.assert_close(saved['torch_rng'],rng_snapshots[-1]['torch_rng'],atol=0,rtol=0)
    recovery=saved['emergency_recovery']
    assert recovery['schema']=='natlang.text-warmup-emergency-recovery/1'
    assert recovery['safe_to_resume'] is True and recovery['failure_stage']=='before_optimizer_step'
    assert recovery['failed_attempt_step']==3 and recovery['last_committed_step']==2
    assert recovery['optimizer_step_started'] is False and recovery['partial_gradients_cleared'] is True
    assert recovery['pre_attempt_rng_saved_for_replay'] is True and recovery['schedule_state_is_last_committed'] is True
    assert recovery['error_type']=='RuntimeError' and recovery['error']=='injected failure after backward, before optimizer step'
    # The last held probe predates the committed weights: no stale result is
    # attached as a qualification of this emergency checkpoint.
    assert saved['qualification'] is None
    assert recovery['phase']=='projection_only'
    assert all(p.grad is None for p in engines[-1].backbone.hf.parameters())
    assert all(p.grad is None for p in engines[-1].heads.parameters())
    failed_batch_draw=sampled[-1]
    assert len(sampled)==3 and len(rng_snapshots)>=5

    # Exact resume starts from the captured pre-attempt stream and repeats its sample.
    monkeypatch.setattr(module,'clip_finite_gradients',original_clip)
    module.main(args)
    assert sampled[-1]==failed_batch_draw
    resumed=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert resumed['step']==3 and 'emergency_recovery' not in resumed


def test_optimizer_step_exception_never_writes_a_safe_emergency_checkpoint(tmp_path,monkeypatch):
    import hashlib
    import pathlib
    import torch.optim
    from natlang_neuralese.train import text_warmup
    module,args,engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=3)
    original_adamw=torch.optim.AdamW
    class PartiallyFailingAdamW(original_adamw):
        calls=0
        def step(self,closure=None):
            type(self).calls+=1
            if type(self).calls==3:
                parameter=self.param_groups[0]['params'][0]
                with torch.no_grad():parameter.add_(.125)
                raise RuntimeError('injected partial optimizer mutation')
            return super().step(closure)
    monkeypatch.setattr(torch.optim,'AdamW',PartiallyFailingAdamW)
    checkpoint=tmp_path/'run'/'checkpoint.pt'
    pre_failure_hash=None
    original_step=PartiallyFailingAdamW.step
    # Capture the last valid disk checkpoint immediately before the mutating failure.
    def step(self,closure=None):
        nonlocal pre_failure_hash
        if type(self).calls==2:
            # checkpoint-every=2 writes step two after its optimizer update.
            pass
        if type(self).calls==2 and checkpoint.exists():
            pre_failure_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return original_step(self,closure)
    monkeypatch.setattr(PartiallyFailingAdamW,'step',step)
    with pytest.raises(RuntimeError,match='injected partial optimizer mutation'):
        module.main(args)
    assert pre_failure_hash is not None
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==pre_failure_hash
    saved=torch.load(checkpoint,weights_only=False)
    assert saved['step']==2 and 'emergency_recovery' not in saved
    # Live memory may be partially mutated; the persisted older checkpoint is the only safe resume point.
    live={**{'backbone.'+name:value.detach().cpu() for name,value in engines[-1].backbone.hf.named_parameters()},
          **{'heads.'+name:value.detach().cpu() for name,value in engines[-1].heads.named_parameters()}}
    assert any(not torch.equal(value,live[name]) for name,value in saved['student_parameters'].items() if name in live)


def test_postcommit_telemetry_failure_saves_current_model_optimizer_and_rng(tmp_path,monkeypatch):
    import copy
    import errno
    from pathlib import Path
    from natlang_neuralese.train import text_warmup
    module,args,engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=3)
    original_open=Path.open
    def fail_training_log(path,*open_args,**kwargs):
        mode=(open_args[0] if open_args else kwargs.get('mode','r'))
        if path.name=='train.jsonl' and mode=='a':
            raise OSError(errno.ENOSPC,'injected telemetry disk full')
        return original_open(path,*open_args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_training_log)

    original_capture=module.capture_training_rng_state
    rng_snapshots=[]
    def capture(device):
        value=original_capture(device)
        rng_snapshots.append(copy.deepcopy(value))
        return value
    monkeypatch.setattr(module,'capture_training_rng_state',capture)

    original_step=torch.optim.AdamW.step
    optimizer_snapshots=[]
    def capture_step(optimizer,*step_args,**step_kwargs):
        result=original_step(optimizer,*step_args,**step_kwargs)
        optimizer_snapshots.append(copy.deepcopy(optimizer.state_dict()))
        return result
    monkeypatch.setattr(torch.optim.AdamW,'step',capture_step)

    with pytest.raises(SystemExit) as stopped:
        module.main(args)
    assert stopped.value.code==2
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert saved['step']==1
    assert len(optimizer_snapshots)==1 and len(rng_snapshots)==3
    _assert_nested_state_equal(saved['optimizer'],optimizer_snapshots[0])
    live={**{'backbone.'+name:value.detach().cpu() for name,value in engines[-1].backbone.hf.named_parameters()},
          **{'heads.'+name:value.detach().cpu() for name,value in engines[-1].heads.named_parameters()}}
    for name,value in saved['student_parameters'].items():
        torch.testing.assert_close(value,live[name],atol=0,rtol=0)
    assert saved['python_rng']==rng_snapshots[-1]['python_rng']
    torch.testing.assert_close(saved['torch_rng'],rng_snapshots[-1]['torch_rng'],atol=0,rtol=0)
    recovery=saved['emergency_recovery']
    assert recovery['safe_to_resume'] is True
    assert recovery['failure_stage']=='after_optimizer_commit'
    assert recovery['last_committed_step']==recovery['failed_attempt_step']==1
    assert recovery['optimizer_step_committed'] is True
    assert recovery['current_rng_saved_for_resume'] is True
    assert recovery['persistence_failure'] is True
    assert recovery['error_type']=='OSError' and 'injected telemetry disk full' in recovery['error']
    assert torch.load(tmp_path/'run'/'heads.pt',weights_only=False)['warmup']['step']==saved['step']
    assert not (tmp_path/'run'/'.checkpoint-space.reserve').exists()


def test_postcommit_serving_export_failure_does_not_fail_full_recovery(tmp_path,monkeypatch,capsys):
    import errno,json
    from pathlib import Path
    from natlang_neuralese.train import text_warmup
    module,args,_engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=3)
    original_open=Path.open
    def fail_training_log(path,*open_args,**kwargs):
        mode=(open_args[0] if open_args else kwargs.get('mode','r'))
        if path.name=='train.jsonl' and mode=='a':
            raise OSError(errno.ENOSPC,'injected telemetry disk full')
        return original_open(path,*open_args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_training_log)
    atomic=module.atomic_checkpoint;head_exports=0
    def fail_emergency_heads(path,state):
        nonlocal head_exports
        if Path(path).name=='heads.pt':
            head_exports+=1
            if head_exports==2:
                raise OSError(errno.ENOSPC,'injected optional heads export full')
        return atomic(path,state)
    monkeypatch.setattr(module,'atomic_checkpoint',fail_emergency_heads)

    with pytest.raises(SystemExit) as stopped:
        module.main(args)
    assert stopped.value.code==2
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert saved['step']==1 and saved['emergency_recovery']['safe_to_resume'] is True
    assert saved['emergency_recovery']['failure_stage']=='after_optimizer_commit'
    assert torch.load(tmp_path/'run'/'heads.pt',weights_only=False)['warmup']['step']==0
    status=json.loads((tmp_path/'run'/'heads-export-status.json').read_text())
    assert status['checkpoint_step']==1 and status['heads_step']==0
    assert status['heads_current'] is False
    assert status['checkpoint_authoritative_for_resume'] is True
    assert status['export_error']=={'type':'OSError','message':'[Errno 28] injected optional heads export full'}
    event=next(json.loads(line) for line in capsys.readouterr().out.splitlines()
               if 'postcommit_emergency_checkpoint_saved' in line)
    assert event['safe_to_resume'] is True and event['serving_heads_exported'] is False
    assert event['heads_export_status_written'] is True
    assert not (tmp_path/'run'/'.checkpoint-space.reserve').exists()


def test_resume_marks_heads_step_unknown_without_matching_status_receipt(tmp_path,monkeypatch):
    import errno,json
    from pathlib import Path
    from natlang_neuralese.train import text_warmup
    module,args,_engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=4)
    original_open=Path.open
    train_writes=0
    def fail_third_train_log(path,*open_args,**kwargs):
        nonlocal train_writes
        mode=(open_args[0] if open_args else kwargs.get('mode','r'))
        if path.name=='train.jsonl' and mode=='a':
            train_writes+=1
            if train_writes==3:
                raise OSError(errno.ENOSPC,'injected first-run telemetry failure')
        return original_open(path,*open_args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_third_train_log)
    with pytest.raises(SystemExit):module.main(args)
    assert torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)['step']==3

    # Simulate a missed/stale status update and a lagging serving artifact.
    heads_path=tmp_path/'run'/'heads.pt'
    heads=torch.load(heads_path,weights_only=False)
    heads['warmup']['step']=0
    torch.save(heads,heads_path)
    (tmp_path/'run'/'heads-export-status.json').unlink()
    def fail_resumed_train_log(path,*open_args,**kwargs):
        mode=(open_args[0] if open_args else kwargs.get('mode','r'))
        if path.name=='train.jsonl' and mode=='a':
            raise OSError(errno.ENOSPC,'injected resumed telemetry failure')
        return original_open(path,*open_args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_resumed_train_log)
    atomic=module.atomic_checkpoint
    def fail_resumed_heads(path,state):
        if Path(path).name=='heads.pt':
            raise OSError(errno.ENOSPC,'injected resumed optional export failure')
        return atomic(path,state)
    monkeypatch.setattr(module,'atomic_checkpoint',fail_resumed_heads)
    with pytest.raises(SystemExit):module.main(args)

    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert saved['step']==4 and saved['emergency_recovery']['safe_to_resume'] is True
    status=json.loads((tmp_path/'run'/'heads-export-status.json').read_text())
    assert status['checkpoint_step']==4 and status['heads_step']==-1
    assert status['heads_step_known'] is False and status['heads_current'] is False
    assert status['export_error']['message']=='[Errno 28] injected resumed optional export failure'


def test_checkpoint_space_preflight_refuses_before_first_update(tmp_path,monkeypatch):
    from natlang_neuralese.train import text_warmup
    module,args,engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=2)
    original_step=torch.optim.AdamW.step
    calls=[]
    def count_step(optimizer,*step_args,**step_kwargs):
        calls.append(True)
        return original_step(optimizer,*step_args,**step_kwargs)
    monkeypatch.setattr(torch.optim.AdamW,'step',count_step)
    def refuse(_self):
        raise CheckpointReserveError('injected insufficient reserve capacity')
    monkeypatch.setattr(CheckpointDiskReserve,'acquire',refuse)

    with pytest.raises(SystemExit) as stopped:
        module.main(args)
    assert stopped.value.code==2
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert saved['step']==0
    assert calls==[]
    assert (tmp_path/'run'/'baseline.json').is_file()


def test_lost_reserve_saves_committed_update_then_exits_nonzero(tmp_path,monkeypatch):
    import copy
    from natlang_neuralese.train import text_warmup
    module,args,engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=3)
    original_ensure=CheckpointDiskReserve.ensure
    calls=0
    def lose_reserve_after_first_update(reserve):
        nonlocal calls
        calls+=1
        if calls==2:
            raise CheckpointReserveError('injected reserve lost to filesystem race')
        return original_ensure(reserve)
    monkeypatch.setattr(CheckpointDiskReserve,'ensure',lose_reserve_after_first_update)
    original_capture=module.capture_training_rng_state
    snapshots=[]
    def capture(device):
        value=original_capture(device)
        snapshots.append(copy.deepcopy(value))
        return value
    monkeypatch.setattr(module,'capture_training_rng_state',capture)

    with pytest.raises(SystemExit) as stopped:
        module.main(args)
    assert stopped.value.code==2
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert saved['step']==1
    assert saved['python_rng']==snapshots[-1]['python_rng']
    torch.testing.assert_close(saved['torch_rng'],snapshots[-1]['torch_rng'],atol=0,rtol=0)
    recovery=saved['emergency_recovery']
    assert recovery['safe_to_resume'] is True
    assert recovery['last_committed_step']==1
    assert recovery['error']=='injected reserve lost to filesystem race'
    assert not (tmp_path/'run'/'.checkpoint-space.reserve').exists()


def test_failed_postcommit_emergency_save_reports_unsafe_and_exits_nonzero(tmp_path,monkeypatch,capsys):
    import errno
    import json
    from pathlib import Path
    from natlang_neuralese.train import text_warmup
    module,args,_engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=3)
    original_open=Path.open
    def fail_telemetry(path,*open_args,**kwargs):
        mode=(open_args[0] if open_args else kwargs.get('mode','r'))
        if path.name=='train.jsonl' and mode=='a':
            raise OSError(errno.ENOSPC,'injected telemetry disk full')
        return original_open(path,*open_args,**kwargs)
    monkeypatch.setattr(Path,'open',fail_telemetry)
    atomic=module.atomic_checkpoint
    def fail_emergency(path,state):
        if 'emergency_recovery' in state:
            raise OSError('injected emergency checkpoint failure')
        return atomic(path,state)
    monkeypatch.setattr(module,'atomic_checkpoint',fail_emergency)

    with pytest.raises(RuntimeError,match='emergency checkpoint could not be saved'):
        module.main(args)
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    failure=next(event for event in events if event.get('event')=='postcommit_emergency_checkpoint_failed')
    assert failure['safe_to_resume'] is False
    assert failure['last_committed_step']==1
    assert failure['checkpoint_error']=='injected emergency checkpoint failure'
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',weights_only=False)
    assert saved['step']==0 and 'emergency_recovery' not in saved


def test_checkpoint_reserve_allocates_releases_rearms_and_cleans(tmp_path):
    path=tmp_path/'.checkpoint-space.reserve'
    reserve=CheckpointDiskReserve(path,1024*1024)
    assert reserve.acquire()==1024*1024
    assert path.stat().st_size==1024*1024
    reserve.release_space()
    assert path.stat().st_size==len(reserve._MAGIC)
    assert reserve.ensure()==1024*1024
    reserve.cleanup()
    assert not path.exists()


def test_checkpoint_reserve_preflight_refuses_when_free_space_is_too_small(tmp_path,monkeypatch):
    import shutil
    reserve=CheckpointDiskReserve(tmp_path/'.checkpoint-space.reserve',1024*1024)
    usage_type=type(shutil.disk_usage(tmp_path))
    monkeypatch.setattr('natlang_neuralese.train.checkpoint_safety.shutil.disk_usage',
                        lambda _path:usage_type(4096,4096,0))
    with pytest.raises(CheckpointReserveError,match='only 0 are free'):
        reserve.acquire()
    assert not reserve.path.exists()


def test_failed_atomic_checkpoint_removes_partial_owned_pending_file(tmp_path,monkeypatch):
    from natlang_neuralese.train.trajectory_state import atomic_checkpoint
    path=tmp_path/'checkpoint.pt'
    path.write_bytes(b'last-valid-checkpoint')
    original=path.read_bytes()
    def partial_then_fail(_state,stream):
        stream.write(b'partial checkpoint')
        raise OSError('injected checkpoint write failure')
    monkeypatch.setattr(torch,'save',partial_then_fail)
    with pytest.raises(OSError,match='injected checkpoint write failure'):
        atomic_checkpoint(path,{'step':2})
    assert path.read_bytes()==original
    assert not path.with_suffix('.pending').exists()


def test_checkpoint_size_upper_bound_includes_weights_heads_and_optimizer_slots():
    parameter=torch.nn.Parameter(torch.zeros(10,dtype=torch.bfloat16))
    heads=torch.nn.Linear(4,2,bias=True).to(dtype=torch.bfloat16)
    optimizer=torch.optim.AdamW([parameter,*heads.parameters()])
    estimated=warmup_checkpoint_size_upper_bound([('backbone.weight',parameter)],heads,optimizer)
    weight_bytes=parameter.numel()*parameter.element_size()
    head_bytes=sum(value.numel()*value.element_size() for value in heads.state_dict().values())
    assert estimated > weight_bytes+head_bytes


def _optimizer_state_tensor_bytes(state):
    return sum(value.numel()*value.element_size() for value in state.values()
               if isinstance(value,torch.Tensor))


def test_adamw_checkpoint_reserve_matches_initialized_and_lazy_state_layouts():
    parameter=torch.nn.Parameter(torch.zeros(5,dtype=torch.bfloat16))
    optimizer=torch.optim.AdamW([parameter])
    lazy_bound=optimizer_state_size_upper_bound(optimizer)
    assert lazy_bound==2*parameter.numel()*parameter.element_size()+4

    parameter.grad=torch.ones_like(parameter)
    optimizer.step()
    initialized_bytes=_optimizer_state_tensor_bytes(optimizer.state[parameter])
    assert initialized_bytes==lazy_bound
    assert optimizer_state_size_upper_bound(optimizer)==initialized_bytes


def test_muon_checkpoint_reserve_matches_initialized_and_lazy_state_layouts():
    muon_type=getattr(torch.optim,'Muon',None)
    if muon_type is None:
        pytest.skip('installed PyTorch has no torch.optim.Muon')
    parameter=torch.nn.Parameter(torch.zeros((4,4),dtype=torch.bfloat16))
    optimizer=muon_type([parameter],lr=0.01)
    lazy_bound=optimizer_state_size_upper_bound(optimizer)
    assert lazy_bound==parameter.numel()*parameter.element_size()

    parameter.grad=torch.ones_like(parameter)
    optimizer.step()
    initialized_bytes=_optimizer_state_tensor_bytes(optimizer.state[parameter])
    assert initialized_bytes==lazy_bound
    assert optimizer_state_size_upper_bound(optimizer)==initialized_bytes


def test_port_muon_adamw_reserve_uses_each_partition_layout():
    if not hasattr(torch.optim,'Muon'):
        pytest.skip('installed PyTorch has no torch.optim.Muon')
    matrix=torch.nn.Parameter(torch.zeros((4,4),dtype=torch.bfloat16))
    vector=torch.nn.Parameter(torch.zeros(4,dtype=torch.bfloat16))
    optimizer=PortMuonAdamW([('backbone.matrix',matrix),('heads.bias',vector)],
                            lr=0.01,vocab_size=100)
    expected=(matrix.numel()*matrix.element_size()
              +2*vector.numel()*vector.element_size()+4)
    assert optimizer_state_size_upper_bound(optimizer)==expected

    matrix.grad=torch.ones_like(matrix)
    vector.grad=torch.ones_like(vector)
    optimizer.step()
    initialized=(_optimizer_state_tensor_bytes(optimizer.muon.state[matrix])
                 +_optimizer_state_tensor_bytes(optimizer.auxiliary.state[vector]))
    assert initialized==expected
    assert optimizer_state_size_upper_bound(optimizer)==initialized


def test_checkpoint_reserve_refuses_unknown_optimizer_layout():
    parameter=torch.nn.Parameter(torch.zeros(4))
    optimizer=torch.optim.SGD([parameter],lr=0.1)
    with pytest.raises(CheckpointReserveError,match='does not know optimizer state layout'):
        optimizer_state_size_upper_bound(optimizer)


def test_restart_before_first_checkpoint_preserves_partial_files(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup
    def load(*_args):
        backbone,heads=tiny_student()
        return SimpleNamespace(backbone=backbone,heads=heads,tokenizer=None,_tokens=lambda _text:[9,3,5,8]),None
    monkeypatch.setattr(text_warmup,'load_initial',load)
    heads_path=tmp_path/'heads.pt';torch.save({},heads_path)
    records=tmp_path/'records.jsonl';records.write_text('')
    text=tmp_path/'text.jsonl'
    text.write_text('\n'.join(json.dumps({'text':s,'split':split,'source_groups':[s]})
                              for s,split in [('train','train'),('held','test')])+'\n')
    out=tmp_path/'run';out.mkdir();(out/'plan.json').write_text('{"interrupted": true}\n')
    text_warmup.main(['--heads',str(heads_path),'--records',str(records),'--text-data',str(text),
        '--out',str(out),'--device','cpu','--steps','1','--tokens','8','--prefix-tokens','2','--batch','1',
        '--eval-batch','1','--held-documents','1','--eval-every','1','--checkpoint-every','1',
        '--optimizer','adamw','--backbone-training','full'])
    aborted=[x for x in out.iterdir() if x.name.startswith('aborted-')]
    assert len(aborted)==1 and json.loads((aborted[0]/'plan.json').read_text())=={'interrupted':True}
    assert (out/'checkpoint.pt').exists()
    # A live run profiles its next update on request; extending --steps is an operational resume.
    (out/'profile-request').touch()
    text_warmup.main(['--heads',str(heads_path),'--records',str(records),'--text-data',str(text),
        '--out',str(out),'--device','cpu','--steps','2','--tokens','8','--prefix-tokens','2','--batch','1',
        '--eval-batch','1','--held-documents','1','--eval-every','1','--checkpoint-every','1',
        '--optimizer','adamw','--backbone-training','full'])
    assert not (out/'profile-request').exists()
    assert 'Self CPU' in (out/'profile-step2.txt').read_text()


def test_rollout_stage_unfreezes_after_sketch_plateau_and_restores():
    from natlang_neuralese.train.foundation_schedule import RolloutStage
    stage=RolloutStage(passes=6,start_passes=6,min_evals=2,patience=2)
    phases=[stage.observe(v)['phase'] for v in (2.0,1.5,1.2,1.19,1.195,1.18)]
    assert phases==['sketch_only']*4+['whole_stack']*2
    again=RolloutStage(passes=6,start_passes=6,min_evals=2,patience=2);again.load_state_dict(stage.state_dict())
    assert again.controls()==stage.controls()
    import pytest
    with pytest.raises(ValueError):RolloutStage(passes=4).load_state_dict(stage.state_dict())
    assert RolloutStage(passes=3,sketch_first=False).controls()['sketch_only'] is False
    negative=RolloutStage(passes=4,start_passes=4,min_evals=1,patience=1,min_relative_improvement=10)
    assert [negative.observe(v)['phase'] for v in (-0.03,-0.04)]==['sketch_only','whole_stack']
    ramp=RolloutStage(passes=6,start_passes=4,min_evals=1,patience=1)
    assert [(c['phase'],c['passes']) for c in map(ramp.observe,(2.,2.,1.,1.,.5,.5))]==[
        ('sketch_only',4),('sketch_only',5),('sketch_only',5),('sketch_only',6),('sketch_only',6),('whole_stack',6)]
    legacy={'schema':'natlang.sketch-rollout-stage/1','config':{'passes':6,'sketch_first':True},'phase':'sketch_only',
            'history':[3.0],'best':3.0,'last_significant':1,'unfrozen_at_eval':None}
    restarted=RolloutStage(passes=6,start_passes=4);restarted.load_state_dict(legacy)
    assert restarted.controls()['passes']==4 and restarted.controls()['sketch_only']


def test_sketch_rollout_trains_only_the_sketch_at_depth_then_evaluates_every_pass(tmp_path,monkeypatch):
    import json
    module,args,_engines=_tiny_warmup_run_inputs(tmp_path,monkeypatch,steps=10)
    args=[x for x in args]
    args[args.index('--eval-every')+1]='1'
    # Every improvement is insignificant, so both plateaus (projection, then sketch rollout) arrive early.
    args+=['--projection-min-evals','1','--projection-patience','1','--projection-min-improvement','10',
           '--rollout-passes','4','--rollout-start-passes','3']
    module.main(args)
    run=tmp_path/'run'
    rows=[json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
    sketch_rows=[r for r in rows if r['schedule'].get('rollout',{}).get('sketch_only')]
    assert sketch_rows, [r['schedule'] for r in rows]
    for row in sketch_rows:
        assert row['schedule']['sequence_passes'] in (3,4)
        assert row['backbone_gradient_norm']==0
        assert row['sketch_gradient_norm']>0
    evals=[json.loads(line) for line in (run/'eval.jsonl').read_text().splitlines()]
    assert evals[-1]['evaluation_passes']==4
    assert any(key.startswith('pass-3-') for key in evals[-1]['strata'])
    assert 'rollout' in evals[-1]
    assert {r['schedule']['sequence_passes'] for r in sketch_rows}=={3,4}
    assert any(r['schedule'].get('rollout',{}).get('phase')=='whole_stack' and r['backbone_gradient_norm']>0 for r in rows)
