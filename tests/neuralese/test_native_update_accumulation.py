"""Accumulated native updates: several packed micro-batches, one optimizer step on their summed gradient."""
import copy
import json

import torch

from test_native_optimizer_checkpointing import examples
from test_shared_state import tiny


def build():
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    model = SharedStateModel(tiny('lfm'), (58, 59, 60, 61), rank=4)
    optimizer, _ = make_native_optimizer(model, lr=1e-3, learned_row_lr=.003, in_backward=True)
    return model, optimizer


def stepped_gradients(model, optimizer):
    """Record the gradient each parameter carries when the optimizer consumes it."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    seen = {}
    latent_update = optimizer.latent._update

    def latent(parameter, group):
        assert names[id(parameter)] not in seen, 'a latent stepped twice in one update'
        seen[names[id(parameter)]] = parameter.grad.detach().clone()
        return latent_update(parameter, group)
    optimizer.latent._update = latent
    step = optimizer.step

    def ordinary(*args, **kwargs):
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                seen[name] = parameter.grad.detach().clone()
        return step(*args, **kwargs)
    optimizer.step = ordinary
    return seen


def test_two_micro_batches_step_once_like_one_batch_of_all_graphs():
    from natlang_neuralese.train.native_state_sft import native_training_update
    whole, whole_optimizer = build()
    split, split_optimizer = build()
    split.load_state_dict(whole.state_dict())
    split.port.hf.load_state_dict(whole.port.hf.state_dict())
    before = {name: value.detach().clone() for name, value in whole.named_parameters()}
    expected_gradients = stepped_gradients(whole, whole_optimizer)
    actual_gradients = stepped_gradients(split, split_optimizer)
    graphs = examples()
    expected, expected_details = native_training_update(whole, whole_optimizer, graphs,
                                                        checkpoint_layers=True, chunk_tokens=3)
    actual, actual_details = native_training_update(split, split_optimizer, graphs[1:], accumulate=[graphs[:1]],
                                                    checkpoint_layers=True, chunk_tokens=3)
    torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-7)
    # The optimizer consumes the same summed gradient, up to FP32 summation order.
    assert actual_gradients.keys() == expected_gradients.keys() and len(expected_gradients) > 10
    for name, gradient in expected_gradients.items():
        torch.testing.assert_close(actual_gradients[name], gradient, rtol=1e-5, atol=1e-8, msg=name)
    # Newton-Schulz runs in BF16, so the committed steps agree in direction rather than bit for bit.
    for name, value in split.named_parameters():
        reference = dict(whole.named_parameters())[name].detach() - before[name]
        delta = value.detach() - before[name]
        if reference.abs().max() > 0:
            cosine = torch.nn.functional.cosine_similarity(delta.flatten(), reference.flatten(), dim=0)
            assert cosine > .999, (name, float(cosine))
    # One update: every latent stepped exactly once and holds no gradient afterwards.
    assert all(parameter.grad is None for parameter in split_optimizer.latent.param_groups[0]['params'])
    assert len(actual_details) == len(expected_details) == 3
    assert actual_details[0]['original_physical_tokens'] == expected_details[0]['original_physical_tokens']


def test_efficiency_only_controls_actual_muon_and_adam_update_with_history(monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.rl.native_replay import efficiency_only_update_scale
    torch.manual_seed(818)
    model, optimizer=build(); other, other_optimizer=build()
    other.load_state_dict(model.state_dict())
    other.port.hf.load_state_dict(model.port.hf.state_dict())
    graphs=examples()[:2]
    def loss(model, micro, **kwargs):
        value=sum((parameter.float().square().sum() for parameter in model.parameters() if parameter.requires_grad))
        return value, {'objective':'test'}
    monkeypatch.setattr(trainer,'packed_graph_loss',loss)
    # Establish the same nonzero actual optimizer history before comparing.
    trainer.native_training_update(model,optimizer,graphs,checkpoint_layers=False)
    trainer.native_training_update(other,other_optimizer,graphs,checkpoint_layers=False)
    rows={example.id:{'efficiency_only_update_scale':.01} for example in graphs}
    spec={'efficiency_update_control':{'schema':'natlang.efficiency-only-update-scaling/1','maximum':.05}}
    objective=SimpleNamespace(spec=spec,rows=rows,policy_loss=None)
    assert efficiency_only_update_scale(objective,graphs)==.01
    before={name:value.detach().clone() for name,value in model.named_parameters()}
    original_rates=[group['lr'] for group in optimizer.param_groups]
    _,details=trainer.native_training_update(model,optimizer,graphs[1:],accumulate=[graphs[:1]],
                                             checkpoint_layers=False,objective=objective)
    trainer.native_training_update(other,other_optimizer,graphs[1:],accumulate=[graphs[:1]],
                                   checkpoint_layers=False)
    assert [group['lr'] for group in optimizer.param_groups]==original_rates
    assert details['efficiency_update_control']['scale']==.01
    for child in (optimizer.latent,optimizer.auxiliary):
        ids={id(p) for group in child.param_groups for p in group['params']}
        pairs=[(name,p) for name,p in model.named_parameters() if id(p) in ids]
        actual=torch.cat([(p.detach()-before[name]).flatten() for name,p in pairs])
        reference=torch.cat([(dict(other.named_parameters())[name].detach()-before[name]).flatten() for name,_ in pairs])
        assert reference.norm()>0
        torch.testing.assert_close(actual.norm()/reference.norm(),torch.tensor(.01),rtol=.03,atol=.001)
    objective.policy_loss={'schema':'jolt-plus/1','beta_s':1.,'beta_t':.5}
    rows[graphs[0].id]['jolt']={'role':'student','privileged':{'actual':'twin'}}
    assert efficiency_only_update_scale(objective,graphs)==1.
    del rows[graphs[0].id]['jolt'];del rows[graphs[0].id]['efficiency_only_update_scale']
    assert efficiency_only_update_scale(objective,graphs)==1.


def test_accumulated_loss_weights_micro_batches_by_scored_graphs():
    from natlang_neuralese.train.native_state_sft import native_training_update, packed_graph_loss
    model, optimizer = build()
    graphs = examples()
    reference = copy.deepcopy(model)
    with torch.no_grad():
        first, _ = packed_graph_loss(reference, graphs[:2], checkpoint_layers=True, chunk_tokens=3,
                                     collect_accuracy=False)
        second, _ = packed_graph_loss(reference, graphs[2:], checkpoint_layers=True, chunk_tokens=3,
                                      collect_accuracy=False)
    loss, _ = native_training_update(model, optimizer, graphs[2:], accumulate=[graphs[:2]],
                                     checkpoint_layers=True, chunk_tokens=3)
    torch.testing.assert_close(loss, (2 * first + second) / 3, rtol=1e-6, atol=1e-7)


def test_merge_update_details_sums_packed_totals_and_weights_objective_means():
    from natlang_neuralese.train.native_state_sft import merge_update_details
    first = [{'packed_physical_tokens': 10, 'original_physical_tokens': 12, 'supervised_tokens': 3, 'weight': 1.}]
    second = [{'packed_physical_tokens': 5, 'original_physical_tokens': 5, 'supervised_tokens': 2, 'weight': 2.}] * 2
    merged = merge_update_details([first, second], [1, 2])
    assert [item['supervised_tokens'] for item in merged] == [3, 2, 2]
    assert all(item['packed_physical_tokens'] == 15 and item['original_physical_tokens'] == 17 for item in merged)
    objective = merge_update_details([{'objective': 'dpo', 'preference_pairs': 1, 'preference_accuracy': 1.},
                                      {'objective': 'dpo', 'preference_pairs': 3, 'preference_accuracy': 0.}], [1, 3])
    assert objective == {'objective': 'dpo', 'preference_pairs': 4, 'preference_accuracy': .25}


def test_trainer_steps_once_per_accumulated_group_and_records_the_transition(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_state_sft as trainer
    from test_native_source_fetch_recovery import source
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    base = tmp_path / 'base'
    paths = source(base, 'lfm')
    row = json.loads(paths['train'].read_text())
    rows = []
    for name in 'abcde':
        item = json.loads(json.dumps(row))
        item['id'] = name
        rows.append(item)
    paths['train'].write_text(''.join(json.dumps(item) + '\n' for item in rows))
    observed = []
    update = trainer.native_training_update

    def counted(model, optimizer, batch, **kwargs):
        observed.append([len(micro) for micro in (*kwargs.get('accumulate', ()), batch)])
        return update(model, optimizer, batch, **kwargs)
    monkeypatch.setattr(trainer, 'native_training_update', counted)

    def run(out, accumulate, resume=False, epochs='1'):
        args = ['--base', str(base), '--dtype', 'bfloat16', '--rank', '4', '--device', 'cpu',
                '--native-graphs', str(paths['train']), '--held-native-graphs', str(paths['test']),
                '--epochs', epochs, '--batch-graphs', '1', '--accumulate-batches', str(accumulate),
                '--eval-every', '1', '--checkpoint-every', '1', '--checkpoint-seconds', '0', '--seed', '23',
                '--out', str(out)]
        if resume:
            args += ['--continue-from', str(out / 'checkpoint.pt')]
        assert trainer.main(args) == 0
        return torch.load(out / 'checkpoint.pt', weights_only=False, map_location='cpu')['training']

    out = tmp_path / 'run'
    saved = run(out, 1, epochs='0.4')
    assert observed == [[1], [1]] and saved['global_step'] == 2
    observed.clear()
    saved = run(out, 2, resume=True)
    # Three remaining single-graph batches: one update of two micro-batches, then one of the last.
    assert observed == [[1, 1], [1]]
    assert saved['global_step'] == 4
    assert sorted(saved['exposure_counts'].values()) == [1, 1, 1, 1, 1]
    history = saved['options']['accumulate_batches_history']
    assert [(item['from'], item['to'], item['at_global_step']) for item in history] == [(1, 2, 2)]


def test_real_jolt_multi_microbatch_summarizes_and_steps_once(tmp_path):
    from test_rl_jolt import load,episode_with_top_k,rollouts,prepare,objective_for,forward
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
    model,roles,identity=load('lfm')
    episode=episode_with_top_k(model,roles,identity)
    rows=rollouts(episode.turns,rewards=(1,0))
    graphs,spec=prepare(rows,episode,roles,identity)
    objective=objective_for(tmp_path,spec)
    examples_=[forward(model,graph,identity)[0] for graph in graphs]
    optimizer,_=make_native_optimizer(model,lr=1e-3,in_backward=True)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    loss,details=native_training_update(model,optimizer,examples_[1:],accumulate=[examples_[:1]],
        objective=objective,checkpoint_layers=True,chunk_tokens=8)
    assert torch.isfinite(loss) and details['episodes']==2
    assert sum(details['jolt']['positions'].values())==details['sampled_tokens']+details['sampled_decisions']
    assert details['jolt']['unprivileged_episodes']+details['jolt']['privileged_episodes']==2
    assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_bad_accumulation_telemetry_fails_before_any_optimizer_step(monkeypatch):
    import pytest
    from natlang_neuralese.train import native_state_sft as trainer
    model,optimizer=build();graphs=examples();before={n:p.detach().clone() for n,p in model.named_parameters()}
    def fail(*args,**kwargs):raise ValueError('bad telemetry')
    monkeypatch.setattr(trainer,'merge_update_details',fail)
    with pytest.raises(ValueError,match='bad telemetry'):
        trainer.native_training_update(model,optimizer,graphs[1:],accumulate=[graphs[:1]],checkpoint_layers=True,chunk_tokens=3)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
    assert all(not child['state'] for child in (optimizer.state_dict()['muon'],optimizer.state_dict()['adamw'],optimizer.state_dict().get(optimizer.latent_key)) if child)


def test_real_jolt_student_teacher_microbatches_retain_distinct_role_telemetry(tmp_path):
    from test_rl_jolt import load,episode_with_top_k,rollouts,DECLARATION,rendering,custody,train_case,objective_for,forward
    from natlang_neuralese.rl.jolt import privileged_source
    from natlang_neuralese.rl.native_replay import prepare_objective
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update,packed_graph_loss
    model,roles,identity=load('lfm')
    student=episode_with_top_k(model,roles,identity,mode='write-only')
    teacher=episode_with_top_k(model,roles,identity,mode='text')
    rows=rollouts(student.turns,rewards=(0,))+rollouts(teacher.turns,role='teacher',rewards=(1,))
    rows[1]['sample']=1
    states={**student.states,**teacher.states}
    priv=privileged_source(rows,cases={train_case()['id']:train_case()},rendering=rendering(),custody=custody(),fetch_text=lambda key:states[key].text)
    graphs,spec=prepare_objective(rows,identity=identity,roles=roles,fetch_block=states.__getitem__,policy_loss=DECLARATION,privileged=priv)
    objective=objective_for(tmp_path,spec)
    batch=[forward(model,graph,identity)[0] for graph in graphs]
    reference=copy.deepcopy(model)
    # The policy loss is one action mean over the complete update: the accumulated two-micro-batch update must
    # equal the same graphs in ONE micro-batch (packing-invariant), not a mean of per-micro-batch means.
    value,_=packed_graph_loss(reference,batch,objective=objective,checkpoint_layers=True,chunk_tokens=8,collect_accuracy=False)
    value.backward()
    reference_grad={n:p.grad.detach().clone() for n,p in reference.named_parameters() if p.grad is not None}
    optimizer,_=make_native_optimizer(model,lr=1e-3,in_backward=True)
    seen=stepped_gradients(model,optimizer)
    _,details=native_training_update(model,optimizer,batch[1:],accumulate=[batch[:1]],objective=objective,checkpoint_layers=True,chunk_tokens=8)
    assert details['episodes']==2 and details['jolt']['positions']['student']>0 and details['jolt']['positions']['teacher']>0
    assert details['jolt']['mean_kl']['student'] is not None and details['jolt']['mean_kl']['teacher'] is not None
    from natlang_neuralese.train.native_state_sft import record_objective_details
    progress={};record_objective_details(progress,details)
    saved=json.loads(json.dumps(progress))['objective_details']
    assert saved['sampled_tokens']+saved['sampled_decisions']==sum(saved['jolt']['positions'].values())
    assert all(name in saved['jolt'] for name in ('student_outcome','student_opsd','teacher_outcome','teacher_kl'))
    assert seen.keys() <= reference_grad.keys() and len(seen)>10
    for name in seen:
        expected_gradient=reference_grad[name]
        torch.testing.assert_close(seen[name],expected_gradient,rtol=1e-5,atol=1e-8,msg=name)


def test_accumulated_centering_uses_position_weighted_means():
    from natlang_neuralese.train.native_state_sft import merge_update_details
    a={'positions':2,'rho_mean':1.,'rho_max':3.,'w_y_zero_fraction':0.}
    b={'positions':6,'rho_mean':3.,'rho_max':5.,'w_y_zero_fraction':.5}
    result=merge_update_details([{'score_centering':a},{'score_centering':b}],[1,1])['score_centering']
    assert result=={'positions':8,'rho_mean':2.5,'rho_max':5.,'w_y_zero_fraction':.375}


def test_sft_budget_scales_every_actual_group_and_restores_on_success_and_failure(monkeypatch):
    import pytest
    from natlang_neuralese.train import native_state_sft as trainer
    model, optimizer=build(); graphs=examples()[:2]
    nominal=[(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups]
    observed=[]
    original=trainer.packed_graph_loss
    def inspect(*args,**kwargs):
        observed.append([(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups])
        return original(*args,**kwargs)
    monkeypatch.setattr(trainer,'packed_graph_loss',inspect)
    _,details=trainer.native_training_update(model,optimizer,graphs,checkpoint_layers=True,chunk_tokens=3,sft_update_reference=8)
    assert observed and all(abs(rate[0]-base[0]/4)<1e-12 for row in observed for rate,base in zip(row,nominal))
    assert all(qat is None or abs(qat-baseqat/4)<1e-12 for row in observed for (_,qat),(_,baseqat) in zip(row,nominal))
    assert [(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups]==nominal
    assert details[0]['sft_update_budget']['scored_graphs']==2
    assert details[0]['sft_update_budget']['multiplier']==.25
    monkeypatch.setattr(trainer,'packed_graph_loss',lambda *a,**k: (_ for _ in ()).throw(RuntimeError('failure')))
    with pytest.raises(RuntimeError,match='failure'):
        trainer.native_training_update(model,optimizer,graphs,checkpoint_layers=True,chunk_tokens=3,sft_update_reference=8)
    assert [(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups]==nominal


def test_sft_budget_caps_large_windows_and_rejects_zero_scored():
    import pytest
    from natlang_neuralese.train.native_state_sft import native_training_update
    model,optimizer=build(); graphs=examples()[:2]
    _,details=native_training_update(model,optimizer,graphs,checkpoint_layers=True,chunk_tokens=3,sft_update_reference=1)
    assert details[0]['sft_update_budget']['multiplier']==1
    for graph in graphs:
        graph.target_weights=[0 for _ in graph.target_weights]; graph.decisions=[]
    with pytest.raises(ValueError,match='at least one scored'):
        native_training_update(model,optimizer,graphs,checkpoint_layers=True,chunk_tokens=3,sft_update_reference=1)


def test_earned_accumulation_decrease_requires_reference_and_carries_captured_reference(tmp_path, monkeypatch):
    import pytest
    from natlang_neuralese.train import native_state_sft as trainer
    from test_native_source_fetch_recovery import source
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    paths=source(tmp_path/'base','lfm'); row=json.loads(paths['train'].read_text())
    paths['train'].write_text(''.join(json.dumps(dict(row,id=f'g{i}'))+'\n' for i in range(4)))
    out=tmp_path/'out'
    args=['--base',str(tmp_path/'base'),'--dtype','bfloat16','--rank','4','--device','cpu',
          '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
          '--epochs','0.5','--batch-graphs','1','--accumulate-batches','2',
          '--eval-every','100','--out',str(out)]
    assert trainer.main(args)==0
    slot=out/'checkpoint.pt'; state=torch.load(slot,weights_only=False,map_location='cpu')
    assert state['training']['options']['sft_update_budget_reference']['scored_graphs']==2
    changed=copy.deepcopy(state);changed['training']['options'].pop('sft_update_budget_reference')
    for frame in (changed['training'].get('sft_progress') or {}).values():
        (frame.get('options') or {}).pop('sft_update_budget_reference',None)
    torch.save(changed,slot)
    continuation=args.copy(); continuation[continuation.index('--accumulate-batches')+1]='1'
    continuation[continuation.index('--epochs')+1]='1'; continuation+=['--continue-from',str(slot)]
    with pytest.raises(ValueError,match='decreasing SFT accumulation requires'):
        trainer.main(continuation)
    from natlang_neuralese.train.native_resident import binding
    pin=binding(slot); frame=changed['training'];counts=frame['exposure_counts']; rowbindings=frame['source_row_bindings']
    rows=[{'id':key,'delta':value,'scored':True,'previous_count':0,'current_count':value,'binding':rowbindings[key]}
          for key,value in counts.items() if value]
    delta={'schema':'natlang.sft-update-budget-row-delta/1','global_step':frame['global_step'],
           'checkpoint_sha256':pin['sha256'],'train_sha256':frame['source_receipts'][0]['sha256'],'rows':rows}
    delta_path=tmp_path/'delta.json';delta_path.write_text(json.dumps(delta))
    predecessor=tmp_path/'predecessor';predecessor.write_text('fixture')
    receipt={'schema':'natlang.sft-update-budget-reference/1','checkpoint_sha256':pin['sha256'],
             'train_sha256':delta['train_sha256'],'global_step':delta['global_step'],
             'scored_graphs':sum(row['delta'] for row in rows),'row_delta':binding(delta_path),
             'predecessor_checkpoint':binding(predecessor),'train_source':binding(paths['train'])}
    proof=tmp_path/'reference.json';proof.write_text(json.dumps(receipt))
    appended=tmp_path/'appended.jsonl'
    appended.write_text(paths['train'].read_text()+json.dumps(dict(row,id='additional'))+'\n')
    with_reference=continuation+['--sft-update-budget-reference',json.dumps(binding(proof)),
                                '--continued-native-graphs',str(paths['train'])]
    with_reference[with_reference.index('--native-graphs')+1]=str(appended)
    assert trainer.main(with_reference)==0
    explicit=torch.load(slot,weights_only=False,map_location='cpu')['training']
    saved_reference=explicit['options']['sft_update_budget_reference']
    assert saved_reference['evidence']==binding(proof)
    assert saved_reference['reference_train_sha256']==receipt['train_sha256']
    appended_again=tmp_path/'appended-again.jsonl'
    appended_again.write_text(appended.read_text()+json.dumps(dict(row,id='later'))+'\n')
    inherited=with_reference.copy()
    inherited[inherited.index('--native-graphs')+1]=str(appended_again)
    inherited[inherited.index('--continued-native-graphs')+1]=str(appended)
    inherited[inherited.index('--epochs')+1]='2'
    unrelated=tmp_path/'unrelated-reference.json';unrelated.write_text(json.dumps(receipt)+' ')
    bad=inherited.copy();bad[bad.index('--sft-update-budget-reference')+1]=json.dumps(binding(unrelated))
    with pytest.raises(ValueError,match='reference differs'):
        trainer.main(bad)
    original=proof.read_text();proof.write_text(original+' ')
    with pytest.raises(ValueError,match='sha|binding|changed'):
        trainer.main(inherited)
    proof.write_text(original)
    assert trainer.main(inherited)==0
    later=torch.load(slot,weights_only=False,map_location='cpu')['training']
    assert later['options']['sft_update_budget_reference']==saved_reference
    implicit=inherited.copy()
    for flag in ('--sft-update-budget-reference','--continued-native-graphs'):
        index=implicit.index(flag);del implicit[index:index+2]
    implicit[implicit.index('--epochs')+1]='3'
    proof.write_text(original+' ')
    with pytest.raises(ValueError,match='sha|binding|changed'):
        trainer.main(implicit)
    proof.write_text(original)
    assert trainer.main(implicit)==0
    assert torch.load(slot,weights_only=False,map_location='cpu')['training']['options']['sft_update_budget_reference']==saved_reference
    torch.save(state,slot)
    assert trainer.main(continuation)==0
    final=torch.load(slot,weights_only=False,map_location='cpu')['training']
    assert final['options']['sft_update_budget_reference']['scored_graphs']==2
    assert final['options']['sft_update_budget_last_update']['multiplier']==.5


def test_sft_budget_scales_real_muon_and_adam_displacements(monkeypatch):
    from natlang_neuralese.train import native_state_sft as trainer
    torch.manual_seed(821)
    model,optimizer=build(); other,other_optimizer=build()
    other.load_state_dict(model.state_dict());other.port.hf.load_state_dict(model.port.hf.state_dict())
    graphs=examples()[:2]
    def loss(model,micro,**kwargs):
        return sum(p.float().square().sum() for p in model.parameters() if p.requires_grad), {'objective':'fixture'}
    monkeypatch.setattr(trainer,'packed_graph_loss',loss)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    trainer.native_training_update(model,optimizer,graphs,checkpoint_layers=False,sft_update_reference=8)
    trainer.native_training_update(other,other_optimizer,graphs,checkpoint_layers=False)
    for child in (optimizer.latent,optimizer.auxiliary):
        ids={id(p) for group in child.param_groups for p in group['params']}
        selected=[(name,p) for name,p in model.named_parameters() if id(p) in ids]
        actual=torch.cat([(p.detach()-before[name]).flatten() for name,p in selected])
        reference=torch.cat([(dict(other.named_parameters())[name].detach()-before[name]).flatten() for name,_ in selected])
        assert reference.norm()>0
        torch.testing.assert_close(actual.norm()/reference.norm(),torch.tensor(.25),rtol=.03,atol=.001)


def test_sft_budget_composes_with_warmup_and_restores_nominal_groups(monkeypatch):
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.lr_anneal import LrWarmup
    model,optimizer=build();graphs=examples()[:2]
    nominal=[(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups]
    warmup=LrWarmup(optimizer,10); assert warmup.scale(4)==.5
    warmed=[(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups]
    original=trainer.packed_graph_loss;observed=[]
    def inspect(*args,**kwargs):
        observed.append([(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups])
        return original(*args,**kwargs)
    monkeypatch.setattr(trainer,'packed_graph_loss',inspect)
    trainer.native_training_update(model,optimizer,graphs,checkpoint_layers=True,chunk_tokens=3,sft_update_reference=8)
    assert all(abs(actual[0]-base[0]*.25)<1e-12 for row in observed for actual,base in zip(row,warmed))
    assert [(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups]==warmed
    warmup.restore()
    assert [(group['lr'],group.get('qat_latent_lr')) for group in optimizer.param_groups]==nominal


def test_warmup_from_a_declared_rate_transition():
    from natlang_neuralese.train.lr_anneal import LrWarmup
    _,optimizer=build()
    nominal=[group['lr'] for group in optimizer.param_groups]
    warmup=LrWarmup(optimizer,64,start=67)
    assert warmup.factor(67)==1/64 and warmup.factor(98)==.5 and warmup.factor(130)==1. and warmup.factor(500)==1.
    assert warmup.factor(10)==0.
    assert warmup.scale(98)==.5
    assert [group['lr'] for group in optimizer.param_groups]==[rate*.5 for rate in nominal]
    warmup.restore()
    assert [group['lr'] for group in optimizer.param_groups]==nominal
    assert LrWarmup(optimizer,10).factor(4)==.5


def test_sft_budget_control_is_recorded_for_mapping_details(monkeypatch):
    from natlang_neuralese.train import native_state_sft as trainer
    model,optimizer=build();graphs=examples()[:1]
    def loss(model,micro,**kwargs):
        return sum(p.float().square().sum() for p in model.parameters() if p.requires_grad), {'objective':'fixture'}
    monkeypatch.setattr(trainer,'packed_graph_loss',loss)
    _,details=trainer.native_training_update(model,optimizer,graphs,checkpoint_layers=False,sft_update_reference=4)
    assert details['sft_update_budget']['scored_graphs']==1
    assert details['sft_update_budget']['multiplier']==.25


def test_merge_update_details_combines_engine_guard_summaries():
    """Each micro-batch reports its own engine-mismatch summary (bf95-lowlr w2 round 0 failed on 'disagree')."""
    from natlang_neuralese.train.native_state_sft import merge_update_details
    guard = lambda episodes, rejected, tokens, max_abs, mean_abs, k3: {
        'episodes': episodes, 'rejected': rejected, 'tokens': tokens, 'max_abs': max_abs, 'mean_abs': mean_abs,
        'max_mean_k3': k3}
    first = {'sampled_tokens': 10, 'episodes': 1, 'objective': 'grpo', 'engine_guard': guard(1, 0, 10, .2, .01, .001)}
    second = {'sampled_tokens': 30, 'episodes': 1, 'objective': 'grpo', 'engine_guard': guard(1, 1, 30, .9, .05, .02)}
    native = {'sampled_tokens': 5, 'episodes': 1, 'objective': 'grpo'}
    merged = merge_update_details([first, second, native], [1, 1, 1])
    assert merged['engine_guard'] == {'episodes': 2, 'rejected': 1, 'tokens': 40, 'max_abs': .9,
                                      'mean_abs': (.01 * 10 + .05 * 30) / 40, 'max_mean_k3': .02}
    assert merged['sampled_tokens'] == 45 and merged['objective'] == 'grpo'


def test_merge_update_details_accepts_a_micro_batch_without_conditional_terms():
    """A micro-batch whose episode the engine guard rejected carries only counts (bf95-lowlr w2 round 1 update 5
    failed on 'different detail fields'); its zero JOLT terms keep their optimizer share."""
    from natlang_neuralese.train.native_state_sft import merge_update_details
    jolt = {'student_outcome': .4, 'student_opsd': .2, 'teacher_outcome': .1, 'teacher_kl': .3,
            'unprivileged_episodes': 1, 'privileged_episodes': 1, 'positions': {'student': 10, 'teacher': 10},
            'guidance': {'blocks': 1, 'writer_blocks': 0, 'text_fallbacks': 1, 'verbatim_reference': 0},
            'mean_kl': {'student': .05, 'teacher': .07}}
    guard = {'episodes': 1, 'rejected': 0, 'tokens': 10, 'max_abs': .1, 'mean_abs': .01, 'max_mean_k3': .001}
    full = {'sampled_tokens': 10, 'sampled_decisions': 0, 'episodes': 1, 'objective': 'grpo',
            'policy_loss': 'jolt-plus/1', 'score_centering': {'positions': 10, 'rho_max': 1.5, 'w_y_zero_fraction': .1},
            'jolt': jolt, 'engine_guard': guard}
    rejected = {'sampled_tokens': 0, 'sampled_decisions': 0, 'episodes': 0, 'objective': 'grpo',
                'engine_guard': {**guard, 'rejected': 1, 'max_abs': 3.}}
    merged = merge_update_details([full, rejected], [1, 1])
    assert merged['policy_loss'] == 'jolt-plus/1' and merged['score_centering'] == full['score_centering']
    assert merged['jolt']['student_outcome'] == .4 * .5 and merged['jolt']['positions'] == {'student': 10, 'teacher': 10}
    assert merged['engine_guard']['rejected'] == 1 and merged['episodes'] == 1


def test_policy_update_is_one_action_mean_whatever_the_micro_batch_partition(monkeypatch):
    """Review F1 (2026-11-09): micro-batch action means weighted by graph count reversed this update's sign."""
    from natlang_neuralese.train import native_state_sft as trainer
    model,optimizer=build();graphs=examples()[:2]
    parameter=next(p for p in model.parameters() if p.requires_grad)
    value={graphs[0].id:(1.,1),graphs[1].id:(-.02,100)}  # (mean action loss, planned actions) per graph

    class Objective:
        def planned_actions(self,micro):return sum(value[example.id][1] for example in micro)

    def packed(model_,micro,**kwargs):
        mean=sum(value[e.id][0]*value[e.id][1] for e in micro)/sum(value[e.id][1] for e in micro)
        return parameter.sum()*0.+mean,{'episodes':len(micro)}
    monkeypatch.setattr(trainer,'packed_graph_loss',packed)
    monkeypatch.setattr(trainer,'merge_update_details',lambda parts,scored:parts[-1])
    monkeypatch.setattr('natlang_neuralese.rl.native_replay.efficiency_only_update_scale',lambda *a:1.)
    split,_=trainer.native_training_update(model,optimizer,graphs[1:],accumulate=[graphs[:1]],objective=Objective(),
                                           checkpoint_layers=True,chunk_tokens=3)
    whole,_=trainer.native_training_update(model,optimizer,graphs,objective=Objective(),checkpoint_layers=True,chunk_tokens=3)
    assert abs(float(split)-(1.-2.)/101)<1e-7 and abs(float(whole)-(1.-2.)/101)<1e-7
