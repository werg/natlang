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
    expected=[]
    for example in batch:
        value,_=packed_graph_loss(reference,[example],objective=objective,checkpoint_layers=True,chunk_tokens=8,collect_accuracy=False)
        expected.append(value)
    ((expected[0]+expected[1])/2).backward()
    reference_grad={n:p.grad.detach().clone() for n,p in reference.named_parameters() if p.grad is not None}
    optimizer,_=make_native_optimizer(model,lr=1e-3,in_backward=True)
    seen=stepped_gradients(model,optimizer)
    _,details=native_training_update(model,optimizer,batch[1:],accumulate=[batch[:1]],objective=objective,checkpoint_layers=True,chunk_tokens=8)
    assert details['episodes']==2 and details['jolt']['positions']['student']>0 and details['jolt']['positions']['teacher']>0
    assert details['jolt']['mean_kl']['student'] is not None and details['jolt']['mean_kl']['teacher'] is not None
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
