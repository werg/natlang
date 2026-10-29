"""MuonSR Kahan rounding: BF16 matrices follow the FP32 trajectory instead of a stochastic-rounding random walk."""
import copy

import torch

from natlang_neuralese.train.optim import MuonSR


def run(dtype, rounding, steps=300, lr=2e-5):
    torch.manual_seed(5)
    weight = (torch.randn(64, 128) * .02).to(torch.bfloat16).float()  # one representable start for every run
    parameter = torch.nn.Parameter(weight.to(dtype))
    optimizer = MuonSR([{'params': [parameter], 'lr': lr, 'weight_decay': .1, 'name': 'w'}], lr=lr,
                       rounding=rounding)
    generator = torch.Generator().manual_seed(9)
    for _ in range(steps):
        # Identical BF16-representable gradients for every precision, so only the weight writes differ.
        parameter.grad = torch.randn(64, 128, generator=generator).to(torch.bfloat16).to(dtype)
        optimizer.step()
    return parameter.detach().float(), optimizer


def test_kahan_tracks_the_fp32_trajectory_where_stochastic_rounding_drifts():
    reference, _ = run(torch.float32, 'stochastic')
    kahan, optimizer = run(torch.bfloat16, 'kahan')
    stochastic, _ = run(torch.bfloat16, 'stochastic')
    # The represented value of a Kahan latent is its BF16 weight plus the carried residual.
    state = optimizer.state[optimizer.param_groups[0]['params'][0]]
    kahan_error = (kahan + state['compensation'].float() - reference).square().mean().sqrt()
    stochastic_error = (stochastic - reference).square().mean().sqrt()
    assert kahan_error < .1 * stochastic_error, (float(kahan_error), float(stochastic_error))
    # The BF16 weight itself stays within rounding of the reference instead of drifting.
    assert (kahan - reference).abs().max() < 2 ** -7 * reference.abs().max()
    assert state['compensation'].dtype == torch.bfloat16 and state['compensation'].abs().max() > 0


def test_kahan_residual_survives_a_state_round_trip_and_rounding_is_validated():
    _, optimizer = run(torch.bfloat16, 'kahan', steps=5)
    saved = copy.deepcopy(optimizer.state_dict())
    parameter = torch.nn.Parameter(optimizer.param_groups[0]['params'][0].detach().clone())
    restored = MuonSR([{'params': [parameter], 'lr': 2e-5, 'weight_decay': .1, 'name': 'w'}], lr=2e-5, rounding='kahan')
    restored.load_state_dict(saved)
    original = optimizer.state[optimizer.param_groups[0]['params'][0]]['compensation']
    assert torch.equal(restored.state[parameter]['compensation'], original)
    assert restored.state[parameter]['compensation'].dtype == torch.bfloat16
    try:
        MuonSR([{'params': [parameter], 'lr': 1., 'name': 'w'}], lr=1., rounding='floor')
    except ValueError as error:
        assert 'rounding' in str(error)
    else:
        raise AssertionError('unknown rounding accepted')

def test_outer_fragment_install_preserves_kahan_residual_and_rounding_across_restart():
    from types import SimpleNamespace
    from natlang_neuralese.train.diloco.protocol import Fragment,Slice,TensorSurface,install_fragment
    from natlang_neuralese.train.native_diloco import compensated_install_hook
    p=torch.nn.Parameter(torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16))
    latent=MuonSR([{'params':[p],'lr':1e-4,'name':'weight'}],lr=1e-4,rounding='kahan')
    latent.set_state_storage('cpu')
    latent.state[p]['compensation']=torch.tensor([[.001,.002],[.003,.004]],dtype=torch.bfloat16)
    latent.state[p]['momentum']=torch.ones_like(p)
    owner=SimpleNamespace(latent=latent);named=[('weight',p)]
    surface=TensorSurface(dict(named),install_hook=compensated_install_hook(named,owner))
    fragment=Fragment(0,(Slice('weight',1,2),));values=torch.tensor([2.003,3.006])
    old_p=p.detach().clone();old_c=latent.state[p]['compensation'].clone()
    install_fragment(surface,fragment,values)
    comp=latent.state[p]['compensation']
    assert torch.equal(p.flatten()[[0,3]],old_p.flatten()[[0,3]])
    assert torch.equal(comp.flatten()[[0,3]],old_c.flatten()[[0,3]])
    expected=(old_c.flatten()[1:3].float()+values-p.detach().flatten()[1:3].float()).to(torch.bfloat16)
    assert torch.equal(comp.flatten()[1:3],expected)
    assert torch.equal(latent.state[p]['momentum'],torch.ones_like(p))
    saved=latent.state_dict();p2=torch.nn.Parameter(p.detach().clone())
    resumed=MuonSR([{'params':[p2],'lr':1e-4,'name':'weight'}],lr=1e-4,rounding='kahan')
    resumed.load_state_dict(saved);resumed.set_state_storage('cpu')
    assert torch.equal(resumed.state[p2]['compensation'],comp)
    s2=TensorSurface({'weight':p2},install_hook=compensated_install_hook([('weight',p2)],SimpleNamespace(latent=resumed)))
    another=Fragment(1,(Slice('weight',0,1),));v2=torch.tensor([1.003])
    install_fragment(surface,another,v2);install_fragment(s2,another,v2)
    assert torch.equal(p,p2) and torch.equal(latent.state[p]['compensation'],resumed.state[p2]['compensation'])


def test_outer_fragment_install_refuses_bad_compensation_before_any_write():
    import pytest
    from types import SimpleNamespace
    from natlang_neuralese.train.diloco.protocol import Fragment,Slice,TensorSurface,install_fragment
    from natlang_neuralese.train.native_diloco import compensated_install_hook
    first=torch.nn.Parameter(torch.ones((2,2),dtype=torch.bfloat16))
    second=torch.nn.Parameter(torch.ones((2,2),dtype=torch.bfloat16))
    latent=MuonSR([{'params':[first,second],'lr':1e-4}],lr=1e-4,rounding='kahan')
    latent.state[second]['compensation']=torch.zeros(1,dtype=torch.bfloat16)
    named=[('first',first),('second',second)];surface=TensorSurface(dict(named),install_hook=compensated_install_hook(named,SimpleNamespace(latent=latent)))
    fragment=Fragment(0,(Slice('first',0,2),Slice('second',0,2)))
    with pytest.raises(ValueError,match='shape or dtype'):
        install_fragment(surface,fragment,torch.tensor([1.003,1.004,1.005,1.006]))
    assert torch.equal(first,torch.ones_like(first)) and torch.equal(second,torch.ones_like(second))
    assert 'compensation' not in latent.state[first]

def test_actual_trainer_rounding_switch_clears_stale_compensation_and_same_mode_retains(tmp_path,monkeypatch):
    import json
    from test_native_source_fetch_recovery import source
    from natlang_neuralese.train import native_state_sft as trainer
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';paths=source(base,'lfm');row=json.loads(paths['train'].read_text())
    paths['train'].write_text(''.join(json.dumps({**row,'id':str(i),'source_groups':[str(i)]})+'\n' for i in range(5)))
    out=tmp_path/'run';args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu',
        '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),'--epochs','1',
        '--batch-graphs','1','--accumulate-batches','1','--eval-every','1','--checkpoint-every','1',
        '--checkpoint-seconds','0','--seed','23','--refresh-after-updates','1','--out',str(out)]
    def run(mode):
        assert trainer.main([*args,'--muon-rounding',mode])==0
        return torch.load(out/'checkpoint.pt',weights_only=False)['training']
    first=run('kahan');states=first['optimizer']['muonsr']['state']
    assert any('compensation' in state for state in states.values())
    second=run('stochastic');states=second['optimizer']['muonsr']['state']
    assert all('compensation' not in state for state in states.values())
    assert second['options']['muon_rounding_history'][-1]['cleared_compensation_buffers']>0
    third=run('kahan');states=third['optimizer']['muonsr']['state']
    assert any('compensation' in state for state in states.values())
    assert third['options']['muon_rounding_history'][-1]['cleared_compensation_buffers']==0
    fourth=run('kahan');assert len(fourth['options']['muon_rounding_history'])==2
    assert fourth['global_step']==4
