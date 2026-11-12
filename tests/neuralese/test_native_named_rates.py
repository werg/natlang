"""Exact named native optimizer slots survive adaptive rate groups and resume."""
import copy
import pytest
import torch
from natlang_neuralese.train.optim import PortMuonAdamW
from natlang_neuralese.train.optim_restore import set_learning_rates,OptimizerRestoreError


def mixed(order=('row_a','matrix','row_b')):
    params={name:torch.nn.Parameter(torch.ones((16,16) if name=='matrix' else (4,))) for name in order}
    optimizer=PortMuonAdamW([(name,params[name]) for name in order],lr=.001,vocab_size=999)
    return params,optimizer


def named_history(optimizer):
    return {optimizer._parameter_names[id(p)]:(group,child.state[p])
        for child in (optimizer.muon,optimizer.auxiliary,optimizer.latent) if child is not None
        for group in child.param_groups for p in group['params']}


def test_raw_port_named_slots_survive_schema_child_order_split_reorder_and_rates():
    params,opt=mixed()
    for index,p in enumerate(params.values()):p.grad=torch.full_like(p,index+1.)
    opt.step();opt.zero_grad()
    set_learning_rates(opt,params,{'row_b':.003})
    saved=copy.deepcopy(opt.state_dict())
    assert saved['schema'][0]['name']=='row_a'
    assert saved['param_names']['muon']==['matrix']
    assert saved['param_names']['adamw']==['row_a','row_b']
    changed,new=mixed(('row_b','matrix','row_a'))
    new.load_state_dict(saved)
    actual=named_history(new);expected=named_history(opt)
    for name in params:
        assert actual[name][0]['lr']==expected[name][0]['lr']
        for key,value in expected[name][1].items():
            if torch.is_tensor(value):assert torch.equal(actual[name][1][key],value),(name,key)
            else:assert actual[name][1][key]==value
    assert new.state_dict()['param_names']['adamw']==['row_b','row_a']
    for name in params:changed[name].data.copy_(params[name])
    for name in params:
        params[name].grad=torch.full_like(params[name],.4)
        changed[name].grad=torch.full_like(changed[name],.4)
    opt.step();new.step()
    for name in params:assert torch.equal(params[name],changed[name]),name


def test_legacy_unsplit_schema_is_recovered_but_ambiguous_split_is_rejected():
    params,opt=mixed();legacy=opt.state_dict();legacy.pop('param_names')
    _,new=mixed(('row_b','matrix','row_a'));new.load_state_dict(legacy)
    set_learning_rates(opt,params,{'row_b':.003})
    ambiguous=opt.state_dict();ambiguous.pop('param_names')
    for child in (ambiguous['muon'],ambiguous['adamw']):
        for group in child['param_groups']:group.pop('param_names',None)
    with pytest.raises(OptimizerRestoreError,match='historical split'):new.load_state_dict(ambiguous)


def test_latent_backward_hooks_follow_actual_parameter_after_group_reorder():
    def build():
        values=[torch.nn.Parameter(torch.arange(256).reshape(16,16).float()/1000+i) for i in (0,1)]
        optimizer=PortMuonAdamW([],lr=.001,vocab_size=999,
            latent=[(name,p,.001,1) for name,p in zip(('a','b'),values)],latent_optimizer='muonsr')
        return values,optimizer
    actual,opt=build();expected,reference=build()
    opt.latent.step_in_backward(gated=True)
    opt.latent.param_groups.reverse();opt.param_groups=opt._groups()
    set_learning_rates(opt,dict(zip(('a','b'),actual)),{'a':.002,'b':.005})
    set_learning_rates(reference,dict(zip(('a','b'),expected)),{'a':.002,'b':.005})
    with opt.latent.in_backward():(actual[0].sum()+2*actual[1].sum()).backward()
    expected[0].grad=torch.ones_like(expected[0]);expected[1].grad=torch.full_like(expected[1],2.)
    reference.step()
    for a,b in zip(actual,expected):assert torch.equal(a,b)
    for group in opt.latent.param_groups:assert group['qat_latent_lr']==group['lr']


def test_serialized_names_cannot_alias_two_parameters_to_one_saved_slot():
    _,optimizer=mixed();saved=optimizer.state_dict()
    group=saved['adamw']['param_groups'][0]
    assert len(group['params'])==2
    group['params'][1]=group['params'][0]
    with pytest.raises(OptimizerRestoreError,match='slot-name inventory'):
        optimizer.load_state_dict(saved)
