import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.eval import native_memory as memory
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.data.native_sources import NativeGraphSource
from test_native_memory_capacity import prepared_fixture, assert_exact
from test_native_state_sft import graph


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_seven_actual_cases_share_only_the_two_reference_backwards(tmp_path, monkeypatch, family):
    model, optimizer, train, roles, config, args = prepared_fixture(tmp_path, monkeypatch, family=family)
    source_path = tmp_path/'seven-capacity-cases.jsonl'
    source_path.write_text(''.join(json.dumps(graph('case-'+str(index)))+'\n' for index in range(7)))
    source = NativeGraphSource(source_path, 'tokenizer-sha', required_split='train', role_ids=roles.ids)
    template = json.loads(resident.bound(json.loads(args.memory_qualification)).read_text())
    template.update(source=resident.binding(source_path), indexes=resident._source_witness(source))
    requests = []
    for index,entry in enumerate(source.entries):
        request = {**template, 'cases':[{'index':index,'id':entry['id'],'binding':entry['binding']}]}
        path = tmp_path/('request-'+str(index)+'.json');path.write_text(json.dumps(request));requests.append(resident.binding(path))
    args.memory_qualification=json.dumps(requests)
    weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    calls=[];original=memory.qualify_capacity
    def observed(*a,**kw):
        calls.append('reference' if kw.get('gradient_observer') is not None else 'case')
        return original(*a,**kw)
    monkeypatch.setattr(memory,'qualify_capacity',observed)
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert result['passed'] and calls.count('reference')==2 and calls.count('case')==7
    assert len(result['checks'])==7 and result['checks'][0]['numerical_reference_execution']['measured']
    for check in result['checks'][1:]:
        execution=check['numerical_reference_execution']
        assert execution['measured'] is False and execution['same_owned_boundary']
        assert execution['original_request']==requests[0]
    assert [check['capacity']['cases'][0]['id'] for check in result['checks']]==['case-'+str(i)for i in range(7)]
    assert_exact(weights,model.state_dict());assert_exact(state,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())


class Backend:
    def __init__(self):
        self.scale=torch.ones(())
        self.row_counts={'calls':0}
        self.has_executed=False
    def configuration(self):return {'precision':'test'}
    def extra_state(self):return {'scale':self.scale}


def boundary_fixture():
    model=torch.nn.Linear(3,2)
    model.port=SimpleNamespace(ffn_chunk_tokens=0, checkpoint_layers=True)
    model.register_buffer('scale',torch.ones(1))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    parameter=next(model.parameters());optimizer.state[parameter]['moment']=torch.ones_like(parameter)
    backend=Backend()
    invocation={'parent':'a'*64,'runtime':'b'*64,'source':{'sha256':'c'*64},
        'reference':{'id':'reference','binding':'d'*64},'roles':[1,2,3,4],
        'identity':{'precision':'float32','checkpoint_layers':True,'ffn_chunk_tokens':0}}
    return model,optimizer,backend,invocation


@pytest.mark.parametrize('mutation', [
    'parameter','parameter_storage','requires_grad','buffer','optimizer_moment','optimizer_rate',
    'optimizer_parameters','module_mode','module_dispatch','port_execution','backend_state','backend_identity',
    'rng','parent','runtime','source','row','precision','checkpoint_flags','roles'])
def test_every_tracked_boundary_or_invocation_change_requires_new_reference(mutation):
    model,optimizer,backend,invocation=boundary_fixture();reuse=memory._NumericalReferenceReuse();calls=[]
    def measure():calls.append(1);return {'passed':True,'different_gradient_names':[]}
    def run():return reuse.measure(invocation,model=model,optimizer=optimizer,backend=backend,
        request={'path':'request','sha256':'e'*64},operation=measure)
    first,_=run();first['different_gradient_names'].append('caller mutation')
    same,evidence=run();assert same['different_gradient_names']==[] and not evidence['measured'] and len(calls)==1
    parameter=next(model.parameters())
    with torch.no_grad():
        if mutation=='parameter':parameter.add_(1)
        elif mutation=='parameter_storage':parameter.data=parameter.clone()
        elif mutation=='requires_grad':parameter.requires_grad_(False)
        elif mutation=='buffer':model.scale.add_(1)
        elif mutation=='optimizer_moment':optimizer.state[parameter]['moment'].add_(1)
        elif mutation=='optimizer_rate':optimizer.param_groups[0]['lr']=.02
        elif mutation=='optimizer_parameters':optimizer.param_groups[0]['params']=list(reversed(optimizer.param_groups[0]['params']))
        elif mutation=='module_mode':model.eval()
        elif mutation=='module_dispatch':model.forward=lambda value:torch.nn.functional.linear(value,model.weight,model.bias)+1
        elif mutation=='port_execution':model.port.ffn_chunk_tokens=4096
        elif mutation=='backend_state':backend.scale.add_(1)
        elif mutation=='backend_identity':backend=Backend()
        elif mutation=='rng':torch.rand(1)
        elif mutation in('parent','runtime'):invocation[mutation]='f'*64
        elif mutation=='source':invocation['source']['sha256']='f'*64
        elif mutation=='row':invocation['reference']['binding']='f'*64
        elif mutation=='precision':invocation['identity']['precision']='bf16'
        elif mutation=='checkpoint_flags':invocation['identity']['checkpoint_layers']=False
        elif mutation=='roles':invocation['roles']=[5,6,7,8]
    _,evidence=run();assert evidence['measured'] and len(calls)==2


def test_no_failed_or_mutating_measurement_is_reused():
    model,optimizer,backend,invocation=boundary_fixture();reuse=memory._NumericalReferenceReuse();calls=[]
    def failed():calls.append(1);return {'passed':False}
    for _ in range(2):
        _,evidence=reuse.measure(invocation,model=model,optimizer=optimizer,backend=backend,request={},operation=failed)
        assert evidence['measured']
    assert len(calls)==2 and not reuse.completed
    def mutating():
        with torch.no_grad():next(model.parameters()).add_(1)
        return {'passed':True}
    with pytest.raises(memory.CapacityStateChangedError,match='owned model/optimizer/backend/RNG boundary'):
        reuse.measure(invocation,model=model,optimizer=optimizer,backend=backend,request={},operation=mutating)
    assert not reuse.completed


def test_unknown_metadata_and_another_owned_call_do_not_authorize_reuse():
    model,optimizer,backend,invocation=boundary_fixture();calls=[]
    def measured():calls.append(1);return {'passed':True}
    for _ in range(2):
        memory._NumericalReferenceReuse().measure(invocation,model=model,optimizer=optimizer,
            backend=backend,request={},operation=measured)
    assert len(calls)==2
    invocation['unsupported']=object();reuse=memory._NumericalReferenceReuse()
    for _ in range(2):
        _,evidence=reuse.measure(invocation,model=model,optimizer=optimizer,backend=backend,request={},operation=measured)
        assert evidence=={'measured':True,'reuse_supported':False}
    assert len(calls)==4 and not reuse.completed


def test_restored_backend_buffer_versions_disable_reuse_without_a_new_failure():
    model,optimizer,backend,invocation=boundary_fixture()
    backend.scale=model.scale
    before=backend.scale.clone();reuse=memory._NumericalReferenceReuse();calls=[]
    def restored():
        calls.append(1)
        backend.scale.add_(1);backend.scale.sub_(1)
        return {'passed':True}
    for _ in range(2):
        proof,evidence=reuse.measure(invocation,model=model,optimizer=optimizer,
            backend=backend,request={},operation=restored)
        assert proof['passed'] and evidence['measured'] and not evidence['reuse_supported']
    assert len(calls)==2 and not reuse.completed and torch.equal(before,backend.scale)


def test_new_nonparameter_execution_cache_must_settle_before_reuse():
    model,optimizer,backend,invocation=boundary_fixture();reuse=memory._NumericalReferenceReuse();calls=[]
    def measured():
        calls.append(1)
        if not hasattr(model,'execution_cache'):model.register_buffer('execution_cache',torch.ones(1))
        return {'passed':True}
    def run():return reuse.measure(invocation,model=model,optimizer=optimizer,
        backend=backend,request={},operation=measured)
    assert run()[1]['reuse_supported'] is False
    assert run()[1]['measured']
    assert run()[1]['measured'] is False
    assert len(calls)==2
