import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.data.native_graph import graph_loss
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.eval import native_memory
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_packed_capacity import geometry_for_source, index_source, maximum_pending_groups, qualify_pending
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from test_native_state_sft import graph
from test_shared_state import tiny


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_actual_packed_maximum_runs_full_backward_and_matches_graph_mean(tmp_path, monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path=tmp_path/'train.jsonl'
    path.write_text(''.join(json.dumps(graph(name,weights))+'\n' for name,weights in [
        ('a',[0,1,.5]),('b',[0,.25,1]),('c',[0,1,1])]))
    source=NativeGraphSource(path,'tokenizer-sha',required_split='train',role_ids=(58,59,60,61))
    model=SharedStateModel(tiny('lfm').to(device=device),(58,59,60,61),rank=4).to(device)
    reference=copy.deepcopy(model)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    reference_optimizer,_=make_native_optimizer(reference,lr=3e-6,in_backward=True)
    examples=list(source.iter_indices([0,1]))
    loss=torch.stack([graph_loss(reference,example,checkpoint_layers=True,chunk_tokens=3)[0]
                      for example in examples]).mean()
    loss.backward()
    expected={name:p.grad.detach().clone() for name,p in reference_optimizer._native_named_parameters if p.grad is not None}
    observed={}; real=native_memory.qualify_capacity
    def measure(*args,**kwargs):
        return real(*args,**kwargs,gradient_observer=lambda name,g:observed.__setitem__(name,g.detach().clone()))
    monkeypatch.setattr(native_memory,'qualify_capacity',measure)
    args=SimpleNamespace(out=tmp_path/'out',checkpoint_layers=True,chunk_tokens=3)
    _,initial,_=geometry_for_source(source,args.out,allow_build=True)
    before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    result=qualify_pending(model,optimizer,source,[(0,[0,1],False),(0,[2],True)],args,
        backend=None,checkpoint_sha256='a'*64,global_step=7,prior_geometry=initial)
    proof=json.loads(open(result['proof']['path']).read())
    assert proof['passed'] and proof['optimizer_updates']==proof['earned_exposure']==0
    assert proof['groups'][0][0]['id']=='a' and proof['groups'][0][1]['id']=='b'
    case=proof['capacity']['cases'][0]
    assert case['example_ids']==['a','b'] and case['forward_tokens']==6 and case['kv_slots']==6
    assert case['forward_tokens']>max(example.plan.token_count for example in examples)
    assert case['loss']==pytest.approx(float(loss.detach()),rel=1e-5,abs=1e-6)
    assert observed.keys()==expected.keys()
    for name in observed:torch.testing.assert_close(observed[name],expected[name],rtol=2e-4,atol=2e-6)
    for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),rng)
    if cuda_rng is not None: assert torch.equal(torch.cuda.get_rng_state(),cuda_rng)
    _,_,reused=geometry_for_source(source,args.out,prior=result['geometry'])
    assert reused=={'parsed_rows':0,'indexed_rows':0,'reused_prefix_rows':3}
    # A resident update can be newer than the last durable parent. Qualify the
    # actual live weights and mark that boundary honestly.
    with torch.no_grad():model.port.embedding_weight[1].add_(.1)
    live_before=copy.deepcopy(model.state_dict());observed.clear()
    live=qualify_pending(model,optimizer,source,[(0,[0,1],False)],args,
        backend=None,checkpoint_sha256='a'*64,global_step=8,prior_geometry=result['geometry'],live_uncommitted=True)
    live_proof=json.loads(open(live['proof']['path']).read())
    assert live_proof['model_boundary']=='live-uncommitted' and live_proof['global_step']==8
    assert live_proof['capacity']['cases'][0]['loss'] != case['loss']
    for key,value in model.state_dict().items():torch.testing.assert_close(value,live_before[key],rtol=0,atol=0)


def test_cold_index_geometry_and_verified_append_prefix(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path=tmp_path/'old.jsonl';path.write_text(json.dumps(graph('a'))+'\n'+json.dumps(graph('b'))+'\n')
    source,indexed=index_source(path,'tokenizer-sha',(58,59,60,61))
    _,prior,stats=geometry_for_source(source,tmp_path,allow_build=True,indexed_geometry=indexed)
    assert stats=={'parsed_rows':0,'indexed_rows':2,'reused_prefix_rows':0}
    other_roles=NativeGraphSource(path,'tokenizer-sha',required_split='train',role_ids=(1,2,3,4))
    with pytest.raises(ValueError,match='parser or source identity'):
        geometry_for_source(other_roles,tmp_path,prior=prior)
    expanded=tmp_path/'new.jsonl';expanded.write_text(path.read_text()+json.dumps(graph('c'))+'\n')
    next_source=NativeGraphSource(expanded,'tokenizer-sha',required_split='train',role_ids=(58,59,60,61))
    with pytest.raises(ValueError,match='independently prepared'):
        geometry_for_source(next_source,tmp_path,prior=prior)
    _,_,stats=geometry_for_source(next_source,tmp_path,prior=prior,allow_build=True)
    assert stats=={'parsed_rows':1,'indexed_rows':0,'reused_prefix_rows':2}
    changed=tmp_path/'changed.jsonl';changed.write_text(json.dumps(graph('b'))+'\n'+json.dumps(graph('a'))+'\n')
    changed_source=NativeGraphSource(changed,'tokenizer-sha',required_split='train',role_ids=(58,59,60,61))
    with pytest.raises(ValueError,match='unchanged actual source prefix'):
        geometry_for_source(changed_source,tmp_path,prior=prior,allow_build=True)
    Path(prior['path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        geometry_for_source(source,tmp_path,prior=prior)


def test_actual_pending_maxima_include_graph_count_and_distinct_kv_peak():
    entries=[{'forward_tokens':v} for v in (9,9,5,5,2,2,2,2)]
    kv=[9,9,15,15,2,2,2,2]
    batches=[(0,[0,1],False),(0,[2,3],False),(0,[4,5,6,7],True)]
    assert maximum_pending_groups(entries,kv,batches)==[(0,1),(2,3),(4,5,6,7)]


def test_geometry_builder_preserves_verified_prefix_on_interruption(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'indexes'))
    path=tmp_path/'train.jsonl';path.write_text(''.join(json.dumps(graph(name))+'\n'for name in ('a','b','c','d')))
    source=NativeGraphSource(path,'tokenizer-sha',required_split='train',role_ids=(58,59,60,61))
    original=source.iter_indices
    def interrupted(indices):
        for count,example in enumerate(original(indices)):
            if count==3:raise InterruptedError('simulated builder stop')
            yield example
    monkeypatch.setattr(source,'iter_indices',interrupted)
    saved=[]
    with pytest.raises(InterruptedError,match='builder stop'):
        geometry_for_source(source,tmp_path,allow_build=True,checkpoint_rows=2,
            progress_callback=lambda ref,count,total:saved.append((ref,count,total)))
    assert [count for _,count,_ in saved]==[2,3]
    assert all(total==4 for _,_,total in saved)
    prior=saved[-1][0]
    assert len(json.loads(Path(prior['path']).read_text())['kv_slots'])==3
    parsed=[]
    def resumed(indices):
        parsed.extend(indices)
        yield from original(indices)
    monkeypatch.setattr(source,'iter_indices',resumed)
    _,_,stats=geometry_for_source(source,tmp_path,allow_build=True,prior=prior)
    assert parsed==[3] and stats['reused_prefix_rows']==3 and stats['parsed_rows']==1


def test_packed_proof_and_geometry_are_checkpoint_publication_dependencies(tmp_path):
    from scripts.coreweave.provider_dependencies import dependency_closure
    from natlang_neuralese.train.native_resident import binding
    source=tmp_path/'source.jsonl';source.write_text('{}\n')
    geometry=tmp_path/'geometry.json'
    geometry.write_text(json.dumps({'schema':'natlang.native-packed-geometry/1','source':binding(source)}))
    proof=tmp_path/'proof.json'
    proof.write_text(json.dumps({'schema':'natlang.native-sft-packed-capacity/1','geometry':binding(geometry),
        'source':binding(source),'capacity_identity':{},'runtime_code_sha256':'a'*64,
        'passed':True,'optimizer_updates':0,'earned_exposure':0}))
    payload={'training':{'options':{'sft_packed_capacity':{'geometry':binding(geometry),'proof':binding(proof)}}}}
    closure=dependency_closure({},namespace=str(tmp_path),_checkpoint_payload=payload)
    assert {item['path'] for item in closure['files']}=={str(source),str(geometry),str(proof)}
    from natlang_neuralese.train.recipe import stage_parameter_args
    args=stage_parameter_args({'sft_packed_geometry':binding(geometry)})
    assert args[0]=='--sft-packed-geometry' and json.loads(args[1])==binding(geometry)
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'trainer_parameters':{'sft_packed_geometry':binding(geometry)}}))
    config_closure=dependency_closure({'config':binding(config)},namespace=str(tmp_path))
    assert {item['path'] for item in config_closure['files']}=={str(config),str(source),str(geometry)}
    geometry.write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        dependency_closure({},namespace=str(tmp_path),_checkpoint_payload=payload)
