"""Exact logical coefficient identity survives numeric convergence."""
import torch
from test_native_serving_adapters import engine_fixture

def test_distinct_coefficient_histories_do_not_coalesce_when_values_converge(tmp_path):
    from dataclasses import replace
    from natlang_neuralese.model.native_adapters import create_adapter,target_parameters,optimize_adapter
    from natlang_neuralese.model.native_artifact import NativeTensorArtifact
    engine=engine_fixture(tmp_path,'lfm')
    name=next(name for name in target_parameters(engine.shared_model) if name.endswith('v_proj.weight'))
    seed=create_adapter(engine,{'kind':'xs','rank':2,'parameter_names':[name]})
    parents=[engine.store.put_native_tensor_artifact(replace(seed.artifact,tensors={name:torch.full_like(seed.artifact.tensors[name],value)})) for value in (1.,2.)]
    options={'name':'sgd','learning_rate':1.,'momentum':.5}
    updates=[]
    for parent,value in zip(parents,(1.,2.)):
        gradient=NativeTensorArtifact('adapter-gradient',engine.native_identity,{name:torch.full_like(parent.artifact.tensors[name],value)},parent.id)
        updates.append(optimize_adapter(engine,gradient,options,None))
    first,second=[engine.store.get_native_tensor_artifact(update['adapter']['id']).artifact for update in updates]
    assert torch.equal(first.tensors[name],second.tensors[name]) and first.tensors[name].count_nonzero()==0
    assert updates[0]['adapter']['id']!=updates[1]['adapter']['id'],'equal values must not silently alias independent moment histories'
    assert first.provenance['coefficient_lineage']==parents[0].id
    assert second.provenance['coefficient_lineage']==parents[1].id
    for update,expected in zip(updates,(-.5,-1.)):
        ident=update['adapter']['id']
        gradient=NativeTensorArtifact('adapter-gradient',engine.native_identity,{name:torch.zeros_like(first.tensors[name])},ident)
        result=optimize_adapter(engine,gradient,options,update['optimizer_state_id'])
        artifact=engine.store.get_native_tensor_artifact(result['adapter']['id']).artifact
        assert torch.equal(artifact.tensors[name],torch.full_like(artifact.tensors[name],expected))
    engine.store.close()
