import copy,json
import pytest
from natlang_neuralese.train.recipe import (native_recipe_binding,native_recipe_provenance,
    preserve_recipe_external_provenance,external_source_provenances)


def provenance(path,sha):
    return {'schema':'natlang.external-source-publication-provenance/1',
            'closure':{'path':path,'sha256':sha},'custody':{'source':'exact'}}


def test_reconstruction_and_append_preserve_all_external_histories(tmp_path):
    first=provenance('/archive/first','a'*64);second=provenance('/archive/second','b'*64)
    recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps({'schema':'fixture'}))
    original={**native_recipe_binding(recipe,'c'*64),'external_provenance':first}
    reconstructed=native_recipe_binding(recipe,'c'*64,previous=original)
    assert external_source_provenances(reconstructed)==[first]
    recipe.write_text(json.dumps({'schema':'fixture','external_provenances':[first,second]}))
    appended=native_recipe_binding(recipe,'c'*64,previous=reconstructed)
    assert external_source_provenances(appended)==[first,second]
    saved=json.loads(json.dumps(appended));assert native_recipe_provenance(saved)==appended
    recreated=native_recipe_binding(recipe,'c'*64,previous=saved)
    assert external_source_provenances(recreated)==[first,second]
    with pytest.raises(ValueError,match='dropped or changed'):
        preserve_recipe_external_provenance(appended,{**native_recipe_binding(recipe,'c'*64),'external_provenances':[second]})
    altered=copy.deepcopy(first);altered['custody']['source']='changed'
    with pytest.raises(ValueError,match='dropped or changed'):
        preserve_recipe_external_provenance(appended,{**native_recipe_binding(recipe,'c'*64),'external_provenances':[altered,second]})
    with pytest.raises(ValueError,match='historical provenance changed'):
        external_source_provenances({**native_recipe_binding(recipe,'c'*64),'external_provenances':[first,altered]})


def test_actual_sft_resume_saves_reconstructed_external_provenance(tmp_path,monkeypatch):
    import torch
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base);_,_,_,config=load_native_model(base,dtype=torch.float32,rank=4);fp=config['tokenizer_fingerprint']
    old=tmp_path/'old.jsonl';new=tmp_path/'new.jsonl';held=tmp_path/'held.jsonl'
    rows=[dict(graph(name),tokenizer_fingerprint=fp) for name in ('a','b','c')]
    old.write_text(''.join(json.dumps(row)+'\n' for row in rows[:2]));new.write_text(''.join(json.dumps(row)+'\n' for row in rows));held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    first=provenance('/history/first','a'*64);rp=tmp_path/'recipe.json';rp.write_text(json.dumps({'schema':'fixture'}));binding={**native_recipe_binding(rp,'c'*64),'external_provenance':first}
    out=tmp_path/'run';args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(old),'--held-native-graphs',str(held),'--epochs','1','--eval-every','100','--held-eval-tokens','4','--out',str(out),'--recipe-binding',json.dumps(binding)]
    def stop(policy,step,**kwargs):policy.stop.requested=step==1;return False
    with monkeypatch.context() as patch:
        patch.setattr(trainer.CheckpointPolicy,'due',stop);assert trainer.main(args)==0
    parent=torch.load(out/'checkpoint.pt',weights_only=False)['training'];assert parent['global_step']==1
    reconstructed=native_recipe_binding(rp,'c'*64);next_args=list(args);next_args[next_args.index('--native-graphs')+1]=str(new);next_args[next_args.index('--recipe-binding')+1]=json.dumps(reconstructed);next_args+=['--continued-native-graphs',str(old)]
    update=trainer.native_training_update;observed=[]
    def check(model,optimizer,batch,**kwargs):
        if not observed:
            assert torch.equal(torch.get_rng_state(),parent['rng']['torch_rng']);assert optimizer.state_dict().keys()==parent['optimizer'].keys()
        observed.extend(item.id for item in batch);return update(model,optimizer,batch,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(trainer,'native_training_update',check);assert trainer.main(next_args)==0
    saved=torch.load(out/'checkpoint.pt',weights_only=False)['training'];assert external_source_provenances(saved['options']['recipe_binding'])==[first]
    assert observed and all(v==1 for v in saved['exposure_counts'].values()) and saved['global_step']>parent['global_step']
