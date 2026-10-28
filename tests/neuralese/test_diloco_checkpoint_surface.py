import json
import pytest
import torch


def test_mmap_coordinator_surface_matches_actual_trainable_parameters(tmp_path):
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer, native_diloco
    from natlang_neuralese.train.diloco.protocol import parameter_identity, plan_fragments
    base=tmp_path/'base';tiny_source(base);_,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint'];train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text(json.dumps(dict(graph('train'),tokenizer_fingerprint=fp))+'\n')
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    out=tmp_path/'run';assert trainer.main(['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(train),'--held-native-graphs',str(held),'--epochs','1','--out',str(out)])==0
    checkpoint=out/'checkpoint.pt';model,named=native_diloco._trainable(checkpoint,base);actual=dict(named)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);state['model_config']['architecture']='lfm3'
    fixture=tmp_path/'mapped.pt';torch.save(state,fixture)
    mapped=native_diloco._checkpoint_surface(fixture,base)
    assert set(mapped)==set(actual)
    for name,tensor in mapped.items():assert torch.equal(tensor,actual[name].detach())
    assert parameter_identity(mapped)==parameter_identity(actual)
    assert plan_fragments(mapped,target_bytes=1024)==plan_fragments(actual,target_bytes=1024)
    state['training']['optimizer']['param_names']['adamw'].append('backbone.missing')
    torch.save(state,fixture)
    with pytest.raises(ValueError,match='missing floating coordinator parameter'):
        native_diloco._checkpoint_surface(fixture,base)
