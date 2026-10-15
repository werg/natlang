"""Real named backbone updates retain identity, optimizer history and RNG state."""
import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.model.native_model import load_native_model, model_state, CHECKPOINT_SCHEMA
from natlang_neuralese.model.native_artifact import decode_native_tensor_artifact
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.native_grad import native_graph_gradient
from natlang_neuralese.serve.native_optim import native_optimizer_step
from natlang_neuralese.train.loop import capture_training_rng_state


def source(path,family):
    path.mkdir()
    if family == 'lfm':
        from test_native_serve import _tiny_source
        _tiny_source(path)
        return torch.float32
    from test_mellum_port import tiny_mellum
    from transformers import PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    tiny_mellum(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f't{i}':i for i in range(128)},unk_token='t0')),unk_token='t0',eos_token='t2')
    tokenizer.chat_template = "{% for message in messages %}{{ message.get('reasoning_content','') }}{{ message['content'] }}{% endfor %}assistant:"
    tokenizer.save_pretrained(path)
    return torch.bfloat16


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('algorithm',['adam','sgd'])
def test_actual_backbone_update_reload_and_sparse_target_resume(tmp_path,family,algorithm):
    dtype = source(tmp_path/'base',family)
    model,_,roles,config = load_native_model(tmp_path/'base',dtype=dtype,rank=4)
    rng = capture_training_rng_state('cpu')
    parent = tmp_path/'native.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),'training':{'rng':rng}},parent)
    engine = load_native_engine(str(parent),dtype=dtype)
    backbone = next(name for name,value in engine.shared_model.named_parameters()
                    if name.startswith('port.hf.') and name.endswith('q_proj.weight'))
    names = [backbone,'adapter.up.weight']
    reference = {name:torch.nn.Parameter(dict(engine.shared_model.named_parameters())[name].detach().clone()) for name in names}
    options = {'name':algorithm,'learning_rate':.01,**({'momentum':.8} if algorithm=='sgd' else {})}
    reference_optimizer = (torch.optim.Adam(reference.values(),lr=.01) if algorithm=='adam' else
                           torch.optim.SGD(reference.values(),lr=.01,momentum=.8))
    row = {'schema':'natlang.neuralese-native-graph/1','id':'real-service-update','split':'train',
        'source_groups':['service-fixture'],'tokenizer_fingerprint':config['tokenizer_fingerprint'],
        'nodes':[{'name':'answer','token_ids':[4,roles.produce_open,5,6,roles.produce_close,7],
                  'target_weights':[0,1,1,1,1,1],'producing':[[2,4]]}]}
    state_id,history = None,{}
    for step,target in enumerate([backbone,'adapter.up.weight',backbone],1):
        engine.store.put_native_tensor_artifact(history[state_id]) if state_id else None
        result = native_graph_gradient(engine,{'schema':'natlang.native-gradient-request/1','graph':row,
            'native_identity':engine.native_identity,'target':{'kind':'model-parameters','names':[target]}},None)
        gradient = engine.store.get_native_tensor_artifact(result['gradient']['id']).artifact
        assert gradient.tensors[target].norm()>0
        reference_optimizer.zero_grad(set_to_none=True)
        reference[target].grad = gradient.tensors[target].to(reference[target]).clone()
        reference_optimizer.step()
        old_values = {name:value.detach().clone() for name,value in engine.shared_model.named_parameters()}
        updated = native_optimizer_step(engine,{'schema':'natlang.native-optimizer-request/1',
            'native_identity':engine.native_identity,'gradient_id':result['gradient']['id'],
            'optimizer_state_id':state_id,'optimizer':options},None)
        assert updated['step']==step
        assert all(torch.equal(old_values[name],value) for name,value in engine.shared_model.named_parameters())
        state_id = updated['optimizer_state_id']
        history[state_id] = engine.store.get_native_tensor_artifact(state_id).artifact
        next_engine = load_native_engine(updated['checkpoint']['path'],dtype=dtype)
        assert next_engine.native_identity == updated['native_identity']
        actual = dict(next_engine.shared_model.named_parameters())
        assert all(torch.equal(actual[name],value) for name,value in reference.items())
        assert any(not torch.equal(actual[name],old_values[name]) for name in names)
        assert torch.equal(next_engine.native_training_state['rng']['torch_rng'],rng['torch_rng'])
        assert next_engine.native_training_state['rng']['python_rng']==rng['python_rng']
        assert next_engine.native_training_state['service_optimizer_step']==step
        manifest = json.loads(Path(updated['checkpoint']['path']).read_text())
        assert Path(updated['checkpoint']['path']).parent == tmp_path/'native-updates'
        assert manifest['parameter_scope']=='model-parameters'
        assert manifest['updated_parameter_names']==[target]
        assert Path(manifest['base_checkpoint']['path']).is_file()
        engine.store.close()
        engine = next_engine
    from natlang_neuralese.model.native_model import load_native_checkpoint
    _,_,_,_,restored_state = load_native_checkpoint(updated['checkpoint']['path'],dtype=dtype)
    assert restored_state['base_state']['schema']==CHECKPOINT_SCHEMA
    assert restored_state['optimizer']['step']==3
    state = history[state_id].tensors
    if algorithm=='adam':
        assert state['step.'+backbone].item()==2 and state['step.adapter.up.weight'].item()==1
    else:
        assert 'momentum.'+backbone in state and 'momentum.adapter.up.weight' in state
    engine.store.close()


@pytest.mark.parametrize('extra',[{'order':2},{'create_graph':True},{'derived':[{'kind':'grad'}]}])
def test_native_service_does_not_silently_downgrade_requested_second_order(extra):
    from types import SimpleNamespace
    from natlang_neuralese.serve.chat import RequestError
    with pytest.raises(RequestError,match='first-order'):
        native_graph_gradient(SimpleNamespace(),{'schema':'natlang.native-gradient-request/1',**extra},None)


def test_native_service_does_not_ignore_requested_adapter():
    from types import SimpleNamespace
    from natlang_neuralese.serve.chat import RequestError
    with pytest.raises(RequestError,match='identity-bound checkpoint'):
        native_graph_gradient(SimpleNamespace(),{'schema':'natlang.native-gradient-request/1','adapters':[{'id':'unsupported'}]},None)
