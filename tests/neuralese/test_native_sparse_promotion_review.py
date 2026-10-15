"""Back-review of sparse masters across a carried legacy CE schedule."""
import copy
import json
import torch
from natlang_neuralese.data.native_graph import SCHEMA
from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint,upgrade_reasoning_tokens,SHARED_CHECKPOINT_SCHEMA,model_state
from natlang_neuralese.train.native_continuation import collect_sft_progress
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from test_native_model import tiny_source


def test_promoted_shared_parent_can_resume_legacy_ce_frame_without_repromoting(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    path=base/'tokenizer.json';raw=json.loads(path.read_text())
    raw['model']['vocab']['<think>']=raw['model']['vocab'].pop('t54')
    raw['model']['vocab']['</think>']=raw['model']['vocab'].pop('t55')
    path.write_text(json.dumps(raw))
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4,token_override_ids=())
    def row(name,split='train'):
        return {'schema':SCHEMA,'id':name,'split':split,'source_groups':[name],
            'tokenizer_fingerprint':config['tokenizer_fingerprint'],'nodes':[{'name':'answer',
                'token_ids':[1,54,5,55,2],'target_weights':[0,1,1,1,1]}]}
    rows=[row('a'),row('b')]
    source=tmp_path/'train.jsonl';source.write_text(''.join(json.dumps(value)+'\n' for value in rows))
    held=tmp_path/'held.jsonl';held.write_text(json.dumps(row('held','test'))+'\n')
    common=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(source),'--held-native-graphs',str(held),'--epochs','1',
        '--stage-id','full','--eval-every','1','--chunk-tokens','1','--weight-decay','0']
    with monkeypatch.context() as patch:
        patch.setattr(trainer,'load_native_model',lambda *a,**kw:load_native_model(*a,**kw,token_override_ids=()))
        assert trainer.main([*common,'--out',str(tmp_path/'old')])==0
    state=torch.load(tmp_path/'old/checkpoint.pt',map_location='cpu',weights_only=False)
    # Released legacy formats did not declare sparse dtype. Values and all
    # actual trainer schedule/moment/RNG receipts remain untouched.
    state['model_config'].pop('sparse_row_dtype')
    state['interface_state'].pop('sparse_row_dtype')
    state['training']['identity']['config'].pop('sparse_row_dtype',None)
    parent=tmp_path/'legacy.pt';torch.save(state,parent)
    model,tokenizer,_,config,_=load_native_checkpoint(parent)
    frames=collect_sft_progress(state['training'])
    event=upgrade_reasoning_tokens(model,tokenizer,config)
    optimizer,_=make_native_optimizer(model,lr=1e-5,in_backward=False)
    optimizer.load_state_dict(state['training']['optimizer'],
        allow_added_names=('interface.extra_input_rows','interface.extra_output_rows'),
        allow_promoted_names=tuple('interface.'+name for name in event.get('promoted_parameters',{})))
    shared=tmp_path/'shared.pt'
    torch.save({'schema':SHARED_CHECKPOINT_SCHEMA,'participants':{'student':{
        'model_config':config,'model_state':model_state(model),'optimizer':optimizer.state_dict(),
        'fp8':None,'sft_progress':frames,'global_step':state['training']['global_step']}},
        'training':{'step':3,'rng':state['training']['rng']}},shared)
    rows.append(row('c'));source.write_text(''.join(json.dumps(value)+'\n' for value in rows))
    assert trainer.main([*common,'--out',str(tmp_path/'resumed'),'--continue-from',str(shared),
        '--native-participant','student','--objective-continuation','--learn-reasoning-boundaries'])==0
    resumed=torch.load(tmp_path/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
    assert resumed['training']['exposure_counts']=={'a':1,'b':1,'c':1}
    assert resumed['training']['global_step']==state['training']['global_step']+1
