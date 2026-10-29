import pytest
import torch
from natlang_neuralese.train import native_state_sft as trainer
from test_native_source_fetch_recovery import source
from test_native_resident_ce import fixture


def test_fresh_initialization_saves_zero_update_and_refuses_parent(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';source(base,'lfm')
    _,_,_,config=trainer.load_native_model(base,dtype=torch.bfloat16,rank=4)
    paths,_,_=fixture(tmp_path,config['tokenizer_fingerprint'])
    args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu',
          '--native-graphs',str(paths['old_train']),'--held-native-graphs',str(paths['held']),
          '--ce-supervision-policy',str(paths['old_ce']),'--epochs','2','--seed','23',
          '--learning-rate','0.000001','--learned-row-learning-rate','0.0001',
          '--interface-row-learning-rate','0.0001','--out',str(tmp_path/'out'),'--initialize-only']
    monkeypatch.setattr(trainer,'native_training_update',lambda *a,**k:pytest.fail('initialization ran an update'))
    trainer.main(args)
    ck=torch.load(tmp_path/'out/checkpoint.pt',map_location='cpu',weights_only=False)
    state=ck['training']
    assert state['global_step']==state['step']==0 and not any(state['exposure_counts'].values())
    assert state['optimizer'] and state['rng'] and state['sft_progress']
    assert state['options']['fresh_initialization']['parent_checkpoint'] is None
    import hashlib
    from natlang_neuralese.rl.jolt_handoff import checked_fresh_parent
    checkpoint=tmp_path/'out/checkpoint.pt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert checked_fresh_parent(checkpoint,expected_sha256=digest,
        source_sha256=state['source_receipts'][0]['sha256'])==checkpoint
    with pytest.raises(ValueError,match='fresh output'):
        trainer.main(args)


def test_fresh_handoff_rejects_real_nested_updated_optimizer(tmp_path,monkeypatch):
    import hashlib
    from natlang_neuralese.rl.jolt_handoff import checked_fresh_parent
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';source(base,'lfm')
    _,_,_,config=trainer.load_native_model(base,dtype=torch.bfloat16,rank=4)
    paths,_,_=fixture(tmp_path,config['tokenizer_fingerprint'])
    trainer.main(['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu',
        '--native-graphs',str(paths['old_train']),'--held-native-graphs',str(paths['held']),
        '--ce-supervision-policy',str(paths['old_ce']),'--epochs','2','--seed','23',
        '--accumulate-batches','1','--refresh-after-updates','1','--eval-every','1',
        '--checkpoint-every','1','--checkpoint-seconds','0','--out',str(tmp_path/'updated')])
    value=torch.load(tmp_path/'updated/checkpoint.pt',map_location='cpu',weights_only=False)
    training=value['training'];assert training['global_step']==1
    assert training['optimizer']['adamw']['state']
    source_sha=training['source_receipts'][0]['sha256']
    training['step']=training['global_step']=0;training['exposure_counts']={}
    training['options']['fresh_initialization']={'schema':'natlang.fresh-native-initialization/1',
        'parent_checkpoint':None,'optimizer_updates':0,'earned_exposure':0}
    checkpoint=tmp_path/'forged.pt';torch.save(value,checkpoint)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='inherited training'):
        checked_fresh_parent(checkpoint,expected_sha256=digest,source_sha256=source_sha)
