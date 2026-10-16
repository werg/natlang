from pathlib import Path
import pytest
import torch
from test_native_serve import _tiny_source
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state, native_identity
from natlang_neuralese.rl.native_raw_policy import prepare_raw_policy
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.loop import capture_training_rng_state


def parent_training():
    return {'schema':TRAINING_SCHEMA,'rng':capture_training_rng_state('cpu'),
            'optimizer':{'state':{'exact-parent':'retained'}}}


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('attention',['ieee','tf32x3'])
def test_raw_policy_identity_and_tokenizer_match_actual_loaded_model_without_construction(tmp_path,monkeypatch,family,attention):
    import natlang_neuralese.model.native_model as native
    from transformers import AutoTokenizer
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from test_native_serving_adapters import engine_fixture
    engine=engine_fixture(tmp_path,family)
    model,tokenizer,roles,config=engine.shared_model,engine.tokenizer,engine.role_tokens,engine.model_config
    base=Path(config['base'])
    if attention!='ieee':
        from natlang_neuralese.model.native_precision import set_native_attention_precision
        set_native_attention_precision(model,attention);config['native_attention_precision']=attention
    with torch.no_grad():
        model.input_rows.add_(.013)
        model.adapter.up.weight.add_(.012)
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),
                'training':parent_training()},checkpoint)
    expected=native_identity(model,config,checkpoint_sha256=sha256_file_hex(checkpoint))
    before=torch.get_rng_state().clone()
    monkeypatch.setattr(native,'load_native_model',lambda *a,**kw:pytest.fail('second model construction'))
    actual=prepare_raw_policy(checkpoint,base=base,tokenizer_out=tmp_path/'tokenizer')
    assert actual['native_identity']==expected
    assert actual['native_participant'] is None and actual['semantic_adapter_sha256'] is None
    assert tokenizer_fingerprint(AutoTokenizer.from_pretrained(actual['tokenizer']))==config['tokenizer_fingerprint']
    assert torch.equal(before,torch.get_rng_state())
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    saved=CheckpointPolicy.load(checkpoint)
    monkeypatch.setattr(CheckpointPolicy,'load',lambda *a,**kw:pytest.fail('repeated full-parent deserialization'))
    reused=prepare_raw_policy(checkpoint,base=base,tokenizer_out=tmp_path/'reused-tokenizer',
        verified_checkpoint=({'path':str(checkpoint),'sha256':sha256_file_hex(checkpoint)},saved))
    assert reused['native_identity']==expected and torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError,match='another exact parent'):
        prepare_raw_policy(checkpoint,base=base,tokenizer_out=tmp_path/'bad-tokenizer',
            verified_checkpoint=({'path':str(checkpoint),'sha256':'0'*64},saved))


@pytest.mark.parametrize('tamper',['weights-only','legacy','roles','tokenizer','rng','schedule'])
def test_raw_policy_rejects_nonresumable_or_mismatched_parent(tmp_path,tamper):
    base=tmp_path/'base';base.mkdir();_tiny_source(base)
    model,tokenizer,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    state={'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),
           'training':parent_training()}
    if tamper=='weights-only':state['optimizer_pruned']=True
    elif tamper=='legacy':state['schema']='legacy'
    elif tamper=='roles':config['marker_ids']=[4,5,6,7]
    elif tamper=='rng':state['training'].pop('rng')
    elif tamper=='schedule':state['training'].pop('schema')
    else:config['tokenizer_fingerprint']='0'*64
    checkpoint=tmp_path/'checkpoint.pt';torch.save(state,checkpoint)
    with pytest.raises(ValueError):prepare_raw_policy(checkpoint,base=base,tokenizer_out=tmp_path/'tokenizer')


@pytest.mark.parametrize('cadence',[0,5000])
def test_bare_native_factory_keeps_actual_recipe_and_omits_shared_artifact_requirements(tmp_path,monkeypatch,cadence):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.rl import production
    from natlang_neuralese.train import recipe
    base=tmp_path/'base';base.mkdir();_tiny_source(base)
    model,tokenizer,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    checkpoint=tmp_path/'checkpoint.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),
                'training':parent_training()},checkpoint)
    declared={'id':'native_full','kind':'native_state_sft','parameters':{
        'epochs':3,'dtype':'float32','communication_foundation':True,'task_eval':{'cases':'registered-held'},
        'task_eval_every':cadence}}
    monkeypatch.setattr(recipe,'resolve_recipe_data',lambda path:{'stages':[declared]})
    monkeypatch.setattr(recipe,'resolve_stage_inputs',lambda *a:{'foundation_native_graphs':{'path':str(tmp_path/'foundation')}})
    args=SimpleNamespace(model='lfm',checkpoint=checkpoint,base=base,semantic_adapter=None,native_participant=None,
        semantic_refresh_config=None,production_stage='native_full',port=8097,
        **{key:tmp_path/key for key in ('seed_native_graphs','held_native_graphs','production_recipe','out','config','cases','runtime')})
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    saved=CheckpointPolicy.load(checkpoint)
    args.verified_native_checkpoint=({'path':str(checkpoint),'sha256':sha256_file_hex(checkpoint)},saved)
    with monkeypatch.context() as reused:
        reused.setattr(CheckpointPolicy,'load',lambda *a,**kw:pytest.fail('factory reloaded verified native parent'))
        result=production.configure_native(args)
    assert result['native_actor_mode']=='raw-state' and 'semantic_adapter' not in result
    for command in ('actor_command','learner_command','seed_command'):
        assert '--native-participant' not in result[command] and '--semantic-adapter' not in result[command]
    assert result['trainer_parameters']['epochs']==3 and result['trainer_parameters']['task_eval_every']==cadence
    assert result['foundation_native_graphs']==str(tmp_path/'foundation')
    assert '--foundation-native-graphs' in result['seed_command']
    bound=prepare_raw_policy(checkpoint,base=base,tokenizer_out=tmp_path/'second-tokenizer')
    receipt={'schema':'natlang.rl-policy-binding/2','communication':'native-state',
        'checkpoint_sha256':bound['native_identity']['checkpoint_sha256'],'native_identity':bound['native_identity']}
    production.validate_actor_receipts({'local':receipt},policy_id=sha256_file_hex(checkpoint),
        neuralese_server=True,expected_native=bound)
    forged={**receipt,'native_identity':{**receipt['native_identity'],'embedding_weight_sha256':'f'*64}}
    with pytest.raises(ValueError):production.validate_actor_receipts({'local':forged},policy_id=sha256_file_hex(checkpoint),
        neuralese_server=True,expected_native=bound)


def test_all_correct_ordinary_native_batch_updates_policy_without_semantic_refresh(tmp_path,monkeypatch):
    """Controller behavior only; exact real optimizer continuation is tested separately."""
    import json
    from natlang_neuralese.rl import production,native_raw_policy,native_refresh
    source=tmp_path/'cases.jsonl';source.write_text('{"id":"ordinary","split":"train"}\n')
    parent=tmp_path/'parent.pt';parent.write_bytes(b'exact-parent')
    out=tmp_path/'out';config=tmp_path/'config.json'
    config.write_text(json.dumps({'native_state':True,'checkpoint':str(parent),'base':'base',
        'out':str(out),'cases':str(source),'endpoints':['unused'],'learner_command':['learn'],
        'after_update':['publish']}))
    monkeypatch.setattr(native_raw_policy,'prepare_raw_policy',lambda *a,**kw:{'tokenizer':'exact-tokenizer'})
    monkeypatch.setattr(native_refresh,'refresh',lambda **kw:pytest.fail('raw policy must not wait for shared VAE'))
    def collect(**kwargs):
        (kwargs['out']/'rollouts.jsonl').write_text(json.dumps({'task_id':'ordinary','split':'train',
            'reward':{'quality':1},'turns':[{'invocation_id':'ordinary','sampling':{'prompt':[]}}]})+'\n')
        return {'scored':1,'successes':1,'groups_with_spread':0}
    monkeypatch.setattr(production,'collect',collect)
    calls=[]
    def command(argv,values):
        calls.append((argv[0],values['mode'],values['checkpoint']))
        if argv[0]=='publish':return
        destination=Path(values['learner_out']);destination.mkdir(parents=True)
        (destination/'checkpoint.pt').write_bytes(b'updated-policy')
        (destination/'report.json').write_text(json.dumps({'completed':True,'step':1}))
        (destination/'rl-data-receipt.json').write_text(json.dumps({'tasks_selected':[
            {'task_id':'ordinary','split':'train'}]}))
        (destination/'train.jsonl').write_text('{"step":1,"target_tokens":2,"supervised_tokens":2}\n')
    monkeypatch.setattr(production,'run_command',command)
    state=production.run(config,rounds=1)
    current=out/'round-000000/learner/checkpoint.pt'
    assert calls==[('learn','rft',str(parent)),('publish','rft',str(current))]
    assert state['checkpoint']==str(current) and state['trained_task_ids']==['ordinary']
    assert state['trained_target_tokens']==2 and not state.get('pending_native_checkpoint')
