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
@pytest.mark.parametrize('shared',[None,'source-v1','stable-v1'])
def test_raw_policy_identity_and_tokenizer_match_actual_loaded_model_without_construction(tmp_path,monkeypatch,family,attention,shared):
    import natlang_neuralese.model.native_model as native
    from transformers import AutoTokenizer
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from test_native_serving_adapters import engine_fixture
    engine=engine_fixture(tmp_path,family)
    model,tokenizer,roles,config=engine.shared_model,engine.tokenizer,engine.role_tokens,engine.model_config
    base=Path(config['base'])
    if shared is not None:
        from natlang_neuralese.model.shared_execution import configure_shared_execution
        configure_shared_execution(model,shared);config['native_shared_execution']=shared
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
    reloaded=AutoTokenizer.from_pretrained(actual['tokenizer'])
    assert tokenizer_fingerprint(reloaded)==config['tokenizer_fingerprint']
    from natlang_neuralese.rl.native_raw_policy import bind_saved_history_reasoning
    rebound=bind_saved_history_reasoning(reloaded,tokenizer_path=actual['tokenizer'],
        checkpoint_sha256=expected['checkpoint_sha256'],tokenizer_fingerprint=expected['tokenizer_fingerprint'])
    assert rebound==actual['history_reasoning']
    assert actual['history_reasoning_binding']['sha256']==sha256_file_hex(actual['history_reasoning_binding']['path'])
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


def test_raw_policy_persists_explicit_keep_profile_across_tokenizer_reload(tmp_path):
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.rl.native_raw_policy import bind_saved_history_reasoning
    from transformers import AutoTokenizer
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    engine=engine_fixture(tmp_path,'lfm',service_template=True)
    model,tokenizer,roles,config=engine.shared_model,engine.tokenizer,engine.role_tokens,engine.model_config
    # Give this fixture a genuinely switch-capable LFM template: the default
    # service fixture drops earlier reasoning by construction.
    tokenizer.chat_template=tokenizer.chat_template.replace(' and loop.last', '')
    tokenizer.save_pretrained(config['base'])
    config['tokenizer_fingerprint']=tokenizer_fingerprint(tokenizer)
    config['history_reasoning_policy']='keep'
    checkpoint=tmp_path/'keep-parent.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),
                'training':parent_training()},checkpoint)
    prepared=prepare_raw_policy(checkpoint,base=Path(config['base']),
                                tokenizer_out=tmp_path/'keep-tokenizer',require_optimizer=False)
    assert set(prepared['native_identity']) == {'checkpoint_sha256','tokenizer_fingerprint',
        'embedding_weight_sha256','architecture'}
    reloaded=AutoTokenizer.from_pretrained(prepared['tokenizer'])
    history=bind_saved_history_reasoning(reloaded,tokenizer_path=prepared['tokenizer'],
        checkpoint_sha256=prepared['native_identity']['checkpoint_sha256'],
        tokenizer_fingerprint=prepared['native_identity']['tokenizer_fingerprint'],
        expected_history_reasoning_policy=prepared['history_reasoning']['policy'],
        expected_binding_sha256=prepared['history_reasoning_binding']['sha256'])
    assert history['policy']=='keep'
    assert tokenizer_fingerprint(reloaded)==prepared['native_identity']['tokenizer_fingerprint']
    with pytest.raises(ValueError,match='not bound to this checkpoint/tokenizer'):
        bind_saved_history_reasoning(reloaded,tokenizer_path=prepared['tokenizer'],
            checkpoint_sha256='0'*64,tokenizer_fingerprint=prepared['native_identity']['tokenizer_fingerprint'])
    sidecar=Path(prepared['tokenizer'])/'natlang-history-reasoning.json'
    original=sidecar.read_text()
    sidecar.unlink()
    with pytest.raises(ValueError,match='requires its saved tokenizer sidecar'):
        bind_saved_history_reasoning(reloaded,tokenizer_path=prepared['tokenizer'],
            checkpoint_sha256=prepared['native_identity']['checkpoint_sha256'],
            tokenizer_fingerprint=prepared['native_identity']['tokenizer_fingerprint'],
            expected_history_reasoning_policy='keep')
    import json
    tampered=json.loads(original)
    tampered['history_reasoning']['policy']='last_turn_only'
    sidecar.write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='differs from checkpoint profile'):
        bind_saved_history_reasoning(reloaded,tokenizer_path=prepared['tokenizer'],
            checkpoint_sha256=prepared['native_identity']['checkpoint_sha256'],
            tokenizer_fingerprint=prepared['native_identity']['tokenizer_fingerprint'],
            expected_history_reasoning_policy='keep')
    with pytest.raises(ValueError,match='differs from the authenticated checkpoint control'):
        bind_saved_history_reasoning(reloaded,tokenizer_path=prepared['tokenizer'],
            checkpoint_sha256=prepared['native_identity']['checkpoint_sha256'],
            tokenizer_fingerprint=prepared['native_identity']['tokenizer_fingerprint'],
            expected_history_reasoning_policy='keep', expected_binding_sha256=prepared['history_reasoning_binding']['sha256'])


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
        actor_runtime=Path(__file__).resolve().parents[2],
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
    assert result['history_reasoning_policy']=='last_turn_only'
    assert result['history_reasoning_binding_sha256']==sha256_file_hex(
        Path(result['rl_tokenizer'])/'natlang-history-reasoning.json')
    assert result['learner_command'][-6:] == ['--history-reasoning-policy','{history_reasoning_policy}',
        '--history-reasoning-binding-sha256','{history_reasoning_binding_sha256}',
        '--history-reasoning-binding-path','{history_reasoning_binding_path}']
    bound=prepare_raw_policy(checkpoint,base=base,tokenizer_out=tmp_path/'second-tokenizer')
    receipt={'schema':'natlang.rl-policy-binding/2','communication':'native-state',
        'checkpoint_sha256':bound['native_identity']['checkpoint_sha256'],'native_identity':bound['native_identity'],
        'history_reasoning':bound['history_reasoning']}
    production.validate_actor_receipts({'local':receipt},policy_id=sha256_file_hex(checkpoint),
        neuralese_server=True,expected_native=bound)
    forged={**receipt,'native_identity':{**receipt['native_identity'],'embedding_weight_sha256':'f'*64}}
    with pytest.raises(ValueError):production.validate_actor_receipts({'local':forged},policy_id=sha256_file_hex(checkpoint),
        neuralese_server=True,expected_native=bound)


def test_synthetic_learner_result_cannot_complete_raw_policy_round(tmp_path,monkeypatch):
    """A controller-only fake checkpoint/report must not receive completed-round credit."""
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
    with pytest.raises(ValueError,match='exact published checkpoint'):
        production.run(config,rounds=1)
    state=json.loads((out/'state.json').read_text())
    current=out/'round-000000/learner/checkpoint.pt'
    assert calls==[('learn','rft',str(parent)),('publish','rft',str(current))]
    assert state['round']==0 and state['trained_rounds']==0
    assert 'last_completed_update' not in state and state['active_update_round'].endswith('round-000000')

@pytest.mark.parametrize('family',['lfm','mellum'])
def test_sampled_actor_profile_overrides_future_seed_profile_in_actual_stage_cli(tmp_path,monkeypatch,family):
    import json
    from transformers import AutoTokenizer
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.rl import production,native_stage
    from natlang_neuralese.rl.native_raw_policy import bind_saved_history_reasoning
    from natlang_neuralese.train.recipe import stage_parameter_args
    engine=engine_fixture(tmp_path,family,service_template=True)
    model,config=engine.shared_model,engine.model_config
    checkpoint=tmp_path/'parent.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':config,**model_state(model),
                'training':parent_training()},checkpoint)
    actual=prepare_raw_policy(checkpoint,base=config['base'],tokenizer_out=tmp_path/'actor-tokenizer')
    sampled=actual['history_reasoning']['policy']
    future='keep' if sampled=='last_turn_only' else 'last_turn_only'
    parameters={'learning_rate':3e-6,'history_reasoning_policy':future}
    original=parameters.copy()
    rl_args=production.native_policy_trainer_args(parameters,actual)
    seed_args=stage_parameter_args(parameters)
    assert parameters==original
    assert seed_args[seed_args.index('--history-reasoning-policy')+1]==future
    assert rl_args[rl_args.index('--history-reasoning-policy')+1]==sampled
    seen={}
    def admit(args,trainer):
        seen.update(profile=args.history_reasoning_policy,sha=args.history_reasoning_binding_sha256,
                    path=args.history_reasoning_binding_path,trainer=trainer)
        tokenizer=AutoTokenizer.from_pretrained(actual['tokenizer'],local_files_only=True)
        return bind_saved_history_reasoning(tokenizer,tokenizer_path=actual['tokenizer'],
            checkpoint_sha256=actual['native_identity']['checkpoint_sha256'],
            tokenizer_fingerprint=actual['native_identity']['tokenizer_fingerprint'],
            expected_history_reasoning_policy=args.history_reasoning_policy,
            expected_binding_sha256=args.history_reasoning_binding_sha256,
            expected_binding_path=args.history_reasoning_binding_path)
    monkeypatch.setattr(native_stage,'_admit_and_train',admit)
    command=['--rollouts',str(tmp_path/'rollouts.jsonl'),'--rl-tokenizer',actual['tokenizer'],
        '--continue-from',str(checkpoint),'--held-native-graphs',str(tmp_path/'held.jsonl'),
        '--out',str(tmp_path/'learner')]
    assert native_stage.main(command+rl_args)==actual['history_reasoning']
    assert seen['sha']==actual['history_reasoning_binding']['sha256']
    assert seen['trainer']==['--learning-rate','3e-06']
    with pytest.raises(ValueError,match='differs from checkpoint profile'):
        native_stage.main(command+seed_args)
    tampered=json.loads(Path(actual['history_reasoning_binding']['path']).read_text())
    tampered['history_reasoning']['policy']=future
    Path(actual['history_reasoning_binding']['path']).write_text(json.dumps(tampered))
    with pytest.raises(ValueError,match='authenticated checkpoint control'):
        native_stage.main(command+rl_args)
