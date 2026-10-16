"""Production coefficient learning uses the existing graph and named optimizers."""
import pytest
import torch
from natlang_neuralese.common.paths import root as logical_root
from test_native_serving_adapters import engine_fixture
from test_native_invocation_actors import adapters
from natlang_neuralese.serve.native_grad import native_graph_gradient,native_coefficient_gradients


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_all_selected_actor_gradients_share_one_preupdate_forward(tmp_path,family):
    torch.manual_seed(1741)
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    role=engine.role_tokens
    graph={'schema':'natlang.neuralese-native-graph/1','id':'same-preupdate-all-actors','split':'train',
        'source_groups':['coefficient-train-fixture'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'writer','token_ids':[4,role.produce_open,5,6,role.produce_close,7],
                  'producing':[[2,4]],'target_weights':[0]*6},
                 {'name':'reader','token_ids':[8,role.consume_open,role.consume_close,9,10],
                  'imports':[{'at':2,'source':'writer','start':2,'stop':4}],'target_weights':[0,0,0,1,1]}]}
    bindings={'writer':[source.id],'reader':[target.id]}
    name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    body={'schema':'natlang.native-gradient-request/1','graph':graph,'invocation_adapters':bindings,
          'native_identity':engine.native_identity,'adapter_gradient_precision':'fp32'}
    targets=[{'id':actor.id,'names':[name]} for actor in (source,target)]
    count=[]
    handle=engine.shared_model.register_forward_hook(lambda *args:count.append(1))
    try:actual=native_coefficient_gradients(engine,body,targets)
    finally:handle.remove()
    assert len(count)==1,'all actor gradients must use one actual shared graph forward'
    assert len(actual['gradients'])==2
    for declaration,record in zip(targets,actual['gradients']):
        gradient=engine.store.get_native_tensor_artifact(record['id']).artifact
        expected=native_graph_gradient(engine,{**body,'target':{'kind':'adapter',**declaration}},None)
        reference=engine.store.get_native_tensor_artifact(expected['gradient']['id']).artifact
        assert gradient.source_artifact_id==declaration['id']
        assert gradient.provenance==reference.provenance
        assert torch.equal(gradient.tensors[name],reference.tensors[name])
        assert gradient.tensors[name].norm()>0
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_whole_policy_stop_restart_preserves_all_moments_rng_and_exposure(tmp_path,monkeypatch,family):
    import copy,json
    from test_native_adapted_rl import capture_actor
    from natlang_neuralese.model.native_coefficient_policy import initialize_policy,restore_policy
    from natlang_neuralese.rl.native_replay import invocation_behavior_policy,prepare_objective
    from natlang_neuralese.rl.graph_replay import save_bundle
    from natlang_neuralese.serve.store import encode_native_state,TensorStore
    from natlang_neuralese.model.native_artifact import encode_native_tensor_artifact
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
    from natlang_neuralese.rl.native_coefficient_stage import train
    from natlang_neuralese.train.loop import StopSignal
    torch.manual_seed(1743)
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    initial=initialize_policy(engine,{source.id:[name],target.id:[name]},defaults=[],out=tmp_path/'initial',state_transport='same-base-adapter-state')
    base_digest=sha256_file_hex_cached(engine.native_checkpoint)
    _,turns=capture_actor(engine,[source.id],monkeypatch,reader_entries=[target.id],explicit_state_mode=True)
    rows=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],'sampling':turn['sampling'],
           'replay':{'raw_calls_retained':True}} for turn in turns]
    episodes=[{'task_id':'coefficient-production-train','split':'train','group':'one',
        'sample':sample,'reward':{'quality':quality},'policy':{'checkpoint_sha256':engine.native_identity['checkpoint_sha256']},
        'turns':copy.deepcopy(rows)} for sample,quality in enumerate((1,0))]
    fetch_block=lambda ident:engine.store.get_native_state(ident).block
    behavior=invocation_behavior_policy(episodes,identity=engine.native_identity,roles=engine.role_tokens,fetch_block=fetch_block)
    def fetch(ident):
        return encode_native_tensor_artifact(engine.store.get_native_tensor_artifact(ident).artifact) if ident.startswith('nnt1_') else encode_native_state(fetch_block(ident))
    save_bundle(episodes,tmp_path/'rollout-bundle',policy_id=engine.native_identity['checkpoint_sha256'],fetch=fetch)
    source_manifest=tmp_path/'rollout-bundle/manifest.json';source_bytes=source_manifest.read_bytes()
    from natlang_neuralese.rl.native_stage import prepare
    engine.tokenizer.save_pretrained(tmp_path/'rl-tokenizer')
    graphs,spec=prepare(episodes,checkpoint=engine.native_checkpoint,tokenizer=tmp_path/'rl-tokenizer',
        bundle=source_manifest,coefficient_policy=initial)
    assert source_manifest.read_bytes()==source_bytes
    assert spec['schema']=='natlang.native-policy-objective/2'
    rebound=json.loads(__import__('pathlib').Path(spec['block_bundle']).read_text())
    assert rebound['source_collection_manifest']['sha256']==sha256_file_hex_cached(source_manifest)
    objective=tmp_path/'objective.json';objective.write_text(json.dumps(spec))
    evaluations=[]
    def evaluate(engine,path,final,coverage=None):
        # Mechanism test executes an actual adapted held-loss observation. It is
        # not a task-quality qualification, and production must also run tasks.
        from natlang_neuralese.model.native_adapters import run_native
        from natlang_neuralese.model.shared_state import Invocation,compile_graph
        policy=restore_policy(engine,path)
        entry=policy['handles'][source.id]
        with torch.no_grad():
            loss=run_native(engine,lambda:engine.shared_model.logits(engine.shared_model(compile_graph([Invocation('held',(11,12,13))]))['h_final']).float().square().mean(),[entry['current_id']])
        evaluations.append((str(path),float(loss)))
        return {'passed':bool(torch.isfinite(loss)),'diagnostic_held_readout_loss':float(loss),'task_quality_claim':False}
    options=dict(policy=initial,graphs=graphs,objective=objective,epochs=2,optimizer={'name':'adam','learning_rate':.01},evaluate=evaluate)
    from natlang_neuralese.train.loop import capture_training_rng_state,restore_training_rng_state
    import random
    # Consume real process RNG at each forward and during held execution, so
    # exact RNG continuation is exercised rather than metadata-only equality.
    rng_hook=engine.shared_model.register_forward_hook(lambda *args:(torch.rand(1),random.random()) and None)
    initial_rng=capture_training_rng_state(engine.device)
    stop=StopSignal()
    partial=train(engine,out=tmp_path/'resume',stop=stop,after_commit=lambda state:stop.request(),
                  **{**options,'graphs':graphs[:1]})
    assert partial['step']==1 and not partial['completed'] and not evaluations
    assert len(list((tmp_path/'resume/policies').glob('*/policy.json')))==1
    assert len(list((tmp_path/'resume/training-state').glob('*.pt')))==1
    saved=torch.load(tmp_path/'resume/checkpoint.pt',weights_only=False)
    assert sum(item['visits'] for item in saved['obligations'])==1
    from natlang_neuralese.model.native_coefficient_policy import policy_dependencies
    pending_items=policy_dependencies(partial['policy']['path'])
    pending_closure={entry['sha256']:__import__('pathlib').Path(entry['path']) for entry in pending_items}
    assert len(pending_items)>len(pending_closure),'identical bytes at distinct required archive paths must retain both aliases'
    assert pending_closure[sha256_file_hex_cached(objective)]==objective
    assert pending_closure[sha256_file_hex_cached(source_manifest)]==source_manifest
    assert sha256_file_hex_cached(engine.native_checkpoint) in pending_closure

    engine.store.close();engine.store=TensorStore(tmp_path/'restart-store')
    resumed=train(engine,out=tmp_path/'portable-resume',**{**options,'policy':partial['policy']['path']})
    assert resumed['step']==4 and resumed['completed'] and resumed['qualification_complete']
    assert len(evaluations)==1
    final=restore_policy(engine,resumed['policy']['path'])
    final_closure={entry['sha256']:__import__('pathlib').Path(entry['path']) for entry in policy_dependencies(resumed['policy']['path'])}
    assert sha256_file_hex_cached(objective) not in final_closure,'completed objective is not a live deployment dependency'
    assert sha256_file_hex_cached(__import__('pathlib').Path(final['training_state']['path'])) in final_closure
    states={}
    for handle,entry in final['handles'].items():
        assert entry['current_id']!=handle and entry['optimizer_state_id'] is not None
        states[handle]=engine.store.get_native_tensor_artifact(entry['optimizer_state_id']).artifact
        assert states[handle].provenance['step']==4
    # A complete resumed stage does not repeat a single optimizer step.
    repeated=train(engine,out=tmp_path/'portable-resume',**{**options,'policy':partial['policy']['path']})
    assert repeated['step']==4 and repeated['coefficient_policy_sha256']==resumed['coefficient_policy_sha256']
    assert __import__('pathlib').Path(resumed['policy']['path']).exists(),'qualified published parent must survive metadata refresh'
    # Independent uninterrupted continuation starts from the same exact initial
    # coefficients; neither process initialization nor held inference may alter
    # the actual resulting coefficients/moments/RNG.
    final_rng=torch.load(tmp_path/'portable-resume/checkpoint.pt',weights_only=False)['rng']
    restore_training_rng_state(initial_rng,engine.device)
    uninterrupted=train(engine,out=tmp_path/'whole',**options)
    other_rng=torch.load(tmp_path/'whole/checkpoint.pt',weights_only=False)['rng']
    assert final_rng['python_rng']==other_rng['python_rng']
    assert torch.equal(final_rng['torch_rng'],other_rng['torch_rng'])
    assert all(torch.equal(left,right) for left,right in zip(final_rng['cuda_rng'],other_rng['cuda_rng']))
    rng_hook.remove()
    other=restore_policy(engine,uninterrupted['policy']['path'])
    for handle,entry in final['handles'].items():
        assert other['handles'][handle]==entry
        actual=engine.store.get_native_tensor_artifact(entry['optimizer_state_id']).artifact
        assert all(torch.equal(actual.tensors[key],states[handle].tensors[key]) for key in actual.tensors)
    assert sha256_file_hex_cached(engine.native_checkpoint)==base_digest
    engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_runtime_policy_binds_generation_decision_and_actual_default_coefficients(tmp_path,family):
    import json,threading,subprocess
    from pathlib import Path
    from natlang_neuralese.model.native_coefficient_policy import initialize_policy,policy_core,publish_policy,runtime_binding
    from natlang_neuralese.rl.native_coefficient_stage import bind_policy
    from natlang_neuralese.serve.http import serve
    from dataclasses import replace
    torch.manual_seed(1751)
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    initial=initialize_policy(engine,{source.id:[name]},defaults=[{'id':source.id,'scale':.75}],out=tmp_path/'initial')
    core=policy_core(json.loads(initial.read_text()))
    changed=engine.store.put_native_tensor_artifact(replace(source.artifact,
        tensors={key:value+.01 for key,value in source.artifact.tensors.items()}))
    core['handles'][source.id]['current_id']=changed.id
    current=publish_policy(engine,core,tmp_path/'current')
    policy=bind_policy(engine,current)
    runtime=runtime_binding(policy,current)
    fixture=tmp_path/'runtime.json';fixture.write_text(json.dumps({'runtime':runtime,'handle':source.id,'current':changed.id,
        'defaults':[{'id':changed.id,'scale':.75}]}))
    engine.start();server=serve(engine,'127.0.0.1',0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        result=subprocess.run(['node',str(Path(__file__).resolve().parents[2]/'ts-host/test/fixtures/native-coefficient-runtime.mjs'),
            f'http://127.0.0.1:{server.server_address[1]}',str(fixture)],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+'\n'+result.stderr
    finally:server.shutdown();server.server_close();engine.stop();engine.store.close()



@pytest.mark.parametrize('family',['lfm','mellum'])
def test_explicit_new_base_migration_preserves_basis_coefficients_and_all_moments(tmp_path,family):
    import json
    from pathlib import Path
    from natlang_neuralese.model.native_coefficient_policy import initialize_policy,policy_core,publish_policy,rebase_policy,restore_policy
    from natlang_neuralese.model.native_adapters import optimize_adapter
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA,model_state
    from natlang_neuralese.serve import load_native_engine
    engine=engine_fixture(tmp_path,family,service_template=True)
    source,target=adapters(engine)
    name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    initial=initialize_policy(engine,{source.id:[name],target.id:[name]},defaults=[source.id],out=tmp_path/'initial',state_transport='same-base-adapter-state')
    core=policy_core(json.loads(initial.read_text()))
    roles=engine.role_tokens
    graph={'schema':'natlang.neuralese-native-graph/1','id':'coefficient-base-migration','split':'train',
        'source_groups':['native-coefficient-migration-train'],'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],
        'nodes':[{'name':'writer','token_ids':[4,roles.produce_open,5,6,roles.produce_close,7],'producing':[[2,4]],'target_weights':[0]*6},
                 {'name':'reader','token_ids':[8,roles.consume_open,roles.consume_close,9,10],
                  'imports':[{'at':2,'source':'writer','start':2,'stop':4}],'target_weights':[0,0,0,1,1]}]}
    body={'schema':'natlang.native-gradient-request/1','native_identity':engine.native_identity,'graph':graph,
          'invocation_adapters':{'writer':[source.id],'reader':[target.id]},'adapter_gradient_precision':'fp32'}
    result=native_coefficient_gradients(engine,body,[{'id':ident,'names':[name]} for ident in (source.id,target.id)])
    for record in result['gradients']:
        gradient=engine.store.get_native_tensor_artifact(record['id']).artifact
        update=optimize_adapter(engine,gradient,{'name':'adam','learning_rate':.01},None)
        entry=core['handles'][gradient.source_artifact_id]
        entry['current_id']=update['adapter']['id'];entry['optimizer_state_id']=update['optimizer_state_id']
    from natlang_neuralese.model.native_coefficient_policy import write_training_state
    from natlang_neuralese.train.loop import capture_training_rng_state
    progress={'schema':'natlang.native-coefficient-training/1','base_native_identity':dict(engine.native_identity),
        'rng':capture_training_rng_state(engine.device),'step':3,'cursor':1,
        'actor_policy_sha256':__import__('natlang_neuralese.model.native_coefficient_policy',fromlist=['actor_policy_sha256']).actor_policy_sha256(core),
        'obligations':[{'key':'finished','visits':2,'required_visits':2,'objective':{'path':'completed-not-needed','sha256':'0'*64}},
                       {'key':'owed','visits':1,'required_visits':4,'graph':{'old':'actual behavior is not rebound'},
                        'objective':{'path':'pending-not-replayed','sha256':'1'*64}}]}
    core['training_state']=write_training_state(progress,tmp_path/'original-progress')
    current=publish_policy(engine,core,tmp_path/'old-current')
    previous={handle:{key:engine.store.get_native_tensor_artifact(entry[key]).artifact
              for key in ('original_id','current_id','optimizer_state_id')} for handle,entry in core['handles'].items()}
    old_write=engine.native_write_text('t5 t6',adapters=[core['handles'][source.id]['current_id']])
    old_block=engine.store.get_native_state(old_write['id']).block
    # A genuine backbone continuation changes actual weights and checkpoint.
    with torch.no_grad():next(engine.shared_model.port.hf.parameters()).add_(.01)
    parent=Path(engine.native_checkpoint)
    new=tmp_path/'continued-native.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':engine.model_config,**model_state(engine.shared_model)},new)
    newer=load_native_engine(str(new),device=engine.device)
    try:
        with pytest.raises(ValueError,match='another exact native base'):restore_policy(newer,current)
        with pytest.raises(ValueError,match='explicit supervised cancellation'):
            rebase_policy(newer,current,tmp_path/'refuse-pending')
        reason='Backbone improved; preserve completed visits and recollect the untrained stale behavior.'
        migrated=rebase_policy(newer,current,tmp_path/'migrated',cancel_stale_reason=reason)
        actual=restore_policy(newer,migrated)
        from natlang_neuralese.model.native_coefficient_policy import verify_base_transition
        with pytest.raises(ValueError,match='explicit supervised cancellation'):
            verify_base_transition(current,migrated,identity=newer.native_identity)
        verify_base_transition(current,migrated,identity=newer.native_identity,cancel_stale_reason=reason)
        from natlang_neuralese.model.native_composite import resolve_composite_artifact
        carried=torch.load(resolve_composite_artifact(actual['training_state'],migrated),weights_only=False)
        assert carried['obligations'][0]==progress['obligations'][0]
        assert carried['obligations'][1]['visits']==1 and carried['obligations'][1]['required_visits']==4
        assert 'graph' not in carried['obligations'][1] and carried['obligations'][1]['cancelled']
        assert carried['cancellations'][-1]['rows'][0]['uncredited_visits']==3
        assert torch.equal(carried['rng']['torch_rng'],progress['rng']['torch_rng'])
        assert actual['base_native_identity']==newer.native_identity!=engine.native_identity
        assert set(actual['handles'])==set(core['handles']),'program handles remain stable; actual artifacts are rebound explicitly'
        for handle,entry in actual['handles'].items():
            for key in ('original_id','current_id','optimizer_state_id'):
                before=previous[handle][key];after=newer.store.get_native_tensor_artifact(entry[key]).artifact
                assert after.native_identity==newer.native_identity and entry[key]!=core['handles'][handle][key]
                assert all(torch.equal(value,after.tensors[name]) for name,value in before.tensors.items())
                assert after.provenance['base_transition']['qualified'] is False
            before=engine.store.get_native_tensor_artifact(previous[handle]['current_id'].source_artifact_id).artifact
            after=newer.store.get_native_tensor_artifact(newer.store.get_native_tensor_artifact(entry['current_id']).artifact.source_artifact_id).artifact
            assert all(torch.equal(value,after.tensors[name]) for name,value in before.tensors.items())
        from natlang_neuralese.model.native_artifact import NativeTensorArtifact
        from dataclasses import replace
        forged_core=policy_core(actual)
        actor=forged_core['handles'][source.id]
        good=newer.store.get_native_tensor_artifact(actor['current_id']).artifact
        wrong=newer.store.put_native_tensor_artifact(replace(good,tensors={key:value+.001 for key,value in good.tensors.items()}))
        old_state=newer.store.get_native_tensor_artifact(actor['optimizer_state_id']).artifact
        wrong_state=newer.store.put_native_tensor_artifact(replace(old_state,source_artifact_id=wrong.id))
        actor.update(current_id=wrong.id,optimizer_state_id=wrong_state.id)
        forged=publish_policy(newer,forged_core,tmp_path/'forged-preservation')
        with pytest.raises(ValueError,match='changed exact values'):
            verify_base_transition(current,forged,identity=newer.native_identity,cancel_stale_reason=reason)
        # Even explicit same-base transport must reject actual old-base KV.
        from natlang_neuralese.serve.engine import GenerationRequest
        newer.store.put_native_state(old_block)
        old_id=old_write['id']
        with pytest.raises(Exception,match='same base checkpoint'):
            newer.generate(GenerationRequest(messages=[{'role':'user','content':[{'type':'native_state','id':old_id}]}],
                adapters=[actual['handles'][source.id]['current_id']],max_tokens=1,
                native_state_modes={old_id:'same-base-adapter-state'}))
        # Both new-base actors execute through the actual shared graph and the
        # independent per-token Engine/cache reference, including producer dcoef.
        bindings={'writer':[actual['handles'][source.id]['current_id']], 'reader':[actual['handles'][target.id]['current_id']]}
        from test_native_invocation_policy import actual_engine_gradient
        from natlang_neuralese.model.native_actor_graph import run_invocation_actors
        from natlang_neuralese.data.native_graph import parse_graph
        from natlang_neuralese.rl.native_replay import action_logps
        from natlang_neuralese.rl.full_policy import clipped_token_loss
        selected=bindings['writer'][0]
        row={'actions':[{'query':8,'token_id':9,'temperature':1.,'allowed_token_ids':None,'behavior_logp':0.},
                        {'query':9,'token_id':10,'temperature':1.,'allowed_token_ids':None,'behavior_logp':0.}], 'advantage':1.}
        cached_loss,cached_grad,_,_=actual_engine_gradient(newer,graph,row,bindings,selected,name)
        artifact=newer.store.get_native_tensor_artifact(selected).artifact
        leaf=artifact.tensors[name].clone().requires_grad_(True)
        def flat(bank):
            result=newer.shared_model(parse_graph(graph,tokenizer_fingerprint=newer.native_identity['tokenizer_fingerprint']).plan)
            new_logps,old_logps=action_logps(newer.shared_model,result['h_final'][0],row['actions'])
            loss=clipped_token_loss(new_logps,old_logps,1.,clip_low=.2,clip_high=.28).mean()
            return loss.detach(),torch.autograd.grad(loss,leaf)[0]
        flat_loss,flat_grad=run_invocation_actors(newer,list(bindings),bindings,flat,
            leaves={selected:{**artifact.tensors,name:leaf}},gradient_precision='fp32')
        torch.testing.assert_close(flat_loss,cached_loss,atol=3e-6,rtol=2e-5)
        torch.testing.assert_close(flat_grad,cached_grad,atol=3e-6,rtol=2e-5)
        assert (flat_grad-cached_grad).norm()/cached_grad.norm()<2e-5
    finally:newer.store.close();engine.store.close()


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_real_task_collector_receives_current_coefficient_policy(tmp_path,monkeypatch,family):
    import json
    from pathlib import Path
    from natlang_neuralese.model.native_coefficient_policy import initialize_policy
    from natlang_neuralese.rl.native_coefficient_stage import bind_policy
    from natlang_neuralese.eval.task_cadence import TaskEval,parse_task_eval
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.model.native_adapters import execution_binding
    engine=engine_fixture(tmp_path,family,service_template=True,context=8192)
    source,target=adapters(engine)
    name=next(name for name in source.artifact.tensors if name.endswith('v_proj.weight'))
    policy=initialize_policy(engine,{source.id:[name]},defaults=[source.id],out=tmp_path/'policy')
    bind_policy(engine,policy)
    observed=[]
    original=engine._native_prefill_impl
    def prefill(sequence):
        observed.append(execution_binding(engine))
        return original(sequence)
    monkeypatch.setattr(engine,'_native_prefill_impl',prefill)
    # Bound this mechanism fixture's output and resource demand. The action has
    # its real actual-model probability; this is no task-quality/lift claim.
    monkeypatch.setattr(engine,'_native_token',lambda sequence:(2,float(sequence.logits[0].float().log_softmax(-1)[2])))
    original_packet=logical_root('repo')/'runs/natlang-held-task-packet-v3-20261012/cases.jsonl'
    packet=tmp_path/'held-case.jsonl';packet.write_text(original_packet.open().readline())
    spec=parse_task_eval({'packet':{'path':str(packet),'sha256':sha256_file_hex_cached(packet)},'cases':[0],
        'workers':1,'noise_repeats':1,'neuralese_server':True,'runtime':str(Path(__file__).resolve().parents[2]/'ts-host'),
        'max_model_requests':1,'max_turns':4,'case_seconds':120,'min_successes':0})
    try:
        result=TaskEval(spec,tmp_path/'held',engine,prompt_rendering='full').run(1)
        assert observed and all(entries and entries[0]['id']==source.id for entries in observed)
        assert result['coefficient_policy_sha256']==__import__('natlang_neuralese.model.native_coefficient_policy',fromlist=['actor_policy_sha256']).actor_policy_sha256(json.loads(policy.read_text()))
        assert result['outcome_counts']['infrastructure_error']==0,result
        assert result['eligible'] is True,result
    finally:engine.store.close()
