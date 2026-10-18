"""Durable raw coordinator composition; fixtures here attest protocol, not lift."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from natlang_neuralese.rl import native_raw_rounds as raw,native_rounds
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.hashing import sha256_file_hex


def fixture(tmp_path,monkeypatch):
    parent=tmp_path/'parent.pt';parent.write_bytes(b'exact-current-native-parent')
    train=tmp_path/'train';train.write_text('{}\n');held=tmp_path/'held';held.write_text('{}\n')
    recipe=tmp_path/'recipe.json'
    recipe.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1','id':'raw-fixture',
        'input_bindings':{'train':{'path':str(train),'sha256':sha256_file_hex(train)},
                          'held':{'path':str(held),'sha256':sha256_file_hex(held)}},
        'stages':[{'id':'full','kind':'native_state_sft','requires':[],
            'inputs':{'native_graphs':'train','held_native_graphs':'held'},
            'parameters':{'epochs':3,'checkpoint_every':250,'communication_foundation':True}}]}))
    directory=tmp_path/'student';directory.mkdir()
    config={'native_state':True,'native_actor_mode':'raw-state','out':str(directory/'production'),
        'checkpoint':str(parent),'production_recipe':str(recipe),'production_stage':'full',
        'seed_command':['python','-m','natlang_neuralese.train.native_state_sft'],
        'trainer_parameters':{'checkpoint_every':250},'foundation_native_graphs':str(train)}
    config_path=directory/'config.json';write_json_durable(config_path,config)
    monkeypatch.setattr(raw,'configure_model',lambda *a:config_path)
    monkeypatch.setattr(raw,'_progress',lambda cp:{'stages':{}})
    stop=SimpleNamespace(requested=False);calls=[]
    def job(path,declaration,out,active_stop):
        name='sft' if declaration.get('mandatory_sft_only') else 'rl'
        calls.append((name,declaration))
        target=Path(declaration['out']);target.mkdir(parents=True,exist_ok=True)
        state_path=target/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'round':0}
        if name=='rl':
            checkpoint=target/f"policy-{state['round']}.pt";checkpoint.write_bytes(('policy'+str(state['round'])).encode())
            state.update(round=state['round']+1,phase='collect',checkpoint=str(checkpoint))
            write_json_durable(state_path,state)
            if getattr(stop,'interrupt_after_policy',False):stop.requested=True
            return state
        assert (not declaration['trainer_parameters']['refresh_at_checkpoint']
            or declaration['trainer_parameters'].get('refresh_after_updates') is not None)
        assert declaration['trainer_parameters']['checkpoint_every']==250
        checkpoint=target/'learner/checkpoint.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        # Exact output slots are reused after controller restart.
        if not checkpoint.exists():checkpoint.write_bytes(declaration['checkpoint'].encode()+b'+full-row-SFT')
        ready={'checkpoint':{'path':str(checkpoint),'sha256':sha256_file_hex(checkpoint)},
               'foundation_ready':True,'completed':False,'communication_foundation':{'complete':True}}
        write_json_durable(checkpoint.parent/'native-training-receipt.json',ready)
        return {'pending_native_checkpoint':str(checkpoint)}
    monkeypatch.setattr(native_rounds,'round_job',job)
    return directory,config,stop,calls


def test_raw_policy_publish_stop_restart_does_not_recollect_and_interleaves_same_parent(tmp_path,monkeypatch):
    directory,config,stop,calls=fixture(tmp_path,monkeypatch)
    stop.interrupt_after_policy=True
    raw.student_cycle('plan',{},None,directory,stop=stop)
    cursor=json.loads((directory/'cycle.json').read_text())
    assert cursor['pending_rl_round']==1 and cursor['phase']=='rl'
    published=raw._current_checkpoint(config)
    stop.requested=False;stop.interrupt_after_policy=False
    result=raw.student_cycle('plan',{},None,directory,stop=stop)
    assert [kind for kind,_ in calls].count('rl')==1
    first_sft=next(value for kind,value in calls if kind=='sft')
    assert first_sft['checkpoint']==published['path'] and 'semantic_adapter' not in first_sft
    assert first_sft['trainer_parameters']['communication_foundation'] is True
    assert result['cycle']==1 and result['foundation_qualified']
    assert raw._current_checkpoint(config)['path']==json.loads(Path(result['readiness']['path']).read_text())['checkpoint']['path']
    # The next cycle advances the same current policy, and ordinary deficits
    # remain useful even though the declared communication cohort is complete.
    result=raw.student_cycle('plan',{},None,directory,stop=stop)
    later=[value for kind,value in calls if kind=='sft'][2:]
    assert later[0]['trainer_parameters']['communication_foundation'] is False
    assert later[-1]['trainer_parameters']['communication_foundation'] is True
    assert later[-1]['trainer_parameters']['refresh_foundation_readiness'] is True
    assert 'refresh_foundation_readiness' not in later[0]['trainer_parameters']
    assert result['cycle']==2 and [kind for kind,_ in calls].count('rl')==2


def test_raw_supervised_publish_restart_uses_same_output_and_retains_obligations(tmp_path,monkeypatch):
    directory,config,stop,calls=fixture(tmp_path,monkeypatch)
    actual=raw._publish_checkpoint;seen=[]
    def publish(config,checkpoint):
        actual(config,checkpoint);seen.append(checkpoint)
        if len(seen)==1:raise RuntimeError('controller lost after exact SFT publication')
    monkeypatch.setattr(raw,'_publish_checkpoint',publish)
    with pytest.raises(RuntimeError,match='controller lost'):raw.student_cycle('plan',{},None,directory,stop=stop)
    monkeypatch.setattr(raw,'_publish_checkpoint',actual)
    first=raw._current_checkpoint(config)
    result=raw.student_cycle('plan',{},None,directory,stop=stop)
    assert [kind for kind,_ in calls].count('rl')==1
    sft=[value for kind,value in calls if kind=='sft']
    assert sft[0]['out']==sft[1]['out']
    assert seen[0]==first and result['cycle']==1


def test_actual_raw_bridge_worker_runs_real_shared_native_learner_and_exact_resume(tmp_path,monkeypatch):
    """Only collection is a fixture; model/optimizer/RNG/carry and bridge are real."""
    from natlang_neuralese.rl import production,native_bridge,native_stage
    from test_native_rl_production import test_native_stage_real_shared_parent_update_and_exact_noop_resume as actual_trainer
    original=native_stage.main;observed=[]
    def through_bridge(argv):
        out=Path(argv[argv.index('--out')+1]);out.mkdir(parents=True,exist_ok=True)
        parent=Path(argv[argv.index('--continue-from')+1]);state_path=out/'state.json'
        config=tmp_path/'bridge-config.json'
        write_json_durable(config,{'native_state':True,'native_actor_mode':'raw-state',
            'checkpoint':str(parent),'out':str(out)})
        def worker(path,**kwargs):
            assert kwargs['defer_native_refresh'] is True
            assert original(argv)==0
            prior=json.loads(state_path.read_text()) if state_path.exists() else {'round':0}
            value={**prior,'phase':'collect','checkpoint':str(out/'checkpoint.pt'),'round':prior['round']+1}
            write_json_durable(state_path,value);return value
        monkeypatch.setattr(production,'run',worker)
        request=native_bridge.write_request(tmp_path/f'bridge-{len(observed)}',kind='round',out=out,
            config=config,state=state_path if state_path.exists() else None)
        result=request.with_suffix('.result.json')
        assert native_bridge.execute(request,result)['status']=='completed'
        value=native_bridge.verify_result(request,result)
        assert value['status']=='completed' and value['phase']=='collect'
        observed.append(value)
        return 0
    monkeypatch.setattr(native_stage,'main',through_bridge)
    actual_trainer(tmp_path,False)
    assert len(observed)==2 and observed[0]['checkpoint']
    assert 'checkpoint' not in observed[1]  # exact already-current no-op publication


def test_live_raw_owner_blocks_rendezvous_and_exact_released_parents_replace_originals(tmp_path,monkeypatch):
    from test_native_readiness import ready
    from natlang_neuralese.train.native_resident import process_owner
    from natlang_neuralese.train.native_readiness import ready_participants
    models=[];values={};requirements=[];directory=tmp_path/'raw';directory.mkdir()
    for name in ('lfm','mellum'):
        path,receipt,requirement=ready(tmp_path,name+':'+'a'*64)
        requirements.append(requirement);models.append({'model':name,'handoff_requirement':requirement})
        values[name]={'readiness':raw.binding(path),'checkpoint':{key:receipt['checkpoint'][key] for key in ('path','sha256')},
                      'handoff':{'path':'protocol-fixture','sha256':'a'*64}}
    plan=tmp_path/'plan.json';write_json_durable(plan,{'schema':'natlang.native-raw-rl-bootstrap/1','models':models})
    for name in values:
        production=directory/(name+'-production.json')
        write_json_durable(production,{'checkpoint':values[name]['checkpoint']['path']})
        config=directory/(name+'-config.json')
        write_json_durable(config,{'bootstrap_binding':{'plan_sha256':sha256_file_hex(plan)}})
        values[name].update(production_state=raw.binding(production),config=raw.binding(config))
    # Protocol-only original-owner fixture; actual boot/process/pin verification
    # has separate real-process tests. Here the existing readiness verifier runs.
    monkeypatch.setattr(raw,'verify_handoff',lambda *a,**kw:({},{}))
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
    monkeypatch.setattr(CheckpointPolicy,'load',lambda path:{'schema':CHECKPOINT_SCHEMA,
        'model_config':{'architecture':Path(path).parent.name.split(':')[0],'base_inventory_sha256':'a'*64},
        'training':{'schema':TRAINING_SCHEMA,'optimizer':{'protocol':True},'rng':{'protocol':True}}})
    state=directory/'state.json'
    record={'schema':raw.SCHEMA,'released':True,'owner':process_owner(),
            'plan_sha256':sha256_file_hex(plan),'models':values}
    write_json_durable(state,record)
    config={'out':str(tmp_path/'shared'),'raw_rl_out':str(directory),
            'raw_rl_bootstrap':raw.binding(plan),'participants':requirements}
    selected,pending=ready_participants(config)
    assert not selected and len(pending)==2
    # The same currently owned process may inspect eligibility, but the public
    # controller cannot use it until the optimizer coordinator actually exits.
    assert set(raw.validate_release(state,plan,allow_live=True))=={item['model_id'] for item in requirements}
    record['owner']={**record['owner'],'start_ticks':'0'}
    write_json_durable(state,record)
    selected,pending=ready_participants(config)
    assert {value['model_id'] for value in selected}=={item['model_id'] for item in requirements} and not pending
    changed=Path(values['lfm']['production_state']['path'])
    write_json_durable(changed,{'checkpoint':'substituted-current-parent'})
    with pytest.raises(ValueError,match='artifact changed'):raw.validate_release(state,plan)


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_real_rl_sft_rl_bridge_uses_latest_weights_and_retains_full_source_credits(tmp_path,monkeypatch,family):
    """Actual tiny native training throughout; rewards/actions are test fixtures."""
    import torch
    import os
    if family=='mellum' and os.environ.get('NATLANG_NATIVE_ADAPTER_TEST_DEVICE')!='cuda':
        pytest.skip('actual bare Mellum FP8 continuation requires declared CUDA execution')
    device='cuda' if family=='mellum' else 'cpu'
    from test_native_model import tiny_source
    from test_native_rl_replay import actor_rollout
    from test_rl_episode_graph import attach_execution_graph
    from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint,native_identity
    from natlang_neuralese.train import native_state_sft
    from natlang_neuralese.rl import native_bridge,native_stage,production
    from natlang_neuralese.rl.graph_replay import save_bundle
    from natlang_neuralese.serve.store import encode_native_state
    base=tmp_path/'base'
    if family=='lfm':tiny_source(base);dtype=torch.float32
    else:
        from test_native_full_updates import source
        dtype=source(base,family)
    _,_,roles,model_config=load_native_model(base,dtype=dtype,device=device,rank=4)
    backend_args=(['--fp8-training','triton-experts','--fp8-recipe','current','--native-linear-precision','bf16x6']
                  if family=='mellum' else [])
    op,close,co,cc=roles.ids
    row={'schema':'natlang.neuralese-native-graph/1','id':'full-row','split':'train',
        'source_groups':['train-source'],'tokenizer_fingerprint':model_config['tokenizer_fingerprint'],
        'nodes':[{'name':'writer','token_ids':[1,op,5,6,close],'target_weights':[0,1,1,1,1],'producing':[[2,4]]},
            {'name':'reader','token_ids':[3,co,cc,11,2],'target_weights':[0,1,0,1,1],
             'imports':[{'at':2,'source':'writer','start':2,'stop':4}]}]}
    held={**row,'id':'held-row','split':'test','source_groups':['protected-held']}
    train_path=tmp_path/'train.jsonl';held_path=tmp_path/'held.jsonl'
    train_path.write_text(json.dumps(row)+'\n');held_path.write_text(json.dumps(held)+'\n')
    sft_args=['--base',str(base),'--native-graphs',str(train_path),'--held-native-graphs',str(held_path),
        '--device',device,'--dtype',str(dtype).removeprefix('torch.'),'--rank','4','--stage-id','full','--epochs','2','--seed','1',
        '--learning-rate','1e-3','--weight-decay','0','--eval-every','1','--checkpoint-every','1','--refresh-at-checkpoint',
        '--forward-atol','1e-4','--gradient-atol','1e-5',*backend_args]
    from natlang_neuralese.train.recipe import native_recipe_binding
    policy=tmp_path/'policy.json'
    write_json_durable(policy,{'schema':'natlang.neuralese-training-recipe/1','id':'saved-lineage',
        'identity_qualification':'native_state_runtime_qualified',
        'stages':[{'id':'full','kind':'native_state_sft','requires':[],'parameters':{'epochs':2}}]})
    recipe_binding=native_recipe_binding(policy,'e'*64)
    assert recipe_binding['sha256']!=recipe_binding['lineage_sha256']
    sft_args+=['--recipe-binding',json.dumps(recipe_binding)]
    initial=tmp_path/'initial'
    assert native_state_sft.main([*sft_args,'--out',str(initial)])==0
    parent=initial/'checkpoint.pt';saved=torch.load(parent,weights_only=False)
    assert saved['training']['exposure_counts']['full-row']==1 and saved['training']['global_step']==1
    def executed(out):
        if family=='mellum':
            metrics=json.loads((out/'metrics.json').read_text())
            assert metrics['fp8']['has_executed'] and metrics['fp8']['row_counts']['calls']>0
            assert torch.load(out/'checkpoint.pt',weights_only=False)['training']['fp8']['backend']=='triton-grouped-experts-v1'
    executed(initial)
    master={'native_state':True,'native_actor_mode':'raw-state','checkpoint':str(parent),'out':str(tmp_path/'production')}
    Path(master['out']).mkdir();master_path=tmp_path/'master.json';write_json_durable(master_path,master)
    def policy_round(index):
        current=raw._current_checkpoint(master)
        model,tokenizer,role,config,_=load_native_checkpoint(current['path'],base=base,device=device)
        identity=native_identity(model,config,checkpoint_sha256=current['sha256'])
        _,_,_,ident,block,turns=actor_rollout(family,loaded=(model,role,identity),text=tokenizer.decode([5,6]))
        episodes=[{'task_id':'diagnostic-train-fixture','sample':index,'group':index,'split':'train',
            'policy':identity,'reward':{'quality':1},'turns':[
                {'messages':[],'target':{},'invocation_id':turn['graph_turn'],'sampling':turn['sampling'],
                 'replay':{'raw_calls_retained':True}} for turn in turns]}]
        attach_execution_graph(episodes)
        source=tmp_path/f'rollouts-{index}.jsonl';source.write_text(json.dumps(episodes[0])+'\n')
        archive=tmp_path/f'archive-{index}'
        save_bundle(episodes,archive,policy_id=current['sha256'],fetch=lambda key:encode_native_state(block))
        token_dir=tmp_path/'tokenizer';tokenizer.save_pretrained(token_dir)
        out=Path(master['out'])/f'round-{index}'/'learner'
        argv=['--rollouts',str(source),'--rl-tokenizer',str(token_dir),'--rl-mode','rft',
            '--rl-block-bundle',str(archive/'manifest.json'),'--continue-from',current['path'],
            '--held-native-graphs',str(held_path),'--out',str(out),'--device',device,'--epochs','1',
            '--rank','4','--learning-rate','1e-3','--weight-decay','0','--eval-every','1',
            '--stage-id',f'policy-{index}',*backend_args]
        state_path=Path(master['out'])/'state.json'
        def worker(path,**kwargs):
            assert native_stage.main(argv)==0
            state=json.loads(state_path.read_text()) if state_path.exists() else {'round':0,'trained_task_ids':[]}
            state.update(checkpoint=str(out/'checkpoint.pt'),round=state['round']+1,phase='collect',
                trained_task_ids=['diagnostic-train-fixture'])
            write_json_durable(state_path,state);return state
        monkeypatch.setattr(production,'run',worker)
        request=native_bridge.write_request(tmp_path/f'bridge-{index}',kind='round',out=master['out'],
            config=master_path,state=state_path if state_path.exists() else None)
        assert native_bridge.read_request(request)['parent_checkpoint']==native_bridge.binding(current['path'])
        result=request.with_suffix('.result.json');native_bridge.execute(request,result)
        assert native_bridge.verify_result(request,result)['status']=='completed'
        executed(out)
        return out/'checkpoint.pt'
    first=policy_round(0);saved=torch.load(first,weights_only=False)
    assert saved['training']['sft_progress']['full']['exposure_counts']['full-row']==1
    assert saved['training']['global_step']==2
    assert json.loads(saved['training']['options']['recipe_binding'])==recipe_binding
    continued=tmp_path/'continued'
    assert native_state_sft.main([*sft_args,'--out',str(continued),'--continue-from',str(first)])==0
    executed(continued)
    sft=continued/'checkpoint.pt';saved=torch.load(sft,weights_only=False)
    assert saved['training']['exposure_counts']['full-row']==2 and saved['training']['global_step']==3
    raw._publish_checkpoint(master,raw.binding(sft))
    state=json.loads((Path(master['out'])/'state.json').read_text())
    assert state['round']==1 and state['trained_task_ids']==['diagnostic-train-fixture']
    second=policy_round(1);saved=torch.load(second,weights_only=False)
    assert saved['training']['sft_progress']['full']['exposure_counts']['full-row']==2
    assert saved['training']['global_step']==4 and saved['training']['optimizer'] and saved['training']['rng']
    assert json.loads(saved['training']['options']['recipe_binding'])==recipe_binding
    # Explicit metadata overrides cannot silently relabel the same optimizer.
    wrong={**recipe_binding,'lineage_sha256':'f'*64}
    original=parent.read_bytes()
    rejected=tmp_path/'rejected-lineage'
    with pytest.raises(ValueError,match='preserved recipe lineage'):
        native_state_sft.main([*sft_args,'--recipe-binding',json.dumps(wrong),
            '--out',str(rejected),'--continue-from',str(parent)])
    assert parent.read_bytes()==original and not (rejected/'checkpoint.pt').exists()


def test_raw_release_follows_actual_monotone_resident_source_policy_without_moving_foundation(tmp_path,monkeypatch):
    import copy
    from test_native_resident import setup
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.train.recipe import native_recipe_binding
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    _,directory,manager,new_recipe=setup(tmp_path)
    original=manager.recipe_binding
    resident.prepare(directory,new_recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert manager.poll(7) is not None
    declaration={'handoff_requirement':{'model_id':'fixture','recipe_sha256':original['sha256'],'stage_ids':['sft']}}
    result=raw._release_requirement(declaration,{'resident_current':raw.binding(directory/'current.json')})
    assert result['recipe_sha256']==resident.current(directory)['payload']['recipe']['sha256']
    # Rebinding every receipt/hash cannot turn an unapproved parameter change
    # into resident data-only continuation: reuse the exact existing contract.
    state=copy.deepcopy(resident.current(directory))
    recipe=json.loads(new_recipe.read_text());recipe['stages'][0]['parameters']['epochs']=9
    changed=tmp_path/'changed-policy.json';write_json_durable(changed,recipe)
    bound=native_recipe_binding(changed,original['lineage_sha256'])
    state['payload']['recipe']=bound
    state['payload']['plan']['recipe']=recipe
    state['history'][-1]['recipe']=bound
    state['generation']=resident.digest(state['payload'])
    forged=tmp_path/'forged';forged.mkdir();write_json_durable(forged/'current.json',state)
    with pytest.raises(ValueError,match='parameters|runtime|objective|policy'):
        raw._release_requirement(declaration,{'resident_current':raw.binding(forged/'current.json')})


def test_live_raw_task_plan_adopts_after_sealed_operation_without_resetting_current_policy(tmp_path,monkeypatch):
    from natlang_neuralese.rl import production
    parent=tmp_path/'parent.pt';parent.write_bytes(b'original full checkpoint protocol binding')
    runtime=tmp_path/'runtime';(runtime/'dist/teacher').mkdir(parents=True)
    (runtime/'dist/teacher/cli.js').write_text('/* protocol fixture; not executed */')
    files={}
    for name in ('recipe','seed','held','cases1','cases2'):
        path=tmp_path/name;path.write_text(json.dumps({'split':'train','id':name})+'\n');files[name]=raw.binding(path)
    declaration={'model':'lfm','production_stage':'full','runtime':str(runtime),'base':str(tmp_path/'base'),'port':8097,
        'handoff_requirement':{'model_id':'lfm2:'+'a'*64},'production_recipe':files['recipe'],
        'seed_native_graphs':files['seed'],'held_native_graphs':files['held'],'cases':files['cases1']}
    witness={'path':'original-handoff','sha256':'b'*64}
    monkeypatch.setattr(raw,'verify_handoff',lambda *a,**kw:({'checkpoint':raw.binding(parent)},
        {'model_config':{'architecture':'lfm2','base_inventory_sha256':'a'*64},
         'training':{'options':{'recipe_binding':{'sha256':files['recipe']['sha256'],'lineage_sha256':'c'*64}}}}))
    calls=[]
    def configure(args):
        calls.append(args.cases)
        return {'native_state':True,'checkpoint':str(args.checkpoint),'out':str(args.out),
            'cases':str(args.cases),'production_recipe':str(args.production_recipe),
            'production_stage':args.production_stage,
            'seed_command':['python','seed','{trainer_args}'],'rows_per_round':32,
            'seed_trainer_parameters':{},'trainer_parameters':{'checkpoint_every':250}}
    monkeypatch.setattr(production,'configure_native',configure)
    plan=tmp_path/'plan';write_json_durable(plan,{'models':[declaration]})
    directory=tmp_path/'student'
    path=raw.configure_model(plan,declaration,witness,directory)
    config=json.loads(path.read_text());out=Path(config['out']);out.mkdir()
    state={'checkpoint':str(parent),'round':7,'trained_task_ids':['earlier-credit'],'trained_rounds':4}
    write_json_durable(out/'state.json',state);state_bytes=(out/'state.json').read_bytes()
    declaration={**declaration,'cases':files['cases2'],'rows_per_round':4,'supervised_quantum_updates':16}
    write_json_durable(plan,{'models':[declaration]})
    write_json_durable(directory/'cycle.json',{'phase':'sft','cycle':7,'next_stage':1})
    raw.configure_model(plan,declaration,witness,directory)
    held=json.loads(path.read_text())
    assert len(calls)==1 and held['cases']==files['cases1']['path']
    assert held['rows_per_round']==32 and held['seed_trainer_parameters']['checkpoint_every']==16
    assert held['seed_command'][-1]=='{seed_trainer_args}'
    assert held['bootstrap_binding']['plan_sha256']!=raw.binding(plan)['sha256']
    assert held['supervised_quantum_control']=={'plan':raw.binding(plan),'updates':16}
    write_json_durable(directory/'cycle.json',{'phase':'rl','cycle':8,'next_stage':0})
    raw.configure_model(plan,declaration,witness,directory)
    assert len(calls)==2 and json.loads(path.read_text())['cases']==files['cases2']['path']
    assert json.loads(path.read_text())['rows_per_round']==4
    assert json.loads(path.read_text())['trainer_parameters']['checkpoint_every']==250
    assert (out/'state.json').read_bytes()==state_bytes
    # Older raw configs acquire the authenticated lineage only after their
    # already sealed supervised operation completes, without resampling tasks.
    upgraded=json.loads(path.read_text());upgraded.pop('recipe_lineage_sha256')
    for key in ('trainer_parameters','seed_trainer_parameters'):upgraded[key].pop('recipe_binding')
    write_json_durable(path,upgraded)
    write_json_durable(directory/'cycle.json',{'phase':'sft','cycle':8,'next_stage':0})
    raw.configure_model(plan,declaration,witness,directory)
    assert 'recipe_lineage_sha256' not in json.loads(path.read_text())
    write_json_durable(directory/'cycle.json',{'phase':'rl','cycle':9,'next_stage':0})
    raw.configure_model(plan,declaration,witness,directory)
    assert json.loads(path.read_text())['recipe_lineage_sha256']=='c'*64
    assert (out/'state.json').read_bytes()==state_bytes and len(calls)==2


def test_independent_students_keep_working_until_two_ready_then_drain(tmp_path,monkeypatch):
    from threading import Event,Lock
    second_fast=Event();slow_ready=Event();lock=Lock();running=set();visits={'fast':0,'slow':0}
    plan=tmp_path/'plan.json';out=tmp_path/'out'
    write_json_durable(plan,{'schema':'natlang.native-raw-rl-bootstrap/1','models':[
        {'model':name,'handoff_receipt':{'path':'protocol'},'handoff_requirement':{'model_id':name}}
        for name in visits]})
    def student(path,declaration,witness,directory,*,stop):
        name=declaration['model'];directory.mkdir(parents=True,exist_ok=True)
        with lock:
            assert name not in running;running.add(name);visits[name]+=1;number=visits[name]
        if name=='slow':assert second_fast.wait(3),'fast student idled behind the slow cycle'
        elif number>=2:
            second_fast.set();assert slow_ready.wait(3)
        parent=directory/'parent';parent.write_text(name+str(number))
        state=directory/'production/state.json';state.parent.mkdir(exist_ok=True)
        write_json_durable(state,{'checkpoint':str(parent)})
        write_json_durable(directory/'config.json',{'out':str(state.parent)})
        if name=='slow':slow_ready.set()
        with lock:running.remove(name)
        return {'readiness':{'path':'protocol'}}
    monkeypatch.setattr(raw,'student_cycle',student)
    def released(path,plan,allow_live=False):
        if not Path(path).exists():return None
        record=json.loads(Path(path).read_text())
        if not record.get('released'):return None
        assert not running and set(record['models'])==set(visits)
        return record
    monkeypatch.setattr(raw,'validate_release',released)
    result=raw.run(plan,out,stop=SimpleNamespace(requested=False),poll_seconds=.01)
    assert result['released'] and visits['fast']>=2 and visits['slow']>=1


def test_completed_nonfoundation_stage_does_not_qualify_foundation(tmp_path,monkeypatch):
    directory,config,stop,calls=fixture(tmp_path,monkeypatch)
    original=native_rounds.round_job
    def completed(*args):
        result=original(*args)
        path=Path(result['pending_native_checkpoint']).parent/'native-training-receipt.json'
        value=json.loads(path.read_text());value.update(completed=True,foundation_ready=False)
        write_json_durable(path,value);return result
    monkeypatch.setattr(native_rounds,'round_job',completed)
    cursor={};continuation={'mandatory_sft_only':True,'out':str(directory/'ordinary-job'),
        'checkpoint':config['checkpoint'],'trainer_parameters':{'checkpoint_every':250,'refresh_at_checkpoint':True,'refresh_after_updates':250}}
    raw._supervised_job(continuation,directory/'ordinary-job',config,cursor,stop)
    assert 'foundation_qualified' not in cursor and 'readiness' not in cursor


def test_worker_refuses_actual_live_resident_owner_before_production(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_bridge,production
    from natlang_neuralese.train.native_resident import process_owner
    resident=tmp_path/'resident';resident.mkdir()
    write_json_durable(resident/'initial.json',{'owner':process_owner()})
    parent=tmp_path/'parent';parent.write_bytes(b'protocol')
    config=tmp_path/'config.json';out=tmp_path/'worker'
    write_json_durable(config,{'native_state':True,'native_actor_mode':'raw-state',
        'out':str(out),'checkpoint':str(parent),'resident_data_directory':str(resident)})
    request=native_bridge.write_request(tmp_path/'request',kind='round',out=out,config=config)
    monkeypatch.setattr(production,'run',lambda *a,**k:pytest.fail('allocated competing production worker'))
    result=native_bridge.execute(request,tmp_path/'result.json',stop=SimpleNamespace(requested=False))
    assert result['status']=='failed' and 'still alive' in result['error']


def test_immutable_recipe_copy_never_overwrites_changed_original(tmp_path):
    source=tmp_path/'authority/recipe.json';copy=tmp_path/'copy.json';copy.write_text('{"recipe":"exact"}')
    reference={'path':str(source),'sha256':sha256_file_hex(copy)}
    raw._restore_recipe_alias(reference,raw.binding(copy))
    assert source.read_bytes()==copy.read_bytes()
    source.write_text('changed-authority')
    with pytest.raises(ValueError):raw._restore_recipe_alias(reference,raw.binding(copy))
    assert source.read_text()=='changed-authority'


def test_hot_plan_failure_stops_owned_sibling_before_executor_drains(tmp_path,monkeypatch):
    import time
    from threading import Event
    started=Event();drained=Event();stop=SimpleNamespace(requested=False)
    plan=tmp_path/'plan.json';out=tmp_path/'out'
    write_json_durable(plan,{'schema':'natlang.native-raw-rl-bootstrap/1','models':[
        {'model':name,'handoff_receipt':{'path':'protocol'},'handoff_requirement':{'model_id':name}}
        for name in ('slow','hot-plan')]})
    monkeypatch.setattr(raw,'validate_release',lambda *a,**k:None)
    def student(path,declaration,witness,directory,*,stop):
        if declaration['model']=='slow':
            started.set();deadline=time.monotonic()+3
            while not stop.requested and time.monotonic()<deadline:drained.wait(.005)
            assert stop.requested,'coordinator failed without forwarding stop to its owned worker'
            drained.set();return {}
        assert started.wait(3)
        write_json_durable(plan,{'schema':'changed-invalid-plan','models':[]})
        directory.mkdir(parents=True,exist_ok=True);parent=directory/'parent';parent.write_bytes(b'protocol')
        state=directory/'production/state.json';state.parent.mkdir()
        write_json_durable(state,{'checkpoint':str(parent)})
        write_json_durable(directory/'config.json',{'out':str(state.parent)})
        return {}
    monkeypatch.setattr(raw,'student_cycle',student)
    with pytest.raises(ValueError,match='live raw plan'):
        raw.run(plan,out,stop=stop,poll_seconds=.01)
    assert drained.is_set() and stop.requested


def test_raw_continuation_does_not_execute_inactive_future_preference_stage(tmp_path,monkeypatch):
    directory,config,stop,calls=fixture(tmp_path,monkeypatch)
    path=Path(config['production_recipe']);recipe=json.loads(path.read_text())
    recipe['stages'].insert(0,{'id':'future-preference','kind':'native_offline_preference',
        'admitted':False,'requires':[],'inputs':{},'parameters':{}})
    write_json_durable(path,recipe)
    result=raw.student_cycle('plan',{},None,directory,stop=stop)
    assert result['cycle']==1
    supervised=[value for kind,value in calls if kind=='sft']
    assert supervised and all(value['production_stage']=='full'for value in supervised)


def test_declared_stage_quanta_keep_preference_and_foundation_cadences_distinct():
    config={'production_stage':'ordinary','supervised_quantum_updates':128,
            'supervised_stage_quantum_updates':{'ordinary':1000,'preference':250}}
    ordinary={'stage_id':'ordinary','eval_every':250}
    preference={'stage_id':'preference','eval_every':50}
    from natlang_neuralese.rl.native_raw_rounds import _supervised_quantum
    _supervised_quantum(config,ordinary);_supervised_quantum(config,preference)
    assert ordinary=={'stage_id':'ordinary','eval_every':250,'checkpoint_every':1000,'refresh_at_checkpoint':True,'refresh_after_updates':1000}
    assert preference=={'stage_id':'preference','eval_every':50,'checkpoint_every':250,'refresh_at_checkpoint':True,'refresh_after_updates':250}
    fallback={'stage_id':'legacy','eval_every':16};_supervised_quantum(config,fallback)
    assert fallback['checkpoint_every']==128


@pytest.mark.parametrize('mapping',[[],{'ordinary':True},{'ordinary':0},{'':250}])
def test_bad_stage_quanta_reject_before_worker_or_checkpoint_read(mapping):
    from natlang_neuralese.rl.native_raw_rounds import _supervised_quantum
    with pytest.raises(ValueError,match='stage quanta'):
        _supervised_quantum({'supervised_stage_quantum_updates':mapping},{'stage_id':'ordinary','eval_every':250})


def test_supervised_retry_rejects_output_mismatch_before_config_or_request(tmp_path,monkeypatch):
    continuation={'mandatory_sft_only':True,'out':str(tmp_path/'old-operation')}
    def never(*args,**kwargs):raise AssertionError('mismatched output must reject before sealing a request')
    monkeypatch.setattr(native_rounds,'round_job',never)
    with pytest.raises(ValueError,match='owned job directory'):
        raw._supervised_job(continuation,tmp_path/'fresh-retry',{}, {},SimpleNamespace(requested=False))
    assert not(tmp_path/'fresh-retry').exists()
    assert not(tmp_path/'old-operation').exists()
