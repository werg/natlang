"""Production scheduling plumbing; receipt fixtures do not claim task quality."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from natlang_neuralese.rl import native_supervised as supervised
from natlang_neuralese.common.hashing import sha256_file_hex


def test_mandatory_worker_uses_existing_seed_command_and_retains_partial_pending_update(tmp_path,monkeypatch):
    from natlang_neuralese.rl import production
    parent=tmp_path/'parent';parent.write_bytes(b'actual exact parent binding')
    config={'native_state':True,'mandatory_sft_only':True,'out':str(tmp_path/'job'),
        'checkpoint':str(parent),'seed_command':['same-shared-trainer','{trainer_args}'],
        'trainer_parameters':{'refresh_at_checkpoint':True,'checkpoint_every':7}}
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    calls=[]
    def trainer(command,values):
        calls.append((command,values))
        assert '--refresh-at-checkpoint' in values['trainer_args']
        out=Path(values['learner_out']);out.mkdir(parents=True)
        (out/'checkpoint.pt').write_bytes(b'actual updated weights fixture')
        (out/'report.json').write_text(json.dumps({'completed':False,'numerics_ok':True,
            'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':7},
            'mandatory_sft':{'complete':False,'remaining_graph_visits':10000}}))
    monkeypatch.setattr(production,'run_command',trainer)
    result=production.run(config_path,rounds=1,defer_native_refresh=True)
    assert len(calls)==1 and result['phase']=='awaiting_semantic_refresh'
    assert result['checkpoint']==str(parent) and result['pending_update_kind']=='mandatory_sft'
    assert result['mandatory_sft']['remaining_graph_visits']==10000
    # Restart before refresh never repeats completed optimizer work.
    again=production.run(config_path,rounds=1,defer_native_refresh=True)
    assert again==result and len(calls)==1


def test_pending_source_comparison_checks_real_source_epochs_and_completed_credits():
    sources={'native_graphs':{'sha256':'train'},'held_native_graphs':{'sha256':'held'}}
    frame={'complete':True,'train_source_sha256':'train','held_source_sha256':'held','epochs':1}
    adapter={'models':{'student':{'mandatory_sft':{'stages':{'full':frame}}}}}
    assert not supervised.stage_pending(adapter,'student','full',sources,1)
    assert supervised.stage_pending(adapter,'student','full',sources,2)
    assert supervised.stage_pending(adapter,'student','full',{**sources,'native_graphs':{'sha256':'append'}},1)
    frame['complete']=False
    assert supervised.stage_pending(adapter,'student','full',sources,1)


@pytest.mark.parametrize('kind',['native_state_sft','native_offline_preference'])
def test_interleave_shares_parent_merges_both_partial_updates_and_does_not_repeat_cycle(tmp_path,monkeypatch,kind):
    import threading
    from natlang_neuralese.rl import native_rounds
    parent=tmp_path/'parent.pt';parent.write_bytes(b'joint parent')
    adapter=tmp_path/'adapter.json'
    adapter.write_text(json.dumps({'checkpoint':{'path':str(parent),'sha256':sha256_file_hex(parent)},
        'models':{name:{} for name in ('a','b')}}))
    train=tmp_path/'train.jsonl';train.write_text('{}\n')
    held=tmp_path/'held.jsonl';held.write_text('{}\n')
    recipe=tmp_path/'recipe.json'
    objective=tmp_path/'objective.json'
    objective.write_text(json.dumps({'schema':'natlang.native-offline-pairwise-objective/1',
        'kind':'native-offline-pairwise','beta':.1,'margin':0.,'rows':{'exact-pair':{}}}))
    preference={'objective_spec':'objective'} if kind=='native_offline_preference' else {}
    recipe.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1','id':'mechanism-fixture',
        'input_bindings':{'train':{'path':str(train),'sha256':sha256_file_hex(train)},
                          'held':{'path':str(held),'sha256':sha256_file_hex(held)},
                          'objective':{'path':str(objective),'sha256':sha256_file_hex(objective)}},
        'stages':[{'id':'full','kind':kind,'requires':[],
            'inputs':{'native_graphs':'train','held_native_graphs':'held',**preference},
            'parameters':{'epochs':1,'checkpoint_every':7,'eval_every':7,'communication_foundation':True}}]}))
    configs=[{'native_state':True,'native_participant':name,'checkpoint':str(parent),
        'semantic_adapter':str(adapter),'production_recipe':str(recipe),'production_stage':'full',
        'semantic_refresh_config':str(tmp_path/'semantic.json'),'out':str(tmp_path/name),
        'seed_command':['python','-m','natlang_neuralese.train.native_state_sft']} for name in ('a','b')]
    barrier=threading.Barrier(2);calls=[]
    def job(path,config,directory,stop):
        assert config['mandatory_sft_only'] and config['checkpoint']==str(parent)
        assert config['seed_native_graphs']==str(train)
        if kind=='native_offline_preference':
            assert config['seed_command'][2]=='natlang_neuralese.train.native_preference'
            assert config['objective_spec']==str(objective)
        assert config['trainer_parameters']['refresh_at_checkpoint']
        assert 'communication_foundation' not in config['trainer_parameters']
        barrier.wait(timeout=10)
        child=Path(config['out'])/'child.pt';child.write_bytes(config['native_participant'].encode())
        calls.append(config['native_participant'])
        return {'pending_native_checkpoint':str(child)}
    monkeypatch.setattr(native_rounds,'round_job',job)
    refreshed=tmp_path/'joint.pt';refreshed.write_bytes(b'updated joint parent')
    next_adapter=tmp_path/'next-adapter.json'
    next_adapter.write_text(json.dumps({'checkpoint':{'path':str(refreshed),'sha256':sha256_file_hex(refreshed)},
        'models':{name:{'mandatory_sft':{'complete':False}} for name in ('a','b')}}))
    refreshes=[]
    def refresh(configs,source,updates,*args):
        assert source==str(adapter) and set(updates)=={'a','b'}
        refreshes.append(updates)
        return {'checkpoint':{'path':str(refreshed),'sha256':sha256_file_hex(refreshed)},
            'semantic_adapter':{'path':str(next_adapter),'sha256':sha256_file_hex(next_adapter)},
            'updated_participants':{name:{'sha256':sha256_file_hex(path)} for name,path in updates.items()}}
    monkeypatch.setattr(native_rounds,'refresh_job',refresh)
    latest=tmp_path/'latest.json';stop=SimpleNamespace(requested=False)
    result=supervised.interleave([],configs,str(adapter),tmp_path/'work',cycle=0,latest=latest,device='cpu',stop=stop)
    assert result==str(next_adapter) and sorted(calls)==['a','b'] and len(refreshes)==1
    assert json.loads(latest.read_text())['supervised_cycle']==0
    for config in configs:
        state=json.loads((Path(config['out'])/'state.json').read_text())
        assert state['checkpoint']==str(refreshed) and state['semantic_adapter']==str(next_adapter)
    assert supervised.interleave([],configs,result,tmp_path/'work',cycle=0,latest=latest,device='cpu',stop=stop)==result
    assert len(calls)==2  # resume does not redo either completed quantum


def test_pending_metadata_admission_is_not_claimed_as_optimizer_work_and_is_resumable(tmp_path,monkeypatch):
    from natlang_neuralese.rl import production
    parent=tmp_path/'parent.pt';parent.write_bytes(b'actual parent numeric fixture')
    declared=tmp_path/'pending.json';declared.write_text('{}')
    out=tmp_path/'run';config={'schema':'natlang.native-rl-production/1','out':str(out),
        'mandatory_sft_only':True,'checkpoint':str(parent),'seed_command':['shared-trainer'],
        'pending_native_sources':str(declared),'trainer_parameters':{},'native_state':True}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));calls=[]
    def trainer(command,values):
        calls.append(True);directory=Path(values['learner_out']);directory.mkdir(parents=True)
        child=directory/'checkpoint.pt';child.write_bytes(b'same numeric state plus mandatory source metadata')
        digest=sha256_file_hex(declared)
        report={'completed':False,'numerics_ok':True,'continuation_boundary':{'reached':False,'new_optimizer_updates':0},
            'mandatory_sft':{'complete':False,'stages':{'full':{'pending_source_declarations':{digest:{}}}}},
            'pending_source_admission':{'schema':'natlang.native-pending-source-admission/1',
                'parent_checkpoint_sha256':sha256_file_hex(parent),'checkpoint_sha256':sha256_file_hex(child),
                'new_optimizer_updates':0,'declarations':[digest]}}
        (directory/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(production,'run_command',trainer)
    state=production.run(path,rounds=1,defer_native_refresh=True)
    assert state['phase']=='awaiting_semantic_refresh' and not state['mandatory_sft']['complete']
    assert state['last_learner_progress']['pending_source_admission']['new_optimizer_updates']==0
    assert production.run(path,rounds=1,defer_native_refresh=True)==state and len(calls)==1


def test_offline_continuation_binds_actual_objective_and_effective_revision(tmp_path):
    from natlang_neuralese.train.native_preference import SCHEMA,KIND
    from natlang_neuralese.train.native_preference import NativePreferenceObjective
    objective=tmp_path/'objective.json'
    objective.write_text(json.dumps({'schema':SCHEMA,'kind':KIND,'beta':.1,'margin':0.,'rows':{'bound-pair':{}}}))
    train=tmp_path/'train';train.write_text('exact pair source')
    held=tmp_path/'held';held.write_text('protected ordinary source')
    recipe=tmp_path/'recipe.json'
    stage={'id':'offline','kind':'native_offline_preference','requires':[],
        'inputs':{'native_graphs':'train','held_native_graphs':'held','objective_spec':'objective'},
        'parameters':{'epochs':2,'preference_beta':.7,'preference_margin':.2,'checkpoint_every':5}}
    recipe.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1','id':'test-offline',
        'input_bindings':{name:{'path':str(path),'sha256':sha256_file_hex(path)}
                          for name,path in [('train',train),('held',held),('objective',objective)]},'stages':[stage]}))
    config={'recipe_lineage_sha256':'a'*64,'production_recipe':str(recipe),'seed_command':['python','-m','natlang_neuralese.train.native_state_sft',
        '--native-graphs','{seed_native_graphs}','{trainer_args}']}
    declaration,sources=supervised.continuation_config(config,stage,checkpoint='current',adapter='adapter',out=tmp_path/'out')
    assert declaration['seed_command'][2]=='natlang_neuralese.train.native_preference'
    assert declaration['trainer_parameters']['recipe_binding']['lineage_sha256']=='a'*64
    assert declaration['trainer_parameters']['recipe_binding']['sha256']==sha256_file_hex(recipe)
    assert declaration['objective_spec']==str(objective)
    assert '--objective-spec' in declaration['seed_command']
    contract=NativePreferenceObjective(objective,beta=.7,margin=.2).binding()
    frame={'complete':True,'train_source_sha256':sha256_file_hex(train),
           'held_source_sha256':sha256_file_hex(held),'epochs':2,'objective':contract}
    adapter={'models':{'student':{'mandatory_sft':{'stages':{'offline':frame}}}}}
    assert not supervised.stage_pending(adapter,'student','offline',sources,2,objective=contract)
    assert supervised.stage_pending(adapter,'student','offline',sources,2,objective={**contract,'beta':.8})
    assert supervised.stage_pending(adapter,'student','offline',sources,2,objective={**contract,'sha256':'new-file'})
    assert 'native_offline_preference' in supervised.MANDATORY_STAGE_KINDS


@pytest.mark.parametrize('kind',['native_state_sft','native_offline_preference'])
def test_explicit_preference_capacity_survives_emitted_command_and_prepared_request(tmp_path,monkeypatch,kind):
    """Exercise command/descriptor plumbing; the capacity runner is a numerical fixture."""
    from natlang_neuralese.rl import production
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.train import native_resident as resident
    from natlang_neuralese.train.native_preference import NativePreferenceObjective,SCHEMA,KIND
    parent=tmp_path/'parent.pt';parent.write_bytes(b'unchanged full optimizer fixture')
    train=tmp_path/'train';train.write_text('{}\n');held=tmp_path/'held';held.write_text('{}\n')
    objective_path=tmp_path/'objective.json'
    objective_path.write_text(json.dumps({'schema':SCHEMA,'kind':KIND,'beta':.1,'margin':0.,
        'rows':{'pair':{'graph_sha256':'b'*64},'reference':{'graph_sha256':'e'*64}}}))
    objective=NativePreferenceObjective(objective_path)
    source=SimpleNamespace(path=train,entries=[{'id':'pair','binding':{'sha256':'b'*64}},
        {'id':'reference','binding':{'sha256':'e'*64}}])
    examples=[SimpleNamespace(id=name,split='train',plan=SimpleNamespace(token_count=count,slot_sources=(1,)*count))
              for name,count in [('pair',8),('reference',3)]]
    source.iter_indices=lambda indices:iter(examples[index]for index in indices)
    live_objective=SimpleNamespace(kind=KIND,rows=objective.rows,binding=objective.binding,forward_inputs=lambda context:{})
    descriptor=tmp_path/'descriptor.json';descriptor.write_text(json.dumps({
        'schema':'natlang.native-preference-capacity-case/1','source':resident.binding(train),
        'objective':objective.binding(),
        'case':{'id':'pair','graph_sha256':'b'*64,'split':'train','geometry':{'forward_tokens':8,'kv_slots':8}},
        'numerical_reference':{'id':'reference','graph_sha256':'e'*64,'split':'train',
                               'geometry':{'forward_tokens':3,'kv_slots':3}}}))
    qualification=resident.binding(descriptor)
    stage={'id':'qualified-stage','kind':kind,'requires':[],
        'inputs':{'native_graphs':'train','held_native_graphs':'held',
                  **({'objective_spec':'objective'}if kind=='native_offline_preference'else{})},
        'parameters':{'epochs':1,'checkpoint_every':4,'eval_every':4,'memory_qualification':qualification}}
    recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1',
        'id':'qualification-forwarding','input_bindings':{name:resident.binding(path)for name,path in
            [('train',train),('held',held),('objective',objective_path)]},'stages':[stage]}))
    config={'native_state':True,'production_recipe':str(recipe),'production_stage':stage['id'],
        'recipe_lineage_sha256':'a'*64,'seed_command':['python','-m','natlang_neuralese.train.native_state_sft',
            '--native-graphs','{seed_native_graphs}','--out','{learner_out}','{seed_trainer_args}']}
    declared,_=supervised.continuation_config(config,stage,checkpoint=str(parent),adapter=None,out=tmp_path/'job')
    monkeypatch.setattr(resident,'_source_witness',lambda value:{'source':resident.binding(train),'indexes':[]})
    monkeypatch.setattr(memory,'capacity_identity',lambda *args:{'restored':'same-parent'})
    prepared=[];commands=[]
    def capacity(*args,**kwargs):
        reference=json.loads(args[5].memory_qualification)
        request=json.loads(resident.bound(reference).read_text())
        prepared.append(request)
        assert kwargs['checkpoint_sha256']==sha256_file_hex(parent)
        assert kwargs['objective'] is live_objective
        return {'passed':True}
    monkeypatch.setattr(memory,'_run_one_prepared',capacity)
    def trainer(command,check):
        commands.append(command)
        if kind=='native_offline_preference':
            assert command[2]=='natlang_neuralese.train.native_preference'
            assert command[command.index('--objective-spec')+1]==str(objective_path)
            args=SimpleNamespace(memory_qualification=command[command.index('--memory-qualification')+1],
                                 out=tmp_path/'job/learner')
            assert json.loads(args.memory_qualification)==qualification
            memory.run_prepared(None,None,source,None,{},args,backend=None,
                checkpoint_sha256=sha256_file_hex(parent),objective=live_objective)
        else:assert '--memory-qualification'not in command
        directory=tmp_path/'job/learner';directory.mkdir(parents=True,exist_ok=True)
        (directory/'checkpoint.pt').write_bytes(b'published update fixture')
        (directory/'report.json').write_text(json.dumps({'completed':True,'numerics_ok':True,'mandatory_sft':{'complete':True}}))
    monkeypatch.setattr(production.subprocess,'run',trainer)
    state={'checkpoint':str(parent)}
    supervised.execute(declared,state,tmp_path/'state.json',None)
    assert len(commands)==1
    if kind=='native_offline_preference':
        assert prepared[0]['cases'][0]['id']=='pair'
        assert prepared[0]['reference']['id']=='reference'
        assert prepared[0]['objective']==objective.binding()
        assert prepared[0]['identity']=={'restored':'same-parent'}
    else:assert prepared==[]
    assert parent.read_bytes()==b'unchanged full optimizer fixture'


@pytest.mark.parametrize('module',['natlang_neuralese.train.native_state_sft','natlang_neuralese.train.native_preference'])
def test_actual_resident_append_then_cold_resume_uses_current_source_roles(tmp_path,monkeypatch,module):
    from test_native_resident import setup
    from natlang_neuralese.train import native_resident as resident
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path,manifest_train=True)
    extra=tmp_path/'append-only-row.jsonl'
    extra.write_text(paths['new'].read_text().splitlines()[-1]+'\n')
    manifest=tmp_path/'actual-appended-sources.json'
    manifest.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs',**resident.binding(paths['old'])},
        {'kind':'native-graphs',**resident.binding(extra)}]}))
    value=json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(manifest)
    recipe.write_text(json.dumps(value))
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert manager.poll(7) is not None
    actual=resident.current(directory)
    stage=actual['payload']['plan']['recipe']['stages'][0]
    old_command=['python','-m',module,'--native-graphs',str(paths['old']),
        '--held-native-graphs=obsolete-held','--continued-native-graphs=obsolete-predecessor',
        '--continued-native-graphs','duplicate-obsolete-predecessor',
        '--pending-native-sources','obsolete-pending','--foundation-native-graphs','obsolete-foundation',
        '--withdrawn-native-graphs=obsolete-withdrawal','--task-eval','']
    config={'production_recipe':str(recipe),'recipe_lineage_sha256':'a'*64,'seed_command':old_command}
    continued,sources=supervised.continuation_config(config,stage,checkpoint='same-full-parent',adapter=None,out=tmp_path/'cold')
    command=continued['seed_command']
    for role in ('native_graphs','held_native_graphs','continued_native_graphs'):
        flag,key=supervised.SOURCE_ARGUMENTS[role]
        assert command.count(flag)==1
        assert command[command.index(flag)+1]=='{'+key+'}'
        assert continued[key]==sources[role]['path']
    assert continued['seed_native_graphs']==actual['payload']['train']['path']
    assert continued['continued_native_graphs']==str(paths['old'])
    for flag in ('--pending-native-sources','--foundation-native-graphs','--withdrawn-native-graphs'):
        assert not any(value==flag or value.startswith(flag+'=')for value in command)
    assert command[-2:]==['--continued-native-graphs','{continued_native_graphs}']
    assert command[command.index('--task-eval')+1]==''
    assert config['seed_command']==old_command
    assert resident.current(directory)==actual


@pytest.mark.parametrize('tail',[['--continued-native-graphs'],['--continued-native-graphs','--task-eval'],['--continued-native-graphs=']])
def test_source_normalizer_rejects_incomplete_arguments(tail):
    with pytest.raises(ValueError,match='lacks a value'):
        supervised.bind_seed_sources({'seed_command':['python','-m','natlang_neuralese.train.native_state_sft',*tail]}, {})


def test_custom_seed_contract_retains_implicit_core_sources_and_current_optional_roles():
    config={'seed_command':['custom-launcher','{trainer_args}'],'foundation_native_graphs':'stale'}
    supervised.bind_seed_sources(config,{'native_graphs':{'path':'current'},'held_native_graphs':{'path':'held'},
        'continued_native_graphs':{'path':'current-predecessor'}})
    assert config['seed_command']==['custom-launcher','{trainer_args}','--continued-native-graphs','{continued_native_graphs}']
    assert config['seed_native_graphs']=='current'and config['continued_native_graphs']=='current-predecessor'
    assert 'foundation_native_graphs'not in config
