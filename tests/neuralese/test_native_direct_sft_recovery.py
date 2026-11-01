import json
from pathlib import Path
import pytest
from natlang_neuralese.rl.native_direct_sft_recovery import recover
from natlang_neuralese.train.native_resident import binding


def test_direct_sft_recovery_preserves_history_and_rejects_unqualified(tmp_path):
    def file(name, data):
        p=tmp_path/name;p.write_text(json.dumps(data));return binding(p)
    parent=file('parent',{'old':True});child=file('child',{'new':True})
    state=tmp_path/'state.json';state.write_text(json.dumps({'phase':'mandatory_sft_pending','pending_update_kind':'mandatory_sft','checkpoint':parent['path'],'round':11,'offset':57,'trained_rounds':11}))
    boundary={'kind':'semantic-refresh','reached':False,'new_optimizer_updates':1,'checkpoint_sha256':child['sha256']}
    report=file('report',{'checkpoint':child['path'],'checkpoint_sha256':child['sha256'],'continuation_boundary':boundary,'progress_numerics':{'checkpoint_sha256':child['sha256'],'passed':True,'nonfinite':[]}})
    receipt=file('receipt',{'checkpoint':{'sha256':child['sha256']},'continuation':{'checkpoint_sha256':parent['sha256']}})
    custody=file('custody',{'schema':'natlang.earned-sft-checkpoint-custody/1','parent':parent|{'global_step':55,'optimizer_present':True,'rng_present':True},'successor':child|{'global_step':56,'optimizer_present':True,'rng_present':True},'old_bindings_and_quotas_preserved':True,'old_exposure_not_decreased':True})
    bad=file('bad',{'checkpoint_sha256':child['sha256'],'optimizer_updates':0,'report':{'checkpoint_sha256':child['sha256'],'passed':False}})
    score=file('score',{'step':56,'cases':4})
    eval_binding=file('eval-binding',{'checkpoint_sha256':child['sha256']})
    evaluation=file('evaluation',{'schema':'natlang.direct-sft-evaluation/1','checkpoint_sha256':child['sha256'],'score':score,'binding':eval_binding})
    args=dict(report_ref=report,receipt_ref=receipt,custody_ref=custody,qualification_ref=bad,evaluation_ref=evaluation,out=tmp_path/'result')
    with pytest.raises(ValueError, match='gradient qualification'):recover(state,**args)
    assert json.loads(state.read_text())['checkpoint']==parent['path']
    args['qualification_ref']=file('good',{'checkpoint_sha256':child['sha256'],'optimizer_updates':0,'report':{'checkpoint_sha256':child['sha256'],'passed':True,'complete_gradient_qualification':{'passed':True}}})
    recover(state,**args)
    saved=json.loads(state.read_text());assert saved['checkpoint']==child['path'] and saved['phase']=='collect'
    assert (saved['round'],saved['offset'],saved['trained_rounds'])==(11,57,11)
    assert json.loads((tmp_path/'result').read_text())['stage_completion_granted'] is False


def test_completed_direct_sft_refresh_publishes_only_qualified_pending_child(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    def file(name, data):
        p=tmp_path/name;p.write_text(json.dumps(data));return binding(p)
    parent=file('parent',{'old':True});child=file('child',{'new':True})
    state=tmp_path/'state.json';state.write_text(json.dumps({'phase':'awaiting_semantic_refresh','checkpoint':parent['path'],'pending_native_checkpoint':child['path'],'round':12,'offset':57,'trained_rounds':12}))
    boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':1,'checkpoint_sha256':child['sha256']}
    report=file('report',{'numerics_ok':True,'checkpoint_sha256':child['sha256'],'continuation_boundary':boundary})
    receipt=file('receipt',{'checkpoint':{'sha256':child['sha256']},'continuation':{'checkpoint_sha256':parent['sha256']}})
    common={'optimizer_present':True,'rng_present':True};custody=file('custody',{'schema':'natlang.earned-sft-checkpoint-custody/1','parent':parent|common|{'global_step':57},'successor':child|common|{'global_step':58},'old_bindings_and_quotas_preserved':True,'old_exposure_not_decreased':True})
    publish_completed(state,report_ref=report,receipt_ref=receipt,custody_ref=custody,out=tmp_path/'publication')
    s=json.loads(state.read_text());assert s['phase']=='collect' and s['checkpoint']==child['path'] and 'pending_native_checkpoint' not in s
    assert (s['round'],s['offset'],s['trained_rounds'])==(12,57,12)


def test_completed_refresh_accepts_authenticated_same_rolling_slot(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    def file(name, data):
        p=tmp_path/name;p.write_text(json.dumps(data));return binding(p)
    parent=file('parent-pin',{'old':True});child=file('rolling',{'new':True})
    prior=file('prior',{'schema':'natlang.direct-sft-refresh-publication/1','parent':{'sha256':'older'},'checkpoint':{'path':child['path'],'sha256':parent['sha256']}})
    state=tmp_path/'state.json';state.write_text(json.dumps({'phase':'awaiting_semantic_refresh','checkpoint':child['path'],'pending_native_checkpoint':child['path'],'direct_sft_refresh_history':[prior]}))
    boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':1,'checkpoint_sha256':child['sha256']}
    report=file('report',{'numerics_ok':True,'checkpoint_sha256':child['sha256'],'continuation_boundary':boundary})
    receipt=file('receipt',{'checkpoint':{'sha256':child['sha256']},'continuation':{'checkpoint_sha256':'older'}})
    common={'optimizer_present':True,'rng_present':True};custody=file('custody',{'schema':'natlang.earned-sft-checkpoint-custody/1','parent':parent|common|{'global_step':58},'successor':child|common|{'global_step':59},'old_bindings_and_quotas_preserved':True,'old_exposure_not_decreased':True})
    publish_completed(state,report_ref=report,receipt_ref=receipt,custody_ref=custody,out=tmp_path/'publication')
    assert json.loads(state.read_text())['checkpoint']==child['path']


def _session_reconciliation_fixture(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import _strong_binding, _strong_path_binding, _session_control
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.common.custody import ProtectedWorlds
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(value if isinstance(value,str) else json.dumps(value,sort_keys=True))
        return p
    def ref(path):return _strong_path_binding(path)
    out=tmp_path/'production';out.mkdir()
    checkpoint=write('production/learner/checkpoint.pt','saved full model optimizer RNG and exposure bytes')
    cp=ref(checkpoint)
    ancestor='1'*64
    launch_parent_path=write('production/learner/parent72.pt','authenticated immediate predecessor bytes')
    immediate=_strong_binding(launch_parent_path)['sha256']
    mandatory={'schema':'natlang.native-sft-progress/1','complete':False,'remaining_graph_visits':17,
        'stages':{'next':{'complete':False,'remaining_graph_visits':17}},'source_sha256':'train-source-sha'}
    state_data={'schema':'natlang.rl-production-state/1','phase':'awaiting_semantic_refresh',
        'pending_update_kind':'mandatory_sft','checkpoint':str(checkpoint),'pending_native_checkpoint':str(checkpoint),
        'pending_native_round_dir':str(out/'learner'),'round':12,'offset':41,'trained_rounds':12,
        'collection_attempt':3,'group_stats':[{'old':1}],'attempted_task_ids':['a'],'collected_task_ids':['b'],
        'trained_task_ids':['c'],'trained_supervised_tokens':987,'trained_target_tokens':654,
        'task_curriculum_history':[{'source':'old-curriculum'}],'production_complete':False,
        'mandatory_sft':{'schema':'natlang.native-sft-progress/1','complete':False,'remaining_graph_visits':999,
            'stages':{'old':{'complete':False,'remaining_graph_visits':999}},'source_sha256':'old-source'},
        'last_learner_progress':{'checkpoint_sha256':ancestor,'mandatory_sft':{'old':True},
            'continuation_boundary':{'kind':'semantic-refresh','reached':False,'new_optimizer_updates':3,
                'checkpoint_sha256':cp['sha256']}},
        'actor_policy_pending_load':'old-actor-hash','latest_reward':{'actor':'old','score':-1},
        'readiness':{'checkpoint_sha256':'old'} }
    hist=write('production/history.json',{'schema':'natlang.direct-sft-refresh-publication/1',
        'parent':{'sha256':ancestor},'checkpoint':{'sha256':ancestor}})
    state_data['direct_sft_refresh_history']=[ref(hist)]
    state_path=write('production/state.json',state_data)
    (out/'controller.lock').touch()
    common_files={}
    for key,value in {'old-runtime':'old runtime','old-recipe':'old recipe','old-cases':'old cases',
            'old-held':'old held','old-ce':'old ce','old-tokenizer':'old tokenizer','old-base':'old base'}.items():
        common_files[key]=write('old/'+key,value)
    old_config={'schema':'natlang.rl-production-config/1','id':'previous-session','native_state':True,
        'out':str(out),'checkpoint':str(common_files['old-base']),'runtime':'/old/runtime',
        'production_recipe':str(common_files['old-recipe']),'cases':str(common_files['old-cases']),
        'held_native_graphs':str(common_files['old-held']),'ce_supervision_policy':str(common_files['old-ce']),
        'rl_tokenizer':str(common_files['old-tokenizer']),'base':str(common_files['old-base']),
        'foundation_native_graphs':'/old/source','seed_native_graphs':'/old/source','task_curriculum':{'schema':'curr/1'}}
    old_config_path=write('old/config.json',old_config)
    report={'schema':'natlang.neuralese-native-sft-report/1','checkpoint':str(checkpoint),
        'checkpoint_sha256':cp['sha256'],'completed':False,'numerics_ok':True,
        'mandatory_sft':mandatory,
        'continuation':{'checkpoint_sha256':ancestor},
        'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':3,
            'checkpoint_sha256':cp['sha256']},
        'progress_numerics':{'schema':'natlang.native-finite-training-state/1','checkpoint_sha256':cp['sha256'],
            'passed':True,'nonfinite':[],'floating_tensors':{'model':1,'optimizer':1},
            'runtime_qualification_granted':False}}
    report_path=write('production/learner/report.json',report);report_ref=ref(report_path)
    receipt={'schema':'natlang.native-training-readiness/1','checkpoint':{'path':str(checkpoint),'sha256':cp['sha256']},
        'training_receipt':{'path':str(report_path),'sha256':report_ref['sha256']},
        'continuation':{'checkpoint_sha256':ancestor},'continuation_boundary':report['continuation_boundary'],
        'native_identity':{'checkpoint_sha256':cp['sha256']},'mandatory_sft':mandatory,
        'exposure':{'consumed_updates':3,'supervised_token_visits':11,'consumed_rows':37,'train_rows':190154}}
    receipt_path=write('production/learner/native-training-receipt.json',receipt);receipt_ref=ref(receipt_path)
    artifacts={}
    runtime_root=tmp_path/'runtime';runtime_root.mkdir()
    write('runtime/ts-host/node_modules/typescript/bin/tsc','compiler fixture')
    write('runtime/ts-host/package.json',{'name':'fixture-host'})
    write('runtime/ts-host/dist/teacher/collector.js','export {};')
    dep_file=write('runtime/node_modules/just-bash/package.json',{'version':'3.4.2'})
    import hashlib
    dependency_hash=hashlib.sha256(b'just-bash/package.json\0'+hashlib.sha256(dep_file.read_bytes()).hexdigest().encode()+b'\n').hexdigest()
    source_file=write('runtime/fixture.ts','source fixture')
    control_file=write('runtime/prompt-control.json',{'prompt':'pinned'})
    source_inputs={'fixture.ts':_strong_binding(source_file)['sha256']}
    tokenizer_dir=tmp_path/'assets/tokenizer';write('assets/tokenizer/tokenizer.json','tokenizer bytes')
    train_path=write('assets/train.json','train manifest')
    mandatory['stages']['next']['train_source_sha256']=ref(train_path)['sha256']
    mandatory['stages']['next']['source_hash']='sourcehash'
    report['mandatory_sft']=mandatory
    write('production/learner/report.json',report)
    report_ref=ref(report_path)
    receipt['mandatory_sft']=mandatory
    receipt['training_receipt']={'path':str(report_path),'sha256':report_ref['sha256']}
    write('production/learner/native-training-receipt.json',receipt)
    receipt_ref=ref(receipt_path)
    recipe_path=write('assets/recipe.json',{'schema':'natlang.multi-stage-recipe/1','id':'next'})
    ce_path=write('assets/ce.json','ce policy')
    held_path=write('assets/held.json','held manifest')
    runtime_bound=[control_file,train_path,recipe_path,ce_path,held_path]
    runtime_closure=write('runtime/runtime-input-closure.json',{
        'schema':'natlang.protected-runtime-input-closure/1','destination':str(runtime_root),
        'commit':'test-commit','protected_worlds_sha256':ProtectedWorlds.load().digest,
        'bound_files':[{'path':str(path),'sha256':_strong_binding(path)['sha256']} for path in runtime_bound],
        'node_build':{'mode':'isolated_typescript','compiler':'ts-host/node_modules/typescript/bin/tsc',
            'compiler_version':'test','source_inputs':source_inputs,
            'source_inputs_sha256':canonical_json_sha256_hex(source_inputs)},
        'node_dependencies':{'sha256':dependency_hash,'files':1,'bytes':dep_file.stat().st_size,
            'just_bash_version':'3.4.2','collector_module_import':'passed'}})
    admission_path=write('assets/admission.json',{'schema':'natlang.acquired-positive-graph-successor/1',
        'source':ref(train_path),'recipe':ref(recipe_path),'ce_policy':ref(ce_path),
        'current_protection_digest':ProtectedWorlds.load().digest})
    checkpoint_audit=write('assets/checkpoint-audit.json',{'schema':'natlang.earned-full-parent-cutover-audit/1',
        'status':'actual saved fullstate verified; not a capability qualification',
        'optimizer_present':True,'rng_present':True,
        'parent':{'path':str(checkpoint),'sha256':cp['sha256'],'bytes':cp['bytes'],'global_step':73,
            'source_hash':'sourcehash','source_rows':190154},
        'source_exposure':{'226':{'rows':226,'visited_rows':20,'visits':20},
            '534':{'rows':534,'visited_rows':17,'visits':17}}})
    paths={'runtime_manifest':runtime_closure,
        'production_recipe':recipe_path,
        'train_source':train_path,'rl_cases':write('assets/cases.jsonl','{}\n'),
        'held_source':held_path,'ce_policy':ce_path,
        'tokenizer_manifest':tokenizer_dir,'base_manifest':write('assets/base.json','base identity'),
        'source_admission':admission_path,
        'protection_union':runtime_closure,
        'checkpoint_audit':checkpoint_audit,
        'continuation_preflight':write('assets/preflight.json',{'schema':'natlang.native-continuation-cpu-preflight/1',
            'passed':True,'parent_sha256':immediate,'global_step':72})}
    artifacts={key:ref(path) for key,path in paths.items()}
    next_config={'schema':'natlang.rl-production-config/1','id':'successor-session','native_state':True,
        'mandatory_sft_only':False,
        'out':str(out),'checkpoint':str(checkpoint),'runtime':str(runtime_root/'ts-host'),
        'production_recipe':artifacts['production_recipe']['path'],'cases':artifacts['rl_cases']['path'],
        'held_native_graphs':artifacts['held_source']['path'],'ce_supervision_policy':artifacts['ce_policy']['path'],
        'rl_tokenizer':artifacts['tokenizer_manifest']['path'],'base':str(paths['base_manifest']),
        'foundation_native_graphs':artifacts['train_source']['path'],'seed_native_graphs':artifacts['train_source']['path'],
        'actor_environment':{'NATLANG_REPO':str(runtime_root)},
        'task_curriculum':{'schema':'curr/1','policy':'preserve'},'production_stage':'native_joint_sft_v4',
        'seed_command':['python','-m','trainer'],'trainer_parameters':{'epochs':1}}
    next_config_path=write('successor/config.json',next_config);next_config_ref=ref(next_config_path)
    plan={'schema':'natlang.rl-production-session-plan/1',
        'predecessor':{'state':ref(state_path),'config':ref(old_config_path),'checkpoint':cp,
            'report':report_ref,'training_receipt':receipt_ref},
        'successor':{'config':next_config_ref,'out':str(out),
            'control_sha256':canonical_json_sha256_hex(_session_control(next_config)),
            'immediate_parent':{'sha256':immediate,'global_step':72},'selected_next_mode':'joint_sft_and_rl',
            'artifacts':artifacts}}
    plan_path=write('plan.json',plan)
    return {'state_path':state_path,'old_config_path':old_config_path,'state_before':ref(state_path),
        'checkpoint_ref':cp,'report_ref':report_ref,'receipt_ref':receipt_ref,'next_config_ref':next_config_ref,
        'plan_ref':ref(plan_path),'out':tmp_path/'transaction','out_path':out,
        'launch_parent_path':launch_parent_path,'immediate_parent_sha256':immediate}


def test_production_session_reconciliation_stages_and_applies_exact_earned_checkpoint(tmp_path,monkeypatch):
    from natlang_neuralese.rl.native_direct_sft_recovery import (
        reconcile_production_session,apply_production_session_reconciliation)
    from natlang_neuralese.common.hashing import sha256_file_hex
    f=_session_reconciliation_fixture(tmp_path)
    old_state_bytes=f['state_path'].read_bytes()
    result=reconcile_production_session(f['state_path'],f['old_config_path'],plan_ref=f['plan_ref'],
        report_ref=f['report_ref'],training_receipt_ref=f['receipt_ref'],checkpoint_ref=f['checkpoint_ref'],
        successor_config_ref=f['next_config_ref'],out=f['out'])
    assert result['prepared_only'] is True and result['controller_state_mutated'] is False
    assert f['state_path'].read_bytes()==old_state_bytes
    staged=json.loads((f['out']/'proposed-state.json').read_text())
    assert staged['phase']=='collect' and staged['checkpoint']==f['checkpoint_ref']['path']
    assert staged['round']==12 and staged['offset']==41 and staged['trained_rounds']==12
    assert staged['collection_attempt']==3 and staged['group_stats']==[{'old':1}]
    assert staged['trained_supervised_tokens']==987 and staged['trained_target_tokens']==654
    assert staged['attempted_task_ids']==['a'] and staged['collected_task_ids']==['b']
    assert staged['task_curriculum_history']==[{'source':'old-curriculum'}]
    assert staged['mandatory_sft']['remaining_graph_visits']==17
    assert staged['historical_mandatory_sft_obligations'][-1]['obligations']['remaining_graph_visits']==999
    assert staged['last_learner_progress']['checkpoint_sha256']==f['checkpoint_ref']['sha256']
    assert staged['production_session_reconciliation']['selected_next_mode']=='joint_sft_and_rl'
    assert staged['production_session_reconciliation']['optimizer_updates_credited']==0
    assert staged['production_session_reconciliation']['capability_evaluation_pending'] is True
    assert 'actor_policy_pending_load' not in staged and 'readiness' not in staged
    assert result['archived_pending_operation']['pending_update_kind']=='mandatory_sft'
    assert result['archived_pending_operation']['last_learner_progress']['checkpoint_sha256']=='1'*64
    assert result['archived_mandatory_sft_obligations']['remaining_graph_visits']==999
    assert result['checkpoint_optimizer_rng_decoded'] is False
    assert sha256_file_hex(Path(f['checkpoint_ref']['path']))==f['checkpoint_ref']['sha256']
    assert reconcile_production_session(f['state_path'],f['old_config_path'],plan_ref=f['plan_ref'],
        report_ref=f['report_ref'],training_receipt_ref=f['receipt_ref'],checkpoint_ref=f['checkpoint_ref'],
        successor_config_ref=f['next_config_ref'],out=f['out'])==result

    applied=apply_production_session_reconciliation(f['state_path'],f['out'])
    current=json.loads(f['state_path'].read_text())
    assert applied['controller_state_mutated'] is True
    assert current['production_session_reconciliation']['applied'] is True
    assert current['production_session_reconciliation']['optimizer_updates_credited']==0
    assert current['checkpoint']==f['checkpoint_ref']['path']
    assert apply_production_session_reconciliation(f['state_path'],f['out'])['transition_id']==result['transition_id']
    from natlang_neuralese.rl.production import run
    with pytest.raises(ValueError,match='exact successor config'):
        run(f['old_config_path'],rounds=0)
    import natlang_neuralese.rl.native_supervised as supervised
    monkeypatch.setattr(supervised,'execute',lambda config,state,state_path,control:state)
    assert run(tmp_path/'successor/config.json',rounds=0)['production_session_reconciliation']['applied'] is True
    assert sha256_file_hex(Path(f['checkpoint_ref']['path']))==f['checkpoint_ref']['sha256']


def test_production_session_reconciliation_rejects_stale_state_and_parent_preflight(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import reconcile_production_session
    f=_session_reconciliation_fixture(tmp_path)
    with f['state_path'].open('a') as stream:stream.write(' ')
    with pytest.raises(ValueError,match='differs from its exact binding'):
        reconcile_production_session(f['state_path'],f['old_config_path'],plan_ref=f['plan_ref'],
            report_ref=f['report_ref'],training_receipt_ref=f['receipt_ref'],checkpoint_ref=f['checkpoint_ref'],
            successor_config_ref=f['next_config_ref'],out=f['out'])

    tmp_path2=tmp_path/'other';tmp_path2.mkdir()
    f=_session_reconciliation_fixture(tmp_path2)
    preflight=json.loads(f['plan_ref'] and Path(f['plan_ref']['path']).read_text())
    p=Path(preflight['successor']['artifacts']['continuation_preflight']['path'])
    p.write_text(json.dumps({'schema':'natlang.native-continuation-cpu-preflight/1','passed':True,
        'parent_sha256':'3'*64,'global_step':72}))
    # Re-pin the changed plan artifact but retain the planned parent identity.
    from natlang_neuralese.rl.native_direct_sft_recovery import _strong_binding
    preflight_ref=_strong_binding(p)
    preflight_plan=preflight
    preflight_plan['successor']['artifacts']['continuation_preflight']=preflight_ref
    plan_path=Path(f['plan_ref']['path']);plan_path.write_text(json.dumps(preflight_plan,sort_keys=True))
    f['plan_ref']=_strong_binding(plan_path)
    with pytest.raises(ValueError,match='immediate checkpoint parent'):
        reconcile_production_session(f['state_path'],f['old_config_path'],plan_ref=f['plan_ref'],
            report_ref=f['report_ref'],training_receipt_ref=f['receipt_ref'],checkpoint_ref=f['checkpoint_ref'],
            successor_config_ref=f['next_config_ref'],out=f['out'])


def test_production_session_reconciliation_accepts_bound_launch_parent_when_preflight_was_overwritten(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import reconcile_production_session, _strong_binding
    f=_session_reconciliation_fixture(tmp_path)
    plan=json.loads(Path(f['plan_ref']['path']).read_text())
    train_ref=plan['successor']['artifacts']['train_source']
    checkpoint=f['checkpoint_ref']
    manifest=tmp_path/'actual-launch.json'
    launch_parent_path=f['launch_parent_path']
    immediate=f['immediate_parent_sha256']
    manifest.write_text(json.dumps({'schema':'natlang.native-live-source-runtime-continuation/1',
        'parent':{'path':str(launch_parent_path),'sha256':immediate,'global_step':72},
        'source':train_ref,'runtime_commit':'test-commit',
        'argv':['python','-m','natlang_neuralese.train.native_state_sft','--continue-from',str(launch_parent_path),
            '--objective-continuation','--continued-native-graphs',train_ref['path']]}))
    old_preflight=tmp_path/'previous-preflight.json'
    old_preflight.write_text(json.dumps({'schema':'natlang.native-continuation-cpu-preflight/1',
        'passed':True,'parent_sha256':'2'*64,'global_step':72}))
    launch=tmp_path/'launch-receipt.json'
    launch.write_text(json.dumps({'schema':'natlang.live-runtime-adoption/1','status':'checks passed; launching actual GPU trainer',
        'manifest':_strong_binding(manifest),'preflight':_strong_binding(old_preflight),
        'parent':{'path':str(launch_parent_path),'sha256':immediate,'global_step':72},
        'runtime_commit':'test-commit','source':train_ref}))
    plan['successor']['artifacts']['continuation_preflight']=_strong_binding(launch)
    plan_path=Path(f['plan_ref']['path']);plan_path.write_text(json.dumps(plan,sort_keys=True))
    f['plan_ref']=_strong_binding(plan_path)
    valid_launch=launch.read_text()
    bad_receipt=json.loads(valid_launch);bad_receipt['source']['sha256']='f'*64
    launch.write_text(json.dumps(bad_receipt))
    plan['successor']['artifacts']['continuation_preflight']=_strong_binding(launch)
    plan_path.write_text(json.dumps(plan,sort_keys=True));f['plan_ref']=_strong_binding(plan_path)
    with pytest.raises(ValueError,match='preflight or launch evidence'):
        reconcile_production_session(f['state_path'],f['old_config_path'],plan_ref=f['plan_ref'],
            report_ref=f['report_ref'],training_receipt_ref=f['receipt_ref'],checkpoint_ref=f['checkpoint_ref'],
            successor_config_ref=f['next_config_ref'],out=f['out'])
    launch.write_text(valid_launch)
    plan['successor']['artifacts']['continuation_preflight']=_strong_binding(launch)
    plan_path.write_text(json.dumps(plan,sort_keys=True));f['plan_ref']=_strong_binding(plan_path)
    result=reconcile_production_session(f['state_path'],f['old_config_path'],plan_ref=f['plan_ref'],
        report_ref=f['report_ref'],training_receipt_ref=f['receipt_ref'],checkpoint_ref=f['checkpoint_ref'],
        successor_config_ref=f['next_config_ref'],out=f['out'])
    assert result['prepared_only'] is True
