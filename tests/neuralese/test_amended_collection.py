import hashlib
import json
from pathlib import Path
import pytest
from natlang_neuralese.rl.amended_collection import resume_amended_collection
from natlang_neuralese.rl.production import adopt_completed_collection_withdrawal


def test_interrupted_group_projection_is_authenticated(tmp_path):
    out=tmp_path/'round';out.mkdir();policy='a'*64
    def bind(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    def write(name,value):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value)+'\n');return p
    cases=write('selected-cases.jsonl',{'id':'task'})
    binding={'schema':'natlang.rl-collection-binding/1','policy_id':policy,'cases_sha256':bind(cases)['sha256'],
        'rows':1,'offset':0,'group':2,'root_mode':'student','seed_base':700000,'case_seconds':600,
        'max_model_requests':128,'max_turns':160,'neuralese_server':False,'temperature':1.,'guided':False,'endpoints':['actor']}
    write('collection-binding.json',binding)
    write('score.json',{'collection_complete':False,'groups':[{'task_id':'task','scored':2,'spread':True}],
         'actor_receipts':{}})
    for name in ['packet.jsonl','journal.jsonl','preparation.json','sampling-config.json']:write(name,{})
    entries=[];rows=[]
    for sample in range(2):
        result=write(f'result-{sample}.jsonl',{})
        entries.append({'group':0,'sample':sample,'index':0,'task_id':'task','output':str(result)})
        rows.append({'group':0,'sample':sample,'task_id':'task','policy':{'checkpoint_sha256':policy},
                     'reward':{'quality':sample},'turns':[{}]})
    (out/'queue.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
    (out/'rollouts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    projected=tmp_path/'projection.jsonl';projected.write_bytes((out/'rollouts.jsonl').read_bytes())
    names=['collection-binding.json','score.json','rollouts.jsonl','packet.jsonl','queue.jsonl','journal.jsonl',
           'preparation.json','sampling-config.json','result-0.jsonl','result-1.jsonl']
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'schema':'natlang.amended-collection-continuation/1',
         'out':str(out),'groups':[0],'files':{n:bind(out/n) for n in names},'filtered_rollouts':bind(projected)}))
    request={'out':out,'cases':cases,'policy_id':policy,'rows':1,'group':2,'endpoints':['actor'],'neuralese_server':False}
    result=resume_amended_collection(bind(proof),request,validate_actors=lambda *a,**kw:None)
    assert result['scored']==2 and result['groups_with_spread']==1 and result['collection_complete'] is False
    rows[0]['reward']['quality']=1;projected.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    p=json.loads(proof.read_text());p['filtered_rollouts']=bind(projected);proof.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='projection'):
        resume_amended_collection(bind(proof),request,validate_actors=lambda *a,**kw:None)

def test_policy_arguments_do_not_apply_sft_communication_quota():
    from natlang_neuralese.rl.production import native_policy_trainer_args
    from natlang_neuralese.train.recipe import stage_parameter_args
    parameters={'communication_min_visits':7,'pending_focus':'focus.json','sft_packed_geometry':{'path':'sft'},'foundation_communication_batches':1,'learning_rate':1e-6}
    args=native_policy_trainer_args(parameters,{})
    assert all(flag not in args for flag in ('--communication-min-visits','--pending-focus','--sft-packed-geometry','--foundation-communication-batches'))
    assert '--communication-min-visits' in stage_parameter_args(parameters)
    assert parameters['communication_min_visits']==7

def test_policy_quota_frame_excludes_prior_sft_obligations():
    from natlang_neuralese.train.native_continuation import policy_pending_frame
    frame={'source_row_bindings':{'policy':'binding'},'source_row_order':['policy'],
           'exposure_counts':{},'options':{'epochs':1,'_objective_required_visits':{'old-sft':7},
           '_pending_native_source_declarations':[{'binding':{'sha256':'old'},'rows':{}}]}}
    projected=policy_pending_frame(frame)
    assert projected['required_visits']=={'policy':1}
    assert not projected.get('pending_source_declarations')
    assert frame['options']['_objective_required_visits']=={'old-sft':7}

def test_policy_cli_rejects_direct_sft_override_and_unknown_control():
    from natlang_neuralese.rl.production import validate_native_policy_trainer_cli
    validate_native_policy_trainer_cli(['--batch-tokens','32768','--async-checkpoint'])
    for flag in ('--pending-focus','--sft-packed-geometry','--communication-min-visits','--unknown-control'):
        with pytest.raises(ValueError,match='unclassified native RL trainer flag'):
            validate_native_policy_trainer_cli([flag,'x'])

def test_captured_logprob_probe_reports_exact_deltas(tmp_path):
    import torch
    from natlang_neuralese.rl.native_replay import behavior_logprob_probe
    class Model:
        def logits(self,states):return states
    hidden=torch.tensor([[1.,2.,3.],[0.,1.,0.]])
    actions=[{'query':0,'token_id':2,'temperature':1.,'allowed_token_ids':None,'behavior_logp':-1.},
             {'query':1,'token_id':1,'temperature':1.,'allowed_token_ids':None,'behavior_logp':-2.}]
    p=tmp_path/'probe.json';r=behavior_logprob_probe(Model(),hidden,actions,graph_id='g',policy={'checkpoint_sha256':'a'*64},destination=p)
    assert len(r['positions'])==2 and r['max_abs_delta']>0
    assert json.loads(p.read_text())==r


def test_zero_signal_jolt_projection_preserves_active_and_privileged_graphs():
    from natlang_neuralese.rl.native_stage import skip_zero_signal_jolt
    graphs=[{'id':x} for x in ('zero','outcome','privileged')]
    rows={'zero':{'advantage':0.,'jolt':{'role':'student','privileged':None}},
          'outcome':{'advantage':1.,'jolt':{'role':'student','privileged':None}},
          'privileged':{'advantage':0.,'jolt':{'role':'teacher','privileged':{'state':'x'}}}}
    objective={'mode':'grpo','policy_loss':{'schema':'jolt-plus/1'},'rows':rows}
    kept,projected,receipt=skip_zero_signal_jolt(graphs,objective)
    assert [x['id'] for x in kept]==['outcome','privileged']
    assert set(projected['rows'])=={'outcome','privileged'} and set(rows)=={'zero','outcome','privileged'}
    assert receipt['omitted_graph_visits']==0 and receipt['omitted_graph_ids']==['zero']
    assert 'active episodes' in receipt['normalization']
    empty,_,all_zero=skip_zero_signal_jolt(graphs[:1],{**objective,'rows':{'zero':rows['zero']}})
    assert not empty and all_zero['omitted_graph_visits']==0


def test_no_signal_round_advances_collection_without_training_credit(tmp_path):
    from natlang_neuralese.rl.production import commit_no_signal_round
    from natlang_neuralese.common.hashing import sha256_file_hex
    rollout=tmp_path/'rollouts.jsonl';rollout.write_text('{}\n');parent=tmp_path/'parent.pt';parent.write_bytes(b'parent')
    policy='a'*64;loss={'schema':'jolt-plus/1'};report=tmp_path/'zero-signal-report.json'
    report.write_text(json.dumps({'schema':'natlang.native-rl-no-signal/1','policy_sha256':policy,'policy_loss':json.dumps(loss),
        'rollouts':{str(rollout):sha256_file_hex(rollout)},'optimizer_updates':0,'graph_visits':0,
        'projection':{'active_count':0,'omitted_graph_visits':0}}))
    state={'round':10,'offset':57,'trained_rounds':7,'checkpoint':str(parent),'active_update_round':'x','update_mode':'grpo'}
    commit_no_signal_round(state,report_path=report,rollouts_path=rollout,policy_id=policy,policy_loss=loss,next_offset=58)
    assert state['round']==11 and state['offset']==58 and state['trained_rounds']==7 and state['checkpoint']==str(parent)
    assert state['no_signal_history'][0]['optimizer_updates']==0 and 'active_update_round' not in state


def test_completed_update_accepts_only_exact_amendment_projection(tmp_path):
    from natlang_neuralese.rl.production import verify_amended_rollout
    from natlang_neuralese.common.hashing import sha256_file_hex
    round_dir=tmp_path/'round';round_dir.mkdir();projection=tmp_path/'projection';projection.write_text('sample')
    def bind(path):return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'schema':'natlang.amended-collection-continuation/1',
        'out':str(round_dir),'filtered_rollouts':bind(projection)}));ref=bind(proof)
    assert verify_amended_rollout(ref,round_directory=round_dir,rollout_path=projection)['out']==str(round_dir)
    with pytest.raises(ValueError,match='authenticated amendment'):
        verify_amended_rollout(None,round_directory=round_dir,rollout_path=projection)
    projection.write_text('tamper')
    with pytest.raises(ValueError):verify_amended_rollout(ref,round_directory=round_dir,rollout_path=projection)


def test_completed_collection_withdrawal_keeps_other_complete_groups_and_binds_stale_toolace(tmp_path):
    from natlang_neuralese.rl.task_source import _canonical_row_sha256
    out=tmp_path/'round';out.mkdir();policy='b'*64
    def bind(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    def write(name,value):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value if isinstance(value,str) else json.dumps(value)+'\n');return path
    retained={'id':'math-task','split':'train','source_ids':['math:1'],'source_groups':['math:1'],
        'source_revisions':['current'],'curriculum':{'external':{'source':'math'}},'semantics':{}}
    stale={'id':'toolace-task','split':'train','source_ids':['Team-ACE/ToolACE:3509'],'source_groups':['toolace:3509'],
        'source_revisions':['natlang.broad_external_rl_tasks/1','Team-ACE/ToolACE@6bda777'],
        'curriculum':{'external':{'source':'toolace'}},'semantics':{}}
    cases=tmp_path/'cases.jsonl';cases.write_text(''.join(json.dumps(row)+'\n' for row in [retained,stale]))
    selected=write('selected-cases.jsonl',''.join(json.dumps(row)+'\n' for row in [retained,stale]))
    write('packet.jsonl',selected.read_text())
    binding={'schema':'natlang.rl-collection-binding/1','policy_id':policy,'cases_sha256':bind(cases)['sha256'],
        'rows':2,'offset':0,'group':2,'root_mode':'student','seed_base':700000,'case_seconds':600,
        'max_model_requests':128,'max_turns':160,'neuralese_server':False,'temperature':1.,'guided':False,'endpoints':['actor']}
    write('collection-binding.json',binding)
    score={'collection_complete':True,'groups':[{'task_id':'math-task','scored':2,'outcomes':['task_success']*2,'spread':True},
        {'task_id':'toolace-task','scored':2,'outcomes':['task_success']*2,'spread':True}],'actor_receipts':{},'scored':4,'rollouts':4}
    write('score.json',score)
    for name in ['journal.jsonl','preparation.json','sampling-config.json']:write(name,{})
    queue=[];rollouts=[];names=['collection-binding.json','score.json','rollouts.jsonl','packet.jsonl','queue.jsonl',
        'journal.jsonl','preparation.json','sampling-config.json','selected-cases.jsonl']
    for group,task_id in enumerate(['math-task','toolace-task']):
        for sample in range(2):
            result=write(f'results/{group}-{sample}.json',{})
            names.append(result.relative_to(out).as_posix())
            queue.append({'group':group,'sample':sample,'index':group,'task_id':task_id,'output':str(result)})
            rollouts.append({'group':group,'sample':sample,'task_id':task_id,'policy':{'checkpoint_sha256':policy},
                'outcome':'task_success','reward':{'quality':1},'turns':[{}]})
    write('queue.jsonl',''.join(json.dumps(row)+'\n' for row in queue))
    write('rollouts.jsonl',''.join(json.dumps(row)+'\n' for row in rollouts))
    filtered=tmp_path/'retained-rollouts.jsonl';filtered.write_text(''.join(json.dumps(row)+'\n' for row in rollouts[:2]))
    withdrawal={'group':1,'task_id':'toolace-task','row_sha256':_canonical_row_sha256(stale),
        'reason':'stale-toolace-schema-contract','required_schema_contract':'natlang.toolace-schema-contract/1'}
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'schema':'natlang.completed-collection-withdrawal/1',
        'out':str(out),'groups':[0],'withdrawals':[withdrawal],'files':{name:bind(out/name) for name in names},
        'source_cases':bind(cases),
        'filtered_rollouts':bind(filtered)}))
    request={'out':out,'cases':cases,'policy_id':policy,'rows':2,'group':2,'endpoints':['actor'],'neuralese_server':False}
    result=resume_amended_collection(bind(proof),request,validate_actors=lambda *a,**kw:None)
    assert result['collection_complete'] is True and result['historical_collection_complete'] is True
    assert result['amended_complete_groups']==[0] and result['withdrawn_groups']==[withdrawal]
    assert result['scored']==2 and result['successes']==2
    bad=json.loads(proof.read_text());bad['withdrawals'][0]['row_sha256']='0'*64
    proof.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='withdrawal proof'):
        resume_amended_collection(bind(proof),request,validate_actors=lambda *a,**kw:None)


def test_completed_withdrawal_preserves_privileged_rows_and_unknown_native_failure(tmp_path):
    from natlang_neuralese.rl.task_source import _canonical_row_sha256
    out=tmp_path/'round';out.mkdir()
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'checkpoint-parent')
    policy=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    def bind(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    def write(name,value,raw=False):
        path=out/name;path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,bytes):data=value
        elif raw or isinstance(value,str):data=value.encode()
        else:data=(json.dumps(value)+'\n').encode()
        path.write_bytes(data)
        return path
    ids=['math-task','toolace-task','kk-task','gsm-task']
    sources=['math','toolace','knights-knaves','gsm8k']
    cases_rows=[]
    for ident,source in zip(ids,sources):
        revisions=['natlang.broad_external_rl_tasks/1']
        source_ids=[source+':1']
        external={'source':source}
        if source=='toolace':source_ids=['Team-ACE/ToolACE:3509']
        cases_rows.append({'id':ident,'split':'train','source_ids':source_ids,'source_groups':[source+':1'],
            'source_revisions':revisions,'curriculum':{'external':external},'semantics':{}})
    cases=tmp_path/'cases.jsonl';cases.write_text(''.join(json.dumps(row)+'\n' for row in cases_rows))
    selected=write('selected-cases.jsonl',''.join(json.dumps(row)+'\n' for row in cases_rows))
    write('packet.jsonl',selected.read_text())
    binding={'schema':'natlang.rl-collection-binding/1','policy_id':policy,'cases_sha256':bind(cases)['sha256'],
        'rows':4,'offset':0,'group':4,'jolt_teacher_attempts':1,'root_mode':'student','seed_base':700000,
        'case_seconds':600,'max_model_requests':128,'max_turns':160,'neuralese_server':False,
        'temperature':1.,'guided':False,'endpoints':['actor']}
    write('collection-binding.json',binding)
    groups=[{'task_id':'math-task','scored':4,'outcomes':['task_success']*4,'spread':False},
        {'task_id':'toolace-task','scored':4,'outcomes':['invalid_action']+['semantic_mismatch']*3,'spread':False},
        {'task_id':'kk-task','scored':3,'outcomes':['task_success']*3+['unresolved'],'spread':False},
        {'task_id':'gsm-task','scored':4,'outcomes':['task_success']*4,'spread':False}]
    write('score.json',{'collection_complete':True,'groups':groups,'actor_receipts':{},'scored':19,'rollouts':20})
    write('preparation.json',{});write('sampling-config.json',{})
    queue=[];rollouts=[];names=['collection-binding.json','score.json','rollouts.jsonl','packet.jsonl','queue.jsonl',
        'journal.jsonl','preparation.json','sampling-config.json','selected-cases.jsonl']
    for group,ident in enumerate(ids):
        for sample in range(5):
            is_unresolved=(group==2 and sample==0)
            name=f'row-{group:03}/sample-{sample:02}/results.jsonl'
            result=write(name,b'',raw=True) if is_unresolved else write(name,'{}\n',raw=True)
            names.append(result.relative_to(out).as_posix())
            entry={'group':group,'sample':sample,'index':group,'task_id':ident,'output':str(result),
                'key':f'task-{group}-{sample}'}
            if sample==4:entry['jolt_role']='teacher'
            queue.append(entry)
            unresolved=is_unresolved
            outcome=('unresolved' if unresolved else 'invalid_action' if group==1 and sample==0
                     else 'semantic_mismatch' if group==1 else 'task_success')
            rollouts.append({'group':group,'sample':sample,'task_id':ident,'policy':{'checkpoint_sha256':policy},
                'outcome':outcome,'unscored':'unresolved' if unresolved else None,
                'jolt_role':'teacher' if sample==4 else 'student',
                'reward':None if unresolved else {'quality':1.0},'turns':[] if unresolved else [{}]})
    error_text='Error: neuralese-native-grad-invalid: native reader lacks its authenticated in-episode writer'
    error_path=write('row-002/sample-00/jobs/000002.error.json',{'index':2,'program_ir_sha256':'d'*64,
        'source_sha256':binding['cases_sha256'],'error':error_text})
    names.append(error_path.relative_to(out).as_posix())
    error_ref=bind(error_path)
    terminal={'path':str(error_path),'sha256':error_ref['sha256'],'error':error_text,
        'program_ir_sha256':'d'*64,'source_sha256':binding['cases_sha256'],'record_binding_verified':True}
    journal=write('journal.jsonl',json.dumps({'event':'finish','key':'task-2-0',
        'output_accounting':{'job_states':[{'terminal_error':terminal}]}})+'\n')
    write('queue.jsonl',''.join(json.dumps(row)+'\n' for row in queue))
    write('rollouts.jsonl',''.join(json.dumps(row)+'\n' for row in rollouts))
    filtered=tmp_path/'retained-rollouts.jsonl'
    filtered.write_text(''.join(json.dumps(row)+'\n' for row in rollouts if row['group'] in {0,3}))
    withdrawals=[{'group':1,'task_id':'toolace-task','row_sha256':_canonical_row_sha256(cases_rows[1]),
        'reason':'stale-toolace-schema-contract','required_schema_contract':'natlang.toolace-schema-contract/1'},
        {'group':2,'task_id':'kk-task','row_sha256':_canonical_row_sha256(cases_rows[2]),
         'reason':'incomplete-native-execution-group','unresolved_samples':[{'sample':0,
            'error_path':'row-002/sample-00/jobs/000002.error.json','error_sha256':error_ref['sha256'],
            'error':error_text,'program_ir_sha256':'d'*64,'source_sha256':binding['cases_sha256'],
            'record_binding_verified':True}]}]
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'schema':'natlang.completed-collection-withdrawal/1',
        'out':str(out),'groups':[0,3],'withdrawals':withdrawals,'files':{name:bind(out/name) for name in names},
        'source_cases':bind(cases),'filtered_rollouts':bind(filtered)}))
    request={'out':out,'cases':cases,'policy_id':policy,'rows':4,'group':4,'endpoints':['actor'],'neuralese_server':False}
    result=resume_amended_collection(bind(proof),request,validate_actors=lambda *a,**kw:None)
    assert result['collection_complete'] is True and result['amended_complete_groups']==[0,3]
    assert result['scored']==10 and result['successes']==10 and len(result['groups'])==2
    assert len(result['withdrawn_groups'])==2

    guard=tmp_path/'hold.json'
    guard.write_text(json.dumps({'schema':'natlang.live-invalid-task-update-hold/1',
        'optimizer_updates_credited':0,'config_before_sha256':'a'*64,'config_after_sha256':'b'*64,
        'affected_task_id':'toolace-task'}))
    state={'round':10,'phase':'updating','active_update_round':str(out),
        'checkpoint':str(checkpoint),'update_rollouts':str(out/'rollouts.jsonl'),
        'update_mode':'grpo','update_policy_loss':{'schema':'jolt-plus/1'},
        'update_stages':['all'],'last_learner_progress':{'checkpoint_sha256':policy},
        'task_curriculum_history':{
            'math-task':{'task_id':'math-task','route':'retire','scored':4,'observed_round':9},
            'toolace-task':{'task_id':'toolace-task','route':'rejection-sampling','scored':4,
                            'outcomes':['invalid_action','semantic_mismatch'],'observed_round':10},
            'kk-task':{'task_id':'kk-task','route':'retire','scored':3,'outcomes':['task_success']*3+['unresolved'],
                       'observed_round':10}},
        'group_stats':[{'task_id':'math-task','scored':4}, {'task_id':'toolace-task','scored':4},
                       {'task_id':'kk-task','scored':3}]}
    reference=bind(proof)
    adopt_completed_collection_withdrawal(state,round_directory=out,report=result,
        amendment_ref=reference,no_update_evidence=bind(guard))
    assert state['update_rollouts']==str(filtered.resolve())
    assert state['update_amendment_proof']==reference
    assert state['update_mode']=='grpo' and state['update_policy_loss']=={'schema':'jolt-plus/1'}
    assert state['collection_withdrawal_history'][0]['optimizer_updates_credited']==0
    assert state['collection_withdrawal_history'][0]['original_rollouts']['bytes']==(out/'rollouts.jsonl').stat().st_size
    assert set(state['task_curriculum_history']) == {'math-task'}
    assert [row['task_id'] for row in state['group_stats']] == ['math-task']
    archived=state['collection_withdrawal_history'][0]['withdrawn_selector_observations']
    assert set(archived['task_curriculum_history']) == {'toolace-task','kk-task'}
    assert [row['task_id'] for row in archived['group_stats']] == ['toolace-task','kk-task']
    before=len(state['collection_withdrawal_history'])
    adopt_completed_collection_withdrawal(state,round_directory=out,report=result,
        amendment_ref=reference,no_update_evidence=bind(guard))
    assert len(state['collection_withdrawal_history'])==before

    blocked={**state,'update_rollouts':str(out/'rollouts.jsonl'),'collection_withdrawal_history':[]}
    (out/'learner').mkdir()
    with pytest.raises(ValueError,match='learner outputs'):
        adopt_completed_collection_withdrawal(blocked,round_directory=out,report=result,
            amendment_ref=reference,no_update_evidence=bind(guard))


def test_full_behavior_logprob_gate_checks_late_action_before_return(tmp_path):
    import torch
    from natlang_neuralese.rl.native_replay import behavior_logprob_probe
    class Model:
        def logits(self,states):return states
    hidden=torch.tensor([[1.,2.,3.]]).repeat(20,1)
    expected=float(torch.log_softmax(hidden[0],dim=0)[2])
    actions=[{'query':i,'token_id':2,'temperature':1.,'allowed_token_ids':None,'behavior_logp':expected}
             for i in range(20)]
    passed=behavior_logprob_probe(Model(),hidden,actions,graph_id='full',policy={'checkpoint_sha256':'a'*64},
        destination=tmp_path/'passed.json',limit=None,atol=1e-4)
    assert passed['passed'] and passed['all_episode_actions'] and passed['action_count']==20
    actions[-1]['behavior_logp']-=0.1
    with pytest.raises(ValueError,match='parity failed before SGD'):
        behavior_logprob_probe(Model(),hidden,actions,graph_id='full',policy={'checkpoint_sha256':'a'*64},
            destination=tmp_path/'failed.json',limit=None,atol=1e-4)
    failed=json.loads((tmp_path/'failed.json').read_text())
    assert not failed['passed'] and failed['action_count']==20 and failed['positions'][-1]['action_index']==19


def test_behavior_parity_only_applies_to_exact_loaded_behavior_parent():
    from natlang_neuralese.rl.native_replay import NativePolicyObjective
    obj=NativePolicyObjective.__new__(NativePolicyObjective)
    obj.identity={'checkpoint_sha256':'a'*64};obj.evaluation_identity=obj.identity.copy()
    obj.adapter_binding=None;obj.evaluation_adapter_binding=None;obj.behavior_policy=None
    obj.bind_initial_checkpoint('a'*64)
    assert obj.can_compare_behavior_actor({})
    assert not obj.can_compare_behavior_actor({'actor_checkpoint_sha256':'c'*64})
    obj.bind_initial_checkpoint('b'*64)
    assert not obj.can_compare_behavior_actor({})
    with pytest.raises(ValueError,match='loaded checkpoint'):
        obj.bind_initial_checkpoint('not-a-sha')


def test_zero_signal_projection_omits_twins_whose_role_coefficient_is_zero():
    """Review F5 (2026-11-09): with beta_s 0 a zero-advantage student twin carries no loss term."""
    from natlang_neuralese.rl.native_stage import skip_zero_signal_jolt
    graphs=[{'id':x} for x in ('student','teacher')]
    rows={'student':{'advantage':0.,'jolt':{'role':'student','privileged':{'state':'x'}}},
          'teacher':{'advantage':0.,'jolt':{'role':'teacher','privileged':{'state':'y'}}}}
    objective={'mode':'grpo','policy_loss':{'schema':'jolt-plus/1','beta_s':0.,'beta_t':.5},'rows':rows}
    kept,_,receipt=skip_zero_signal_jolt(graphs,objective)
    assert [x['id'] for x in kept]==['teacher'] and receipt['omitted_graph_ids']==['student']
