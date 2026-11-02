import copy
import pytest
import torch
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.native_execution_transition import derive, verify_states


def parent():
    return {'schema': CHECKPOINT_SCHEMA, 'model_config': {'dtype':'torch.bfloat16','architecture':'lfm3'},
        'interface_state': {'adapter': {'up':torch.tensor([1.])}},
        'backbone_state': {'weight':torch.tensor([3.])},
        'training': {'schema':TRAINING_SCHEMA,'global_step':67,'optimizer':{'state':torch.tensor([2.])},
                     'rng':{'torch':torch.tensor([42])},'exposure_counts':{'row':4},
                     'sft_progress':{'stage':{'quota':99}},'options':{'source':'bound'}}}


def test_exact_transition_and_tamper_rejection():
    before=parent(); source=copy.deepcopy(before); child=derive(before,parent_sha256='a'*64)
    report=verify_states(source,child,parent_sha256='a'*64)
    assert report['optimizer_updates']==0 and report['fresh_collection_required'] is True
    assert before==source
    assert child['training']['exposure_counts']=={'row':4}
    for key in ('backbone_state','interface_state','training'):
        altered=copy.deepcopy(child)
        if key=='backbone_state':altered[key]['weight'][0]=4
        elif key=='interface_state':altered[key]['adapter']['up'][0]=3
        else:altered[key]['optimizer']['state'][0]=9
        with pytest.raises(ValueError):verify_states(source,altered,parent_sha256='a'*64)
    with pytest.raises(ValueError):derive(child,parent_sha256='a'*64)


def test_file_verification_reloads_and_rejects_changed_bytes(tmp_path, monkeypatch):
    import hashlib
    from natlang_neuralese.rl import native_raw_policy
    from natlang_neuralese.train import native_execution_transition as tx
    from natlang_neuralese.train.native_execution_transition import verify_files
    source=parent();child=derive(source,parent_sha256='a'*64)
    def ref(path):
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    a=tmp_path/'parent.pt';b=tmp_path/'child.pt';torch.save(source,a)
    parent_ref=ref(a)
    child=derive(source,parent_sha256=parent_ref['sha256']);torch.save(child,b)
    monkeypatch.setattr(native_raw_policy,'saved_native_identity',lambda state,checkpoint_sha256:
        {'checkpoint_sha256':checkpoint_sha256,'profile':state['model_config'].get('native_shared_execution','source-v1')})
    comparisons=[]
    compare=tx.verify_states
    def observed_compare(*args, **kwargs):
        comparisons.append(True)
        return compare(*args, **kwargs)
    monkeypatch.setattr(tx, 'verify_states', observed_compare)
    child_ref=ref(b)
    proof=verify_files(parent_ref,child_ref)
    assert proof['actor_identity']['old']['profile']=='source-v1'
    assert proof['actor_identity']['new']['profile']=='ffn-stable-v1'
    proof['actor_identity']['new']['profile']='caller mutation'
    assert verify_files(parent_ref,child_ref)['actor_identity']['new']['profile']=='ffn-stable-v1'
    assert len(comparisons)==1
    # Even identical bytes require a new tensor comparison after an inode's
    # checked metadata changes. A wrong digest never gets a cache hit.
    import os
    stat=b.stat();os.utime(b, ns=(stat.st_atime_ns,stat.st_mtime_ns+1))
    verify_files(parent_ref,child_ref)
    assert len(comparisons)==2
    with pytest.raises(ValueError):verify_files(parent_ref,{**child_ref,'sha256':'0'*64})
    bad=copy.deepcopy(child);bad['training']['exposure_counts']['row']+=1;torch.save(bad,b)
    with pytest.raises(ValueError):verify_files(parent_ref,ref(b))


def test_diagnostic_publication_does_not_replace_controller_head(tmp_path, monkeypatch):
    import hashlib, json
    from natlang_neuralese.rl import native_raw_policy
    from natlang_neuralese.train.native_execution_transition import verify_files, publish_diagnostic, verify_candidate_publication
    def ref(path):
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    source=parent(); old=tmp_path/'parent.pt';child=tmp_path/'child.pt';torch.save(source,old)
    old_ref=ref(old);torch.save(derive(source,parent_sha256=old_ref['sha256']),child)
    monkeypatch.setattr(native_raw_policy,'saved_native_identity',lambda state,checkpoint_sha256:
        {'checkpoint_sha256':checkpoint_sha256,'profile':state['model_config'].get('native_shared_execution','source-v1')})
    proof=verify_files(old_ref,ref(child));proof_path=tmp_path/'proof.json';proof_path.write_text(json.dumps(proof))
    state_path=tmp_path/'controller.json';state_path.write_text(json.dumps({'checkpoint':str(old),'phase':'collect'}))
    publication=publish_diagnostic(state_path,proof_ref=ref(proof_path),out=tmp_path/'candidate.json')
    state=json.loads(state_path.read_text())
    assert state['checkpoint']==str(old) and publication['selection_hold'] is True
    assert state['execution_transition_candidate']['sha256']==ref(tmp_path/'candidate.json')['sha256']
    assert verify_candidate_publication(ref(tmp_path/'candidate.json'))['proof']==proof
    with pytest.raises(ValueError,match='candidate or receipt'):
        publish_diagnostic(state_path,proof_ref=ref(proof_path),out=tmp_path/'other.json')


def test_held_execution_transition_cannot_enter_training():
    from natlang_neuralese.train.native_execution_transition import require_qualified_for_training
    require_qualified_for_training(parent()['training'])
    child=derive(parent(),parent_sha256='a'*64)
    with pytest.raises(ValueError,match='held pending'):
        require_qualified_for_training(child['training'])


def test_replay_gate_requires_every_authenticated_action_and_exact_deltas():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_execution_transition import validate_complete_replay_probe
    identity={'checkpoint_sha256':'b'*64}
    replay=SimpleNamespace(identity=identity, decision_actions=(), actions=(
        {'token_id':7,'behavior_logp':-1.0}, {'token_id':8,'behavior_logp':-2.0}))
    rows=[{'action_index':0,'token_id':7,'behavior_logp':-1.0,'learner_logp':-1.0,'delta':0.0},
          {'action_index':1,'token_id':8,'behavior_logp':-2.0,'learner_logp':-2.0,'delta':0.0}]
    receipt={'schema':'natlang.native-policy-behavior-logprob-probe/1','policy':identity,'graph_id':'g',
             'execution_device':'cuda:0',
             'all_episode_actions':True,'passed':True,'atol':1e-4,'action_count':2,'positions':rows,'max_abs_delta':0.0}
    assert validate_complete_replay_probe(receipt,identity=identity,graph_id='g',compiled_replay=replay)
    for change in ('partial','token','delta','identity'):
        bad=copy.deepcopy(receipt)
        if change=='partial':bad['positions'].pop()
        elif change=='token':bad['positions'][1]['token_id']=9
        elif change=='delta':bad['positions'][1]['delta']=0.01
        else:bad['policy']['checkpoint_sha256']='c'*64
        with pytest.raises(ValueError):validate_complete_replay_probe(bad,identity=identity,graph_id='g',compiled_replay=replay)


def test_numerical_gate_rejects_cpu_or_wrong_policy_before_certificate_admission():
    from natlang_neuralese.train.native_execution_transition import validate_numerical_certificate
    identity={'checkpoint_sha256':'b'*64}
    bad={'schema':'natlang.native-runtime-certificate/1','passed':True,'native_identity':identity,
         'checkpoint_sha256':'b'*64,'execution_device':'cpu','channel_restore_exact':True,
         'gradient_scope':'complete-production-trainables'}
    with pytest.raises(ValueError,match='actual CUDA'):
        validate_numerical_certificate(bad,identity=identity,model_config={'native_shared_execution':'ffn-stable-v1'})


def test_protected_dev_behavior_binds_packet_and_reports_per_case_changes(tmp_path, monkeypatch):
    import json, hashlib
    from pathlib import Path
    from types import SimpleNamespace
    from natlang_neuralese.eval import portable_task_evidence
    from natlang_neuralese.common.custody import ProtectedWorlds
    from natlang_neuralese.train import native_execution_transition as transition
    monkeypatch.setattr(portable_task_evidence,'validate',lambda receipt, identity, packet_path:
        {'score':json.loads(Path(receipt['score_path']).read_text())} if receipt.get('native_identity')==identity
        and receipt.get('passed') is True else (_ for _ in ()).throw(ValueError('bad actor')))
    packet_path=tmp_path/'packet/cases.jsonl';packet_path.parent.mkdir();packet_path.write_text('case')
    packet=hashlib.sha256(packet_path.read_bytes()).hexdigest()
    monkeypatch.setattr(transition,'DEV_PACKET_SHA256',packet)
    (tmp_path/'training').mkdir()
    (tmp_path/'training/neuralese_corpora.json').write_text(json.dumps({'corpora':[{'id':transition.DEV_PACKET_ID,
        'path':'packet','admission':'evaluation-only-never-training','training_admission':False,
        'split':'test (generated); role development'}]}))
    source={'id':'registered-evaluation-only-packets','packets':[{'id':transition.DEV_PACKET_ID,'cases_sha256':packet}]}
    monkeypatch.setattr(ProtectedWorlds,'load',lambda **kwargs:SimpleNamespace(sources=(json.dumps(source),),digest='d'*64))
    identity={'checkpoint_sha256':'b'*64}; old_identity={'checkpoint_sha256':'a'*64}
    def evidence(path, successes, outcomes):
        path.write_text(json.dumps({'packet_sha256':packet,'case_indices':[0,1],
            'cases_detail':[{'id':name,'outcome':outcome} for name,outcome in zip(('x','y'),outcomes)],
            'task_successes':successes}))
        return {'schema':'natlang.native-task-execution-receipt/2','passed':True,'score_path':str(path),
                'score_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    candidate=evidence(tmp_path/'new.json',1,['task_failure','task_success']);candidate['native_identity']=identity
    baseline=evidence(tmp_path/'old.json',1,['task_success','task_failure']);baseline['native_identity']=old_identity
    pref={'path':str(packet_path),'sha256':packet}
    result=transition.validate_dev_behavior(candidate,baseline,candidate_identity=identity,baseline_identity=old_identity,
                                              packet_ref=pref,protection_root=tmp_path)
    assert result['role']=='protected-development' and result['untouched_confirmation'] is False
    assert len(result['case_outcome_changes'])==2
    bad=copy.deepcopy(pref);bad['sha256']='a'*64
    with pytest.raises(ValueError):
        transition.validate_dev_behavior(candidate,baseline,candidate_identity=identity,baseline_identity=old_identity,
                                         packet_ref=bad,protection_root=tmp_path)


def test_sealed_replay_rejects_collection_bound_to_other_actor(tmp_path):
    import hashlib, json
    from natlang_neuralese.train.native_execution_transition import validate_fresh_sealed_replay
    def ref(name, obj):
        path=tmp_path/name;path.write_text(json.dumps(obj))
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    identity={'checkpoint_sha256':'b'*64}
    request=ref('request.json',{'policy_id':'a'*64,'expected_native':{'native_identity':identity}})
    replay=ref('replay.json',{})
    with pytest.raises(ValueError,match='candidate native actor'):
        validate_fresh_sealed_replay({},request,replay,identity=identity,roles=None)


@pytest.mark.parametrize('qualification_has_size', [True, False])
def test_selected_publication_preserves_old_round_and_never_counts_an_update(tmp_path, monkeypatch, qualification_has_size):
    import hashlib, json
    from natlang_neuralese.train import native_execution_transition as tx
    from natlang_neuralese.common import custody
    def write(name, value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    old=write('parent.pt',{'old':1}); child=write('child.pt',{'new':1}); inner=write('proof.json',{'inner':1})
    candidate=write('candidate.json',{'proof':inner}); cases=write('cases.json',{'cases':1})
    binding=write('source/collection-binding.json',{'cases_sha256':cases['sha256']})
    rollouts=write('source/rollouts.jsonl',{'x':1})
    collection=write('source/sealed.json',{'files':{'collection-binding.json':binding,'rollouts.jsonl':rollouts}})
    request=write('source/request.json',{'cases':cases['path']})
    from natlang_neuralese.eval.portable_task_evidence import FILES, SCHEMA
    scores=[]
    for label in ('new','old'):
        files={}
        for name in FILES:
            ref=write('dev-'+label+'/'+name,{'fixture':label})
            files[name]={'sha256':ref['sha256'],'bytes':ref['bytes']}
        scores.append(write('dev-'+label+'/evidence.json',{'schema':SCHEMA,'files':files}))
    evidence=write('evidence.json',{'collection_proof':collection,'collection_request':request,'dev_candidate':scores[0],
        'dev_parent':scores[1]})
    qual_ref=write('qualified.json',{'q':1})
    qualified_candidate = dict(candidate)
    if not qualification_has_size:
        qualified_candidate.pop('bytes')
    report={'candidate':qualified_candidate,'evidence':evidence,'protection_root':str(tmp_path),
            'protected_development':{'role':'protected-development'}}
    checked={'proof':{'parent':old,'checkpoint':child,'actor_identity':{'old':{},'new':{}}},
             'publication':{'proof':inner}}
    monkeypatch.setattr(tx,'verify_qualification',lambda ref:report)
    monkeypatch.setattr(tx,'verify_candidate_publication',lambda ref:checked)
    monkeypatch.setattr(custody,'_read_protected_registry',lambda path:{'sources':[]})
    monkeypatch.setattr(custody,'_protected_worlds_snapshot',lambda *args:('stable',))
    state=tmp_path/'state.json';state.write_text(json.dumps({'checkpoint':old['path'],'phase':'stopped_collecting',
        'round':0,'offset':3,'trained_rounds':7,'execution_transition_candidate':candidate}))
    selection=write('round-000000-selection.json',{'preserve':'old'})
    partial=write('partial.pt',{'earned':True})
    prior=json.loads(state.read_text());state.write_text(json.dumps({**prior,'partial_checkpoint':partial['path']}))
    with pytest.raises(ValueError,match='different earned partial'):
        tx.publish_selected(state,qualification_ref=qual_ref,out=tmp_path/'blocked.json',decision_reason='paired review')
    for field in ('active_update_round', 'partial_update_round'):
        state.write_text(json.dumps({**prior, field:0}))
        with pytest.raises(ValueError, match='pending update'):
            tx.publish_selected(state,qualification_ref=qual_ref,out=tmp_path/'blocked-zero.json',decision_reason='paired review')
    state.write_text(json.dumps({**prior,'execution_transition_candidate':{**candidate,'bytes':candidate['bytes']+1}}))
    with pytest.raises(ValueError, match='declared size'):
        tx.publish_selected(state,qualification_ref=qual_ref,out=tmp_path/'blocked-size.json',decision_reason='paired review')
    state.write_text(json.dumps(prior))
    result=tx.publish_selected(state,qualification_ref=qual_ref,out=tmp_path/'selected.json',decision_reason='paired review')
    changed=json.loads(state.read_text())
    assert result['preserved_old_collection']['selection']==selection
    assert changed['checkpoint']==child['path'] and changed['round']==1 and changed['offset']==3
    assert changed['trained_rounds']==7 and result['optimizer_updates']==0 and result['graph_visits']==0
    assert json.loads(open(selection['path']).read())=={'preserve':'old'}


def test_training_admission_preserves_candidate_bytes_and_survives_earned_step(tmp_path, monkeypatch):
    import hashlib
    from natlang_neuralese.train import native_execution_transition as tx
    candidate=tmp_path/'candidate.pt';candidate.write_bytes(b'immutable candidate')
    child_sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
    admission={'path':'selected.json','sha256':'c'*64}
    selected={'proof':{'parent':{'sha256':'a'*64},'checkpoint':{'sha256':child_sha}}}
    monkeypatch.setattr(tx,'verify_selected_publication',lambda ref:selected if ref==admission else (_ for _ in ()).throw(ValueError('bad')))
    source=parent();child=derive(source,parent_sha256='a'*64)
    admitted=tx.require_qualified_for_training(child['training'],admission_ref=admission,checkpoint_path=candidate)
    assert child['training']['options']['native_shared_execution_history'][-1]['qualification']=='pending'
    assert admitted['options']['native_shared_execution_history'][-1]['qualification']['admission']==admission
    resumed={**admitted,'global_step':68}
    assert tx.require_qualified_for_training(resumed)==resumed
    bad=tmp_path/'bad.pt';bad.write_bytes(b'different')
    with pytest.raises(ValueError,match='different immutable candidate'):
        tx.require_qualified_for_training(child['training'],admission_ref=admission,checkpoint_path=bad)


def test_selected_execution_publication_is_typed_refresh_ancestry(tmp_path, monkeypatch):
    import hashlib, json
    from natlang_neuralese.rl.native_direct_sft_recovery import _authenticated_refresh_publication
    from natlang_neuralese.train import native_execution_transition as tx
    path=tmp_path/'selected.json';path.write_text(json.dumps({'schema':'natlang.native-execution-selection-publication/1'}))
    ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    proof={'parent':{'sha256':'a'*64},'checkpoint':{'sha256':'b'*64},'global_step':67}
    monkeypatch.setattr(tx,'verify_selected_publication',lambda reference:
        {'proof':proof,'authenticated_ancestors':['a'*64,'b'*64]})
    checked=_authenticated_refresh_publication(ref)
    assert checked['authenticated_ancestors']=={'a'*64,'b'*64}
    assert checked['custody']['successor']['global_step']==67


def test_selection_admission_requires_durable_controller_history(tmp_path, monkeypatch):
    import hashlib, json
    from natlang_neuralese.train import native_execution_transition as tx
    def ref(path):
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    controller=tmp_path/'state.json';candidate={'path':'candidate','sha256':'c'*64};qual={'path':'qualified','sha256':'d'*64}
    proof={'parent':{'sha256':'a'*64},'checkpoint':{'sha256':'b'*64},'actor_identity':{'old':{},'new':{}}}
    protected={'role':'protected-development'}
    monkeypatch.setattr(tx,'verify_qualification',lambda _: {'candidate':candidate,'protected_development':protected})
    monkeypatch.setattr(tx,'verify_candidate_publication',lambda _: {'proof':proof})
    publication={'schema':'natlang.native-execution-selection-publication/1','selection_hold':False,
        'optimizer_updates':0,'graph_visits':0,'decision_reason':'reviewed paired development outcomes',
        'candidate':candidate,'qualification':qual,'parent':proof['parent'],'checkpoint':proof['checkpoint'],
        'actor_identity':proof['actor_identity'],'protected_development':protected,'untouched_confirmation':False,
        'controller_state_path':str(controller)}
    path=tmp_path/'selection.json';path.write_text(json.dumps(publication));binding=ref(path)
    controller.write_text(json.dumps({'execution_transition_history':[binding],'direct_sft_refresh_history':[binding]}))
    assert tx.verify_selected_publication(binding)['proof']==proof
    controller.write_text(json.dumps({'execution_transition_history':[],'direct_sft_refresh_history':[binding]}))
    with pytest.raises(ValueError,match='durable controller'):
        tx.verify_selected_publication(binding)


def test_qualification_envelope_rechecks_every_bound_evidence_file(tmp_path, monkeypatch):
    import hashlib, json
    from natlang_neuralese.train import native_execution_transition as tx
    def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    source=parent();source['model_config']['marker_ids']=[1,2,3,4]
    child=derive(source,parent_sha256='a'*64);child_path=tmp_path/'child.pt';torch.save(child,child_path)
    child_ref=ref(child_path);candidate_path=tmp_path/'candidate.json';candidate_path.write_text('{}');candidate_ref=ref(candidate_path)
    identity={'checkpoint_sha256':child_ref['sha256']};old={'checkpoint_sha256':'a'*64}
    checked={'proof':{'checkpoint':child_ref,'actor_identity':{'old':old,'new':identity}}}
    monkeypatch.setattr(tx,'verify_candidate_publication',lambda _:checked)
    seen=[]
    monkeypatch.setattr(tx,'validate_numerical_certificate',lambda *args,**kw:seen.append('runtime'))
    monkeypatch.setattr(tx,'validate_fresh_sealed_replay',lambda *args,**kw:seen.append('replay'))
    monkeypatch.setattr(tx,'validate_dev_behavior',lambda *args,**kw: {'role':'protected-development','untouched_confirmation':False})
    names=('runtime_certificate','collection_proof','collection_request','replay_probe','dev_candidate','dev_parent','dev_packet')
    evidence={'schema':'natlang.native-execution-qualification-evidence/1'}
    for name in names:
        path=tmp_path/(name+'.json');path.write_text('{}');evidence[name]=ref(path)
    envelope=tmp_path/'evidence.json';envelope.write_text(json.dumps(evidence));envelope_ref=ref(envelope)
    result=tx.qualify_candidate(candidate_ref,envelope_ref,protection_root=tmp_path)
    assert seen==['runtime','replay'] and result['selection_hold'] is True and result['untouched_confirmation'] is False
    report=tmp_path/'qualification.json';report.write_text(json.dumps(result));report_ref=ref(report)
    assert tx.verify_qualification(report_ref)==result
    (tmp_path/'runtime_certificate.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):tx.verify_qualification(report_ref)


def test_rl_trainer_admission_is_controller_owned(tmp_path):
    import hashlib
    from natlang_neuralese.rl.production import (native_policy_trainer_args, validate_native_policy_trainer_cli,
                                                  bind_seed_execution_admission)
    path=tmp_path/'selected.json';path.write_text('{}')
    ref={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    args=native_policy_trainer_args({}, {}, ref)
    assert args==['--execution-transition-admission',str(path),'--execution-transition-admission-sha256',ref['sha256']]
    validate_native_policy_trainer_cli(args)
    assert bind_seed_execution_admission([], ref)==args
    with pytest.raises(ValueError,match='owned by the controller'):
        bind_seed_execution_admission(['--execution-transition-admission','other'], ref)
    with pytest.raises(ValueError,match='owned by the controller'):
        native_policy_trainer_args({'execution_transition_admission':str(path)}, {}, ref)


def test_replay_admission_checks_all_decisions_and_authenticated_option_branches():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_execution_transition import validate_complete_replay_probe
    identity={'checkpoint_sha256':'a'*64}
    decision={'turn':'decide','chosen':0,'temperature':1.,'option_nodes':['decide:0','decide:1'],
              'options':[[{'token_id':3,'query':10}],[{'token_id':4,'query':20}]],'behavior_logps':[-.5,-1.]}
    replay=SimpleNamespace(identity=identity,actions=(),decision_actions=(decision,))
    row={'decision_index':0,'turn':'decide','chosen':0,'temperature':1.,'option_nodes':['decide:0','decide:1'],
         'option_token_ids':[[3],[4]],'option_queries':[[10],[20]],'behavior_logps':[-.5,-1.], 'learner_logps':[-.5,-1.], 'deltas':[0.,0.]}
    receipt={'schema':'natlang.native-policy-behavior-logprob-probe/1','policy':identity,'graph_id':'g',
        'execution_device':'cuda:0','all_episode_actions':True,'passed':True,'atol':1e-4,
        'positions':[],'action_count':0,'decision_positions':[row],'decision_count':1,'max_abs_delta':0.}
    assert validate_complete_replay_probe(receipt,identity=identity,graph_id='g',compiled_replay=replay)
    for mutate in (lambda x:x['decision_positions'].clear(),
                   lambda x:x['decision_positions'][0]['option_nodes'].__setitem__(1,'foreign'),
                   lambda x:x['decision_positions'][0]['option_token_ids'].pop(),
                   lambda x:x['decision_positions'][0]['learner_logps'].__setitem__(1,-.8)):
        bad=copy.deepcopy(receipt);mutate(bad)
        with pytest.raises(ValueError):validate_complete_replay_probe(bad,identity=identity,graph_id='g',compiled_replay=replay)
