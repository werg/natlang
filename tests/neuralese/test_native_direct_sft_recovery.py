import copy
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


def _same_slot_publication_fixture(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    tmp_path.mkdir(parents=True,exist_ok=True)
    def file(name, data):
        p=tmp_path/name;p.write_text(json.dumps(data));return binding(p)
    rolling=tmp_path/'rolling';rolling.write_text(json.dumps({'step':57,'old':True}))
    old_parent=file('parent-pin',{'step':53})
    prior_child=binding(rolling)
    state=tmp_path/'state.json';state.write_text(json.dumps({'phase':'awaiting_semantic_refresh','checkpoint':old_parent['path'],
        'pending_native_checkpoint':str(rolling),'round':7,'trained_rounds':6,
        'trained_task_ids':['already-earned'],'trained_supervised_tokens':123}))
    common={'optimizer_present':True,'rng_present':True}
    prior_boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':4,
        'checkpoint_sha256':prior_child['sha256']}
    prior_report=file('prior-report',{'numerics_ok':True,'checkpoint_sha256':prior_child['sha256'],
        'continuation_boundary':prior_boundary})
    prior_receipt=file('prior-receipt',{'checkpoint':{'sha256':prior_child['sha256']},
        'continuation':{'checkpoint_sha256':old_parent['sha256']}})
    prior_custody=file('prior-custody',{'schema':'natlang.earned-sft-checkpoint-custody/1',
        'parent':old_parent|common|{'global_step':53},'successor':prior_child|common|{'global_step':57},
        'old_bindings_and_quotas_preserved':True,'old_exposure_not_decreased':True})
    publish_completed(state,report_ref=prior_report,receipt_ref=prior_receipt,custody_ref=prior_custody,
        out=tmp_path/'prior-publication')
    historical_sha=prior_child['sha256']
    rolling.write_text(json.dumps({'step':61,'new':True}))
    child=binding(rolling);parent=prior_child|common|{'global_step':57}
    receipt=file('receipt',{'checkpoint':{'sha256':child['sha256']},
        'continuation':{'checkpoint_sha256':old_parent['sha256']},
        'operation_parent':{'path':str(rolling.resolve()),'sha256':historical_sha,'global_step':57}})
    boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':4,'checkpoint_sha256':child['sha256']}
    report=file('report',{'numerics_ok':True,'checkpoint_sha256':child['sha256'],'continuation_boundary':boundary})
    custody=file('custody',{'schema':'natlang.earned-sft-checkpoint-custody/1','parent':parent,
        'successor':child|common|{'global_step':61},'old_bindings_and_quotas_preserved':True,
        'old_exposure_not_decreased':True})
    saved=json.loads(state.read_text());saved.update({'phase':'awaiting_semantic_refresh',
        'pending_native_checkpoint':str(rolling)});state.write_text(json.dumps(saved))
    return {'state':state,'report':report,'receipt':receipt,'custody':custody,
        'out':tmp_path/'publication','historical_sha':historical_sha,'rolling':rolling,
        'root_sha':old_parent['sha256'],'prior_publication':tmp_path/'prior-publication'}


def test_completed_refresh_accepts_authenticated_same_rolling_slot(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    def file(root,name,data):
        path=root/name;path.write_text(json.dumps(data));return binding(path)
    before=json.loads(fixture['state'].read_text())
    result=publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
        custody_ref=fixture['custody'],out=fixture['out'])
    assert json.loads(fixture['state'].read_text())['checkpoint']==str(fixture['rolling'].resolve())
    assert json.loads(fixture['state'].read_text())['round']==before['round']
    after=json.loads(fixture['state'].read_text())
    assert after['trained_rounds']==before['trained_rounds']
    assert after['trained_task_ids']==before['trained_task_ids']
    assert after['trained_supervised_tokens']==before['trained_supervised_tokens']
    assert result['stage_completion_granted'] is False
    assert result['behavioral_qualification_granted'] is False
    assert result['additional_training_credit']==0
    assert result['parent']['sha256']==fixture['historical_sha']
    proof=result['historical_parent_proof']
    assert proof['bytes_verified_currently'] is False
    assert proof['identity']=={'path':str(fixture['rolling'].resolve()),
        'sha256':fixture['historical_sha'],'global_step':57}
    assert proof['reported_continuation_ancestor_sha256']==json.loads(
        Path(fixture['receipt']['path']).read_text())['continuation']['checkpoint_sha256']
    assert result['checkpoint']['sha256']==binding(fixture['rolling'])['sha256']
    # Second same-slot update proves the ordered 53 -> 57 -> 61 -> 65 chain.
    state=json.loads(fixture['state'].read_text());state.update({'phase':'awaiting_semantic_refresh',
        'pending_native_checkpoint':str(fixture['rolling'])});fixture['state'].write_text(json.dumps(state))
    parent61=result['checkpoint']|{'global_step':61,'optimizer_present':True,'rng_present':True}
    fixture['rolling'].write_text(json.dumps({'step':65,'second_sft_child':True}));child65=binding(fixture['rolling'])
    receipt65=file(tmp_path,'receipt-65',{'checkpoint':{'sha256':child65['sha256']},
        'continuation':{'checkpoint_sha256':fixture['root_sha']},
        'operation_parent':{'path':parent61['path'],'sha256':parent61['sha256'],'global_step':61}})
    report65=file(tmp_path,'report-65',{'numerics_ok':True,'checkpoint_sha256':child65['sha256'],
        'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':4,
            'checkpoint_sha256':child65['sha256']}})
    custody65=file(tmp_path,'custody-65',{'schema':'natlang.earned-sft-checkpoint-custody/1',
        'parent':parent61,'successor':child65|{'global_step':65,'optimizer_present':True,'rng_present':True},
        'old_bindings_and_quotas_preserved':True,'old_exposure_not_decreased':True})
    result65=publish_completed(fixture['state'],report_ref=report65,receipt_ref=receipt65,
        custody_ref=custody65,out=tmp_path/'publication-65')
    assert result65['parent']['sha256']==parent61['sha256']
    assert result65['checkpoint']['sha256']==child65['sha256']
    assert result65['historical_parent_proof']['reported_continuation_ancestor_sha256']==fixture['root_sha']


def test_completed_refresh_rejects_missing_or_nonlast_same_slot_history(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    state=json.loads(fixture['state'].read_text());state['direct_sft_refresh_history']=[]
    fixture['state'].write_text(json.dumps(state))
    with pytest.raises(ValueError,match='last authenticated parent'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])
    state=json.loads(fixture['state'].read_text())
    stale=tmp_path/'stale-last-publication'
    stale.write_text(json.dumps({'schema':'natlang.direct-sft-refresh-publication/1',
        'checkpoint':{'path':str(fixture['rolling'].resolve()),'sha256':'f'*64}}))
    state['direct_sft_refresh_history'].append(binding(stale));fixture['state'].write_text(json.dumps(state))
    with pytest.raises(ValueError,match='omits exact checkpoint identities'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])


def test_completed_refresh_rejects_tampered_historical_evidence_and_parent_step(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    prior=json.loads(fixture['prior_publication'].read_text())
    Path(prior['receipt']['path']).write_text('{}')
    with pytest.raises(ValueError,match='exact file binding'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])
    # A fresh fixture is needed because the tampered prior evidence is immutable.
    fixture=_same_slot_publication_fixture(tmp_path/'step')
    custody=json.loads(Path(fixture['custody']['path']).read_text());custody['parent']['global_step']=56
    Path(fixture['custody']['path']).write_text(json.dumps(custody))
    fixture['custody']=binding(Path(fixture['custody']['path']))
    with pytest.raises(ValueError,match='immediate parent step'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])


def test_completed_refresh_rejects_same_slot_child_hash_mismatch(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    fixture['rolling'].write_text('changed after receipt')
    with pytest.raises(ValueError,match='native readiness artifact changed'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])


def test_completed_refresh_rejects_unrelated_same_slot_continuation_ancestor(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    receipt_path=Path(fixture['receipt']['path'])
    receipt=json.loads(receipt_path.read_text())
    receipt['continuation']['checkpoint_sha256']='f'*64
    receipt_path.write_text(json.dumps(receipt));fixture['receipt']=binding(receipt_path)
    with pytest.raises(ValueError,match='not the exact parent or its authenticated publication ancestor'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])


def test_completed_refresh_rejects_operation_parent_mismatch(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    receipt_path=Path(fixture['receipt']['path']);receipt=json.loads(receipt_path.read_text())
    receipt['operation_parent']['sha256']='f'*64
    receipt_path.write_text(json.dumps(receipt));fixture['receipt']=binding(receipt_path)
    with pytest.raises(ValueError,match='qualified full-state custody'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])


def test_completed_refresh_accepts_rl_checkpoint_as_immediate_parent(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    # An RL update replaced the rolling slot after the last SFT publication.
    fixture['rolling'].write_text(json.dumps({'step':59,'rl_parent':True}))
    rl_parent=binding(fixture['rolling'])|{'global_step':59,'optimizer_present':True,'rng_present':True}
    child_data={'step':63,'sft_child':True};fixture['rolling'].write_text(json.dumps(child_data))
    child=binding(fixture['rolling'])
    receipt_path=Path(fixture['receipt']['path']);receipt=json.loads(receipt_path.read_text())
    receipt['checkpoint']['sha256']=child['sha256']
    receipt['operation_parent']={'path':rl_parent['path'],'sha256':rl_parent['sha256'],'global_step':59}
    receipt_path.write_text(json.dumps(receipt));fixture['receipt']=binding(receipt_path)
    report_path=Path(fixture['report']['path']);report=json.loads(report_path.read_text())
    report['checkpoint_sha256']=child['sha256'];report['continuation_boundary'].update(
        {'new_optimizer_updates':4,'checkpoint_sha256':child['sha256']})
    report_path.write_text(json.dumps(report));fixture['report']=binding(report_path)
    custody_path=Path(fixture['custody']['path']);custody=json.loads(custody_path.read_text())
    custody['parent']=rl_parent;custody['successor']=child|{'global_step':63,'optimizer_present':True,'rng_present':True}
    custody_path.write_text(json.dumps(custody));fixture['custody']=binding(custody_path)
    result=publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
        custody_ref=fixture['custody'],out=fixture['out'])
    assert result['parent']['sha256']==rl_parent['sha256']
    assert result['parent']['global_step']==59
    assert result['checkpoint']['sha256']==child['sha256']
    assert result['historical_parent_proof']['reported_continuation_ancestor_sha256']==fixture['root_sha']


def test_completed_refresh_rejects_rl_parent_without_operation_parent_receipt(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    fixture=_same_slot_publication_fixture(tmp_path)
    fixture['rolling'].write_text(json.dumps({'step':59,'rl_parent':True}))
    rl_parent=binding(fixture['rolling'])|{'global_step':59,'optimizer_present':True,'rng_present':True}
    fixture['rolling'].write_text(json.dumps({'step':63,'sft_child':True}));child=binding(fixture['rolling'])
    receipt_path=Path(fixture['receipt']['path']);receipt=json.loads(receipt_path.read_text())
    receipt['checkpoint']['sha256']=child['sha256'];receipt.pop('operation_parent')
    receipt_path.write_text(json.dumps(receipt));fixture['receipt']=binding(receipt_path)
    report_path=Path(fixture['report']['path']);report=json.loads(report_path.read_text())
    report['checkpoint_sha256']=child['sha256'];report['continuation_boundary']['checkpoint_sha256']=child['sha256']
    report_path.write_text(json.dumps(report));fixture['report']=binding(report_path)
    custody_path=Path(fixture['custody']['path']);custody=json.loads(custody_path.read_text())
    custody['parent']=rl_parent;custody['successor']=child|{'global_step':63,'optimizer_present':True,'rng_present':True}
    custody_path.write_text(json.dumps(custody));fixture['custody']=binding(custody_path)
    with pytest.raises(ValueError,match='exact overwritten parent identity'):
        publish_completed(fixture['state'],report_ref=fixture['report'],receipt_ref=fixture['receipt'],
            custody_ref=fixture['custody'],out=fixture['out'])


def test_completed_refresh_accepts_exact_registered_rolling_alias_of_rl_parent(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    def file(name, data):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,sort_keys=True));return binding(path)
    round_parent=file('round-11/learner/checkpoint.pt',{'step':63,'optimizer':'kept','rng':'kept'})
    rolling=tmp_path/'production/learner/checkpoint.pt';rolling.parent.mkdir(parents=True)
    rolling.write_bytes(Path(round_parent['path']).read_bytes())
    rolling_parent=binding(rolling)
    assert rolling_parent['sha256']==round_parent['sha256']
    state_path=tmp_path/'production/state.json'
    alias={'schema':'natlang.rolling-sft-resume/1','checkpoint_path':str(rolling.resolve()),
        'checkpoint_sha256':rolling_parent['sha256'],'previous_checkpoint_path':round_parent['path'],
        'previous_checkpoint_sha256':round_parent['sha256']}
    state_path.write_text(json.dumps({'phase':'awaiting_semantic_refresh',
        'checkpoint':round_parent['path'],'pending_native_checkpoint':str(rolling),
        'rolling_sft_resume_history':[alias]}))
    parent=rolling_parent|{'global_step':63,'optimizer_present':True,'rng_present':True}
    rolling.write_text(json.dumps({'step':67,'optimizer':'updated','rng':'updated'}))
    child=binding(rolling)|{'global_step':67,'optimizer_present':True,'rng_present':True}
    report=file('report.json',{'numerics_ok':True,'checkpoint_sha256':child['sha256'],
        'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':4,
            'checkpoint_sha256':child['sha256']}})
    receipt=file('receipt.json',{'checkpoint':{'sha256':child['sha256']},
        'continuation':{'checkpoint_sha256':parent['sha256']},
        'operation_parent':{'path':rolling_parent['path'],'sha256':rolling_parent['sha256'],'global_step':63}})
    custody=file('custody.json',{'schema':'natlang.earned-sft-checkpoint-custody/1',
        'parent':parent,'successor':child,'old_bindings_and_quotas_preserved':True,
        'old_exposure_not_decreased':True})
    (tmp_path/'production/controller.lock').touch()
    result=publish_completed(state_path,report_ref=report,receipt_ref=receipt,custody_ref=custody,
        out=tmp_path/'publication')
    proof=result['rolling_parent_alias_proof']
    assert result['parent']=={'path':str(rolling.resolve()),'sha256':rolling_parent['sha256'],'global_step':63}
    assert proof['schema']=='natlang.rolling-sft-parent-alias-proof/1'
    assert proof['state_checkpoint_source']['path']==round_parent['path']
    assert proof['state_checkpoint_source']['sha256']==rolling_parent['sha256']
    assert proof['same_checkpoint_bytes_and_step'] is True


@pytest.mark.parametrize('tamper', ['missing-history','wrong-path','wrong-hash'])
def test_completed_refresh_rejects_unregistered_or_mismatched_rolling_alias(tmp_path,tamper):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    def file(name, data):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,sort_keys=True));return binding(path)
    round_parent=file('round/checkpoint.pt',{'step':63})
    rolling=tmp_path/'prod/checkpoint.pt';rolling.parent.mkdir(parents=True)
    rolling.write_bytes(Path(round_parent['path']).read_bytes())
    parent_bind=binding(rolling)
    alias={'schema':'natlang.rolling-sft-resume/1','checkpoint_path':str(rolling.resolve()),
        'checkpoint_sha256':parent_bind['sha256'],'previous_checkpoint_path':round_parent['path'],
        'previous_checkpoint_sha256':round_parent['sha256']}
    if tamper=='wrong-path':alias['checkpoint_path']=str(tmp_path/'elsewhere')
    if tamper=='wrong-hash':alias['previous_checkpoint_sha256']='f'*64
    history=[] if tamper=='missing-history' else [alias]
    state=tmp_path/'prod/state.json';state.write_text(json.dumps({'phase':'awaiting_semantic_refresh',
        'checkpoint':round_parent['path'],'pending_native_checkpoint':str(rolling),
        'rolling_sft_resume_history':history}))
    parent=parent_bind|{'global_step':63,'optimizer_present':True,'rng_present':True}
    rolling.write_text(json.dumps({'step':67}));child=binding(rolling)|{'global_step':67,
        'optimizer_present':True,'rng_present':True}
    report=file('report.json',{'checkpoint_sha256':child['sha256'],
        'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':4,
            'checkpoint_sha256':child['sha256']}})
    receipt=file('receipt.json',{'checkpoint':{'sha256':child['sha256']},
        'continuation':{'checkpoint_sha256':parent['sha256']},
        'operation_parent':{'path':parent_bind['path'],'sha256':parent_bind['sha256'],'global_step':63}})
    custody=file('custody.json',{'schema':'natlang.earned-sft-checkpoint-custody/1',
        'parent':parent,'successor':child,'old_bindings_and_quotas_preserved':True,
        'old_exposure_not_decreased':True})
    (tmp_path/'prod/controller.lock').touch()
    with pytest.raises(ValueError,match='rolling SFT parent alias'):
        publish_completed(state,report_ref=report,receipt_ref=receipt,custody_ref=custody,out=tmp_path/'publication')


@pytest.mark.parametrize('tamper',['none','interrupted','interrupted-changed-binding','interrupted-decreased-exposure','missing-archive','wrong-intermediate','forged-report','wrong-controller-parent',
                                   'changed-binding','decreased-exposure','changed-quota','missing-endpoint'])
def test_two_earned_windows_require_exact_archived_intermediate_and_full_endpoint_custody(tmp_path,tamper,monkeypatch):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed, _authenticated_refresh_publication
    def file(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,sort_keys=True));return binding(path)
    parent=file('round/checkpoint.pt',{'step':72,'optimizer':'present','rng':'present'})
    slot=tmp_path/'production/learner/checkpoint.pt';slot.parent.mkdir(parents=True)
    slot.write_text(json.dumps({'step':75,'earned':'first-window'}));middle=binding(slot)
    middle_boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':3,'checkpoint_sha256':middle['sha256']}
    middle_report=file('archive/report.json',{'schema':'natlang.neuralese-native-sft-report/1',
        'checkpoint_sha256':middle['sha256'],'continuation_boundary':middle_boundary,'numerics_ok':True})
    middle_receipt=file('archive/native-training-receipt.json',{'schema':'natlang.native-training-readiness/1',
        'checkpoint':middle,'continuation':{'checkpoint_sha256':parent['sha256']},
        'operation_parent':{'path':str(slot),'sha256':parent['sha256'],'global_step':72},
        'continuation_boundary':middle_boundary,'training_receipt':middle_report})
    archive=file('archive/receipt.json',{'schema':'natlang.stopped-window-evidence/1',
                                         'artifacts':[middle_report,middle_receipt]})
    slot.write_text(json.dumps({'step':79,'earned':'both-windows'}));child=binding(slot)
    transition={'schema':'natlang.rolling-sft-resume/1','checkpoint_path':str(slot),
        'checkpoint_sha256':middle['sha256'],'previous_checkpoint_path':parent['path'],
        'previous_checkpoint_sha256':parent['sha256']}
    if tamper=='wrong-intermediate':transition['checkpoint_sha256']='f'*64
    if tamper=='wrong-controller-parent':transition['previous_checkpoint_sha256']='f'*64
    if tamper=='forged-report':Path(middle_report['path']).write_text('{}')
    state=tmp_path/'production/state.json';state.write_text(json.dumps({'phase':'awaiting_semantic_refresh',
        'checkpoint':parent['path'],'pending_native_checkpoint':str(slot),'rolling_sft_resume_history':[transition]}))
    boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':4,'checkpoint_sha256':child['sha256']}
    report=file('new-report.json',{'checkpoint_sha256':child['sha256'],'continuation_boundary':boundary,'numerics_ok':True})
    receipt=file('new-receipt.json',{'checkpoint':child,'continuation':{'checkpoint_sha256':middle['sha256']},
        'operation_parent':{'path':str(slot),'sha256':middle['sha256'],'global_step':75}})
    common={'optimizer_present':True,'rng_present':True}
    held={'path':str(tmp_path/'held'),'sha256':'d'*64}
    frame={'source_row_order':['a'],'source_row_bindings':{'a':'a-binding'},
        'required_visits':{'a':2},'exposure_counts':{'a':1},'pending_source_rows':{},
        'held_row_bindings':{'held':'held-binding'},'source_receipts':[{'sha256':'old'},held]}
    new_frame=copy.deepcopy(frame);new_frame['source_row_order'].append('b')
    new_frame['source_row_bindings']['b']='b-binding';new_frame['required_visits']['b']=1
    new_frame['exposure_counts'].update(a=2,b=1)
    if tamper=='changed-binding':new_frame['source_row_bindings']['a']='forged'
    if tamper=='decreased-exposure':new_frame['exposure_counts']['a']=0
    if tamper=='changed-quota':new_frame['required_visits']['a']=1
    def snapshot(name,checkpoint,step,frames):
        return file(name,{'schema':'natlang.rolling-sft-parent-custody-snapshot/1',
            'checkpoint':{**checkpoint,'bytes':Path(checkpoint['path']).stat().st_size},
            'global_step':step,'optimizer_present':True,'rng_present':True,'model_config_sha256':'a'*64,
            'sft_frames':{'full':frames}})
    old_snapshot=snapshot('old-snapshot.json',parent,72,frame)
    new_snapshot=snapshot('new-snapshot.json',child,79,new_frame)
    if tamper.startswith('interrupted'):
        middle_frame=copy.deepcopy(frame)
        if tamper=='interrupted-changed-binding': middle_frame['source_row_bindings']['a']='forged'
        if tamper=='interrupted-decreased-exposure': middle_frame['exposure_counts']['a']=0
        middle_snapshot=snapshot('middle-snapshot.json',middle,75,middle_frame)
        transition['custody_snapshot']=middle_snapshot
        def interrupted(declaration, selected, old, new, actual_receipt, **kwargs):
            return {'schema':'natlang.rolling-sft-intermediate-window-proof/1','kind':'interrupted-saved-window',
                'registered_transition':selected,'resolved_previous_checkpoint_path':parent['path'],
                'intermediate':{'path':str(slot),'sha256':middle['sha256'],'global_step':75},
                'intermediate_updates':3,'intermediate_checkpoint_bytes_verified_currently':True}
        monkeypatch.setattr('natlang_neuralese.rl.native_direct_sft_recovery._interrupted_rolling_window',interrupted)
        state.write_text(json.dumps({'phase':'awaiting_semantic_refresh','checkpoint':parent['path'],
            'pending_native_checkpoint':str(slot),'rolling_sft_resume_history':[transition]}))
    custody_data={'schema':'natlang.earned-sft-checkpoint-custody/1','parent':parent|common|{'global_step':72},
        'successor':child|common|{'global_step':79},'old_bindings_and_quotas_preserved':True,
        'old_exposure_not_decreased':True,'rolling_intermediate_window':{
            'schema':'natlang.rolling-sft-intermediate-window/1','evidence':archive},
        'rolling_endpoint_snapshots':{'schema':'natlang.rolling-sft-endpoint-snapshots/1',
                                      'parent':old_snapshot,'successor':new_snapshot}}
    if tamper.startswith('interrupted'):
        custody_data['rolling_intermediate_window']['schema']='natlang.interrupted-sft-intermediate-window/1'
    if tamper=='missing-archive':custody_data['rolling_intermediate_window']['evidence']['sha256']='0'*64
    if tamper=='missing-endpoint':del custody_data['rolling_endpoint_snapshots']
    custody=file('custody.json',custody_data);out=tmp_path/'publication.json'
    if tamper not in ('none','interrupted'):
        with pytest.raises(ValueError):publish_completed(state,report_ref=report,receipt_ref=receipt,custody_ref=custody,out=out)
        assert json.loads(state.read_text())['checkpoint']==parent['path']
        return
    result=publish_completed(state,report_ref=report,receipt_ref=receipt,custody_ref=custody,out=out)
    proof=result['rolling_intermediate_proof']
    assert proof['intermediate']['sha256']==middle['sha256'] and proof['intermediate']['global_step']==75
    assert proof['intermediate_updates']==3 and proof['intermediate_checkpoint_bytes_verified_currently'] is (tamper=='interrupted')
    if tamper=='interrupted':
        assert set(result['rolling_endpoint_custody_proof']['interrupted_segments']) == {'parent_to_intermediate','intermediate_to_successor'}
    assert result['rolling_endpoint_custody_proof']['stages']['full']['old_graph_visits']==1
    assert result['parent']['sha256']==parent['sha256'] and result['checkpoint']['sha256']==child['sha256']
    assert _authenticated_refresh_publication(binding(out))['publication']==result
    Path(parent['path']).unlink()
    assert _authenticated_refresh_publication(binding(out))['publication']==result


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


def test_completed_direct_sft_refresh_requires_exact_multiupdate_delta(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_completed
    def file(name,data):
        path=tmp_path/name;path.write_text(json.dumps(data));return binding(path)
    parent=file('multi-parent',{'old':True});child=file('multi-child',{'new':True})
    state=tmp_path/'multi-state.json';state.write_text(json.dumps({'phase':'awaiting_semantic_refresh','checkpoint':parent['path'],'pending_native_checkpoint':child['path'],'round':11,'trained_rounds':11}))
    boundary={'kind':'semantic-refresh','reached':True,'new_optimizer_updates':3,'checkpoint_sha256':child['sha256']}
    report=file('multi-report',{'numerics_ok':True,'checkpoint_sha256':child['sha256'],'continuation_boundary':boundary})
    receipt=file('multi-receipt',{'checkpoint':{'sha256':child['sha256']},'continuation':{'checkpoint_sha256':parent['sha256']}})
    common={'optimizer_present':True,'rng_present':True}
    custody_data={'schema':'natlang.earned-sft-checkpoint-custody/1','parent':parent|common|{'global_step':53},'successor':child|common|{'global_step':57},'old_bindings_and_quotas_preserved':True,'old_exposure_not_decreased':True}
    custody=file('multi-custody',custody_data)
    with pytest.raises(ValueError,match='qualified full-state custody'):
        publish_completed(state,report_ref=report,receipt_ref=receipt,custody_ref=custody,out=tmp_path/'multi-out')
    boundary['new_optimizer_updates']=4; report=file('multi-report',{'numerics_ok':True,'checkpoint_sha256':child['sha256'],'continuation_boundary':boundary})
    publish_completed(state,report_ref=report,receipt_ref=receipt,custody_ref=custody,out=tmp_path/'multi-out')
    result=json.loads(state.read_text());assert result['checkpoint']==child['path'] and result['trained_rounds']==11


def test_completed_refresh_accepts_only_bound_unearned_stage_replacement(tmp_path):
    from natlang_neuralese.rl.native_direct_sft_recovery import _completed_refresh_obligations
    parent={'sha256':'a'*64};history=tmp_path/'policy.json';history.write_text(json.dumps({'schema':'natlang.cumulative-policy-exposure-custody/1','status':'verified','through_checkpoint':parent}))
    policy=binding(history);proof=tmp_path/'replacement.json';proof.write_text(json.dumps({'parent_sha256':parent['sha256'],'old_stage':'old','old_source_sha256':'b'*64,'new_source_sha256':'c'*64}));proof_ref=binding(proof)
    record={'schema':'natlang.unearned-sft-source-replacement/1','parent_sha256':parent['sha256'],'old_stage':'old','old_source_sha256':'b'*64,'new_source_sha256':'c'*64,'old_graph_visits':0,'weights_optimizer_rng_rl_exposure':'preserved','cumulative_policy_history':policy,'proof_path':proof_ref['path'],'proof_sha256':proof_ref['sha256']}
    custody={'old_obligations_transition':'unearned-sft-supersession-verified','unearned_replacement':record}
    assert _completed_refresh_obligations(custody,parent)
    record['old_graph_visits']=1
    assert not _completed_refresh_obligations(custody,parent)


@pytest.mark.parametrize('defect',[None,'unreviewed-binding','quota-loss','wrong-parent','missing-digest','wrong-source','missing-addition'])
def test_completed_refresh_accepts_only_bound_source_correction(tmp_path,defect):
    from natlang_neuralese.rl.native_direct_sft_recovery import _completed_refresh_obligations
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
    from natlang_neuralese.train.native_resident import binding
    def save(name,obj):
        p=tmp_path/name;p.write_text(json.dumps(obj));return binding(p)
    prior=save('prior.json',{'old':True});current=save('current.json',{'new':True})
    rows=tmp_path/'rows.jsonl';rows.write_text(json.dumps({'id':'repair','old_binding':'old','new_binding':'new','disposition':'eligibility-decreased'})+'\n'+json.dumps({'id':'addition','new_binding':'added','disposition':'added'})+'\n')
    proof=save('proof.json',{'schema':'natlang.native-source-correction/1',
        'prior_source':{**prior,'rows':2},'current_source':{**current,'rows':3},
        'row_bindings_path':rows.name,'row_bindings_sha256':binding(rows)['sha256']})
    parent={'sha256':'a'*64,'rows':2,'train_source_sha256':prior['sha256'],
        'bindings_sha256':'b'*64,'quota_sha256':'c'*64,'visits_sha256':'d'*64}
    child={'sha256':'e'*64,'rows':3,'train_source_sha256':current['sha256'],
        'bindings_sha256':'f'*64,'quota_sha256':'1'*64,'visits_sha256':'2'*64}
    verification={'schema':'natlang.earned-sft-source-transition-verification/1',
        'parent_sha256':parent['sha256'],'successor_sha256':child['sha256'],'proof':proof,
        'parent_source_sha256':prior['sha256'],'successor_source_sha256':current['sha256'],
        'old_quotas_preserved':True,'old_exposure_preserved':True,'old_rows':2,'new_rows':3,
        'changed_bindings':[{'id':'repair','binding_before':'old','binding_after':'new'}],
        'added_rows':1,'added_ids_sha256':digest(['addition'])}
    for side,checkpoint in [('before',parent),('after',child)]:
        for kind,key in [('bindings','bindings_sha256'),('quotas','quota_sha256'),('visits','visits_sha256')]:
            verification[kind+'_'+side+'_sha256']=checkpoint[key]
    if defect=='unreviewed-binding':verification['changed_bindings'][0]['binding_after']='invented'
    elif defect=='quota-loss':verification['old_quotas_preserved']=False
    elif defect=='wrong-parent':verification['parent_sha256']='3'*64
    elif defect=='missing-digest':parent.pop('bindings_sha256');verification.pop('bindings_before_sha256')
    elif defect=='wrong-source':child['train_source_sha256']='4'*64
    elif defect=='missing-addition':verification['added_rows']=0
    custody={'old_bindings_and_quotas_preserved':False,'old_obligations_transition':'authenticated-source-correction',
        'source_correction_transition':{'schema':'natlang.earned-sft-source-correction-custody/1',
            'parent_sha256':parent['sha256'],'successor_sha256':child['sha256'],'proof':proof,
            'verification':save('verification.json',verification)}}
    assert _completed_refresh_obligations(custody,parent,child) is (defect is None)


@pytest.mark.parametrize('tamper', ['none', 'missing', 'duplicate', 'nan', 'nested-nan', 'regressed', 'progress', 'changed-file'])
def test_interrupted_update_telemetry_checks_exact_order_and_finite_progress(tmp_path, tamper):
    from natlang_neuralese.rl.native_direct_sft_recovery import _interrupted_update_telemetry
    rows = [{'schema': 'natlang.native-training-progress/1', 'kind': 'native',
             'global_step': step, 'loss': 1.0, 'graphs': 2, 'forward_tokens': 8,
             'supervised_tokens': 4, 'communication_exposure': {'graph_visits': step}}
            for step in (65, 66, 67)]
    if tamper == 'missing': rows.pop(1)
    if tamper == 'duplicate': rows.insert(1, copy.deepcopy(rows[0]))
    if tamper == 'nan': rows[1]['loss'] = float('nan')
    if tamper == 'nested-nan': rows[1]['learning_rates'] = {'backbone': float('nan')}
    if tamper == 'regressed': rows[1]['communication_exposure']['graph_visits'] = 0
    train = tmp_path / 'train.jsonl'
    train.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    progress = tmp_path / 'progress.json'
    progress.write_text(json.dumps({} if tamper == 'progress' else rows[-1]))
    train_ref, progress_ref = binding(train), binding(progress)
    if tamper == 'changed-file': train.write_text('{}\n')
    if tamper == 'none':
        proof = _interrupted_update_telemetry(train_ref, progress_ref, 64, 67)
        assert proof['observed_updates'] == 3
    else:
        with pytest.raises(ValueError):
            _interrupted_update_telemetry(train_ref, progress_ref, 64, 67)


@pytest.mark.parametrize('tamper', ['none', 'alias', 'rolling-alias', 'rolling-alias-forged-journal', 'alias-corrupt', 'alias-path-forgery', 'checkpoint', 'snapshot', 'both-snapshots', 'source', 'journal', 'capacity', 'step'])
def test_interrupted_rolling_window_binds_retained_snapshot_launch_and_telemetry(tmp_path, tamper, monkeypatch):
    from natlang_neuralese.rl.native_direct_sft_recovery import _rolling_intermediate_window
    def file(name, data, *, raw=False):
        p = tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data if raw else json.dumps(data, sort_keys=True)); return binding(p)
    parent = file('parent.pt', 'parent', raw=True) | {'global_step': 64}
    slot = tmp_path/'learner/checkpoint.pt'; slot.parent.mkdir()
    retained = file('retained.pt', 'saved-full-state', raw=True)
    middle = {'path': str(slot), 'sha256': retained['sha256'], 'bytes': Path(retained['path']).stat().st_size}
    snapshot = {'schema':'natlang.rolling-sft-parent-custody-snapshot/1', 'checkpoint':middle,
                'global_step':67, 'optimizer_present':True, 'rng_present':True,
                'model_config_sha256':'a'*64, 'sft_frames':{'full':{'actual':'state'}}}
    actual = copy.deepcopy(snapshot); actual['checkpoint']['path'] = retained['path']
    if tamper == 'both-snapshots': snapshot['sft_frames']['full']['actual'] = 'forged'
    original = file('snapshot.json', snapshot)
    extracted = copy.deepcopy(snapshot); extracted['checkpoint']['path'] = retained['path']
    if tamper == 'snapshot': extracted['sft_frames']['full']['actual'] = 'forged'
    alias_path = None
    if tamper in ('alias', 'alias-corrupt', 'alias-path-forgery'):
        alias_path=tmp_path/'external'/'retained.pt';alias_path.parent.mkdir()
        alias_path.write_bytes(Path(retained['path']).read_bytes())
        Path(retained['path']).unlink();Path(retained['path']).symlink_to(alias_path)
        actual['checkpoint']['path']=str(alias_path.resolve())
        if tamper=='alias-corrupt':alias_path.write_bytes(b'forged')
        if tamper=='alias-path-forgery':extracted['checkpoint']['path']=str(alias_path)
    extracted_ref = file('extracted.json', extracted)
    calls = []
    def derive(checkpoint, sha256, directory):
        calls.append((str(checkpoint), sha256))
        path = Path(directory)/'derived.json'
        path.write_text(json.dumps(actual, sort_keys=True))
        return binding(path)
    monkeypatch.setattr('natlang_neuralese.rl.native_supervised._rolling_parent_custody_snapshot', derive)
    rows = [{'schema':'natlang.native-training-progress/1', 'kind':'native', 'global_step':n,
             'loss':1., 'graphs':2, 'forward_tokens':3, 'supervised_tokens':1,
             'communication_exposure':{'graph_visits':n}} for n in (65, 66, 67)]
    train=file('train.jsonl',''.join(json.dumps(row)+'\n' for row in rows),raw=True)
    progress=file('progress.json',rows[-1]);config=file('config.json',{});recipe=file('recipe.json',{})
    runtime=file('runtime/closure.json',{});launch=file('launch.sh',f"{Path(runtime['path']).parent} NATLANG_BF16_REDUCTION=fp32-accumulate {config['path']}",raw=True)
    request=file('request.json',{}); proof=file('proof.json',{'schema':'natlang.native-memory-qualification/1',
        'passed':True,'optimizer_updates':0,'parent_checkpoint_sha256':parent['sha256'],'request':request})
    capacity=file('capacity.json',{'passed':tamper!='capacity','optimizer_updates':0,
        'parent_checkpoint_sha256':parent['sha256'],'proof':proof,'request':request})
    roles={k:file(k+'.json',{'role':k}) for k in ('native_graphs','continued_native_graphs',
        'foundation_native_graphs','held_native_graphs','ce_supervision_policy')}
    manifest=file('manifest.json',{'schema':'natlang.behavior-selected-lineage-continuation/1',
        'parent':parent,'config':config,'recipe':recipe,'runtime_receipt':runtime,'roles':roles})
    argv=['python','-m','natlang_neuralese.train.native_state_sft','--continue-from',parent['path'],
          '--out',str(slot.parent),'--recipe-binding',json.dumps({'sha256':recipe['sha256']})]
    if tamper in ('rolling-alias', 'rolling-alias-forged-journal'):
        relocated = tmp_path/'external-rolling'/'checkpoint.pt'
        relocated.parent.mkdir()
        relocated.write_bytes(Path(retained['path']).read_bytes())
        slot.symlink_to(relocated)
        if tamper == 'rolling-alias-forged-journal':
            argv[argv.index('--out')+1] = str(relocated.parent)
    for role,flag in [('native_graphs','--native-graphs'),('continued_native_graphs','--continued-native-graphs'),
        ('foundation_native_graphs','--foundation-native-graphs'),('held_native_graphs','--held-native-graphs'),
        ('ce_supervision_policy','--ce-supervision-policy')]: argv += [flag,roles[role]['path']]
    if tamper == 'journal': argv[argv.index('--continue-from')+1] = 'wrong'
    journal=file('journal.txt',f"subprocess.CalledProcessError: Command '{argv!r}' died with signal 11.\n",raw=True)
    if tamper == 'source': Path(roles['native_graphs']['path']).write_text('forged')
    if tamper == 'checkpoint': Path(retained['path']).write_text('forged')
    refs={'manifest':manifest,'config':config,'recipe':recipe,'runtime':runtime,'launch':launch,
          'capacity':capacity,'capacity_request':request,'capacity_proof':proof,'journal':journal,
          'train':train,'progress':progress,'retained_checkpoint':retained,'extracted_snapshot':extracted_ref,
          'source_controls':roles}
    evidence=file('evidence.json',{'schema':'natlang.interrupted-sft-window-evidence/1','preflight':'not-observed','artifacts':refs})
    transition={'schema':'natlang.rolling-sft-resume/1','previous_checkpoint_path':parent['path'],
        'previous_checkpoint_sha256':parent['sha256'],'checkpoint_path':str(slot),
        'checkpoint_sha256':retained['sha256'],'custody_snapshot':original}
    receipt={'operation_parent':{'path':str(slot),'sha256':retained['sha256'],
                                 'global_step':68 if tamper == 'step' else 67}}
    args=({'schema':'natlang.interrupted-sft-intermediate-window/1','evidence':evidence},transition,
          parent,{'path':str(slot)},receipt)
    if tamper in ('none','alias','rolling-alias'):
        result=_rolling_intermediate_window(*args)
        assert result['intermediate_updates']==3 and result['intermediate_checkpoint_bytes_verified_currently'] is True
        assert result['intermediate']['path'] == str(slot)
    else:
        with pytest.raises(ValueError): _rolling_intermediate_window(*args)
    if tamper not in ('checkpoint','alias-corrupt'):
        assert calls == [(str(alias_path.resolve()) if alias_path else retained['path'], retained['sha256'])]
