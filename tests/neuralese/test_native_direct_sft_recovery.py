import json
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
