import json
import pytest
from natlang_neuralese.rl.native_direct_policy_publish import publish
from natlang_neuralese.train.native_resident import binding


def test_direct_policy_publication_requires_real_privileged_update_and_preserves_breadth(tmp_path):
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return binding(p)
    parent=write('parent',{'step':56});child=write('child',{'step':57});state=tmp_path/'state.json'
    state.write_text(json.dumps({'phase':'collect','checkpoint':parent['path'],'round':11,'offset':57,'trained_rounds':11}))
    cert=write('cert',{'passed':True});report=write('report',{'completed':True,'numerics_ok':True,'checkpoint_sha256':child['sha256'],'runtime_certificate':cert|{'passed':True}})
    receipt=write('receipt',{'checkpoint':{'sha256':child['sha256']},'continuation':{'checkpoint_sha256':parent['sha256']}})
    common={'optimizer_present':True,'rng_present':True};custody=write('custody',{'schema':'natlang.earned-rl-checkpoint-custody/1','parent':parent|common|{'global_step':56},'successor':child|common|{'global_step':57},'sft_source_quota_exposure_unchanged':True})
    task='one-task';p=tmp_path/'rollouts';p.write_text(''.join(json.dumps({'task_id':task,'group':1,'sample':i,'policy':{'checkpoint_sha256':parent['sha256']}})+'\n' for i in range(6)))
    score=write('score',{'groups':[{'task_id':task,'scored':6}]});provenance=write('provenance',{'schema':'natlang.completed-same-policy-group/1','checkpoint_sha256':parent['sha256'],'rollouts':binding(p),'score':score})
    objective=write('objective',{'rows':{str(i):{'task_id':task,'sample':i,'jolt':{'privileged':{'turn':{}}}} for i in range(6)}})
    log=tmp_path/'log';log.write_text(json.dumps({'global_step':57,'objective_details':{'jolt':{'privileged_episodes':6,'student_opsd':0,'student_outcome':0}}})+'\n')
    args=dict(report_ref=report,receipt_ref=receipt,custody_ref=custody,provenance_ref=provenance,objective_ref=objective,log_ref=binding(log),out=tmp_path/'publication')
    with pytest.raises(ValueError,match='declared text-twin'):publish(state,**args)
    log.write_text(json.dumps({'global_step':57,'objective_details':{'jolt':{'privileged_episodes':6,'student_opsd':0.1,'student_outcome':0}}})+'\n');args['log_ref']=binding(log)
    publish(state,**args);s=json.loads(state.read_text());assert (s['round'],s['offset'],s['trained_rounds'])==(12,57,12)
    assert s['checkpoint']==child['path'] and s['trained_task_ids']==[task]
