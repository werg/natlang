import hashlib
import json
from pathlib import Path

import pytest

from scripts.coreweave.teacher_batch_completion import batch_completion


def fixture(tmp_path, legacy=False):
    source=tmp_path/'source.jsonl';source.write_text('immutable source\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    jobs=tmp_path/'jobs';jobs.mkdir();error=jobs/'000000.error.json'
    content={'index':0,'program_id':'case','code':'NATLANG_MODEL_REQUEST_BUDGET'}
    if not legacy:content.update(source_sha256=sha(source),program_ir_sha256='a'*64)
    error.write_text(json.dumps(content));output=tmp_path/'results.jsonl'
    Path(str(output)+'.manifest.json').write_text(json.dumps({'version':'natlang.teacher_batch.native/1',
        'source':str(source),'source_sha256':sha(source),'range':{'start':0,'count':1},'missing':[0]}))
    entry={'key':'budget-case','index':0,'source':str(source),'jobs':str(jobs),'output':str(output)}
    state={'index':0,'program_id':'case','digest':'a'*64,'state':'failed',
        'resource_limit_reason':'model_request_budget','terminal_error':{'path':str(error),'sha256':sha(error)}}
    finish={'event':'finish','key':entry['key'],'status':'incomplete','resource_limit_reason':'model_request_budget',
        'output_accounting':{'version':'natlang.supervisor_output_accounting/2','complete':False,
            'manifest_valid':True,'expected_jobs':1,'job_states':[state]}}
    score={'schema':'natlang.s7-task-rollout-score/1','rollouts':1,'outcome_counts':{'budget_exhaustion':1}}
    return entry,finish,score,error


@pytest.mark.parametrize('legacy',[False,True])
def test_recorded_budget_failures_do_not_idle_continuous_teacher(tmp_path,legacy):
    entry,finish,score,_=fixture(tmp_path,legacy)
    proof=batch_completion([entry],[finish],[2],score)
    assert proof['can_advance'] and not proof['training_admission']
    assert len(proof['model_budget_failures'])==1 and not proof['repair_required']
    assert proof['model_budget_failures'][0]['errors'][0]['per_error_source_digests_present'] is not legacy


@pytest.mark.parametrize('mutation',['missing-finish','error-bytes','error-kind','source','manifest','transport','worker-crash','score'])
def test_incomplete_transport_or_unbound_failures_still_require_repair(tmp_path,mutation):
    entry,finish,score,error=fixture(tmp_path);journal=[finish];codes=[2]
    if mutation=='missing-finish':journal=[]
    elif mutation=='error-bytes':error.write_text('{}')
    elif mutation=='error-kind':finish['output_accounting']['job_states'][0]['state']='transport_failed'
    elif mutation=='source':Path(entry['source']).write_text('changed source\n')
    elif mutation=='manifest':Path(entry['output']+'.manifest.json').write_text('{}')
    elif mutation=='transport':finish['resource_limit_reason']='case_wall_budget'
    elif mutation=='worker-crash':codes=[1]
    else:score['rollouts']=0
    assert not batch_completion([entry],journal,codes,score)['can_advance']


def test_complete_exact_model_failure_is_a_closed_collection_not_a_success_label(tmp_path):
    entry,finish,score,_=fixture(tmp_path)
    finish['output_accounting'].update(complete=True,job_states=[{'state':'result'}])
    score['outcome_counts']={'semantic_mismatch':1}
    proof=batch_completion([entry],[finish],[2],score)
    assert proof['can_advance'] and proof['exact_completed_queue_keys']==1
    assert proof['model_budget_failures']==[] and proof['training_admission'] is False
