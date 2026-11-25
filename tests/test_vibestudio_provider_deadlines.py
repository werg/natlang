"""Native provider failure accounting: no model/provider calls."""
import copy
import json
import os
from pathlib import Path
import gzip
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'neuralese_data'))
import vibestudio_campaign as campaign


def native_deadline(kind='generation'):
    return {'rootCallIds':['root'],'rootOutcome':'error','captureErrors':[],
            'executionError':{'name':'NatlangCallError','message':'natlang child: failed: The operation was aborted due to timeout'},
            'calls':[{'record':{'call_id':'root','parent_call_id':None,'outcome':'error'}},
                     {'record':{'call_id':'child','parent_call_id':'root','outcome':'failed',
                                'executor':{'kind':'agent','model_id':'test:provider'},
                                'model_turns':[{'schema':'natlang.model-exchange/1','kind':kind,
                                                'request':{'invocation_id':'child','messages':[{'role':'user','content':'test'}]},
                                                'error':{'name':'Error','message':'The operation was aborted due to timeout'}}]},
                      'events':[{'kind':'model_request','phase':'error','call_id':'child','actor':{'id':'test:provider'}}]}]}

@pytest.mark.parametrize('kind',['generation','score'])
def test_captured_deadline_is_infrastructure_without_changing_sampling_evidence(kind):
    teacher=native_deadline(kind)
    teacher['calls'][1]['record']['cost']={'model_requests':2,'tokens':37}
    before=copy.deepcopy(teacher)
    evidence=campaign.captured_provider_transport_errors(teacher)
    assert len(evidence)==1 and evidence[0]['call_id']=='child' and evidence[0]['kind']==kind
    assert teacher==before
    outcome={'status':'test-failed','reason':teacher['executionError']['message'],'captured_provider_transport_errors':evidence}
    assert campaign.infrastructure_failure(outcome,{})

@pytest.mark.parametrize('change',['quoted-model-answer','tool-timeout','wrong-request','wrong-model','no-request-events',
                                   'completed-request-event','completed-call','recovered-request','unknown-error-name',
                                   'authored-abort','authored-timeout','budget-cancellation','budget-quiescence','cancelled-root','wrong-schema'])
def test_timeout_text_alone_never_classifies_a_goal_as_infrastructure(change):
    teacher=native_deadline();call=teacher['calls'][1];record=call['record'];exchange=record['model_turns'][0]
    if change=='quoted-model-answer':exchange.pop('error');exchange['reply']={'text':'The operation was aborted due to timeout'}
    elif change=='tool-timeout':record['model_turns']=[];record['detail']='The operation was aborted due to timeout'
    elif change=='wrong-request':exchange['request']['invocation_id']='foreign'
    elif change=='wrong-model':call['events'][0]['actor']['id']='other:model'
    elif change=='no-request-events':call['events']=[]
    elif change=='completed-request-event':call['events'][0]['phase']='end'
    elif change=='completed-call':record['outcome']='done'
    elif change=='recovered-request':record['model_turns'].append({'kind':'generation','reply':{'text':'actual model reply'}})
    elif change=='unknown-error-name':exchange['error']['name']='ToolTimeoutError'
    elif change=='authored-abort':teacher['executionError']['name']='AbortError'
    elif change=='authored-timeout':teacher['executionError']['name']='NatlangTimeoutError'
    elif change=='budget-cancellation':teacher['executionError']['name']='NatlangModelBudgetError'
    elif change=='budget-quiescence':record['outcome']='quiesced'
    elif change=='cancelled-root':teacher['rootOutcome']='cancelled';teacher['calls'][0]['record']['outcome']='cancelled'
    else:exchange['schema']='untrusted.report/1'
    assert campaign.captured_provider_transport_errors(teacher)==[]
    assert not campaign.provider_transport_failure({'status':'test-failed','reason':teacher['executionError']['message'],
                                                    'captured_provider_transport_errors':[]})


def test_legacy_root_timeout_prose_is_not_a_provider_witness():
    assert not campaign.provider_transport_failure({'status':'test-failed','reason':'natlang user: failed: The operation was aborted due to timeout'})
