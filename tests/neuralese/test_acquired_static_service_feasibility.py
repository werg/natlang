import copy
from scripts.neuralese_data.audit_acquired_static_service_feasibility import inspect_session


def sample():
    return {'trajectory':{'id':'upstream', 'tools':[{'type':'function','function':{'name':'browser.search','parameters':{'type':'object'}}}],
        'messages':[{'role':'assistant','tool_calls':[{'id':'a','function':{'name':'browser.search','arguments':'{"query":"x"}'}}]},
                    {'role':'tool','tool_call_id':'a','name':'browser.search','content':'original observation'}],
        'outcome':{'label':'teacher','checked':None}}}


def test_pairs_original_tool_observation_without_claiming_native_execution():
    result=inspect_session(sample())
    assert result['structural_pairing_errors']==[]
    assert result['observations'][0]['result']=='original observation'
    assert result['source_observations_not_native_execution'] and not result['training_admission']


def test_rejects_missing_result_and_undeclared_tool():
    row=sample();row['trajectory']['messages'].pop();row['trajectory']['messages'][0]['tool_calls'][0]['function']['name']='browser.ghost'
    errors=inspect_session(row)['structural_pairing_errors']
    assert any('declared tool schema' in x for x in errors)
    assert any('lack a recorded observation' in x for x in errors)
