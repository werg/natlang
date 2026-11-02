import pytest
from scripts.neuralese_data.audit_acquired_source_call_schemas import inspect


def fixture():
    return {'schema':'natlang.acquired-static-service-feasibility/1','training_admission':False,
        'tool_declarations':[{'type':'function','function':{'name':'browser.search','parameters':{
            'type':'object','properties':{'query':{'type':'string'}},'required':['query']}}}],
        'observations':[{'name':'browser.search','call_id':'a','arguments':{'query':'x'}}]}


def test_validates_observed_arguments_against_original_schema():
    data=fixture();assert inspect(data)['argument_schema_failures']==[]
    data['observations'][0]['arguments']={'query':7}
    assert inspect(data)['argument_schema_failures'][0]['call_id']=='a'


def test_remote_schema_references_are_rejected():
    data=fixture();data['tool_declarations'][0]['function']['parameters']={'$ref':'https://example.invalid/schema'}
    with pytest.raises(ValueError,match='external schema'):inspect(data)
