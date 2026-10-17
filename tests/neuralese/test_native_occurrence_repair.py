import copy
import json
import pytest
from natlang_neuralese.rl.native_occurrence_repair import repair_tool_history_reads
from natlang_neuralese.rl.episode_graph import validate_rollout_graph


def fixture():
    identity={'architecture':'lfm2','checkpoint_sha256':'a'*64,'tokenizer_fingerprint':'b'*64,'embedding_weight_sha256':'c'*64}
    block='nzs1_'+'d'*52
    part={'type':'native_state','id':block,'source_occurrence':'call#write','stop_gradient':True}
    capture={'kind':'native_state','id':block,'source_occurrence':'call#write','stop_gradient':True,'native_identity':identity}
    return {'policy':{'checkpoint_sha256':'a'*64},'reward':{'quality':0},'turns':[
        {'invocation_id':'call','messages':[],'target':{'content':'actual'},'sampling':{'writes':[{'state_id':block}]}},
        {'invocation_id':'call','messages':[{'role':'assistant','tool_calls':[{'function':{'name':'return_result','arguments':json.dumps({'value':[part],'again':[part]})}}]}],
         'target':{'content':'sampled unchanged'},'sampling':{'prompt':[capture,copy.deepcopy(capture)],'samples':[{'token_id':42}]}}],
        'execution_graph':[{'kind':'model_turn','node':'call#turn1','call_id':'call'},
            {'kind':'block_write','node':'call#write','turn':'call#turn1','block':block,'native_identity':identity},
            {'kind':'model_turn','node':'call#turn2','call_id':'call'}]}


def test_exact_tool_json_history_continuation_preserves_samples_rewards_and_original_graph():
    original=fixture();before=copy.deepcopy(original)
    with pytest.raises(ValueError,match='inventories'):validate_rollout_graph(original)
    repaired,added=repair_tool_history_reads(original)
    assert len(added)==2
    validate_rollout_graph(repaired)
    assert original==before
    assert repaired['turns']==before['turns']and repaired['reward']==before['reward']and repaired['policy']==before['policy']
    assert repaired['execution_graph'][:3]==before['execution_graph']
    assert all(e['stop_gradient']for e in repaired['execution_graph'][3:])
    assert repair_tool_history_reads(repaired)==(repaired,[])


@pytest.mark.parametrize('damage',['argument-id','argument-source','argument-gradient','capture-identity','missing-writer','wrong-writer','duplicate-node'])
def test_missing_read_requires_actual_argument_and_exact_known_writer(damage):
    row=fixture()
    if damage.startswith('argument-'):
        fn=row['turns'][1]['messages'][0]['tool_calls'][0]['function'];value=json.loads(fn['arguments'])
        for part in (value['value'][0],value['again'][0]):
            if damage=='argument-id':part['id']='nzs1_'+'x'*52
            elif damage=='argument-source':part['source_occurrence']='foreign'
            else:part.pop('stop_gradient')
        fn['arguments']=json.dumps(value)
    elif damage=='capture-identity':row['turns'][1]['sampling']['prompt'][0]['native_identity']={}
    elif damage=='missing-writer':row['execution_graph'].pop(1)
    elif damage=='wrong-writer':row['execution_graph'][1]['block']='nzs1_'+'x'*52
    else:row['execution_graph'].append(copy.deepcopy(row['execution_graph'][0]))
    with pytest.raises(ValueError):repair_tool_history_reads(row)
