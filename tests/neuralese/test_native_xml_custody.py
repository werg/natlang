import copy,hashlib,json
from pathlib import Path
import pytest
from natlang_neuralese.data.native_xml_custody import authenticate,derived_tool_calls,validate_replay_turn,render_packet,POLICY_SHA256
from natlang_neuralese.rl.task_rollouts import rollout_turns
from natlang_neuralese.common import paths

LFM3=Path.home()/'.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
CAPTURED_GRAPH80=paths.resolve('data_nvme','dgxops/claude-recovery-20261028/data/training-bug-audit-v1/synthetic-wrapper-real-xml-turn-schema-v2.json')


def fixture():
    xml='<tool_call>\n<function=eval>\n<parameter=code>\nconst a = 1 < 2;\n</parameter>\n</function>\n</tool_call>'
    tools=[{'type':'function','function':{'name':'eval','parameters':{'type':'object','properties':{'code':{'type':'string'}},'required':['code'],'additionalProperties':False}}}]
    request={'invocation_id':'i','messages':[{'role':'user','content':'x'}],'tools':[],'response_tools':tools,'seed':3,'max_tokens':100}
    raw={'choices':[{'finish_reason':'stop','message':{'content':xml,'reasoning_content':'thinking'}}]}
    encode=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    req=encode({k:v for k,v in request.items() if k!='invocation_id'});response=encode(raw);sha=lambda x:hashlib.sha256(x.encode()).hexdigest()
    calls=[['eval',{'code':'const a = 1 < 2;'}]]
    return {'phase':'action','invocation_id':'i','context':copy.deepcopy(request['messages']),'tools_offered':[],
      'response_tools':copy.deepcopy(tools),'model_request':request,'model_request_canonical':req,'request_sha256':sha(req),
      'raw_response_sha256':sha(response),'model_response':{'calls':calls,'text':'','raw_calls':[],
        'raw_response':raw,'raw_response_canonical':response,'tool_parse':{'schema':'natlang.native-xml-tool-parse/1',
         'source':'assistant-content','schema_custody':'response_tools','policy_sha256':POLICY_SHA256,'wire_tool_aliases':{}}},
      'assistant':{'content':'','reasoning':'thinking','calls':[{'tool':'eval','source_tool':'eval','arguments':calls[0][1],'call_id':None}]}}


def test_authenticate_rejects_changed_source_schemas_policy_and_claimed_calls():
    turn=fixture();assert authenticate(turn)==[['eval',{'code':'const a = 1 < 2;'}]]
    for change in (lambda x:x['model_response']['raw_response']['choices'][0]['message'].update(content='bad'),
                   lambda x:x['response_tools'][0]['function'].update(name='other'),
                   lambda x:x['assistant']['calls'][0]['arguments'].update(code='wrong'),
                   lambda x:x['model_response']['tool_parse'].update(policy_sha256='0'*64),
                   lambda x:x['model_request'].update(seed=4),lambda x:x['model_response']['raw_calls'].append({})):
        broken=copy.deepcopy(turn);change(broken)
        with pytest.raises(ValueError):authenticate(broken)
    broken=copy.deepcopy(turn);broken['model_response']['raw_response']['choices'][0]['message']['content']='prefix '+broken['model_response']['raw_response']['choices'][0]['message']['content']
    encoded=json.dumps(broken['model_response']['raw_response'],sort_keys=True,separators=(',',':'));broken['model_response']['raw_response_canonical']=encoded;broken['raw_response_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
    with pytest.raises(ValueError):authenticate(broken)


def test_rollout_projection_retains_actual_empty_prompt_tools_and_reauthenticates():
    source=fixture();row=rollout_turns({'trajectory':[source]})[0]
    assert row['tools'] is None and row['response_tools']==source['response_tools']
    assert row['target']['tool_calls']==derived_tool_calls(source)
    assert validate_replay_turn(row)==authenticate(source)
    forged=copy.deepcopy(row);forged['target']['tool_calls'][0]['function']['arguments']='{"code":"wrong"}'
    with pytest.raises(ValueError):validate_replay_turn(forged)


def test_render_exact_actual_xml_and_refuse_noncanonical_frame():
    from transformers import AutoTokenizer
    from natlang_neuralese.serve.chat import bind_history_reasoning
    base=LFM3
    if not base.is_dir():pytest.skip('pinned real tokenizer unavailable')
    tokenizer=AutoTokenizer.from_pretrained(base,trust_remote_code=False);bind_history_reasoning(tokenizer,policy='keep')
    turn=fixture();text,ids,boundary=render_packet(tokenizer,turn)
    assert text.endswith(turn['model_response']['raw_response']['choices'][0]['message']['content']+'<|im_end|>\n')
    assert 0<boundary<len(ids)
    assert '<tools>' not in text


def test_authenticated_xml_admission_uses_exact_action():
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import admitted_teacher_turn
    turn=fixture();row={'id':'row'};decisions={('row',0):{'approved':True,'invocation_id':'i','raw_response_sha256':turn['raw_response_sha256']}}
    assert admitted_teacher_turn(row,0,turn,decisions)[0] is True
    forged=copy.deepcopy(turn);forged['model_response']['raw_response']['choices'][0]['message']['content']+='x'
    with pytest.raises(ValueError):admitted_teacher_turn(row,0,forged,decisions)


def test_dpo_uses_exact_authenticated_xml_projection_and_preserves_prompt_tools():
    from transformers import AutoTokenizer
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from natlang_neuralese.data.native_preference import pair_graph,SCHEMA_PREFERENCE,KIND
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    base=LFM3
    if not base.is_dir():pytest.skip('pinned real tokenizer unavailable')
    tokenizer=AutoTokenizer.from_pretrained(base,trust_remote_code=False);bind_history_reasoning(tokenizer,policy='keep')
    turn=fixture();target={'role':'assistant','content':'','reasoning_content':'thinking','tool_calls':derived_tool_calls(turn)}
    alternatives={'chosen':{'split':'train','split_groups':['world'],'tools':[], 'messages':turn['context']+[target],'native_xml_turns':{'1':turn}},
                  'rejected':{'split':'train','split_groups':['world'],'tools':[], 'messages':turn['context']+[{'role':'assistant','content':'wrong'}]}}
    record={'schema':SCHEMA_PREFERENCE,'required_objective':KIND,'offline':True,'native_on_policy':False,'split':'train','source_groups':['world'],'common_message_count':1,'source':{'kind':'test'},'alternatives':alternatives}
    graph,_=pair_graph(record,tokenizer,tokenizer_fingerprint=tokenizer_fingerprint(tokenizer))
    assert len(graph['nodes'])==2 and all(not any(n['target_weights']) for n in graph['nodes'])
    forged=copy.deepcopy(record);forged['alternatives']['chosen']['native_xml_turns']['1']['assistant']['calls'][0]['arguments']['code']='wrong'
    with pytest.raises(ValueError):pair_graph(forged,tokenizer,tokenizer_fingerprint=tokenizer_fingerprint(tokenizer))


def test_real_captured_assistant_content_with_synthetic_custody_envelope():
    # The content is the saved graph-80 model response; its request envelope was
    # reconstructed for this test and is not claimed as the original transport.
    path=CAPTURED_GRAPH80
    if not path.is_file():pytest.skip('captured graph-80 response unavailable')
    turn=json.loads(path.read_text())
    assert authenticate(turn) and turn['model_response']['raw_calls']==[]
    forged=copy.deepcopy(turn);forged['assistant']['calls'][0]['arguments']={}
    with pytest.raises(ValueError):authenticate(forged)


@pytest.mark.parametrize('captured', [False, True])
def test_capture_to_offline_pair_to_dpo_and_forged_evidence(captured):
    from scripts.neuralese_data.trajectory import Trajectory,append_captured_native_xml_turn
    from scripts.neuralese_data.acquired_preferences import bound_preference_record
    from natlang_neuralese.data.native_preference import pair_graph
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from transformers import AutoTokenizer
    base=LFM3
    if not base.is_dir():pytest.skip('pinned real tokenizer unavailable')
    tok=AutoTokenizer.from_pretrained(base,trust_remote_code=False);bind_history_reasoning(tok,policy='keep')
    source=fixture()
    if captured:
        path=CAPTURED_GRAPH80
        if not path.is_file():pytest.skip('captured graph-80 response unavailable')
        source=json.loads(path.read_text())
    def trajectory(name):
        return Trajectory(id=name,family='actual-capture',messages=copy.deepcopy(source['context']),head=1,noun='task',
            lineage={'actor':'captured-actor'},license={},split='train',split_groups=['world'],outcome={},tools=[])
    chosen=trajectory('chosen');append_captured_native_xml_turn(chosen,source)
    rejected=trajectory('rejected');rejected.messages.append({'role':'assistant','content':'wrong'})
    binding={'source_file':{'sha256':'a'*64},'source_row':'pair','source_revision':'actual-revision'}
    record=bound_preference_record({'chosen':chosen,'rejected':rejected},binding)
    graph,_=pair_graph(record,tok,tokenizer_fingerprint=tokenizer_fingerprint(tok))
    assert len(graph['nodes'])==2 and record['alternatives']['chosen']['native_xml_turns']['1']==source
    assert source['model_response']['raw_calls']==[] and record['source']==binding
    for field in ('request','schema','action'):
        broken=copy.deepcopy(chosen)
        turn=broken.native_xml_turns['1']
        if field=='request':turn['model_request']['seed']='forged'
        elif field=='schema':turn['response_tools'][0]['function']['name']='forged'
        else:turn['assistant']['calls'][0]['arguments']['code']='forged'
        with pytest.raises(ValueError):bound_preference_record({'chosen':broken,'rejected':rejected},binding)


def test_failure_repair_retains_and_rechecks_sampled_xml_evidence():
    from test_native_failure_repair_dpo import record
    from natlang_neuralese.data.native_failure_repair import _merge_graphs
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    pair=record();alt=pair['alternatives']['failed']
    turn=rollout_turns({'trajectory':[fixture()]})[0]
    alt['native_xml_turns']=[turn]
    capture=alt['execution_receipt'];capture['native_xml_turns_sha256']=canonical_json_sha256_hex([turn])
    alt['execution_capture_sha256']=canonical_json_sha256_hex(capture)
    replay=alt['graph_replay_receipt'];replay['execution_capture_sha256']=alt['execution_capture_sha256']
    alt['graph_replay_sha256']=canonical_json_sha256_hex(replay)
    _merge_graphs(pair,'tok')
    forged=copy.deepcopy(pair);forged['alternatives']['failed']['native_xml_turns'][0]['target']['tool_calls'][0]['function']['arguments']='{}'
    with pytest.raises(ValueError):_merge_graphs(forged,'tok')


def test_schema_semantics_reference_structural_equality_and_boolean_values():
    from natlang_neuralese.data.native_xml_custody import _valid
    schema={'$defs':{'item':{'anyOf':[{'type':'null'},{'type':'object','const':{'a':1,'b':'😀'}}]}},'$ref':'#/$defs/item'}
    assert _valid(schema,{'b':'😀','a':1}) and _valid(schema,None)
    assert not _valid(schema,{'a':True,'b':'😀'})
    assert not _valid({'type':'array','uniqueItems':True,'items':True},[{'a':1,'b':2},{'b':2,'a':1}])
    assert _valid({'type':'array','items':True},[1,None]) and not _valid({'type':'array','items':False},[1])
    assert _valid({'type':['null','string'],'maxLength':1},'😀')
    assert not _valid({'type':'string','maxLength':1},'😀x')


def test_schema_reference_authenticates_typed_response():
    turn=fixture();params=turn['response_tools'][0]['function']['parameters']
    params['$defs']={'source':{'type':'string','minLength':1}}
    params['properties']['code']={'$ref':'#/$defs/source'}
    turn['model_request']['response_tools']=copy.deepcopy(turn['response_tools'])
    request={k:v for k,v in turn['model_request'].items() if k!='invocation_id'}
    encoded=json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    turn['model_request_canonical']=encoded;turn['request_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
    assert authenticate(turn)==[['eval',{'code':'const a = 1 < 2;'}]]


def test_typed_additional_properties_match_native_parser():
    turn=fixture();params=turn['response_tools'][0]['function']['parameters']
    params['additionalProperties']={'type':'number'}
    turn['model_request']['response_tools']=copy.deepcopy(turn['response_tools'])
    request={k:v for k,v in turn['model_request'].items() if k!='invocation_id'}
    encoded=json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    turn['model_request_canonical']=encoded;turn['request_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
    raw=turn['model_response']['raw_response'];raw['choices'][0]['message']['content']=raw['choices'][0]['message']['content'].replace('</function>', '<parameter=count>\n3\n</parameter>\n</function>')
    encoded=json.dumps(raw,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    turn['model_response']['raw_response_canonical']=encoded;turn['raw_response_sha256']=hashlib.sha256(encoded.encode()).hexdigest()
    turn['model_response']['calls'][0][1]['count']=3;turn['assistant']['calls'][0]['arguments']['count']=3
    assert authenticate(turn)[0][1]['count']==3
