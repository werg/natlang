import json
from types import SimpleNamespace

import pytest

from scripts.neuralese_data import verify_acquired_diagnosis_targets as verify
from scripts.neuralese_data.audit_acquired_learning_signal import row_signal
from natlang_neuralese.train.own_output_supervision import policy_objective
from natlang_neuralese.common.hashing import canonical_json_sha256_hex


def fixture(tmp_path,monkeypatch):
    from transformers import AutoTokenizer
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:SimpleNamespace(
        decode=lambda ids,**kwargs:''.join(chr(token-100) for token in ids if token>=100)))
    def write(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return path
    roles={'tokenizer_fingerprint':'f'*64,'chat_template_sha256':'d'*64,'start':10,'end':17,
        'roles':{'system':11,'user':12,'assistant':13,'tool':14},'think_open':15,'think_close':16,
        'generation_prompt':[10,13],'tool_response_prefixes':[[30,31]]}
    policy=write('policy.json',{'objective':{'name':'own-outputs/1','mass':'token','reasoning_share':.1,'role_tokens':roles}})
    objective=policy_objective(json.loads(policy.read_text()))
    text=json.dumps({'schema':'natlang.source-execution-diagnosis/1',
                     'source_annotations':{'mast_annotation':'retained-source-label'}})
    ids=[10,12,200,17,10,13]+[ord(char)+100 for char in text]+[17]
    row={'id':'diagnosis','split':'train','source_groups':['world:diagnosis'],'tokenizer_fingerprint':'f'*64,
         'nodes':[{'name':'text','token_ids':ids,'target_weights':[0]+[1]*(len(ids)-1)}],
         'conversion':{'source_row_id':'source:diagnosis','acquired_session':{'source_lineage':{
             'store':'mcemri/MAST-Data','notes':{'source_execution_diagnosis':{
                 'failed_actions_positive_targets':False,'source_labels_unchanged':True,
                 'source_metadata':{'mast_annotation':'retained-source-label'}}}}}}}
    graph=tmp_path/'graphs.jsonl';graph.write_text(json.dumps(row)+'\n')
    screen=write('screen.json',{'output':{**verify.bind(graph),'rows':1}})
    signals=tmp_path/'signals.jsonl';signals.write_text(json.dumps(row_signal(row,objective))+'\n')
    census=write('census.json',{'screen':verify.bind(screen),'ce_policy':verify.bind(policy),
        'objective_sha256':canonical_json_sha256_hex(objective),'row_signals':verify.bind(signals),
        'counts':{'source_label_diagnosis_rows':1}})
    return screen,census,policy,row


def test_verifies_actual_effective_reply_against_source_annotations(tmp_path,monkeypatch):
    screen,census,policy,row=fixture(tmp_path,monkeypatch)
    result=verify.verify(screen,census,policy,'fake-base',tmp_path/'proof.json')
    proof=json.loads((tmp_path/'proof.json').read_text())
    assert result['verified_rows']==1 and proof['rows'][0]['id']=='diagnosis'
    assert proof['census']==verify.bind(census)


def test_census_graph_mismatch_never_publishes_target_proof(tmp_path,monkeypatch):
    screen,census,policy,row=fixture(tmp_path,monkeypatch)
    row['nodes'][0]['token_ids'][-2]+=1
    graph=tmp_path/'graphs.jsonl';graph.write_text(json.dumps(row)+'\n')
    value=json.loads(screen.read_text());value['output']={**verify.bind(graph),'rows':1};screen.write_text(json.dumps(value))
    value=json.loads(census.read_text());value['screen']=verify.bind(screen);census.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='graph differs from completed census'):
        verify.verify(screen,census,policy,'fake-base',tmp_path/'proof.json')
    assert not (tmp_path/'proof.json').exists()
