"""Actual pinned tokenizers retain full context, tool results and earlier reasoning."""
from pathlib import Path
import copy
import pytest
from transformers import AutoTokenizer
from scripts.neuralese_data.build_acquired_full_session_graphs import session_graphs, held_session_targets
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint

BASES=[Path('/home/werg/data/dgxops/qad-drift-probe/base'),
       Path('/home/werg/data/models/mellum21-12b-a2.5b-thinking')]


def value():
    messages=[{'role':'user','content':'SOURCE_BEGIN '+('useful context '*10000)},
        {'role':'assistant','content':None,'reasoning':'EARLIER_REASONING_TARGET',
         'tool_calls':[{'id':'c1','type':'function','function':{'name':'inspect','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'c1','content':'EXACT_OBSERVATION'},
        {'role':'assistant','reasoning':'LATEST_REASONING_TARGET','content':'COMPLETE_ANSWER'}]
    return {'schema':'natlang.normalized-agent-trajectory/1','whole_trajectory':True,'windowed':False,
        'positive_candidate':True,'source_row':'source.parquet:17',
        'source_file':{'path':'source.parquet','sha256':'a'*64},
        'source_revision':'revision','trajectory':{'id':'source-session','split_groups':['source-task'],
        'messages':messages,'tools':[{'type':'function','function':{'name':'inspect','description':'inspect',
        'parameters':{'type':'object','properties':{}}}}],
        'outcome':{'label':'teacher'},'lineage':{'teacher':'upstream'},'license':{'name':'upstream'}}}


@pytest.mark.parametrize('base',BASES,ids=['lfm','mellum'])
def test_both_real_templates_cover_full_context_and_every_reasoning_target(base):
    if not (base/'tokenizer.json').is_file():
        pytest.skip('pinned production tokenizer unavailable on this machine')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    source_fp=tokenizer_fingerprint(tokenizer);provision(tokenizer)
    original=value();saved=copy.deepcopy(original)
    graphs=list(session_graphs(original,tokenizer,source_fingerprint=source_fp,
        expanded_fingerprint=tokenizer_fingerprint(tokenizer),split='test'))
    assert original==saved and len(graphs)==3
    texts=[tokenizer.decode(graph['nodes'][0]['token_ids'],skip_special_tokens=False) for graph in graphs]
    assert 'SOURCE_BEGIN ' in texts[0] and texts[0].count('useful context ')==10000
    assert 'EXACT_OBSERVATION' in texts[0] and 'COMPLETE_ANSWER' in texts[0]
    assert 'LATEST_REASONING_TARGET' in texts[0] and 'EARLIER_REASONING_TARGET' in texts[1]
    for graph in graphs:
        assert graph['split']=='test' and graph['source_groups']==['source-task']
        node=graph['nodes'][0]
        assert len(node['token_ids'])==len(node['target_weights']) and node['target_weights'][0]==0
        assert max(node['target_weights'])>0
        assert max(node['target_weights'])>min(weight for weight in node['target_weights'] if weight>0)
        assert graph['conversion']['acquired_session']['oracle_verified'] is False
        assert graph['conversion']['acquired_session']['truncated'] is False


def test_failure_is_never_positive_sft():
    row=value();row['trajectory']['outcome']['label']='failed'
    with pytest.raises(ValueError,match='non-positive'):
        list(session_graphs(row,None,source_fingerprint='a'*64,expanded_fingerprint='b'*64,split='train'))


def test_original_target_identity_closes_decision_hold_descendants():
    row=value()
    assert held_session_targets(row,{'source-session:1'})==['source-session:1']
    assert held_session_targets(row,{'unrelated:1'})==[]


def test_cli_routes_complete_groups_and_omits_held_targets(tmp_path,monkeypatch):
    import json,sys
    from scripts.neuralese_data import build_acquired_full_session_graphs as builder
    from natlang_neuralese.common.jsonio import write_jsonl_text,open_jsonl_text
    from natlang_neuralese.common.hashing import sha256_file_hex
    rows=[]
    for name,split,group in [('a','train','group-a'),('b','test','group-a'),
                             ('c','train','group-c'),('d','train','group-d')]:
        row=value();row['trajectory'].update(id=name,split=split,split_groups=[group]);rows.append(row)
    rows[3]['positive_candidate']=False
    source=tmp_path/'full-trajectories.jsonl.zst'
    with write_jsonl_text(source) as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    receipt=tmp_path/'export.json'
    receipt.write_text(json.dumps({'schema':'natlang.normalized-agent-source-file/1',
        'contract':{'source':rows[0]['source_file'],'revision':rows[0]['source_revision']},
        'outputs':{source.name:sha256_file_hex(source)},'complete_sessions':len(rows)}))
    monkeypatch.setattr(builder.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    monkeypatch.setattr(builder,'tokenizer_fingerprint',lambda t:'a'*64)
    monkeypatch.setattr(builder,'provision',lambda t:None)
    monkeypatch.setattr(builder,'current_protection',lambda p:{'held':set(),'quality':set(),'pending':set(),'proof':{}})
    holds=sorted((Path(builder.__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    monkeypatch.setattr(builder,'load_decision_holds',lambda paths:({'c:decision:0000'},[]))
    monkeypatch.setattr(builder,'session_graphs',lambda row,t,**options:iter([{'id':row['trajectory']['id'],'split':options['split']}]))
    output=tmp_path/'graphs'
    argv=['builder','--export-receipt',str(receipt),'--base-tokenizer','fixture',
          '--protection','fixture','--output',str(output)]
    for hold in holds:argv+=['--decision-hold',str(hold)]
    monkeypatch.setattr(sys,'argv',argv);builder.main()
    with open_jsonl_text(output/'test/native-graphs.jsonl.zst') as stream:
        assert [json.loads(line)['id'] for line in stream]==['a','b']
    omissions=[json.loads(line) for line in (output/'omissions.jsonl').read_text().splitlines()]
    assert [(r['id'],r['reason']) for r in omissions]==[
        ('c','current-decision-hold'),('d','source-reported-non-positive')]
    rows[0]['trajectory']['split']='unknown'
    with write_jsonl_text(source) as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    data=json.loads(receipt.read_text());data['outputs'][source.name]=sha256_file_hex(source);receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='unknown source session split'):
        builder.main()
