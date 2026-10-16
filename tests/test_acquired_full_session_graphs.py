"""Actual pinned tokenizers retain full context, tool results and earlier reasoning."""
from pathlib import Path
import copy
import pytest
from transformers import AutoTokenizer
from scripts.neuralese_data.build_acquired_full_session_graphs import session_graphs, held_session_targets
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from natlang_neuralese.data.text_corpus import (native_gold_document, native_gold_rendered_text,
    tokenizer_fingerprint)
from natlang_neuralese.common.paths import resolve

BASES=[resolve('data_nvme','dgxops/qad-drift-probe/base'),
       resolve('models','mellum21-12b-a2.5b-thinking')]


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
    assert original==saved
    coverage=graphs[0]['conversion']['acquired_session']['reasoning_coverage']
    omitted=[int(ordinal) for ordinal, detail in coverage.items() if not detail['present_in_whole_view']]
    aux_ordinals=[graph['conversion']['acquired_session']['assistant_turn'] for graph in graphs[1:]]
    assert aux_ordinals==omitted
    texts=[tokenizer.decode(graph['nodes'][0]['token_ids'],skip_special_tokens=False) for graph in graphs]
    assert 'SOURCE_BEGIN ' in texts[0] and texts[0].count('useful context ')==10000
    assert 'EXACT_OBSERVATION' in texts[0] and 'COMPLETE_ANSWER' in texts[0]
    assert 'LATEST_REASONING_TARGET' in texts[0]
    for graph in graphs[1:]:
        ordinal=graph['conversion']['acquired_session']['assistant_turn']
        marker='EARLIER_REASONING_TARGET' if ordinal==0 else 'LATEST_REASONING_TARGET'
        assert marker in tokenizer.decode(graph['nodes'][0]['token_ids'],skip_special_tokens=False)
        assert graph['conversion']['acquired_session']['view']=='full-prefix-reasoning-target'
    assert graphs[0]['conversion']['acquired_session']['reasoning_coverage_probe'] == \
        'per-assistant-unique-sentinel/full-serving-render/1'
    for graph in graphs:
        assert graph['split']=='test' and graph['source_groups']==['source-task']
        node=graph['nodes'][0]
        assert len(node['token_ids'])==len(node['target_weights']) and node['target_weights'][0]==0
        assert max(node['target_weights'])>0
        assert max(node['target_weights'])>min(weight for weight in node['target_weights'] if weight>0)
        assert graph['conversion']['acquired_session']['oracle_verified'] is False
        assert graph['conversion']['acquired_session']['truncated'] is False


@pytest.mark.parametrize('base',BASES,ids=['lfm','mellum'])
def test_reasoning_coverage_uses_actual_full_render_for_multiturn_and_trailing_messages(base):
    if not (base/'tokenizer.json').is_file():
        pytest.skip('pinned production tokenizer unavailable on this machine')
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    source_fp=tokenizer_fingerprint(tokenizer);provision(tokenizer)
    row=value()
    row['trajectory']['messages']=[
        {'role':'system','content':'echo DUPLICATE_REASONING'},
        {'role':'user','content':'first user DUPLICATE_REASONING'},
        {'role':'assistant','reasoning':'DUPLICATE_REASONING','content':'first answer'},
        {'role':'tool','tool_call_id':'c1','content':'tool observation'},
        {'role':'user','content':'second user DUPLICATE_REASONING'},
        {'role':'assistant','reasoning':'DUPLICATE_REASONING','content':'second answer'},
        {'role':'tool','tool_call_id':'c2','content':'trailing tool'},
        {'role':'user','content':'trailing user DUPLICATE_REASONING'},
    ]
    original=copy.deepcopy(row)
    expected_text,_=native_gold_document(tokenizer,row['trajectory']['messages'][:-1],
        row['trajectory']['messages'][-1],row['trajectory'].get('tools'))
    assert native_gold_rendered_text(tokenizer,row['trajectory']['messages'],
        row['trajectory'].get('tools')) == expected_text
    graphs=list(session_graphs(row,tokenizer,source_fingerprint=source_fp,
        expanded_fingerprint=tokenizer_fingerprint(tokenizer),split='train'))
    assert row==original
    metadata=graphs[0]['conversion']['acquired_session']
    coverage=metadata['reasoning_coverage']
    assert set(coverage)=={'0','1'}
    assert all(item['sentinel_occurrences'] in {0,1} for item in coverage.values())
    aux_by_ordinal={graph['conversion']['acquired_session']['assistant_turn']:graph for graph in graphs[1:]}
    assert set(aux_by_ordinal)=={int(key) for key, item in coverage.items()
                                  if not item['present_in_whole_view']}
    whole_text=tokenizer.decode(graphs[0]['nodes'][0]['token_ids'],skip_special_tokens=False)
    assert 'trailing tool' in whole_text and 'trailing user DUPLICATE_REASONING' in whole_text
    for ordinal, graph in aux_by_ordinal.items():
        text=tokenizer.decode(graph['nodes'][0]['token_ids'],skip_special_tokens=False)
        assert text.count('DUPLICATE_REASONING') >= 1
        assert graph['nodes'][0]['target_weights'][-1] > 0


def test_failure_is_never_positive_sft():
    row=value();row['trajectory']['outcome']['label']='failed'
    with pytest.raises(ValueError,match='non-positive'):
        list(session_graphs(row,None,source_fingerprint='a'*64,expanded_fingerprint='b'*64,split='train'))


def test_original_target_identity_closes_decision_hold_descendants():
    row=value()
    assert held_session_targets(row,{'source-session:1'})==['source-session:1']
    assert held_session_targets(row,{'unrelated:1'})==[]


def test_runtime_resume_fingerprint_is_independent_of_checkout_path(tmp_path):
    from scripts.neuralese_data.build_acquired_full_session_graphs import runtime_bindings
    first=tmp_path/'checkout-a';second=tmp_path/'checkout-b'
    names=['training/neuralese/natlang_neuralese/data/text_corpus.py',
        'training/neuralese/natlang_neuralese/data/native_graph_conversion.py',
        'training/neuralese/natlang_neuralese/data/prompt_rendering.py',
        'training/neuralese/natlang_neuralese/serve/chat.py',
        'training/neuralese/natlang_neuralese/common/jsonio.py',
        'scripts/neuralese_data/build_native_graph_text_corpus.py',
        'scripts/neuralese_data/build_acquired_full_session_graphs.py',
        'scripts/neuralese_data/build_native_semantic_pairs_e2e.py',
        'scripts/neuralese_data/conversion_output_lease.py',
        'training/neuralese/natlang_neuralese/model/native_model.py']
    for root in (first,second):
        for name in names:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixed runtime bytes')
    a=runtime_bindings(first);b=runtime_bindings(second)
    assert a['fingerprint']==b['fingerprint']
    assert a['files']==b['files']
    assert a['observed_paths']!=b['observed_paths']


def test_output_lease_rejects_second_writer_before_shard_cleanup(tmp_path):
    from scripts.neuralese_data.build_acquired_full_session_graphs import acquire_output_lease
    output=tmp_path/'graphs';unfinished=output/'session-shards'/'shard-00000000'
    unfinished.mkdir(parents=True);(unfinished/'partial').write_text('still owned')
    lease=acquire_output_lease(output)
    try:
        with pytest.raises(RuntimeError,match='already owns'):
            acquire_output_lease(output)
        assert (unfinished/'partial').read_text()=='still owned'
    finally:
        lease.close()


def test_unreceipted_shard_cleanup_is_durably_listed_first(tmp_path):
    from scripts.neuralese_data.build_acquired_full_session_graphs import completed_shards
    import json
    output=tmp_path/'graphs';shards=output/'session-shards';unfinished=shards/'shard-00000000'
    unfinished.mkdir(parents=True);payload=unfinished/'partial.bin';payload.write_bytes(b'partial output')
    frontier,receipts=completed_shards(shards,{},0)
    assert (frontier,receipts)==(0,[])
    assert not unfinished.exists()
    line=(output/'cleanup-receipt.jsonl').read_text().splitlines()[0]
    receipt=json.loads(line)
    assert receipt['directory']=='session-shards/shard-00000000'
    assert receipt['files']==[{'path':'session-shards/shard-00000000/partial.bin','bytes':14,
        'sha256':__import__('hashlib').sha256(b'partial output').hexdigest()}]


def test_cli_routes_complete_groups_and_omits_held_targets(tmp_path,monkeypatch):
    import json,sys
    from scripts.neuralese_data import build_acquired_full_session_graphs as builder
    from natlang_neuralese.common.jsonio import write_jsonl_text,open_jsonl_text
    from natlang_neuralese.common.hashing import sha256_file_hex
    rows=[]
    for name,split,group in [('a','train','group-a'),('b','test','group-a'),
                             ('c','train','group-c'),('d','train','group-d'),
                             ('e','train','group-e')]:
        row=value();row['trajectory'].update(id=name,split=split,split_groups=[group]);rows.append(row)
    rows[3]['positive_candidate']=False
    source=tmp_path/'full-trajectories.jsonl.zst'
    with write_jsonl_text(source) as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    receipt=tmp_path/'export.json'
    receipt.write_text(json.dumps({'schema':'natlang.normalized-agent-source-file/1',
        'contract':{'source':rows[0]['source_file'],'revision':rows[0]['source_revision']},
        'outputs':{source.name:sha256_file_hex(source)},'complete_sessions':len(rows)}))
    from types import SimpleNamespace
    prepared=SimpleNamespace(tokenizer=object(),source_fingerprint='a'*64,expanded_fingerprint='a'*64,
        binding={'schema':'test-prepared-tokenizer'},validate=lambda base:None)
    monkeypatch.setattr(builder,'prepare_tokenizer',lambda base:prepared)
    leases=[];acquire=builder.acquire_output_lease
    def acquire_tracked(output):
        lease=acquire(output);leases.append(lease);return lease
    monkeypatch.setattr(builder,'acquire_output_lease',acquire_tracked)
    monkeypatch.setattr(builder.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    monkeypatch.setattr(builder,'tokenizer_fingerprint',lambda t:'a'*64)
    monkeypatch.setattr(builder,'provision',lambda t:None)
    monkeypatch.setattr(builder,'current_protection',lambda p:{'held':set(),'quality':set(),'pending':set(),'proof':{}})
    holds=sorted((Path(builder.__file__).resolve().parents[2]/'training/decision-holds').glob('*.jsonl'))
    monkeypatch.setattr(builder,'load_decision_holds',lambda paths:({'c:decision:0000'},[]))
    should_interrupt={'value':True}
    def session_graphs_stub(row,t,**options):
        if row['trajectory']['id']=='e' and should_interrupt['value']:
            should_interrupt['value']=False
            raise RuntimeError('simulated interruption after first committed shard')
        return iter([{'id':row['trajectory']['id'],'split':options['split']}])
    monkeypatch.setattr(builder,'session_graphs',session_graphs_stub)
    output=tmp_path/'graphs'
    argv=['builder','--export-receipt',str(receipt),'--base-tokenizer','fixture',
          '--protection','fixture','--output',str(output),'--chunk-sessions','2']
    for hold in holds:argv+=['--decision-hold',str(hold)]
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        builder.main()
    assert leases[-1].closed,'failed callable conversion must explicitly release its output lease'
    interrupted=json.loads((output/'resume-state.json').read_text())
    assert interrupted['completed_source_record_frontier']==4
    assert len(interrupted['shards'])==2
    cleanup=[json.loads(line) for line in (output/'cleanup-receipt.jsonl').read_text().splitlines()]
    assert [item['directory'] for item in cleanup]==['session-shards/shard-00000004']
    assert cleanup[0]['files']
    builder.main()
    assert leases[-1].closed,'successful callable conversion must explicitly release its output lease'
    with open_jsonl_text(output/'test/native-graphs.jsonl.zst') as stream:
        assert [json.loads(line)['id'] for line in stream]==['a','b']
    omissions=[json.loads(line) for line in (output/'omissions.jsonl').read_text().splitlines()]
    assert [(r['id'],r['reason']) for r in omissions]==[
        ('c','current-decision-hold'),('d','source-reported-non-positive')]
    completed=json.loads((output/'receipt.json').read_text())
    assert completed['source_record_count']==5 and len(completed['shards'])==3
    assert completed['counts']['positive_sessions']==3
    rows[0]['trajectory']['split']='unknown'
    with write_jsonl_text(source) as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    data=json.loads(receipt.read_text());data['outputs'][source.name]=sha256_file_hex(source);receipt.write_text(json.dumps(data))
    argv[argv.index(str(output))]=str(tmp_path/'invalid-graphs')
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='unknown source session split'):
        builder.main()
