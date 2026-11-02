import copy
import json
from natlang_neuralese.common.jsonio import write_jsonl_text
from scripts.neuralese_data.parallel_native_graph_index import parallel_rows
from scripts.neuralese_data.acquired_content_review import content_review
from test_static_native_context import row,objective,FP


def test_content_review_detects_exact_tokens_without_equating_new_id():
    graph=row();same=copy.deepcopy(graph);same['id']='different-source';other=copy.deepcopy(graph)
    other['nodes'][0]['token_ids'][10]+=1
    a=content_review(graph,objective(mass='token'));b=content_review(same,objective(mass='token'));c=content_review(other,objective(mass='token'))
    assert a['exact_token_weight_sha256']==b['exact_token_weight_sha256']
    assert a['exact_token_weight_sha256']!=c['exact_token_weight_sha256']
    assert a['role_structure_sha256']==b['role_structure_sha256']


def test_parallel_census_reuses_actual_parser_and_effective_weight_census(tmp_path):
    graph=row();path=tmp_path/'source.jsonl.zst'
    with write_jsonl_text(path) as stream:stream.write(json.dumps(graph)+'\n')
    result=list(parallel_rows(path,FP,[58,59,60,61],workers=1,max_pending=1,audit=objective(mass='token')))
    assert len(result)==1
    entry,slots,decisions,signal,review=result[0]
    assert entry['id']==signal['id']==review['id']==graph['id']
    assert slots>=entry['forward_tokens'] and decisions==0
    assert signal['eligible'] and signal['category_mass'].get('context',0)==0
    assert review['forward_tokens']==entry['forward_tokens']


def test_sample_contains_only_actual_effective_positive_tokens():
    import hashlib
    from natlang_neuralese.train.own_output_supervision import graph_weights
    graph=row()
    graph['id']=next(str(i) for i in range(100000)
        if int(hashlib.sha256(str(i).encode()).hexdigest()[:8],16)%512==0)
    obj=objective(mass='token')
    review=content_review(graph,obj)
    actual=[token for node in graph['nodes'] for token in node['token_ids']]
    weights,_=graph_weights(graph,obj)
    expected=[token for token,weight in zip(actual,weights,strict=True) if weight>0][:8192]
    assert [token for run in review['sample']['effective_positive_token_segments'] for token in run]==expected


def test_tool_markup_mass_counts_only_complete_assistant_reply_pairs(monkeypatch):
    from scripts.neuralese_data import audit_acquired_learning_signal as audit
    graph=row();graph['nodes'][0]['token_ids']=[125019,7,125020,125019,8,125020,125019,9]
    graph['nodes'][0]['target_weights']=[1]*8
    monkeypatch.setattr(audit,'graph_weights',lambda *_:([0,0,0,0.5,1,0.5,2,3],[0,0,0,1,1,1,1,1]))
    signal=audit.row_signal(graph,{},tool_call_markers=(125019,125020))
    measured=signal['tool_call_markup']
    assert measured['balanced_pairs']==1
    assert measured['effective_mass']==2
    assert measured['effective_positive_tokens']==3
    assert measured['unmatched_markers']==1
