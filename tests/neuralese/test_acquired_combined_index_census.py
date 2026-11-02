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
