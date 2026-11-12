import copy
import math
import pytest
from test_own_output_supervision import objective,turn,USER,ASSISTANT,FP
from natlang_neuralese.data.ordinary_context_windows import window_ordinary_row,verify_ordinary_windows
from natlang_neuralese.data.native_graph_schema import parse_graph
from natlang_neuralese.data.native_sources import _row_bindings
from natlang_neuralese.train.own_output_supervision import graph_weights,apply_to_example


def row():
    ids=turn(USER,[40]*2)+turn(ASSISTANT,[41]*18,reasoning=[42]*12)+turn(USER,[43])+turn(ASSISTANT,[44]*12)
    raw=[.05]*len(ids);raw[0]=0.;raw[12]=0.;raw[29]=0.
    return {'schema':'natlang.neuralese-native-graph/1','id':'source:exact','split':'train',
        'source_groups':['world:shared'],'tokenizer_fingerprint':FP,
        'nodes':[{'name':'text','token_ids':ids,'target_weights':raw}],
        'conversion':{'kind':'static-transformation','role_interpretation':'whole-trajectory-chat-roles',
                      'source_row_id':'original','runtime_opening_assistant_turns':0,
                      'supervision_policy':{'name':'original-source-mask'}}}


def test_windows_preserve_one_graph_exact_coverage_masks_categories_and_mass():
    original=row(); obj=objective(mass='token');child,proof=window_ordinary_row(original,obj,max_tokens=17,overlap=3)
    assert child['id']==original['id'] and child['split']==original['split'] and child['source_groups']==original['source_groups']
    assert len(child['nodes'])>1 and max(len(node['token_ids']) for node in child['nodes'])<=17
    assert all(not node.get('imports') and not node.get('producing') for node in child['nodes'])
    assert verify_ordinary_windows(original,child,obj)
    before_weights,before_categories=graph_weights(original,obj);after_weights,after_categories=graph_weights(child,obj)
    assert math.fsum(before_weights)==pytest.approx(math.fsum(after_weights))
    assert sum(v>0 for v in before_weights)==sum(v>0 for v in after_weights)
    assert math.fsum(original['nodes'][0]['target_weights'])==pytest.approx(sum(sum(n['target_weights']) for n in child['nodes']))
    assert proof['effective_positive_targets']==sum(v>0 for v in before_weights)
    assert proof['raw_positive_targets']==sum(v>0 for v in original['nodes'][0]['target_weights'])
    parsed=parse_graph(child,tokenizer_fingerprint=FP)
    applied=apply_to_example(parsed,child,obj)
    assert list(applied.target_weights)==after_weights and list(applied.objective_categories)==after_categories
    assert max(parsed.plan.own_positions,default=0)<17
    assert max(parsed.plan.slot_positions,default=0)<17
    original_binding=_row_bindings([parsed])[child['id']]
    forged=copy.deepcopy(child);forged['conversion']['ordinary_context_windows']['windows'][0]['categories'][0]=3
    assert _row_bindings([parse_graph(forged,tokenizer_fingerprint=FP)])[child['id']]==original_binding
    with pytest.raises(ValueError,match='span or node binding'):graph_weights(forged,obj)


@pytest.mark.parametrize('tamper',['token','raw','overlap','weight','category','provenance','objective'])
def test_window_audit_rejects_changed_source_or_supervision(tamper):
    original=row();obj=objective(mass='token');child,_=window_ordinary_row(original,obj,max_tokens=17,overlap=3)
    if tamper=='token':child['nodes'][0]['token_ids'][2]+=1
    if tamper=='raw':child['nodes'][0]['target_weights'][3]=9
    if tamper=='overlap':child['nodes'][1]['token_ids'][0]+=1
    if tamper=='weight':child['conversion']['ordinary_context_windows']['windows'][0]['effective_weights'][3]=999
    if tamper=='category':child['conversion']['ordinary_context_windows']['windows'][0]['categories'][3]=3
    if tamper=='provenance':child['conversion']['source_row_id']='forged'
    if tamper=='objective':obj=objective(.2,mass='token')
    with pytest.raises(ValueError):verify_ordinary_windows(original,child,obj)


def test_rejects_native_communication_and_empty_learning_windows():
    original=row();obj=objective(mass='token')
    for key,value in [('imports',[{'at':1,'source':'other','start':0,'stop':1}]),('producing',[[2,4]])]:
        altered=copy.deepcopy(original);altered['nodes'][0][key]=value
        with pytest.raises(ValueError):window_ordinary_row(altered,obj,max_tokens=17,overlap=3)
    altered=copy.deepcopy(original);altered['nodes'].append(copy.deepcopy(altered['nodes'][0]));altered['nodes'][1]['name']='child'
    with pytest.raises(ValueError,match='one ordinary node'):window_ordinary_row(altered,obj,max_tokens=17,overlap=3)
    altered=copy.deepcopy(original);altered['nodes'][0]['target_weights']=[0.]*len(altered['nodes'][0]['token_ids'])
    with pytest.raises(ValueError,match='targetless'):window_ordinary_row(altered,obj,max_tokens=17,overlap=3)


def test_small_rows_and_invalid_limits_are_explicit():
    original=row();unchanged,proof=window_ordinary_row(original,objective(),max_tokens=100,overlap=2)
    assert unchanged==original and proof is None
    for limit,overlap in [(262145,2),(8,0),(8,8),(True,1)]:
        with pytest.raises(ValueError):window_ordinary_row(original,objective(),max_tokens=limit,overlap=overlap)
