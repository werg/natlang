import copy
import math
from pathlib import Path
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


def test_static_candidate_correction_preserves_row_and_rechecks_current_protection(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
    from natlang_neuralese.common.jsonio import open_jsonl_text
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    from scripts.neuralese_data import transform_acquired_ordinary_context as tool
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    original=row();candidate_path=tmp_path/'candidate/native-graphs.jsonl.zst'
    writer=GraphWriter(candidate_path);writer.write(original);source=writer.finish()
    previous_source=write('previous-source.json',{'source':'actual'})
    family=write('families.json',{'schema':'natlang.logical-family-map/1','source_sha256':previous_source['sha256'],
        'rows':[{'id':'prior:distinct','binding':'a'*64,'family':'general-agent'}]})
    previous=write('previous.json',{'source':previous_source,'family_map':family})
    policy=write('policy.json',{'objective':objective(mass='token')})
    screen=write('screen.json',{'output':{**source,'tokenizer_fingerprint':FP,
        'graph_ids_sha256':canonical_json_sha256_hex([original['id']])}})
    from natlang_neuralese.train.own_output_supervision import CATEGORIES
    weights,categories=graph_weights(original,objective(mass='token'))
    signal={'id':original['id'],'graph_sha256':canonical_json_sha256_hex(original),
        'source_store':'fixture','eligible':True,
        'forward_tokens':len(weights),'raw_positive_tokens':sum(w>0 for w in original['nodes'][0]['target_weights']),
        'effective_positive_tokens':sum(w>0 for w in weights),
        'category_mass':{name:sum(w for w,c in zip(weights,categories) if c==i) for i,name in enumerate(CATEGORIES)}}
    signals_path=tmp_path/'signals.jsonl';signals_path.write_text(json.dumps(signal)+'\n')
    signals={'path':str(signals_path),'sha256':sha256_file_hex(signals_path),'bytes':signals_path.stat().st_size}
    from scripts.neuralese_data import audit_acquired_learning_signal as audited_census
    audit_path=Path(audited_census.__file__).resolve()
    audit_ref={'path':str(audit_path),'sha256':sha256_file_hex(audit_path),'bytes':audit_path.stat().st_size}
    census=write('census.json',{'schema':'natlang.acquired-effective-learning-census/1','rows':1,
        'training_admission':False,'screen':screen,'row_signals':signals,'ce_policy':policy,
        'builder':audit_ref,
        'objective_sha256':canonical_json_sha256_hex(objective(mass='token'))})
    candidate=write('candidate.json',{'schema':'natlang.acquired-streaming-source-successor/1',
        'training_admission':False,'appended_graphs':source,'new_rows':1,'source_screen':screen,
        'predecessor':previous,'predecessor_rows':1,'ce_policy':policy,'effective_loss_census':census,'row_signals':signals})
    checkpoint={'path':str(tmp_path/'checkpoint.pt'),'sha256':'f'*64}
    snapshot=write('snapshot.json',{'schema':'natlang.rolling-sft-parent-custody-snapshot/1',
        'checkpoint':checkpoint,'optimizer_present':True,'rng_present':True,
        'sft_frames':{'full':{'source_row_bindings':{},'pending_source_rows':{},'exposure_counts':{}}}})
    publication={'checkpoint':checkpoint,'rolling_endpoint_custody_proof':{'successor':snapshot}}
    import natlang_neuralese.rl.native_direct_sft_recovery as recovery
    monkeypatch.setattr(recovery,'_authenticated_refresh_publication',lambda ref:{'publication':publication})
    publication_ref={'path':str(tmp_path/'publication.json'),'sha256':'e'*64}
    worlds=SimpleNamespace(identities=frozenset(),digest='d'*64)
    monkeypatch.setattr(tool.ProtectedWorlds,'load',lambda:worlds)
    monkeypatch.setattr(tool,'world_identities',lambda *a,**k:set())
    monkeypatch.setattr(tool.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    receipt=tool.transform(candidate,tmp_path/'out',parent_publication=publication_ref,max_tokens=17,overlap=3,representation='ordinary-windows')
    assert receipt['corrected_rows']==1 and receipt['input_rows']==receipt['output']['rows']==1
    assert receipt['protected_worlds_digest']==worlds.digest and receipt['row_ids_preserved'] is True
    with open_jsonl_text(receipt['output']['path']) as stream: bounded=json.loads(next(stream))
    assert verify_ordinary_windows(original,bounded,objective(mass='token'))
    monkeypatch.setattr(tool,'world_identities',lambda *a,**k:{('id','protected')})
    worlds.identities=frozenset({('id','protected')})
    with pytest.raises(ValueError,match='protected-world'):
        tool.transform(candidate,tmp_path/'blocked',parent_publication=publication_ref,max_tokens=17,overlap=3,representation='ordinary-windows')
    assert not (tmp_path/'blocked').exists()

