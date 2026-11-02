import copy
import json
from types import SimpleNamespace
import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from scripts.neuralese_data import prepare_acquired_manifest_source_stage as stage
from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as protection_module
from scripts.neuralese_data import decision_holds


def write(path, value):
    path.write_text(json.dumps(value))
    return stage.bind(path)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, 'require_output_headroom', lambda _: None)
    monkeypatch.setattr(stage, 'ProtectedWorlds', SimpleNamespace(load=lambda:SimpleNamespace(digest='worlds')))
    quality={'path':str(tmp_path/'quality.json'),'sha256':'quality'}
    monkeypatch.setattr(protection_module,'current_protection',lambda _: {'proof':quality})
    monkeypatch.setattr(decision_holds,'load_decision_holds',lambda _: (set(),[]))
    src=write(tmp_path/'old-source.json',{'schema':'natlang.native-graph-sources/1','sources':[]})
    old={'id':'old','binding':'old-binding','forward_tokens':4}
    new={'id':'new','binding':'new-binding','forward_tokens':5}
    fmap=write(tmp_path/'map.json',{'schema':'natlang.logical-family-map/1','source_sha256':src['sha256'],'rows':[old]})
    geo=write(tmp_path/'old-geometry.json',{'identity':{'tokenizer':'tok','roles':[1,2], 'parser_policy_sha256':'parser'}})
    ce=write(tmp_path/'ce.json',{})
    pred=write(tmp_path/'predecessor.json',{'source':src,'family_map':fmap,'geometry':geo,'ce_policy':ce,
        'rows':1,'current_protection_digest':'worlds'})
    leaf=write(tmp_path/'leaf.bin',{'payload':'graph'})
    leaf_with_rows={**leaf,'rows':1}
    manifest=write(tmp_path/'combined-source.json',{'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs',**leaf,'training_priority':1}]})
    export=write(tmp_path/'export.json',{})
    converter=write(tmp_path/'converter.json',{'outputs':[leaf_with_rows],'training_admission':False,
        'export_receipt':export,'explicit_history_reasoning_policy':'keep'})
    scr=write(tmp_path/'screen.json',{'schema':'natlang.new-acquired-source-screen/1','training_admission':False,
        'prepared_source_sha256':src['sha256'],'protected_worlds_sha256':'worlds','quality_protection':quality,
        'decision_holds':[],'active':{'rows':1},'pending':{'rows':0},'counts':{'active':1,'raw':1},
        'output_inventory':[leaf_with_rows]})
    overlap=write(tmp_path/'overlap.json',{'schema':'natlang.acquired-source-occurrence-overlap/1',
        'training_admission':False,'screen':scr,'predecessor':pred,'active_rows':1,
        'source_occurrence_suffix_missing':0,'duplicate_source_occurrences_within_active':0})
    combined=write(tmp_path/'combined.json',{'schema':'natlang.combined-acquired-screened-delta/1',
        'training_admission':False,'predecessor':pred,'source':manifest,'rows':1,'source_leaves':1,
        'tokenizer_fingerprint':'tok','protected_worlds_sha256':'worlds','quality_protection':quality,
        'decision_holds':[],'screens':[scr],'occurrence_proofs':[overlap],
        'pending_render_obligations':{'rows':0,'evidence':[]},'carried_over_context_obligations':None})
    proof=write(tmp_path/'proof.json',{'schema':'natlang.parallel-native-source-index/1','source':leaf,
        'decision_readouts':0,'parser_policy_sha256':'parser','rows':1,
        'ordered_metadata_sha256':canonical_json_sha256_hex([new]),'ordered_geometry_sha256':canonical_json_sha256_hex([7])})
    signals=write(tmp_path/'signals.json',[]);review=write(tmp_path/'review.json',{});pending=write(tmp_path/'pending.json',{})
    names=('prepare_acquired_manifest_index_census.py','parallel_native_graph_index.py','acquired_content_review.py',
        'audit_acquired_learning_signal.py','build_acquired_streaming_successor.py')
    code=[stage.bind(stage.Path(stage.__file__).with_name(name)) for name in names]
    census_value={'schema':'natlang.acquired-effective-learning-census/1','training_admission':False,'screen':combined,
        'rows':1,'counts':{'zero_effective_rows':0},'ce_policy':ce,'row_signals':signals,'content_review':review,
        'pending_census':pending,'code_bindings':code,'leaf_index_proofs':[proof],
        'source_format_history':[{'source':leaf,'converter':converter,'export':export,'history_reasoning_policy':'keep'}],
        'ordered_kv_slots':[7],'ordered_graph_ids_sha256':canonical_json_sha256_hex(['new'])}
    census=write(tmp_path/'census.json',census_value)
    monkeypatch.setattr(stage.capacity,'index_source',lambda *args:(SimpleNamespace(entries=[old,new]),{}))
    def geometry(_source,out,*,prior,allow_build,indexed_geometry):
        assert indexed_geometry=={('new','new-binding'):7}
        return [4,7],write(out/'geometry.json',{'slots':[4,7]}),{'reused':True}
    monkeypatch.setattr(stage.capacity,'geometry_for_source',geometry)
    return pred,combined,census,census_value


def test_manifest_stage_preserves_prefix_and_uses_production_leaf_geometry(tmp_path,monkeypatch):
    pred,combined,census,_=fixture(tmp_path,monkeypatch)
    result=stage.prepare(pred['path'],combined['path'],census['path'],tmp_path/'out')
    assert result['rows']==2 and result['new_rows']==1 and result['training_admission'] is False
    assert result['combined_leaf_index_proofs'] and result['geometry_scan']=={'reused':True}
    manifest=json.loads((tmp_path/'out'/'train-sources.json').read_text())
    assert manifest['sources'][-1]['sha256']==json.loads(open(combined['path']).read())['source']['sha256']


def test_manifest_stage_rejects_forged_geometry_even_if_census_is_self_consistent(tmp_path,monkeypatch):
    pred,combined,census,value=fixture(tmp_path,monkeypatch)
    bad=copy.deepcopy(value);bad['ordered_kv_slots']=[8]
    altered=write(tmp_path/'bad-census.json',bad)
    with pytest.raises(ValueError,match='bindings or geometry'):
        stage.prepare(pred['path'],combined['path'],altered['path'],tmp_path/'out')


def test_chained_finalization_binds_review_and_uses_fresh_canonical_builder(tmp_path):
    semantic=tmp_path/'semantic.json';semantic.write_text('{}');runtime=tmp_path/'runtime';runtime.mkdir()
    argv=stage.finalization_argv('pred','combined','census','prepared',tmp_path/'new',semantic,[runtime])
    assert argv[1].endswith('/build_acquired_streaming_successor.py')
    assert argv[argv.index('--prepared-source-receipt')+1]=='prepared'
    assert argv[argv.index('--new-family-share')+1]=='0.07'
    assert argv[argv.index('--reviewed-converter-runtime')+1]==str(runtime)
    with pytest.raises(ValueError,match='requires an output'):
        stage.finalization_argv('p','c','x','s',tmp_path/'new',None,[runtime])
