import copy
import pytest
from scripts.neuralese_data.finalize_earned_static_context_source import reconcile_geometry
from natlang_neuralese.train.native_packed_capacity import _rows_digest


def test_geometry_reuses_only_exact_unchanged_rows_and_new_native_binding():
    old=[{'id':'a','binding':'old-a','family':'general-agent'}, {'id':'b','binding':'old-b','family':'general-agent'}]
    original=[{'id':'a','binding':'old-a','forward_tokens':100}, {'id':'b','binding':'old-b','forward_tokens':20}]
    geometry={'rows_sha256':_rows_digest(original),'kv_slots':[128,32]}
    current=[{'id':'b','binding':'old-b','forward_tokens':20}, {'id':'a','binding':'new-a','forward_tokens':100,
        'producer_consumer':True,'external_state_consumption':False}]
    ledger={'a':{'old_binding':'old-a','new_binding':'new-a'}}
    _,values=reconcile_geometry(current,old,geometry,ledger,['a'],{'a':160})
    assert values==[32,160]
    changed=copy.deepcopy(current);changed[0]['binding']='forged'
    with pytest.raises(ValueError,match='undeclared'):reconcile_geometry(changed,old,geometry,ledger,['a'],{'a':160})
    changed=copy.deepcopy(current);changed[1]['forward_tokens']=99
    with pytest.raises(ValueError,match='geometry rows'):reconcile_geometry(changed,old,geometry,ledger,['a'],{'a':160})
    with pytest.raises(ValueError,match='order'):reconcile_geometry(list(reversed(current)),old,geometry,ledger,['a'],{'a':160})


def test_finalizer_builds_bound_source_geometry_policy_and_receipt(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from natlang_neuralese.common.hashing import sha256_file_hex,canonical_json_sha256_hex,sha256_file_hex_cached
    from scripts.neuralese_data import finalize_earned_static_context_source as tool
    def write(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    old_source=write('old-source.json',{'sources':[]});new_source=write('new-source.json',{'sources':[]})
    old_map=write('old-map.json',{'schema':'natlang.logical-family-map/1','source_sha256':old_source['sha256'],
        'rows':[{'id':'a','binding':'old-a','family':'general-agent'}]})
    original=[{'id':'a','binding':'old-a','forward_tokens':3}]
    package=Path(tool.capacity.__file__).resolve().parents[1]
    names=('data/native_sources.py','data/native_graph_schema.py','data/decision_graph.py','model/native_graph_plan.py',
        'data/native_pair_schema.py','common/row_index.py','common/jsonio.py','common/hashing.py')
    parser=canonical_json_sha256_hex({name:sha256_file_hex_cached(package/name) for name in names})
    geometry=write('old-geometry.json',{'schema':'natlang.native-packed-geometry/1','source':old_source,
       'identity':{'tokenizer':'f'*64,'roles':[58,59,60,61],'split':'train','parser_policy_sha256':parser},
       'rows_sha256':_rows_digest(original),'kv_slots':[128]})
    old_leaf=write('old-leaf.json',{});unchanged=write('unchanged.json',{});derived=write('derived.json',{})
    old_policy=write('old-policy.json',{'sources':[{'sha256':old_leaf['sha256'],'context_weight':0.05,'encoding':'native-row-objectives-v1'}],
      'source_contract':{'active_sources':[old_leaf],'active_source_count':1,'inventory':[old_source]},
      'objective':{'family_balance':{'map':old_map,'key':'logical-family','scales':{'general-agent':1.}}}})
    graph={'schema':'natlang.neuralese-native-graph/1','id':'a','split':'train','source_groups':['g'],'tokenizer_fingerprint':'f'*64,
       'nodes':[{'name':'p','token_ids':[1,2,3],'target_weights':[0,0,0],'producing':[[2,3]]},
       {'name':'c','token_ids':[4,5,6],'target_weights':[0,0,1],'imports':[{'at':1,'source':'p','start':2,'stop':3}]}]}
    pairs_file=tmp_path/'pairs.jsonl';pairs_file.write_text(json.dumps({'child':graph})+'\n')
    pairs=write('pairs-receipt.json',{'source_leaf':old_leaf,'graph_pairs':{'path':str(pairs_file),'sha256':sha256_file_hex(pairs_file)}})
    ledger_file=tmp_path/'ledger.jsonl';ledger_file.write_text(json.dumps({'id':'a','old_binding':'old-a','new_binding':'new-a'})+'\n')
    correction=write('correction.json',{'stage_id':'stage','prior_source':old_source,'current_source':new_source,
       'row_bindings_path':str(ledger_file),'row_bindings_sha256':sha256_file_hex(ledger_file),
       'static_native_context_correction':{'ce_policy':old_policy}})
    prep=write('preparation.json',{'schema':'natlang.earned-static-context-source-preparation/1','previous_source':old_source,
        'train_source':new_source,'source_correction':correction,'corrected_ids':['a'],'graph_pairs':pairs,
        'unchanged_leaf':unchanged,'derived_leaf':derived})
    recipe=write('recipe.json',{'stages':[{'id':'stage','inputs':{},'parameters':{}}],'input_bindings':{}})
    prior=write('prior.json',{'schema':'natlang.acquired-streaming-source-successor/1','source':old_source,'rows':1,
        'family_map':old_map,'geometry':geometry,'ce_policy':old_policy,'recipe':recipe,'family_balance':{},'current_protection_digest':'p'})
    entry={'id':'a','binding':'new-a','forward_tokens':3,'producer_consumer':True,'external_state_consumption':False}
    monkeypatch.setattr(tool,'NativeGraphSource',lambda *a,**kw:SimpleNamespace(entries=[entry]))
    monkeypatch.setattr(tool,'require_output_headroom',lambda _:None)
    monkeypatch.setattr(tool.ProtectedWorlds,'load',lambda:SimpleNamespace(digest='p'))
    monkeypatch.setattr(tool,'source_correction_dependencies',lambda _:[])
    monkeypatch.setattr(tool,'seed_unchanged_index',lambda *a:{'rows':0,'scope':'fixture'})
    monkeypatch.setattr(tool,'policy_objective',lambda policy:policy['objective'])
    monkeypatch.setattr(tool,'identity',lambda obj:{'name':'test','sha256':'a'*64})
    monkeypatch.setattr(tool,'load_policy',lambda ref:json.loads(Path(ref['path']).read_text()))
    monkeypatch.setattr(tool,'prepare_policy',lambda policy,source:{**policy,'_family_census':{'sha256':'b'*64}})
    monkeypatch.setattr(tool,'continuation_identity',lambda old,new:({}, {'to':{'sha256':'c'*64}}))
    monkeypatch.setattr(tool,'predecessor_mass_census',lambda *a:({'schema':'natlang.own-output-mass-census/1',
       'objective_without_family_scales':True,'families':{'general-agent':{'rows':1,'unscaled_mass':2.}},'unscaled_mass_total':2.},{}))
    monkeypatch.setattr(tool,'load_recipe',lambda path:json.loads(Path(path).read_text()))
    monkeypatch.setattr(tool,'resolve_stage_inputs',lambda *a:{})
    monkeypatch.setattr(tool,'_source_correction_continuation',lambda *a:True)
    result=tool.finalize(prep['path'],prior['path'],tmp_path/'final')
    assert result['rows']==1 and result['training_admission'] is False
    new_map=json.loads(Path(result['family_map']['path']).read_text())
    assert new_map['rows']==[{'id':'a','binding':'new-a','family':'general-agent'}]
    new_geo=json.loads(Path(result['geometry']['path']).read_text())
    assert new_geo['kv_slots'][0]>0 and new_geo['source']['sha256']==new_source['sha256']


def test_seed_index_reuses_only_byte_identical_authenticated_rows(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from natlang_neuralese.common.hashing import sha256_file_hex_cached,canonical_json_sha256_hex
    from natlang_neuralese.common import row_index
    from scripts.neuralese_data import finalize_earned_static_context_source as tool
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    def write_graphs(name,rows):
        w=GraphWriter(tmp_path/name)
        for row in rows:w.write(row)
        return w.finish()
    old_rows=[{'id':'keep','value':[1,2]},{'id':'replace','value':[3,4]}]
    original=write_graphs('original.zst',old_rows);remaining=write_graphs('remaining.zst',[old_rows[0]])
    def write(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}
    pairs=write('pairs.json',{'source_leaf':original,'rows':[{'id':'replace','parent_row_sha256':canonical_json_sha256_hex(old_rows[1])}]})
    source=write('source.json',{});entries=[{'id':'keep','binding':'b1','groups':['g'],'forward_tokens':2},
                                            {'id':'replace','binding':'b2','groups':['g'],'forward_tokens':2}]
    fake=SimpleNamespace(entries=entries,children=[SimpleNamespace(file_sha256=original['sha256'],entries=entries)])
    monkeypatch.setattr(tool,'NativeGraphSource',lambda *a,**k:fake)
    monkeypatch.setattr(row_index,'cache_dir',lambda:tmp_path/'cache')
    package=Path(tool.capacity.__file__).resolve().parents[1]
    names=('data/native_sources.py','data/native_graph_schema.py','data/decision_graph.py','model/native_graph_plan.py',
        'data/native_pair_schema.py','common/row_index.py','common/jsonio.py','common/hashing.py')
    parser=canonical_json_sha256_hex({name:sha256_file_hex_cached(package/name) for name in names})
    geometry={'identity':{'tokenizer':'f'*64,'roles':[1,2,3,4],'parser_policy_sha256':parser}}
    prep={'graph_pairs':pairs,'unchanged_leaf':remaining,'corrected_ids':['replace']}
    proof=tool.seed_unchanged_index(prep,{'source':source},geometry,entries)
    assert proof['rows']==1 and row_index.load(Path(proof['index']['path']))['entries']==[entries[0]]
    Path(remaining['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='differs from its binding'):tool.seed_unchanged_index(prep,{'source':source},geometry,entries)


def test_replaced_recipe_bindings_are_removed_only_when_unreferenced():
    from scripts.neuralese_data.finalize_earned_static_context_source import prune_replaced_recipe_bindings
    recipe={'input_bindings':{'old':{},'shared':{},'new':{}},'stages':[{'inputs':{'native_graphs':'new'}},{'inputs':{'held_native_graphs':'shared'}}]}
    prune_replaced_recipe_bindings(recipe,{'old','shared'})
    assert set(recipe['input_bindings'])=={'shared','new'}
