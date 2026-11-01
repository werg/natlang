import copy
import pytest

FP='f'*64
def row():
    return {'schema':'natlang.neuralese-native-graph/1','id':'source:exact','split':'train',
      'source_groups':['world:shared'],'tokenizer_fingerprint':FP,
      'nodes':[{'name':'text','token_ids':list(range(40)),'target_weights':[0.]+[1.]*39}],
      'conversion':{'role_interpretation':'ordinary-text','source_row_id':'original'}}


def test_pending_partition_preserves_source_and_rejects_forgery(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    from scripts.neuralese_data import partition_acquired_context_pending as part
    import natlang_neuralese.rl.native_direct_sft_recovery as recovery
    def write(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
        return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha256_file_hex(path)}
    first=row();second=copy.deepcopy(first);second['id']='source:short';second['nodes'][0]['token_ids']=second['nodes'][0]['token_ids'][:10];second['nodes'][0]['target_weights']=second['nodes'][0]['target_weights'][:10]
    writer=GraphWriter(tmp_path/'input.zst');writer.write(first);writer.write(second);source=writer.finish()
    screen=write('screen.json',{'schema':'natlang.acquired-current-protection-screen/1','output':{**source,'tokenizer_fingerprint':FP,
        'graph_ids_sha256':canonical_json_sha256_hex(sorted([first['id'],second['id']]))},'accounting':{'eligible_train_graphs':2},
        'protected_worlds':{'identity_set_sha256':'d'*64,'matches_expected':True}})
    previous_source=write('source.json',{});family=write('map.json',{'source_sha256':previous_source['sha256'],'rows':[{'id':'old'}]})
    previous=write('previous.json',{'source':previous_source,'family_map':family})
    signals=[]
    for r in (first,second):
        signals.append({'id':r['id'],'graph_sha256':canonical_json_sha256_hex(r),'forward_tokens':len(r['nodes'][0]['token_ids']),
          'raw_positive_tokens':sum(w>0 for w in r['nodes'][0]['target_weights']),'effective_positive_tokens':1,
          'category_mass':{'reply':1.},'source_store':'fixture','eligible':True})
    signal_path=tmp_path/'signals.jsonl';signal_path.write_text(''.join(json.dumps(x)+'\n' for x in signals));sig=part.bind(signal_path)
    policy=write('policy.json',{});builder=write('audit.py',{})
    census=write('census.json',{'schema':'natlang.acquired-effective-learning-census/1','training_admission':False,'rows':2,
        'screen':screen,'row_signals':sig,'ce_policy':policy,'builder':builder,'objective_sha256':'a'*64})
    candidate=write('candidate.json',{'schema':'natlang.acquired-streaming-source-successor/1','training_admission':False,
        'new_rows':2,'predecessor_rows':1,'predecessor':previous,'source_screen':screen,'appended_graphs':source,
        'effective_loss_census':census,'row_signals':sig})
    checkpoint={'sha256':'e'*64};snap=write('snapshot.json',{'schema':'natlang.rolling-sft-parent-custody-snapshot/1',
        'checkpoint':checkpoint,'optimizer_present':True,'rng_present':True,'sft_frames':{'x':{}}})
    monkeypatch.setattr(recovery,'_authenticated_refresh_publication',lambda _: {'publication':{'checkpoint':checkpoint,'rolling_endpoint_custody_proof':{'successor':snap}}})
    monkeypatch.setattr(part.ProtectedWorlds,'load',lambda:SimpleNamespace(digest='d'*64,identities=set()))
    monkeypatch.setattr(part,'world_identities',lambda *a,**kw:set())
    monkeypatch.setattr(part.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    proof=part.partition(candidate,{'path':'publication','sha256':'f'*64},tmp_path/'out',limit=17)
    assert proof['active_graph']['rows']==proof['pending_graph']['rows']==1
    new_screen=json.loads((tmp_path/'out/screen.json').read_text());new_census=json.loads((tmp_path/'out/census.json').read_text())
    assert part.verify_partition(new_screen,previous['path'])[1]==1
    part.verify_partition_census(new_screen,new_census)
    target=tmp_path/'out/row-signals.jsonl';target.write_text('{}\n')
    with pytest.raises(ValueError,match='differs from its binding'):
        part.verify_partition_census(new_screen,new_census)


def test_successor_builder_calls_static_correction_verifier_before_admission(tmp_path,monkeypatch):
    import json
    from scripts.neuralese_data import build_acquired_streaming_successor as builder
    from scripts.neuralese_data import transform_acquired_ordinary_context as correction
    previous=tmp_path/'previous.json';previous.write_text('{}')
    screen=tmp_path/'screen.json';screen.write_text(json.dumps({'static_context_correction':{'path':'proof'}}))
    census=tmp_path/'census.json';census.write_text('{}')
    def reject(*args,**kwargs):raise ValueError('central static proof rejected')
    monkeypatch.setattr(correction,'verify_context_correction',reject)
    with pytest.raises(ValueError,match='central static proof rejected'):
        builder.prepare_source_stage(previous,screen,tmp_path/'prepare')
    with pytest.raises(ValueError,match='central static proof rejected'):
        builder.build(previous,screen,census,tmp_path/'build')


def test_pending_rebase_preserves_family_and_verifies_exact_correction(tmp_path,monkeypatch):
    import json,hashlib
    from scripts.neuralese_data import rebase_acquired_pending_partition as tool
    from scripts.neuralese_data.partition_acquired_context_pending import bind
    import natlang_neuralese.train.native_static_context_correction as static_correction
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return bind(p)
    oldmap=write('old-map.json',{'source_sha256':'old','rows':[{'id':'x','binding':'before','family':'general-agent'}]})
    newmap=write('new-map.json',{'source_sha256':'new','rows':[{'id':'x','binding':'after','family':'general-agent'}]})
    ce=write('ce.json',{'objective':None});new_ce=write('new-ce.json',{'objective':None})
    prior=write('prior.json',{'source':{'sha256':'old'},'family_map':oldmap,'ce_policy':ce,'current_protection_digest':'p'})
    ledger=tmp_path/'ledger.jsonl';ledger.write_text(json.dumps({'id':'x','old_binding':'before','new_binding':'after','disposition':'corrected'})+'\n')
    correction=write('correction.json',{'prior_source':{'sha256':'old'},'current_source':{'sha256':'new'},
       'static_native_context_correction':{'schema':'validated-by-source-correction'},'row_bindings_path':str(ledger),
       'row_bindings_sha256':hashlib.sha256(ledger.read_bytes()).hexdigest()})
    current=write('current.json',{'source':{'sha256':'new'},'family_map':newmap,'ce_policy':new_ce,'source_correction':correction,'current_protection_digest':'p'})
    partition=write('partition.json',{'predecessor':prior,'protected_worlds_sha256':'p'})
    screen=write('screen.json',{'pending_partition':partition,'output':{'rows':2}})
    census=write('census.json',{'screen':screen,'pending_partition':partition})
    monkeypatch.setattr(tool,'source_correction_dependencies',lambda ref:[])
    monkeypatch.setattr(static_correction,'verify_static_context_correction',lambda *a:({'x':{'old_binding':'before','new_binding':'after'}},[]))
    result=tool.rebase(screen['path'],census['path'],current['path'],tmp_path/'rebased')
    assert result['corrected_ids']==['x']
    assert json.loads((tmp_path/'rebased/census.json').read_text())['ce_policy']==new_ce
    updated=json.loads((tmp_path/'rebased/screen.json').read_text())
    assert tool.verify_rebase(updated,current['path'])[1]==prior
    (tmp_path/'new-map.json').write_text('{}')
    with pytest.raises(ValueError,match='differs from its binding'):tool.verify_rebase(updated,current['path'])
