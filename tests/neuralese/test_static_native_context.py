import copy
import pytest
import torch
from test_own_output_supervision import objective,turn,USER,ASSISTANT,FP
from test_shared_state import tiny
from natlang_neuralese.data.static_native_context import graphify,verify_graphification
from natlang_neuralese.data.native_graph import parse_graph,graph_loss
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.own_output_supervision import graph_weights,apply_to_example

def row():
    ids=turn(USER,[20]*25+[21]*25+[22]*25)+turn(ASSISTANT,[23,24,25])
    return {'schema':'natlang.neuralese-native-graph/1','id':'original','split':'train','source_groups':['world'],
      'tokenizer_fingerprint':FP,'nodes':[{'name':'text','token_ids':ids,'target_weights':[0.]+[.05]*(len(ids)-1)}],
      'conversion':{'role_interpretation':'whole-trajectory-chat-roles','runtime_opening_assistant_turns':0}}

def test_static_graph_preserves_tokens_objective_and_imports_causally():
    original=row();obj=objective(mass='token');child,proof=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2)
    assert proof['producer_count']>=2 and child['id']==original['id']
    assert verify_graphification(original,child,obj)
    assert graph_weights(original,obj)==graph_weights(child,obj)
    assert [t for n in child['nodes'] for t in n['token_ids']]==original['nodes'][0]['token_ids']
    ex=apply_to_example(parse_graph(child,tokenizer_fingerprint=FP),child,obj)
    model=SharedStateModel(tiny('lfm'),marker_ids=(58,59,60,61),rank=4)
    model.port.hf.requires_grad_(True)
    loss,_=graph_loss(model,ex,checkpoint_layers=False,chunk_tokens=32)
    loss.backward()
    grad=model.port.hf.get_input_embeddings().weight.grad
    assert grad is not None
    assert all(grad[token].abs().sum()>0 for token in (20,21,22))

@pytest.mark.parametrize('mutation',['edge','producer','weights','provenance'])
def test_static_graph_rejects_forgery(mutation):
    original=row();obj=objective(mass='token');child,_=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2)
    bad=copy.deepcopy(child)
    if mutation=='edge':bad['nodes'][-1]['imports'][0]['start']+=1
    elif mutation=='producer':bad['nodes'][0]['token_ids'][0]+=1
    elif mutation=='weights':bad['conversion']['static_native_context']['effective_weights'][-2]+=1
    else:bad['source_groups']=['different']
    with pytest.raises(ValueError):verify_graphification(original,bad,obj)

def test_static_source_correction_replays_parent_and_rejects_changed_proof(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    from scripts.neuralese_data import audit_acquired_learning_signal as audited_census
    from scripts.neuralese_data.audit_acquired_learning_signal import row_signal
    from scripts.neuralese_data import transform_acquired_ordinary_context as tool
    import natlang_neuralese.rl.native_direct_sft_recovery as recovery
    def write(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha256_file_hex(p)}
    original=row();original['conversion']['source_row_id']='original'
    writer=GraphWriter(tmp_path/'source.zst');writer.write(original);source=writer.finish()
    source_prev=write('previous-source.json',{});family=write('family.json',{'schema':'natlang.logical-family-map/1',
        'source_sha256':source_prev['sha256'],'rows':[{'id':'prior'}]});previous=write('previous.json',{'source':source_prev,'family_map':family})
    policy=write('policy.json',{'objective':objective(mass='token')})
    screen=write('screen.json',{'output':{**source,'tokenizer_fingerprint':FP,'graph_ids_sha256':canonical_json_sha256_hex([original['id']])}})
    signal=row_signal(original,objective(mass='token'));path=tmp_path/'signals.jsonl';path.write_text(json.dumps(signal)+'\n')
    signals={'path':str(path),'bytes':path.stat().st_size,'sha256':sha256_file_hex(path)}
    census=write('census.json',{'schema':'natlang.acquired-effective-learning-census/1','rows':1,'training_admission':False,
        'screen':screen,'row_signals':signals,'ce_policy':policy,'builder':{
            'path':audited_census.__file__,'sha256':sha256_file_hex(audited_census.__file__)},
        'objective_sha256':canonical_json_sha256_hex(objective(mass='token'))})
    candidate=write('candidate.json',{'schema':'natlang.acquired-streaming-source-successor/1','training_admission':False,
        'appended_graphs':source,'new_rows':1,'source_screen':screen,'predecessor':previous,'predecessor_rows':1,
        'ce_policy':policy,'effective_loss_census':census,'row_signals':signals})
    checkpoint={'sha256':'f'*64};snapshot=write('snapshot.json',{'schema':'natlang.rolling-sft-parent-custody-snapshot/1',
        'checkpoint':checkpoint,'optimizer_present':True,'rng_present':True,'sft_frames':{'x':{}}})
    pub={'checkpoint':checkpoint,'rolling_endpoint_custody_proof':{'successor':snapshot}}
    monkeypatch.setattr(recovery,'_authenticated_refresh_publication',lambda _:{'publication':pub})
    monkeypatch.setattr(tool.ProtectedWorlds,'load',lambda:SimpleNamespace(identities=set(),digest='d'*64))
    monkeypatch.setattr(tool,'world_identities',lambda *a,**kw:set())
    monkeypatch.setattr(tool.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    result=tool.transform(candidate,tmp_path/'output',parent_publication={'path':'pub','sha256':'f'*64},max_tokens=64,overlap=6)
    assert result['representation']=='static-native'
    corrected_screen=json.loads((tmp_path/'output/screen.json').read_text());corrected_census=json.loads((tmp_path/'output/census.json').read_text())
    assert len(tool.verify_context_correction(corrected_screen,previous['path'],corrected_census)[1])==1
    path=tmp_path/'output/row-signals.jsonl';path.write_text('{}\n')
    with pytest.raises(ValueError,match='differs from its binding'):tool.verify_context_correction(corrected_screen,previous['path'],corrected_census)

def test_earned_static_correction_authenticates_bindings_and_preserves_zero_exposure(tmp_path):
    import json,hashlib
    from test_native_source_correction import _receipt,_write_proof,_row
    from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
    from natlang_neuralese.data.native_sources import _row_bindings
    from natlang_neuralese.train.native_source_correction import apply_source_correction
    original=row();child,_=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    old=_row_bindings([indexed_graph_metadata(original,tokenizer_fingerprint=FP)])[original['id']]
    new=_row_bindings([indexed_graph_metadata(child,tokenizer_fingerprint=FP)])[child['id']]
    prior=_receipt('/old/source','old',[original['id']],['old-leaf'])
    current=_receipt('/new/source','new',[original['id']],['new-leaf'])
    frame={'source_receipts':[prior,{'path':'/held','sha256':'held'}],'source_row_bindings':{original['id']:old},
        'source_row_order':[original['id']],'exposure_counts':{original['id']:0},'required_visits':{original['id']:1},
        'options':{'stage_id':'native_full_sft','epochs':1,'_communication_ids':[],
                   'communication_quota_policy':{'stage_id':'native_full_sft','visits':7},
                   'supervision_objective':__import__('natlang_neuralese.train.own_output_supervision',fromlist=['identity']).identity(objective(mass='token'))}}
    frame['sft_progress']={'native_full_sft':copy.deepcopy(frame)}
    proof_path=_write_proof(tmp_path,prior,current,[_row(original['id'],old,new,'old-leaf','new-leaf','corrected',old_communication=False)])
    pairs=tmp_path/'pairs.jsonl';pairs.write_text(json.dumps({'parent':original,'child':child})+'\n')
    policy=tmp_path/'policy.json';policy.write_text(json.dumps({'objective':objective(mass='token')}))
    ref=lambda path:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    proof=json.loads(proof_path.read_text());proof['static_native_context_correction']={
        'schema':'natlang.static-native-context-source-correction/1','graph_pairs':ref(pairs),'ce_policy':ref(policy),'rows':1}
    proof_path.write_text(json.dumps(proof))
    args=dict(model_id='lfm2:inventory',stage_id='native_full_sft',current_source_receipt=current,
        current_bindings={original['id']:new},current_order=[original['id']],current_communication_ids=[original['id']])
    result,_=apply_source_correction(frame,proof_path,**args)
    assert result['exposure_counts'][original['id']]==0
    assert result['required_visits'][original['id']]==7
    assert result['options']['_objective_required_visits'][original['id']]==7
    carried=result['sft_progress']['native_full_sft']
    assert carried['required_visits'][original['id']]==7
    assert carried['options']['_objective_required_visits'][original['id']]==7
    assert result['options']['source_correction_history'][-1]['static_native_context_rows']==[original['id']]
    changed=copy.deepcopy(frame);changed['exposure_counts'][original['id']]=1
    with pytest.raises(ValueError,match='exact changed communication row'):apply_source_correction(changed,proof_path,**args)
    changed=copy.deepcopy(frame);changed['options']['supervision_objective']['sha256']='0'*64
    with pytest.raises(ValueError,match='earned supervision policy'):apply_source_correction(changed,proof_path,**args)
    pairs.write_text('{}\n')
    with pytest.raises(ValueError,match='artifact changed'):apply_source_correction(frame,proof_path,**args)

def test_rolling_endpoint_authenticates_static_correction_and_quota(tmp_path):
    import json,hashlib
    from test_native_source_correction import _receipt,_write_proof,_row
    from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
    from natlang_neuralese.data.native_sources import _row_bindings
    from natlang_neuralese.rl.native_direct_sft_recovery import _rolling_endpoint_snapshots
    original=row();child,_=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    old=_row_bindings([indexed_graph_metadata(original,tokenizer_fingerprint=FP)])[original['id']]
    new=_row_bindings([indexed_graph_metadata(child,tokenizer_fingerprint=FP)])[child['id']]
    prior=_receipt('/old/source','old',[original['id']],['old-leaf']);current=_receipt('/new/source','new',[original['id']],['new-leaf'])
    proof_path=_write_proof(tmp_path,prior,current,[_row(original['id'],old,new,'old-leaf','new-leaf','corrected',old_communication=False)])
    pairs=tmp_path/'pairs.jsonl';pairs.write_text(json.dumps({'parent':original,'child':child})+'\n')
    policy=tmp_path/'policy.json';policy.write_text(json.dumps({'objective':objective(mass='token')}))
    ref=lambda path:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    proof=json.loads(proof_path.read_text());proof['static_native_context_correction']={'schema':'natlang.static-native-context-source-correction/1',
        'graph_pairs':ref(pairs),'ce_policy':ref(policy),'rows':1};proof_path.write_text(json.dumps(proof));proof_ref=ref(proof_path)
    stage='native_full_sft';held={'path':'/held','sha256':'held'};base={'source_row_order':[original['id']],
        'source_row_bindings':{original['id']:old},'exposure_counts':{original['id']:0},'required_visits':{original['id']:1},
        'source_receipts':[prior,held],'held_row_bindings':{},'pending_source_rows':{}}
    after=copy.deepcopy(base);after.update({'source_row_bindings':{original['id']:new},'required_visits':{original['id']:7},
        'source_receipts':[current,held],'source_correction_binding':proof_ref,'source_correction_history':[{'proof_sha256':proof_ref['sha256'],
        'static_native_context_rows':[original['id']],'static_native_context_required_visits':{original['id']:7}}]})
    parent={'path':str(tmp_path/'parent.pt'),'sha256':'1'*64,'global_step':4};successor={'path':str(tmp_path/'child.pt'),'sha256':'2'*64,'global_step':5}
    def snapshot(name,checkpoint,frame):
        p=tmp_path/name;p.write_text(json.dumps({'schema':'natlang.rolling-sft-parent-custody-snapshot/1',
          'checkpoint':{**checkpoint,'bytes':1},'global_step':checkpoint['global_step'],'optimizer_present':True,'rng_present':True,
          'model_config_sha256':'a'*64,'sft_frames':{stage:frame}}));return ref(p)
    declaration={'schema':'natlang.rolling-sft-endpoint-snapshots/1','parent':snapshot('parent.json',parent,base),
        'successor':snapshot('child.json',successor,after),'source_correction':proof_ref}
    result=_rolling_endpoint_snapshots(declaration,parent,successor,verify_current_endpoints=False)
    assert result['corrected_rows']==[original['id']]
    bad=copy.deepcopy(base);bad['exposure_counts'][original['id']]=1
    declaration['parent']=snapshot('bad-parent.json',parent,bad)
    with pytest.raises(ValueError,match='zero-exposure'):_rolling_endpoint_snapshots(declaration,parent,successor,verify_current_endpoints=False)

def test_earned_graph_pair_preparation_checks_saved_binding_and_zero_exposure(tmp_path,monkeypatch):
    import json,hashlib
    from types import SimpleNamespace
    from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
    from natlang_neuralese.data.native_sources import _row_bindings
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    from scripts.neuralese_data import prepare_acquired_static_graph_pairs as prep
    import natlang_neuralese.rl.native_direct_sft_recovery as recovery
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    original=row();child,proof=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    other=copy.deepcopy(original);other['id']='ordinary-other';other['nodes'][0]['token_ids']=other['nodes'][0]['token_ids'][:12];other['nodes'][0]['target_weights']=other['nodes'][0]['target_weights'][:12]
    writer=GraphWriter(tmp_path/'source.zst');writer.write(other);writer.write(original);source=writer.finish()
    policy=write('policy.json',{'objective':objective(mass='token')});signals=write('signals.json',{})
    manifest=write('train-sources.json',{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**{k:source[k] for k in ('path','sha256','bytes')},'training_priority':1}]})
    geometry=write('geometry.json',{'source':{'sha256':manifest['sha256']},
        'identity':{'split':'train','roles':[58,59,60,61],'tokenizer':FP}})
    candidate=write('candidate.json',{'schema':'natlang.acquired-streaming-source-successor/1','appended_graphs':source,
        'source':manifest,'ce_policy':policy,'row_signals':signals,'geometry':geometry,'new_rows':2})
    audit=write('audit.json',{'schema':'natlang.acquired-static-native-context-candidate-audit/1','source':source,'ce_policy':policy,
        'row_signals':signals,'proofs':[proof],'max_tokens':32,'consumer_context':6,'state_tokens':2})
    old=_row_bindings([indexed_graph_metadata(original,tokenizer_fingerprint=FP)])[original['id']]
    other_binding=_row_bindings([indexed_graph_metadata(other,tokenizer_fingerprint=FP)])[other['id']]
    checkpoint={'sha256':'e'*64};snapshot=write('snapshot.json',{'schema':'natlang.rolling-sft-parent-custody-snapshot/1',
        'checkpoint':checkpoint,'optimizer_present':True,'rng_present':True,'sft_frames':{'stage':{'source_receipts':[manifest],
        'stage_identity':{'config':{'architecture':'model','base_inventory_sha256':'exact'}},
        'source_row_order':[other['id'],original['id']],'source_row_bindings':{other['id']:other_binding,original['id']:old},
        'exposure_counts':{original['id']:0},'required_visits':{original['id']:1}}}})
    monkeypatch.setattr(recovery,'_authenticated_refresh_publication',lambda _:{'publication':{'checkpoint':checkpoint,'rolling_endpoint_custody_proof':{'successor':snapshot}}})
    monkeypatch.setattr(prep.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    result=prep.prepare(candidate,audit,{'path':'publication','sha256':'f'*64},tmp_path/'pairs')
    assert result['rows'][0]['old_binding']==old and result['rows'][0]['prior_exposure']==0
    from natlang_neuralese.train.native_static_context_correction import verify_static_context_correction
    verified,_=verify_static_context_correction(tmp_path/'pairs/receipt.json',result['source_correction_declaration'])
    assert verified[original['id']]['old_binding']==old
    from scripts.neuralese_data import prepare_earned_static_context_source as source_prep
    monkeypatch.setattr(source_prep,'NativeGraphSource',lambda *a,**k:SimpleNamespace(entries=[{'id':other['id'],'binding':other_binding},
        {'id':original['id'],'binding':old}]))
    source_result=source_prep.prepare(prep.bind(tmp_path/'pairs/receipt.json'),tmp_path/'staged-source',model_id='model:exact')
    assert source_result['corrected_ids']==[original['id']]
    from natlang_neuralese.common.jsonio import open_jsonl_text
    with open_jsonl_text(source_result['unchanged_leaf']['path']) as stream:
        assert [json.loads(line) for line in stream]==[other]
    from natlang_neuralese.train.native_source_correction import source_correction_dependencies
    assert source_correction_dependencies(source_result['source_correction'])

def test_static_recipe_transition_requires_preserved_family_membership(tmp_path,monkeypatch):
    import json,hashlib
    from test_native_source_correction import _receipt,_write_proof,_row
    from natlang_neuralese.train.recipe import _source_correction_continuation
    import natlang_neuralese.train.native_source_correction as correction
    import natlang_neuralese.train.native_supervision as supervision
    import natlang_neuralese.train.native_logical_families as families
    import natlang_neuralese.train.native_static_context_correction as static_correction
    from natlang_neuralese.train.own_output_supervision import identity,policy_objective
    prior=_receipt('/old','old',["x"],['old-leaf']);current=_receipt('/new','new',["x"],['new-leaf'])
    proof_path=_write_proof(tmp_path,prior,current,[_row('x','old-binding','new-binding','old-leaf','new-leaf','corrected',old_communication=False)])
    proof=json.loads(proof_path.read_text());proof['static_native_context_correction']={'schema':'evidence-tested-separately'};proof_path.write_text(json.dumps(proof))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    row_path=tmp_path/'source-correction-rows.jsonl'
    monkeypatch.setattr(correction,'source_correction_dependencies',lambda ref:[{'path':str(proof_path)},{'path':str(row_path),'sha256':digest(row_path)}])
    monkeypatch.setattr(static_correction,'verify_static_context_correction',lambda *a:({'x':{'old_binding':'old-binding','new_binding':'new-binding'}},[]))
    old_map=tmp_path/'old-map.json';new_map=tmp_path/'new-map.json'
    old_map.write_text(json.dumps({'source_sha256':'old','rows':[{'id':'x','binding':'old-binding','family':'general-agent'}]}))
    new_map.write_text(json.dumps({'source_sha256':'new','rows':[{'id':'x','binding':'new-binding','family':'general-agent'}]}))
    monkeypatch.setattr(families,'checked',lambda ref: __import__('pathlib').Path(ref['path']))
    base=objective(mass='token');before=copy.deepcopy(base);after=copy.deepcopy(base)
    balance={'key':'logical-family','max_scale':8,'normalization':'declared/1','scales':{'general-agent':1.0}}
    before['family_balance']={**balance,'map':{'path':str(old_map),'sha256':'a'*64}}
    after['family_balance']={**balance,'map':{'path':str(new_map),'sha256':'b'*64}}
    prior_policy={'objective':before};next_policy={'objective':after,'inventory':[{'sha256':'new'}],
        'objective_transition':{'from':identity(policy_objective(prior_policy))['sha256']}}
    monkeypatch.setattr(supervision,'load_policy',lambda ref:prior_policy if ref['path']=='old-ce' else next_policy)
    old={'native_graphs':{'path':'/old','sha256':'old','binding':'native_graphs'},'ce_supervision_policy':{'path':'old-ce','sha256':'c','binding':'ce'}}
    new={'native_graphs':{'path':'/new','sha256':'new','binding':'native_graphs'},
       'continued_native_graphs':{'path':'/old','sha256':'old','binding':'continued_native_graphs'},
       'ce_supervision_policy':{'path':'new-ce','sha256':'d','binding':'ce'},
       'source_correction':{'path':str(proof_path),'sha256':digest(proof_path),'binding':'source_correction'}}
    assert _source_correction_continuation(old,new)
    new_map.write_text(json.dumps({'source_sha256':'new','rows':[{'id':'x','binding':'new-binding','family':'different'}]}))
    with pytest.raises(ValueError,match='family correction ledger differs from row inventory'):_source_correction_continuation(old,new)


def test_static_recipe_transition_admits_only_declared_family_reassignment(tmp_path,monkeypatch):
    import json,hashlib
    from test_native_source_correction import _receipt,_write_proof,_row,_write_ledger,_with_section
    from natlang_neuralese.train.recipe import _source_correction_continuation
    import natlang_neuralese.train.native_source_correction as correction
    import natlang_neuralese.train.native_supervision as supervision
    import natlang_neuralese.train.native_logical_families as families
    from natlang_neuralese.train.own_output_supervision import identity,policy_objective
    prior=_receipt('/old','old',['x','y'],['old-leaf','y-leaf']);current=_receipt('/new','new',['y'],['y-leaf'])
    proof_path=_write_proof(tmp_path,prior,current,[_row('x','x-binding',None,'old-leaf',None,'deferred','duplicate label')])
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    row_path=tmp_path/'source-correction-rows.jsonl'
    monkeypatch.setattr(correction,'source_correction_dependencies',lambda ref:[{'path':str(proof_path)},{'path':str(row_path),'sha256':digest(row_path)}])
    old_map=tmp_path/'old-map.json';new_map=tmp_path/'new-map.json'
    old_map.write_text(json.dumps({'source_sha256':'old','rows':[{'id':'x','binding':'x-binding','family':'general-agent'},
        {'id':'y','binding':'y-binding','family':'natlang'}]}))
    new_map.write_text(json.dumps({'source_sha256':'new','rows':[{'id':'y','binding':'y-binding','family':'port-record-wrapper'}]}))
    monkeypatch.setattr(families,'checked',lambda ref: __import__('pathlib').Path(ref['path']))
    base=objective(mass='token');before=copy.deepcopy(base);after=copy.deepcopy(base)
    balance={'key':'logical-family','max_scale':8,'normalization':'declared/1'}
    before['family_balance']={**balance,'scales':{'general-agent':1.0,'natlang':1.0},'map':{'path':str(old_map),'sha256':'a'*64}}
    after['family_balance']={**balance,'scales':{'port-record-wrapper':1.0},'map':{'path':str(new_map),'sha256':'b'*64}}
    prior_policy={'objective':before};next_policy={'objective':after,'inventory':[{'sha256':'new'}],
        'objective_transition':{'from':identity(policy_objective(prior_policy))['sha256']},
        'quota_aware_basis':{'schema':'natlang.quota-aware-ce-basis/1','candidate_source_sha256':'new'}}
    monkeypatch.setattr(supervision,'load_policy',lambda ref:prior_policy if ref['path']=='old-ce' else next_policy)
    old={'native_graphs':{'path':'/old','sha256':'old','binding':'native_graphs'},'ce_supervision_policy':{'path':'old-ce','sha256':'c','binding':'ce'}}
    new={'native_graphs':{'path':'/new','sha256':'new','binding':'native_graphs'},
       'continued_native_graphs':{'path':'/old','sha256':'old','binding':'continued_native_graphs'},
       'ce_supervision_policy':{'path':'new-ce','sha256':'d','binding':'ce'},
       'source_correction':{'path':str(proof_path),'sha256':digest(proof_path),'binding':'source_correction'}}
    def refresh():new['source_correction']['sha256']=digest(proof_path)
    with pytest.raises(ValueError,match='membership or row binding changed outside its correction'):_source_correction_continuation(old,new)
    schema=correction.FAMILY_REASSIGNMENT_SCHEMA
    wrong=_write_ledger(tmp_path,'wrong.jsonl',[{'id':'y','binding':'y-binding','old_family':'coding','new_family':'port-record-wrapper'}])
    _with_section(proof_path,'family_reassignment',schema,wrong);refresh()
    with pytest.raises(ValueError,match='declared family reassignment differs'):_source_correction_continuation(old,new)
    ledger=_write_ledger(tmp_path,'move.jsonl',[{'id':'y','binding':'y-binding','old_family':'natlang','new_family':'port-record-wrapper'}])
    _with_section(proof_path,'family_reassignment',schema,ledger);refresh()
    assert _source_correction_continuation(old,new)


def test_pending_static_graphs_are_staged_without_discharge(tmp_path,monkeypatch):
    import json,hashlib
    from types import SimpleNamespace
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    from scripts.neuralese_data import prepare_pending_static_graphs as tool
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    original=row();child,proof=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    writer=GraphWriter(tmp_path/'pending.zst');writer.write(original);source=writer.finish()
    policy=write('policy.json',{'objective':objective(mass='token')});signals=write('signals.json',{})
    candidate=write('candidate.json',{'ce_policy':policy,'row_signals':signals})
    partition=write('partition.json',{'schema':'natlang.acquired-position-pending-partition/1','input_graph':source,
        'pending_graph':source,'candidate':candidate,'pending_rows':[{'id':original['id'],'graph_sha256':canonical_json_sha256_hex(original)}]})
    audit=write('audit.json',{'schema':'natlang.acquired-static-native-context-candidate-audit/1','source':source,
        'ce_policy':policy,'row_signals':signals,'proofs':[proof],'max_tokens':32,'consumer_context':6,'state_tokens':2})
    monkeypatch.setattr(tool.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    result=tool.prepare(partition,audit,tmp_path/'derived')
    assert result['original_pending_graphs']==source and result['derived_graphs']['rows']==1
    assert 'obligations remain' in result['status']


class WordTokenizer:
    """Deterministic text/token map: every token id decodes to a word, and words encode into small in-vocab ids."""
    def decode(self,ids,skip_special_tokens=False):
        return ' '.join(f'w{i}' for i in ids if not (skip_special_tokens and i<20))+'.'
    def encode(self,text,add_special_tokens=False):
        import hashlib
        return [40+int(hashlib.md5(word.encode()).hexdigest(),16)%12 for word in text.split()]


def test_seeded_static_graph_exports_a_chunk_summary_with_provenance_under_bounded_producer_loss():
    original=row();obj=objective(mass='token')
    child,proof=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2,tokenizer=WordTokenizer(),seed_tokens=4)
    meta=child['conversion']['static_native_context']
    assert meta['schema']=='natlang.static-native-context-graph/2' and proof['schema']=='natlang.static-native-context-proof/2'
    assert verify_graphification(original,child,obj)
    producers=child['nodes'][:-1];spans=meta['producer_spans']
    assert len(producers)>=2 and all(len(n['token_ids'])<=32 for n in producers)
    for node,(a,b),seed in zip(producers,spans,meta['seeds']):
        n=b-a;k=seed['tokens']
        # Original chunk tokens come first and stay unsupervised; the summary is the exported, supervised span.
        assert node['token_ids'][:n]==original['nodes'][0]['token_ids'][a:b]
        assert node['target_weights']==[0.]*n+[1.]*k and node['producing']==[[n,n+k]]
        assert 1<=k<=4 and seed['provenance']=='mechanical:chunk-summary' and seed['parts']
    assert [t for n,(a,b) in zip(producers,spans) for t in n['token_ids'][:b-a]]+child['nodes'][-1]['token_ids']==original['nodes'][0]['token_ids']
    assert [(e['start'],e['stop']) for e in child['nodes'][-1]['imports']]==[tuple(n['producing'][0]) for n in producers]
    weights,categories=graph_weights(child,obj)
    ex=apply_to_example(parse_graph(child,tokenizer_fingerprint=FP),child,obj)
    assert len(weights)==len(ex.target_weights)==sum(len(n['token_ids']) for n in child['nodes'])
    total=sum(s['tokens'] for s in meta['seeds']);seed_mass=meta['seed_weight']*total
    parent_mass=sum(graph_weights(original,obj)[0])
    assert 0<seed_mass<=meta['seed_share']*parent_mass*(1+1e-9)
    assert proof['imported_state_tokens']==total and proof['seed_provenance']=={'mechanical:chunk-summary':len(producers)}
    model=SharedStateModel(tiny('lfm'),marker_ids=(58,59,60,61),rank=4)
    model.port.hf.requires_grad_(True)
    loss,_=graph_loss(model,ex,checkpoint_layers=False,chunk_tokens=32)
    loss.backward()
    grad=model.port.hf.get_input_embeddings().weight.grad
    assert grad is not None and all(grad[token].abs().sum()>0 for token in (20,21,22))


def test_seed_is_question_selected_and_names_exact_content():
    from natlang_neuralese.data.operator_seeds import chunk_summary_seed
    text='The vault opens at nine. Gears turn slowly. The pump P-17 failed on 2026-03-04. Nothing else happened.'
    focused=chunk_summary_seed(text,'when did pump P-17 fail')
    assert focused['parts'][0]=='question-overlap' and 'P-17 failed' in focused['text']
    assert focused['provenance']=='mechanical:chunk-summary'
    blind=chunk_summary_seed(text)
    assert 'question-overlap' not in blind['parts'] and 'P-17' in blind['text']
    assert chunk_summary_seed(text,'when did pump P-17 fail')==focused


@pytest.mark.parametrize('mutation',['seed-token','seed-weight','seed-provenance','import','producing','tail-fragment'])
def test_seeded_static_graph_rejects_forgery(mutation):
    original=row();obj=objective(mass='token')
    child,_=graphify(original,obj,max_tokens=32,consumer_context=6,state_tokens=2,tokenizer=WordTokenizer(),seed_tokens=4)
    bad=copy.deepcopy(child);meta=bad['conversion']['static_native_context']
    if mutation=='seed-token':bad['nodes'][0]['token_ids'][-1]+=1
    elif mutation=='seed-weight':bad['nodes'][0]['target_weights'][-1]=0.
    elif mutation=='seed-provenance':meta['seeds'][0]['provenance']=''
    elif mutation=='import':bad['nodes'][-1]['imports'][0]['stop']-=1
    elif mutation=='producing':bad['nodes'][0]['producing']=[[len(bad['nodes'][0]['token_ids'])-2,len(bad['nodes'][0]['token_ids'])]]
    else:bad['nodes'][0]['token_ids'][0]+=1
    with pytest.raises(ValueError):verify_graphification(original,bad,obj)


def test_seeded_geometry_falls_back_through_the_same_position_error():
    original=row()
    with pytest.raises(ValueError,match='exceed the position limit'):
        graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=8,tokenizer=WordTokenizer(),seed_tokens=8)
    with pytest.raises(ValueError,match='invalid static native context geometry'):
        graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2,tokenizer=WordTokenizer(),seed_tokens=20)


def test_tokenizerless_graphify_keeps_the_legacy_tail_export_for_replay():
    original=row();child,_=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    assert child['conversion']['static_native_context']['schema']=='natlang.static-native-context-graph/1'
    assert 'seeds' not in child['conversion']['static_native_context']
