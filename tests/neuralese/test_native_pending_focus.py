import hashlib,json
import pytest
from natlang_neuralese.train.native_pending_focus import dependencies,resolve,promote
from natlang_neuralese.train.native_source_streams import stream_pending_batches


def test_focus_preserves_packed_credits_and_monotone_streams(tmp_path):
    entries=[{'id':str(i),'communication':False,'training_priority':0} for i in range(8)]
    batches=[(0,[0,1],False),(0,[2,3],False),(0,[4,5],False),(0,[6,7],True)]
    result=promote(batches,entries,{'4'})
    assert [b[1] for b in result]==[[4,5],[0,1],[2,3],[6,7]]
    assert sorted((v,tuple(ix)) for v,ix,_ in result)==sorted((v,tuple(ix)) for v,ix,_ in batches)
    class Source:
        def __init__(self):self.entries=entries
        def iter_indices(self,indices):
            for i in indices:yield type('Row',(),{'id':str(i)})()
    assert [[x.id for x in batch] for _,_,_,batch in stream_pending_batches(Source(),result,focus_ids={'4'})]==[['4','5'],['0','1'],['2','3'],['6','7']]


def test_focus_requires_source_binding_and_explicit_withdrawal(tmp_path):
    proof=tmp_path/'proof';proof.write_text('authenticated')
    evidence={'path':str(proof),'sha256':hashlib.sha256(proof.read_bytes()).hexdigest()}
    doc={'schema':'natlang.native-pending-focus/1','rows':[{'id':'x','binding':'a'*64,'source_sha256':'b'*64,'split':'train','evidence':[evidence]}]}
    request=tmp_path/'request.json';request.write_text(json.dumps(doc));ref={'path':str(request),'sha256':hashlib.sha256(request.read_bytes()).hexdigest()}
    assert dependencies(ref)[0]==doc
    assert resolve(doc,[{'id':'x'}],{'x':'a'*64},['b'*64])==({'x'},[])
    with pytest.raises(ValueError):resolve(doc,[{'id':'x'}],{'x':'c'*64},['b'*64])
    with pytest.raises(ValueError):resolve(doc,[],{},[])
    assert resolve(doc,[],{},[],[{'rows':[{'id':'x','binding':'a'*64}]}])==(set(),['x'])
    proof.write_text('tampered')
    with pytest.raises(ValueError):dependencies(ref)


def test_actual_tiny_sgd_focus_matches_manual_order_and_resume(tmp_path,monkeypatch):
    import torch
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.native_cohort_schedule import source_cohort_ids
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from test_native_source_fetch_recovery import source,equal
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';paths=source(base,'lfm')
    row=json.loads(paths['train'].read_text());rows=[]
    for name in ('a','b','c'):
        item=json.loads(json.dumps(row));item['id']=name;item['source_groups']=[name];rows.append(item)
    paths['train'].write_text(''.join(json.dumps(x)+'\n' for x in rows))
    _,_,_,cfg=trainer.load_native_model(base,dtype=torch.bfloat16,rank=4)
    indexed=NativeGraphSource(paths['train'],cfg['tokenizer_fingerprint'],required_split='train',role_ids=tuple(cfg['marker_ids']))
    cohort=source_cohort_ids(indexed);assert len(set(cohort))==1
    proof=tmp_path/'proof';proof.write_text('oracle accepted')
    doc={'schema':'natlang.native-pending-focus/1','rows':[{'id':'c','binding':indexed.bindings['c'],'source_sha256':cohort[2],'split':'train','evidence':[{'path':str(proof),'sha256':hashlib.sha256(proof.read_bytes()).hexdigest()}]}]}
    request=tmp_path/'focus.json';request.write_text(json.dumps(doc))
    def run(out,focus=False,resume=False):
        args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu','--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),'--epochs','1','--batch-graphs','1','--eval-every','1','--checkpoint-every','1','--checkpoint-seconds','0','--seed','23','--refresh-after-updates','1','--out',str(out)]
        if focus:args+=['--pending-focus',str(request)]
        if resume:args+=['--continue-from',str(out/'checkpoint.pt')]
        assert trainer.main(args)==0
    focused=tmp_path/'focused';run(focused,focus=True)
    actual=torch.load(focused/'checkpoint.pt',weights_only=False,map_location='cpu')
    assert actual['training']['exposure_counts']['c']==1
    assert actual['training']['options']['pending_focus_history'][-1]['state']=='complete'
    original=trainer.pending_exposure_batches;stream=trainer.stream_pending_batches
    def manual(*args,**kwargs):
        required,batches=original(*args,**kwargs)
        match=next(i for i,b in enumerate(batches) if 2 in b[1]);chosen=batches.pop(match);batches.insert(0,chosen)
        last={v:i for i,(v,_,_) in enumerate(batches)}
        return required,[(v,indices,last[v]==i) for i,(v,indices,_) in enumerate(batches)]
    def manual_stream(*args,**kwargs):
        kwargs['focus_ids']={'c'};return stream(*args,**kwargs)
    reference=tmp_path/'reference'
    with monkeypatch.context() as patch:
        patch.setattr(trainer,'pending_exposure_batches',manual);patch.setattr(trainer,'stream_pending_batches',manual_stream)
        run(reference)
    expected=torch.load(reference/'checkpoint.pt',weights_only=False,map_location='cpu')
    for key in actual.keys()-{'training'}:equal(actual[key],expected[key])
    for key in ('optimizer','rng','global_step','exposure_counts'):equal(actual['training'][key],expected['training'][key])
    run(focused,focus=True,resume=True)
    resumed=torch.load(focused/'checkpoint.pt',weights_only=False,map_location='cpu')
    assert resumed['training']['exposure_counts']['c']==1
    assert sum(resumed['training']['exposure_counts'].values())==2
    assert len(resumed['training']['options']['pending_focus_history'])==1


def test_recipe_and_provider_preserve_focus_evidence(tmp_path):
    from natlang_neuralese.train.recipe import resolve_stage_inputs,stage_input_args,_pending_focus_continuation
    from test_coreweave_provider_dependencies import deps,write
    evidence=write(tmp_path/'evidence','teacher/oracle')
    focus=write(tmp_path/'focus.json',{'schema':'natlang.native-pending-focus/1','rows':[{'id':'x','binding':'a'*64,'source_sha256':'b'*64,'split':'train','evidence':[{'path':evidence['path'],'sha256':evidence['sha256']}]}]})
    recipe={'input_bindings':{'focus':focus}};stage={'kind':'native_state_sft','inputs':{'pending_focus':'focus'}}
    resolved=resolve_stage_inputs(recipe,stage,{})
    assert resolved['pending_focus']['dependencies']==[{'path':evidence['path'],'sha256':evidence['sha256']}]
    assert stage_input_args(resolved,'native_state_sft')==['--pending-focus',focus['path']]
    assert _pending_focus_continuation({'native_graphs':{'sha256':'old'}},{'native_graphs':{'sha256':'old'},**resolved})
    assert not _pending_focus_continuation({'native_graphs':{'sha256':'old'}},{'native_graphs':{'sha256':'changed'},**resolved})
    config=write(tmp_path/'config.json',{'pending_focus':focus})
    closure=deps.dependency_closure({'config':config},namespace=str(tmp_path))
    assert {v['path'] for v in closure['files']}=={focus['path'],evidence['path'],config['path']}
    checkpoint=deps.dependency_closure({},namespace=str(tmp_path),_checkpoint_payload={'training':{'options':{'pending_focus_history':[{'request':{'path':focus['path'],'sha256':focus['sha256']},'state':'complete'}]}}})
    assert {v['path'] for v in checkpoint['files']}=={focus['path'],evidence['path']}


def test_resident_same_source_focus_transition_is_authenticated(tmp_path):
    import copy
    from test_native_resident_ce import fixture
    from natlang_neuralese.train import native_resident as resident
    paths,old,_=fixture(tmp_path);new=copy.deepcopy(old)
    focus=tmp_path/'focus.json';focus.write_text('{}')
    new['recipe']['input_bindings']['focus']=resident.binding(focus)
    new['recipe']['stages'][0]['inputs']['pending_focus']='focus'
    new['stage_inputs']['sft']['pending_focus']={**resident.binding(focus),'binding':'focus'}
    resident._contract(old,new,'sft')
    broken=copy.deepcopy(new);broken['stage_inputs']['sft']['pending_focus']=None
    with pytest.raises(ValueError):resident._contract(old,broken,'sft')


def test_actual_focus_resume_with_quality_withdrawal_and_append(tmp_path,monkeypatch):
    import torch
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.native_cohort_schedule import source_cohort_ids
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from test_native_source_fetch_recovery import source
    from test_native_withdrawal import proof
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';paths=source(base,'lfm');template=json.loads(paths['train'].read_text())
    rows=[]
    for ident in ('a','b','c'):
        row=json.loads(json.dumps(template));row['id']=ident;row['source_groups']=[ident];rows.append(row)
    paths['train'].write_text(''.join(json.dumps(row)+'\n' for row in rows))
    _,_,_,cfg=trainer.load_native_model(base,dtype=torch.bfloat16,rank=4)
    native=NativeGraphSource(paths['train'],cfg['tokenizer_fingerprint'],required_split='train',role_ids=tuple(cfg['marker_ids']))
    cohort=source_cohort_ids(native);evidence=tmp_path/'evidence';evidence.write_text('accepted')
    rowrefs=[{'id':name,'binding':native.bindings[name],'source_sha256':cohort[index],'split':'train','evidence':[{'path':str(evidence),'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}]} for index,name in ((1,'b'),(2,'c'))]
    focus=tmp_path/'focus.json';focus.write_text(json.dumps({'schema':'natlang.native-pending-focus/1','rows':rowrefs}))
    out=tmp_path/'out';args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu','--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),'--epochs','1','--eval-every','1','--checkpoint-every','1','--checkpoint-seconds','0','--seed','23','--refresh-after-updates','1','--out',str(out),'--pending-focus',str(focus)]
    assert trainer.main(args)==0
    before=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu')['training']
    assert before['exposure_counts']['b']==1 and before['exposure_counts']['c']==0
    extra=json.loads(json.dumps(template));extra['id']='extra';extra['source_groups']=['extra']
    newer=tmp_path/'new.jsonl';newer.write_text(''.join(json.dumps(row)+'\n' for row in (rows[0],rows[1],extra)))
    withdrawal=proof(tmp_path,[{'id':'c','binding':native.bindings['c'],'source_sha256':cohort[2],'reason':'quality'}])
    doc=json.loads(withdrawal.read_text());doc['model_id']=cfg['architecture']+':'+cfg['base_inventory_sha256'];withdrawal.write_text(json.dumps(doc))
    args[args.index('--native-graphs')+1]=str(newer)
    args+=['--continue-from',str(out/'checkpoint.pt'),'--continued-native-graphs',str(paths['train']),'--withdrawn-native-graphs',str(withdrawal)]
    assert trainer.main(args)==0
    after=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu')['training']
    record=after['options']['pending_focus_history'][-1]
    assert record['state']=='complete' and record['withdrawn']==['c'] and record['consumed']==['b']
    assert 'c' not in after['exposure_counts'] and after['exposure_counts']['b']==1
    assert after['global_step']==before['global_step']+1


def test_real_resident_adopts_focus_and_saves_history(tmp_path,monkeypatch):
    import copy
    import torch
    from natlang_neuralese.train import native_state_sft as trainer,native_resident as resident
    from natlang_neuralese.train.recipe import native_recipe_binding
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.common.jsonio import write_json_durable as write_json
    from test_native_source_fetch_recovery import source
    from test_native_resident_ce import fixture
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';source(base,'lfm')
    _,_,_,cfg=trainer.load_native_model(base,dtype=torch.bfloat16,rank=4)
    paths,old,_=fixture(tmp_path,cfg['tokenizer_fingerprint'])
    old_source=NativeGraphSource(paths['old_train'],cfg['tokenizer_fingerprint'],required_split='train',role_ids=tuple(cfg['marker_ids']))
    from natlang_neuralese.train.native_cohort_schedule import source_cohort_ids
    evidence=tmp_path/'evidence';evidence.write_text('accepted')
    focus=tmp_path/'focus.json';write_json(focus,{'schema':'natlang.native-pending-focus/1','rows':[{'id':'old','binding':old_source.bindings['old'],'source_sha256':source_cohort_ids(old_source)[0],'split':'train','evidence':[{'path':str(evidence),'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}]}]})
    recipe=copy.deepcopy(old['recipe']);recipe['input_bindings']['focus']=resident.binding(focus);recipe['stages'][0]['inputs']['pending_focus']='focus'
    new_recipe=tmp_path/'focus-recipe.json';write_json(new_recipe,recipe)
    directory=tmp_path/'resident'/'resident-data';directory.parent.mkdir();write_json(directory.parent/'recipe-plan.json',old)
    init=resident.ResidentAdoption.__init__;save=trainer._save;published=[]
    def initialize(self,*args,**kwargs):
        evidence=tmp_path/'capacity-proof.json';write_json(evidence,{'passed':True})
        capacity=tmp_path/'capacity.json';write_json(capacity,{'schema':'natlang.native-resident-geometry-qualification/1','passed':True,'identity':kwargs['geometry_identity'],'runtime_code_sha256':resident.digest(old['code']),'limits':{'forward_tokens':1000000,'kv_slots':1000000},'evidence':resident.binding(evidence)})
        kwargs['geometry_qualification']=capacity;init(self,*args,**kwargs)
    def capture(policy,*args,**kwargs):
        result=save(policy,*args,**kwargs)
        if kwargs['step']==1 and not published:
            published.append(resident.prepare(directory,new_recipe,'sft',tokenizer_fingerprint=cfg['tokenizer_fingerprint'],role_ids=tuple(cfg['marker_ids'])))
        return result
    args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu','--native-graphs',str(paths['old_train']),'--held-native-graphs',str(paths['held']),'--ce-supervision-policy',str(paths['old_ce']),'--epochs','2','--eval-every','1','--checkpoint-every','1','--checkpoint-seconds','0','--seed','23','--out',str(tmp_path/'out'),'--stage-id','sft','--resident-data-directory',str(directory),'--recipe-binding',json.dumps(native_recipe_binding(paths['old_recipe'],old['recipe_sha256']))]
    with monkeypatch.context() as patch:
        patch.setattr(resident.ResidentAdoption,'__init__',initialize);patch.setattr(trainer,'_save',capture)
        assert trainer.main(args)==0
    saved=torch.load(tmp_path/'out/checkpoint.pt',weights_only=False,map_location='cpu')['training']
    assert published and saved['exposure_counts']['old']==2
    record=saved['options']['pending_focus_history'][-1]
    assert record['at_global_step']==1 and record['state']=='complete' and record['consumed']==['old']
    with monkeypatch.context() as patch:
        patch.setattr(resident.ResidentAdoption,'__init__',initialize)
        assert trainer.main(args+['--continue-from',str(tmp_path/'out/checkpoint.pt')])==0
    restarted=torch.load(tmp_path/'out/checkpoint.pt',weights_only=False,map_location='cpu')['training']
    assert restarted['exposure_counts']==saved['exposure_counts']
    assert restarted['options']['pending_focus_history']==saved['options']['pending_focus_history']


def test_started_recipe_handoff_records_focus_without_changing_source(tmp_path):
    import copy
    from test_native_resident_ce import fixture
    from natlang_neuralese.train.recipe import unstarted_stage_input_changes,_without_unstarted_bindings
    from natlang_neuralese.train import native_resident as resident
    paths,old,_=fixture(tmp_path);new=copy.deepcopy(old)
    evidence=tmp_path/'evidence';evidence.write_text('accepted')
    request=tmp_path/'focus.json';request.write_text(json.dumps({'schema':'natlang.native-pending-focus/1','rows':[{'id':'old','binding':'a'*64,'source_sha256':'b'*64,'split':'train','evidence':[{'path':str(evidence),'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}]}]}))
    binding=resident.binding(request);new['recipe']['input_bindings']['focus']=binding
    new['recipe']['stages'][0]['inputs']['pending_focus']='focus'
    new['stage_inputs']['sft']['pending_focus']={**binding,'binding':'focus'}
    changes=unstarted_stage_input_changes(old,new,tmp_path,started_stage_ids=('sft',))
    assert changes['sft']['kind']=='pending_focus_continuation'
    before,after=_without_unstarted_bindings(old['recipe'],new['recipe'],changes)
    assert before==after


def test_focus_composes_with_authenticated_source_and_ce_append(tmp_path):
    import copy
    import pytest
    from test_native_resident_ce import fixture
    from natlang_neuralese.train.recipe import unstarted_stage_input_changes
    from natlang_neuralese.train import native_resident as resident
    paths, old, new = fixture(tmp_path)
    evidence = tmp_path/'focus-evidence'; evidence.write_text('verified')
    request = tmp_path/'focus-combined.json'
    request.write_text(json.dumps({'schema':'natlang.native-pending-focus/1','rows':[{
        'id':'extra','binding':'a'*64,'source_sha256':'b'*64,'split':'train',
        'evidence':[resident.binding(evidence)]}]}))
    binding = resident.binding(request)
    new['recipe']['input_bindings']['focus'] = binding
    new['recipe']['stages'][0]['inputs']['pending_focus'] = 'focus'
    new['stage_inputs']['sft']['pending_focus'] = {**binding, 'binding':'focus'}
    changes = unstarted_stage_input_changes(old,new,tmp_path,started_stage_ids=('sft',))
    assert changes['sft']['kind'] == 'pending_focus_continuation'
    bad = copy.deepcopy(new)
    bad['stage_inputs']['sft']['continued_native_graphs']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='handoff refused'):
        unstarted_stage_input_changes(old,bad,tmp_path,started_stage_ids=('sft',))
