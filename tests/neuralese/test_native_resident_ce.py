"""CE coverage follows a real resident append, never a prior-objective rewrite."""
import copy
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train import native_state_sft as trainer
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs, native_recipe_binding
from natlang_neuralese.common.jsonio import write_json_durable as write_json
from scripts.neuralese_data.build_native_ce_supervision_policy import complete_source_contract
from test_native_supervision_mass import ordinary, declared
from test_native_source_fetch_recovery import source, equal


def fixture(tmp_path, fingerprint='tokenizer-sha'):
    paths={}
    for name in ('old','extra','held'):
        row=ordinary();row.update(id=name,tokenizer_fingerprint=fingerprint,
            split='test' if name=='held' else 'train',source_groups=[name])
        if name=='extra':row['nodes'][0]['token_ids'][-1]=5
        paths[name]=tmp_path/(name+'.jsonl');paths[name].write_text(json.dumps(row)+'\n')
    paths['old_train']=tmp_path/'old-train.json';write_json(paths['old_train'],{
        'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**resident.binding(paths['old'])}]})
    paths['train']=tmp_path/'train.json';write_json(paths['train'],{
        'schema':'natlang.native-graph-sources/1','sources':[
            {'kind':'native-graphs',**resident.binding(paths[name])}for name in ('old_train','extra')]})
    old=complete_source_contract(declared(digest=resident.sha(paths['old'])),
        [resident.binding(paths['old'])],context_weight=.05,inventory=[resident.binding(paths['old_train'])])
    old['held_sources']=[dict(sha256=resident.sha(paths['held']),encoding='native-row-objectives-v1',context_weight=.05)]
    new=complete_source_contract(old,[resident.binding(paths[name])for name in ('old','extra')],
        context_weight=.05,inventory=[resident.binding(paths['train'])])
    paths['old_ce']=tmp_path/'old-ce.json';paths['new_ce']=tmp_path/'new-ce.json'
    write_json(paths['old_ce'],old);write_json(paths['new_ce'],new)
    recipe={'schema':'natlang.neuralese-training-recipe/1','id':'resident-ce',
        'identity_qualification':'native_state_runtime_qualified','input_bindings':{
            'train':resident.binding(paths['old_train']),'held':resident.binding(paths['held']),
            'ce':resident.binding(paths['old_ce'])},'stages':[{'id':'sft','kind':'native_state_sft',
            'requires':[],'parameters':{'epochs':2.},'inputs':{'native_graphs':'train',
                'held_native_graphs':'held','ce_supervision_policy':'ce'}}]}
    paths['old_recipe']=tmp_path/'old-recipe.json';write_json(paths['old_recipe'],recipe)
    new_recipe=copy.deepcopy(recipe);new_recipe['input_bindings'].update(
        train=resident.binding(paths['train']),prior=resident.binding(paths['old_train']),ce=resident.binding(paths['new_ce']))
    new_recipe['stages'][0]['inputs']['continued_native_graphs']='prior'
    paths['new_recipe']=tmp_path/'new-recipe.json';write_json(paths['new_recipe'],new_recipe)
    def plan(path):
        value=load_recipe(path)
        return dict(schema='natlang.neuralese-recipe-plan/1',recipe=value,
            recipe_sha256=resident.sha(path),device='cpu',code=resident.code_inventory(Path(resident.__file__).parents[1]),
            inputs={},stage_inputs={'sft':resolve_stage_inputs(value,value['stages'][0],{})})
    return paths,plan(paths['old_recipe']),plan(paths['new_recipe'])


def test_real_recipe_contract_accepts_complete_ce_addition(tmp_path):
    paths,old,new=fixture(tmp_path)
    resident._contract(old,new,'sft')
    loaded=extension_policy(old['stage_inputs']['sft'],new['stage_inputs']['sft'])
    assert {s['sha256']for s in loaded['sources']}=={resident.sha(paths['old']),resident.sha(paths['extra'])}


@pytest.mark.parametrize('change',['coefficient','encoding','delete','held','mask','coverage','inventory','default','parameters','otherstage','same_source','drop_policy'])
def test_objective_or_custody_change_rejected(tmp_path,change):
    paths,old,new=fixture(tmp_path);doc=json.loads(paths['new_ce'].read_text())
    if change=='coefficient':doc['sources'][0]['context_weight']=.2
    elif change=='encoding':doc['sources'][0]['encoding']='ordinary-suffix-raw-v1'
    elif change=='delete':doc['sources']=[x for x in doc['sources']if x['sha256']!=resident.sha(paths['old'])]
    elif change=='held':doc['held_sources']=[]
    elif change=='mask':doc['failed_history_masks']=[]
    elif change=='coverage':doc['source_contract']['active_sources'].pop()
    elif change=='inventory':doc['source_contract']['inventory']=[resident.binding(paths['old'])]
    elif change=='default':doc['source_contract']['default_raw_context_weight']=.2
    elif change=='parameters':new['recipe']['stages'][0]['parameters']['epochs']=3.
    elif change=='otherstage':old['stage_inputs']['another']={'native_graphs':{}};new['stage_inputs']['another']={}
    elif change=='same_source':new['stage_inputs']['sft']['native_graphs']=old['stage_inputs']['sft']['native_graphs']
    elif change=='drop_policy':new['stage_inputs']['sft'].pop('ce_supervision_policy')
    if change not in ('parameters','otherstage','same_source','drop_policy'):
        # Change the exact immutable reference too: exercise semantic guards,
        # rather than merely triggering a stale file digest.
        write_json(paths['new_ce'],doc);ref=resident.binding(paths['new_ce'])
        new['recipe']['input_bindings']['ce']=ref
        new['stage_inputs']['sft']['ce_supervision_policy']={**ref,'binding':'ce'}
    with pytest.raises(ValueError):resident._contract(old,new,'sft')


def cpu_backend(monkeypatch):
    from natlang_neuralese.train import fp8_backend
    class CPUBackend:
        def reset_row_counts(self):pass
        def autocast(self):return contextlib.nullcontext()
        def native_modules(self):return contextlib.nullcontext()
        def checkpoint(self,function,*args,**kwargs):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(function,*args,**kwargs)
        def extra_state(self):return {'cpu_test_backend':True}
        def restore_extra_state(self,value):assert value==self.extra_state()
        def inventory(self):return {'cpu_test_backend':True}
    monkeypatch.setattr(fp8_backend,'make_backend',lambda *a,**k:CPUBackend())


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_trainer_live_append_matches_explicit_complete_source_updates(tmp_path,monkeypatch,family):
    if family=='mellum':cpu_backend(monkeypatch)
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';source(base,family)
    _,_,_,config=trainer.load_native_model(base,dtype=torch.bfloat16,rank=4)
    paths,old,new=fixture(tmp_path,config['tokenizer_fingerprint'])
    directory=tmp_path/'resident'/'resident-data';directory.parent.mkdir();write_json(directory.parent/'recipe-plan.json',old)
    original_init=resident.ResidentAdoption.__init__;original_save=trainer._save
    original_update=trainer.native_training_update;published=[];orders={'live':[],'reference':[]};mode='live'
    def initialize(self,*args,**kwargs):
        proof=tmp_path/'capacity-evidence.json';write_json(proof,{'passed':True,'scope':'tiny CPU transaction fixture'})
        capacity=tmp_path/'capacity.json';write_json(capacity,{
            'schema':'natlang.native-resident-geometry-qualification/1','passed':True,
            'identity':kwargs['geometry_identity'],'runtime_code_sha256':resident.digest(old['code']),
            'limits':{'forward_tokens':1000000,'kv_slots':1000000},'evidence':resident.binding(proof)})
        kwargs['geometry_qualification']=capacity;original_init(self,*args,**kwargs)
    def save(policy,*args,**kwargs):
        result=original_save(policy,*args,**kwargs)
        if mode=='live'and kwargs['step']==1 and not published:
            before=trainer.capture_training_rng_state('cpu')
            published.append(resident.prepare(directory,paths['new_recipe'],'sft',
                tokenizer_fingerprint=config['tokenizer_fingerprint'],role_ids=tuple(config['marker_ids'])))
            equal(before,trainer.capture_training_rng_state('cpu'))
        return result
    def update(model,optimizer,batch,**kwargs):
        orders[mode].extend(row.id for row in batch)
        return original_update(model,optimizer,batch,**kwargs)
    def run(out,resident_mode,resume=None):
        args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu',
            '--native-graphs',str(paths['old_train'] if resident_mode else paths['train']),
            '--held-native-graphs',str(paths['held']),'--ce-supervision-policy',str(paths['old_ce']if resident_mode else paths['new_ce']),
            '--epochs','2','--eval-every','1','--checkpoint-every','1','--checkpoint-seconds','0',
            '--seed','23','--out',str(out),'--stage-id','sft']
        if family=='mellum':args+=['--fp8-training','triton-experts']
        if resident_mode:args+=['--resident-data-directory',str(directory),'--recipe-binding',json.dumps(native_recipe_binding(paths['old_recipe'],old['recipe_sha256']))]
        if resume is not None:args+=['--continue-from',str(resume)]
        return trainer.main(args)
    with monkeypatch.context()as patch:
        patch.setattr(resident.ResidentAdoption,'__init__',initialize)
        patch.setattr(trainer,'_save',save);patch.setattr(trainer,'native_training_update',update)
        assert run(tmp_path/'live',True)==0
        mode='reference';assert run(tmp_path/'reference',False)==0
    assert published and resident.current(directory)
    assert orders['live']==orders['reference']and 'extra'in orders['live']
    live=torch.load(tmp_path/'live/checkpoint.pt',weights_only=False,map_location='cpu')
    reference=torch.load(tmp_path/'reference/checkpoint.pt',weights_only=False,map_location='cpu')
    for key in live.keys()-{'training'}:equal(live[key],reference[key])
    for key in ('optimizer','rng','global_step','step','exposure_counts'):
        equal(live['training'][key],reference['training'][key])
    assert live['training']['exposure_counts']=={'old':2,'extra':2}
    options=live['training']['options'];assert options['ce_supervision_policy_binding']==resident.binding(paths['new_ce'])
    event=options['ce_supervision_policy_history'][-1]
    assert event['at_global_step']==1 and event['kind']=='resident-additive-source-coverage'
    assert event['from']==resident.binding(paths['old_ce'])and event['to']==resident.binding(paths['new_ce'])
    # Restart with the OLD CLI CE path: the committed resident plan is the
    # authority, so no eligible new row falls back to the stale declaration.
    mode='restart';orders[mode]=[]
    with monkeypatch.context()as patch:
        patch.setattr(resident.ResidentAdoption,'__init__',initialize)
        assert run(tmp_path/'live',True,resume=tmp_path/'live/checkpoint.pt')==0
    restarted=torch.load(tmp_path/'live/checkpoint.pt',weights_only=False,map_location='cpu')
    for key in restarted.keys()-{'training'}:equal(restarted[key],live[key])
    for key in ('optimizer','rng','global_step','step','exposure_counts'):
        equal(restarted['training'][key],live['training'][key])
    assert restarted['training']['options']['ce_supervision_policy_binding']==resident.binding(paths['new_ce'])

@pytest.mark.parametrize('tamper',[False,True])
def test_real_resident_commit_carries_preloaded_policy_or_rejects_tamper(tmp_path,monkeypatch,tamper):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.native_supervision import load_policy
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,old,new=fixture(tmp_path);directory=tmp_path/'resident'/'resident-data'
    directory.parent.mkdir();write_json(directory.parent/'recipe-plan.json',old)
    evidence=tmp_path/'evidence.json';write_json(evidence,{'passed':True,'scope':'tiny CPU fixture'})
    capacity=tmp_path/'capacity.json';write_json(capacity,{
        'schema':'natlang.native-resident-geometry-qualification/1','passed':True,'identity':None,
        'runtime_code_sha256':resident.digest(old['code']),
        'limits':{'forward_tokens':1000000,'kv_slots':1000000},'evidence':resident.binding(evidence)})
    train=NativeGraphSource(paths['old_train'],'tokenizer-sha',required_split='train')
    held=NativeGraphSource(paths['held'],'tokenizer-sha',required_split='test')
    train.set_ce_policy(load_policy(resident.binding(paths['old_ce'])))
    manager=resident.ResidentAdoption(directory,stage_id='sft',
        recipe_binding=native_recipe_binding(paths['old_recipe'],old['recipe_sha256']),
        train=train,held=held,geometry_qualification=capacity)
    resident.prepare(directory,paths['new_recipe'],'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    if tamper:
        paths['new_ce'].write_text(paths['new_ce'].read_text()+' ')
        assert manager.poll(9)is None
        assert resident.current(directory)is None and manager.train is train and manager.ce_policy_update is None
        assert list((directory/'rejections').glob('*.json'))
    else:
        expanded=manager.poll(9);assert expanded and expanded[0].ordered_ids()==['old','extra']
        ref,policy=manager.ce_policy_update
        assert ref['sha256']==resident.sha(paths['new_ce'])and len(policy['sources'])==2
        assert resident.current(directory)['at_global_step']==9
        assert next(iter(train)).target_weights!=next(iter(NativeGraphSource(paths['old'],'tokenizer-sha',required_split='train'))).target_weights


def test_loaded_policy_cannot_disagree_with_resident_plan():
    from natlang_neuralese.train.native_resident_ce import require_loaded_policy
    require_loaded_policy(None,None)
    require_loaded_policy({'sha256':'a'*64,'path':'/alias'},{'sha256':'a'*64,'path':'/original'})
    for binding,reference in [(None,{'sha256':'a'*64}),({'sha256':'a'*64},None),
                              ({'sha256':'a'*64},{'sha256':'b'*64})]:
        with pytest.raises(ValueError,match='loaded CE'):require_loaded_policy(binding,reference)


@pytest.mark.parametrize('where',['active_sources','inventory'])
@pytest.mark.parametrize('byte_count',[True,-1,0,999999])
def test_complete_contract_optional_byte_counts_are_strict(tmp_path,where,byte_count):
    paths,old,new=fixture(tmp_path)
    doc=json.loads(paths['new_ce'].read_text());doc['source_contract'][where][0]['bytes']=byte_count
    write_json(paths['new_ce'],doc);ref=resident.binding(paths['new_ce'])
    new['recipe']['input_bindings']['ce']=ref
    new['stage_inputs']['sft']['ce_supervision_policy']={**ref,'binding':'ce'}
    with pytest.raises(ValueError,match='byte count'):resident._contract(old,new,'sft')


@pytest.mark.parametrize('preserve_both_aliases',[False,True])
def test_same_content_source_aliases_accept_both_builder_inventory_forms(tmp_path,preserve_both_aliases):
    paths,old,new=fixture(tmp_path)
    alias=tmp_path/'extra-alias.jsonl';alias.write_bytes(paths['extra'].read_bytes())
    dag=json.loads(paths['train'].read_text());dag['sources'].append({'kind':'native-graphs',**resident.binding(alias)})
    write_json(paths['train'],dag)
    refs=[resident.binding(paths[name])for name in ('old','extra')]
    if preserve_both_aliases:refs.append(resident.binding(alias))
    policy=complete_source_contract(json.loads(paths['old_ce'].read_text()),refs,
        context_weight=.05,inventory=[resident.binding(paths['train'])])
    write_json(paths['new_ce'],policy)
    for role,name in [('native_graphs','train'),('ce_supervision_policy','new_ce')]:
        ref=resident.binding(paths[name]);new['stage_inputs']['sft'][role]={**ref,'binding':role}
    assert extension_policy(old['stage_inputs']['sft'],new['stage_inputs']['sft']) is not None
