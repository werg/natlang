"""Resident append is a metadata transaction, never an optimizer/checkpoint fork."""
import copy
import json
from pathlib import Path

import pytest

from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs, native_recipe_binding
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.common.jsonio import write_json_durable as write_json
from test_native_state_sft import graph


def setup(tmp_path, fingerprint='tokenizer-sha', role_ids=(), geometry_identity=None, manifest_train=False):
    paths = {}
    for name, ids, split in [('old', ['a','b'], 'train'), ('new', ['a','b','c'], 'train'),
                             ('held', ['held'], 'test')]:
        rows = []
        for key in ids:
            row = graph(key);row.update(split=split,tokenizer_fingerprint=fingerprint)
            rows.append(row)
        paths[name] = tmp_path/(name+'.jsonl')
        paths[name].write_text(''.join(json.dumps(row)+'\n' for row in rows))
    if manifest_train:
        leaf=paths['old'];paths['old']=tmp_path/'old-manifest.json'
        write_json(paths['old'],{'schema':'natlang.native-graph-sources/1','sources':[
            {'kind':'native-graphs',**resident.binding(leaf)}]})
    old_recipe = {'schema':'natlang.neuralese-training-recipe/1','id':'resident','identity_qualification':'native_state_runtime_qualified',
        'input_bindings': {'train':resident.binding(paths['old']), 'held':resident.binding(paths['held'])},
        'stages':[{'id':'sft','kind':'native_state_sft','requires':[],
            'parameters':{'epochs':1.},'inputs':{'native_graphs':'train','held_native_graphs':'held'}}]}
    old_path=tmp_path/'old-recipe.json';write_json(old_path,old_recipe)
    recipe=load_recipe(old_path)
    plan={'schema':'natlang.neuralese-recipe-plan/1','recipe':recipe,
        'recipe_sha256':resident.sha(old_path),'device':'cpu','code':resident.code_inventory(Path(resident.__file__).parents[1]),
        'stage_inputs':{'sft':resolve_stage_inputs(recipe,recipe['stages'][0],{})},
        'inputs':{str(p):resident.sha(p) for p in (paths['old'],paths['held'])}}
    out=tmp_path/'run';out.mkdir();directory=out/'resident-data';write_json(out/'recipe-plan.json',plan)
    old=NativeGraphSource(paths['old'],fingerprint,required_split='train',role_ids=role_ids)
    held=NativeGraphSource(paths['held'],fingerprint,required_split='test',role_ids=role_ids)
    evidence=tmp_path/'geometry-proof.json';write_json(evidence,{'passed':True,'scope':'tiny CPU control fixture'})
    capacity=tmp_path/'geometry-capacity.json';write_json(capacity,
        {'schema':'natlang.native-resident-geometry-qualification/1','passed':True,
         'identity':geometry_identity,'runtime_code_sha256':resident.digest(plan['code']),
         'limits':{'forward_tokens':1000000,'kv_slots':1000000},'evidence':resident.binding(evidence)})
    manager=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=native_recipe_binding(old_path,plan['recipe_sha256']),
                                      train=old,held=held,geometry_qualification=capacity,geometry_identity=geometry_identity)
    new_recipe=copy.deepcopy(old_recipe)
    new_recipe['input_bindings']['train']=resident.binding(paths['new'])
    new_recipe['input_bindings']['prior']=resident.binding(paths['old'])
    new_recipe['stages'][0]['inputs']['continued_native_graphs']='prior'
    new_path=tmp_path/'new-recipe.json';write_json(new_path,new_recipe)
    return paths,directory,manager,new_path


def test_actual_indexes_append_commit_and_crash_projection_recovery(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    original=resident.repair
    with monkeypatch.context() as patch:
        patch.setattr(resident,'repair',lambda *a: (_ for _ in ()).throw(OSError('crash after atomic commit')))
        with pytest.raises(RuntimeError,match='needs recovery'):
            manager.poll(7)
    assert resident.current(directory)['at_global_step']==7
    assert not (directory/'rejections').exists()
    state=resident.recover(directory)
    assert json.loads((directory.parent/'recipe-plan.json').read_text())==state['payload']['plan']
    assert json.loads((directory/'active-binding.json').read_text())['generation']==state['generation']
    assert state['payload']['prior_recipe']['path'].endswith('old-recipe.json')
    # Committed data generation is recoverable; it does not invent saved row visits.
    assert 'exposure_counts' not in state
    loaded=NativeGraphSource(paths['new'],'tokenizer-sha',required_split='train')
    resumed=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=state['payload']['recipe'],train=loaded,held=manager.held)
    assert resumed.poll(7) is None


def test_resident_append_reuses_loaded_tree_without_reopening_old_leaf(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path,manifest_train=True)
    old_train,old_held=manager.train,manager.held
    extra=tmp_path/'extra.jsonl'
    extra.write_text(paths['new'].read_text().splitlines()[-1]+'\n')
    source=tmp_path/'append.json'
    write_json(source,{'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs',**resident.binding(paths['old'])},
        {'kind':'native-graphs',**resident.binding(extra)}]})
    value=json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(source)
    write_json(recipe,value)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    original=NativeGraphSource.__init__
    opened=[]
    def track(self,path,*args,**kwargs):
        opened.append(Path(path).resolve())
        assert Path(path).resolve() not in {old_train.path.resolve(),old_held.path.resolve(),
            old_train.children[0].path.resolve()}
        return original(self,path,*args,**kwargs)
    monkeypatch.setattr(NativeGraphSource,'__init__',track)
    train,held=manager.poll(9)
    assert train.children[0] is old_train and held is old_held
    assert train.entries[0] is old_train.entries[0]
    assert train.ordered_ids()==['a','b','c']
    assert opened==[extra.resolve()]


def test_reused_resident_tree_rejects_changed_leaf_before_commit(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    # The new root is valid; the unchanged held root must still be reverified.
    paths['held'].write_text(paths['held'].read_text().replace('"id": "held"','"id": "edited"'))
    assert manager.poll(9) is None
    assert resident.current(directory) is None
    assert manager.train.ordered_ids()==['a','b']


@pytest.mark.parametrize('prior_pending',[False,True])
def test_prepare_without_publication_preserves_live_pending_and_sgd(tmp_path,monkeypatch,prior_pending):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    pointer=directory/'pending.json'
    if prior_pending:
        pointer.write_text('{"untouched":"existing request"}\n')
    before=pointer.read_bytes() if pointer.exists() else None
    prepared=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),publish=False)
    assert (pointer.read_bytes() if pointer.exists() else None)==before
    assert resident.bound(prepared).is_file()
    assert resident.current(directory) is None and manager.train.ordered_ids()==['a','b']
    # Repeating normal preparation explicitly publishes this same validated request.
    published=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert published==prepared and json.loads(pointer.read_text())==published
    assert manager.poll(10)[0].ordered_ids()==['a','b','c']


def test_rejection_does_not_commit_or_change_loaded_source_and_same_pending_is_not_retried(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    witness=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    paths['new'].write_text(paths['new'].read_text().replace('"id": "a"','"id": "changed"'))
    assert manager.poll(10) is None
    assert manager.train.ordered_ids()==['a','b']
    assert resident.current(directory) is None
    error,=(directory/'rejections').glob('*.json')
    first=error.read_bytes()
    assert manager.poll(11) is None and error.read_bytes()==first


@pytest.mark.parametrize('change',['held','foundation','rate','runtime','priority','prefix'])
def test_strict_data_only_rejects_nonappend_changes_before_publication(tmp_path,monkeypatch,change):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    value=json.loads(recipe.read_text())
    if change=='held':
        value['input_bindings']['held']=resident.binding(paths['new'])
    elif change=='foundation':
        value['input_bindings']['foundation']=resident.binding(paths['old'])
        value['stages'][0]['inputs']['foundation_native_graphs']='foundation'
    elif change=='rate':value['stages'][0]['parameters']['learning_rate']=.03
    elif change=='runtime':value['runtime']={'changed':True}
    elif change=='prefix':paths['new'].write_text(paths['new'].read_text().replace('"id": "a"','"id": "changed"'))
    elif change=='priority':
        manifest=tmp_path/'priority.json';write_json(manifest,{'schema':'natlang.native-graph-sources/1',
            'sources':[{'kind':'native-graphs',**resident.binding(paths['new']),'training_priority':9}]})
        value['input_bindings']['train']=resident.binding(manifest)
    write_json(recipe,value)
    with pytest.raises(ValueError):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert not (directory/'pending.json').exists() and resident.current(directory) is None


def test_cached_witness_required_and_no_model_or_rng_action(tmp_path,monkeypatch):
    import torch
    import random
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    before=torch.random.get_rng_state().clone();python_rng=random.getstate()
    expanded=manager.poll(23)
    assert expanded[0].ordered_ids()==['a','b','c']
    assert torch.equal(before,torch.random.get_rng_state()) and python_rng==random.getstate()
    assert not list(directory.parent.rglob('*.pt'))
    assert manager.recipe_binding['path']==str(recipe)


def test_same_forward_class_metrics_are_exact_weighted_graph_aggregates(monkeypatch):
    import torch
    from types import SimpleNamespace
    from natlang_neuralese.train.native_state_sft import _evaluate
    from natlang_neuralese.data import native_graph
    calls=[]
    def loss(model,example,**kwargs):
        calls.append(example.id)
        return torch.tensor(example.loss),{'boundary_sums':{'7': {'count':1, 'weight':example.weight,
            'nll_sum':example.loss*example.weight, 'correct_weight':example.accuracy*example.weight,
            'probability_sum':.25*example.weight}},'weight':example.weight,
            'accuracy':example.accuracy,'supervised_tokens':3}
    monkeypatch.setattr(native_graph,'graph_loss',loss)
    examples=[SimpleNamespace(id='a',loss=2.,weight=2.,accuracy=.5,plan=SimpleNamespace(token_count=4)),
              SimpleNamespace(id='b',loss=4.,weight=1.,accuracy=1.,plan=SimpleNamespace(token_count=4))]
    model=torch.nn.Linear(1,1)
    model.marker_ids=(7,)
    result=_evaluate(model,examples,checkpoint_layers=False,chunk_tokens=128,
        graph_classes={'a':'ordinary','b':'communication'})
    assert calls==['a','b'] and model.training
    assert result['classes']['ordinary']['loss']==2.
    assert result['classes']['communication']['weighted_accuracy']==1.
    assert result['loss']==pytest.approx(8/3)
    assert result['classes']['ordinary']['boundaries']['7']['mean_nll']==2.
    assert result['classes']['communication']['boundaries']['7']['mean_nll']==4.
    assert result['classes']['ordinary']['boundaries']['7']['count']==1
    assert result['boundaries']['7']['weight']==3.


@pytest.mark.parametrize('crash_after_commit',[False,True])
@pytest.mark.parametrize('held_append',[False,True])
@pytest.mark.parametrize('pending_append',[False,True])
def test_real_resident_sgd_matches_all_data_run_and_recovers_old_checkpoint(tmp_path,monkeypatch,crash_after_commit,held_append,pending_append):
    import torch
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    identity={'model_config':config,'batch_graphs':1,'batch_tokens':0,'checkpoint_layers':True,'chunk_tokens':128,'checkpoint_input_offload':False,
        'fp8':{'kind':'off','recipe':'delayed','amax_history_len':16,'input_gradient_precision':None},'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':'cpu'}}
    paths,directory,manager,recipe=setup(tmp_path,config['tokenizer_fingerprint'],roles.ids,identity)
    pending_path = None
    if pending_append:
        from natlang_neuralese.train.native_pending_sources import declaration
        row=graph('long');row.update(split='train',tokenizer_fingerprint=config['tokenizer_fingerprint'])
        long=tmp_path/'long.jsonl';long.write_text(json.dumps(row)+'\n')
        pending_path=tmp_path/'pending.json';write_json(pending_path,declaration(long,long,
            tokenizer_fingerprint=config['tokenizer_fingerprint'],role_ids=roles.ids))
        value=json.loads(recipe.read_text());value['input_bindings']['pending']=resident.binding(pending_path)
        value['stages'][0]['inputs']['pending_native_sources']='pending';write_json(recipe,value)
    held_path=paths['held']
    if held_append:
        row=graph('held-extra');row.update(split='test',tokenizer_fingerprint=config['tokenizer_fingerprint'])
        held_path=tmp_path/'held-expanded.jsonl';held_path.write_text(paths['held'].read_text()+json.dumps(row)+'\n')
        declaration=json.loads(recipe.read_text());declaration['input_bindings']['held']=resident.binding(held_path)
        declaration['input_bindings']['prior-held']=resident.binding(paths['held'])
        declaration['stages'][0]['inputs']['continued_held_native_graphs']='prior-held';write_json(recipe,declaration)
    out=directory.parent/'sft'
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
          '--native-graphs',str(paths['old']),'--held-native-graphs',str(paths['held']),
          '--epochs','1','--eval-every','1','--checkpoint-every','1','--out',str(out),
          '--resident-data-directory',str(directory),'--resident-geometry-qualification',str(tmp_path/'geometry-capacity.json'),'--recipe-binding',json.dumps(manager.recipe_binding),'--stage-id','sft']
    original_due=trainer.CheckpointPolicy.due
    prepared=[]
    def due(policy,step,**kwargs):
        if step==1 and not prepared:
            prepared.append(resident.prepare(directory,recipe,'sft',
                tokenizer_fingerprint=config['tokenizer_fingerprint'],role_ids=roles.ids))
        return original_due(policy,step,**kwargs)
    class Crash(BaseException):pass
    original_poll=resident.ResidentAdoption.poll
    def crash_poll(self,step):
        result=original_poll(self,step)
        if result is not None:raise Crash()
        return result
    with monkeypatch.context() as patch:
        patch.setattr(trainer.CheckpointPolicy,'due',due)
        if crash_after_commit:
            patch.setattr(resident.ResidentAdoption,'poll',crash_poll)
            with pytest.raises(Crash):trainer.main(argv)
        else:assert trainer.main(argv)==0
    if crash_after_commit:
        before=torch.load(out/'checkpoint.pt',weights_only=False)
        assert before['training']['global_step']==1
        assert before['training']['source_row_order']==['a','b']
        assert resident.current(directory) is not None
        assert trainer.main(argv)==0
    actual=torch.load(out/'checkpoint.pt',weights_only=False)
    assert actual['training']['exposure_counts']=={'a':1,'b':1,'c':1}
    assert actual['training']['global_step']==3
    if pending_append:
        frame=actual['training']['sft_progress']['sft']
        assert frame['pending_source_rows']['long']['required_visits']==1
        assert frame['exposure_counts']['long']==0
        assert frame['required_visits']['long']==1
    assert actual['training']['options']['source_predecessors']=={
        'continued_native_graphs':resident.sha(paths['old']),
        'continued_held_native_graphs':resident.sha(paths['held'])}
    assert actual['training']['options']['resident_adoption']['generation']==resident.current(directory)['generation']
    baseline=tmp_path/'baseline'
    clean=argv[:]
    for flag in ('--resident-data-directory','--resident-geometry-qualification','--recipe-binding','--stage-id'):
        i=clean.index(flag);del clean[i:i+2]
    clean[clean.index('--native-graphs')+1]=str(paths['new'])
    clean[clean.index('--out')+1]=str(baseline)
    clean[clean.index('--held-native-graphs')+1]=str(held_path)
    if pending_path is not None:clean += ['--pending-native-sources',str(pending_path)]
    assert trainer.main(clean)==0
    expected=torch.load(baseline/'checkpoint.pt',weights_only=False)
    def exact(a,b):
        if torch.is_tensor(a):assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for key in a:exact(a[key],b[key])
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for x,y in zip(a,b):exact(x,y)
        else:assert a==b
    for key in ('backbone_state','interface_state'):exact(actual[key],expected[key])
    exact(actual['training']['optimizer'],expected['training']['optimizer'])
    exact(actual['training']['rng'],expected['training']['rng'])


def test_new_geometry_exceeding_bound_runtime_memory_policy_is_pending_not_committed(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    capacity=tmp_path/'geometry-capacity.json';value=json.loads(capacity.read_text())
    value['limits']['forward_tokens']=2;write_json(capacity,value)
    initial=json.loads((directory/'initial.json').read_text());initial['geometry_qualification']=resident.binding(capacity)
    write_json(directory/'initial.json',initial)
    with pytest.raises(ValueError,match='exceeds qualified'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    assert resident.current(directory) is None and manager.train.ordered_ids()==['a','b']


@pytest.mark.parametrize('pending_kind',['adopted','newer','corrupt'])
def test_retirement_needs_actual_adopted_checkpoint_and_drained_owner(tmp_path,monkeypatch,pending_kind):
    import torch,time
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    manager.poll(7)
    initial=json.loads((directory/'initial.json').read_text());initial['owner']={'pid':-1}
    write_json(directory/'initial.json',initial)
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    checkpoint=tmp_path/'checkpoint.pt';torch.save({'schema':CHECKPOINT_SCHEMA,'training':{
        'schema':TRAINING_SCHEMA,'global_step':7,'optimizer':{'test':1},'rng':{'test':1},'options':{},'source_receipts':[{}]}},checkpoint)
    terminal=tmp_path/'exit.json';write_json(terminal,{'at':time.time(),'returncode':0})
    with pytest.raises(ValueError,match='checkpoint of the adopted'):
        resident.retire(directory,resident.binding(checkpoint),resident.binding(terminal))
    state=resident.current(directory)
    torch.save({'schema':CHECKPOINT_SCHEMA,'training':{'schema':TRAINING_SCHEMA,'global_step':7,
        'optimizer':{'test':1},'rng':{'test':1},'options':{'resident_adoption':{'generation':state['generation']}},
        'source_receipts':[state['payload']['train']]}},checkpoint)
    write_json(terminal,{'at':0,'returncode':0})
    with pytest.raises(ValueError,match='drained optimizer'):
        resident.retire(directory,resident.binding(checkpoint),resident.binding(terminal))
    write_json(terminal,{'at':time.time(),'returncode':0})
    pointer=directory/'pending.json'
    if pending_kind=='newer':
        next_payload={**state['payload'],'stage_id':'another-stage'}
        request=directory/'next-request.json'
        write_json(request,{'schema':resident.SCHEMA,'payload':next_payload,'generation':resident.digest(next_payload)})
        write_json(pointer,resident.binding(request))
    elif pending_kind=='corrupt':
        value=json.loads(pointer.read_text());value['sha256']='0'*64;write_json(pointer,value)
        with pytest.raises(ValueError,match='resident artifact changed'):
            resident.retire(directory,resident.binding(checkpoint),resident.binding(terminal))
        assert resident.current(directory)==state
        assert not (directory/'retired'/(state['generation']+'.json')).exists()
        return
    resident.retire(directory,resident.binding(checkpoint),resident.binding(terminal))
    assert pointer.exists()==(pending_kind=='newer')
    assert resident.current(directory) is None
    assert json.loads((directory.parent/'recipe-plan.json').read_text())==state['payload']['plan']
    assert (directory/'retired'/(state['generation']+'.json')).exists()


def test_held_append_preserves_old_rows_and_class_coverage_and_checks_cross_split_groups(tmp_path,monkeypatch):
    from natlang_neuralese.train.native_held import held_selection
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    old_held=manager.held
    indices,coverage,selection=held_selection(old_held.entries,{},source_sha256=old_held.file_sha256,
                                            graph_budget=1)
    extra=graph('new-held');extra.update(split='test',tokenizer_fingerprint='tokenizer-sha')
    path=tmp_path/'held-expanded.jsonl';path.write_text(paths['held'].read_text()+json.dumps(extra)+'\n')
    value=json.loads(recipe.read_text());value['input_bindings']['held']=resident.binding(path)
    value['input_bindings']['prior-held']=resident.binding(paths['held'])
    value['stages'][0]['inputs']['continued_held_native_graphs']='prior-held';write_json(recipe,value)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    train,held=manager.poll(9)
    assert held.ordered_ids()==['held','new-held']
    selected,next_coverage,_=held_selection(held.entries,coverage,source_sha256=held.file_sha256,graph_budget=1)
    assert selected==[1]  # New held row before repeating any already visited old row.
    state=resident.current(directory)
    assert state['payload']['held']==resident.binding(path)
    assert state['history'][-1]['held']==resident.binding(path)
    assert next_coverage['graph_visits']==2


def test_precommit_failure_keeps_old_generation_and_restart_retries_exact_request(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    original=resident.write_json
    def fail_commit(path,value):
        if Path(path).name=='current.json':raise OSError('crash before rename')
        return original(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(resident,'write_json',fail_commit)
        assert manager.poll(9) is None
    assert resident.current(directory) is None and manager.train.ordered_ids()==['a','b']
    retry=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=manager.recipe_binding,
        train=manager.train,held=manager.held)
    assert retry.poll(9)[0].ordered_ids()==['a','b','c']


def test_previous_generation_ready_receipt_is_not_current_data_qualification(tmp_path):
    from test_native_readiness import ready
    from natlang_neuralese.train.native_readiness import validate_ready
    path,receipt,requirement=ready(tmp_path,'lfm')
    directory=tmp_path/'resident-data';directory.mkdir()
    payload={'stage_id':'graph','recipe':receipt['recipe']}
    generation=resident.digest(payload)
    write_json(directory/'current.json',{'schema':resident.SCHEMA,'payload':payload,'generation':generation})
    assert validate_ready(path,requirement) is None
    receipt['resident_adoption']={'generation':generation};write_json(path,receipt)
    assert validate_ready(path,requirement)['model_id']=='lfm'


def test_resident_readiness_accepts_original_or_restarted_launch_but_not_unrelated_owner(tmp_path):
    import time
    from test_native_readiness import ready
    from natlang_neuralese.train.native_readiness import validate_ready
    path,receipt,requirement=ready(tmp_path,'lfm')
    directory=tmp_path/'resident-data';directory.mkdir()
    old=tmp_path/'original-recipe.json';old.write_text('{}')
    payload={'stage_id':'graph','recipe':receipt['recipe'],'prior_recipe':resident.binding(old)}
    generation=resident.digest(payload)
    write_json(directory/'current.json',{'schema':resident.SCHEMA,'payload':payload,'generation':generation,
        'history':[{'recipe':receipt['recipe']}]})
    receipt['resident_adoption']={'generation':generation};write_json(path,receipt)
    launch=tmp_path/'launch.json';terminal=tmp_path/'exit.json';write_json(terminal,{'at':time.time(),'returncode':0})
    requirement.update(trainer_exit=str(terminal),trainer_recipe=receipt['recipe']['path'],
                       resident_adoption=resident.binding(directory/'current.json'))
    for actual in (str(old),receipt['recipe']['path']):
        write_json(launch,{'at':1,'argv':['python','--recipe',actual]})
        requirement['trainer_launch']=resident.binding(launch)
        assert validate_ready(path,requirement)['model_id']=='lfm'
    write_json(launch,{'at':1,'argv':['python','--recipe','unrelated']})
    requirement['trainer_launch']=resident.binding(launch)
    with pytest.raises(ValueError,match='different trainer launch'):
        validate_ready(path,requirement)



def delta_fixture(tmp_path, monkeypatch, *, duplicate=False):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    paths, directory, manager, recipe = setup(tmp_path, manifest_train=True)
    delta = tmp_path/'delta.jsonl'
    delta.write_text(paths['new'].read_text().splitlines()[0 if duplicate else -1]+'\n')
    wrapper = tmp_path/'delta-wrapper.json'
    write_json(wrapper, {'schema':'natlang.native-graph-sources/1','sources':[
        {'kind':'native-graphs', **resident.binding(paths['old'])},
        {'kind':'native-graphs', **resident.binding(delta)}]})
    value = json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(wrapper)
    write_json(recipe,value)
    return paths,directory,manager,recipe,delta,wrapper


def test_delta_prepare_loads_only_new_graphs_then_retained_learner_validates_and_resumes(tmp_path, monkeypatch):
    paths,directory,manager,recipe,delta,wrapper=delta_fixture(tmp_path,monkeypatch)
    old_train, old_held = manager.train, manager.held
    original=NativeGraphSource.__init__;opened=[]
    def track(self,path,*args,**kwargs):
        opened.append(Path(path).resolve())
        assert Path(path).resolve() not in {old_train.path.resolve(), old_held.path.resolve(),
            old_train.children[0].path.resolve()}
        return original(self,path,*args,**kwargs)
    monkeypatch.setattr(NativeGraphSource,'__init__',track)
    request=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',
        role_ids=(),delta_sources={'train':resident.binding(delta)})
    payload=json.loads(resident.bound(request).read_text())['payload']
    assert 'entries_sha256' not in payload and 'held_entries_sha256' not in payload
    assert payload['delta_validation']['prior_rows']==2 and payload['rows']==3
    assert set(opened)=={delta.resolve()}
    with pytest.raises(ValueError,match='not a full-source'):
        resident.geometry_from_evidence(manager.train,0,request,package=manager.package)
    train,held=manager.poll(9)
    assert train.children[0] is old_train and held is old_held
    assert [row['id']for row in train.entries]==['a','b','c']
    committed=resident.current(directory)
    assert committed['validation']=={'entries_sha256':resident.digest(train.entries),
        'held_entries_sha256':resident.digest(held.entries)}
    assert 'exposure_counts'not in committed
    resumed=resident.ResidentAdoption(directory,stage_id='sft',recipe_binding=committed['payload']['recipe'],
        train=train,held=held)
    assert resumed.poll(10)is None
    assert resident.recover(directory)['generation']==committed['generation']


def test_delta_duplicate_old_identity_is_rejected_by_actual_union_without_credit(tmp_path,monkeypatch):
    _,directory,manager,recipe,delta,_=delta_fixture(tmp_path,monkeypatch,duplicate=True)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
        delta_sources={'train':resident.binding(delta)})
    assert manager.poll(9)is None
    assert resident.current(directory)is None
    assert [row['id']for row in manager.train.entries]==['a','b']
    assert list((directory/'rejections').glob('*.json'))


@pytest.mark.parametrize('change',['priority','held','suffix','source'])
def test_delta_preparation_or_adoption_rejects_contract_drift(tmp_path,monkeypatch,change):
    paths,directory,manager,recipe,delta,wrapper=delta_fixture(tmp_path,monkeypatch)
    if change=='priority':
        value=json.loads(wrapper.read_text());value['sources'][0]['training_priority']=4;write_json(wrapper,value)
        value=json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(wrapper);write_json(recipe,value)
    elif change=='held':
        value=json.loads(recipe.read_text());value['input_bindings']['held']=resident.binding(delta);write_json(recipe,value)
    if change in ('priority','held'):
        with pytest.raises(ValueError,match='resident'):
            resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
                delta_sources={'train':resident.binding(delta)})
    else:
        request=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
        if change=='source':delta.write_text(delta.read_text()+' ')
        else:
            value=json.loads(resident.bound(request).read_text());value['payload']['delta_validation']['entries_sha256']='0'*64
            value['generation']=resident.digest(value['payload']);forged=directory/'forged.json';write_json(forged,value)
            write_json(directory/'pending.json',resident.binding(forged))
        assert manager.poll(9)is None and resident.current(directory)is None



def test_delta_geometry_limit_and_lost_prior_index_fail_closed(tmp_path,monkeypatch):
    paths,directory,manager,recipe,delta,wrapper=delta_fixture(tmp_path,monkeypatch)
    initial=json.loads((directory/'initial.json').read_text())
    capacity=json.loads(resident.bound(initial['geometry_qualification']).read_text())
    capacity['limits']={'forward_tokens':1,'kv_slots':1}
    low=tmp_path/'too-small-capacity.json';write_json(low,capacity)
    initial['geometry_qualification']=resident.binding(low);write_json(directory/'initial.json',initial)
    with pytest.raises(ValueError,match='whole-graph geometry'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
    assert not(directory/'pending.json').exists()



def test_delta_helper_defers_full_held_separation_to_actual_resident(tmp_path,monkeypatch):
    paths,directory,manager,recipe,delta,wrapper=delta_fixture(tmp_path,monkeypatch)
    row=json.loads(delta.read_text());row['source_groups']=['source-held'];delta.write_text(json.dumps(row)+'\n')
    value=json.loads(wrapper.read_text());value['sources'][1]={'kind':'native-graphs',**resident.binding(delta)};write_json(wrapper,value)
    value=json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(wrapper);write_json(recipe,value)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
        delta_sources={'train':resident.binding(delta)})
    assert manager.poll(9)is None and resident.current(directory)is None
    errors=[json.loads(p.read_text())['error']for p in(directory/'rejections').glob('*.json')]
    assert errors and 'protected append'in errors[0]


@pytest.mark.parametrize('explicit_side',['certificate','actual'])
def test_geometry_identity_accepts_only_implicit_ieee_default(tmp_path,explicit_side):
    import copy
    identity={'model_config':{'architecture':'lfm2','native_linear_precision':'bf16x6'},
              'batch_tokens':65536,'runtime':{'device':'cuda'}}
    explicit=copy.deepcopy(identity);explicit['model_config']['native_attention_precision']='ieee'
    evidence=tmp_path/'proof.json';write_json(evidence,{'passed':True})
    certificate=tmp_path/'capacity.json'
    value={'schema':'natlang.native-resident-geometry-qualification/1','passed':True,
        'identity':explicit if explicit_side=='certificate'else identity,
        'runtime_code_sha256':resident.digest({'module':'bound'}),
        'limits':{'forward_tokens':94236,'kv_slots':157075},'evidence':resident.binding(evidence)}
    write_json(certificate,value);original=certificate.read_bytes()
    actual=identity if explicit_side=='certificate'else explicit
    assert resident.geometry_capacity(resident.binding(certificate),identity=actual,code={'module':'bound'})==value['limits']
    assert certificate.read_bytes()==original
    for mode in ('tf32x3',None,'invalid'):
        wrong=copy.deepcopy(actual);wrong['model_config']['native_attention_precision']=mode
        with pytest.raises(ValueError,match='differs'):
            resident.geometry_capacity(resident.binding(certificate),identity=wrong,code={'module':'bound'})
    wrong=copy.deepcopy(actual);wrong['batch_tokens']+=1
    with pytest.raises(ValueError,match='differs'):
        resident.geometry_capacity(resident.binding(certificate),identity=wrong,code={'module':'bound'})
