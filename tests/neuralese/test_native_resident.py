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


def test_packed_candidate_qualification_precedes_resident_commit(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    geom=tmp_path/'prepared-geometry.json';write_json(geom,{'source':'fixture'})
    ref=resident.binding(geom)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),packed_geometry=ref)
    seen=[]
    def reject(train,held,payload,ce_update,declarations):
        assert resident.current(directory) is None
        assert payload['sft_packed_geometry']==ref and train.ordered_ids()==['a','b','c']
        seen.append(True)
        raise ValueError('actual packed case did not fit')
    assert manager.poll(13,candidate_check=reject) is None
    assert seen and resident.current(directory) is None and manager.train.ordered_ids()==['a','b']
    # A new immutable request can be retried and committed only after its check.
    manager.rejected.clear()
    def accept(train,held,payload,ce_update,declarations):
        assert resident.current(directory) is None
        return {'proof':'live-model-measured'}
    assert manager.poll(14,candidate_check=accept)[0].ordered_ids()==['a','b','c']
    assert manager.packed_capacity_update=={'proof':'live-model-measured'}
    assert resident.current(directory)['at_global_step']==14


def test_resident_commits_only_after_actual_live_packed_backward(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
    from natlang_neuralese.train.native_packed_capacity import geometry_for_source,qualify_pending
    from test_native_optimizer_checkpointing import assert_exact
    from test_shared_state import tiny
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    paths,directory,manager,recipe=setup(tmp_path)
    candidate=NativeGraphSource(paths['new'],'tokenizer-sha',required_split='train')
    _,geometry,_=geometry_for_source(candidate,tmp_path,allow_build=True)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),packed_geometry=geometry)
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    native_training_update(model,optimizer,list(candidate.iter_indices([0])),checkpoint_layers=True,chunk_tokens=3)
    gradients={name:(parameter.grad,parameter.grad.detach().clone())
               for name,parameter in optimizer._native_named_parameters if parameter.grad is not None}
    assert gradients
    with torch.no_grad():model.port.embedding_weight[1].add_(.1)
    before=copy.deepcopy(model.state_dict());optimizer_before=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    args=SimpleNamespace(out=tmp_path/'measurement',checkpoint_layers=True,chunk_tokens=3)
    def actual(train,held,payload,ce_update,declarations):
        assert resident.current(directory) is None
        return qualify_pending(model,optimizer,train,[(0,[0,1],False)],args,backend=None,
            checkpoint_sha256='a'*64,global_step=17,prior_geometry=payload['sft_packed_geometry'],live_uncommitted=True)
    assert manager.poll(17,candidate_check=actual)[0].ordered_ids()==['a','b','c']
    proof=json.loads(Path(manager.packed_capacity_update['proof']['path']).read_text())
    assert proof['model_boundary']=='live-uncommitted' and proof['global_step']==17
    assert proof['capacity']['cases'][0]['example_ids']==['a','b'] and proof['passed']
    for key,value in model.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert_exact(optimizer.state_dict(),optimizer_before)
    for name,parameter in optimizer._native_named_parameters:
        if name in gradients:
            assert parameter.grad is gradients[name][0]
            torch.testing.assert_close(parameter.grad,gradients[name][1],rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())


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
    identity={'model_config':config,'batch_graphs':1,'batch_tokens':0,'checkpoint_layers':True,'chunk_tokens':1024,'checkpoint_input_offload':False,'cotangent_accumulator_backend':'plain',
        'share_causal_prefixes':True,  # on by default for a new stage (owner 2026-10-28)
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
    original=NativeGraphSource._parse;parsed=[]
    def track(self,row):
        parsed.append(Path(self.path).resolve())
        return original(self,row)
    monkeypatch.setattr(NativeGraphSource,'_parse',track)
    request=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',
        role_ids=(),delta_sources={'train':resident.binding(delta)})
    payload=json.loads(resident.bound(request).read_text())['payload']
    assert 'entries_sha256' not in payload and 'held_entries_sha256' not in payload
    assert payload['delta_validation']['prior_rows']==2 and payload['rows']==3
    # The explicit new source is parsed once for validation and again by the
    # exact geometry reader; neither pass touches the prior or held leaf.
    assert parsed==[delta.resolve(),delta.resolve()]
    assert len(payload['indexes'])==3  # old, delta and held leaves, all FCB-bound
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


def test_delta_exact_duplicate_identity_is_rejected_before_publication(tmp_path,monkeypatch):
    _,directory,manager,recipe,delta,_=delta_fixture(tmp_path,monkeypatch,duplicate=True)
    with pytest.raises(ValueError,match='repeats an already represented row identity'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
    assert resident.current(directory)is None
    assert not (directory/'pending.json').exists()
    assert [row['id']for row in manager.train.entries]==['a','b']


def test_delta_conflicting_same_id_is_rejected_before_publication(tmp_path,monkeypatch):
    _,directory,manager,recipe,delta,wrapper=delta_fixture(tmp_path,monkeypatch)
    changed=graph('a');changed['nodes'][0]['target_weights']=[0,.25,1]
    delta.write_text(json.dumps(changed)+'\n')
    value=json.loads(wrapper.read_text());value['sources'][1]={'kind':'native-graphs',**resident.binding(delta)}
    write_json(wrapper,value)
    value=json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(wrapper)
    write_json(recipe,value)
    with pytest.raises(ValueError,match='conflicts with an existing row identity'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
    assert resident.current(directory)is None and not (directory/'pending.json').exists()
    assert manager.train.ordered_ids()==['a','b']


def test_delta_prior_prefix_preflight_fails_closed_without_warm_index(tmp_path,monkeypatch):
    paths,directory,manager,recipe,delta,_=delta_fixture(tmp_path,monkeypatch)
    witness,=resident._source_witness(manager.train.children[0],manager.package)
    Path(witness['cache']['path']).unlink()
    original=NativeGraphSource._parse
    def forbid_old_parse(self,row):
        if Path(self.path).resolve()==paths['old'].resolve():
            raise AssertionError('missing prior index must not trigger a full-source reparse')
        return original(self,row)
    monkeypatch.setattr(NativeGraphSource,'_parse',forbid_old_parse)
    with pytest.raises(ValueError,match='exact completed source index'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
    assert not (directory/'pending.json').exists()


def test_delta_prior_prefix_does_not_fall_back_if_index_disappears_after_preflight(tmp_path,monkeypatch):
    paths,directory,manager,recipe,delta,_=delta_fixture(tmp_path,monkeypatch)
    witness,=resident._source_witness(manager.train.children[0],manager.package)
    cache=Path(witness['cache']['path'])
    from natlang_neuralese.common import row_index
    original_load=row_index.load
    removed=False
    def remove_after_preflight(path):
        nonlocal removed
        value=original_load(path)
        if Path(path)==cache and not removed:
            removed=True
            cache.unlink()
        return value
    monkeypatch.setattr(row_index,'load',remove_after_preflight)
    with pytest.raises(ValueError,match='disappeared before cache-only load'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
    assert removed and paths['old'].is_file()
    assert not (directory/'pending.json').exists()


def test_delta_rejects_held_row_identity_even_with_forged_disjoint_group(tmp_path,monkeypatch):
    _,directory,manager,recipe,delta,wrapper=delta_fixture(tmp_path,monkeypatch)
    row=graph('held');row.update(split='train',tokenizer_fingerprint='tokenizer-sha',source_groups=['forged-new-group'])
    delta.write_text(json.dumps(row)+'\n')
    value=json.loads(wrapper.read_text());value['sources'][1]={'kind':'native-graphs',**resident.binding(delta)}
    write_json(wrapper,value)
    value=json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(wrapper);write_json(recipe,value)
    with pytest.raises(ValueError,match='repeats a protected held row identity'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
    assert manager.held.ordered_ids()==['held']
    assert resident.current(directory)is None and not (directory/'pending.json').exists()


@pytest.mark.parametrize('mutation',['withdraw','replace'])
def test_pending_request_change_cannot_commit_after_validation(tmp_path,monkeypatch,mutation):
    from contextlib import contextmanager

    _,directory,manager,recipe,delta,_=delta_fixture(tmp_path,monkeypatch)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
        delta_sources={'train':resident.binding(delta)})
    original_locked=resident.locked
    @contextmanager
    def withdraw_while_holding_commit_lock(path):
        with original_locked(path):
            if Path(path)==directory:
                pointer=directory/'pending.json'
                if mutation=='withdraw':
                    pointer.unlink()
                else:
                    write_json(pointer,{'path':'replaced-request.json','sha256':'0'*64})
            yield
    monkeypatch.setattr(resident,'locked',withdraw_while_holding_commit_lock)
    assert manager.poll(9)is None
    assert resident.current(directory)is None
    if mutation=='withdraw':
        assert not (directory/'pending.json').exists()
    else:
        assert json.loads((directory/'pending.json').read_text())=={
            'path':'replaced-request.json','sha256':'0'*64}
    errors=[json.loads(path.read_text()) for path in(directory/'rejections').glob('*.json')]
    expected='withdrawn before commit' if mutation=='withdraw' else 'changed before commit'
    assert len(errors)==1 and expected in errors[0]['error']
    assert manager.train.ordered_ids()==['a','b']


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



def test_delta_preparation_rejects_held_group_overlap_before_publication(tmp_path,monkeypatch):
    paths,directory,manager,recipe,delta,wrapper=delta_fixture(tmp_path,monkeypatch)
    row=json.loads(delta.read_text());row['source_groups']=['source-held'];delta.write_text(json.dumps(row)+'\n')
    value=json.loads(wrapper.read_text());value['sources'][1]={'kind':'native-graphs',**resident.binding(delta)};write_json(wrapper,value)
    value=json.loads(recipe.read_text());value['input_bindings']['train']=resident.binding(wrapper);write_json(recipe,value)
    with pytest.raises(ValueError,match='overlaps protected held source groups'):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)})
    assert resident.current(directory)is None and not (directory/'pending.json').exists()


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


@pytest.mark.parametrize('architecture,dtype',[('lfm2','torch.float32'),('mellum','torch.bfloat16')])
@pytest.mark.parametrize('explicit_side',['certificate','actual'])
def test_geometry_capacity_ignores_only_explicit_keep_rendering_metadata(tmp_path,architecture,dtype,explicit_side):
    import copy
    identity={'model_config':{'architecture':architecture,'dtype':dtype,'native_linear_precision':'bf16x6'},
              'batch_tokens':65536,'checkpoint_input_offload':False}
    keep=copy.deepcopy(identity);keep['model_config']['history_reasoning_policy']='keep'
    evidence=tmp_path/'proof.json';write_json(evidence,{'passed':True})
    certificate=tmp_path/'capacity.json';value={'schema':'natlang.native-resident-geometry-qualification/1',
        'passed':True,'identity':keep if explicit_side=='certificate' else identity,
        'runtime_code_sha256':resident.digest({'module':'same'}),'limits':{'forward_tokens':94236,'kv_slots':157075},
        'evidence':resident.binding(evidence)}
    write_json(certificate,value);original=certificate.read_bytes();actual=identity if explicit_side=='certificate' else keep
    assert resident.geometry_capacity(resident.binding(certificate),identity=actual,code={'module':'same'})==value['limits']
    assert certificate.read_bytes()==original
    for field,bad in [('history_reasoning_policy',None),('history_reasoning_policy','last_turn_only'),
                      ('history_reasoning_policy','invented'),('dtype','torch.float16'),
                      ('native_linear_precision','ieee'),('native_attention_precision','tf32x3')]:
        wrong=copy.deepcopy(actual);wrong['model_config'][field]=bad
        with pytest.raises(ValueError,match='differs'):
            resident.geometry_capacity(resident.binding(certificate),identity=wrong,code={'module':'same'})
    for field,bad in [('batch_tokens',32768),('checkpoint_input_offload',True)]:
        wrong=copy.deepcopy(actual);wrong[field]=bad
        with pytest.raises(ValueError,match='differs'):
            resident.geometry_capacity(resident.binding(certificate),identity=wrong,code={'module':'same'})
    with pytest.raises(ValueError,match='differs'):
        resident.geometry_capacity(resident.binding(certificate),identity=actual,code={'module':'unreviewed'})


def whole_delta_geometry(tmp_path,delta,manager):
    source=NativeGraphSource(delta,'tokenizer-sha',required_split='train')
    measured=[]
    for index,example in enumerate(source.iter_indices(range(len(source)))):
        measured.append({'index':index,'id':example.id,'binding':source.entries[index]['binding'],
            'forward_tokens':example.plan.token_count,'kv_slots':len(example.plan.slot_sources)})
    producer=tmp_path/'completed-geometry-producer.json';write_json(producer,{'passed':True,'rows':measured})
    report={'schema':'natlang.native-whole-delta-geometry/1','passed':True,'source':resident.binding(delta),
        'prior_train':None,'old_train_rows':0,'new_train_rows':len(source),'entries_sha256':resident.digest(source.entries),
        'tokenizer_fingerprint':source.tokenizer_fingerprint,'role_ids':list(source.role_ids),
        'indexes':resident._source_witness(source,manager.package),'provenance':[resident.binding(producer)],
        'maxima':{key:max(measured,key=lambda row:row[key])for key in('forward_tokens','kv_slots')}}
    path=tmp_path/'whole-delta-geometry.json';write_json(path,report)
    return path


def d31_reference_qa_part(tmp_path, *, model='lfm25', mutation=None):
    import hashlib
    roles=[64402,64403,64404,64405]
    fingerprint='qa-tokenizer'
    raw_path=tmp_path/'rendered-whole.jsonl'
    graph_path=tmp_path/'graphs.jsonl'
    raw_rows=[];graph_rows=[]
    for index in range(2):
        raw_id=f'whole-trajectory:teacher-program:p{index}:decision:0001:source-sha256:{index:064x}'
        text=f'question {index} answer {index}'
        groups=[f'qa-question:{index}',f'qa-task:{index}']
        records=[{'id':f'teacher-program:p{index}:decision:0001','sha256':f'{index+1:064x}'}]
        record_ids=[row['id'] for row in records]
        source_record_groups={record_ids[0]:groups}
        raw={'corpus':'qa-source','id':raw_id,'split':'train','text':text,
            'tokenizer_sha256':'source-tokenizer','source_groups':groups,
            'source_record_ids':record_ids,'source_records':records,
            'source_record_groups':source_record_groups}
        graph_row=graph(f'text:{raw_id}')
        graph_row['tokenizer_fingerprint']=fingerprint
        graph_row['nodes'][0]['target_weights']=[0.,0.05,1.]
        graph_row['source_groups']=groups
        graph_row['conversion']={'source_row_id':raw_id,
            'source_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'source_record_ids':record_ids,'source_records':records,
            'source_record_groups':source_record_groups,
            'source_tokenizer_fingerprint':'source-tokenizer',
            'supervision_policy':{'failed_assistant_tokens_masked':0,
                'runtime_opening_assistant_turns':1,'runtime_opening_assistant_tokens':1,
                'assistant_target_tokens':1}}
        raw_rows.append(raw);graph_rows.append(graph_row)
    raw_path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in raw_rows))
    graph_path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in graph_rows))
    def ref(path): return resident.binding(path)|{'bytes':path.stat().st_size}
    qa_root=tmp_path/'d31-source-root';qa_item=qa_root/'data'/'qa_sft.json'
    qa_item.parent.mkdir(parents=True);qa_item.write_text('{"sha":"source"}\n')
    source_manifest=tmp_path/'d31-source-manifest.json'
    qa_member={'path':'data/qa_sft.json','bytes':qa_item.stat().st_size,'sha256':resident.sha(qa_item)}
    if mutation=='qa-input-manifest': qa_member['sha256']='0'*64
    write_json(source_manifest,{'schema':'natlang.corpus-snapshot/1','id':'external-expansion-d31-source-20261014-v1',
        'path':'d31-source-root','files':[qa_member]})
    render_manifest=tmp_path/'render-manifest.json'
    write_json(render_manifest,{'schema':'natlang.corpus-snapshot/1','id':'qa-render','files':[
        {**ref(raw_path),'path':f'whole-trajectory/{model}/text-data.jsonl.zst'}]})
    render_policy=tmp_path/'render-policy.json'
    write_json(render_policy,{'schema':'natlang.render-parts-policy-binding-file/1','binding':{'controls':[]}})
    policy=json.loads(render_policy.read_text());policy['sha256']=resident.digest(policy['binding']);write_json(render_policy,policy)
    qa_admission={'schema':'natlang.external-static-admission/1','approved':True,'training_exposure':0,
        'source_manifest':ref(source_manifest),'qa_inputs':[ref(qa_item)],
        'closure':{'protected_hits':0,'closure_violations':0,'invalid':0}}
    audit_path=tmp_path/'geometry-audit.json';write_json(audit_path,{'schema':'natlang.d31-qa-native-whole-graph-geometry-audit/1',
        'training_admission':False,'training_exposure':0,'models':{model:{'rows':2,'sha256':resident.sha(graph_path)}}})
    supervision={'rows':2,'failed_assistant_tokens_masked':0,
        'runtime_opening_assistant_turns':2,'runtime_opening_assistant_tokens':2,
        'assistant_target_tokens':2}
    conversion_path=tmp_path/'conversion-receipt.json';write_json(conversion_path,{
        'schema':'natlang.native-graph-text-build/1','model_id':'qa-model',
        'source':ref(raw_path)|{'rows':2},'expanded_tokenizer_fingerprint':fingerprint,
        'role_token_ids':roles,'outputs':[ref(graph_path)|{'path':f'{model}/train/native-graphs.jsonl.zst','rows':2}],
        'whole_trajectory_supervision':{'train':supervision}})
    review={'schema':'natlang.d31-reference-qa-whole-trajectory-native-graphs/1',
        'source_class':'static-reference-answer-supervision','source_corpus_id':'qa-source',
        'render_corpus_id':'qa-render','training_admission':True,'training_exposure':0,
        'whole_trajectory_only':True,'approved_source_custody':True,
        'quality_boundary':'Static reference QA plumbing only; not independently verified research execution, tool success, on-policy generation, communication-foundation credit, RL credit, or model-quality claim.',
        'omissions':{'conversion':0,'failed_assistant_targets':0,'protected_hits':0},
        'qa_source_admission':qa_admission,'source_manifest':ref(render_manifest),
        'render_manifest':ref(render_manifest),'render_policy_binding':ref(render_policy),
        'geometry_audit':ref(audit_path),
        'program_header_crosscheck':{'source_record_ids_equal':True},
        'models':{model:{'tokenizer_fingerprint':fingerprint,'role_token_ids':roles,
            'model_id':'qa-model','source':ref(raw_path)|{'rows':2},
            'graph_output':ref(graph_path)|{'rows':2},'receipt':ref(conversion_path),
            'whole_trajectory_supervision':supervision}}}
    if mutation=='held': review['qa_source_admission']['closure']['protected_hits']=1
    review_path=tmp_path/'admission-review.json';write_json(review_path,review)
    corpus_path=tmp_path/'corpus-manifest.json'
    entries={
        'keep-whole-graphs-v1/admission-review.json':review_path,
        'keep-whole-graphs-v1/geometry-audit.json':audit_path,
        f'keep-whole-graphs-v1/{model}/receipt.json':conversion_path,
        f'keep-whole-graphs-v1/{model}/train/native-graphs.jsonl.zst':graph_path}
    write_json(corpus_path,{'schema':'natlang.corpus-snapshot/1',
        'id':'external-expansion-deepdive-reference-qa-native-graphs-20261018-v1',
        'files':[{**ref(path),'path':name}for name,path in entries.items()]})
    source=NativeGraphSource(graph_path,fingerprint,required_split='train',role_ids=roles)
    part={'schema':resident._D31_REFERENCE_QA_PART,'model':model,'source':resident.binding(graph_path),
        'corpus_manifest':resident.binding(corpus_path),'review':resident.binding(review_path),
        'geometry_audit':resident.binding(audit_path),'conversion_receipt':resident.binding(conversion_path),
        'render_manifest':resident.binding(render_manifest),'render_policy_binding':resident.binding(render_policy),
        'render_source':resident.binding(raw_path),'qa_source_manifest':resident.binding(source_manifest)}
    if mutation=='held':
        doc=json.loads(corpus_path.read_text())
        for member in doc['files']:
            if member['path'].endswith('admission-review.json'):
                member.update(resident.binding(review_path),bytes=review_path.stat().st_size)
        write_json(corpus_path,doc);part['corpus_manifest']=resident.binding(corpus_path)
    return source,part


def test_d31_reference_qa_geometry_part_binds_rendered_rows_and_actual_consumer_geometry(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source,part=d31_reference_qa_part(tmp_path)
    output=tmp_path/'qa-whole-delta.json'
    proof=resident.prepare_whole_delta_geometry(source,[part],output,package=Path(resident.__file__).parents[1])
    report=json.loads(output.read_text())
    assert report['new_train_rows']==2
    assert report['maxima']['forward_tokens']['forward_tokens']==3
    assert report['maxima']['kv_slots']['kv_slots']==3
    assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={
        'forward_tokens':3,'kv_slots':3}
    assert report['provenance'][-1]==part['qa_source_manifest']


@pytest.mark.parametrize('mutation',['held','render-source','graph-lineage','qa-input-manifest','target-count'])
def test_d31_reference_qa_geometry_rejects_changed_hold_or_row_lineage(tmp_path,monkeypatch,mutation):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source,part=d31_reference_qa_part(tmp_path,mutation=mutation if mutation in ('held','qa-input-manifest') else None)
    if mutation=='render-source':
        raw=Path(part['render_source']['path']);rows=raw.read_text().splitlines();row=json.loads(rows[0]);row['split']='test'
        rows[0]=json.dumps(row,separators=(',',':'));raw.write_text('\n'.join(rows)+'\n')
        part['render_source']=resident.binding(raw)
        part['render_source']['bytes']=raw.stat().st_size
    elif mutation=='graph-lineage':
        graph_file=Path(part['source']['path']);rows=graph_file.read_text().splitlines();row=json.loads(rows[0])
        row['conversion']['source_row_id']='different-source-row'
        rows[0]=json.dumps(row,separators=(',',':'));graph_file.write_text('\n'.join(rows)+'\n')
        # Rebind every receipt so this reaches semantic source-row validation.
        conversion=Path(part['conversion_receipt']['path']);conversion_doc=json.loads(conversion.read_text())
        for output in conversion_doc['outputs']:
            if output['path'].endswith('/train/native-graphs.jsonl.zst'):
                output.update(sha256=resident.sha(graph_file),bytes=graph_file.stat().st_size)
        write_json(conversion,conversion_doc)
        review=Path(part['review']['path']);review_doc=json.loads(review.read_text())
        model_review=review_doc['models'][part['model']]
        model_review['graph_output']=resident.binding(graph_file)|{'bytes':graph_file.stat().st_size,'rows':2}
        model_review['receipt']=resident.binding(conversion)|{'bytes':conversion.stat().st_size}
        write_json(review,review_doc)
        audit=Path(part['geometry_audit']['path']);audit_doc=json.loads(audit.read_text())
        audit_doc['models'][part['model']]['sha256']=resident.sha(graph_file)
        write_json(audit,audit_doc)
        review_doc['geometry_audit']=resident.binding(audit)|{'bytes':audit.stat().st_size}
        write_json(review,review_doc)
        part['source']=resident.binding(graph_file)|{'bytes':graph_file.stat().st_size}
        part['conversion_receipt']=resident.binding(conversion)|{'bytes':conversion.stat().st_size}
        part['review']=resident.binding(review)|{'bytes':review.stat().st_size}
        part['geometry_audit']=resident.binding(audit)|{'bytes':audit.stat().st_size}
        corpus=Path(part['corpus_manifest']['path']);doc=json.loads(corpus.read_text())
        changed={
            'admission-review.json':review,'geometry-audit.json':audit,
            f"{part['model']}/receipt.json":conversion,
            f"{part['model']}/train/native-graphs.jsonl.zst":graph_file}
        for member in doc['files']:
            relative=member['path'].removeprefix('keep-whole-graphs-v1/')
            if relative in changed:
                path=changed[relative]
                member.update(sha256=resident.sha(path),bytes=path.stat().st_size)
        write_json(corpus,doc);part['corpus_manifest']=resident.binding(corpus)
    elif mutation=='target-count':
        conversion=Path(part['conversion_receipt']['path']);conversion_doc=json.loads(conversion.read_text())
        conversion_doc['whole_trajectory_supervision']['train']['assistant_target_tokens']+=1
        write_json(conversion,conversion_doc)
        review=Path(part['review']['path']);review_doc=json.loads(review.read_text())
        model_review=review_doc['models'][part['model']]
        model_review['whole_trajectory_supervision']['assistant_target_tokens']+=1
        model_review['receipt']=resident.binding(conversion)|{'bytes':conversion.stat().st_size}
        write_json(review,review_doc)
        part['conversion_receipt']=resident.binding(conversion)|{'bytes':conversion.stat().st_size}
        part['review']=resident.binding(review)|{'bytes':review.stat().st_size}
        corpus=Path(part['corpus_manifest']['path']);doc=json.loads(corpus.read_text())
        for member in doc['files']:
            if member['path'].endswith('/receipt.json'):
                member.update(sha256=resident.sha(conversion),bytes=conversion.stat().st_size)
            elif member['path'].endswith('admission-review.json'):
                member.update(sha256=resident.sha(review),bytes=review.stat().st_size)
        write_json(corpus,doc);part['corpus_manifest']=resident.binding(corpus)
    output=tmp_path/'rejected.json'
    expected_error = {
        'graph-lineage': 'D31 QA row identity, source hash, split, or group closure differs',
        'target-count': 'D31 QA complete source/target accounting differs',
    }.get(mutation)
    if expected_error is None:
        error = pytest.raises(ValueError)
    else:
        error = pytest.raises(ValueError, match=expected_error)
    with error:
        resident.prepare_whole_delta_geometry(source,[part],output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


def test_delta_reuses_authenticated_complete_geometry_and_retains_full_live_validation(tmp_path,monkeypatch):
    paths,directory,manager,recipe,delta,_=delta_fixture(tmp_path,monkeypatch)
    evidence=whole_delta_geometry(tmp_path,delta,manager)
    old_train=manager.train
    def no_full_compile(*args):raise AssertionError('completed delta evidence must avoid a full geometry pass')
    monkeypatch.setattr(resident,'new_geometry',no_full_compile)
    request=resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
        delta_sources={'train':resident.binding(delta)},geometry_evidence={'train':resident.binding(evidence)})
    payload=json.loads(resident.bound(request).read_text())['payload']
    assert payload['geometry_evidence']=={'train':resident.binding(evidence)}
    assert payload['geometry']['kv_slots']>0 and payload['delta_validation']['prior_rows']==2
    assert 'entries_sha256'not in payload and not resident.current(directory)
    assert manager.poll(9)is not None
    assert manager.train.children[0]is old_train and manager.train.ordered_ids()==['a','b','c']
    assert 'exposure_counts'not in resident.current(directory)


@pytest.mark.parametrize('change',['metadata','tokenizer','roles','count','source','index','provenance','maximum','legacy-null-prior','held-scope'])
def test_delta_geometry_reuse_rejects_unbound_or_incomplete_evidence(tmp_path,monkeypatch,change):
    _,directory,manager,recipe,delta,_=delta_fixture(tmp_path,monkeypatch)
    evidence=whole_delta_geometry(tmp_path,delta,manager);value=json.loads(evidence.read_text())
    if change=='metadata':value['entries_sha256']='0'*64
    elif change=='tokenizer':value['tokenizer_fingerprint']='another-tokenizer'
    elif change=='roles':value['role_ids']=[1,2,3,4]
    elif change=='count':value['new_train_rows']+=1
    elif change=='source':value['source']['sha256']='0'*64
    elif change=='index':value['indexes'][0]['metadata_sha256']='0'*64
    elif change=='provenance':value['provenance'][0]['sha256']='0'*64
    elif change=='maximum':value['maxima']['kv_slots']['kv_slots']+=1
    elif change=='legacy-null-prior':value['schema']='legacy-tiered-source-geometry'
    write_json(evidence,value)
    scope='test'if change=='held-scope'else'train'
    with pytest.raises(ValueError):
        resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=(),
            delta_sources={'train':resident.binding(delta)},geometry_evidence={scope:resident.binding(evidence)})
    assert not(directory/'pending.json').exists()and resident.current(directory)is None


def ordinary_coverage_part(tmp_path,kind):
    rows=[graph('small'),graph('large')]
    rows[1]['nodes'][0]['token_ids']=[1,2,3,4]
    rows[1]['nodes'][0]['target_weights']=[0,1,1,1]
    leaf=tmp_path/'fit.jsonl';leaf.write_text(''.join(json.dumps(row)+'\n'for row in rows))
    manifest=tmp_path/'fit-sources.json';write_json(manifest,{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**resident.binding(leaf)}]})
    source=NativeGraphSource(manifest,'tokenizer-sha',required_split='train')
    coverage=tmp_path/'coverage.jsonl'
    covered=[]
    for index,example in enumerate(source.iter_indices(range(len(source)))):
        covered.append({'id':example.id,'model':'fixture','split':'train','partition':'fit','disposition':'fit',
            'partition_row_index':index,'geometry':{'forward_tokens':example.plan.token_count,'kv_slots':len(example.plan.slot_sources)}})
    coverage.write_text(''.join(json.dumps(row)+'\n'for row in covered))
    receipt={'schema':kind,'passed':True,'source_eligibility_unchanged':True,'exact_original_row_bytes':True,
        'files':[{**resident.binding(path),'path':path.name,'bytes':path.stat().st_size}for path in(leaf,manifest,coverage)],
        'coverage':coverage.name,'manifests':{'fixture:train:fit':manifest.name},
        'counts':{'fixture':{'train:fit':{'rows':2}}},'binding':{'limits':{'fixture':{'forward_tokens':10,'kv_slots':10}}}}
    if kind=='natlang.admitted-ordinary-source-capacity-preparation/1':
        receipt.update(fit_sources={'fixture':resident.binding(manifest)},counts={'fixture':{'one-corpus':{'fit':2}}},
            limits={'fixture':{'forward_tokens':10,'kv_slots':10}})
    path=tmp_path/'receipt.json';write_json(path,receipt)
    return source,{'model':'fixture','source':resident.binding(manifest),'coverage':resident.binding(coverage),'receipt':resident.binding(path)}


def source03_bridge_fixture(tmp_path, monkeypatch, *, with_ordinary=False,
                            old_geometry_override=None, old_forward_override=None,
                            invented_current_component=False, boolean_geometry=False,
                            omit_keep_coverage_bytes=False, wrong_final_keep_coverage_sha=False,
                            legacy_fcb=False, source_ordinals=None,
                            old_coverage_mutation=None, wrong_keep_parent=False,
                            malformed_auxiliary_id=False, wrong_pending_parent_digest=False):
    import hashlib
    import pickle
    import subprocess
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
    from natlang_neuralese.common import row_index

    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    tokenizer='tokenizer-sha';roles=[64402,64403,64404,64405]
    base=tmp_path/'base';model=base/'lfm25';model.mkdir(parents=True)
    graphs=[graph('parent-0' if malformed_auxiliary_id else 'parent-0:reasoning-turn:0'),
            graph('parent-1' if malformed_auxiliary_id else 'parent-1:reasoning-turn:1')]
    source_ordinals = source_ordinals or list(range(len(graphs)))
    assert len(source_ordinals) == len(graphs)
    if boolean_geometry:
        graphs[0]['nodes'][0]['token_ids'] = [1]
        graphs[0]['nodes'][0]['target_weights'] = [0]
    raw_graphs=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in graphs).encode()
    bridge_leaf=model/'00-fit.jsonl.zst'
    with bridge_leaf.open('wb') as compressed:
        subprocess.run(['zstd','-q','-c'],input=raw_graphs,stdout=compressed,check=True)
    sidecar=model/'bridge-row-bindings.jsonl'
    old=tmp_path/'old';oldfit=old/'old-fit';(oldfit/'row-index').mkdir(parents=True)
    old_leaf=oldfit/'00-fit.jsonl.zst'
    with old_leaf.open('wb') as compressed:
        subprocess.run(['zstd','-q','-c'],input=raw_graphs,stdout=compressed,check=True)
    bridge_source=NativeGraphSource(bridge_leaf,tokenizer,required_split='train',role_ids=roles)
    old_source=NativeGraphSource(old_leaf,tokenizer,required_split='train',role_ids=roles)
    old_coverage=oldfit/'coverage.jsonl'
    side_rows=[];old_cov_rows=[];bridge_cov_rows=[]
    for i,(row,entry) in enumerate(zip(graphs,bridge_source.entries,strict=True)):
        line=(json.dumps(row,separators=(',',':'))+'\n').encode()
        raw_sha=hashlib.sha256(line).hexdigest();graph_sha=canonical_json_sha256_hex(row)
        parent_id='parent-'+str(i)
        keep_parent_id=('wrong-parent-'+str(i) if wrong_keep_parent else parent_id) + ':history-reasoning:keep'
        side={'id':entry['id'],'binding':entry['binding'],'keep_parent_id':keep_parent_id,
            'keep_partition':'remaining-long','keep_source_graph_sha256':'1'*64,
            'keep_source_row_bytes_sha256':'2'*64,'old_fit_source_row':source_ordinals[i],
            'old_source_graph_sha256':graph_sha,'source_row_bytes_sha256':raw_sha,
            'source_sha256':old_source.file_sha256}
        side_rows.append(side)
        row_geometry = {'forward_tokens':entry['forward_tokens'],'kv_slots':entry['forward_tokens']}
        if boolean_geometry and i == 0:
            row_geometry = {'forward_tokens':True,'kv_slots':True}
        if old_geometry_override is not None:
            row_geometry = old_geometry_override
        old_cov_rows.append({'model':'lfm25','split':'train','disposition':'fit','source_row':source_ordinals[i],'id':entry['id'],
            'source_graph_sha256':graph_sha,'source_row_bytes_sha256':raw_sha,
            'geometry':row_geometry,'represented_by':{'fit':True}})
        bridge_cov_rows.append({'schema':'natlang.d32-source03-aux-bridge-geometry-row/1','bridge_row_index':i,
            'id':entry['id'],'binding':entry['binding'],'source_graph_sha256':graph_sha,
            'source_row_bytes_sha256':raw_sha,'geometry':row_geometry,
            'old_fit_coverage_row':{'source_sha256':old_source.file_sha256,
                'coverage_sha256':'','old_source_row_index':source_ordinals[i]},'old_row_index_binding':entry['binding']})
    if source_ordinals != list(range(len(graphs))):
        for original_row in sorted(set(range(max(source_ordinals) + 1)) - set(source_ordinals)):
            old_cov_rows.append({'model':'lfm25','split':'train','disposition':'remaining-long',
                'source_row':original_row,'id':'pending-original-row-'+str(original_row),
                'source_graph_sha256':'3'*64,'source_row_bytes_sha256':'4'*64,
                'geometry':{'forward_tokens':100,'kv_slots':100},'represented_by':{'fit':False}})
    if old_coverage_mutation == 'wrong-id':
        old_cov_rows[0]['id'] = 'different-old-graph'
    elif old_coverage_mutation == 'wrong-sha':
        old_cov_rows[0]['source_graph_sha256'] = 'f' * 64
    elif old_coverage_mutation == 'duplicate-coordinate':
        old_cov_rows.append(copy.deepcopy(old_cov_rows[0]))
    sidecar.write_text(''.join(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n' for row in side_rows))
    old_coverage.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in old_cov_rows))
    old_cov_sha=sha256_file_hex(old_coverage)
    for row in bridge_cov_rows:row['old_fit_coverage_row']['coverage_sha256']=old_cov_sha
    coverage=tmp_path/'geometry-supplement'/'coverage.jsonl';coverage.parent.mkdir()
    coverage.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in bridge_cov_rows))

    package=Path(resident.__file__).parents[1]
    names=('data/native_sources.py','data/native_graph_schema.py','model/native_graph_plan.py',
        'data/native_pair_schema.py','common/row_index.py','common/jsonio.py','common/hashing.py')
    current_components={name:sha256_file_hex(package/name) for name in names}
    old_components=dict(current_components);old_components['common/hashing.py']='a'*64
    policy_old=canonical_json_sha256_hex(old_components);policy_new=canonical_json_sha256_hex(current_components)
    transition={'old_policy_sha256':policy_old,'new_policy_sha256':policy_new,
        'changed_components':['common/hashing.py'],'old_components':old_components,
        'new_components':current_components,'unchanged_parser_components':[n for n in names if n!='common/hashing.py']}
    if legacy_fcb:
        old_components=dict(current_components)
        policy_old=policy_new=canonical_json_sha256_hex(current_components)
        transition=None
    def key_for(source,policy):
        return {'source_sha256':source.file_sha256,'tokenizer':tokenizer,'split':'train',
            'policy_sha256':policy,'kind':'native-graphs','pair_model':None,'role_ids':roles}
    if invented_current_component:
        current_components['data/native_sources.py'] = 'f' * 64
        policy_new = canonical_json_sha256_hex(current_components)
    old_key=key_for(old_source,policy_old);bridge_key=key_for(bridge_source,policy_new)
    def index_file(path,key,entries):
        value={'schema':'natlang.row-index/1','key':key,'entries':entries,
            'metadata_sha256':canonical_json_sha256_hex(entries)}
        with path.open('wb') as stream:pickle.dump(value,stream)
    old_entries=copy.deepcopy(old_source.entries)
    if old_forward_override is not None:
        old_entries[0]['forward_tokens'] = old_forward_override
    old_index=oldfit/'row-index/native-graphs-436a086fe6a15ca3669c145776184b1089fdfc41.pkl';index_file(old_index,old_key,old_entries)
    bridge_index=coverage.parent/'row-index/native-graphs-bridge.pkl';bridge_index.parent.mkdir()
    index_file(bridge_index,bridge_key,bridge_source.entries)

    old_files=[]
    for p,name in ((old_leaf,'old-fit/00-fit.jsonl.zst'),(old_coverage,'old-fit/coverage.jsonl'),
                   (old_index,'old-fit/row-index/native-graphs-436a086fe6a15ca3669c145776184b1089fdfc41.pkl')):
        old_files.append({'path':name,'sha256':sha256_file_hex(p),'bytes':p.stat().st_size})
    def artifact(path):return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    keep_coverage=tmp_path/'keep-coverage.jsonl';keep_coverage.write_text('{}\n')
    join={'schema':'natlang.d32-source03-aux-bridge-old-geometry-join/1','rows':2,
        'mapping_sha256':'b'*64,'matched_old_fit_graph_sha':2,'old_fit_match_errors':[],'keep_parent_errors':[],
        'artifacts':{'old_fit_source':artifact(old_leaf),'old_coverage':artifact(old_coverage),
            'old_row_index':artifact(old_index),'bridge_source':artifact(bridge_leaf),
            'row_binding_sidecar':artifact(sidecar),'keep_coverage':artifact(keep_coverage)},
        'training_admission':False,'training_exposure':0}
    join_path=old/'join-report.json';write_json(join_path,join)
    old_files.append({'path':'join-report.json','sha256':sha256_file_hex(join_path),'bytes':join_path.stat().st_size})
    provenance={'schema':'natlang.d32-source03-aux-bridge-old-join-provenance/1','files':old_files,
        'geometry_forward_tokens_sum':6,'geometry_kv_slots_sum':6,'join_mapping_sha256':join['mapping_sha256'],
        'matched_old_fit_graph_sha':2,'rows':2,'source_row_index_key':bridge_key,
        'training_admission':False,'training_exposure':0,'unique_keep_parents':2}
    if transition is not None:
        provenance['parser_policy_transition']=transition
    provenance_path=old/'provenance.json';write_json(provenance_path,provenance)
    def ref(path):return {'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size}
    fit_sources=model/'fit-sources.json';write_json(fit_sources,{'schema':'natlang.native-graph-sources/1',
        'sources':[{'kind':'native-graphs',**resident.binding(bridge_leaf)}]})
    keep_coverage_binding=artifact(keep_coverage)
    if omit_keep_coverage_bytes:keep_coverage_binding.pop('bytes')
    if wrong_final_keep_coverage_sha:keep_coverage_binding['sha256']='f'*64
    evidence=model/'bridge-evidence-final.json';write_json(evidence,{'schema':'natlang.d32-history-keep-aux-bridge-final/1',
        'source03':{'rows':2,'unique_keep_parents':2,'training_admission':False,'training_exposure':0,
            'old_fit_source':artifact(old_leaf),'old_coverage':artifact(old_coverage),'old_row_index':artifact(old_index),
            'keep_coverage':keep_coverage_binding,
            'bridge_source':{**artifact(bridge_leaf),'raw_selected_rows_sha256':hashlib.sha256(raw_graphs).hexdigest()},
            'row_binding_sidecar':{**artifact(sidecar),'canonical_rows_sha256':hashlib.sha256(sidecar.read_bytes()).hexdigest()},
            'keep_pending_parent_ids_sha256':resident.digest(
                ['parent-0','parent-1'] if not wrong_pending_parent_digest else ['parent-0','parent-elsewhere'])}})
    registration=model/'registration-audit.json';write_json(registration,{'schema':'natlang.d32-history-keep-aux-bridge-registration/1',
        'training_admission':False,'training_exposure':0})
    manifest=tmp_path/'corpus-manifest.json'
    files=[]
    for p,rel in ((bridge_leaf,'lfm25/00-fit.jsonl.zst'),(sidecar,'lfm25/bridge-row-bindings.jsonl'),
                  (evidence,'lfm25/bridge-evidence-final.json'),(fit_sources,'lfm25/fit-sources.json'),
                  (registration,'lfm25/registration-audit.json')):
        files.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha256_file_hex(p)})
    write_json(manifest,{'schema':'natlang.corpus-snapshot/1','id':'d32-source03-history-keep-aux-bridge-v1',
        'training_admission':False,'files':files})
    supplement={'schema':'natlang.d32-source03-aux-bridge-geometry-supplement/1','source':ref(bridge_leaf),
        'source_row_binding_sidecar':ref(sidecar),'old_fit_source':ref(old_leaf),'old_fit_coverage':ref(old_coverage),
        'old_source_row_index':ref(old_index),'join_report':ref(join_path),'join_mapping_sha256':join['mapping_sha256'],
        'coverage':ref(coverage),'source_row_index':ref(bridge_index),'row_index_key':bridge_key,
        'row_index_entries_sha256':canonical_json_sha256_hex(bridge_source.entries),
        'rows':2,'forward_tokens_sum':6,'kv_slots_sum':6,'unique_keep_parents':2,
        'training_admission':False,'training_exposure':0}
    if transition is not None:
        supplement['parser_policy_transition']=transition
    supplement_path=coverage.parent/'supplement.json';write_json(supplement_path,supplement)
    # The verifier resolves the staged technical package by sibling names, while
    # each embedded reference retains its original source-host path and digest.
    parts=[{'schema':'natlang.d32-source03-aux-bridge-geometry-part/1','model':'lfm25',
        'source':resident.binding(fit_sources),'corpus_manifest':resident.binding(manifest),
        'supplement':resident.binding(supplement_path),'old_join_provenance':resident.binding(provenance_path),
        'limits':{'forward_tokens':10,'kv_slots':10}}]
    main=bridge_leaf
    if with_ordinary:
        ordinary_row=graph('ordinary');ordinary_row['nodes'][0]['token_ids']=[1,2,3,4]
        ordinary_row['nodes'][0]['target_weights']=[0,1,1,1]
        ordinary=tmp_path/'ordinary.jsonl';ordinary.write_text(json.dumps(ordinary_row)+'\n')
        main=tmp_path/'combined.jsonl';main.write_bytes(raw_graphs+ordinary.read_bytes())
        ordinary_source=NativeGraphSource(ordinary,tokenizer,required_split='train',role_ids=roles)
        combined=NativeGraphSource(main,tokenizer,required_split='train',role_ids=roles)
        coverage2=tmp_path/'ordinary-coverage.jsonl'
        coverage2.write_text(json.dumps({'id':ordinary_source.entries[0]['id'],'model':'fixture','split':'train',
            'partition':'fit','disposition':'fit','partition_row_index':0,
            'geometry':{'forward_tokens':4,'kv_slots':4}})+'\n')
        receipt={'schema':'natlang.acquired-native-capacity-partition/1','source_eligibility_unchanged':True,
            'exact_original_row_bytes':True,'files':[{'path':'ordinary.jsonl','sha256':sha256_file_hex(ordinary),'bytes':ordinary.stat().st_size},
                {'path':'ordinary-coverage.jsonl','sha256':sha256_file_hex(coverage2),'bytes':coverage2.stat().st_size}],
            'coverage':'ordinary-coverage.jsonl','manifests':{'fixture:train:fit':'ordinary.jsonl'},
            'counts':{'fixture':{'train:fit':{'rows':1}}},'binding':{'limits':{'fixture':{'forward_tokens':10,'kv_slots':10}}}}
        receipt_path=tmp_path/'ordinary-receipt.json';write_json(receipt_path,receipt)
        parts.append({'model':'fixture','source':resident.binding(ordinary),'coverage':resident.binding(coverage2),
            'receipt':resident.binding(receipt_path)})
        return combined,parts
    manifest_source=NativeGraphSource(fit_sources,tokenizer,required_split='train',role_ids=roles)
    return manifest_source,parts


@pytest.mark.parametrize('kind',['natlang.acquired-native-capacity-partition/1','natlang.admitted-ordinary-source-capacity-preparation/1'])
def test_whole_delta_producer_authenticates_all_standard_coverage_and_derives_actual_maxima(tmp_path,monkeypatch,kind):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source,part=ordinary_coverage_part(tmp_path,kind)
    output=tmp_path/'geometry.json';proof=resident.prepare_whole_delta_geometry(source,[part],output,
        package=Path(resident.__file__).parents[1])
    report=json.loads(output.read_text())
    assert report['maxima']['kv_slots']['id']=='large'and report['maxima']['kv_slots']['kv_slots']==4
    assert report['new_train_rows']==2 and report['provenance']==[part['receipt'],part['coverage']]
    assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={'forward_tokens':4,'kv_slots':4}


@pytest.mark.parametrize('bad',['unbound-coverage','nonmaximum-kv','over-limit','wrong-id','missing-row','duplicate-row','wrong-disposition','wrong-fit'])
def test_whole_delta_producer_rejects_coverage_or_geometry_changes_before_publication(tmp_path,monkeypatch,bad):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source,part=ordinary_coverage_part(tmp_path,'natlang.acquired-native-capacity-partition/1')
    coverage=Path(part['coverage']['path']);rows=[json.loads(line)for line in coverage.read_text().splitlines()]
    receipt_path=Path(part['receipt']['path']);receipt=json.loads(receipt_path.read_text())
    if bad=='nonmaximum-kv':rows[0]['geometry']['kv_slots']=2
    elif bad=='over-limit':receipt['binding']['limits']['fixture']['kv_slots']=3
    elif bad=='wrong-id':rows[0]['id']='another-row'
    elif bad=='missing-row':rows.pop()
    elif bad=='duplicate-row':rows[1]['id']=rows[0]['id']
    elif bad=='wrong-disposition':rows[0]['partition']='remaining-long'
    elif bad=='wrong-fit':receipt['manifests']['fixture:train:fit']='another-manifest.json'
    else:rows[0]['geometry']['kv_slots']=1
    coverage.write_text(''.join(json.dumps(row)+'\n'for row in rows));part['coverage']=resident.binding(coverage)
    if bad!='unbound-coverage':
        item=next(item for item in receipt['files']if item['path']==coverage.name)
        item.update(sha256=part['coverage']['sha256'],bytes=coverage.stat().st_size)
    write_json(receipt_path,receipt);part['receipt']=resident.binding(receipt_path)
    output=tmp_path/'not-published.json'
    with pytest.raises(ValueError):resident.prepare_whole_delta_geometry(source,[part],output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


def test_delta_geometry_producer_does_not_publish_completed_report_when_maximum_read_fails(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source,part=ordinary_coverage_part(tmp_path,'natlang.acquired-native-capacity-partition/1')
    def unavailable(*args):raise OSError('actual maximum-row source read unavailable')
    monkeypatch.setattr(source,'iter_indices',unavailable)
    output=tmp_path/'geometry.json'
    with pytest.raises(OSError,match='maximum-row'):
        resident.prepare_whole_delta_geometry(source,[part],output,package=Path(resident.__file__).parents[1])
    assert not output.exists()and not list(tmp_path.glob('.geometry.json.preparing-*'))


def absolute_teacher_part(tmp_path,monkeypatch):
    import shutil
    original=tmp_path/'original';original.mkdir();monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    _,part=ordinary_coverage_part(original,'natlang.admitted-ordinary-source-capacity-preparation/1')
    # The real producer stores per-model fit manifests below its original root.
    model=original/'fixture';model.mkdir();manifest=original/'fit-sources.json'
    doc=json.loads(manifest.read_text());doc['sources'][0]['path']='../fit.jsonl';write_json(model/'fit-sources.json',doc)
    receipt=json.loads((original/'receipt.json').read_text())
    receipt['fit_sources']['fixture']=resident.binding(model/'fit-sources.json')
    receipt['files']=[{**row,'path':str(original/row['path'])}for row in receipt['files']if row['path']!='fit-sources.json']
    fit=model/'fit-sources.json';receipt['files'].append({**resident.binding(fit),'bytes':fit.stat().st_size})
    write_json(original/'receipt.json',receipt)
    staged=tmp_path/'staged';shutil.copytree(original,staged);shutil.rmtree(original)
    source=NativeGraphSource(staged/'fixture/fit-sources.json','tokenizer-sha',required_split='train')
    part={'model':'fixture','source':resident.binding(staged/'fixture/fit-sources.json'),
        'coverage':resident.binding(staged/'coverage.jsonl'),'receipt':resident.binding(staged/'receipt.json')}
    return source,part,receipt


def test_whole_delta_teacher_absolute_original_namespace_is_authenticated_after_relocation(tmp_path,monkeypatch):
    source,part,_=absolute_teacher_part(tmp_path,monkeypatch)
    proof=resident.prepare_whole_delta_geometry(source,[part],tmp_path/'geometry.json',package=Path(resident.__file__).parents[1])
    assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={'forward_tokens':4,'kv_slots':4}


@pytest.mark.parametrize('mutation',['root','traversal','duplicate','mixed','outside','sha','bytes','acquired-absolute'])
def test_teacher_absolute_namespace_drift_cannot_publish_geometry(tmp_path,monkeypatch,mutation):
    source,part,receipt=absolute_teacher_part(tmp_path,monkeypatch)
    if mutation=='root':receipt['fit_sources']['fixture']['path']=receipt['fit_sources']['fixture']['path'].replace('/original/','/different/')
    elif mutation=='traversal':receipt['files'][0]['path']=receipt['files'][0]['path'].replace('/original/','/original/../original/')
    elif mutation=='duplicate':receipt['files'].append(copy.deepcopy(receipt['files'][0]))
    elif mutation=='mixed':receipt['files'][0]['path']='coverage.jsonl'
    elif mutation=='outside':receipt['files'][0]['path']='/another-root/coverage.jsonl'
    elif mutation=='sha':next(row for row in receipt['files']if row['path'].endswith('/coverage.jsonl'))['sha256']='f'*64
    elif mutation=='bytes':next(row for row in receipt['files']if row['path'].endswith('/coverage.jsonl'))['bytes']+=1
    else:receipt['schema']='natlang.acquired-native-capacity-partition/1'
    p=Path(part['receipt']['path']);write_json(p,receipt);part['receipt']=resident.binding(p);out=tmp_path/'geometry.json'
    with pytest.raises(ValueError):resident.prepare_whole_delta_geometry(source,[part],out,package=Path(resident.__file__).parents[1])
    assert not out.exists()


def test_source03_bridge_geometry_is_verified_and_can_be_bridge_only_or_ordered_prefix(tmp_path,monkeypatch):
    for with_ordinary in (False, True):
        root=tmp_path/str(with_ordinary);root.mkdir()
        source,parts=source03_bridge_fixture(root,monkeypatch,with_ordinary=with_ordinary)
        output=root/'geometry.json'
        proof=resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
        report=json.loads(output.read_text())
        assert report['new_train_rows']==len(source)
        assert report['maxima']['kv_slots']['kv_slots']==(4 if with_ordinary else 3)
        assert report['maxima']['kv_slots']['index']==(2 if with_ordinary else 0)
        assert report['provenance'][:4]==[
            parts[0]['source'],parts[0]['corpus_manifest'],parts[0]['supplement'],parts[0]['old_join_provenance']]
        assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={
            'forward_tokens':4 if with_ordinary else 3,'kv_slots':4 if with_ordinary else 3}


def test_source03_bridge_v1_requires_exact_current_fcb_policy(tmp_path,monkeypatch):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,legacy_fcb=True)
    output=tmp_path/'legacy-fcb-geometry.json'
    proof=resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert json.loads(output.read_text())['new_train_rows']==2
    assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={
        'forward_tokens':3,'kv_slots':3}


def test_source03_bridge_maps_historical_ordinals_to_compact_fit_positions(tmp_path,monkeypatch):
    # The prepartition source had pending gaps: original source rows 7 and 11
    # are compact FIT positions 0 and 1 in the old admitted FIT leaf.
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,source_ordinals=[7,11])
    output=tmp_path/'mapped-geometry.json'
    proof=resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert json.loads(output.read_text())['new_train_rows']==2
    assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={
        'forward_tokens':3,'kv_slots':3}


@pytest.mark.parametrize('mutation',['wrong-id','wrong-sha','duplicate-coordinate'])
def test_source03_bridge_rejects_ambiguous_historical_to_compact_fit_mapping(tmp_path,monkeypatch,mutation):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,source_ordinals=[7,11],
        old_coverage_mutation=mutation)
    output=tmp_path/'rejected-mapping.json'
    with pytest.raises(ValueError,match='historical source coordinate'):
        resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


def test_source03_bridge_rejects_duplicate_historical_coordinate_across_distinct_fit_rows(tmp_path,monkeypatch):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,source_ordinals=[7,7])
    output=tmp_path/'rejected-duplicate-original-coordinate.json'
    with pytest.raises(ValueError,match='historical source coordinate'):
        resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


@pytest.mark.parametrize('mutation',['wrong-keep-parent','malformed-auxiliary-id'])
def test_source03_bridge_binds_keep_parent_to_exact_auxiliary_source_id(tmp_path,monkeypatch,mutation):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,
        wrong_keep_parent=mutation=='wrong-keep-parent',
        malformed_auxiliary_id=mutation=='malformed-auxiliary-id')
    output=tmp_path/'rejected-parent-binding.json'
    with pytest.raises(ValueError,match='(KEEP parent ID|reasoning-turn parent)'):
        resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


def test_source03_bridge_rejects_wrong_pending_parent_digest_after_valid_parent_joins(tmp_path,monkeypatch):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,wrong_pending_parent_digest=True)
    output=tmp_path/'rejected-pending-parent-digest.json'
    with pytest.raises(ValueError,match='final unsuffixed pending-parent digest'):
        resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


def test_source03_bridge_v1_rejects_a_different_parser_package(tmp_path,monkeypatch):
    import shutil
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,legacy_fcb=True)
    actual=Path(resident.__file__).parents[1]
    different=tmp_path/'different-parser-package';shutil.copytree(actual,different)
    (different/'common/hashing.py').write_bytes((different/'common/hashing.py').read_bytes()+b'\n# changed\n')
    output=tmp_path/'rejected-fcb-geometry.json'
    with pytest.raises(ValueError,match='old row index contract'):
        resident.prepare_whole_delta_geometry(source,parts,output,package=different)
    assert not output.exists()


def test_source03_bridge_accepts_registered_keep_coverage_sha_without_redundant_bytes(tmp_path,monkeypatch):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,omit_keep_coverage_bytes=True)
    output=tmp_path/'geometry.json'
    proof=resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert json.loads(output.read_text())['new_train_rows']==2
    assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={
        'forward_tokens':3,'kv_slots':3}


def test_source03_bridge_still_rejects_different_keep_coverage_sha(tmp_path,monkeypatch):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,omit_keep_coverage_bytes=True,
        wrong_final_keep_coverage_sha=True)
    output=tmp_path/'geometry.json'
    with pytest.raises(ValueError,match='current KEEP coverage binding differs'):
        resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


def test_source03_bridge_graph_rows_are_consumed_as_a_single_pass_stream(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from natlang_neuralese.common import jsonio

    source,parts=source03_bridge_fixture(tmp_path,monkeypatch)
    original_open=jsonio.open_jsonl_text
    consumed=[]

    class StreamingRows:
        def __init__(self,rows):
            self.rows=iter(rows)
        def __iter__(self):
            return self
        def __next__(self):
            row=next(self.rows)
            consumed.append(row)
            return row
        def __length_hint__(self):
            raise AssertionError('bridge graph stream must not be materialized')

    @contextmanager
    def open_with_stream_guard(path):
        with original_open(path) as rows:
            if Path(path).name=='00-fit.jsonl.zst' and Path(path).parent.name=='lfm25':
                yield StreamingRows(rows)
            else:
                yield rows

    monkeypatch.setattr(jsonio,'open_jsonl_text',open_with_stream_guard)
    output=tmp_path/'streamed-geometry.json'
    proof=resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert len(consumed)==2
    assert json.loads(output.read_text())['new_train_rows']==2
    assert resident.geometry_from_evidence(source,0,proof,package=Path(resident.__file__).parents[1])=={
        'forward_tokens':3,'kv_slots':3}


@pytest.mark.parametrize('mutation',['interleaved','duplicate','coverage','missing-row','wrong-geometry','boolean-geometry',
                                    'wrong-policy','wrong-binding','old-source','old-geometry',
                                    'old-forward-tokens','invented-current-parser'])
def test_source03_bridge_geometry_rejects_unbound_or_nonprefix_bridge(tmp_path,monkeypatch,mutation):
    source,parts=source03_bridge_fixture(tmp_path,monkeypatch,with_ordinary=True,
        old_geometry_override={'forward_tokens':2,'kv_slots':2} if mutation=='old-geometry' else None,
        old_forward_override=2 if mutation=='old-forward-tokens' else None,
        invented_current_component=mutation=='invented-current-parser',
        boolean_geometry=mutation=='boolean-geometry')
    bridge=parts[0]
    if mutation=='interleaved':parts=[parts[1],bridge]
    elif mutation=='duplicate':parts=[bridge,copy.deepcopy(bridge),parts[1]]
    elif mutation=='coverage':
        path=Path(bridge['supplement']['path']).parent/'coverage.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['geometry']['kv_slots']=2
        path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
    elif mutation in {'missing-row','wrong-geometry'}:
        path=Path(bridge['supplement']['path']);supplement=json.loads(path.read_text())
        coverage=Path(supplement['coverage']['path'])
        rows=[json.loads(line) for line in coverage.read_text().splitlines()]
        if mutation=='missing-row':rows.pop()
        else:rows[0]['geometry']['kv_slots']=4
        coverage.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
        supplement['coverage']=resident.binding(coverage);write_json(path,supplement)
        bridge['supplement']=resident.binding(path)
    elif mutation=='wrong-policy':
        path=Path(bridge['supplement']['path']);value=json.loads(path.read_text())
        value['row_index_key']['policy_sha256']='f'*64;write_json(path,value)
        bridge['supplement']=resident.binding(path)
    elif mutation in {'old-geometry','old-forward-tokens','invented-current-parser','boolean-geometry'}:
        pass  # Fully bound malformed fixtures are constructed at creation time.
    elif mutation=='wrong-binding':
        path=Path(bridge['source']['path']).parent/'bridge-row-bindings.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['binding']='f'*64
        path.write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows))
    else:
        path=Path(bridge['source']['path']).parent/'00-fit.jsonl.zst'
        path.write_bytes(path.read_bytes()+b'changed')
    output=tmp_path/'rejected-geometry.json'
    with pytest.raises(ValueError):
        resident.prepare_whole_delta_geometry(source,parts,output,package=Path(resident.__file__).parents[1])
    assert not output.exists()


def test_diloco_resident_control_stages_publishes_and_adopts_at_update_boundary(tmp_path, monkeypatch):
    import importlib.util
    import hashlib
    path = Path(__file__).resolve().parents[2] / 'scripts/coreweave/diloco_resident_worker.py'
    spec = importlib.util.spec_from_file_location('resident_worker_test', path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    paths, directory, manager, recipe = setup(tmp_path)
    staged, installed = tmp_path / 'staged', tmp_path / 'immutable' / 'object'
    staged.write_bytes(b'content')
    digest = hashlib.sha256(b'content').hexdigest()
    assert helper.install(staged, installed, digest, 7)['sha256'] == digest
    staged.write_bytes(b'changed')
    with pytest.raises(ValueError, match='staged content'):
        helper.install(staged, installed, digest, 7)
    prepared = helper.prepare(directory, recipe, 'sft', 'tokenizer-sha', [])
    assert helper.status(directory)['pending'] is None
    published = helper.publish(directory, prepared['request'])
    assert helper.status(directory)['pending'] == prepared['request']
    old_train = manager.train
    assert manager.poll(12) is not None
    assert manager.train is not old_train and len(manager.train) == 3
    status = helper.status(directory)
    assert status['generation'] == published['generation'] and status['at_global_step'] == 12
    assert 'exposure_counts' not in resident.current(directory)
