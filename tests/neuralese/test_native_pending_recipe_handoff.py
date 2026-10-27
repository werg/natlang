import copy
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
from natlang_neuralese.train.native_pending_sources import declaration, pending_source_dependencies
from natlang_neuralese.train.recipe import (_pending_source_continuation, _data_continuation,
    _priority_only_native_source_continuation, unstarted_stage_input_changes,
    _without_unstarted_bindings, _combined_native_source_continuation)
from test_native_state_sft import graph


def declared(tmp_path, name, ids, *, evidence=(), mutate=None):
    graphs=tmp_path/(name+'.jsonl')
    graphs.write_text(''.join(json.dumps(dict(graph(i),tokenizer_fingerprint='1'*64))+'\n' for i in ids))
    value=declaration(graphs,graphs,tokenizer_fingerprint='1'*64,role_ids=(58,59,60,61),evidence=evidence)
    if mutate:mutate(value)
    path=tmp_path/(name+'.json');path.write_text(json.dumps(value))
    return {'path':str(path),'sha256':sha(path),'binding':'pending'}


def inputs():
    return {'native_graphs':{'path':'train','sha256':'a'*64,'binding':'train'},
        'continued_native_graphs':{'path':'previous','sha256':'b'*64,'binding':'previous'},
        'held_native_graphs':{'path':'held','sha256':'c'*64,'binding':'held'},
        'foundation_native_graphs':{'path':'foundation','sha256':'d'*64,'binding':'foundation'}}


def test_new_pending_obligations_keep_exact_active_sources_and_historical_predecessor(tmp_path):
    old=inputs();new={**old,'pending_native_sources':declared(tmp_path,'new',['long'])}
    assert _pending_source_continuation(old,new)
    assert not _data_continuation(old,new)  # Resident scope is deliberately unchanged.
    assert not _pending_source_continuation(new,new)
    assert not _pending_source_continuation(new,old)


@pytest.mark.parametrize('role',['native_graphs','held_native_graphs','foundation_native_graphs','continued_native_graphs'])
def test_pending_handoff_cannot_change_other_inputs(tmp_path,role):
    old=inputs();new={**copy.deepcopy(old),'pending_native_sources':declared(tmp_path,'new',['long'])}
    new[role]['sha256']='e'*64
    assert not _pending_source_continuation(old,new)


def test_expansion_retains_old_rows_quotas_and_dependency_evidence(tmp_path):
    prior=declared(tmp_path,'prior',['long']);old={**inputs(),'pending_native_sources':prior}
    evidence=pending_source_dependencies(prior['path'])
    added=declared(tmp_path,'added',['long','later'],evidence=evidence)
    # The shared semantic binding is independent of the leaf filename.
    assert _pending_source_continuation(old,{**inputs(),'pending_native_sources':added})
    reduced=declared(tmp_path,'reduced',['later'],evidence=evidence)
    assert not _pending_source_continuation(old,{**inputs(),'pending_native_sources':reduced})
    changed=declared(tmp_path,'changed',['long','later'],evidence=evidence,
        mutate=lambda v:v['sources'][0]['rows'][0].update(required_visits=2))
    assert not _pending_source_continuation(old,{**inputs(),'pending_native_sources':changed})
    missing=declared(tmp_path,'missing',['long','later'])
    assert not _pending_source_continuation(old,{**inputs(),'pending_native_sources':missing})


def test_corrupt_declaration_and_dependency_fail_before_handoff(tmp_path):
    pending=declared(tmp_path,'new',['long']);new={**inputs(),'pending_native_sources':pending}
    path=tmp_path/'new.json';original=path.read_text();path.write_text(original+'\n')
    with pytest.raises(ValueError,match='declaration changed'):_pending_source_continuation(inputs(),new)
    path.write_text(original);(tmp_path/'new.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='immutable binding'):_pending_source_continuation(inputs(),new)


def test_started_recipe_uses_existing_data_continuation_projection(tmp_path):
    old=inputs();pending=declared(tmp_path,'pending',['long']);new={**old,'pending_native_sources':pending}
    def plan(resolved):
        names={k:v['binding'] for k,v in resolved.items()}
        return {'recipe':{'stages':[{'id':'full','kind':'native_state_sft','inputs':names,'parameters':{}}],
                'input_bindings':{v['binding']:{k:v[k] for k in ('path','sha256')} for v in resolved.values()}},
            'stage_inputs':{'full':resolved}}
    before,after=plan(old),plan(new);stage=tmp_path/'full';stage.mkdir();(stage/'checkpoint.pt').write_text('existing stage marker')
    changes=unstarted_stage_input_changes(before,after,tmp_path)
    assert changes['full']['kind']=='data_continuation'
    assert _without_unstarted_bindings(before['recipe'],after['recipe'],changes)[0]==\
        _without_unstarted_bindings(before['recipe'],after['recipe'],changes)[1]


def test_drained_priority_revision_binds_exact_same_sources_and_predecessor(tmp_path):
    leaf=tmp_path/'leaf.jsonl';leaf.write_text('{}\n')
    def reference(name, priority):
        source={'kind':'native-graphs','path':str(leaf),'sha256':sha(leaf),
                'training_priority':priority}
        path=tmp_path/name
        path.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[source]}))
        return {'path':str(path),'sha256':sha(path),'binding':'native'}
    old_train=reference('old.json',0);new_train=reference('new.json',1)
    old={**inputs(),'native_graphs':old_train}
    new={**old,'native_graphs':new_train,'continued_native_graphs':old_train}
    assert _priority_only_native_source_continuation(old,new)
    # Train and foundation share one named binding in production recipes.
    shared_old={**old,'foundation_native_graphs':old_train}
    shared_new={**new,'foundation_native_graphs':new_train}
    assert _priority_only_native_source_continuation(shared_old,shared_new)
    assert not _priority_only_native_source_continuation(shared_old,{**shared_new,'foundation_native_graphs':old_train})
    independent={**old,'foundation_native_graphs':inputs()['held_native_graphs']}
    assert _priority_only_native_source_continuation(independent,{**new,'foundation_native_graphs':independent['foundation_native_graphs']})
    recipe={'stages':[{'id':'full','kind':'native_state_sft','inputs':{
        key:value['binding'] for key,value in old.items()},'parameters':{}}],
        'input_bindings':{value['binding']:{'path':value['path'],'sha256':value['sha256']}
            for value in old.values()}}
    before={'recipe':recipe,'stage_inputs':{'full':old}}
    after_recipe=copy.deepcopy(recipe)
    after_recipe['stages'][0]['inputs']={key:value['binding'] for key,value in new.items()}
    after_recipe['input_bindings'].update({value['binding']:{'path':value['path'],'sha256':value['sha256']}
        for value in new.values()})
    after={'recipe':after_recipe,'stage_inputs':{'full':new}}
    stage=tmp_path/'full';stage.mkdir();(stage/'checkpoint.pt').write_text('started')
    assert unstarted_stage_input_changes(before,after,tmp_path)['full']['kind']=='scheduling_priority_continuation'
    changed=reference('changed.json',1)
    changed_path=Path(changed['path']);value=json.loads(changed_path.read_text())
    altered_leaf=tmp_path/'different-leaf.jsonl';altered_leaf.write_text('{"different":true}\n')
    value['sources'][0].update(path=str(altered_leaf),sha256=sha(altered_leaf))
    changed_path.write_text(json.dumps(value))
    changed={'path':str(changed_path),'sha256':sha(changed_path),'binding':'native'}
    assert not _priority_only_native_source_continuation(old,{**new,'native_graphs':changed})
    lowered=reference('lowered.json',0)
    assert not _priority_only_native_source_continuation(new,{**new,'native_graphs':lowered,
        'continued_native_graphs':new_train})


def test_priority_revision_can_follow_nested_source_manifest_closure(tmp_path):
    leaf=tmp_path/'leaf.jsonl';leaf.write_text('{}\n')
    def write_manifest(name, sources):
        path=tmp_path/name
        path.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':sources}))
        return {'path':str(path),'sha256':sha(path),'binding':name}
    leaf_row={'kind':'native-graphs','path':str(leaf),'sha256':sha(leaf),
              'training_priority':0}
    old_child=write_manifest('old-child.json',[leaf_row])
    new_child=write_manifest('new-child.json',[{**leaf_row,'training_priority':1}])
    old_train=write_manifest('old-train.json',[{'kind':'native-graphs',**{key:old_child[key] for key in ('path','sha256')}}])
    new_train=write_manifest('new-train.json',[{'kind':'native-graphs',**{key:new_child[key] for key in ('path','sha256')}}])
    old={**inputs(),'native_graphs':old_train}
    new={**old,'native_graphs':new_train,'continued_native_graphs':old_train}
    assert _priority_only_native_source_continuation(old,new)
    overridden_train=write_manifest('overridden-train.json',[{'kind':'native-graphs',
        'path':new_child['path'],'sha256':new_child['sha256'],'training_priority':0}])
    ineffective={**old,'native_graphs':overridden_train,'continued_native_graphs':old_train}
    assert not _priority_only_native_source_continuation(old,ineffective)


def test_priority_revision_resolves_authenticated_staged_source_map(tmp_path):
    leaf=tmp_path/'staged.jsonl';leaf.write_text('{}\n')
    original=str(tmp_path/'missing-origin'/'source.jsonl')
    (tmp_path/'.natlang-source-map.json').write_text(json.dumps({
        'schema':'natlang.staged-source-map/1','paths':{original:str(leaf)}}))
    binding={'kind':'native-graphs','path':original,'sha256':sha(leaf)}
    def reference(name, priority):
        path=tmp_path/name
        path.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[
            {**binding,'training_priority':priority}]}))
        return {'path':str(path),'sha256':sha(path),'binding':'train'}
    before=reference('before.json',0);after=reference('after.json',1)
    old={**inputs(),'native_graphs':before,'foundation_native_graphs':before}
    new={**old,'native_graphs':after,'foundation_native_graphs':after,'continued_native_graphs':before}
    assert _priority_only_native_source_continuation(old,new)
    other=tmp_path/'other.jsonl';other.write_text('{}\n')
    (tmp_path/'.natlang-source-map.json').write_text(json.dumps({
        'schema':'natlang.staged-source-map/1','paths':{original:str(other)}}))
    # A changed staged mapping is visible to normal source verification; it
    # does not by itself alter declared priority-only source identities.
    assert _priority_only_native_source_continuation(old,new)


def expanded_active(old, pending):
    return {**old, 'native_graphs': {'path':'expanded','sha256':'e'*64,'binding':'train'},
        'continued_native_graphs':old['native_graphs'],
        'held_native_graphs': {'path':'expanded-held','sha256':'f'*64,'binding':'held'},
        'continued_held_native_graphs':old['held_native_graphs'],
        'pending_native_sources':pending}


def test_combined_started_handoff_preserves_full_pending_and_exact_active_predecessors(tmp_path):
    prior=declared(tmp_path,'prior',['long'])
    old={**inputs(),'pending_native_sources':prior}
    added=declared(tmp_path,'added',['long','later'],evidence=pending_source_dependencies(prior['path']))
    new=expanded_active(old,added)
    assert not _data_continuation(old,new)
    assert not _pending_source_continuation(old,new)
    assert _combined_native_source_continuation(old,new)
    recipe={'stages':[{'id':'full','kind':'native_state_sft','parameters':{}}]}
    before={'recipe':recipe,'stage_inputs':{'full':old}}
    after={'recipe':recipe,'stage_inputs':{'full':new}}
    stage=tmp_path/'full';stage.mkdir();(stage/'checkpoint.pt').write_text('started')
    assert unstarted_stage_input_changes(before,after,tmp_path)['full']['kind']=='data_continuation'


@pytest.mark.parametrize('mutation',['prior_train','prior_held','foundation','unrelated','drop_rows','quota','dependencies'])
def test_combined_handoff_rejects_loss_of_any_existing_invariant(tmp_path,mutation):
    prior=declared(tmp_path,'prior',['long']);old={**inputs(),'pending_native_sources':prior}
    evidence=pending_source_dependencies(prior['path'])
    ids=['later'] if mutation=='drop_rows' else ['long','later']
    added=declared(tmp_path,'added',ids,evidence=() if mutation=='dependencies' else evidence,
        mutate=(lambda v:v['sources'][0]['rows'][0].update(required_visits=2)) if mutation=='quota' else None)
    new=expanded_active(old,added)
    if mutation=='prior_train':new['continued_native_graphs']={**old['native_graphs'],'sha256':'0'*64}
    if mutation=='prior_held':new['continued_held_native_graphs']={**old['held_native_graphs'],'sha256':'0'*64}
    if mutation=='foundation':new['foundation_native_graphs']={**old['foundation_native_graphs'],'sha256':'0'*64}
    if mutation=='unrelated':new['arbitrary_new_role']={'path':'x','sha256':'0'*64,'binding':'x'}
    assert not _combined_native_source_continuation(old,new)


def test_combined_handoff_can_add_first_pending_but_checks_its_immutable_bytes(tmp_path):
    old=inputs();pending=declared(tmp_path,'new',['long']);new=expanded_active(old,pending)
    assert _combined_native_source_continuation(old,new)
    (tmp_path/'new.json').write_text('{}')
    with pytest.raises(ValueError,match='declaration changed'):_combined_native_source_continuation(old,new)
