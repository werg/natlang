"""Completed no-credit retry bridges the exact old runtime recipe into adoption."""
import copy
from pathlib import Path
import pytest
from natlang_neuralese.rl import native_runtime_handoff as runtime
from natlang_neuralese.rl import native_failed_supervised_retry as retry
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.common.jsonio import write_json_durable


def setup(tmp_path, monkeypatch):
    old=tmp_path/'old.json';old.write_text('{}')
    new=tmp_path/'new.json';new.write_text('{"retry":true}')
    authority={'path':'/exact/runtime-proof.json','sha256':'a'*64}
    predecessor=tmp_path/'config.json';write_json_durable(predecessor,{'production_recipe':str(old)})
    intent=tmp_path/'intent.json';write_json_durable(intent,{'snapshots':{'config':binding(predecessor)}})
    receipt=tmp_path/'receipt.json';value={'intent':binding(intent),'stage_id':'full','checkpoint':{'path':'/exact/model.pt','sha256':'b'*64}};write_json_durable(receipt,value)
    reference=binding(receipt)
    declaration={'runtime_handoff':authority,'failed_supervised_retry':reference}
    marker={'receipt':reference,'stage_id':'full','checkpoint':value['checkpoint'],'training_credit':0,'readiness_granted':False}
    cursor={'failed_supervised_retry_history':[marker]}
    release={'prior_runtime_handoff':authority,'recipe_lineage_sha256':'lineage','recipe_sha256':binding(new)['sha256']}
    calls=[]
    def verify(ref,decl):
        calls.append((ref,decl));assert ref==reference;assert decl is declaration
        return release
    monkeypatch.setattr(retry,'verify_release',verify)
    adopted={'payload':{'prior_recipe':binding(new)},'history':[]}
    return adopted,{'recipe':binding(old)},{'lineage':'lineage'},declaration,cursor,release,calls


def test_consumed_retry_is_exact_bridge_and_does_not_mutate_state(tmp_path,monkeypatch):
    args=setup(tmp_path,monkeypatch);before=copy.deepcopy(args[:5])
    assert runtime._resident_recipe_ancestry(*args[:5])
    assert args[:5]==before and len(args[-1])==1


@pytest.mark.parametrize('broken',['unconsumed','lineage','authority','new_recipe','old_recipe','changed_intent'])
def test_retry_cannot_authorize_unrelated_history(tmp_path,monkeypatch,broken):
    adopted,transition,proof,declaration,cursor,release,calls=setup(tmp_path,monkeypatch)
    if broken=='unconsumed':cursor['failed_supervised_retry_history']=[]
    elif broken=='lineage':release['recipe_lineage_sha256']='wrong'
    elif broken=='authority':release['prior_runtime_handoff']={'path':'/other','sha256':'c'*64}
    elif broken=='new_recipe':release['recipe_sha256']='d'*64
    elif broken=='old_recipe':
        other=tmp_path/'other.json';other.write_text('{"wrong":true}');transition['recipe']=binding(other)
    else:
        (tmp_path/'intent.json').write_text('{}')
        with pytest.raises(ValueError):runtime._resident_recipe_ancestry(adopted,transition,proof,declaration,cursor)
        return
    assert runtime._resident_recipe_ancestry(adopted,transition,proof,declaration,cursor) is False


def test_direct_ancestry_does_not_invoke_retry_verifier(tmp_path,monkeypatch):
    args=setup(tmp_path,monkeypatch);args[0]['history']=[{'recipe':args[1]['recipe']}]
    assert runtime._resident_recipe_ancestry(*args[:5]);assert not args[-1]


def test_authenticated_current_adoption_keeps_new_resident_namespace_and_plan(tmp_path):
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.recipe import recipe_identity_sha256,resolve_recipe_data
    old=tmp_path/'old'/'recipe.json';old.parent.mkdir()
    new=tmp_path/'current'/'recipe.json';new.parent.mkdir()
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    write_json_durable(train,{'source':'train'});write_json_durable(held,{'source':'held'})
    for path,name in ((old,'drained'),(new,'retry-adopted')):
        write_json_durable(path,{'schema':'natlang.neuralese-training-recipe/1','id':name,
            'identity_qualification':'native_state_runtime_qualified',
            'input_bindings':{'train':binding(train),'held':binding(held)},
            'stages':[{'id':'full','kind':'native_state_sft','requires':[],
                'inputs':{'native_graphs':'train','held_native_graphs':'held'},'parameters':{'epochs':1}}]})
    directory=tmp_path/'runtime-root'/'resident-data';directory.mkdir(parents=True)
    current_file=directory/'current.json';write_json_durable(current_file,{'resident':'current'})
    code={'native_runtime.py':'a'*64}
    plan={'recipe':resolve_recipe_data(new),'recipe_sha256':'lineage-current','code':code,
        'stage_inputs':{'full':{'native_graphs':binding(train),'held_native_graphs':binding(held)}}}
    write_json_durable(directory.parent/'recipe-plan.json',plan)
    recipe_ref={'path':str(new.resolve()),'sha256':recipe_identity_sha256(new),
        'file_sha256':binding(new)['sha256']};current_ref=binding(current_file)
    adopted={'payload':{'recipe':recipe_ref,'plan':plan}}
    transition={'new_code':code}
    config={
        'production_recipe':str(new),
        'resident_data_directory':str(directory),
        'production_stage':'full','seed_native_graphs':binding(train),'held_native_graphs':binding(held),
        'trainer_parameters':{'seed_native_graphs':binding(train)},
        'seed_trainer_parameters':{'seed_native_graphs':binding(train)},
    }
    selected=runtime._current_adopted_resident_directory(config,current_ref,adopted,'lineage-current',transition)
    assert selected==directory.resolve()
    # A live append may keep its active resident root outside the recipe's
    # directory; the committed recipe, plan, code and sources bind that root.
    assert selected!=new.parent/'resident-data'
    for field,value in (
        ('resident_data_directory',str(old.parent/'resident-data')),
        ('production_recipe',str(old)),
    ):
        changed={**config,field:value}
        with pytest.raises(ValueError,match='authenticated resident namespace'):
            runtime._current_adopted_resident_directory(changed,current_ref,adopted,'lineage-current',transition)
    # The current plan must still match the exact config source bindings.
    with pytest.raises(ValueError,match='train/held source binding'):
        runtime._current_adopted_resident_directory({**config,'seed_native_graphs':binding(held)},
            current_ref,adopted,'lineage-current',transition)


def test_fresh_retry_namespace_uses_retry_plan_and_prior_root_only_as_ancestry(tmp_path,monkeypatch):
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.train.recipe import recipe_identity_sha256
    from natlang_neuralese.rl import native_failed_supervised_retry as retry
    prior_recipe=tmp_path/'prior'/'recipe.json';prior_recipe.parent.mkdir()
    current_recipe=tmp_path/'retry'/'recipe.json';current_recipe.parent.mkdir()
    directory=current_recipe.parent/'resident-data';directory.mkdir()
    current_file=tmp_path/'prior'/'resident-data'/'current.json';current_file.parent.mkdir();write_json_durable(current_file,{'resident':'prior-root'})
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    write_json_durable(train,{'source':'train'});write_json_durable(held,{'source':'held'})
    for path,name in ((prior_recipe,'drained'),(current_recipe,'retry-future')):
        write_json_durable(path,{'schema':'natlang.neuralese-training-recipe/1','id':name,
            'identity_qualification':'native_state_runtime_qualified',
            'input_bindings':{'train':binding(train),'held':binding(held)},
            'stages':[{'id':'full','kind':'native_state_sft','requires':[],
                'inputs':{'native_graphs':'train','held_native_graphs':'held'},'parameters':{'epochs':1}}]})
    lineage_path=tmp_path/'lineage.json'
    from natlang_neuralese.train.recipe import resolve_recipe_data
    plan={'recipe':resolve_recipe_data(current_recipe),'stage_inputs':{'full':{
        'native_graphs':binding(train),'held_native_graphs':binding(held)}},'recipe_sha256':'lineage-current'}
    write_json_durable(lineage_path,plan)
    write_json_durable(directory.parent/'recipe-plan.json',plan)
    predecessor=tmp_path/'predecessor.json';write_json_durable(predecessor,{'production_recipe':str(prior_recipe)})
    intent=tmp_path/'intent.json';write_json_durable(intent,{'snapshots':{'config':binding(predecessor)}})
    receipt=tmp_path/'receipt.json';write_json_durable(receipt,{'intent':binding(intent)})
    retry_ref=binding(receipt);runtime_authority={'path':'/runtime-handoff.json','sha256':'a'*64}
    declaration={'failed_supervised_retry':retry_ref,'runtime_handoff':runtime_authority}
    proof={'lineage':'lineage-current'};transition={'recipe':binding(prior_recipe)}
    cursor={'resident_current':binding(current_file)}
    adopted={'payload':{'prior_recipe':{'sha256':'prior'},'plan':{'recipe_sha256':'lineage-current'}},
        'history':[{'recipe':{'sha256':recipe_identity_sha256(prior_recipe)}}]}
    release={'prior_runtime_handoff':runtime_authority,'recipe_lineage_sha256':'lineage-current',
        'recipe_sha256':recipe_identity_sha256(current_recipe),'prior_resident_current':cursor['resident_current'],
        'resident_plan':{'lineage_handoff':binding(lineage_path)}}
    monkeypatch.setattr(retry,'verify_release',lambda reference,decl:release)
    config={'production_recipe':str(current_recipe),'resident_data_directory':str(directory),
        'production_stage':'full','seed_native_graphs':binding(train),'held_native_graphs':binding(held)}
    assert runtime._retry_resident_directory(config,cursor['resident_current'],adopted,
        transition,proof,declaration,cursor)==directory
    assert directory!=Path(cursor['resident_current']['path']).parent
    config['seed_native_graphs']=binding(tmp_path/'held.jsonl')
    with pytest.raises(ValueError,match='train/held source binding'):
        runtime._retry_resident_directory(config,cursor['resident_current'],adopted,
            transition,proof,declaration,cursor)
