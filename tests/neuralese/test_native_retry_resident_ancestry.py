"""Completed no-credit retry bridges the exact old runtime recipe into adoption."""
import copy
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
