"""Missing-plan recovery authenticates saved state; it never fabricates old authority."""
import copy
import json
from pathlib import Path
import pytest
import torch
from natlang_neuralese.train import recipe as runner
from natlang_neuralese.train.native_plan_recovery import checkpoint_plan_recovery
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.common.hashing import sha256_file_hex as sha


def fixture(tmp_path):
    train=tmp_path/'train.jsonl';train.write_text('{}\n')
    held=tmp_path/'held.jsonl';held.write_text('{}\n')
    recipe={'schema':'natlang.neuralese-training-recipe/1','id':'recovery',
        'identity_qualification':'native_state_runtime_qualified',
        'input_bindings':{'train':{'path':str(train),'sha256':sha(train)},'held':{'path':str(held),'sha256':sha(held)}},
        'stages':[{'id':'sft','kind':'native_state_sft','requires':[],'parameters':{'epochs':1.},
            'inputs':{'native_graphs':'train','held_native_graphs':'held'}}]}
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe))
    resolved=runner.load_recipe(path)
    inputs={'sft':runner.resolve_stage_inputs(resolved,resolved['stages'][0],{})}
    lineage='7'*64
    state={'schema':CHECKPOINT_SCHEMA,'backbone_state':{'weight':torch.tensor([2.])},
        'interface_state':{'input_rows':torch.tensor([[3.]])},'model_config':{'architecture':'mellum'},
        'training':{'schema':TRAINING_SCHEMA,'optimizer':{'moment':torch.tensor([.25])},
            'rng':{'cpu':torch.get_rng_state()},'global_step':11368,
            'options':{'recipe_binding':json.dumps(runner.native_recipe_binding(path,lineage)),'stage_id':'sft'},
            'source_receipts':[{'path':str(train),'sha256':sha(train)},{'path':str(held),'sha256':sha(held)}],
            'source_row_bindings':{'row':'bound-row'},'exposure_counts':{'row':3},'sft_progress':{'pending':{'credits':1}}}}
    parent=tmp_path/'checkpoint.pt';torch.save(state,parent)
    return parent,path,resolved,inputs,state


def test_new_plan_install_resume_uses_saved_lineage_and_exact_numeric_parent(tmp_path,monkeypatch):
    parent,path,recipe,inputs,state=fixture(tmp_path)
    checksum=sha(parent)
    proof=checkpoint_plan_recovery(parent,path,recipe,inputs,expected_sha256=checksum)
    assert proof['lineage_sha256']=='7'*64!=proof['current_recipe']['sha256']
    assert proof['historical_plan_reconstructed'] is False
    out=tmp_path/'recovered';seen=[]
    def launch(command,**kwargs):
        seen.append(command)
        raise RuntimeError('launch captured before allocating model')
    monkeypatch.setattr(runner._ChildSignalForwarder,'run',lambda self,command,**kwargs:launch(command,**kwargs))
    args=['--recipe',str(path),'--out',str(out),'--device','cpu','--recover-native-checkpoint',str(parent),
        '--recover-native-checkpoint-sha256',checksum]
    with pytest.raises(RuntimeError,match='launch captured'):runner.main(args)
    plan=json.loads((out/'recipe-plan.json').read_text())
    assert plan['recipe_sha256']=='7'*64 and plan['checkpoint_recovery']==proof
    assert not (out/'resident-data/current.json').exists()
    command=seen[0]
    assert command[command.index('--continue-from')+1]==str(parent)
    binding=json.loads(command[command.index('--recipe-binding')+1])
    assert binding['lineage_sha256']=='7'*64 and binding['sha256']==sha(path)
    assert runner.main(['--recipe',str(path),'--out',str(out),'--device','cpu','--plan-only'])==0
    assert sha(parent)==checksum  # No tensor, moment, RNG or credit was rewritten.
    with pytest.raises(ValueError,match='missing plan'):runner.main(args+['--plan-only'])
    broken=copy.deepcopy(plan);broken['checkpoint_recovery']['lineage_sha256']='9'*64
    (out/'recipe-plan.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError,match='authority changed'):
        runner.main(['--recipe',str(path),'--out',str(out),'--device','cpu','--plan-only'])


@pytest.mark.parametrize('change',['wrong-sha','recipe','lineage-policy','pruned','rng','optimizer','stage','preceding','train','held','credits'])
def test_recovery_rejects_wrong_or_missing_exact_authority(tmp_path,change):
    parent,path,recipe,inputs,state=fixture(tmp_path)
    expected=sha(parent)
    if change=='wrong-sha':expected='0'*64
    elif change=='recipe':path.write_text(path.read_text()+'\n')
    elif change=='lineage-policy':
        old=json.loads(state['training']['options']['recipe_binding']);old['sha256']='8'*64
        state['training']['options']['recipe_binding']=json.dumps(old)
    elif change=='pruned':state['optimizer_pruned']=True
    elif change in ('rng','optimizer'):state['training'].pop(change)
    elif change=='stage':state['training']['options']['stage_id']='invented'
    elif change=='preceding':recipe['stages'][0]['requires']=['unproven-stage']
    elif change in ('train','held'):inputs['sft']['native_graphs' if change=='train' else 'held_native_graphs']['sha256']='1'*64
    elif change=='credits':state['training']['exposure_counts']['row']=-1
    if change in ('lineage-policy','pruned','rng','optimizer','stage','credits'):
        torch.save(state,parent);expected=sha(parent)
    with pytest.raises(ValueError):checkpoint_plan_recovery(parent,path,recipe,inputs,expected_sha256=expected)
