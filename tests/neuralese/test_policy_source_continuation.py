import json
from pathlib import Path
import pytest
from natlang_neuralese.common.jsonio import write_json_durable, write_bytes_durable
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.rl.policy_source_continuation import arrange_and_seal, finish_seal, verify_seal


def test_interrupted_policy_append_preserves_old_order_and_objective(tmp_path):
    out=tmp_path
    slot=out/'checkpoint.pt'; slot.write_bytes(b'earned-optimizer-rng')
    old_graphs=[{'id':'a','nodes':[1]},{'id':'b','nodes':[2]}]
    old={'schema':'native-policy','policy':{'actor':'actual'},'rows':{'a':{'actions':[1]},'b':{'actions':[2]}}}
    graph_path=out/'rl-native-graphs.jsonl'; objective_path=out/'rl-native-objective.json'
    write_bytes_durable(graph_path, ''.join(json.dumps(g)+'\n' for g in old_graphs).encode())
    write_json_durable(objective_path,old)
    old_hash=sha256_file_hex_cached(objective_path)
    new={'schema':'native-policy','policy':{'actor':'actual'},'rows':{'a':{'actions':[1]},'b':{'actions':[2]},'c':{'actions':[3]}}}
    graphs=[{'id':'c','nodes':[3]},*old_graphs]
    record=arrange_and_seal(out,graphs,new)
    assert [g['id'] for g in graphs]==['a','b','c'] and record['added_rows']==1
    write_bytes_durable(graph_path, ''.join(json.dumps(g)+'\n' for g in graphs).encode())
    write_json_durable(objective_path,new)
    proof=finish_seal(out,record,graph_path,objective_path)
    result=verify_seal(proof,previous={'sha256':old_hash},current={'sha256':sha256_file_hex_cached(objective_path)},
                       checkpoint=slot,train_sha=sha256_file_hex_cached(graph_path))
    assert result['old_rows']==2
    retry_graphs=[{'id':'c','nodes':[3]},*old_graphs]
    retry=arrange_and_seal(out,retry_graphs,new)
    assert retry==result and [row['id'] for row in retry_graphs]==['a','b','c']
    # A crash between the two source replacements keeps the predecessor proof.
    graph_path.write_bytes(Path(result['predecessor']['graphs']['path']).read_bytes())
    partial=arrange_and_seal(out,[{'id':'c','nodes':[3]},*old_graphs],new)
    assert partial==result
    write_bytes_durable(graph_path, ''.join(json.dumps(g)+'\n' for g in graphs).encode())
    with pytest.raises(ValueError,match='checkpoint or objective'):
        verify_seal(proof,previous={'sha256':'false'},current={'sha256':sha256_file_hex_cached(objective_path)},
                    checkpoint=slot,train_sha=sha256_file_hex_cached(graph_path))
    predecessor=Path(result['predecessor']['objective']['path'])
    assert json.loads(predecessor.read_text())==old


def test_policy_append_rejects_changed_earned_actions_and_graphs(tmp_path):
    (tmp_path/'checkpoint.pt').write_bytes(b'earned')
    (tmp_path/'rl-native-graphs.jsonl').write_text(json.dumps({'id':'a','nodes':[1]})+'\n')
    write_json_durable(tmp_path/'rl-native-objective.json',{'schema':'x','rows':{'a':{'actions':[1]}}})
    with pytest.raises(ValueError,match='changed or removed an earned row'):
        arrange_and_seal(tmp_path,[{'id':'a','nodes':[1]}],{'schema':'x','rows':{'a':{'actions':[2]}}})
    with pytest.raises(ValueError,match='changed or removed an earned graph'):
        arrange_and_seal(tmp_path,[{'id':'a','nodes':[2]}],{'schema':'x','rows':{'a':{'actions':[1]}}})


def test_actual_policy_trainer_resumes_earned_state_and_trains_only_added_row(tmp_path):
    import torch
    from test_native_rl_production import test_native_stage_real_shared_parent_update_and_exact_noop_resume
    from natlang_neuralese.rl.native_stage import main
    test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path, False)
    source=tmp_path/'rollouts.jsonl'; first=json.loads(source.read_text())
    second=json.loads(json.dumps(first)); second['sample']=1
    source.write_text(json.dumps(second)+'\n'+json.dumps(first)+'\n')
    out=tmp_path/'learner'; before=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)['training']
    argv=['--rollouts',str(source),'--rl-tokenizer',str(tmp_path/'tokenizer'),'--rl-mode','rft',
          '--continue-from',str(tmp_path/'shared.pt'),'--held-native-graphs',str(tmp_path/'held.jsonl'),
          '--out',str(out),'--device','cpu','--epochs','1','--rank','4','--learning-rate','1e-3',
          '--weight-decay','0','--eval-every','1']
    assert main(argv)==0
    after=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)['training']
    assert after['global_step']==before['global_step']+1
    assert all(after['exposure_counts'][key]==value for key,value in before['exposure_counts'].items())
    assert sum(after['exposure_counts'].values())==sum(before['exposure_counts'].values())+1
    history=after['options']['policy_source_continuation_history'][-1]
    assert history['old_rows']==1 and history['added_rows']==1
    assert history['at_global_step']==before['global_step']
