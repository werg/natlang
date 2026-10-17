import copy
import json

import pytest
import torch

from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
from natlang_neuralese.train.native_continuation import withdraw_ordinary_rows, collect_sft_progress, progress_obligations, reject_withdrawn_targets


def proof(tmp_path, rows):
    evidence=tmp_path/'audit.json';evidence.write_text('{"decision":"rejected original actions"}')
    path=tmp_path/'withdrawal.json'
    path.write_text(json.dumps({'schema':'natlang.native-graph-withdrawal/1','model_id':'model:pin',
        'evidence':{'path':'audit.json','sha256':sha(evidence)},'rows':rows}))
    return path


def state():
    return {'schema':'natlang.neuralese-native-sft-training/1','step':7,'global_step':11,
        'identity':{'objective':None},'optimizer':{'moments':torch.tensor([3.])},'rng':{'stream':42},
        'source_receipts':[{'sources':[{'sha256':'physical'}]}],
        'source_row_bindings':{'good':'g','bad':'b','comm':'c'},
        'source_row_order':['good','bad','comm'],'exposure_counts':{'good':1,'bad':2,'comm':0},
        'options':{'epochs':1,'stage_id':'full','_communication_ids':['comm']}}


def test_exact_withdrawal_preserves_optimizer_rng_good_credits_and_carried_obligations(tmp_path):
    old=state();old['sft_progress']=collect_sft_progress(old)
    older=copy.deepcopy(old['sft_progress']['full']);older['exposure_counts']['bad']=5
    old['sft_progress']['earlier']=older
    path=proof(tmp_path,[{'id':'bad','binding':'b','source_sha256':'physical','reason':'unexecuted'}])
    new=withdraw_ordinary_rows(old,path,current_bindings={'good':'g','comm':'c','extra':'e'},
        model_id='model:pin',source_origins={'bad':'physical'})
    assert new['optimizer'] is old['optimizer'] and new['rng'] is old['rng']
    assert new['step']==7 and new['global_step']==11 and old['exposure_counts']['bad']==2
    assert new['source_row_order']==['good','comm'] and new['exposure_counts']=={'good':1,'comm':0}
    assert progress_obligations(new['sft_progress'])['remaining_graph_visits']==2
    history=new['options']['withdrawal_history'][0]
    assert history['rows'][0]['consumed_visits']==2 and not history['prior_updates_undone']
    assert history['rows'][0]['consumed_visits_by_stage']=={'full':2,'earlier':5}
    assert old['sft_progress']['full']['exposure_counts']['bad']==2
    assert old['sft_progress']['earlier']['exposure_counts']['bad']==5
    assert 'withdrawal_history' not in old['options']
    assert withdraw_ordinary_rows(new,path,current_bindings={'good':'g','comm':'c'},model_id='model:pin') is new
    with pytest.raises(ValueError,match='reintroduced'):
        reject_withdrawn_targets(new,{'bad':'b'})
    reject_withdrawn_targets(new,{'good':'g'})


@pytest.mark.parametrize('change',[
    {'binding':'wrong'},{'source_sha256':'other'},{'id':'comm','binding':'c'},
    {'id':'unknown'},{'reason':''}])
def test_withdrawal_rejects_wrong_identity_source_communication_and_missing_reason(tmp_path,change):
    row={'id':'bad','binding':'b','source_sha256':'physical','reason':'rejected'};row.update(change)
    with pytest.raises(ValueError,match='exact removed ordinary'):
        withdraw_ordinary_rows(state(),proof(tmp_path,[row]),current_bindings={'good':'g'},
            model_id='model:pin',source_origins={'bad':'physical','comm':'physical'})


def test_withdrawal_cannot_reintroduce_targets_or_accept_corrupt_evidence(tmp_path):
    path=proof(tmp_path,[{'id':'bad','binding':'b','source_sha256':'physical','reason':'rejected'}])
    with pytest.raises(ValueError):
        withdraw_ordinary_rows(state(),path,current_bindings={'bad':'b'},model_id='model:pin',source_origins={'bad':'physical'})
    (tmp_path/'audit.json').write_text('changed')
    with pytest.raises(ValueError,match='evidence hash'):
        withdraw_ordinary_rows(state(),path,current_bindings={},model_id='model:pin',source_origins={'bad':'physical'})


def test_actual_optimizer_resume_with_removed_target_and_new_good_row(tmp_path):
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.native_state_sft import main
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint'];old=tmp_path/'old.jsonl';new=tmp_path/'new.jsonl';held=tmp_path/'held.jsonl'
    rows=[dict(graph(key),tokenizer_fingerprint=fp) for key in ('good','bad')]
    old.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    new.write_text(json.dumps(rows[0])+'\n'+json.dumps(dict(graph('extra'),tokenizer_fingerprint=fp))+'\n')
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(old),
        '--held-native-graphs',str(held),'--epochs','1','--eval-every','100','--out',str(tmp_path/'out')]
    assert main(args)==0
    prior=torch.load(tmp_path/'out/checkpoint.pt',weights_only=False)['training']
    path=proof(tmp_path,[{'id':'bad','binding':prior['source_row_bindings']['bad'],
        'source_sha256':sha(old),'reason':'rejected'}])
    d=json.loads(path.read_text());d['model_id']=config['architecture']+':'+config['base_inventory_sha256'];path.write_text(json.dumps(d))
    args[args.index('--native-graphs')+1]=str(new)
    args+=['--continued-native-graphs',str(old),'--withdrawn-native-graphs',str(path)]
    assert main(args)==0
    actual=torch.load(tmp_path/'out/checkpoint.pt',weights_only=False)['training']
    assert actual['step']==3 and actual['exposure_counts']=={'good':1,'extra':1}
    assert actual['options']['withdrawal_history'][0]['rows'][0]['consumed_visits']==1
    # Restart does not reapply withdrawals or repeat good targets.
    assert main(args)==0
    assert torch.load(tmp_path/'out/checkpoint.pt',weights_only=False)['training']['step']==3
    # A new owner resumes the filtered current source without the obsolete
    # predecessor flag. The immutable proof still authenticates its prior use.
    del args[args.index('--continued-native-graphs'):args.index('--continued-native-graphs')+2]
    old.unlink()
    assert main(args)==0
    resumed=torch.load(tmp_path/'out/checkpoint.pt',weights_only=False)['training']
    assert resumed['step']==3 and resumed['exposure_counts']==actual['exposure_counts']
    assert resumed['options']['withdrawal_history']==actual['options']['withdrawal_history']
    (tmp_path/'audit.json').write_text('changed after application')
    with pytest.raises(ValueError,match='evidence hash'):
        main(args)


def test_recipe_withdrawal_keeps_exact_predecessor_and_evidence_closure(tmp_path):
    from natlang_neuralese.train.recipe import _data_continuation, _bind_native_source_dependencies, stage_input_args
    path=proof(tmp_path,[{'id':'bad','binding':'b','source_sha256':'physical','reason':'rejected'}])
    binding=lambda name,digest:{'path':name,'sha256':digest,'binding':{'declared':True}}
    old={'native_graphs':binding('old','old-sha'),'held_native_graphs':binding('held','held-sha'),
         'foundation_native_graphs':binding('foundation','fixed')}
    new={**old,'native_graphs':binding('filtered','new-sha'),
         'continued_native_graphs':old['native_graphs'],
         'withdrawn_native_graphs':binding(str(path),sha(path))}
    assert _data_continuation(old,new)
    bad=copy.deepcopy(new);bad['continued_native_graphs']['sha256']='wrong'
    assert not _data_continuation(old,bad)
    bad=copy.deepcopy(new);bad['foundation_native_graphs']['sha256']='moved'
    assert not _data_continuation(old,bad)
    resolved=_bind_native_source_dependencies({'withdrawn_native_graphs':new['withdrawn_native_graphs']})
    assert resolved['withdrawn_native_graphs']['dependencies'][0]['sha256']==sha(tmp_path/'audit.json')
    assert stage_input_args(resolved,'native_state_sft')==['--withdrawn-native-graphs',str(path)]
