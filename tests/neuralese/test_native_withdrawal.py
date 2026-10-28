import copy
import json

import pytest
import torch

from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha
from natlang_neuralese.train.native_continuation import withdraw_native_rows, collect_sft_progress, progress_obligations, reject_withdrawn_targets


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
    new=withdraw_native_rows(old,path,current_bindings={'good':'g','comm':'c','extra':'e'},
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
    assert withdraw_native_rows(new,path,current_bindings={'good':'g','comm':'c'},model_id='model:pin') is new
    with pytest.raises(ValueError,match='reintroduced'):
        reject_withdrawn_targets(new,{'bad':'b'})
    reject_withdrawn_targets(new,{'good':'g'})


@pytest.mark.parametrize('change',[
    {'binding':'wrong'},{'source_sha256':'other'},{'id':'comm','binding':'c'},
    {'id':'unknown'},{'reason':''}])
def test_withdrawal_rejects_wrong_identity_source_communication_and_missing_reason(tmp_path,change):
    row={'id':'bad','binding':'b','source_sha256':'physical','reason':'rejected'};row.update(change)
    with pytest.raises(ValueError,match='exact removed native'):
        withdraw_native_rows(state(),proof(tmp_path,[row]),current_bindings={'good':'g'},
            model_id='model:pin',source_origins={'bad':'physical','comm':'physical'})


def test_communication_withdrawal_preserves_history_and_removes_classification(tmp_path):
    old=state();old['exposure_counts']['comm']=2
    old['options']['_producer_consumer_ids']=['comm']
    old['sft_progress']=collect_sft_progress(old)
    row={'id':'comm','binding':'c','source_sha256':'physical','reason':'held connected','communication':True}
    new=withdraw_native_rows(old,proof(tmp_path,[row]),current_bindings={'good':'g','bad':'b'},
        model_id='model:pin',source_origins={'comm':'physical'})
    assert new['optimizer'] is old['optimizer'] and new['rng'] is old['rng']
    assert new['options']['_communication_ids']==[] and new['options']['_producer_consumer_ids']==[]
    assert new['sft_progress']['full']['options']['_communication_ids']==[]
    assert new['options']['withdrawal_history'][0]['rows'][0]['consumed_visits']==2
    assert old['options']['_communication_ids']==['comm']
    with pytest.raises(ValueError,match='reintroduced'):
        reject_withdrawn_targets(new,{'comm':'c'})


def test_communication_withdrawal_rejects_false_classification_and_carried_mismatch(tmp_path):
    row={'id':'comm','binding':'c','source_sha256':'physical','reason':'held connected','communication':False}
    with pytest.raises(ValueError,match='exact removed native'):
        withdraw_native_rows(state(),proof(tmp_path,[row]),current_bindings={},model_id='model:pin',source_origins={'comm':'physical'})
    row['communication']=True
    old=state();old['sft_progress']=collect_sft_progress(old)
    old['sft_progress']['full']['options']['_communication_ids']=[]
    with pytest.raises(ValueError,match='carried stage classification'):
        withdraw_native_rows(old,proof(tmp_path,[row]),current_bindings={},model_id='model:pin',source_origins={'comm':'physical'})


def test_withdrawal_cannot_reintroduce_targets_or_accept_corrupt_evidence(tmp_path):
    path=proof(tmp_path,[{'id':'bad','binding':'b','source_sha256':'physical','reason':'rejected'}])
    with pytest.raises(ValueError):
        withdraw_native_rows(state(),path,current_bindings={'bad':'b'},model_id='model:pin',source_origins={'bad':'physical'})
    (tmp_path/'audit.json').write_text('changed')
    with pytest.raises(ValueError,match='evidence hash'):
        withdraw_native_rows(state(),path,current_bindings={},model_id='model:pin',source_origins={'bad':'physical'})


@pytest.mark.parametrize("communication", [False, True])
def test_actual_optimizer_resume_with_removed_target_and_new_good_row(tmp_path, communication):
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.native_state_sft import main
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint'];old=tmp_path/'old.jsonl';new=tmp_path/'new.jsonl';held=tmp_path/'held.jsonl'
    rows=[dict(graph(key),tokenizer_fingerprint=fp) for key in ('good','bad')]
    if communication:
        from test_native_graph import row as communication_row
        rows[1]={**communication_row(), 'id':'bad', 'source_groups':['source-bad'], 'tokenizer_fingerprint':fp}
    old.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    new.write_text(json.dumps(rows[0])+'\n'+json.dumps(dict(graph('extra'),tokenizer_fingerprint=fp))+'\n')
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(old),
        '--held-native-graphs',str(held),'--epochs','1','--eval-every','100','--out',str(tmp_path/'out')]
    assert main(args)==0
    prior=torch.load(tmp_path/'out/checkpoint.pt',weights_only=False)['training']
    path=proof(tmp_path,[{'id':'bad','binding':prior['source_row_bindings']['bad'],
        'source_sha256':sha(old),'reason':'rejected', **({'communication':True} if communication else {})}])
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
    pos=args.index('--withdrawn-native-graphs');args[pos]='--retired-native-withdrawal'
    assert main(args)==0
    retired=torch.load(tmp_path/'out/checkpoint.pt',weights_only=False)['training']
    assert retired['step']==3 and retired['exposure_counts']==actual['exposure_counts']
    assert retired['options']['withdrawal_history']==actual['options']['withdrawal_history']
    (tmp_path/'audit.json').write_text('changed after application')
    with pytest.raises(ValueError,match='evidence hash'):
        main(args)


def test_retirement_requires_committed_exact_proof_and_rejects_reintroduction(tmp_path):
    from natlang_neuralese.train.native_continuation import verify_retired_withdrawal
    row={'id':'bad','binding':'b','source_sha256':'physical','reason':'rejected'}
    path=proof(tmp_path,[row]);old=state()
    with pytest.raises(ValueError,match='not committed'):
        verify_retired_withdrawal(old,path,current_bindings={'good':'g'},model_id='model:pin')
    earned=withdraw_native_rows(old,path,current_bindings={'good':'g','comm':'c'},model_id='model:pin',source_origins={'bad':'physical'})
    assert verify_retired_withdrawal(earned,path,current_bindings={'good':'g','comm':'c'},model_id='model:pin')['rows']==1
    with pytest.raises(ValueError,match='reintroduced'):
        verify_retired_withdrawal(earned,path,current_bindings={'bad':'b'},model_id='model:pin')
    forged=copy.deepcopy(earned);forged['options']['withdrawal_history'][0]['rows'][0]['binding']='forged'
    with pytest.raises(ValueError,match='history differs'):
        verify_retired_withdrawal(forged,path,current_bindings={},model_id='model:pin')


def test_recipe_can_retire_only_the_exact_prior_withdrawal(tmp_path):
    from natlang_neuralese.train.recipe import _data_continuation,_bind_native_source_dependencies,stage_input_args
    path=proof(tmp_path,[{'id':'bad','binding':'b','source_sha256':'physical','reason':'rejected'}])
    bind=lambda name,digest:{'path':name,'sha256':digest,'binding':{'declared':True}}
    old={'native_graphs':bind('old','old-sha'),'held_native_graphs':bind('held','held-sha'),
         'withdrawn_native_graphs':bind(str(path),sha(path))}
    new={'native_graphs':bind('new','new-sha'),'held_native_graphs':old['held_native_graphs'],
         'continued_native_graphs':old['native_graphs'],'retired_native_withdrawal':old['withdrawn_native_graphs']}
    assert _data_continuation(old,new)
    wrong=copy.deepcopy(new);wrong['retired_native_withdrawal']['sha256']='wrong'
    assert not _data_continuation(old,wrong)
    missing=copy.deepcopy(new);del missing['retired_native_withdrawal']
    assert not _data_continuation(old,missing)
    resolved=_bind_native_source_dependencies({'retired_native_withdrawal':new['retired_native_withdrawal']})
    assert resolved['retired_native_withdrawal']['dependencies'][0]['sha256']==sha(tmp_path/'audit.json')
    args=stage_input_args(resolved,'native_state_sft')
    assert '--retired-native-withdrawal' in args and '--withdrawn-native-graphs' not in args


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


def test_real_diloco_full_checkpoint_withdrawal_and_append_keeps_coordinator_identity(tmp_path,monkeypatch):
    import socket,threading,time
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer, native_diloco
    from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,serve_coordinator
    from natlang_neuralese.train.diloco.worker import DilocoSettings,Transport
    base=tmp_path/'base';tiny_source(base);_,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint'];good=dict(graph('good'),tokenizer_fingerprint=fp);bad=dict(graph('bad'),tokenizer_fingerprint=fp);extra=dict(graph('extra'),tokenizer_fingerprint=fp)
    seed=tmp_path/'seed.jsonl';old=tmp_path/'old.jsonl';new=tmp_path/'new.jsonl';held=tmp_path/'held.jsonl'
    seed.write_text(json.dumps(good)+'\n');old.write_text(json.dumps(good)+'\n'+json.dumps(bad)+'\n');new.write_text(json.dumps(good)+'\n'+json.dumps(extra)+'\n');held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    common=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--held-native-graphs',str(held),'--epochs','1','--eval-every','100','--stage-id','native_full_sft']
    assert trainer.main([*common,'--native-graphs',str(seed),'--out',str(tmp_path/'seed')])==0
    parent=tmp_path/'seed/checkpoint.pt';out=tmp_path/'worker';model,named=native_diloco._trainable(parent,str(base))
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}';token='identity-test-token-'+'x'*32
    settings={'run_id':'actual','workers':1,'worker_id':0,'coordinator_url':url,'local_steps':1,'wire_dtype':'fp32','heartbeat_seconds':.05,'allow_private_http':True}
    argv=[*common,'--native-graphs',str(old),'--continued-native-graphs',str(seed),'--continue-from',str(parent),'--out',str(out),'--diloco',json.dumps(settings)]
    identity=native_diloco.identity(parent,argv,DilocoSettings.from_dict(settings))
    root=tmp_path/'coord';initialize_coordinator(root,run_id='actual',masters={name:p.detach() for name,p in named},identity=identity,workers=1,min_quorum=1,fragment_bytes=1024,grace_seconds=0)
    stop=threading.Event();server=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,token),kwargs={'stop':stop},daemon=True);server.start();monkeypatch.setenv('NATLANG_DILOCO_TOKEN',token)
    for _ in range(100):
        try:Transport(url,token,5).json('GET','/v1/status');break
        except OSError:time.sleep(.02)
    try:
        assert trainer.main(argv)==0
        prior=torch.load(out/'checkpoint.pt',weights_only=False)['training'];assert prior['exposure_counts']['bad']==1
        evidence=tmp_path/'audit.json';evidence.write_text('{"decision":"withdraw"}')
        withdrawal=tmp_path/'withdrawal.json';withdrawal.write_text(json.dumps({'schema':'natlang.native-graph-withdrawal/1','model_id':config['architecture']+':'+config['base_inventory_sha256'],
            'evidence':{'path':'audit.json','sha256':sha(evidence)},'rows':[{'id':'bad','binding':prior['source_row_bindings']['bad'],'source_sha256':sha(old),'reason':'rejected'}]}))
        resumed=list(argv);resumed[resumed.index('--native-graphs')+1]=str(new);resumed[resumed.index('--continued-native-graphs')+1]=str(old);resumed += ['--withdrawn-native-graphs',str(withdrawal)]
        assert trainer.main(resumed)==0
        result=torch.load(out/'checkpoint.pt',weights_only=False)['training']
        assert result['global_step']>prior['global_step'] and result['exposure_counts']=={'good':1,'extra':1}
        assert result['options']['withdrawal_history'][0]['rows'][0]['consumed_visits']==1
        assert result['optimizer'] and result['rng'] and result['options']['diloco_identity_transition_history']
        progress=json.loads((root/'progress.json').read_text());assert len(progress['identity_transitions']['0'])==1
        assert progress['completion_history']['0'][0]['completion']['drained']
        assert trainer.main(resumed)==0  # Same checkpoint/source replay keeps the recorded identity transition.
        retired=list(resumed)
        for flag in ('--withdrawn-native-graphs','--continued-native-graphs'):
            n=retired.index(flag);del retired[n:n+2]
        assert trainer.main(retired)==0  # Consumed proof switches may be retired without new exposure.
        later=tmp_path/'later.jsonl';later.write_text(new.read_text()+json.dumps(dict(graph('later'),tokenizer_fingerprint=fp))+'\n')
        appended=list(resumed);appended[appended.index('--native-graphs')+1]=str(later)
        appended[appended.index('--continued-native-graphs')+1]=str(new)
        n=appended.index('--withdrawn-native-graphs');del appended[n:n+2]
        assert trainer.main(appended)==0
        final=torch.load(out/'checkpoint.pt',weights_only=False)['training']
        assert final['exposure_counts']=={'good':1,'extra':1,'later':1} and final['global_step']>result['global_step']
        assert final['options']['withdrawal_history'][0]['rows'][0]['consumed_visits']==1
        assert len(json.loads((root/'progress.json').read_text())['identity_transitions']['0'])==3
    finally:stop.set();server.join(timeout=3)
