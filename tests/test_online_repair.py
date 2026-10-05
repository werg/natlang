from pathlib import Path
import json
import pytest
from scripts.online_repair import RepairOutbox


def test_outbox_idempotence_checkpoint_visibility_and_identity(tmp_path):
    box=RepairOutbox(tmp_path/'queue.db', {'data':'pinned'})
    box.record_step(3,[(42,{'cutoff':2})]);box.record_step(3,[(42,{'cutoff':4})])
    assert box.export(2)==[]
    assert len(box.export(3))==1 and box.export(3)[0]['cutoff']==4
    with pytest.raises(ValueError,match='identity mismatch'):
        RepairOutbox(tmp_path/'queue.db',{'data':'different'})


def gate_fixture():
    torch=pytest.importorskip('torch')
    from scripts.online_repair import gated_loss
    # The first target is hard. Full action includes two tokens; later target
    # is separated by ignored observation context.
    logits=torch.tensor([[[0.,4.],[4.,0.],[0.,0.],[4.,0.],[0.,0.]]],requires_grad=True)
    class Model:
        calls=0
        def __call__(self,**kwargs):
            self.calls+=1
            return type('Output',(),{'logits':logits})()
    encoded={'input_ids':torch.zeros((1,5),dtype=torch.long),
             'attention_mask':torch.ones((1,5),dtype=torch.long),
             'labels':torch.tensor([[-100,0,0,-100,0]])}
    return torch,gated_loss,Model(),encoded,logits


def test_gate_uses_one_forward_keeps_complete_corrective_action_and_gradients():
    torch,gate,model,encoded,logits=gate_fixture()
    loss,rows=gate(model,encoded,[[2,4]],mean_nll=2,token_nll=3)
    loss.backward()
    assert model.calls==1 and rows[0]['cutoff']==2
    assert rows[0]['retained_tokens']==2
    assert logits.grad[0,:2].abs().sum()>0
    assert logits.grad[0,2:].abs().sum()==0
    expected=torch.nn.functional.cross_entropy(logits[0,:2],torch.tensor([0,0]))
    torch.testing.assert_close(loss,expected)


def test_easy_sample_is_exact_vanilla_loss():
    torch,gate,model,encoded,logits=gate_fixture()
    loss,rows=gate(model,encoded,[[2,4]],mean_nll=100,token_nll=100)
    assert rows[0]['cutoff'] is None
    expected=torch.nn.functional.cross_entropy(logits[0,[0,1,3]],torch.tensor([0,0,0]))
    torch.testing.assert_close(loss,expected)


def test_mean_gate_and_boundaries_fail_closed():
    torch,gate,model,encoded,logits=gate_fixture()
    _,rows=gate(model,encoded,[[2,4]],mean_nll=.5,token_nll=100)
    assert rows[0]['cutoff']==2
    with pytest.raises(ValueError,match='cover all'):
        gate(model,encoded,[[2]],mean_nll=2,token_nll=3)


def test_full_gold_exposure_still_flags_hard_samples():
    torch,gate,model,encoded,logits=gate_fixture()
    loss,rows=gate(model,encoded,[[2,4]],mean_nll=2,token_nll=3,full_gold=[True])
    loss.backward()
    assert rows[0]['cutoff']==2 and rows[0]['full_gold']
    assert rows[0]['retained_tokens']==3 and logits.grad[0,3].abs().sum()>0


def test_prepare_batch_filters_only_current_checkpoint_visible_hard_rows(tmp_path):
    from scripts.prepare_online_repair_batch import prepare,sha
    corpus=tmp_path/'corpus.jsonl';rows=[{'id':'a','program_id':'p','split':'train'},
        {'id':'b','program_id':'q','split':'train'}]
    lines=[json.dumps(r)+'\n' for r in rows];corpus.write_text(''.join(lines))
    policy={'version':1};identity={'data_sha256':sha(corpus),'policy':policy}
    box=RepairOutbox(tmp_path/'queue.db',identity)
    box.record_step(1,[(0,{'cutoff':2}),(len(lines[0].encode()),{'cutoff':2})])
    box.record_step(2,[(0,{'cutoff':None})]) # easy now: do not repair stale flag
    box.record_step(4,[(0,{'cutoff':2})]) # not checkpoint-visible
    ck=tmp_path/'checkpoint';ck.mkdir();(ck/'adapter_model.safetensors').write_bytes(b'weights')
    (ck/'state.json').write_text(json.dumps({'step':2,'corpus':{'data_sha256':sha(corpus),'online_repair':policy}}))
    artifacts=[]
    for program in ('p','q'):
        artifact=tmp_path/(program+'.json');artifact.write_text(json.dumps({'task':{'program_ir':{'id':program}}}))
        artifacts.append({'path':str(artifact),'sha256':sha(artifact)})
    source=tmp_path/'source-plan.json';source.write_text(json.dumps({
        'schema':'natlang.student_chunk_rewrite_plan/1','selected_program_ids':['p','q'],
        'teacher_artifacts':artifacts,
        'student':{'adapter':str(ck),'weight_pins':{str(ck/'adapter_model.safetensors'):sha(ck/'adapter_model.safetensors')}},
        'online_repair_corpus':str(corpus)}))
    result=prepare(tmp_path/'queue.db',ck,source,tmp_path/'batch')
    plan=json.loads(Path(result['plan']).read_text())
    assert plan['selected_program_ids']==['q']
    assert plan['online_repair']['flags'][0]['row_id']=='b'


def test_gate_with_real_lfm_model_preserves_easy_loss_and_updates(tmp_path):
    torch=pytest.importorskip('torch')
    transformers=pytest.importorskip('transformers')
    if not hasattr(transformers,'Lfm2Config'):pytest.skip('requires training-image Transformers')
    from scripts.train_lora import collate_completions,batch_completion_loss
    from scripts.online_repair import gated_loss
    cfg=transformers.Lfm2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,full_attn_idxs=[0,1],use_cache=False)
    model=transformers.Lfm2ForCausalLM(cfg).train()
    encoded=collate_completions([([1,2],[3,4,5,6],[True,True,False,True])],device='cpu')
    ordinary=batch_completion_loss(model,encoded)
    gated,receipts=gated_loss(model,encoded,[[2,4]],mean_nll=100,token_nll=100)
    torch.testing.assert_close(gated,ordinary)
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    for _ in range(3):
        opt.zero_grad()
        loss,receipts=gated_loss(model,encoded,[[2,4]],mean_nll=.01,token_nll=.01)
        loss.backward();opt.step()
        assert receipts[0]['retained_tokens']==2
