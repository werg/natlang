import copy
import pytest
import torch
from natlang_neuralese.train.native_state_sft import packed_graph_loss,main
from natlang_neuralese.model.shared_state import SharedStateModel
from test_shared_state import tiny
from test_native_communication_prefix import episodes


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('limit,expected',[(0,True),(10000,False),(1,True)])
def test_physical_graph_checkpoint_choice_preserves_entire_graph_gradients(family,limit,expected):
    torch.manual_seed(20)
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    direct=copy.deepcopy(model);model.checkpoint_direct_token_limit=limit
    rows=episodes()
    loss,_=packed_graph_loss(model,rows,checkpoint_layers=True,share_causal_prefixes=True,chunk_tokens=7)
    reference,_=packed_graph_loss(direct,rows,checkpoint_layers=False,share_causal_prefixes=True,chunk_tokens=7)
    loss.backward();reference.backward()
    assert model.port.checkpoint_layers is expected
    torch.testing.assert_close(loss,reference,atol=1e-6,rtol=1e-5)
    assert [r.id for r in rows]==['episode:0','episode:1']
    for (name,p),(other,q) in zip(model.named_parameters(),direct.named_parameters()):
        assert name==other
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4)


def test_invalid_limit_rejects_before_allocating_model(tmp_path):
    with pytest.raises(ValueError,match='checkpoint-direct-token-limit'):
        main(['--native-graphs','missing','--held-native-graphs','missing','--out',str(tmp_path),'--checkpoint-direct-token-limit','-1'])


def test_capacity_receipt_binds_the_actual_checkpoint_threshold(tmp_path):
    from test_native_policy_capacity import actual
    from natlang_neuralese.eval.native_memory import capacity_identity
    engine,model,opt,objective,train,args,config,backend,batches=actual(tmp_path,'lfm')
    try:
        uniform=capacity_identity(config,args,backend)
        args.checkpoint_direct_token_limit=12000
        adaptive=capacity_identity(config,args,backend)
        assert 'checkpoint_direct_token_limit' not in uniform
        assert adaptive['checkpoint_direct_token_limit']==12000
        assert adaptive!=uniform
    finally:engine.stop();engine.store.close()
