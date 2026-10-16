import random
import pytest
import torch

from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.eval.native_train_benchmark import read_only_comparison
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import make_native_optimizer


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_backbone_read_only_comparison_preserves_weights_and_rng(tmp_path,family):
    from test_native_full_updates import source
    from natlang_neuralese.model.native_model import load_native_model
    dtype=source(tmp_path/"base",family)
    model,_,roles,config=load_native_model(tmp_path/"base",dtype=dtype,rank=4)
    model.port.checkpoint_layers=True
    model.eval()
    optimizer,named=make_native_optimizer(model,lr=3e-6,in_backward=False)
    example=parse_graph({'schema':'natlang.neuralese-native-graph/1','id':'read-only-whole',
        'split':'train','source_groups':['fixture'],'tokenizer_fingerprint':config['tokenizer_fingerprint'],
        'nodes':[{'name':'writer','token_ids':[4,roles.produce_open,5,6,roles.produce_close,7],
                  'target_weights':[0,1,1,1,1,1],'producing':[[2,4]]}]},
        tokenizer_fingerprint=config['tokenizer_fingerprint'])
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    rng=torch.get_rng_state().clone();python_rng=random.getstate()
    result=read_only_comparison(model,[example],layouts=((True,2),(False,4 if family=="lfm" else 2)),repeats=2)
    assert result['optimizer_updates']==0 and result['whole_graphs']
    assert all(row['status']=='passed' and row['gradient_parameters']>1 for row in result['cases'])
    assert all(torch.equal(before[name],p) for name,p in model.named_parameters())
    assert all(p.grad is None for p in model.parameters())
    assert torch.equal(rng,torch.get_rng_state()) and python_rng==random.getstate()
    assert not model.training and model.port.checkpoint_layers


def test_read_only_comparison_rejects_invalid_probe_before_forward():
    from test_shared_state import tiny
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    with pytest.raises(ValueError,match='at least two'):
        read_only_comparison(model,[],repeats=1)
