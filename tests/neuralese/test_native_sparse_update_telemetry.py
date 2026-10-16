"""Actual update telemetry must expose BF16 rounding rather than only gradients."""
import pytest
import torch
from test_shared_state import tiny
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.data.native_graph import parse_graph, SCHEMA
from natlang_neuralese.train.native_state_sft import make_native_optimizer, native_training_update


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('master_dtype',[torch.float32,torch.bfloat16])
def test_sparse_row_telemetry_matches_actual_committed_parameters(family,master_dtype):
    port=tiny(family).to(dtype=torch.bfloat16)
    if family=='mellum':port.hf.natlang_precision='bf16'
    model=SharedStateModel(port,(58,59,60,61),rank=4,token_override_ids=(54,55),
        sparse_row_dtype=master_dtype,token_override_dtype=master_dtype)
    example=parse_graph({'schema':SCHEMA,'id':'sparse-telemetry','split':'train',
        'source_groups':['train'],'tokenizer_fingerprint':'fixture','nodes':[
            {'name':'answer','token_ids':[1,58,54,59,60,55,61,2],
             'target_weights':[0,1,1,1,1,1,1,1]}]},tokenizer_fingerprint='fixture')
    if family=='mellum':
        # Deliberate FP32/BF16 counterfactual: 1.0 is exact in both dtypes,
        # while a first AdamW update at this LR is below half a BF16 ULP.
        # Random near-zero rows can now move after the correct AdamW routing.
        with torch.no_grad():model.extra_output_rows.fill_(1.0)
    optimizer,_=make_native_optimizer(model,lr=1e-5)
    names=('input_rows','extra_input_rows') if model.tied else (
        'input_rows','output_rows','extra_input_rows','extra_output_rows')
    before={name:getattr(model,name).detach().clone() for name in names}
    native_training_update(model,optimizer,[example],checkpoint_layers=True,chunk_tokens=3)
    metrics=model.native_sparse_updates
    assert set(metrics)==set(names)
    for name,previous in before.items():
        parameter=getattr(model,name)
        ids=model.token_override_ids if name.startswith('extra_') else model.marker_ids
        assert metrics[name]['dtype']==str(master_dtype)
        for index,row in enumerate(metrics[name]['rows']):
            change=parameter[index].detach().float()-previous[index].float()
            assert row['token_id']==ids[index]
            assert row['changed_elements']==int((change!=0).sum())
            assert row['elements']==parameter.shape[-1]
            assert row['gradient_l2']==float(parameter.grad[index].float().norm())
            assert row['update_l2']==float(change.norm())
            assert row['gradient_l2']>0
            if master_dtype==torch.float32:assert row['update_l2']>0
    if family=='mellum' and master_dtype==torch.bfloat16:
        assert torch.all(before['extra_output_rows']==1.0)
        # Real CE gradients still exist, but every deliberately representable
        # BF16 output coordinate rounds away this production-small Adam step.
        assert all(row['gradient_l2']>0 and row['changed_elements']==0
                   for row in metrics['extra_output_rows']['rows'])
