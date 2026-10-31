"""Shared raw contexts retain complete real producer/consumer dependencies."""
import copy

import pytest
import torch

from natlang_neuralese.data.native_graph import parse_graph,SCHEMA,graph_loss_from_result
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
from natlang_neuralese.train.native_state_sft import packed_graph_loss
from test_shared_state import tiny


def episodes(*,stop=False,empty_relay=False):
    rows=[]
    for index in range(2):
        writer={'name':'writer','token_ids':[2,3,4,5,6,58,10+index,11,59,12],
                'producing':[[6,8]]}
        reader={'name':'reader','token_ids':[2,3,4,5,6,7,60,14,15,61,16],
            'imports':[{'at':7,'source':'writer','start':6,'stop':8,'stop_gradient':stop},
                       {'at':9,'source':'writer','start':6,'stop':8}],
            'producing':[[8,10]]}
        fanout={'name':'fanout','token_ids':[2,3,4,5,6,60,17,18,61],
            'imports':[{'at':6,'source':'writer','start':6,'stop':8}]}
        chained={'name':'chained','token_ids':[2,3,4,5,6,60,21,61],
            'imports':[{'at':6,'source':'reader','start':8,'stop':10}]}
        nodes=[writer,reader,fanout,chained]
        if empty_relay:
            nodes+=[{'name':'relay','token_ids':[2,3,4,5],
                'imports':[{'at':4,'source':'writer','start':6,'stop':8}],
                'logical_producing':[[4,6]]},
                {'name':'final','token_ids':[2,3,4,5,6,60,19,61],
                 'imports':[{'at':6,'source':'relay','start':4,'stop':6,'logical':True}]}]
        for n,node in enumerate(nodes):
            weights=[0.]+[.05 if local<5 else (index+1.) for local in range(1,len(node['token_ids']))]
            for edge in node.get('imports',[]):
                if edge['at']<len(weights):weights[edge['at']]=0.
            if len(weights)>3:weights[3]=0.
            node['target_weights']=weights
        row={'schema':SCHEMA,'id':f'episode:{index}','split':'train','source_groups':[f'train:{index}'],
            'tokenizer_fingerprint':'f'*64,'nodes':nodes}
        rows.append(parse_graph(row,tokenizer_fingerprint='f'*64))
    return rows


@pytest.mark.parametrize('stop',[False,True])
@pytest.mark.parametrize('empty_relay',[False,True])
def test_all_original_histories_positions_flags_and_occurrences_are_preserved(stop,empty_relay):
    rows=episodes(stop=stop,empty_relay=empty_relay)
    plan,maps=pack_causal_prefixes(rows)
    assert plan.token_count < sum(row.plan.token_count for row in rows)
    assert any(node.cache_prefix for node in plan.invocations)
    assert all(not node.producing and not node.logical_producing for node in plan.invocations if node.cache_prefix)
    for row,mapping in zip(rows,maps):
        oldtokens=[token for node in row.plan.invocations for token in node.token_ids]
        newtokens=[token for node in plan.invocations for token in node.token_ids]
        for old,new in enumerate(mapping):
            assert oldtokens[old]==newtokens[new]
            assert row.plan.own_positions[old]==plan.own_positions[new]
            assert row.plan.producing[old]==plan.producing[new]
            assert tuple(mapping[p]for p in row.plan.history(old))==plan.history(new)
            assert row.plan.history_stops(old)==plan.history_stops(new)
    # Intra-episode reuse is useful even when one long graph fills the batch.
    one=pack_causal_prefixes(rows[:1])
    assert one is not None and one[0].token_count < rows[0].plan.token_count
    if empty_relay:
        relay=[node for node in plan.invocations if node.name.endswith(':relay')]
        assert len(relay)==2 and all(not node.token_ids and len(node.imports)==2 for node in relay)


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('stop',[False,True])
def test_outputs_kv_hybrid_and_complete_ce_gradients_match_explicit_cached_execution(family,stop):
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    for parameter in model.port.parameters():parameter.requires_grad_(True)
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    reference=copy.deepcopy(model)
    rows=episodes(stop=stop,empty_relay=True)
    plan,maps=pack_causal_prefixes(rows)
    with torch.no_grad():
        actual=model(plan)
        for row,mapping in zip(rows,maps):
            expected=reference(row.plan,reference=True)
            torch.testing.assert_close(actual['h_final'][:,list(mapping)],expected['h_final'],atol=2e-6,rtol=2e-5)
            for layer_index,(left,right) in enumerate(zip(actual['states'],expected['states'])):
                dim=2 if model.port.is_attention(layer_index)else 1
                index=torch.tensor(mapping,dtype=torch.long)
                for a,b in zip(left,right):
                    torch.testing.assert_close(a.index_select(dim,index),b,atol=2e-6,rtol=2e-5)
    expected=torch.stack([graph_loss_from_result(reference,row,reference(row.plan,reference=True),chunk_tokens=3)[0]
                          for row in rows]).mean()
    actual,details=packed_graph_loss(model,rows,share_causal_prefixes=True,checkpoint_layers=True,chunk_tokens=3)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    assert len(details)==len(rows)
    actual.backward();expected.backward()
    for (name,a),(other,b) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other and (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,atol=3e-6,rtol=2e-4,msg=name)
    assert model.adapter.up.weight.grad.norm()>0


def test_bf16_sft_shares_only_raw_prefixes_and_preserves_producer_consumer_supervision():
    model=SharedStateModel(tiny('lfm').to(dtype=torch.bfloat16),(58,59,60,61),rank=4)
    for parameter in model.port.parameters():parameter.requires_grad_(True)
    reference=copy.deepcopy(model);rows=episodes(empty_relay=True)
    actual,actual_details=packed_graph_loss(model,rows,share_causal_prefixes=True,
        checkpoint_layers=False,chunk_tokens=2)
    expected,expected_details=packed_graph_loss(reference,rows,share_causal_prefixes=False,
        checkpoint_layers=False,chunk_tokens=2)
    actual.backward();expected.backward()
    assert actual_details[0]['packed_physical_tokens'] < actual_details[0]['original_physical_tokens']
    assert [item['supervised_tokens'] for item in actual_details] == [item['supervised_tokens'] for item in expected_details]
    assert [item['boundary_targets'] for item in actual_details] == [item['boundary_targets'] for item in expected_details]
    torch.testing.assert_close(actual.detach(),expected.detach(),atol=0,rtol=0)
    actual_gradients=[];expected_gradients=[]
    for (name,left),(other_name,right) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other_name and (left.grad is None)==(right.grad is None),name
        if left.grad is not None:
            actual_gradients.append(left.grad.detach().float().flatten())
            expected_gradients.append(right.grad.detach().float().flatten())
    actual_gradients=torch.cat(actual_gradients);expected_gradients=torch.cat(expected_gradients)
    relative_error=(actual_gradients-expected_gradients).norm()/expected_gradients.norm().clamp_min(1e-12)
    cosine=torch.nn.functional.cosine_similarity(actual_gradients,expected_gradients,dim=0)
    assert float(relative_error)<.02
    assert float(cosine)>.999


def test_qat_sampled_precision_gates_communication_prefix_packing_on_tiny_graphs():
    from natlang_neuralese.train.native_state_sft import _communication_prefix_allowed
    model=SharedStateModel(tiny('lfm').to(dtype=torch.bfloat16),(58,59,60,61),rank=4)
    rows=episodes(empty_relay=True)
    bf16_precision=(object(),'bf16',0.0,1.5)
    fp8_precision=(object(),'fp8',.066,1.5)
    assert _communication_prefix_allowed(None,bf16_precision)
    assert not _communication_prefix_allowed(None,fp8_precision)
    _,bf16_details=packed_graph_loss(model,rows,share_causal_prefixes=True,
        allow_communication_prefixes=_communication_prefix_allowed(None,bf16_precision))
    _,fp8_details=packed_graph_loss(model,rows,share_causal_prefixes=True,
        allow_communication_prefixes=_communication_prefix_allowed(None,fp8_precision))
    assert bf16_details[0]['packed_physical_tokens'] < bf16_details[0]['original_physical_tokens']
    assert fp8_details[0]['packed_physical_tokens'] == fp8_details[0]['original_physical_tokens']


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_cuda_native_stateful_prefixes_against_cached_reference(family,dtype):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from natlang_neuralese.model.native_parameter_precision import configure_norm_precision
    from natlang_neuralese.model.native_precision import set_native_linear_precision
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    model=SharedStateModel(tiny(family).to(device='cuda',dtype=dtype),(58,59,60,61),rank=4,
        native_norm_execution='stable-v1')
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    configure_norm_precision(model,'fp32-norms-v1')
    set_native_linear_precision(model,'bf16x6')
    configure_backbone_training(model.port,'latent')
    with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    old=copy.deepcopy(model)
    rows=episodes(empty_relay=True)
    outputs=[]
    for owner,shared in [(old,False),(model,True)]:
        with ParameterCotangentScope.for_model(owner,mode='vjp') as scope:
            loss,_=packed_graph_loss(owner,rows,share_causal_prefixes=shared,checkpoint_layers=True,chunk_tokens=3)
            loss.backward();scope.finish()
        outputs.append(loss.detach())
    torch.testing.assert_close(outputs[0],outputs[1],atol=0 if dtype==torch.bfloat16 else 1e-6,
                               rtol=0 if dtype==torch.bfloat16 else 1e-6)
    actual_gradients=[];expected_gradients=[]
    for (name,p),(other,q)in zip(model.named_parameters(),old.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            if dtype==torch.bfloat16:
                actual_gradients.append(p.grad.detach().float().flatten())
                expected_gradients.append(q.grad.detach().float().flatten())
            else:
                torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4,msg=name)
    if dtype==torch.bfloat16:
        actual_gradients=torch.cat(actual_gradients);expected_gradients=torch.cat(expected_gradients)
        relative_error=(actual_gradients-expected_gradients).norm()/expected_gradients.norm().clamp_min(1e-12)
        cosine=torch.nn.functional.cosine_similarity(actual_gradients,expected_gradients,dim=0)
        assert float(relative_error)<.02
        assert float(cosine)>.999


@pytest.mark.parametrize('family',['lfm'])
def test_two_actual_shared_updates_preserve_complete_optimizer_history(family,monkeypatch):
    import test_native_prefix_packing as checks
    # Reuse the established per-parameter SR draw and exact BF16 momentum-cast
    # oracle; only replace its ordinary examples with complete real-edge DAGs.
    monkeypatch.setattr(checks,'examples',lambda:episodes(empty_relay=True))
    checks.test_two_actual_muon_updates_and_optimizer_histories_match_disjoint_reference(family)


@pytest.mark.parametrize('family',['lfm'])
def test_cuda_two_fp32_production_profile_updates_preserve_optimizer_law(family,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    import test_native_prefix_packing as checks
    from natlang_neuralese.model.native_parameter_precision import configure_norm_precision
    from natlang_neuralese.model.native_precision import set_native_linear_precision
    from natlang_neuralese.train.backbone_policy import configure_backbone_training
    from natlang_neuralese.model.shared_execution import configure_shared_execution
    def production(port,markers,**kwargs):
        model=SharedStateModel(port.to(device='cuda',dtype=torch.float32),markers,
            native_norm_execution='stable-v1',**kwargs)
        if family=='mellum':model.port.hf.natlang_precision='bf16'
        configure_norm_precision(model,'fp32-norms-v1')
        set_native_linear_precision(model,'bf16x6')
        configure_shared_execution(model,'stable-v1')
        configure_backbone_training(model.port,'latent')
        with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
        return model
    monkeypatch.setattr(checks,'SharedStateModel',production)
    monkeypatch.setattr(checks,'examples',lambda:episodes(empty_relay=True))
    # Existing exact per-matrix BF16 momentum recurrence/cast oracle and strict
    # every-parameter raw-gradient/update/state tolerances remain unchanged.
    checks.test_two_actual_muon_updates_and_optimizer_histories_match_disjoint_reference(family)
