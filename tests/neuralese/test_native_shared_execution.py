"""Shared fixed-row/state arithmetic is explicit, complete and resumable."""
import copy
import json
import pytest
import torch
from natlang_neuralese.model.shared_execution import configure_shared_execution,validate_shared_execution
from natlang_neuralese.eval.native_attention_transition import qualify_ffn_chunk_transition
from test_native_attention_transition import fixture,Tokenizer,equal


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('checkpointed',[False,True])
@pytest.mark.parametrize('device',['cpu','cuda'])
def test_complete_graph_cached_and_chunk_gradients_preserve_actual_state(family,checkpointed,device):
    if device=='cuda'and not torch.cuda.is_available():pytest.skip('actual CUDA reference required')
    model,optimizer,named,roles,config=fixture(family,device=device)
    model.port.checkpoint_layers=checkpointed
    before={n:p.detach().clone()for n,p in named};history=copy.deepcopy(optimizer.state_dict())
    rng=torch.get_rng_state().clone()
    configure_shared_execution(model,'stable-v1');config['native_shared_execution']='stable-v1'
    validate_shared_execution(model,config)
    report=qualify_ffn_chunk_transition(model,Tokenizer(),roles,config,optimizer=optimizer,
        previous_chunk_tokens=0,candidate_chunk_tokens=2,checkpoint_sha256='a'*64,
        forward_atol=1e-4,gradient_atol=1e-5)
    assert report['passed']and report['transition_equivalence']['passed']
    for mode in('graph','cached'):
        assert set(report['cross_mode'][mode]['comparisons'])=={'input_embeddings',*(n for n,_ in named)}
    equal(history,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())
    for name,p in named:assert torch.equal(p,before[name])
    configure_shared_execution(model,'source-v1')
    validate_shared_execution(model,{**config,'native_shared_execution':'source-v1'})


def test_unknown_profile_and_live_identity_mismatch_reject():
    model,*_=fixture('lfm')
    with pytest.raises(ValueError):configure_shared_execution(model,'unbound')
    configure_shared_execution(model,'stable-v1')
    with pytest.raises(ValueError):validate_shared_execution(model,{})


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('higher_order',[False,True])
def test_maple_norm_preserves_private_gain_and_reference_derivatives(device,higher_order):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('actual CUDA norm required')
    from types import SimpleNamespace
    import contextlib
    from natlang_neuralese.maple.model import RMSNorm,_rms_norm
    from natlang_neuralese.maple.ternary import STATE
    from natlang_neuralese.common.native_higher_order import reference_arithmetic
    norm=RMSNorm(8,1e-5).to(device);norm.native_stable_norm=True
    norm.private['7']=torch.nn.Parameter(torch.linspace(.01,.08,8,device=device))
    model=SimpleNamespace(port=torch.nn.Sequential(norm))
    configure_shared_execution(model,'stable-v1')
    prior={key:STATE[key]for key in ('enabled','size')}
    try:
        STATE.update(enabled=True,size=7)
        x=torch.randn(2,3,8,device=device,requires_grad=True)
        with reference_arithmetic()if higher_order else contextlib.nullcontext():
            actual=norm(x);expected=_rms_norm(x,norm.gain(),norm.eps)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
            parameters=(x,norm.weight,norm.private['7'])
            a=torch.autograd.grad(actual.square().sum(),parameters,create_graph=higher_order)
            b=torch.autograd.grad(expected.square().sum(),parameters,create_graph=higher_order)
            for left,right in zip(a,b):torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)
            assert a[-1].abs().sum()>0
            if higher_order:
                left=torch.autograd.grad(sum(g.square().sum()for g in a),parameters)
                right=torch.autograd.grad(sum(g.square().sum()for g in b),parameters)
                for value,reference in zip(left,right):
                    assert torch.isfinite(value).all()
                    torch.testing.assert_close(value,reference,atol=2e-5,rtol=2e-5)
    finally:
        STATE.update(prior)


@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference'])
def test_recipe_profile_is_shared_across_objectives(kind):
    from natlang_neuralese.train.recipe import HANDLERS,validate_stage_parameters,stage_parameter_args
    assert 'native_shared_execution'in HANDLERS[kind]['parameters']
    for value in('source-v1','stable-v1','ffn-stable-v1'):
        validate_stage_parameters(kind,{'native_shared_execution':value})
        assert stage_parameter_args({'native_shared_execution':value})==['--native-shared-execution',value]
    with pytest.raises(ValueError):validate_stage_parameters(kind,{'native_shared_execution':'unknown'})


def test_ffn_only_profile_preserves_norm_and_convolution_state_policy():
    from natlang_neuralese.model.shared_execution import convolution_state_dtype, _fixed_ffn
    model, *_ = fixture('lfm')
    norm = next(module for module in model.port.modules() if type(module).__name__ in {'RMSNorm','Lfm2RMSNorm','Lfm3RMSNorm'})
    old_norm = norm.forward
    configure_shared_execution(model, 'ffn-stable-v1')
    validate_shared_execution(model, {'native_shared_execution':'ffn-stable-v1'})
    assert norm.forward == old_norm
    assert any(getattr(module.forward, '__func__', None) is _fixed_ffn for module in model.port.modules())
    with torch.enable_grad():
        assert convolution_state_dtype(model.port) == torch.float32
    configure_shared_execution(model, 'stable-v1')
    configure_shared_execution(model, 'ffn-stable-v1')
    validate_shared_execution(model, {'native_shared_execution':'ffn-stable-v1'})
    assert norm.forward == old_norm


@pytest.mark.parametrize('source_transition',[False,True])
def test_real_checkpoint_resume_inherits_profile_and_exact_optimizer_rng_exposure(tmp_path,monkeypatch,source_transition):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.data.native_graph import SCHEMA
    from test_native_relative_quantum import _same
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    row={'schema':SCHEMA,'id':'train','split':'train','source_groups':['train'],
        'tokenizer_fingerprint':config['tokenizer_fingerprint'],
        'nodes':[{'name':'answer','token_ids':[1,2,3,4,5],'target_weights':[0,1,1,1,1]}]}
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    train.write_text(json.dumps(row)+'\n')
    held.write_text(json.dumps({**row,'id':'held','split':'test','source_groups':['held']})+'\n')
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu','--native-graphs',str(train),
        '--held-native-graphs',str(held),'--epochs','3','--eval-every','1','--seed','1','--checkpoint-every','1',
        '--forward-atol','0.0001','--gradient-atol','0.00001','--refresh-after-updates','1']
    run=tmp_path/'run'
    assert trainer.main([*args,'--out',str(run),*(['--native-shared-execution','stable-v1']if not source_transition else[])])==0
    first=torch.load(run/'checkpoint.pt',weights_only=False)
    if source_transition:
        from natlang_neuralese.model.native_model import model_state
        from natlang_neuralese.train.native_state_sft import capture_training_rng_state
        original=trainer.qualify_precision_transition
        observed=[]
        def inspect_transition(model,*a,**kw):
            _same(model_state(model)['backbone_state'],first['backbone_state'])
            interface=model_state(model)['interface_state']
            assert interface.pop('native_shared_execution')=='stable-v1'
            _same(interface,first['interface_state'])
            _same(kw['optimizer'].state_dict(),first['training']['optimizer'])
            _same(capture_training_rng_state('cpu'),first['training']['rng'])
            result=original(model,*a,**kw)
            _same(kw['optimizer'].state_dict(),first['training']['optimizer'])
            _same(capture_training_rng_state('cpu'),first['training']['rng'])
            observed.append(result)
            return result
        monkeypatch.setattr(trainer,'qualify_precision_transition',inspect_transition)
        assert trainer.main([*args,'--out',str(run),'--native-shared-execution','stable-v1'])==0
        assert len(observed)==1 and observed[0]['shared_execution']=={'from':'source-v1','to':'stable-v1'}
        monkeypatch.setattr(trainer,'qualify_precision_transition',original)
        first=torch.load(run/'checkpoint.pt',weights_only=False)
    assert first['model_config']['native_shared_execution']=='stable-v1'
    assert first['training']['options']['native_shared_execution_history'][0]['reference_check']['complete_gradient_qualification']['passed']
    baseline=tmp_path/'baseline';baseline.mkdir();torch.save(first,baseline/'checkpoint.pt')
    assert trainer.main([*args,'--out',str(run)])==0
    assert trainer.main([*args,'--out',str(baseline),'--native-shared-execution','stable-v1'])==0
    resumed=torch.load(run/'checkpoint.pt',weights_only=False)
    explicit=torch.load(baseline/'checkpoint.pt',weights_only=False)
    assert resumed['model_config']['native_shared_execution']=='stable-v1'
    for key in('backbone_state','interface_state'):
        assert key in resumed;_same(resumed[key],explicit[key])
    for key in('optimizer','rng','global_step','exposure_counts','source_row_bindings','source_row_order'):
        _same(resumed['training'][key],explicit['training'][key])


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('profile',['stable-v1','ffn-stable-v1'])
def test_stable_saved_tensor_identity_preserves_runtime_validation_and_rejects_profile_conflict(tmp_path,monkeypatch,family,profile):
    from test_native_serving_adapters import engine_fixture
    from test_native_raw_policy import parent_training
    from natlang_neuralese.model import native_model as native
    from natlang_neuralese.rl.native_raw_policy import prepare_raw_policy
    from natlang_neuralese.common.hashing import sha256_file_hex
    engine=engine_fixture(tmp_path,family);model=engine.shared_model;config=engine.model_config
    configure_shared_execution(model,profile);config['native_shared_execution']=profile
    saved={'schema':native.CHECKPOINT_SCHEMA,'step':9,'model_config':config,**native.model_state(model),'training':parent_training()}
    assert saved['interface_state']['native_shared_execution']==profile
    parent=tmp_path/'stable.pt';torch.save(saved,parent)
    expected=native.native_identity(model,config,checkpoint_sha256=sha256_file_hex(parent))
    actual=prepare_raw_policy(parent,base=config['base'],tokenizer_out=tmp_path/'tokens')
    assert actual['native_identity']==expected
    # An older explicitly stable diagnostic interface omitted this duplicate
    # declaration. Its already-bound model_config still defines the same mode.
    legacy=copy.deepcopy(saved);legacy['interface_state'].pop('native_shared_execution')
    legacy_path=tmp_path/'legacy-stable.pt';torch.save(legacy,legacy_path)
    restored,_,_,restored_config,_=native.load_native_checkpoint(legacy_path)
    assert native.native_identity(restored,restored_config,checkpoint_sha256=sha256_file_hex(legacy_path))==prepare_raw_policy(
        legacy_path,base=config['base'],tokenizer_out=tmp_path/'legacy-tokens')['native_identity']
    corrupted=copy.deepcopy(saved);corrupted['interface_state']['native_shared_execution']='source-v1'
    bad=tmp_path/'mismatch.pt';torch.save(corrupted,bad)
    with pytest.raises(ValueError,match='native_shared_execution'):
        native.load_native_checkpoint(bad)
    with pytest.raises(ValueError,match='shared execution differs'):
        prepare_raw_policy(bad,base=config['base'],tokenizer_out=tmp_path/'bad-tokens')
    # Actual served-model identity must still reject tampered installed kernels.
    module=next(m for m in model.port.modules()if hasattr(m,'_native_shared_source_forward'))
    module.forward=module._native_shared_source_forward
    with pytest.raises(ValueError,match='module differs'):
        native.native_identity(model,config,checkpoint_sha256=sha256_file_hex(parent))


def test_ffn_invariant_policy_disables_plain_training_vendor_fastpath(monkeypatch):
    from natlang_neuralese.maple import fused_moe
    from natlang_neuralese.model import native_weight_carrier
    x=torch.randn(2,4);weight=torch.randn(3,4)
    marker=object()
    monkeypatch.setattr(native_weight_carrier,'plain_training_active',lambda:True)
    monkeypatch.setattr(fused_moe._InvariantLinear,'apply',lambda *args:marker)
    assert fused_moe.invariant_linear(x,weight,allow_plain_fastpath=False) is marker
    assert torch.equal(fused_moe.invariant_linear(x,weight),torch.nn.functional.linear(x,weight))


def test_canonical_profile_forces_shared_and_adapter_projection_and_readout(monkeypatch):
    import torch.nn.functional as F
    from natlang_neuralese.model import shared_state
    model, *_ = fixture('lfm')
    configure_shared_execution(model,'ffn-stable-v1')
    observed=[]
    def linear(x, weight, *, precision='ieee', force_invariant=False):
        observed.append(force_invariant)
        return F.linear(x,weight)
    monkeypatch.setattr(shared_state,'_linear_weight',linear)
    module=next(module for name,module in model.port.named_modules()
                if type(module) is torch.nn.Linear and '.feed_forward.' not in name)
    shared_state._linear(module,torch.randn(2,module.in_features,dtype=module.weight.dtype))
    adapter=model.adapter.down
    shared_state._linear(adapter,torch.randn(2,adapter.in_features,dtype=adapter.weight.dtype))
    width=model.port.output_weight.shape[1]
    model.logits(torch.randn(1,width,dtype=model.port.output_weight.dtype))
    assert len(observed)>=4 and all(observed)
    configure_shared_execution(model,'source-v1')
    observed.clear()
    shared_state._linear(module,torch.randn(2,module.in_features,dtype=module.weight.dtype))
    assert observed==[False]


def test_canonical_profile_uses_eager_pointwise_even_in_plain_training(monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.model import shared_state, native_weight_carrier
    calls=[]
    monkeypatch.setattr(native_weight_carrier,'plain_training_active',lambda:True)
    monkeypatch.setenv('NATLANG_NATIVE_COMPILE','1')
    monkeypatch.setattr(torch,'compile',lambda fn,**kw:(lambda *a,**k:(calls.append('compiled'),fn(*a,**k))[1]))
    def fn(owner, value):return 'result'
    wrapped=shared_state._plain_compiled(fn)
    class PretendCuda(torch.Tensor):
        @property
        def is_cuda(self): return True
    fake_tensor=torch.zeros(1).as_subclass(PretendCuda)
    assert wrapped(SimpleNamespace(native_shared_execution='ffn-stable-v1'),fake_tensor)=='result'
    assert wrapped(SimpleNamespace(_native_canonical_pointwise=True),fake_tensor)=='result'
    assert calls==[]
    assert wrapped(SimpleNamespace(native_shared_execution='source-v1'),fake_tensor)=='result'
    assert calls==['compiled']
