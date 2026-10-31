import contextlib,json
from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_packed_capacity import _rows_digest, oversize_baseline, qualify_oversize_batches


def test_oversize_baseline_authenticates_exact_prefix(tmp_path):
    sourcefile=tmp_path/'old.json';sourcefile.write_text('{}')
    entries=[{'id':'old','binding':'a'*64,'forward_tokens':7},{'id':'new','binding':'b'*64,'forward_tokens':50}]
    geometry={'identity':{'parser':'pinned'},'source':{'path':str(sourcefile),'sha256':sha256_file_hex(sourcefile)},'kv_slots':[9],
              'rows_sha256':_rows_digest(entries[:1])}
    p=tmp_path/'geometry.json';p.write_text(json.dumps(geometry))
    result=oversize_baseline(SimpleNamespace(entries=entries),{'path':str(p),'sha256':sha256_file_hex(p)},fallback_tokens=3)
    assert result['forward_tokens']==7 and result['kv_slots']==9 and result['prefix_rows']==1
    changed=tmp_path/'changed.json';changed.write_text(json.dumps({'identity':{'parser':'different'}}))
    with pytest.raises(ValueError,match='parser or tokenizer'):
        oversize_baseline(SimpleNamespace(entries=entries),{'path':str(p),'sha256':sha256_file_hex(p)},fallback_tokens=3,current_geometry={'path':str(changed),'sha256':sha256_file_hex(changed)})
    entries[0]['binding']='c'*64
    with pytest.raises(ValueError,match='exact source prefix'):
        oversize_baseline(SimpleNamespace(entries=entries),{'path':str(p),'sha256':sha256_file_hex(p)},fallback_tokens=3)


def test_oversize_gate_uses_actual_point_and_restores_prior_gradients(tmp_path, monkeypatch):
    from natlang_neuralese.eval import native_memory
    calls=[]
    def fake(model,optimizer,cases,**kwargs):
        calls.append(kwargs)
        for _,parameter in optimizer._native_named_parameters:assert parameter.grad is None
        return {'passed':True,'optimizer_updates':0}
    monkeypatch.setattr(native_memory,'qualify_capacity',fake)
    parameter=torch.nn.Parameter(torch.ones(2));parameter.grad=torch.tensor([3.,4.]);gradient=parameter.grad
    model=torch.nn.Linear(2,2);optimizer=SimpleNamespace(_native_named_parameters=[('p',parameter)])
    example=SimpleNamespace(plan=SimpleNamespace(token_count=100,slot_sources=range(101)))
    source=tmp_path/'source.json';source.write_text('{}')
    args=SimpleNamespace(cotangent_accumulator_backend='plain',checkpoint_layers=True,chunk_tokens=16,accumulate_batches=16,out=tmp_path)
    precision=(object(),'fp8',0.06,1.5)
    ref=qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,precision=precision,
        source_path=source,checkpoint_sha256='a'*64,global_step=62,baseline={'forward_tokens':10,'kv_slots':11})
    assert ref and calls[0]['cotangent_mode']=='plain' and calls[0]['precision'] is precision
    assert calls[0]['retained_gradients'] and parameter.grad is gradient
    assert json.loads(open(ref['path']).read())['optimizer_updates']==0
    assert qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,precision=precision,
        source_path=source,checkpoint_sha256='a'*64,global_step=62,baseline={'forward_tokens':100,'kv_slots':101}) is None


def test_capacity_uses_selected_precision_context_and_plain_backend(monkeypatch):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples,assert_exact
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.eval.native_memory import qualify_capacity
    from natlang_neuralese.train import native_state_sft
    import copy
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    before=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    flag={'active':False,'observed':False};original=native_state_sft.packed_graph_loss
    class Qat:
        @contextlib.contextmanager
        def context(self,point,mix):
            assert point=='fp8' and mix==0.2
            flag['active']=True
            try:yield
            finally:flag['active']=False
    def checked(*args,**kwargs):
        assert flag['active'] and not kwargs['allow_communication_prefixes']
        flag['observed']=True
        return original(*args,**kwargs)
    monkeypatch.setattr(native_state_sft,'packed_graph_loss',checked)
    report=qualify_capacity(model,optimizer,examples()[:1],checkpoint_sha256='a'*64,source_binding={},
        cotangent_mode='plain',precision=(Qat(),'fp8',0.2,1.5),production_packing=True)
    assert report['passed'] and flag['observed'] and report['qat_point']['point']=='fp8'
    assert_exact(before,model.state_dict());assert_exact(state,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_capacity_with_real_inline_quantization_preserves_state():
    import copy
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples,assert_exact
    from test_quantization import component
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.quantization import Quantization
    from natlang_neuralese.eval.native_memory import qualify_capacity
    model=SharedStateModel(tiny('lfm'),(58,59,60,61),rank=4)
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    qat=Quantization(component());assert qat.install(model,inline=True)
    before=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict());qat_state=copy.deepcopy(qat.state_dict());rng=torch.get_rng_state().clone()
    result=qualify_capacity(model,optimizer,examples()[:1],checkpoint_sha256='a'*64,source_binding={},
        cotangent_mode='plain',precision=(qat,'q4',0.5,1.5),production_packing=True)
    assert result['passed'] and result['qat_point']=={'point':'q4','mix':0.5,'loss_weight':1.5}
    assert_exact(before,model.state_dict());assert_exact(state,optimizer.state_dict());assert_exact(qat_state,qat.state_dict());assert torch.equal(rng,torch.get_rng_state())


def test_failed_oversize_gate_preserves_gradients_and_records_no_update(tmp_path, monkeypatch):
    from natlang_neuralese.eval import native_memory
    def fail(*args,**kwargs):return {'passed':False,'optimizer_updates':0,'cases':[{'status':'out_of_memory'}]}
    monkeypatch.setattr(native_memory,'qualify_capacity',fail)
    parameter=torch.nn.Parameter(torch.ones(2));parameter.grad=torch.tensor([3.,4.]);gradient=parameter.grad
    model=torch.nn.Linear(2,2);optimizer=SimpleNamespace(_native_named_parameters=[('p',parameter)])
    example=SimpleNamespace(plan=SimpleNamespace(token_count=100,slot_sources=range(101)))
    source=tmp_path/'source.json';source.write_text('{}')
    args=SimpleNamespace(cotangent_accumulator_backend='plain',checkpoint_layers=True,chunk_tokens=16,accumulate_batches=16,out=tmp_path)
    with pytest.raises(ValueError,match='failed live-model/QAT capacity'):
        qualify_oversize_batches(model,optimizer,[[example]],args=args,backend=None,precision=None,
            source_path=source,checkpoint_sha256='a'*64,global_step=62,baseline={'forward_tokens':10,'kv_slots':11})
    assert parameter.grad is gradient
    record=json.loads(next((tmp_path/'oversize-capacity').glob('*.json')).read_text())
    assert not record['passed'] and record['optimizer_updates']==record['earned_exposure']==0


def test_gate_failure_before_update_preserves_durable_checkpoint(tmp_path, monkeypatch):
    from test_native_source_fetch_recovery import source,equal
    from natlang_neuralese.train import native_state_sft as trainer,native_packed_capacity
    base=tmp_path/'base';paths=source(base,'lfm');out=tmp_path/'out'
    args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu',
          '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
          '--epochs','2','--batch-graphs','1','--accumulate-batches','1','--eval-every','1','--checkpoint-every','1','--checkpoint-seconds','0',
          '--seed','23','--lr-warmup','8','--refresh-after-updates','1','--out',str(out)]
    assert trainer.main(args)==0
    checkpoint=out/'checkpoint.pt';before=torch.load(checkpoint,map_location='cpu',weights_only=False)
    def fail(*args,**kwargs):raise ValueError('injected oversize capacity failure')
    monkeypatch.setattr(native_packed_capacity,'qualify_oversize_batches',fail)
    with pytest.raises(ValueError,match='injected oversize capacity'):
        trainer.main(args+['--continue-from',str(checkpoint)])
    after=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert after['training']['global_step']==before['training']['global_step']
    equal(before['training']['optimizer'],after['training']['optimizer'])
    equal(before['training']['rng'],after['training']['rng'])
    assert before['training']['exposure_counts']==after['training']['exposure_counts']
