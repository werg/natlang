import copy
import gc
import json
import mmap

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from natlang_neuralese.common.checkpoint_file_storage import CheckpointFileStore, SCHEMA
from natlang_neuralese.common.checkpoint_offload import CheckpointInputOffload, configure_checkpoint_input_offload


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32,torch.bfloat16,torch.float16,torch.int64,torch.bool,torch.complex64])
def test_exact_strided_values_repeated_reads_and_holder_lifetime(tmp_path,dtype):
    store=CheckpointFileStore(tmp_path)
    original=torch.arange(42).reshape(6,7).to(dtype).T
    saved=store.pack(original)
    path=saved.path
    assert path.stat().st_size==original.numel()*original.element_size()
    first=saved.unpack();second=saved.unpack()
    torch.testing.assert_close(first,original,atol=0,rtol=0)
    torch.testing.assert_close(second,original,atol=0,rtol=0)
    first.zero_()
    torch.testing.assert_close(saved.unpack(),original,atol=0,rtol=0)
    assert store.snapshot()['read_bytes']==3*path.stat().st_size
    assert store.snapshot()['live_files']==1
    directory=store.path
    del store;gc.collect()
    assert path.is_file()  # Holder keeps the store and its lease alive.
    del saved;gc.collect()
    assert not directory.exists()


def test_live_stores_survive_cleanup_and_abandoned_store_is_removed(tmp_path):
    live=CheckpointFileStore(tmp_path)
    saved=live.pack(torch.ones(8))
    orphan=tmp_path/'natlang-checkpoint-input-abandoned';orphan.mkdir()
    (orphan/'store.json').write_text(json.dumps({'schema':SCHEMA}))
    (orphan/'.lock').touch();(orphan/'tensor.bin').write_bytes(b'abandoned')
    unrelated=tmp_path/'natlang-checkpoint-input-unknown';unrelated.mkdir()
    (unrelated/'store.json').write_text(json.dumps({'schema':'unrelated'}))
    other=CheckpointFileStore(tmp_path)
    assert not orphan.exists() and saved.path.exists() and unrelated.exists()
    assert live.path.exists() and other.path.exists()
    saved.cleanup();assert live.snapshot()['live_bytes']==0


def test_failed_pack_releases_partial_file_and_counters(tmp_path,monkeypatch):
    store=CheckpointFileStore(tmp_path)
    monkeypatch.setattr(torch,'from_file',lambda *a,**k:(_ for _ in ()).throw(OSError('disk error')))
    with pytest.raises(OSError,match='disk error'):store.pack(torch.ones(8))
    assert not list(store.path.glob('tensor-*')) and store.snapshot()['live_bytes']==0


def test_switching_wrappers_keeps_live_graph_files_and_restores_backend(tmp_path):
    from types import SimpleNamespace
    port=SimpleNamespace(activation_checkpoint=checkpoint)
    wrapper=configure_checkpoint_input_offload(port,enabled=True,directory=tmp_path)
    assert configure_checkpoint_input_offload(port,enabled=True,directory=tmp_path) is wrapper
    x=torch.randn(9,requires_grad=True)
    result=wrapper(lambda v:v.sin().square(),x,use_reentrant=False)
    directory=wrapper.file_store.path
    assert configure_checkpoint_input_offload(port,enabled=False) is checkpoint
    del wrapper;gc.collect()
    assert directory.exists()
    result.sum().backward()
    torch.testing.assert_close(x.grad,2*x.detach().sin()*x.detach().cos(),atol=0,rtol=0)
    del result;gc.collect()
    assert not directory.exists()
    with pytest.raises(ValueError,match='requires enabled'):
        configure_checkpoint_input_offload(port,enabled=False,directory=tmp_path)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_checkpoint_repeated_unpack_and_second_derivative_are_exact(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    x=torch.randn(12,device=device,dtype=torch.float64,requires_grad=True)
    original=x.detach().clone().requires_grad_()
    fn=lambda v:v.sin().square()
    baseline=checkpoint(fn,original,use_reentrant=False).sum()
    reference=torch.autograd.grad(baseline,original,create_graph=True)[0]
    reference2=torch.autograd.grad(reference.sum(),original)[0]
    wrapper=CheckpointInputOffload(checkpoint,directory=tmp_path)
    result=wrapper(fn,x,use_reentrant=False).sum()
    first=torch.autograd.grad(result,x,create_graph=True,retain_graph=True)[0]
    second=torch.autograd.grad(first.sum(),x,retain_graph=True)[0]
    again=torch.autograd.grad(result,x)[0]
    for left,right in [(result,baseline),(first,reference),(again,reference),(second,reference2)]:
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    assert wrapper.snapshot()['unpacks']>=2


@pytest.mark.parametrize('family',['lfm'])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('group_size',[1,4])
def test_both_native_graphs_gradients_next_update_and_rng(tmp_path,family,device,group_size):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.data.native_graph import parse_graph,graph_loss
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4).to(device)
    model.port.checkpoint_layer_group_size=group_size
    reference=copy.deepcopy(model)
    row={'schema':'natlang.neuralese-native-graph/1','id':'file-storage','split':'train',
        'source_groups':['file-storage'],'tokenizer_fingerprint':'a'*64,'nodes':[
        {'name':'producer','token_ids':[2,3,4,5,6],'producing':[[1,5]],'target_weights':[0,.1,.1,1,1]},
        {'name':'consumer','token_ids':[7,8,9,10],'imports':[{'at':1,'source':'producer','start':1,'stop':5}],
         'target_weights':[0,0,1,1]}]}
    graph=parse_graph(row,tokenizer_fingerprint='a'*64)
    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if device=='cuda' else None
    expected,_=graph_loss(model,graph,checkpoint_layers=True,chunk_tokens=2);expected.backward()
    after=torch.get_rng_state();cuda_after=torch.cuda.get_rng_state() if device=='cuda' else None
    torch.set_rng_state(rng)
    if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
    wrapper=configure_checkpoint_input_offload(reference.port,enabled=True,directory=tmp_path)
    actual,_=graph_loss(reference,graph,checkpoint_layers=True,chunk_tokens=2);actual.backward()
    torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    for (name,left),(other,right) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other and (left.grad is None)==(right.grad is None)
        if left.grad is not None:torch.testing.assert_close(left.grad,right.grad,atol=0,rtol=0,msg=name)
    assert torch.equal(after,torch.get_rng_state())
    if cuda_after is not None:assert torch.equal(cuda_after,torch.cuda.get_rng_state())
    a=torch.optim.AdamW(model.parameters(),lr=.001);b=torch.optim.AdamW(reference.parameters(),lr=.001)
    a.step();b.step()
    for left,right in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(left,right,atol=0,rtol=0)
    for left,right in zip(a.state.values(),b.state.values()):
        for name in left:torch.testing.assert_close(left[name],right[name],atol=0,rtol=0)
    assert wrapper.snapshot()['written_bytes']>0 and wrapper.snapshot()['read_bytes']>0


def test_sft_directory_inherits_then_changes_without_losing_visits(tmp_path):
    from test_native_model import tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    paths={}
    for split in ('train','test'):
        row=graph(split);row.update(split=split,tokenizer_fingerprint=config['tokenizer_fingerprint'],source_groups=[split])
        paths[split]=tmp_path/(split+'.jsonl');paths[split].write_text(json.dumps(row)+'\n')
    out=tmp_path/'run';directory=str((tmp_path/'scratch').resolve())
    argv=['--base',str(base),'--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
        '--out',str(out),'--device','cpu','--dtype','float32','--rank','4','--epochs','1','--eval-every','1','--checkpoint-every','1']
    assert trainer.main(argv+['--checkpoint-input-offload','--checkpoint-input-offload-directory',directory])==0
    def saved():return torch.load(out/'checkpoint.pt',weights_only=False)['training']
    first=saved();assert first['options']['checkpoint_input_offload_directory']==directory
    argv[argv.index('--epochs')+1]='2'
    assert trainer.main(argv)==0
    second=saved();assert second['options']['checkpoint_input_offload_directory']==directory
    assert second['exposure_counts']=={'train':2} and second['global_step']==first['global_step']+1
    telemetry=json.loads((out/'progress.json').read_text())['checkpoint_input_offload']
    assert telemetry['storage']=='file' and telemetry['written_bytes']>0 and telemetry['read_bytes']>0
    argv[argv.index('--epochs')+1]='3'
    assert trainer.main(argv+['--checkpoint-input-offload-directory',''])==0
    third=saved();assert third['options']['checkpoint_input_offload'] and third['options']['checkpoint_input_offload_directory'] is None
    assert third['exposure_counts']=={'train':3}
    argv[argv.index('--epochs')+1]='4'
    assert trainer.main(argv+['--no-checkpoint-input-offload'])==0
    fourth=saved();assert not fourth['options']['checkpoint_input_offload']
    assert fourth['options']['checkpoint_input_offload_directory'] is None and fourth['exposure_counts']=={'train':4}


def test_file_mode_bound_in_capacity_identity_and_explicit_settings_are_checked(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.eval.native_memory import capacity_identity
    from natlang_neuralese.common.checkpoint_offload import configure_checkpoint_storage_options
    args=SimpleNamespace(batch_graphs=1,batch_tokens=0,checkpoint_layers=True,chunk_tokens=2,
        checkpoint_input_offload=True,device='cpu',fp8_training='off',fp8_recipe='current',fp8_amax_history_len=16)
    old=capacity_identity({},args,None)
    assert 'checkpoint_input_offload_directory' not in old
    args.checkpoint_input_offload_directory=str(tmp_path)
    new=capacity_identity({},args,None)
    assert new!=old and new['checkpoint_input_offload_directory']==str(tmp_path)
    port=SimpleNamespace(activation_checkpoint=checkpoint)
    configure_checkpoint_storage_options(port,{'checkpoint_input_offload_directory':str(tmp_path)},
        {'checkpoint_input_offload':True})
    assert port.activation_checkpoint.directory==str(tmp_path)
    configure_checkpoint_storage_options(port,{'checkpoint_input_offload_directory':''})
    assert port.activation_checkpoint.directory is None
    with pytest.raises(ValueError,match='optional directory string'):
        configure_checkpoint_storage_options(port,{'checkpoint_input_offload_directory':False})


def test_malformed_abandoned_marker_does_not_break_training_or_delete_unknown_files(tmp_path):
    orphan=tmp_path/'natlang-checkpoint-input-interrupted-marker';orphan.mkdir()
    (orphan/'store.json').write_text('{')
    store=CheckpointFileStore(tmp_path)
    assert orphan.exists() and store.pack(torch.ones(2)).unpack().sum()==2


@pytest.mark.parametrize('family',['lfm'])
def test_actual_file_capacity_reference_preserves_optimizer_and_original_hook(tmp_path,family):
    from test_shared_state import tiny
    from test_native_optimizer_checkpointing import examples,assert_exact
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.eval.native_memory import offload_numerical_parity
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    model=SharedStateModel(tiny(family),(58,59,60,61),rank=4)
    if family=='mellum':model.port.hf.natlang_precision='bf16'
    optimizer,_=make_native_optimizer(model,lr=3e-6,in_backward=True)
    before=copy.deepcopy(optimizer.state_dict());weights=copy.deepcopy(model.state_dict())
    hook=configure_checkpoint_input_offload(model.port,enabled=True,directory=tmp_path)
    report=offload_numerical_parity(model,optimizer,examples()[0],chunk_tokens=3,
        checkpoint_sha256='a'*64,source_binding={},checkpoint_input_offload=True,
        checkpoint_input_offload_directory=str(tmp_path))
    assert report['passed'] and report['gradient_parameters']>0
    assert report['candidate_checkpoint_input_offload_directory']==str(tmp_path)
    assert model.port.activation_checkpoint is hook
    assert_exact(before,optimizer.state_dict());assert_exact(weights,model.state_dict())


@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference'])
def test_recipe_handoff_explicitly_tunes_storage_without_changing_the_stage(tmp_path,kind):
    from natlang_neuralese.train.recipe import native_stage_parameter_handoffs,_without_operational
    old={'stages':[{'id':'native','kind':kind,'parameters':{'checkpoint_input_offload':True}}]}
    new=copy.deepcopy(old)
    new['stages'][0]['parameters']['checkpoint_input_offload_directory']=str(tmp_path/'scratch')
    changes,before,after=native_stage_parameter_handoffs(old,new,tmp_path)
    assert changes['native']['checkpoint_input_offload_directory']==[None,str(tmp_path/'scratch')]
    assert _without_operational(before)==_without_operational(after)


@pytest.mark.parametrize('layout',['transpose','offset_slice','expanded','overlap','channels_last'])
@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('arena',[False,True])
def test_kernel_relevant_strides_survive_file_storage_without_unused_prefix(tmp_path,layout,device,arena):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    base=torch.arange(120,dtype=torch.float32,device=device)
    if layout=='transpose':tensor=base.reshape(12,10).T
    elif layout=='offset_slice':tensor=base[17:91:3]
    elif layout=='expanded':tensor=base[17:18].expand(7,9)
    elif layout=='overlap':tensor=base.as_strided((6,7),(2,1),storage_offset=17)
    else:tensor=base.reshape(2,3,4,5).to(memory_format=torch.channels_last)
    store=CheckpointFileStore(tmp_path)
    if arena:store.reserve_arena(1<<20)
    saved=store.pack(tensor);restored=saved.unpack()
    assert restored.stride()==tensor.stride()
    assert restored.storage_offset()==0
    torch.testing.assert_close(restored,tensor,atol=0,rtol=0)
    span=1+sum((n-1)*s for n,s in zip(tensor.shape,tensor.stride()))
    saved_bytes=saved.size if arena else saved.path.stat().st_size
    assert saved_bytes==span*tensor.element_size()
    if layout=='expanded':assert saved_bytes==tensor.element_size()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_bounded_pinned_buffer_reuse_preserves_all_saved_values_and_layouts(tmp_path):
    store=CheckpointFileStore(tmp_path, transfer_chunk_bytes=64)
    holders=[]
    for dtype in (torch.float64, torch.bfloat16, torch.int64, torch.bool):
        original=torch.arange(120, device='cuda').to(dtype).reshape(12,10)[1::2].T
        holders.append((store.pack(original), original.clone(), original.stride()))
    # Several later packs have overwritten the one buffer, including its dtype
    # view. Each holder must still restore independent, exact file-backed data.
    for saved, original, stride in reversed(holders):
        for _ in range(2):
            restored=saved.unpack()
            assert restored.device==original.device and restored.stride()==stride
            torch.testing.assert_close(restored,original,atol=0,rtol=0)
            restored.zero_()
    snapshot=store.snapshot()
    assert store.transfer_buffer.is_pinned()
    assert snapshot['pinned_transfer_bytes']==64
    assert snapshot['pinned_pack_chunks']>len(holders)
    assert snapshot['pinned_unpack_chunks']>2*len(holders)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_file_wrapper_honors_explicitly_disabled_pinning(tmp_path,device):
    if device=='cuda'and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    wrapper=CheckpointInputOffload(checkpoint, pin_memory=False, directory=tmp_path)
    x=torch.randn(13, device=device, dtype=torch.float64, requires_grad=True)
    result=wrapper(torch.sin,x,use_reentrant=False).sum()
    result.backward()
    torch.testing.assert_close(x.grad,x.detach().cos(),atol=0,rtol=0)
    assert wrapper.file_store.pin_memory is False
    assert wrapper.snapshot()['pinned_transfer_bytes']==0


@pytest.mark.parametrize('value',[True,0,-16,15,17])
def test_invalid_transfer_allocation_is_rejected_before_creating_store(tmp_path,value):
    with pytest.raises(ValueError,match='transfer chunk'):
        CheckpointFileStore(tmp_path,transfer_chunk_bytes=value)
    assert not list(tmp_path.iterdir())


def test_file_scratch_refuses_to_spend_recovery_floor_before_mapping(tmp_path,monkeypatch):
    from natlang_neuralese.train import checkpoint_safety as safety
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES',str(40*1024**3))
    store=CheckpointFileStore(tmp_path)
    assert store.minimum_free_bytes==40*1024**3
    from types import SimpleNamespace
    monkeypatch.setattr(safety.shutil,'disk_usage',lambda p:SimpleNamespace(free=store.minimum_free_bytes+31))
    monkeypatch.setattr(torch,'from_file',lambda *a,**k:pytest.fail('must reject before mapping or writing'))
    with pytest.raises(safety.CheckpointSpaceUnavailable):store.pack(torch.ones(8))
    assert not list(store.path.glob('tensor-*'))
    assert store.snapshot()['live_bytes']==0


def test_physical_scratch_reservation_failure_removes_partial_file(tmp_path,monkeypatch):
    import errno
    import natlang_neuralese.common.checkpoint_file_storage as storage
    store=CheckpointFileStore(tmp_path)
    def fail(*args):raise OSError(errno.ENOSPC,'physical reservation exhausted')
    monkeypatch.setattr(storage.os,'posix_fallocate',fail)
    monkeypatch.setattr(torch,'from_file',lambda *a,**k:pytest.fail('reserve before mapping'))
    with pytest.raises(OSError,match='physical reservation'):store.pack(torch.ones(8))
    assert not list(store.path.glob('tensor-*'))
    assert store.snapshot()['live_files']==0


def test_scratch_honors_stronger_checkpoint_floor(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES',str(50*1024**3))
    store=CheckpointFileStore(tmp_path)
    assert store.minimum_free_bytes==50*1024**3


def test_reserved_arena_protects_graph_owned_inputs_and_reuses_only_released_ranges(tmp_path,monkeypatch):
    import mmap
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES','0')
    store=CheckpointFileStore(tmp_path,pin_memory=False)
    page=mmap.PAGESIZE
    store.reserve_arena(2*page)
    assert (store.path/'arena.bin').stat().st_blocks*512>=2*page
    first=torch.arange(8,dtype=torch.float64)
    second=first[::2]
    a=store.pack(first);b=store.pack(second)
    assert a.offset!=b.offset and not list(store.path.glob('tensor-*'))
    first.fill_(100)
    torch.testing.assert_close(a.unpack(),torch.arange(8,dtype=torch.float64),rtol=0,atol=0)
    torch.testing.assert_close(b.unpack(),torch.arange(8,dtype=torch.float64)[::2],rtol=0,atol=0)
    with pytest.raises(RuntimeError,match='arena exhausted'):store.pack(torch.ones(2))
    with pytest.raises(ValueError,match='live graph'):store.release_arena()
    old=a.offset;a.cleanup()
    c=store.pack(torch.full((3,),7.))
    assert c.offset==old
    torch.testing.assert_close(b.unpack(),torch.arange(8,dtype=torch.float64)[::2],rtol=0,atol=0)
    b.cleanup();c.cleanup();store.release_arena()
    assert not (store.path/'arena.bin').exists()


def test_reserved_scratch_and_future_checkpoint_compete_above_same_floor(tmp_path,monkeypatch):
    import shutil
    from natlang_neuralese.train.checkpoint_safety import PersistentCheckpointReservation, CheckpointSpaceUnavailable
    monkeypatch.setenv('NATLANG_CHECKPOINT_TRANSFER_HEADROOM_FACTOR','0')
    floor=shutil.disk_usage(tmp_path).free-6*1024*1024
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES',str(floor))
    future=PersistentCheckpointReservation(tmp_path/'checkpoint.pt',transfer_bytes=0,minimum_free_bytes=floor)
    future.acquire(4*1024*1024)
    store=CheckpointFileStore(tmp_path/'scratch',pin_memory=False)
    try:
        with pytest.raises(CheckpointSpaceUnavailable):store.reserve_arena(4*1024*1024)
        assert not (store.path/'arena.bin').exists()
        assert future.pending.stat().st_blocks*512>=4*1024*1024
    finally:future.discard_uncommitted()


def test_scratch_peak_uses_exact_oversize_measured_cases(tmp_path):
    import hashlib
    from natlang_neuralese.train.native_state_sft import _measured_checkpoint_scratch_peak
    def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def report(peak):return {'passed':True,'cases':[{'checkpoint_input_offload':{'peak_live_bytes':peak}}]}
    base={'passed':True,'capacity':report(100)}
    measured=tmp_path/'measured.json';measured.write_text(json.dumps({'passed':True,'capacity':report(700)}))
    reuse=tmp_path/'reuse.json';reuse.write_text(json.dumps({'schema':'natlang.native-oversize-capacity-envelope-reuse/1',
        'passed':True,'case_matches':[{'envelope_source':ref(measured)}]}))
    assert _measured_checkpoint_scratch_peak(base,ref(reuse))==700
    measured.write_text('{}')
    with pytest.raises(ValueError):_measured_checkpoint_scratch_peak(base,ref(reuse))
    with pytest.raises(ValueError,match='measured file peak'):
        _measured_checkpoint_scratch_peak({'passed':True,'capacity':{'passed':True,'cases':[{}]}},None)


@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('pin_memory',[False,True])
def test_reserved_arena_retains_checkpoint_inputs_for_higher_derivatives(tmp_path,monkeypatch,device,pin_memory):
    from types import SimpleNamespace
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES','0')
    port=SimpleNamespace(activation_checkpoint=checkpoint)
    wrapper=configure_checkpoint_input_offload(port,enabled=True,pin_memory=pin_memory,directory=tmp_path)
    wrapper.file_store.reserve_arena(1<<20)
    x=torch.randn(12,dtype=torch.float64,device=device,requires_grad=True)
    y=x.detach().clone().requires_grad_()
    result=wrapper(lambda v:v.sin().square(),x,use_reentrant=False).sum()
    baseline=checkpoint(lambda v:v.sin().square(),y,use_reentrant=False).sum()
    grad=torch.autograd.grad(result,x,create_graph=True)[0]
    expected=torch.autograd.grad(baseline,y,create_graph=True)[0]
    torch.testing.assert_close(grad,expected,rtol=0,atol=0)
    second=torch.autograd.grad(grad.sum(),x)[0]
    expected_second=torch.autograd.grad(expected.sum(),y)[0]
    torch.testing.assert_close(second,expected_second,rtol=0,atol=0)


def test_failed_arena_copy_does_not_retain_a_live_slot(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES','0')
    store=CheckpointFileStore(tmp_path,pin_memory=False)
    store.reserve_arena(8192)
    monkeypatch.setattr(store,'copy_to_file',lambda *args:(_ for _ in ()).throw(OSError('copy failed')))
    with pytest.raises(OSError,match='copy failed'):store.pack(torch.ones(4))
    assert store.snapshot()['live_bytes']==0 and store.snapshot()['live_files']==0
    store.release_arena()


def test_scratch_bound_scales_case_scoped_peaks_to_actual_update_microbatches():
    from types import SimpleNamespace
    from natlang_neuralese.train.native_state_sft import _measured_checkpoint_scratch_peak
    def case(peak,tokens,scope='case'):
        return {'checkpoint_input_offload':{'peak_live_bytes':peak},'forward_tokens':tokens,
                **({'checkpoint_input_peak_scope':scope} if scope else {})}
    def micro(*tokens):return [SimpleNamespace(plan=SimpleNamespace(token_count=t)) for t in tokens]
    large={'passed':True,'capacity':{'passed':True,'cases':[case(49152*524650,524650),case(49152*16384+7,16384)]}}
    # Largest per-token rate times the update's largest microbatch, not the oversize graph.
    assert _measured_checkpoint_scratch_peak(large,None,[micro(8000,8000),micro(100)])==(-(-(49152*16384+7)//16384))*16000
    # Without the update geometry, keep the whole-evidence maximum.
    assert _measured_checkpoint_scratch_peak(large,None)==49152*524650
    # Historical lifetime-scoped peaks cannot be rescaled.
    legacy={'passed':True,'capacity':{'passed':True,'cases':[case(25787596800,524650,None),case(25787596800,16384,None)]}}
    assert _measured_checkpoint_scratch_peak(legacy,None,[micro(16384)])==25787596800


def test_case_peak_reset_and_oversized_arena_shrinks(tmp_path,monkeypatch):
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES','0')
    store=CheckpointFileStore(tmp_path,pin_memory=False)
    saved=store.pack(torch.ones(1024))
    assert store.snapshot()['peak_live_bytes']>0
    del saved
    store.reset_peak()
    assert store.snapshot()['peak_live_bytes']==store.snapshot()['live_bytes']==0
    page=mmap.PAGESIZE
    assert store.reserve_arena(8*page)==8*page
    assert store.reserve_arena(5*page)==8*page
    assert store.reserve_arena(2*page)==2*page
    assert (store.path/'arena.bin').stat().st_size==2*page
    store.release_arena()
