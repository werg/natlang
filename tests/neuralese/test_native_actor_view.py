import copy
import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.model.native_bundle import write_actor_view,load_actor_view
from natlang_neuralese.model.native_model import load_native_checkpoint
from natlang_neuralese.model.shared_state import Invocation,ImportSpan,compile_graph
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.native_optim import _native_optimizer_step_impl
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


@pytest.fixture(params=['lfm'])
def source(tmp_path,request):
    from test_native_serving_adapters import engine_fixture
    engine=engine_fixture(tmp_path,request.param)
    state=CheckpointPolicy.load(engine.native_checkpoint)
    state.update(step=71,training={'optimizer':{'actual_moment':torch.arange(19.)},
        'rng':{'cpu':torch.get_rng_state()},'source_receipts':{'train':{'sha256':'a'*64}},
        'sft_progress':{'native_full_sft':{'exposure_counts':{'row-'+str(i):1 for i in range(10000)}}}})
    path=tmp_path/'full.pt';torch.save(state,path)
    stat=path.stat();signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    state=CheckpointPolicy.load(path)
    yield engine,path,state,signature
    engine.store.close()


def test_both_ports_exact_actor_view_without_schedule_decode(source,tmp_path,monkeypatch):
    engine,path,state,signature=source
    before=sha256_file_hex_cached(path)
    ref=write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)
    assert write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)==ref
    receipt=json.loads(Path(ref['path']).read_text())
    assert receipt['source']['sha256']==before and receipt['training_resume']is False
    assert receipt['view']['bytes']<path.stat().st_size
    original_load=CheckpointPolicy.load
    seen=[]
    def view_only(p,*a,**kw):
        assert Path(p).resolve()!=path.resolve(),'Actor must not decode the full training state'
        seen.append(Path(p));return original_load(p,*a,**kw)
    monkeypatch.setattr(CheckpointPolicy,'load',view_only)
    restored=load_native_engine(str(path),device='cpu',actor_view=ref)
    try:
        assert len(seen)==1
        assert restored.native_identity['checkpoint_sha256']==before
        assert restored.rl_policy_binding['actor_view']['view']['sha256']==receipt['view']['sha256']
        assert restored.native_training_state=={}
        for name,value in engine.shared_model.named_parameters():
            target=dict(restored.shared_model.named_parameters())[name]
            assert value.dtype==target.dtype and torch.equal(value,target)
        plan=compile_graph([Invocation('p',(4,5,6,7),producing=((1,3),)),
            Invocation('c',(8,9),imports=(ImportSpan(1,'p',1,3),))])
        with torch.no_grad():
            a=engine.shared_model.logits(engine.shared_model(plan)['h_final'])
            b=restored.shared_model.logits(restored.shared_model(plan)['h_final'])
        assert torch.equal(a,b)
        with pytest.raises(ValueError,match='read-only'):
            load_native_checkpoint(path,actor_view=ref)
        with pytest.raises(Exception,match='complete parent checkpoint'):
            _native_optimizer_step_impl(restored,{},'test')
    finally:restored.store.close()
    assert sha256_file_hex_cached(path)==before
    assert state['training']['sft_progress']['native_full_sft']['exposure_counts']['row-9999']==1


@pytest.mark.parametrize('field',['source','receipt','view','profile','training'])
def test_bound_actor_view_tampering_fails(source,tmp_path,field):
    _,path,state,signature=source
    ref=write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)
    proof=Path(ref['path']);receipt=json.loads(proof.read_text())
    if field=='source':
        other=tmp_path/'other.pt';torch.save({**state,'step':72},other)
        with pytest.raises(ValueError,match='another complete'):
            load_actor_view(other,ref)
        return
    if field=='receipt':
        proof.write_text(proof.read_text()+' ')
        with pytest.raises(ValueError,match='receipt binding'):
            load_actor_view(path,ref)
        return
    view=proof.parent/receipt['view']['path']
    if field=='view':
        view.write_bytes(b'changed')
        with pytest.raises((ValueError,FileNotFoundError)):
            load_actor_view(path,ref)
        return
    value=CheckpointPolicy.load(view)
    if field=='profile':value['model_config']={**value['model_config'],'native_norm_execution':'wrong-profile'}
    else:value['training']={'optimizer':state['training']['optimizer']}
    replacement=view.with_suffix('.replacement');torch.save(value,replacement);replacement.replace(view)
    receipt['view'].update(sha256=sha256_file_hex_cached(view),bytes=view.stat().st_size)
    proof.write_text(json.dumps(receipt));ref.update(sha256=sha256_file_hex_cached(proof),bytes=proof.stat().st_size)
    with pytest.raises(ValueError,match='payload differs'):
        load_actor_view(path,ref)


def test_unsupported_view_retains_existing_full_loader(tmp_path):
    source=tmp_path/'composite.json';source.write_text('{}')
    assert write_actor_view(source,tmp_path/'views',validated_state={'schema':'unsupported-shared'})is None
    assert not(tmp_path/'views').exists()


def test_replaced_source_after_decode_cannot_receive_old_weights(source,tmp_path):
    _,path,state,signature=source
    other=path.with_suffix('.replacement');torch.save({**state,'step':72},other);other.replace(path)
    with pytest.raises(ValueError,match='unchanged file'):
        write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)
    assert not(tmp_path/'views').exists()


def test_physical_inference_export_cannot_publish_optimizer_updates(source,tmp_path):
    _,path,state,signature=source
    before=sha256_file_hex_cached(path)
    ref=write_actor_view(path,tmp_path/'physical-views',validated_state=state,decoded_source_stat=signature)
    receipt=json.loads(Path(ref['path']).read_text())
    view=Path(ref['path']).parent/receipt['view']['path']
    restored=load_native_engine(str(view),device='cpu')
    try:
        assert restored.native_identity['checkpoint_sha256']==receipt['view']['sha256']
        assert restored.rl_policy_binding['inference_source']['training_checkpoint_sha256']==before
        assert restored.native_training_state=={}
        with pytest.raises(Exception,match='complete parent checkpoint'):
            _native_optimizer_step_impl(restored,{},'test')
    finally:restored.store.close()
    assert sha256_file_hex_cached(path)==before
    assert state['training']['sft_progress']['native_full_sft']['exposure_counts']['row-9999']==1


@pytest.mark.parametrize('mutation',[None,'weights','step','truncated','symlink','outside'])
def test_interrupted_view_reuses_only_exact_fully_verified_output(source,tmp_path,monkeypatch,mutation):
    from natlang_neuralese.train import trajectory_state
    _,path,state,signature=source
    ref=write_actor_view(path,tmp_path/'original',validated_state=state,decoded_source_stat=signature)
    receipt=json.loads(Path(ref['path']).read_text())
    original=Path(ref['path']).parent/receipt['view']['path']
    out=tmp_path/'recovery';folder=out/receipt['source']['sha256'];folder.mkdir(parents=True)
    pending=folder/'.actor-view-interrupted.pt'
    import shutil
    shutil.copyfile(original,pending)
    if mutation in ('weights','step'):
        bad=copy.deepcopy(CheckpointPolicy.load(pending))
        if mutation=='step':bad['step']+=1
        else:
            key=next(iter(bad['backbone_state']))
            bad['backbone_state'][key]=bad['backbone_state'][key]+1
        torch.save(bad,pending)
    elif mutation=='truncated':pending.write_bytes(b'partial checkpoint')
    elif mutation=='symlink':
        pending.unlink();pending.symlink_to(original)
    elif mutation=='outside':
        pending=tmp_path/'.actor-view-outside.pt';shutil.copyfile(original,pending)
    def forbidden(*args,**kwargs):raise AssertionError('Recovery must not serialize actor weights again')
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',forbidden)
    source_hash=sha256_file_hex_cached(path)
    if mutation:
        with pytest.raises((ValueError,RuntimeError,AssertionError,EOFError,IndexError,__import__('pickle').UnpicklingError)):
            write_actor_view(path,out,validated_state=state,decoded_source_stat=signature,prepared_view=pending)
        assert pending.exists()
        assert not list(folder.glob('*.json'))
    else:
        recovered=write_actor_view(path,out,validated_state=state,decoded_source_stat=signature,prepared_view=pending)
        value=json.loads(Path(recovered['path']).read_text())
        assert value['view']['sha256']==receipt['view']['sha256']
        assert value['source']==receipt['source']
        assert not pending.exists()
        assert load_actor_view(path,recovered)['step']==state['step']
    assert sha256_file_hex_cached(path)==source_hash


def test_recovered_actor_view_retains_candidate_if_receipt_publication_fails(source,tmp_path,monkeypatch):
    import shutil
    from natlang_neuralese.model import native_composite
    _,path,state,signature=source
    ref=write_actor_view(path,tmp_path/'original',validated_state=state,decoded_source_stat=signature)
    value=json.loads(Path(ref['path']).read_text())
    folder=tmp_path/'recover'/value['source']['sha256'];folder.mkdir(parents=True)
    pending=folder/'.actor-view-recover.pt'
    shutil.copyfile(Path(ref['path']).parent/value['view']['path'],pending)
    def fail(*args,**kwargs):raise OSError('simulated receipt storage failure')
    monkeypatch.setattr(native_composite,'_durable_immutable',fail)
    with pytest.raises(OSError,match='receipt storage'):
        write_actor_view(path,tmp_path/'recover',validated_state=state,decoded_source_stat=signature,prepared_view=pending)
    assert pending.exists()
