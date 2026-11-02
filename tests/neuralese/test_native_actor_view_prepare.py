"""natlang pins prepare a compact actor view in one call; the server loads it under the full checkpoint's identity."""
import json
import subprocess
import sys

import pytest
import torch

from natlang_neuralese.model.native_bundle import load_actor_view, prepare_actor_view
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.loop import capture_training_rng_state


def complete(tmp_path):
    from test_native_serving_adapters import engine_fixture
    engine = engine_fixture(tmp_path, 'lfm')
    state = CheckpointPolicy.load(engine.native_checkpoint)
    state.update(step=12, training={'schema': TRAINING_SCHEMA, 'global_step': 12,
        'optimizer': {'state': {'moment': torch.arange(7.)}}, 'rng': capture_training_rng_state('cpu'),
        'source_receipts': {'source': 'retained'}})
    path = tmp_path/'full.pt'
    torch.save(state, path)
    engine.store.close()
    return path


def test_prepared_view_serves_the_full_checkpoint_identity(tmp_path):
    path = complete(tmp_path)
    reference = prepare_actor_view(path, tmp_path/'views')
    assert prepare_actor_view(path, tmp_path/'views') == reference, 'a second preparation reuses the immutable view'
    view = load_actor_view(path, reference)
    assert 'training' not in view and view['actor_view']['training_resume'] is False
    full = load_native_engine(str(path), device='cpu')
    compact = load_native_engine(str(path), device='cpu', actor_view=reference)
    assert compact.native_inference_only and not full.native_inference_only
    prompt = list(range(5, 17))
    with torch.no_grad():
        a = full._native_logits(full._native_forward(prompt, None)['h_final'])
        b = compact._native_logits(compact._native_forward(prompt, None)['h_final'])
    assert torch.equal(a, b)


def test_cli_prints_the_receipt_and_refuses_incomplete_sources(tmp_path):
    path = complete(tmp_path)
    out = subprocess.run([sys.executable, '-m', 'natlang_neuralese.model.native_bundle', 'actor-view',
                          '--checkpoint', str(path), '--out', str(tmp_path/'views')], capture_output=True, text=True, check=True)
    reference = json.loads(out.stdout.strip().splitlines()[-1])
    assert set(reference) == {'path', 'sha256', 'bytes'}
    state = CheckpointPolicy.load(path)
    del state['training']['optimizer']
    torch.save(state, tmp_path/'partial.pt')
    with pytest.raises(ValueError, match='optimizer'):
        prepare_actor_view(tmp_path/'partial.pt', tmp_path/'views')


def tiny_complete(tmp_path):
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    state={'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'fixture'},
        'backbone_state':{'weight':torch.arange(64,dtype=torch.float32).reshape(8,8)},
        'interface_state':{'adapter':torch.arange(8,dtype=torch.float32)},'step':12,
        'training':{'optimizer':{'state':{'moment':torch.ones(8)}},'rng':{'seed':12},
                    'source_receipts':{'source':'retained'}}}
    path=tmp_path/'full.pt';torch.save(state,path);return path


def test_reuse_rechecks_values_without_second_serialization(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    path=tiny_complete(tmp_path);root=tmp_path/'views'
    first=prepare_actor_view(path,root)
    def no_write(*args,**kwargs):raise AssertionError('cached view must not be serialized again')
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',no_write)
    second=prepare_actor_view(path,root)
    assert second==first
    assert len(list((root/first_source_sha(path)).glob('*.pt')))==1


def first_source_sha(path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    return sha256_file_hex(path)


def test_cached_receipt_and_view_corruption_fail_closed(tmp_path):
    path=tiny_complete(tmp_path);root=tmp_path/'views';ref=prepare_actor_view(path,root)
    proof=__import__('pathlib').Path(ref['path']);original=proof.read_bytes()
    proof.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='receipt digest'):prepare_actor_view(path,root)
    proof.write_bytes(original)
    receipt=json.loads(original);view=proof.parent/receipt['view']['path']
    with view.open('r+b') as stream:
        stream.seek(-1,2);old=stream.read(1);stream.seek(-1,2);stream.write(bytes([old[0]^1]))
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):
        prepare_actor_view(path,root)


def test_bounded_comparison_handles_strides_and_detects_late_difference():
    from natlang_neuralese.model.native_bundle import _same_values_bounded
    a=torch.arange(8192,dtype=torch.float32).reshape(64,128).t()
    b=a.clone();_same_values_bounded({'tensor':a},{'tensor':b},max_bytes=256)
    b[-1,-1]+=1
    with pytest.raises(RuntimeError,match='actor view differs'):
        _same_values_bounded({'tensor':a},{'tensor':b},max_bytes=256)


def test_forged_receipt_for_changed_source_fails_tensor_comparison(tmp_path):
    import hashlib,os
    from pathlib import Path
    path=tiny_complete(tmp_path);root=tmp_path/'views';ref=prepare_actor_view(path,root)
    old_proof=Path(ref['path']);old=json.loads(old_proof.read_text())
    state=CheckpointPolicy.load(path);name=next(k for k,v in state['backbone_state'].items() if v.is_floating_point() and v.numel())
    state['backbone_state'][name]=state['backbone_state'][name].clone();state['backbone_state'][name].view(-1)[0]+=1
    changed=tmp_path/'changed.pt';torch.save(state,changed)
    newdir=root/first_source_sha(changed);newdir.mkdir()
    os.link(old_proof.parent/old['view']['path'],newdir/old['view']['path'])
    old['source']={'path':str(changed.resolve()),'sha256':first_source_sha(changed),'bytes':changed.stat().st_size}
    raw=(json.dumps(old,sort_keys=True,indent=2)+'\n').encode()
    (newdir/(hashlib.sha256(raw).hexdigest()+'.json')).write_bytes(raw)
    with pytest.raises(RuntimeError,match='actor view differs'):prepare_actor_view(changed,root)


def test_transport_reuses_view_but_binds_new_runtime(tmp_path,monkeypatch):
    from natlang_neuralese.model.native_bundle import prepare_actor_transport
    from natlang_neuralese.common import runtime_inventory
    from natlang_neuralese.train import trajectory_state
    path=tiny_complete(tmp_path);root=tmp_path/'views'
    monkeypatch.setattr(runtime_inventory,'python_runtime_binding',lambda:{'sha256':'a'*64})
    first=prepare_actor_transport(path,root,scope='diagnostic-local-pin')
    def no_write(*args,**kwargs):raise AssertionError('unexpected second serialization')
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',no_write)
    monkeypatch.setattr(runtime_inventory,'python_runtime_binding',lambda:{'sha256':'b'*64})
    second=prepare_actor_transport(path,root,scope='diagnostic-local-pin')
    before=json.loads(__import__('pathlib').Path(first['path']).read_text())
    after=json.loads(__import__('pathlib').Path(second['path']).read_text())
    assert before['view']==after['view'] and first['sha256']!=second['sha256']
    assert after['python_runtime']['sha256']=='b'*64


def test_snapshot_actor_view_uses_selected_owner_state_without_source_reload(tmp_path,monkeypatch):
    from natlang_neuralese.model.native_bundle import write_snapshot_actor_view
    from natlang_neuralese.train import trajectory_state
    path=tiny_complete(tmp_path);state=CheckpointPolicy.load(path);st=path.stat()
    signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1024**3)
    real=CheckpointPolicy.load
    def load_view_only(candidate,*args,**kwargs):
        if __import__('pathlib').Path(candidate).resolve()==path.resolve():raise AssertionError('full source reread')
        return real(candidate,*args,**kwargs)
    monkeypatch.setattr(CheckpointPolicy,'load',load_view_only)
    result=write_snapshot_actor_view(path,tmp_path/'snapshot-views',snapshot=state,decoded_source_stat=signature)
    assert result['maximum_cuda_storage_bytes']==0 and result['tensor_versions_unchanged']
    view=load_actor_view(path,result['view_receipt'])
    assert view['step']==12 and 'training' not in view


def test_snapshot_actor_view_rejects_insufficient_transfer_headroom(tmp_path,monkeypatch):
    from natlang_neuralese.model.native_bundle import write_snapshot_actor_view
    from natlang_neuralese.train import trajectory_state
    path=tiny_complete(tmp_path);state=CheckpointPolicy.load(path);st=path.stat()
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:0)
    with pytest.raises(ValueError,match='headroom'):
        write_snapshot_actor_view(path,tmp_path/'snapshot-views',snapshot=state,
                                  decoded_source_stat=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns))


def test_snapshot_view_preserves_storage_aliases_and_rejects_mutation(tmp_path,monkeypatch):
    from natlang_neuralese.model.native_bundle import write_snapshot_actor_view
    from natlang_neuralese.train import trajectory_state
    source=tiny_complete(tmp_path);state=CheckpointPolicy.load(source)
    storage=torch.arange(64,dtype=torch.float32).reshape(8,8)
    state['backbone_state']={'weight':storage,'transpose':storage.t()}
    path=tmp_path/'alias.pt';torch.save(state,path);state=CheckpointPolicy.load(path);st=path.stat()
    signature=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1024**3)
    result=write_snapshot_actor_view(path,tmp_path/'valid',snapshot=state,decoded_source_stat=signature)
    restored=load_actor_view(path,result['view_receipt'])['backbone_state']
    assert restored['weight'].untyped_storage().data_ptr()==restored['transpose'].untyped_storage().data_ptr()
    assert restored['transpose'].stride()==state['backbone_state']['transpose'].stride()
    storage=state['backbone_state']['weight'].clone()
    state['backbone_state']={'weight':storage,'transpose':storage.t()}
    original=trajectory_state.atomic_checkpoint
    def mutate(*args,**kwargs):
        original(*args,**kwargs)
        state['backbone_state']['weight'].view(-1)[0]+=1
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',mutate)
    with pytest.raises(ValueError,match='snapshot tensor changed'):
        write_snapshot_actor_view(path,tmp_path/'mutated',snapshot=state,decoded_source_stat=signature)
    assert not list((tmp_path/'mutated'/first_source_sha(path)).glob('*.json'))


def test_snapshot_owner_pins_full_source_and_bounds_pending_queue(tmp_path,monkeypatch):
    import json,os
    from types import SimpleNamespace
    from natlang_neuralese.model.native_snapshot_publication import publish_saved_snapshot
    from natlang_neuralese.train import trajectory_state
    source=tiny_complete(tmp_path);state=CheckpointPolicy.load(source)
    journal=tmp_path/'recovery';journal.mkdir();queue=SimpleNamespace(directory=journal)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1024**3)
    result=publish_saved_snapshot(queue,source,state)
    assert result['status']=='ready'
    pending=json.loads((journal/'actor-view-pending.json').read_text());pin=__import__('pathlib').Path(pending['pin'])
    assert pending['full_source_pinned'] and os.stat(pin).st_ino==os.stat(source).st_ino
    source.rename(tmp_path/'old-slot');source.write_bytes(b'new checkpoint')
    assert pin.read_bytes()==(tmp_path/'old-slot').read_bytes()
    assert publish_saved_snapshot(queue,source,state)['status']=='deferred'
    leases=json.loads((journal/'standalone-retention-consumer-leases.json').read_text())['leases']
    assert len(leases)==1 and leases[0]['role']==pending['lease']['role']


def test_snapshot_owner_failure_keeps_full_source_pin_and_retry_record(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from natlang_neuralese.model import native_snapshot_publication as publisher
    source=tiny_complete(tmp_path);state=CheckpointPolicy.load(source)
    journal=tmp_path/'recovery';journal.mkdir()
    actual=publisher.write_snapshot_actor_view
    monkeypatch.setattr(publisher,'write_snapshot_actor_view',lambda *a,**k:(_ for _ in ()).throw(OSError('disk full')))
    result=publisher.publish_saved_snapshot(SimpleNamespace(directory=journal),source,state)
    pending=json.loads((journal/'actor-view-pending.json').read_text())
    assert result['status']=='pending' and pending['full_source_pinned'] and pending['last_error']['message']=='disk full'
    assert __import__('pathlib').Path(pending['pin']).read_bytes()==source.read_bytes()
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(publisher,'write_snapshot_actor_view',actual)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1024**3)
    assert publisher.retry_pending(journal)['status']=='ready'


def test_owner_transport_and_explicit_release_require_verified_full_source(tmp_path,monkeypatch):
    import json,shutil,tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from natlang_neuralese.model import native_snapshot_publication as owner
    from natlang_neuralese.model.native_bundle import _fresh_file_binding
    from natlang_neuralese.common import runtime_inventory
    from natlang_neuralese.train import trajectory_state
    source=tiny_complete(tmp_path);state=CheckpointPolicy.load(source);journal=tmp_path/'recovery';journal.mkdir()
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1024**3)
    assert owner.publish_saved_snapshot(SimpleNamespace(directory=journal),source,state)['status']=='ready'
    record=json.loads((journal/'actor-view-pending.json').read_text());pin=Path(record['pin'])
    monkeypatch.setattr(runtime_inventory,'python_runtime_binding',lambda:{'sha256':'a'*64})
    def no_load(*args,**kwargs):raise AssertionError('transport must not decode full source')
    monkeypatch.setattr(CheckpointPolicy,'load',no_load)
    transport=owner.prepare_pending_transport(journal);manifest=json.loads(Path(transport['path']).read_text())
    delivery=tmp_path/'delivery.json';delivery.write_text(json.dumps({'schema':'natlang.actor-transfer-readback/1',
        'transport_sha256':transport['sha256'],'verified':{'sha256':manifest['view']['sha256'],
            'source_checkpoint_sha256':record['source']['sha256']}}))
    with tempfile.TemporaryDirectory(dir='/dev/shm') as external:
        recovered=Path(external)/'checkpoint.pt';shutil.copyfile(source,recovered)
        st=pin.stat();receipt=tmp_path/'recovery-proof.json'
        receipt.write_text(json.dumps({'schema':'natlang.local-full-recovery-export/1','full_state':True,
            'independent_readback':True,'source_identity':{'device':st.st_dev,'inode':st.st_ino,'size':st.st_size,'mtime_ns':st.st_mtime_ns},
            'checkpoint':_fresh_file_binding(recovered)}))
        actual_write=owner.write_json_durable;failed=[]
        def crash_after_retirement(path,value):
            if Path(path).name=='standalone-retention-consumer-leases.json' and not failed:
                failed.append(True);raise OSError('interrupted release')
            return actual_write(path,value)
        monkeypatch.setattr(owner,'write_json_durable',crash_after_retirement)
        with pytest.raises(OSError,match='interrupted release'):
            owner.release_pending(journal,recovery_ref=_fresh_file_binding(receipt),transport_ref=transport,delivery_ref=_fresh_file_binding(delivery))
        assert pin.exists() and (journal/'actor-view-pending.json').exists()
        result=owner.release_pending(journal,recovery_ref=_fresh_file_binding(receipt),transport_ref=transport,delivery_ref=_fresh_file_binding(delivery))
    assert result['source_sha256']==record['source']['sha256'] and not pin.exists()
    assert not (journal/'actor-view-pending.json').exists()


@pytest.mark.parametrize('asynchronous',[False,True])
def test_checkpoint_policy_default_publishes_from_its_committed_owner_state(tmp_path,monkeypatch,asynchronous):
    from natlang_neuralese.train import trajectory_state,checkpoint_safety
    source=tiny_complete(tmp_path);state=CheckpointPolicy.load(source)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1024**3)
    monkeypatch.setattr(checkpoint_safety,'native_async_snapshot_headroom',lambda *args,**kwargs:{'eligible':True})
    policy=CheckpointPolicy(tmp_path/'learner',prune_best=False,async_checkpoint=asynchronous)
    try:
        policy.reserve_next(state,torch.optim.AdamW([state['backbone_state']['weight']]))
        event=policy.save(state,async_allowed=asynchronous,async_optimizer=object() if asynchronous else None)
        policy.drain_async()
        pending=json.loads((tmp_path/'learner/actor-view-owner/actor-view-pending.json').read_text())
        assert pending['status']=='ready' and pending['full_source_pinned']
        assert event['actor_snapshot_view']['status']=='ready'
        assert pending['source']['sha256']==first_source_sha(policy.slot)
    finally:policy.close()


def test_primary_checkpoint_mutation_rejects_actor_publication_but_keeps_full_commit(tmp_path,monkeypatch):
    from natlang_neuralese.train import checkpoint_policy
    source=tiny_complete(tmp_path);state=CheckpointPolicy.load(source)
    state['backbone_state']['weight']=state['backbone_state']['weight'].clone()
    policy=CheckpointPolicy(tmp_path/'learner',prune_best=False,async_checkpoint=False)
    actual=checkpoint_policy.atomic_checkpoint
    def write_then_mutate(path,values,**kwargs):
        actual(path,values,**kwargs)
        values['backbone_state']['weight'].view(-1)[0]+=1
    monkeypatch.setattr(checkpoint_policy,'atomic_checkpoint',write_then_mutate)
    try:
        with pytest.raises(ValueError,match='primary checkpoint write'):
            policy.save(state)
        assert policy.slot.exists() and CheckpointPolicy.load(policy.slot)['training']['optimizer']
        assert not (tmp_path/'learner/actor-view-owner/actor-view-pending.json').exists()
    finally:policy.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA checkpoint-owner transfer')
@pytest.mark.parametrize('asynchronous', [False, True])
def test_cuda_committed_owner_view_survives_live_update_with_aliases(tmp_path, asynchronous):
    from natlang_neuralese.model.native_bundle import _same_values_bounded
    source = tiny_complete(tmp_path)
    state = CheckpointPolicy.load(source)
    weight = torch.arange(2048 * 1024, device='cuda', dtype=torch.float32).reshape(2048, 1024).to(torch.bfloat16)
    state['backbone_state'] = {'weight': weight, 'transpose': weight.t(), 'offset': weight[1:, 3:]}
    state['interface_state'] = {'adapter': torch.arange(8, device='cuda', dtype=torch.float32)}
    expected = {key: value.cpu().clone() for key, value in state['backbone_state'].items()}
    optimizer = torch.optim.AdamW([weight])
    policy = CheckpointPolicy(tmp_path/'learner', prune_best=False, async_checkpoint=asynchronous)
    try:
        event = policy.save(state, async_allowed=asynchronous, async_optimizer=optimizer)
        # The next live update must not alter the committed source or detached view.
        weight.add_(4)
        policy.drain_async()
        if asynchronous:
            assert event['async_mode'] == 'background'
        publication = event['actor_snapshot_view']
        assert publication['status'] == 'ready'
        view = load_actor_view(policy.slot, publication['view_receipt'])
        restored = view['backbone_state']
        _same_values_bounded(expected, restored)
        assert restored['weight'].untyped_storage().data_ptr() == restored['transpose'].untyped_storage().data_ptr()
        assert restored['offset'].storage_offset() == 1027
        assert restored['transpose'].stride() == (1, 1024)
        assert publication['source_sha256'] == first_source_sha(policy.slot)
    finally:
        policy.close()


def test_snapshot_pin_reuses_only_live_exact_checkpoint_lock(tmp_path):
    from natlang_neuralese.train.checkpoint_safety import (
        PersistentCheckpointReservation, checkpoint_target_lock, CheckpointReserveError)
    target=tmp_path/'checkpoint.pt'
    owner=PersistentCheckpointReservation(target)
    owner.acquire(4096)
    try:
        with checkpoint_target_lock(target,reservation=owner):
            with pytest.raises(BlockingIOError):
                with checkpoint_target_lock(target):pass
        # A publication hook must not release the trainer's lifetime lock.
        with pytest.raises(BlockingIOError):
            with checkpoint_target_lock(target):pass
        with pytest.raises(CheckpointReserveError,match='same slot'):
            with checkpoint_target_lock(tmp_path/'other.pt',reservation=owner):pass
        owner.discard_uncommitted()
    finally:owner.close()
    with pytest.raises(CheckpointReserveError,match='live'):
        with checkpoint_target_lock(target,reservation=owner):pass
    with checkpoint_target_lock(target):pass
