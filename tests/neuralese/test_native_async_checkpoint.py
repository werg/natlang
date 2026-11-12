import copy
import errno
import json
import random
import threading
import time

import pytest
import torch

from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


@pytest.fixture(autouse=True)
def cpu_checkpoint_device_state(monkeypatch):
    # These states/optimizers are CPU fixtures. RNG capture can initialize CUDA
    # on a GPU workstation without a measured allocator peak; that must not
    # silently switch writer lifecycle tests to synchronous fallback. Tests of
    # device headroom explicitly override these observations below.
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)


def _full_state(model, optimizer, step, exposure):
    return {'step':step, 'model':model.state_dict(), 'training':{
        'optimizer':optimizer.state_dict(), 'rng':{'python':random.getstate(),
        'torch':torch.get_rng_state().clone()}, 'exposure_counts':dict(exposure)}}


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and torch.equal(actual,expected)
    elif isinstance(expected, dict):
        assert actual.keys()==expected.keys()
        for key,value in expected.items():_assert_tree_equal(actual[key],value)
    elif isinstance(expected, (tuple,list)):
        assert type(actual) is type(expected) and len(actual)==len(expected)
        for left,right in zip(actual,expected):_assert_tree_equal(left,right)
    else:
        assert actual==expected


def test_insufficient_memory_falls_back_to_synchronous_full_save(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda **_: 1)
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    state={'step':4,'model':{'weight':torch.ones(8)},'training':{'optimizer':{},'rng':{},'exposure_counts':{'a':4}}}
    event=policy.save(state,async_allowed=True,async_optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]))
    assert event['async_fallback']['eligible'] is False
    assert not policy.async_pending and event['committed']
    saved=torch.load(policy.slot,weights_only=False)
    assert saved['training']['exposure_counts']=={'a':4}
    assert saved['training']['options']['async_checkpoint_admission']['mode']=='synchronous-fallback'


def test_async_publication_hook_runs_after_durable_exact_snapshot(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda **_: 1<<50)
    observed=[]
    def publish(path,state,event):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['step']==state['step']
        observed.append((saved['step'],event['path']))
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True,on_saved=publish)
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    event=policy.save({'step':8,'training':{'optimizer':optimizer.state_dict(),'rng':{},
        'exposure_counts':{'a':8},'options':{}}},async_allowed=True,async_optimizer=optimizer)
    policy.drain_async()
    assert observed==[(8,str(policy.slot))]
    assert event['committed'] is True


def test_async_admission_preserves_previous_slot_when_atomic_disk_headroom_is_short(tmp_path,monkeypatch):
    from collections import namedtuple
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda **_: 1<<50)
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    previous={'step':1,'model':{'weight':torch.ones(4)},'training':{'optimizer':{},'rng':{},'exposure_counts':{}}}
    policy.save(previous)
    Usage=namedtuple('Usage','total used free')
    monkeypatch.setattr('natlang_neuralese.train.checkpoint_policy.shutil.disk_usage',
        lambda _path:Usage(10,10,1))
    parameter=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([parameter])
    with pytest.raises(OSError) as refused:
        policy.save({'step':2,'model':{'weight':torch.zeros(4)},'training':{
            'optimizer':optimizer.state_dict(),'rng':{},'exposure_counts':{}}},
            async_allowed=True,async_optimizer=optimizer)
    assert refused.value.errno==errno.ENOSPC
    assert torch.load(policy.slot,weights_only=False)['step']==1
    assert not policy.async_pending


def test_failed_async_write_recovers_latest_state_before_returning(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    real_write=trajectory_state.atomic_checkpoint
    calls=[]
    def fail_first(path,state):
        calls.append(state['step'])
        if len(calls)==1:raise OSError('simulated checkpoint device failure')
        return real_write(path,state)
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',fail_first)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda **_: 1<<50)
    parameter=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([parameter])
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    first={'step':1,'model':{'p':parameter.detach().clone()},'training':{'optimizer':optimizer.state_dict(),
        'rng':{'torch':torch.get_rng_state()},'exposure_counts':{'a':1},'options':{}}}
    policy.save(first,async_allowed=True,async_optimizer=optimizer)
    assert parameter.device.type=='cpu'
    current=copy.deepcopy(first)
    current['step']=2;current['training']['exposure_counts']['a']=2
    event=policy.save(current)  # detects writer error, then synchronously commits current state
    assert event['async_writer_recovery']['recovered_synchronously'] is True
    saved=torch.load(policy.slot,weights_only=False)
    assert saved['step']==2 and saved['training']['exposure_counts']=={'a':2}
    assert current['training']['options']['async_checkpoint_recovery_history']


def test_forced_sync_save_and_finalize_drain_writer_before_consumption(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    entered,release=threading.Event(),threading.Event()
    real_write=trajectory_state.atomic_checkpoint
    def held(path,state):
        entered.set()
        if not release.wait(10):raise TimeoutError('checkpoint test did not release writer')
        return real_write(path,state)
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',held)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda **_: 1<<50)
    parameter=torch.nn.Parameter(torch.ones(3))
    optimizer=torch.optim.AdamW([parameter])
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    policy.save({'step':1,'training':{'optimizer':optimizer.state_dict(),'rng':{},'exposure_counts':{}}},
        async_allowed=True,async_optimizer=optimizer)
    assert entered.wait(5)
    released=[]
    def delayed_release():
        time.sleep(.1)
        release.set()
        released.append(True)
    threading.Thread(target=delayed_release,daemon=True).start()
    event=policy.save({'step':2,'training':{'optimizer':optimizer.state_dict(),'rng':{},'exposure_counts':{}}})
    assert event['committed'] and released
    policy.finalize({'weights':torch.ones(1)})
    assert torch.load(policy.slot,weights_only=False)['step']==2

# Written for one optimizer update per packed batch; accumulation is tested in test_native_update_accumulation.py.
pytestmark = pytest.mark.usefixtures('per_batch_native_updates')


def test_reserved_background_write_retains_separate_next_slot(tmp_path, monkeypatch):
    from natlang_neuralese.train import trajectory_state
    entered, release = threading.Event(), threading.Event()
    original = trajectory_state.atomic_checkpoint
    def delayed(path, state, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(path, state, **kwargs)
    monkeypatch.setattr(trajectory_state, 'atomic_checkpoint', delayed)
    monkeypatch.setattr(trajectory_state, 'available_system_memory_bytes', lambda **_: 1 << 50)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    state = {'step': 1, 'training': {'optimizer': optimizer.state_dict(), 'rng': {},
                                    'exposure_counts': {'graph': 1}, 'options': {}}}
    policy = CheckpointPolicy(tmp_path, async_checkpoint=True)
    policy.reserve_next(state, optimizer)
    first = policy._persistent_reservation
    event = policy.save(state, async_allowed=True, async_optimizer=optimizer)
    assert event['async_pending'] and entered.wait(5)
    second = policy._persistent_reservation
    assert first.pending != second.pending
    assert first.pending.stat().st_ino != second.pending.stat().st_ino
    policy.reserve_next(state, optimizer)
    state['training']['exposure_counts']['graph'] = 2
    release.set()
    policy.drain_async()
    assert torch.load(policy.slot, weights_only=False)['training']['exposure_counts']['graph'] == 1
    assert second.pending.exists() and second.bound > 0
    from natlang_neuralese.train.checkpoint_safety import checkpoint_target_lock
    with pytest.raises(BlockingIOError):
        with checkpoint_target_lock(policy.slot):
            pass
    second.close()
    with checkpoint_target_lock(policy.slot):
        pass


def test_policy_close_drains_live_writer_and_retires_empty_spare(tmp_path, monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setenv('NATLANG_CHECKPOINT_TRANSFER_HEADROOM_FACTOR', '0')
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES', '0')
    monkeypatch.setattr(trajectory_state, 'available_system_memory_bytes', lambda **_: 1 << 50)
    entered, release = threading.Event(), threading.Event()
    original = trajectory_state.atomic_checkpoint
    def delayed(path, state, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(path, state, **kwargs)
    monkeypatch.setattr(trajectory_state, 'atomic_checkpoint', delayed)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    state = {'step': 1, 'training': {'optimizer': optimizer.state_dict(), 'rng': {},
                                    'exposure_counts': {'graph': 1}, 'options': {}}}
    policy = CheckpointPolicy(tmp_path, async_checkpoint=True)
    policy.reserve_next(state, optimizer)
    event = policy.save(state, async_allowed=True, async_optimizer=optimizer)
    assert event['async_pending'] and entered.wait(5)
    pending = policy._persistent_reservation.pending
    closed = threading.Event()
    def close_policy():
        policy.close()
        closed.set()
    closer = threading.Thread(target=close_policy)
    closer.start()
    assert not closed.wait(.1)  # close must wait for the owned background write
    release.set()
    closer.join(5)
    assert closed.is_set() and not closer.is_alive()
    assert torch.load(policy.slot, weights_only=False)['training']['exposure_counts'] == {'graph': 1}
    assert not pending.exists()
    roles = json.loads(policy.slot.with_name(policy.slot.name + '.reservation-roles.json').read_text())
    assert roles['writer'] is None and roles['next'] is None
    successor = CheckpointPolicy(tmp_path)
    successor.reserve_next(state, optimizer)
    assert successor._persistent_reservation.pending.is_file()
    successor.close()


def test_reserved_submit_failure_recovers_current_state_synchronously(tmp_path, monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state, 'available_system_memory_bytes', lambda **_: 1 << 50)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    state = {'step': 2, 'training': {'optimizer': optimizer.state_dict(), 'rng': {},
                                    'exposure_counts': {'graph': 2}, 'options': {}}}
    policy = CheckpointPolicy(tmp_path, async_checkpoint=True)
    policy.reserve_next(state, optimizer)
    def fail(*args, **kwargs):
        raise RuntimeError('snapshot submission failed')
    monkeypatch.setattr(policy._async_writer, 'submit', fail)
    event = policy.save(state, async_allowed=True, async_optimizer=optimizer)
    assert event['async_writer_recovery']['recovered_synchronously']
    assert torch.load(policy.slot, weights_only=False)['step'] == 2


def test_reserved_hook_failure_surfaces_after_durable_commit(tmp_path, monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state, 'available_system_memory_bytes', lambda **_: 1 << 50)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    state = {'step': 3, 'training': {'optimizer': optimizer.state_dict(), 'rng': {},
                                    'exposure_counts': {'graph': 3}, 'options': {}}}
    def fail(*args):
        raise RuntimeError('post-commit hook failed')
    policy = CheckpointPolicy(tmp_path, async_checkpoint=True, on_saved=fail)
    policy.reserve_next(state, optimizer)
    policy.save(state, async_allowed=True, async_optimizer=optimizer)
    with pytest.raises(RuntimeError, match='post-commit hook failed'):
        policy.drain_async()
    assert torch.load(policy.slot, weights_only=False)['step'] == 3
    policy._persistent_reservation.close()


def test_failed_reserved_writer_recovers_newer_full_state_in_spare(tmp_path, monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state, 'available_system_memory_bytes', lambda **_: 1 << 50)
    original = trajectory_state.atomic_checkpoint
    calls = []
    def fail_first(path, state, **kwargs):
        calls.append(state['step'])
        if len(calls) == 1:
            raise OSError(errno.EIO, 'injected writer error')
        return original(path, state, **kwargs)
    monkeypatch.setattr(trajectory_state, 'atomic_checkpoint', fail_first)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    state = {'step': 4, 'training': {'optimizer': optimizer.state_dict(), 'rng': {},
                                    'exposure_counts': {'graph': 4}, 'options': {}}}
    policy = CheckpointPolicy(tmp_path, async_checkpoint=True)
    policy.reserve_next(state, optimizer)
    policy.save(state, async_allowed=True, async_optimizer=optimizer)
    state['step'] = 5
    state['training']['exposure_counts']['graph'] = 5
    event = policy.save(state, async_allowed=True, async_optimizer=optimizer)
    assert event['async_writer_recovery']['recovered_synchronously']
    saved = torch.load(policy.slot, weights_only=False)
    assert saved['step'] == 5 and saved['training']['exposure_counts']['graph'] == 5
    assert calls == [4]


def test_async_admission_reserves_measured_future_device_growth(monkeypatch):
    from natlang_neuralese.train import trajectory_state
    from natlang_neuralese.train.checkpoint_safety import native_async_snapshot_headroom
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(trajectory_state, 'page_cache_reclaimable_for_host_allocations', lambda: True)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 1000)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 500)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    state = {'model': torch.ones(2)}
    monkeypatch.setattr(trajectory_state, 'available_system_memory_bytes', lambda **_: 1 << 40)
    result = native_async_snapshot_headroom(state, optimizer)
    assert result['future_device_growth_allowance_bytes'] == 750
    assert result['required_headroom_bytes'] > result['snapshot_size_bound_bytes'] + 750
    monkeypatch.setattr(trajectory_state, 'available_system_memory_bytes',
                        lambda **_: result['required_headroom_bytes'] - 1)
    assert not native_async_snapshot_headroom(state, optimizer)['eligible']


def test_async_admission_preserves_lifetime_reserved_peak_across_step_peak_reset(monkeypatch):
    from natlang_neuralese.train import trajectory_state
    from natlang_neuralese.train.checkpoint_safety import native_async_snapshot_headroom
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(trajectory_state, 'page_cache_reclaimable_for_host_allocations', lambda: True)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 1000)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda: 500)
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    state={'model':torch.ones(2),'training':{'options':{'_cuda_lifetime_peak_reserved_bytes':2000}}}
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda **_: 1<<40)
    result=native_async_snapshot_headroom(state,optimizer)
    assert result['device_peak_reserved_bytes']==2000
    assert result['future_device_growth_allowance_bytes']==2000


def test_low_disk_spare_falls_back_to_current_reserved_checkpoint(tmp_path,monkeypatch):
    from natlang_neuralese.train import checkpoint_safety
    model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters())
    state=_full_state(model,optimizer,1,{'first':1});policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    monkeypatch.setenv('NATLANG_CHECKPOINT_TRANSFER_HEADROOM_FACTOR','0')
    monkeypatch.setenv('NATLANG_CHECKPOINT_MIN_FREE_BYTES','0')
    policy.reserve_next(state,optimizer);old_inode=policy._persistent_reservation.pending.stat().st_ino
    def limited(self):
        raise checkpoint_safety.CheckpointSpaceUnavailable('floor would be crossed')
    monkeypatch.setattr(checkpoint_safety.CheckpointWriteAdmission,'__enter__',limited)
    monkeypatch.setattr(checkpoint_safety,'native_async_snapshot_headroom',lambda *_a,**_k:{'eligible':True})
    event=policy.save(state,async_allowed=True,async_optimizer=optimizer)
    assert not event.get('async_pending') and event['async_fallback']['reason']=='insufficient_checkpoint_spare_space'
    assert policy.slot.stat().st_ino==old_inode and torch.load(policy.slot,weights_only=False)['step']==1
    assert not policy.slot.with_name(policy.slot.name+'.alternate.reserved-pending').exists()
    policy.close()
