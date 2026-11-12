import errno
import os

import pytest
import torch



def test_evaluation_points_are_declared_steps_and_include_every_write_and_the_end():
    from natlang_neuralese.train.loop import Cadence, check_declared_points
    evals = Cadence(512)
    assert [s for s in range(0, 8193) if evals.due(s)] == list(range(512, 8193, 512))  # 16 points, none at step 0
    check_declared_points(8192, 512, 2048)
    with pytest.raises(ValueError, match="multiple of eval_every"):
        check_declared_points(8000, 512, 2048)  # the end would not be an evaluation point
    with pytest.raises(ValueError, match="checkpoint_every 1000 is not a multiple"):
        check_declared_points(8192, 512, 1000)  # a write without its own evaluation


def test_the_trainers_atomic_writer_leaves_the_previous_file_until_the_replacement_is_complete(tmp_path, monkeypatch):
    from natlang_neuralese.train.trajectory_state import atomic_checkpoint
    target = tmp_path / "checkpoint.pt"
    atomic_checkpoint(target, {"v": torch.ones(2)})
    monkeypatch.setattr(torch, "save", lambda state, stream: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full")))
    with pytest.raises(OSError):
        atomic_checkpoint(target, {"v": torch.zeros(2)})
    monkeypatch.undo()
    assert torch.equal(torch.load(target)["v"], torch.ones(2))
    assert not list(tmp_path.glob("*.pending"))


def test_disk_reserve_holds_the_next_write_and_grows_with_optimizer_state(tmp_path):
    """run-v5 left an 80 GB reserve for a 23.6 GB state: the reserve is sized to the state as it is (5% margin), not
    to every future lazy optimizer slot, and grows when slots appear."""
    from natlang_neuralese.train.checkpoint_safety import CheckpointDiskReserve, warmup_checkpoint_size_upper_bound
    weight = torch.nn.Parameter(torch.zeros(1000, 100))
    heads = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW([weight])
    now = warmup_checkpoint_size_upper_bound([('w', weight)], heads, optimizer, lazy=False, margin=1.05)
    future = warmup_checkpoint_size_upper_bound([('w', weight)], heads, optimizer)
    assert now < future and now < 1.1 * (weight.numel() * 4 + 16 * 1024 * 1024)
    reserve = CheckpointDiskReserve(tmp_path / '.checkpoint-space.reserve', now)
    reserve.acquire()
    assert (tmp_path / '.checkpoint-space.reserve').stat().st_size == now
    weight.grad = torch.ones_like(weight)
    optimizer.step()  # two moments appear
    grown = warmup_checkpoint_size_upper_bound([('w', weight)], heads, optimizer, lazy=False, margin=1.05)
    assert grown > now and reserve.grow_to(grown) == grown and reserve.grow_to(now) == grown  # never shrinks
    reserve.cleanup()
    assert not (tmp_path / '.checkpoint-space.reserve').exists()


def test_atomic_checkpoint_admission_sizes_first_write_and_holds_transfer_reserve(tmp_path, monkeypatch):
    from natlang_neuralese.train.checkpoint_safety import CheckpointWriteAdmission, checkpoint_size_upper_bound
    from natlang_neuralese.train.trajectory_state import atomic_checkpoint

    tensor = torch.zeros(4096, dtype=torch.float32)
    state = {'model': {'w': tensor, 'view': tensor[:32]}, 'step': 3}
    bound = checkpoint_size_upper_bound(state, margin=1.0, minimum_overhead=0)
    assert bound >= tensor.numel() * tensor.element_size()
    assert bound < tensor.numel() * tensor.element_size() * 2
    target = tmp_path / 'new' / 'checkpoint.pt'
    monkeypatch.setenv('NATLANG_CHECKPOINT_ADMISSION_LOCK_ROOT', str(tmp_path/'shared-locks'))
    with CheckpointWriteAdmission(target, 2 * 1024**2, transfer_bytes=1024**2,
                                  minimum_free_bytes=0) as admission:
        reserve = target.parent / f'.natlang-transfer-reserve-{target.parent.stat().st_dev}'
        assert admission._reserve is not None
        assert reserve.stat().st_size >= 1024**2
    assert not reserve.exists()
    atomic_checkpoint(target, state)
    assert torch.load(target, weights_only=True)['step'] == 3


def test_checkpoint_admission_refuses_insufficient_first_write_without_target(tmp_path):
    from natlang_neuralese.train.checkpoint_safety import CheckpointReserveError, CheckpointWriteAdmission

    target = tmp_path / 'missing' / 'checkpoint.pt'
    with pytest.raises(CheckpointReserveError, match='atomic checkpoint admission needs'):
        with CheckpointWriteAdmission(target, 10**30, transfer_bytes=0, minimum_free_bytes=0):
            pytest.fail('admission unexpectedly passed')
    assert not target.exists()


def test_atomic_checkpoint_physically_reserves_pending_bytes_before_serialization(tmp_path, monkeypatch):
    from natlang_neuralese.train import trajectory_state as state_module
    from natlang_neuralese.train.checkpoint_safety import checkpoint_size_upper_bound

    target = tmp_path / 'checkpoint.pt'
    saved = {'weights': torch.arange(32), 'optimizer': {'age': 17}, 'rng': torch.get_rng_state()}
    bound = checkpoint_size_upper_bound(saved)
    real_save = torch.save
    observed = []

    def inspect_save(state, writer):
        allocated = os.fstat(writer.stream.fileno())
        assert allocated.st_size == bound
        assert allocated.st_blocks * 512 >= bound
        assert writer.stream.tell() == 0
        observed.append(bound)
        return real_save(state, writer)

    monkeypatch.setattr(torch, 'save', inspect_save)
    state_module.atomic_checkpoint(target, saved)
    loaded = torch.load(target, weights_only=True)
    assert observed == [bound]
    assert target.stat().st_size < bound
    assert torch.equal(loaded['weights'], saved['weights'])
    assert torch.equal(loaded['rng'], saved['rng'])
    assert loaded['optimizer'] == saved['optimizer']


def test_pending_preallocation_failure_preserves_the_previous_complete_state(tmp_path, monkeypatch):
    from natlang_neuralese.train import trajectory_state as state_module

    target = tmp_path / 'checkpoint.pt'
    state_module.atomic_checkpoint(target, {'age': 17})
    before = target.read_bytes()
    real_allocate = os.posix_fallocate

    def fail_pending(fd, offset, length):
        if os.readlink(f'/proc/self/fd/{fd}').endswith('.pending'):
            raise OSError(errno.ENOSPC, 'another writer consumed unreserved space')
        return real_allocate(fd, offset, length)

    monkeypatch.setattr(os, 'posix_fallocate', fail_pending)
    with pytest.raises(OSError, match='unreserved space'):
        state_module.atomic_checkpoint(target, {'age': 18})
    assert target.read_bytes() == before
    assert not target.with_suffix('.pending').exists()


def test_checkpoint_digest_writer_never_extends_beyond_its_admitted_bound(tmp_path):
    from natlang_neuralese.train.trajectory_state import _CheckpointDigestWriter
    from natlang_neuralese.train.checkpoint_safety import CheckpointReserveError

    with (tmp_path / 'pending').open('wb') as stream:
        os.posix_fallocate(stream.fileno(), 0, 8)
        writer = _CheckpointDigestWriter(stream, write_bound_bytes=8)
        writer.write(b'first')
        with pytest.raises(CheckpointReserveError, match='preallocated byte bound'):
            writer.write(b'overrun')
        assert writer.bytes == 5
        assert stream.tell() == 5
        assert os.fstat(stream.fileno()).st_size == 8


def test_checkpoint_admission_serializes_same_filesystem_writers(tmp_path):
    from natlang_neuralese.train.checkpoint_safety import CheckpointWriteAdmission

    lock_root = tmp_path/'shared-locks'
    os.environ['NATLANG_CHECKPOINT_ADMISSION_LOCK_ROOT'] = str(lock_root)
    first_target = tmp_path/'left'/'checkpoint.pt'
    second_target = tmp_path/'right'/'checkpoint.pt'
    first = CheckpointWriteAdmission(first_target, 1, transfer_bytes=0, minimum_free_bytes=0)
    second = CheckpointWriteAdmission(second_target, 1, transfer_bytes=0, minimum_free_bytes=0,
                                      lock_timeout_seconds=0.05)
    from natlang_neuralese.train.checkpoint_safety import _checkpoint_admission_lock_path
    assert _checkpoint_admission_lock_path(tmp_path.stat().st_dev) == \
        _checkpoint_admission_lock_path(second_target.parent.parent.stat().st_dev)
    first.__enter__()
    try:
        with pytest.raises(Exception, match='timed out waiting'):
            second.__enter__()
    finally:
        first.__exit__(None, None, None)


def test_checkpoint_size_bound_supports_serialized_sparse_native_state(tmp_path, monkeypatch):
    from natlang_neuralese.train.checkpoint_safety import checkpoint_size_upper_bound

    monkeypatch.setenv('NATLANG_CHECKPOINT_ADMISSION_LOCK_ROOT', str(tmp_path/'shared-locks'))
    indices = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    sparse_coo = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
    sparse_csr = sparse_coo.to_sparse_csr()
    state = {'schema': 'natlang.native-state-checkpoint/1',
             'model': {'sparse_rows': sparse_coo, 'sparse_optimizer_history': sparse_csr},
             'training': {'global_step': 12, 'optimizer': {'state': {'sparse': sparse_coo}}}}
    bound = checkpoint_size_upper_bound(state, margin=1.0, minimum_overhead=0)
    storages = {}
    for tensor in (sparse_coo._indices(), sparse_coo._values(), sparse_csr.crow_indices(),
                   sparse_csr.col_indices(), sparse_csr.values()):
        storage = tensor.untyped_storage()
        storages[(storage.data_ptr(), storage.nbytes(), tensor.device.type, tensor.device.index)] = storage.nbytes()
    expected = sum(storages.values())
    assert bound >= expected
    path = tmp_path/'native.pt'
    torch.save(state, path)
    restored = torch.load(path, map_location='cpu', weights_only=True)
    assert restored['model']['sparse_rows'].layout == torch.sparse_coo
    assert restored['model']['sparse_optimizer_history'].layout == torch.sparse_csr
    assert checkpoint_size_upper_bound(restored, margin=1.0, minimum_overhead=0) >= expected


def test_weights_digest_is_a_cheap_on_device_fingerprint_that_sees_every_element():
    from natlang_neuralese.train.trajectory_state import weights_digest
    backbone = {'w': torch.randn(3, 5).to(torch.bfloat16), 'b': torch.zeros(4)}
    heads = {'h': torch.arange(6, dtype=torch.int64), 'm': torch.tensor([True, False])}
    first = weights_digest(backbone, heads)
    assert first == weights_digest({k: v.clone() for k, v in backbone.items()}, heads)
    for name in ('w', 'b'):
        changed = {k: v.clone() for k, v in backbone.items()}
        changed[name].view(-1)[-1] += 1
        assert weights_digest(changed, heads) != first
    swapped = {'w': backbone['w'].flip(0), 'b': backbone['b']}  # same values, other positions
    assert weights_digest(swapped, heads) != first


def test_port_optimizer_steps_latents_with_lionsr_and_round_trips_its_state():
    """The latent policy's memory-lean path: BF16 latents under LionSR (one BF16 momentum), the rest Muon/AdamW."""
    pytest.importorskip('torch.optim', reason='needs torch.optim.Muon').__dict__.get('Muon') or pytest.skip('no Muon')
    from natlang_neuralese.train.optim import PortMuonAdamW
    from natlang_neuralese.train.checkpoint_safety import optimizer_state_size_upper_bound
    torch.manual_seed(0)
    head = torch.nn.Parameter(torch.randn(8, 8))
    norm = torch.nn.Parameter(torch.ones(8))
    latent = torch.nn.Parameter(torch.randn(16, 8).to(torch.bfloat16))
    scale = torch.full((16, 1), 0.5)

    def build():
        return PortMuonAdamW([('heads.w', head), ('backbone.norm', norm)], lr=1e-3, vocab_size=100,
                             latent=[('backbone.w', latent, 1e-2, scale)])
    optimizer = build()
    assert [row['optimizer'] for row in optimizer.schema] == ['muon', 'adamw', 'lion']
    assert optimizer_state_size_upper_bound(optimizer) >= latent.numel() * 2  # its lazy BF16 momentum
    before = latent.detach().clone()
    for parameter in (head, norm, latent):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    moved = (latent.detach().float() - before.float()) / 0.5
    # Sign step of lr per row-scale unit, written with unbiased stochastic rounding: -0.01 on average.
    assert abs(float(moved.mean()) + 0.01) < 0.005 and optimizer.latent.state[latent]['momentum'].dtype == torch.bfloat16
    again = build()
    again.load_state_dict(optimizer.state_dict())
    assert torch.equal(again.latent.state[latent]['momentum'], optimizer.latent.state[latent]['momentum'])


def test_committed_state_save_waits_for_space_without_replaying_update():
    import errno
    from natlang_neuralese.train.checkpoint_safety import CheckpointReserveError, CheckpointSpaceUnavailable, save_with_space_retry
    state = {'global_step': 54, 'exposure': {'row': 2}, 'optimizer': object(), 'rng': object()}
    attempts = []
    waits = []
    def save():
        attempts.append(state)
        if len(attempts) == 1:
            raise CheckpointSpaceUnavailable('full')
        if len(attempts) == 2:
            raise OSError(errno.ENOSPC, 'full')
        return 'committed'
    assert save_with_space_retry(save, retry_seconds=0.1, sleep=waits.append) == 'committed'
    assert attempts == [state, state, state] and waits == [0.1, 0.1]
    assert state['global_step'] == 54 and state['exposure'] == {'row': 2}
    with pytest.raises(CheckpointReserveError, match='unowned'):
        save_with_space_retry(lambda: (_ for _ in ()).throw(CheckpointReserveError('unowned')), sleep=waits.append)
    with pytest.raises(OSError):
        save_with_space_retry(lambda: (_ for _ in ()).throw(OSError(errno.EIO, 'io')), sleep=waits.append)


def test_future_reservation_can_stop_but_dirty_checkpoint_save_keeps_retrying():
    from natlang_neuralese.train.checkpoint_safety import (CheckpointSpaceUnavailable,
        reserve_future_with_space_retry, save_with_space_retry)
    stopped=[]; attempts=[]
    def reserve():
        attempts.append(1)
        raise CheckpointSpaceUnavailable('no room')
    def wait(_):stopped.append(True)
    assert reserve_future_with_space_retry(reserve,stop_requested=lambda:bool(stopped),sleep=wait) is False
    assert attempts==[1]
    writes=[]
    def write():
        writes.append(1)
        if len(writes)==1:raise CheckpointSpaceUnavailable('no room')
        return 'durable'
    assert save_with_space_retry(write,retry_seconds=.1,sleep=lambda _:None)=='durable'
    assert len(writes)==2
